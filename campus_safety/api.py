"""FastAPI application: HTTP routes, WebSocket broadcast, and the demo wiring.

Every incident — whatever its source — is scored by the same risk engine and
persisted with its signal breakdown, then pushed to the dashboard over a
WebSocket. The two incident sources (student SOS and AI video events) meet in
`_create_incident`, which is where the fusion (the verification bonus) happens.
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any

from fastapi import Body, FastAPI, Header, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import audit, config, db, risk
from .detectors import active_backend
from .events import VideoEvent
from .models import Incident, STATUS_ORDER, Zone
from .pipeline import Pipeline
from .zones import (
    CAMPUS_ZONES,
    DEFAULT_ZONE_ID,
    ZONES_BY_ID,
    normalized_history_risk,
    zone_centroid,
)

log = logging.getLogger("campus_safety.api")


# --- WebSocket hub ---------------------------------------------------------


class Hub:
    """Fan-out to every connected dashboard client.

    Simple by design: one process, a handful of clients. A multi-worker
    deployment would swap this for the Redis pub/sub noted in Slide 7.
    """

    def __init__(self) -> None:
        self._clients: set[WebSocket] = set()
        self._lock = asyncio.Lock()

    async def connect(self, ws: WebSocket) -> None:
        await ws.accept()
        async with self._lock:
            self._clients.add(ws)

    async def disconnect(self, ws: WebSocket) -> None:
        async with self._lock:
            self._clients.discard(ws)

    async def broadcast(self, message: dict[str, Any]) -> None:
        async with self._lock:
            clients = list(self._clients)
        dead: list[WebSocket] = []
        for ws in clients:
            try:
                await ws.send_json(message)
            except Exception:  # noqa: BLE001
                dead.append(ws)
        if dead:
            async with self._lock:
                for ws in dead:
                    self._clients.discard(ws)

    @property
    def client_count(self) -> int:
        return len(self._clients)


hub = Hub()
_pipeline: Pipeline | None = None
_loop: asyncio.AbstractEventLoop | None = None


# --- request models --------------------------------------------------------


class SosRequest(BaseModel):
    zone_id: str = Field(default=DEFAULT_ZONE_ID)
    note: str = ""
    confidence: float = 1.0


class StatusUpdate(BaseModel):
    status: str
    actor: str = "guard"
    role: str = "security"


class DetectionRequest(BaseModel):
    type: str = Field(description="fall | crowd")
    zone_id: str | None = None
    confidence: float = 0.8
    held_seconds: float = 5.0
    detail: str = ""


# --- scoring + persistence -------------------------------------------------


def _zone_history_risk(zone_id: str) -> float:
    count = db.zone_history(zone_id, days=30)
    busiest = db.max_zone_history(days=30)
    return normalized_history_risk(zone_id, count, busiest)


def _severity_label(priority: float) -> str:
    for threshold, name in config.SEVERITY_BANDS:
        if priority >= threshold:
            return name
    return "low"


def _create_incident(
    *,
    type_: str,
    zone_id: str,
    source: str,
    sos: bool,
    video_event: str | None,
    held_seconds: float,
    confidence: float,
    note: str,
    crowd_level: float = 0.0,
    ts: str | None = None,
) -> Incident:
    """Score, persist, and return one incident. The single fusion point."""
    ts = ts or db.now_iso()
    hour = datetime.fromisoformat(ts).astimezone().hour

    # Corroboration: an SOS is reinforced by a *different-source* event in the
    # same zone inside the window; a video event is reinforced by a recent SOS.
    corroborating = False
    near = db.incidents_near(ts, zone_id, config.CORROBORATION_WINDOW_S)
    if sos:
        corroborating = any(
            i.source in ("video", "sos+video") for i in near
        ) or db.recent_video_event_near(zone_id, config.CORROBORATION_WINDOW_S)
    elif video_event:
        corroborating = any(i.source in ("sos", "sos+video") for i in near)

    assessment = risk.assess(
        risk.RiskInputs(
            sos=sos,
            video_event=video_event,
            held_seconds=held_seconds,
            hour=hour,
            zone_history_risk=_zone_history_risk(zone_id),
            crowd_level=crowd_level,
            corroborating=corroborating,
        )
    )

    resolved_source = source
    if corroborating and source in ("sos", "video"):
        resolved_source = "sos+video"

    incident = Incident(
        id=None,
        ts=ts,
        type=type_,
        zone_id=zone_id,
        source=resolved_source,  # type: ignore[arg-type]
        severity=assessment.severity,
        priority=assessment.priority,
        status="new",
        note=note,
        confidence=confidence,
        signals=assessment.signals,
    )
    db.insert_incident(incident, assessment.signals)
    db.log_action(
        actor="system",
        role="system",
        action="incident.create",
        target=str(incident.id),
        detail=f"{type_} in {zone_id} priority={assessment.priority:.0f}",
    )
    return incident


def _broadcast_incident(incident: Incident) -> None:
    """Schedule a push to dashboards from a possibly non-async context."""
    payload = {
        "kind": "incident",
        "incident": incident.as_dict(),
        "stats": _pipeline.stats.as_dict() if _pipeline else {},
    }
    _schedule(hub.broadcast(payload))


def _schedule(coro) -> None:  # noqa: ANN001
    if _loop is None:
        return
    try:
        asyncio.run_coroutine_threadsafe(coro, _loop)
    except RuntimeError:
        log.debug("no event loop to schedule broadcast on")


def _on_video_event(event: VideoEvent) -> None:
    """Pipeline sink: a confirmed fall/crowd becomes an incident."""
    db.insert_raw_event(
        ts=event.ts_iso,
        camera_id="cam-01",
        type_=event.kind,
        zone_id=event.zone_id,
        confidence=event.confidence,
        held_seconds=event.held_seconds,
        meta={"detail": event.detail, "track_id": event.track_id},
    )
    incident = _create_incident(
        type_=f"video_{event.kind}",
        zone_id=event.zone_id if event.zone_id in ZONES_BY_ID else DEFAULT_ZONE_ID,
        source="video",
        sos=False,
        video_event=event.kind,
        held_seconds=event.held_seconds,
        confidence=event.confidence,
        note=event.detail,
        crowd_level=event.confidence if event.kind == "crowd" else 0.0,
        ts=event.ts_iso,
    )
    _broadcast_incident(incident)


# --- app lifecycle ---------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):  # noqa: ANN201
    global _pipeline, _loop
    db.init_db()
    if not db.list_zones():
        db.upsert_zones(CAMPUS_ZONES)
    _loop = asyncio.get_running_loop()
    _pipeline = Pipeline(source=None, sink=_on_video_event)
    _pipeline.start()
    log.info("campus safety system up — detector backend: %s", active_backend())
    try:
        yield
    finally:
        if _pipeline:
            _pipeline.stop()


app = FastAPI(title="Smart Campus Safety System", version="0.1.0-mvp", lifespan=lifespan)


# --- student SOS -----------------------------------------------------------


@app.post("/api/sos")
def raise_sos(req: SosRequest, x_role: str | None = Header(default=None)) -> dict[str, Any]:
    role = audit.role_from_header(x_role)
    audit.require(role, "sos.raise")
    if req.zone_id not in ZONES_BY_ID:
        raise HTTPException(status_code=400, detail=f"unknown zone {req.zone_id!r}")
    incident = _create_incident(
        type_="sos",
        zone_id=req.zone_id,
        source="sos",
        sos=True,
        video_event=None,
        held_seconds=0.0,
        confidence=req.confidence,
        note=req.note or "Student SOS",
    )
    audit.log_action(role, role, "sos.raise", str(incident.id), req.zone_id)
    _broadcast_incident(incident)
    return incident.as_dict()


# --- external detection ingest (also drives the scripted demo) -------------


@app.post("/api/detect/{camera_id}")
def ingest_detection(
    camera_id: str,
    req: DetectionRequest,
    x_role: str | None = Header(default=None),
) -> dict[str, Any]:
    role = audit.role_from_header(x_role)
    if req.type not in ("fall", "crowd"):
        # Violence is deliberately out of scope — see the deck's discipline note.
        raise HTTPException(
            status_code=400,
            detail="supported types: 'fall', 'crowd' (violence is out of scope)",
        )
    zone_id = req.zone_id or DEFAULT_ZONE_ID
    if zone_id not in ZONES_BY_ID:
        raise HTTPException(status_code=400, detail=f"unknown zone {zone_id!r}")
    incident = _create_incident(
        type_=f"video_{req.type}",
        zone_id=zone_id,
        source="video",
        sos=False,
        video_event=req.type,
        held_seconds=req.held_seconds,
        confidence=req.confidence,
        note=req.detail or f"{req.type} detected on {camera_id}",
        crowd_level=req.confidence if req.type == "crowd" else 0.0,
    )
    db.log_action(role, role, "detect.ingest", camera_id, f"{req.type} @ {zone_id}")
    _broadcast_incident(incident)
    return incident.as_dict()


# --- incidents -------------------------------------------------------------


@app.get("/api/incidents")
def get_incidents(
    status: str | None = Query(default=None),
    zone_id: str | None = Query(default=None),
    since_hours: int | None = Query(default=None),
    limit: int = Query(default=200, le=1000),
) -> list[dict[str, Any]]:
    if status and status not in STATUS_ORDER:
        raise HTTPException(status_code=400, detail=f"invalid status {status!r}")
    return [
        i.as_dict()
        for i in db.list_incidents(
            status=status, zone_id=zone_id, since_hours=since_hours, limit=limit
        )
    ]


@app.get("/api/incidents/{incident_id}")
def get_incident(incident_id: int) -> dict[str, Any]:
    incident = db.get_incident(incident_id)
    if incident is None:
        raise HTTPException(status_code=404, detail="incident not found")
    return incident.as_dict()


@app.get("/api/incidents/{incident_id}/explain")
def explain_incident(incident_id: int) -> dict[str, Any]:
    """The 'why' breakdown — the per-signal contribution bars on the dashboard."""
    incident = db.get_incident(incident_id)
    if incident is None:
        raise HTTPException(status_code=404, detail="incident not found")
    return {
        "incident_id": incident.id,
        "priority": round(incident.priority, 2),
        "severity": incident.severity,
        "zone_id": incident.zone_id,
        "signals": [s.as_dict() for s in incident.signals],
        "explanation": _narrate(incident),
    }


def _narrate(incident: Incident) -> str:
    """Plain-English reason, so a guard (and a judge) reads the score instantly."""
    parts = [
        f"{s.label} added {s.contribution:.0f}"
        for s in incident.signals
        if s.contribution > 0
    ]
    if not parts:
        return "No strong signals; recorded for the log."
    return f"Priority {incident.priority:.0f} ({incident.severity}) because " + "; ".join(parts) + "."


@app.patch("/api/incidents/{incident_id}")
def update_incident(
    incident_id: int,
    update: StatusUpdate,
    x_role: str | None = Header(default=None),
) -> dict[str, Any]:
    role = audit.role_from_header(x_role)
    if update.status not in STATUS_ORDER:
        raise HTTPException(status_code=400, detail=f"invalid status {update.status!r}")
    audit.require(role, "incident.update")
    if db.get_incident(incident_id) is None:
        raise HTTPException(status_code=404, detail="incident not found")
    updated = db.update_status(incident_id, update.status)  # type: ignore[arg-type]
    db.log_action(role, role, "incident.update", str(incident_id), update.status)
    assert updated is not None
    _broadcast_incident(updated)
    return updated.as_dict()


# --- zones, weights, audit, stats ------------------------------------------


@app.get("/api/zones")
def get_zones() -> list[dict[str, Any]]:
    busiest = db.max_zone_history(days=30)
    out: list[dict[str, Any]] = []
    for zone in db.list_zones() or CAMPUS_ZONES:
        count = db.zone_history(zone.id, days=30)
        risk_val = normalized_history_risk(zone.id, count, busiest)
        cx, cy = zone_centroid(zone)
        item = zone.as_dict()
        item.update(
            {
                "incidents_30d": count,
                "zone_risk": round(risk_val, 3),
                "centroid": [round(cx, 4), round(cy, 4)],
                "open_incidents": len(
                    [i for i in db.list_incidents(zone_id=zone.id, with_signals=False)
                     if i.status != "resolved"]
                ),
            }
        )
        out.append(item)
    return out


@app.get("/api/weights")
def get_weights() -> dict[str, Any]:
    """Publish the risk weights — transparency is the differentiator."""
    payload = risk.published_weights()
    payload["detector"] = {
        "active_backend": active_backend(),
        "licenses": config.DETECTOR_LICENSES,
        "privacy": (
            "No face recognition. Person detection returns an anonymous box; we "
            "store event metadata, not footage."
        ),
    }
    return payload


@app.get("/api/stats")
def get_stats() -> dict[str, Any]:
    return {
        "pipeline": _pipeline.stats.as_dict() if _pipeline else {},
        "dashboard_clients": hub.client_count,
        "detector_backend": active_backend(),
    }


@app.get("/api/audit")
def get_audit(
    limit: int = Query(default=200, le=1000),
    x_role: str | None = Header(default=None),
) -> list[dict[str, Any]]:
    role = audit.role_from_header(x_role)
    audit.require(role, "audit.read")
    audit.log_action(role, role, "audit.read", "", f"limit={limit}")
    return db.list_audit(limit=limit)


@app.get("/api/events/raw")
def get_raw_events(limit: int = Query(default=100, le=1000)) -> list[dict[str, Any]]:
    return db.list_raw_events(limit=limit)


# --- live dashboard + demo controls ----------------------------------------


@app.post("/api/demo/simulate")
def simulate_scenario(
    scenario: str = Query(default="fall_then_sos"),
    zone_id: str = Query(default="sports_ground"),
) -> dict[str, Any]:
    """Hands-free demo driver: fires a scripted sequence of real incidents.

    Useful as a stage backup — it drives the same code path the detector does.
    """
    if scenario not in ("fall_then_sos", "crowd", "sos_only"):
        raise HTTPException(status_code=400, detail="unknown scenario")
    created: list[dict[str, Any]] = []

    if scenario in ("fall_then_sos", "sos_only"):
        fall = _create_incident(
            type_="video_fall",
            zone_id=zone_id,
            source="video",
            sos=False,
            video_event="fall",
            held_seconds=config.FALL_HOLD_S + 1,
            confidence=0.88,
            note="Simulated: person down (scripted replay)",
        )
        _broadcast_incident(fall)
        created.append(fall.as_dict())

    if scenario == "fall_then_sos":
        # Same zone, inside the corroboration window -> the verification bonus.
        sos = _create_incident(
            type_="sos",
            zone_id=zone_id,
            source="sos",
            sos=True,
            video_event=None,
            held_seconds=0.0,
            confidence=1.0,
            note="Simulated: student SOS corroborating the video event",
        )
        _broadcast_incident(sos)
        created.append(sos.as_dict())

    if scenario == "crowd":
        crowd = _create_incident(
            type_="video_crowd",
            zone_id="canteen",
            source="video",
            sos=False,
            video_event="crowd",
            held_seconds=config.CROWD_HOLD_S + 2,
            confidence=0.75,
            note="Simulated: crowd above zone capacity",
            crowd_level=0.8,
        )
        _broadcast_incident(crowd)
        created.append(crowd.as_dict())

    return {"scenario": scenario, "created": created}


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket) -> None:
    await hub.connect(ws)
    try:
        await ws.send_json(
            {
                "kind": "hello",
                "stats": _pipeline.stats.as_dict() if _pipeline else {},
                "zones": [z.as_dict() for z in db.list_zones() or CAMPUS_ZONES],
            }
        )
        while True:
            # Client messages are only keepalives; the server pushes events.
            await ws.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        await hub.disconnect(ws)


# --- static frontend -------------------------------------------------------


@app.get("/")
def index() -> FileResponse:
    return FileResponse(config.FRONTEND_DIR / "index.html")


@app.get("/dashboard")
def dashboard() -> FileResponse:
    return FileResponse(config.FRONTEND_DIR / "dashboard.html")


@app.get("/admin")
def admin() -> FileResponse:
    return FileResponse(config.FRONTEND_DIR / "admin.html")


app.mount("/static", StaticFiles(directory=config.FRONTEND_DIR), name="static")


@app.exception_handler(audit.AccessDenied)
async def access_denied_handler(_request, exc: audit.AccessDenied) -> JSONResponse:  # noqa: ANN001
    return JSONResponse(status_code=403, content={"detail": str(exc)})
