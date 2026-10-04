"""SQLite persistence: schema, queries, and the audit trail.

Portable by design — the SQL below is plain enough to move to PostgreSQL by
swapping the connection helper (see config.DB_URL). We store event *metadata*,
never video frames; raw footage never reaches this layer.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from typing import Any, Iterable, Iterator

from . import config
from .models import Incident, IncidentStatus, Signal, Zone

SCHEMA = """
CREATE TABLE IF NOT EXISTS zones (
    id          TEXT PRIMARY KEY,
    name        TEXT NOT NULL,
    polygon     TEXT NOT NULL,      -- JSON [[x, y], ...] normalised to [0,1]
    capacity    INTEGER NOT NULL DEFAULT 10,
    risk_prior  REAL NOT NULL DEFAULT 0.15
);

CREATE TABLE IF NOT EXISTS incidents (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    ts              TEXT NOT NULL,
    type            TEXT NOT NULL,
    zone_id         TEXT NOT NULL,
    source          TEXT NOT NULL,
    severity        TEXT NOT NULL,
    priority        REAL NOT NULL,
    status          TEXT NOT NULL DEFAULT 'new',
    note            TEXT NOT NULL DEFAULT '',
    confidence      REAL NOT NULL DEFAULT 1.0,
    acknowledged_at TEXT,
    resolved_at     TEXT
);
CREATE INDEX IF NOT EXISTS idx_incidents_ts ON incidents(ts DESC);
CREATE INDEX IF NOT EXISTS idx_incidents_zone ON incidents(zone_id);

-- Per-signal contribution frozen at scoring time, so an old incident can still
-- be explained even after the weights change.
CREATE TABLE IF NOT EXISTS incident_signals (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    incident_id   INTEGER NOT NULL REFERENCES incidents(id) ON DELETE CASCADE,
    signal        TEXT NOT NULL,
    label         TEXT NOT NULL,
    weight        REAL NOT NULL,
    value         REAL NOT NULL,
    contribution  REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_signals_incident ON incident_signals(incident_id);

-- Detection output before fusion. Kept separate from incidents so the demo can
-- show "raw events" vs "scored incidents".
CREATE TABLE IF NOT EXISTS events_raw (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    ts           TEXT NOT NULL,
    camera_id    TEXT NOT NULL,
    type         TEXT NOT NULL,
    zone_id      TEXT,
    confidence   REAL NOT NULL DEFAULT 0.0,
    held_seconds REAL NOT NULL DEFAULT 0.0,
    meta         TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_events_ts ON events_raw(ts DESC);

-- Role-based access trail: every status change and every read of the log.
CREATE TABLE IF NOT EXISTS audit_log (
    id      INTEGER PRIMARY KEY AUTOINCREMENT,
    ts      TEXT NOT NULL,
    actor   TEXT NOT NULL,
    role    TEXT NOT NULL,
    action  TEXT NOT NULL,
    target  TEXT NOT NULL DEFAULT '',
    detail  TEXT NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_audit_ts ON audit_log(ts DESC);
"""


def now_iso() -> str:
    """UTC timestamp, second precision — the single time format in the DB."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _connect() -> sqlite3.Connection:
    # PostgreSQL note: this is the one function that would change.
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(config.DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    return conn


@contextmanager
def connection() -> Iterator[sqlite3.Connection]:
    conn = _connect()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db() -> None:
    """Create the schema if it does not exist. Safe to call on every boot."""
    with connection() as conn:
        conn.executescript(SCHEMA)


def reset_db() -> None:
    """Drop everything and recreate. Used by the seeder."""
    with connection() as conn:
        conn.executescript(
            "DROP TABLE IF EXISTS audit_log; DROP TABLE IF EXISTS events_raw;"
            "DROP TABLE IF EXISTS incident_signals; DROP TABLE IF EXISTS incidents;"
            "DROP TABLE IF EXISTS zones;"
        )
        conn.executescript(SCHEMA)


# --- Zones -----------------------------------------------------------------


def upsert_zones(zones: Iterable[Zone]) -> None:
    with connection() as conn:
        conn.executemany(
            "INSERT INTO zones (id, name, polygon, capacity, risk_prior)"
            " VALUES (?, ?, ?, ?, ?)"
            " ON CONFLICT(id) DO UPDATE SET name=excluded.name,"
            " polygon=excluded.polygon, capacity=excluded.capacity,"
            " risk_prior=excluded.risk_prior",
            [
                (z.id, z.name, json.dumps(z.polygon), z.capacity, z.risk_prior)
                for z in zones
            ],
        )


def list_zones() -> list[Zone]:
    with connection() as conn:
        rows = conn.execute("SELECT * FROM zones ORDER BY id").fetchall()
    return [
        Zone(
            id=r["id"],
            name=r["name"],
            polygon=[tuple(p) for p in json.loads(r["polygon"])],
            capacity=r["capacity"],
            risk_prior=r["risk_prior"],
        )
        for r in rows
    ]


# --- Incidents -------------------------------------------------------------


def insert_incident(incident: Incident, signals: list[Signal] | None = None) -> int:
    """Persist an incident plus its frozen signal breakdown. Returns the id."""
    with connection() as conn:
        cur = conn.execute(
            "INSERT INTO incidents (ts, type, zone_id, source, severity, priority,"
            " status, note, confidence, acknowledged_at, resolved_at)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                incident.ts,
                incident.type,
                incident.zone_id,
                incident.source,
                incident.severity,
                incident.priority,
                incident.status,
                incident.note,
                incident.confidence,
                incident.acknowledged_at,
                incident.resolved_at,
            ),
        )
        incident_id = int(cur.lastrowid)
        payload = signals if signals is not None else incident.signals
        if payload:
            conn.executemany(
                "INSERT INTO incident_signals (incident_id, signal, label, weight,"
                " value, contribution) VALUES (?, ?, ?, ?, ?, ?)",
                [
                    (
                        incident_id,
                        s.signal,
                        s.label,
                        s.weight,
                        s.value,
                        s.contribution,
                    )
                    for s in payload
                ],
            )
    incident.id = incident_id
    return incident_id


def _row_to_incident(row: sqlite3.Row, signals: list[Signal]) -> Incident:
    return Incident(
        id=row["id"],
        ts=row["ts"],
        type=row["type"],
        zone_id=row["zone_id"],
        source=row["source"],
        severity=row["severity"],
        priority=row["priority"],
        status=row["status"],
        note=row["note"],
        confidence=row["confidence"],
        acknowledged_at=row["acknowledged_at"],
        resolved_at=row["resolved_at"],
        signals=signals,
    )


def _signals_for(conn: sqlite3.Connection, incident_id: int) -> list[Signal]:
    rows = conn.execute(
        "SELECT signal, label, weight, value, contribution"
        " FROM incident_signals WHERE incident_id = ?"
        " ORDER BY contribution DESC",
        (incident_id,),
    ).fetchall()
    return [
        Signal(
            signal=r["signal"],
            label=r["label"],
            weight=r["weight"],
            value=r["value"],
            contribution=r["contribution"],
        )
        for r in rows
    ]


def list_incidents(
    status: str | None = None,
    zone_id: str | None = None,
    since_hours: int | None = None,
    limit: int = 200,
    with_signals: bool = True,
) -> list[Incident]:
    clauses: list[str] = []
    params: list[Any] = []
    if status:
        clauses.append("status = ?")
        params.append(status)
    if zone_id:
        clauses.append("zone_id = ?")
        params.append(zone_id)
    if since_hours is not None:
        cutoff = (datetime.now(timezone.utc) - timedelta(hours=since_hours)).replace(
            microsecond=0
        ).isoformat()
        clauses.append("ts >= ?")
        params.append(cutoff)
    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    params.append(limit)
    with connection() as conn:
        rows = conn.execute(
            f"SELECT * FROM incidents {where} ORDER BY ts DESC LIMIT ?", params
        ).fetchall()
        out = [
            _row_to_incident(r, _signals_for(conn, r["id"]) if with_signals else [])
            for r in rows
        ]
    return out


def get_incident(incident_id: int) -> Incident | None:
    with connection() as conn:
        row = conn.execute(
            "SELECT * FROM incidents WHERE id = ?", (incident_id,)
        ).fetchone()
        if row is None:
            return None
        return _row_to_incident(row, _signals_for(conn, incident_id))


def update_status(incident_id: int, status: IncidentStatus) -> Incident | None:
    ts = now_iso()
    with connection() as conn:
        sets = "status = ?"
        params: list[Any] = [status]
        if status == "acknowledged":
            sets += ", acknowledged_at = COALESCE(acknowledged_at, ?)"
            params.append(ts)
        if status == "resolved":
            sets += ", resolved_at = ?"
            params.append(ts)
        params.append(incident_id)
        conn.execute(f"UPDATE incidents SET {sets} WHERE id = ?", params)
    return get_incident(incident_id)


def zone_history(zone_id: str, days: int = 30) -> int:
    """Incident count for a zone over the trailing window — feeds zone risk."""
    cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).replace(
        microsecond=0
    ).isoformat()
    with connection() as conn:
        row = conn.execute(
            "SELECT COUNT(*) AS n FROM incidents WHERE zone_id = ? AND ts >= ?",
            (zone_id, cutoff),
        ).fetchone()
    return int(row["n"]) if row else 0


def max_zone_history(days: int = 30) -> int:
    """Busiest zone's count, so zone risk can be normalised into [0, 1]."""
    cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).replace(
        microsecond=0
    ).isoformat()
    with connection() as conn:
        row = conn.execute(
            "SELECT COALESCE(MAX(c), 0) AS m FROM ("
            " SELECT COUNT(*) AS c FROM incidents WHERE ts >= ? GROUP BY zone_id)",
            (cutoff,),
        ).fetchone()
    return int(row["m"]) if row else 0


def incidents_near(ts_iso: str, zone_id: str, window_s: float) -> list[Incident]:
    """Incidents in the same zone within the corroboration window.

    This is the fusion query behind the verification bonus: a student SOS and a
    nearby video event reinforcing each other.
    """
    try:
        anchor = datetime.fromisoformat(ts_iso)
    except ValueError:
        return []
    lo = (anchor - timedelta(seconds=window_s)).isoformat()
    hi = (anchor + timedelta(seconds=window_s)).isoformat()
    with connection() as conn:
        rows = conn.execute(
            "SELECT * FROM incidents WHERE zone_id = ? AND ts BETWEEN ? AND ?"
            " ORDER BY ts DESC",
            (zone_id, lo, hi),
        ).fetchall()
    return [_row_to_incident(r, []) for r in rows]


def recent_video_event_near(zone_id: str, window_s: float) -> bool:
    """True if a video-sourced incident landed in this zone recently."""
    cutoff = (datetime.now(timezone.utc) - timedelta(seconds=window_s)).replace(
        microsecond=0
    ).isoformat()
    with connection() as conn:
        row = conn.execute(
            "SELECT 1 FROM incidents WHERE zone_id = ? AND ts >= ?"
            " AND source IN ('video', 'sos+video') LIMIT 1",
            (zone_id, cutoff),
        ).fetchone()
    return row is not None


# --- Raw events ------------------------------------------------------------


def insert_raw_event(
    ts: str,
    camera_id: str,
    type_: str,
    zone_id: str | None,
    confidence: float,
    held_seconds: float,
    meta: dict[str, Any] | None = None,
) -> int:
    with connection() as conn:
        cur = conn.execute(
            "INSERT INTO events_raw (ts, camera_id, type, zone_id, confidence,"
            " held_seconds, meta) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                ts,
                camera_id,
                type_,
                zone_id,
                confidence,
                held_seconds,
                json.dumps(meta or {}),
            ),
        )
        return int(cur.lastrowid)


def list_raw_events(limit: int = 100) -> list[dict[str, Any]]:
    with connection() as conn:
        rows = conn.execute(
            "SELECT * FROM events_raw ORDER BY ts DESC LIMIT ?", (limit,)
        ).fetchall()
    return [dict(r) for r in rows]


# --- Audit -----------------------------------------------------------------


def log_action(
    actor: str, role: str, action: str, target: str = "", detail: str = ""
) -> None:
    with connection() as conn:
        conn.execute(
            "INSERT INTO audit_log (ts, actor, role, action, target, detail)"
            " VALUES (?, ?, ?, ?, ?, ?)",
            (now_iso(), actor, role, action, target, detail),
        )


def list_audit(limit: int = 200) -> list[dict[str, Any]]:
    with connection() as conn:
        rows = conn.execute(
            "SELECT * FROM audit_log ORDER BY ts DESC LIMIT ?", (limit,)
        ).fetchall()
    return [dict(r) for r in rows]
