"""End-to-end API tests through FastAPI's TestClient.

Covers the paths a live demo depends on: raising an SOS, scoring it, explaining
it, the status workflow, the verification bonus when an SOS meets a video event,
and the role-based audit guard.
"""

import pytest
from fastapi.testclient import TestClient

from campus_safety import config
from campus_safety.api import app


@pytest.fixture
def client(seeded):
    with TestClient(app) as c:
        yield c


def test_sos_creates_scored_incident(client):
    r = client.post("/api/sos", json={"zone_id": "parking", "note": "test"})
    assert r.status_code == 200
    body = r.json()
    assert body["source"] == "sos"
    assert body["zone_id"] == "parking"
    # SOS override floor must apply.
    assert body["priority"] >= config.OVERRIDES["sos_min_priority"]
    assert body["signals"], "signal breakdown must be present"


def test_unknown_zone_rejected(client):
    r = client.post("/api/sos", json={"zone_id": "atlantis"})
    assert r.status_code == 400


def test_explain_returns_contributions(client):
    inc = client.post("/api/sos", json={"zone_id": "library"}).json()
    r = client.get(f"/api/incidents/{inc['id']}/explain")
    assert r.status_code == 200
    data = r.json()
    assert data["incident_id"] == inc["id"]
    assert any(s["contribution"] > 0 for s in data["signals"])
    assert "because" in data["explanation"] or "recorded" in data["explanation"]


def test_status_workflow(client):
    inc = client.post("/api/sos", json={"zone_id": "canteen"}).json()
    r = client.patch(
        f"/api/incidents/{inc['id']}",
        json={"status": "acknowledged"},
        headers={"X-Role": "security"},
    )
    assert r.status_code == 200
    assert r.json()["status"] == "acknowledged"
    assert r.json()["acknowledged_at"] is not None


def test_invalid_status_rejected(client):
    inc = client.post("/api/sos", json={"zone_id": "canteen"}).json()
    r = client.patch(
        f"/api/incidents/{inc['id']}",
        json={"status": "exploded"},
        headers={"X-Role": "security"},
    )
    assert r.status_code == 400


def test_violence_detection_is_rejected(client):
    # Violence is explicitly out of scope; the API must refuse it.
    r = client.post("/api/detect/cam-01", json={"type": "violence", "zone_id": "parking"})
    assert r.status_code == 400
    assert "out of scope" in r.json()["detail"]


def test_verification_bonus_on_fusion(client):
    # A video fall then a co-located SOS inside the window should corroborate.
    client.post(
        "/api/detect/cam-01",
        json={"type": "fall", "zone_id": "sports_ground", "held_seconds": 6},
    )
    sos = client.post("/api/sos", json={"zone_id": "sports_ground"}).json()
    assert sos["source"] == "sos+video"
    assert any(s["signal"] == "corroboration" and s["contribution"] > 0 for s in sos["signals"])


def test_audit_requires_admin_role(client):
    # Security role may not read the audit log.
    assert client.get("/api/audit", headers={"X-Role": "security"}).status_code == 403
    # Admin may.
    assert client.get("/api/audit", headers={"X-Role": "admin"}).status_code == 200


def test_weights_endpoint_publishes_formula(client):
    r = client.get("/api/weights")
    assert r.status_code == 200
    body = r.json()
    assert body["weights"] == config.RISK_WEIGHTS
    assert body["detector"]["privacy"].lower().startswith("no face")


def test_simulate_scenario_creates_incidents(client):
    r = client.post("/api/demo/simulate?scenario=fall_then_sos&zone_id=sports_ground")
    assert r.status_code == 200
    created = r.json()["created"]
    assert len(created) == 2
    # The SOS in the pair should be corroborated by the fall.
    assert any(i["source"] == "sos+video" for i in created)
