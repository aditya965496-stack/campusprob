"""Domain types shared across the backend."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

IncidentSource = Literal["sos", "video", "sos+video", "system"]
IncidentStatus = Literal["new", "acknowledged", "dispatched", "resolved"]
Severity = Literal["low", "medium", "high", "critical"]

STATUS_ORDER: tuple[IncidentStatus, ...] = (
    "new",
    "acknowledged",
    "dispatched",
    "resolved",
)


@dataclass(slots=True)
class Signal:
    """One contributing factor in a risk score.

    Published individually so the dashboard can render contribution bars and a
    judge can add the numbers up by hand.
    """

    signal: str          # e.g. "sos", "video_fall", "corroboration"
    label: str           # human-readable, for the UI
    weight: float        # the published weight (0 when the signal is inactive)
    value: float         # the signal's own input value in [0, 1] (or 0/1 flag)
    contribution: float  # weight * value — what it actually added to priority

    def as_dict(self) -> dict[str, Any]:
        return {
            "signal": self.signal,
            "label": self.label,
            "weight": round(self.weight, 2),
            "value": round(self.value, 3),
            "contribution": round(self.contribution, 2),
        }


@dataclass(slots=True)
class RiskAssessment:
    """The output of the risk engine — score, band, and the full breakdown."""

    priority: float
    severity: Severity
    signals: list[Signal] = field(default_factory=list)
    overrides: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "priority": round(self.priority, 2),
            "severity": self.severity,
            "signals": [s.as_dict() for s in self.signals],
            "overrides": self.overrides,
        }


@dataclass(slots=True)
class Incident:
    id: int | None
    ts: str
    type: str
    zone_id: str
    source: IncidentSource
    severity: Severity
    priority: float
    status: IncidentStatus
    note: str = ""
    confidence: float = 1.0
    acknowledged_at: str | None = None
    resolved_at: str | None = None
    signals: list[Signal] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "ts": self.ts,
            "type": self.type,
            "zone_id": self.zone_id,
            "source": self.source,
            "severity": self.severity,
            "priority": round(self.priority, 2),
            "status": self.status,
            "note": self.note,
            "confidence": round(self.confidence, 3),
            "acknowledged_at": self.acknowledged_at,
            "resolved_at": self.resolved_at,
            "signals": [s.as_dict() for s in self.signals],
        }


@dataclass(slots=True)
class Zone:
    id: str
    name: str
    polygon: list[tuple[float, float]]  # normalised (x, y) in [0, 1] map space
    capacity: int
    risk_prior: float = 0.15

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "polygon": [[round(x, 4), round(y, 4)] for x, y in self.polygon],
            "capacity": self.capacity,
            "risk_prior": round(self.risk_prior, 3),
        }
