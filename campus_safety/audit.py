"""Role-based access and the audit trail.

Kept intentionally small: one helper that records an action, and one guard that
refuses privileged reads to ordinary roles. The dashboard is security-only; the
audit log is admin-only. Every status change is written here — the transparency
claim on Slide 8 has to be true in the code, not just on the slide.
"""

from __future__ import annotations

from typing import Literal

Role = Literal["student", "security", "admin"]

ROLE_RANK: dict[str, int] = {"student": 0, "security": 1, "admin": 2}

# What each role is allowed to do. Read as: role -> permitted actions.
PERMISSIONS: dict[str, set[str]] = {
    "student": {"sos.raise", "incident.read.own"},
    "security": {"sos.raise", "incident.read", "incident.update", "dashboard.view"},
    "admin": {
        "sos.raise",
        "incident.read",
        "incident.update",
        "dashboard.view",
        "audit.read",
        "zones.manage",
        "weights.read",
    },
}


class AccessDenied(Exception):
    """Raised when a role attempts an action it does not hold."""


def can(role: str, action: str) -> bool:
    return action in PERMISSIONS.get(role, set())


def require(role: str, action: str) -> None:
    if not can(role, action):
        raise AccessDenied(f"role {role!r} may not {action!r}")


def role_from_header(value: str | None) -> str:
    """Map a request header onto a known role. Unknown values downgrade to
    student rather than failing open."""
    if value in PERMISSIONS:
        return value
    return "student"


def log_action(actor: str, role: str, action: str, target: str = "", detail: str = "") -> None:
    from . import db

    db.log_action(actor=actor, role=role, action=action, target=target, detail=detail)

