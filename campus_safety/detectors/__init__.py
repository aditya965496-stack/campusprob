"""Detector backends and graceful selection.

`get_detector()` tries backends in order of fidelity and NEVER raises — if the
heavy ML stack is missing it drops to a lighter one, and if there is no camera
at all it falls back to a synthetic source so the whole pipeline still runs
end to end on stage. Which backend won is logged and surfaced in the UI.
"""

from __future__ import annotations

import logging

from .. import config
from .base import Box, Detector

log = logging.getLogger("campus_safety.detectors")

__all__ = ["Box", "Detector", "get_detector", "active_backend"]


class _NoDetector:
    """Explicit null object, used only if synthetic generation is disabled."""

    name = "none"
    available = False

    def detect(self, frame):  # noqa: ANN001, ANN201
        return []


# Cached after first selection so we do not re-import torch on every call.
_ACTIVE: Detector | None = None


def _try(module_name: str, factory) -> Detector | None:  # noqa: ANN001
    try:
        detector = factory()
    except Exception as exc:  # noqa: BLE001 - a failed backend must never crash
        log.info("detector backend %s unavailable: %s", module_name, exc)
        return None
    if getattr(detector, "available", False):
        log.info("detector backend selected: %s", module_name)
        return detector
    log.info("detector backend %s initialised but reported unavailable", module_name)
    return None


def _select() -> Detector:
    wanted = config.DETECTOR_BACKEND

    # An explicit request that is not "auto" pins the backend, but we still fall
    # through to synthetic if it is unavailable — the demo must not die.
    candidates: list[tuple[str, object]] = []
    if wanted in ("auto", "ultralytics"):
        from . import yolo

        candidates.append(("ultralytics", yolo.build))
    if wanted in ("auto", "opencv-dnn"):
        from . import dnn

        candidates.append(("opencv-dnn", dnn.build))
    if wanted in ("auto", "scripted"):
        from . import scripted

        candidates.append(("scripted", scripted.build))

    for name, factory in candidates:
        detector = _try(name, factory)
        if detector is not None:
            return detector

    if config.ALLOW_SYNTHETIC:
        from . import scripted

        log.info("falling back to synthetic person source")
        return scripted.build_synthetic()

    log.warning("no detector backend available and synthetic generation is off")
    return _NoDetector()


def get_detector() -> Detector:
    """Return the active detector, selecting one on first call."""
    global _ACTIVE
    if _ACTIVE is None:
        _ACTIVE = _select()
    return _ACTIVE


def active_backend() -> str:
    """Name of the live backend — shown as an on-camera badge in the dashboard."""
    return getattr(get_detector(), "name", "none")


def reset() -> None:
    """Force re-selection. Used by tests and by a manual backend switch."""
    global _ACTIVE
    _ACTIVE = None
