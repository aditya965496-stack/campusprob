"""Detector interface.

The single abstraction that keeps the vision layer swappable: Ultralytics YOLO,
OpenCV-DNN, and the scripted/synthetic sources all return `list[Box]`, so nothing
downstream knows or cares which model produced them. This is also the licence
escape hatch — dropping YOLO for YOLOX means adding one module, not a refactor.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np


@dataclass(slots=True)
class Box:
    """One detected person.

    `bbox` is (x, y, w, h) in pixels. There is deliberately NO identity field:
    `track_id` is a *transient* association id for the fall timer, not a
    recognition of who the person is. Person detection returns an anonymous
    box, nothing more — this is the privacy claim on Slide 8, held in code.
    """

    bbox: tuple[float, float, float, float]
    conf: float
    track_id: int | None = None


@runtime_checkable
class Detector(Protocol):
    """Anything that can turn a frame into person boxes."""

    name: str

    def detect(self, frame: np.ndarray) -> list[Box]:
        """Return person detections for one BGR frame."""
        ...

    @property
    def available(self) -> bool:
        """False if the backend failed to initialise and should be skipped."""
        ...
