"""Scripted and synthetic detection sources — the demo-safety net.

Two backends, both with zero ML dependencies:

  * ScriptedDetector  - replays a fixed, hand-authored timeline of person boxes
                        so a fall + crowd scenario runs deterministically,
                        every single time, on any machine. This is what the
                        live demo uses on stage.
  * SyntheticDetector - draws a person that walks in, crouches (a fall), then a
                        crowd that builds up, generated procedurally. Used when
                        there is no script and no camera.

Neither recognises anyone; both emit anonymous boxes, exactly like the real
detectors.
"""

from __future__ import annotations

import math
import time

import numpy as np

from .. import config
from .base import Box

W, H = 960, 540  # synthetic frame size, matching the demo canvas


class _Timeline(tuple):
    """A tiny immutable script: (seconds, boxes) pairs, replayed in order."""


def _standing(x: float, y: float, scale: float = 1.0, tid: int = 1) -> Box:
    """A tall person box (h > w) centred near (x, y)."""
    h = 110 * scale
    w = 42 * scale
    return Box(bbox=(x - w / 2, y - h, w, h), conf=0.86, track_id=tid)


def _fallen(x: float, y: float, scale: float = 1.0, tid: int = 1) -> Box:
    """A wide person box (w >= h) — someone on the ground."""
    h = 40 * scale
    w = 120 * scale
    return Box(bbox=(x - w / 2, y - h, w, h), conf=0.82, track_id=tid)


class SyntheticDetector:
    """Procedurally generates a person who walks, falls, then a gathering crowd.

    The script repeats on a loop so an unattended demo keeps producing events.
    """

    name = "synthetic"
    license = "n/a"
    available = True

    def __init__(self, cycle_s: float = 60.0) -> None:
        self._cycle = cycle_s
        self._t0 = time.time()

    def detect(self, frame: np.ndarray) -> list[Box]:
        t = (time.time() - self._t0) % self._cycle
        boxes: list[Box] = []

        # Phase 1 (0-22s): a walker crosses the sports ground, then falls at ~15s.
        if t < 22:
            progress = min(t / 15.0, 1.0)
            x = 320 + progress * 200
            if t < 15:
                boxes.append(_standing(x, H * 0.72, tid=1))
            else:
                # Down and staying down — the persistence timer will fire.
                boxes.append(_fallen(x, H * 0.80, tid=1))

        # Phase 2 (25-45s): crowd builds in the canteen polygon.
        if 25 <= t < 45:
            build = min((t - 25) / 8.0, 1.0)
            n = int(2 + build * 20)
            for i in range(n):
                angle = (i / max(n, 1)) * math.tau
                cx = 720 + 90 * math.cos(angle) + (i % 3) * 12
                cy = H * 0.55 + 40 * math.sin(angle)
                boxes.append(_standing(cx, cy, scale=0.8, tid=100 + i))
        return boxes


class ScriptedDetector:
    """Replays a fixed timeline of boxes, advancing one step per detect() call.

    The timeline is expressed in *frames* (not wall-clock) so a demo is fully
    deterministic regardless of how fast the machine runs: call detect() enough
    times and the fall + crowd events fire, every time, in order.
    """

    name = "scripted"
    license = "n/a"
    available = True

    def __init__(self) -> None:
        # Each entry: list of boxes for one processed frame.
        self._frames: list[list[Box]] = self._build_timeline()
        self._i = 0

    @staticmethod
    def _build_timeline() -> list[list[Box]]:
        frames: list[list[Box]] = []
        # Frames 0-9: person walking upright across the sports ground.
        for k in range(10):
            x = 320 + k * 20
            frames.append([_standing(x, H * 0.72, tid=1)])
        # Frames 10-14: still upright but slowing (builds a "standing" history).
        for k in range(5):
            frames.append([_standing(520 + k * 4, H * 0.72, tid=1)])
        # Frames 15-24: fallen and staying down (hold lasts 10 processed frames,
        # comfortably past FALL_HOLD_S at demo frame rate).
        for k in range(10):
            frames.append([_fallen(540, H * 0.80, tid=1)])
        # Frames 25-29: alone again.
        for k in range(5):
            frames.append([_standing(560 + k * 10, H * 0.72, tid=1)])
        # Frames 30-49: crowd builds in the canteen (well over capacity).
        for k in range(20):
            n = min(4 + k, 22)
            crowd: list[Box] = []
            for i in range(n):
                angle = (i / n) * math.tau
                cx = 720 + 80 * math.cos(angle)
                cy = H * 0.55 + 35 * math.sin(angle)
                crowd.append(_standing(cx, cy, scale=0.8, tid=100 + i))
            frames.append(crowd)
        return frames

    def detect(self, frame: np.ndarray) -> list[Box]:  # noqa: ARG002, ANN201
        if not self._frames:
            return []
        boxes = self._frames[self._i % len(self._frames)]
        self._i += 1
        # Return copies so downstream state cannot mutate the script.
        return [Box(bbox=b.bbox, conf=b.conf, track_id=b.track_id) for b in boxes]


def build() -> ScriptedDetector:
    return ScriptedDetector()


def build_synthetic() -> SyntheticDetector:
    return SyntheticDetector()


def blank_frame() -> np.ndarray:
    """A neutral frame for backends that ignore pixel content."""
    return np.zeros((H, W, 3), dtype=np.uint8)
