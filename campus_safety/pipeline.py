"""Capture + inference pipeline.

Design follows the deck's Slide 7: detection is decoupled from capture by a
bounded queue so inference never lags real time, and every confirmed event is
debounced before it fires. One worker handles one camera group — the honest
CPU-laptop limit — with a documented path to one worker per camera.

The pipeline is deliberately transport-agnostic: it calls `emit(event)` for every
confirmed event and lets the API layer decide how to score and broadcast it.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from queue import Empty, Full, Queue

import numpy as np

from . import config
from .detectors import active_backend, get_detector
from .detectors.scripted import blank_frame
from .events import CrowdCounter, FallDetector, VideoEvent

log = logging.getLogger("campus_safety.pipeline")

EventSink = Callable[[VideoEvent], None]


@dataclass
class PipelineStats:
    """Live counters, surfaced in the dashboard so judges see real numbers."""

    frames_captured: int = 0
    frames_processed: int = 0
    frames_dropped: int = 0
    events_fired: int = 0
    started_at: float = field(default_factory=time.time)
    fps: float = 0.0

    def as_dict(self) -> dict[str, float | int]:
        elapsed = max(time.time() - self.started_at, 1e-6)
        return {
            "backend": active_backend(),
            "frames_captured": self.frames_captured,
            "frames_processed": self.frames_processed,
            "frames_dropped": self.frames_dropped,
            "events_fired": self.events_fired,
            "uptime_s": round(elapsed, 1),
            "process_fps": round(self.frames_processed / elapsed, 2),
        }


class Pipeline:
    """Reads frames, samples every Nth, and runs the event state machines.

    Capture runs in its own thread and keeps only the latest frame; the worker
    consumes from a bounded queue. `source` is anything indexable by frame
    number, or None for a synthetic source (the demo default).
    """

    def __init__(
        self,
        source: str | int | None = None,
        sink: EventSink | None = None,
        camera_id: str = "cam-01",
        realtime_factor: float = 1.0,
    ) -> None:
        self.source = source
        self.camera_id = camera_id
        self.sink = sink or (lambda event: None)
        # realtime_factor >1 speeds the demo up; ==1 is real time; 0 is as fast
        # as possible (used by the eval harness).
        self.realtime_factor = realtime_factor

        self._queue: Queue[np.ndarray] = Queue(maxsize=config.QUEUE_MAXSIZE)
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []
        self._fall = FallDetector()
        self._crowd = CrowdCounter()
        self.stats = PipelineStats()
        self._last_frame: np.ndarray | None = None
        self._lock = threading.Lock()

    # --- frame sources -----------------------------------------------------

    def _open_capture(self):  # noqa: ANN201
        if self.source is None:
            return None
        import cv2  # noqa: PLC0415

        if isinstance(self.source, int):
            cap = cv2.VideoCapture(self.source)
        else:
            cap = cv2.VideoCapture(str(self.source))
        return cap if cap.isOpened() else None

    def _capture_loop(self) -> None:
        cap = self._open_capture()
        frame_idx = 0
        try:
            while not self._stop.is_set():
                if cap is None:
                    frame = blank_frame()
                    self._publish(frame)
                    if self.realtime_factor > 0:
                        time.sleep(1.0 / 30.0 / max(self.realtime_factor, 1e-6))
                else:
                    ok, frame = cap.read()
                    if not ok:
                        cap.set(1, 0)  # loop the clip for an unattended demo
                        continue
                    if frame_idx % config.FRAME_STRIDE == 0:
                        self._publish(frame)
                    frame_idx += 1
        finally:
            if cap is not None:
                cap.release()

    def _publish(self, frame: np.ndarray) -> None:
        self.stats.frames_captured += 1
        try:
            self._queue.put_nowait(frame)
        except Full:
            # Drop the oldest frame rather than fall behind real time.
            try:
                self._queue.get_nowait()
                self._queue.put_nowait(frame)
            except Empty:
                pass
            self.stats.frames_dropped += 1

    # --- inference worker --------------------------------------------------

    def _process_loop(self) -> None:
        detector = get_detector()
        while not self._stop.is_set():
            try:
                frame = self._queue.get(timeout=0.2)
            except Empty:
                continue
            with self._lock:
                self._last_frame = frame
            boxes = detector.detect(frame)
            now = time.time()
            events = self._fall.update(boxes, now=now) + self._crowd.update(
                boxes, now=now
            )
            self.stats.frames_processed += 1
            for event in events:
                self.stats.events_fired += 1
                try:
                    self.sink(event)
                except Exception:  # noqa: BLE001 - a sink bug must not stop capture
                    log.exception("event sink failed for %s", event)

    # --- lifecycle ---------------------------------------------------------

    def start(self) -> None:
        if self._threads:
            return
        self._stop.clear()
        self._threads = [
            threading.Thread(target=self._capture_loop, name="capture", daemon=True),
            threading.Thread(target=self._process_loop, name="infer", daemon=True),
        ]
        for thread in self._threads:
            thread.start()
        log.info("pipeline started (backend=%s)", active_backend())

    def stop(self) -> None:
        self._stop.set()
        for thread in self._threads:
            thread.join(timeout=2.0)
        self._threads = []

    def snapshot(self) -> np.ndarray | None:
        """Most recent raw frame (for the optional MJPEG debug view)."""
        with self._lock:
            return None if self._last_frame is None else self._last_frame.copy()

    def process_once(self, frame: np.ndarray, now: float | None = None) -> list[VideoEvent]:
        """Synchronous single-frame step. Used by the eval harness and tests."""
        detector = get_detector()
        boxes = detector.detect(frame)
        return self._fall.update(boxes, now=now) + self._crowd.update(boxes, now=now)
