"""Ultralytics YOLO person detector.

LICENCE: Ultralytics YOLO is AGPL-3.0. Its network clause means that if you
expose it as a hosted service you must open-source the whole application under
AGPL, or buy a commercial licence. For a hackathon demo on staged footage this
is fine. In production we would either open-source under AGPL or licence
commercially, or swap to a permissive model (YOLOX Apache-2.0, torchvision
BSD-3) behind the same Detector interface.

This module imports torch/ultralytics lazily so the rest of the system runs with
neither installed.
"""

from __future__ import annotations

import numpy as np

from .. import config
from .base import Box

PERSON_CLASS_INDEX = 0  # COCO: person is class 0


class YoloDetector:
    name = "ultralytics"
    license = "AGPL-3.0"

    def __init__(self, weights: str = "yolov8n.pt", imgsz: int = 480) -> None:
        from ultralytics import YOLO  # noqa: PLC0415 - lazy by design

        self._model = YOLO(weights)
        self._imgsz = imgsz
        self.available = True
        self._next_track_id = 1
        # Greedy IoU tracker: enough to hold a fall timer on one camera, and it
        # avoids pulling ByteTrack in as a hard dependency. (ByteTrack is the
        # upgrade path noted in the deck.)
        self._tracks: dict[int, tuple[float, float, float, float]] = {}

    @staticmethod
    def _iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
        ax, ay, aw, ah = a
        bx, by, bw, bh = b
        x1, y1 = max(ax, bx), max(ay, by)
        x2, y2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
        inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        union = aw * ah + bw * bh - inter
        return inter / union if union > 0 else 0.0

    def _assign_track(self, bbox: tuple[float, float, float, float]) -> int:
        best_id, best_iou = None, 0.3
        for tid, tb in self._tracks.items():
            score = self._iou(bbox, tb)
            if score > best_iou:
                best_id, best_iou = tid, score
        if best_id is None:
            best_id = self._next_track_id
            self._next_track_id += 1
        self._tracks[best_id] = bbox
        return best_id

    def detect(self, frame: np.ndarray) -> list[Box]:
        results = self._model.predict(
            frame, imgsz=self._imgsz, conf=config.PERSON_CONF, verbose=False
        )
        boxes: list[Box] = []
        seen: set[int] = set()
        for result in results:
            for det in result.boxes:
                if int(det.cls) != PERSON_CLASS_INDEX:
                    continue
                x1, y1, x2, y2 = (float(v) for v in det.xyxy[0])
                bbox = (x1, y1, x2 - x1, y2 - y1)
                tid = self._assign_track(bbox)
                seen.add(tid)
                boxes.append(Box(bbox=bbox, conf=float(det.conf), track_id=tid))
        # Drop tracks not seen this frame so ids are not recycled onto new people.
        self._tracks = {t: b for t, b in self._tracks.items() if t in seen}
        return boxes


def build() -> YoloDetector:
    return YoloDetector()
