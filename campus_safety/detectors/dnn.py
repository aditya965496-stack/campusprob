"""OpenCV-DNN person detector — the offline fallback with no ML framework.

Uses cv2.dnn with a MobileNet-SSD style Caffe/Caffe2 model if the model files are
present in data/models/. Unlike Ultralytics this needs no torch, so it is the
fallback when the ML stack is missing but the user still wants real inference on
a frame.

If the model files are absent, `available` is False and selection moves on to
the scripted/synthetic source.
"""

from __future__ import annotations

import numpy as np

from .. import config
from .base import Box

PERSON_CLASS_INDEX = 15  # VOC-style class id used by MobileNet-SSD

MODEL_DIR = config.DATA_DIR / "models"
PROTOTXT = MODEL_DIR / "MobileNetSSD_deploy.prototxt"
CAFFEMODEL = MODEL_DIR / "MobileNetSSD_deploy.caffemodel"


class DnnDetector:
    name = "opencv-dnn"
    license = "BSD-3 (OpenCV)"

    def __init__(self, confidence: float = config.PERSON_CONF) -> None:
        import cv2  # noqa: PLC0415 - lazy so selection can fail cleanly

        self._cv2 = cv2
        self._confidence = confidence
        self._net = None
        self._next_track_id = 1
        self._tracks: dict[int, tuple[float, float, float, float]] = {}

        if PROTOTXT.exists() and CAFFEMODEL.exists():
            try:
                self._net = cv2.dnn.readNetFromCaffe(
                    str(PROTOTXT), str(CAFFEMODEL)
                )
            except Exception:  # noqa: BLE001
                self._net = None
        self.available = self._net is not None

    @staticmethod
    def _iou(a, b) -> float:  # noqa: ANN001
        ax, ay, aw, ah = a
        bx, by, bw, bh = b
        x1, y1 = max(ax, bx), max(ay, by)
        x2, y2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
        inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        union = aw * ah + bw * bh - inter
        return inter / union if union > 0 else 0.0

    def _assign_track(self, bbox) -> int:  # noqa: ANN001
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
        if self._net is None:
            return []
        cv2 = self._cv2
        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(
            cv2.resize(frame, (300, 300)),
            scalefactor=0.007843,
            size=(300, 300),
            mean=127.5,
        )
        self._net.setInput(blob)
        detections = self._net.forward()
        boxes: list[Box] = []
        seen: set[int] = set()
        for i in range(detections.shape[2]):
            conf = float(detections[0, 0, i, 2])
            if conf < self._confidence:
                continue
            if int(detections[0, 0, i, 1]) != PERSON_CLASS_INDEX:
                continue
            x1 = float(detections[0, 0, i, 3] * w)
            y1 = float(detections[0, 0, i, 4] * h)
            x2 = float(detections[0, 0, i, 5] * w)
            y2 = float(detections[0, 0, i, 6] * h)
            bbox = (x1, y1, x2 - x1, y2 - y1)
            tid = self._assign_track(bbox)
            seen.add(tid)
            boxes.append(Box(bbox=bbox, conf=conf, track_id=tid))
        self._tracks = {t: b for t, b in self._tracks.items() if t in seen}
        return boxes


def build() -> DnnDetector:
    return DnnDetector()
