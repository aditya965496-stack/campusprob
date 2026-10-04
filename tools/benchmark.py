"""Measure real inference FPS on THIS machine — the deck's "one number".

Runs the active detector over N frames and reports the frames-per-second you can
actually sustain here. If a clip is given it uses real frames from it; otherwise
it uses the synthetic source so the number is still honest about the backend
(it just won't reflect decode cost).

Run:  python -m tools.benchmark [--clip data/clips/fall/sample.mp4] [--frames 200]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from campus_safety.detectors import active_backend, get_detector  # noqa: E402
from campus_safety.detectors.scripted import blank_frame  # noqa: E402


def _frames_from_clip(clip: Path, limit: int):
    import cv2  # noqa: PLC0415

    cap = cv2.VideoCapture(str(clip))
    out = []
    while len(out) < limit:
        ok, frame = cap.read()
        if not ok:
            break
        out.append(frame)
    cap.release()
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Measure detector FPS on this machine")
    ap.add_argument("--clip", default=None, help="optional video to read real frames from")
    ap.add_argument("--frames", type=int, default=200)
    args = ap.parse_args()

    detector = get_detector()
    backend = active_backend()
    print(f"Active detector backend: {backend}")

    if args.clip:
        frames = _frames_from_clip(Path(args.clip), args.frames)
        if not frames:
            print(f"Could not read frames from {args.clip}; using synthetic frames.")
            frames = [blank_frame() for _ in range(args.frames)]
    else:
        frames = [blank_frame() for _ in range(args.frames)]

    # Warm-up (first inference loads weights / allocates buffers).
    detector.detect(frames[0])

    start = time.perf_counter()
    total_boxes = 0
    for frame in frames:
        total_boxes += len(detector.detect(frame))
    elapsed = time.perf_counter() - start

    fps = len(frames) / elapsed if elapsed else 0.0
    print(f"Processed {len(frames)} frames in {elapsed:.2f}s")
    print(f"  -> {fps:.1f} inference FPS on this machine ({backend})")
    print(f"  -> {total_boxes} total person detections")
    print("\nQuote this with the machine spec. With frame sampling at stride "
          f"{__import__('campus_safety.config', fromlist=['FRAME_STRIDE']).FRAME_STRIDE}, "
          "one camera needs only a few inference-FPS to keep up in real time.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
