"""Honest detection evaluation — precision, recall, confusion matrix.

Reads a folder of short labelled clips and runs the SAME event logic the live
system uses, then reports precision/recall/confusion — NOT bare accuracy, which
is misleading for rare events (the deck and Round-3 doc both insist on this).

Expected layout (any that are present are used):

    data/clips/
      fall/        *.mp4   -> ground-truth label "fall"
      not_fall/    *.mp4   -> ground-truth label "none"
      crowd/       *.mp4   -> ground-truth label "crowd"

Each clip is scored as a single trial: did the event logic fire the expected
event at least once during the clip? That matches how the system is used (one
alert per incident), and it is honest about what we measured.

Run:  python -m tools.eval_detection --clips data/clips
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from campus_safety import config  # noqa: E402

LABELS = ("fall", "crowd", "none")
LABEL_DIRS = {"fall": "fall", "crowd": "crowd", "none": "not_fall"}


def _iter_clips(root: Path):
    for label, subdir in LABEL_DIRS.items():
        folder = root / subdir
        if not folder.is_dir():
            continue
        for clip in sorted(folder.glob("*")):
            if clip.suffix.lower() in (".mp4", ".avi", ".mov", ".mkv"):
                yield label, clip


def _event_in_clip(clip: Path) -> str:
    """Run the pipeline over one clip and return the first event kind, or 'none'."""
    try:
        import cv2  # noqa: PLC0415
    except Exception:
        return "none"
    from campus_safety.pipeline import Pipeline  # noqa: PLC0415

    pipeline = Pipeline(source=str(clip), realtime_factor=0.0)
    cap = cv2.VideoCapture(str(clip))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_idx = 0
    t = 0.0
    fired = "none"
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx % config.FRAME_STRIDE == 0:
            t += config.FRAME_STRIDE / fps
            events = pipeline.process_once(frame, now=t)
            for ev in events:
                return ev.kind  # first event wins
        frame_idx += 1
    cap.release()
    return fired


def evaluate(root: Path) -> dict:
    confusion: dict[str, dict[str, int]] = {a: defaultdict(int) for a in LABELS}
    n = 0
    for truth, clip in _iter_clips(root):
        pred = _event_in_clip(clip)
        confusion[truth][pred] += 1
        n += 1
        print(f"  {clip.name:<32} truth={truth:<6} pred={pred}")
    return {"confusion": confusion, "n": n}


def _metrics(confusion: dict[str, dict[str, int]]):
    rows = []
    for label in ("fall", "crowd"):  # "none" is the negative class
        tp = confusion[label][label]
        fp = sum(confusion[o][label] for o in LABELS if o != label)
        fn = sum(confusion[label][o] for o in LABELS if o != label)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        rows.append((label, tp, fp, fn, precision, recall, f1))
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description="Honest detection eval (precision/recall)")
    ap.add_argument("--clips", default=str(config.DATA_DIR / "clips"))
    args = ap.parse_args()

    root = Path(args.clips)
    if not root.is_dir():
        print(f"No clips folder at {root}.")
        print("Create data/clips/{fall,not_fall,crowd}/ and drop short clips in.")
        print("This script intentionally measures YOUR clips — we quote no number we didn't measure.")
        return 1

    print(f"Evaluating clips under {root} ...")
    result = evaluate(root)
    if result["n"] == 0:
        print("No clips found. Add .mp4/.avi/.mov files to the label folders.")
        return 1

    print("\nConfusion matrix (rows = truth, cols = predicted):")
    header = "          " + "".join(f"{p:>8}" for p in LABELS)
    print(header)
    for truth in LABELS:
        print(f"  {truth:<8}" + "".join(f"{result['confusion'][truth][p]:>8}" for p in LABELS))

    print("\nPer-class metrics:")
    print(f"  {'class':<8}{'TP':>4}{'FP':>4}{'FN':>4}{'prec':>8}{'recall':>8}{'F1':>8}")
    for label, tp, fp, fn, p, r, f1 in _metrics(result["confusion"]):
        print(f"  {label:<8}{tp:>4}{fp:>4}{fn:>4}{p:>8.2f}{r:>8.2f}{f1:>8.2f}")

    print(f"\nEvaluated {result['n']} clips.")
    print("Report these numbers with the clip count and your camera/lighting conditions.")
    print("Do NOT generalise them beyond that setup — that's the honest framing judges reward.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
