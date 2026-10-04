"""Configuration: paths, runtime settings, and the published risk weights.

Everything a judge might ask "how did you decide that?" about lives here so
there is exactly one place to point at.
"""

from __future__ import annotations

import os
from pathlib import Path

# --- Paths -----------------------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("CAMPUS_DATA_DIR", ROOT / "data"))
FRONTEND_DIR = ROOT / "frontend"
DB_PATH = DATA_DIR / "campus_safety.db"

# --- Storage ---------------------------------------------------------------
# SQLite for the MVP (stdlib, zero setup, survives a dead venue wifi).
# Production would move to PostgreSQL: set CAMPUS_DB_URL and swap db.py's
# connection helper. The schema below is deliberately portable.
DB_URL = os.environ.get("CAMPUS_DB_URL", f"sqlite:///{DB_PATH}")

# --- Detection -------------------------------------------------------------
# "auto" tries ultralytics -> opencv-dnn -> scripted, and never crashes.
DETECTOR_BACKEND = os.environ.get("CAMPUS_DETECTOR", "auto")

# Process every Nth frame. Events persist over seconds, so ~5 inference-FPS is
# plenty and keeps a CPU laptop alive on one camera.
FRAME_STRIDE = int(os.environ.get("CAMPUS_FRAME_STRIDE", "3"))

# Bounded queue between the capture thread and the inference worker. If the
# worker falls behind we drop frames rather than lag behind real time.
QUEUE_MAXSIZE = int(os.environ.get("CAMPUS_QUEUE_MAXSIZE", "8"))

# --- Event logic -----------------------------------------------------------

# A fall must stay "down" this long before it fires. This timer is what stops
# single-frame flicker on stage. The deck quotes 3-5s; we default to 4s.
FALL_HOLD_S = float(os.environ.get("CAMPUS_FALL_HOLD_S", "4.0"))
FALL_COOLDOWN_S = float(os.environ.get("CAMPUS_FALL_COOLDOWN_S", "20.0"))

# Standing = tall box; fallen = wide box. w/h ratio at or above this counts as
# "wide". 0.8 tolerates perspective without firing on a crouch.
FALL_ASPECT_RATIO = float(os.environ.get("CAMPUS_FALL_ASPECT", "0.8"))

# Minimum downward centroid movement (in box-heights) to count as a drop.
FALL_CENTROID_DROP = float(os.environ.get("CAMPUS_FALL_DROP", "0.15"))

# Crowd: >capacity people in one zone polygon, held for CROWD_HOLD_S.
CROWD_HOLD_S = float(os.environ.get("CAMPUS_CROWD_HOLD_S", "6.0"))
CROWD_COOLDOWN_S = float(os.environ.get("CAMPUS_CROWD_COOLDOWN_S", "60.0"))
CROWD_THRESHOLD = int(os.environ.get("CAMPUS_CROWD_THRESHOLD", "12"))

# Person-detection confidence floor.
PERSON_CONF = float(os.environ.get("CAMPUS_PERSON_CONF", "0.35"))

# If no detection backend is available, fall back to a synthetic person
# walk-through so the pipeline still produces frames end to end. The on-camera
# badge in the dashboard tells the presenter which mode is live.
ALLOW_SYNTHETIC = os.environ.get("CAMPUS_ALLOW_SYNTHETIC", "1") == "1"

# --- Risk engine -----------------------------------------------------------
# THE published weights. Hand-set for the MVP; the README and the admin UI both
# say out loud: "these would be calibrated from real incident data." A judge can
# read this dict and reproduce any score by hand.
#
# Formula (Slide 6 / Round 3 §3):
#   priority = w_sos*sos + w_video*video_event + w_time*time_risk
#            + w_zone*zone_history + w_crowd*crowd + w_corrob*(sos AND video)
RISK_WEIGHTS: dict[str, float] = {
    "sos": 45.0,          # a panic press on its own is already serious
    "video_fall": 50.0,   # a confirmed fall is the highest-weight video event
    "video_crowd": 18.0,  # crowding alone is a signal, not an emergency
    "time": 10.0,         # multiplied by time-of-day risk in [0,1]
    "zone": 15.0,         # multiplied by 30-day zone incident frequency
    "crowd": 12.0,        # multiplied by normalised crowd level
    "corroboration": 25.0,  # the verification bonus: SOS + nearby video event
}

# Hard overrides so the model can never under-rate an emergency the way a
# learned scorer might. Mirrors risk = likelihood x impact with a floor.
OVERRIDES = {
    # Any SOS forces at least this priority regardless of other signals.
    "sos_min_priority": 60.0,
    # A fall held for the persistence timer forces at least this priority.
    "fall_min_priority": 55.0,
}

SCORE_MAX = 100.0

# Severity banding on the 0-100 priority. Kept coarse on purpose — a
# three-band score is something a guard can act on; a float is not.
SEVERITY_BANDS = [
    (75.0, "critical"),
    (50.0, "high"),
    (25.0, "medium"),
    (0.0, "low"),
]

# Time-of-day risk priors (hour -> risk in [0,1]). Campus rhythm: late night and
# the small hours are higher risk; teaching hours are low.
TIME_OF_DAY_RISK = {
    0: 0.9, 1: 0.95, 2: 0.95, 3: 0.9, 4: 0.85, 5: 0.7,
    6: 0.4, 7: 0.25, 8: 0.2, 9: 0.15, 10: 0.15, 11: 0.15,
    12: 0.2, 13: 0.2, 14: 0.15, 15: 0.2, 16: 0.25, 17: 0.35,
    18: 0.45, 19: 0.5, 20: 0.6, 21: 0.7, 22: 0.8, 23: 0.85,
}

# Default zone-history risk when we have no incidents on record for a zone.
DEFAULT_ZONE_RISK = 0.15

# Two incidents in the same zone count as "corroborating" if they are within
# this many seconds of each other (the fusion window).
CORROBORATION_WINDOW_S = float(os.environ.get("CAMPUS_CORROB_WINDOW_S", "120"))

# Licences surfaced in the UI / Q&A so the answer is one click away on stage.
DETECTOR_LICENSES = {
    "ultralytics": {
        "name": "Ultralytics YOLO (YOLOv8 / YOLO11)",
        "license": "AGPL-3.0",
        "note": (
            "Fine for this demo. AGPL's network clause means a hosted product "
            "must open-source under AGPL or buy a commercial licence. Swappable "
            "for YOLOX (Apache-2.0) or torchvision (BSD-3) behind the same API."
        ),
    },
    "opencv-dnn": {
        "name": "OpenCV DNN (MobileNet-SSD, COCO person class)",
        "license": "BSD-3 (OpenCV) / model terms vary",
        "note": "No ML framework dependency; used as the offline fallback.",
    },
    "scripted": {
        "name": "Scripted replay detector",
        "license": "n/a (no model)",
        "note": "Deterministic event source so the demo runs with zero ML deps.",
    },
    "synthetic": {
        "name": "Synthetic person tracker",
        "license": "n/a (no model)",
        "note": "No camera and no model — draws a person box so the loop is live.",
    },
    "none": {
        "name": "No detector",
        "license": "n/a",
        "note": "No backend available and synthetic generation disabled.",
    },
}
