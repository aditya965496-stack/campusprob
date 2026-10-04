# Smart Campus Safety System

**Detection + Verification + Prediction + Response — in one explainable loop.**

An MVP for the Vibeathon AI/ML track. A student SOS and AI video detection
(fall / crowd) are fused into a **transparent, rule-based risk score** and pushed
live to a security dashboard on an offline campus map.

This repo is built to the discipline the pitch deck sets for itself: honest
scope tags, a privacy stance that survives the hardest Q&A, and **no accuracy
number we did not measure ourselves**.

---

## What's real (MVP) vs roadmap

| Feature | Status |
|---|---|
| Student SOS with zone-level location | ✅ MVP |
| Video **fall** detection (aspect-ratio flip + centroid drop, persistence timer) | ✅ MVP |
| Video **crowd** detection (per-zone polygon counting) | ✅ MVP |
| Explainable risk score with per-signal contribution bars | ✅ MVP |
| Multi-signal **verification** (SOS ⇄ video fusion) | ✅ MVP |
| Live security dashboard (map, queue, status workflow) | ✅ MVP |
| Role-based access + audit log | ✅ MVP |
| Swappable detector backends (YOLO / OpenCV-DNN / scripted) | ✅ MVP |
| Predictive risk zones (historical frequency) | 🛠 Roadmap |
| Safe-route recommendation | 🛠 Roadmap |
| Automatic personnel assignment / dispatch | 🛠 Roadmap |
| Violence / fight detection | 🛠 Roadmap — deliberately out of scope |

> **Violence detection is intentionally excluded.** It is the least reliable CV
> component, trivially confused with play/sport, and ethically loaded. The API
> refuses it by design.

---

## Quick start

```bash
pip install -r requirements.txt
python tools/seed.py
uvicorn campus_safety.api:app --reload
```

Then open:

- **Student SOS** — http://localhost:8000/
- **Security dashboard** — http://localhost:8000/dashboard
- **Admin / transparency** — http://localhost:8000/admin

The system runs with **zero ML dependencies** — detection falls back to a
deterministic scripted source so the full loop works on any machine. To enable
live YOLO person-detection:

```bash
pip install -r requirements-ml.txt
```

It will be picked up automatically on next launch (the active backend is shown
as a badge on the dashboard).

---

## The demo (3 minutes, de-risked)

1. **Open the dashboard.** It is already alive — ~50 seeded incidents, zones
   shaded by 30-day risk. Never demo on empty state.
2. **Raise a real SOS.** Open the student page on a phone/second tab, pick a
   zone, hit SOS. Watch it appear on the dashboard map **live** over WebSocket,
   with its priority and a reason.
3. **Show the fusion.** Click **▶ Demo: fall + SOS**. A video fall fires, then a
   co-located SOS — the score jumps because the two **corroborate**. Click
   **Why?** to show the contribution bars. This is the differentiator: a video
   event *plus* an SOS in the same zone scores higher than either alone.
4. **Open Admin.** The risk weights are **published** — any score on screen can
   be reproduced by hand. The privacy panel states "no face recognition" and the
   licensing of the active detector. The audit log shows every action.

**Stage backup:** every "Demo" button drives the exact same code path as the
detector, so the pitch works even with no camera and no ML installed.

---

## The one number we measured

The deck requires a number we measured ourselves — not a paper figure.

```bash
# precision / recall / confusion matrix on YOUR labelled clips
python tools/eval_detection.py --clips data/clips

# real inference FPS on THIS machine
python tools/benchmark.py
```

Report them with the clip count and your camera/lighting conditions, and do not
generalise beyond that setup. That honesty is the credibility lever.

---

## How the risk score works

A transparent weighted sum over named signals (published in `config.py` and at
`/api/weights`):

```
priority = w_sos·sos + w_video·video_event + w_time·time_of_day
         + w_zone·zone_history + w_crowd·crowd + w_corrob·(sos AND nearby video)
```

Plus **hard overrides**: an SOS, or a fall confirmed past the persistence timer,
forces a minimum priority so the score can never under-rate an emergency.
Weights are hand-set for the MVP — with real incident data we would calibrate
them. We deliberately do **not** use a learned severity model: we have no
labelled incident data, and a glass box staff can audit beats a black box
trained on nothing.

---

## Privacy & responsible AI

- **No face recognition, anywhere.** Person detection returns an anonymous box;
  `track_id` is a transient association for the fall timer, not an identity.
- **Event metadata, not footage.** We persist `{type, zone, time, confidence}`;
  raw frames never reach the database.
- **Role-based access + audit log.** The dashboard is security-only; the audit
  log is admin-only; every status change is recorded.
- Designed to **DPDP 2023** principles: notice, consent, purpose limitation,
  retention. (Verify the current Rules status before quoting dates on a slide.)

---

## Architecture

```
Student App ─┐
CCTV / clip ─┼─► Backend (FastAPI) ─► AI Engine ─► Risk Engine ─► Alert (WebSocket) ─► Dashboard
Location ────┘         │              (detectors/)   (risk.py)                          (map, queue)
                       └─ SQLite (event metadata, audit)         ▲
                                                                 └─ feedback: Ack / Dispatch / Resolve
```

- `campus_safety/detectors/` — the swappable vision layer. `get_detector()`
  tries Ultralytics → OpenCV-DNN → scripted and **never crashes**.
- `campus_safety/pipeline.py` — capture thread + bounded queue + inference
  worker; samples every Nth frame; debounces events before firing.
- `campus_safety/events.py` — the fall state machine and crowd counter.
- `campus_safety/risk.py` — the explainable risk engine.
- `campus_safety/api.py` — HTTP + WebSocket, and the single fusion point.
- `frontend/` — three surfaces, vanilla JS, **no build step, no CDN, no tiles**.

### Detector licensing

The default live backend is **Ultralytics YOLO (AGPL-3.0)** — fine for a demo;
a hosted product would open-source under AGPL or buy a commercial licence, or
swap to **YOLOX (Apache-2.0)** / **torchvision (BSD-3)** behind the same
`Detector` interface. The active backend and its licence are shown on the Admin
page.

---

## Tests

```bash
pytest
```

Covers the parts where a silent bug would embarrass the demo: risk-score
reproducibility and overrides, the debouncer (flicker rejection + cool-down),
the fall state machine, zone geometry, and the API end to end.

---

## Configuration

Everything is environment-overridable (see `campus_safety/config.py`): detector
backend (`CAMPUS_DETECTOR`), frame stride, fall/crowd timers and thresholds,
corroboration window, and the data directory. Storage is SQLite by default;
`CAMPUS_DB_URL` documents the PostgreSQL path for production.

---

## Likely judge questions — honest answers

- **How accurate is detection?** We quote only our own eval (`eval_detection.py`)
  with its clip count and conditions. No paper numbers.
- **Did you train a model?** No — pretrained person detector, event logic on top.
- **What about privacy?** No face recognition; event metadata not footage;
  role-based access; DPDP-aligned.
- **How is this different from existing apps?** SOS apps are reactive; enterprise
  video analytics don't connect to a student's alert. We *fuse* the two into one
  explainable, auditable score — affordable and student-facing.
- **Real-time?** On this machine the pipeline sustains the FPS `benchmark.py`
  reports; one camera with frame sampling is the honest MVP limit.
