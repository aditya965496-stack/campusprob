"""Smart Campus Safety System — detection + verification + explainable risk.

An MVP for the Vibeathon AI/ML track. See README.md for scope tags: everything
the runtime path needs is implemented; prediction / safe-route / auto-dispatch
are roadmap and are labelled as such in the UI and in /api/weights metadata.

Privacy posture (non-negotiable, see Slide 8 of the deck):
  - No face recognition anywhere in this codebase. Person detection returns an
    anonymous bounding box, nothing more.
  - We persist event *metadata* (type, zone, timestamp, confidence) — never raw
    frames and never identification data.
"""

__version__ = "0.1.0-mvp"
