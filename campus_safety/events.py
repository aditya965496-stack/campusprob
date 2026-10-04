"""Event logic: detections -> debounced incidents.

This is the layer the deck describes on Slide 6 — rule-based interpretation on
top of raw detections, with a persistence timer that stops single-frame flicker
from firing an event on stage. Two state machines live here:

  * FallDetector  - box aspect-ratio flip + centroid drop, held for N seconds,
                    tracked per person so the timer follows one body.
  * CrowdCounter  - people-count per zone polygon, held for N seconds.

Both feed a shared Debouncer, and both translate detections into a
`VideoEvent` the rest of the system understands.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Literal

from . import config
from .detectors.base import Box
from .zones import ZONES_BY_ID, zone_for_point


def _utc_now() -> float:
    return datetime.now(timezone.utc).timestamp()


@dataclass(slots=True)
class VideoEvent:
    """A confirmed video event, ready to be scored and shown."""

    kind: Literal["fall", "crowd"]
    zone_id: str
    timestamp: float
    confidence: float
    held_seconds: float
    track_id: int | None = None
    detail: str = ""

    @property
    def ts_iso(self) -> str:
        return (
            datetime.fromtimestamp(self.timestamp, tz=timezone.utc)
            .replace(microsecond=0)
            .isoformat()
        )


@dataclass
class _Pending:
    """A condition that is holding but has not yet fired."""

    since: float
    last_seen: float
    confidence: float
    detail: str = ""


class Debouncer:
    """Fires once a condition has held for `hold_s`, with a cool-down after.

    Keyed by an arbitrary string so the same object can debounce falls per
    track and crowds per zone. The `now` argument on every call keeps it fully
    testable without sleeping.

    A hold is considered *broken* if the condition stops being observed for
    `grace_s` — that is what rejects a one-frame flicker.
    """

    def __init__(self, hold_s: float, cooldown_s: float, grace_s: float = 1.0):
        self.hold_s = hold_s
        self.cooldown_s = cooldown_s
        self.grace_s = grace_s
        self._pending: dict[str, _Pending] = {}
        self._last_fired: dict[str, float] = {}

    def update(
        self, key: str, active: bool, confidence: float, now: float, detail: str = ""
    ) -> float | None:
        """Report whether `key` is active. Returns held_seconds when it fires.

        Call once per processed frame for every key of interest. Returns a float
        exactly on the frame the event should fire (and only then), else None.
        """
        if not active:
            # Break the hold if the condition has been absent past the grace.
            pending = self._pending.get(key)
            if pending and (now - pending.last_seen) > self.grace_s:
                del self._pending[key]
            return None

        # Condition held long enough — but respect the cool-down.
        last = self._last_fired.get(key)
        if last is not None and (now - last) < self.cooldown_s:
            self._pending.pop(key, None)
            return None

        pending = self._pending.get(key)
        if pending is None:
            self._pending[key] = _Pending(
                since=now, last_seen=now, confidence=confidence, detail=detail
            )
            return None

        pending.last_seen = now
        pending.confidence = max(pending.confidence, confidence)
        if detail:
            pending.detail = detail

        held = now - pending.since
        if held < self.hold_s:
            return None

        self._last_fired[key] = now
        self._pending.pop(key, None)
        return held

    def active_keys(self) -> list[str]:
        return list(self._pending.keys())


@dataclass
class _TrackState:
    """Per-person memory for the fall rule."""

    was_standing: bool
    is_down: bool
    last_centroid_y: float
    last_box_height: float
    first_seen: float


class FallDetector:
    """Aspect-ratio flip + centroid drop, held for a persistence timer.

    Standing -> tall box (h > w). Fallen -> wide box (w >= h) *and* the centroid
    drops. We require the flip to be a *transition* (the person was standing) so
    someone already lying on the grass is not reported as falling over and over.
    """

    def __init__(
        self,
        hold_s: float = config.FALL_HOLD_S,
        cooldown_s: float = config.FALL_COOLDOWN_S,
        aspect_ratio: float = config.FALL_ASPECT_RATIO,
        centroid_drop: float = config.FALL_CENTROID_DROP,
    ):
        self.aspect_ratio = aspect_ratio
        self.centroid_drop = centroid_drop
        self._tracks: dict[int, _TrackState] = {}
        self._debouncer = Debouncer(hold_s, cooldown_s)

    def _is_wide(self, box: Box) -> bool:
        _, _, w, h = box.bbox
        return h > 0 and (w / h) >= self.aspect_ratio

    def update(self, boxes: list[Box], now: float | None = None) -> list[VideoEvent]:
        """Consume one frame of detections; return any fall events that fire."""
        now = _utc_now() if now is None else now
        events: list[VideoEvent] = []
        seen_ids: set[int] = set()

        for box in boxes:
            tid = box.track_id
            if tid is None:
                continue
            seen_ids.add(tid)
            cx, cy, w, h = box.bbox
            cy_center = cy + h / 2.0
            wide = self._is_wide(box)

            state = self._tracks.get(tid)
            if state is None:
                self._tracks[tid] = _TrackState(
                    was_standing=not wide,
                    is_down=False,
                    last_centroid_y=cy_center,
                    last_box_height=h,
                    first_seen=now,
                )
                continue

            if not wide:
                state.was_standing = True
                state.is_down = False
            else:
                if not state.is_down and state.was_standing:
                    dropped = (
                        cy_center - state.last_centroid_y
                    ) > (self.centroid_drop * max(state.last_box_height, 1e-6))
                    if dropped:
                        state.is_down = True
                        state.was_standing = False

            state.last_centroid_y = cy_center
            state.last_box_height = h

            zone_id = zone_for_point(cx + w / 2.0, cy_center) or "unknown"
            key = f"fall:{tid}"
            held = self._debouncer.update(
                key, active=state.is_down, confidence=box.conf, now=now,
                detail=f"track {tid}",
            )
            if held is not None:
                zone = ZONES_BY_ID.get(zone_id)
                events.append(
                    VideoEvent(
                        kind="fall",
                        zone_id=zone_id,
                        timestamp=now,
                        confidence=box.conf,
                        held_seconds=held,
                        track_id=tid,
                        detail=(
                            f"Person down in {zone.name if zone else 'unknown zone'}"
                            f" for {held:.1f}s"
                        ),
                    )
                )

        # Forget tracks that left the frame, so ids can be recycled safely.
        for tid in list(self._tracks):
            if tid not in seen_ids:
                self._tracks.pop(tid, None)
                self._debouncer._pending.pop(f"fall:{tid}", None)
        return events


class CrowdCounter:
    """People-count per zone polygon, held for a persistence timer."""

    def __init__(
        self,
        threshold: int = config.CROWD_THRESHOLD,
        hold_s: float = config.CROWD_HOLD_S,
        cooldown_s: float = config.CROWD_COOLDOWN_S,
    ):
        self.threshold = threshold
        self._debouncer = Debouncer(hold_s, cooldown_s)

    def counts(self, boxes: list[Box]) -> dict[str, int]:
        counts: dict[str, int] = {}
        for box in boxes:
            cx, cy, w, h = box.bbox
            zone_id = zone_for_point(cx + w / 2.0, cy + h / 2.0)
            if zone_id:
                counts[zone_id] = counts.get(zone_id, 0) + 1
        return counts

    def update(self, boxes: list[Box], now: float | None = None) -> list[VideoEvent]:
        now = _utc_now() if now is None else now
        events: list[VideoEvent] = []
        counts = self.counts(boxes)

        for zone_id, count in counts.items():
            zone = ZONES_BY_ID.get(zone_id)
            threshold = zone.capacity if zone else self.threshold
            # Normalise against the zone's own capacity, not a global number.
            over = count >= max(threshold, self.threshold)
            held = self._debouncer.update(
                f"crowd:{zone_id}",
                active=over,
                confidence=min(1.0, count / max(threshold, 1)),
                now=now,
                detail=f"{count} people",
            )
            if held is not None:
                events.append(
                    VideoEvent(
                        kind="crowd",
                        zone_id=zone_id,
                        timestamp=now,
                        confidence=min(1.0, count / max(threshold, 1)),
                        held_seconds=held,
                        detail=(
                            f"{count} people in {zone.name if zone else zone_id}"
                            f" (capacity {threshold})"
                        ),
                    )
                )
        return events
