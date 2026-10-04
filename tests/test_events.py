"""Tests for the debounce / persistence logic and the fall state machine.

The persistence timer is what stops a single-frame flicker from firing an event
on stage, so these tests pin exactly that behaviour: a blip must not fire, a
sustained condition must fire once, and the cool-down must suppress a re-fire.
"""

from campus_safety.detectors.base import Box
from campus_safety.events import CrowdCounter, Debouncer, FallDetector


# --- Debouncer -------------------------------------------------------------


def test_debouncer_requires_hold():
    d = Debouncer(hold_s=3.0, cooldown_s=10.0)
    assert d.update("k", True, 0.9, now=0.0) is None   # just started
    assert d.update("k", True, 0.9, now=1.0) is None   # still holding
    assert d.update("k", True, 0.9, now=2.9) is None   # not yet
    held = d.update("k", True, 0.9, now=3.1)           # crossed the threshold
    assert held is not None and held >= 3.0


def test_debouncer_flicker_does_not_fire():
    d = Debouncer(hold_s=3.0, cooldown_s=10.0, grace_s=1.0)
    d.update("k", True, 0.9, now=0.0)
    # A single frame of "active" then it vanishes past the grace window.
    assert d.update("k", False, 0.0, now=2.0) is None
    # Reappears but the hold restarts; a short reappearance must not fire.
    assert d.update("k", True, 0.9, now=2.1) is None
    assert d.update("k", True, 0.9, now=2.5) is None


def test_debouncer_cooldown_blocks_refire():
    d = Debouncer(hold_s=1.0, cooldown_s=10.0)
    d.update("k", True, 0.9, now=0.0)
    assert d.update("k", True, 0.9, now=1.1) is not None  # fires
    # Immediately holding again, but within cool-down -> no second fire.
    d.update("k", True, 0.9, now=1.2)
    assert d.update("k", True, 0.9, now=2.5) is None
    # After the cool-down, a fresh sustained hold fires again.
    d.update("k", True, 0.9, now=12.0)
    assert d.update("k", True, 0.9, now=13.1) is not None


def test_debouncer_keys_are_independent():
    d = Debouncer(hold_s=1.0, cooldown_s=5.0)
    d.update("a", True, 0.9, now=0.0)
    d.update("b", True, 0.9, now=0.0)
    assert d.update("a", True, 0.9, now=1.1) is not None
    assert d.update("b", True, 0.9, now=1.1) is not None


# --- FallDetector ----------------------------------------------------------


def _standing(tid=1, y=400.0):
    # tall box: h > w, centred so its centroid is near y
    return Box(bbox=(500.0, y - 100, 40.0, 100.0), conf=0.9, track_id=tid)


def _fallen(tid=1, y=460.0):
    # wide box: w > h, centroid dropped
    return Box(bbox=(460.0, y - 30, 120.0, 30.0), conf=0.9, track_id=tid)


def test_fall_requires_transition_and_hold():
    fd = FallDetector(hold_s=3.0, cooldown_s=20.0)
    # Establish a standing history.
    assert fd.update([_standing()], now=0.0) == []
    assert fd.update([_standing()], now=0.2) == []
    # Now the person falls and stays down.
    assert fd.update([_fallen()], now=0.4) == []       # just went down
    assert fd.update([_fallen()], now=2.0) == []       # holding
    events = fd.update([_fallen()], now=3.6)           # held long enough
    assert len(events) == 1
    assert events[0].kind == "fall"
    assert events[0].track_id == 1


def test_already_lying_person_does_not_fire():
    fd = FallDetector(hold_s=2.0, cooldown_s=20.0)
    # First observation is already "wide" -> no standing->fallen transition.
    fd.update([_fallen()], now=0.0)
    fd.update([_fallen()], now=1.0)
    assert fd.update([_fallen()], now=3.0) == []


def test_fall_single_frame_blip_does_not_fire():
    fd = FallDetector(hold_s=3.0, cooldown_s=20.0)
    fd.update([_standing()], now=0.0)
    fd.update([_standing()], now=0.2)
    fd.update([_fallen()], now=0.4)       # one wide frame
    fd.update([_standing()], now=0.6)     # back up immediately
    assert fd.update([_standing()], now=4.0) == []


# --- CrowdCounter ----------------------------------------------------------


def _person_in_canteen(tid):
    # Canteen polygon spans x in [0.60,0.90], y in [0.46,0.68] of a 960x540 map.
    # Put the centroid squarely inside: ~ (0.75*960, 0.57*540) = (720, 308).
    return Box(bbox=(700.0, 290.0, 40.0, 36.0), conf=0.8, track_id=tid)


def test_crowd_fires_over_capacity_after_hold():
    cc = CrowdCounter(threshold=12, hold_s=2.0, cooldown_s=30.0)
    boxes = [_person_in_canteen(100 + i) for i in range(60)]  # way over capacity
    assert cc.update(boxes, now=0.0) == []    # just started holding
    events = cc.update(boxes, now=2.5)
    assert len(events) == 1
    assert events[0].kind == "crowd"
    assert events[0].zone_id == "canteen"


def test_crowd_below_threshold_does_not_fire():
    cc = CrowdCounter(threshold=12, hold_s=1.0, cooldown_s=30.0)
    boxes = [_person_in_canteen(100 + i) for i in range(3)]
    assert cc.update(boxes, now=0.0) == []
    assert cc.update(boxes, now=2.0) == []
