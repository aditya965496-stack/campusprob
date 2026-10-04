"""Tests for zone geometry and zone-risk normalisation."""

from campus_safety import zones
from campus_safety.zones import ZONES_BY_ID, normalized_history_risk, point_in_polygon


def test_point_in_square():
    square = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
    assert point_in_polygon(0.5, 0.5, square)
    assert not point_in_polygon(1.5, 0.5, square)
    assert not point_in_polygon(-0.1, 0.5, square)


def test_zone_for_point_matches_known_zone():
    # Centroid of the canteen polygon must resolve back to the canteen.
    canteen = ZONES_BY_ID["canteen"]
    cx = sum(p[0] for p in canteen.polygon) / len(canteen.polygon)
    cy = sum(p[1] for p in canteen.polygon) / len(canteen.polygon)
    assert zones.zone_for_point(cx, cy) == "canteen"


def test_point_outside_all_zones_is_none():
    # Top-left corner is deliberately empty space on the map.
    assert zones.zone_for_point(0.005, 0.005) is None


def test_zones_do_not_overlap_at_their_centroids():
    # Each zone's centroid should fall only in its own polygon — a sanity check
    # that the hand-authored map has no accidental overlaps.
    for zone in zones.CAMPUS_ZONES:
        cx = sum(p[0] for p in zone.polygon) / len(zone.polygon)
        cy = sum(p[1] for p in zone.polygon) / len(zone.polygon)
        assert zones.zone_for_point(cx, cy) == zone.id


def test_history_risk_blends_prior_when_no_data():
    # With no incidents anywhere, risk falls back to the zone's prior.
    r = normalized_history_risk("parking", count=0, busiest=0)
    assert r == ZONES_BY_ID["parking"].risk_prior


def test_history_risk_in_unit_interval():
    r = normalized_history_risk("library", count=5, busiest=10)
    assert 0.0 <= r <= 1.0


def test_busier_zone_scores_higher():
    low = normalized_history_risk("library", count=1, busiest=10)
    high = normalized_history_risk("library", count=9, busiest=10)
    assert high > low
