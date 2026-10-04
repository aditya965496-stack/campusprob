"""Campus zones: polygons, assignment, and zone-level risk priors.

Zones are the unit of location for this whole system. Indoor GPS is poor, so we
deliberately locate at the zone level rather than pretending to street accuracy
(see the deck's "Location accuracy" risk). Polygons live in normalised map space
([0, 1] x [0, 1]) and are shared with frontend/assets/campus.svg so the map and
the backend agree by construction.
"""

from __future__ import annotations

import math

from . import config
from .models import Zone

# Coordinates are (x, y) in normalised map space, y increasing downward to match
# SVG. Keep the count small: a guard needs to reason about zones, not pixels.
CAMPUS_ZONES: tuple[Zone, ...] = (
    Zone(
        id="main_gate",
        name="Main Gate / Entrance",
        polygon=[(0.06, 0.72), (0.24, 0.72), (0.24, 0.94), (0.06, 0.94)],
        capacity=25,
        risk_prior=0.35,
    ),
    Zone(
        id="library",
        name="Central Library",
        polygon=[(0.34, 0.10), (0.58, 0.10), (0.58, 0.34), (0.34, 0.34)],
        capacity=40,
        risk_prior=0.12,
    ),
    Zone(
        id="academic_block",
        name="Academic Block",
        polygon=[(0.62, 0.08), (0.92, 0.08), (0.92, 0.38), (0.62, 0.38)],
        capacity=60,
        risk_prior=0.10,
    ),
    Zone(
        id="canteen",
        name="Canteen / Mess",
        polygon=[(0.60, 0.46), (0.90, 0.46), (0.90, 0.68), (0.60, 0.68)],
        capacity=50,
        risk_prior=0.25,
    ),
    Zone(
        id="sports_ground",
        name="Sports Ground",
        polygon=[(0.28, 0.56), (0.56, 0.56), (0.56, 0.92), (0.28, 0.92)],
        capacity=80,
        risk_prior=0.20,
    ),
    Zone(
        id="hostel_a",
        name="Hostel Block A",
        polygon=[(0.08, 0.16), (0.28, 0.16), (0.28, 0.44), (0.08, 0.44)],
        capacity=45,
        risk_prior=0.30,
    ),
    Zone(
        id="parking",
        name="Parking / Rear Lot",
        polygon=[(0.62, 0.74), (0.92, 0.74), (0.92, 0.94), (0.62, 0.94)],
        capacity=30,
        risk_prior=0.45,
    ),
    Zone(
        id="admin_lawn",
        name="Admin Lawn",
        polygon=[(0.30, 0.38), (0.56, 0.38), (0.56, 0.52), (0.30, 0.52)],
        capacity=35,
        risk_prior=0.08,
    ),
)

ZONES_BY_ID: dict[str, Zone] = {z.id: z for z in CAMPUS_ZONES}
DEFAULT_ZONE_ID = "admin_lawn"


def point_in_polygon(x: float, y: float, polygon: list[tuple[float, float]]) -> bool:
    """Ray-casting test. Points on the boundary are treated as inside.

    Deliberately dependency-free: this runs per detected person per frame, and
    NumPy/shapely would be overkill for eight convex-ish quadrilaterals.
    """
    inside = False
    n = len(polygon)
    if n < 3:
        return False
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        # Half-open rule on y avoids double-counting shared vertices.
        if (yi > y) != (yj > y):
            # x coordinate of the edge at height y
            x_cross = (xj - xi) * (y - yi) / (yj - yi) + xi
            if x < x_cross:
                inside = not inside
        j = i
    return inside


def zone_for_point(x: float, y: float, frame_w: float = 960.0, frame_h: float = 540.0) -> str | None:
    """Which zone contains a point, or None if outside all zones.

    Supports both normalised coordinates in [0, 1] and pixel coordinates on a
    standard frame canvas (default 960x540).
    """
    if x > 1.0 or y > 1.0:
        x = x / frame_w
        y = y / frame_h
    for zone in CAMPUS_ZONES:
        if point_in_polygon(x, y, list(zone.polygon)):
            return zone.id
    return None


def zone_centroid(zone: Zone) -> tuple[float, float]:
    """Average of polygon vertices — good enough for drawing a marker."""
    xs = [p[0] for p in zone.polygon]
    ys = [p[1] for p in zone.polygon]
    return (sum(xs) / len(xs), sum(ys) / len(ys))


def normalized_history_risk(zone_id: str, count: int, busiest: int) -> float:
    """Map a zone's incident count onto a risk prior in [0, 1].

    Blends the observed frequency with the zone's hand-set prior so a brand-new
    zone is not treated as risk-free just because it has no history yet.
    """
    zone = ZONES_BY_ID.get(zone_id)
    prior = zone.risk_prior if zone else config.DEFAULT_ZONE_RISK
    if busiest <= 0:
        return prior
    observed = min(1.0, count / busiest)
    # 60% observed frequency, 40% prior — stated plainly, not a hidden constant.
    return round(0.6 * observed + 0.4 * prior, 4)


def haversine_m(
    lat1: float, lon1: float, lat2: float, lon2: float
) -> float:
    """Great-circle distance in metres. Used only for the (future) real map."""
    r = 6_371_000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))
