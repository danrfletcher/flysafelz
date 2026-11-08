"""Plain geometry for the landing zone (LZ): distances in NM, over-the-LZ tests, bearings, the UK guard.

Everything here is pure. No network, no model, no state. The only thresholds
(the UK bounds) come from rules.yaml.

Distances and bearings between two points use a local flat map built from the
WGS84 radii of curvature at the mid latitude of the pair. Against WGS84
geodesics this is accurate to well under 0.1 % out to 25 NM, which is far
inside the 0.5 % the callers need. It is not meant for long distances.

Polygon tests put every vertex on one flat map with its scale fixed at the
LZ's latitude, so an edge that is straight in (lat, lon) stays straight and an
LZ on it is found on it, however long the edge. Edges are read as straight
lines in (lat, lon).

Conventions:
- Points are (lat, lon) in degrees, east positive. GeoJSON is [lon, lat] and
  is converted at the edge (shape_from_geojson), nowhere else.
- Bearings are degrees clockwise from true north, in [0, 360).
- Fail closed: an LZ exactly on a boundary is over the shape, and the
  distance to the edge of a shape the LZ is inside is 0.
"""

import math
from collections.abc import Sequence
from itertools import pairwise

from flysafelz.rules import RULES

METRES_PER_NM = 1852
METRES_PER_KM = 1000  # unit: the radius in an E text may be written in km
MIN_POLYGON_POINTS = 3  # structure: fewest distinct points that make an area (an E text with fewer is not a polygon)
FULL_CIRCLE = 360  # degrees
HALF_TURN = 180  # degrees
QUARTER_TURN = 90  # degrees: the pole is this far from the equator
WGS84_A = 6378137.0  # equatorial radius, metres
WGS84_E2 = 0.00669437999014  # first eccentricity squared
ON_EDGE_M = 1e-6  # an LZ this close to an edge counts as on it (float noise only)

OUTSIDE_UK_MESSAGE = "outside UK coverage"

COMPASS_POINTS = (
    "N",
    "NNE",
    "NE",
    "ENE",
    "E",
    "ESE",
    "SE",
    "SSE",
    "S",
    "SSW",
    "SW",
    "WSW",
    "W",
    "WNW",
    "NW",
    "NNW",
)

Point = tuple[float, float]  # (lat, lon)
Ring = list[Point]  # closed: the last point repeats the first
Poly = list[Ring]  # the outer ring first, then any holes
Shape = list[Poly]  # one or more polygons (a MultiPolygon)
_Xy = tuple[float, float]  # (east, north) metres from the LZ


class GeometryError(ValueError):
    """A shape cannot be used (too few points, bad coordinates, unknown GeoJSON type)."""


class OutsideUKCoverage(ValueError):
    """The point is outside the UK bounds in rules.yaml."""


def check_uk_bounds(lat: float, lon: float) -> None:
    """Raise OutsideUKCoverage unless lat and lon are inside the UK bounds (edges inclusive).

    NaN and infinity are outside. The four bounds are the uk_bounds_* rules.
    """
    if not (math.isfinite(lat) and math.isfinite(lon)):
        raise OutsideUKCoverage(OUTSIDE_UK_MESSAGE)
    if (
        RULES.uk_bounds_lat_min.breached(lat)
        or RULES.uk_bounds_lat_max.breached(lat)
        or RULES.uk_bounds_lon_min.breached(lon)
        or RULES.uk_bounds_lon_max.breached(lon)
    ):
        raise OutsideUKCoverage(OUTSIDE_UK_MESSAGE)


# --- points -----------------------------------------------------------------


def _offset_m(lat0: float, lon0: float, lat: float, lon: float) -> _Xy:
    """(east, north) metres from (lat0, lon0) to (lat, lon) on the local flat map."""
    mid = math.radians((lat0 + lat) * 0.5)
    sin_mid = math.sin(mid)
    w = math.sqrt(1 - WGS84_E2 * sin_mid * sin_mid)
    north_radius = WGS84_A * (1 - WGS84_E2) / (w * w * w)
    east_radius = WGS84_A / w
    dlon = (lon - lon0 + HALF_TURN) % FULL_CIRCLE - HALF_TURN
    north = math.radians(lat - lat0) * north_radius
    east = math.radians(dlon) * east_radius * math.cos(mid)
    return east, north


def _bearing_of(xy: _Xy) -> float:
    """Bearing of an (east, north) offset. A zero offset has no direction and gives 0.0."""
    return math.degrees(math.atan2(xy[0], xy[1])) % FULL_CIRCLE


def distance_nm(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Distance between two points in nautical miles."""
    return math.hypot(*_offset_m(lat1, lon1, lat2, lon2)) / METRES_PER_NM


def bearing_deg(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Bearing from point 1 to point 2, degrees clockwise from north, in [0, 360)."""
    return _bearing_of(_offset_m(lat1, lon1, lat2, lon2))


def compass16(bearing: float) -> str:
    """The 16-point compass name for a bearing: 22.5 degree sectors centred on N, NNE, NE and so on.

    A bearing exactly on a sector boundary takes the clockwise side (11.25 is NNE).
    360 is N and negative bearings wrap (-10 is N).
    """
    if not math.isfinite(bearing):
        raise ValueError(f"bearing must be finite, got {bearing!r}")
    sector = FULL_CIRCLE / len(COMPASS_POINTS)
    return COMPASS_POINTS[int((bearing % FULL_CIRCLE + sector * 0.5) // sector) % len(COMPASS_POINTS)]


# --- circles ----------------------------------------------------------------


def circle_over(lat: float, lon: float, clat: float, clon: float, radius_nm: float) -> bool:
    """True when the LZ is inside the circle or exactly on its edge."""
    return distance_nm(lat, lon, clat, clon) <= radius_nm


def circle_edge_distance_nm(lat: float, lon: float, clat: float, clon: float, radius_nm: float) -> float:
    """NM from the LZ to the circle's edge. 0 when the LZ is over the circle."""
    return max(0.0, distance_nm(lat, lon, clat, clon) - radius_nm)


def circle_edge_bearing_deg(lat: float, lon: float, clat: float, clon: float, radius_nm: float) -> float:
    """Bearing from the LZ to the nearest point on the circle's edge.

    From outside this points at the centre. From inside it points directly away
    from the centre. At the exact centre every edge point is equally near and
    the result is 0.0.
    """
    toward_centre = _offset_m(lat, lon, clat, clon)
    bearing = _bearing_of(toward_centre)
    if not any(toward_centre):
        return 0.0
    if math.hypot(*toward_centre) < radius_nm * METRES_PER_NM:
        return (bearing + HALF_TURN) % FULL_CIRCLE
    return bearing


# --- polygons ---------------------------------------------------------------


def _checked_point(lat: float, lon: float) -> Point:
    if not (math.isfinite(lat) and math.isfinite(lon)):
        raise GeometryError(f"coordinates must be finite, got ({lat!r}, {lon!r})")
    if abs(math.radians(lat)) > math.pi * 0.5:
        raise GeometryError(f"latitude {lat!r} is out of range (GeoJSON is [lon, lat])")
    return float(lat), float(lon)


def ring_from_points(points: Sequence[Sequence[float]]) -> Ring:
    """A closed ring from (lat, lon) points. An open ring is closed.

    Raises GeometryError for fewer than three distinct points or bad coordinates,
    so the caller can fail closed.
    """
    ring = [_checked_point(p[0], p[1]) for p in points]
    try:
        _first, _second, _third, *_rest = dict.fromkeys(ring)
    except ValueError as exc:
        raise GeometryError("a ring needs at least three distinct points") from exc
    if ring[0] != ring[-1]:
        ring.append(ring[0])
    return ring


def shape_from_points(points: Sequence[Sequence[float]]) -> Shape:
    """A one-polygon shape from (lat, lon) points, as the contracts store them."""
    return [[ring_from_points(points)]]


def _geojson_ring(ring: Sequence[Sequence[float]]) -> Ring:
    return ring_from_points([(position[1], position[0]) for position in ring])  # [lon, lat] -> (lat, lon)


def shape_from_geojson(geometry: dict) -> Shape:
    """A Shape from a GeoJSON Polygon or MultiPolygon, whose positions are [lon, lat].

    The first ring of each polygon is the outer boundary and any others are holes.
    Raises GeometryError for any other type or a ring that is not usable.
    """
    kind = geometry.get("type")
    coordinates = geometry.get("coordinates")
    if kind == "Polygon":
        polygons = [coordinates]
    elif kind == "MultiPolygon":
        polygons = coordinates
    else:
        raise GeometryError(f"unsupported GeoJSON type {kind!r}; need Polygon or MultiPolygon")
    if not polygons:
        raise GeometryError("GeoJSON geometry has no polygons")
    shape = []
    for rings in polygons:
        if not rings:
            raise GeometryError("GeoJSON polygon has no rings")
        shape.append([_geojson_ring(ring) for ring in rings])
    return shape


def _fixed_offset_m(lat0: float, lon0: float, lat: float, lon: float) -> _Xy:
    """(east, north) metres from (lat0, lon0) on a flat map whose scale is fixed at lat0.

    Unlike _offset_m, the scale does not depend on the other point, so a
    straight edge in (lat, lon) stays a straight line on this map. Polygon
    tests need that: an LZ on the edge must land on the segment, not near it.
    """
    phi = math.radians(lat0)
    sin_phi = math.sin(phi)
    w = math.sqrt(1 - WGS84_E2 * sin_phi * sin_phi)
    north_radius = WGS84_A * (1 - WGS84_E2) / (w * w * w)
    east_radius = WGS84_A / w
    dlon = (lon - lon0 + HALF_TURN) % FULL_CIRCLE - HALF_TURN
    return math.radians(dlon) * east_radius * math.cos(phi), math.radians(lat - lat0) * north_radius


def _ring_xy(lat: float, lon: float, ring: Ring) -> list[_Xy]:
    """The ring on one flat map centred on the LZ, so every edge is a straight segment."""
    return [_fixed_offset_m(lat, lon, p_lat, p_lon) for p_lat, p_lon in ring]


def _inside(ring_xy: list[_Xy]) -> bool:
    """Even-odd test: is the origin (the LZ) inside this closed ring?"""
    inside = False
    for (x1, y1), (x2, y2) in pairwise(ring_xy):
        if (y1 > 0) != (y2 > 0):
            x_at_zero = x1 - y1 * (x2 - x1) / (y2 - y1)
            if x_at_zero > 0:
                inside = not inside
    return inside


def _nearest_on_segment(a: _Xy, b: _Xy) -> _Xy:
    """The point on segment a-b nearest the origin."""
    dx, dy = b[0] - a[0], b[1] - a[1]
    length_sq = dx * dx + dy * dy
    if length_sq == 0:
        return a
    t = max(0.0, min(1.0, -(a[0] * dx + a[1] * dy) / length_sq))
    return a[0] + t * dx, a[1] + t * dy


def _nearest_on_ring(ring_xy: list[_Xy]) -> _Xy:
    candidates = [_nearest_on_segment(a, b) for a, b in pairwise(ring_xy)]
    return min(candidates, key=lambda p: math.hypot(*p))


def _nearest_on_shape(lat: float, lon: float, shape: Shape) -> _Xy:
    """The boundary point nearest the LZ, as (east, north) metres from the LZ."""
    candidates = [_nearest_on_ring(_ring_xy(lat, lon, ring)) for poly in shape for ring in poly]
    return min(candidates, key=lambda p: math.hypot(*p))


def shape_over(lat: float, lon: float, shape: Shape) -> bool:
    """True when the LZ is inside the shape (not in a hole) or exactly on any edge."""
    for poly in shape:
        rings_xy = [_ring_xy(lat, lon, ring) for ring in poly]
        if any(math.hypot(*_nearest_on_ring(r)) <= ON_EDGE_M for r in rings_xy):
            return True
        outer, *holes = rings_xy
        if _inside(outer) and not any(_inside(h) for h in holes):
            return True
    return False


def shape_edge_distance_nm(lat: float, lon: float, shape: Shape) -> float:
    """NM from the LZ to the shape's edge. 0 when the LZ is over the shape.

    An LZ in a hole is not over the shape, so this is the distance to the hole's edge.
    """
    if shape_over(lat, lon, shape):
        return 0.0
    return math.hypot(*_nearest_on_shape(lat, lon, shape)) / METRES_PER_NM


def shape_edge_bearing_deg(lat: float, lon: float, shape: Shape) -> float:
    """Bearing from the LZ to the nearest point on the shape's boundary (also when the LZ is inside)."""
    return _bearing_of(_nearest_on_shape(lat, lon, shape))
