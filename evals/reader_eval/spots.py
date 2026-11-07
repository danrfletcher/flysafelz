"""The places and the hour a NOTAM is judged at (G13, R3.4, R3.11).

    eval_spots(area, parsed) -> the spots: the area's centre, 8 just inside its edge, 8 just outside it, 8 near the local area's edge
    plan_spots(area, parsed) -> the same spots and how many candidates were dropped for being on the wrong side
    eval_segment(parsed, reading) -> the one-hour segment every spot is judged in

The area is code's: the circle or polygon `build_area` made from the E text, or the NOTAM's Q-circle when there is none. The
centre is a circle's own centre, or shapely's `representative_point()` for a polygon, which lies inside a concave shape too.
From the centre, along each of 8 compass bearings (`i * 360 / 8`, computed), the edge is where the ray first leaves the area.
The inside spot is `ring_inner_nm` short of that, the outside spot `ring_inner_nm` past it and the far spot `ring_outer_nm` past it.
Every inside spot is checked with `geo.circle_over` or `geo.shape_over` and every outside and far spot is checked to be outside.
A spot on the wrong side (which a concave shape can cause) is dropped and counted, never scored. An area whose edge is no
further than `ring_inner_nm` from the centre has no inside spots: the centre stands for them.

The segment is one hour centred on the middle of the first interval in which the NOTAM is active within B-C (item D decides,
with SR and SS resolved at the centre). With no item D and no end time (C open-ended or PERM) it is B to B + 1 h. Pressure is
the reference pressure of the rules.
"""

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import timedelta
from enum import StrEnum

import shapely
from shapely import make_valid
from shapely.geometry import LineString, Point
from shapely.geometry import Polygon as ShapelyPolygon
from shapely.geometry.base import BaseGeometry
from shapely.ops import nearest_points

from evals.reader_eval.settings import SETTINGS, EvalSettings, reference_pressure_hpa
from flysafelz.contracts import Circle, NotamReading, ParsedNotam, Polygon
from flysafelz.geo import FULL_CIRCLE, GeometryError, circle_over, compass16, distance_nm, shape_from_points, shape_over
from flysafelz.notam_action import ActionSegment, LandingZone, active_intervals
from flysafelz.notam_area import Area

BEARING_COUNT = 8  # compass points around the area (structure)
REFINEMENTS = 3  # passes that pull a computed spot onto its exact distance from the centre
SEARCH_DAYS = 7  # how far past B to look for the first active interval of a NOTAM with no end time but an item D
HOUR = timedelta(hours=1)


class SpotKind(StrEnum):
    CENTRE = "centre"
    INSIDE = "inside"
    OUTSIDE = "outside"
    FAR = "far"


@dataclass(frozen=True)
class Spot:
    """A place to judge the NOTAM from. offset_nm is the distance from the edge (0 for the centre); bearing_deg is from the centre."""

    kind: SpotKind
    lat: float
    lon: float
    bearing_deg: float | None = None
    offset_nm: float = 0.0

    def describe(self) -> str:
        """Where the spot is, in words a pilot can place: "the centre", "0.1 NM outside the edge, NE of the centre"."""
        if self.kind is SpotKind.CENTRE or self.bearing_deg is None:
            return "the centre"
        side = "inside" if self.kind is SpotKind.INSIDE else "outside"
        return f"{self.offset_nm:g} NM {side} the edge, {compass16(self.bearing_deg)} of the centre"


@dataclass(frozen=True)
class SpotPlan:
    spots: tuple[Spot, ...]
    dropped: int  # candidate spots on the wrong side of the edge, left out


@dataclass(frozen=True)
class _Region:
    """The area as the spots need it: its centre, whether a point is over it, and how far from the centre its edge is on a bearing."""

    lat: float
    lon: float
    over: Callable[[float, float], bool]
    edge_nm: Callable[[float], float | None]  # a bearing in degrees -> NM from the centre to the edge on it; None when the ray finds none


def _circle_of(area: Area, parsed: ParsedNotam) -> Circle:
    if isinstance(area, Circle):
        return area
    return Circle(lat=parsed.centre_lat, lon=parsed.centre_lon, radius_nm=parsed.radius_nm)


def _polygon_shape(area: Area) -> tuple[BaseGeometry, list] | None:
    """The polygon as a shapely shape (lon, lat) and as geo.py's shape, or None when it is not a usable polygon."""
    if not isinstance(area, Polygon):
        return None
    try:
        shape = shape_from_points(area.points)
    except GeometryError:
        return None
    flat = ShapelyPolygon([(lon, lat) for lat, lon in area.points])
    return (flat if flat.is_valid else make_valid(flat)), shape


def area_centre(area: Area, parsed: ParsedNotam) -> tuple[float, float]:
    """(lat, lon) of the area's centre: a circle's own, a polygon's representative point, else the Q-circle's."""
    found = _polygon_shape(area)
    if found is None:
        circle = _circle_of(area, parsed)
        return circle.lat, circle.lon
    point = found[0].representative_point()
    return point.y, point.x


def _destination(lat: float, lon: float, bearing: float, nm: float) -> tuple[float, float]:
    """The point nm NM from (lat, lon) on the bearing, measured with geo.distance_nm so that geo's tests see exactly nm."""
    north = distance_nm(lat, lon, lat + 1, lon)  # NM in a degree of latitude
    east = distance_nm(lat, lon, lat, lon + 1)  # NM in a degree of longitude here
    theta = math.radians(bearing)
    dlat, dlon = nm * math.cos(theta) / north, nm * math.sin(theta) / east
    for _ in range(REFINEMENTS):
        got = distance_nm(lat, lon, lat + dlat, lon + dlon)
        if got == 0:
            break
        dlat, dlon = dlat * nm / got, dlon * nm / got
    return lat + dlat, lon + dlon


def _circle_region(circle: Circle) -> _Region:
    return _Region(
        circle.lat,
        circle.lon,
        lambda lat, lon: circle_over(lat, lon, circle.lat, circle.lon, circle.radius_nm),
        lambda bearing: circle.radius_nm,
    )


def _polygon_region(flat: BaseGeometry, shape: list) -> _Region:
    centre = flat.representative_point()
    lat0, lon0 = centre.y, centre.x
    east = distance_nm(lat0, lon0, lat0, lon0 + 1)  # NM in a degree of longitude here
    north = distance_nm(lat0, lon0, lat0 + 1, lon0)  # NM in a degree of latitude
    local = shapely.transform(flat, lambda coords: (coords - [lon0, lat0]) * [east, north])  # NM east and north of the centre
    boundary = local.boundary
    min_x, min_y, max_x, max_y = local.bounds
    reach = math.hypot(max_x - min_x, max_y - min_y) + 1  # long enough to leave the shape from a point inside it

    def edge_nm(bearing: float) -> float | None:
        theta = math.radians(bearing)
        hit = LineString([(0.0, 0.0), (reach * math.sin(theta), reach * math.cos(theta))]).intersection(boundary)
        if hit.is_empty:
            return None
        nearest = nearest_points(Point(0.0, 0.0), hit)[1]
        return math.hypot(nearest.x, nearest.y)

    return _Region(lat0, lon0, lambda lat, lon: shape_over(lat, lon, shape), edge_nm)


def _region(area: Area, parsed: ParsedNotam) -> _Region:
    found = _polygon_shape(area)
    if found is not None:
        return _polygon_region(*found)
    return _circle_region(_circle_of(area, parsed))


def plan_spots(area: Area, parsed: ParsedNotam, settings: EvalSettings = SETTINGS) -> SpotPlan:
    """The spots (the centre first, then each bearing's inside, outside and far spot) and how many candidates were dropped."""
    region = _region(area, parsed)
    spots = [Spot(SpotKind.CENTRE, region.lat, region.lon)]
    dropped = 0
    for i in range(BEARING_COUNT):
        bearing = i * FULL_CIRCLE / BEARING_COUNT
        edge = region.edge_nm(bearing)
        if edge is None:
            dropped += 3  # the ray found no edge, so none of the three spots on it can be placed
            continue
        wanted = (
            (SpotKind.INSIDE, edge - settings.ring_inner_nm, settings.ring_inner_nm, True),
            (SpotKind.OUTSIDE, edge + settings.ring_inner_nm, settings.ring_inner_nm, False),
            (SpotKind.FAR, edge + settings.ring_outer_nm, settings.ring_outer_nm, False),
        )
        for kind, from_centre, offset, should_be_over in wanted:
            if kind is SpotKind.INSIDE and edge <= settings.ring_inner_nm:
                continue  # the edge is this close to the centre: the centre stands for the inner ring
            lat, lon = _destination(region.lat, region.lon, bearing, from_centre)
            if bool(region.over(lat, lon)) is should_be_over:
                spots.append(Spot(kind, lat, lon, bearing, offset))
            else:
                dropped += 1
    return SpotPlan(tuple(spots), dropped)


def eval_spots(area: Area, parsed: ParsedNotam, settings: EvalSettings = SETTINGS) -> list[Spot]:
    """The spots to judge a NOTAM from. See the module text."""
    return list(plan_spots(area, parsed, settings).spots)


def eval_segment(parsed: ParsedNotam, reading: NotamReading | None, settings: EvalSettings = SETTINGS) -> ActionSegment:
    """The one-hour segment every spot is judged in, for this NOTAM (the reading supplies its area and item D). See the module text."""
    lat, lon = area_centre(reading.geometry if reading else None, parsed)
    lz = LandingZone(lat=lat, lon=lon, elevation_ft=settings.site_elevation_ft)
    schedule = (reading.schedule if reading and reading.schedule.strip() else parsed.item_d) or ""
    begin = parsed.valid_from
    start = begin
    if parsed.valid_to is not None or schedule.strip():
        horizon = parsed.valid_to if parsed.valid_to is not None else begin + timedelta(days=SEARCH_DAYS)
        intervals: Sequence = active_intervals(parsed, reading, lz, begin, horizon)
        if intervals:
            first_start, first_end = intervals[0]
            start = first_start + (first_end - first_start) / 2 - HOUR / 2
    return ActionSegment(start=start, end=start + HOUR, pressure_msl=reference_pressure_hpa())
