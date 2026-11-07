"""G13 spots: where a NOTAM is judged from. Real geometry (geo.circle_over / shape_over), areas from tests/factories.py."""

import math

import pytest
from factories import LZ, circle_at, offset, square_at
from reader_eval_helpers import q_notam

from evals.reader_eval.settings import SETTINGS
from evals.reader_eval.spots import BEARING_COUNT, SpotKind, area_centre, eval_spots, plan_spots
from flysafelz.contracts import Circle, Polygon
from flysafelz.geo import circle_over, distance_nm, shape_from_points, shape_over
from flysafelz.rules import RULES

TOLERANCE_NM = 1e-6  # the spots are placed to the metre or better


def of_kind(spots, kind):
    return [spot for spot in spots if spot.kind is kind]


def test_circle_has_a_centre_and_three_rings_of_eight():
    circle = circle_at(0, radius_nm=3.0)
    spots = eval_spots(circle, q_notam())
    assert len(spots) == 1 + 3 * BEARING_COUNT
    assert [len(of_kind(spots, kind)) for kind in SpotKind] == [1, 8, 8, 8]  # the enum is centre, inside, outside, far
    assert (spots[0].lat, spots[0].lon) == (circle.lat, circle.lon)


def test_rings_are_at_the_settings_distances_from_the_edge_on_computed_bearings():
    circle = circle_at(0, radius_nm=3.0)
    spots = eval_spots(circle, q_notam())
    expected = {SpotKind.INSIDE: -SETTINGS.ring_inner_nm, SpotKind.OUTSIDE: SETTINGS.ring_inner_nm, SpotKind.FAR: SETTINGS.ring_outer_nm}
    for kind, past_edge in expected.items():
        ring = of_kind(spots, kind)
        assert [spot.bearing_deg for spot in ring] == [i * 360 / 8 for i in range(8)]
        for spot in ring:
            assert distance_nm(circle.lat, circle.lon, spot.lat, spot.lon) == pytest.approx(circle.radius_nm + past_edge, abs=TOLERANCE_NM)
            assert spot.offset_nm == pytest.approx(abs(past_edge))


def test_each_spot_is_on_the_right_side_of_the_edge():
    circle = circle_at(0, radius_nm=3.0)
    spots = eval_spots(circle, q_notam())
    for spot in spots:
        over = circle_over(spot.lat, spot.lon, circle.lat, circle.lon, circle.radius_nm)
        assert over is (spot.kind in (SpotKind.CENTRE, SpotKind.INSIDE))


def test_the_bearings_point_where_they_say():
    circle = circle_at(0, radius_nm=3.0)
    spots = eval_spots(circle, q_notam())
    east = next(spot for spot in of_kind(spots, SpotKind.OUTSIDE) if spot.bearing_deg == 90)
    north = next(spot for spot in of_kind(spots, SpotKind.OUTSIDE) if spot.bearing_deg == 0)
    assert east.lon > circle.lon and east.lat == pytest.approx(circle.lat, abs=1e-6)
    assert north.lat > circle.lat and north.lon == pytest.approx(circle.lon, abs=1e-6)


def test_far_spots_stay_inside_the_local_area_by_a_margin():
    local_area = RULES.local_area.value
    assert not isinstance(local_area, list)
    assert SETTINGS.ring_outer_nm < local_area
    circle = circle_at(0, radius_nm=3.0)
    far = of_kind(eval_spots(circle, q_notam()), SpotKind.FAR)
    for spot in far:
        edge_gap = distance_nm(circle.lat, circle.lon, spot.lat, spot.lon) - circle.radius_nm
        assert edge_gap < local_area  # the prefilter's `> local_area` must not flip on float noise
        assert edge_gap == pytest.approx(SETTINGS.ring_outer_nm, abs=TOLERANCE_NM)


def test_a_circle_no_bigger_than_the_inner_ring_has_no_inner_spots():
    circle = circle_at(0, radius_nm=SETTINGS.ring_inner_nm)
    plan = plan_spots(circle, q_notam())
    assert of_kind(plan.spots, SpotKind.INSIDE) == []  # the centre stands for them, so none lies past the centre
    assert len(of_kind(plan.spots, SpotKind.OUTSIDE)) == 8
    assert len(of_kind(plan.spots, SpotKind.FAR)) == 8
    assert plan.dropped == 0
    smaller = plan_spots(circle_at(0, radius_nm=SETTINGS.ring_inner_nm / 2), q_notam())
    assert of_kind(smaller.spots, SpotKind.INSIDE) == []
    assert all(spot.kind is not SpotKind.INSIDE for spot in smaller.spots)


def test_no_area_uses_the_q_circle():
    parsed = q_notam(radius_nm=2.0)
    spots = eval_spots(None, parsed)
    assert len(spots) == 25
    assert (spots[0].lat, spots[0].lon) == (parsed.centre_lat, parsed.centre_lon)
    for spot in of_kind(spots, SpotKind.OUTSIDE):
        assert distance_nm(parsed.centre_lat, parsed.centre_lon, spot.lat, spot.lon) == pytest.approx(2.0 + SETTINGS.ring_inner_nm, abs=TOLERANCE_NM)


def test_convex_polygon_spots_are_on_the_right_side_and_none_is_dropped():
    square = square_at(0, half_nm=2.0)
    plan = plan_spots(square, q_notam())
    shape = shape_from_points(square.points)
    assert plan.dropped == 0
    assert len(plan.spots) == 25
    for spot in plan.spots:
        assert shape_over(spot.lat, spot.lon, shape) is (spot.kind in (SpotKind.CENTRE, SpotKind.INSIDE))
    # the edge is nearer on the sides than at the corners, and the spots follow it: compare east (a side) with north-east (a corner)
    outside = {spot.bearing_deg: spot for spot in of_kind(plan.spots, SpotKind.OUTSIDE)}
    side = distance_nm(plan.spots[0].lat, plan.spots[0].lon, outside[90].lat, outside[90].lon)
    corner = distance_nm(plan.spots[0].lat, plan.spots[0].lon, outside[45].lat, outside[45].lon)
    assert side == pytest.approx(2.0 + SETTINGS.ring_inner_nm, rel=0.02)
    assert corner > side


def u_shape(notch_nm: float) -> Polygon:
    """A U, 2 NM wide and 2 NM tall, open at the top: a notch notch_nm wide cut down from the top to 0.6 NM above the bottom."""

    def at(east: float, north: float) -> tuple[float, float]:
        lat, lon = offset(LZ.lat, LZ.lon, 0, north)
        return offset(lat, lon, 90, east)

    left, right = 1.0 - notch_nm / 2, 1.0 + notch_nm / 2
    return Polygon(points=[at(*xy) for xy in [(0, 0), (2, 0), (2, 2), (right, 2), (right, 0.6), (left, 0.6), (left, 2), (0, 2)]])


def test_the_centre_of_a_concave_polygon_is_inside_it():
    u = u_shape(0.6)
    lat, lon = area_centre(u, q_notam())
    assert shape_over(lat, lon, shape_from_points(u.points))
    spots = eval_spots(u, q_notam())
    assert (spots[0].lat, spots[0].lon) == (lat, lon)


def test_concave_polygon_spots_on_the_wrong_side_are_dropped_and_counted():
    u = u_shape(0.06)  # a notch narrower than the 0.1 NM ring: an outside spot on a notch wall lands in the far arm, which is inside the shape
    plan = plan_spots(u, q_notam())
    shape = shape_from_points(u.points)
    assert plan.dropped >= 1
    assert len(plan.spots) + plan.dropped == 25
    for spot in plan.spots:
        assert shape_over(spot.lat, spot.lon, shape) is (spot.kind in (SpotKind.CENTRE, SpotKind.INSIDE))


def test_a_wide_notch_loses_no_spot():
    plan = plan_spots(u_shape(0.8), q_notam())
    assert plan.dropped == 0
    assert len(plan.spots) == 25


def test_a_polygon_that_is_not_a_shape_falls_back_to_the_q_circle():
    two_points = Polygon.model_construct(points=[(LZ.lat, LZ.lon), (LZ.lat + 0.01, LZ.lon)])
    parsed = q_notam(radius_nm=2.0)
    spots = eval_spots(two_points, parsed)
    assert (spots[0].lat, spots[0].lon) == (parsed.centre_lat, parsed.centre_lon)
    assert len(spots) == 25


def test_describe_names_the_place_in_words():
    spots = eval_spots(circle_at(0, radius_nm=3.0), q_notam())
    assert spots[0].describe() == "the centre"
    outside_ne = next(spot for spot in spots if spot.kind is SpotKind.OUTSIDE and spot.bearing_deg == 45)
    assert outside_ne.describe() == "0.1 NM outside the edge, NE of the centre"
    inside_s = next(spot for spot in spots if spot.kind is SpotKind.INSIDE and spot.bearing_deg == 180)
    assert inside_s.describe() == "0.1 NM inside the edge, S of the centre"


def test_the_area_may_be_a_circle_built_anywhere():
    lat, lon = offset(LZ.lat, LZ.lon, 270, 30.0)
    far_away = Circle(lat=lat, lon=lon, radius_nm=1.0)
    spots = eval_spots(far_away, q_notam())
    assert len(spots) == 25
    for spot in of_kind(spots, SpotKind.OUTSIDE):
        assert math.isclose(distance_nm(lat, lon, spot.lat, spot.lon), 1.0 + SETTINGS.ring_inner_nm, abs_tol=TOLERANCE_NM)
