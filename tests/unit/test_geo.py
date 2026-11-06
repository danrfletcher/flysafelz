"""geo.py: distances, over-the-LZ tests, bearings, compass names and the UK guard (E16, E21, Geometry)."""

import math

import pytest

from flysafelz import geo
from flysafelz.geo import (
    GeometryError,
    OutsideUKCoverage,
    bearing_deg,
    check_uk_bounds,
    circle_edge_bearing_deg,
    circle_edge_distance_nm,
    circle_over,
    compass16,
    distance_nm,
    ring_from_points,
    shape_edge_bearing_deg,
    shape_edge_distance_nm,
    shape_from_geojson,
    shape_from_points,
    shape_over,
)

BLUE_MILL = (53.802797, -0.086644)
WEST_SLEEKBURN = (55.158, -1.570)
# H7171/26 E text: 0.5 NM radius of 534732N 0000459W
H7171_CENTRE = (53 + 47 / 60 + 32 / 3600, -(4 / 60 + 59 / 3600))
H7171_RADIUS_NM = 0.5
# EGD496A Q-line circle: 5356N00001E, 10 NM
EGD496A_CENTRE = (53 + 56 / 60, 1 / 60)
EGD496A_RADIUS_NM = 10

# WGS84 geodesic reference values (pyproj Geod("WGS84").inv, one-off): (from, to, NM, initial bearing deg).
GEODESICS = [
    (BLUE_MILL, H7171_CENTRE, 0.64822, 168.639),
    (BLUE_MILL, EGD496A_CENTRE, 8.66087, 25.027),
    ((53.8, -0.1), (53.8 + 25 / 60, -0.1), 25.04166, 0.0),
    ((53.8, -0.1), (53.8, 0.4), 17.78876, 89.798),
    ((55.158, -1.570), (55.355, -1.265), 15.83152, 40.465),
    ((60.15, -1.15), (60.2, -1.0), 5.40905, 56.148),
]


@pytest.mark.parametrize(("a", "b", "nm", "bearing"), GEODESICS)
def test_distance_matches_wgs84_geodesic_within_half_a_percent(a, b, nm, bearing):
    assert distance_nm(*a, *b) == pytest.approx(nm, rel=0.005)
    assert distance_nm(*b, *a) == pytest.approx(distance_nm(*a, *b), rel=1e-9)


@pytest.mark.parametrize(("a", "b", "nm", "bearing"), GEODESICS)
def test_bearing_matches_wgs84_geodesic(a, b, nm, bearing):
    got = bearing_deg(*a, *b)
    # The flat local map gives the constant-direction bearing; a geodesic's initial bearing differs by up to ~0.2 degrees at 18 NM.
    assert abs((got - bearing + 180) % 360 - 180) < 0.3


def test_bearing_covers_the_four_directions_and_stays_in_range():
    lat, lon = 54.0, -1.0
    assert bearing_deg(lat, lon, lat + 0.1, lon) == pytest.approx(0.0, abs=1e-9)
    assert bearing_deg(lat, lon, lat, lon + 0.1) == pytest.approx(90.0, abs=0.1)
    assert bearing_deg(lat, lon, lat - 0.1, lon) == pytest.approx(180.0, abs=1e-9)
    assert bearing_deg(lat, lon, lat, lon - 0.1) == pytest.approx(270.0, abs=0.1)
    assert 0 <= bearing_deg(lat, lon, lat + 0.1, lon - 0.0001) < 360


# --- circles ----------------------------------------------------------------


def test_blue_mill_against_the_h7171_e_text_circle():
    """E21: the LZ is 0.65 NM from the centre, so 0.15 NM outside the 0.5 NM circle, bearing 168.6 from the LZ."""
    centre_nm = distance_nm(*BLUE_MILL, *H7171_CENTRE)
    assert centre_nm == pytest.approx(0.648, abs=0.01)
    assert not circle_over(*BLUE_MILL, *H7171_CENTRE, H7171_RADIUS_NM)
    assert circle_edge_distance_nm(*BLUE_MILL, *H7171_CENTRE, H7171_RADIUS_NM) == pytest.approx(0.148, abs=0.01)
    assert circle_edge_bearing_deg(*BLUE_MILL, *H7171_CENTRE, H7171_RADIUS_NM) == pytest.approx(168.6, abs=0.5)


def test_blue_mill_is_inside_the_egd496a_q_circle_and_the_coded_h7171_circle():
    assert distance_nm(*BLUE_MILL, *EGD496A_CENTRE) == pytest.approx(8.66, abs=0.05)
    assert circle_over(*BLUE_MILL, *EGD496A_CENTRE, EGD496A_RADIUS_NM)
    assert circle_edge_distance_nm(*BLUE_MILL, *EGD496A_CENTRE, EGD496A_RADIUS_NM) == 0
    assert circle_over(*BLUE_MILL, *H7171_CENTRE, 1)  # the coded 1 NM Q-circle covers the LZ


def test_lz_exactly_on_a_circle_edge_is_over():
    radius = distance_nm(*BLUE_MILL, *H7171_CENTRE)
    assert circle_over(*BLUE_MILL, *H7171_CENTRE, radius)
    assert not circle_over(*BLUE_MILL, *H7171_CENTRE, radius - 0.001)
    assert circle_edge_distance_nm(*BLUE_MILL, *H7171_CENTRE, radius) == 0


def test_inside_a_circle_the_edge_distance_is_zero_and_the_nearest_edge_is_away_from_the_centre():
    centre = (54.0, -1.0)
    lz = (54.0, -0.99)  # east of the centre, 0.35 NM, inside a 1 NM circle
    assert circle_over(*lz, *centre, 1)
    assert circle_edge_distance_nm(*lz, *centre, 1) == 0
    assert circle_edge_bearing_deg(*lz, *centre, 1) == pytest.approx(90.0, abs=0.1)  # away from the centre
    assert circle_edge_bearing_deg(*lz, *centre, 0.1) == pytest.approx(270.0, abs=0.1)  # outside: towards it


def test_at_the_exact_centre_the_edge_bearing_is_zero_rather_than_an_error():
    assert circle_edge_bearing_deg(54.0, -1.0, 54.0, -1.0, 1) == 0.0


# --- polygons ---------------------------------------------------------------

# Deliberately asymmetric: 0.10 degrees of longitude wide, 0.02 of latitude tall.
# GeoJSON positions are [lon, lat]. Swapping them puts the box at latitude -0.10..0.0, far from the LZ.
BOX_GEOJSON = {
    "type": "Polygon",
    "coordinates": [[[-0.15, 53.79], [-0.05, 53.79], [-0.05, 53.81], [-0.15, 53.81], [-0.15, 53.79]]],
}


def test_geojson_is_lon_lat_so_an_asymmetric_box_is_read_the_right_way_round():
    shape = shape_from_geojson(BOX_GEOJSON)
    assert shape_over(*BLUE_MILL, shape)  # lat 53.8, lon -0.0866
    assert shape[0][0][0] == (53.79, -0.15)  # stored as (lat, lon)
    swapped = {"type": "Polygon", "coordinates": [[[lat, lon] for lon, lat in BOX_GEOJSON["coordinates"][0]]]}
    assert not shape_over(*BLUE_MILL, shape_from_geojson(swapped))


def test_a_latitude_out_of_range_is_a_geometry_error_which_catches_most_swapped_input():
    with pytest.raises(GeometryError):
        shape_from_geojson({"type": "Polygon", "coordinates": [[[53.0, 95.0], [54.0, 95.0], [54.0, 96.0], [53.0, 95.0]]]})


def test_inside_a_polygon_the_edge_distance_is_zero():
    shape = shape_from_geojson(BOX_GEOJSON)
    assert shape_edge_distance_nm(*BLUE_MILL, shape) == 0


def test_distance_to_a_polygon_edge_and_bearing_to_the_nearest_boundary_point():
    shape = shape_from_geojson(BOX_GEOJSON)  # lat 53.79..53.81, lon -0.15..-0.05
    south = (53.77, -0.10)  # 0.02 degrees of latitude south of the bottom edge = 1.2 NM
    assert not shape_over(*south, shape)
    assert shape_edge_distance_nm(*south, shape) == pytest.approx(distance_nm(*south, 53.79, -0.10), rel=1e-3)
    assert shape_edge_distance_nm(*south, shape) == pytest.approx(1.2, abs=0.01)
    assert shape_edge_bearing_deg(*south, shape) == pytest.approx(0.0, abs=0.1)
    east = (53.80, -0.03)
    assert shape_edge_bearing_deg(*east, shape) == pytest.approx(270.0, abs=0.1)
    inside_near_west = (53.80, -0.14)
    assert shape_over(*inside_near_west, shape)
    assert shape_edge_bearing_deg(*inside_near_west, shape) == pytest.approx(270.0, abs=0.1)  # nearest boundary is west


def test_nearest_boundary_point_can_be_the_middle_of_an_edge_or_a_corner():
    shape = shape_from_geojson(BOX_GEOJSON)
    corner = (53.77, -0.02)  # south-east of the SE corner (53.79, -0.05)
    assert shape_edge_distance_nm(*corner, shape) == pytest.approx(distance_nm(*corner, 53.79, -0.05), rel=1e-3)
    assert 270 < shape_edge_bearing_deg(*corner, shape) < 360  # towards the NW


def test_lz_exactly_on_a_polygon_edge_or_corner_is_over():
    shape = shape_from_geojson(BOX_GEOJSON)
    assert shape_over(53.79, -0.10, shape)  # on the south edge
    assert shape_over(53.79, -0.05, shape)  # on a corner
    assert shape_edge_distance_nm(53.79, -0.10, shape) == 0
    assert not shape_over(53.7899, -0.10, shape)


def test_lz_on_a_diagonal_edge_of_a_large_polygon_is_over_all_along_it():
    # The hypotenuse runs (53.7, -0.2) to (53.9, 0.0): neither north-south nor east-west, 0.2 degrees long.
    triangle = shape_from_points([(53.7, -0.2), (53.9, 0.0), (53.9, -0.2)])
    for fraction in (0.1, 0.25, 0.5, 0.75, 0.9):
        lat, lon = 53.7 + 0.2 * fraction, -0.2 + 0.2 * fraction
        assert shape_over(lat, lon, triangle), fraction
        assert shape_edge_distance_nm(lat, lon, triangle) == 0
    outside = (53.8 - 0.0005, -0.1 + 0.0005)  # about 40 m off the edge, on the outside
    assert not shape_over(*outside, triangle)
    # Same edge walked the other way round (reversed ring) and a bigger, 0.8 degree one.
    assert shape_over(53.8, -0.1, shape_from_points([(53.9, -0.2), (53.9, 0.0), (53.7, -0.2)]))
    big = shape_from_points([(53.0, -1.0), (53.8, 0.0), (53.8, -1.0)])
    assert shape_over(53.4, -0.5, big)


def test_a_multipolygon_is_over_when_any_part_is():
    far = [[[-3.0, 55.0], [-2.9, 55.0], [-2.9, 55.1], [-3.0, 55.1], [-3.0, 55.0]]]
    multi = {"type": "MultiPolygon", "coordinates": [far, BOX_GEOJSON["coordinates"]]}
    shape = shape_from_geojson(multi)
    assert len(shape) == 2
    assert shape_over(*BLUE_MILL, shape)
    assert shape_over(55.05, -2.95, shape)
    assert not shape_over(54.5, -2.0, shape)
    # The distance is to the nearest part.
    assert shape_edge_distance_nm(55.05, -2.8, shape) == pytest.approx(distance_nm(55.05, -2.8, 55.05, -2.9), rel=1e-3)


HOLEY = {
    "type": "Polygon",
    "coordinates": [
        [[-0.20, 53.70], [0.0, 53.70], [0.0, 53.90], [-0.20, 53.90], [-0.20, 53.70]],
        [[-0.12, 53.78], [-0.06, 53.78], [-0.06, 53.82], [-0.12, 53.82], [-0.12, 53.78]],  # around Blue Mill
    ],
}


def test_a_polygon_with_a_hole_is_not_over_the_hole():
    shape = shape_from_geojson(HOLEY)
    assert len(shape[0]) == 2
    assert not shape_over(*BLUE_MILL, shape)  # in the hole
    assert shape_over(53.75, -0.10, shape)  # in the solid part
    assert shape_over(53.80, -0.15, shape)  # solid, west of the hole
    # In the hole the distance is to the hole's nearest edge: the east one, at longitude -0.06.
    d = shape_edge_distance_nm(*BLUE_MILL, shape)
    assert d == pytest.approx(distance_nm(BLUE_MILL[0], BLUE_MILL[1], BLUE_MILL[0], -0.06), rel=1e-3)
    assert shape_edge_bearing_deg(*BLUE_MILL, shape) == pytest.approx(90.0, abs=0.5)  # east edge is nearest


def test_lz_exactly_on_a_hole_edge_is_over():
    shape = shape_from_geojson(HOLEY)
    assert shape_over(53.78, -0.09, shape)


def test_an_open_ring_is_closed_and_gives_the_same_answers():
    open_points = [(53.79, -0.15), (53.79, -0.05), (53.81, -0.05), (53.81, -0.15)]
    closed = ring_from_points(open_points)
    assert closed[0] == closed[-1]
    assert len(closed) == len(open_points) + 1
    assert ring_from_points(closed) == closed  # an already closed ring is left alone
    assert shape_over(*BLUE_MILL, shape_from_points(open_points))
    assert shape_over(*BLUE_MILL, shape_from_points(closed))


@pytest.mark.parametrize(
    "points",
    [
        [],
        [(53.0, -1.0)],
        [(53.0, -1.0), (54.0, -1.0)],
        [(53.0, -1.0), (54.0, -1.0), (53.0, -1.0)],  # closed line: only two distinct points
        [(53.0, -1.0)] * 5,
    ],
)
def test_a_ring_with_fewer_than_three_distinct_points_raises_geometry_error(points):
    with pytest.raises(GeometryError):
        ring_from_points(points)


def test_bad_coordinates_and_unsupported_geojson_raise_geometry_error():
    with pytest.raises(GeometryError):
        ring_from_points([(53.0, -1.0), (54.0, math.nan), (54.0, -2.0)])
    for bad in (
        {"type": "Point", "coordinates": [0, 0]},
        {"type": "MultiPolygon", "coordinates": []},
        {"type": "Polygon", "coordinates": []},
        {},
    ):
        with pytest.raises(GeometryError):
            shape_from_geojson(bad)
    with pytest.raises(GeometryError):
        shape_from_geojson({"type": "Polygon", "coordinates": [[[0, 0], [1, 1], [0, 0]]]})


# --- compass16 --------------------------------------------------------------

NAMES = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE", "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]
SECTOR = 22.5


@pytest.mark.parametrize("index", range(16))
def test_compass16_every_sector_centre(index):
    assert compass16(index * SECTOR) == NAMES[index]


@pytest.mark.parametrize("index", range(16))
def test_compass16_both_sides_of_every_boundary(index):
    boundary = index * SECTOR + SECTOR / 2
    assert compass16(boundary - 0.01) == NAMES[index]
    assert compass16(boundary + 0.01) == NAMES[(index + 1) % 16]
    assert compass16(boundary) == NAMES[(index + 1) % 16]  # exactly on the boundary: the clockwise side


@pytest.mark.parametrize(
    ("bearing", "name"),
    [(0, "N"), (360, "N"), (-10, "N"), (-11.26, "NNW"), (-90, "W"), (720, "N"), (359.99, "N"), (11.25, "NNE"), (168.6, "SSE"), (180, "S")],
)
def test_compass16_edges_of_the_range(bearing, name):
    assert compass16(bearing) == name


def test_compass16_rejects_a_bearing_that_is_not_a_number():
    for bad in (math.nan, math.inf, -math.inf):
        with pytest.raises(ValueError):
            compass16(bad)


def test_compass8_is_not_shipped():
    assert not hasattr(geo, "compass8")


# --- UK bounds guard (E16) ----------------------------------------------------


@pytest.mark.parametrize(("lat", "lon"), [(49.8, -8.7), (61.0, 2.0), (49.8, 2.0), (61.0, -8.7), BLUE_MILL, WEST_SLEEKBURN])
def test_uk_bounds_accept_exactly_on_a_bound_and_the_two_lzs(lat, lon):
    check_uk_bounds(lat, lon)


@pytest.mark.parametrize(
    ("lat", "lon"),
    [(49.79, -1.0), (61.01, -1.0), (54.0, -8.71), (54.0, 2.01), (0.0, 0.0), (48.85, 2.35), (math.nan, -1.0), (54.0, math.nan), (math.inf, -1.0), (54.0, -math.inf)],
)
def test_uk_bounds_reject_outside_and_non_finite_input(lat, lon):
    with pytest.raises(OutsideUKCoverage, match="^outside UK coverage$"):
        check_uk_bounds(lat, lon)


def test_the_bounds_come_from_rules_yaml(monkeypatch):
    """Moving a bound in the rules moves the guard: the numbers are not in the Python."""
    from types import SimpleNamespace

    from flysafelz.rules import RULES

    tighter = RULES.uk_bounds_lat_max.model_copy(update={"value": 56.0})
    monkeypatch.setattr(geo, "RULES", SimpleNamespace(**{**{n: getattr(RULES, n) for n in type(RULES).model_fields}, "uk_bounds_lat_max": tighter}))
    check_uk_bounds(*WEST_SLEEKBURN)
    with pytest.raises(OutsideUKCoverage):
        check_uk_bounds(56.5, -1.0)
