"""shapes.notam_geometry (I8): the one GeoJSON shape of a NOTAM, byte-identical each time, with approx saying whether it is the Q-circle standing in."""

import ast
import json
import math
from pathlib import Path

import pytest
from factories import coordinate, notam

from flysafelz import shapes
from flysafelz.contracts import Circle
from flysafelz.geo import distance_nm
from flysafelz.notam_area import build_area, forget_areas
from flysafelz.shapes import CIRCLE_VERTICES, airspace_geometry, circle_ring, notam_geometry

Q_CIRCLE = Circle(lat=53.8, lon=-0.0833, radius_nm=2.0)
SQUARE = [(53.80, -0.10), (53.80, -0.07), (53.82, -0.07), (53.82, -0.10)]
POLYGON_TEXT = "UAS OPR WI AREA BOUNDED BY " + " - ".join(coordinate(*c) for c in [*SQUARE, SQUARE[0]]) + "."
CIRCLE_TEXT = "UAS OPR WI 1NM RADIUS OF " + coordinate(53.8, -0.0833) + "."
NO_AREA_TEXT = "UAS OPR SFC-700FT AGL."


def with_e(text: str):
    return notam("H7001/26", item_e=text, circle=Q_CIRCLE)


def ring_of(geometry: dict) -> list[list[float]]:
    assert geometry["type"] == "Polygon" and len(geometry["coordinates"]) == 1
    return geometry["coordinates"][0]


def test_a_circle_in_the_e_text_is_a_polygon_of_the_fixed_vertex_count_and_not_approx():
    parsed = with_e(CIRCLE_TEXT)
    geometry, approx = notam_geometry(parsed)
    assert approx is False
    ring = ring_of(geometry)
    assert len(ring) == CIRCLE_VERTICES + 1 and ring[0] == ring[-1]
    area, _ = build_area(parsed)
    assert isinstance(area, Circle)
    assert all(abs(distance_nm(area.lat, area.lon, lat, lon) - area.radius_nm) < 0.01 * area.radius_nm for lon, lat in ring)


def test_a_polygon_in_the_e_text_keeps_its_corners_as_lon_lat_and_is_closed_and_not_approx():
    geometry, approx = notam_geometry(with_e(POLYGON_TEXT))
    assert approx is False
    ring = ring_of(geometry)
    assert ring[0] == ring[-1]
    for (lat, lon), (x, y) in zip(SQUARE, ring, strict=False):
        assert x == pytest.approx(lon, abs=1e-5) and y == pytest.approx(lat, abs=1e-5)


def test_a_notam_with_no_area_is_its_q_circle_and_approx():
    geometry, approx = notam_geometry(with_e(NO_AREA_TEXT))
    assert approx is True
    ring = ring_of(geometry)
    assert len(ring) == CIRCLE_VERTICES + 1
    assert all(abs(distance_nm(Q_CIRCLE.lat, Q_CIRCLE.lon, lat, lon) - Q_CIRCLE.radius_nm) < 0.02 for lon, lat in ring)


def test_a_parser_miss_is_replaced_by_the_q_circle_and_approx():
    forget_areas()
    parsed = with_e("UAS OPR WI 1NM RADIUS OF " + coordinate(54.5, -0.0833) + ".")  # 40 NM from a 2 NM Q-circle: far past the slack
    area, note = build_area(parsed)
    assert area is None and "parser miss" in note
    geometry, approx = notam_geometry(parsed)
    assert approx is True
    assert all(abs(distance_nm(Q_CIRCLE.lat, Q_CIRCLE.lon, lat, lon) - Q_CIRCLE.radius_nm) < 0.02 for lon, lat in ring_of(geometry))


@pytest.mark.parametrize("text", [NO_AREA_TEXT, CIRCLE_TEXT, POLYGON_TEXT], ids=["q-circle", "circle", "polygon"])
def test_calling_twice_gives_byte_identical_output_even_after_the_area_memory_is_cleared(text):
    first = json.dumps(notam_geometry(with_e(text)))
    assert json.dumps(notam_geometry(with_e(text))) == first
    forget_areas()
    assert json.dumps(notam_geometry(with_e(text))) == first


def test_the_circle_starts_due_north_and_steps_evenly():
    ring = circle_ring(53.8, -0.0833, 1.0)
    assert ring[0][0] == pytest.approx(-0.0833) and ring[0][1] > 53.8
    bearings = [math.degrees(math.atan2((x + 0.0833) * math.cos(math.radians(53.8)), y - 53.8)) % 360 for x, y in ring[:-1]]
    steps = [(b - a) % 360 for a, b in zip(bearings, [*bearings[1:], bearings[0]], strict=True)]
    assert all(step == pytest.approx(360 / CIRCLE_VERTICES, abs=0.1) for step in steps)


def test_airspace_geometry_is_a_polygon_for_one_and_a_multipolygon_for_more_and_closes_rings():
    square = [(53.0, -1.0), (53.0, -0.9), (53.1, -0.9), (53.1, -1.0)]
    one = airspace_geometry([[square]])
    assert one["type"] == "Polygon" and one["coordinates"][0][0] == one["coordinates"][0][-1]
    assert one["coordinates"][0][0] == [-1.0, 53.0]  # [lon, lat]
    two = airspace_geometry([[square], [square]])
    assert two["type"] == "MultiPolygon" and len(two["coordinates"]) == 2


def test_the_shapes_module_imports_nothing_from_agent_or_a_framework():
    tree = ast.parse(Path(shapes.__file__).read_text(encoding="utf-8"))
    modules = {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module} | {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
    banned = {"langchain", "langchain_core", "langgraph", "fastapi", "starlette", "flask"}
    assert not [m for m in modules if m.split(".")[0] in banned or m.startswith("flysafelz.agent")]
