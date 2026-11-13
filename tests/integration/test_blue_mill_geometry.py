"""E21: the Blue Mill geometry, measured from the committed fixtures with geo.py.

The LZ sits inside H7171/26's coded 1 NM Q-circle, 0.15 NM outside the 0.5 NM circle its E text gives (centre 0.65 NM away).

Not checkable: "the LZ is inside the EGD496A polygon, about 95 m from its edge". The committed airspace.json (a keyed
openAIP capture) holds no EGD496A polygon near Blue Mill, so there is no polygon to measure. What the fixtures do hold
is J3857/26's coded Q-circle, which contains the LZ, and that is what assess falls back on (the assumption it prints).
test_egd496a_has_no_polygon_in_the_committed_capture fails if a recapture ever adds one, so this test gets updated then.
"""

import cases
import pytest

from flysafelz.clients.openaip import Airspace
from flysafelz.contracts import Circle
from flysafelz.geo import circle_edge_distance_nm, circle_over, distance_nm
from flysafelz.pib import parse_pib

LZ = (53.802797, -0.086644)


def notams():
    _issued, parsed = parse_pib((cases.case_dir(cases.BLUE_MILL) / "pib.xml").read_bytes())
    return {n.id: n for n in parsed}


def drone_circle() -> Circle:
    _case, _forecast, _airspace, _pib, readings = cases.load(cases.BLUE_MILL)
    geometry = readings["H7171/26"].geometry
    assert isinstance(geometry, Circle)
    return geometry


def test_the_case_lz_is_the_one_the_plan_names():
    case, *_rest = cases.load(cases.BLUE_MILL)
    assert (case.lat, case.lon) == LZ


def test_the_lz_is_inside_h7171s_coded_one_nm_circle():
    coded = notams()["H7171/26"]
    assert coded.radius_nm == 1
    assert circle_over(*LZ, coded.centre_lat, coded.centre_lon, coded.radius_nm)


def test_the_lz_is_015_nm_outside_the_half_nm_e_text_circle_whose_centre_is_065_nm_away():
    drone = drone_circle()
    assert drone.radius_nm == 0.5
    assert not circle_over(*LZ, drone.lat, drone.lon, drone.radius_nm)
    assert distance_nm(*LZ, drone.lat, drone.lon) == pytest.approx(0.65, abs=0.02)
    assert circle_edge_distance_nm(*LZ, drone.lat, drone.lon, drone.radius_nm) == pytest.approx(0.15, abs=0.02)


def test_the_lz_is_inside_j3857s_coded_q_circle_which_is_what_stands_in_for_egd496a():
    coded = notams()["J3857/26"]
    assert coded.radius_nm == 10
    assert circle_over(*LZ, coded.centre_lat, coded.centre_lon, coded.radius_nm)


def names_egd496a(area: Airspace) -> bool:
    """True when the area is EGD496A by its designator or by its name. openAIP often leaves the designator empty and puts
    the identity in the name ('EASINGTON'), so both are read, with the spaces and dashes taken out ('EG D 496A', 'D496A')."""
    squashed = "".join(ch for text in (area.designator, area.name) for ch in (text or "").upper() if ch.isalnum())
    return "D496A" in squashed


@pytest.mark.parametrize(
    ("designator", "name", "expected"),
    [("EGD496A", None, True), (None, "EGD496A BLUE MILL", True), (None, "D 496A", True), (None, "EG-D496A", True), (None, "EASINGTON", False), ("EGD496B", "D496B", False), (None, None, False)],
)
def test_the_guard_finds_egd496a_by_designator_or_by_name(designator, name, expected):
    _case, _forecast, airspace, _pib, _readings = cases.load(cases.BLUE_MILL)
    area = airspace.areas[0].model_copy(update={"designator": designator, "name": name})
    assert names_egd496a(area) is expected


def test_egd496a_has_no_polygon_in_the_committed_capture():
    _case, _forecast, airspace, _pib, _readings = cases.load(cases.BLUE_MILL)
    areas: list[Airspace] = airspace.areas
    assert not any(names_egd496a(a) for a in areas)
