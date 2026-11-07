"""E20, E21 and the committed cases: the two fixtures in fixtures/ load offline and hold what the plan says.

Everything here reads committed files. Sockets are disabled for the whole suite, so a parse that
tried to fetch anything would fail the run.

airspace.json was captured by Vector with the openAIP key (make fixtures). Both files are committed, so a
missing one is a failure here, not a skip.
"""

import json
import re
from datetime import UTC, datetime
from pathlib import Path

import httpx
import pytest
import yaml

from flysafelz.capture import load_case
from flysafelz.clients.open_meteo import parse_response
from flysafelz.clients.openaip import KEY_HEADER, KEY_VARIABLE, Airspace, fetch_airspace
from flysafelz.contracts import Circle, NotamReading, SourceState
from flysafelz.geo import circle_edge_distance_nm, circle_over, distance_nm, shape_over
from flysafelz.pib import parse_pib
from flysafelz.weather import REQUIRED

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
BLUE_MILL = "blue_mill_2026-10-31"
WEST_SLEEKBURN = "west_sleekburn_2026-10-31"
CASES = [BLUE_MILL, WEST_SLEEKBURN]

SHA = "bdab14658f0270dc41460eebcd03332abfb230ec"
TODAY = datetime(2026, 11, 6, 12, 0, tzinfo=UTC)  # frozen
LZ = (53.802797, -0.086644)
FILES = ["case.yaml", "weather.json", "airspace.json", "pib.xml", "readings.yaml"]


def case_dir(name: str) -> Path:
    return FIXTURES / name


def airspaces(name: str) -> list[Airspace]:
    """The case's openAIP airspaces, read through PR-5's client around the case's own LZ."""
    path = case_dir(name) / "airspace.json"
    assert path.is_file(), f"{path} is missing: capture it with make fixtures CASE={name}"
    body = json.loads(path.read_text(encoding="utf-8"))
    case, _raw = load_case(case_dir(name) / "case.yaml", today=TODAY)
    client = httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(200, json=body)))
    got = fetch_airspace(case.lat, case.lon, api_key="not-a-real-key", client=client, now=TODAY)
    assert got.status.state is SourceState.OK, got.status.reason
    return got.areas


def readings(name: str) -> list[NotamReading]:
    data = yaml.safe_load((case_dir(name) / "readings.yaml").read_text(encoding="utf-8"))
    assert isinstance(data, list)
    return [NotamReading.model_validate(item) for item in data]


# --- both cases load offline (E20) -----------------------------------------------------------------------------------


@pytest.mark.parametrize("name", CASES)
@pytest.mark.parametrize("file", FILES)
def test_every_file_exists(name, file):
    assert (case_dir(name) / file).is_file()


@pytest.mark.parametrize("name", CASES)
def test_case_yaml_weather_and_pib_parse_through_the_client_parsers(name):
    case, _raw = load_case(case_dir(name) / "case.yaml", today=TODAY)
    forecast = parse_response(json.loads((case_dir(name) / "weather.json").read_text(encoding="utf-8")), case.now)
    assert forecast.status.state is SourceState.OK and forecast.elevation_m is not None
    issued, notams = parse_pib((case_dir(name) / "pib.xml").read_bytes())
    assert case.capture is not None and issued == case.capture.pib_issued and notams
    readings(name)


@pytest.mark.parametrize("name", CASES)
def test_airspace_json_parses_through_the_openaip_client(name):
    areas = airspaces(name)
    assert areas
    # the committed file is already the cut-down capture: the client keeps every item in it
    assert len(areas) == len(json.loads((case_dir(name) / "airspace.json").read_text(encoding="utf-8"))["items"])
    assert all(a.shape for a in areas)


@pytest.mark.parametrize("name", CASES)
def test_the_case_day_and_window_and_the_pinned_sha(name):
    case, _raw = load_case(case_dir(name) / "case.yaml", today=TODAY)
    assert case.now == datetime(2026, 10, 31, 7, 30, tzinfo=UTC)
    assert (case.window.start, case.window.end) == (datetime(2026, 10, 31, 7, tzinfo=UTC), datetime(2026, 10, 31, 18, tzinfo=UTC))
    assert case.archive_sha == SHA
    assert case.capture is not None and case.capture.sha == SHA


@pytest.mark.parametrize("name", CASES)
def test_weather_has_every_required_variable_for_the_window_plus_next_day_midnight(name):
    case, _raw = load_case(case_dir(name) / "case.yaml", today=TODAY)
    forecast = parse_response(json.loads((case_dir(name) / "weather.json").read_text(encoding="utf-8")), case.now)
    wanted = [datetime(2026, 10, 31, hour, tzinfo=UTC) for hour in range(7, 19)] + [datetime(2026, 11, 1, 0, tzinfo=UTC)]
    assert len(wanted) == 13  # 07:00 to 18:00 inclusive (the 17:00 row needs 18:00), and the 23:00 row's closing 00:00
    for stamp in wanted:
        assert stamp in forecast.samples, stamp
        for variable in REQUIRED:
            assert getattr(forecast.samples[stamp], variable) is not None, (stamp, variable)
    assert sorted(forecast.samples) == [datetime(2026, 10, 31, h, tzinfo=UTC) for h in range(24)] + [datetime(2026, 11, 1, tzinfo=UTC)]


@pytest.mark.parametrize("name", CASES)
def test_no_fixture_holds_the_key_variable_or_a_request_header(name):
    for path in case_dir(name).iterdir():
        text = path.read_text(encoding="utf-8")
        assert KEY_HEADER not in text.lower(), path.name
        assert KEY_VARIABLE not in text, path.name


def test_fixtures_dir_holds_only_the_two_cases_and_nothing_with_a_key():
    assert sorted(p.name for p in FIXTURES.iterdir()) == sorted(CASES)
    for path in FIXTURES.rglob("*"):
        assert path.is_dir() or path.name in FILES, path


# --- Blue Mill ---------------------------------------------------------------------------------------------------------


def test_blue_mill_case_yaml_values():
    case, _raw = load_case(case_dir(BLUE_MILL) / "case.yaml", today=TODAY)
    assert (case.lat, case.lon) == LZ
    assert case.capture is not None and case.capture.pib_issued == datetime(2026, 10, 31, 7, 0, 37, tzinfo=UTC)


def test_blue_mill_pib_is_the_07_00_issue_and_has_the_three_notams():
    issued, notams = parse_pib((case_dir(BLUE_MILL) / "pib.xml").read_bytes())
    assert issued == datetime(2026, 10, 31, 7, 0, 37, tzinfo=UTC)
    ids = {n.id for n in notams}
    assert {"J3857/26", "J3864/26", "H7171/26"} <= ids


def test_the_pib_is_trimmed_to_25_nm():
    _issued, notams = parse_pib((case_dir(BLUE_MILL) / "pib.xml").read_bytes())
    assert len(notams) < 100  # the full PIB holds well over a thousand
    for n in notams:
        assert circle_edge_distance_nm(*LZ, n.centre_lat, n.centre_lon, n.radius_nm) <= 25, n.id


def test_blue_mill_readings_validate_as_notam_readings_and_hold_the_two_plan_values():
    by_id = {r.id: r for r in readings(BLUE_MILL)}
    assert set(by_id) == {"J3857/26", "H7171/26"}

    tda = by_id["J3857/26"]
    assert tda.category.value == "cooperative"
    assert tda.activates_designators == ["EGD496A"]
    assert (tda.contact.kind.value, tda.contact.value) == ("phone", "Flylogix 07784 546062")
    assert tda.prohibitive_reason.value == "none"
    assert tda.schedule == "0800-1800"
    assert (tda.floor.datum.value, tda.ceiling.value, tda.ceiling.datum.value) == ("SFC", 1300, "AMSL")

    drone = by_id["H7171/26"]
    assert drone.activity == "drone ops"
    assert isinstance(drone.geometry, Circle) and drone.geometry.radius_nm == 0.5
    assert (drone.floor.datum.value, drone.ceiling.value, drone.ceiling.unit.value, drone.ceiling.datum.value) == ("SFC", 700, "ft", "AGL")
    assert drone.contact.kind.value == "none"


def test_e21_the_lz_is_inside_h7171s_coded_circle_and_just_outside_its_e_text_circle():
    _issued, notams = parse_pib((case_dir(BLUE_MILL) / "pib.xml").read_bytes())
    coded = next(n for n in notams if n.id == "H7171/26")
    assert coded.radius_nm == 1
    assert circle_over(*LZ, coded.centre_lat, coded.centre_lon, coded.radius_nm)

    drone = next(r for r in readings(BLUE_MILL) if r.id == "H7171/26").geometry
    assert isinstance(drone, Circle)
    # the E text says: 0.5 NM radius of 534732N 0000459W
    assert drone.lat == pytest.approx(53 + 47 / 60 + 32 / 3600, abs=1e-6)
    assert drone.lon == pytest.approx(-(4 / 60 + 59 / 3600), abs=1e-6)
    assert not circle_over(*LZ, drone.lat, drone.lon, drone.radius_nm)
    assert distance_nm(*LZ, drone.lat, drone.lon) == pytest.approx(0.65, abs=0.01)
    assert circle_edge_distance_nm(*LZ, drone.lat, drone.lon, drone.radius_nm) == pytest.approx(0.15, abs=0.01)


def test_e21_egd496a_is_not_in_blue_mills_airspace_so_the_q_circle_fallback_applies():
    """The keyed capture found no EGD496A polygon near Blue Mill, so no 95 m edge distance can be checked.

    Its danger areas are EGD323K and EGD323D. J3857/26 (the TDA that activates EGD496A) is then judged on
    its coded Q-circle, which holds the LZ.
    """
    areas = airspaces(BLUE_MILL)
    assert not any(a.designator == "EGD496A" for a in areas)
    assert {a.designator for a in areas if a.type == "danger"} == {"EGD323K", "EGD323D"}
    _issued, notams = parse_pib((case_dir(BLUE_MILL) / "pib.xml").read_bytes())
    tda = next(n for n in notams if n.id == "J3857/26")
    assert circle_over(*LZ, tda.centre_lat, tda.centre_lon, tda.radius_nm)


@pytest.mark.parametrize(
    ("name", "designators", "notam_set"),
    [(BLUE_MILL, {"EGD323K", "EGD323D"}, {"EGD323K"}), (WEST_SLEEKBURN, {"EGD514"}, set())],
)
def test_every_danger_area_has_an_activation_field_set(name, designators, notam_set):
    """byNotam or hoursOfOperation is set on every danger area in both captures (0 empty)."""
    danger = [a for a in airspaces(name) if a.type == "danger"]
    assert {a.designator for a in danger} == designators
    assert all(a.by_notam or a.hours for a in danger)
    assert {a.designator for a in danger if a.by_notam} == notam_set


# --- West Sleekburn -------------------------------------------------------------------------------------------------------


def test_west_sleekburn_case_yaml_and_empty_readings():
    case, _raw = load_case(case_dir(WEST_SLEEKBURN) / "case.yaml", today=TODAY)
    assert (case.lat, case.lon) == (55.1576, -1.5699)
    assert yaml.safe_load((case_dir(WEST_SLEEKBURN) / "readings.yaml").read_text(encoding="utf-8")) == []
    assert re.match(r"#", (case_dir(WEST_SLEEKBURN) / "readings.yaml").read_text(encoding="utf-8"))  # header comment


def test_west_sleekburn_pib_is_trimmed_around_its_own_lz():
    case, _raw = load_case(case_dir(WEST_SLEEKBURN) / "case.yaml", today=TODAY)
    _issued, notams = parse_pib((case_dir(WEST_SLEEKBURN) / "pib.xml").read_bytes())
    assert notams
    for n in notams:
        assert circle_edge_distance_nm(case.lat, case.lon, n.centre_lat, n.centre_lon, n.radius_nm) <= 25, n.id


def test_the_newcastle_cta_lies_over_the_west_sleekburn_lz_with_a_floor_of_1500_ft_amsl():
    case, _raw = load_case(case_dir(WEST_SLEEKBURN) / "case.yaml", today=TODAY)
    over = {a.name: a for a in airspaces(WEST_SLEEKBURN) if shape_over(case.lat, case.lon, a.shape)}
    assert set(over) == {"NEWCASTLE CTA 2", "BORDERS CTA 12"}
    cta = over["NEWCASTLE CTA 2"]
    assert (cta.lower.value, cta.lower.unit.value, cta.lower.datum.value) == (1500, "ft", "AMSL")
    assert (cta.upper.value, cta.upper.unit.value) == (105, "FL")
    # the CTR is beside the LZ, not over it
    assert "NEWCASTLE CTR" not in over
