"""get_airspace (G10): over the LZ, within 2 NM and beyond it, checked against what `assess_airspace` itself says for the same inputs."""

import json
import re
from datetime import UTC, datetime

from airspace_helpers import LZ, fl, offset, square, volume
from factories import SFC, airspace_fetch, amsl, pib_result, reading, weather_forecast
from fakes import FakeToolChatModel
from pib_helpers import parsed
from tool_helpers import calls
from tool_world import FIXTURES, WINDOW, World, areas_world, go

from flysafelz.agent.tools import notams_tool
from flysafelz.agent.tools.common import elevation_ft, lowest_pressure
from flysafelz.agent.window import Window
from flysafelz.airspace import Outcome, TimeSpan, assess_airspace
from flysafelz.clients.openaip import Airspace, parse_airspace
from flysafelz.contracts import NotamReading, SourceName, SourceState
from flysafelz.reader.contract import ReaderResult, ReaderStatus
from flysafelz.shapes import notam_geometry
from flysafelz.sources import CHECK_BY_HAND

AIRSPACE = [calls("get_airspace", **WINDOW)]
START, END = datetime(2026, 10, 31, 10, tzinfo=UTC), datetime(2026, 10, 31, 16, tzinfo=UTC)
NEARBY_NM = 1.0  # centre-to-edge offsets below are chosen so the edge is inside 2 NM
OUT_OF_RANGE_NM = 6.0


def page1_areas() -> list[Airspace]:
    items = json.loads((FIXTURES / "openaip" / "page1.json").read_text(encoding="utf-8"))["items"]
    return [parse_airspace(item) for item in items]


def expected(world: World):
    """assess_airspace over the same areas, forecast and window the tool gets: the oracle."""
    forecast = world.forecast or weather_forecast(datetime(2026, 10, 31, 7, tzinfo=UTC), datetime(2026, 11, 8, tzinfo=UTC))
    height, _known = elevation_ft(forecast)
    window = Window(start_utc=START, end_utc=END, words="", assumed=False)
    return assess_airspace(world.airspace.areas, LZ, TimeSpan(start=START, end=END), site_elevation_ft=height, pressure_msl=lowest_pressure(forecast, window), activations=[])


def run_with(*areas: Airspace, **kwargs):
    world = areas_world(*areas, **kwargs)
    return world, go(world, AIRSPACE)


def test_a_volume_over_the_lz_is_listed_with_its_datums_and_over_the_lz():
    world, run = run_with(volume("ctr", name="OVER CTR"))
    item = next(i for i in run.artifact("get_airspace").findings if i.name == "OVER CTR")
    assert item.over_lz is True and item.distance_nm is None
    assert (item.floor, item.ceiling) == ("SFC", "UNL")
    line = next(line for line in run.text("get_airspace").splitlines() if "OVER CTR" in line)
    assert "Floor SFC, ceiling UNL" in line and "Over the LZ" in line
    finding = next(f for f in expected(world).findings if f.area == "OVER CTR")
    assert (item.outcome, item.reason) == (finding.outcome.value, finding.reason)
    assert item.reason in line


def assert_one_distance(line: str, distance_nm: float | None) -> None:
    """The line states the volume's distance once, at the finding's two decimals, and the artifact has that same number."""
    assert distance_nm is not None
    stated = re.findall(r"(\d+(?:\.\d+)?) NM", line)
    assert stated == [f"{distance_nm:.2f}"], line


def test_a_volume_within_2_nm_has_its_distance_and_bearing_from_the_finding():
    near = square(*offset(*LZ, 90.0, NEARBY_NM + 0.5), 0.5)
    world, run = run_with(volume("tma", name="EAST TMA", lower=fl(30), shape=near))
    item = next(i for i in run.artifact("get_airspace").findings if i.name == "EAST TMA")
    finding = next(f for f in expected(world).findings if f.area == "EAST TMA")
    assert item.over_lz is False
    assert item.distance_nm is not None and 0.5 < item.distance_nm < 2.0
    assert (item.bearing_deg, item.compass) == (finding.bearing_deg, finding.compass)
    assert (item.floor, item.floor_qnh_ft, item.ceiling_qnh_ft) == ("FL30", finding.floor_qnh_ft, finding.ceiling_qnh_ft)
    line = next(line for line in run.text("get_airspace").splitlines() if "EAST TMA" in line)
    assert finding.bearing_deg is not None
    assert f"Bearing {finding.bearing_deg:.0f}° ({finding.compass})" in line
    assert_one_distance(line, item.distance_nm)


def test_a_volume_beyond_2_nm_is_not_listed_and_the_none_line_says_so():
    far = square(*offset(*LZ, 90.0, OUT_OF_RANGE_NM), 0.5)
    world, run = run_with(volume("ctr", name="FAR CTR", shape=far))
    assert all(f.area != "FAR CTR" or f.outcome is Outcome.IGNORE for f in expected(world).findings)
    assert run.artifact("get_airspace").findings == []
    assert "Findings: none." in run.text("get_airspace")


def test_an_empty_airspace_source_prints_none_lines_not_blank_sections():
    items = json.loads((FIXTURES / "openaip" / "empty.json").read_text(encoding="utf-8"))["items"]
    assert items == []
    run = go(World(airspace=airspace_fetch(*(parse_airspace(item) for item in items))), AIRSPACE)
    text = run.text("get_airspace")
    assert "Findings: none." in text
    assert "Lowest ceiling over the LZ (ceiling_qnh_ft): none." in text
    assert "Notes: none." in text


def test_ceiling_qnh_ft_and_the_notes_are_printed_as_assess_airspace_gives_them():
    world, run = run_with(volume("tma", name="LOW TMA", lower=fl(10), shape=square(*LZ, 0.5)))
    truth = expected(world)
    artifact = run.artifact("get_airspace")
    assert artifact.ceiling_qnh_ft == truth.ceiling_qnh_ft
    assert artifact.notes == truth.notes
    text = run.text("get_airspace")
    for note in truth.notes:
        assert f"- {note}" in text
    if truth.ceiling_qnh_ft is not None:
        assert f"(ceiling_qnh_ft): {truth.ceiling_qnh_ft:g} ft QNH." in text


def test_page1_of_the_openaip_fixture_gives_the_same_findings_as_assess_airspace():
    world, run = run_with(*page1_areas())
    truth = {f.area: f for f in expected(world).findings if f.outcome is not Outcome.IGNORE}
    listed = {i.name: i for i in run.artifact("get_airspace").findings}
    assert truth, "the fixture has airspace near Blue Mill"
    # EGD496A is activated by a NOTAM, which this PIB-less run does not have: it is UNVERIFIED in both.
    assert set(listed) == set(truth)
    for name, finding in truth.items():
        assert (listed[name].outcome, listed[name].reason, listed[name].bearing_deg) == (finding.outcome.value, finding.reason, finding.bearing_deg)
    assert "EDINBURGH CTR" not in run.text("get_airspace")


def test_the_artifact_has_a_feature_per_listed_volume_and_the_lz_circle():
    _, run = run_with(*page1_areas())
    artifact = run.artifact("get_airspace")
    kinds = [f.properties.kind for f in artifact.overlays]
    assert kinds[0] == "lz_radius" and kinds[1:] == ["airspace"] * len(artifact.findings)
    assert {f.properties.id for f in artifact.overlays[1:]} == {i.name for i in artifact.findings}
    assert all(f.properties.text is None for f in artifact.overlays)
    assert all(f.geometry.type in {"Polygon", "MultiPolygon"} for f in artifact.overlays[1:])


def test_a_failed_airspace_source_is_named_with_the_manual_check_and_lists_nothing():
    run = go(World(airspace=airspace_fetch(state=SourceState.FAILED)), AIRSPACE)
    text = run.text("get_airspace")
    assert "FAILED" in text and CHECK_BY_HAND[SourceName.AIRSPACE] in text
    assert "Findings" not in text
    assert run.artifact("get_airspace").findings == []


def test_it_writes_the_fixed_airspace_line_and_the_fetch_time_in_the_zone():
    run = go(areas_world(), AIRSPACE)
    assert run.progress == ["Checking airspace…"]
    assert "Fetched: 2026-10-31 07:30 GMT." in run.text("get_airspace")


# --- a danger area openAIP does not have, placed by the NOTAM that activates it ----------------------------------

TDA = "TEMPO DANGER AREA (TDA) EGD496A ACTIVATED. SFC-1300FT AMSL."


def activated_by(notam_lat: float, notam_lon: float, radius_nm: float):
    notam = parsed("H7201/26", subject="RD", centre_lat=notam_lat, centre_lon=notam_lon, radius_nm=radius_nm, item_e=TDA)
    return notam, go(World(pib=pib_result(notam)), AIRSPACE)


def test_an_activated_danger_area_openaip_lacks_is_placed_by_its_notam_beside_the_lz():
    notam, run = activated_by(*offset(*LZ, 90.0, 1.5), 0.5)  # its Q-circle's edge is 1 NM east
    item = next(i for i in run.artifact("get_airspace").findings if i.name == "EGD496A")
    assert (item.type, item.from_notam, item.approx) == ("danger", "H7201/26", True)
    assert item.over_lz is False and item.distance_nm is not None and 0.8 < item.distance_nm < 1.2
    assert item.bearing_deg is not None and 85 < item.bearing_deg < 95
    line = next(line for line in run.text("get_airspace").splitlines() if "EGD496A" in line and line.startswith("- EGD496A"))
    assert f"Bearing {item.bearing_deg:.0f}° ({item.compass})" in line and "Q-circle (approximate)" in line
    assert_one_distance(line, item.distance_nm)
    feature = next(f for f in run.artifact("get_airspace").overlays if f.properties.id == "EGD496A")
    geometry, approx = notam_geometry(notam)
    assert approx is True and feature.geometry.model_dump(mode="json") == geometry  # the one NOTAM shape
    assert feature.properties.kind == "airspace" and feature.properties.action == item.outcome and feature.properties.bearing == item.bearing_deg


def test_an_activated_danger_area_openaip_lacks_is_over_the_lz_when_the_finding_says_so():
    _, run = activated_by(*LZ, 0.5)
    item = next(i for i in run.artifact("get_airspace").findings if i.name == "EGD496A")
    assert item.over_lz is True and item.distance_nm is None and item.bearing_deg is None
    line = next(line for line in run.text("get_airspace").splitlines() if line.startswith("- EGD496A"))
    assert "Over the LZ" in line
    assert [f.properties.id for f in run.artifact("get_airspace").overlays][-1] == "EGD496A"


def test_an_unread_activating_notam_is_said_not_to_have_been_read_not_to_state_no_limits():
    _, run = activated_by(*LZ, 0.5)
    item = next(i for i in run.artifact("get_airspace").findings if i.name == "EGD496A")
    assert (item.from_notam, item.notam_read, item.floor, item.ceiling) == ("H7201/26", False, None, None)
    line = next(line for line in run.text("get_airspace").splitlines() if line.startswith("- EGD496A"))
    assert "Floor and ceiling: NOTAM H7201/26 was not read, see its raw text (get_notams); the floor is taken as SFC." in line
    assert "not stated" not in line  # the E text does state them: the tool just has not read it


def read_then_airspace(monkeypatch, tmp_path, read: NotamReading):
    """The thread reads the activating NOTAM with get_notams (the reader answers `read`), then asks for the airspace."""

    def reader(parsed_notams, lz, window, llm, store, *, now=None):
        return {p.id: ReaderResult(notam_id=p.id, reading=read, status=ReaderStatus.READ, failed_checks=[], model="fake", prompt_version="x", latency_s=0.0, tokens=None) for p in parsed_notams}

    monkeypatch.setattr(notams_tool, "read_for_lz", reader)
    notam = parsed("H7201/26", subject="RD", centre_lat=LZ[0], centre_lon=LZ[1], radius_nm=0.5, item_e=TDA)
    steps = [calls("get_notams", **WINDOW), calls("get_airspace", **WINDOW)]
    run = go(World(pib=pib_result(notam)), steps, reader_model=lambda: FakeToolChatModel(script=[]), reader_store=tmp_path / "readings.sqlite")
    return next(i for i in run.artifact("get_airspace").findings if i.name == "EGD496A"), run


def test_a_read_activating_notam_gives_the_limits_the_reading_states(monkeypatch, tmp_path):
    item, run = read_then_airspace(monkeypatch, tmp_path, reading(notam_id="H7201/26", value=None, designators=["EGD496A"], floor=SFC, ceiling=amsl(1300)))
    assert (item.notam_read, item.floor, item.ceiling) == (True, "SFC", "1300 ft AMSL")
    assert "Floor SFC, ceiling 1300 ft AMSL (as NOTAM H7201/26 states them)." in run.text("get_airspace")


def test_not_stated_is_kept_for_a_read_notam_whose_reading_has_no_ceiling(monkeypatch, tmp_path):
    read = reading(notam_id="H7201/26", value=None, designators=["EGD496A"], floor=SFC).model_copy(update={"ceiling": None})
    item, run = read_then_airspace(monkeypatch, tmp_path, read)
    assert (item.notam_read, item.floor, item.ceiling) == (True, "SFC", None)
    assert "Floor SFC, ceiling not stated (as NOTAM H7201/26 states them)." in run.text("get_airspace")
