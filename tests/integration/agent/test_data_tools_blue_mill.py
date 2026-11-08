"""The three data tools in a real graph on the Blue Mill case: what the model can pass, what it gets back, what the pilot's UI gets, and what neither may see."""

import json
import re
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest
from tool_helpers import ALL_THREE, BLUE_MILL_LZ, BLUE_MILL_NOW, BLUE_MILL_WINDOW, ToolRun, calls, run_tools

from flysafelz.agent.input_check import RequestInvalid
from flysafelz.agent.sources import SourceCache
from flysafelz.agent.state import RunContext
from flysafelz.agent.tools.airspace_tool import build_get_airspace
from flysafelz.agent.tools.common import NO_LANDING_ZONE, Redactor, prepare
from flysafelz.agent.tools.notams_tool import TEXT_BEGIN, build_get_notams
from flysafelz.agent.tools.weather_tool import build_get_weather
from flysafelz.agent.window import TOO_FAR_OUT
from flysafelz.contracts import SourceState
from flysafelz.redact import redact_phones
from flysafelz.shapes import notam_geometry
from flysafelz.sources import load_case_sources

CTX = RunContext(now=BLUE_MILL_NOW, case="blue_mill_2026-10-31", reader="fixture")


def blue_mill(steps=None, **kwargs) -> ToolRun:
    return run_tools(SourceCache(), steps or [calls(*ALL_THREE, **BLUE_MILL_WINDOW)], ctx=CTX, **kwargs)


@pytest.mark.parametrize("builder", [build_get_weather, build_get_airspace, build_get_notams], ids=ALL_THREE)
def test_the_schema_the_model_sees_has_a_window_and_no_position_zone_clock_or_case(builder):
    tool = builder(SourceCache())
    schema = tool.tool_call_schema.model_json_schema()
    assert set(schema["properties"]) == {"start", "end"}
    assert tool.description


@pytest.mark.parametrize("name", ALL_THREE)
@pytest.mark.parametrize("extra", [{"lat": 51.0}, {"lon": 0.0}, {"tz": "UTC"}, {"now": "2026-10-31T00:00:00Z"}, {"case": "other"}])
def test_a_model_that_passes_a_position_zone_clock_or_case_is_refused_and_nothing_is_fetched(name, extra):
    fetched = []

    def never(*args):
        fetched.append(args)
        raise AssertionError("fetched")

    run = run_tools(SourceCache(fetch=never), [[(name, {**BLUE_MILL_WINDOW, **extra})]], ctx=RunContext(now=BLUE_MILL_NOW))
    message = run.of(name)[0]
    assert message.status == "error" and fetched == []
    assert next(iter(extra)) in str(message.content)  # it names the argument it refused


def test_the_tools_use_the_threads_landing_zone_not_one_the_model_supplies():
    run = blue_mill()
    for name in ALL_THREE:
        centre = run.artifact(name).overlays[0].geometry.coordinates[0][0]
        assert abs(centre[1] - BLUE_MILL_LZ["lat"]) < 0.05 and abs(centre[0] - BLUE_MILL_LZ["lon"]) < 0.05


def test_a_request_with_no_landing_zone_never_reaches_a_tool():
    with pytest.raises(RequestInvalid, match="lat is required"):
        run_tools(SourceCache(), [calls("get_weather", **BLUE_MILL_WINDOW)], ctx=CTX, lz={})


def test_a_tool_that_somehow_has_no_landing_zone_says_so_and_fetches_nothing():
    def never(*args):
        raise AssertionError("fetched")

    runtime = SimpleNamespace(state={}, context=CTX)
    answer = prepare(runtime, SourceCache(fetch=never), Redactor(None), "2026-10-31 10:00", "2026-10-31 16:00")  # type: ignore[arg-type]
    assert answer == NO_LANDING_ZONE


def test_blue_mill_three_tools_answer_in_one_step_with_one_artifact_each():
    run = blue_mill()
    assert sorted(m.name for m in run.messages) == sorted(ALL_THREE)
    assert {run.artifact(n).tool for n in ALL_THREE} == set(ALL_THREE)
    assert all(run.artifact(n).source.state == SourceState.OK for n in ALL_THREE)
    assert min(run.progress).startswith("Checking") and any(line.startswith("Reading") for line in run.progress)


def test_blue_mill_notams_are_listed_with_the_cases_readings_and_no_model_is_built():
    built = []

    def never_built():
        built.append(1)
        raise AssertionError("the model was built")

    run = run_tools(SourceCache(reader_model=never_built), [calls("get_notams", **BLUE_MILL_WINDOW)], ctx=CTX)
    ids = {n.id for n in run.artifact("get_notams").notams}
    assert {"H7171/26", "J3857/26"} <= ids and built == []
    assert all(n.reading is not None for n in run.artifact("get_notams").notams if n.action != "AWARENESS")
    assert TEXT_BEGIN in run.text("get_notams")


def test_blue_mill_every_phone_number_is_a_placeholder_in_the_text_the_artifact_and_the_state_keeps_them():
    run = blue_mill()
    numbers = list(run.state.get("phone_map", {}).values())
    assert numbers, "Blue Mill's NOTAMs name phone numbers"
    for name in ALL_THREE:
        seen = run.text(name) + json.dumps(run.artifact(name).model_dump(mode="json"))
        compact = seen.replace(" ", "")
        assert all(number not in seen and number.replace(" ", "") not in compact for number in numbers)
    assert all(key.startswith("<PHONE_") for key in run.state["phone_map"])


def test_the_same_run_twice_gives_the_same_text():
    first, second = blue_mill(), blue_mill(thread_id="other-thread")
    for name in ALL_THREE:
        assert first.text(name) == second.text(name)


def test_a_window_after_the_cases_forecast_is_relayed_not_fetched():
    run = blue_mill([calls("get_weather", start="2027-01-01 10:00", end="2027-01-01 16:00")])
    assert run.artifact("get_weather") is None and run.text("get_weather")


# --- the golden-path checkpoints and the case-mode edges ---------------------------------------------------


def counting_loader():
    seen: list[str] = []

    def load(name, **kwargs):
        seen.append(name)
        return load_case_sources(name, **kwargs)

    return load, seen


def test_gp3_the_progress_stream_shows_weather_then_the_two_reader_lines_in_order():
    one_at_a_time = [calls("get_weather", **BLUE_MILL_WINDOW), calls("get_airspace", **BLUE_MILL_WINDOW), calls("get_notams", **BLUE_MILL_WINDOW)]
    run = blue_mill(one_at_a_time)
    wanted = ["Checking weather…", "Reading 2 NOTAMs (1/2)…", "Reading 2 NOTAMs (2/2)…"]
    assert [line for line in run.progress if line in wanted] == wanted
    assert run.progress == ["Checking weather…", "Checking airspace…", *wanted[1:]]


def test_gp6_a_second_get_notams_in_the_thread_reads_nothing_and_keeps_the_numbering():
    run = blue_mill([calls("get_notams", **BLUE_MILL_WINDOW), calls("get_notams", **BLUE_MILL_WINDOW)])
    assert [line for line in run.progress if line.startswith("Reading")] == ["Reading 2 NOTAMs (1/2)…", "Reading 2 NOTAMs (2/2)…"]  # the first call's, none from the second
    first, second = run.text("get_notams", 0), run.text("get_notams", 1)
    assert first == second and "<PHONE_1>" in second and "<PHONE_2>" not in second
    assert run.state["phone_map"].keys() == {"<PHONE_1>"}
    j3857 = next(n for n in run.artifact("get_notams", 1).notams if n.id == "J3857/26")
    assert j3857.action == "CONTACT_FIRST"
    assert j3857.reading is not None
    assert (j3857.reading.category, j3857.reading.activity, j3857.reading.activates) == ("cooperative", "UAS BVLOS ops", ["EGD496A"])
    assert "TEMPO DANGER AREA (TDA) EGD496A ACTIVATED" in j3857.e_text and "<PHONE_1>" in j3857.e_text


def test_blue_mill_says_2_notams_within_2_nm_and_the_gmt_labels_and_limits_beside_values():
    run = blue_mill()
    assert "2 NOTAMs within 2 NM." in run.text("get_notams") and "ignored" in run.text("get_notams")
    weather = run.text("get_weather")
    assert "Sat 31 Oct 10:00 GMT:" in weather and "BST" not in weather
    assert "wind at 10 m 4.9 mph (caution >= 12 mph: under the 12 mph limit, no-go > 16 mph: under the 16 mph limit)" in weather


def test_the_real_contact_number_is_in_no_text_and_no_artifact_but_is_in_phone_map():
    """The number is read from the fixture here, at test time; it is never written in this file."""
    pib = load_case_sources("blue_mill_2026-10-31").pib
    j3857 = next(n for n in pib.notams if n.id == "J3857/26")
    _, numbers = redact_phones(j3857.item_e)
    assert numbers, "J3857/26 names a phone number"
    run = blue_mill()
    for name in ALL_THREE:
        shown = run.text(name) + json.dumps(run.artifact(name).model_dump(mode="json"))
        for number in numbers.values():
            assert number not in shown and re.sub(r"\D", "", number) not in re.sub(r"\D", "", shown)
    assert run.state["phone_map"] == {"<PHONE_1>": next(iter(numbers.values()))}


def test_each_source_is_loaded_once_when_the_three_tools_run_in_one_turn_and_again_in_the_next():
    load, seen = counting_loader()
    run = run_tools(SourceCache(load_case=load), [calls(*ALL_THREE, **BLUE_MILL_WINDOW), calls(*ALL_THREE, **BLUE_MILL_WINDOW)], ctx=CTX)
    assert seen == ["blue_mill_2026-10-31"]  # one load serves the six calls
    assert len({run.artifact(name, index).source.fetched_at for name in ALL_THREE for index in (0, 1)}) == 1
    assert len(run.messages) == 6


def test_a_window_wholly_beyond_the_horizon_is_relayed_as_the_window_error_and_nothing_is_loaded():
    load, seen = counting_loader()
    run = run_tools(SourceCache(load_case=load), [calls(*ALL_THREE, start="2026-11-20 10:00", end="2026-11-20 16:00")], ctx=CTX)
    assert seen == [] and run.progress == []
    for name in ALL_THREE:
        assert run.text(name) == TOO_FAR_OUT and run.artifact(name) is None


def test_a_window_across_the_clock_change_is_labelled_bst_then_gmt_and_notam_text_keeps_its_published_utc():
    ctx = RunContext(now=datetime(2026, 10, 24, 20, tzinfo=UTC), case="blue_mill_2026-10-31", reader="fixture")
    run = run_tools(SourceCache(), [calls("get_weather", start="2026-10-25 00:00", end="2026-10-25 03:00")], ctx=ctx)
    labels = [hour.label for hour in run.artifact("get_weather").hours]
    assert labels[0].endswith("BST") and labels[-1].endswith("GMT") and {label.rsplit(" ", 1)[1] for label in labels} == {"BST", "GMT"}
    notams = blue_mill()
    published = next(n for n in load_case_sources("blue_mill_2026-10-31").pib.notams if n.id == "J3857/26").item_e
    assert "128.925MHZ" in published and "128.925MHZ" in " ".join(n.e_text for n in notams.artifact("get_notams").notams)


def test_the_notam_features_carry_the_published_text_and_the_others_carry_none():
    run = blue_mill()
    features = {f.properties.id: f for f in run.artifact("get_notams").overlays if f.properties.kind == "notam"}
    published = {n.id: n.e_text for n in run.artifact("get_notams").notams}
    assert set(features) == {"H7171/26", "J3857/26"}
    for notam_id, feature in features.items():
        assert feature.properties.text == published[notam_id]
        assert isinstance(feature.properties.approx, bool)
    assert "<PHONE_1>" in features["J3857/26"].properties.text
    for name in ALL_THREE:
        assert all(f.properties.text is None for f in run.artifact(name).overlays if f.properties.kind != "notam")


def test_blue_mill_the_activated_danger_area_openaip_lacks_is_placed_by_its_notam():
    """EGD496A is in no openAIP page: J3857/26 activates it over the LZ, so the item and the overlay come from that NOTAM, not from nothing."""
    run = blue_mill()
    artifact = run.artifact("get_airspace")
    item = next(i for i in artifact.findings if i.name == "EGD496A")
    assert (item.outcome, item.from_notam, item.type) == ("CONTACT_FIRST", "J3857/26", "danger")
    assert item.over_lz is True and item.distance_nm is None and item.bearing_deg is None
    assert (item.floor, item.ceiling) == ("SFC", "1300 ft AMSL")  # the reading's, as the NOTAM states them
    line = next(line for line in run.text("get_airspace").splitlines() if line.startswith("- EGD496A"))
    assert "Over the LZ" in line and "Floor SFC, ceiling 1300 ft AMSL (as NOTAM J3857/26 states them)" in line and "<PHONE_1>" in line
    parsed = next(n for n in load_case_sources("blue_mill_2026-10-31").pib.notams if n.id == "J3857/26")
    geometry, approx = notam_geometry(parsed)
    feature = next(f for f in artifact.overlays if f.properties.id == "EGD496A")
    assert (feature.properties.kind, feature.properties.action) == ("airspace", "CONTACT_FIRST")
    assert feature.geometry.model_dump(mode="json") == geometry and item.approx is approx
