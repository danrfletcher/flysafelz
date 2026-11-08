"""The assess tool on the committed cases, through a real graph (GP4): Blue Mill and West Sleekburn in case mode.

The tool adds no verdict logic, so each verdict, display string, time and count is compared with what ticket 2's `assess()` gives for the same inputs and window.
"""

import json
import re
from datetime import UTC, datetime

import cases
import pytest
from reader_replies import fixture_readings, model_reply
from reader_set_helpers import ScriptedChat
from tool_helpers import BLUE_MILL_LZ, BLUE_MILL_NOW, BLUE_MILL_WINDOW, ToolRun, calls, run_tools
from unit.reader_helpers import blue_mill_notams

from flysafelz.agent.sources import SourceCache
from flysafelz.agent.state import RunContext
from flysafelz.assess import display
from flysafelz.contracts import Verdict, display_verdict
from flysafelz.redact import redact_phones
from flysafelz.sources import load_case_sources
from flysafelz.verdict_sheet import DROPPED_REASON

BLUE_MILL = cases.BLUE_MILL
WEST_SLEEKBURN = cases.WEST_SLEEKBURN
CTX = RunContext(now=BLUE_MILL_NOW, case=BLUE_MILL, reader="fixture")
LONDON = "Europe/London"
GP4_SEGMENT = "PILOT DISCRETION \u2013 APPROACHING LIMITS \u00b7 WITH CONDITIONS"
START = datetime(2026, 10, 31, 10, tzinfo=UTC)
END = datetime(2026, 10, 31, 16, tzinfo=UTC)


class NoFetch:
    def __call__(self, *args):
        raise AssertionError("a case-mode run fetched live")


def blue_mill(steps=None, ctx=CTX, **kwargs) -> ToolRun:
    return run_tools(SourceCache(fetch=NoFetch()), steps or [calls("assess", **BLUE_MILL_WINDOW)], ctx=ctx, **kwargs)


def tickets_assessment(case: str, window, tz: str | None = LONDON, **changes):
    """Ticket 2's own answer for the same case, window and zone: what the tool must agree with."""
    return cases.run(case, window=window, tz=tz, live=False, **changes)


def segments_of(assessment):
    return [s for h in assessment.hours for s in h.segments]


def with_numbers(text: str, phone_map: dict[str, str]) -> str:
    """The tool's text with the thread's placeholders put back, to compare with ticket 2's strings, which hold the real number."""
    for placeholder, number in phone_map.items():
        text = text.replace(placeholder, number)
    return text


# --- GP4 -------------------------------------------------------------------------------------------------------------


def test_gp4_the_first_segment_is_pilot_discretion_approaching_limits_with_conditions_and_the_text_gives_counts_and_fetch_time():
    run = blue_mill()
    text, artifact = run.text("assess"), run.artifact("assess")
    first = artifact.segments[0]
    assert (first.start, first.end, first.tz_abbr) == ("2026-10-31T10:00:00+00:00", "2026-10-31T11:00:00+00:00", "GMT")
    assert (first.verdict, first.approaching_limits, first.with_conditions, first.display) == ("PILOT_DISCRETION", True, True, GP4_SEGMENT)
    assert f"10:00\u201311:00 GMT  {GP4_SEGMENT}" in text
    assert "2 NOTAMs within 2 NM." in text and "Data as of 07:30 GMT." in text
    assert "6 h assessed, 0 OUTLOOK, 0 too far." in text and "flyable for 6 h 00 min" in text
    assert text.splitlines()[0] == "Window: Saturday 31 Oct 10:00 \u2013 16:00 GMT."
    assert "assumed window" not in text and "No window given" not in text


def test_gp4_every_segment_display_is_display_verdict_of_its_own_three_fields_and_with_conditions_is_ticket_2s():
    """I7: `display` is never written on its own; it is `display_verdict` of the artifact's verdict, approaching_limits and with_conditions."""
    segments = blue_mill().artifact("assess").segments
    expected = segments_of(tickets_assessment(BLUE_MILL, (START, END)))
    assert len(segments) == len(expected) == 6
    for segment, want in zip(segments, expected, strict=True):
        assert segment.display == display_verdict(Verdict(segment.verdict), segment.approaching_limits, segment.with_conditions)
        assert segment.with_conditions == bool(want.conditions)
        assert segment.display == GP4_SEGMENT


def test_gp4_the_danger_area_condition_comes_first_with_the_placeholder_and_the_sector_condition_follows():
    first = blue_mill().artifact("assess").segments[0]
    assert first.conditions[0].startswith("Danger area EGD496A active ") and "<PHONE_1>" in first.conditions[0]
    assert "avoid that sector" in first.conditions[1]
    assert len(first.conditions) == 2


def test_gp4_the_danger_area_condition_shows_the_activation_span_ticket_2_gives_for_the_window():
    """Ticket 2 clips an activation to the window assessed, so a 10:00-16:00 window reads `active 10:00-16:00`; the whole NOTAM day reads `08:00-18:00`."""
    narrow = blue_mill().artifact("assess").segments[0].conditions[0]
    assert narrow.startswith("Danger area EGD496A active 10:00\u201316:00 GMT (")
    wide = blue_mill([calls("assess", start="2026-10-31 07:00", end="2026-10-31 18:00")]).artifact("assess")
    ten = next(s for s in wide.segments if s.start == "2026-10-31T10:00:00+00:00")
    assert ten.conditions[0].startswith("Danger area EGD496A active 08:00\u201318:00 GMT (")


def test_gp4_the_number_is_a_placeholder_everywhere_and_phone_map_holds_it():
    pib = load_case_sources(BLUE_MILL).pib
    _, numbers = redact_phones(next(n for n in pib.notams if n.id == "J3857/26").item_e)
    run = blue_mill()
    shown = run.text("assess") + json.dumps(run.artifact("assess").model_dump(mode="json"))
    assert set(numbers.values()) and "<PHONE_1>" in shown
    for number in numbers.values():
        assert number not in shown and re.sub(r"\D", "", number) not in re.sub(r"\D", "", shown)
    assert run.state["phone_map"] == {"<PHONE_1>": next(iter(numbers.values()))}
    assert "<PHONE_2>" not in shown


def test_gp4_progress_is_one_assessing_line_beside_the_reader_lines_and_it_has_no_verdict_word_or_number():
    run = blue_mill()
    assert run.progress.count("Assessing\u2026") == 1
    line = "Assessing\u2026"
    assert not re.search(r"\d", line) and not any(word in line.upper() for word in ("GO", "UNVERIFIED", "DISCRETION"))
    assert run.progress == ["Assessing\u2026", "Reading 2 NOTAMs (1/2)\u2026", "Reading 2 NOTAMs (2/2)\u2026"]


def test_two_calls_in_one_turn_write_one_assessing_line_each():
    run = blue_mill([calls("assess", **BLUE_MILL_WINDOW), calls("assess", **BLUE_MILL_WINDOW)])
    assert run.progress.count("Assessing\u2026") == 2
    assert [line for line in run.progress if line.startswith("Reading")] == ["Reading 2 NOTAMs (1/2)\u2026", "Reading 2 NOTAMs (2/2)\u2026"]  # the second call reads nothing
    assert run.text("assess", 0) == run.text("assess", 1)
    assert run.state["phone_map"].keys() == {"<PHONE_1>"}


# --- the tool adds no logic ------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("case", "lz", "now", "window"),
    [
        (BLUE_MILL, BLUE_MILL_LZ, BLUE_MILL_NOW, BLUE_MILL_WINDOW),
        (BLUE_MILL, BLUE_MILL_LZ, BLUE_MILL_NOW, {"start": "2026-10-31 07:00", "end": "2026-10-31 18:00"}),
        (WEST_SLEEKBURN, {"lat": 55.1576, "lon": -1.5699}, BLUE_MILL_NOW, {"start": "2026-10-31 07:00", "end": "2026-10-31 18:00"}),
    ],
    ids=["blue-mill-gp4", "blue-mill-whole-day", "west-sleekburn"],
)
def test_every_segment_overall_and_span_is_what_tickets_2_assess_gives_for_the_same_window(case, lz, now, window):
    ctx = RunContext(now=now, case=case, reader="fixture")
    run = run_tools(SourceCache(fetch=NoFetch()), [calls("assess", **window)], ctx=ctx, lz=lz)
    artifact, text = run.artifact("assess"), run.text("assess")
    first = datetime.fromisoformat(window["start"].replace(" ", "T")).replace(tzinfo=UTC)
    last = datetime.fromisoformat(window["end"].replace(" ", "T")).replace(tzinfo=UTC)
    theirs = tickets_assessment(case, (first, last))
    mine = [s for s in artifact.segments if s.tier == "VERDICT"]
    expected = segments_of(theirs)
    assert len(mine) == len(expected)
    phones = run.state.get("phone_map", {})
    for got, want in zip(mine, expected, strict=True):
        assert datetime.fromisoformat(got.start) == want.start and datetime.fromisoformat(got.end) == want.end
        assert (got.verdict, got.approaching_limits, got.display) == (want.verdict.value, want.approaching_limits, display(want))
        assert got.with_conditions == bool(want.conditions)
        assert got.display == display_verdict(Verdict(got.verdict), got.approaching_limits, got.with_conditions)
        assert [with_numbers(c, phones) for c in got.conditions] == want.conditions
        assert [(c.rule_id, c.value, c.limit, c.text) for c in got.cautions] == [(c.rule_id, c.value, c.limit, c.text) for c in want.cautions]
        assert [(n.id, n.action, n.bearing_deg) for n in got.notams] == [(a.notam_id, a.action.value, a.bearing_deg) for a in want.notam_actions]
    assert artifact.overall is not None and theirs.overall_verdict is not None and artifact.overall.verdict == theirs.overall_verdict.value
    assert [(datetime.fromisoformat(s.start), datetime.fromisoformat(s.end)) for s in artifact.flyable] == [(w.start, w.end) for w in theirs.flyable_windows]
    assert artifact.counts.notams_within == len({a.notam_id for s in expected for a in s.notam_actions})
    assert [(s.source, s.state) for s in artifact.sources] == [(s.source.value, s.state.value) for s in theirs.source_statuses]
    assert f"{artifact.counts.notams_within} NOTAMs within 2 NM." in text


def test_the_tool_text_holds_no_verdict_string_that_ticket_2_did_not_give_for_the_window():
    run = blue_mill()
    theirs = {display(s) for s in segments_of(tickets_assessment(BLUE_MILL, (START, END)))}
    segment_lines = [line for line in run.text("assess").splitlines() if re.match(r"\d\d:\d\d\u2013\d\d:\d\d \w+  ", line)]
    assert len(segment_lines) == 6
    assert {line.split("  ", 1)[1] for line in segment_lines} == theirs == {GP4_SEGMENT}


def test_the_overall_display_is_the_one_the_command_line_prints_for_the_same_assessment():
    from flysafelz.cli import overall_display

    run = blue_mill()
    assert run.artifact("assess").overall.display == overall_display(tickets_assessment(BLUE_MILL, (START, END)))
    assert f"Overall: {run.artifact('assess').overall.display}" in run.text("assess")


# --- readings: one thread, one set -----------------------------------------------------------------------------------


def blue_mill_model() -> ScriptedChat:
    by_id, readings = blue_mill_notams(), fixture_readings()
    scripts = {key: [model_reply(by_id[key.replace("_", "/")], readings[key.replace("_", "/")])] for key in ("H7171_26", "J3857_26")}
    return ScriptedChat(list(by_id.values()), **scripts)


def test_the_verdict_does_not_depend_on_get_notams_having_run_first():
    alone = blue_mill([calls("assess", **BLUE_MILL_WINDOW)])
    after_notams = blue_mill([calls("get_notams", **BLUE_MILL_WINDOW), calls("assess", **BLUE_MILL_WINDOW)])
    assert alone.text("assess") == after_notams.text("assess")
    assert alone.artifact("assess") == after_notams.artifact("assess")


def test_assess_and_get_notams_in_one_turn_read_each_notam_once_and_agree_on_the_numbering():
    run = blue_mill([calls("assess", **BLUE_MILL_WINDOW), calls("get_notams", **BLUE_MILL_WINDOW)])
    assert [line for line in run.progress if line.startswith("Reading")] == ["Reading 2 NOTAMs (1/2)\u2026", "Reading 2 NOTAMs (2/2)\u2026"]
    assert "<PHONE_1>" in run.text("get_notams") and "<PHONE_2>" not in run.text("get_notams")
    assert run.state["phone_map"].keys() == {"<PHONE_1>"}


def test_a_live_reader_case_assess_reads_through_the_same_store_as_get_notams(tmp_path):
    llm = blue_mill_model()
    ctx = RunContext(now=BLUE_MILL_NOW, case=BLUE_MILL, reader="live", reader_cache=tmp_path / "readings.sqlite")
    run = run_tools(SourceCache(fetch=NoFetch(), reader_model=lambda: llm), [calls("assess", **BLUE_MILL_WINDOW), calls("get_notams", **BLUE_MILL_WINDOW)], ctx=ctx)
    reads = len(llm.calls)
    assert reads >= 2
    assert run.artifact("assess").segments[0].display == GP4_SEGMENT  # the fake model's readings are readings.yaml's own
    assert len(llm.calls) == reads and len(set(llm.calls)) == reads  # get_notams read nothing more: every NOTAM was read once
    assert (tmp_path / "readings.sqlite").exists()


def test_a_live_reader_case_run_keeps_live_false_so_the_case_hours_that_have_ended_still_count():
    ctx = RunContext(now=datetime(2026, 10, 31, 13, 30, tzinfo=UTC), case=BLUE_MILL, reader="fixture")
    run = run_tools(SourceCache(fetch=NoFetch()), [calls("assess", start="2026-10-31 10:00", end="2026-10-31 16:00")], ctx=ctx)
    assert run.artifact("assess").counts.hours_assessed == 6  # a live run would leave the ended hours out (FR2)


def test_a_live_reader_that_is_not_configured_leaves_the_notams_unread_and_the_hours_unverified(tmp_path):
    ctx = RunContext(now=BLUE_MILL_NOW, case=BLUE_MILL, reader="live", reader_cache=tmp_path / "readings.sqlite")
    run = run_tools(SourceCache(fetch=NoFetch()), [calls("assess", **BLUE_MILL_WINDOW)], ctx=ctx)
    assert {s.verdict for s in run.artifact("assess").segments} <= {"UNVERIFIED", "NO_GO"}
    assert run.artifact("assess").segments[0].verdict == "UNVERIFIED"


# --- drop ------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(("dropped", "name"), [("weather", "weather"), ("airspace", "airspace"), ("pib", "notams")])
def test_a_dropped_source_reaches_assess_as_failed_and_the_segments_read_unverified(dropped, name):
    ctx = RunContext(now=BLUE_MILL_NOW, case=BLUE_MILL, reader="fixture", drop=(dropped,))
    run = blue_mill(ctx=ctx)
    artifact, text = run.artifact("assess"), run.text("assess")
    states = {s.source: s for s in artifact.sources}
    assert states[name].state == "failed" and states[name].reason == DROPPED_REASON and states[name].check_by_hand
    verdicts = {s.verdict for s in artifact.segments}
    assert "UNVERIFIED" in verdicts and verdicts <= {"UNVERIFIED", "NO_GO"}  # NO_GO only where another source forbids it
    assert Verdict.PILOT_DISCRETION.value not in json.dumps(artifact.model_dump(mode="json"))
    assert "PILOT DISCRETION" not in text and "Do not fly on this assessment." in text
    assert f"- {name} source: FAILED ({DROPPED_REASON})" in text


# --- the zone and the window -----------------------------------------------------------------------------------------


def test_the_tool_cannot_be_given_the_threads_position_zone_clock_or_case():
    run = run_tools(SourceCache(fetch=NoFetch()), [calls("assess", lat=51.0, **BLUE_MILL_WINDOW)], ctx=CTX)
    assert run.of("assess")[0].status == "error" and "lat" in str(run.of("assess")[0].content)


def test_a_new_york_pilot_asking_about_the_blue_mill_day_gets_the_hours_on_new_york_time():
    run = blue_mill([calls("assess", start="2026-10-31 06:00", end="2026-10-31 12:00")], tz="America/New_York")
    artifact = run.artifact("assess")
    assert artifact.segments[0].start == "2026-10-31T06:00:00-04:00" and {s.tz_abbr for s in artifact.segments} == {"EDT"}
    london = blue_mill([calls("assess", start="2026-10-31 10:00", end="2026-10-31 16:00")])
    assert [(s.verdict, s.display) for s in artifact.segments] == [(s.verdict, s.display) for s in london.artifact("assess").segments]
    assert "Data as of 03:30 EDT." in run.text("assess")


def test_the_overlays_are_the_lz_circle_and_the_notams_with_the_shape_and_raw_text_get_notams_gives():
    run = blue_mill([calls("assess", **BLUE_MILL_WINDOW), calls("get_notams", **BLUE_MILL_WINDOW)])
    mine = {o.properties.id: o for o in run.artifact("assess").overlays if o.properties.kind == "notam"}
    theirs = {o.properties.id: o for o in run.artifact("get_notams").overlays if o.properties.kind == "notam"}
    assert set(mine) == {"H7171/26", "J3857/26"} and set(mine) <= set(theirs)
    for notam_id, overlay in mine.items():
        assert overlay.geometry == theirs[notam_id].geometry
        assert overlay.properties.approx == theirs[notam_id].properties.approx
        assert overlay.properties.text == theirs[notam_id].properties.text
        assert overlay.properties.action == theirs[notam_id].properties.action
    published = next(n for n in load_case_sources(BLUE_MILL).pib.notams if n.id == "J3857/26").item_e
    assert mine["J3857/26"].properties.text == redact_phones(published)[0]  # the published E text as it was: its UTC times are not converted (P2)
    assert "128.925MHZ" in mine["J3857/26"].properties.text
    assert run.artifact("assess").overlays[0].properties.kind != "notam"
