"""The assess tool at the edges of time: an OUTLOOK slot (E7), a slot beyond the horizon, the default window (E4), the clock changes, and the live-mode edges.

All live runs (no case) over `tool_world.World`, whose fetcher counts its calls, with `now` frozen at the conftest clock.
"""

from datetime import UTC, datetime, timedelta

from factories import NOW, airspace_fetch, pib_result, utc, weather_forecast
from tool_helpers import calls
from unit.agent.tool_world import World, go

from flysafelz.agent.state import RunContext
from flysafelz.agent.window import TOO_FAR_OUT, parse_window
from flysafelz.contracts import SourceName
from flysafelz.daylight import utc_hours
from flysafelz.rules import RULES

SIX_STRINGS = ("NO-GO", "UNVERIFIED", "PILOT DISCRETION", "APPROACHING LIMITS", "WITH CONDITIONS")  # the words every one of display_verdict's six strings is made of
LONDON = "Europe/London"


def lines(text: str) -> list[str]:
    return text.splitlines()


# --- E7: a slot five days out ----------------------------------------------------------------------------------------


def test_a_slot_five_days_out_is_outlook_with_a_trend_the_day_to_ask_again_and_no_verdict_anywhere():
    run = go(World(), [calls("assess", start="2026-11-05 10:00", end="2026-11-05 13:00")])
    text, artifact = run.text("assess"), run.artifact("assess")
    horizon = int(RULES.verdict_horizon_day.value)  # type: ignore[arg-type]
    ask = datetime(2026, 11, 5, tzinfo=UTC) - timedelta(days=horizon)
    assert f"Ask again on {ask:%A} {ask.day} {ask:%b}." in text and "OUTLOOK (no verdict)" in text
    assert "Trend: wind " in text and " mph" in text and "rain chance" in text
    assert "Overall: No verdict: OUTLOOK only" in text
    for word in SIX_STRINGS:
        assert word not in text and word not in artifact.model_dump_json(), word
    assert artifact.overall is None and artifact.overall_tier == "OUTLOOK"
    assert [s.ask_again_on for s in artifact.segments] == [ask.date().isoformat()] * 3
    assert artifact.counts.hours_outlook == 3 and artifact.counts.hours_assessed == 0


def test_the_last_verdict_day_still_gets_a_verdict_and_the_next_day_does_not():
    last_verdict = go(World(), [calls("assess", start="2026-11-03 10:00", end="2026-11-03 12:00")]).artifact("assess")
    first_outlook = go(World(), [calls("assess", start="2026-11-04 10:00", end="2026-11-04 12:00")]).artifact("assess")
    assert {s.tier for s in last_verdict.segments} == {"VERDICT"}
    assert {s.tier for s in first_outlook.segments} == {"OUTLOOK"}


# --- E8: beyond the horizon ------------------------------------------------------------------------------------------


def test_a_slot_wholly_beyond_the_horizon_is_the_window_error_and_nothing_is_fetched():
    world = World()
    run = go(world, [calls("assess", start="2026-11-09 10:00", end="2026-11-09 13:00")])
    assert run.text("assess") == TOO_FAR_OUT and run.artifact("assess") is None
    assert world.calls == [] and run.progress == []


def test_a_window_that_crosses_the_horizon_prints_outlook_and_too_far_blocks_and_never_a_verdict_beyond_day_3():
    run = go(World(), [calls("assess", start="2026-11-06 20:00", end="2026-11-09 04:00")])
    text, artifact = run.text("assess"), run.artifact("assess")
    assert "OUTLOOK hours (no verdict):" in lines(text) and "Hours too far out (too far out to assess, no verdict):" in lines(text)
    assert {s.tier for s in artifact.segments} == {"OUTLOOK", "TOO_FAR"}
    assert all(s.verdict is None and s.display is None for s in artifact.segments)
    assert artifact.overall is None and artifact.overall_tier == "OUTLOOK"
    for word in SIX_STRINGS:
        assert word not in text and word not in artifact.model_dump_json(), word


def test_hours_not_fetched_because_they_are_past_the_horizon_do_not_make_a_source_look_failed():
    run = go(World(), [calls("assess", start="2026-11-06 20:00", end="2026-11-09 04:00")])
    text, artifact = run.text("assess"), run.artifact("assess")
    assert [s.state for s in artifact.sources] == ["ok", "ok", "ok"] and all(s.check_by_hand is None for s in artifact.sources)
    assert "FAILED" not in text and "STALE" not in text and "Do not fly" not in text


# --- E4: the default window ------------------------------------------------------------------------------------------


def test_with_no_start_or_end_the_window_is_the_default_one_and_the_first_line_says_it_was_assumed():
    run = go(World(), [calls("assess")])
    text, artifact = run.text("assess"), run.artifact("assess")
    window = parse_window(None, None, now=NOW, tz=LONDON)
    assert not isinstance(window, str) and window.assumed
    assert lines(text)[0] == window.words and lines(text)[0].startswith("assumed window: ")
    assert lines(text)[0] == "assumed window: Saturday 31 Oct 07:00 \u2013 Sunday 1 Nov 24:00 GMT"
    assert artifact.window_assumed is True and (artifact.window_start, artifact.window_end) == (window.start_utc, window.end_utc)
    hours = utc_hours(window.start_utc, window.end_utc)
    assert artifact.counts.hours_assessed + artifact.counts.hours_outlook + artifact.counts.hours_too_far == len(hours) == 41
    assert "No window given" not in text and not any("No window given" in a for a in artifact.assumptions)  # ticket 2's UTC line cannot sit beside the tz one


def test_an_explicit_window_never_lets_ticket_2_assume_a_utc_one():
    run = go(World(), [calls("assess", start="2026-10-31 10:00", end="2026-10-31 12:00")])
    assert "No window given" not in run.text("assess")
    assert lines(run.text("assess"))[0] == "Window: Saturday 31 Oct 10:00 \u2013 12:00 GMT."


# --- the clock changes -----------------------------------------------------------------------------------------------


def clock_change_run(now: datetime, start: str, end: str):
    world = World(forecast=weather_forecast(now.replace(minute=0, second=0, microsecond=0), now + timedelta(days=5)))
    return go(world, [calls("assess", start=start, end=end)], ctx=RunContext(now=now))


def test_a_window_over_the_bst_to_gmt_change_labels_each_hour_for_itself_and_loses_none():
    run = clock_change_run(datetime(2026, 10, 24, 20, tzinfo=UTC), "2026-10-25 00:00", "2026-10-25 03:00")
    artifact, text = run.artifact("assess"), run.text("assess")
    assert lines(text)[0] == "Window: Sunday 25 Oct 00:00 BST \u2013 03:00 GMT."
    assert [s.tz_abbr for s in artifact.segments] == ["BST", "BST", "GMT", "GMT"]
    assert [datetime.fromisoformat(s.start) for s in artifact.segments] == utc_hours(utc(23, day=24), utc(3, day=25))  # the same four UTC hours, none twice
    assert artifact.counts.hours_assessed == 4


def test_a_window_over_the_gmt_to_bst_change_labels_each_hour_for_itself_and_loses_none():
    run = clock_change_run(datetime(2026, 3, 28, 20, tzinfo=UTC), "2026-03-29 00:00", "2026-03-29 04:00")
    artifact = run.artifact("assess")
    assert [s.tz_abbr for s in artifact.segments] == ["GMT", "BST", "BST"]
    assert [s.start[11:16] for s in artifact.segments] == ["00:00", "02:00", "03:00"]  # 01:00 does not exist that night
    assert [datetime.fromisoformat(s.start) for s in artifact.segments] == utc_hours(utc(0, day=29, month=3), utc(3, day=29, month=3))


# --- one fetch per turn ----------------------------------------------------------------------------------------------


def test_two_calls_in_one_turn_make_one_fetch_per_source_and_share_the_fetch_time():
    world = World()
    run = go(world, [calls("assess", start="2026-10-31 10:00", end="2026-10-31 12:00"), calls("assess", start="2026-10-31 12:00", end="2026-10-31 14:00")])
    assert len(world.calls) == 1
    first, second = run.artifact("assess", 0), run.artifact("assess", 1)
    assert first.fetched_at == second.fetched_at == NOW and first.data_as_of == second.data_as_of == "07:30 GMT"
    assert {s.fetched_at for s in first.sources + second.sources} == {NOW}


def test_assess_beside_the_three_data_tools_still_makes_one_fetch():
    world = World()
    run = go(world, [calls("get_weather", "get_airspace", "get_notams", "assess", start="2026-10-31 10:00", end="2026-10-31 12:00")])
    assert len(world.calls) == 1 and len(run.messages) == 4


def test_the_fetch_time_is_the_oldest_of_the_sources_in_a_live_run():
    older = NOW - timedelta(minutes=20)
    airspace = airspace_fetch()
    airspace = airspace.model_copy(update={"status": airspace.status.model_copy(update={"fetched_at": older})})
    run = go(World(airspace=airspace), [calls("assess", start="2026-10-31 10:00", end="2026-10-31 12:00")])
    assert "Data as of 07:10 GMT." in run.text("assess") and run.artifact("assess").fetched_at == older


# --- zones and live hours --------------------------------------------------------------------------------------------


def test_a_new_york_run_keeps_the_london_day_tiers_for_the_same_utc_hours():
    window = {"start": "2026-11-04 06:00", "end": "2026-11-04 10:00"}  # 06:00-10:00 EST is 11:00-15:00 UTC (the US clocks went back on 1 Nov): day 4 in London
    london = go(World(), [calls("assess", start="2026-11-04 11:00", end="2026-11-04 15:00")], tz=LONDON).artifact("assess")
    new_york = go(World(), [calls("assess", **window)], tz="America/New_York").artifact("assess")
    assert [s.tier for s in new_york.segments] == [s.tier for s in london.segments] == ["OUTLOOK"] * 4
    assert {s.tz_abbr for s in new_york.segments} == {"EST"}
    assert [datetime.fromisoformat(s.start) for s in new_york.segments] == [datetime.fromisoformat(s.start) for s in london.segments]


def test_a_live_run_leaves_out_the_hours_that_have_already_ended():
    now = datetime(2026, 10, 31, 13, 30, tzinfo=UTC)
    run = go(World(), [calls("assess", start="2026-10-31 10:00", end="2026-10-31 16:00")], ctx=RunContext(now=now))
    assert run.artifact("assess").counts.hours_assessed == 3  # 13:00, 14:00, 15:00 (the 13:00 hour has not ended)


def test_every_source_is_named_in_the_artifact_in_the_order_ticket_2_lists_them():
    run = go(World(), [calls("assess", start="2026-10-31 10:00", end="2026-10-31 12:00")])
    assert [s.source for s in run.artifact("assess").sources] == [name.value for name in SourceName]


def test_the_notam_reader_is_never_asked_in_a_live_run_with_no_notams():
    asked = []

    def model():
        asked.append(1)
        raise AssertionError("no NOTAM, no reader")

    run = go(World(pib=pib_result()), [calls("assess", start="2026-10-31 10:00", end="2026-10-31 12:00")], reader_model=model)
    assert asked == [] and run.progress == ["Assessing\u2026"]
