"""The assess tool's text: its formatting rules, one table-driven case at a time over assessments from ticket 2's real `assess()`.

`build` is the tool's pure part (an Assessment in, text and artifact out), so each row changes one input and reads one line.
"""

import re
from datetime import UTC, datetime, timedelta

import pytest
from assess_tool_helpers import LONDON, NOW, assess_calm, render, utc
from factories import airspace_fetch, forecast_failed, notam, pib_result, run_assess, status, wind
from factories import weather_forecast as forecast_between

from flysafelz.agent.tools.assess_tool import Inputs, build
from flysafelz.agent.window import TOO_FAR_OUT, Window, parse_window
from flysafelz.contracts import FlyableWindow as TimeSpan
from flysafelz.contracts import SourceName, SourceState
from flysafelz.rules import RULES
from flysafelz.sources import CHECK_BY_HAND

HOUR = timedelta(hours=1)
NEW_YORK = "America/New_York"


def lines(text: str) -> list[str]:
    return text.splitlines()


def after(text: str, heading: str) -> list[str]:
    """The lines under a heading, up to the next unindented heading that ends in a colon."""
    rows = lines(text)
    start = rows.index(heading) + 1
    out = []
    for row in rows[start:]:
        if row.endswith(":") and not row.startswith((" ", "-")):
            break
        out.append(row)
    return out


# --- the header and the counts ---------------------------------------------------------------------------------------


def test_a_clear_window_reads_window_fetch_time_overall_counts_and_flyable_span():
    text, _ = render(assess_calm(utc(9), utc(12)), (utc(9), utc(12)))
    assert lines(text)[:5] == [
        "Window: test window.",
        "Data as of 07:30 GMT.",
        "Overall: PILOT DISCRETION",
        "0 NOTAMs within 2 NM. 3 h assessed, 0 OUTLOOK, 0 too far.",
        "Flyable windows:",
    ]
    assert "- Saturday 31 Oct 09:00\u201312:00 GMT, flyable for 3 h 00 min" in lines(text)


def test_an_assumed_window_is_printed_as_it_was_given_to_the_tool():
    window = parse_window(None, None, now=NOW, tz=LONDON)
    assert isinstance(window, Window) and window.assumed
    text, artifact = build(assess_calm(window.start_utc, window.end_utc), Inputs(tz=LONDON, window=window, now=NOW, overlays=[]))
    assert lines(text)[0] == window.words and window.words.startswith("assumed window: ")
    assert artifact.window_assumed is True


def test_one_notam_is_counted_in_the_singular():
    assessment = assess_calm(utc(9), utc(12), pib=pib_result(notam("H7171/26")))
    ids = {a.notam_id for h in assessment.hours for s in h.segments for a in s.notam_actions}
    assert ids == {"H7171/26"}  # the factory's NOTAM bears on the LZ in every segment
    text, artifact = render(assessment, (utc(9), utc(12)))
    assert "1 NOTAM within 2 NM." in text and artifact.counts.notams_within == 1


@pytest.mark.parametrize(
    ("changes", "assessed", "outlook", "too_far"),
    [
        pytest.param((utc(9), utc(12)), 3, 0, 0, id="all-verdict"),
        pytest.param((datetime(2026, 11, 3, 10, tzinfo=UTC), datetime(2026, 11, 3, 12, tzinfo=UTC)), 2, 0, 0, id="day-3-is-the-last-verdict-day"),
        pytest.param((datetime(2026, 11, 4, 10, tzinfo=UTC), datetime(2026, 11, 4, 12, tzinfo=UTC)), 0, 2, 0, id="day-4-is-outlook"),
        pytest.param((datetime(2026, 11, 7, 23, tzinfo=UTC), datetime(2026, 11, 8, 1, tzinfo=UTC)), 0, 1, 1, id="outlook-then-too-far"),
        pytest.param((datetime(2026, 11, 9, 10, tzinfo=UTC), datetime(2026, 11, 9, 12, tzinfo=UTC)), 0, 0, 2, id="all-too-far"),
    ],
)
def test_the_hour_counts_by_tier_follow_the_assessment(changes, assessed, outlook, too_far):
    text, artifact = render(run_assess(window=changes), changes)
    assert f"{assessed} h assessed, {outlook} OUTLOOK, {too_far} too far." in text
    assert (artifact.counts.hours_assessed, artifact.counts.hours_outlook, artifact.counts.hours_too_far) == (assessed, outlook, too_far)


# --- times in the pilot's zone ---------------------------------------------------------------------------------------


def test_a_window_over_the_bst_to_gmt_change_labels_each_hour_for_its_own_instant_and_loses_none():
    start, end = utc(23, day=24), utc(3, day=25)  # the clocks go back at 01:00 UTC on 25 Oct 2026
    now = utc(12, day=24)
    forecast = forecast_between(start - HOUR, end + HOUR)
    text, artifact = render(run_assess(window=(start, end), now=now, forecast=forecast), (start, end), now=now)
    segments = artifact.segments
    assert [s.tz_abbr for s in segments] == ["BST", "BST", "GMT", "GMT"]
    assert [s.start for s in segments] == [
        "2026-10-25T00:00:00+01:00",
        "2026-10-25T01:00:00+01:00",
        "2026-10-25T01:00:00+00:00",
        "2026-10-25T02:00:00+00:00",
    ]  # the repeated 01:00 is two different hours
    assert len({s.start for s in segments}) == 4
    for label in ("00:00\u201301:00 BST", "01:00 BST\u201301:00 GMT", "01:00\u201302:00 GMT", "02:00\u201303:00 GMT"):
        assert any(row.startswith(f"{label}  ") for row in lines(text)), label


def test_a_window_over_the_gmt_to_bst_change_labels_each_hour_for_its_own_instant_and_loses_none():
    start, end = utc(23, day=28, month=3), utc(3, day=29, month=3)  # the clocks go forward at 01:00 UTC on 29 Mar 2026
    now = utc(12, day=28, month=3)
    forecast = forecast_between(start - HOUR, end + HOUR)
    _, artifact = render(run_assess(window=(start, end), now=now, forecast=forecast), (start, end), now=now)
    assert [s.tz_abbr for s in artifact.segments] == ["GMT", "GMT", "BST", "BST"]
    assert [s.start[11:16] for s in artifact.segments] == ["23:00", "00:00", "02:00", "03:00"]  # 01:00 does not exist
    assert len({s.start for s in artifact.segments}) == 4


def test_a_new_york_pilot_gets_the_same_hours_on_new_york_time_with_its_abbreviation():
    start, end = utc(14), utc(16)  # 31 Oct: New York is still on EDT (UTC-4)
    london_text, london = render(assess_calm(start, end), (start, end))
    new_york_text, new_york = render(assess_calm(start, end), (start, end), tz=NEW_YORK)
    assert [s.tz_abbr for s in new_york.segments] == ["EDT"] * 2
    assert [s.start[11:16] for s in new_york.segments] == ["10:00", "11:00"]
    assert "10:00\u201311:00 EDT" in new_york_text and "Data as of 03:30 EDT." in new_york_text
    # the day tiers and the verdicts stay Europe/London's: the same UTC hour is the same
    assert [(s.tier, s.verdict) for s in new_york.segments] == [(s.tier, s.verdict) for s in london.segments]
    assert "GMT" not in new_york_text and "EDT" not in london_text


def test_a_span_over_midnight_in_the_pilots_zone_is_written_with_its_dates():
    start, end = utc(22), utc(2, day=1, month=11)
    text, artifact = render(assess_calm(start, end, now=utc(12)), (start, end))
    assert any(row.startswith("22:00\u201323:00 GMT  ") for row in lines(text))
    assert "Saturday 31 Oct:" in lines(text) and "Sunday 1 Nov:" in lines(text)  # each segment is listed under the local day it starts on
    assert [s.start[:10] for s in artifact.segments] == ["2026-10-31"] * 2 + ["2026-11-01"] * 2


# --- OUTLOOK and TOO_FAR hours ---------------------------------------------------------------------------------------


def outlook_assessment(hours_wind: list[float]):
    first = datetime(2026, 11, 5, 10, tzinfo=UTC)  # day 5 in London
    changes = {first + n * HOUR: {**wind(mph), "precipitation_probability": 10.0 * (n + 1)} for n, mph in enumerate(hours_wind)}
    forecast = forecast_between(utc(6), datetime(2026, 11, 8, 23, tzinfo=UTC), changes=changes)
    window = (first, first + len(hours_wind) * HOUR)
    return run_assess(window=window, forecast=forecast), window


def test_an_outlook_block_prints_its_trend_and_the_day_to_ask_again_with_no_verdict():
    assessment, window = outlook_assessment([8.0, 14.0, 11.0])
    text, artifact = render(assessment, window)
    weather = [h.weather for h in assessment.hours]  # the trend is the min and max of ticket 2's own per-hour values
    winds, gusts, rains = ([getattr(w, name) for w in weather] for name in ("wind_10m", "gust_10m", "precipitation_probability"))
    low, high, gust_low, gust_high, rain_low, rain_high = min(winds), max(winds), min(gusts), max(gusts), min(rains), max(rains)
    assert low < high and gust_low < gust_high and rain_low < rain_high  # the case is a real range, not one value three times
    block = after(text, "OUTLOOK hours (no verdict):")
    trend = f"wind {low:g}\u2013{high:g} mph, gust {gust_low:g}\u2013{gust_high:g} mph, rain chance {rain_low:g}\u2013{rain_high:g} %"
    assert block == [f"- Thursday 5 Nov 10:00\u201313:00 GMT: OUTLOOK (no verdict). Trend: {trend}. Ask again on Monday 2 Nov."]
    assert "Overall: No verdict: OUTLOOK only" in text
    assert [s.tier for s in artifact.segments] == ["OUTLOOK"] * 3
    assert all(s.verdict is None and s.display is None for s in artifact.segments)
    assert [(s.wind_10m, s.gust_10m, s.rain_chance) for s in artifact.segments] == [(w.wind_10m, w.gust_10m, w.precipitation_probability) for w in weather]
    b = artifact.blocks[0]
    assert (b.wind_min, b.wind_max, b.gust_min, b.gust_max, b.rain_min, b.rain_max) == (low, high, gust_low, gust_high, rain_low, rain_high)
    assert (b.wind_unit, b.gust_unit, b.rain_unit) == ("mph", "mph", "%")


def test_the_ask_again_day_is_the_hours_london_day_less_the_verdict_horizon_day():
    assessment, window = outlook_assessment([8.0])
    _, artifact = render(assessment, window)
    horizon = int(RULES.verdict_horizon_day.value)  # type: ignore[arg-type]
    assert (artifact.blocks[0].ask_again_on, artifact.segments[0].ask_again_on) == (f"2026-11-{5 - horizon:02d}",) * 2


def test_an_outlook_hour_inside_the_verdict_horizon_has_no_day_to_ask_again_and_says_the_forecast_is_incomplete():
    window = (utc(9), utc(12))  # today; the forecast stops before the window, so the hours are OUTLOOK for want of data
    text, artifact = render(run_assess(window=window, forecast=forecast_between(utc(6), utc(8))), window)
    assert artifact.blocks[0].ask_again_on is None and artifact.segments[0].ask_again_on is None
    assert "not complete enough for a verdict yet" in text and "Ask again on" not in text


def test_an_outlook_block_with_no_forecast_values_says_not_forecast_rather_than_a_number():
    window = (utc(9), utc(12))
    text, artifact = render(run_assess(window=window, forecast=forecast_between(utc(6), utc(8))), window)  # the data ends before the window
    assert "Trend: wind not forecast, gust not forecast, rain chance not forecast." in text
    assert artifact.blocks[0].wind_min is None and artifact.blocks[0].rain_max is None


def test_outlook_hours_of_two_london_days_are_two_blocks_each_with_its_own_day():
    first = datetime(2026, 11, 5, 22, tzinfo=UTC)  # 22:00-24:00 on day 5, then day 6
    forecast = forecast_between(utc(6), datetime(2026, 11, 8, 23, tzinfo=UTC))
    window = (first, first + 4 * HOUR)
    text, artifact = render(run_assess(window=window, forecast=forecast), window)
    assert [b.start[:10] for b in artifact.blocks] == ["2026-11-05", "2026-11-06"]
    assert [b.ask_again_on for b in artifact.blocks] == ["2026-11-02", "2026-11-03"]
    assert len(after(text, "OUTLOOK hours (no verdict):")) == 2


def test_too_far_hours_say_too_far_out_to_assess_and_carry_no_verdict():
    window = (datetime(2026, 11, 9, 10, tzinfo=UTC), datetime(2026, 11, 9, 13, tzinfo=UTC))
    text, artifact = render(run_assess(window=window), window)
    assert "Overall: No verdict: too far out to assess" in text
    assert after(text, "Hours too far out (too far out to assess, no verdict):") == ["- Monday 9 Nov 10:00\u201313:00 GMT: too far out to assess."]
    assert [(s.tier, s.verdict, s.display) for s in artifact.segments] == [("TOO_FAR", None, None)] * 3
    assert artifact.overall is None and artifact.overall_tier == "TOO_FAR"
    assert TOO_FAR_OUT  # the tool-level refusal for a window that starts beyond the horizon is PR-4.F1's


def test_a_window_with_all_three_tiers_prints_each_under_its_own_heading_and_the_overall_is_the_worst_verdict_tier_segment():
    window = (datetime(2026, 11, 3, 22, tzinfo=UTC), datetime(2026, 11, 9, 2, tzinfo=UTC))
    forecast = forecast_between(utc(6), datetime(2026, 11, 8, 23, tzinfo=UTC))
    text, artifact = render(run_assess(window=window, forecast=forecast), window)
    headings = [row for row in lines(text) if row.endswith(":") and not row.startswith((" ", "-"))]
    assert headings.index("Segments:") < headings.index("OUTLOOK hours (no verdict):") < headings.index("Hours too far out (too far out to assess, no verdict):")
    assert artifact.overall is not None and artifact.overall.verdict == "NO_GO"  # the night hours are NO-GO; no outlook hour has a say
    verdict_segments = [s for s in artifact.segments if s.tier == "VERDICT"]
    assert verdict_segments and all(s.verdict for s in verdict_segments)
    assert {s.tier for s in artifact.segments} == {"VERDICT", "OUTLOOK", "TOO_FAR"}


# --- spans -----------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("start", "end", "words"),
    [
        pytest.param(utc(10), utc(11), "1 h 00 min", id="one-hour"),
        pytest.param(utc(10, 7), utc(11, 7), "1 h 00 min", id="whole-hour-off-the-clock"),
        pytest.param(utc(10), utc(10, 59), "0 h 59 min", id="under-an-hour"),
        pytest.param(utc(9, 30), utc(12, 5), "2 h 35 min", id="minutes-at-both-edges"),
        pytest.param(utc(8), utc(16), "8 h 00 min", id="long"),
    ],
)
def test_a_flyable_window_prints_its_exact_length_in_whole_minutes(start, end, words):
    assessment = assess_calm(utc(8), utc(17))
    assessment = assessment.model_copy(update={"flyable_windows": [TimeSpan(start=start, end=end)]})
    text, artifact = render(assessment, (utc(8), utc(17)))
    assert f", flyable for {words}" in text
    assert artifact.flyable[0].minutes == int((end - start).total_seconds() // 60)
    assert artifact.counts.flyable_windows == 1


def test_two_flyable_windows_print_in_order_and_are_counted():
    assessment = assess_calm(utc(8), utc(17)).model_copy(update={"flyable_windows": [TimeSpan(start=utc(8), end=utc(10)), TimeSpan(start=utc(13, 30), end=utc(15))]})
    text, artifact = render(assessment, (utc(8), utc(17)))
    assert after(text, "Flyable windows:") == [
        "- Saturday 31 Oct 08:00\u201310:00 GMT, flyable for 2 h 00 min",
        "- Saturday 31 Oct 13:30\u201315:00 GMT, flyable for 1 h 30 min",
    ]
    assert artifact.counts.flyable_windows == 2


def test_no_flyable_window_says_so_and_counts_none():
    assessment = assess_calm(utc(9), utc(12), forecast=forecast_between(utc(8), utc(13), changes={utc(9): wind(40), utc(10): wind(40), utc(11): wind(40)}))
    text, artifact = render(assessment, (utc(9), utc(12)))
    assert "Flyable windows: no flyable window in this window." in text
    assert artifact.flyable == [] and artifact.counts.flyable_windows == 0


# --- the fetch time --------------------------------------------------------------------------------------------------


def statuses_with(assessment, **fetched: datetime | None):
    """The assessment with each named source's fetch time replaced."""
    changed = [s.model_copy(update={"fetched_at": fetched.get(s.source.value, s.fetched_at)}) for s in assessment.source_statuses]
    return assessment.model_copy(update={"source_statuses": changed})


def test_the_fetch_time_is_the_oldest_of_the_sources_and_is_written_in_the_pilots_zone():
    assessment = statuses_with(assess_calm(utc(9), utc(12)), weather=utc(7, 30), airspace=utc(7, 10), notams=utc(7, 20))
    text, artifact = render(assessment, (utc(9), utc(12)))
    assert "Data as of 07:10 GMT." in text
    assert artifact.fetched_at == utc(7, 10) and artifact.data_as_of == "07:10 GMT"
    assert {s.fetched_at for s in artifact.sources} == {utc(7, 10)}  # every source carries the turn's one fetch time
    assert "Data as of 03:10 EDT." in render(assessment, (utc(9), utc(12)), tz=NEW_YORK)[0]


def test_a_source_without_a_fetch_time_does_not_stand_for_the_oldest():
    assessment = statuses_with(assess_calm(utc(9), utc(12)), weather=utc(7, 30), airspace=None, notams=None)
    assert "Data as of 07:30 GMT." in render(assessment, (utc(9), utc(12)))[0]


def test_no_fetch_time_at_all_is_said_plainly():
    assessment = statuses_with(assess_calm(utc(9), utc(12)), weather=None, airspace=None, notams=None)
    text, artifact = render(assessment, (utc(9), utc(12)))
    assert "Data as of: no source returned data." in text
    assert artifact.fetched_at is None and artifact.data_as_of is None


# --- failed and stale sources ----------------------------------------------------------------------------------------


@pytest.mark.parametrize("source", list(SourceName), ids=lambda s: s.value)
def test_a_failed_source_is_named_with_its_reason_the_manual_check_and_the_warning(source):
    failed = {
        SourceName.WEATHER: {"forecast": forecast_failed()},
        SourceName.AIRSPACE: {"airspace": airspace_fetch(state=SourceState.FAILED)},
        SourceName.NOTAMS: {"pib": pib_result(state=SourceState.FAILED)},
    }[source]
    text, artifact = render(assess_calm(utc(9), utc(12), **failed), (utc(9), utc(12)))
    row = next(r for r in lines(text) if r.startswith(f"- {source.value} source:"))
    assert "FAILED" in row and f"Check by hand: {CHECK_BY_HAND[source]}." in row
    assert "Do not fly on this assessment." in lines(text)
    info = next(i for i in artifact.sources if i.source == source.value)
    assert (info.state, info.check_by_hand) == ("failed", CHECK_BY_HAND[source])
    assert [i.check_by_hand for i in artifact.sources if i.source != source.value] == [None, None]


def test_a_pib_too_old_is_stale_from_the_assessment_not_from_the_cache():
    old = NOW - timedelta(hours=float(RULES.pib_stale_hours.value) + 1)  # type: ignore[arg-type]
    assessment = assess_calm(utc(9), utc(12), pib=pib_result(issued=old), stale_rule=True)
    text, artifact = render(assessment, (utc(9), utc(12)))
    row = next(r for r in lines(text) if r.startswith("- notams source:"))
    assert "STALE" in row and CHECK_BY_HAND[SourceName.NOTAMS] in row
    assert "Do not fly on this assessment." in lines(text)
    assert next(i for i in artifact.sources if i.source == "notams").state == "stale"
    assert status(SourceName.NOTAMS).state is SourceState.OK  # the status the cache held says ok


def test_when_every_source_has_failed_night_stays_no_go_the_rest_is_unverified_and_all_three_are_named():
    window = (utc(14), utc(20))  # sunset is about 16:55 UTC on 31 Oct
    assessment = run_assess(
        window=window,
        forecast=forecast_failed(),
        airspace=airspace_fetch(state=SourceState.FAILED),
        pib=pib_result(state=SourceState.FAILED),
    )
    text, artifact = render(assessment, window)
    for source in SourceName:
        assert any(r.startswith(f"- {source.value} source: FAILED") for r in lines(text))
    shown = {s.display for s in artifact.segments}
    assert shown == {"NO-GO", "UNVERIFIED"}
    night = [s for s in artifact.segments if s.start >= utc(17, 6).isoformat()]
    assert night and all(s.display == "NO-GO" for s in night)
    day = [s for s in artifact.segments if s.end <= utc(17).isoformat()]
    assert day and all(s.display == "UNVERIFIED" for s in day)


def test_a_clean_run_has_no_warning_and_no_manual_check():
    text, artifact = render(assess_calm(utc(9), utc(12)), (utc(9), utc(12)))
    assert "Do not fly" not in text and "Check by hand" not in text
    assert all(i.check_by_hand is None and i.state == "ok" for i in artifact.sources)


def test_the_text_never_contains_a_placeholder_that_a_phone_book_left_unplaced():
    text, _ = render(assess_calm(utc(9), utc(12)), (utc(9), utc(12)))
    assert not re.search(r"<PHONE_\d+>", text)
