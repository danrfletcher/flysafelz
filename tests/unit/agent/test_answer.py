"""PR-11.F1: what code puts in a checked answer (answer.py): the re-check line (FR8, FR9), the limit lines (I10), the `assessment` field (I7), rehydration (P1).

The artifacts are the real ones: the Blue Mill tools run in a graph (guard_world), or `assess()` over the factories' sources written through the tool's own `build`.
"""

import logging
from datetime import UTC, datetime, timedelta

import pytest
from assess_tool_helpers import assess_calm, render
from factories import NOW, utc, weather_forecast
from guard_world import blue_mill_turn
from tool_helpers import calls
from tool_world import WINDOW, World, go

from flysafelz.agent.answer import RECHECK_SENTENCE, assessment_of, fetch_time, limit_lines, recheck_line, rehydrate, rehydrate_assessment, rehydrate_overlays
from flysafelz.agent.state import OverlayFeature, OverlayProperties, PointGeometry
from flysafelz.agent.tools.artifacts import AssessArtifact, WeatherArtifact

LONDON = "Europe/London"
ONE_HOUR = timedelta(hours=1)


def blue_mill_artifacts():
    return {result.name: result.artifact for result in blue_mill_turn().results}


# --- the re-check line ----------------------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("now", "start", "expected"),
    [
        (datetime(2026, 10, 31, 7, 30, tzinfo=UTC), datetime(2026, 10, 31, 10, tzinfo=UTC), True),  # after the current hour
        (datetime(2026, 10, 31, 7, 30, tzinfo=UTC), datetime(2026, 10, 31, 8, tzinfo=UTC), True),  # the very next hour counts
        (datetime(2026, 10, 31, 7, 30, tzinfo=UTC), datetime(2026, 10, 31, 7, tzinfo=UTC), False),  # the current hour: the default window starts here
        (datetime(2026, 10, 31, 7, 30, tzinfo=UTC), datetime(2026, 10, 31, 6, tzinfo=UTC), False),  # before it
        (datetime(2026, 10, 31, 7, 0, tzinfo=UTC), datetime(2026, 10, 31, 7, tzinfo=UTC), False),  # on the hour, equal
    ],
)
def test_the_line_needs_a_window_that_starts_after_the_current_hour(now, start, expected):
    artifact = blue_mill_artifacts()["assess"].model_copy(update={"window_start": start})
    line = recheck_line([artifact], now, LONDON, "assess")
    assert (line is not None) is expected
    if expected:
        assert line == f"Based on data as of 07:30 GMT. {RECHECK_SENTENCE}"


def test_the_current_hour_is_the_current_hour_in_the_threads_zone():
    """23:30Z on 24 Oct is 00:30 BST on the 25th: the current hour in London starts at 23:00Z, so a window from 00:00Z is after it."""
    now = datetime(2026, 10, 24, 23, 30, tzinfo=UTC)
    artifact = blue_mill_artifacts()["assess"].model_copy(update={"window_start": datetime(2026, 10, 25, 0, tzinfo=UTC)})
    assert recheck_line([artifact], now, LONDON, "assess") is not None
    assert recheck_line([artifact.model_copy(update={"window_start": datetime(2026, 10, 24, 23, 59, tzinfo=UTC)})], now, LONDON, "assess") is None


@pytest.mark.parametrize("path", ["clarify", "decline"])
def test_a_clarify_or_decline_never_gets_the_line(path):
    assert recheck_line([blue_mill_artifacts()["assess"]], NOW, LONDON, path) is None


@pytest.mark.parametrize("path", ["assess", "explain", None])
def test_other_paths_and_an_unknown_one_get_it(path):
    assert recheck_line([blue_mill_artifacts()["assess"]], datetime(2026, 10, 31, 7, 30, tzinfo=UTC), LONDON, path) is not None


def test_no_data_tool_means_no_line():
    assert recheck_line([], datetime(2026, 10, 31, 7, 30, tzinfo=UTC), LONDON, "assess") is None


@pytest.mark.parametrize("tool", ["get_weather", "get_airspace", "get_notams", "assess"])
def test_each_of_the_four_data_tools_can_trigger_it(tool):
    artifact = blue_mill_artifacts()[tool]
    assert recheck_line([artifact], datetime(2026, 10, 31, 7, 30, tzinfo=UTC), LONDON, "assess") is not None


def test_the_time_is_the_oldest_fetch_among_the_sources_used_in_the_zone_with_its_abbreviation():
    artifacts = blue_mill_artifacts()
    weather = artifacts["get_weather"]
    older = weather.model_copy(update={"source": weather.source.model_copy(update={"fetched_at": datetime(2026, 10, 31, 6, 45, tzinfo=UTC)})})
    newer = artifacts["get_notams"].model_copy(update={"source": artifacts["get_notams"].source.model_copy(update={"fetched_at": datetime(2026, 10, 31, 7, 20, tzinfo=UTC)})})
    now = datetime(2026, 10, 31, 7, 30, tzinfo=UTC)
    assert fetch_time(older) == datetime(2026, 10, 31, 6, 45, tzinfo=UTC)
    assert recheck_line([newer, older], now, LONDON, "assess") == f"Based on data as of 06:45 GMT. {RECHECK_SENTENCE}"
    # in a BST zone the same instant reads in BST
    summer = datetime(2026, 10, 24, 6, 45, tzinfo=UTC)
    in_bst = older.model_copy(update={"window_start": datetime(2026, 10, 24, 12, tzinfo=UTC), "source": older.source.model_copy(update={"fetched_at": summer})})
    assert recheck_line([in_bst], datetime(2026, 10, 24, 6, 50, tzinfo=UTC), LONDON, "assess") == f"Based on data as of 07:45 BST. {RECHECK_SENTENCE}"


def test_an_assess_artifact_holds_the_oldest_of_its_three_sources():
    assess = blue_mill_artifacts()["assess"]
    assert isinstance(assess, AssessArtifact) and fetch_time(assess) == assess.fetched_at


def test_sources_with_no_fetch_time_give_the_second_sentence_alone():
    weather = blue_mill_artifacts()["get_weather"]
    unfetched = weather.model_copy(update={"source": weather.source.model_copy(update={"fetched_at": None})})
    assert isinstance(unfetched, WeatherArtifact)
    assert recheck_line([unfetched], datetime(2026, 10, 31, 7, 30, tzinfo=UTC), LONDON, "assess") == RECHECK_SENTENCE


def test_a_missing_fetch_time_does_not_hide_an_older_one_that_exists():
    artifacts = blue_mill_artifacts()
    weather = artifacts["get_weather"]
    unfetched = weather.model_copy(update={"source": weather.source.model_copy(update={"fetched_at": None})})
    line = recheck_line([unfetched, artifacts["assess"]], datetime(2026, 10, 31, 7, 30, tzinfo=UTC), LONDON, "assess")
    assert line is not None and line.startswith("Based on data as of ")


# --- the limit lines --------------------------------------------------------------------------------------------------------------------------------


def test_blue_mill_10_to_11_has_its_approaching_limits_lines():
    lines = limit_lines(blue_mill_artifacts().values())
    assert "Over caution limit: gust spread 5.2 mph (caution at or above 4 mph)." in lines
    assert "Over caution limit: low cloud 98 % (caution at or above 90 %)." in lines
    assert "Over caution limit: visibility 2920 m (caution at or below 5000 m)." in lines


def test_a_value_over_a_no_go_limit_reads_over_limit_with_the_verdict_it_triggers():
    """Through the real tool: the weather blocker carries its numbers, so the line is built from them."""
    forecast = weather_forecast(utc(8), utc(16), changes={utc(h): {"gust_10m": 14.0, "wind_10m": 6.0, "wind_80m": 7.0, "wind_120m": 8.0} for h in (10, 11)})
    artifact = go(World(forecast=forecast), [calls("assess", **WINDOW)]).artifact("assess")
    assert "Over limit: gust spread 8 mph (NO-GO above 6 mph)." in limit_lines([artifact])


def test_every_value_clear_of_its_limits_gives_no_line():
    _, artifact = render(assess_calm(utc(10), utc(12)), (utc(10), utc(12)))
    assert limit_lines([artifact]) == []


def test_a_turn_with_no_assess_artifact_gives_no_line():
    artifacts = blue_mill_artifacts()
    assert limit_lines([artifacts["get_weather"], artifacts["get_airspace"], artifacts["get_notams"]]) == []
    assert limit_lines([]) == []


def test_a_value_exactly_on_a_caution_limit_reads_at_caution_limit():
    """A gust spread of exactly 4 mph is on its caution limit (`>=`), not over it."""
    forecast = weather_forecast(utc(9), utc(13), changes={utc(10): {"wind_10m": 6.0, "gust_10m": 10.0, "wind_80m": 7.0, "wind_120m": 8.0}})
    _, artifact = render(assess_calm(utc(10), utc(12), forecast=forecast), (utc(10), utc(12)))
    assert "At caution limit: gust spread 4 mph (caution at or above 4 mph)." in limit_lines([artifact])


def test_one_line_for_each_rule_whatever_the_number_of_hours():
    lines = limit_lines(blue_mill_artifacts().values())
    labels = [line.split(":")[1].split(" ")[1] for line in lines]
    assert len(labels) == len(set(labels))


# --- the assessment field ---------------------------------------------------------------------------------------------------------------------------


def test_no_assess_artifact_means_no_assessment():
    artifacts = blue_mill_artifacts()
    assert assessment_of([artifacts["get_weather"]]) is None and assessment_of([]) is None


def test_the_assessment_holds_the_tools_segments_and_overall():
    artifact = blue_mill_artifacts()["assess"]
    result = assessment_of([artifact])
    assert result is not None
    assert [(s.start, s.end) for s in result.segments] == [(s.start, s.end) for s in artifact.segments]
    assert result.overall is not None and result.overall.display == artifact.overall.display


def test_two_assess_calls_merge_in_start_order_and_the_later_replaces_the_same_span():
    first = blue_mill_artifacts()["assess"]
    other_hours = first.model_copy(update={"segments": [first.segments[0].model_copy(update={"start": "2026-10-31T16:00:00+00:00", "end": "2026-10-31T17:00:00+00:00"})]})
    changed = first.model_copy(update={"segments": [first.segments[0].model_copy(update={"conditions": ["later call"]})]})
    result = assessment_of([other_hours, first, changed])
    assert result is not None
    starts = [s.start for s in result.segments]
    assert starts == sorted(starts) and len(starts) == len(set(starts))
    assert next(s for s in result.segments if s.start == first.segments[0].start).conditions == ["later call"]


def test_the_worst_overall_among_the_calls_is_the_assessments():
    pilot = blue_mill_artifacts()["assess"]
    _, calm = render(assess_calm(utc(10), utc(12)), (utc(10), utc(12)))
    both = assessment_of([calm, pilot])
    assert both is not None and both.overall is not None
    assert both.overall.display == pilot.overall.display  # PILOT DISCRETION is worse than GO


# --- rehydration ------------------------------------------------------------------------------------------------------------------------------------

PHONES = {"<PHONE_1>": "01632 960001", "<PHONE_2>": "01632 960002"}


def test_each_known_placeholder_is_replaced_including_a_repeated_one():
    assert rehydrate("Call <PHONE_1>, or <PHONE_2>; again <PHONE_1>.", PHONES) == "Call 01632 960001, or 01632 960002; again 01632 960001."


def test_text_with_no_placeholder_is_unchanged():
    assert rehydrate("No numbers here.", PHONES) == "No numbers here."
    assert rehydrate("", PHONES) == ""


def test_an_unknown_placeholder_is_left_as_it_is_and_logged_by_name_only(caplog):
    with caplog.at_level(logging.WARNING, logger="flysafelz.agent.answer"):
        assert rehydrate("Call <PHONE_3> or <PHONE_1>.", PHONES) == "Call <PHONE_3> or 01632 960001."
    assert "<PHONE_3>" in caplog.text and "01632" not in caplog.text


def test_no_phone_map_leaves_every_placeholder():
    assert rehydrate("Call <PHONE_1>.", None) == "Call <PHONE_1>."
    assert rehydrate("Call <PHONE_1>.", {}) == "Call <PHONE_1>."


def test_the_assessment_and_overlays_get_the_numbers_back_and_the_originals_keep_their_placeholders():
    artifact = blue_mill_artifacts()["assess"]
    held = artifact.model_copy(update={"segments": [artifact.segments[0].model_copy(update={"conditions": ["phone <PHONE_1> first"]}), *artifact.segments[1:]]})
    assessment = assessment_of([held])
    assert assessment is not None
    restored = rehydrate_assessment(assessment, PHONES)
    assert restored is not None
    assert restored.segments[0].conditions == ["phone 01632 960001 first"]
    assert assessment.segments[0].conditions == ["phone <PHONE_1> first"] and held.segments[0].conditions == ["phone <PHONE_1> first"]
    assert rehydrate_assessment(None, PHONES) is None

    def feature(text):
        return OverlayFeature(geometry=PointGeometry(coordinates=[-0.08, 53.8]), properties=OverlayProperties(kind="notam", id="J3857/26", text=text, action="show"))

    plain, with_phone = feature("no number"), feature("ring <PHONE_1>")
    out = rehydrate_overlays([plain, with_phone], PHONES)
    assert out[0] is plain and out[1].properties.text == "ring 01632 960001" and with_phone.properties.text == "ring <PHONE_1>"
