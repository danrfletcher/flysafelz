"""The assess tool's artifact: its schema, that every number the text prints is in it, and that no verdict string stands where there is no verdict (G12, E7, E9, I7)."""

import json
import re
from datetime import UTC, datetime, timedelta
from itertools import product

import cases
import pytest
from assess_tool_helpers import LONDON, NOW, SIX, assess_calm, render, utc
from factories import airspace_fetch, forecast_failed, pib_result, run_assess, status, wind
from factories import weather_forecast as forecast_between

from flysafelz.agent.sources import SourceCache
from flysafelz.agent.tools.artifacts import AssessArtifact
from flysafelz.agent.tools.assess_tool import build_assess
from flysafelz.contracts import (
    Assessment,
    Blocker,
    Caution,
    HourAssessment,
    Segment,
    SourceName,
    SourceState,
    Tier,
    Verdict,
    display_verdict,
)

HOUR = timedelta(hours=1)
PILOT_DISCRETION_WORDS = "PILOT DISCRETION"


def dump(artifact: AssessArtifact) -> str:
    return json.dumps(artifact.model_dump(mode="json"), ensure_ascii=False)


def numbers_in(text: str) -> set[float]:
    return {float(n) for n in re.findall(r"\d+(?:\.\d+)?", text)}


def strip_non_measures(text: str) -> str:
    """Drop what is not a measured number: clock times, ISO dates, day-of-month in `31 Oct`, NOTAM ids, `<PHONE_n>`, `H7171/26`, `EGD496A`."""
    text = re.sub(r"\d{4}-\d{2}-\d{2}", " ", text)
    text = re.sub(r"\b\d{1,2}:\d{2}\b", " ", text)
    text = re.sub(r"\b\d{1,2} (?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\b", " ", text)
    text = re.sub(r"\b[A-Z]\d{4}/\d{2}\b", " ", text)
    text = re.sub(r"\b[A-Z]{2,}\d+[A-Z]?\b", " ", text)
    text = re.sub(r"<PHONE_\d+>", " ", text)
    return re.sub(r"\bAIP SUP \d+/\d+\b", " ", text)


# --- the schema the model sees ---------------------------------------------------------------------------------------


def test_the_tool_takes_exactly_start_and_end_and_nothing_the_thread_or_the_run_owns():
    tool = build_assess(SourceCache())
    assert set(tool.args) == {"start", "end"}
    schema = tool.tool_call_schema.model_json_schema()
    assert set(schema["properties"]) == {"start", "end"} and not schema.get("required")
    for owned in ("lat", "lon", "tz", "now", "case"):
        assert owned not in tool.args
    assert tool.description and "only source of a verdict" in tool.description


# --- display strings -------------------------------------------------------------------------------------------------


def synthetic(verdict: Verdict, approaching: bool, conditions: list[str]) -> Assessment:
    """One VERDICT-tier hour holding one hand-built segment."""
    hour = utc(9)
    segment = Segment(
        start=hour,
        end=hour + HOUR,
        verdict=verdict,
        approaching_limits=approaching,
        conditions=conditions,
        cautions=[Caution(rule_id="gust_spread_caution", text="gust spread 5 mph, limit 4", value=5.0, limit=4.0)] if approaching else [],
        blockers=[Blocker(source="weather", text="wind 20 mph")] if verdict is Verdict.NO_GO else [],
    )
    statuses = [status(name) for name in SourceName]
    return Assessment(
        window_start=hour,
        window_end=hour + HOUR,
        hours=[HourAssessment(hour=hour, tier=Tier.VERDICT, segments=[segment])],
        overall_verdict=verdict,
        overall_tier=Tier.VERDICT,
        source_statuses=statuses,
    )


COMBINATIONS = [
    pytest.param(Verdict.NO_GO, False, [], id="no-go"),
    pytest.param(Verdict.UNVERIFIED, False, ["UNVERIFIED: weather source failed"], id="unverified"),
    pytest.param(Verdict.PILOT_DISCRETION, False, [], id="discretion"),
    pytest.param(Verdict.PILOT_DISCRETION, True, [], id="discretion-limits"),
    pytest.param(Verdict.PILOT_DISCRETION, False, ["phone the range"], id="discretion-conditions"),
    pytest.param(Verdict.PILOT_DISCRETION, True, ["phone the range"], id="discretion-both"),
]


@pytest.mark.parametrize(("verdict", "approaching", "conditions"), COMBINATIONS)
def test_a_segments_display_is_display_verdict_of_its_own_three_machine_fields(verdict, approaching, conditions):
    text, artifact = render(synthetic(verdict, approaching, conditions), (utc(9), utc(10)))
    [segment] = artifact.segments
    assert (segment.verdict, segment.approaching_limits, segment.with_conditions) == (verdict.value, approaching, bool(conditions))
    assert segment.display == display_verdict(verdict, approaching, bool(conditions))
    assert segment.display in SIX
    assert artifact.overall is not None and artifact.overall.display == segment.display
    assert (artifact.overall.verdict, artifact.overall.approaching_limits, artifact.overall.with_conditions) == (verdict.value, approaching, bool(conditions))
    assert f"Overall: {segment.display}" in text


def test_the_six_combinations_give_six_different_display_strings():
    shown = {display_verdict(Verdict.NO_GO, False, False), display_verdict(Verdict.UNVERIFIED, False, False)}
    shown |= {display_verdict(Verdict.PILOT_DISCRETION, a, c) for a, c in product((False, True), repeat=2)}
    assert shown == SIX and len(SIX) == 6


def test_the_overall_machine_fields_are_those_of_the_worst_segments_only():
    hour = utc(9)
    worse = Segment(start=hour, end=hour + HOUR / 2, verdict=Verdict.UNVERIFIED, conditions=["UNVERIFIED: x"])
    better = Segment(start=hour + HOUR / 2, end=hour + HOUR, verdict=Verdict.PILOT_DISCRETION, approaching_limits=True, conditions=["c"])
    assessment = synthetic(Verdict.UNVERIFIED, False, []).model_copy(update={"hours": [HourAssessment(hour=hour, tier=Tier.VERDICT, segments=[better, worse])]})
    _, artifact = render(assessment, (hour, hour + HOUR))
    assert artifact.overall is not None
    assert (artifact.overall.verdict, artifact.overall.approaching_limits, artifact.overall.with_conditions, artifact.overall.display) == (
        "UNVERIFIED",
        False,
        True,
        "UNVERIFIED",
    )


# --- no verdict where there is none ----------------------------------------------------------------------------------


def contains_a_verdict_string(value: str) -> bool:
    return any(six in value for six in SIX) or PILOT_DISCRETION_WORDS in value or "NO-GO" in value or "UNVERIFIED" in value.upper().replace("UNVERIFIED:", "")


def test_an_outlook_only_window_has_no_verdict_string_in_the_text_or_the_artifact():
    first = datetime(2026, 11, 5, 10, tzinfo=UTC)
    window = (first, first + 3 * HOUR)
    text, artifact = render(run_assess(window=window, forecast=forecast_between(utc(6), datetime(2026, 11, 8, 23, tzinfo=UTC), changes={first: wind(8)})), window)
    for word in (*SIX, PILOT_DISCRETION_WORDS, "NO-GO", "UNVERIFIED", "GO WITH CONDITIONS", "MARGINAL"):
        assert word not in text and word not in dump(artifact), word
    assert artifact.overall is None and artifact.overall_tier == "OUTLOOK"
    assert all(s.verdict is None and s.display is None and not s.conditions and not s.cautions and not s.blockers for s in artifact.segments)


def test_a_too_far_only_window_has_no_verdict_string_in_the_text_or_the_artifact():
    first = datetime(2026, 11, 9, 10, tzinfo=UTC)
    text, artifact = render(run_assess(window=(first, first + 2 * HOUR)), (first, first + 2 * HOUR))
    for word in (*SIX, PILOT_DISCRETION_WORDS, "NO-GO", "UNVERIFIED"):
        assert word not in text and word not in dump(artifact), word
    assert artifact.overall is None and artifact.overall_tier == "TOO_FAR"


FAILED = {
    SourceName.WEATHER: {"forecast": forecast_failed()},
    SourceName.AIRSPACE: {"airspace": airspace_fetch(state=SourceState.FAILED)},
    SourceName.NOTAMS: {"pib": pib_result(state=SourceState.FAILED)},
}


@pytest.mark.parametrize("source", list(SourceName), ids=lambda s: s.value)
def test_a_failed_source_leaves_no_pilot_discretion_string_anywhere(source):
    window = (utc(9), utc(12))
    text, artifact = render(assess_calm(*window, **FAILED[source]), window)
    assert PILOT_DISCRETION_WORDS not in text and PILOT_DISCRETION_WORDS not in dump(artifact)
    assert PILOT_DISCRETION_WORDS.replace(" ", "_") not in dump(artifact)  # nor the enum value
    assert {s.display for s in artifact.segments if s.tier == "VERDICT"} <= {"UNVERIFIED", "NO-GO"}
    assert {s.verdict for s in artifact.segments} <= {"UNVERIFIED", "NO_GO"}
    assert artifact.overall is not None and artifact.overall.display in {"UNVERIFIED", "NO-GO"}


# --- every printed number is in the artifact -------------------------------------------------------------------------


BEARING_IN_TEXT = re.compile(r"\bbearing (\d+(?:\.\d+)?)\u00b0")
SPAN_IN_TEXT = re.compile(r"flyable for (\d+) h (\d{2}) min")


def measured_numbers_in_text(text: str) -> set[float]:
    """The measured numbers the text prints, except a NOTAM's bearing (its own rounding rule, `bearings_in_text`) and a span's `X h YY min` (the artifact's minutes, `spans_in_text`)."""
    return numbers_in(strip_non_measures(SPAN_IN_TEXT.sub(" ", BEARING_IN_TEXT.sub(" ", text))))


def spans_in_text(text: str) -> list[int]:
    return [int(h) * 60 + int(m) for h, m in SPAN_IN_TEXT.findall(text)]


def bearings_in_text(text: str) -> list[str]:
    return BEARING_IN_TEXT.findall(text)


def blue_mill_case(first: datetime, last: datetime):
    """Ticket 2's own answer for the committed Blue Mill case over [first, last): a NOTAM that gives a bearing (H7171/26, avoid that sector)."""
    return render(cases.run(cases.BLUE_MILL, window=(first, last), tz=LONDON, live=False), (first, last), now=cases.load(cases.BLUE_MILL)[0].now)


def window_cases():
    first = datetime(2026, 11, 5, 10, tzinfo=UTC)
    outlook_forecast = forecast_between(utc(6), datetime(2026, 11, 8, 23, tzinfo=UTC), changes={first: wind(8), first + HOUR: wind(14)})
    mixed = (datetime(2026, 11, 3, 22, tzinfo=UTC), datetime(2026, 11, 9, 2, tzinfo=UTC))
    return [
        pytest.param(lambda: (assess_calm(utc(9), utc(12)), (utc(9), utc(12))), id="clear"),
        pytest.param(lambda: (run_assess(window=(first, first + 3 * HOUR), forecast=outlook_forecast), (first, first + 3 * HOUR)), id="outlook"),
        pytest.param(lambda: (run_assess(window=mixed, forecast=forecast_between(utc(6), datetime(2026, 11, 8, 23, tzinfo=UTC))), mixed), id="mixed"),
        pytest.param(
            lambda: (
                assess_calm(
                    utc(9),
                    utc(13),
                    forecast=forecast_between(utc(8), utc(14), changes={utc(10): wind(11), utc(11): {"visibility": 3000.0}}),
                ),
                (utc(9), utc(13)),
            ),
            id="cautions",
        ),
    ]


@pytest.mark.parametrize("make", window_cases())
def test_every_number_the_text_prints_sits_in_the_artifact_as_a_number(make):
    assessment, window = make()
    text, artifact = render(assessment, window)
    in_artifact = numbers_in(json.dumps(artifact.model_dump(mode="json")))
    missing = measured_numbers_in_text(text) - in_artifact
    assert missing == set(), missing
    assert spans_in_text(text) == [s.minutes for s in artifact.flyable]
    assert bearings_in_text(text) == [f"{n.bearing_deg:.0f}" for s in artifact.segments for n in s.notams if n.bearing_deg is not None]


@pytest.mark.parametrize(("first", "last"), [(utc(10), utc(16)), (utc(7), utc(18))], ids=["blue-mill-gp4", "blue-mill-whole-day"])
def test_blue_mill_every_number_the_text_prints_sits_in_the_artifact_and_a_bearing_is_the_artifacts_in_whole_degrees(first, last):
    """The one rounding rule: the text prints a NOTAM's bearing in whole degrees (as `get_notams` does); every other number is in the artifact as printed."""
    text, artifact = blue_mill_case(first, last)
    in_artifact = numbers_in(json.dumps(artifact.model_dump(mode="json")))
    assert measured_numbers_in_text(text) - in_artifact == set()
    held = [n.bearing_deg for s in artifact.segments for n in s.notams if n.bearing_deg is not None]
    assert held and any(b != round(b) for b in held), "the case must hold a bearing that is not already whole"
    assert bearings_in_text(text) == [f"{b:.0f}" for b in held]  # one printed bearing per NOTAM line, in order
    assert spans_in_text(text) == [s.minutes for s in artifact.flyable]
    assert "bearing 169\u00b0" in text and "168.6" not in text


def test_the_counts_flyable_spans_and_hours_in_the_text_equal_the_artifacts():
    assessment, window = assess_calm(utc(9), utc(12)), (utc(9), utc(12))
    text, artifact = render(assessment, window)
    counts = artifact.counts
    assert f"{counts.notams_within} NOTAMs within {counts.local_area_nm:g} NM. {counts.hours_assessed} h assessed, {counts.hours_outlook} OUTLOOK, {counts.hours_too_far} too far." in text
    assert [f"flyable for {s.minutes // 60} h {s.minutes % 60:02d} min" in text for s in artifact.flyable] == [True] * len(artifact.flyable)
    assert counts.flyable_windows == len(artifact.flyable) and counts.hours_assessed == sum(1 for s in artifact.segments if s.tier == "VERDICT")


def test_a_caution_carries_its_rule_value_limit_and_the_units_of_the_rule():
    window = (utc(9), utc(13))
    forecast = forecast_between(utc(8), utc(14), changes={utc(10): {**wind(11), "gust_10m": 17.0}, utc(11): {"visibility": 3000.0}})
    _, artifact = render(assess_calm(*window, forecast=forecast), window)
    cautions = [c for s in artifact.segments for c in s.cautions]
    assert cautions, "the forecast should approach a limit"
    for caution in cautions:
        assert caution.rule_id and caution.value is not None and caution.limit is not None
        assert caution.unit  # every caution's rule has a unit in rules.yaml
        assert f"{caution.value:g}" in caution.text and f"{caution.limit:g}" in caution.text


def test_the_artifact_is_json_and_msgpack_safe_and_keeps_the_zone():
    _, artifact = render(assess_calm(utc(9), utc(12)), (utc(9), utc(12)))
    again = AssessArtifact.model_validate(json.loads(artifact.model_dump_json()))
    assert again == artifact
    assert artifact.tz == LONDON and artifact.window_start == utc(9) and artifact.window_end == utc(12)
    assert artifact.tool == "assess" and artifact.fetched_at == NOW
