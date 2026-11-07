"""E1 / E2: Blue Mill with each source failed in turn, and with a stale PIB. A failed source never gives PILOT_DISCRETION."""

from datetime import timedelta

import cases
import pytest
from factories import airspace_fetch, forecast_failed, pib_result, utc

from flysafelz.contracts import SourceName, SourceState, Verdict
from flysafelz.rules import RULES

CASE = cases.BLUE_MILL


def segments(assessment):
    return [s for h in assessment.hours for s in h.segments]


def only_night(segment) -> bool:
    return [b.source for b in segment.blockers] == ["daylight"]


def status_of(assessment, source):
    return next(s for s in assessment.source_statuses if s.source is source)


def failed(source: SourceName):
    if source is SourceName.WEATHER:
        return {"forecast": forecast_failed()}
    if source is SourceName.AIRSPACE:
        return {"airspace": airspace_fetch(state=SourceState.FAILED)}
    return {"pib": pib_result(state=SourceState.FAILED)}


@pytest.mark.parametrize("source", [SourceName.WEATHER, SourceName.AIRSPACE, SourceName.NOTAMS])
def test_a_failed_source_makes_every_verdict_tier_segment_unverified_unless_another_source_gives_no_go(source):
    assessment = cases.run(CASE, **failed(source))
    found = segments(assessment)
    assert found
    for segment in found:
        assert segment.verdict is not Verdict.PILOT_DISCRETION, segment.start
        if segment.verdict is Verdict.NO_GO:
            assert segment.blockers  # a working source said no
            assert all(b.source != source.value for b in segment.blockers)
        else:
            assert segment.verdict is Verdict.UNVERIFIED
            assert any(c.startswith("UNVERIFIED: ") and f"{source.value} source failed" in c for c in segment.conditions)
    assert assessment.flyable_windows == []


@pytest.mark.parametrize("source", [SourceName.WEATHER, SourceName.AIRSPACE, SourceName.NOTAMS])
def test_the_failed_source_is_listed_with_its_reason_and_the_others_are_ok(source):
    assessment = cases.run(CASE, **failed(source))
    states = {s.source: s.state for s in assessment.source_statuses}
    assert states.pop(source) is SourceState.FAILED
    assert set(states.values()) == {SourceState.OK}
    assert status_of(assessment, source).reason


@pytest.mark.parametrize("source", [SourceName.AIRSPACE, SourceName.NOTAMS])
def test_night_is_still_no_go_with_the_airspace_or_notam_source_failed(source):
    assessment = cases.run(CASE, **failed(source))
    night = [s for s in segments(assessment) if s.start >= utc(17, 6)]
    assert night and all(s.verdict is Verdict.NO_GO and only_night(s) for s in night)
    assert status_of(assessment, source).state is SourceState.FAILED  # still listed


def test_night_is_still_no_go_with_the_weather_source_failed():
    assessment = cases.run(CASE, **failed(SourceName.WEATHER))
    night = [s for s in segments(assessment) if s.start >= utc(17, 6)]
    assert night and all(s.verdict is Verdict.NO_GO and only_night(s) for s in night)
    day = [s for s in segments(assessment) if s.end <= utc(17)]
    assert day and all(s.verdict is Verdict.UNVERIFIED for s in day if not s.blockers)


def test_bad_weather_is_still_no_go_with_the_notam_source_failed():
    assessment = cases.run(CASE, **failed(SourceName.NOTAMS))
    eight = next(s for s in segments(assessment) if s.start == utc(8))
    assert eight.verdict is Verdict.NO_GO and [b.source for b in eight.blockers] == ["weather"]
    assert assessment.overall_verdict is Verdict.NO_GO


def test_a_stale_pib_is_treated_like_a_failed_one():
    hours = RULES.pib_stale_hours.value
    assert not isinstance(hours, list)
    case, _forecast, _airspace, pib, _readings = cases.load(CASE)
    old = pib.status.model_copy(update={"issued_at": case.now - timedelta(hours=hours + 1)})
    assessment = cases.run(CASE, pib=pib.model_copy(update={"status": old}))
    status = status_of(assessment, SourceName.NOTAMS)
    assert status.state is SourceState.STALE
    assert "PIB issued" in status.reason
    for segment in segments(assessment):
        assert segment.verdict in {Verdict.NO_GO, Verdict.UNVERIFIED}, segment.start
        if segment.verdict is Verdict.UNVERIFIED:
            assert any("notams source stale" in c for c in segment.conditions)
    assert assessment.flyable_windows == []


def test_a_pib_exactly_at_the_stale_limit_is_not_stale():
    hours = RULES.pib_stale_hours.value
    assert not isinstance(hours, list)
    case, _forecast, _airspace, pib, _readings = cases.load(CASE)
    edge = pib.status.model_copy(update={"issued_at": case.now - timedelta(hours=hours)})
    assessment = cases.run(CASE, pib=pib.model_copy(update={"status": edge}))
    assert status_of(assessment, SourceName.NOTAMS).state is SourceState.OK


def test_with_every_source_ok_the_listed_statuses_are_ok():
    assert {s.state for s in cases.run(CASE).source_statuses} == {SourceState.OK}
