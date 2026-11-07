"""G11 / E2 / E17, end to end: capture a `days` case into tmp_path, replay it, and assess its whole window.

The capture goes through capture.main with a MultiDayServer (a fake Open-Meteo, openAIP and GitHub that
builds its hourly rows from the request's own dates; tests/capture_helpers.py), then
sources.load_case_sources and cases.load_case_inputs replay the files and assess() cuts the window.
The canned openAIP pages and PIB are the ones in tests/data/capture/. No network, no key, no clock.
"""

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from capture_helpers import case_yaml, make_root, run, server_for

from flysafelz.assess import assess
from flysafelz.cases import load_case_inputs, load_case_inputs_by_name
from flysafelz.contracts import Assessment, SourceState, Tier
from flysafelz.sources import CaseSources, load_case_sources

ONE_HOUR = timedelta(hours=1)


def utc(*parts: int) -> datetime:
    return datetime(*parts, tzinfo=UTC)


def stamp(t: datetime) -> str:
    return t.strftime("%Y-%m-%dT%H:%M:%SZ")


def capture_case(tmp_path: Path, name: str, *, now: datetime, start: datetime, end: datetime, days: int) -> Path:
    root = make_root(tmp_path, name, case_yaml(now=stamp(now), start=stamp(start), end=stamp(end), days=days))
    assert run(root, name, server_for(now)) == 0
    return root


def assess_case(sources: CaseSources) -> Assessment:
    case = sources.case
    return assess(
        case.lat,
        case.lon,
        now=case.now,
        forecast=sources.forecast,
        airspace=sources.airspace,
        pib=sources.pib,
        readings=sources.readings,
        window=(case.window.start, case.window.end),
    )


def hour_of(assessment: Assessment, hour: datetime):
    return next(h for h in assessment.hours if h.hour == hour)


def hours_between(assessment: Assessment, first: datetime, end: datetime):
    return [h for h in assessment.hours if first <= h.hour < end]


@pytest.fixture
def gmt(tmp_path):
    """Blue Mill, now 2026-10-31T07:30Z (GMT), five London days: 31 Oct to 4 Nov."""
    name = "blue_mill_2026-10-31_5day"
    root = capture_case(tmp_path, name, now=utc(2026, 10, 31, 7, 30), start=utc(2026, 10, 31, 7), end=utc(2026, 11, 5), days=5)
    return root, name


@pytest.fixture
def bst(tmp_path):
    """Blue Mill, now 2026-09-01T07:30Z (BST), five London days: 1 to 5 Sep."""
    name = "blue_mill_2026-09-01_5day"
    root = capture_case(tmp_path, name, now=utc(2026, 9, 1, 7, 30), start=utc(2026, 9, 1, 7), end=utc(2026, 9, 5, 23), days=5)
    return root, name


def test_e2_day_3_hours_carry_a_verdict_and_every_day_4_hour_is_outlook_with_none(gmt):
    root, name = gmt
    assessment = assess_case(load_case_sources(name, root=root))
    day_3 = hours_between(assessment, utc(2026, 11, 3), utc(2026, 11, 4))  # 3 Nov, London day 3 (GMT)
    day_4 = hours_between(assessment, utc(2026, 11, 4), utc(2026, 11, 5))
    assert len(day_3) == 24 and len(day_4) == 24
    for hour in day_3:
        assert hour.tier is Tier.VERDICT and hour.segments, hour.hour
        assert all(segment.verdict is not None for segment in hour.segments)
    for hour in day_4:
        assert hour.tier is Tier.OUTLOOK and hour.segments == [], hour.hour
        assert hour.weather.wind_10m is not None  # the raw weather is kept for the outlook hours
    assert assessment.overall_tier is Tier.VERDICT and assessment.overall_verdict is not None


def test_assess_returns_a_row_for_every_hour_of_the_multi_day_window(gmt):
    root, name = gmt
    assessment = assess_case(load_case_sources(name, root=root))
    start, end = utc(2026, 10, 31, 7), utc(2026, 11, 5)
    assert [h.hour for h in assessment.hours] == [start + i * ONE_HOUR for i in range(int((end - start) / ONE_HOUR))]
    assert len(assessment.hours) == 113
    assert {h.tier for h in assessment.hours} == {Tier.VERDICT, Tier.OUTLOOK}
    assert all(h.tier is Tier.VERDICT for h in assessment.hours if h.hour < utc(2026, 11, 4))


def test_e2_e17_in_bst_the_day_3_to_day_4_boundary_is_at_23_00_utc_not_00_00(bst):
    root, name = bst
    assessment = assess_case(load_case_sources(name, root=root))
    last_of_day_3 = hour_of(assessment, utc(2026, 9, 4, 22))  # 23:00 BST on 4 Sep
    first_of_day_4 = hour_of(assessment, utc(2026, 9, 4, 23))  # 00:00 BST on 5 Sep
    assert last_of_day_3.tier is Tier.VERDICT and last_of_day_3.segments
    assert all(segment.verdict is not None for segment in last_of_day_3.segments)
    assert first_of_day_4.tier is Tier.OUTLOOK and first_of_day_4.segments == []
    # a one-hour shift either way would move one of these
    assert hour_of(assessment, utc(2026, 9, 4, 21)).tier is Tier.VERDICT
    assert hour_of(assessment, utc(2026, 9, 5, 0)).tier is Tier.OUTLOOK
    day_4 = hours_between(assessment, utc(2026, 9, 4, 23), utc(2026, 9, 5, 23))
    assert len(day_4) == 24 and all(h.tier is Tier.OUTLOOK and h.segments == [] for h in day_4)
    assert hour_of(assessment, utc(2026, 9, 5, 22)).tier is Tier.OUTLOOK  # the window's last hour


def test_the_replay_side_loads_a_days_case_through_both_loaders(gmt):
    root, name = gmt
    sources = load_case_sources(name, root=root)
    assert sources.case.days == 5 and sources.case.window.end == utc(2026, 11, 5)
    assert len(sources.forecast.samples) == 121 and sources.forecast.status.state is SourceState.OK
    assert sources.pib.status.sha is not None  # case.yaml's recorded capture sha, as for any captured case
    case, forecast, _airspace, _pib, _readings = load_case_inputs_by_name(name, root=root)  # today = the case's own `now`
    assert case.days == 5 and len(forecast.samples) == 121
    same = load_case_inputs(root / "fixtures" / name)
    assert same[0] == case and same[1] == forecast
