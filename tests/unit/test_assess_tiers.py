"""G16 / E3 / E13: the tier of each hour, from a frozen `now` on day 0 (31 Oct 2026, Europe/London on GMT).

Day 3 is 3 Nov and day 7 is 7 Nov. The caps come from rules.yaml.
"""

from datetime import UTC, datetime, timedelta

import pytest
from factories import NOW, ONE_HOUR, forecast_failed, run_assess, utc, weather_forecast
from weather_helpers import rules_with

from flysafelz import assess as assess_module
from flysafelz.assess import fetch_horizon
from flysafelz.contracts import Tier, Verdict
from flysafelz.rules import RULES


def tiers(assessment) -> dict[datetime, Tier]:
    return {h.hour: h.tier for h in assessment.hours}


def hour_at(day: int, hour: int, month: int = 11) -> datetime:
    return utc(hour, day=day, month=month)


def test_the_caps_in_rules_yaml_are_day_3_and_day_7():
    assert RULES.verdict_horizon_day.value == 3
    assert RULES.outlook_horizon_day.value == 7


@pytest.mark.parametrize(
    ("hour", "tier"),
    [
        (utc(7), Tier.VERDICT),  # day 0
        (hour_at(3, 23), Tier.VERDICT),  # the last hour of day 3
        (hour_at(4, 0), Tier.OUTLOOK),  # the first hour of day 4
        (hour_at(7, 23), Tier.OUTLOOK),  # the last hour of day 7
        (hour_at(8, 0), Tier.TOO_FAR),  # the first hour of day 8
        (hour_at(9, 12), Tier.TOO_FAR),
    ],
)
def test_the_day_boundaries(hour, tier):
    window = (hour, hour + ONE_HOUR)
    assessment = run_assess(window=window)
    assert tiers(assessment) == {hour: tier}


def test_fetch_horizon_is_the_end_of_day_7_in_london_as_utc():
    assert fetch_horizon(NOW) == hour_at(8, 0)
    assert fetch_horizon(utc(23, 59)) == hour_at(8, 0)  # still day 0
    assert fetch_horizon(hour_at(1, 0)) == hour_at(9, 0)


def test_fetch_horizon_follows_the_london_date_across_the_clock_change():
    before_change = datetime(2026, 10, 24, 12, 0, tzinfo=UTC)  # BST. Day 7 is 31 Oct, which is on GMT.
    assert fetch_horizon(before_change) == datetime(2026, 11, 1, 0, 0, tzinfo=UTC)
    just_after_bst_midnight = datetime(2026, 10, 24, 23, 30, tzinfo=UTC)  # 00:30 BST on 25 Oct: that is day 0
    assert fetch_horizon(just_after_bst_midnight) == datetime(2026, 11, 2, 0, 0, tzinfo=UTC)


def test_a_window_wholly_beyond_day_7_is_all_too_far_with_no_verdict_and_no_error():
    assessment = run_assess(window=(hour_at(8, 0), hour_at(8, 6)))
    assert set(tiers(assessment).values()) == {Tier.TOO_FAR}
    assert len(assessment.hours) == 6
    assert assessment.overall_verdict is None
    assert assessment.overall_tier is Tier.TOO_FAR
    assert assessment.flyable_windows == []
    assert all(h.segments == [] for h in assessment.hours)


def test_a_too_far_window_with_a_failed_weather_source_is_still_too_far():
    assessment = run_assess(window=(hour_at(8, 0), hour_at(8, 3)), forecast=forecast_failed())
    assert set(tiers(assessment).values()) == {Tier.TOO_FAR}
    assert assessment.overall_verdict is None


def test_outlook_hours_keep_their_raw_weather_and_have_no_segments():
    hour = hour_at(5, 12)
    values = {"wind_10m": 11.0, "gust_10m": 14.0, "precipitation_probability": 35.0}
    forecast = weather_forecast(hour - ONE_HOUR, hour + ONE_HOUR, changes={hour: values, hour + ONE_HOUR: values})  # row HH reads samples HH and HH+1
    assessment = run_assess(window=(hour, hour + ONE_HOUR), forecast=forecast)
    (outlook,) = assessment.hours
    assert outlook.tier is Tier.OUTLOOK
    assert outlook.segments == []
    assert (outlook.weather.wind_10m, outlook.weather.gust_10m, outlook.weather.precipitation_probability) == (11.0, 14.0, 35.0)
    assert assessment.overall_verdict is None
    assert assessment.overall_tier is Tier.OUTLOOK


def test_data_that_ends_mid_day_2_gives_verdict_up_to_the_last_hour_then_outlook():
    # The last sample is 12:00, and row HH reads samples HH and HH+1, so row 11 is the last complete one.
    forecast = weather_forecast(hour_at(2, 8), hour_at(2, 12))
    assessment = run_assess(window=(hour_at(2, 10), hour_at(2, 16)), forecast=forecast)
    assert [t.value for t in tiers(assessment).values()] == ["VERDICT", "VERDICT", "OUTLOOK", "OUTLOOK", "OUTLOOK", "OUTLOOK"]
    assert assessment.overall_tier is Tier.VERDICT


def test_a_required_variable_missing_mid_run_makes_the_hours_that_read_it_unverified_with_the_weather_as_the_reason():
    hole = hour_at(2, 11)  # no CAPE in the 11:00 sample: rows 10 and 11 both read it
    forecast = weather_forecast(hour_at(2, 8), hour_at(2, 15), changes={hole: {"cape": None}})
    assessment = run_assess(window=(hour_at(2, 9), hour_at(2, 13)), forecast=forecast)
    by_hour = {h.hour.hour: h for h in assessment.hours}
    assert all(h.tier is Tier.VERDICT for h in assessment.hours)
    verdicts = {hour: h.segments[0].verdict for hour, h in by_hour.items()}
    assert verdicts == {9: Verdict.PILOT_DISCRETION, 10: Verdict.UNVERIFIED, 11: Verdict.UNVERIFIED, 12: Verdict.PILOT_DISCRETION}
    assert any("weather data missing" in c and "cape" in c for c in by_hour[11].segments[0].conditions)


def test_a_variable_missing_at_the_tail_makes_the_hours_from_there_on_outlook():
    forecast = weather_forecast(hour_at(2, 8), hour_at(2, 15), after={hour_at(2, 13): {"cape": None}})
    assessment = run_assess(window=(hour_at(2, 10), hour_at(2, 15)), forecast=forecast)
    assert [t.value for t in tiers(assessment).values()] == ["VERDICT", "VERDICT", "OUTLOOK", "OUTLOOK", "OUTLOOK"]


def test_a_failed_weather_source_keeps_days_0_to_3_as_verdict_and_the_rest_outlook():
    forecast = forecast_failed()
    assessment = run_assess(window=(hour_at(3, 22), hour_at(4, 2)), forecast=forecast)
    assert [t.value for t in tiers(assessment).values()] == ["VERDICT", "VERDICT", "OUTLOOK", "OUTLOOK"]
    assert all(s.verdict is Verdict.NO_GO or s.verdict is Verdict.UNVERIFIED for h in assessment.hours for s in h.segments)


def test_a_window_across_midnight_and_the_clock_change_has_one_row_for_each_utc_hour():
    now = datetime(2026, 10, 24, 22, 30, tzinfo=UTC)  # 23:30 BST on 24 Oct: day 0 is 24 Oct
    start, end = datetime(2026, 10, 24, 22, 0, tzinfo=UTC), datetime(2026, 10, 25, 4, 0, tzinfo=UTC)  # the change is at 01:00 UTC
    assessment = run_assess(window=(start, end), now=now)
    hours = [h.hour for h in assessment.hours]
    assert hours == [start + timedelta(hours=i) for i in range(6)]
    assert len(set(hours)) == 6


def test_day_numbering_follows_london_across_the_clock_change():
    # Day 0 is 24 Oct (BST) so day 3 is 27 Oct (GMT) and its last hour is 23:00 UTC; 28 Oct 00:00 UTC is day 4.
    now = datetime(2026, 10, 24, 22, 30, tzinfo=UTC)
    start, end = datetime(2026, 10, 27, 22, 0, tzinfo=UTC), datetime(2026, 10, 28, 2, 0, tzinfo=UTC)
    assessment = run_assess(window=(start, end), now=now)
    assert [t.value for t in tiers(assessment).values()] == ["VERDICT", "VERDICT", "OUTLOOK", "OUTLOOK"]


def test_the_outlook_cap_is_the_rules_file_not_a_number_in_the_code(tmp_path, monkeypatch):
    monkeypatch.setattr(assess_module, "RULES", rules_with(tmp_path, "outlook_horizon_day", 5))
    assert fetch_horizon(NOW) == hour_at(6, 0)
    assessment = run_assess(window=(hour_at(5, 23), hour_at(6, 1)))
    assert [t.value for t in tiers(assessment).values()] == ["OUTLOOK", "TOO_FAR"]
