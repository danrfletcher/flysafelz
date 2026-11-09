"""G5 / E3: the data tier of each hour, from the rows of the run."""

import json
from datetime import UTC, datetime

import pytest
from weather_helpers import FIXTURES, NOW, calm, calm_forecast, forecast_of, rules_with

from flysafelz import weather
from flysafelz.clients.open_meteo import parse_response
from flysafelz.contracts import Tier
from flysafelz.daylight import london_day_offset
from flysafelz.weather import REQUIRED, build_rows, weather_tier


def rows_from_fixture(name, start, end):
    body = json.loads((FIXTURES / f"{name}.json").read_text(encoding="utf-8"))
    return build_rows(parse_response(body, NOW), start, end)


def utc(*args):
    return datetime(*args, tzinfo=UTC)


# --- tail missing: OUTLOOK, not UNVERIFIED ------------------------------------------------------------


def test_upper_wind_missing_at_the_tail_makes_those_hours_outlook_not_unverified():
    rows = rows_from_fixture("tail_missing_80m", utc(2026, 11, 1, 0, 0), utc(2026, 11, 2, 0, 0))
    # 80 m and 120 m are null from 21:00, so rows 20:00 to 23:00 lack the closing value.
    last_complete = utc(2026, 11, 1, 19, 0)
    assert [r.hour for r in rows if not r.missing_required][-1] == last_complete
    for hour in (utc(2026, 11, 1, 20, 0), utc(2026, 11, 1, 21, 0), utc(2026, 11, 1, 23, 0)):
        tier = weather_tier(hour, rows, NOW)
        assert tier.tier is Tier.OUTLOOK
        assert tier.unverified is False and tier.missing == []
    for hour in (utc(2026, 11, 1, 0, 0), last_complete):
        tier = weather_tier(hour, rows, NOW)
        assert tier.tier is Tier.VERDICT and tier.unverified is False


def test_outlook_hours_keep_their_raw_values_for_the_printout():
    rows = rows_from_fixture("tail_missing_80m", utc(2026, 11, 1, 0, 0), utc(2026, 11, 2, 0, 0))
    tail = next(r for r in rows if r.hour == utc(2026, 11, 1, 22, 0))
    assert weather_tier(tail.hour, rows, NOW).tier is Tier.OUTLOOK
    assert tail.values.wind_10m is not None
    assert tail.values.gust_10m is not None
    assert tail.values.precipitation_probability is not None
    assert tail.values.wind_80m is None and tail.missing_required == ["wind_80m", "wind_120m"]


# --- a hole inside the run: UNVERIFIED ----------------------------------------------------------------


def test_a_missing_cape_inside_the_run_makes_only_the_affected_hours_unverified():
    rows = rows_from_fixture("hole_in_cape", utc(2026, 10, 31, 0, 0), utc(2026, 11, 2, 0, 0))
    # cape is null at 12:00, which the 11:00 row (closing) and the 12:00 row (opening) both need.
    unverified = [r.hour for r in rows if weather_tier(r.hour, rows, NOW).unverified]
    assert unverified == [utc(2026, 10, 31, 11, 0), utc(2026, 10, 31, 12, 0)]
    tier = weather_tier(utc(2026, 10, 31, 12, 0), rows, NOW)
    assert tier.tier is Tier.VERDICT
    assert tier.missing == ["cape"]
    assert "cape" in tier.reason and "weather" in tier.reason
    for hour in (utc(2026, 10, 31, 10, 0), utc(2026, 10, 31, 13, 0)):
        assert weather_tier(hour, rows, NOW).unverified is False


def test_a_hole_in_the_last_window_rows_is_unverified_when_the_data_goes_on_after_the_window():
    # The same hole as above (cape null at 12:00), but the window ends at 13:00, so the
    # 11:00 and 12:00 rows are the last two of the window while the response continues to 2 Nov.
    rows = rows_from_fixture("hole_in_cape", utc(2026, 10, 31, 7, 0), utc(2026, 10, 31, 13, 0))
    assert rows[-1].hour == utc(2026, 10, 31, 12, 0)
    for hour in (utc(2026, 10, 31, 11, 0), utc(2026, 10, 31, 12, 0)):
        tier = weather_tier(hour, rows, NOW)
        assert tier.tier is Tier.VERDICT and tier.unverified is True
        assert tier.missing == ["cape"]
    assert weather_tier(utc(2026, 10, 31, 10, 0), rows, NOW).unverified is False

    # A wider window over the same response gives the same answer for those hours.
    wide = rows_from_fixture("hole_in_cape", utc(2026, 10, 31, 0, 0), utc(2026, 11, 2, 0, 0))
    for hour in (utc(2026, 10, 31, 11, 0), utc(2026, 10, 31, 12, 0)):
        assert weather_tier(hour, rows, NOW) == weather_tier(hour, wide, NOW)


def test_a_tail_hole_stays_outlook_when_the_window_ends_inside_the_tail():
    # 80 m and 120 m are null from 21:00 to the end of the response: a tail, whatever the window.
    rows = rows_from_fixture("tail_missing_80m", utc(2026, 11, 1, 18, 0), utc(2026, 11, 1, 22, 0))
    assert [weather_tier(r.hour, rows, NOW).tier for r in rows] == [Tier.VERDICT, Tier.VERDICT, Tier.OUTLOOK, Tier.OUTLOOK]
    assert not any(weather_tier(r.hour, rows, NOW).unverified for r in rows)


def test_a_window_past_the_end_of_the_response_is_outlook():
    rows = rows_from_fixture("hole_in_cape", utc(2026, 11, 2, 20, 0), utc(2026, 11, 3, 4, 0))
    assert {weather_tier(r.hour, rows, NOW).tier for r in rows} == {Tier.OUTLOOK}


def test_two_missing_variables_in_one_hour_are_both_named():
    samples = calm_forecast(utc(2026, 10, 31, 8, 0), utc(2026, 10, 31, 14, 0)).samples
    samples[utc(2026, 10, 31, 10, 0)] = calm(visibility=None, wind_120m=None)
    rows = build_rows(forecast_of(samples), utc(2026, 10, 31, 8, 0), utc(2026, 10, 31, 14, 0))
    tier = weather_tier(utc(2026, 10, 31, 10, 0), rows, NOW)
    assert tier.unverified and tier.missing == ["wind_120m", "visibility"]


def test_an_optional_variable_missing_keeps_the_tier():
    samples = {t: calm(precipitation_probability=None, cloud_cover=None, direction_10m=None) for t in calm_forecast(utc(2026, 10, 31, 8, 0), utc(2026, 10, 31, 12, 0)).samples}
    rows = build_rows(forecast_of(samples), utc(2026, 10, 31, 8, 0), utc(2026, 10, 31, 11, 0))
    for row in rows:
        tier = weather_tier(row.hour, rows, NOW)
        assert tier.tier is Tier.VERDICT and not tier.unverified
        assert row.notes == ["rain chance not available", "total cloud not available", "wind direction not available"]


def test_an_hour_with_no_row_inside_the_run_is_unverified_with_every_variable_named():
    rows = build_rows(calm_forecast(utc(2026, 10, 31, 8, 0), utc(2026, 10, 31, 14, 0)), utc(2026, 10, 31, 8, 0), utc(2026, 10, 31, 14, 0))
    gap = [r for r in rows if r.hour != utc(2026, 10, 31, 10, 0)]
    tier = weather_tier(utc(2026, 10, 31, 10, 0), gap, NOW)
    assert tier.unverified and tier.missing == list(REQUIRED)


def test_when_no_hour_has_every_variable_every_hour_is_outlook():
    rows = build_rows(forecast_of({}), utc(2026, 10, 31, 8, 0), utc(2026, 10, 31, 10, 0))
    assert [weather_tier(r.hour, rows, NOW).tier for r in rows] == [Tier.OUTLOOK, Tier.OUTLOOK]


# --- the day 3 cap (day 0 is today in Europe/London) -------------------------------------------------------


def test_the_verdict_tier_stops_after_day_three_even_with_full_data():
    start, end = utc(2026, 10, 31, 0, 0), utc(2026, 11, 5, 0, 0)
    rows = build_rows(calm_forecast(start, end), start, end)
    # NOW is 2026-10-31, day 0. Day 3 is 3 Nov, day 4 is 4 Nov. The clocks are already on GMT.
    assert weather_tier(utc(2026, 11, 3, 23, 0), rows, NOW).tier is Tier.VERDICT
    assert weather_tier(utc(2026, 11, 4, 0, 0), rows, NOW).tier is Tier.OUTLOOK
    assert weather_tier(utc(2026, 11, 4, 12, 0), rows, NOW).tier is Tier.OUTLOOK
    assert weather_tier(utc(2026, 10, 31, 0, 0), rows, NOW).tier is Tier.VERDICT


def test_day_zero_is_the_london_date_near_midnight():
    # 23:30Z on 24 Oct is 00:30 BST on 25 Oct, so for this `now` day 0 is 25 Oct.
    now = utc(2026, 10, 24, 23, 30)
    start, end = utc(2026, 10, 24, 23, 0), utc(2026, 10, 30, 0, 0)
    rows = build_rows(calm_forecast(start, end), start, end)
    assert london_day_offset(utc(2026, 10, 24, 23, 0), now) == 0  # 00:00 BST on the 25th
    # Day 3 is 28 Oct (GMT by then), so 28 Oct 23:00Z is the last verdict hour and 29 Oct 00:00Z is day 4.
    assert weather_tier(utc(2026, 10, 28, 23, 0), rows, now).tier is Tier.VERDICT
    assert weather_tier(utc(2026, 10, 29, 0, 0), rows, now).tier is Tier.OUTLOOK
    # Counted by UTC date instead, 29 Oct 00:00Z would be a day later than it is and 24 Oct 23:00Z a day earlier.
    assert weather_tier(utc(2026, 10, 24, 23, 0), rows, now).tier is Tier.VERDICT


def test_the_day_number_follows_the_london_date_across_the_clock_change(tmp_path, monkeypatch):
    now = utc(2026, 10, 24, 22, 0)  # 23:00 BST on 24 Oct: day 0 is 24 Oct
    late = utc(2026, 10, 24, 23, 30)  # 00:30 BST on 25 Oct
    assert london_day_offset(late, now) == 1
    assert london_day_offset(utc(2026, 10, 24, 22, 59), now) == 0
    start, end = utc(2026, 10, 24, 22, 0), utc(2026, 10, 26, 2, 0)
    rows = build_rows(calm_forecast(start, end), start, end)
    # With the verdict horizon moved to day 0 (a changed rules.yaml), 23:30Z is already past it.
    monkeypatch.setattr(weather, "RULES", rules_with(tmp_path, "verdict_horizon_day", 0))
    assert weather_tier(utc(2026, 10, 24, 22, 30), rows, now).tier is Tier.VERDICT
    assert weather_tier(late, rows, now).tier is Tier.OUTLOOK
    # Hours after the change (GMT) are numbered by the London date too: 26 Oct 01:00Z is day 2.
    assert london_day_offset(utc(2026, 10, 26, 1, 0), now) == 2


def test_every_hour_of_a_run_across_the_clock_change_is_checked_once():
    start, end = utc(2026, 10, 24, 22, 0), utc(2026, 10, 26, 2, 0)
    rows = build_rows(calm_forecast(start, end), start, end)
    tiers = [weather_tier(r.hour, rows, start) for r in rows]
    assert len(tiers) == 28
    assert all(t.tier is Tier.VERDICT and not t.unverified for t in tiers)


@pytest.mark.parametrize("horizon", [0, 1])
def test_the_cap_is_the_rules_file_not_a_number_in_the_code(tmp_path, monkeypatch, horizon):
    start, end = utc(2026, 10, 31, 0, 0), utc(2026, 11, 3, 0, 0)
    rows = build_rows(calm_forecast(start, end), start, end)
    monkeypatch.setattr(weather, "RULES", rules_with(tmp_path, "verdict_horizon_day", horizon))
    verdict_days = {(r.hour.month, r.hour.day) for r in rows if weather_tier(r.hour, rows, NOW).tier is Tier.VERDICT}
    assert verdict_days == ({(10, 31)} if horizon == 0 else {(10, 31), (11, 1)})
