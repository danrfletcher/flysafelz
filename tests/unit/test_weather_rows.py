"""G4: mapping forecast timestamps onto UTC hour rows, for each kind of value."""

import itertools
import json
from datetime import UTC, datetime

import pytest
from weather_helpers import FIXTURES, NOW, calm, calm_forecast, forecast_of

from flysafelz.clients.open_meteo import ONE_HOUR, parse_response
from flysafelz.weather import build_rows, check_row

H = datetime(2026, 10, 31, 9, 0, tzinfo=UTC)
START, END = H, datetime(2026, 10, 31, 10, 0, tzinfo=UTC)  # exactly one row: 09:00


def one_row(here, closing):
    """The 09:00 row from two samples: HH (09:00) and HH+1 (10:00)."""
    forecast = forecast_of({H: here, H + ONE_HOUR: closing})
    [row] = build_rows(forecast, START, END)
    return row


# --- instantaneous values: the worse of HH and HH+1 ----------------------------------------


@pytest.mark.parametrize("field", ["wind_10m", "wind_80m", "wind_120m", "cape", "cloud_cover_low", "cloud_cover"])
def test_instantaneous_values_take_the_higher_of_the_two_timestamps(field):
    low, high = 2.0, 9.0
    assert getattr(one_row(calm(**{field: low}), calm(**{field: high})).values, field) == high
    assert getattr(one_row(calm(**{field: high}), calm(**{field: low})).values, field) == high


@pytest.mark.parametrize("field", ["visibility", "pressure_msl"])
def test_visibility_and_pressure_take_the_lower_of_the_two_timestamps(field):
    low, high = 2000.0, 9000.0
    assert getattr(one_row(calm(**{field: low}), calm(**{field: high})).values, field) == low
    assert getattr(one_row(calm(**{field: high}), calm(**{field: low})).values, field) == low


# --- preceding-hour values: the HH+1 value only --------------------------------------------


@pytest.mark.parametrize("field", ["gust_10m", "precipitation", "precipitation_probability", "shortwave_radiation"])
def test_preceding_hour_values_use_only_the_closing_timestamp(field):
    assert getattr(one_row(calm(**{field: 90.0}), calm(**{field: 3.0})).values, field) == 3.0
    assert getattr(one_row(calm(**{field: 3.0}), calm(**{field: 90.0})).values, field) == 90.0


def test_direction_reads_the_opening_timestamp():
    assert one_row(calm(direction_10m=90), calm(direction_10m=270)).values.direction_10m == 90


# --- derived values: per timestamp first, then the worse ------------------------------------


def test_gust_spread_is_the_worse_per_timestamp_spread_not_max_gust_minus_max_wind():
    here = calm(wind_10m=14, gust_10m=17)  # spread 3
    closing = calm(wind_10m=6, gust_10m=12)  # spread 6
    row = one_row(here, closing)
    assert row.gust_spread == 6
    assert row.values.gust_10m == 12 and row.values.wind_10m == 14
    # Max gust minus max wind would give a smaller number: 17 - 14 = 3.
    assert max(here.gust_10m, closing.gust_10m) - max(here.wind_10m, closing.wind_10m) == 3
    assert row.gust_spread != 3


def test_gradient_is_the_worse_per_timestamp_gradient():
    here = calm(wind_10m=10, wind_120m=12)  # 2
    closing = calm(wind_10m=5, wind_120m=15)  # 10
    row = one_row(here, closing)
    assert row.gradient == 10
    # |max(120 m) - max(10 m)| would be 5.
    assert abs(max(here.wind_120m, closing.wind_120m) - max(here.wind_10m, closing.wind_10m)) == 5


def test_the_gradient_is_an_absolute_difference():
    assert one_row(calm(wind_10m=15, wind_120m=5), calm(wind_10m=15, wind_120m=5)).gradient == 10


def test_a_spread_of_exactly_the_limit_survives_float_noise():
    # 12.3 - 6.3 is 6.000000000000001 in floating point; the data has one decimal, so it is 6.0.
    row = one_row(calm(wind_10m=6.3, gust_10m=12.3), calm(wind_10m=6.3, gust_10m=12.3))
    assert row.gust_spread == 6.0
    assert [f.rule_id for f in check_row(row)] == ["gust_spread_caution"]  # not NO_GO under "> 6"


# --- missing values: nothing interpolated or filled forward ------------------------------------


def test_a_null_at_the_closing_timestamp_makes_the_instantaneous_value_missing_for_the_row():
    row = one_row(calm(cape=100), calm(cape=None))
    assert row.values.cape is None
    assert row.missing_required == ["cape"]


def test_a_null_at_the_opening_timestamp_also_makes_it_missing():
    row = one_row(calm(wind_80m=None), calm(wind_80m=10))
    assert row.values.wind_80m is None
    assert row.missing_required == ["wind_80m"]


def test_a_missing_closing_timestamp_counts_as_missing_for_every_required_instantaneous_value():
    forecast = forecast_of({H: calm()})  # there is no 10:00
    [row] = build_rows(forecast, START, END)
    assert row.missing_required == [
        "wind_10m",
        "wind_80m",
        "wind_120m",
        "gust_10m",
        "precipitation",
        "cape",
        "visibility",
        "shortwave_radiation",
        "cloud_cover_low",
    ]
    assert row.values.wind_10m is None  # not the 09:00 value carried forward
    assert row.values.gust_10m is None and row.values.shortwave_radiation is None


def test_an_hour_with_no_samples_at_all_is_a_row_with_everything_missing():
    [row] = build_rows(forecast_of({}), START, END)
    assert len(row.missing_required) == 9
    assert row.gust_spread is None and row.gradient is None


def test_a_missing_closing_gust_is_missing_even_when_the_opening_gust_is_there():
    row = one_row(calm(gust_10m=20), calm(gust_10m=None))
    assert row.values.gust_10m is None
    assert row.missing_required == ["gust_10m"]
    assert row.gust_spread == 15  # from the 09:00 timestamp alone; nothing is made up for 10:00


def test_a_missing_optional_variable_adds_a_note_and_keeps_the_row_complete():
    row = one_row(calm(precipitation_probability=None, direction_10m=None, cloud_cover=None), calm(precipitation_probability=None))
    assert row.missing_required == []
    assert row.notes == ["rain chance not available", "total cloud not available", "wind direction not available"]
    assert row.values.precipitation_probability is None


def test_an_all_present_row_has_no_notes_and_nothing_missing():
    row = one_row(calm(), calm())
    assert row.notes == [] and row.missing_required == []


# --- the 23:00 row takes next-day 00:00 ---------------------------------------------------------


def full_day_forecast():
    return parse_response(json.loads((FIXTURES / "full_48h.json").read_text(encoding="utf-8")), NOW)


def test_the_2300_row_takes_the_next_days_0000_values_from_the_response():
    forecast = full_day_forecast()
    day_start, day_end = datetime(2026, 10, 31, 0, 0, tzinfo=UTC), datetime(2026, 11, 1, 0, 0, tzinfo=UTC)
    assert len(build_rows(forecast, day_start, day_end)) == 24
    midnight = datetime(2026, 11, 1, 0, 0, tzinfo=UTC)
    at_2300 = forecast.samples[datetime(2026, 10, 31, 23, 0, tzinfo=UTC)]
    # Make the next-day 00:00 values differ from 23:00 in every way the row maps.
    forecast.samples[midnight] = calm(wind_10m=11.0, gust_10m=13.0, cape=250.0, visibility=3000.0, shortwave_radiation=40.0)
    last = build_rows(forecast, day_start, day_end)[-1]
    assert last.hour == datetime(2026, 10, 31, 23, 0, tzinfo=UTC)
    assert last.values.wind_10m == 11.0 and at_2300.wind_10m < 11.0
    assert last.values.gust_10m == 13.0  # closing timestamp only
    assert last.values.shortwave_radiation == 40.0
    assert last.values.cape == 250.0
    assert last.values.visibility == 3000.0  # the lower of the two
    assert last.missing_required == []


def test_a_response_cut_off_at_2300_leaves_the_2300_row_missing():
    forecast = full_day_forecast()
    del forecast.samples[datetime(2026, 11, 1, 0, 0, tzinfo=UTC)]
    rows = build_rows(forecast, datetime(2026, 10, 31, 22, 0, tzinfo=UTC), datetime(2026, 11, 1, 0, 0, tzinfo=UTC))
    assert rows[0].missing_required == []
    assert "wind_10m" in rows[1].missing_required


# --- windows ----------------------------------------------------------------------------------------


def test_an_empty_window_gives_an_empty_list():
    assert build_rows(calm_forecast(START, END), START, START) == []
    assert build_rows(calm_forecast(START, END), END, START) == []


def test_the_end_is_excluded_and_a_start_inside_an_hour_rounds_down():
    rows = build_rows(calm_forecast(START, END), datetime(2026, 10, 31, 9, 30, tzinfo=UTC), datetime(2026, 10, 31, 12, 0, tzinfo=UTC))
    assert [r.hour.hour for r in rows] == [9, 10, 11]


def test_a_window_across_midnight_and_the_clock_change_has_each_utc_hour_once():
    start = datetime(2026, 10, 24, 22, 0, tzinfo=UTC)
    end = datetime(2026, 10, 26, 2, 0, tzinfo=UTC)
    forecast = calm_forecast(start, end)
    rows = build_rows(forecast, start, end)
    hours = [r.hour for r in rows]
    assert len(hours) == 28  # 22:00 on the 24th to 01:00 on the 26th
    assert len(set(hours)) == len(hours)
    assert all(b - a == ONE_HOUR for a, b in itertools.pairwise(hours))
    assert hours[0] == start and hours[-1] == datetime(2026, 10, 26, 1, 0, tzinfo=UTC)
    assert all(h.utcoffset().total_seconds() == 0 for h in hours)
    assert all(r.missing_required == [] for r in rows)
    # The 25 Oct local day has 25 hours: 01:00 BST and 01:00 GMT are both there, as 00:00Z and 01:00Z.
    assert datetime(2026, 10, 25, 0, 0, tzinfo=UTC) in hours and datetime(2026, 10, 25, 1, 0, tzinfo=UTC) in hours
