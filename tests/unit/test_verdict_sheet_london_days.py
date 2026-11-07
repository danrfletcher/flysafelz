"""G7 / E14 / E17: the weather-coverage check uses Europe/London days, so it is off by no hours in BST or GMT.

Blue Mill. A forecast holds a calm sample at every UTC hour from `first` to `last`, both included. An hour is
covered when its own sample and the one after it are there. Hours up to London day 3 (the verdict horizon)
need weather; later hours do not. In BST the London day starts at 23:00Z the evening before, so every
boundary here sits one hour earlier in UTC than in GMT.
"""

from datetime import UTC, datetime

import pytest
from factories import weather_forecast

from flysafelz.verdict_sheet import Example, SheetError, check_weather_coverage

CASE = "blue_mill_2026-10-31"
GMT_NOW = datetime(2026, 10, 31, 7, 30, tzinfo=UTC)  # London is on GMT: London days are UTC days
BST_NOW = datetime(2026, 9, 1, 7, 30, tzinfo=UTC)  # London is on BST: London day 1 Sep is 31 Aug 23:00Z to 1 Sep 23:00Z


def utc(month, day, hour=0):
    return datetime(2026, month, day, hour, tzinfo=UTC)


def check(now, first, last, window):
    example = Example(id="ex", path="assess", case=CASE, now=now, window=window)
    check_weather_coverage(example, weather_forecast(first, last), now, window)


def message(now, first, last, window) -> str:
    with pytest.raises(SheetError) as excinfo:
        check(now, first, last, window)
    return str(excinfo.value)


# --- GMT ---------------------------------------------------------------------------------------------------

GMT_DEFAULT = (utc(10, 31, 7), utc(11, 2))  # now's hour to the end of tomorrow (1 Nov) in London


def test_gmt_a_single_day_case_misses_the_default_windows_second_day_and_the_error_says_which():
    text = message(GMT_NOW, utc(10, 31), utc(11, 1), GMT_DEFAULT)
    assert text.startswith("example ex:") and "London day 2026-11-01" in text
    assert "24 hours missing" in text  # 1 Nov 00:00Z to 23:00Z: the 00:00 hour has its own sample but no closing one
    assert "blue_mill_2026-10-31_2day" in text and "days: 2" in text and "\n" not in text


def test_gmt_the_closing_row_matters_one_hour_short_is_still_an_error_naming_the_last_hour():
    text = message(GMT_NOW, utc(10, 31), utc(11, 1, 23), GMT_DEFAULT)
    assert "London day 2026-11-01" in text and "1 hours missing" in text


def test_gmt_covered_to_the_end_of_tomorrow_passes():
    check(GMT_NOW, utc(10, 31), utc(11, 2), GMT_DEFAULT)


def test_gmt_day_3_needs_weather_to_the_end_of_that_day_and_day_4_needs_none():
    window = (utc(10, 31, 7), utc(11, 8))  # to the end of 7 Nov: London days 0 to 7
    check(GMT_NOW, utc(10, 31), utc(11, 4), window)  # the end of day 3 (3 Nov), plus the closing row
    text = message(GMT_NOW, utc(10, 31), utc(11, 3, 23), window)
    assert "London day 2026-11-03" in text and "1 hours missing" in text
    assert "_8day" in text  # the window's last London day is day 7, the outlook day: eight days


# --- BST ---------------------------------------------------------------------------------------------------

BST_DEFAULT = (utc(9, 1, 7), utc(9, 2, 23))  # to the end of 2 Sep in London, which is 23:00Z


def test_bst_the_london_day_ends_at_23_00z_so_the_default_window_needs_weather_to_2_sep_23_00z():
    check(BST_NOW, utc(8, 31, 23), utc(9, 2, 23), BST_DEFAULT)
    text = message(BST_NOW, utc(8, 31, 23), utc(9, 2, 22), BST_DEFAULT)
    assert "London day 2026-09-02" in text and "1 hours missing" in text


def test_bst_a_utc_day_case_runs_out_one_hour_before_the_london_day_does():
    # 1 Sep 00:00Z to 2 Sep 00:00Z is the UTC day. 23:00Z on 1 Sep is already 00:00 on 2 Sep in London,
    # so its London date is 2 Sep, not the 1 Sep a UTC date would say.
    text = message(BST_NOW, utc(9, 1), utc(9, 2), BST_DEFAULT)
    assert "London day 2026-09-02" in text
    assert "blue_mill_2026-10-31_2day" in text


def test_bst_a_utc_day_case_that_stops_at_22_00z_still_misses_the_last_hour_of_london_day_1():
    text = message(BST_NOW, utc(9, 1), utc(9, 1, 22), BST_DEFAULT)
    assert "London day 2026-09-01" in text  # the 22:00Z hour (23:00 BST on 1 Sep) has no closing row


def test_bst_day_3_ends_one_hour_earlier_in_utc_than_in_gmt():
    window = (utc(9, 1, 7), utc(9, 7))
    check(BST_NOW, utc(8, 31, 23), utc(9, 4, 23), window)  # 4 Sep is London day 3: it ends 4 Sep 23:00Z; a GMT-style 5 Sep 00:00Z is not needed
    text = message(BST_NOW, utc(8, 31, 23), utc(9, 4, 22), window)
    assert "London day 2026-09-04" in text and "1 hours missing" in text


def test_bst_a_two_day_london_span_is_covered_exactly_to_its_end():
    first, end = utc(8, 31, 23), utc(9, 2, 23)  # days: 2 from a 1 Sep now
    check(BST_NOW, first, end, BST_DEFAULT)
    assert "1 hours missing" in message(BST_NOW, first, utc(9, 2, 22), BST_DEFAULT)


# --- hours that need no weather -----------------------------------------------------------------------------


def test_hours_past_the_verdict_horizon_need_no_weather():
    window = (utc(10, 31, 7), utc(11, 20))  # 20 Nov is far beyond the outlook day: TOO_FAR
    check(GMT_NOW, utc(10, 31), utc(11, 4), window)


def test_the_suggested_days_stop_at_the_outlook_horizon_and_a_numbered_case_name_is_not_stacked():
    window = (utc(10, 31, 7), utc(11, 20))
    example = Example(id="ex", path="assess", case="blue_mill_2026-10-31_3day", now=GMT_NOW, window=window)
    with pytest.raises(SheetError) as excinfo:
        check_weather_coverage(example, weather_forecast(utc(10, 31), utc(11, 1)), GMT_NOW, window)
    text = str(excinfo.value)
    assert "blue_mill_2026-10-31_8day" in text and "days: 8" in text  # the outlook day is 7: eight London days
    assert "_3day_" not in text
