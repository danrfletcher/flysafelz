"""daylight.py: phases and minute-exact edges (G6), Europe/London day counting and UTC hours (E13)."""

from datetime import UTC, date, datetime, timedelta
from types import SimpleNamespace

import pytest
from astral import Observer
from astral.sun import dawn, dusk

from flysafelz import daylight
from flysafelz.daylight import (
    Phase,
    daylight_edges,
    daylight_phase,
    london_date,
    london_day_date,
    london_day_offset,
    london_day_utc,
    london_days_utc,
    round_down_minute,
    round_up_minute,
    sunrise_sunset,
    utc_hours,
)
from flysafelz.rules import RULES

BLUE_MILL = (53.802797, -0.086644)
LERWICK = (60.15, -1.15)  # dusk exists on 21 June
NORTH_SHETLAND = (60.8, -1.15)  # the sun never gets 6 degrees below the horizon on 21 June, still inside the UK bounds


def utc(*args) -> datetime:
    return datetime(*args, tzinfo=UTC)


# --- Blue Mill, 31 October 2026 ----------------------------------------------

BLUE_MILL_EDGES = [
    (utc(2026, 10, 31, 6, 21), Phase.TWILIGHT),  # civil dawn, astral 06:20:40, rounded up
    (utc(2026, 10, 31, 6, 59), Phase.DAY),  # sunrise, astral 06:58:34, rounded up
    (utc(2026, 10, 31, 16, 28), Phase.TWILIGHT),  # sunset, astral 16:28:27, rounded down
    (utc(2026, 10, 31, 17, 6), Phase.NIGHT),  # civil dusk, astral 17:06:19, rounded down (GP6)
]


def test_blue_mill_on_31_october_has_all_four_edges_to_the_minute():
    edges = daylight_edges(*BLUE_MILL, utc(2026, 10, 31, 0, 0), utc(2026, 11, 1, 0, 0))
    assert edges == BLUE_MILL_EDGES
    assert all(t.second == 0 and t.microsecond == 0 for t, _ in edges)


def test_the_gp6_row_splits_at_1706():
    edges = daylight_edges(*BLUE_MILL, utc(2026, 10, 31, 17, 0), utc(2026, 10, 31, 18, 0))
    assert edges == [(utc(2026, 10, 31, 17, 6), Phase.NIGHT)]
    assert daylight_phase(*BLUE_MILL, utc(2026, 10, 31, 17, 0)) is Phase.TWILIGHT


@pytest.mark.parametrize(
    ("when", "phase"),
    [
        ((2026, 10, 31, 0, 0, 0), Phase.NIGHT),
        ((2026, 10, 31, 6, 20, 59), Phase.NIGHT),
        ((2026, 10, 31, 6, 21, 0), Phase.TWILIGHT),
        ((2026, 10, 31, 6, 58, 59), Phase.TWILIGHT),
        ((2026, 10, 31, 6, 59, 0), Phase.DAY),
        ((2026, 10, 31, 12, 0, 0), Phase.DAY),
        ((2026, 10, 31, 16, 27, 59), Phase.DAY),
        ((2026, 10, 31, 16, 28, 0), Phase.TWILIGHT),
        ((2026, 10, 31, 17, 5, 59), Phase.TWILIGHT),
        ((2026, 10, 31, 17, 6, 0), Phase.NIGHT),
        ((2026, 10, 31, 23, 59, 0), Phase.NIGHT),
    ],
)
def test_phase_changes_exactly_on_the_edge_minute(when, phase):
    assert daylight_phase(*BLUE_MILL, utc(*when)) is phase


def test_phase_accepts_any_timezone_and_rejects_naive_datetimes():
    from zoneinfo import ZoneInfo

    assert daylight_phase(*BLUE_MILL, datetime(2026, 10, 31, 17, 6, tzinfo=ZoneInfo("Europe/London"))) is Phase.NIGHT  # GMT
    assert daylight_phase(*BLUE_MILL, datetime(2026, 10, 31, 18, 6, tzinfo=ZoneInfo("Europe/Paris"))) is Phase.NIGHT  # 17:06Z
    with pytest.raises(ValueError, match="timezone-aware"):
        daylight_phase(*BLUE_MILL, datetime(2026, 10, 31, 12, 0))
    with pytest.raises(ValueError, match="timezone-aware"):
        daylight_edges(*BLUE_MILL, datetime(2026, 10, 31), utc(2026, 11, 1))


def test_an_edge_at_the_start_or_end_of_the_window_is_not_a_change_inside_it():
    dusk_minute = utc(2026, 10, 31, 17, 6)
    assert daylight_edges(*BLUE_MILL, dusk_minute, utc(2026, 10, 31, 18, 0)) == []  # already NIGHT at the start
    assert daylight_edges(*BLUE_MILL, utc(2026, 10, 31, 17, 0), dusk_minute) == []  # end is exclusive
    assert daylight_edges(*BLUE_MILL, utc(2026, 10, 31, 17, 0), dusk_minute + timedelta(minutes=1)) == [(dusk_minute, Phase.NIGHT)]


# --- rounding direction (G6) -------------------------------------------------


def test_edges_round_so_that_the_caution_zone_grows():
    """Dawn and sunrise up, sunset and dusk down. Compare each rounded edge with astral's exact time."""
    observer = Observer(*BLUE_MILL)
    day = date(2026, 10, 31)
    raw_dawn = dawn(observer, day, depression=6, tzinfo=UTC)
    raw_dusk = dusk(observer, day, depression=6, tzinfo=UTC)
    raw_sunrise, raw_sunset = sunrise_sunset(*BLUE_MILL, day)
    assert (raw_dawn.second, raw_dusk.second) == (40, 19)  # astral: 06:20:40 and 17:06:19
    dawn_edge, sunrise_edge = BLUE_MILL_EDGES[0][0], BLUE_MILL_EDGES[1][0]
    sunset_edge, dusk_edge = BLUE_MILL_EDGES[2][0], BLUE_MILL_EDGES[3][0]
    assert dawn_edge > raw_dawn and dawn_edge - raw_dawn < timedelta(minutes=1)
    assert sunrise_edge > raw_sunrise and sunrise_edge - raw_sunrise < timedelta(minutes=1)
    assert sunset_edge < raw_sunset and raw_sunset - sunset_edge < timedelta(minutes=1)
    assert dusk_edge < raw_dusk and raw_dusk - dusk_edge < timedelta(minutes=1)
    assert dusk_edge == utc(2026, 10, 31, 17, 6)  # GP6 says 17:06, not 17:07


def test_rounding_helpers_keep_a_whole_minute_and_move_a_partial_one_the_right_way():
    exact = utc(2026, 10, 31, 6, 21)
    assert round_up_minute(exact) == exact == round_down_minute(exact)
    partial = datetime(2026, 10, 31, 6, 20, 40, 1, tzinfo=UTC)
    assert round_up_minute(partial) == utc(2026, 10, 31, 6, 21)
    assert round_down_minute(partial) == utc(2026, 10, 31, 6, 20)
    just_over = datetime(2026, 10, 31, 6, 20, 0, 1, tzinfo=UTC)
    assert round_up_minute(just_over) == utc(2026, 10, 31, 6, 21)


def test_the_civil_twilight_angle_comes_from_rules_yaml(monkeypatch):
    """With a 12 degree angle in the rules the dusk moves later. The 6 is not in the Python."""
    deeper = RULES.civil_twilight_depression.model_copy(update={"value": 12})
    fake = SimpleNamespace(**{**{name: getattr(RULES, name) for name in type(RULES).model_fields}, "civil_twilight_depression": deeper})
    monkeypatch.setattr(daylight, "RULES", fake)
    edges = daylight_edges(*BLUE_MILL, utc(2026, 10, 31, 0, 0), utc(2026, 11, 1, 0, 0))
    night_start = [t for t, phase in edges if phase is Phase.NIGHT]
    assert night_start and night_start[0] > utc(2026, 10, 31, 17, 40)


# --- windows across UTC midnight (G6) -----------------------------------------


def test_a_window_crossing_midnight_uses_both_dates_and_keeps_an_early_dawn():
    edges = daylight_edges(*LERWICK, utc(2026, 6, 20, 22, 0), utc(2026, 6, 21, 3, 0))
    assert edges == [
        (utc(2026, 6, 20, 23, 28), Phase.NIGHT),  # dusk of the 20th, astral 23:28:xx
        (utc(2026, 6, 21, 0, 45), Phase.TWILIGHT),  # dawn of the 21st, astral 00:44:07, early and just after midnight
        (utc(2026, 6, 21, 2, 40), Phase.DAY),  # sunrise, astral 02:39:13
    ]


def test_a_window_crossing_midnight_at_blue_mill_finds_next_mornings_edges():
    edges = daylight_edges(*BLUE_MILL, utc(2026, 10, 31, 23, 0), utc(2026, 11, 1, 8, 0))
    assert edges == [(utc(2026, 11, 1, 6, 23), Phase.TWILIGHT), (utc(2026, 11, 1, 7, 1), Phase.DAY)]


def test_a_window_of_several_days_lists_every_edge_in_order():
    edges = daylight_edges(*BLUE_MILL, utc(2026, 10, 30, 0, 0), utc(2026, 11, 2, 0, 0))
    assert len(edges) == 8 + 4
    assert [t for t, _ in edges] == sorted(t for t, _ in edges)
    phases = [p for _, p in edges]
    assert phases == [Phase.TWILIGHT, Phase.DAY, Phase.TWILIGHT, Phase.NIGHT] * 3


# --- high latitude in summer --------------------------------------------------


def test_lerwick_in_midsummer_has_a_civil_dusk_and_a_short_night():
    assert daylight_phase(*LERWICK, utc(2026, 6, 21, 12, 0)) is Phase.DAY
    assert daylight_phase(*LERWICK, utc(2026, 6, 21, 22, 0)) is Phase.TWILIGHT  # after sunset 21:33
    assert daylight_phase(*LERWICK, utc(2026, 6, 21, 23, 30)) is Phase.NIGHT  # after dusk 23:28
    assert daylight_phase(*LERWICK, utc(2026, 6, 22, 0, 30)) is Phase.NIGHT  # before dawn 00:4x


def test_at_60_8_north_there_is_no_dusk_so_the_night_is_twilight_and_nothing_crashes():
    # astral itself raises here.
    with pytest.raises(ValueError):
        dusk(Observer(*NORTH_SHETLAND), date(2026, 6, 21), depression=6, tzinfo=UTC)
    phases = {daylight_phase(*NORTH_SHETLAND, utc(2026, 6, 21, hour, 0)) for hour in range(24)}
    assert phases == {Phase.DAY, Phase.TWILIGHT}
    assert Phase.NIGHT not in phases
    assert daylight_phase(*NORTH_SHETLAND, utc(2026, 6, 21, 23, 30)) is Phase.TWILIGHT
    assert daylight_phase(*NORTH_SHETLAND, utc(2026, 6, 22, 1, 0)) is Phase.TWILIGHT


def test_at_60_8_north_the_edges_are_only_sunset_and_sunrise():
    edges = daylight_edges(*NORTH_SHETLAND, utc(2026, 6, 20, 12, 0), utc(2026, 6, 22, 12, 0))
    assert [p for _, p in edges] == [Phase.TWILIGHT, Phase.DAY, Phase.TWILIGHT, Phase.DAY]


# --- E13: London days and UTC hours -------------------------------------------


@pytest.mark.parametrize(
    ("t", "now", "offset"),
    [
        (utc(2026, 10, 24, 22, 30), utc(2026, 10, 24, 22, 30), 0),  # 24 Oct in London (BST): day 0
        (utc(2026, 10, 24, 23, 0), utc(2026, 10, 24, 22, 30), 1),  # 25 Oct 00:00 BST: day 1
        (utc(2026, 10, 25, 23, 30), utc(2026, 10, 24, 22, 30), 1),  # still 25 Oct (GMT after the change)
        (utc(2026, 10, 26, 0, 0), utc(2026, 10, 24, 22, 30), 2),  # 26 Oct 00:00 GMT
        (utc(2026, 10, 24, 22, 59), utc(2026, 10, 24, 22, 30), 0),
        (utc(2026, 10, 24, 12, 0), utc(2026, 10, 24, 22, 30), 0),
        (utc(2026, 10, 23, 12, 0), utc(2026, 10, 24, 22, 30), -1),  # the past is negative
        (utc(2026, 3, 28, 23, 30), utc(2026, 3, 28, 12, 0), 0),  # before the spring change, still 28 March
        (utc(2026, 3, 29, 0, 0), utc(2026, 3, 28, 12, 0), 1),  # 29 March 00:00 GMT
        (utc(2026, 3, 29, 23, 0), utc(2026, 3, 28, 12, 0), 2),  # 30 March 00:00 BST
        (utc(2026, 3, 29, 22, 59), utc(2026, 3, 28, 12, 0), 1),
    ],
)
def test_london_day_offset_across_midnight_and_both_clock_changes(t, now, offset):
    assert london_day_offset(t, now) == offset


def test_london_day_offset_uses_the_london_date_not_the_utc_date():
    # 00:30 BST on 1 July is 23:30 UTC on 30 June. It is day 1 for a user at 12:00 on 30 June.
    assert london_day_offset(utc(2026, 6, 30, 23, 30), utc(2026, 6, 30, 12, 0)) == 1
    assert london_day_offset(utc(2026, 6, 30, 22, 59), utc(2026, 6, 30, 12, 0)) == 0


def test_london_day_offset_rejects_naive_datetimes():
    with pytest.raises(ValueError):
        london_day_offset(datetime(2026, 10, 24, 12), utc(2026, 10, 24, 12))


@pytest.mark.parametrize(
    ("day", "hours", "start", "end"),
    [
        (date(2026, 10, 25), 25, utc(2026, 10, 24, 23, 0), utc(2026, 10, 26, 0, 0)),  # clocks go back
        (date(2026, 3, 29), 23, utc(2026, 3, 29, 0, 0), utc(2026, 3, 29, 23, 0)),  # clocks go forward
        (date(2026, 10, 24), 24, utc(2026, 10, 23, 23, 0), utc(2026, 10, 24, 23, 0)),
        (date(2026, 1, 15), 24, utc(2026, 1, 15, 0, 0), utc(2026, 1, 16, 0, 0)),
    ],
)
def test_the_london_day_has_the_right_number_of_utc_hours(day, hours, start, end):
    assert london_day_utc(day) == (start, end)
    listed = utc_hours(*london_day_utc(day))
    assert len(listed) == hours
    assert listed[0] == start
    assert listed[-1] == end - timedelta(hours=1)


def test_utc_hours_excludes_the_end_and_starts_at_the_hour_containing_start():
    assert utc_hours(utc(2026, 10, 31, 7, 0), utc(2026, 10, 31, 10, 0)) == [utc(2026, 10, 31, h, 0) for h in (7, 8, 9)]
    assert utc_hours(utc(2026, 10, 31, 7, 30), utc(2026, 10, 31, 9, 0)) == [utc(2026, 10, 31, 7, 0), utc(2026, 10, 31, 8, 0)]
    assert utc_hours(utc(2026, 10, 31, 7, 30), utc(2026, 10, 31, 9, 1)) == [utc(2026, 10, 31, h, 0) for h in (7, 8, 9)]
    assert utc_hours(utc(2026, 10, 31, 7, 0), utc(2026, 10, 31, 7, 0)) == []
    assert utc_hours(utc(2026, 10, 31, 8, 0), utc(2026, 10, 31, 7, 0)) == []


def test_utc_hours_crosses_midnight_in_utc():
    got = utc_hours(utc(2026, 10, 31, 22, 0), utc(2026, 11, 1, 2, 0))
    assert got == [utc(2026, 10, 31, 22, 0), utc(2026, 10, 31, 23, 0), utc(2026, 11, 1, 0, 0), utc(2026, 11, 1, 1, 0)]


def test_utc_hours_works_for_aware_non_utc_input_and_returns_utc():
    from zoneinfo import ZoneInfo

    start = datetime(2026, 10, 31, 8, 0, tzinfo=ZoneInfo("Europe/Paris"))  # 07:00Z
    got = utc_hours(start, utc(2026, 10, 31, 9, 0))
    assert got == [utc(2026, 10, 31, 7, 0), utc(2026, 10, 31, 8, 0)]
    assert all(h.tzinfo is UTC for h in got)


# --- London day N: the span a `days` capture covers (G11, E17) -------------------------------------------------


@pytest.mark.parametrize(
    ("t", "expected"),
    [
        (utc(2026, 10, 24, 22, 59), date(2026, 10, 24)),  # BST: still 24 Oct
        (utc(2026, 10, 24, 23, 0), date(2026, 10, 25)),  # BST midnight is 23:00 UTC
        (utc(2026, 10, 25, 23, 30), date(2026, 10, 25)),  # GMT after the change: 25 Oct until 00:00 UTC
        (utc(2026, 10, 26, 0, 0), date(2026, 10, 26)),
        (utc(2026, 3, 28, 23, 30), date(2026, 3, 28)),  # before the spring change, GMT
        (utc(2026, 3, 29, 0, 0), date(2026, 3, 29)),
        (utc(2026, 3, 29, 23, 0), date(2026, 3, 30)),  # BST from 01:00 UTC on 29 Mar; midnight is 23:00 UTC
        (utc(2026, 1, 15, 12, 0), date(2026, 1, 15)),
    ],
)
def test_london_date_is_the_london_calendar_date(t, expected):
    assert london_date(t) == expected


@pytest.mark.parametrize(
    ("now", "day", "expected"),
    [
        (utc(2026, 10, 24, 7, 30), 0, date(2026, 10, 24)),
        (utc(2026, 10, 24, 7, 30), 1, date(2026, 10, 25)),  # the 25-hour day
        (utc(2026, 10, 24, 22, 45), 0, date(2026, 10, 24)),
        (utc(2026, 10, 24, 23, 15), 0, date(2026, 10, 25)),  # already 25 Oct in London
        (utc(2026, 3, 28, 12, 0), 2, date(2026, 3, 30)),  # across the spring change
        (utc(2026, 10, 31, 7, 30), 7, date(2026, 11, 7)),
    ],
)
def test_london_day_date_counts_london_days_from_now(now, day, expected):
    assert london_day_date(now, day) == expected
    assert london_day_offset(london_day_utc(expected)[0], now) == day  # the same day the offset helper calls `day`


@pytest.mark.parametrize(
    ("now", "days", "start", "end", "hours"),
    [
        (utc(2026, 10, 31, 7, 30), 1, utc(2026, 10, 31, 0), utc(2026, 11, 1, 0), 24),  # GMT
        (utc(2026, 10, 31, 7, 30), 5, utc(2026, 10, 31, 0), utc(2026, 11, 5, 0), 120),
        (utc(2026, 9, 1, 7, 30), 1, utc(2026, 8, 31, 23), utc(2026, 9, 1, 23), 24),  # BST
        (utc(2026, 9, 1, 7, 30), 5, utc(2026, 8, 31, 23), utc(2026, 9, 5, 23), 120),
        (utc(2026, 10, 24, 7, 30), 3, utc(2026, 10, 23, 23), utc(2026, 10, 27, 0), 73),  # BST into the 25-hour day
        (utc(2026, 10, 25, 7, 30), 1, utc(2026, 10, 24, 23), utc(2026, 10, 26, 0), 25),  # day 0 is the 25-hour day
        (utc(2026, 3, 29, 7, 30), 1, utc(2026, 3, 29, 0), utc(2026, 3, 29, 23), 23),  # day 0 is the 23-hour day
        (utc(2026, 3, 28, 12, 0), 3, utc(2026, 3, 28, 0), utc(2026, 3, 30, 23), 71),
        (utc(2026, 6, 30, 23, 30), 2, utc(2026, 6, 30, 23), utc(2026, 7, 2, 23), 48),  # 23:30 UTC is already 1 July in London
    ],
)
def test_london_days_utc_runs_from_the_start_of_day_0_to_the_end_of_day_n_minus_1(now, days, start, end, hours):
    assert london_days_utc(now, days) == (start, end)
    assert (end - start) == timedelta(hours=hours)


def test_london_days_utc_rejects_fewer_than_one_day_and_naive_times():
    with pytest.raises(ValueError):
        london_days_utc(utc(2026, 10, 31, 7, 30), 0)
    with pytest.raises(ValueError):
        london_days_utc(datetime(2026, 10, 31, 7, 30), 1)
