"""schedule.py: the D-item parser (E8). Table-driven, exact expected intervals. All times are UTC."""

from datetime import UTC, date, datetime, timedelta

import pytest

from flysafelz.daylight import sunrise_sunset
from flysafelz.schedule import Schedule, parse_schedule

BLUE_MILL = (53.802797, -0.086644)
# 2026-10-31 is a Saturday, so 1 Nov is a Sunday and 2 Nov a Monday.
W1 = ("2026-10-31 00:00", "2026-11-03 00:00")


def at(text: str) -> datetime:
    return datetime.fromisoformat(text).replace(tzinfo=UTC)


def iv(*pairs: tuple[str, str]) -> list[tuple[datetime, datetime]]:
    return [(at(a), at(b)) for a, b in pairs]


def run(text, window=W1, lat=BLUE_MILL[0], lon=BLUE_MILL[1]) -> Schedule:
    return parse_schedule(text, at(window[0]), at(window[1]), lat, lon)


# (id, D text, (B, C), expected intervals)
FORMATS = [
    ("hhmm-hhmm", "0800-1700", W1, [("2026-10-31 08:00", "2026-10-31 17:00"), ("2026-11-01 08:00", "2026-11-01 17:00"), ("2026-11-02 08:00", "2026-11-02 17:00")]),
    (
        "date-list-with-months",
        "OCT 26 27 28 NOV 02 04 09 10 11 12 18 0900-2030",
        ("2026-10-26 00:00", "2026-11-05 00:00"),
        [(f"2026-10-{d} 09:00", f"2026-10-{d} 20:30") for d in (26, 27, 28)] + [("2026-11-02 09:00", "2026-11-02 20:30"), ("2026-11-04 09:00", "2026-11-04 20:30")],
    ),
    (
        "cross-month-range",
        "SEP 29-OCT 05 0800-1700",
        ("2026-10-03 00:00", "2026-10-07 00:00"),
        [(f"2026-10-0{d} 08:00", f"2026-10-0{d} 17:00") for d in (3, 4, 5)],
    ),
    (
        "bare-day-list",
        "27 0900-1600, 28 0800-1000",
        ("2026-10-26 00:00", "2026-10-30 00:00"),
        [("2026-10-27 09:00", "2026-10-27 16:00"), ("2026-10-28 08:00", "2026-10-28 10:00")],
    ),
    ("daily-alone-is-all-day", "DAILY", ("2026-10-31 06:00", "2026-11-01 18:00"), [("2026-10-31 06:00", "2026-11-01 18:00")]),
    ("daily-with-times", "DAILY 0900-1000", W1, [(f"2026-{d} 09:00", f"2026-{d} 10:00") for d in ("10-31", "11-01", "11-02")]),
    (
        "mon-fri",
        "MON-FRI 0800-1600",
        ("2026-10-30 00:00", "2026-11-03 00:00"),
        [("2026-10-30 08:00", "2026-10-30 16:00"), ("2026-11-02 08:00", "2026-11-02 16:00")],
    ),
    (
        "day-sets",
        "TUE WED THU SAT 0800-0830 1530-1600, FRI SUN 0800-1600",
        ("2026-10-27 00:00", "2026-11-02 00:00"),
        [
            ("2026-10-27 08:00", "2026-10-27 08:30"), ("2026-10-27 15:30", "2026-10-27 16:00"),
            ("2026-10-28 08:00", "2026-10-28 08:30"), ("2026-10-28 15:30", "2026-10-28 16:00"),
            ("2026-10-29 08:00", "2026-10-29 08:30"), ("2026-10-29 15:30", "2026-10-29 16:00"),
            ("2026-10-30 08:00", "2026-10-30 16:00"),
            ("2026-10-31 08:00", "2026-10-31 08:30"), ("2026-10-31 15:30", "2026-10-31 16:00"),
            ("2026-11-01 08:00", "2026-11-01 16:00"),
        ],
    ),
    ("weekday-range-wraps-the-week", "FRI-MON 0900-1000", W1, [(f"2026-{d} 09:00", f"2026-{d} 10:00") for d in ("10-31", "11-01", "11-02")]),
    ("h24", "H24", W1, [W1]),
    ("sun-h24", "SUN H24", W1, [("2026-11-01 00:00", "2026-11-02 00:00")]),
    ("sr-ss", "SR-SS", ("2026-10-31 00:00", "2026-11-01 00:00"), [("2026-10-31 06:58", "2026-10-31 16:29")]),
    ("sr-30-to-ss+30", "SR-30 TO SS+30", ("2026-10-31 00:00", "2026-11-01 00:00"), [("2026-10-31 06:28", "2026-10-31 16:59")]),
    ("sr-plus-offset-and-clock-time", "SR+15-1500", ("2026-10-31 00:00", "2026-11-01 00:00"), [("2026-10-31 07:13", "2026-10-31 15:00")]),
    # Shapes the plan's list misses (found in the 31 Oct PIB).
    ("overnight-wrap", "2300-0600", ("2026-10-31 12:00", "2026-11-01 12:00"), [("2026-10-31 23:00", "2026-11-01 06:00")]),
    ("overnight-wrap-from-the-night-before", "2300-0600", ("2026-10-31 03:00", "2026-10-31 12:00"), [("2026-10-31 03:00", "2026-10-31 06:00")]),
    (
        "sun-thu-overnight-belongs-to-its-start-day",
        "SUN-THU 2300-0600",
        ("2026-10-31 00:00", "2026-11-06 00:00"),
        [
            ("2026-11-01 23:00", "2026-11-02 06:00"),
            ("2026-11-02 23:00", "2026-11-03 06:00"),
            ("2026-11-03 23:00", "2026-11-04 06:00"),
            ("2026-11-04 23:00", "2026-11-05 06:00"),
            ("2026-11-05 23:00", "2026-11-06 00:00"),
        ],
    ),
    (
        "bare-day-wrap-ends-next-day",
        "02 1200-0900, 03 1500-0900",
        ("2026-11-01 00:00", "2026-11-05 00:00"),
        [("2026-11-02 12:00", "2026-11-03 09:00"), ("2026-11-03 15:00", "2026-11-04 09:00")],
    ),
    ("exc-sun", "0800-1700 EXC SUN", W1, [("2026-10-31 08:00", "2026-10-31 17:00"), ("2026-11-02 08:00", "2026-11-02 17:00")]),
    (
        "exc-sat-sun",
        "0800-1800 EXC SAT SUN",
        ("2026-10-30 00:00", "2026-11-03 00:00"),
        [("2026-10-30 08:00", "2026-10-30 18:00"), ("2026-11-02 08:00", "2026-11-02 18:00")],
    ),
    (
        "two-time-pairs-in-one-group",
        "0730-0900 1700-1900",
        ("2026-10-31 00:00", "2026-11-02 00:00"),
        [("2026-10-31 07:30", "2026-10-31 09:00"), ("2026-10-31 17:00", "2026-10-31 19:00"), ("2026-11-01 07:30", "2026-11-01 09:00"), ("2026-11-01 17:00", "2026-11-01 19:00")],
    ),
    (
        "month-carried-over-to-bare-days",
        "NOV 02 1700-1900, 03 04 05 0730-0900",
        ("2026-11-01 00:00", "2026-11-07 00:00"),
        [("2026-11-02 17:00", "2026-11-02 19:00")] + [(f"2026-11-0{d} 07:30", f"2026-11-0{d} 09:00") for d in (3, 4, 5)],
    ),
    (
        "range-into-the-next-year",
        "OCT 25-JAN 01 0800-1630",
        ("2026-12-30 00:00", "2027-01-03 00:00"),
        [("2026-12-30 08:00", "2026-12-30 16:30"), ("2026-12-31 08:00", "2026-12-31 16:30"), ("2027-01-01 08:00", "2027-01-01 16:30")],
    ),
    ("odd-spacing-after-dash", "OCT 26- DEC 08 0800-1800", ("2026-11-06 00:00", "2026-11-08 00:00"), [("2026-11-06 08:00", "2026-11-06 18:00"), ("2026-11-07 08:00", "2026-11-07 18:00")]),
    ("odd-spacing-around-dash", "OCT 25 - NOV 04 0600-2300", ("2026-11-03 00:00", "2026-11-04 12:00"), [("2026-11-03 06:00", "2026-11-03 23:00"), ("2026-11-04 06:00", "2026-11-04 12:00")]),
    ("space-before-trailing-comma", "0700-2000 ,", ("2026-10-31 00:00", "2026-11-01 00:00"), [("2026-10-31 07:00", "2026-10-31 20:00")]),
    ("space-inside-the-time-pair", "31 1400 -1700", ("2026-10-30 00:00", "2026-11-01 00:00"), [("2026-10-31 14:00", "2026-10-31 17:00")]),
    (
        "embedded-newlines",
        "24 0900-1300, 25 0830-1600, 26-29 0900-1600, 30 0900-2200, 31 \n1400 -1700, 01 0930-1630",
        ("2026-10-30 00:00", "2026-11-02 00:00"),
        [("2026-10-30 09:00", "2026-10-30 22:00"), ("2026-10-31 14:00", "2026-10-31 17:00"), ("2026-11-01 09:30", "2026-11-01 16:30")],
    ),
    (
        "newline-between-month-and-days",
        "OCT 30 0730-1700, NOV 02 1700-1900, NOV 03 04 05 0730-0900 \n1700-1900",
        ("2026-11-02 00:00", "2026-11-04 00:00"),
        [("2026-11-02 17:00", "2026-11-02 19:00"), ("2026-11-03 07:30", "2026-11-03 09:00"), ("2026-11-03 17:00", "2026-11-03 19:00")],
    ),
    (
        "ends-at-2359",
        "0000-2359",
        ("2026-10-31 00:00", "2026-11-02 00:00"),
        [("2026-10-31 00:00", "2026-10-31 23:59"), ("2026-11-01 00:00", "2026-11-01 23:59")],
    ),
    (
        "ends-at-0059-next-day",
        "SEP 24-OCT 24 0600-2359, OCT 25-31 0700-0059",
        ("2026-10-24 00:00", "2026-10-27 00:00"),
        [("2026-10-24 06:00", "2026-10-24 23:59"), ("2026-10-25 07:00", "2026-10-26 00:59"), ("2026-10-26 07:00", "2026-10-27 00:00")],
    ),
    ("ends-at-2400", "0800-2400", ("2026-10-31 00:00", "2026-11-02 12:00"), [("2026-10-31 08:00", "2026-11-01 00:00"), ("2026-11-01 08:00", "2026-11-02 00:00"), ("2026-11-02 08:00", "2026-11-02 12:00")]),
    (
        "month-end-run-over-two-months",
        "26 27 28 29 1700-0900, 30 1600-2359, 31 01 0000-2359, 02 0000-0900",
        ("2026-10-30 00:00", "2026-11-03 00:00"),
        [
            ("2026-10-30 00:00", "2026-10-30 09:00"),
            ("2026-10-30 16:00", "2026-10-30 23:59"),
            ("2026-10-31 00:00", "2026-10-31 23:59"),
            ("2026-11-01 00:00", "2026-11-01 23:59"),
            ("2026-11-02 00:00", "2026-11-02 09:00"),
        ],
    ),
    (
        "times-only-clause-joins-the-clause-before",
        "04 08 0800-0830, 1630-1700",
        ("2026-11-03 00:00", "2026-11-10 00:00"),
        [("2026-11-04 08:00", "2026-11-04 08:30"), ("2026-11-04 16:30", "2026-11-04 17:00"), ("2026-11-08 08:00", "2026-11-08 08:30"), ("2026-11-08 16:30", "2026-11-08 17:00")],
    ),
    (
        "date-spans-with-different-times",
        "OCT 01-24 0500-1800, OCT 25 - NOV 30 0600-1900",
        ("2026-10-24 00:00", "2026-10-26 00:00"),
        [("2026-10-24 05:00", "2026-10-24 18:00"), ("2026-10-25 06:00", "2026-10-25 19:00")],
    ),
]


@pytest.mark.parametrize(("name", "text", "window", "expected"), FORMATS, ids=[f[0] for f in FORMATS])
def test_format_gives_exact_intervals(name, text, window, expected):
    schedule = run(text, window)
    assert schedule.parsed is True
    assert schedule.note is None
    assert schedule.intervals == iv(*expected)


# --- SR and SS ---------------------------------------------------------------


def test_sr_ss_come_from_astral_and_round_outward_so_the_notam_is_never_shorter():
    for day in (date(2026, 10, 30), date(2026, 10, 31), date(2026, 11, 2)):
        raw_rise, raw_set = sunrise_sunset(*BLUE_MILL, day)
        schedule = parse_schedule("SR-SS", datetime.combine(day, datetime.min.time(), UTC), datetime.combine(day + timedelta(days=1), datetime.min.time(), UTC), *BLUE_MILL)
        ((start, end),) = schedule.intervals
        assert start <= raw_rise and raw_rise - start < timedelta(minutes=1)  # SR rounded down
        assert end >= raw_set and end - raw_set < timedelta(minutes=1)  # SS rounded up
        assert start.second == end.second == 0


def test_sr_ss_with_offsets_round_outward_too():
    day = date(2026, 10, 31)
    raw_rise, raw_set = sunrise_sunset(*BLUE_MILL, day)
    ((start, end),) = run("SR-30 TO SS+30", ("2026-10-31 00:00", "2026-11-01 00:00")).intervals
    assert start <= raw_rise - timedelta(minutes=30) and (raw_rise - timedelta(minutes=30)) - start < timedelta(minutes=1)
    assert end >= raw_set + timedelta(minutes=30) and end - (raw_set + timedelta(minutes=30)) < timedelta(minutes=1)


def test_sr_ss_uses_each_days_own_times_and_the_lz_position():
    two_days = run("SR-SS", ("2026-10-31 00:00", "2026-11-02 00:00")).intervals
    assert len(two_days) == 2
    assert two_days[0][0].time() != two_days[1][0].time()  # the next day's sunrise is a different minute
    west = parse_schedule("SR-SS", at("2026-10-31 00:00"), at("2026-11-01 00:00"), 54.0, -6.0)  # ~24 minutes later than at Blue Mill
    east = run("SR-SS", ("2026-10-31 00:00", "2026-11-01 00:00"))
    assert west.intervals[0][0] > east.intervals[0][0] + timedelta(minutes=20)


def test_weekday_with_sr_ss():
    schedule = run("MON-FRI SR-SS", ("2026-10-30 00:00", "2026-11-03 00:00"))
    assert [s.date() for s, _ in schedule.intervals] == [date(2026, 10, 30), date(2026, 11, 2)]


def test_lower_case_text_reads_the_same_as_upper_case():
    window = ("2026-10-30 00:00", "2026-11-03 00:00")
    assert run("mon-fri sr-ss", window) == run("MON-FRI SR-SS", window)
    assert run("oct 30 0800-1700", window) == run("OCT 30 0800-1700", window)
    assert run("oct 30 0800-1700", window).intervals == iv(("2026-10-30 08:00", "2026-10-30 17:00"))


def test_a_sunrise_that_astral_cannot_compute_fails_closed():
    """The sun never sets at 89.5 N in midsummer: astral raises, so the schedule is unparsed and covers B-C."""
    window = (at("2026-06-21 00:00"), at("2026-06-22 00:00"))
    schedule = parse_schedule("SR-SS", *window, 89.5, 0.0)
    assert schedule.parsed is False
    assert schedule.intervals == [window]
    assert "SR-SS" in (schedule.note or "")
    assert parse_schedule("0800-1700", *window, 89.5, 0.0).parsed is True  # clock times need no sun


# --- fail closed, empty text, clipping, merging, PERM ------------------------

UNPARSEABLE = [
    "TEMPO",
    "0800-2500",
    "2460-2500",
    "08:00-17:00",
    "0800",
    "FEB 30 0800-1700",
    "MON- 0800-1700",
    "MON-FRI 0800-1700 BY APPOINTMENT",
    "OCT 25-OCT 05 0800-1700",
    "29-OCT 05 0800-1700",  # no month before the 29th: never-active would fail open
    "EXC",
    "0800-1700 EXC",
    "0800-1700 EXC 31",
    "AUG 0800-1700",
    "32 0800-1700",
    "0800-1700 + 0900",
    "SR-SS+",
    "1200-0900-1000",
]


@pytest.mark.parametrize("text", UNPARSEABLE)
def test_unparseable_text_is_active_for_all_of_b_to_c_with_a_note_quoting_it(text):
    schedule = run(text)
    assert schedule.parsed is False
    assert schedule.intervals == iv(W1)
    assert f'"{text}"' in (schedule.note or "")


def test_the_note_quotes_multi_line_text_on_one_line():
    schedule = run("NOT A\n SCHEDULE")
    assert schedule.parsed is False
    assert '"NOT A SCHEDULE"' in (schedule.note or "")
    assert "\n" not in (schedule.note or "")


@pytest.mark.parametrize("text", ["", "   ", "\n", " \n\t ", None])
def test_empty_or_blank_text_means_no_schedule_so_active_for_all_of_b_to_c(text):
    schedule = run(text)
    assert schedule.parsed is True
    assert schedule.note is None
    assert schedule.intervals == iv(W1)


def test_an_interval_outside_b_to_c_is_dropped_and_zero_intervals_is_valid():
    schedule = run("OCT 25 0800-1700")
    assert schedule == Schedule(intervals=[], parsed=True, note=None)
    assert run("MON 0800-1700", ("2026-10-31 00:00", "2026-11-02 00:00")).intervals == []  # Monday is C itself


def test_intervals_are_clipped_to_b_to_c():
    schedule = run("0800-1700", ("2026-10-31 12:00", "2026-11-01 10:00"))
    assert schedule.intervals == iv(("2026-10-31 12:00", "2026-10-31 17:00"), ("2026-11-01 08:00", "2026-11-01 10:00"))


def test_an_interval_that_clips_to_nothing_is_dropped():
    assert run("0800-1200", ("2026-10-31 12:00", "2026-10-31 20:00")).intervals == []  # ends exactly at B
    assert run("1200-1500", ("2026-10-31 06:00", "2026-10-31 12:00")).intervals == []  # starts exactly at C


def test_overlapping_touching_and_unsorted_intervals_are_merged_and_sorted():
    window = ("2026-10-31 00:00", "2026-11-01 00:00")
    assert run("0800-1200, 1100-1500", window).intervals == iv(("2026-10-31 08:00", "2026-10-31 15:00"))
    assert run("0800-1200, 1200-1500", window).intervals == iv(("2026-10-31 08:00", "2026-10-31 15:00"))
    assert run("1400-1500, 0800-0900", window).intervals == iv(("2026-10-31 08:00", "2026-10-31 09:00"), ("2026-10-31 14:00", "2026-10-31 15:00"))
    assert run("0800-1200 1000-1300", window).intervals == iv(("2026-10-31 08:00", "2026-10-31 13:00"))


def test_overnight_spans_on_consecutive_days_merge_with_the_next_days_span():
    schedule = run("1800-0800", ("2026-10-31 00:00", "2026-11-02 00:00"))
    assert schedule.intervals == iv(("2026-10-31 00:00", "2026-10-31 08:00"), ("2026-10-31 18:00", "2026-11-01 08:00"), ("2026-11-01 18:00", "2026-11-02 00:00"))
    assert run("0800-0800", ("2026-10-31 00:00", "2026-11-02 00:00")).intervals == iv(("2026-10-31 00:00", "2026-11-02 00:00"))  # equal ends: a full day


def test_perm_needs_horizon_end_and_uses_it_as_the_end():
    b = at("2026-10-31 00:00")
    with pytest.raises(ValueError, match="horizon_end"):
        parse_schedule("0800-1700", b, None, *BLUE_MILL)
    schedule = parse_schedule("0800-1700", b, None, *BLUE_MILL, horizon_end=at("2026-11-02 00:00"))
    assert schedule.intervals == iv(("2026-10-31 08:00", "2026-10-31 17:00"), ("2026-11-01 08:00", "2026-11-01 17:00"))
    assert parse_schedule("", b, None, *BLUE_MILL, horizon_end=at("2026-11-02 00:00")).intervals == iv(("2026-10-31 00:00", "2026-11-02 00:00"))
    unreadable = parse_schedule("TEMPO", b, None, *BLUE_MILL, horizon_end=at("2026-11-02 00:00"))
    assert unreadable.parsed is False and unreadable.intervals == iv(("2026-10-31 00:00", "2026-11-02 00:00"))


def test_a_given_c_wins_over_horizon_end():
    schedule = parse_schedule("H24", at("2026-10-31 00:00"), at("2026-11-01 00:00"), *BLUE_MILL, horizon_end=at("2026-12-01 00:00"))
    assert schedule.intervals == iv(("2026-10-31 00:00", "2026-11-01 00:00"))


def test_datetimes_in_other_timezones_are_converted_to_utc():
    from zoneinfo import ZoneInfo

    b = datetime(2026, 10, 31, 9, 0, tzinfo=ZoneInfo("Europe/Paris"))  # 08:00Z
    c = datetime(2026, 10, 31, 20, 0, tzinfo=ZoneInfo("Europe/Paris"))  # 19:00Z
    schedule = parse_schedule("0700-1700", b, c, *BLUE_MILL)
    assert schedule.intervals == iv(("2026-10-31 08:00", "2026-10-31 17:00"))
    assert all(t.tzinfo is UTC for pair in schedule.intervals for t in pair)


def test_naive_datetimes_are_refused():
    with pytest.raises(ValueError, match="timezone-aware"):
        parse_schedule("0800-1700", datetime(2026, 10, 31), at("2026-11-01 00:00"), *BLUE_MILL)


def test_the_clock_change_does_not_move_notam_times():
    """NOTAM times are UTC all year, so 0800-1700 is 08:00Z on both sides of 25 October."""
    schedule = run("0800-1700", ("2026-10-24 00:00", "2026-10-27 00:00"))
    assert [(s.hour, e.hour) for s, e in schedule.intervals] == [(8, 17)] * 3
