"""parse_window: the model's local start/end in the thread's tz, the default window, the errors the model relays (G8, FR1, FR15, E8, E18).

Every case calls `parse_window` with a frozen `now` and reads what it returns. No clock, no network.
"""

import inspect
from datetime import UTC, datetime

import pytest

from flysafelz.agent import window
from flysafelz.agent.window import Window, WindowError, local_label, parse_window
from flysafelz.assess import fetch_horizon

LONDON = "Europe/London"
NEW_YORK = "America/New_York"
SATURDAY_MORNING = datetime(2026, 10, 31, 7, 30, tzinfo=UTC)  # the frozen clock: London is on GMT
DAY_BEFORE_THE_CLOCKS_GO_BACK = datetime(2026, 10, 24, 12, 0, tzinfo=UTC)
DAY_BEFORE_THE_CLOCKS_GO_FORWARD = datetime(2026, 3, 28, 12, 0, tzinfo=UTC)


def utc(month: int, day: int, hour: int, minute: int = 0) -> datetime:
    return datetime(2026, month, day, hour, minute, tzinfo=UTC)


# (id, start, end, tz, now, start_utc, end_utc, words, assumed)
WINDOWS = [
    ("explicit, one day, GMT", "2026-10-31 10:00", "2026-10-31 16:00", LONDON, SATURDAY_MORNING, utc(10, 31, 10), utc(10, 31, 16), "Saturday 31 Oct 10:00 – 16:00 GMT", False),
    ("explicit, across two days", "2026-10-31 22:00", "2026-11-01 02:00", LONDON, SATURDAY_MORNING, utc(10, 31, 22), utc(11, 1, 2), "Saturday 31 Oct 22:00 – Sunday 1 Nov 02:00 GMT", False),
    ("an end at midnight closes the day before", "2026-10-31 18:00", "2026-11-01 00:00", LONDON, SATURDAY_MORNING, utc(10, 31, 18), utc(11, 1, 0), "Saturday 31 Oct 18:00 – 24:00 GMT", False),
    ("whitespace around a time is ignored", " 2026-10-31 10:00 ", "2026-10-31 16:00\n", LONDON, SATURDAY_MORNING, utc(10, 31, 10), utc(10, 31, 16), "Saturday 31 Oct 10:00 – 16:00 GMT", False),
    (
        "a summer time window is read in BST",
        "2026-07-01 10:00",
        "2026-07-01 12:00",
        LONDON,
        datetime(2026, 7, 1, 6, 0, tzinfo=UTC),
        utc(7, 1, 9),
        utc(7, 1, 11),
        "Wednesday 1 Jul 10:00 – 12:00 BST",
        False,
    ),
    (
        "another zone: New York is on EDT on 30 Oct",
        "2026-10-30 10:00",
        "2026-10-30 12:00",
        NEW_YORK,
        datetime(2026, 10, 30, 6, 0, tzinfo=UTC),
        utc(10, 30, 14),
        utc(10, 30, 16),
        "Friday 30 Oct 10:00 – 12:00 EDT",
        False,
    ),
    # E18: the clocks go back at 02:00 BST on 25 Oct 2026 (01:00 UTC)
    (
        "E18 autumn: BST before the change, GMT after",
        "2026-10-25 00:30",
        "2026-10-25 03:00",
        LONDON,
        DAY_BEFORE_THE_CLOCKS_GO_BACK,
        utc(10, 24, 23, 30),
        utc(10, 25, 3),
        "Sunday 25 Oct 00:30 BST – 03:00 GMT",
        False,
    ),
    (
        "E18 autumn: the ambiguous 01:30 is the first one (BST, 00:30Z)",
        "2026-10-25 01:30",
        "2026-10-25 03:00",
        LONDON,
        DAY_BEFORE_THE_CLOCKS_GO_BACK,
        utc(10, 25, 0, 30),
        utc(10, 25, 3),
        "Sunday 25 Oct 01:30 BST – 03:00 GMT",
        False,
    ),
    (
        "E18 autumn: after the change it is all GMT",
        "2026-10-25 02:00",
        "2026-10-25 04:00",
        LONDON,
        DAY_BEFORE_THE_CLOCKS_GO_BACK,
        utc(10, 25, 2),
        utc(10, 25, 4),
        "Sunday 25 Oct 02:00 – 04:00 GMT",
        False,
    ),
    # E18: the clocks go forward at 01:00 GMT on 29 Mar 2026
    (
        "E18 spring: GMT before the change, BST after",
        "2026-03-29 00:00",
        "2026-03-29 03:00",
        LONDON,
        DAY_BEFORE_THE_CLOCKS_GO_FORWARD,
        utc(3, 29, 0),
        utc(3, 29, 2),
        "Sunday 29 Mar 00:00 GMT – 03:00 BST",
        False,
    ),
    (
        "E18 spring: after the change it is all BST",
        "2026-03-29 02:00",
        "2026-03-29 04:00",
        LONDON,
        DAY_BEFORE_THE_CLOCKS_GO_FORWARD,
        utc(3, 29, 1),
        utc(3, 29, 3),
        "Sunday 29 Mar 02:00 – 04:00 BST",
        False,
    ),
    # FR15 / FR1: a window that starts inside the horizon and ends beyond it is not touched
    (
        "FR15 a window across the horizon passes through unchanged",
        "2026-11-07 10:00",
        "2026-11-09 12:00",
        LONDON,
        SATURDAY_MORNING,
        utc(11, 7, 10),
        utc(11, 9, 12),
        "Saturday 7 Nov 10:00 – Monday 9 Nov 12:00 GMT",
        False,
    ),
    ("a window that has begun but not ended is fine", "2026-10-31 05:00", "2026-10-31 09:00", LONDON, SATURDAY_MORNING, utc(10, 31, 5), utc(10, 31, 9), "Saturday 31 Oct 05:00 – 09:00 GMT", False),
    # the default window: from now rounded down to the hour to the end of tomorrow in tz
    ("the default window in London", None, None, LONDON, SATURDAY_MORNING, utc(10, 31, 7), utc(11, 2, 0), "assumed window: Saturday 31 Oct 07:00 – Sunday 1 Nov 24:00 GMT", True),
    ("empty strings count as left out", "", "  ", LONDON, SATURDAY_MORNING, utc(10, 31, 7), utc(11, 2, 0), "assumed window: Saturday 31 Oct 07:00 – Sunday 1 Nov 24:00 GMT", True),
    (
        "the default window in New York ends at the end of tomorrow there",
        None,
        None,
        NEW_YORK,
        SATURDAY_MORNING,
        utc(10, 31, 7),
        utc(11, 2, 5),
        "assumed window: Saturday 31 Oct 03:00 EDT – Sunday 1 Nov 24:00 EST",
        True,
    ),
    (
        "the default window in London across the clocks going back",
        None,
        None,
        LONDON,
        DAY_BEFORE_THE_CLOCKS_GO_BACK,
        utc(10, 24, 12),
        utc(10, 26, 0),
        "assumed window: Saturday 24 Oct 13:00 BST – Sunday 25 Oct 24:00 GMT",
        True,
    ),
]


@pytest.mark.parametrize(
    ("start", "end", "tz", "now", "start_utc", "end_utc", "words", "assumed"),
    [pytest.param(*row[1:], id=row[0]) for row in WINDOWS],
)
def test_a_window_is_read_in_the_threads_zone_and_described_in_words(start, end, tz, now, start_utc, end_utc, words, assumed):
    got = parse_window(start, end, now=now, tz=tz)
    assert got == Window(start_utc, end_utc, words, assumed)


NOT_A_WINDOW = [
    ("only start", "2026-10-31 10:00", None, LONDON, "Give both start and end, or neither to use the default window."),
    ("only end", None, "2026-10-31 16:00", LONDON, "Give both start and end, or neither to use the default window."),
    ("only start, end empty", "2026-10-31 10:00", "", LONDON, "Give both start and end, or neither to use the default window."),
    ("10am", "10am", "2026-10-31 16:00", LONDON, "The start must be a local time written YYYY-MM-DD HH:MM, for example 2026-10-31 10:00."),
    ("an end in words", "2026-10-31 10:00", "4pm", LONDON, "The end must be a local time written YYYY-MM-DD HH:MM, for example 2026-10-31 10:00."),
    ("a T and a Z instead of a space", "2026-10-31T10:00Z", "2026-10-31 16:00", LONDON, "The start must be a local time written YYYY-MM-DD HH:MM, for example 2026-10-31 10:00."),
    ("single digits", "2026-10-31 9:00", "2026-10-31 16:00", LONDON, "The start must be a local time written YYYY-MM-DD HH:MM, for example 2026-10-31 10:00."),
    ("month 13", "2026-13-01 10:00", "2026-13-01 12:00", LONDON, "The start is not a real date and time; write it as YYYY-MM-DD HH:MM."),
    ("hour 24", "2026-10-31 10:00", "2026-10-31 24:00", LONDON, "The end is not a real date and time; write it as YYYY-MM-DD HH:MM."),
    ("31 November", "2026-11-31 10:00", "2026-12-01 10:00", LONDON, "The start is not a real date and time; write it as YYYY-MM-DD HH:MM."),
    ("end equals start", "2026-10-31 10:00", "2026-10-31 10:00", LONDON, "The end must be after the start."),
    ("end before start", "2026-10-31 16:00", "2026-10-31 10:00", LONDON, "The end must be after the start."),
    ("wholly past", "2026-10-31 04:00", "2026-10-31 07:00", LONDON, "That time has passed: the window ended before now."),
    ("ends exactly now", "2026-10-31 06:00", "2026-10-31 07:30", LONDON, "That time has passed: the window ended before now."),
    ("yesterday", "2026-10-30 10:00", "2026-10-30 16:00", LONDON, "That time has passed: the window ended before now."),
    # E8: the horizon is 7 days on: fetch_horizon(2026-10-31T07:30Z) is 2026-11-08T00:00Z
    ("E8 a slot 9 days out", "2026-11-09 10:00", "2026-11-09 12:00", LONDON, "That window is too far out to assess: it starts beyond the forecast horizon."),
    ("E8 a window starting exactly at the horizon", "2026-11-08 00:00", "2026-11-08 02:00", LONDON, "That window is too far out to assess: it starts beyond the forecast horizon."),
    # E18: 01:30 on 29 Mar 2026 does not exist in London
    ("E18 the nonexistent 01:30", "2026-03-29 01:30", "2026-03-29 03:00", LONDON, "The start 2026-03-29 01:30 does not exist in Europe/London: the clocks go forward then, so that hour is skipped."),
    ("E18 a nonexistent end", "2026-03-29 00:00", "2026-03-29 01:30", LONDON, "The end 2026-03-29 01:30 does not exist in Europe/London: the clocks go forward then, so that hour is skipped."),
]


@pytest.mark.parametrize(
    ("start", "end", "tz", "text"),
    [pytest.param(*row[1:], id=row[0]) for row in NOT_A_WINDOW],
)
def test_a_window_that_cannot_be_assessed_gives_one_sentence_for_the_model_to_relay(start, end, tz, text):
    now = DAY_BEFORE_THE_CLOCKS_GO_FORWARD if "2026-03-29" in f"{start} {end}" else SATURDAY_MORNING
    assert parse_window(start, end, now=now, tz=tz) == WindowError(text)


def test_a_new_york_window_is_judged_against_the_london_horizon():
    """FR1: the day tiers are Europe/London days, so the horizon is one instant for every zone (00:00Z on 8 Nov here).

    19:30 EST on 7 Nov is 00:30Z on 8 Nov and 19:00 EST is 00:00Z: both at or after the horizon, so too far out, although 7 Nov is
    still 'today plus six' in New York. 18:59 EST is 23:59Z, inside it.
    """
    assert fetch_horizon(SATURDAY_MORNING) == utc(11, 8, 0)
    too_far = WindowError("That window is too far out to assess: it starts beyond the forecast horizon.")
    assert parse_window("2026-11-07 19:30", "2026-11-07 21:00", now=SATURDAY_MORNING, tz=NEW_YORK) == too_far
    assert parse_window("2026-11-07 19:00", "2026-11-07 21:00", now=SATURDAY_MORNING, tz=NEW_YORK) == too_far
    inside = parse_window("2026-11-07 18:59", "2026-11-07 21:00", now=SATURDAY_MORNING, tz=NEW_YORK)
    assert isinstance(inside, Window) and inside.start_utc == utc(11, 7, 23, 59)


@pytest.mark.parametrize("moment", [utc(10, 25, 0, 30), utc(10, 25, 1, 30)])
def test_local_label_takes_the_abbreviation_from_each_instant(moment):
    labelled = local_label(moment, LONDON)
    assert labelled == ("01:30 BST" if moment == utc(10, 25, 0, 30) else "01:30 GMT")


def test_local_label_before_and_after_each_change():
    assert local_label(utc(10, 25, 0, 59), LONDON) == "01:59 BST"
    assert local_label(utc(10, 25, 1, 0), LONDON) == "01:00 GMT"
    assert local_label(utc(3, 29, 0, 59), LONDON) == "00:59 GMT"
    assert local_label(utc(3, 29, 1, 0), LONDON) == "02:00 BST"
    assert local_label(utc(10, 31, 10), LONDON) == "10:00 GMT"
    assert local_label(utc(7, 1, 9), LONDON) == "10:00 BST"
    assert local_label(utc(10, 31, 10), "UTC") == "10:00 UTC"


def test_local_label_and_the_parser_refuse_a_naive_time():
    with pytest.raises(ValueError, match="timezone-aware"):
        local_label(datetime(2026, 10, 31, 10, 0), LONDON)
    with pytest.raises(ValueError, match="timezone-aware"):
        parse_window(None, None, now=datetime(2026, 10, 31, 7, 30), tz=LONDON)


def test_the_parser_takes_the_clock_and_the_zone_from_its_caller_and_no_place():
    """F2: no lat/lon, and tz and now are keyword-only arguments the caller reads from the thread and the run context, never the model's."""
    parameters = inspect.signature(parse_window).parameters
    assert list(parameters) == ["start", "end", "now", "tz"]
    assert parameters["now"].kind is inspect.Parameter.KEYWORD_ONLY and parameters["tz"].kind is inspect.Parameter.KEYWORD_ONLY
    assert not {"lat", "lon", "latitude", "longitude"} & set(parameters)


def test_the_parser_uses_no_clock_of_its_own():
    """The run's `now` is the only clock: the module never asks the system for the time."""
    source = inspect.getsource(window)
    assert "datetime.now" not in source and "utcnow" not in source and "time.time" not in source
