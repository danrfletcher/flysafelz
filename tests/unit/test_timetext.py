"""timetext: times written into sentences, in UTC (the default) or in a zone. Table-driven, exact strings, aware UTC inputs.

Rows: default byte-identity, local clock, the abbreviation at each instant, the two 2026 London clock changes, the repeated
hour, :30 and :45 offsets, a range that crosses local midnight, and unknown / empty zone names. No network, no clock read.
"""

import ast
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path

import pytest

from flysafelz import timetext
from flysafelz.clients.pib_source import utc_text
from flysafelz.timetext import EN_DASH, check_tz, clock, clock_span, labelled_clock, stamp


def at(day: int, hour: int, minute: int = 0, *, month: int = 10, second: int = 0) -> datetime:
    return datetime(2026, month, day, hour, minute, second, tzinfo=UTC)


GMT_DAY = at(31, 7, 30)  # 31 Oct 2026: London is on GMT (the clocks went back on 25 Oct at 01:00 UTC)
BST_DAY = at(24, 7, 30)  # 24 Oct 2026: London is still on BST
D = EN_DASH

# --- the default: None and "UTC" are today's text, byte for byte ---------------------------------------------

DEFAULT_ZONES = [None, "UTC"]


@pytest.mark.parametrize("tz", DEFAULT_ZONES, ids=["None", "UTC"])
@pytest.mark.parametrize(
    ("t", "clock_text", "stamp_text"),
    [
        (at(31, 8), "08:00", "2026-10-31T08:00:00Z"),
        (GMT_DAY, "07:30", "2026-10-31T07:30:00Z"),
        (BST_DAY, "07:30", "2026-10-24T07:30:00Z"),
        (at(25, 0, 30), "00:30", "2026-10-25T00:30:00Z"),
        (at(29, 1, month=3), "01:00", "2026-03-29T01:00:00Z"),
        (at(31, 23, 59, second=59), "23:59", "2026-10-31T23:59:59Z"),
        (datetime(2026, 10, 31, 9, 0, tzinfo=timezone(timedelta(hours=2))), "07:00", "2026-10-31T07:00:00Z"),  # any aware zone in, UTC out
    ],
    ids=["morning", "half past", "a BST day", "25 Oct 00:30", "29 Mar 01:00", "seconds", "+02:00 input"],
)
def test_the_default_is_the_utc_text_of_today(tz, t, clock_text, stamp_text):
    assert clock(t, tz) == clock_text
    assert labelled_clock(t, tz) == clock_text  # no zone label in the default
    assert stamp(t, tz) == stamp_text
    assert stamp(t, tz) == utc_text(t)  # the same text pib_source.utc_text gives, which is left unchanged


@pytest.mark.parametrize("tz", DEFAULT_ZONES, ids=["None", "UTC"])
def test_the_default_span_is_hour_minute_to_hour_minute_with_no_label(tz):
    assert clock_span(at(31, 8), at(31, 18), tz) == f"08:00{D}18:00"
    assert clock_span(at(31, 22), at(31, 23), tz) == f"22:00{D}23:00"
    assert clock_span(at(31, 22), at(31, 23), tz) == clock_span(at(31, 22), at(31, 23))  # tz left out is the same


def test_the_default_span_keeps_the_ends_in_order_across_utc_midnight():
    assert clock_span(at(31, 22), at(1, 2, month=11), None) == f"22:00{D}02:00"  # as before: no date, ends not reordered


# --- a zone: local clock, date and the abbreviation tzname() gives for that instant ----------------------------

LONDON = "Europe/London"


@pytest.mark.parametrize(
    ("t", "tz", "clock_text", "stamp_text"),
    [
        (GMT_DAY, LONDON, "07:30", "2026-10-31 07:30 GMT"),
        (BST_DAY, LONDON, "08:30", "2026-10-24 08:30 BST"),
        (GMT_DAY, "Europe/Paris", "08:30", "2026-10-31 08:30 CET"),
        (GMT_DAY, "America/New_York", "03:30", "2026-10-31 03:30 EDT"),
        (at(2, 7, 30, month=11), "America/New_York", "02:30", "2026-11-02 02:30 EST"),  # US clocks changed on 1 Nov
        (GMT_DAY, "Asia/Tokyo", "16:30", "2026-10-31 16:30 JST"),
        (GMT_DAY, "Asia/Kolkata", "13:00", "2026-10-31 13:00 IST"),  # a :30 offset
        (GMT_DAY, "Pacific/Chatham", "21:15", "2026-10-31 21:15 +1345"),  # a :45 offset; no letters, so tzname() is the offset
        (at(31, 23, 30), LONDON, "23:30", "2026-10-31 23:30 GMT"),
        (at(24, 23, 30), LONDON, "00:30", "2026-10-25 00:30 BST"),  # the local date moves with the clock
    ],
    ids=["London GMT", "London BST", "Paris", "New York", "New York after change", "Tokyo", "Kolkata :30", "Chatham :45", "late GMT", "date moves"],
)
def test_a_zone_gives_the_local_time_and_its_own_abbreviation(t, tz, clock_text, stamp_text):
    assert clock(t, tz) == clock_text
    assert stamp(t, tz) == stamp_text
    assert labelled_clock(t, tz) == f"{clock_text} {stamp_text.rsplit(' ', 1)[1]}"


def test_the_zone_offsets_come_from_zoneinfo_not_from_the_code():
    """Same UTC instant, three zones: the shift is whatever zoneinfo says, and it follows the date."""
    assert [clock(GMT_DAY, z) for z in (LONDON, "Europe/Paris", "Asia/Kolkata")] == ["07:30", "08:30", "13:00"]
    assert [clock(BST_DAY, z) for z in (LONDON, "Europe/Paris", "Asia/Kolkata")] == ["08:30", "09:30", "13:00"]


# --- the two 2026 London clock changes (E18) -----------------------------------------------------------------

CHANGES = [
    # (id, start, end, the range in Europe/London)
    ("BST to GMT, ends differ", at(24, 23, 30), at(25, 2), f"00:30 BST{D}02:00 GMT"),
    ("GMT to BST, ends differ", at(29, 0, 30, month=3), at(29, 2, month=3), f"00:30 GMT{D}03:00 BST"),
    ("a day wholly in GMT", at(31, 8), at(31, 18), f"08:00{D}18:00 GMT"),
    ("a day wholly in BST", at(24, 8), at(24, 18), f"09:00{D}19:00 BST"),
    ("a day wholly in GMT, winter", at(15, 8, month=1), at(15, 18, month=1), f"08:00{D}18:00 GMT"),
    ("ends before the BST change", at(28, 8, month=3), at(28, 18, month=3), f"08:00{D}18:00 GMT"),
    ("starts after the BST change", at(30, 8, month=3), at(30, 18, month=3), f"09:00{D}19:00 BST"),
]


@pytest.mark.parametrize(("start", "end", "expected"), [c[1:] for c in CHANGES], ids=[c[0] for c in CHANGES])
def test_a_range_carries_the_abbreviation_that_applies_at_each_end(start, end, expected):
    assert clock_span(start, end, LONDON) == expected


def test_the_change_instants_are_one_hour_apart_in_text_and_in_abbreviation():
    assert stamp(at(25, 0, 59), LONDON) == "2026-10-25 01:59 BST"
    assert stamp(at(25, 1, 0), LONDON) == "2026-10-25 01:00 GMT"  # 01:00 UTC is the first GMT instant
    assert stamp(at(29, 0, 59, month=3), LONDON) == "2026-03-29 00:59 GMT"
    assert stamp(at(29, 1, 0, month=3), LONDON) == "2026-03-29 02:00 BST"  # the 01:00-02:00 local hour does not exist


def test_two_segments_either_side_of_the_change_get_their_own_abbreviation():
    before = stamp(at(24, 23), LONDON)
    after = stamp(at(25, 1), LONDON)
    assert (before, after) == ("2026-10-25 00:00 BST", "2026-10-25 01:00 GMT")


# --- the repeated local hour on 25 Oct ----------------------------------------------------------------------


def test_the_repeated_hour_gives_two_different_texts_for_two_instants():
    first, second = at(25, 0, 30), at(25, 1, 30)  # both are 01:30 on a London clock
    assert labelled_clock(first, LONDON) == "01:30 BST"
    assert labelled_clock(second, LONDON) == "01:30 GMT"
    assert stamp(first, LONDON) == "2026-10-25 01:30 BST"
    assert stamp(second, LONDON) == "2026-10-25 01:30 GMT"
    assert stamp(first, LONDON) != stamp(second, LONDON)
    assert clock(first, LONDON) == clock(second, LONDON) == "01:30"  # the bare clock is the same: only the labelled forms tell them apart


def test_a_range_inside_the_repeated_hour_names_both_abbreviations():
    assert clock_span(at(25, 0, 30), at(25, 1, 30), LONDON) == f"01:30 BST{D}01:30 GMT"


# --- a range that crosses local midnight ---------------------------------------------------------------------


def test_a_range_across_local_midnight_shows_both_dates_and_keeps_its_order():
    # 14:00-18:00 UTC is 23:00 on the 31st to 03:00 on the 1st in Tokyo
    assert clock_span(at(31, 14), at(31, 18), "Asia/Tokyo") == f"2026-10-31 23:00 JST{D}2026-11-01 03:00 JST"
    assert clock(at(31, 14), "Asia/Tokyo") == "23:00" and clock(at(31, 18), "Asia/Tokyo") == "03:00"  # the clock form stays HH:MM


def test_a_range_that_stays_on_one_local_day_has_no_date_even_if_the_utc_days_differ():
    assert clock_span(at(31, 22), at(1, 2, month=11), "Asia/Tokyo") == f"07:00{D}11:00 JST"  # 22:00-02:00 UTC is 07:00-11:00 on one Tokyo day


def test_london_midnight_crossing_in_gmt():
    assert clock_span(at(31, 22), at(1, 2, month=11), LONDON) == f"2026-10-31 22:00 GMT{D}2026-11-01 02:00 GMT"


# --- UTC spelled as a zone is a zone: it says UTC, never Z ---------------------------------------------------


@pytest.mark.parametrize("tz", ["utc", "Etc/UTC"])
def test_utc_spelled_as_a_zone_is_accepted_and_written_as_a_zone(tz):
    # Only None and "UTC" promise today's byte-identical text. These give the zone form, with the abbreviation UTC (not Z).
    assert stamp(GMT_DAY, tz) == "2026-10-31 07:30 UTC"
    assert labelled_clock(GMT_DAY, tz) == "07:30 UTC"
    assert clock_span(at(31, 8), at(31, 18), tz) == f"08:00{D}18:00 UTC"


# --- unknown and empty names fail closed ---------------------------------------------------------------------

BAD_NAMES = ["", "Mars/Olympus", "Europe", "../etc/passwd", "/etc/localtime", "Europe/London\x00", "UTC ", "London"]


@pytest.mark.parametrize("tz", BAD_NAMES)
def test_an_unknown_or_empty_zone_raises_value_error_naming_it(tz):
    for call in (
        lambda: check_tz(tz),
        lambda: clock(GMT_DAY, tz),
        lambda: labelled_clock(GMT_DAY, tz),
        lambda: stamp(GMT_DAY, tz),
        lambda: clock_span(GMT_DAY, GMT_DAY + timedelta(hours=1), tz),
    ):
        with pytest.raises(ValueError, match="unknown time zone") as caught:
            call()
        assert repr(tz) in str(caught.value)


def test_a_bad_zone_never_falls_back_to_london_or_utc():
    with pytest.raises(ValueError):
        stamp(GMT_DAY, "Mars/Olympus")  # not the UTC text, not the London text


@pytest.mark.parametrize("tz", [None, "UTC", "utc", "Etc/UTC", LONDON, "Asia/Kolkata"])
def test_check_tz_accepts_real_zones_and_returns_nothing(tz):
    assert check_tz(tz) is None


# --- a naive time is refused, as it is everywhere else -------------------------------------------------------


@pytest.mark.parametrize("tz", [None, "UTC", LONDON])
def test_a_naive_time_raises_whatever_the_zone(tz):
    naive = datetime(2026, 10, 31, 7, 30)
    for call in (clock, labelled_clock, stamp):
        with pytest.raises(ValueError, match="timezone-aware"):
            call(naive, tz)
    with pytest.raises(ValueError, match="timezone-aware"):
        clock_span(naive, GMT_DAY, tz)
    with pytest.raises(ValueError, match="timezone-aware"):
        clock_span(GMT_DAY, naive, tz)


# --- the fence: stdlib only, no numbers, no abbreviation table, and the CLI stays UTC -------------------------------

SRC = Path(timetext.__file__).parent
TREE = ast.parse(Path(timetext.__file__).read_text(encoding="utf-8"))


def test_timetext_imports_only_datetime_and_zoneinfo():
    modules = {n.module for n in ast.walk(TREE) if isinstance(n, ast.ImportFrom)} | {a.name for n in ast.walk(TREE) if isinstance(n, ast.Import) for a in n.names}
    assert modules == {"datetime", "zoneinfo"}


def test_timetext_holds_no_number_no_offset_and_no_abbreviation_table():
    numbers = [n.value for n in ast.walk(TREE) if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)) and not isinstance(n.value, bool)]
    assert numbers == []  # no threshold, no offset in hours or minutes
    words = {n.value for n in ast.walk(TREE) if isinstance(n, ast.Constant) and isinstance(n.value, str)}
    assert not words & {"GMT", "BST", "CET", "EST", "EDT", "JST", "IST"}  # the abbreviation comes from zoneinfo's tzname()
    assert not [n for n in ast.walk(TREE) if isinstance(n, ast.Dict)]


def test_the_cli_and_the_core_stay_utc():
    cli = (SRC / "cli.py").read_text(encoding="utf-8")
    assert "--tz" not in cli and "tz=" not in cli and "timetext" not in cli
    assert "All times UTC." in cli
    for name in ("daylight.py", "schedule.py", "capture.py"):  # UTC internals the spec keeps
        assert "timetext" not in (SRC / name).read_text(encoding="utf-8"), name
    assert "timetext" not in (SRC / "clients" / "pib_source.py").read_text(encoding="utf-8")  # request parameters stay UTC
