"""E8 corpus: every distinct D-item string in the 31 Oct 2026 PIB parses, inside a synthetic B-C.

The strings come from the NATS PIB archive (Jonty/uk-notam-archive) at commit
bdab14658f0270dc41460eebcd03332abfb230ec. The test reads only the committed text file; rebuild it with
scripts/extract_d_items.py.
"""

import re
from datetime import UTC, datetime
from pathlib import Path

import pytest

from flysafelz.schedule import parse_schedule

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "schedule" / "d_items_2026-10-31.txt"
ARCHIVE_SHA = "bdab14658f0270dc41460eebcd03332abfb230ec"
BLUE_MILL = (53.802797, -0.086644)
B = datetime(2026, 10, 1, tzinfo=UTC)
C = datetime(2027, 1, 31, tzinfo=UTC)

# Strings that are allowed to come back parsed=False. Reviewed by hand: all 201 strings in this PIB parse, so the list is empty.
# Adding a string here needs a reason next to it.
EXPECTED_UNPARSED: list[str] = []


def load() -> list[str]:
    lines = FIXTURE.read_text(encoding="utf-8").splitlines()
    return [line.replace("\\n", "\n") for line in lines if not line.startswith("#")]


STRINGS = load()


def test_the_fixture_names_its_archive_sha_and_holds_201_distinct_strings():
    header = [line for line in FIXTURE.read_text(encoding="utf-8").splitlines() if line.startswith("#")]
    assert any(ARCHIVE_SHA in line for line in header)
    assert len(STRINGS) == 201
    assert len(set(STRINGS)) == 201


def test_the_corpus_has_81_shapes_once_digits_are_masked():
    """The plan said 76. The real number found in the archive PIB is 81."""
    shapes = {re.sub(r"\d", "9", s) for s in STRINGS}
    assert len(shapes) == 81


def test_the_corpus_keeps_its_embedded_newlines():
    assert any("\n" in s for s in STRINGS)


@pytest.mark.parametrize("text", STRINGS, ids=[f"{i:03}" for i in range(len(STRINGS))])
def test_each_d_string_parses_with_sane_intervals_inside_b_to_c(text):
    schedule = parse_schedule(text, B, C, *BLUE_MILL)
    if text in EXPECTED_UNPARSED:
        assert schedule.parsed is False
        return
    assert schedule.parsed is True, schedule.note
    assert schedule.note is None
    for start, end in schedule.intervals:
        assert B <= start < end <= C, (text, start, end)
        assert start.tzinfo is UTC and end.tzinfo is UTC
    # sorted and merged: each interval starts strictly after the one before it ends
    for (_, previous_end), (next_start, _) in zip(schedule.intervals, schedule.intervals[1:], strict=False):
        assert previous_end < next_start, text


def test_no_string_is_silently_empty_over_a_four_month_window():
    """Each real string is active on some day between October and January, so none parses to zero intervals."""
    empty = [s for s in STRINGS if not parse_schedule(s, B, C, *BLUE_MILL).intervals]
    assert empty == []


def test_a_string_outside_the_expected_list_that_fails_would_be_caught():
    schedule = parse_schedule("NOT A SCHEDULE", B, C, *BLUE_MILL)
    assert schedule.parsed is False
    assert "NOT A SCHEDULE" not in EXPECTED_UNPARSED
