"""G14 (b) times and dates: a time must be one a tool printed, a date one the prompt calendar or a tool lists; clarify answers; the stated window."""

from datetime import date

import pytest
from guard_world import findings, text_turn, turn, turn_of

from flysafelz.agent.guard import ToolResult, scan

# --- times and dates ---------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("written", ["10:00", "16:00", "17:02", "10:00\u201316:00", "10:00-16:00", "1000-1600", "10:00:00"])
def test_a_time_a_tool_printed_passes(written):
    assert findings(f"Open from {written} GMT.", turn()) == []


@pytest.mark.parametrize(("written", "shown"), [("17:30", "17:30"), ("5pm", "17:00"), ("4:30 pm", "16:30"), ("0930-1830", "09:30")])
def test_a_time_no_tool_printed_is_a_finding_in_any_form(written, shown):
    found = findings(f"Open from {written} GMT.", turn())
    assert found and found[0].endswith("is not a time any tool printed this turn")


def test_four_pm_passes_when_a_tool_printed_16_00():
    assert findings("Closes at 4pm.", turn()) == []
    assert findings("Closes at 4:00 p.m.", turn()) == []
    assert findings("Closes at 12am.", turn()) == ["'12am' is not a time any tool printed this turn"]


def test_a_height_range_is_not_read_as_a_time_range():
    assert findings("Airspace from 1000-1500 ft.", text_turn("1000-1500 ft")) == []
    assert findings("Airspace from 1000-1500 ft.", text_turn("1000 ft and 1500 ft")) == []


@pytest.mark.parametrize("written", ["Saturday 31 Oct", "Sat 31 Oct", "31 Oct", "31st October", "31st of October 2026", "2026-10-31", "Saturday 31 October"])
def test_a_date_the_calendar_or_a_tool_lists_passes(written):
    assert findings(f"On {written} it is calm.", turn()) == []


def test_the_calendar_lists_days_after_the_tool_text():
    only_calendar = turn_of(ToolResult("get_weather", "no dates here"), calendar=(date(2026, 11, 3),))
    assert findings("Tuesday 3 Nov looks calm.", only_calendar) == []
    assert findings("Wednesday 4 Nov looks calm.", only_calendar) == ["'Wednesday 4 Nov' is not a date the calendar or any tool lists this turn"]


@pytest.mark.parametrize(
    ("written", "shown"),
    [("Sunday 31 Oct", "Sunday 31 Oct"), ("9 Nov", "9 Nov"), ("2026-11-09", "2026-11-09"), ("Saturday 30 Oct", "Saturday 30 Oct")],
)
def test_a_date_that_is_not_listed_or_has_the_wrong_weekday_is_a_finding(written, shown):
    assert findings(f"On {written} it is calm.", turn()) == [f"'{shown}' is not a date the calendar or any tool lists this turn"]


def test_a_year_is_not_checked():
    assert findings("Valid for the 2026 and 2027 seasons.", turn()) == []


def test_the_scan_reads_each_kind_once():
    found = scan("Saturday 31 Oct 10:00-16:00 GMT, 5.2 mph at 53.8028, -0.0866 (H7171/26).")
    assert [d.written for d in found.dates] == ["Saturday 31 Oct"]
    assert [t.clock for t in found.times] == ["10:00", "16:00"] or [t.clock for t in found.times] == ["10:00"]
    assert [n.written for n in found.numbers] == ["5.2 mph"]
    assert [p.written for p in found.places] == ["53.8028, -0.0866"]


def test_a_clarify_answer_naming_a_day_in_the_calendar_passes():
    assert findings("Did you mean Sunday 1 Nov?", turn_of(calendar=(date(2026, 10, 31), date(2026, 11, 1)))) == []
    assert findings("Did you mean Monday 2 Nov?", turn_of(calendar=(date(2026, 10, 31), date(2026, 11, 1)))) == [
        "'Monday 2 Nov' is not a date the calendar or any tool lists this turn"
    ]


@pytest.mark.parametrize("label", ["GMT", "BST", "UTC", "local time", "Z"])
def test_the_stated_window_passes_against_the_printed_window_whatever_the_zone_label(label):
    printed = text_turn("Window Saturday 31 Oct 10:00-16:00 GMT.")
    assert findings(f"Assumed window: Saturday 31 Oct 10:00\u201316:00 {label}.", printed) == []


def test_a_time_inside_the_stated_window_that_no_tool_printed_is_a_finding():
    printed = text_turn("Window Saturday 31 Oct 10:00-16:00 GMT.")
    assert findings("Assumed window: Saturday 31 Oct 10:00\u201317:30 GMT.", printed) == ["'17:30' is not a time any tool printed this turn"]
