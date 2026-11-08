"""G3 and E5, E6, E15, E15b, E15c: the one system prompt, filled in by code from the run's `now` and the thread's `tz`.

The prompt is read as the model gets it: through the real graph for the "rebuilt every turn" and "not stored" checks, and by calling `system_prompt`
for the table over times and zones. A snapshot (tests/golden/system_prompt.txt) holds one whole prompt, so a change to its wording shows in a diff;
`pytest --update-golden` rewrites it.
"""

import re
from datetime import timedelta
from itertools import pairwise
from pathlib import Path

import pytest
from fakes import FakeToolChatModel
from graph_helpers import agent_graph, ask, assess_call, blue_mill_context, system_prompts, utc
from langchain_core.messages import AIMessage, SystemMessage

from flysafelz.agent.graph import TOOL_NAMES
from flysafelz.agent.prompts import BANNED_LINE_START, BANNED_PHRASES, PATH_TAGS, PATHS, banned_line, calendar_lines, system_prompt
from flysafelz.agent.settings import SETTINGS

GOLDEN = Path(__file__).resolve().parents[2] / "golden" / "system_prompt.txt"
GMT, BST, EDT = "Europe/London", "Europe/London", "America/New_York"

# (now in UTC, tz, today's heading, the time now, calendar's first line, calendar's last line)
TABLE = [
    pytest.param(utc(2026, 10, 31, 7, 30), "Europe/London", "Saturday 31 Oct (2026-10-31)", "07:30 GMT", "- today: Saturday 31 Oct (2026-10-31)", "- Saturday 7 Nov (2026-11-07)", id="GMT"),
    pytest.param(utc(2026, 7, 22, 12, 0), "Europe/London", "Wednesday 22 Jul (2026-07-22)", "13:00 BST", "- today: Wednesday 22 Jul (2026-07-22)", "- Wednesday 29 Jul (2026-07-29)", id="BST"),
    pytest.param(
        utc(2026, 10, 24, 23, 30),
        "Europe/London",
        "Sunday 25 Oct (2026-10-25)",
        "00:30 BST",
        "- today: Sunday 25 Oct (2026-10-25)",
        "- Sunday 1 Nov (2026-11-01)",
        id="change day, date from tz not UTC",
    ),
    pytest.param(
        utc(2026, 10, 25, 1, 30),
        "Europe/London",
        "Sunday 25 Oct (2026-10-25)",
        "01:30 GMT",
        "- today: Sunday 25 Oct (2026-10-25)",
        "- Sunday 1 Nov (2026-11-01)",
        id="change day, after the clocks went back",
    ),
    pytest.param(utc(2026, 10, 31, 7, 30), "America/New_York", "Saturday 31 Oct (2026-10-31)", "03:30 EDT", "- today: Saturday 31 Oct (2026-10-31)", "- Saturday 7 Nov (2026-11-07)", id="non-UK zone"),
    pytest.param(
        utc(2026, 10, 31, 2, 0),
        "America/New_York",
        "Friday 30 Oct (2026-10-30)",
        "22:00 EDT",
        "- today: Friday 30 Oct (2026-10-30)",
        "- Friday 6 Nov (2026-11-06)",
        id="non-UK zone, still yesterday in UTC",
    ),
    pytest.param(
        utc(2026, 10, 31, 23, 30),
        "Pacific/Auckland",
        "Sunday 1 Nov (2026-11-01)",
        "12:30 NZDT",
        "- today: Sunday 1 Nov (2026-11-01)",
        "- Sunday 8 Nov (2026-11-08)",
        id="a zone ahead of UTC, already tomorrow in UTC terms",
    ),
]


def lines_of(prompt: str) -> list[str]:
    return prompt.split("\n")


def calendar_of(prompt: str) -> list[str]:
    return [line for line in lines_of(prompt) if re.match(r"- (today: |tomorrow: )?\w+day \d+ \w+ \(\d{4}-\d{2}-\d{2}\)$", line)]


# --- G3: what code fills in -------------------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(("now", "tz", "heading", "clock", "first", "last"), TABLE)
def test_the_prompt_has_todays_date_the_clock_with_its_abbreviation_and_an_eight_day_calendar_in_tz(now, tz, heading, clock, first, last):
    prompt = system_prompt(now, tz)
    assert f"Today is {heading}." in prompt
    assert f"The time now is {clock}." in prompt
    assert f"The pilot's time zone is {tz}." in prompt
    calendar = calendar_of(prompt)
    assert len(calendar) == SETTINGS.calendar_days + 1 == 8
    assert calendar[0] == first and calendar[-1] == last
    assert calendar[1].startswith("- tomorrow: ")


@pytest.mark.parametrize(("now", "tz"), [p.values[:2] for p in TABLE])
def test_the_calendar_days_follow_one_another_and_each_weekday_matches_its_date(now, tz):
    calendar = calendar_of(system_prompt(now, tz))
    dates = [re.search(r"\((\d{4}-\d{2}-\d{2})\)", line).group(1) for line in calendar]
    from datetime import date

    days = [date.fromisoformat(d) for d in dates]
    assert [b - a for a, b in pairwise(days)] == [timedelta(days=1)] * 7
    for line, day in zip(calendar, days, strict=True):
        assert f"{day:%A} {day.day} {day:%b}" in line


def test_e6_with_now_friday_the_calendar_lists_tomorrow_as_saturday_31_oct_and_sunday_1_nov_after_it():
    calendar = calendar_of(system_prompt(utc(2026, 10, 30, 7, 30), "Europe/London"))
    assert calendar[0] == "- today: Friday 30 Oct (2026-10-30)"
    assert calendar[1] == "- tomorrow: Saturday 31 Oct (2026-10-31)"
    assert calendar[2] == "- Sunday 1 Nov (2026-11-01)"


def test_a_calendar_of_more_days_is_as_long_as_asked():
    assert len(calendar_lines(utc(2026, 10, 31, 7, 30), 3)) == 4


# --- G3: the fixed text ----------------------------------------------------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def prompt() -> str:
    return system_prompt(utc(2026, 10, 31, 7, 30), "Europe/London")


def test_the_four_paths_are_named_with_their_triggers(prompt):
    for path in PATHS:
        assert f"- {path}:" in prompt
    assert PATHS == ("assess", "explain", "clarify", "decline")
    text = prompt.lower()
    for trigger in ("whether, when or for how long they can fly", "what a notam", "vague", "off-topic", "out of scope", "landowner", "kit", "ignore your rules"):
        assert trigger in text, trigger


def test_every_tool_the_prompt_names_is_one_of_the_five(prompt):
    named = set(re.findall(r"\b(get_\w+|search_guidance|assess)\b", prompt))
    assert named and named <= set(TOOL_NAMES)
    assert {"assess", "get_notams", "get_airspace", "search_guidance"} <= named


def test_the_number_rule_is_stated(prompt):
    assert "Quote numbers as the tools give them or round them; never compute" in prompt


def test_the_rag_rule_is_stated(prompt):
    assert "Guidance explains, tools give facts" in prompt


def test_the_tool_arguments_rule_says_a_window_only(prompt):
    assert "Never give coordinates or a time zone" in prompt and "YYYY-MM-DD HH:MM" in prompt


def test_the_default_window_sentence_matches_what_the_tool_does(prompt):
    """The prompt says assess covers the rest of today and tomorrow when no time is given; window.py's default window is that."""
    assert "leave start and end out" in prompt and "rest of today and tomorrow" in prompt


def test_the_prompt_tells_the_model_to_say_where_a_value_sits_using_the_tools_phrase(prompt):
    """I10: the tools print `over the 6 mph limit`; the guard reads that phrase, so the model is asked to use it."""
    assert "When a value is near or past a limit, say where it sits using the tool's phrase, e.g. 'over the 6 mph limit'. You don't need to give the value itself." in prompt
    assert prompt.count("limit'") == 1


def test_compound_messages_have_the_precedence_verdict_first_and_a_one_line_decline_of_only_the_out_of_scope_part(prompt):
    assert "clarify > assess > explain > decline" in prompt
    assert "verdict first, then answer every other part" in prompt
    assert "Decline only the out-of-scope part, in one line" in prompt


def test_a_push_attached_to_a_request_to_fly_is_an_assess_and_only_a_pure_instruction_is_declined(prompt):
    assert "still assess" in prompt and "refuse the wording asked for" in prompt
    assert "an instruction with no request to fly" in prompt


# --- E15: the banned wording appears on one line, and nowhere else -------------------------------------------------------------------------------


def banned_in(text: str) -> list[str]:
    found = [phrase for phrase in BANNED_PHRASES if phrase in text.lower()]
    return found + (["GO"] if re.search(r"\bGO\b", text) else [])


def test_the_banned_wording_is_on_exactly_one_line_the_rule_line(prompt):
    holding = [line for line in lines_of(prompt) if banned_in(line)]
    assert holding == [banned_line()]
    assert holding[0].startswith(BANNED_LINE_START)
    assert sorted(banned_in(holding[0])) == sorted([*BANNED_PHRASES, "GO"])


@pytest.mark.parametrize(("now", "tz"), [p.values[:2] for p in TABLE])
def test_the_banned_wording_is_never_outside_the_rule_line_in_any_filled_in_prompt(now, tz):
    other = [line for line in lines_of(system_prompt(now, tz)) if line != banned_line()]
    assert banned_in("\n".join(other)) == []


def test_the_decline_example_wording_holds_none_of_the_banned_wording(prompt):
    example = re.search(r'For example: "(.+?)"', prompt, re.DOTALL)
    assert example and banned_in(example.group(1)) == []


def test_the_test_exempts_only_that_line():
    planted = system_prompt(utc(2026, 10, 31, 7, 30), "Europe/London") + '\nSay "all clear" when it is.'
    assert [line for line in lines_of(planted) if banned_in(line) and line != banned_line()] == ['Say "all clear" when it is.']


# --- I5: the path tag --------------------------------------------------------------------------------------------------------------------------------------


def test_the_prompt_tells_the_model_to_start_every_final_answer_with_one_path_line(prompt):
    assert PATH_TAGS == ("[path: assess]", "[path: explain]", "[path: clarify]", "[path: decline]")
    assert "Start every final answer with exactly one first line" in prompt
    for tag in PATH_TAGS:
        assert tag in prompt
    assert "Do not add the line to a message that calls tools" in prompt


# --- the snapshot -----------------------------------------------------------------------------------------------------------------------------------------------


def test_the_prompt_for_blue_mill_at_the_frozen_clock_matches_the_snapshot(update_golden, prompt):
    if update_golden:
        GOLDEN.write_text(prompt + "\n", encoding="utf-8")
    assert GOLDEN.read_text(encoding="utf-8") == prompt + "\n"


# --- through the graph: what the model is given ----------------------------------------------------------------------------------------------------


def test_the_model_is_given_the_prompt_as_its_system_message_and_the_prompt_is_not_kept_in_messages():
    model = FakeToolChatModel(script=[AIMessage("[path: decline]\nI can only help with whether to fly from this spot.")])
    state = ask(agent_graph(model), "What is a good pub?")
    [seen] = system_prompts(model)
    assert seen == system_prompt(utc(2026, 10, 31, 7, 30), "Europe/London")
    assert not any(isinstance(m, SystemMessage) for m in state["messages"])
    assert not any("THE FOUR PATHS" in str(m.content) for m in state["messages"])


def test_the_prompt_is_rebuilt_every_turn_from_that_turns_now():
    model = FakeToolChatModel(script=[AIMessage("[path: decline]\none."), AIMessage("[path: decline]\ntwo.")])
    graph = agent_graph(model)
    ask(graph, "first", context=blue_mill_context(utc(2026, 10, 31, 7, 30)))
    ask(graph, "second", context=blue_mill_context(utc(2026, 10, 31, 11, 45)))
    first, second = system_prompts(model)
    assert "The time now is 07:30 GMT." in first and "The time now is 11:45 GMT." in second
    assert first != second


def test_every_model_call_of_a_turn_gets_the_prompt_including_the_one_after_a_tool():
    model = FakeToolChatModel(script=[assess_call(), AIMessage("[path: assess]\nDone.")])
    state = ask(agent_graph(model))
    first, second = system_prompts(model)
    assert first == second
    assert not any(isinstance(m, SystemMessage) for m in state["messages"])


def test_the_prompt_uses_the_threads_time_zone():
    model = FakeToolChatModel(script=[AIMessage("[path: decline]\nok.")])
    ask(agent_graph(model), "Can I fly?", tz="America/New_York")
    [seen] = system_prompts(model)
    assert "03:30 EDT" in seen and "The pilot's time zone is America/New_York." in seen


def test_a_thread_without_a_stored_zone_gets_europe_london(prompt):
    assert "The pilot's time zone is Europe/London." in prompt
