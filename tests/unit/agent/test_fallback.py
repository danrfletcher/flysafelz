"""PR-11.F1: `fallback_answer` (G15, FR11): built by code from this turn's tool results, never from a draft.

Table-driven over (did `assess` run this turn, the path the draft named), then the BST to GMT change of 25 Oct 2026 (E18 labels) through the assess tool's own text.
"""

import pytest
from assess_tool_helpers import assess_calm, render
from factories import utc
from guard_world import blue_mill_turn

from flysafelz.agent.guard import CANNOT_HELP, FALLBACK_HEADING, RAW_HEADING, ToolResult, fallback_answer
from flysafelz.agent.state import OverlayFeature, OverlayProperties, PointGeometry

DRAFT = "A DRAFT SENTENCE THE MODEL WROTE"


def results(*names: str) -> tuple[ToolResult, ...]:
    return tuple(r for r in blue_mill_turn().results if r.name in names)


ASSESS_RAN = results("get_weather", "get_airspace", "get_notams", "assess")
NO_ASSESS = results("get_weather", "get_airspace", "get_notams")


def assess_text() -> str:
    return next(r.text for r in ASSESS_RAN if r.name == "assess")


@pytest.mark.parametrize("path", ["assess", "explain", "clarify", "decline", None])
def test_with_assess_this_turn_every_path_gets_the_table_and_the_notams_as_published(path):
    answer = fallback_answer(ASSESS_RAN, path)
    assert answer.startswith(f"{FALLBACK_HEADING}:\n\n")
    assert assess_text() in answer  # the table exactly as the tool printed it, in the thread's zone
    notams = next(r for r in ASSESS_RAN if r.name == "get_notams").artifact.notams
    assert answer.count("NOTAMs as published:") == 1
    for item in notams:
        assert f"{item.id}: {item.e_text}" in answer


def test_the_heading_is_the_one_the_spec_gives():
    assert FALLBACK_HEADING == "The assistant's answer failed its checks; this is the raw assessment"


@pytest.mark.parametrize("path", ["explain", "assess", None])
def test_without_assess_an_explain_gets_the_raw_tool_text_and_the_notams_as_published(path):
    answer = fallback_answer(NO_ASSESS, path)
    assert answer.startswith(f"{RAW_HEADING}:\n\n")
    for result in NO_ASSESS:
        assert result.text in answer
    notams = next(r for r in NO_ASSESS if r.name == "get_notams")
    assert "H7171/26" in notams.text and "H7171/26" in answer


@pytest.mark.parametrize("path", ["clarify", "decline"])
def test_without_assess_a_clarify_or_decline_gets_the_one_fixed_line(path):
    assert fallback_answer(NO_ASSESS, path) == CANNOT_HELP == "I can only help with whether to fly at this spot; ask me about a time."


@pytest.mark.parametrize("path", ["assess", "explain", "clarify", "decline", None])
def test_a_turn_with_no_tool_text_gets_the_fixed_line(path):
    assert fallback_answer((), path) == CANNOT_HELP
    assert fallback_answer((ToolResult("get_weather", "   ", None),), path) == CANNOT_HELP


def test_a_tool_that_failed_with_text_is_still_shown_as_it_was_printed():
    failed = ToolResult("get_weather", "Weather source unavailable: check the Met Office.", None)
    assert fallback_answer((failed,), "explain") == f"{RAW_HEADING}:\n\n{failed.text}"


def test_nothing_of_a_draft_is_in_it_whatever_the_path():
    for path in ("assess", "explain", "clarify", "decline", None):
        for turn in (ASSESS_RAN, NO_ASSESS, ()):
            assert DRAFT not in fallback_answer(turn, path)


def test_an_overlay_notam_the_tool_did_not_list_is_added_as_published():
    feature = OverlayFeature(geometry=PointGeometry(coordinates=[-0.08, 53.8]), properties=OverlayProperties(kind="notam", id="Z9999/26", text="OVERLAY ONLY TEXT", action="show"))
    answer = fallback_answer(ASSESS_RAN, "assess", [feature])
    assert "Z9999/26: OVERLAY ONLY TEXT" in answer and answer.count("H7171/26:") == 1


def test_an_assess_text_that_is_not_an_assess_artifact_does_not_count_as_assess():
    """The check is on the artifact the tool wrote, not on the tool's name alone."""
    lookalike = ToolResult("assess", "Overall: GO", None)
    answer = fallback_answer((lookalike,), "explain")
    assert answer.startswith(f"{RAW_HEADING}:")


def test_the_table_is_in_the_threads_zone_across_the_25_oct_clock_change():
    """The clocks go back at 01:00Z on 25 Oct 2026: the tool's own labels read BST then GMT, and the fallback shows them as printed."""
    start, end = utc(23, day=24), utc(3, day=25)
    text, artifact = render(assess_calm(start, end), (start, end))
    answer = fallback_answer((ToolResult("assess", text, artifact),), "assess")
    assert "BST" in answer and "GMT" in answer
    assert "00:00\u201301:00 BST" in answer or "01:00 BST\u201301:00 GMT" in answer
    assert text in answer
