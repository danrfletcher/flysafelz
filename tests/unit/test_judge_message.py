"""The judge's user message (5a PR-7.F1, G12 and ticket 4's I5): the time, the LZ name, the question, the final answer, and the judge note's line.

The fixtures are two committed DEV examples: dev_05_druridge_bay has a judge note and dev_01_blue_mill has none. No held-out example is built or judged (F7).
"""

from datetime import UTC, datetime

import pytest
from judge_helpers import dev_example
from pydantic import ValidationError

from evals.agent.evaluators.answer import ScoredAnswer
from evals.agent.judge import NOTE_PREFIX, delivered_text, judge_message, london_time, lz_name_from_case, note_text, place_text

WITH_NOTE = "dev_05_druridge_bay"
WITHOUT_NOTE = "dev_01_blue_mill"


def message_of(example_id: str, text: str = "NO-GO this morning.", **more):
    return judge_message(dev_example(example_id), ScoredAnswer(path="assess", text=text), **more)


# --- the time ------------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("utc", "shown"),
    [
        ("2026-10-31T07:30Z", "Saturday 31 October 2026, 07:30 (GMT)"),  # the spec's two examples
        ("2026-09-13T06:30Z", "Sunday 13 September 2026, 07:30 (BST)"),
        ("2026-10-25T00:30Z", "Sunday 25 October 2026, 01:30 (BST)"),  # clocks go back at 01:00 UTC: the first 01:30
        ("2026-10-25T01:30Z", "Sunday 25 October 2026, 01:30 (GMT)"),  # and the second
        ("2026-03-29T00:59Z", "Sunday 29 March 2026, 00:59 (GMT)"),  # clocks go forward at 01:00 UTC
        ("2026-03-29T01:00Z", "Sunday 29 March 2026, 02:00 (BST)"),
        ("2026-12-31T23:30Z", "Thursday 31 December 2026, 23:30 (GMT)"),
        ("2026-06-30T23:30Z", "Wednesday 1 July 2026, 00:30 (BST)"),  # the date and the weekday follow the London clock, not UTC
    ],
)
def test_the_time_is_shown_on_the_london_clock_with_the_weekday(utc, shown):
    assert london_time(datetime.fromisoformat(utc)) == shown


def test_a_time_without_a_zone_is_refused_not_guessed():
    with pytest.raises(ValueError, match="timezone-aware"):
        london_time(datetime(2026, 10, 31, 7, 30))


def test_the_time_does_not_depend_on_the_zone_the_instant_is_written_in():
    from zoneinfo import ZoneInfo

    assert london_time(datetime(2026, 10, 25, 1, 30, tzinfo=UTC)) == london_time(datetime(2026, 10, 24, 20, 30, tzinfo=ZoneInfo("America/Chicago")))


# --- the message ---------------------------------------------------------------------------------------------------------------


def test_a_message_without_a_note_is_exactly_the_four_parts():
    example = dev_example(WITHOUT_NOTE)
    assert "judge_notes" not in example["reference"]
    assert message_of(WITHOUT_NOTE) == (f"Current time: Tuesday 29 September 2026, 07:30 (BST)\nLocation: Blue Mill\nQuestion: {example['inputs']['question']}\n\nFinal answer:\nNO-GO this morning.")


def test_there_is_no_example_specific_line_when_the_example_has_no_note():
    assert "Example-specific" not in message_of(WITHOUT_NOTE)


def test_a_note_ends_the_message_with_exactly_the_one_line():
    example = dev_example(WITH_NOTE)
    note = example["reference"]["judge_notes"]
    text = message_of(WITH_NOTE, "The NOTAM ends at 18:00.")
    expected_line = f"Example-specific check (fold into whichever criterion it fits): {' '.join(note.split())}"
    assert text.endswith(f"\n\n{expected_line}")
    assert text.rsplit("\n", 1)[1] == expected_line  # one line, the last
    assert text.count("Example-specific check") == 1


def test_the_note_prefix_is_the_spec_text():
    assert NOTE_PREFIX == "Example-specific check (fold into whichever criterion it fits): "


@pytest.mark.parametrize(
    ("note", "line"),
    [
        ("plain note", "plain note"),
        ("two\nlines", "two lines"),
        ("three\r\nlines\n\nand a gap", "three lines and a gap"),
        ("  padded  \n", "padded"),
        ("tab\tinside\nand break", "tab\tinside and break"),
        ("quotes \"double\" and 'single'", "quotes \"double\" and 'single'"),
        ("accents: é ñ ü, Zoë, 日本語, ½ NM, 500 ft AGL – at most", "accents: é ñ ü, Zoë, 日本語, ½ NM, 500 ft AGL – at most"),
        ("separators\u2028here\u2029too", "separators here too"),
    ],
)
def test_a_note_becomes_one_line_and_is_otherwise_kept_as_written(note, line):
    example = {**dev_example(WITHOUT_NOTE), "reference": {"judge_notes": note}}
    text = judge_message(example, ScoredAnswer(text="Answer."))
    assert text.endswith(f"\n\n{NOTE_PREFIX}{line}")
    assert text.rsplit("\n", 1)[1] == f"{NOTE_PREFIX}{line}"


@pytest.mark.parametrize("note", [None, "", "   ", "\n\n", 5, ["x"]])
def test_an_empty_or_missing_note_adds_no_line(note):
    assert note_text(note) is None
    example = {**dev_example(WITHOUT_NOTE), "reference": {"judge_notes": note}}
    assert "Example-specific" not in judge_message(example, ScoredAnswer(text="Answer."))


def test_newlines_quotes_and_non_ascii_in_an_answer_are_kept_as_written():
    text = "First line.\n\nSecond: \"quoted\", it's Zoë's – 日本語 ½ NM.\n  indented line"
    assert message_of(WITHOUT_NOTE, text).endswith(f"Final answer:\n{text}")


def test_a_plain_string_answer_is_taken_as_the_text():
    assert judge_message(dev_example(WITHOUT_NOTE), "Just text.").endswith("Final answer:\nJust text.")


# --- the place -----------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("case", "name"),
    [
        ("blue_mill_2026-10-31", "Blue Mill"),  # the spec's example
        ("west_sleekburn_2026-07-22_5day", "West Sleekburn"),
        ("blue_mill_2026-09-29_injected", "Blue Mill"),
        ("town_moor", "Town Moor"),
        ("2026-10-31", None),
        ("", None),
        (None, None),
        (7, None),
    ],
)
def test_the_lz_name_is_built_from_the_case_folder(case, name):
    assert lz_name_from_case(case) == name


def test_the_run_context_has_no_lz_name_so_the_case_folder_is_used():
    """The PR says which: ticket 4's RunContext (now, case, reader, reader_cache, drop) carries no name."""
    from dataclasses import fields

    from flysafelz.agent.state import RunContext

    assert {f.name for f in fields(RunContext)} == {"now", "case", "reader", "reader_cache", "drop"}


def test_the_location_line_uses_the_case_name_by_default():
    assert "\nLocation: Druridge Bay\n" in message_of(WITH_NOTE)


def test_a_name_passed_in_wins_over_the_case_folder():
    assert "\nLocation: Blue Mill Field\n" in message_of(WITHOUT_NOTE, lz_name="Blue Mill Field")


def test_with_no_name_at_all_the_location_is_the_coordinates_to_four_places():
    example = {**dev_example(WITHOUT_NOTE), "fixtures": {}}
    example["inputs"] = {**example["inputs"], "lat": 53.802797, "lon": -0.086644}
    text = judge_message(example, "Answer.")
    assert "\nLocation: the launch zone at 53.8028, -0.0866\n" in text


def test_a_blank_name_falls_back_to_the_coordinates():
    assert place_text("  ", 55.60903936, -1.71282694) == "the launch zone at 55.6090, -1.7128"
    assert place_text(None, 51.5, 0.0) == "the launch zone at 51.5000, 0.0000"


# --- what the message never holds ----------------------------------------------------------------------------------------------


def test_the_message_holds_no_reference_label_except_the_judge_note():
    example = dev_example("dev_17_sherburn")  # has required tools, must_mention and must_not_mention
    reference = example["reference"]
    assert reference.get("required_tools") and reference.get("must_mention")
    text = judge_message(example, "Which day do you mean?")
    for label in ("window", "required_tools", "forbidden_tools", "must_mention", "must_not_mention", "verdict_by_segment", "notam_actions", "tier"):
        assert label not in text
    given = f"{example['inputs']['question']} {text.split(chr(10))[1]}"  # the question and the place are the pilot's own words, so a label may match them
    for value in (*reference["required_tools"], *reference["must_mention"], *reference["must_not_mention"]):
        if value not in given:
            assert value not in text


def test_an_assess_example_message_shows_no_window_or_verdict_rows():
    example = dev_example(WITHOUT_NOTE)
    text = judge_message(example, "Answer.")
    for value in example["reference"]["window"]:
        assert value not in text
    assert "verdict" not in text.lower().replace("answer", "")


def test_only_the_judge_note_is_read_from_the_reference():
    example = {**dev_example(WITH_NOTE)}
    example["reference"] = {"judge_notes": "the note", "must_mention": ["SECRET-MENTION"], "window": ["2026-01-01T00:00Z"]}
    text = judge_message(example, "Answer.")
    assert "the note" in text and "SECRET-MENTION" not in text and "2026-01-01" not in text


# --- I5: the answer as delivered -------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("raw", ["[path: assess]\nNO-GO this morning.", "[path: explain]\n\nNO-GO this morning.", "[Path : assess]\nNO-GO this morning."])
def test_a_raw_model_output_that_starts_with_the_path_tag_gives_a_message_with_no_tag(raw):
    text = judge_message(dev_example(WITHOUT_NOTE), ScoredAnswer(path="assess", text=raw))
    assert "[path:" not in text.lower() and "path :" not in text.lower()
    assert text.endswith("Final answer:\nNO-GO this morning.")


def test_text_with_no_tag_is_left_exactly_as_written():
    assert delivered_text(ScoredAnswer(text="Line one.\n\nLine two: path: not a tag.")) == "Line one.\n\nLine two: path: not a tag."


def test_a_phone_number_in_the_answer_never_reaches_the_judge():
    """A fiction-range number: the redactor masks it before the message is built."""
    text = judge_message(dev_example(WITHOUT_NOTE), ScoredAnswer(text="Call the site on 01632 960123."))
    assert "960123" not in text and "<PHONE_1>" in text


def test_a_phone_number_in_the_question_is_masked_too():
    example = dev_example(WITHOUT_NOTE)
    example = {**example, "inputs": {**example["inputs"], "question": "Ring 07700 900123 about Blue Mill?"}}
    assert "900123" not in judge_message(example, "Answer.")


def test_a_scored_answer_cannot_be_built_with_a_bad_kind():
    with pytest.raises((ValueError, ValidationError, TypeError)):
        ScoredAnswer(text="x", kind="v9")
