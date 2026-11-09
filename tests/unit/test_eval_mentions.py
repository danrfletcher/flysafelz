"""Mention matching (5a PR-6.F1, G11, E10): the normaliser table, its literal extensions, and the two evaluators over real dev examples."""

from pathlib import Path

import pytest
import yaml
from agent_eval.factories import answer, example
from eval_form_helpers import dev, dev_records

from evals.agent.evaluators.mentions import MUST_MENTION_KEY, MUST_NOT_MENTION_KEY, mentions, must_mention, must_not_mention, normalise_mention, reference_list

# --- E10, exactly as planned ---------------------------------------------------------------------------------------------------

PLANNED = [
    ("1500", "base at 1,500 ft", True),
    ("Newcastle CTA 2", "NEWCASTLE CTA2", True),
    ("18:00", "until 6pm", False),
]


@pytest.mark.parametrize(("needle", "text", "found"), PLANNED)
def test_e10_the_planned_table(needle, text, found):
    assert mentions(text, needle) is found


# --- E10 rule extensions: literal, not widened ---------------------------------------------------------------------------------

EXTENSIONS = [
    pytest.param("1\u00a0500", "1500", id="a no-break space is whitespace"),
    pytest.param("1\t500", "1500", id="a tab is whitespace"),
    pytest.param("1,500,000", "1500000", id="every comma between digits goes"),
    pytest.param("1, 500", "1,500", id="a comma not between digits stays, then the space goes"),
    pytest.param("NO-GO", "no-go", id="a hyphen is kept"),
    pytest.param("  Mixed\nCase\r\nText ", "mixedcasetext", id="lower-case, all whitespace"),
    pytest.param("A,B", "a,b", id="a comma between letters stays"),
    pytest.param("1 ,500", "1,500", id="a comma after a space is not between digits"),
]


@pytest.mark.parametrize(("raw", "normalised"), EXTENSIONS)
def test_e10_normalise_mention_table(raw, normalised):
    assert normalise_mention(raw) == normalised


def test_e10_the_no_break_space_makes_1500_match():
    assert mentions("base at 1\u00a0500 ft", "1500")


def test_e10_a_comma_then_space_does_not_match_1500():
    assert not mentions("base at 1, 500 ft", "1500")


def test_e10_a_hyphen_is_a_known_blind_spot_of_must_not_mention():
    assert not mentions("no go today", "NO-GO")
    ex = example(path="clarify", extra={"must_not_mention": ["NO-GO"]})
    assert must_not_mention(ex, answer(path="clarify", text="It is a no go today."))["score"] == 1  # type: ignore[index]


def test_e10_the_order_is_lower_case_then_digit_commas_then_whitespace():
    assert normalise_mention("1,\u00a0500") == "1,500"  # the comma is not between digits while the space is still there
    assert normalise_mention("1,500") == "1500"


def test_e10_the_match_is_a_substring_match_so_burn_is_in_sherburn():
    """Why a check on the word Burn is a judge note and not a mention entry."""
    assert mentions("Sherburn aerodrome is 3 NM away", "Burn")
    assert not mentions("Sherburn aerodrome", "Burn Road")


def test_an_empty_needle_would_match_everything_so_it_raises():
    with pytest.raises(ValueError, match="empty once normalised"):
        mentions("anything", "  \t")


# --- reference_list -------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("value", [None, []])
def test_a_null_or_empty_list_is_absent(value):
    assert reference_list({"reference": {"must_mention": value}}, "must_mention") is None


def test_an_absent_key_and_a_missing_reference_are_absent():
    assert reference_list({"reference": {}}, "must_mention") is None
    assert reference_list({"id": "x"}, "must_mention") is None


@pytest.mark.parametrize("value", ["H6217", ["ok", 3], {"a": 1}])
def test_a_list_that_is_not_a_list_of_strings_raises_naming_the_example(value):
    with pytest.raises(ValueError, match="example syn_1: reference.must_mention must be a list of strings"):
        reference_list({"id": "syn_1", "reference": {"must_mention": value}}, "must_mention")


# --- the evaluators: None on an absent list ------------------------------------------------------------------------------------


@pytest.mark.parametrize("path", ["assess", "explain", "clarify", "decline", "act"])
def test_both_return_none_when_the_example_has_no_list(path):
    ex, ans = example(path=path), answer(path=path, text="NO-GO. Newcastle CTA2 at 1,500 ft")
    assert must_mention(ex, ans) is None and must_not_mention(ex, ans) is None


def test_both_return_none_for_an_empty_list():
    ex = example(path="explain", extra={"must_mention": [], "must_not_mention": []})
    assert must_mention(ex, answer(path="explain")) is None and must_not_mention(ex, answer(path="explain")) is None


# --- the evaluators: scores ----------------------------------------------------------------------------------------------------


def test_must_mention_passes_when_every_entry_is_there_by_the_normalised_match():
    ex = dev("dev_15_west_sleekburn")
    got = must_mention(ex, answer(path="explain", text="Newcastle CTR and NEWCASTLE CTA2 start at 1,500 ft."))
    assert got and got["key"] == MUST_MENTION_KEY and got["score"] == 1 and "dev_15_west_sleekburn" in got["comment"]


def test_must_mention_names_the_entries_that_are_missing():
    ex = dev("dev_15_west_sleekburn")
    got = must_mention(ex, answer(path="explain", text="Newcastle CTR only."))
    assert got and got["score"] == 0 and "'Newcastle CTA 2'" in got["comment"] and "'1500'" in got["comment"] and "'Newcastle CTR'" not in got["comment"]


def test_the_planned_18_00_entry_is_missing_from_until_6pm():
    ex = example(path="explain", extra={"must_mention": ["18:00"]})
    assert must_mention(ex, answer(path="explain", text="Fly until 6pm."))["score"] == 0  # type: ignore[index]


@pytest.mark.parametrize("text", ["", "   ", "\n\t \u00a0"])
def test_an_empty_or_blank_answer_fails_must_mention_and_passes_must_not_mention(text):
    ex = example(path="clarify", extra={"must_mention": ["H6217"], "must_not_mention": ["NO-GO", "PILOT DISCRETION"]})
    assert must_mention(ex, answer(path="clarify", text=text))["score"] == 0  # type: ignore[index]
    assert must_not_mention(ex, answer(path="clarify", text=text))["score"] == 1  # type: ignore[index]


def test_must_not_mention_fails_on_a_listed_phrase_whatever_its_case_or_spacing():
    ex = dev("dev_07_buttermere")
    assert must_not_mention(ex, answer(path="clarify", text="Which day? I will say Pilot  Discretion later."))["score"] == 0  # type: ignore[index]
    got = must_not_mention(ex, answer(path="clarify", text="Which day do you mean, and where?"))
    assert got and got["key"] == MUST_NOT_MENTION_KEY and got["score"] == 1


def test_must_not_mention_names_every_phrase_found():
    got = must_not_mention(dev("dev_07_buttermere"), answer(path="clarify", text="NO-GO now, PILOT DISCRETION later."))
    assert got and got["score"] == 0 and "'NO-GO'" in got["comment"] and "'PILOT DISCRETION'" in got["comment"]


def test_a_notam_id_with_its_year_is_needed_when_the_entry_has_one():
    ex = dev("dev_04_alnwick_pastures")  # must_mention: H5915/26
    assert must_mention(ex, answer(text="Ignore H5915 for now"))["score"] == 0  # type: ignore[index]
    assert must_mention(ex, answer(text="Ignore h5915/26 for now"))["score"] == 1  # type: ignore[index]


def test_an_entry_that_is_blank_raises_naming_the_example_and_the_key():
    ex = example(path="explain", id="syn_9", extra={"must_mention": [" "]})
    with pytest.raises(ValueError, match="example syn_9: reference.must_mention: mention ' ' is empty"):
        must_mention(ex, answer(path="explain", text="anything"))


def test_both_evaluators_score_v0_runs():
    ex = example(path="explain", extra={"must_mention": ["1500"], "must_not_mention": ["NO-GO"]})
    v0 = answer(path="explain", text="base 1,500 ft", kind="v0")
    assert must_mention(ex, v0)["score"] == 1 and must_not_mention(ex, v0)["score"] == 1  # type: ignore[index]


# --- every dev example parses ---------------------------------------------------------------------------------------------------


def test_every_dev_example_with_a_list_gives_a_score_for_a_blank_answer():
    seen = 0
    for record in dev_records():
        blank = answer(path=record["reference"]["path"], text="")
        if record["reference"].get("must_mention"):
            seen += 1
            assert must_mention(record, blank)["score"] == 0  # type: ignore[index]
        if record["reference"].get("must_not_mention"):
            seen += 1
            assert must_not_mention(record, blank)["score"] == 1  # type: ignore[index]
    assert seen == 16  # 8 dev examples carry each list


# --- F7: only dev rows are read -------------------------------------------------------------------------------------------------


def test_f7_no_held_out_example_is_loaded():
    records = dev_records()
    assert len(records) == 29 and all(r["metadata"]["split"] == "dev" and not r["id"].startswith("ho_") for r in records)


def test_f7_none_of_the_pr_6_tests_names_a_held_out_id():
    here = Path(__file__).resolve().parent
    data = yaml.safe_load((here.parents[1] / "evals" / "agent" / "data" / "examples.yaml").read_text(encoding="utf-8"))
    held_out = [e["id"] for e in data if (e.get("metadata") or {}).get("split") == "held_out"]
    assert len(held_out) == 15
    files = [
        here / name for name in ("test_eval_mentions.py", "test_eval_noise.py", "test_eval_trajectory.py", "test_eval_form.py", "test_eval_nulls.py", "test_eval_form_demo.py", "eval_form_helpers.py")
    ]
    assert all(f.exists() for f in files), "a PR-6.F1 test file was not found"
    assert [f"{f.name}: {i}" for f in files for i in held_out if i in f.read_text(encoding="utf-8")] == []
