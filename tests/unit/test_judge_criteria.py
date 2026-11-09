"""The judge's criteria (5a PR-7.F1, G12/E5): the four criteria, which path gets which, the strict output schema and the one-line reason rule."""

import pytest
from judge_helpers import FakeJudgeModel, answer, canned, dev_example
from pydantic import ValidationError

from evals.agent.judge import CRITERIA, CRITERIA_BY_PATH, DEFINITIONS, SELF_PREFERENCE_LINE, Judge, JudgeOutput, criteria_for, feedback_key
from evals.agent.judge_config import JudgeConfigError

PATH_TABLE = {
    "assess": ("answers_the_question", "prioritised", "actionable"),
    "explain": ("answers_the_question", "prioritised", "actionable"),
    "clarify": ("answers_the_question",),
    "decline": ("answers_the_question", "good_decline"),
    "act": (),
    "unknown": (),
    "something_new": (),
    "": (),
}


def test_the_criteria_are_exactly_these_four_in_this_order():
    assert CRITERIA == ("answers_the_question", "prioritised", "actionable", "good_decline")
    assert set(DEFINITIONS) == set(CRITERIA)
    assert set(JudgeOutput.model_fields) == set(CRITERIA)  # the schema has no fifth field


@pytest.mark.parametrize("path", PATH_TABLE)
def test_the_path_table_gives_each_path_its_criteria(path):
    assert criteria_for(path) == PATH_TABLE[path]


@pytest.mark.parametrize("not_a_path", [None, 3, ["assess"], {"assess": 1}])
def test_a_reference_path_that_is_not_a_string_has_no_criteria(not_a_path):
    assert criteria_for(not_a_path) == ()


def test_the_table_lists_only_known_criteria_and_only_the_paths_that_have_any():
    assert set(CRITERIA_BY_PATH) == {"assess", "explain", "clarify", "decline"}
    assert all(set(names) <= set(CRITERIA) for names in CRITERIA_BY_PATH.values())


@pytest.mark.parametrize("path", PATH_TABLE)
@pytest.mark.parametrize("criterion", CRITERIA)
def test_every_path_and_criterion_pair_is_graded_or_null_never_a_fail(path, criterion):
    """E5, through Judge.judge: the model says FAIL for all four; only the criteria the path uses are scored, the others are None."""
    example = {**dev_example("dev_01_blue_mill"), "reference": {"path": path}}
    model = FakeJudgeModel(replies=[canned(**{name: (False, "no") for name in CRITERIA})])
    grades = Judge(model).judge(example, answer("Some prose."))
    found = grades[feedback_key(criterion)]
    if criterion in PATH_TABLE[path]:
        assert found == {"key": f"judge.{criterion}", "score": 0.0, "comment": "no"}
    else:
        assert found is None
    assert len(model.calls) == (1 if PATH_TABLE[path] else 0)  # a path with no criteria never reaches the model


def test_a_reply_that_grades_a_criterion_the_path_does_not_use_has_it_dropped():
    example = dev_example("dev_07_buttermere")  # clarify: answers_the_question only
    grades = Judge(FakeJudgeModel(replies=[canned()])).judge(example, answer("Which day do you mean?"))
    assert [key for key, found in grades.items() if found is not None] == ["judge.answers_the_question"]


# --- the strict schema ---------------------------------------------------------------------------------------------------------


def reply(**changes):
    grades = {name: {"passed": True, "reason": "ok"} for name in CRITERIA}
    grades.update(changes)
    return grades


def test_a_full_reply_is_accepted():
    assert JudgeOutput.model_validate(reply()).prioritised.passed is True


def test_a_fifth_criterion_is_rejected():
    with pytest.raises(ValidationError, match="extra_forbidden"):
        JudgeOutput.model_validate({**reply(), "politeness": {"passed": True, "reason": "ok"}})


@pytest.mark.parametrize("missing", CRITERIA)
def test_a_missing_criterion_is_rejected(missing):
    data = reply()
    del data[missing]
    with pytest.raises(ValidationError, match=missing):
        JudgeOutput.model_validate(data)


def test_an_extra_field_inside_a_criterion_is_rejected():
    with pytest.raises(ValidationError, match="extra_forbidden"):
        JudgeOutput.model_validate(reply(prioritised={"passed": True, "reason": "ok", "score": 5}))


@pytest.mark.parametrize("passed", ["yes", 1, 0, None, "true"])
def test_passed_must_be_a_real_bool(passed):
    with pytest.raises(ValidationError):
        JudgeOutput.model_validate(reply(actionable={"passed": passed, "reason": "ok"}))


@pytest.mark.parametrize("reason", ["", "   ", "\t", "first line\nsecond line", "one\rtwo", "a\u2028b", "a\x85b", "trailing\n"])
def test_the_reason_must_be_one_non_empty_line(reason):
    with pytest.raises(ValidationError):
        JudgeOutput.model_validate(reply(actionable={"passed": True, "reason": reason}))


@pytest.mark.parametrize("reason", ["Opens with the NOTAM that closes the zone.", 'Says "no" in Zoë\'s words: ñ, 日本', "a  b"])
def test_a_one_line_reason_is_kept_as_written(reason):
    assert JudgeOutput.model_validate(reply(actionable={"passed": False, "reason": reason})).actionable.reason == reason


def test_a_reply_that_breaks_the_schema_is_a_judge_error_for_each_applicable_criterion_not_a_fail():
    example = dev_example("dev_08_bamburgh")  # decline: answers_the_question + good_decline
    bad = canned(good_decline=(True, "line one\nline two"))
    grades = Judge(FakeJudgeModel(replies=[bad])).judge(example, answer("I can't help with that."))
    for key in ("judge.answers_the_question", "judge.good_decline"):
        assert grades[key]["score"] is None
        assert grades[key]["comment"].startswith("judge error: ")
    assert grades["judge.prioritised"] is None and grades["judge.actionable"] is None


def test_the_self_preference_sentence_names_the_risk_and_the_check():
    assert "same model family" in SELF_PREFERENCE_LINE
    assert "self-preference" in SELF_PREFERENCE_LINE
    assert "calibration" in SELF_PREFERENCE_LINE
    assert "\n" not in SELF_PREFERENCE_LINE


def test_judge_config_error_is_exported_for_the_run_command():
    assert issubclass(JudgeConfigError, ValueError)
