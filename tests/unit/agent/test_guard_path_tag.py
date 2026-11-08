"""PR-11.F1 / I5: the guard reads the first line of a final message as the path tag, strips it, and saves the path for PR-12.F1.

Table-driven over the valid tags, the broken ones, the retry, and the tag-only failure that is delivered rather than replaced by the fallback.
"""

import pytest
from test_guard_flow import GOOD_BODY, OVERALL, flow, sent, update

from flysafelz.agent.guard import GuardFinding
from flysafelz.agent.guard_flow import PATH_FINDING, split_path
from flysafelz.agent.prompts import PATHS

NO_TOOL_BODY = "I can only help with whether to fly at this spot."


@pytest.mark.parametrize("path", PATHS)
def test_each_valid_tag_is_read_stripped_and_saved(path):
    assert split_path(f"[path: {path}]\nThe body.") == (path, "The body.")


@pytest.mark.parametrize(
    "written",
    [
        "The body, with no tag.",  # missing
        "[path: advise]\nThe body.",  # unknown value
        "path: assess\nThe body.",  # no brackets
        "(path: assess)\nThe body.",  # wrong brackets
        "[Path: assess]\nThe body.",  # wrong case
        "[path:assess]\nThe body.",  # no space
        "[path: assess] The body.",  # the body on the tag's line
        "The body.\n[path: assess]",  # not on the first line
        "\n[path: assess]\nThe body.",  # a blank line first
    ],
)
def test_a_broken_tag_has_no_path(written):
    path, body = split_path(written)
    assert path is None and "[path" not in body.lower()


def test_a_tag_on_line_two_is_removed_from_the_body_but_not_accepted():
    assert split_path("The body.\n[path: explain]\nMore.") == (None, "The body.\nMore.")


def test_a_valid_tag_never_reaches_the_body_even_when_the_model_repeats_it():
    assert split_path("[path: assess]\nThe body.\n[path: assess]") == ("assess", "The body.")


def test_a_good_answer_with_a_valid_tag_passes_with_the_path_saved():
    result, _, outcomes = flow(f"[path: assess]\n{GOOD_BODY}")
    assert outcomes[0].path == "assess" and update(result)["guard"] == "passed"


def test_no_rule_ties_a_path_to_its_tools_a_compound_clarify_after_tools_passes():
    body = f"{OVERALL}, flyable from 10:00 GMT to 16:00 GMT. Which day do you mean after that?"
    result, stub, outcomes = flow(f"[path: clarify]\n{body}")
    assert len(stub.requests) == 1 and update(result)["guard"] == "passed" and outcomes[0].path == "clarify"
    assert "Based on data as of" not in update(result)["answer"] and "Over caution limit" not in update(result)["answer"]  # nothing is added to a clarify


@pytest.mark.parametrize("bad", ["No tag at all.", "[path: advise]\nBad value.", "path: explain\nNo brackets.", "Text\n[path: explain]"])
def test_a_broken_tag_gets_one_retry_and_the_retry_with_a_tag_passes(bad):
    result, stub, outcomes = flow(bad, f"[path: decline]\n{NO_TOOL_BODY}")
    assert len(stub.requests) == 2 and update(result)["guard"] == "retried"
    checker = stub.requests[1].messages[-1]
    assert str(PATH_FINDING) in checker.content
    [attempt] = outcomes[0].attempts
    assert attempt.reason == "path_tag" and outcomes[0].path == "decline"


def test_a_tag_still_missing_after_the_retry_with_a_correct_answer_is_delivered_with_path_unknown():
    untagged = "I can only help with whether to fly at this spot; ask me about a time."
    result, stub, outcomes = flow(untagged, untagged)
    assert len(stub.requests) == 2
    assert update(result)["guard"] == "retried"  # the answer is the retry's; the tag was the only thing wrong
    assert sent(result).content == untagged and update(result)["answer"].startswith(untagged)
    [outcome] = outcomes
    assert outcome.path == "unknown"
    assert [a.reason for a in outcome.attempts] == ["path_tag", "path_tag"]  # a `guard_fail` with reason path_tag for PR-12.F1


def test_a_missing_tag_with_a_real_failure_is_the_fallback_as_before():
    result, _, outcomes = flow("NO-GO from 10:00 GMT.", "NO-GO from 10:00 GMT.")
    assert update(result)["guard"] == "fallback" and outcomes[0].path == "unknown"
    assert [a.reason for a in outcomes[0].attempts] == ["checks", "checks"]


def test_a_valid_tag_with_a_real_failure_twice_is_the_fallback_and_the_path_is_kept():
    _, _, outcomes = flow("[path: assess]\nNO-GO from 10:00 GMT.", "[path: assess]\nNO-GO from 10:00 GMT.")
    assert outcomes[0].result == "fallback" and outcomes[0].path == "assess"


def test_the_path_finding_is_a_guard_finding_in_plain_words():
    assert isinstance(PATH_FINDING, GuardFinding) and PATH_FINDING.check == "path"
    assert all(f"[path: {p}]" in PATH_FINDING for p in PATHS)
