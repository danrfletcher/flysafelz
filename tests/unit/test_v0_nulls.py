"""E5 (5a PR-9.F1): every registered evaluator over one v0 run. The G14 null list returns None; the safety and correctness evaluators return a number; the aggregates read n/a."""

import pytest
from agent_eval.factories import answer, example, row, seg, t

from evals.agent.evaluators.aggregate import EVALUATORS, ScoredRun, evaluate_runs

WINDOW = (t(9), t(11))
NULL_FOR_V0 = ("path_accuracy", "required_tools", "forbidden_tools", "step_cap", "guard_pass", "decision_first", "cited", "recheck_stated")
NUMBERED_FOR_V0 = (
    "harmful_missed_blocker",
    "hard_restriction_recall",
    "verdict_match",
    "approaching_limits_match",
    "window_iou",
    "notam_action_match",
    "noise_rate",
    "no_verdict_beyond_day3",
    "unverified_on_dropped_source",
    "window_stated",
)
REFERENCE = {"required_tools": ["assess"], "forbidden_tools": ["get_weather"], "must_mention": ["H6217"]}


def v0_example(**options):
    return example(
        [row(t(9), t(11), "NO_GO", True)],
        window=WINDOW,
        actions={"H6217/26": "NO_FLY", "V30/26": "IGNORE"},
        drop=("pib",),
        now=t(7, 30),
        extra=REFERENCE,
        **options,
    )


def v0_answer():
    return answer([seg(t(9), t(11), "UNVERIFIED", True)], actions={"H6217/26": "NO_FLY", "V30/26": "IGNORE"}, kind="v0", text="10:00 to 12:00: UNVERIFIED. NOTAM source unavailable")


@pytest.fixture(scope="module")
def scores():
    [run] = evaluate_runs([ScoredRun(v0_example(), v0_answer())]).runs
    return run.feedback


def test_the_test_covers_every_registered_evaluator():
    assert set(NULL_FOR_V0) | set(NUMBERED_FOR_V0) | {"must_mention", "must_not_mention"} == set(EVALUATORS)


@pytest.mark.parametrize("key", NULL_FOR_V0)
def test_an_evaluator_in_the_g14_null_list_returns_none_for_a_v0_run(scores, key):
    assert scores[key] is None  # None, not 0 and not False


@pytest.mark.parametrize("key", NUMBERED_FOR_V0)
def test_a_safety_or_correctness_evaluator_returns_a_number_for_a_v0_run(scores, key):
    found = scores[key]
    assert found is not None and found["key"] == key
    assert isinstance(found["score"], (int, float)) and not isinstance(found["score"], bool)


def test_a_dropped_source_is_scored_on_v0_like_any_other_run(scores):
    assert scores["unverified_on_dropped_source"]["score"] == 1


def test_the_aggregates_skip_null_scores_so_a_nulled_evaluator_reads_n_a_not_0_percent():
    evaluation = evaluate_runs([ScoredRun(v0_example(id="a"), v0_answer(), rep) for rep in (1, 2, 3)])
    for key in NULL_FOR_V0:
        assert (evaluation.mean(key), evaluation.applied(key), evaluation.mean_text(key)) == (None, 0, "n/a")
    for key in NUMBERED_FOR_V0:
        assert evaluation.applied(key) == 3 and evaluation.mean_text(key) != "n/a"
    assert evaluation.mean_text("verdict_match") == "0%"  # this answer says UNVERIFIED where the sheet says NO_GO: a real 0%, which is not the same as n/a


def test_the_path_matrix_leaves_v0_out_since_it_routes_nothing():
    evaluation = evaluate_runs([ScoredRun(v0_example(), v0_answer())])
    assert sum(sum(cells.values()) for cells in evaluation.path_confusion.values()) == 0


def test_must_mention_does_not_score_a_compound_v0_example_and_does_score_a_plain_one():
    plain = evaluate_runs([ScoredRun(v0_example(), v0_answer())]).runs[0].feedback
    compound = evaluate_runs([ScoredRun(v0_example(compound=True), v0_answer())]).runs[0].feedback
    assert plain["must_mention"] is not None
    assert compound["must_mention"] is None and compound["must_not_mention"] is None
    assert compound["verdict_match"] is not None and compound["notam_action_match"] is not None
