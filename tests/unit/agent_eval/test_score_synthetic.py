"""The container check's script (5a PR-5.F1): three synthetic answers against the dev entry blue_mill_today, scoring 0 / 14 / 1 as the spec expects."""

import json

import pytest

from evals.agent import score_synthetic
from evals.agent.evaluators.aggregate import ScoredRun, score_run
from evals.agent.loader import load_dataset


@pytest.fixture(scope="module")
def scores():
    example = next(r for r in load_dataset("dev").records if r["id"] == score_synthetic.EXAMPLE)
    return {name: score_run(ScoredRun(example, answer)).feedback for name, answer in score_synthetic.answers(example).items()}


def test_the_correct_answer_has_no_missed_blocker_and_matches_every_labelled_minute(scores):
    assert scores["correct"]["harmful_missed_blocker"]["score"] == 0  # type: ignore[index]
    assert scores["correct"]["verdict_match"]["score"] == 1  # type: ignore[index]


def test_the_misplaced_edge_disagrees_on_exactly_fourteen_minutes_and_is_a_missed_blocker(scores):
    found = scores["e12"]
    assert "14 min disagree" in found["verdict_match"]["comment"]  # type: ignore[index]
    assert found["verdict_match"]["score"] == pytest.approx((660 - 14) / 660)  # type: ignore[index]
    assert found["harmful_missed_blocker"]["score"] == 1 and "17:06–17:20Z" in found["harmful_missed_blocker"]["comment"]  # type: ignore[index]


def test_the_downgraded_hard_restriction_is_a_missed_blocker_and_a_recall_miss(scores):
    found = scores["downgraded"]
    assert found["harmful_missed_blocker"]["score"] == 1 and found["hard_restriction_recall"]["score"] == 0  # type: ignore[index]


def test_main_prints_each_feedback_dict(capsys):
    score_synthetic.main()
    out = capsys.readouterr().out
    assert [line for line in out.splitlines() if line.startswith("== ")] == ["== correct", "== e12", "== downgraded"]
    printed = [line.split(": ", 1)[1] for line in out.splitlines() if line.startswith("harmful_missed_blocker: ")]
    assert [json.loads(p)["score"] for p in printed] == [0, 1, 1]
    assert "unverified_on_dropped_source: None (does not apply)" in out
