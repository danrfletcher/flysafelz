"""Running the evaluators over a run's examples and aggregating (5a PR-5.F1, G8, G9, E5).

    EVALUATORS                           {key: evaluator}, in the order the report lists them
    ScoredRun(example, answer, rep, errored)
    evaluate_runs(runs, evaluators=EVALUATORS) -> Evaluation

An errored example (PR-4.F1's flag: the harness failed, so there is no answer) is never passed to an evaluator. `evaluate_runs` skips it, lists its id in
`Evaluation.skipped`, and puts nothing in any score: it is not a 0. `score_run` asserts the flag is false and an answer is there, so a caller that bypasses
`evaluate_runs` and hands an errored run to it fails loudly.

Evaluation
    runs         one `RunScores` per scored run: example id, rep, {key: Feedback | None}
    skipped      the ids of the errored runs that were left out, in run order
    mean(key)    the mean score over the runs where the evaluator applied; None when it applied to none (never 0.0)
    applied(key) how many runs it applied to
    missed_blockers      `harmful_missed_blocker` as a count of runs with a miss, with the example ids named (once each, in order)
    path_confusion       the 5 x (5 + unknown) path matrix over every scored run
    action_confusion     the NotamAction x NotamAction matrix over the NOTAMs of every scored assess run with sheet labels
"""

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from evals.agent.evaluators.answer import ScoredAnswer
from evals.agent.evaluators.blocker import MISSED_BLOCKER_KEY, RECALL_KEY, hard_restriction_recall, harmful_missed_blocker
from evals.agent.evaluators.labels import Evaluator, Feedback, example_id
from evals.agent.evaluators.paths import ACTION_KEY, PATH_KEY, action_confusion, action_pairs, notam_action_match, path_accuracy, path_confusion, path_pair
from evals.agent.evaluators.safety import DROPPED_KEY, HORIZON_KEY, no_verdict_beyond_day3, unverified_on_dropped_source
from evals.agent.evaluators.verdict import IOU_KEY, LIMITS_KEY, VERDICT_KEY, approaching_limits_match, verdict_match, window_iou

EVALUATORS: dict[str, Evaluator] = {
    MISSED_BLOCKER_KEY: harmful_missed_blocker,
    RECALL_KEY: hard_restriction_recall,
    PATH_KEY: path_accuracy,
    VERDICT_KEY: verdict_match,
    LIMITS_KEY: approaching_limits_match,
    IOU_KEY: window_iou,
    ACTION_KEY: notam_action_match,
    HORIZON_KEY: no_verdict_beyond_day3,
    DROPPED_KEY: unverified_on_dropped_source,
}


@dataclass(frozen=True)
class ScoredRun:
    """One run to score: the loader's example, the answer (None for an errored run) and the rep."""

    example: Mapping[str, Any]
    answer: ScoredAnswer | None
    rep: int = 1
    errored: bool = False


@dataclass(frozen=True)
class RunScores:
    example_id: str
    rep: int
    feedback: dict[str, Feedback | None]


@dataclass(frozen=True)
class MissedBlockers:
    count: int  # runs with a harmful miss
    examples: tuple[str, ...]  # their example ids, each once, in run order


def score_run(run: ScoredRun, evaluators: Mapping[str, Callable[[Mapping[str, Any], ScoredAnswer], Feedback | None]] = EVALUATORS) -> RunScores:
    """Every evaluator on one run. The run must not be errored and must have an answer: this asserts both."""
    assert not run.errored, f"{example_id(run.example)}: an errored run is never passed to an evaluator"
    assert run.answer is not None, f"{example_id(run.example)}: a scored run has an answer"
    return RunScores(example_id(run.example), run.rep, {key: evaluator(run.example, run.answer) for key, evaluator in evaluators.items()})


@dataclass
class Evaluation:
    runs: list[RunScores] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    path_confusion: dict[str, dict[str, int]] = field(default_factory=dict)
    action_confusion: dict[str, dict[str, int]] = field(default_factory=dict)

    def scores(self, key: str) -> list[float]:
        return [found["score"] for scores in self.runs if (found := scores.feedback.get(key)) is not None]

    def applied(self, key: str) -> int:
        return len(self.scores(key))

    def mean(self, key: str) -> float | None:
        found = self.scores(key)
        return sum(found) / len(found) if found else None

    @property
    def missed_blockers(self) -> MissedBlockers:
        named = [s.example_id for s in self.runs if (found := s.feedback.get(MISSED_BLOCKER_KEY)) is not None and found["score"]]
        return MissedBlockers(len(named), tuple(dict.fromkeys(named)))


def evaluate_runs(runs: Iterable[ScoredRun], evaluators: Mapping[str, Callable[[Mapping[str, Any], ScoredAnswer], Feedback | None]] = EVALUATORS) -> Evaluation:
    """Score every run that was not errored and aggregate. See the module text."""
    result = Evaluation()
    path_pairs: list[tuple[str, str]] = []
    action_pair_list = []
    for run in runs:
        if run.errored:
            result.skipped.append(example_id(run.example))
            continue
        result.runs.append(score_run(run, evaluators))
        assert run.answer is not None
        path_pairs.append(path_pair(run.example, run.answer))
        action_pair_list += action_pairs(run.example, run.answer) or []
    result.path_confusion = path_confusion(path_pairs)
    result.action_confusion = action_confusion(action_pair_list)
    return result


def keys(evaluators: Sequence[str] | None = None) -> list[str]:
    """The evaluator keys, in report order (or the given ones that exist)."""
    return list(EVALUATORS) if evaluators is None else [k for k in evaluators if k in EVALUATORS]
