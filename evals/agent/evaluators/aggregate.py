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
    mean(key)    the mean score over the runs where the evaluator applied and gave a score; None when it did for none (never 0.0). A judge error (PR-7.F1) has no score
    mean_text(key)   the mean as a percentage ("83%"), or "n/a" when there is none: a v0 experiment's nulled evaluators read n/a, never 0% (PR-9.F1, E5)
    applied(key) how many runs it applied to
    missed_blockers      `harmful_missed_blocker` as a count of runs with a miss, with the example ids named (once each, in order)
    path_confusion       the 5 x (5 + unknown) path matrix over every scored v1 run (a v0 run does no routing, so it is not in the matrix)
    action_confusion     the NotamAction x NotamAction matrix over the NOTAMs of every scored assess run with sheet labels
"""

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from evals.agent.evaluators.answer import V0, ScoredAnswer
from evals.agent.evaluators.blocker import MISSED_BLOCKER_KEY, RECALL_KEY, hard_restriction_recall, harmful_missed_blocker
from evals.agent.evaluators.form import CITED_KEY, DECISION_KEY, GUARD_KEY, RECHECK_KEY, WINDOW_KEY, cited, decision_first, guard_pass, recheck_stated, window_stated
from evals.agent.evaluators.labels import Evaluator, Feedback, example_id
from evals.agent.evaluators.mentions import MUST_MENTION_KEY, MUST_NOT_MENTION_KEY, must_mention, must_not_mention
from evals.agent.evaluators.noise import NOISE_KEY, noise_rate
from evals.agent.evaluators.paths import ACTION_KEY, PATH_KEY, action_confusion, action_pairs, notam_action_match, path_accuracy, path_confusion, path_pair
from evals.agent.evaluators.safety import DROPPED_KEY, HORIZON_KEY, no_verdict_beyond_day3, unverified_on_dropped_source
from evals.agent.evaluators.trajectory import FORBIDDEN_KEY, REQUIRED_KEY, STEP_KEY, forbidden_tools, required_tools, step_cap_ok
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
    NOISE_KEY: noise_rate,
    REQUIRED_KEY: required_tools,
    FORBIDDEN_KEY: forbidden_tools,
    STEP_KEY: step_cap_ok,
    GUARD_KEY: guard_pass,
    DECISION_KEY: decision_first,
    CITED_KEY: cited,
    WINDOW_KEY: window_stated,
    RECHECK_KEY: recheck_stated,
    MUST_MENTION_KEY: must_mention,
    MUST_NOT_MENTION_KEY: must_not_mention,
}


NOT_APPLICABLE = "n/a"


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
        return [found["score"] for scores in self.runs if (found := scores.feedback.get(key)) is not None and found["score"] is not None]  # a judge error is a null, not a 0

    def applied(self, key: str) -> int:
        return len(self.scores(key))

    def mean(self, key: str) -> float | None:
        found = self.scores(key)
        return sum(found) / len(found) if found else None

    def mean_text(self, key: str) -> str:
        found = self.mean(key)
        return NOT_APPLICABLE if found is None else f"{found:.0%}"

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
        if run.answer.kind != V0:
            path_pairs.append(path_pair(run.example, run.answer))
        action_pair_list += action_pairs(run.example, run.answer) or []
    result.path_confusion = path_confusion(path_pairs)
    result.action_confusion = action_confusion(action_pair_list)
    return result


def keys(evaluators: Sequence[str] | None = None) -> list[str]:
    """The evaluator keys, in report order (or the given ones that exist)."""
    return list(EVALUATORS) if evaluators is None else [k for k in evaluators if k in EVALUATORS]
