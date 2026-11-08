"""Minute-weighted verdict match, approaching-limits match and window IoU (5a PR-5.F1, G9, E12, E14).

    verdict_match(example, answer)            minutes where the answer's verdict equals the label's, over the labelled VERDICT-tier minutes
    approaching_limits_match(example, answer) the same for `approaching_limits`: reported on its own, never folded into the verdict score
    window_iou(example, answer)               mean over the verdicts present of |predicted ∩ labelled| / |predicted ∪ labelled|, in minutes
    verdict_ious(example, answer)             the per-verdict IoU behind `window_iou`: {Verdict: float}, or None where it gives no score

All three return None for a non-assess example, for one with no finalised sheet entry, and when the window holds no labelled VERDICT-tier minute (a zero-minute
window, or all OUTLOOK): see labels.py. OUTLOOK and TOO_FAR rows are left out, and rows and segments are clipped to `reference.window` (E14). A minute the answer does
not cover is a disagreement; where segments overlap the worst verdict counts (minutes.py). The score domain is the labelled minutes, so an answer that gives a verdict
over an OUTLOOK row or outside the window is not penalised here.

IoU is taken per verdict value over the same minutes. A verdict that neither side has is skipped, not scored 1.0, and the mean is over the verdicts that either side has.
E12: a label of PILOT_DISCRETION to dusk 17:06 then NO_GO, and an answer that extends PILOT_DISCRETION to 17:20, disagree on exactly 14 minutes: the match is (N-14)/N and
the NO_GO IoU is below 1.

The comment of a score names the number of minutes that disagree and the first run of them, with what the answer said there and what the sheet says.
"""

from collections.abc import Mapping
from typing import Any

from evals.agent.evaluators.answer import ScoredAnswer
from evals.agent.evaluators.labels import Feedback, Labels, answer_minutes, feedback, sheet_labels
from evals.agent.evaluators.minutes import Minutes, range_text, runs
from flysafelz.contracts import Verdict

VERDICT_KEY = "verdict_match"
LIMITS_KEY = "approaching_limits_match"
IOU_KEY = "window_iou"
NO_ANSWER = "no answer"


def _scored(example: Mapping[str, Any], answer: ScoredAnswer) -> tuple[Labels, Minutes] | None:
    labels = sheet_labels(example)
    if labels is None or not labels.verdicts:
        return None
    return labels, answer_minutes(answer, labels.window)


def _disagreement(labels: Labels, said: Minutes, wrong: list[int], shown: Any) -> str:
    first = runs(wrong)[0]
    heard = {shown(said[m]) if m in said else NO_ANSWER for m in first}
    meant = {shown((labels.verdicts[m], labels.limits[m])) for m in first}
    return f"{len(wrong)} min disagree, first at {range_text(first)} ({len(first)} min): answer {'/'.join(sorted(heard))}, sheet {'/'.join(sorted(meant))}"


def _match(key: str, example: Mapping[str, Any], answer: ScoredAnswer, same: Any, shown: Any, noun: str) -> Feedback | None:
    scored = _scored(example, answer)
    if scored is None:
        return None
    labels, said = scored
    total = len(labels.verdicts)
    wrong = sorted(m for m in labels.verdicts if m not in said or not same(said[m], (labels.verdicts[m], labels.limits[m])))
    comment = f"{labels.example_id}: {total - len(wrong)} of {total} labelled minutes agree on the {noun}"
    comment += f"; {_disagreement(labels, said, wrong, shown)}" if wrong else ""
    return feedback(key, (total - len(wrong)) / total, comment)


def verdict_match(example: Mapping[str, Any], answer: ScoredAnswer) -> Feedback | None:
    return _match(VERDICT_KEY, example, answer, lambda a, b: a[0] is b[0], lambda held: held[0].value, "verdict")


def approaching_limits_match(example: Mapping[str, Any], answer: ScoredAnswer) -> Feedback | None:
    return _match(LIMITS_KEY, example, answer, lambda a, b: a[1] == b[1], lambda held: f"approaching_limits {str(held[1]).lower()}", "approaching-limits flag")


def verdict_ious(example: Mapping[str, Any], answer: ScoredAnswer) -> dict[Verdict, float] | None:
    """IoU per verdict value that either side has over the labelled minutes (see the module text), or None where `window_iou` gives no score."""
    scored = _scored(example, answer)
    if scored is None:
        return None
    labels, said = scored
    found: dict[Verdict, float] = {}
    for verdict in Verdict:
        labelled = {m for m, v in labels.verdicts.items() if v is verdict}
        predicted = {m for m in labels.verdicts if m in said and said[m][0] is verdict}
        union = labelled | predicted
        if union:
            found[verdict] = len(labelled & predicted) / len(union)
    return found


def window_iou(example: Mapping[str, Any], answer: ScoredAnswer) -> Feedback | None:
    per_verdict = verdict_ious(example, answer)
    if not per_verdict:
        return None
    mean = sum(per_verdict.values()) / len(per_verdict)
    parts = ", ".join(f"{verdict.value} {value:.3f}" for verdict, value in per_verdict.items())
    return feedback(IOU_KEY, mean, f"{example.get('id', 'example')}: IoU in minutes per verdict: {parts}; mean {mean:.3f} over {len(per_verdict)} verdict(s)")
