"""What the evaluators share: the feedback shape, and the sheet's labels of one example as minutes (5a PR-5.F1).

    Feedback                   {"key", "score", "comment"}: LangSmith's feedback shape (score is None only for a judge error, PR-7.F1). An evaluator returns one, or None when it does not apply to the example
    Evaluator                  (example, answer) -> Feedback | None
    sheet_labels(example)      the example's labels as `Labels`, or None when it is not an assess example or has no finalised sheet entry
    answer_minutes(answer, window)   the answer's segments as {minute: (verdict, approaching_limits)} inside the window

An `example` is a record from evals.agent.loader (id, inputs, fixtures, reference, metadata). The evaluators read:
    reference.path             the primary path (a compound example's extra parts are not read here)
    reference.window           [start, end], two aware times, for an assess example
    reference.verdict_by_segment, reference.tier, reference.notam_actions   the finalised sheet's labels, which the loader joins in for an assess example; an assess
                               example that lacks any of the three has no finalised sheet entry
    inputs.now, fixtures.drop  for the two safety checks

Labels (G9, E14): the sheet's rows and the answer's segments are both expanded to UTC minutes (minutes.py) and clipped to the window. A minute is a labelled
VERDICT-tier minute when a row with a verdict covers it and no `tier` row (OUTLOOK, TOO_FAR) does. Those minutes are the domain of every minute score: a window
with none (a zero-minute window, or a window that is all OUTLOOK) gives no score, never a division by zero.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol, TypedDict

from evals.agent.evaluators.answer import ScoredAnswer, action_of, clean_id
from evals.agent.evaluators.minutes import Minutes, excluded_minutes, expand, window_of
from flysafelz.contracts import NotamAction, Verdict

ASSESS = "assess"
SHEET_KEYS = ("verdict_by_segment", "tier", "notam_actions")


class Feedback(TypedDict):
    key: str
    score: float | None  # None only for a judge error (PR-7.F1): a null with a comment, never a fail. The code evaluators always give a number
    comment: str


class Evaluator(Protocol):
    def __call__(self, example: Mapping[str, Any], answer: ScoredAnswer) -> Feedback | None: ...


def feedback(key: str, score: float, comment: str) -> Feedback:
    return {"key": key, "score": score, "comment": comment}


def example_id(example: Mapping[str, Any]) -> str:
    return str(example.get("id", "example"))


def primary_path(example: Mapping[str, Any]) -> str:
    """The example's primary path, `reference.path`. Raises ValueError, naming the example, when there is none."""
    path = (example.get("reference") or {}).get("path")
    if not isinstance(path, str) or not path.strip():
        raise ValueError(f"example {example_id(example)}: reference.path is missing")
    return path.strip()


@dataclass(frozen=True)
class Labels:
    example_id: str
    window: range  # the window's minutes (empty for a zero-minute window)
    verdicts: dict[int, Verdict]  # labelled VERDICT-tier minutes inside the window
    limits: dict[int, bool]  # their approaching_limits labels
    notam_actions: dict[str, NotamAction]  # NOTAM id (whitespace removed) -> the sheet's action; a null label is IGNORE


def sheet_labels(example: Mapping[str, Any]) -> Labels | None:
    """The example's sheet labels as minutes, or None for a non-assess example and for an assess example with no finalised sheet entry."""
    reference = example.get("reference") or {}
    if primary_path(example) != ASSESS or any(key not in reference for key in SHEET_KEYS):
        return None
    name = example_id(example)
    try:
        window = window_of(reference.get("window"), "reference.window")
        rows = reference["verdict_by_segment"] or []
        tiers = reference["tier"] or []
        labelled = expand(rows, window, "sheet row")
        left_out = excluded_minutes(tiers, window)
    except ValueError as error:
        raise type(error)(f"example {name}: {error}") from None
    verdicts = {minute: held[0] for minute, held in labelled.items() if minute not in left_out}
    limits = {minute: labelled[minute][1] for minute in verdicts}
    actions = {clean_id(i): action_of(a if a is not None else NotamAction.IGNORE.value, f"example {name}, NOTAM {i}") for i, a in (reference["notam_actions"] or {}).items()}
    return Labels(name, window, verdicts, limits, actions)


def answer_minutes(answer: ScoredAnswer, window: range) -> Minutes:
    """The answer's segments as {minute: (verdict, approaching_limits)}, clipped to the window; overlaps take the worst verdict, gaps are absent."""
    return expand(answer.segments, window, "answer segment")
