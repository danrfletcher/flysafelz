"""Path accuracy and the NOTAM action confusion (5a PR-5.F1, G9, I5, E9).

    path_accuracy(example, answer) -> Feedback | None  1 when the predicted path is the example's primary `reference.path`, else 0; applies to every path of a v1 run, and is None for a v0 run (v0 does no routing, PR-9.F1)
    path_pair(example, answer) -> (labelled, predicted)
    PATH_ROWS, PATH_COLUMNS                            the labelled paths (assess, explain, act, clarify, decline) and the predicted ones plus "unknown"
    path_confusion(pairs) -> {labelled: {predicted: n}}   5 rows, each with the 5 paths and an `unknown` column, keyed by path name

    notam_action_match(example, answer) -> Feedback | None   the share of the sheet's labelled NOTAMs whose action matches (an assess example with a finalised sheet entry)
    action_pairs(example, answer) -> [(labelled, predicted)] | None
    action_confusion(pairs) -> {labelled: {predicted: n}}     every NotamAction on both axes
    action_matches(labelled, predicted) -> bool        equal, or AWARENESS_MAP predicted for an AWARENESS label

Where the predicted path comes from (I5). It is the path the guard tagged, held in the root run's metadata `path`: `answer.path`, built with
`path_from_metadata` (answer.py). The answer text is never parsed. `unknown` (the tag was missing after the guard's retry) scores 0 and has its own column. For a compound
example (`compound: true, parts: [assess, explain]`) only the primary `path` is compared, so an answer that also explains scores 1 when its primary path is right; the
extra parts, and a missing `compound` key, make no difference here (PR-6 owns required_tools and must_mention).

NOTAM actions. The matrix is label x predicted over the NOTAMs in the sheet's `notam_actions`; a labelled NOTAM the answer does not list is predicted IGNORE, and an id in
the answer that the sheet does not list is left out and named in the comment. AWARENESS_MAP predicted against an AWARENESS label is a match; the reverse (AWARENESS
predicted against an AWARENESS_MAP label) and every other pair that differs is a mismatch. The axes are the members of `NotamAction`, so a member added later (CHECK_SUP
is one) is a label value with no change here.
"""

from collections.abc import Iterable, Mapping
from typing import Any

from evals.agent.evaluators.answer import PATHS, UNKNOWN_PATH, V0, ScoredAnswer
from evals.agent.evaluators.blocker import surfaced, unlisted_ids
from evals.agent.evaluators.labels import Feedback, example_id, feedback, primary_path, sheet_labels
from flysafelz.contracts import NotamAction

PATH_KEY = "path_accuracy"
ACTION_KEY = "notam_action_match"
PATH_ROWS = PATHS
PATH_COLUMNS = (*PATHS, UNKNOWN_PATH)
ACTIONS = tuple(NotamAction)


def path_pair(example: Mapping[str, Any], answer: ScoredAnswer) -> tuple[str, str]:
    """(the example's primary path, the predicted path). A primary path that is not one of PATHS raises ValueError, naming the example."""
    labelled = primary_path(example)
    if labelled not in PATH_ROWS:
        raise ValueError(f"example {example_id(example)}: reference.path {labelled!r} is not one of {', '.join(PATH_ROWS)}")
    return labelled, answer.path


def path_accuracy(example: Mapping[str, Any], answer: ScoredAnswer) -> Feedback | None:
    if answer.kind == V0:
        return None  # the baseline makes one call and routes nothing: it has no path to be right or wrong about (G14)
    labelled, predicted = path_pair(example, answer)
    hit = labelled == predicted
    outcome = "a hit" if hit else "a miss: the path tag was missing (unknown)" if predicted == UNKNOWN_PATH else "a miss"
    return feedback(PATH_KEY, 1 if hit else 0, f"{example_id(example)}: predicted path {predicted}, the example's path is {labelled}: {outcome}")


def path_confusion(pairs: Iterable[tuple[str, str]]) -> dict[str, dict[str, int]]:
    """The labelled x predicted counts. Every row and column is there, zero or not; a pair outside the axes raises ValueError."""
    matrix: dict[str, dict[str, int]] = {row: {column: 0 for column in PATH_COLUMNS} for row in PATH_ROWS}
    for labelled, predicted in pairs:
        if labelled not in matrix or predicted not in PATH_COLUMNS:
            raise ValueError(f"path pair ({labelled!r}, {predicted!r}) is outside the matrix: rows {', '.join(PATH_ROWS)}; columns {', '.join(PATH_COLUMNS)}")
        matrix[labelled][predicted] += 1
    return matrix


def action_matches(labelled: NotamAction, predicted: NotamAction) -> bool:
    """Equal, or AWARENESS_MAP predicted for an AWARENESS label (the map shows what the answer would have said: Dan's rule). Never the other way round."""
    return labelled is predicted or (labelled is NotamAction.AWARENESS and predicted is NotamAction.AWARENESS_MAP)


def action_pairs(example: Mapping[str, Any], answer: ScoredAnswer) -> list[tuple[NotamAction, NotamAction]] | None:
    """(labelled, predicted) for each NOTAM in the sheet's notam_actions, in id order; None when the example has no sheet labels or labels no NOTAM."""
    labels = sheet_labels(example)
    if labels is None or not labels.notam_actions:
        return None
    return [(label, surfaced(answer, notam_id)) for notam_id, label in sorted(labels.notam_actions.items())]


def action_confusion(pairs: Iterable[tuple[NotamAction, NotamAction]]) -> dict[str, dict[str, int]]:
    """The labelled x predicted counts over every NotamAction, keyed by the action's name."""
    matrix = {row.value: {column.value: 0 for column in ACTIONS} for row in ACTIONS}
    for labelled, predicted in pairs:
        matrix[NotamAction(labelled).value][NotamAction(predicted).value] += 1
    return matrix


def notam_action_match(example: Mapping[str, Any], answer: ScoredAnswer) -> Feedback | None:
    labels = sheet_labels(example)
    if labels is None or not labels.notam_actions:
        return None
    given = [(notam_id, label, surfaced(answer, notam_id)) for notam_id, label in sorted(labels.notam_actions.items())]
    wrong = [f"{i} labelled {label.value}, answered {predicted.value}" for i, label, predicted in given if not action_matches(label, predicted)]
    right = len(given) - len(wrong)
    comment = f"{labels.example_id}: {right} of {len(given)} labelled NOTAMs have the labelled action" + (f"; mismatches: {'; '.join(wrong)}" if wrong else "")
    extra = unlisted_ids(labels, answer)
    comment += f". Left out, not in the sheet's notam_actions: {', '.join(extra)}" if extra else ""
    return feedback(ACTION_KEY, right / len(given), comment)
