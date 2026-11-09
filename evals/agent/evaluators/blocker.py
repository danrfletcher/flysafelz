"""The two safety scores for NOTAMs and minutes (5a PR-5.F1, G8): `harmful_missed_blocker` and hard-restriction recall.

    harmful_missed_blocker(example, answer) -> Feedback | None
    hard_restriction_recall(example, answer) -> Feedback | None
    unlisted_ids(labels, answer)             the NOTAM ids the answer surfaced that the sheet does not list

harmful_missed_blocker. Score 1 when the answer has a harmful miss, else 0; the run's count is the number of examples scoring 1 (aggregate.py names them). Two triggers:
  * a minute where the answer says PILOT_DISCRETION and the sheet labels NO_GO or UNVERIFIED (labelled VERDICT-tier minutes inside the window only);
  * a NOTAM the sheet labels NO_FLY or CONTACT_FIRST that the answer surfaces with a weaker action. Weaker is ranked with `evals.reader_eval.downgrade.SEVERITY`
    (strictest first: NO_FLY, UNREAD_OVER_LZ, CHECK_SUP, CONTACT_FIRST, AVOID_SECTOR, AWARENESS, AWARENESS_MAP, IGNORE), so NO_FLY surfaced as CONTACT_FIRST is
    weaker, CONTACT_FIRST surfaced as NO_FLY or CHECK_SUP is not. A labelled NOTAM missing from the answer counts as surfaced with IGNORE.
The comment names the example, the first offending minute range (with the count of offending minutes) and every offending NOTAM id.

Hard-restriction recall = NOTAMs labelled NO_FLY or CONTACT_FIRST that the answer surfaces with that same action, over NOTAMs so labelled. "With that action" is exact:
NO_FLY surfaced as NO_FLY, CONTACT_FIRST as CONTACT_FIRST; a stricter action is not harmful (missed-blocker does not fire) but is not a hit here either. None when
no NOTAM is so labelled: never 1.0 or 0.0.

Both return None for a non-assess example and for one with no finalised sheet entry. A NOTAM id in the answer that the sheet does not list is left out of the score
and named in the comment.
"""

from collections.abc import Mapping
from typing import Any

from evals.agent.evaluators.answer import ScoredAnswer
from evals.agent.evaluators.labels import Feedback, Labels, answer_minutes, feedback, sheet_labels
from evals.agent.evaluators.minutes import range_text, runs
from evals.reader_eval.downgrade import severity_rank
from flysafelz.contracts import NotamAction, Verdict

MISSED_BLOCKER_KEY = "harmful_missed_blocker"
RECALL_KEY = "hard_restriction_recall"
HARD_ACTIONS = (NotamAction.NO_FLY, NotamAction.CONTACT_FIRST)
BLOCKING_LABELS = (Verdict.NO_GO, Verdict.UNVERIFIED)


def unlisted_ids(labels: Labels, answer: ScoredAnswer) -> list[str]:
    """The NOTAM ids the answer surfaced that the sheet's `notam_actions` does not list, sorted."""
    return sorted(i for i in answer.notam_actions if i not in labels.notam_actions)


def surfaced(answer: ScoredAnswer, notam_id: str) -> NotamAction:
    """The action the answer gave a NOTAM; IGNORE when it does not list it."""
    return answer.notam_actions.get(notam_id, NotamAction.IGNORE)


def _extras(labels: Labels, answer: ScoredAnswer) -> str:
    extra = unlisted_ids(labels, answer)
    return f" Left out, not in the sheet's notam_actions: {', '.join(extra)}." if extra else ""


def _offending_minutes(labels: Labels, answer: ScoredAnswer) -> list[int]:
    said = answer_minutes(answer, labels.window)
    return sorted(m for m, label in labels.verdicts.items() if label in BLOCKING_LABELS and said.get(m, (None, False))[0] is Verdict.PILOT_DISCRETION)


def _downgraded(labels: Labels, answer: ScoredAnswer) -> list[tuple[str, NotamAction, NotamAction]]:
    """(id, labelled, surfaced) for each hard-labelled NOTAM the answer surfaces with a weaker action (or not at all), sorted by id."""
    found = []
    for notam_id, label in sorted(labels.notam_actions.items()):
        shown = surfaced(answer, notam_id)
        if label in HARD_ACTIONS and severity_rank(shown) > severity_rank(label):
            found.append((notam_id, label, shown))
    return found


def harmful_missed_blocker(example: Mapping[str, Any], answer: ScoredAnswer) -> Feedback | None:
    labels = sheet_labels(example)
    if labels is None:
        return None
    minutes, notams = _offending_minutes(labels, answer), _downgraded(labels, answer)
    if not minutes and not notams:
        return feedback(MISSED_BLOCKER_KEY, 0, f"{labels.example_id}: no missed blocker.{_extras(labels, answer)}")
    parts = []
    if minutes:
        first = runs(minutes)[0]
        total = "" if len(first) == len(minutes) else f"; {len(minutes)} min in all"
        parts.append(f"PILOT_DISCRETION where the sheet says NO_GO or UNVERIFIED, first at {range_text(first)} ({len(first)} min{total})")
    if notams:
        parts.append("NOTAM " + "; NOTAM ".join(f"{i} labelled {label.value}, surfaced as {shown.value}" for i, label, shown in notams))
    return feedback(MISSED_BLOCKER_KEY, 1, f"{labels.example_id}: missed blocker: {'; '.join(parts)}.{_extras(labels, answer)}")


def hard_restriction_recall(example: Mapping[str, Any], answer: ScoredAnswer) -> Feedback | None:
    labels = sheet_labels(example)
    if labels is None:
        return None
    hard = {i: label for i, label in labels.notam_actions.items() if label in HARD_ACTIONS}
    if not hard:
        return None
    hits = sorted(i for i, label in hard.items() if surfaced(answer, i) is label)
    missed = [f"{i} labelled {hard[i].value}, surfaced as {surfaced(answer, i).value}" for i in sorted(hard) if i not in hits]
    comment = f"{labels.example_id}: {len(hits)} of {len(hard)} hard restrictions surfaced with their labelled action" + (f"; missed: {'; '.join(missed)}" if missed else "")
    return feedback(RECALL_KEY, len(hits) / len(hard), f"{comment}.{_extras(labels, answer)}")
