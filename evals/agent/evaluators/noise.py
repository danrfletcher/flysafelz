"""`noise_rate` (5a PR-6.F1, G10): how many of the NOTAMs the sheet labels IGNORE the answer surfaced anyway.

    noise_rate(example, answer) -> Feedback | None

noise_rate = IGNORE-labelled NOTAMs surfaced / IGNORE-labelled NOTAMs, a rate from 0 (none of the noise shown) to 1 (all of it); lower is better. The labels are the
finalised sheet's `notam_actions`, read through `labels.sheet_labels` like every sheet score; a null label counts as IGNORE there, and a null `notam_actions` is no
NOTAMs. "Surfaced" is PR-5.F1's extraction, `blocker.surfaced`, the one hard-restriction recall uses: the action the answer gave the NOTAM, IGNORE when it does not list it.
So a NOTAM is surfaced when the answer gives it any action but IGNORE. NOTAMs labelled AWARENESS_MAP (or any other action) are not in the denominator.

None for a non-assess example, for one with no finalised sheet entry, and for one with no IGNORE label: never 0.0. It applies to both run kinds: v0 answers carry NOTAM actions too.
"""

from collections.abc import Mapping
from typing import Any

from evals.agent.evaluators.answer import ScoredAnswer
from evals.agent.evaluators.blocker import surfaced
from evals.agent.evaluators.labels import Feedback, feedback, sheet_labels
from flysafelz.contracts import NotamAction

NOISE_KEY = "noise_rate"


def noise_rate(example: Mapping[str, Any], answer: ScoredAnswer) -> Feedback | None:
    labels = sheet_labels(example)
    if labels is None:
        return None
    ignored = sorted(notam_id for notam_id, label in labels.notam_actions.items() if label is NotamAction.IGNORE)
    if not ignored:
        return None
    shown = [notam_id for notam_id in ignored if surfaced(answer, notam_id) is not NotamAction.IGNORE]
    comment = f"{labels.example_id}: {len(shown)} of {len(ignored)} IGNORE-labelled NOTAMs surfaced" + (f": {', '.join(shown)}" if shown else "")
    return feedback(NOISE_KEY, len(shown) / len(ignored), comment)
