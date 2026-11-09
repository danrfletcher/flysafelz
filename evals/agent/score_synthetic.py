"""Scores three synthetic answers against one dev sheet entry and prints each feedback dict (5a PR-5.F1 container check).

    uv run python -m evals.agent.score_synthetic

The entry is the dev example `blue_mill_today` (Blue Mill, 31 Oct 2026; PILOT_DISCRETION to civil dusk at 17:06, then NO_GO; J3857/26 is CONTACT_FIRST).
The three answers are built from the sheet's own rows, so the first is the sheet read back:
  correct       every segment as labelled and every NOTAM with its labelled action
  e12           the same, with PILOT_DISCRETION carried 14 minutes past dusk, to 17:20
  downgraded    the correct segments, with the CONTACT_FIRST NOTAM surfaced as AWARENESS (no dev example labels a NOTAM NO_FLY, so this is the nearest hard restriction)
Reads the committed dataset only. No model, no network, no LangSmith.
"""

import json
from datetime import timedelta

from evals.agent.evaluators.aggregate import ScoredRun, score_run
from evals.agent.evaluators.answer import ScoredAnswer
from evals.agent.evaluators.minutes import aware
from evals.agent.loader import load_dataset

EXAMPLE = "blue_mill_today"
DUSK = "2026-10-31T17:06:00Z"
HARD_NOTAM = "J3857/26"
CARRIED = timedelta(minutes=14)


def segments_of(example: dict, carry: bool) -> list[dict]:
    """The sheet's rows as answer segments. With `carry`, the PILOT_DISCRETION row that ends at dusk runs 14 minutes longer and the NO_GO row starts later."""
    dusk = aware(DUSK, "dusk")
    found = []
    for r in example["reference"]["verdict_by_segment"]:
        start, end = aware(r["start"], "row start"), aware(r["end"], "row end")
        if carry and end == dusk:
            end += CARRIED
        elif carry and start == dusk:
            start += CARRIED
        found.append({"start": start.isoformat(), "end": end.isoformat(), "verdict": r["verdict"], "approaching_limits": r["approaching_limits"]})
    return found


def answers(example: dict) -> dict[str, ScoredAnswer]:
    actions = dict(example["reference"]["notam_actions"])
    kept = {"path": "assess", "text": "synthetic"}
    return {
        "correct": ScoredAnswer.from_dict({**kept, "segments": segments_of(example, False), "notam_actions": actions}),
        "e12": ScoredAnswer.from_dict({**kept, "segments": segments_of(example, True), "notam_actions": actions}),
        "downgraded": ScoredAnswer.from_dict({**kept, "segments": segments_of(example, False), "notam_actions": {**actions, HARD_NOTAM: "AWARENESS"}}),
    }


def main() -> None:
    example = next(r for r in load_dataset("dev").records if r["id"] == EXAMPLE)
    for name, answer in answers(example).items():
        print(f"== {name}")
        for key, found in score_run(ScoredRun(example, answer)).feedback.items():
            print(f"{key}: {json.dumps(found) if found else 'None (does not apply)'}")


if __name__ == "__main__":
    main()
