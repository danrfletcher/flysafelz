"""Shared setup for the PR-6.F1 evaluator tests: the committed dev examples, and a fake run that passes every noise, trajectory and form score.

dev(id)                          a deep copy of the dev example (a loader record), so a test may edit it
dev_records()                    every dev example, in file order
passing_answer(record, kind=)    a ScoredAnswer built from the example's own labels: the sheet's rows and actions, the required tools (search_guidance returns
                                 CAP 1535), a text with a verdict word first, the window, the re-check sentence, the must_mention entries and the source words
Only `split: dev` rows are read (F7): `load_dataset("dev")` never returns a held-out example.
"""

import copy
from datetime import datetime
from functools import cache
from typing import Any

from evals.agent.evaluators.answer import V1, ScoredAnswer, ScoredSegment, ToolCall
from evals.agent.evaluators.form import GUARD_PASSED
from evals.agent.evaluators.mentions import reference_list
from evals.agent.evaluators.trajectory import SEARCH_GUIDANCE
from evals.agent.loader import load_dataset
from flysafelz.agent.answer import RECHECK_SENTENCE
from flysafelz.agent.state import DEFAULT_TZ
from flysafelz.timetext import clock

DOCUMENT = "CAP 1535"
SOURCE_WORDS = "Weather, airspace and NOTAM data."


@cache
def _records() -> dict[str, dict[str, Any]]:
    return {record["id"]: record for record in load_dataset("dev").records}


def dev_records() -> list[dict[str, Any]]:
    return [copy.deepcopy(record) for record in _records().values()]


def dev(example_id: str) -> dict[str, Any]:
    return copy.deepcopy(_records()[example_id])


def sheet_segments(record: dict[str, Any], *, unverified: bool = False) -> list[ScoredSegment]:
    """The sheet's rows that carry a verdict as answer segments. With `unverified`, every verdict but NO_GO becomes UNVERIFIED (what a dropped source should give)."""
    found = []
    for row in record["reference"].get("verdict_by_segment") or []:
        if row.get("verdict") is None:
            continue
        verdict = "UNVERIFIED" if unverified and row["verdict"] != "NO_GO" else row["verdict"]
        found.append(ScoredSegment(datetime.fromisoformat(row["start"]), datetime.fromisoformat(row["end"]), verdict, bool(row.get("approaching_limits"))))  # type: ignore[arg-type]
    return found


def passing_answer(record: dict[str, Any], *, kind: str = V1, now_text: bool = True) -> ScoredAnswer:
    reference = record["reference"]
    path = reference["path"]
    parts = []
    if path == "assess":
        parts.append("NO-GO.")
    window = reference.get("window")
    if path == "assess" and window:
        start, end = datetime.fromisoformat(window[0]), datetime.fromisoformat(window[1])
        parts.append(f"Window {clock(start, DEFAULT_TZ)} to {clock(end, DEFAULT_TZ)}.")
        parts.append(RECHECK_SENTENCE)
    parts.extend(reference_list(record, "must_mention") or [])
    required = reference_list(record, "required_tools") or []
    if SEARCH_GUIDANCE in required:
        parts.append(DOCUMENT)
    parts.append(SOURCE_WORDS)
    calls = [ToolCall(name, (DOCUMENT,) if name == SEARCH_GUIDANCE else ()) for name in required]
    actions = {i: a for i, a in (reference.get("notam_actions") or {}).items() if a is not None}
    return ScoredAnswer(
        path,
        tuple(sheet_segments(record, unverified=bool(record["fixtures"].get("drop")))),
        actions,
        " ".join(parts),
        tuple(calls),
        GUARD_PASSED,
        kind,
    )
