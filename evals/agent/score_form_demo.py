"""Scores canned runs for four dev examples and prints each evaluator's value or None (5a PR-6.F1 container check, GP3 deterministic part).

    uv run python -m evals.agent.score_form_demo
    uv run python -m evals.agent.score_form_demo --mentions     the E10 mention-normaliser cases instead

The examples are dev_14_town_moor (assess + decline), dev_16_buttermere (explain, two tools), dev_17_sherburn (clarify + explain, four forbidden tools) and
dev_27_sherburn (assess with the NOTAM source dropped). For each there are two canned runs: `good`, built from the example's own labels and reference lists, and
`bad`, an empty answer that called nothing, with the guard in its fallback. Reads the committed dataset only. No model, no network, no LangSmith.
"""

import json
import sys
from datetime import datetime

from evals.agent.evaluators.aggregate import ScoredRun, score_run
from evals.agent.evaluators.answer import ScoredAnswer, ToolCall
from evals.agent.evaluators.mentions import mentions, normalise_mention, reference_list
from evals.agent.evaluators.trajectory import SEARCH_GUIDANCE
from evals.agent.loader import load_dataset
from flysafelz.agent.answer import RECHECK_SENTENCE
from flysafelz.agent.state import DEFAULT_TZ
from flysafelz.timetext import clock

EXAMPLES = ("dev_14_town_moor", "dev_16_buttermere", "dev_17_sherburn", "dev_27_sherburn")
SOURCE_WORDS = "Weather, airspace and NOTAM data."
GUIDE = "CAP 1535"
MENTION_CASES = [
    ("1500", "base at 1,500 ft"),
    ("Newcastle CTA 2", "NEWCASTLE CTA2"),
    ("18:00", "until 6pm"),
    ("1500", "1\u00a0500"),
    ("1500", "1\t500"),
    ("1500000", "1,500,000"),
    ("1500", "1, 500"),
    ("NO-GO", "no go"),
    ("Burn", "Sherburn"),
]


def canned(example: dict, good: bool) -> ScoredAnswer:
    """The `good` run reads back the example's labels (UNVERIFIED where a source is dropped); the `bad` one is empty with no calls and a fallback guard."""
    reference = example["reference"]
    path = reference["path"]
    if not good:
        return ScoredAnswer(path, (), {}, "", (), "fallback")
    required = reference_list(example, "required_tools") or []
    drop = bool(example["fixtures"].get("drop"))
    segments = []
    for r in reference.get("verdict_by_segment") or []:
        if r.get("verdict") is not None:
            verdict = "UNVERIFIED" if drop and r["verdict"] != "NO_GO" else r["verdict"]
            segments.append({"start": r["start"], "end": r["end"], "verdict": verdict, "approaching_limits": bool(r.get("approaching_limits"))})
    parts = ["NO-GO."] if path == "assess" else []
    if path == "assess" and reference.get("window"):
        start, end = (datetime.fromisoformat(w) for w in reference["window"])
        parts += [f"Window {clock(start, DEFAULT_TZ)} to {clock(end, DEFAULT_TZ)}.", RECHECK_SENTENCE]
    parts += [*(reference_list(example, "must_mention") or []), SOURCE_WORDS]
    if SEARCH_GUIDANCE in required:
        parts.append(GUIDE)
    actions = {i: a for i, a in (reference.get("notam_actions") or {}).items() if a is not None}
    calls = [ToolCall(name, (GUIDE,) if name == SEARCH_GUIDANCE else ()) for name in required]
    return ScoredAnswer.from_dict({"path": path, "segments": segments, "notam_actions": actions, "text": " ".join(parts), "tool_calls": calls, "guard": "passed", "kind": "v1"})


def print_scores() -> None:
    records = {r["id"]: r for r in load_dataset("dev").records}
    for name in EXAMPLES:
        for label in ("good", "bad"):
            print(f"== {name} ({label})")
            for key, found in score_run(ScoredRun(records[name], canned(records[name], label == "good"))).feedback.items():
                print(f"{key}: {json.dumps(found) if found else 'None (does not apply)'}")


def print_mentions() -> None:
    for needle, text in MENTION_CASES:
        print(f"{needle!r} in {text!r}: {mentions(text, needle)}    ({normalise_mention(needle)!r} in {normalise_mention(text)!r})")


def main(argv: list[str] | None = None) -> None:
    (print_mentions if "--mentions" in (sys.argv[1:] if argv is None else argv) else print_scores)()


if __name__ == "__main__":
    main()
