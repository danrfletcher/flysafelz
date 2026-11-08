"""The two safety checks the plan maps to this PR (5a PR-5.F1, E8, E11): no verdict beyond the horizon day, and UNVERIFIED on a dropped source.

    no_verdict_beyond_day3(example, answer, rules=None) -> Feedback | None
    unverified_on_dropped_source(example, answer) -> Feedback | None
    verdict_horizon_day(rules=None) -> int

no_verdict_beyond_day3 (E11). Score 1 passes, 0 fails. Every answer segment carries a verdict, so a segment fails when any part of it falls on a Europe/London day
greater than `verdict_horizon_day`. Days count from the day of the example's `inputs.now` (day 0), by London calendar date: the horizon day ends at London midnight, which
is 23:00Z in BST and 00:00Z in GMT, and a segment that ends exactly then is still inside it. The horizon is read from rules.yaml through `flysafelz.rules.load_rules`
(or the `rules` given), never written here, which is why the function's name is the plan's and its number is not. None for a non-assess example.

unverified_on_dropped_source (E8). For an example whose `fixtures.drop` names a source (weather, airspace or pib), the answer must not state a verdict from data it did
not have. It fails (score 0) when
  * a labelled VERDICT-tier minute of the window is not UNVERIFIED in the answer (a gap counts), or
  * the answer has UNVERIFIED segments but its text does not name every dropped source: `weather`, `airspace`, `notam` (the SourceName words; pib is the NOTAMs).
A NO_GO minute is the one exception: a NO_GO from a source that was not dropped outranks UNVERIFIED (assess works the same way), so a NO_GO answer passes on a minute the
sheet also labels NO_GO and fails on any other. Without it the correct answer to dev_10 (the sheet has NO_GO weather hours) would fail the check. None when `drop` is
empty, for a non-assess example, for one with no finalised sheet entry, and when the window has no labelled VERDICT-tier minute.
The text is the final answer; matching is case-insensitive, on the words only (code reads closed word lists, never what a sentence claims).
"""

from collections.abc import Mapping
from typing import Any

from evals.agent.evaluators.answer import ScoredAnswer
from evals.agent.evaluators.labels import ASSESS, Feedback, answer_minutes, example_id, feedback, primary_path, sheet_labels
from evals.agent.evaluators.minutes import aware, minute_span, range_text, runs
from flysafelz.contracts import Verdict
from flysafelz.daylight import london_day_date, london_day_utc
from flysafelz.rules import DEFAULT_PATH, Rules, load_rules
from flysafelz.verdict_sheet import DROP_SOURCES

HORIZON_KEY = "no_verdict_beyond_day3"
DROPPED_KEY = "unverified_on_dropped_source"
PLURAL = "s"


def verdict_horizon_day(rules: Rules | None = None) -> int:
    """The last London day (day 0 is the day of `now`) that can carry a verdict, from rules.yaml through load_rules."""
    return int((rules or load_rules(DEFAULT_PATH)).verdict_horizon_day.value)  # type: ignore[arg-type]  # a number rule


def no_verdict_beyond_day3(example: Mapping[str, Any], answer: ScoredAnswer, rules: Rules | None = None) -> Feedback | None:
    if primary_path(example) != ASSESS:
        return None
    horizon = verdict_horizon_day(rules)
    now = aware((example.get("inputs") or {}).get("now"), f"example {example_id(example)} inputs.now")
    first_late_moment = london_day_utc(london_day_date(now, horizon + 1))[0]
    late = [s for s in answer.segments if s.end > s.start and s.end > first_late_moment]
    if not late:
        return feedback(HORIZON_KEY, 1, f"{example_id(example)}: no segment carries a verdict beyond London day {horizon}")
    first = min(late, key=lambda s: s.start)
    shown = range_text(minute_span(first.start, first.end))
    return feedback(HORIZON_KEY, 0, f"{example_id(example)}: {len(late)} segment(s) carry a verdict beyond London day {horizon}, the first {first.verdict.value} at {shown}")


def _words(dropped: list[str], name: str) -> list[str]:
    unknown = [d for d in dropped if d not in DROP_SOURCES]
    if unknown:
        raise ValueError(f"example {name}: fixtures.drop names {', '.join(unknown)}, which is not one of {', '.join(DROP_SOURCES)}")
    return [DROP_SOURCES[d].value.removesuffix(PLURAL) for d in dropped]  # "notams" is matched as "notam", so "NOTAM data" and "NOTAMs" both name it


def _acceptable(said: Verdict | None, label: Verdict) -> bool:
    """UNVERIFIED always; NO_GO only where the sheet says NO_GO too (a source that was not dropped can still block). A gap (None) never."""
    return said is Verdict.UNVERIFIED or (said is Verdict.NO_GO and label is Verdict.NO_GO)


def unverified_on_dropped_source(example: Mapping[str, Any], answer: ScoredAnswer) -> Feedback | None:
    dropped = list((example.get("fixtures") or {}).get("drop") or [])
    name = example_id(example)
    if not dropped:
        return None
    words = _words(dropped, name)
    labels = sheet_labels(example)
    if labels is None or not labels.verdicts:
        return None
    said = answer_minutes(answer, labels.window)
    wrong = sorted(m for m, label in labels.verdicts.items() if not _acceptable(said[m][0] if m in said else None, label))
    problems = []
    if wrong:
        first = runs(wrong)[0]
        problems.append(
            f"{len(wrong)} min are not UNVERIFIED, first at {range_text(first)} (answer {said[first.start][0].value if first.start in said else 'no answer'}, sheet {labels.verdicts[first.start].value})"
        )
    unnamed = [d for d, word in zip(dropped, words, strict=True) if word not in answer.text.lower()]
    has_unverified = any(said[m][0] is Verdict.UNVERIFIED for m in labels.verdicts if m in said)
    if has_unverified and unnamed:
        problems.append(f"the UNVERIFIED answer does not name the dropped source ({', '.join(unnamed)})")
    if problems:
        return feedback(DROPPED_KEY, 0, f"{name}: dropped {', '.join(dropped)}: " + "; ".join(problems))
    return feedback(
        DROPPED_KEY, 1, f"{name}: dropped {', '.join(dropped)}: every labelled minute is UNVERIFIED (or NO_GO where the sheet says so)" + (" and the answer names the source" if has_unverified else "")
    )
