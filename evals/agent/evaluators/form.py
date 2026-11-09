"""Form scores (5a PR-6.F1, G11, E5, E10): how the final answer is written and what code put in it.

    guard_pass(example, answer)         ticket 4's guard passed the first draft
    decision_first(example, answer)     the first sentence holds a verdict word
    cited_notams(example, answer)       every NOTAM-id `must_mention` entry is in the answer
    cited_guidance(example, answer)     a rules answer names a document `search_guidance` returned
    cited(example, answer)              both parts, composed
    window_stated(example, answer)      the answer states the assumed window
    recheck_stated(example, answer)     the code-built re-check line is there when the window starts after the current hour of `now`
    verdict_words()                     the verdict words `decision_first` looks for

Where the rest of G11 lives. "No verdict beyond day 3" and "UNVERIFIED on a dropped source" were built with the safety scores in PR-5.F1
(`safety.no_verdict_beyond_day3`, `safety.unverified_on_dropped_source`) and are registered with them in aggregate.EVALUATORS: they were not written twice. `must_mention`
and `must_not_mention` are in mentions.py.

Every score here returns None when it does not apply, never 0 or False (E5). The rules, per score (a path is the example's primary `reference.path`):
  guard_pass       v1 runs with a guard result, any path. The result is ticket 4's `guard` state field: `passed` scores 1; `retried` (the first draft failed a check and the
                   retry passed) and `fallback` score 0, like ticket 4's own `guard_fail` feedback, which is 0 only when the first draft passed. None for v0 and for a run with no guard result.
  decision_first   v1, the assess path, and a window with at least one VERDICT-tier minute. A window that is all OUTLOOK (e.g. Sat 10 Oct) has no verdict to lead with, so it
                   is None, and so is an assess example with no finalised sheet entry. The first sentence is the text up to the first full stop, question or exclamation
                   mark followed by a space, or the first line break. The words are built from `flysafelz.contracts.display_verdict` (NO-GO, UNVERIFIED, PILOT DISCRETION), so a
                   qualifier after them ("PILOT DISCRETION – APPROACHING LIMITS") still holds the word. Case is ignored. An empty answer fails.
  cited_*          v1. cited_notams applies when `must_mention` holds an entry shaped like a NOTAM id (a letter, two to four digits, an optional /yy: `H6217`, `H6449/26`, `V30/26`); cited_guidance when `search_guidance` is in
                   `required_tools`. `cited` is 1 when every part that applies passes and None when neither applies. Same scores as one combined check; the parts are separate
                   functions so the online eval job can import `cited_guidance` alone. Matching uses the mention normaliser ("CAP1535" is "CAP 1535"). A `search_guidance`
                   that returned no document, or was never called, leaves nothing to name, so cited_guidance fails.
  window_stated    the assess path with a `reference.window`, v1 and v0. Ticket 4's answer builder (answer.py) has no window line: it adds the re-check line and the limit lines
                   only. So this uses the fallback the spec names: the answer carries the window's start and end as Europe/London `HH:MM` (flysafelz.timetext.clock), through
                   the mention normaliser. Both must be there.
  recheck_stated   v1, the assess path with a `reference.window`. The line is ticket 4's own `answer.recheck_line`, called on the window start: when it gives a line (the window
                   starts after the current hour of `inputs.now` in Europe/London) the answer must carry it; when it gives None this is None. The line is not typed here.
"""

import re
from collections.abc import Mapping
from datetime import datetime
from types import SimpleNamespace
from typing import Any

from evals.agent.evaluators.answer import V1, ScoredAnswer
from evals.agent.evaluators.labels import ASSESS, Feedback, example_id, feedback, primary_path, sheet_labels
from evals.agent.evaluators.mentions import mentions, reference_list
from evals.agent.evaluators.minutes import aware, window_of
from evals.agent.evaluators.trajectory import SEARCH_GUIDANCE
from flysafelz.agent.answer import recheck_line
from flysafelz.agent.state import DEFAULT_TZ
from flysafelz.contracts import Verdict, display_verdict
from flysafelz.timetext import clock

GUARD_KEY = "guard_pass"
DECISION_KEY = "decision_first"
CITED_KEY = "cited"
CITED_NOTAMS_KEY = "cited_notams"
CITED_GUIDANCE_KEY = "cited_guidance"
WINDOW_KEY = "window_stated"
RECHECK_KEY = "recheck_stated"
GUARD_PASSED = "passed"
NOTAM_ID = re.compile(r"[A-Za-z]\d{2,4}(?:/\d{2})?")
_SENTENCE_END = re.compile(r"[.!?](?=\s)|\n")


def verdict_words() -> tuple[str, ...]:
    """The verdict words, from display_verdict with no qualifier: NO-GO, UNVERIFIED, PILOT DISCRETION."""
    return tuple(display_verdict(verdict, False, False) for verdict in Verdict)


def _word_pattern(word: str) -> re.Pattern[str]:
    return re.compile(r"\b" + r"\s+".join(re.escape(part) for part in word.split()) + r"\b", re.IGNORECASE)


_VERDICT_PATTERNS = tuple(_word_pattern(word) for word in verdict_words())


def first_sentence(text: str) -> str:
    """The text up to the first sentence end (a full stop, question or exclamation mark followed by a space) or line break; all of it when there is none."""
    stripped = text.strip()
    found = _SENTENCE_END.search(stripped)
    return stripped[: found.start()] if found else stripped


def _is_assess(example: Mapping[str, Any]) -> bool:
    return primary_path(example) == ASSESS


def guard_pass(example: Mapping[str, Any], answer: ScoredAnswer) -> Feedback | None:
    if answer.kind != V1 or answer.guard is None:
        return None
    if answer.guard == GUARD_PASSED:
        return feedback(GUARD_KEY, 1, f"{example_id(example)}: the guard passed the first draft")
    return feedback(GUARD_KEY, 0, f"{example_id(example)}: the guard result was {answer.guard}")


def decision_first(example: Mapping[str, Any], answer: ScoredAnswer) -> Feedback | None:
    if answer.kind != V1 or not _is_assess(example):
        return None
    labels = sheet_labels(example)
    if labels is None or not labels.verdicts:
        return None
    sentence = first_sentence(answer.text)
    if any(pattern.search(sentence) for pattern in _VERDICT_PATTERNS):
        return feedback(DECISION_KEY, 1, f"{labels.example_id}: the first sentence holds a verdict word")
    shown = sentence or "(empty answer)"
    return feedback(DECISION_KEY, 0, f"{labels.example_id}: the first sentence holds no verdict word ({', '.join(verdict_words())}): {shown!r}")


def cited_notams(example: Mapping[str, Any], answer: ScoredAnswer) -> Feedback | None:
    if answer.kind != V1:
        return None
    ids = [entry for entry in reference_list(example, "must_mention") or [] if NOTAM_ID.fullmatch(entry.strip())]
    if not ids:
        return None
    missing = [entry for entry in ids if not mentions(answer.text, entry)]
    if missing:
        return feedback(CITED_NOTAMS_KEY, 0, f"{example_id(example)}: the answer does not cite {', '.join(missing)}")
    return feedback(CITED_NOTAMS_KEY, 1, f"{example_id(example)}: the answer cites {', '.join(ids)}")


def cited_guidance(example: Mapping[str, Any], answer: ScoredAnswer) -> Feedback | None:
    if answer.kind != V1 or SEARCH_GUIDANCE not in (reference_list(example, "required_tools") or []):
        return None
    returned = list(dict.fromkeys(doc for call in answer.tool_calls if call.name == SEARCH_GUIDANCE for doc in call.guides))
    named = [doc for doc in returned if mentions(answer.text, doc)]
    if named:
        return feedback(CITED_GUIDANCE_KEY, 1, f"{example_id(example)}: the answer names {', '.join(named)}, which search_guidance returned")
    detail = f"it returned {', '.join(returned)}" if returned else "it returned no document or was not called"
    return feedback(CITED_GUIDANCE_KEY, 0, f"{example_id(example)}: the answer names no document search_guidance returned ({detail})")


def cited(example: Mapping[str, Any], answer: ScoredAnswer) -> Feedback | None:
    parts = [found for found in (cited_notams(example, answer), cited_guidance(example, answer)) if found is not None]
    if not parts:
        return None
    failed = [part for part in parts if not part["score"]]
    if failed:
        return feedback(CITED_KEY, 0, "; ".join(part["comment"] for part in failed))
    return feedback(CITED_KEY, 1, "; ".join(part["comment"] for part in parts))


def _window(example: Mapping[str, Any]) -> tuple[datetime, datetime] | None:
    """The assess example's window as two aware times, or None for another path or an example with no window."""
    window = (example.get("reference") or {}).get("window")
    if not _is_assess(example) or window is None:
        return None
    window_of(window, f"example {example_id(example)} reference.window")  # a bad window raises ValueError naming the example
    return aware(window[0], "window start"), aware(window[1], "window end")


def window_stated(example: Mapping[str, Any], answer: ScoredAnswer) -> Feedback | None:
    window = _window(example)
    if window is None:
        return None
    start, end = clock(window[0], DEFAULT_TZ), clock(window[1], DEFAULT_TZ)
    missing = [shown for shown in dict.fromkeys((start, end)) if not mentions(answer.text, shown)]
    if missing:
        return feedback(WINDOW_KEY, 0, f"{example_id(example)}: the answer does not state the window {start}-{end} ({DEFAULT_TZ}); missing {', '.join(missing)}")
    return feedback(WINDOW_KEY, 1, f"{example_id(example)}: the answer states the window {start}-{end} ({DEFAULT_TZ})")


def recheck_stated(example: Mapping[str, Any], answer: ScoredAnswer) -> Feedback | None:
    window = _window(example)
    if window is None or answer.kind != V1:
        return None
    now = aware((example.get("inputs") or {}).get("now"), f"example {example_id(example)} inputs.now")
    line = recheck_line([SimpleNamespace(window_start=window[0])], now, DEFAULT_TZ, ASSESS)
    if line is None:
        return None
    if mentions(answer.text, line):
        return feedback(RECHECK_KEY, 1, f"{example_id(example)}: the window starts after the current hour and the answer carries the re-check line")
    return feedback(RECHECK_KEY, 0, f"{example_id(example)}: the window starts after the current hour but the answer lacks the re-check line {line!r}")
