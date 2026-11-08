"""Mention matching (5a PR-6.F1, G11, E10): the one normaliser, and the two evaluators that use it.

    normalise_mention(text)            lower-case, remove commas between digits, remove all whitespace (in that order)
    mentions(haystack, needle)         True when the normalised needle is a substring of the normalised haystack
    reference_list(example, key)       a list of strings from `example.reference[key]`, or None when it is absent or empty (shared by trajectory.py and form.py)
    must_mention(example, answer)      every `reference.must_mention` entry is in the final answer
    must_not_mention(example, answer)  no `reference.must_not_mention` entry is in the final answer

Both sides go through the same normaliser, so `1500` is in "base at 1,500 ft" and `Newcastle CTA 2` is in "NEWCASTLE CTA2", while `18:00` is not in "until 6pm".
The rule is literal and is not widened:
  * the comma goes only between two digits, so "1,500,000" is "1500000" but "1, 500" keeps its comma (the comma is not between digits) and does not hold `1500`;
  * whitespace is anything `\\s` matches, so a tab and a no-break space (U+00A0) count: "1\\u00a0500" is "1500";
  * a hyphen is kept, so `NO-GO` is not in "no go". That is a known blind spot of `must_not_mention`: an answer that writes the words without the hyphen passes it;
  * it is a substring match, not a word match: `Burn` is in "Sherburn" (so a held-out check on that word is a judge note, not a mention entry).

must_mention scores 1 when every entry is there, else 0, and the comment lists the missing ones. must_not_mention scores 1 when none is there, else 0, and the comment lists
the ones found. An empty or whitespace-only answer holds nothing: must_mention fails and must_not_mention passes. Both return None when the list is absent or empty, and
apply to every path and to both run kinds. An entry that normalises to nothing (empty or only whitespace) raises ValueError naming the example: it would match every answer.
"""

import re
from collections.abc import Mapping
from typing import Any

from evals.agent.evaluators.answer import ScoredAnswer
from evals.agent.evaluators.labels import Feedback, example_id, feedback

MUST_MENTION_KEY = "must_mention"
MUST_NOT_MENTION_KEY = "must_not_mention"
_COMMA_BETWEEN_DIGITS = re.compile(r"(?<=\d),(?=\d)")
_WHITESPACE = re.compile(r"\s+")


def normalise_mention(text: object) -> str:
    """Lower-case, then remove commas that sit between two digits, then remove all whitespace. In that order."""
    return _WHITESPACE.sub("", _COMMA_BETWEEN_DIGITS.sub("", str(text).lower()))


def mentions(haystack: object, needle: object) -> bool:
    """True when the normalised `needle` is a substring of the normalised `haystack`. A needle that normalises to nothing raises ValueError."""
    wanted = normalise_mention(needle)
    if not wanted:
        raise ValueError(f"mention {needle!r} is empty once normalised: it would match every answer")
    return wanted in normalise_mention(haystack)


def reference_list(example: Mapping[str, Any], key: str) -> list[str] | None:
    """`example.reference[key]` as a list of strings; None when the key is absent, null or an empty list. Anything else raises ValueError naming the example."""
    value = (example.get("reference") or {}).get(key)
    if value is None or value == []:
        return None
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"example {example_id(example)}: reference.{key} must be a list of strings, not {value!r}")
    return list(value)


def _checked(example: Mapping[str, Any], entries: list[str], text: str, key: str) -> list[str]:
    """The entries found in the text. An entry that normalises to nothing raises ValueError naming the example and the key."""
    try:
        return [entry for entry in entries if mentions(text, entry)]
    except ValueError as error:
        raise ValueError(f"example {example_id(example)}: reference.{key}: {error}") from None


def must_mention(example: Mapping[str, Any], answer: ScoredAnswer) -> Feedback | None:
    entries = reference_list(example, MUST_MENTION_KEY)
    if entries is None:
        return None
    found = _checked(example, entries, answer.text, MUST_MENTION_KEY)
    missing = [entry for entry in entries if entry not in found]
    if missing:
        return feedback(MUST_MENTION_KEY, 0, f"{example_id(example)}: the answer does not mention {', '.join(repr(m) for m in missing)}")
    return feedback(MUST_MENTION_KEY, 1, f"{example_id(example)}: the answer mentions all {len(entries)} required entries")


def must_not_mention(example: Mapping[str, Any], answer: ScoredAnswer) -> Feedback | None:
    entries = reference_list(example, MUST_NOT_MENTION_KEY)
    if entries is None:
        return None
    found = _checked(example, entries, answer.text, MUST_NOT_MENTION_KEY)
    if found:
        return feedback(MUST_NOT_MENTION_KEY, 0, f"{example_id(example)}: the answer mentions {', '.join(repr(f) for f in found)}, which it must not")
    return feedback(MUST_NOT_MENTION_KEY, 1, f"{example_id(example)}: the answer mentions none of the {len(entries)} forbidden entries")
