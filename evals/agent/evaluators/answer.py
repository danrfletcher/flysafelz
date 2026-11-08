"""The one shape every answer is scored in (5a PR-5.F1): `ScoredAnswer`, and the two ways to build it.

    ScoredAnswer(path, segments, notam_actions, text)
    ScoredAnswer.from_assessment(assessment, *, path, text)   from a recorded `flysafelz.contracts.Assessment` (v1: the assess tool's result)
    ScoredAnswer.from_dict(data)                              from a plain v0-style dict: {"path", "segments": [...], "notam_actions": {...}, "text"}
    path_from_metadata(metadata)                              the predicted path of a run: its root run's metadata `path`, else "unknown" (I5)

    path            one of PATHS, or "unknown". Never read from the answer text: it is the guard's tag, which ticket 4 puts in the root run's metadata (I5); a
                    run whose tag is missing after the guard's retry has no `path` there, and scores as a path miss
    segments        ScoredSegment(start, end, verdict, approaching_limits): aware times, a `contracts.Verdict`, a bool. Order and overlap do not matter
    notam_actions   {NOTAM id: NotamAction} for the NOTAMs the answer surfaced. A NOTAM the answer does not list counts as IGNORE, as in the sheet. Ids have
                    their whitespace removed and keep their `/26` year
    text            the final answer text

The v0 baseline's structured output (start, end, verdict, approaching_limits, conditions per segment, and an action per NOTAM) goes through `from_dict`; `conditions`
and any other key are ignored. A segment time with no timezone, a verdict that is not a `Verdict` and an action that is not a `NotamAction` each raise ValueError.
v1's `Assessment` gives its VERDICT-tier hours' segments (OUTLOOK and TOO_FAR hours have none) and, for each NOTAM, the most restrictive action any segment gave it
(evals.reader_eval.downgrade.SEVERITY ranks them). The two builders give identical ScoredAnswers for the same segments.
"""

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from evals.agent.evaluators.minutes import aware
from evals.reader_eval.downgrade import SEVERITY
from flysafelz.contracts import Assessment, NotamAction, Verdict

PATHS = ("assess", "explain", "act", "clarify", "decline")
UNKNOWN_PATH = "unknown"
_WHITESPACE = re.compile(r"\s+")


def clean_id(notam_id: object) -> str:
    """A NOTAM id with its whitespace removed ("J3219 /26" is "J3219/26"). The `/26` year stays: J3219/25 is another NOTAM."""
    return _WHITESPACE.sub("", str(notam_id))


def action_of(value: object, what: str) -> NotamAction:
    try:
        return NotamAction(value)
    except ValueError:
        raise ValueError(f"{what}: {value!r} is not one of {', '.join(a.value for a in NotamAction)}") from None


def normalise_path(value: object) -> str:
    """The path name when it is one of PATHS (any case), else "unknown": a missing tag and a tag that is not a path both score as a miss in the `unknown` column."""
    if isinstance(value, str) and value.strip().lower() in PATHS:
        return value.strip().lower()
    return UNKNOWN_PATH


def path_from_metadata(metadata: Mapping[str, Any] | None) -> str:
    """The predicted path of a run, from its root run's metadata `path` (I5). "unknown" when the key is missing or is not a path. The answer text is never parsed."""
    return normalise_path((metadata or {}).get("path"))


@dataclass(frozen=True)
class ScoredSegment:
    start: datetime
    end: datetime
    verdict: Verdict
    approaching_limits: bool = False

    def __post_init__(self) -> None:
        start, end = aware(self.start, "segment start"), aware(self.end, "segment end")
        if end < start:
            raise ValueError(f"segment ends at {self.end!r}, before it starts at {self.start!r}")
        object.__setattr__(self, "start", start)
        object.__setattr__(self, "end", end)
        try:
            object.__setattr__(self, "verdict", Verdict(self.verdict))
        except ValueError:
            raise ValueError(f"segment verdict {self.verdict!r} is not one of {', '.join(v.value for v in Verdict)}") from None
        object.__setattr__(self, "approaching_limits", bool(self.approaching_limits))


@dataclass(frozen=True)
class ScoredAnswer:
    path: str = UNKNOWN_PATH
    segments: tuple[ScoredSegment, ...] = ()
    notam_actions: Mapping[str, NotamAction] = field(default_factory=dict)
    text: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", normalise_path(self.path))
        object.__setattr__(self, "segments", tuple(self.segments))
        actions: dict[str, NotamAction] = {}
        for notam_id, action in dict(self.notam_actions).items():
            found = action_of(action, f"NOTAM {notam_id}")
            kept = clean_id(notam_id)
            actions[kept] = _stricter(actions.get(kept), found)
        object.__setattr__(self, "notam_actions", actions)

    @classmethod
    def from_assessment(cls, assessment: Assessment, *, path: object, text: str = "") -> "ScoredAnswer":
        """The answer an `Assessment` gives: its segments, and each NOTAM's most restrictive action. `path` is the run's metadata path (see path_from_metadata)."""
        segments = [ScoredSegment(s.start, s.end, s.verdict, s.approaching_limits) for hour in assessment.hours for s in hour.segments]
        actions: dict[str, NotamAction] = {}
        for hour in assessment.hours:
            for segment in hour.segments:
                for result in segment.notam_actions:
                    actions[result.notam_id] = _stricter(actions.get(result.notam_id), result.action)
        return cls(normalise_path(path), tuple(segments), actions, text)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ScoredAnswer":
        """A v0-style dict. `notam_actions` is {id: action} or a list of {"id": ..., "action": ...}; `segments` is a list of {start, end, verdict, approaching_limits}."""
        segments = tuple(
            ScoredSegment(item.get("start"), item.get("end"), item.get("verdict"), bool(item.get("approaching_limits", False)))  # type: ignore[arg-type]
            for item in data.get("segments") or []
        )
        return cls(normalise_path(data.get("path")), segments, _actions_of(data.get("notam_actions")), str(data.get("text") or ""))


def _stricter(held: NotamAction | None, found: NotamAction) -> NotamAction:
    return found if held is None or SEVERITY.index(found) < SEVERITY.index(held) else held


def _actions_of(value: object) -> dict[str, NotamAction]:
    if value is None:
        return {}
    pairs: Iterable[tuple[object, object]]
    if isinstance(value, Mapping):
        pairs = value.items()
    elif isinstance(value, list):
        pairs = [(item.get("id", item.get("notam_id")), item.get("action")) for item in value]
    else:
        raise ValueError(f"notam_actions must be a mapping or a list of {{id, action}}, not {value!r}")  # noqa: TRY004  # every bad input to a builder is a ValueError, so one except clause serves
    found: dict[str, NotamAction] = {}
    for notam_id, action in pairs:
        kept = clean_id(notam_id)
        found[kept] = _stricter(found.get(kept), action_of(action, f"NOTAM {notam_id}"))
    return found
