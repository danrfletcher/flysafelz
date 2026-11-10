"""The synthetic reference of one production run (7 PR-1.F1, G6): what 5a's checks are told about a run that has no labels. Built from the run and its trace, and nothing else.

    build_reference(run, tree) -> OnlineReference

`run` is the root run. `tree` is the runs below it in the same trace (children at any depth; a run without a parent is ignored). Fields:
    path             root metadata `path`, else root outputs `path`, else `unknown` (ticket 4's I5, I6). A value that is not one of 5a's PATHS counts as missing, and `unknown` stays `unknown`:
                     the path is never worked out from the answer
    now              inputs `now` if it parses (a time with no zone is UTC; any other zone is converted to UTC), else the run's `start_time` as UTC
    required_tools   `("search_guidance",)` if a tool run of that name is in the tree, else `()`
    forbidden_tools  all of ticket 4's agent tools (`flysafelz.agent.graph.TOOL_NAMES`) when the path is `decline`, else `()`
    lz_name, lat, lon   root metadata `lz_name` (the seed writes it) and inputs `lat` and `lon`, for the judge's place line; None when absent
    assess           the last `assess` tool run's artifact as an `AssessedWindow` (the window and the VERDICT-tier segments, as 5a's `ScoredSegment`), else None
    assess_reason    `no assess` when `assess` is None (no assess run, or one whose output holds no readable artifact), for PR-2.F1 to turn into nulls; else None

There are no `must_mention`, window or sheet labels: a production run has none, and none is invented. Segments are read the way 5a's `eval_run.scored_answer_of` reads an artifact (a segment with no
verdict, an OUTLOOK or TOO_FAR hour, is left out).

5a has no reference type of its own: its evaluators read an example mapping and a `ScoredAnswer`. `OnlineReference` here only carries what PR-2.F1 needs to build those from a run.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from evals.agent.eval_run import ASSESS_TOOL
from evals.agent.evaluators.answer import UNKNOWN_PATH, ScoredSegment, normalise_path
from evals.agent.evaluators.trajectory import SEARCH_GUIDANCE
from evals.online.runs import PATH_KEY, inputs_of, metadata_of, outputs_of
from flysafelz.agent.graph import TOOL_NAMES

DECLINE = "decline"
TOOL_RUN = "tool"
NO_ASSESS = "no assess"
LZ_NAME_KEY = "lz_name"
MAX_DEPTH = 6  # how deep into a tool run's outputs the artifact is looked for; a real trace has it four levels down


@dataclass(frozen=True)
class AssessedWindow:
    """What the run's `assess` call assessed: its window and the segments that have a verdict."""

    start: datetime
    end: datetime
    segments: tuple[ScoredSegment, ...]


@dataclass(frozen=True)
class OnlineReference:
    path: str
    now: datetime
    required_tools: tuple[str, ...]
    forbidden_tools: tuple[str, ...]
    lz_name: str | None
    lat: float | None
    lon: float | None
    assess: AssessedWindow | None
    assess_reason: str | None


def _utc(moment: datetime) -> datetime:
    return moment.replace(tzinfo=UTC) if moment.tzinfo is None else moment.astimezone(UTC)


def _parsed_time(value: object) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        return _utc(datetime.fromisoformat(value.strip()))
    except ValueError:
        return None


def _now_of(run: Any) -> datetime:
    given = _parsed_time(inputs_of(run).get("now"))
    if given is not None:
        return given
    started = getattr(run, "start_time", None)
    if not isinstance(started, datetime):
        raise ValueError(f"run {getattr(run, 'id', '?')} has no start_time and no usable inputs.now")  # noqa: TRY004  # a run without a start is a broken run, not a wrong type
    return _utc(started)


def path_of(run: Any) -> str:
    """The run's path: root metadata `path`, else root outputs `path`, else `unknown`. The one reading of it: `build_reference` and the seed's output lines both use it."""
    for source in (metadata_of(run), outputs_of(run)):
        path = normalise_path(source.get(PATH_KEY))
        if path != UNKNOWN_PATH:
            return path
    return UNKNOWN_PATH


def _number(value: object) -> float | None:
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def _lz_name(run: Any) -> str | None:
    name = metadata_of(run).get(LZ_NAME_KEY)
    return (name.strip() or None) if isinstance(name, str) else None


def _artifact_in(value: object, depth: int = 0) -> Mapping[str, Any] | None:
    """The assess artifact somewhere in a tool run's outputs: a mapping with `tool == "assess"` and `segments`. The tool returns a Command, so the artifact sits at
    outputs.output.update.messages[*].artifact in a real trace; a plain tool message (outputs.output.artifact) is found the same way."""
    if depth > MAX_DEPTH:
        return None
    if isinstance(value, Mapping):
        if value.get("tool") == ASSESS_TOOL and isinstance(value.get("segments"), list):
            return value
        children: list[object] = list(value.values())
    elif isinstance(value, list):
        children = list(reversed(value))  # the last message is the latest
    else:
        return None
    for item in children:
        found = _artifact_in(item, depth + 1)
        if found is not None:
            return found
    return None


def _assessed(artifact: Mapping[str, Any]) -> AssessedWindow | None:
    try:
        segments = tuple(
            ScoredSegment(datetime.fromisoformat(item["start"]), datetime.fromisoformat(item["end"]), item["verdict"], bool(item.get("approaching_limits")))
            for item in artifact["segments"]
            if item.get("verdict") is not None  # an OUTLOOK or TOO_FAR hour has no verdict to score
        )
        return AssessedWindow(_utc(datetime.fromisoformat(artifact["window_start"])), _utc(datetime.fromisoformat(artifact["window_end"])), segments)
    except (KeyError, TypeError, ValueError, AttributeError):
        return None


def _tool_runs(tree: Sequence[Any], name: str) -> list[Any]:
    return [child for child in tree if getattr(child, "parent_run_id", None) is not None and getattr(child, "run_type", None) == TOOL_RUN and getattr(child, "name", None) == name]


def _started(run: Any) -> datetime:
    started = getattr(run, "start_time", None)
    return _utc(started) if isinstance(started, datetime) else datetime.min.replace(tzinfo=UTC)


def _assess_of(tree: Sequence[Any]) -> AssessedWindow | None:
    for child in sorted(_tool_runs(tree, ASSESS_TOOL), key=_started, reverse=True):  # the last assess call is the one the answer rests on
        artifact = _artifact_in(outputs_of(child))
        window = _assessed(artifact) if artifact is not None else None
        if window is not None:
            return window
    return None


def build_reference(run: Any, tree: Sequence[Any]) -> OnlineReference:
    """The reference for one root run and the runs below it. Raises ValueError only for a run with no start_time and no usable `now` (a run LangSmith never gives)."""
    path = path_of(run)
    inputs = inputs_of(run)
    assess = _assess_of(tree)
    return OnlineReference(
        path=path,
        now=_now_of(run),
        required_tools=(SEARCH_GUIDANCE,) if _tool_runs(tree, SEARCH_GUIDANCE) else (),
        forbidden_tools=tuple(TOOL_NAMES) if path == DECLINE else (),
        lz_name=_lz_name(run),
        lat=_number(inputs.get("lat")),
        lon=_number(inputs.get("lon")),
        assess=assess,
        assess_reason=None if assess is not None else NO_ASSESS,
    )
