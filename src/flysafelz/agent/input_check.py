"""The input check: what a request must be before the model sees it. Runs first in every turn, as the agent's one `before_agent` hook.

    validate_request(state, ctx) -> dict   the state update for an accepted turn, or RequestInvalid (one line) for a rejected one
    check_input                            the same check as a stand-alone `before_agent` hook (the contract tests' graph); the real graph runs `validate_request` from the one middleware class in middleware.py

An accepted turn clears the last turn's `answer`, `guard`, `overlays` and `assessment`, and keeps the thread's LZ and time zone
in `thread_lz`. A request outside the UK is answered here by code, with no model call, and its LZ is kept in `thread_lz` too, so
the thread is tied to it like any other. A rejected request raises, so this hook writes nothing for it: no answer, no overlays, no
phone numbers, and no change to `thread_lz`.

LangGraph saves a turn's input messages before this hook runs, so a rejected question is still in the thread's `messages` when the
rejection is raised. The next accepted turn removes it (and any other question left unanswered before the newest one), so the model
is never shown a rejected question and `messages` do not keep it. A caller therefore sends one question per request.

Checks, in this order, the first failure being the one reported:
    1. the run context (case-mode switches, the case exists, `now` is timezone-aware)
    2. the last message is a human message (the question)
    3. lat and lon: both sent (or both left out in a follow-up, which uses the thread's LZ), finite numbers in range, and the
       same LZ the thread already has. A key sent as None is refused as missing, not read as left out
    4. tz: a time zone name in the bundled tzdata (or the key left out: the thread's, else Europe/London). None is refused
    5. the UK bounds from rules.yaml: outside them is not an error but code's fixed answer
"""

import math
from collections.abc import Mapping
from typing import Any, get_args
from zoneinfo import ZoneInfo, available_timezones

from langchain.agents.middleware import before_agent
from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage
from langgraph.runtime import Runtime

from flysafelz.agent.state import DEFAULT_TZ, LAT_LIMIT, DropName, ReaderName, RequestState, RunContext, ThreadLZ, is_aware
from flysafelz.geo import HALF_TURN, OutsideUKCoverage, check_uk_bounds
from flysafelz.sources import SourceError, load_case_sources

UK_ONLY_ANSWER = "FlySafeLZ only covers the UK."
READERS = get_args(ReaderName)
DROPS = get_args(DropName)

# Every zone name tzdata knows. A name outside this set is refused, so a path ("../etc/passwd") or a spelling a
# case-insensitive disk would let through ("europe/london") never reaches ZoneInfo.
KNOWN_TIME_ZONES = frozenset(available_timezones())


class RequestInvalid(ValueError):
    """The request cannot be answered. The message is one line naming the field."""


def _coordinate(name: str, value: object, limit: float) -> float:
    if value is None:
        raise RequestInvalid(f"{name} is required")
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise RequestInvalid(f"{name} must be a number")
    if not math.isfinite(value):
        raise RequestInvalid(f"{name} must be a finite number")
    if abs(value) > limit:
        raise RequestInvalid(f"{name} must be between -{limit} and {limit}")
    return float(value)


def _time_zone(value: object) -> str:
    if isinstance(value, str) and value in KNOWN_TIME_ZONES:
        try:
            ZoneInfo(value)
            return value
        except (KeyError, ValueError, OSError):
            pass
    raise RequestInvalid(f"unknown time zone {value!r}")


def _check_context(ctx: RunContext) -> None:
    if ctx.reader not in READERS:
        raise RequestInvalid(f"unknown reader {ctx.reader!r}; allowed: {', '.join(READERS)}")
    drop = ctx.drop
    if not isinstance(drop, tuple):
        raise RequestInvalid(f"drop must be a list of: {', '.join(DROPS)}")
    for value in drop:
        if value not in DROPS:
            raise RequestInvalid(f"unknown drop value {value!r}; allowed: {', '.join(DROPS)}")
    if ctx.now is not None and not is_aware(ctx.now):
        raise RequestInvalid("now must be a timezone-aware datetime")
    if ctx.case is None:
        for name, used in (("reader", ctx.reader != "fixture"), ("reader_cache", ctx.reader_cache is not None), ("drop", bool(drop))):
            if used:
                raise RequestInvalid(f"{name} only applies in case mode; set a case")
        return
    if ctx.reader_cache is not None and ctx.reader == "fixture":
        raise RequestInvalid("reader_cache needs reader 'live'; with reader 'fixture' it would be ignored")
    if not isinstance(ctx.case, str):
        raise RequestInvalid("case must be a string")
    try:
        # The case's readings.yaml is only needed when the readings come from it.
        load_case_sources(ctx.case, readings_optional=ctx.reader == "live")
    except SourceError as exc:
        raise RequestInvalid(str(exc)) from None


def _check_question(messages: Any) -> None:
    if not messages or not isinstance(messages[-1], HumanMessage):
        raise RequestInvalid("no question: the last message must be the pilot's")


def _unanswered_questions(messages: Any) -> list[RemoveMessage]:
    """Removals for the pilot's questions just before the newest one with no reply between: what earlier rejected requests left in the thread."""
    stale = []
    for message in reversed(messages[:-1]):
        if not isinstance(message, HumanMessage):
            break
        if message.id is not None:
            stale.append(RemoveMessage(id=message.id))
    return stale[::-1]


def _location(state: Mapping[str, Any], stored: ThreadLZ | None) -> tuple[float, float]:
    """The request's LZ: what it sent, or the thread's own when it left out both keys. A different LZ than the thread's is refused.

    A key that is left out is not a key set to None: None is a value, and it is refused as missing.
    """
    if stored is not None and "lat" not in state and "lon" not in state:
        return stored["lat"], stored["lon"]
    lat = _coordinate("lat", state.get("lat"), LAT_LIMIT)
    lon = _coordinate("lon", state.get("lon"), HALF_TURN)
    if stored is not None and (lat, lon) != (stored["lat"], stored["lon"]):
        raise RequestInvalid("one landing zone per conversation; start a new thread")
    return lat, lon


def _zone(state: Mapping[str, Any], stored: ThreadLZ | None) -> str:
    """The request's time zone: what it sent, else (key left out) the thread's last one, else Europe/London. None is refused."""
    if "tz" in state:
        return _time_zone(state["tz"])
    return stored["tz"] if stored is not None else DEFAULT_TZ


def validate_request(state: Mapping[str, Any], ctx: RunContext | None) -> dict[str, Any]:
    """Check one turn's request. Returns the state update for it, or raises RequestInvalid.

    Inside the UK the update clears the last turn's outputs and stores the thread's LZ and time zone. Outside it, the update is
    code's UK-only answer (one AIMessage, `guard` "passed", no overlays, no assessment), the thread's LZ and time zone, and a jump
    to the end, so the model is never called. Either way, earlier unanswered questions (left by rejected requests) are removed.
    """
    _check_context(RunContext() if ctx is None else ctx)
    _check_question(state.get("messages"))
    stored = state.get("thread_lz")
    lat, lon = _location(state, stored)
    thread_lz = ThreadLZ(lat=lat, lon=lon, tz=_zone(state, stored))
    stale = _unanswered_questions(state["messages"])
    try:
        check_uk_bounds(lat, lon)
    except OutsideUKCoverage:
        reply = {"answer": UK_ONLY_ANSWER, "guard": "passed", "overlays": [], "assessment": None, "thread_lz": thread_lz}
        return {**reply, "messages": [*stale, AIMessage(UK_ONLY_ANSWER)], "jump_to": "end"}
    update = {"answer": None, "guard": None, "overlays": [], "assessment": None, "thread_lz": thread_lz}
    return {**update, "messages": stale} if stale else update


@before_agent(state_schema=RequestState, can_jump_to=["end"])
def check_input(state: RequestState, runtime: Runtime[RunContext]) -> dict[str, Any]:
    """The agent's first step each turn: validate_request on the thread state and this run's context."""
    return validate_request(state, runtime.context)
