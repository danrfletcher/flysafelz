"""What the three data tools share: the arguments the model may give, the steps every tool starts with, and how a tool answers.

    WindowArgs                  the only arguments of a data tool: `start` and `end` (plus the injected `runtime`, which the model never sees)
    prepare(runtime, source_cache, ...) -> Prepared | str   the LZ and zone from state, the clock and case from the run context, the window, the sources
    Redactor                    the one place a tool's text and artifact lose their phone numbers; carries the thread's `phone_map`
    reply(runtime, text, artifact, redactor) -> Command   the tool message (text + artifact) and the `phone_map` update

A tool reads `lat`, `lon` and `tz` from the thread's private `thread_lz` and `now` / `case` from the run context, never from its arguments
(Q8, F2): the argument schema forbids any other key, so a model that sends `lat` is refused before the tool runs.

Everything a tool returns to the model is plain text a pilot could read: times in `tz` with the zone abbreviation, numbers with their limits,
plain counts and the fetch time. A problem the model should relay (a window it cannot assess, a case that does not load, a source that failed)
is text too, never an exception.
"""

import operator
import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import datetime
from itertools import count
from typing import Any
from zoneinfo import ZoneInfo

from langchain.tools import ToolRuntime
from langchain_core.messages import ToolMessage
from langgraph.types import Command
from pydantic import BaseModel, ConfigDict, Field

from flysafelz.agent import progress
from flysafelz.agent.redaction import redact_artifact, redact_text
from flysafelz.agent.sources import SourceCache, Sources, SourcesError, fetch_time, thread_id_of, turn_of
from flysafelz.agent.state import RunContext, run_now
from flysafelz.agent.tools.artifacts import SourceInfo
from flysafelz.agent.window import Window, WindowError, parse_window
from flysafelz.clients.open_meteo import Forecast
from flysafelz.clients.openaip import metres_to_feet
from flysafelz.contracts import Limit, LimitUnit, SourceName, SourceState, SourceStatus, position_word
from flysafelz.daylight import utc_hours
from flysafelz.redact import redact_phones
from flysafelz.sources import CHECK_BY_HAND
from flysafelz.timetext import stamp
from flysafelz.weather import build_rows

NOT_FORECAST = "not forecast"
NONE_LINE = "none"
NO_LANDING_ZONE = "No landing zone is set for this thread, so there is nothing to check."


# `ToolRuntime` alone is `ToolRuntime[None, dict]`: its `context` is typed None, and pydantic then warns on stderr each time it copies the arguments of a
# tool call with a real RunContext in it. Naming the context type here is what keeps a live run's stderr to the progress lines.
AgentRuntime = ToolRuntime[RunContext, Any]


class WindowArgs(BaseModel):
    """The arguments of a data tool. The model gives a window, or leaves both out for the default one. Nothing else.

    `runtime` is filled in by LangGraph and is not part of the schema the model is shown.
    """

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    start: str | None = Field(default=None, description="Window start as a local time, written YYYY-MM-DD HH:MM. Leave out together with `end` for the default window.")
    end: str | None = Field(default=None, description="Window end as a local time, written YYYY-MM-DD HH:MM. Leave out together with `start` for the default window.")
    runtime: AgentRuntime = Field(default=None)  # type: ignore[assignment]  # injected; never part of the model-facing schema


ORDER = {">=": operator.ge, ">": operator.gt, "<=": operator.le, "<": operator.lt}  # the ordering comparisons of rules.yaml


@dataclass(frozen=True)
class Prepared:
    """What a tool needs once the window and the sources are settled."""

    lat: float
    lon: float
    tz: str
    ctx: RunContext
    now: datetime
    window: Window
    sources: Sources
    live: bool  # no case: a live run


def g(value: float) -> str:
    """A number as short as it can be written: 12 not 12.0, 0.1 as 0.1."""
    return f"{value:g}"


def measured(value: float, limits: Iterable[tuple[str, float]] = ()) -> str:
    """A measured value to one decimal place without a trailing zero: 8, 8.4.

    `limits` are the (comparison, limit) pairs it is printed beside. One decimal is used unless the rounded number would sit on the other side
    of a limit from the true one (11.96 rounded to 12 against `>= 12`): then decimals are added until the printed number and the true one
    agree on every limit, so a shown value never contradicts the limit printed next to it.
    """
    tests = [(ORDER[comparison], limit) for comparison, limit in limits if comparison in ORDER]
    for places in count(1):  # ends at the latest when the text is the value itself, which agrees on every limit
        text = f"{value:.{places}f}".rstrip("0").rstrip(".")
        if all(test(float(text), limit) == test(value, limit) for test, limit in tests):
            return text
    raise AssertionError("count() does not end")  # pragma: no cover


def limit_phrase(value: float, limit: float, comparison: str, unit: str) -> str:
    """Where a value sits against its limit, as the one phrase the guard reads (guard.py `_LIMIT_PHRASE`): `over the 6 mph limit`.

    The word is `contracts.position_word` of the true value, the limit and the rule's comparison, never anything read from text. `value` and `limit`
    must be in `unit`.
    """
    return f"{position_word(value, limit, comparison)} the {g(limit)} {unit} limit"


def hour_label(hour: datetime, tz: str) -> str:
    """An hour on the clock of `tz` with its date and the abbreviation of its own instant: Sat 31 Oct 10:00 GMT, Sun 25 Oct 01:00 BST."""
    local = hour.astimezone(ZoneInfo(tz))
    return f"{local:%a %d %b %H:%M} {local.tzname()}"


def limit_text(limit: Limit | None) -> str | None:
    """A floor or ceiling as the data states it: SFC, FL65, 1500 ft AGL. None when there is none."""
    if limit is None:
        return None
    if limit.datum.value == "SFC":
        return limit.datum.value
    if limit.unit is LimitUnit.FL:
        return f"FL{g(limit.value)}"
    return f"{g(limit.value)} {limit.unit.value} {limit.datum.value}"


def elevation_ft(forecast: Forecast) -> tuple[float, bool]:
    """(site elevation in feet, whether the forecast gave one). Without one the site is taken at the ground of the sea, and the caller says so."""
    if forecast.elevation_m is None:
        return 0.0, False
    return metres_to_feet(forecast.elevation_m), True


def lowest_pressure(forecast: Forecast, window: Window) -> float | None:
    """The lowest forecast pressure (hPa) in the window, or None. The lowest is the cautious one for flight levels: the same FL is lower on a low-pressure day."""
    pressures = [row.values.pressure_msl for row in build_rows(forecast, window.start_utc, window.end_utc) if row.values.pressure_msl is not None]
    return min(pressures, default=None)


def window_hours(window: Window) -> list[datetime]:
    return utc_hours(window.start_utc, window.end_utc)


# --- redaction -----------------------------------------------------------------------------------------------------


class Redactor:
    """The thread's `phone_map` while one tool builds its answer. Every string that leaves the tool goes through `text` or `artifact`.

    Tools that run in the same step all start from the same `phone_map`, so each would number a new phone `<PHONE_n>` from the same n. Once
    `share` has been given the thread's Sources they number through its `phones` instead (under its lock), so they never give one placeholder
    to two numbers, and `merge_phone_maps` has nothing to refuse.
    """

    def __init__(self, phone_map: dict[str, str] | None):
        self.start = dict(phone_map or {})
        self.map = dict(self.start)
        self._sources: Sources | None = None

    def share(self, sources: Sources) -> None:
        self._sources = sources

    def _redact(self, redact: Callable[[Any, dict[str, str]], tuple[Any, dict[str, str]]], value: Any) -> Any:
        if self._sources is None:
            redacted, self.map = redact(value, self.map)
            return redacted
        with self._sources.phone_lock:
            known = {**self._sources.phones, **self.map}
            redacted, self.map = redact(value, known)
            self._sources.phones.update(self.map)
        return redacted

    def text(self, value: str) -> str:
        return self._redact(redact_text, value)

    def artifact(self, value: Any) -> Any:
        return self._redact(redact_artifact, value)

    def localiser(self, item_e: str) -> Callable[[str], str]:
        """A function that turns the `<PHONE_n>` of ticket 3's own numbering for this NOTAM's E text into the thread's numbering.

        The reader, `notam_action` and the airspace code number a NOTAM's phones from 1 for that NOTAM alone, so a `<PHONE_1>` in their text
        could stand for another number than the thread's `<PHONE_1>`. Text that came from them goes through this first.
        """
        _redacted, own = redact_phones(item_e)
        swap = {placeholder: self.text(number) for placeholder, number in own.items()}
        if not swap:
            return lambda value: value
        pattern = re.compile("|".join(re.escape(placeholder) for placeholder in swap))
        return lambda value: pattern.sub(lambda match: swap[match.group(0)], value)


def reply(runtime: ToolRuntime, text: str, artifact: Any, redactor: Redactor) -> Command:
    """The tool's answer: the text and artifact as a tool message, and the `phone_map` when this call added a number to it."""
    text = redactor.text(text)
    artifact = None if artifact is None else redactor.artifact(artifact)
    update: dict[str, Any] = {"messages": [ToolMessage(content=text, tool_call_id=runtime.tool_call_id, artifact=artifact)]}
    if redactor.map != redactor.start:
        update["phone_map"] = redactor.map
    return Command(update=update)


# --- steps every tool starts with ------------------------------------------------------------------------------------


def prepare(runtime: ToolRuntime, source_cache: SourceCache, redactor: Redactor, start: str | None, end: str | None, line: str | None = None) -> Prepared | str:
    """The LZ, zone, clock, window and sources for this call, or the one line to give the model instead.

    The window is read first, so one that cannot be assessed (malformed, passed, too far out) fetches nothing. `line` is the progress line to
    write once the window is good and before the sources are fetched.
    """
    thread_lz = runtime.state.get("thread_lz")
    if not thread_lz:
        return NO_LANDING_ZONE
    lat, lon, tz = thread_lz["lat"], thread_lz["lon"], thread_lz["tz"]
    ctx = runtime.context if isinstance(runtime.context, RunContext) else RunContext()
    now = run_now(ctx)
    window = parse_window(start, end, now=now, tz=tz)
    if isinstance(window, WindowError):
        return window.text
    if line is not None:
        progress.emit(line)
    sources = source_cache.get(thread_id=thread_id_of(runtime.config), turn=turn_of(runtime.state["messages"]), lat=lat, lon=lon, ctx=ctx)
    if isinstance(sources, SourcesError):
        return sources.text
    redactor.share(sources)
    return Prepared(lat, lon, tz, ctx, now, window, sources, live=ctx.case is None)


def fetch_line(prepared: Prepared) -> str:
    """When the data was fetched (FR9): the oldest fetch among the three sources, which every tool of a turn shares (I2)."""
    try:
        return f"Fetched: {stamp(fetch_time(prepared.sources.statuses), prepared.tz)}."
    except ValueError:
        return "Fetched: no source returned data."


def window_line(prepared: Prepared) -> str:
    """The window in words. When code chose it, the words already begin `assumed window: `."""
    words = prepared.window.words
    return words if prepared.window.assumed else f"Window: {words}."


def source_info_at(status: SourceStatus, fetched: datetime | None) -> SourceInfo:
    """The artifact's description of one source (see artifacts.SourceInfo), given the turn's fetch time (None: no source has one)."""
    ok = status.state is SourceState.OK
    return SourceInfo(source=status.source.value, state=status.state.value, reason=status.reason, fetched_at=fetched, check_by_hand=None if ok else CHECK_BY_HAND[status.source])


def source_info(prepared: Prepared, status: SourceStatus) -> SourceInfo:
    """The artifact's description of one source, with the turn's fetch time (FR9) taken from the prepared sources."""
    try:
        fetched: datetime | None = fetch_time(prepared.sources.statuses)
    except ValueError:
        fetched = None
    return source_info_at(status, fetched)


def usable(status: SourceStatus) -> bool:
    return status.state is SourceState.OK


def unusable_text(label: str, status: SourceStatus) -> str:
    """The line for a source that is not OK: its state, why, and where the pilot can check by hand. No values follow it."""
    why = f" ({status.reason})" if status.reason else ""
    return f"{label} source: {status.state.value.upper()}{why}. No {label.lower()} values are given. Check by hand: {CHECK_BY_HAND[status.source]}."


SOURCE_LABEL = {SourceName.WEATHER: "Weather", SourceName.AIRSPACE: "Airspace", SourceName.NOTAMS: "NOTAM"}
