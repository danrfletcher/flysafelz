"""The sources a thread's tools share: one fetch per source per turn, live or from a case.

    SourceCache(...)                       one per process; holds each thread's fetched sources
    SourceCache.get(thread_id=, turn=, lat=, lon=, ctx=)   -> Sources | SourcesError
    Sources                                forecast, airspace, pib, readings, unread, and `status(name)`
    fetch_time(statuses)                   the oldest `fetched_at` among the statuses (FR9)
    thread_id_of(config), turn_of(messages)   the thread and the turn a tool is running in

`get_weather`, `get_airspace`, `get_notams` and `assess` all call `get`, so a turn never fetches a source twice and every tool in
it reports the same fetch and the same fetch time.

What an entry is keyed by (I2): the thread, the lat and lon, and `fetch_horizon(now)`. A live fetch covers the whole horizon
(from the hour we are in to `fetch_horizon(now)`), so it serves every window the model may ask for; the window is not in the key.
One fetch returns all three sources, so one entry holds all three. In case mode the key also holds the case, `reader`,
`reader_cache` and `drop` (a run that reads NOTAMs from `readings.yaml` and a run that has the model read them never share readings).

When an entry is used again:
  - it is younger than `source_reuse_minutes` (settings.yaml), measured on the run context's `now` and never on the wall clock;
  - and, if any source in it FAILED, only within the turn it failed in. The next turn fetches again.
A per-key lock makes two tool calls that arrive together (two threads of one graph run) share one fetch.
Memory is bounded. A thread keeps one entry: storing a new one drops that thread's older ones (another horizon, another case
setting). Every store also drops the entries of any thread that are `source_reuse_minutes` old or older on that store's clock: they
could not be reused anyway. So a server whose threads each ask once and never return keeps only the entries of the last few minutes.
`forget(thread_id)` drops a thread's entries at once. A per-key lock exists only while a call is using it.

Case mode (I3). With `ctx.case` set the sources come from `load_case_sources`, never the network, with `ctx.drop` applied by
`apply_drops`. `ctx.reader` then picks the readings: "fixture" is the case's `readings.yaml`, with no model call; "live" leaves
`readings` empty, as in live mode, and `readings.yaml` is never opened. In live mode and in case mode with reader "live" the NOTAM tool
reads for its own window (ticket 3's `read_for_lz`, at most `notam_reader_calls` model calls a request, through the reading store at
`ctx.reader_cache` else the default one) and adds what it got to the thread's `Sources` (`record_reads`), so the next tool call finds it.

Nothing here raises for a bad case: `get` returns a `SourcesError` whose text is the one line the tool gives the model. The reader's
settings are only looked at when a tool first has a NOTAM to read (`reader_llm`).
"""

import threading
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.runnables import RunnableConfig

from flysafelz.agent.settings import SETTINGS, AgentSettings
from flysafelz.agent.state import RunContext, run_now
from flysafelz.assess import fetch_horizon
from flysafelz.clients.open_meteo import Forecast
from flysafelz.clients.openaip import AirspaceFetch
from flysafelz.contracts import NotamReading, SourceName, SourceState, SourceStatus
from flysafelz.pib import PibResult
from flysafelz.reader import config as reader_config
from flysafelz.reader.contract import ReaderStatus
from flysafelz.sources import CaseSources, SourceError, fetch_live_sources, load_case_sources
from flysafelz.verdict_sheet import apply_drops

Fetcher = Callable[[float, float, tuple[datetime, datetime], datetime], tuple[Forecast, AirspaceFetch, PibResult]]
CaseLoader = Callable[..., CaseSources]
ReaderModel = Callable[[], BaseChatModel]
_RECORD_LOCK = threading.Lock()  # tool calls of one step may add readings together


@dataclass(frozen=True)
class Sources:
    """What one fetch gave: the three sources and the NOTAM readings that go with them.

    readings   NOTAM id -> reading, from the case's `readings.yaml` or the NOTAM reader; empty until `get_notams` reads (and adds to it), except in a case run with reader "fixture"
    unread     NOTAM id -> ReaderStatus (FAILED or UNAVAILABLE; their values are "failed" and "unavailable") for a NOTAM the reader was asked about and gave no reading for
    announced  the NOTAM ids a case run with the fixture reader has already written a reader progress line for (live mode needs none: a NOTAM read is kept in `readings`)
    limited    the NOTAM ids left unread because the request's reader calls (`notam_reader_calls`) were used up; they stay unread for the life of this entry
    phones     the `<PHONE_n>` numbering the tools of this thread have given so far (placeholder -> number). Tools that run in one step each start from the
               thread's `phone_map` and cannot see each other's numbers, so they number through this, under `phone_lock`, and always agree

    `readings`, `unread`, `announced`, `limited` and `phones` are what the tools add to; the rest never change.
    """

    forecast: Forecast
    airspace: AirspaceFetch
    pib: PibResult
    readings: Mapping[str, NotamReading]
    unread: Mapping[str, ReaderStatus]
    announced: set[str] = field(default_factory=set)
    limited: set[str] = field(default_factory=set)
    phones: dict[str, str] = field(default_factory=dict)

    def record_reads(
        self,
        readings: Mapping[str, NotamReading],
        unread: Mapping[str, ReaderStatus],
        *,
        announced: Iterable[str] = (),
        limited: Iterable[str] = (),
    ) -> None:
        """Keep what a `get_notams` call just did for this thread: the readings, the statuses of the NOTAMs it got none for, the ones it wrote a progress line for and the ones the reader limit left unread.

        Only a Sources made by `SourceCache` can take readings (its two maps are plain dicts); any other raises TypeError.
        """
        if not (isinstance(self.readings, dict) and isinstance(self.unread, dict)):
            raise TypeError("these readings cannot be added to")
        with _RECORD_LOCK:
            self.readings.update(readings)
            self.unread.update(unread)
            self.announced.update(announced)
            self.limited.update(limited)

    @property
    def phone_lock(self) -> threading.Lock:
        """Hold this while reading or adding to `phones`."""
        return _RECORD_LOCK

    def status(self, name: SourceName) -> SourceStatus:
        """The status of one source."""
        return {SourceName.WEATHER: self.forecast.status, SourceName.AIRSPACE: self.airspace.status, SourceName.NOTAMS: self.pib.status}[name]

    @property
    def statuses(self) -> list[SourceStatus]:
        return [self.forecast.status, self.airspace.status, self.pib.status]


@dataclass(frozen=True)
class SourcesError:
    """The sources cannot be had: one line the tool returns to the model as its text (an unknown case, a reader setting that is missing)."""

    text: str


def fetch_time(sources_used: Iterable[SourceStatus]) -> datetime:
    """The time the data was fetched: the oldest `fetched_at` among the sources a turn used (FR9). Raises ValueError when none has one."""
    times = [status.fetched_at for status in sources_used if status.fetched_at is not None]
    if not times:
        raise ValueError("none of the sources has a fetch time")
    return min(times)


def thread_id_of(config: RunnableConfig) -> str:
    """The graph thread a tool is running in, from the run's configuration. Raises ValueError when the run has none (no thread, no shared fetch)."""
    thread_id = (config.get("configurable") or {}).get("thread_id")
    if not thread_id:
        raise ValueError("the run has no thread_id: invoke the graph with config={'configurable': {'thread_id': ...}}")
    return str(thread_id)


def turn_of(messages: Sequence[BaseMessage]) -> str:
    """Which turn of the thread a tool is running in: the id of the pilot's newest message (every model and tool call of a turn follows it)."""
    pilot = [message for message in messages if isinstance(message, HumanMessage)]
    if not pilot:
        return ""
    newest = pilot[-1]
    return str(newest.id) if newest.id is not None else f"question-{len(pilot)}"


def _hour_floor(moment: datetime) -> datetime:
    return moment.astimezone(UTC).replace(minute=0, second=0, microsecond=0)


@dataclass(frozen=True)
class _Entry:
    sources: Sources
    stored_at: datetime  # the run's `now` when it was fetched
    turn: str  # the turn that fetched it


class SourceCache:
    """The per-thread store of fetched sources. See the module doc for the rules.

    fetch         ticket 2's `fetch_live_sources` (a test passes a counting fake)
    load_case     `load_case_sources`
    reader_model  builds the NOTAM reader's chat model when the model reads the NOTAMs (a test passes a fake chat model);
                  the default builds the configured one, and only then
    reader_store  the stored-readings file the reader uses when the run names none (a test passes a temporary one); the default is the reader's own file
    """

    def __init__(
        self,
        *,
        fetch: Fetcher = fetch_live_sources,
        load_case: CaseLoader = load_case_sources,
        reader_model: ReaderModel | None = None,
        reader_store: Path | None = None,
        settings: AgentSettings = SETTINGS,
    ):
        self._fetch = fetch
        self._load_case = load_case
        self._reader_model = reader_model
        self._reader_store = reader_store
        self.settings = settings
        self._reuse = timedelta(minutes=settings.source_reuse_minutes)
        self._entries: dict[tuple[Any, ...], _Entry] = {}
        self._locks: dict[tuple[Any, ...], tuple[threading.Lock, int]] = {}  # key -> (lock, calls using it); gone when the count reaches 0
        self._guard = threading.Lock()  # protects the two dicts; never held while fetching

    def reader_llm(self) -> BaseChatModel:
        """The NOTAM reader's chat model, built now. Raises ReaderConfigError when the reader is not configured."""
        return (self._reader_model or reader_config.build_llm)()

    def reader_store(self, ctx: RunContext | None) -> Path:
        """The stored-readings file for a run: the context's, else this one's, else the reader's default file."""
        if ctx is not None and ctx.reader_cache is not None:
            return ctx.reader_cache
        return self._reader_store if self._reader_store is not None else reader_config.reading_store()

    def forget(self, thread_id: str) -> None:
        """Drop everything kept for a thread. (Its locks need no dropping: a lock lives only while a call uses it.)"""
        with self._guard:
            for key in [key for key in self._entries if key[0] == thread_id]:
                del self._entries[key]

    @contextmanager
    def _locked(self, key: tuple[Any, ...]) -> Iterator[None]:
        """Hold the key's lock. Calls on one key share one lock; it is dropped when the last of them is done, so none outlives its calls."""
        with self._guard:
            lock, users = self._locks.get(key) or (threading.Lock(), 0)
            self._locks[key] = (lock, users + 1)
        try:
            with lock:
                yield
        finally:
            with self._guard:
                lock, users = self._locks[key]
                if users == 1:
                    del self._locks[key]
                else:
                    self._locks[key] = (lock, users - 1)

    def get(self, *, thread_id: str, turn: str, lat: float, lon: float, ctx: RunContext | None) -> Sources | SourcesError:
        """The thread's sources for this turn: the kept ones if they may be reused, else a fresh fetch (or case load)."""
        ctx = RunContext() if ctx is None else ctx
        now = run_now(ctx)
        horizon = fetch_horizon(now)
        case_part = (ctx.case, ctx.reader, ctx.reader_cache, ctx.drop) if ctx.case is not None else (None, None, None, ())
        key = (thread_id, lat, lon, horizon, *case_part)
        with self._locked(key):
            with self._guard:
                entry = self._entries.get(key)
            if entry is not None and self._may_reuse(entry, now, turn):
                return entry.sources
            made = self._make(lat, lon, ctx, now, horizon)
            if isinstance(made, SourcesError):
                return made
            with self._guard:
                for old, kept in list(self._entries.items()):
                    # this thread has moved on (another horizon or case setting), or the entry is too old for anyone to reuse
                    if old != key and (old[0] == thread_id or now - kept.stored_at >= self._reuse):
                        del self._entries[old]
                self._entries[key] = _Entry(made, now, turn)
            return made

    def _may_reuse(self, entry: _Entry, now: datetime, turn: str) -> bool:
        if not timedelta(0) <= now - entry.stored_at < self._reuse:
            return False
        failed = any(status.state is SourceState.FAILED for status in entry.sources.statuses)
        return not failed or entry.turn == turn

    def _make(self, lat: float, lon: float, ctx: RunContext, now: datetime, horizon: datetime) -> Sources | SourcesError:
        if ctx.case is None:
            forecast, airspace, pib = self._fetch(lat, lon, (_hour_floor(now), horizon), now)
            return Sources(forecast, airspace, pib, {}, {})
        return self._from_case(ctx.case, lat, lon, ctx)

    def _from_case(self, case: str, lat: float, lon: float, ctx: RunContext) -> Sources | SourcesError:
        fixture = ctx.reader == "fixture"
        try:
            loaded = self._load_case(case, with_readings=fixture)
        except SourceError as exc:
            return SourcesError(str(exc))
        case_now = loaded.case.now  # the case's own clock: its PIB, drops and readings are all as of then
        forecast, airspace, pib = apply_drops(ctx.drop, case_now, loaded.forecast, loaded.airspace, loaded.pib)
        return Sources(forecast, airspace, pib, loaded.readings if fixture else {}, {})
