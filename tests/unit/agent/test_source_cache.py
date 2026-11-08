"""SourceCache rules (G7, FR12, I2, FR9): the key, the reuse time, a failed source's life, the lock, case mode and its errors.

The fetcher is a counting fake put in place of `fetch_live_sources`; the clock is the run context's `now` and nothing else.
"""

import threading
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta

import pytest
from factories import airspace_fetch, forecast_failed, pib_result, status, weather_forecast
from langchain_core.messages import AIMessage, HumanMessage

from flysafelz.agent.settings import SETTINGS
from flysafelz.agent.sources import SourceCache, Sources, SourcesError, fetch_time, thread_id_of, turn_of
from flysafelz.agent.state import RunContext
from flysafelz.assess import fetch_horizon
from flysafelz.contracts import SourceName, SourceState
from flysafelz.sources import CaseSources, SourceError

LAT, LON = 53.802797, -0.086644
NOW = datetime(2026, 10, 31, 7, 30, tzinfo=UTC)
REUSE = timedelta(minutes=SETTINGS.source_reuse_minutes)
MINUTE = timedelta(minutes=1)
THREAD = "thread-1"
TURN_1, TURN_2 = "turn-1", "turn-2"


@dataclass
class CountingFetcher:
    """Stands in for `fetch_live_sources`. `weather` is the state of the weather it returns; `calls` is what it was asked, in order."""

    weather: SourceState = SourceState.OK
    calls: list[tuple[float, float, tuple[datetime, datetime], datetime]] = field(default_factory=list)
    hold: threading.Event | None = None  # when set, a call waits for it (up to a limit) before it answers
    entered: threading.Event = field(default_factory=threading.Event)

    def __call__(self, lat, lon, window, now):
        self.calls.append((lat, lon, window, now))
        self.entered.set()
        if self.hold is not None:
            assert self.hold.wait(timeout=10), "the test never released the fetcher"
        first, last = window
        forecast = forecast_failed() if self.weather is SourceState.FAILED else weather_forecast(first, last)
        return forecast, airspace_fetch(), pib_result()


def cache_with(fetcher: CountingFetcher, **kwargs) -> SourceCache:
    return SourceCache(fetch=fetcher, **kwargs)


def ask(cache: SourceCache, *, at: datetime = NOW, thread: str = THREAD, turn: str = TURN_1, lat: float = LAT, lon: float = LON) -> Sources:
    got = cache.get(thread_id=thread, turn=turn, lat=lat, lon=lon, ctx=RunContext(now=at))
    assert isinstance(got, Sources), got
    return got


# --- what a miss fetches --------------------------------------------------------------------------------------


def test_a_miss_fetches_once_for_the_whole_horizon_from_the_hour_we_are_in():
    fetcher = CountingFetcher()
    got = ask(cache_with(fetcher))
    assert fetcher.calls == [(LAT, LON, (datetime(2026, 10, 31, 7, 0, tzinfo=UTC), fetch_horizon(NOW)), NOW)]
    assert fetch_horizon(NOW) == datetime(2026, 11, 8, 0, 0, tzinfo=UTC)
    assert (got.readings, got.unread) == ({}, {})  # live mode: the NOTAM tools read for their own window


def test_the_three_sources_come_from_the_one_fetch_and_report_their_own_status():
    got = ask(cache_with(CountingFetcher()))
    assert [got.status(name).source for name in SourceName] == list(SourceName)
    assert [s.source for s in got.statuses] == [SourceName.WEATHER, SourceName.AIRSPACE, SourceName.NOTAMS]
    assert got.status(SourceName.WEATHER) is got.forecast.status and got.status(SourceName.NOTAMS) is got.pib.status


# --- the key --------------------------------------------------------------------------------------------------


def test_two_calls_in_one_turn_share_one_fetch_and_one_fetch_time():
    fetcher = CountingFetcher()
    cache = cache_with(fetcher)
    first, second = ask(cache), ask(cache)
    assert len(fetcher.calls) == 1
    assert first is second
    assert fetch_time(first.statuses) == fetch_time(second.statuses)


def test_another_thread_has_its_own_fetch():
    fetcher = CountingFetcher()
    cache = cache_with(fetcher)
    ask(cache, thread="a")
    ask(cache, thread="b")
    ask(cache, thread="a")
    ask(cache, thread="b")
    assert len(fetcher.calls) == 2


@pytest.mark.parametrize(("lat", "lon"), [(LAT + 0.01, LON), (LAT, LON + 0.01)])
def test_another_place_in_the_same_thread_is_a_new_fetch(lat, lon):
    fetcher = CountingFetcher()
    cache = cache_with(fetcher)
    ask(cache)
    ask(cache, lat=lat, lon=lon)
    assert [(call[0], call[1]) for call in fetcher.calls] == [(LAT, LON), (lat, lon)]


def test_a_thread_keeps_one_entry_so_going_back_to_an_earlier_place_fetches_again():
    fetcher = CountingFetcher()
    cache = cache_with(fetcher)
    ask(cache)
    ask(cache, lat=LAT + 0.01)
    ask(cache)  # the first place was dropped when the thread moved on
    assert len(fetcher.calls) == 3


def test_the_window_the_model_asks_for_is_not_in_the_key_a_later_turn_in_reuse_time_hits():
    fetcher = CountingFetcher()
    cache = cache_with(fetcher)
    ask(cache, turn=TURN_1)
    ask(cache, turn=TURN_2, at=NOW + MINUTE)
    assert len(fetcher.calls) == 1


def test_a_new_horizon_after_london_midnight_fetches_again_even_inside_the_reuse_time():
    """fetch_horizon(now) moves at midnight in London: 23:55Z on 31 Oct (GMT) and 00:05Z on 1 Nov are ten minutes apart but not one horizon."""
    before, after = datetime(2026, 10, 31, 23, 55, tzinfo=UTC), datetime(2026, 11, 1, 0, 5, tzinfo=UTC)
    assert fetch_horizon(before) != fetch_horizon(after)
    assert after - before < REUSE
    fetcher = CountingFetcher()
    cache = cache_with(fetcher)
    ask(cache, at=before)
    ask(cache, at=after)
    assert [call[3] for call in fetcher.calls] == [before, after]
    assert [call[2][1] for call in fetcher.calls] == [fetch_horizon(before), fetch_horizon(after)]


def test_the_same_horizon_inside_the_reuse_time_does_not_fetch_again():
    first, second = datetime(2026, 10, 31, 22, 0, tzinfo=UTC), datetime(2026, 10, 31, 22, 10, tzinfo=UTC)
    assert fetch_horizon(first) == fetch_horizon(second)
    fetcher = CountingFetcher()
    cache = cache_with(fetcher)
    ask(cache, at=first)
    ask(cache, at=second)
    assert len(fetcher.calls) == 1


# --- the reuse time -------------------------------------------------------------------------------------------


def test_the_reuse_time_is_the_settings_value():
    assert SETTINGS.source_reuse_minutes == 15


@pytest.mark.parametrize(
    ("later", "fetches"),
    [
        (timedelta(0), 1),
        (REUSE - MINUTE, 1),
        (REUSE - timedelta(seconds=1), 1),
        (REUSE, 2),
        (REUSE + MINUTE, 2),
        (timedelta(hours=1), 2),
        (-MINUTE, 2),  # a clock that went backwards is not trusted
    ],
    ids=["same moment", "14 min", "14 min 59 s", "exactly 15 min", "16 min", "an hour", "clock backwards"],
)
def test_a_next_turn_reuses_the_fetch_until_the_reuse_time_is_up(later, fetches):
    fetcher = CountingFetcher()
    cache = cache_with(fetcher)
    ask(cache, turn=TURN_1)
    ask(cache, turn=TURN_2, at=NOW + later)
    assert len(fetcher.calls) == fetches


def test_the_reuse_time_follows_the_settings_it_was_given():
    short = SETTINGS.model_copy(update={"source_reuse_minutes": 5})
    fetcher = CountingFetcher()
    cache = cache_with(fetcher, settings=short)
    ask(cache)
    ask(cache, at=NOW + timedelta(minutes=4))
    ask(cache, at=NOW + timedelta(minutes=6))
    assert len(fetcher.calls) == 2


def test_the_reuse_time_is_counted_from_the_fetch_not_from_the_last_use():
    fetcher = CountingFetcher()
    cache = cache_with(fetcher)
    ask(cache)
    ask(cache, at=NOW + timedelta(minutes=10))  # a hit: does not extend the entry
    ask(cache, at=NOW + timedelta(minutes=20))
    assert len(fetcher.calls) == 2


def test_the_clock_is_the_contexts_not_the_wall_clock():
    """The frozen 2026 `now` is months from the real date; if the wall clock were used the entry would be long expired."""
    fetcher = CountingFetcher()
    cache = cache_with(fetcher)
    ask(cache)
    ask(cache, at=NOW + MINUTE)
    assert len(fetcher.calls) == 1
    assert fetcher.calls[0][3] == NOW


# --- a failed source ------------------------------------------------------------------------------------------


def test_a_failed_weather_status_is_reused_within_its_turn():
    fetcher = CountingFetcher(weather=SourceState.FAILED)
    cache = cache_with(fetcher)
    first = ask(cache, turn=TURN_1)
    second = ask(cache, turn=TURN_1, at=NOW + MINUTE)
    assert first.status(SourceName.WEATHER).state is SourceState.FAILED
    assert first is second
    assert len(fetcher.calls) == 1


def test_a_failed_weather_status_is_fetched_again_in_the_next_turn_inside_the_reuse_time():
    fetcher = CountingFetcher(weather=SourceState.FAILED)
    cache = cache_with(fetcher)
    ask(cache, turn=TURN_1)
    fetcher.weather = SourceState.OK
    second = ask(cache, turn=TURN_2, at=NOW + MINUTE)
    assert len(fetcher.calls) == 2
    assert second.status(SourceName.WEATHER).state is SourceState.OK
    third = ask(cache, turn=TURN_2, at=NOW + 2 * MINUTE)  # the good fetch is kept now
    assert third is second and len(fetcher.calls) == 2


def test_a_good_fetch_is_kept_across_turns():
    fetcher = CountingFetcher()
    cache = cache_with(fetcher)
    ask(cache, turn=TURN_1)
    ask(cache, turn=TURN_2, at=NOW + MINUTE)
    assert len(fetcher.calls) == 1


@pytest.mark.parametrize("failing", [SourceName.AIRSPACE, SourceName.NOTAMS])
def test_any_failed_source_gets_the_same_treatment(failing):
    class OneFailure(CountingFetcher):
        def __call__(self, lat, lon, window, now):
            forecast, airspace, pib = super().__call__(lat, lon, window, now)
            if failing is SourceName.AIRSPACE:
                airspace = airspace_fetch(state=SourceState.FAILED)
            else:
                pib = pib_result(state=SourceState.FAILED)
            return forecast, airspace, pib

    fetcher = OneFailure()
    cache = cache_with(fetcher)
    ask(cache, turn=TURN_1)
    ask(cache, turn=TURN_1, at=NOW + MINUTE)
    assert len(fetcher.calls) == 1
    ask(cache, turn=TURN_2, at=NOW + 2 * MINUTE)
    assert len(fetcher.calls) == 2


def test_a_stale_source_is_not_a_failed_one():
    class StaleNotams(CountingFetcher):
        def __call__(self, lat, lon, window, now):
            forecast, airspace, _ = super().__call__(lat, lon, window, now)
            return forecast, airspace, pib_result(state=SourceState.STALE)

    fetcher = StaleNotams()
    cache = cache_with(fetcher)
    ask(cache, turn=TURN_1)
    ask(cache, turn=TURN_2, at=NOW + MINUTE)
    assert len(fetcher.calls) == 1


# --- the lock -------------------------------------------------------------------------------------------------


def test_forget_drops_a_threads_entries_and_only_that_threads():
    fetcher = CountingFetcher()
    cache = cache_with(fetcher)
    ask(cache, thread="a")
    ask(cache, thread="b")
    cache.forget("a")
    ask(cache, thread="a")
    ask(cache, thread="b")
    assert len(fetcher.calls) == 3


def test_parallel_calls_of_one_thread_share_one_fetch():
    """Eight OS threads ask at once while the one fetch is held open; only the first fetches, the rest wait for it and share it."""
    release = threading.Event()
    fetcher = CountingFetcher(hold=release)
    cache = cache_with(fetcher)
    asked = threading.Barrier(9)
    got: list[Sources | SourcesError] = []

    def worker() -> None:
        asked.wait(timeout=10)
        got.append(cache.get(thread_id=THREAD, turn=TURN_1, lat=LAT, lon=LON, ctx=RunContext(now=NOW)))

    workers = [threading.Thread(target=worker) for _ in range(8)]
    for w in workers:
        w.start()
    asked.wait(timeout=10)
    assert fetcher.entered.wait(timeout=10)
    release.set()
    for w in workers:
        w.join(timeout=10)
        assert not w.is_alive()
    assert len(fetcher.calls) == 1
    assert len(got) == 8 and all(item is got[0] for item in got)
    assert cache._locks == {}  # the last call out dropped the shared lock


def test_parallel_calls_of_two_threads_fetch_separately():
    release = threading.Event()
    fetcher = CountingFetcher(hold=release)
    cache = cache_with(fetcher)
    results: dict[str, Sources | SourcesError] = {}

    def worker(thread: str) -> None:
        results[thread] = cache.get(thread_id=thread, turn=TURN_1, lat=LAT, lon=LON, ctx=RunContext(now=NOW))

    workers = [threading.Thread(target=worker, args=(name,)) for name in ("a", "b")]
    for w in workers:
        w.start()
    release.set()
    for w in workers:
        w.join(timeout=10)
    assert len(fetcher.calls) == 2
    assert results["a"] is not results["b"]


# --- memory stays bounded -------------------------------------------------------------------------------------


def test_a_store_drops_the_expired_entries_of_other_threads():
    fetcher = CountingFetcher()
    cache = cache_with(fetcher)
    for name in ("a", "b", "c"):
        ask(cache, thread=name)
    assert len(cache._entries) == 3
    ask(cache, thread="d", at=NOW + REUSE)  # a, b and c are exactly the reuse time old: no one could reuse them
    assert [key[0] for key in cache._entries] == ["d"]
    ask(cache, thread="a", at=NOW)  # a's entry is really gone: with it kept, this (a clock going back) would have been a hit
    assert len(fetcher.calls) == 5


def test_a_store_keeps_the_entries_of_other_threads_that_can_still_be_reused():
    fetcher = CountingFetcher()
    cache = cache_with(fetcher)
    ask(cache, thread="a")
    ask(cache, thread="b", at=NOW + REUSE - MINUTE)
    assert sorted(key[0] for key in cache._entries) == ["a", "b"]
    ask(cache, thread="a", turn=TURN_2, at=NOW + REUSE - MINUTE)
    assert len(fetcher.calls) == 2


def test_a_run_of_one_shot_threads_never_holds_more_than_the_reuse_time_of_them():
    cache = cache_with(CountingFetcher())
    for i in range(100):
        ask(cache, thread=f"click-{i}", at=NOW + i * MINUTE)
    reuse_minutes = SETTINGS.source_reuse_minutes
    assert len(cache._entries) == reuse_minutes
    assert cache._locks == {}


def test_a_lock_lives_only_while_calls_use_it():
    release = threading.Event()
    fetcher = CountingFetcher(hold=release)
    cache = cache_with(fetcher)
    workers = [threading.Thread(target=lambda: cache.get(thread_id=THREAD, turn=TURN_1, lat=LAT, lon=LON, ctx=RunContext(now=NOW))) for _ in range(2)]
    workers[0].start()
    assert fetcher.entered.wait(timeout=10)
    workers[1].start()  # waits on the first call's lock
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline and sum(users for _, users in cache._locks.values()) < 2:
        time.sleep(0.01)
    assert len(cache._locks) == 1 and sum(users for _, users in cache._locks.values()) == 2  # one lock, shared by both calls
    cache.forget(THREAD)
    assert len(cache._locks) == 1  # forget does not pull a lock from under a running call
    release.set()
    for w in workers:
        w.join(timeout=10)
        assert not w.is_alive()
    assert cache._locks == {}


def test_a_fetch_that_raises_still_releases_the_lock():
    def broken(*args):
        raise RuntimeError("network down")

    cache = SourceCache(fetch=broken)
    with pytest.raises(RuntimeError, match="network down"):
        cache.get(thread_id=THREAD, turn=TURN_1, lat=LAT, lon=LON, ctx=RunContext(now=NOW))
    assert cache._locks == {} and cache._entries == {}


# --- case mode ------------------------------------------------------------------------------------------------


class NoFetch:
    """A live fetcher that must never be called: in case mode nothing is fetched (F5, F9)."""

    def __call__(self, *args):
        raise AssertionError("a case-mode run fetched live")


def test_an_unknown_case_is_the_source_errors_line_not_an_exception_and_nothing_is_fetched():
    attempts = []

    def load(name, **kwargs):
        attempts.append((name, kwargs))
        raise SourceError("no case called nope: fixtures/nope/ does not exist")

    cache = SourceCache(fetch=NoFetch(), load_case=load)
    got = cache.get(thread_id=THREAD, turn=TURN_1, lat=LAT, lon=LON, ctx=RunContext(now=NOW, case="nope"))
    assert got == SourcesError("no case called nope: fixtures/nope/ does not exist")
    assert attempts == [("nope", {"with_readings": True})]
    cache.get(thread_id=THREAD, turn=TURN_1, lat=LAT, lon=LON, ctx=RunContext(now=NOW, case="nope"))
    assert len(attempts) == 2  # an error is not kept


def test_a_case_fixture_run_loads_the_case_with_its_readings_and_never_fetches():
    loaded = CaseSources(
        case=_case_stub(),
        forecast=weather_forecast(NOW, NOW + timedelta(hours=3)),
        airspace=airspace_fetch(),
        pib=pib_result(),
        readings={},
    )
    seen = []

    def load(name, **kwargs):
        seen.append((name, kwargs))
        return loaded

    cache = SourceCache(fetch=NoFetch(), load_case=load)
    ctx = RunContext(now=NOW, case="any")
    first = cache.get(thread_id=THREAD, turn=TURN_1, lat=LAT, lon=LON, ctx=ctx)
    second = cache.get(thread_id=THREAD, turn=TURN_2, lat=LAT, lon=LON, ctx=ctx)
    assert isinstance(first, Sources) and first is second
    assert seen == [("any", {"with_readings": True})]
    assert first.forecast is loaded.forecast


@pytest.mark.parametrize(
    "changes",
    [{"reader": "live"}, {"drop": ("pib",)}, {"case": "other"}],
    ids=["reader", "drop", "case"],
)
def test_the_case_settings_are_part_of_the_key_so_a_fixture_and_a_live_run_never_share_readings(changes):
    loads = []

    def load(name, **kwargs):
        loads.append((name, kwargs))
        return CaseSources(case=_case_stub(), forecast=weather_forecast(NOW, NOW + timedelta(hours=3)), airspace=airspace_fetch(), pib=pib_result(state=SourceState.FAILED), readings={})

    cache = SourceCache(fetch=NoFetch(), load_case=load, reader_model=_no_model)
    base = RunContext(now=NOW, case="any")
    changed = RunContext(now=NOW, **{"case": "any", **changes})
    cache.get(thread_id=THREAD, turn=TURN_1, lat=LAT, lon=LON, ctx=base)
    cache.get(thread_id=THREAD, turn=TURN_1, lat=LAT, lon=LON, ctx=changed)
    assert len(loads) == 2


def _no_model():
    raise AssertionError("the reader's model was built")


def _case_stub():
    """The part of a Case that case-mode reading needs: its own `now`."""
    from flysafelz.capture import Case

    return Case.model_validate({"lat": LAT, "lon": LON, "now": NOW, "window": {"start": NOW, "end": NOW + timedelta(hours=3)}})


# --- the helpers the tools share ------------------------------------------------------------------------------


def test_fetch_time_is_the_oldest_fetched_at_among_the_sources_used():
    old, newer = NOW - timedelta(minutes=9), NOW
    statuses = [
        status(SourceName.WEATHER).model_copy(update={"fetched_at": newer}),
        status(SourceName.AIRSPACE).model_copy(update={"fetched_at": old}),
        status(SourceName.NOTAMS).model_copy(update={"fetched_at": newer}),
    ]
    assert fetch_time(statuses) == old
    assert fetch_time(statuses[:1]) == newer
    assert fetch_time(iter(statuses[::-1])) == old


def test_fetch_time_of_sources_with_no_time_raises():
    with pytest.raises(ValueError, match="fetch time"):
        fetch_time([])
    with pytest.raises(ValueError, match="fetch time"):
        fetch_time([status(SourceName.WEATHER).model_copy(update={"fetched_at": None})])


def test_every_tool_in_a_turn_reports_the_same_fetch_time():
    cache = cache_with(CountingFetcher())
    times = {fetch_time(ask(cache, at=NOW + n * MINUTE).statuses) for n in range(3)}
    assert len(times) == 1


def test_the_thread_id_comes_from_the_runs_configuration():
    assert thread_id_of({"configurable": {"thread_id": "t-9"}}) == "t-9"
    assert thread_id_of({"configurable": {"thread_id": 7}}) == "7"


@pytest.mark.parametrize("config", [{}, {"configurable": {}}, {"configurable": {"thread_id": ""}}, {"configurable": None}])
def test_a_run_without_a_thread_id_is_refused(config):
    with pytest.raises(ValueError, match="thread_id"):
        thread_id_of(config)


def test_the_turn_is_the_id_of_the_pilots_newest_message():
    first = HumanMessage("can I fly", id="m1")
    reply = AIMessage("which window?", id="m2")
    second = HumanMessage("tomorrow", id="m3")
    assert turn_of([first]) == "m1"
    assert turn_of([first, reply]) == "m1"
    assert turn_of([first, reply, second]) == "m3"
    assert turn_of([first, reply, second, AIMessage("ok", id="m4")]) == "m3"


def test_a_pilot_message_with_no_id_is_counted_and_no_pilot_message_is_the_empty_turn():
    assert turn_of([HumanMessage("a"), AIMessage("b"), HumanMessage("c")]) == "question-2"
    assert turn_of([]) == ""
    assert turn_of([AIMessage("hello")]) == ""
