"""PR-4.F1: the source cache in case mode on the committed Blue Mill and West Sleekburn cases (G7, I3).

Sockets are off for the whole suite. The only fakes are the chat model (the reader's) and a live fetcher that fails the test if it is ever called.
"""

import builtins
import io
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

import cases
import pytest
from reader_replies import fixture_readings, model_reply
from reader_set_helpers import ScriptedChat
from unit.reader_helpers import blue_mill_notams

from flysafelz.agent.redaction import redact_artifact
from flysafelz.agent.sources import SourceCache, Sources, SourcesError
from flysafelz.agent.state import RunContext
from flysafelz.assess import assess, fetch_horizon
from flysafelz.contracts import Assessment, SourceName, SourceState, Verdict
from flysafelz.reader import config as reader_config
from flysafelz.reader.reader_set import reader_set
from flysafelz.sources import SourceError, load_case_sources
from flysafelz.verdict_sheet import DROPPED_REASON

BLUE_MILL, WEST_SLEEKBURN = cases.BLUE_MILL, cases.WEST_SLEEKBURN
NOW = datetime(2026, 10, 31, 7, 30, tzinfo=UTC)
REAL_STORE = reader_config._repository_root() / reader_config.STORE_NAME  # data/readings.sqlite: no test may touch it
THREAD, TURN = "thread-1", "turn-1"


class NoFetch:
    """The live fetcher: in case mode it must never be called (F5, F9)."""

    def __call__(self, *args):
        raise AssertionError("a case-mode run fetched live")


def get(cache: SourceCache, case: str, **context) -> Sources | SourcesError:
    loaded = load_case_sources(case, with_readings=False)
    return cache.get(thread_id=THREAD, turn=TURN, lat=loaded.case.lat, lon=loaded.case.lon, ctx=RunContext(now=NOW, case=case, **context))


def sources_of(cache: SourceCache, case: str, **context) -> Sources:
    got = get(cache, case, **context)
    assert isinstance(got, Sources), got
    return got


def assessed(case: str, sources: Sources) -> Assessment:
    c = load_case_sources(case).case
    return assess(c.lat, c.lon, now=c.now, forecast=sources.forecast, airspace=sources.airspace, pib=sources.pib, readings=sources.readings, reader_status=sources.unread, window=(c.window.start, c.window.end), live=False)


def blue_mill_model() -> ScriptedChat:
    """A fake reader model: the two NOTAMs readings.yaml covers answer as readings.yaml says; any other gets the generated good reply."""
    by_id, readings = blue_mill_notams(), fixture_readings()
    scripts = {key: [model_reply(by_id[key.replace("_", "/")], readings[key.replace("_", "/")])] for key in ("H7171_26", "J3857_26")}
    return ScriptedChat(list(by_id.values()), **scripts)


def sqlite_files(folder: Path) -> list[Path]:
    return sorted(path for path in folder.rglob("*") if path.is_file() and path.suffix in {".sqlite", ".sqlite3", ".db"})


def stamp(path: Path) -> tuple[bool, int | None, int | None, bytes | None]:
    """(exists, mtime_ns, size, bytes): what 'unchanged' means for the real store."""
    if not path.exists():
        return (False, None, None, None)
    info = path.stat()
    return (True, info.st_mtime_ns, info.st_size, path.read_bytes())


# --- fixture reader (the default): exactly what load_case_sources gives ---------------------------------------


@pytest.mark.parametrize("case", [BLUE_MILL, WEST_SLEEKBURN])
def test_a_case_comes_through_the_cache_as_load_case_sources_gives_it(case):
    expected = load_case_sources(case)
    got = sources_of(SourceCache(fetch=NoFetch()), case)
    assert got.forecast == expected.forecast
    assert got.airspace == expected.airspace
    assert got.pib == expected.pib
    assert got.readings == expected.readings
    assert got.unread == {}


def test_blue_mills_readings_are_readings_yaml():
    got = sources_of(SourceCache(fetch=NoFetch()), BLUE_MILL)
    assert set(got.readings) == {"H7171/26", "J3857/26"}
    assert got.readings == fixture_readings()


def test_west_sleekburn_has_no_readings_yaml_to_read_so_it_has_no_readings():
    got = sources_of(SourceCache(fetch=NoFetch()), WEST_SLEEKBURN)
    assert got.readings == load_case_sources(WEST_SLEEKBURN).readings


def test_the_fixture_default_is_the_same_whether_the_reader_is_named_or_left_out():
    left_out = sources_of(SourceCache(fetch=NoFetch()), BLUE_MILL)
    named = sources_of(SourceCache(fetch=NoFetch()), BLUE_MILL, reader="fixture")
    assert (left_out.forecast, left_out.airspace, left_out.pib, left_out.readings) == (named.forecast, named.airspace, named.pib, named.readings)


def test_a_fixture_run_calls_no_reader_and_builds_no_model():
    def no_model():
        raise AssertionError("the reader's model was built in a fixture run")

    cache = SourceCache(fetch=NoFetch(), reader_model=no_model)
    assert sources_of(cache, BLUE_MILL).readings


def test_the_case_clock_is_the_cases_when_the_run_sets_none():
    """No `now` on the context: the cache uses the wall clock only for its reuse time. The case's PIB is judged at the case's own now."""
    loaded = load_case_sources(BLUE_MILL)
    got = SourceCache(fetch=NoFetch()).get(thread_id=THREAD, turn=TURN, lat=loaded.case.lat, lon=loaded.case.lon, ctx=RunContext(case=BLUE_MILL))
    assert isinstance(got, Sources) and got.pib == loaded.pib


def test_an_unknown_case_is_the_source_errors_line():
    with pytest.raises(SourceError) as expected:
        load_case_sources("no_such_case")
    got = SourceCache(fetch=NoFetch()).get(thread_id=THREAD, turn=TURN, lat=53.8, lon=-0.08, ctx=RunContext(now=NOW, case="no_such_case"))
    assert got == SourcesError(str(expected.value))
    assert isinstance(got, SourcesError) and "no_such_case" in got.text and "\n" not in got.text


def test_the_assessment_of_a_cached_case_is_the_assessment_of_the_loaded_one():
    """Everything but the source statuses equals ticket 2's `cases.run`; the NOTAM status also carries the archive commit `load_case_sources` adds."""
    got = assessed(BLUE_MILL, sources_of(SourceCache(fetch=NoFetch()), BLUE_MILL))
    expected = cases.run(BLUE_MILL)
    assert got.model_dump(exclude={"source_statuses"}) == expected.model_dump(exclude={"source_statuses"})
    assert [(s.source, s.state, s.reason) for s in got.source_statuses] == [(s.source, s.state, s.reason) for s in expected.source_statuses]


# --- reader: live ---------------------------------------------------------------------------------------------


def horizon_ids() -> set[str]:
    """The NOTAM ids `read_for_lz` selects for Blue Mill over the whole horizon, found with the reader-set rule itself."""
    c = load_case_sources(BLUE_MILL, with_readings=False)
    chosen = reader_set(c.pib.notams, (c.case.lat, c.case.lon), (c.case.now.replace(minute=0), fetch_horizon(c.case.now)))
    return {notam.id for notam in chosen}


def test_a_live_reader_run_reads_the_selected_notams_and_nothing_else(tmp_path):
    llm = blue_mill_model()
    store = tmp_path / "readings.sqlite"
    got = sources_of(SourceCache(fetch=NoFetch(), reader_model=lambda: llm), BLUE_MILL, reader="live", reader_cache=store)
    assert horizon_ids() >= {"H7171/26", "J3857/26"}
    assert set(llm.calls) == horizon_ids()
    assert {"H7171/26", "J3857/26"} <= set(got.readings)
    assert set(got.readings) | set(got.unread) == horizon_ids()
    assert set(got.readings).isdisjoint(got.unread)
    whole = load_case_sources(BLUE_MILL)
    assert (got.forecast, got.airspace, got.pib) == (whole.forecast, whole.airspace, whole.pib)


def test_a_live_reader_run_never_opens_readings_yaml(tmp_path, monkeypatch):
    llm = blue_mill_model()  # built first: the fake's answers are made from readings.yaml, and that is the test's own read, not the code's
    opened: list[str] = []
    real_open, real_io_open, real_path_open = builtins.open, io.open, Path.open
    real_read_text, real_read_bytes = Path.read_text, Path.read_bytes

    def noting(real):
        def wrapper(target, *args, **kwargs):
            opened.append(str(target))
            return real(target, *args, **kwargs)

        return wrapper

    def noting_method(real):
        def wrapper(self, *args, **kwargs):
            opened.append(str(self))
            return real(self, *args, **kwargs)

        return wrapper

    monkeypatch.setattr(builtins, "open", noting(real_open))
    monkeypatch.setattr(io, "open", noting(real_io_open))
    monkeypatch.setattr(Path, "open", noting_method(real_path_open))
    monkeypatch.setattr(Path, "read_text", noting_method(real_read_text))
    monkeypatch.setattr(Path, "read_bytes", noting_method(real_read_bytes))
    from flysafelz import sources as case_sources

    def fixture_reader_must_not_run(*args, **kwargs):
        raise AssertionError("the fixture reader ran in a live reader run")

    monkeypatch.setattr(case_sources, "_readings", fixture_reader_must_not_run)
    sources_of(SourceCache(fetch=NoFetch(), reader_model=lambda: llm), BLUE_MILL, reader="live", reader_cache=tmp_path / "readings.sqlite")
    assert opened, "the spy saw no file at all: it is not watching"
    assert any(path.endswith("pib.xml") for path in opened)  # the spy works: the case's other files were read
    assert not [path for path in opened if path.endswith("readings.yaml")]


def test_a_live_reader_run_writes_only_the_given_store_and_leaves_the_real_one_alone(tmp_path):
    before = stamp(REAL_STORE)
    store = tmp_path / "somewhere" / "mine.sqlite"
    store.parent.mkdir()
    llm = blue_mill_model()
    sources_of(SourceCache(fetch=NoFetch(), reader_model=lambda: llm), BLUE_MILL, reader="live", reader_cache=store)
    assert sqlite_files(tmp_path) == [store]
    assert stamp(REAL_STORE) == before
    with sqlite3.connect(store) as db:
        stored = db.execute("select count(*) from sqlite_master where type = 'table'").fetchone()[0]
    assert stored >= 1


def test_a_second_run_with_the_same_store_is_all_cache_hits_and_reads_nothing(tmp_path):
    store = tmp_path / "readings.sqlite"
    first_model, second_model = blue_mill_model(), blue_mill_model()
    first = sources_of(SourceCache(fetch=NoFetch(), reader_model=lambda: first_model), BLUE_MILL, reader="live", reader_cache=store)
    assert first_model.calls
    second = sources_of(SourceCache(fetch=NoFetch(), reader_model=lambda: second_model), BLUE_MILL, reader="live", reader_cache=store)
    assert second_model.calls == []
    assert second.readings == first.readings and second.unread == first.unread


def test_a_live_reader_run_without_a_store_uses_the_default_one(tmp_path, monkeypatch):
    default = tmp_path / "default" / "readings.sqlite"
    default.parent.mkdir()
    monkeypatch.setattr(reader_config, "reading_store", lambda: default)
    before = stamp(REAL_STORE)
    llm = blue_mill_model()
    sources_of(SourceCache(fetch=NoFetch(), reader_model=lambda: llm), BLUE_MILL, reader="live")
    assert sqlite_files(tmp_path) == [default]
    assert stamp(REAL_STORE) == before


def test_the_reader_model_is_built_only_in_a_live_run_and_only_when_there_are_notams_to_read(tmp_path):
    built = []

    def model():
        built.append(1)
        return blue_mill_model()

    cache = SourceCache(fetch=NoFetch(), reader_model=model)
    sources_of(cache, BLUE_MILL, reader="fixture")
    sources_of(cache, BLUE_MILL, reader="live", reader_cache=tmp_path / "a.sqlite", drop=("pib",))
    assert built == []
    sources_of(cache, BLUE_MILL, reader="live", reader_cache=tmp_path / "b.sqlite")
    assert built == [1]


def test_a_live_run_with_the_reader_not_configured_gives_the_readers_line_not_an_exception(tmp_path):
    """No READER_* settings (the suite clears them): the reader's own one line, naming each missing one, reaches the model as the tool text."""
    got = get(SourceCache(fetch=NoFetch()), BLUE_MILL, reader="live", reader_cache=tmp_path / "readings.sqlite")
    assert isinstance(got, SourcesError)
    assert "READER_" in got.text and "\n" not in got.text
    assert sqlite_files(tmp_path) == []


def test_fixture_and_live_runs_in_one_thread_never_share_readings(tmp_path):
    llm = blue_mill_model()
    cache = SourceCache(fetch=NoFetch(), reader_model=lambda: llm)
    from_file = sources_of(cache, BLUE_MILL, reader="fixture")
    from_reader = sources_of(cache, BLUE_MILL, reader="live", reader_cache=tmp_path / "readings.sqlite")
    again = sources_of(cache, BLUE_MILL, reader="fixture")
    assert from_file is not from_reader
    assert llm.calls and again.readings == from_file.readings
    assert again is not from_file  # the thread keeps one entry: the live run replaced the fixture one


# --- drop -----------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("dropped", "name"),
    [("pib", SourceName.NOTAMS), ("weather", SourceName.WEATHER), ("airspace", SourceName.AIRSPACE)],
)
def test_a_dropped_source_is_failed_and_named_and_no_hour_it_could_decide_is_left_as_a_verdict(dropped, name):
    baseline = assessed(BLUE_MILL, sources_of(SourceCache(fetch=NoFetch()), BLUE_MILL))
    got = assessed(BLUE_MILL, sources_of(SourceCache(fetch=NoFetch()), BLUE_MILL, drop=(dropped,)))
    statuses = {status.source: status for status in got.source_statuses}
    assert (statuses[name].state, statuses[name].reason) == (SourceState.FAILED, DROPPED_REASON)
    assert all(status.state is not SourceState.FAILED for source, status in statuses.items() if source is not name)
    verdicts = {segment.verdict for hour in got.hours for segment in hour.segments}
    assert verdicts == {Verdict.NO_GO, Verdict.UNVERIFIED}  # NO_GO stays where something else (the dark, a blocker) already forbids it
    assert Verdict.PILOT_DISCRETION in {segment.verdict for hour in baseline.hours for segment in hour.segments}  # what the drop took away


def test_a_dropped_source_is_part_of_the_key():
    cache = SourceCache(fetch=NoFetch())
    whole = sources_of(cache, BLUE_MILL)
    dropped = sources_of(cache, BLUE_MILL, drop=("pib",))
    assert whole.pib.status.state is SourceState.OK and dropped.pib.status.state is SourceState.FAILED


def test_nothing_is_dropped_from_the_cached_case_sources_themselves():
    """apply_drops returns new objects: dropping in one run leaves what the next run loads whole."""
    cache = SourceCache(fetch=NoFetch())
    sources_of(cache, BLUE_MILL, drop=("pib", "weather", "airspace"))
    again = sources_of(SourceCache(fetch=NoFetch()), BLUE_MILL)
    assert again.status(SourceName.WEATHER).state is SourceState.OK and again.pib.notams


def test_drop_composes_with_the_live_reader_and_a_dropped_pib_means_no_reader_call(tmp_path):
    llm = blue_mill_model()
    store = tmp_path / "readings.sqlite"
    got = sources_of(SourceCache(fetch=NoFetch(), reader_model=lambda: llm), BLUE_MILL, reader="live", reader_cache=store, drop=("pib",))
    assert llm.calls == []
    assert not store.exists()
    assert got.readings == {} and got.unread == {}
    assert got.pib.status.reason == DROPPED_REASON


@pytest.mark.parametrize("dropped", ["weather", "airspace"])
def test_drop_of_another_source_with_the_live_reader_still_reads(tmp_path, dropped):
    llm = blue_mill_model()
    got = sources_of(SourceCache(fetch=NoFetch(), reader_model=lambda: llm), BLUE_MILL, reader="live", reader_cache=tmp_path / "readings.sqlite", drop=(dropped,))
    assert set(llm.calls) == horizon_ids()
    assert got.readings


# --- phone numbers --------------------------------------------------------------------------------------------


def test_blue_mills_contact_number_reaches_the_model_only_as_phone_1():
    sources = sources_of(SourceCache(fetch=NoFetch()), BLUE_MILL)
    number = fixture_readings()["J3857/26"].contact.value.removeprefix("Flylogix ")
    assert number == "07784 546062"
    assessment = assessed(BLUE_MILL, sources)
    assert number in assessment.model_dump_json()  # the cache and assess hold the real one (it never leaves the server) ...
    redacted, phone_map = redact_artifact(assessment, {})
    dumped = redacted.model_dump_json()
    assert phone_map == {"<PHONE_1>": number}  # ... and the one boundary hides it
    assert number not in dumped and "07784546062" not in dumped.replace(" ", "")
    assert "phone Flylogix <PHONE_1> and fly only if they agree" in dumped
    raw_e, raw_map = redact_artifact(next(n for n in sources.pib.notams if n.id == "J3857/26"), {})
    assert raw_map == {"<PHONE_1>": number}
    assert "07784" not in raw_e.item_e and "<PHONE_1>" in raw_e.item_e
