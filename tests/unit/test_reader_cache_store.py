"""G9: the store and the load. Round trip, semantic-only rows, area and schedule rebuilt, what is never written, bad rows
and bad files, bypass, concurrency.

Every test goes through read_notam (the fake model) and then store / lookup / read_cached on a database under tmp_path.
"""

import json
import logging
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import pytest
from cache_helpers import fixture_reply
from reader_helpers import FAKE_MODEL, Steps, blue_mill_notams, reply_fields

from flysafelz import notam_area
from flysafelz.contracts import Circle
from flysafelz.reader import prompt as prompt_module
from flysafelz.reader.cache import cache_key, key_text, lookup, record_failure, store, was_failed
from flysafelz.reader.cached_read import model_calls, read_cached
from flysafelz.reader.contract import ReaderOutput, ReaderStatus
from flysafelz.reader.read import read_notam

NOTAMS = blue_mill_notams()
H7171 = NOTAMS["H7171/26"]
J3857 = NOTAMS["J3857/26"]
NO_CONTACT = {"kind": "none", "value": None}


def read_h7171(**overrides):
    return read_notam(H7171, Steps({**fixture_reply("H7171/26"), **overrides}))


def table(path) -> list[tuple[str, str, str]]:
    with sqlite3.connect(path) as connection:
        return connection.execute("SELECT key, json, created_at FROM readings ORDER BY key").fetchall()


def corrupt_the_row(path, new_json: str) -> None:
    with sqlite3.connect(path) as connection:
        connection.execute("UPDATE readings SET json = ?", (new_json,))


# --- round trip and what a row holds -----------------------------------------------------------------------------------


def test_a_stored_reading_comes_back_as_a_stored_result_equal_to_the_read(tmp_path, now):
    database, key = tmp_path / "readings.sqlite", cache_key(H7171, FAKE_MODEL)
    original = read_h7171()
    assert store(database, key, original, now) is True

    loaded = lookup(database, H7171, key)

    assert loaded is not None
    assert loaded.status is ReaderStatus.STORED and loaded.tokens is None and loaded.failed_checks == []
    assert loaded.model == key.model == FAKE_MODEL
    assert loaded.notam_id == "H7171/26" and loaded.prompt_version == prompt_module.PROMPT_VERSION
    assert loaded.reading == original.reading  # validates as NotamReading, field for field


def test_a_row_holds_the_semantic_fields_and_created_at_and_nothing_coded(tmp_path, now):
    database, key = tmp_path / "readings.sqlite", cache_key(H7171, FAKE_MODEL)
    store(database, key, read_h7171(), now)

    ((stored_key, stored_json, created_at),) = table(database)

    assert stored_key == key_text(key) and json.loads(stored_key) == ["H7171/26", key.input_sha256, FAKE_MODEL, key.prompt_sha256]
    assert set(json.loads(stored_json)) == set(ReaderOutput.model_fields)  # category .. prohibitive_reason
    assert not {"geometry", "schedule", "id"} & set(json.loads(stored_json))
    assert created_at == now.isoformat()


def test_area_and_schedule_are_built_by_code_on_every_load(tmp_path, now):
    database, key = tmp_path / "readings.sqlite", cache_key(H7171, FAKE_MODEL)
    original = read_h7171()
    store(database, key, original, now)

    loaded = lookup(database, H7171, key)

    assert loaded is not None and loaded.reading is not None
    assert loaded.reading.geometry == notam_area.build_area(H7171)[0]
    assert isinstance(loaded.reading.geometry, Circle) and loaded.reading.geometry.radius_nm == 0.5
    assert loaded.reading.schedule == H7171.item_d == "0800-1800"
    assert loaded.reading.id == "H7171/26"


def test_a_code_fix_takes_effect_without_clearing_the_cache(tmp_path, now, monkeypatch):
    database, key = tmp_path / "readings.sqlite", cache_key(H7171, FAKE_MODEL)
    store(database, key, read_h7171(), now)
    before = lookup(database, H7171, key)
    fixed = Circle(lat=51.0, lon=-1.0, radius_nm=0.25)

    monkeypatch.setattr(notam_area, "build_area", lambda parsed: (fixed, "fixed by a later release"))
    after = lookup(database, H7171, key)

    assert before is not None and before.reading is not None and before.reading.geometry != fixed
    assert after is not None and after.reading is not None and after.reading.geometry == fixed
    assert len(table(database)) == 1  # the same row, not re-read


def test_an_exemption_quote_with_a_phone_placeholder_loads_back_with_the_placeholder_intact(tmp_path, now):
    database, key = tmp_path / "readings.sqlite", cache_key(H7171, FAKE_MODEL)
    quote = "FOR INFO <PHONE_1>"
    result = read_h7171(exemptions=[{"quote": quote, "covers": "OTHER"}])
    assert result.status is ReaderStatus.READ  # the quote is in the redacted E text, so the checks pass
    store(database, key, result, now)

    loaded = lookup(database, H7171, key)

    assert loaded is not None and loaded.reading is not None
    assert [e.quote for e in loaded.reading.exemptions] == [quote]
    assert "<PHONE_1>" in table(database)[0][1]  # and the row holds the placeholder, never the number


def test_overwrite_keeps_one_row_and_the_latest_wins(tmp_path, now):
    database, key = tmp_path / "readings.sqlite", cache_key(H7171, FAKE_MODEL)
    store(database, key, read_h7171(activity="first"), now)
    later = now.replace(hour=now.hour + 1)
    store(database, key, read_h7171(activity="second"), later)

    ((_, _, created_at),) = table(database)
    loaded = lookup(database, H7171, key)

    assert loaded is not None and loaded.reading is not None and loaded.reading.activity == "second"
    assert created_at == later.isoformat()


def test_the_database_and_its_directory_are_made_on_the_first_write(tmp_path, now):
    database = tmp_path / "data" / "deeper" / "readings.sqlite"
    assert lookup(database, H7171, cache_key(H7171, FAKE_MODEL)) is None
    assert not database.parent.exists()  # reading makes nothing

    store(database, cache_key(H7171, FAKE_MODEL), read_h7171(), now)

    assert database.is_file()


# --- what is never written ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("steps", "status"),
    [
        ([reply_fields(exemptions=[{"quote": "NOT IN THE TEXT", "covers": "OTHER"}])] * 2, ReaderStatus.FAILED),
        ([ConnectionError("down")], ReaderStatus.UNAVAILABLE),
        (["", ""], ReaderStatus.UNAVAILABLE),
    ],
    ids=["failed", "unavailable-connection", "unavailable-empty-body"],
)
def test_failed_and_unavailable_results_are_never_written(tmp_path, now, steps, status):
    database = tmp_path / "readings.sqlite"

    result = read_cached(H7171, Steps(*steps), database, now)

    assert result.result.status is status and result.result.reading is None
    assert table(database) == []  # no reading row; the id alone is noted in `failures`, to count the retry
    assert store(database, cache_key(H7171, FAKE_MODEL), result.result, now) is False  # and a direct store refuses too
    assert table(database) == []
    assert lookup(database, H7171, cache_key(H7171, FAKE_MODEL)) is None


def test_a_failed_result_next_to_a_stored_one_leaves_only_the_stored_one(tmp_path, now):
    database = tmp_path / "readings.sqlite"
    read_cached(J3857, Steps(fixture_reply("J3857/26")), database, now)
    read_cached(H7171, Steps(ConnectionError("down")), database, now)

    assert [json.loads(key)[0] for key, _, _ in table(database)] == ["J3857/26"]


def test_a_repaired_result_is_stored(tmp_path, now):
    database = tmp_path / "readings.sqlite"
    bad = {**fixture_reply("H7171/26"), "exemptions": [{"quote": "NOT IN THE TEXT", "covers": "OTHER"}]}

    first = read_cached(H7171, Steps(bad, fixture_reply("H7171/26")), database, now)
    second = read_cached(H7171, Steps(), database, now)

    assert first.result.status is ReaderStatus.REPAIRED and first.calls == 2
    assert second.hit and second.result.status is ReaderStatus.STORED and second.calls == 0
    assert second.result.reading == first.result.reading


def test_model_calls_matches_what_the_model_was_actually_asked(tmp_path, now):
    """calls is derived from the result, so it is checked against the double's own count for every way a read can end."""
    bad = {**fixture_reply("H7171/26"), "exemptions": [{"quote": "NOT IN THE TEXT", "covers": "OTHER"}]}
    good = fixture_reply("H7171/26")
    scripts = {
        ReaderStatus.READ: [good],
        ReaderStatus.REPAIRED: [bad, good],
        ReaderStatus.FAILED: [bad, bad],
        ReaderStatus.UNAVAILABLE: [ConnectionError("first call")],
    }
    for status, steps in scripts.items():
        llm = Steps(*steps)
        result = read_notam(H7171, llm)
        assert result.status is status and model_calls(result) == len(llm.calls), status
    llm = Steps(bad, ConnectionError("the repair"))
    result = read_notam(H7171, llm)
    assert result.status is ReaderStatus.UNAVAILABLE and model_calls(result) == len(llm.calls)
    database = tmp_path / "readings.sqlite"
    store(database, cache_key(H7171, FAKE_MODEL), read_h7171(), now)
    loaded = lookup(database, H7171, cache_key(H7171, FAKE_MODEL))
    assert loaded is not None and model_calls(loaded) == 0  # a stored reading cost no call


# --- bad rows and bad files --------------------------------------------------------------------------------------------

GOOD_ROW = ReaderOutput(**fixture_reply("H7171/26")).model_dump(mode="json")
BAD_ROWS = {
    "not json": "{not json",
    "an unknown enum": json.dumps({**GOOD_ROW, "category": "alarming"}),
    "an extra key": json.dumps({**GOOD_ROW, "mood": "cheerful"}),
    "a geometry the model must not fill (F1)": json.dumps({**GOOD_ROW, "geometry": {"kind": "circle", "lat": 51.0, "lon": -1.0, "radius_nm": 9.0}}),
    "a schedule the model must not fill (F1)": json.dumps({**GOOD_ROW, "schedule": "0000-2359"}),
    "a field missing after a contract change": json.dumps({k: v for k, v in GOOD_ROW.items() if k != "prohibitive_reason"}),
    "a quote that is not in the text": json.dumps({**GOOD_ROW, "exemptions": [{"quote": "EXCEPT PARAGLIDERS", "covers": "PARAMOTOR_CLASS"}]}),
    "a placeholder the NOTAM does not have": json.dumps({**GOOD_ROW, "contact": {"kind": "phone", "value": "<PHONE_7>"}}),
    "a json null": "null",
}


@pytest.mark.parametrize("what", BAD_ROWS)
def test_a_corrupt_row_is_a_miss_that_is_read_again_and_replaced(tmp_path, now, caplog, what):
    database, key = tmp_path / "readings.sqlite", cache_key(H7171, FAKE_MODEL)
    store(database, key, read_h7171(), now)
    corrupt_the_row(database, BAD_ROWS[what])
    llm = Steps(fixture_reply("H7171/26"))

    with caplog.at_level(logging.DEBUG):
        assert lookup(database, H7171, key) is None
        again = read_cached(H7171, llm, database, now)

    assert not again.hit and again.result.status is ReaderStatus.READ and len(llm.calls) == 1
    assert len(table(database)) == 1 and json.loads(table(database)[0][1]) == GOOD_ROW  # the row was replaced, not added to
    assert lookup(database, H7171, key) is not None
    warnings = [r.getMessage() for r in caplog.records if r.name == "flysafelz.reader.cache"]
    assert warnings and all("H7171/26" in line for line in warnings)
    assert not any(bad in line for line in warnings for bad in ("alarming", "cheerful", "EXCEPT PARAGLIDERS", "<PHONE_7>", "0000-2359"))  # the id, never the row


def test_a_garbage_file_is_read_around_with_one_warning_and_no_exception(tmp_path, now, caplog):
    database = tmp_path / "readings.sqlite"
    garbage = bytes(range(256)) * 8
    database.write_bytes(garbage)
    llm = Steps(fixture_reply("H7171/26"))

    with caplog.at_level(logging.WARNING):
        result = read_cached(H7171, llm, database, now)

    assert result.result.status is ReaderStatus.READ and not result.hit and len(llm.calls) == 1
    assert len([r for r in caplog.records if r.name == "flysafelz.reader.cached_read"]) == 1
    assert database.read_bytes() == garbage  # it is left as it was


def test_a_directory_where_the_file_should_be_is_read_around_too(tmp_path, now, caplog):
    database = tmp_path / "readings.sqlite"
    database.mkdir()

    with caplog.at_level(logging.WARNING):
        result = read_cached(H7171, Steps(fixture_reply("H7171/26")), database, now)

    assert result.result.status is ReaderStatus.READ
    assert len([r for r in caplog.records if r.levelno >= logging.WARNING and "cache" in r.getMessage().lower()]) == 1


def test_a_path_under_a_plain_file_is_read_around_with_one_warning(tmp_path, now, caplog):
    (tmp_path / "data").write_text("a file where the folder should be", encoding="utf-8")

    with caplog.at_level(logging.WARNING):
        result = read_cached(H7171, Steps(fixture_reply("H7171/26")), tmp_path / "data" / "readings.sqlite", now)

    assert result.result.status is ReaderStatus.READ
    assert len([r for r in caplog.records if "not stored" in r.getMessage()]) == 1


def test_a_failure_is_recorded_by_id_and_a_later_store_of_that_id_clears_it(tmp_path, now):
    database = tmp_path / "readings.sqlite"
    assert not was_failed(database, "H7171/26") and not database.exists()  # no file: no failure, and nothing made

    record_failure(database, "H7171/26", now)
    record_failure(database, "H7171/26", now)  # twice is still one record
    assert was_failed(database, "H7171/26") and not was_failed(database, "J3857/26")
    assert lookup(database, H7171, cache_key(H7171, FAKE_MODEL)) is None  # a failure is not a reading

    result = read_h7171()
    store(database, cache_key(H7171, FAKE_MODEL), result, now)
    assert not was_failed(database, "H7171/26")


def test_recording_a_failure_in_a_garbage_file_is_for_the_caller_and_read_cached_reads_around_it(tmp_path, now, caplog):
    database = tmp_path / "readings.sqlite"
    database.write_bytes(bytes(range(256)) * 8)

    with pytest.raises(sqlite3.Error):
        record_failure(database, "H7171/26", now)
    with caplog.at_level(logging.WARNING):
        result = read_cached(H7171, Steps(ConnectionError("down")), database, now)

    assert result.result.status is ReaderStatus.UNAVAILABLE and not result.retry
    assert len([r for r in caplog.records if r.name == "flysafelz.reader.cached_read"]) == 1


def test_an_empty_file_is_a_database_with_no_rows(tmp_path, now):
    database = tmp_path / "readings.sqlite"
    database.touch()

    first = read_cached(H7171, Steps(fixture_reply("H7171/26")), database, now)
    second = read_cached(H7171, Steps(), database, now)

    assert not first.hit and second.hit


# --- bypass --------------------------------------------------------------------------------------------------------------


def test_bypass_neither_reads_nor_writes(tmp_path, now):
    database, key = tmp_path / "readings.sqlite", cache_key(H7171, FAKE_MODEL)
    store(database, key, read_h7171(activity="stored earlier"), now)
    before = table(database)
    llm = Steps(fixture_reply("H7171/26"))

    result = read_cached(H7171, llm, None, now)

    assert not result.hit and result.result.status is ReaderStatus.READ and len(llm.calls) == 1  # the stored reading for the same key was not returned
    assert result.result.reading is not None and result.result.reading.activity != "stored earlier"
    assert table(database) == before  # and the model's answer was not written
    assert list(tmp_path.iterdir()) == [database]


def test_bypass_writes_nothing_when_there_is_no_database(tmp_path, now, monkeypatch):
    monkeypatch.chdir(tmp_path)

    read_cached(H7171, Steps(fixture_reply("H7171/26")), None, now)

    assert list(tmp_path.iterdir()) == []


# --- concurrency (PR-5's shared pool of three) ------------------------------------------------------------------------

THREADS = 3
ROUNDS = 12


def quiet_reply(activity: str) -> dict:
    return reply_fields(activity=activity, contact=NO_CONTACT)


def write_together(database, notams, now) -> list[bool]:
    """THREADS writers of different NOTAMs and THREADS writers of the first NOTAM's key, all released at the same moment."""
    shared, barrier = notams[0], Barrier(THREADS * 2)

    def write_distinct(parsed):
        result = read_notam(parsed, Steps(quiet_reply(f"distinct {parsed.id}")))
        barrier.wait()
        return store(database, cache_key(parsed, FAKE_MODEL), result, now)

    def write_shared(n):
        result = read_notam(shared, Steps(quiet_reply(f"shared writer {n}")))
        barrier.wait()
        return store(database, cache_key(shared, FAKE_MODEL), result, now)

    with ThreadPoolExecutor(max_workers=THREADS * 2) as pool:
        futures = [pool.submit(write_distinct, parsed) for parsed in notams[1 : THREADS + 1]]
        futures += [pool.submit(write_shared, n) for n in range(THREADS)]
        return [future.result() for future in futures]  # a ProgrammingError or "database is locked" would be raised here


def test_three_threads_storing_different_notams_and_three_storing_the_same_key(tmp_path, now):
    notams = [NOTAMS[i] for i in NOTAMS if i not in ("H7171/26", "J3857/26")][: THREADS * 2]
    for round_ in range(ROUNDS):
        database = tmp_path / f"round{round_}" / "readings.sqlite"  # a fresh file each time: the first write races to make the table

        assert write_together(database, notams, now) == [True] * (THREADS * 2)

        assert len(table(database)) == THREADS + 1  # three NOTAMs and the one shared key
        loaded = lookup(database, notams[0], cache_key(notams[0], FAKE_MODEL))
        assert loaded is not None and loaded.reading is not None
        assert loaded.reading.activity in {f"shared writer {n}" for n in range(THREADS)}  # one whole, valid reading


def test_three_threads_reading_through_the_cache_at_once_give_no_error_and_one_row_each(tmp_path, now):
    database = tmp_path / "readings.sqlite"
    notams = [NOTAMS[i] for i in list(NOTAMS)[:THREADS]]
    barrier = Barrier(THREADS)

    def read(parsed):
        barrier.wait()
        return read_cached(parsed, Steps(quiet_reply("quiet")), database, now)

    with ThreadPoolExecutor(max_workers=THREADS) as pool:
        results = [f.result() for f in [pool.submit(read, parsed) for parsed in notams]]

    assert [r.result.status for r in results] == [ReaderStatus.READ] * THREADS
    assert len(table(database)) == THREADS
