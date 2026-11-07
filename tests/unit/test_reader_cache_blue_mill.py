"""GP5 at function level: the Blue Mill reader set (H7171/26, J3857/26) read twice through the cache.

The model is PerNotam (tests/unit/cache_helpers.py), which counts every `invoke`; its answers are readings.yaml's fields
with the number as its placeholder. The database is under tmp_path. The CLI's print is PR-6's.
"""

import sqlite3

from cache_helpers import PerNotam, bad_reply, blue_mill_set, fixture_reply

from flysafelz.reader.cached_read import read_set
from flysafelz.reader.contract import ReaderStatus

H7171, J3857 = "H7171/26", "J3857/26"


def rows(path) -> list[str]:
    with sqlite3.connect(path) as connection:
        return [key for (key,) in connection.execute("SELECT key FROM readings ORDER BY key")]


def statuses(read_set_result) -> dict[str, ReaderStatus]:
    return {notam_id: read.result.status for notam_id, read in read_set_result.reads.items()}


def test_second_run_all_ok_makes_no_model_call_and_every_result_is_stored(tmp_path, now):
    notams, database = blue_mill_set(), tmp_path / "readings.sqlite"
    llm = PerNotam(notams, H7171_26=[fixture_reply(H7171)], J3857_26=[fixture_reply(J3857)])

    first = read_set(notams, llm, database, now)
    asked_by_run_1 = len(llm.calls)
    second = read_set(notams, llm, database, now)

    assert (first.calls, first.retries) == (2, 0) and asked_by_run_1 == 2
    assert statuses(first) == {H7171: ReaderStatus.READ, J3857: ReaderStatus.READ}
    assert (second.calls, second.retries) == (0, 0)
    assert len(llm.calls) == asked_by_run_1  # the fake was not asked again
    assert statuses(second) == {H7171: ReaderStatus.STORED, J3857: ReaderStatus.STORED}
    for notam_id in (H7171, J3857):
        assert second.reads[notam_id].result.reading == first.reads[notam_id].result.reading  # field for field
        assert second.reads[notam_id].hit and not first.reads[notam_id].hit


def test_second_run_after_a_failure_sends_only_the_failed_notam(tmp_path, now):
    notams, database = blue_mill_set(), tmp_path / "readings.sqlite"
    llm = PerNotam(notams, H7171_26=[bad_reply(H7171), bad_reply(H7171)], J3857_26=[fixture_reply(J3857)])

    first = read_set(notams, llm, database, now)
    assert statuses(first) == {H7171: ReaderStatus.FAILED, J3857: ReaderStatus.READ}
    assert (first.calls, first.retries) == (3, 0)  # H7171 read and repair, then J3857
    assert len(llm.calls) == 3
    assert first.reads[H7171].result.reading is None
    assert len(rows(database)) == 1  # J3857 only: a failure is never filed

    llm.script(H7171, fixture_reply(H7171))
    llm.calls.clear()
    second = read_set(notams, llm, database, now)

    assert llm.calls == [H7171]  # J3857 is a hit and is not sent
    assert (second.calls, second.retries) == (1, 1)
    assert statuses(second) == {H7171: ReaderStatus.READ, J3857: ReaderStatus.STORED}
    assert len(rows(database)) == 2


def test_a_retry_that_needs_its_own_repair_costs_two_calls(tmp_path, now):
    notams, database = blue_mill_set(), tmp_path / "readings.sqlite"
    llm = PerNotam(notams, H7171_26=[bad_reply(H7171), bad_reply(H7171)], J3857_26=[fixture_reply(J3857)])
    read_set(notams, llm, database, now)

    llm.script(H7171, bad_reply(H7171), fixture_reply(H7171))
    llm.calls.clear()
    second = read_set(notams, llm, database, now)

    assert llm.calls == [H7171, H7171]
    assert (second.calls, second.retries) == (2, 1)
    assert second.reads[H7171].result.status is ReaderStatus.REPAIRED


def test_an_unavailable_notam_is_not_stored_and_is_retried_next_time(tmp_path, now):
    notams, database = blue_mill_set(), tmp_path / "readings.sqlite"
    llm = PerNotam(notams, H7171_26=[ConnectionError("endpoint refused the connection")], J3857_26=[fixture_reply(J3857)])

    first = read_set(notams, llm, database, now)
    assert statuses(first) == {H7171: ReaderStatus.UNAVAILABLE, J3857: ReaderStatus.READ}
    assert first.reads[H7171].result.reading is None
    assert (first.calls, first.retries) == (2, 0)
    assert len(rows(database)) == 1

    llm.script(H7171, fixture_reply(H7171))
    llm.calls.clear()
    second = read_set(notams, llm, database, now)

    assert llm.calls == [H7171]  # retried
    assert statuses(second) == {H7171: ReaderStatus.READ, J3857: ReaderStatus.STORED}
    assert (second.calls, second.retries) == (1, 1)


def failed_ids(path) -> list[str]:
    with sqlite3.connect(path) as connection:
        return [notam_id for (notam_id,) in connection.execute("SELECT notam_id FROM failures ORDER BY notam_id")]


def test_a_first_run_where_nothing_is_readable_stores_no_reading_and_the_next_run_retries_both(tmp_path, now):
    """No hit at all in run 2's cache, yet both NOTAMs are retries: they failed, they were not new (GP5, R3.2)."""
    notams, database = blue_mill_set(), tmp_path / "data" / "readings.sqlite"
    down = [ConnectionError("down")]
    llm = PerNotam(notams, H7171_26=list(down), J3857_26=list(down))

    first = read_set(notams, llm, database, now)

    assert set(statuses(first).values()) == {ReaderStatus.UNAVAILABLE}
    assert (first.calls, first.retries) == (2, 0)  # the first ask of each is not a retry
    assert rows(database) == []  # no reading was filed
    assert failed_ids(database) == [H7171, J3857]

    llm.script(H7171, fixture_reply(H7171))
    llm.script(J3857, fixture_reply(J3857))
    second = read_set(notams, llm, database, now)

    assert (second.calls, second.retries) == (2, 2)
    assert set(statuses(second).values()) == {ReaderStatus.READ}
    assert failed_ids(database) == []  # read now, so nothing left to retry
    third = read_set(notams, llm, database, now)
    assert (third.calls, third.retries) == (0, 0)


def test_a_notam_never_seen_before_is_a_miss_but_not_a_retry(tmp_path, now):
    notams, database = blue_mill_set(), tmp_path / "readings.sqlite"
    llm = PerNotam(notams, H7171_26=[fixture_reply(H7171)], J3857_26=[fixture_reply(J3857)])
    first = read_set(notams[:1], llm, database, now)
    assert (first.calls, first.retries) == (1, 0)

    second = read_set(notams, llm, database, now)

    assert llm.calls == [H7171, J3857]  # J3857 is new: sent once, and H7171 was a hit
    assert (second.calls, second.retries) == (1, 0)
    assert not second.reads[J3857].retry and not second.reads[J3857].hit


def test_a_reissue_or_changed_text_after_a_good_reading_is_not_a_retry(tmp_path, now):
    notams, database = blue_mill_set(), tmp_path / "readings.sqlite"
    llm = PerNotam(notams, H7171_26=[fixture_reply(H7171)], J3857_26=[fixture_reply(J3857)])
    read_set(notams, llm, database, now)
    edited = notams[0].model_copy(update={"item_e": notams[0].item_e.replace("700FT", "600FT")})
    reissued = notams[0].model_copy(update={"id": "H7172/26"})
    llm = PerNotam([edited, reissued], H7171_26=[fixture_reply(H7171)], H7172_26=[fixture_reply(H7171)])

    second = read_set([edited, reissued], llm, database, now)

    assert [read.hit for read in second.reads.values()] == [False, False]
    assert (second.calls, second.retries) == (2, 0)


def test_a_notam_that_failed_and_is_edited_before_the_next_ask_is_still_a_retry(tmp_path, now):
    """The record is by id: the same NOTAM asked again after a failure, whatever its text now says."""
    notams, database = blue_mill_set(), tmp_path / "readings.sqlite"
    llm = PerNotam(notams, H7171_26=[bad_reply(H7171), bad_reply(H7171)], J3857_26=[fixture_reply(J3857)])
    read_set(notams, llm, database, now)
    edited = notams[0].model_copy(update={"item_e": notams[0].item_e.replace("700FT", "600FT")})
    llm = PerNotam([edited], H7171_26=[fixture_reply(H7171)])

    second = read_set([edited], llm, database, now)

    assert (second.calls, second.retries) == (1, 1)


def test_the_failures_table_holds_ids_and_dates_only_and_is_never_returned_as_a_reading(tmp_path, now):
    notams, database = blue_mill_set(), tmp_path / "readings.sqlite"
    llm = PerNotam(notams, H7171_26=[bad_reply(H7171), bad_reply(H7171)], J3857_26=[ConnectionError("down")])

    read_set(notams, llm, database, now)

    with sqlite3.connect(database) as connection:
        assert [column[1] for column in connection.execute("PRAGMA table_info(failures)")] == ["notam_id", "created_at"]
        assert connection.execute("SELECT notam_id, created_at FROM failures ORDER BY notam_id").fetchall() == [(H7171, now.isoformat()), (J3857, now.isoformat())]
    assert rows(database) == []
    llm.script(H7171, bad_reply(H7171), bad_reply(H7171))
    llm.script(J3857, ConnectionError("still down"))
    again = read_set(notams, llm, database, now)
    assert not any(read.hit for read in again.reads.values())  # a failure is never served as a reading
    assert (again.calls, again.retries) == (3, 2)
    assert failed_ids(database) == [H7171, J3857]


def test_bypass_records_no_failure_and_counts_no_retry(tmp_path, now):
    notams, database = blue_mill_set(), tmp_path / "readings.sqlite"
    llm = PerNotam(notams, H7171_26=[ConnectionError("down"), fixture_reply(H7171)], J3857_26=[ConnectionError("down"), fixture_reply(J3857)])

    first, second = read_set(notams, llm, None, now), read_set(notams, llm, None, now)

    assert (first.retries, second.retries) == (0, 0)
    assert not database.exists()


def test_the_readings_of_the_set_pass_the_real_checks_and_match_the_fixture_areas(tmp_path, now):
    """The answers are built from readings.yaml, so a stored reading comes back as the fixture's, with code's area."""
    notams, database = blue_mill_set(), tmp_path / "readings.sqlite"
    llm = PerNotam(notams, H7171_26=[fixture_reply(H7171)], J3857_26=[fixture_reply(J3857)])
    read_set(notams, llm, database, now)

    loaded = read_set(notams, llm, database, now).reads[H7171].result.reading
    assert loaded is not None
    assert loaded.geometry is not None and loaded.geometry.kind == "circle" and loaded.geometry.radius_nm == 0.5  # built by code from the E text
    assert loaded.schedule == "0800-1800"
