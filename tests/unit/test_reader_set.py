"""G8, G23: the on-demand read for one landing zone and window (`read_for_lz`).

The model is ScriptedChat (tests/unit/reader_set_helpers.py): thread-safe, scripted per NOTAM id, and it records the peak
number of calls in flight. The cache is PR-4's, in a SQLite file under tmp_path. The NOTAMs are PR-4's `notam()` factory
and the committed Blue Mill PIB. Nothing here opens a socket or reads the environment.
"""

import logging
import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import ClassVar

import pytest
from cache_helpers import blue_mill_set, fixture_reply
from factories import notam
from pib_helpers import LZ, SPAN
from reader_helpers import BLUE_MILL_PIB, blue_mill_notams, reply_fields
from reader_set_helpers import ScriptedChat, bad_reply, good_reply

from flysafelz.pib import SortAction, prefilter, sort_notam
from flysafelz.reader import reader_set as module
from flysafelz.reader.contract import ReaderStatus
from flysafelz.reader.reader_set import read_for_lz, reader_set
from flysafelz.reader.settings import ReaderSettings, load_settings

READ, REPAIRED, FAILED, UNAVAILABLE, STORED = ReaderStatus.READ, ReaderStatus.REPAIRED, ReaderStatus.FAILED, ReaderStatus.UNAVAILABLE, ReaderStatus.STORED
H7171, J3857, J3864 = "H7171/26", "J3857/26", "J3864/26"
FAR_AWAY = {"centre_lat": 55.5}  # a Q-circle well beyond the local area: the prefilter drops it
VIA_E = "TEMPO DANGER AREA (TDA) EGD496A ACTIVATED."


def to_reader(notam_id: str = "H7171/26", **overrides):
    """A TO_READER NOTAM. Its E text names its id, so no two have the same text: the fake finds a NOTAM by the text it is sent."""
    overrides.setdefault("item_e", f"UAS OPR WI 0.5NM RADIUS OF 534732N 0000459W. REF {notam_id}.")
    return notam(notam_id, **overrides)


def via_airspace(notam_id: str = "J3857/26"):
    return notam(notam_id, subject="RD", radius_nm=10.0, item_e=VIA_E)


def many(count: int):
    """`count` distinct TO_READER NOTAMs."""
    return [to_reader(f"H{7000 + index}/26") for index in range(count)]


def statuses(results) -> dict[str, ReaderStatus]:
    return {notam_id: result.status for notam_id, result in results.items()}


def rows(path) -> int:
    with sqlite3.connect(path) as connection:
        return connection.execute("SELECT COUNT(*) FROM readings").fetchone()[0]


def read(notams, llm, cache=None, now=None):
    return read_for_lz(notams, LZ, SPAN, llm, cache, now=now)


def sort_action(parsed) -> SortAction:
    return sort_notam(parsed, SPAN, LZ).action


# --- G8, G23: which NOTAMs are read ------------------------------------------------------------------------------------


# (name, notam, what ticket 2's sort says, is it read). One factory NOTAM per outcome, and the prefilter's own drops.
SELECTION = [
    ("TO_READER: a UAS NOTAM near the LZ", to_reader(), SortAction.TO_READER, True),
    (
        "TO_READER: a restriction with no designator",
        notam("J1458/26", subject="RT", item_e="RESTRICTED AREA (TEMPORARY) ACT WI 2.2NM RADIUS OF PSN 520528N 0001055E (DUXFORD)"),
        SortAction.TO_READER,
        True,
    ),
    ("VIA_AIRSPACE: a danger area trigger, read for its contact", via_airspace(), SortAction.VIA_AIRSPACE, True),
    ("IGNORE: IFR traffic only", notam("H100/26", traffic="I"), SortAction.IGNORE, False),
    ("IGNORE: item D has no time in the window", notam("H101/26", item_d="0000-0100"), SortAction.IGNORE, False),
    ("AWARENESS: GNSS jamming", notam("B1104/26", subject="GW", radius_nm=25.0, item_e="GNSS INTERFERENCE"), SortAction.AWARENESS, False),
    ("AWARENESS: an area-wide radius", notam("H7297/26", radius_nm=50.0), SortAction.AWARENESS, False),
]


@pytest.mark.parametrize(("name", "parsed", "action", "is_read"), SELECTION, ids=[row[0] for row in SELECTION])
def test_only_to_reader_and_via_airspace_notams_reach_the_model(name, parsed, action, is_read):
    survives_the_prefilter = prefilter([parsed], LZ, SPAN)[0] == [parsed]
    assert survives_the_prefilter  # so the sort, not the prefilter, is what this row tests
    assert sort_action(parsed) is action, name
    llm = ScriptedChat([parsed])

    results = read([parsed], llm)

    assert list(results) == ([parsed.id] if is_read else [])
    assert llm.calls == ([parsed.id] if is_read else [])
    if is_read:
        assert results[parsed.id].status is READ


def test_a_notam_the_prefilter_drops_is_not_read_even_though_the_sort_would_send_it_to_the_reader():
    far = to_reader("H200/26", **FAR_AWAY)
    assert prefilter([far], LZ, SPAN)[0] == []
    assert sort_action(far) is SortAction.TO_READER  # the sort alone would read it; the prefilter runs first
    llm = ScriptedChat([far])

    assert read([far], llm) == {}
    assert llm.calls == []


def test_all_the_outcomes_together_read_exactly_the_two_kinds_and_in_input_order():
    parsed = [row[1] for row in SELECTION] + [to_reader("H200/26", **FAR_AWAY)]
    llm = ScriptedChat(parsed)

    results = read(parsed, llm)

    assert list(results) == ["H7171/26", "J1458/26", "J3857/26"]
    assert sorted(llm.calls) == ["H7171/26", "J1458/26", "J3857/26"]
    assert reader_set(parsed, LZ, SPAN) == [row[1] for row in SELECTION[:3]]


def test_the_set_is_decided_by_prefilter_then_sort_with_the_whole_window_as_the_span(monkeypatch):
    order, spans = [], []
    real_prefilter, real_sort = module.prefilter, module.sort_notam

    def prefilter_spy(notams, lz, window, **kwargs):
        order.append("prefilter")
        return real_prefilter(notams, lz, window, **kwargs)

    def sort_spy(parsed, span, lz, **kwargs):
        order.append(f"sort {parsed.id}")
        spans.append((span, lz))
        return real_sort(parsed, span, lz, **kwargs)

    monkeypatch.setattr(module, "prefilter", prefilter_spy)
    monkeypatch.setattr(module, "sort_notam", sort_spy)
    parsed = [to_reader("H200/26", **FAR_AWAY), to_reader(), notam("H100/26", traffic="I")]

    assert [n.id for n in reader_set(parsed, LZ, SPAN)] == [H7171]

    assert order == ["prefilter", "sort H7171/26", "sort H100/26"]  # the dropped NOTAM is never sorted
    assert spans == [(SPAN, LZ), (SPAN, LZ)]


def test_a_reading_that_calls_a_notam_unimportant_drops_nothing():
    """Facts can downgrade, opinions can't: an informational reading that applies to UAS only is still in the dict."""
    parsed = [to_reader(), via_airspace()]
    unimportant = reply_fields(category="informational", applies_to="UAS_ONLY", contact={"kind": "none", "value": None})
    llm = ScriptedChat(parsed, H7171_26=[unimportant], J3857_26=[unimportant])

    results = read(parsed, llm)

    assert statuses(results) == {H7171: READ, J3857: READ}
    assert results[H7171].reading is not None and results[H7171].reading.applies_to == "UAS_ONLY"


# --- the Blue Mill case --------------------------------------------------------------------------------------------------


def test_blue_mill_reader_set_is_h7171_and_j3857_and_j3864_is_dropped_by_the_prefilter():
    notams = list(blue_mill_notams().values())

    kept, dropped = prefilter(notams, LZ, SPAN)
    chosen = reader_set(notams, LZ, SPAN)

    assert {n.id for n in chosen} == {H7171, J3857}
    assert sort_action(next(n for n in chosen if n.id == H7171)) is SortAction.TO_READER
    assert sort_action(next(n for n in chosen if n.id == J3857)) is SortAction.VIA_AIRSPACE
    assert J3864 in {d.notam_id for d in dropped}
    assert J3864 not in {n.id for n in kept}


def test_blue_mill_is_read_through_the_whole_pib(tmp_path, now):
    notams = list(blue_mill_notams().values())
    llm = ScriptedChat(notams, H7171_26=[fixture_reply(H7171)], J3857_26=[fixture_reply(J3857)])

    results = read(notams, llm, tmp_path / "readings.sqlite", now)

    assert sorted(results) == [H7171, J3857]
    assert sorted(llm.calls) == [H7171, J3857]
    assert statuses(results) == {H7171: READ, J3857: READ}
    assert BLUE_MILL_PIB.is_file()


# --- nothing to read --------------------------------------------------------------------------------------------------------


class NoPool(ThreadPoolExecutor):
    def __init__(self, *args, **kwargs):
        raise AssertionError("a pool was made for an empty reader set")


@pytest.mark.parametrize(
    "parsed",
    [[], [notam("H100/26", traffic="I"), notam("B1104/26", subject="GW", radius_nm=25.0), notam("H7297/26", radius_nm=50.0)], [to_reader("H200/26", **FAR_AWAY)]],
    ids=["no NOTAMs", "everything IGNORE or AWARENESS", "everything dropped by the prefilter"],
)
def test_an_empty_reader_set_returns_nothing_with_no_model_call_and_no_pool(monkeypatch, parsed):
    monkeypatch.setattr(module, "ThreadPoolExecutor", NoPool)
    llm = ScriptedChat(parsed)

    assert read(parsed, llm) == {}
    assert llm.calls == []


# --- the cache ----------------------------------------------------------------------------------------------------------


def test_a_miss_reads_and_stores_and_a_hit_is_stored_with_no_model_call(tmp_path, now):
    notams, database = blue_mill_set(), tmp_path / "readings.sqlite"
    llm = ScriptedChat(notams, H7171_26=[fixture_reply(H7171)], J3857_26=[fixture_reply(J3857)])

    first = read(notams, llm, database, now)
    second = read(notams, llm, database, now)

    assert statuses(first) == {H7171: READ, J3857: READ}
    assert rows(database) == 2
    assert statuses(second) == {H7171: STORED, J3857: STORED}
    assert len(llm.calls) == 2  # the second run asked the model nothing
    for notam_id in (H7171, J3857):
        assert second[notam_id].reading == first[notam_id].reading


def test_a_repaired_result_is_stored_too(tmp_path, now):
    parsed, database = [to_reader()], tmp_path / "readings.sqlite"
    llm = ScriptedChat(parsed, H7171_26=[bad_reply(), good_reply()])

    assert statuses(read(parsed, llm, database, now)) == {H7171: REPAIRED}
    assert rows(database) == 1
    assert statuses(read(parsed, llm, database, now)) == {H7171: STORED}
    assert llm.calls == [H7171, H7171]


def test_failed_and_unavailable_results_come_back_but_are_never_stored(tmp_path, now):
    parsed, database = [to_reader("H1/26"), to_reader("H2/26")], tmp_path / "readings.sqlite"
    llm = ScriptedChat(parsed, H1_26=[bad_reply(), bad_reply()], H2_26=[TimeoutError("took too long")])

    results = read(parsed, llm, database, now)

    assert statuses(results) == {"H1/26": FAILED, "H2/26": UNAVAILABLE}
    assert results["H1/26"].reading is None and results["H1/26"].failed_checks
    assert results["H2/26"].reading is None
    assert not database.exists() or rows(database) == 0


def test_second_run_after_a_mixed_first_run_asks_the_model_only_about_the_failed_and_unavailable(tmp_path, now):
    read_ok, repaired, failed, unavailable = (to_reader(f"H{n}/26") for n in (11, 12, 13, 14))
    parsed, database = [read_ok, repaired, failed, unavailable], tmp_path / "readings.sqlite"
    llm = ScriptedChat(parsed, H12_26=[bad_reply(), good_reply()], H13_26=[bad_reply(), bad_reply()], H14_26=[ConnectionError("down")])

    first = read(parsed, llm, database, now)

    assert statuses(first) == {"H11/26": READ, "H12/26": REPAIRED, "H13/26": FAILED, "H14/26": UNAVAILABLE}
    assert rows(database) == 2

    llm.calls.clear()
    second = read(parsed, llm, database, now)

    assert sorted(llm.calls) == ["H13/26", "H14/26"]
    assert statuses(second) == {"H11/26": STORED, "H12/26": STORED, "H13/26": READ, "H14/26": READ}
    assert rows(database) == 4


def test_no_cache_means_nothing_is_read_from_or_written_to_a_file(tmp_path, now):
    parsed = [to_reader()]
    llm = ScriptedChat(parsed)

    assert statuses(read(parsed, llm, None, now)) == {H7171: READ}
    assert statuses(read(parsed, llm, None, now)) == {H7171: READ}
    assert llm.calls == [H7171, H7171]
    assert list(tmp_path.iterdir()) == []


def test_a_row_is_stamped_with_the_now_it_was_given(tmp_path, now):
    database = tmp_path / "readings.sqlite"
    read([to_reader()], ScriptedChat([to_reader()]), database, now)

    with sqlite3.connect(database) as connection:
        (created,) = connection.execute("SELECT created_at FROM readings").fetchone()
    assert created == now.isoformat()


def test_without_a_now_the_clock_stamps_the_row(tmp_path):
    database = tmp_path / "readings.sqlite"
    read([to_reader()], ScriptedChat([to_reader()]), database)

    with sqlite3.connect(database) as connection:
        (created,) = connection.execute("SELECT created_at FROM readings").fetchone()
    assert datetime.fromisoformat(created).tzinfo is not None


# --- the same id twice ----------------------------------------------------------------------------------------------------


def test_the_same_id_twice_is_read_once_and_the_first_wins():
    first, second = to_reader(), to_reader(item_e="UAS OPR WI 0.5NM RADIUS OF 534732N 0000459W. A DIFFERENT SECOND COPY.")
    llm = ScriptedChat([first])  # a read of the second copy would show in `calls` as an unknown text

    results = read([first, second, first], llm)

    assert llm.calls == [H7171]
    assert list(results) == [H7171]
    assert [n.item_e for n in reader_set([first, second], LZ, SPAN)] == [first.item_e]


# --- the shared limit -------------------------------------------------------------------------------------------------------


def run_in_time(call, seconds: float = 60.0):
    """Run `call` in a thread and return its result; fail the test if it is still running after `seconds` (a deadlock)."""
    box = {}
    worker = threading.Thread(target=lambda: box.update(result=call()), daemon=True)
    worker.start()
    worker.join(timeout=seconds)
    assert not worker.is_alive(), f"no result in {seconds}s: the pool is deadlocked"
    return box["result"]


class CountingPool(ThreadPoolExecutor):
    """The real pool, noting its size and every submit."""

    made: ClassVar[list["CountingPool"]] = []

    def __init__(self, max_workers=None, *args, **kwargs):
        super().__init__(max_workers, *args, **kwargs)
        self.size, self.submitted = max_workers, 0
        self.made.append(self)

    def submit(self, fn, /, *args, **kwargs):
        self.submitted += 1
        return super().submit(fn, *args, **kwargs)


def test_ten_notams_that_all_need_a_repair_run_at_most_three_calls_at_once_and_finish(monkeypatch):
    CountingPool.made = []
    monkeypatch.setattr(module, "ThreadPoolExecutor", CountingPool)
    parsed = many(10)
    scripts = {parsed_.id.replace("/", "_"): [bad_reply(), good_reply()] for parsed_ in parsed}
    llm = ScriptedChat(parsed, rendezvous=3, pause_s=0.01, **scripts)

    results = run_in_time(lambda: read(parsed, llm))

    assert statuses(results) == {parsed_.id: REPAIRED for parsed_ in parsed}
    assert len(llm.calls) == 20  # ten first reads and ten repairs
    assert llm.peak == 3  # three ran together (the rendezvous needed them to), and never a fourth: first reads and repairs counted together
    assert [(pool.size, pool.submitted) for pool in CountingPool.made] == [(3, 10)]  # one pool, ten submits: a repair is never submitted back


def test_the_limit_is_the_settings_concurrency(monkeypatch, tmp_path):
    two = tmp_path / "settings.yaml"
    two.write_text("timeout_s: 45\nconcurrency: 2\nruns: 3\n", encoding="utf-8")
    monkeypatch.setattr(module, "SETTINGS", load_settings(two))
    parsed = many(6)
    llm = ScriptedChat(parsed, rendezvous=2, pause_s=0.01)

    results = run_in_time(lambda: read(parsed, llm))

    assert set(statuses(results).values()) == {READ}
    assert llm.peak == 2


def test_one_worker_does_a_first_read_and_its_repair_in_its_own_slot_so_a_pool_of_one_cannot_deadlock(monkeypatch, tmp_path):
    one = tmp_path / "settings.yaml"
    one.write_text("timeout_s: 45\nconcurrency: 1\nruns: 3\n", encoding="utf-8")
    monkeypatch.setattr(module, "SETTINGS", load_settings(one))
    parsed = many(4)
    scripts = {parsed_.id.replace("/", "_"): [bad_reply(), good_reply()] for parsed_ in parsed}
    llm = ScriptedChat(parsed, pause_s=0.005, **scripts)

    results = run_in_time(lambda: read(parsed, llm))

    assert set(statuses(results).values()) == {REPAIRED}
    assert llm.peak == 1
    assert llm.calls == [parsed_.id for parsed_ in parsed for _ in (1, 2)]  # each NOTAM's first read and repair back to back


# --- one NOTAM cannot stop the others -------------------------------------------------------------------------------------------


def test_a_notam_whose_model_call_raises_is_unavailable_and_the_others_are_read():
    parsed = many(5)
    llm = ScriptedChat(parsed, H7002_26=[RuntimeError("the endpoint fell over")], H7004_26=[ValueError("not json"), ValueError("still not")])

    results = read(parsed, llm)

    assert list(results) == [p.id for p in parsed]  # one entry per NOTAM in the set, in order
    assert statuses(results) == {"H7000/26": READ, "H7001/26": READ, "H7002/26": UNAVAILABLE, "H7003/26": READ, "H7004/26": UNAVAILABLE}


def test_a_read_that_raises_inside_the_worker_costs_only_its_own_entry(monkeypatch, caplog, tmp_path, now):
    """read_notam and the cache layer never raise; if something past them did, the entry is UNAVAILABLE, logged without its text, and not stored."""
    real = module.read_cached

    def raising_for_h7001(parsed, llm, cache, stamp):
        if parsed.id == "H7001/26":
            raise RuntimeError("boom, call 07700 900123")
        return real(parsed, llm, cache, stamp)

    monkeypatch.setattr(module, "read_cached", raising_for_h7001)
    parsed, database = many(3), tmp_path / "readings.sqlite"
    llm = ScriptedChat(parsed)

    with caplog.at_level(logging.WARNING):
        results = read(parsed, llm, database, now)

    assert statuses(results) == {"H7000/26": READ, "H7001/26": UNAVAILABLE, "H7002/26": READ}
    assert results["H7001/26"].reading is None and results["H7001/26"].model == llm.model_name
    assert "H7001/26" not in llm.calls
    assert rows(database) == 2
    assert "900123" not in caplog.text and "<PHONE_1>" in caplog.text


def test_the_settings_object_is_the_one_the_limit_comes_from():
    assert isinstance(module.SETTINGS, ReaderSettings) and module.SETTINGS.concurrency >= 1
