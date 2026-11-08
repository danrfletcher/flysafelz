"""Minute expansion (5a PR-5.F1): half-open spans, clipping (E14), overlap and gap rules, timezone rejection, every row shape of the sheet."""

from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path

import pytest
import yaml

from agent_eval.factories import row, t, tier_row
from evals.agent.evaluators.minutes import (
    WORST_FIRST,
    NaiveDatetimeError,
    aware,
    excluded_minutes,
    expand,
    minute_span,
    minute_time,
    range_text,
    runs,
    window_of,
)
from flysafelz.contracts import Verdict

SHEET = Path(__file__).resolve().parents[3] / "evals" / "agent" / "data" / "verdict_sheet.yaml"
NO_GO, UNVERIFIED, PD = Verdict.NO_GO, Verdict.UNVERIFIED, Verdict.PILOT_DISCRETION


def whole(start, end) -> range:
    return minute_span(start, end)


def test_a_span_is_half_open_and_adjacent_spans_share_no_minute():
    first, second = whole(t(10), t(10, 6)), whole(t(10, 6), t(10, 20))
    assert len(first) == 6 and len(second) == 14
    assert minute_time(first.stop - 1) == t(10, 5) and minute_time(second.start) == t(10, 6)
    assert not set(first) & set(second)
    assert len(whole(t(10), t(11))) == 60


def test_a_span_starting_or_ending_between_two_minutes_holds_the_minute_starts_inside_it():
    assert [minute_time(m) for m in minute_span(t(10).replace(second=30), t(10, 3).replace(second=30))] == [t(10, 1), t(10, 2), t(10, 3)]


def test_an_empty_or_backwards_span_holds_no_minute():
    assert len(whole(t(10), t(10))) == 0 and len(whole(t(11), t(10))) == 0


def test_z_and_an_offset_naming_the_same_instant_give_identical_minutes():
    assert whole("2026-10-31T10:00:00Z", "2026-10-31T11:00:00Z") == whole("2026-10-31T11:00:00+01:00", "2026-10-31T12:00:00+01:00")
    assert aware("2026-10-31T10:00Z", "x") == aware(datetime(2026, 10, 31, 11, tzinfo=timezone(timedelta(hours=1))), "x")


@pytest.mark.parametrize("naive", ["2026-10-31T10:00:00", datetime(2026, 10, 31, 10)])
def test_a_naive_time_is_refused_and_the_message_says_what_it_was_for(naive):
    with pytest.raises(NaiveDatetimeError, match="segment start.*no timezone"):
        aware(naive, "segment start")
    with pytest.raises(NaiveDatetimeError, match="row 3 end"):
        minute_span(t(10), naive, "row 3")
    with pytest.raises(NaiveDatetimeError, match="row 2 start"):  # expand numbers the items from 1
        expand([row(t(8), t(9), "NO_GO"), {"start": naive, "end": t(11), "verdict": "NO_GO"}], whole(t(8), t(12)), "row")


@pytest.mark.parametrize("bad", ["tomorrow", 12, None])
def test_something_that_is_not_a_time_is_a_value_error(bad):
    with pytest.raises(ValueError, match="what"):
        aware(bad, "what")


def test_a_window_is_two_aware_times_with_the_end_not_before_the_start():
    assert len(window_of(["2026-10-31T10:00Z", "2026-10-31T10:30Z"])) == 30
    assert len(window_of([t(10), t(10)])) == 0  # a zero-minute window is allowed: scores say None, never a division by zero
    for bad in (None, "10:00", [t(10)], [t(10), t(11), t(12)], [t(11), t(10)]):
        with pytest.raises(ValueError, match="window"):
            window_of(bad)
    with pytest.raises(NaiveDatetimeError):
        window_of([datetime(2026, 10, 31, 10), t(11)])


# --- E14: clipping ---------------------------------------------------------------------------------------------------------------


def test_a_row_that_straddles_the_window_edge_gives_only_its_in_window_part():
    window = whole(t(10), t(12))
    early = expand([row(t(9), t(11), "NO_GO")], window)
    late = expand([row(t(11, 30), t(13), "NO_GO")], window)
    assert len(early) == 60 and min(early) == window.start
    assert len(late) == 30 and max(late) == window.stop - 1


def test_a_row_wholly_outside_the_window_gives_no_minutes():
    window = whole(t(10), t(12))
    assert expand([row(t(8), t(10), "NO_GO"), row(t(12), t(14), "NO_GO")], window) == {}


# --- overlap and gaps ------------------------------------------------------------------------------------------------------------


def test_the_worst_first_order_is_the_contract_order():
    assert WORST_FIRST == tuple(Verdict)


@pytest.mark.parametrize("pair", [(NO_GO, UNVERIFIED), (NO_GO, PD), (UNVERIFIED, PD)])
@pytest.mark.parametrize("reverse", [False, True])
def test_two_items_on_one_minute_give_the_worst_verdict_whatever_the_order(pair, reverse):
    items = [row(t(10), t(11), v.value) for v in (reversed(pair) if reverse else pair)]
    got = expand(items, whole(t(10), t(11)))
    assert {v for v, _ in got.values()} == {pair[0]}


def test_approaching_limits_on_an_overlap_comes_from_an_item_with_the_worst_verdict():
    window = whole(t(10), t(11))
    assert set(expand([row(t(10), t(11), "NO_GO", False), row(t(10), t(11), "PILOT_DISCRETION", True)], window).values()) == {(NO_GO, False)}
    assert set(expand([row(t(10), t(11), "NO_GO", False), row(t(10), t(11), "NO_GO", True)], window).values()) == {(NO_GO, True)}


def test_a_minute_no_item_covers_is_absent():
    got = expand([row(t(10), t(10, 10), "NO_GO"), row(t(10, 20), t(10, 30), "NO_GO")], whole(t(10), t(10, 30)))
    assert len(got) == 20 and not any(minute_time(m) >= t(10, 10) and minute_time(m) < t(10, 20) for m in got)


def test_an_unknown_verdict_is_a_value_error_naming_the_item():
    with pytest.raises(ValueError, match="row 2.*MAYBE"):
        expand([row(t(10), t(11), "NO_GO"), row(t(10), t(11), "MAYBE")], whole(t(10), t(11)), "row")


# --- the row shapes of the sheet -------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("shape", "item", "verdict_minutes", "excluded"),
    [
        ("plain", row(t(10), t(11), "PILOT_DISCRETION"), 60, 0),
        ("edge named", row(t(10), t(10, 30), "NO_GO", True, edge="civil dusk"), 30, 0),
        ("tier", tier_row(t(10), t(11), "OUTLOOK"), 0, 60),
        ("tier TOO_FAR", tier_row(t(10), t(11), "TOO_FAR"), 0, 60),
    ],
)
def test_each_row_shape_expands_to_verdict_minutes_or_excluded_minutes(shape, item, verdict_minutes, excluded):
    window = whole(t(8), t(18))
    assert len(expand([item], window)) == verdict_minutes
    assert len(excluded_minutes([item], window)) == excluded


def test_every_committed_sheet_entry_expands_and_its_rows_sum_to_the_minutes_they_cover():
    entries = yaml.safe_load(SHEET.read_text(encoding="utf-8"))
    assert len(entries) >= 16
    shapes = set()
    for entry in entries:
        rows = entry.get("rows") or []
        if not rows:
            continue
        window = whole(entry["window"]["start"], entry["window"]["end"])
        shapes |= {tuple(sorted(r)) for r in rows}
        covered = set()
        for r in rows:
            covered |= set(whole(r["start"], r["end"])) & set(window)
        expanded, left_out = expand(rows, window), excluded_minutes(rows, window)
        assert set(expanded) | left_out == covered and not set(expanded) & left_out, entry["id"]
    assert {"approaching_limits", "edge", "end", "start", "verdict"} <= {k for shape in shapes for k in shape}


# --- helpers ---------------------------------------------------------------------------------------------------------------------


def test_runs_are_the_contiguous_stretches_in_time_order():
    assert runs({5, 6, 7, 9, 10, 20}) == [range(5, 8), range(9, 11), range(20, 21)]
    assert runs([]) == []


def test_range_text_names_the_minutes_in_utc_and_both_dates_across_midnight():
    assert range_text(whole(t(17, 6), t(17, 20))) == "2026-10-31 17:06–17:20Z"
    assert range_text(whole(t(23), datetime(2026, 11, 1, 1, tzinfo=UTC))) == "2026-10-31 23:00Z–2026-11-01 01:00Z"
