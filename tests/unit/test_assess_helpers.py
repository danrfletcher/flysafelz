"""G2: the two public helpers assess cuts an hour with: notams_in_hour() and cut_points_with_reasons().

Daylight is checked against the daylight module's own rules (daylight_edges, daylight_phase) for every
hour of the day at Blue Mill on 31 Oct 2026. NOTAMs are synthetic, built with the factories.
"""

from datetime import timedelta

import pytest
from factories import LZ, circle_at, notam, reading, utc

from flysafelz.assess import cut_points_with_reasons, notams_in_hour
from flysafelz.daylight import ONE_HOUR, ONE_MINUTE, Phase, daylight_edges, daylight_phase
from flysafelz.pib import SortAction, sort_notam

NIGHT, TWILIGHT, DAY = Phase.NIGHT, Phase.TWILIGHT, Phase.DAY


def cuts(hour, *notams):
    """cut_points_with_reasons() at Blue Mill, each NOTAM with a reading so its schedule is read."""
    readings = {n.id: reading(notam_id=n.id, geometry=circle_at(0)) for n in notams}
    return cut_points_with_reasons(utc(hour), LZ.lat, LZ.lon, list(notams), readings, LZ)


def active(notam_id, start, end):
    return notam(notam_id, circle=circle_at(0), valid_from=start, valid_to=end)


# --- daylight reasons -----------------------------------------------------------------------


@pytest.mark.parametrize("hour", range(24))
def test_with_no_notams_the_cuts_are_the_daylight_edges_inside_the_hour_and_each_reason_is_the_phase_change(hour):
    start = utc(hour)
    expected = [t for t, _phase in daylight_edges(LZ.lat, LZ.lon, start, start + ONE_HOUR) if start < t < start + ONE_HOUR]
    got = cuts(hour)
    assert [t for t, _ in got] == expected
    for t, reasons in got:
        assert reasons == [(daylight_phase(LZ.lat, LZ.lon, t - ONE_MINUTE), daylight_phase(LZ.lat, LZ.lon, t))]


def test_the_day_has_the_four_daylight_changes_in_order_dawn_sunrise_sunset_dusk():
    every = [(t, r) for hour in range(24) for t, r in cuts(hour)]
    assert every == [
        (utc(6, 21), [(NIGHT, TWILIGHT)]),  # civil dawn
        (utc(6, 59), [(TWILIGHT, DAY)]),  # sunrise
        (utc(16, 28), [(DAY, TWILIGHT)]),  # sunset
        (utc(17, 6), [(TWILIGHT, NIGHT)]),  # civil dusk
    ]


def test_the_dawn_hour_has_dawn_and_sunrise_as_in_the_segment_tests():
    assert cuts(6) == [(utc(6, 21), [(NIGHT, TWILIGHT)]), (utc(6, 59), [(TWILIGHT, DAY)])]


def test_an_hour_with_no_daylight_or_notam_edge_has_no_cuts():
    assert cuts(12) == []
    assert cuts(2) == []


# --- NOTAM reasons --------------------------------------------------------------------------


def test_a_notam_starting_inside_the_hour_cuts_there_with_a_starts_reason():
    assert cuts(9, active("J1/26", utc(9, 17), utc(12))) == [(utc(9, 17), [("J1/26", "starts")])]


def test_a_notam_ending_inside_the_hour_cuts_there_with_an_ends_reason():
    assert cuts(10, active("J1/26", utc(8), utc(10, 43))) == [(utc(10, 43), [("J1/26", "ends")])]


def test_a_notam_that_starts_and_ends_inside_one_hour_gives_two_cuts_starts_then_ends():
    assert cuts(9, active("J1/26", utc(9, 17), utc(9, 43))) == [(utc(9, 17), [("J1/26", "starts")]), (utc(9, 43), [("J1/26", "ends")])]


def test_a_notam_active_across_the_whole_hour_adds_no_cut_and_no_reason():
    assert cuts(9, active("J1/26", utc(8), utc(11))) == []


def test_a_notam_starting_exactly_on_the_hour_adds_no_cut():
    # E10: 08:00 is not strictly inside the 08:00 hour. Naming that edge is the sheet's job (G14).
    assert cuts(8, active("J1/26", utc(8), utc(12))) == []


def test_a_notam_ending_exactly_on_the_hour_adds_no_cut_in_the_hour_before_or_after():
    n = active("J1/26", utc(7), utc(9))
    assert cuts(8, n) == []
    assert cuts(9, n) == []


def test_two_notams_on_the_same_minute_share_one_cut_with_the_ids_in_ascending_order():
    got = cuts(9, active("J2/26", utc(9, 30), utc(12)), active("H1/26", utc(9, 30), utc(12)))
    assert got == [(utc(9, 30), [("H1/26", "starts"), ("J2/26", "starts")])]


def test_a_start_and_an_end_on_the_same_minute_share_one_cut():
    got = cuts(9, active("H1/26", utc(8), utc(9, 30)), active("J2/26", utc(9, 30), utc(12)))
    assert got == [(utc(9, 30), [("H1/26", "ends"), ("J2/26", "starts")])]


def test_a_notam_start_on_the_sunset_minute_shares_the_cut_with_daylight_first():
    got = cuts(16, active("A1/26", utc(16, 28), utc(20)))
    assert got == [(utc(16, 28), [(DAY, TWILIGHT), ("A1/26", "starts")])]


def test_several_cuts_are_in_time_order_whatever_their_source():
    got = cuts(16, active("J1/26", utc(16, 45), utc(20)), active("H1/26", utc(8), utc(16, 10)))
    assert [t for t, _ in got] == [utc(16, 10), utc(16, 28), utc(16, 45)]


def test_the_same_minute_gives_the_same_order_whatever_the_input_order():
    first, second = active("H1/26", utc(9, 30), utc(12)), active("J2/26", utc(9, 30), utc(12))
    assert cuts(9, first, second) == cuts(9, second, first)


# --- notams_in_hour -------------------------------------------------------------------------


def synthetic():
    """Four NOTAMs for the 09:00 hour: two apply, one has ended, one starts later."""
    return [
        notam("J1/26", circle=circle_at(0), valid_from=utc(8), valid_to=utc(12)),
        notam("J2/26", circle=circle_at(0), valid_from=utc(1), valid_to=utc(8)),
        notam("J3/26", circle=circle_at(0), valid_from=utc(9, 30), valid_to=utc(12)),
        notam("J4/26", circle=circle_at(0), valid_from=utc(11), valid_to=utc(12)),
    ]


def test_notams_in_hour_keeps_exactly_those_sort_notam_does_not_ignore_in_input_order():
    notams = synthetic()
    kept = notams_in_hour(utc(9), notams, {}, LZ)
    expected = [n for n in notams if sort_notam(n, (utc(9), utc(10)), (LZ.lat, LZ.lon)).action is not SortAction.IGNORE]
    assert [n.id for n in kept] == [n.id for n in expected] == ["J1/26", "J3/26"]
    assert all(a is b for a, b in zip(kept, [notams[0], notams[2]]))


def test_notams_in_hour_with_an_empty_list_returns_nothing_and_leaves_ignored_alone():
    ignored = {"X1/26": "kept"}
    assert notams_in_hour(utc(9), [], {}, LZ, ignored) == []
    assert ignored == {"X1/26": "kept"}


def test_notams_in_hour_records_each_ignore_reason_by_id_and_the_first_reason_wins():
    notams = synthetic()
    ignored: dict[str, str] = {}
    notams_in_hour(utc(9), notams, {}, LZ, ignored)
    assert list(ignored) == ["J2/26", "J4/26"]
    assert ignored["J2/26"].startswith("not active in the span: ended")
    assert ignored["J4/26"].startswith("not active in the span: starts")
    # A later hour where J2/26 is ignored again, differently, leaves the first reason alone.
    first = dict(ignored)
    notams_in_hour(utc(13), notams, {}, LZ, ignored)
    assert ignored["J2/26"] == first["J2/26"]
    assert ignored["J1/26"].startswith("not active in the span: ended")  # a new id is added


def test_notams_in_hour_with_every_notam_ignored_returns_nothing_and_records_each_once():
    ignored: dict[str, str] = {}
    gone = [notam(f"J{i}/26", circle=circle_at(0), valid_from=utc(1), valid_to=utc(2)) for i in (1, 2, 3)]
    assert notams_in_hour(utc(9), gone, {}, LZ, ignored) == []
    assert list(ignored) == ["J1/26", "J2/26", "J3/26"]


def test_notams_in_hour_works_without_an_ignored_argument():
    assert [n.id for n in notams_in_hour(utc(9), synthetic(), {}, LZ)] == ["J1/26", "J3/26"]


def test_notams_in_hour_takes_the_hour_not_the_day():
    n = notam("J1/26", circle=circle_at(0), valid_from=utc(9, 30), valid_to=utc(9, 45))
    assert notams_in_hour(utc(9), [n], {}, LZ) == [n]
    assert notams_in_hour(utc(9) - timedelta(hours=1), [n], {}, LZ) == []
