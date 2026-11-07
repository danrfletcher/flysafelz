"""G2 / G14 / E9 / E10 / E11: the edge name of a sheet row, from assess's own cut reasons.

edge_name() is checked as a table over the reasons it is given. segment_rows() is checked through the real
assess() at Blue Mill with factory NOTAMs, so the names come out of the same public helpers assess cuts with.
Dawn and dusk dates are fixed: Blue Mill on 2026-10-31 has civil dawn 06:21, sunrise 06:59, sunset 16:28, civil
dusk 17:06 (UTC); on 2026-11-03 civil dusk is exactly 17:00; on 2026-03-07 civil dawn is exactly 06:00; on
2026-01-24 sunrise is exactly 08:00; on 2026-01-09 sunset is exactly 16:00.
"""

from datetime import timedelta

import pytest
from factories import LZ, circle_at, notam, pib_result, run_assess, utc

from flysafelz.daylight import Phase
from flysafelz.pib import prefilter
from flysafelz.verdict_sheet import edge_name, segment_rows

NIGHT, TWILIGHT, DAY = Phase.NIGHT, Phase.TWILIGHT, Phase.DAY
ONE_HOUR = timedelta(hours=1)


# --- edge_name over the reasons ---------------------------------------------------------------


@pytest.mark.parametrize(
    ("reasons", "name"),
    [
        ([], "hour"),
        ([(NIGHT, TWILIGHT)], "civil dawn"),
        ([(TWILIGHT, DAY)], "sunrise"),
        ([(DAY, TWILIGHT)], "sunset"),
        ([(TWILIGHT, NIGHT)], "civil dusk"),
        ([("J3857/26", "starts")], "J3857/26 starts"),
        ([("J3857/26", "ends")], "J3857/26 ends"),
        # several in one minute: daylight first, then NOTAM ids sorted, a start before an end of the same id
        ([("B2/26", "starts"), ("A1/26", "starts")], "A1/26 starts + B2/26 starts"),
        ([("B2/26", "ends"), ("A1/26", "starts")], "A1/26 starts + B2/26 ends"),
        ([("A1/26", "ends"), ("A1/26", "starts")], "A1/26 starts + A1/26 ends"),
        ([("A1/26", "starts"), (DAY, TWILIGHT)], "sunset + A1/26 starts"),
        ([("A1/26", "ends"), ("B2/26", "starts"), (TWILIGHT, NIGHT)], "civil dusk + A1/26 ends + B2/26 starts"),
        ([(DAY, NIGHT)], "daylight DAY to NIGHT"),  # not a real change (the phases pass through twilight): still named, never crashes
    ],
)
def test_edge_name_table(reasons, name):
    assert edge_name(reasons) == name


# --- segment_rows through the real assess() ----------------------------------------------------------


def rows_for(*notams, window=None, now=None):
    """The sheet rows of a Blue Mill assessment with these NOTAMs, as (start, end, edge) in UTC minutes of the window's day."""
    window = window or (utc(8), utc(12))
    kwargs = {} if now is None else {"now": now}
    pib = pib_result(*notams)
    assessment = run_assess(window=window, pib=pib, **kwargs)
    span = (assessment.hours[0].hour, assessment.hours[-1].hour + ONE_HOUR)
    near, _dropped = prefilter(pib.notams, (LZ.lat, LZ.lon), span)
    rows = segment_rows(assessment, LZ.lat, LZ.lon, near, {}, LZ)
    return [(r["start"][11:16], r["end"][11:16], r["edge"]) for r in rows]


def over_lz(notam_id, start, end):
    return notam(notam_id, circle=circle_at(0), valid_from=start, valid_to=end)


def test_e9_an_hour_with_no_edge_is_one_row_named_hour_and_the_first_row_is_window_start():
    assert rows_for(window=(utc(9), utc(12))) == [("09:00", "10:00", "window start"), ("10:00", "11:00", "hour"), ("11:00", "12:00", "hour")]


def test_e11_a_window_starting_at_07_30_gives_a_first_row_from_07_00_named_window_start():
    assert rows_for(window=(utc(7, 30), utc(9)))[0] == ("07:00", "08:00", "window start")


def test_the_four_daylight_names_on_31_oct_and_hour_everywhere_else():
    assert rows_for(window=(utc(6), utc(18))) == [
        ("06:00", "06:21", "window start"),
        ("06:21", "06:59", "civil dawn"),
        ("06:59", "07:00", "sunrise"),
        *[(f"{h:02d}:00", f"{h + 1:02d}:00", "hour") for h in range(7, 16)],
        ("16:00", "16:28", "hour"),
        ("16:28", "17:00", "sunset"),
        ("17:00", "17:06", "hour"),
        ("17:06", "18:00", "civil dusk"),
    ]


NOTAM_CASES = {
    # a NOTAM starting exactly on HH:00 cuts nothing, and names that row (E10)
    "starts_on_the_hour": ([over_lz("X1/26", utc(10), utc(14))], [("08:00", "09:00", "window start"), ("09:00", "10:00", "hour"), ("10:00", "11:00", "X1/26 starts"), ("11:00", "12:00", "hour")]),
    # one ending exactly on HH:00 names the row that starts there (G14)
    "ends_on_the_hour": ([over_lz("X1/26", utc(7), utc(10))], [("08:00", "09:00", "window start"), ("09:00", "10:00", "hour"), ("10:00", "11:00", "X1/26 ends"), ("11:00", "12:00", "hour")]),
    # one running across the hour is not an edge there: hour, not starts (G14)
    "runs_across_every_hour": ([over_lz("X1/26", utc(7), utc(13))], [("08:00", "09:00", "window start"), ("09:00", "10:00", "hour"), ("10:00", "11:00", "hour"), ("11:00", "12:00", "hour")]),
    "starts_and_ends_inside_hours": (
        [over_lz("X1/26", utc(9, 20), utc(10, 40))],
        [("08:00", "09:00", "window start"), ("09:00", "09:20", "hour"), ("09:20", "10:00", "X1/26 starts"), ("10:00", "10:40", "hour"), ("10:40", "11:00", "X1/26 ends"), ("11:00", "12:00", "hour")],
    ),
    # several edges in one minute: ids sorted, in the same order whatever the PIB order
    "two_starts_same_minute_ab": ([over_lz("A1/26", utc(9, 20), utc(14)), over_lz("B2/26", utc(9, 20), utc(14))], None),
    "two_starts_same_minute_ba": ([over_lz("B2/26", utc(9, 20), utc(14)), over_lz("A1/26", utc(9, 20), utc(14))], None),
    "an_end_and_a_start_on_the_hour": ([over_lz("B2/26", utc(7), utc(10)), over_lz("A1/26", utc(10), utc(14))], None),
    "a_notam_that_starts_at_the_window_start_does_not_rename_the_first_row": ([over_lz("X1/26", utc(8), utc(14))], [("08:00", "09:00", "window start"), ("09:00", "10:00", "hour"), ("10:00", "11:00", "hour"), ("11:00", "12:00", "hour")]),
}


@pytest.mark.parametrize(("notams", "expected"), [pytest.param(*v, id=k) for k, v in NOTAM_CASES.items() if v[1] is not None])
def test_notam_edges(notams, expected):
    assert rows_for(*notams) == expected


@pytest.mark.parametrize("order", [("A1/26", "B2/26"), ("B2/26", "A1/26")])
def test_two_starts_in_one_minute_are_joined_in_id_order_whatever_the_pib_order(order):
    rows = rows_for(*(over_lz(i, utc(9, 20), utc(14)) for i in order))
    assert rows[2] == ("09:20", "10:00", "A1/26 starts + B2/26 starts")
    assert len(rows) == 5


def test_an_end_and_a_start_on_the_same_hour_are_joined_in_id_order():
    rows = rows_for(over_lz("B2/26", utc(7), utc(10)), over_lz("A1/26", utc(10), utc(14)))
    assert ("10:00", "11:00", "A1/26 starts + B2/26 ends") in rows


def test_a_notam_edge_on_the_sunset_minute_follows_the_daylight_name():
    rows = rows_for(over_lz("X1/26", utc(16, 28), utc(20)), window=(utc(16), utc(18)))
    assert rows == [("16:00", "16:28", "window start"), ("16:28", "17:00", "sunset + X1/26 starts"), ("17:00", "17:06", "hour"), ("17:06", "18:00", "civil dusk")]


# --- a daylight edge exactly on HH:00 (G14) ------------------------------------------------------------


@pytest.mark.parametrize(
    ("day", "hour", "name"),
    [
        (utc(7, 30, day=3, month=11), 17, "civil dusk"),
        (utc(7, 30, day=7, month=3), 6, "civil dawn"),
        (utc(7, 30, day=24, month=1), 8, "sunrise"),
        (utc(7, 30, day=9, month=1), 16, "sunset"),
    ],
)
def test_a_daylight_edge_exactly_on_the_hour_is_named(day, hour, name):
    first = day.replace(hour=hour - 1, minute=0)
    rows = rows_for(window=(first, first + 3 * ONE_HOUR), now=day)
    by_start = {start: (end, edge) for start, end, edge in rows}
    assert by_start[f"{hour:02d}:00"][1] == name  # the edge names the row that opens at HH:00
    assert any(end == f"{hour:02d}:00" for end, _edge in by_start.values())  # the row before ends there: HH:00 is the cut, not a second one
    assert rows[0][2] == "window start"


def test_a_daylight_edge_and_a_notam_start_on_the_same_hour_are_joined_daylight_first():
    day = utc(7, 30, day=3, month=11)
    first = day.replace(hour=16, minute=0)
    rows = rows_for(notam("X1/26", circle=circle_at(0), valid_from=day.replace(hour=17, minute=0), valid_to=day.replace(hour=20, minute=0)), window=(first, first + 3 * ONE_HOUR), now=day)
    assert ("17:00", "18:00", "civil dusk + X1/26 starts") in rows
