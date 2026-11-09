"""assess() over the committed Blue Mill case (fixtures/blue_mill_2026-10-31/): now 07:30Z, window 07:00 to 18:00Z."""

import cases
from factories import utc

from flysafelz.assess import display
from flysafelz.contracts import NotamAction, Verdict

CASE = cases.BLUE_MILL


def segments(assessment):
    return [s for h in assessment.hours for s in h.segments]


def test_the_window_has_11_rows_07_to_17():
    assessment = cases.run(CASE)
    assert [h.hour for h in assessment.hours] == [utc(h) for h in range(7, 18)]
    assert (assessment.window_start, assessment.window_end) == (utc(7), utc(18))


def test_from_08_every_segment_has_j3857_contact_first_and_h7171_avoid_sector():
    late = [s for s in segments(cases.run(CASE)) if s.start >= utc(8)]
    assert late
    for segment in late:
        actions = {a.notam_id: a.action for a in segment.notam_actions}
        assert actions["J3857/26"] is NotamAction.CONTACT_FIRST, segment.start
        assert actions["H7171/26"] is NotamAction.AVOID_SECTOR, segment.start
        # the airspace path gives the call-Flylogix condition (J3857/26 owns EGD496A) and names the SUP, once
        danger = [c for c in segment.conditions if "EGD496A" in c]
        assert len(danger) == 1 and danger[0].startswith("Danger area EGD496A active 08:00") and danger[0].endswith("; details in AIP SUP 108/2026"), segment.start
        assert not any("read the SUP" in c for c in segment.conditions), segment.start
        assert display(segment).endswith("WITH CONDITIONS") or segment.verdict is Verdict.NO_GO, segment.start


def test_before_08_neither_notam_is_active_and_neither_is_listed():
    early = [s for s in segments(cases.run(CASE)) if s.end <= utc(8)]
    assert early
    assert all(s.notam_actions == [] for s in early)


def test_no_segment_is_plain_pilot_discretion():
    assert all(display(s) != "PILOT DISCRETION" for s in segments(cases.run(CASE)))


def test_the_17_00_row_splits_at_17_06_twilight_then_night():
    last = cases.run(CASE).hours[-1]
    assert last.hour == utc(17)
    twilight, night = last.segments
    assert (twilight.start, twilight.end, night.start, night.end) == (utc(17), utc(17, 6), utc(17, 6), utc(18))
    assert twilight.verdict is Verdict.PILOT_DISCRETION and twilight.approaching_limits
    assert any("twilight" in c.text for c in twilight.cautions)
    assert night.verdict is Verdict.NO_GO
    assert [b.source for b in night.blockers] == ["daylight"]


def test_the_last_flyable_window_ends_at_17_06():
    windows = cases.run(CASE).flyable_windows
    assert windows
    assert windows[-1].end == utc(17, 6)
    assert all(w.start < w.end for w in windows)


def test_the_overall_verdict_is_no_go_because_of_the_bad_visibility_and_night():
    assessment = cases.run(CASE)
    assert assessment.overall_verdict is Verdict.NO_GO
    assert [s.verdict for s in segments(assessment) if s.start == utc(8)] == [Verdict.NO_GO]  # visibility 1,340 m


def test_the_danger_area_is_decided_once_by_airspace_and_not_repeated_by_the_notam():
    segment = next(s for s in segments(cases.run(CASE)) if s.start == utc(11))
    danger = [c for c in segment.conditions if "EGD496A" in c]
    assert len(danger) == 1
    assert "phone Flylogix" in danger[0]
    assert danger[0].endswith("; details in AIP SUP 108/2026")
    assert danger[0].count("details in") == 1


def test_the_debug_list_carries_each_ignored_31_oct_notam_with_a_reason():
    _case, _forecast, _airspace, pib, _readings = cases.load(CASE)
    assessment = cases.run(CASE)
    listed = {a.notam_id for s in segments(assessment) for a in s.notam_actions}
    debug = {i.notam_id: i.reason for i in assessment.ignored_notams}
    assert listed == {"H7171/26", "J3857/26"}
    on_the_day = {n.id for n in pib.notams}
    assert on_the_day - listed <= set(debug), sorted(on_the_day - listed - set(debug))
    assert debug and all(debug.values())
    assert not (set(debug) & listed)
