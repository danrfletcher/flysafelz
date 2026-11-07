"""Q10 / E7 / G6: how an hour is cut into segments, to the minute, by daylight edges and NOTAM edges."""

from datetime import timedelta

from factories import (
    circle_at,
    exemption,
    notam,
    pib_result,
    reading,
    run_assess,
    utc,
)

from flysafelz.contracts import NotamAction, Verdict


def segments_of(assessment):
    return [s for h in assessment.hours for s in h.segments]


def spans(assessment):
    return [(s.start, s.end) for s in segments_of(assessment)]


def test_civil_dawn_and_sunrise_in_one_hour_make_three_segments():
    # Blue Mill on 31 Oct 2026: civil dawn 06:21, sunrise 06:59 (see PR-2's helper).
    assessment = run_assess(window=(utc(6), utc(7)))
    segments = segments_of(assessment)
    assert spans(assessment) == [(utc(6), utc(6, 21)), (utc(6, 21), utc(6, 59)), (utc(6, 59), utc(7))]
    assert [s.verdict for s in segments] == [Verdict.NO_GO, Verdict.PILOT_DISCRETION, Verdict.PILOT_DISCRETION]
    assert segments[0].blockers[0].source == "daylight"  # night
    assert segments[1].approaching_limits and segments[1].cautions  # twilight
    assert not segments[2].approaching_limits  # day


def test_dusk_splits_the_hour_at_the_exact_minute_twilight_then_night():
    segments = segments_of(run_assess(window=(utc(17), utc(18))))
    assert [(s.start, s.end) for s in segments] == [(utc(17), utc(17, 6)), (utc(17, 6), utc(18))]
    assert segments[0].approaching_limits and segments[0].verdict is Verdict.PILOT_DISCRETION
    assert segments[1].verdict is Verdict.NO_GO


def test_a_notam_start_and_end_inside_an_hour_split_it_to_the_minute():
    n = notam("J1/26", circle=circle_at(0), valid_from=utc(9, 17), valid_to=utc(10, 43))
    assessment = run_assess(window=(utc(9), utc(11)), pib=pib_result(n), readings={n.id: reading(notam_id=n.id, geometry=circle_at(0))})
    assert spans(assessment) == [(utc(9), utc(9, 17)), (utc(9, 17), utc(10)), (utc(10), utc(10, 43)), (utc(10, 43), utc(11))]
    listed = [[a.notam_id for a in s.notam_actions] for s in segments_of(assessment)]
    assert listed == [[], ["J1/26"], ["J1/26"], []]  # the NOTAM does not leak into its sibling segments


def test_an_edge_exactly_on_the_hour_cuts_nothing_and_leaves_no_zero_length_segment():
    n = notam("J1/26", circle=circle_at(0), valid_from=utc(10), valid_to=utc(11))
    assessment = run_assess(window=(utc(9), utc(12)), pib=pib_result(n), readings={n.id: reading(notam_id=n.id, geometry=circle_at(0))})
    assert spans(assessment) == [(utc(9), utc(10)), (utc(10), utc(11)), (utc(11), utc(12))]
    assert all(s.start < s.end for s in segments_of(assessment))
    assert [len(h.segments) for h in assessment.hours] == [1, 1, 1]


def test_segments_tile_every_hour_with_no_gap_and_no_overlap():
    n = notam("J1/26", circle=circle_at(0), valid_from=utc(6, 40), valid_to=utc(7, 20))
    assessment = run_assess(window=(utc(6), utc(8)), pib=pib_result(n), readings={n.id: reading(notam_id=n.id, geometry=circle_at(0))})
    for hour in assessment.hours:
        edges = [hour.hour, *[s.end for s in hour.segments]]
        assert [s.start for s in hour.segments] == edges[:-1]
        assert edges[-1] == hour.hour + timedelta(hours=1)


def test_an_ignore_in_one_segment_does_not_leak_into_its_sibling():
    # Active only from 09:30. Before then it is IGNORE, after it is CONTACT_FIRST.
    n = notam("J1/26", circle=circle_at(0), valid_from=utc(9, 30), valid_to=utc(18))
    assessment = run_assess(window=(utc(9), utc(10)), pib=pib_result(n), readings={n.id: reading(notam_id=n.id, geometry=circle_at(0))})
    first, second = segments_of(assessment)
    assert (first.start, second.start) == (utc(9), utc(9, 30))
    assert first.notam_actions == [] and first.conditions == []
    assert [a.action for a in second.notam_actions] == [NotamAction.CONTACT_FIRST]
    assert second.conditions


def test_a_notam_ignored_for_the_whole_window_is_in_the_debug_list_and_in_no_segment():
    n = notam("J1/26", circle=circle_at(0), item_e="UAS OPR WI 0.5NM. PARAMOTORS EXEMPT.")
    exempt = reading(notam_id=n.id, geometry=circle_at(0), exemptions=[exemption("PARAMOTORS EXEMPT")])
    assessment = run_assess(pib=pib_result(n), readings={n.id: exempt})
    assert [i.notam_id for i in assessment.ignored_notams] == ["J1/26"]
    assert assessment.ignored_notams[0].reason
    assert segments_of(assessment)  # there are segments, and none of them mentions it
    assert all(a.notam_id != "J1/26" for s in segments_of(assessment) for a in s.notam_actions)
