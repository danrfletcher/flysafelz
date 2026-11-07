"""Q8: Blue Mill with readings.yaml removed. An unread NOTAM over the LZ from the surface is UNVERIFIED with its E text, never NO_FLY.

The E text in an Assessment is the redacted one (<PHONE_n>, R3.5): the CLI puts the real number back when it prints."""

import cases
from factories import utc

from flysafelz.contracts import NotamAction, Verdict
from flysafelz.redact import redact_phones

CASE = cases.BLUE_MILL


def segments(assessment):
    return [s for h in assessment.hours for s in h.segments]


def e_text(notam_id: str) -> str:
    _case, _forecast, _airspace, pib, _readings = cases.load(CASE)
    return " ".join(redact_phones(next(n for n in pib.notams if n.id == notam_id).item_e)[0].split())


def test_the_danger_area_segments_are_unverified_with_the_e_text_unless_weather_or_night_blocks():
    assessment = cases.run(CASE, readings=False)
    text = e_text("J3857/26")
    active = [s for s in segments(assessment) if s.start >= utc(8)]
    assert active
    for segment in active:
        if segment.verdict is Verdict.NO_GO:
            assert {b.source for b in segment.blockers} <= {"weather", "daylight"}, segment.start  # never the NOTAM
        else:
            assert segment.verdict is Verdict.UNVERIFIED, segment.start
        assert any(c.startswith("UNVERIFIED: ") and text in c for c in segment.conditions), segment.start
        assert any("EGD496A" in c and "has not been read" in c for c in segment.conditions), segment.start


def test_no_segment_is_ever_no_fly_without_a_reading():
    for segment in segments(cases.run(CASE, readings=False)):
        assert NotamAction.NO_FLY not in {a.action for a in segment.notam_actions}
        assert not [b for b in segment.blockers if b.source in {"notam", "airspace"}]


def test_unread_notams_are_listed_as_unread_over_lz_and_nothing_is_flyable_while_they_are_active():
    """J3857 draws no area, so its Q-circle (over the LZ) makes it unread. H7171's E text draws a circle 0.15 NM from the LZ: code
    locates it (G16), so with no reading it is a plain sector to avoid, not UNREAD_OVER_LZ."""
    assessment = cases.run(CASE, readings=False)
    for segment in segments(assessment):
        if segment.start >= utc(8):
            assert {a.notam_id: a.action for a in segment.notam_actions} == {"H7171/26": NotamAction.AVOID_SECTOR, "J3857/26": NotamAction.UNREAD_OVER_LZ}
    assert all(w.end <= utc(8) for w in assessment.flyable_windows)


def test_with_the_readings_the_same_segments_are_not_unverified():
    read = segments(cases.run(CASE))
    assert Verdict.UNVERIFIED not in {s.verdict for s in read}
