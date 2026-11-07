"""assess() over the committed West Sleekburn case: the Newcastle CTR ceiling PR-7's capture established."""

import cases

from flysafelz.assess import display

CASE = cases.WEST_SLEEKBURN
CEILING = "stay below 1,300 ft QNH"


def segments(assessment):
    return [s for h in assessment.hours for s in h.segments]


def test_every_daylight_segment_carries_the_ceiling_condition():
    daylight = [s for s in segments(cases.run(CASE)) if not any(b.source == "daylight" for b in s.blockers)]
    assert daylight
    for segment in daylight:
        assert any(CEILING in c for c in segment.conditions), (segment.start, segment.conditions)


def test_no_segment_is_plain_pilot_discretion():
    assert all(display(s) != "PILOT DISCRETION" for s in segments(cases.run(CASE)))


def test_the_ceiling_reaches_the_flyable_windows_as_a_condition_not_a_block():
    assessment = cases.run(CASE)
    assert assessment.flyable_windows
    flyable = [s for s in segments(assessment) if s.verdict.value == "PILOT_DISCRETION"]
    assert flyable and all(any(CEILING in c for c in s.conditions) for s in flyable)
