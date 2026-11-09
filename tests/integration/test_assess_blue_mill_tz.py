"""assess() over the committed Blue Mill case with tz="Europe/London" on 31 Oct (GMT): the same verdicts, the time text labelled GMT.

Offline: tests/cases.py replays fixtures/blue_mill_2026-10-31 (now 07:30Z, window 07:00 to 18:00Z). On 31 Oct London is on GMT, so the
clock reads the same as UTC and only the label is new. The goldens are not touched.
"""

import cases
import pytest

from flysafelz.assess import display

CASE = cases.BLUE_MILL
LONDON = "Europe/London"


def segments(assessment):
    return [s for h in assessment.hours for s in h.segments]


def masked(assessment):
    """The dump with the texts that can hold a time blanked, so everything else is compared."""
    dump = assessment.model_dump()
    dump["assumptions"] = ["<text>"] * len(dump["assumptions"])
    for hour in dump["hours"]:
        for segment in hour["segments"]:
            segment["conditions"] = ["<text>"] * len(segment["conditions"])
            for action in segment["notam_actions"]:
                action["reason"] = "<text>"
    for status in dump["source_statuses"]:
        status["reason"] = "<text>"
    for ignored in dump["ignored_notams"]:
        ignored["reason"] = "<text>"
    return dump


@pytest.mark.parametrize("tz", [None, "UTC", LONDON])
def test_every_verdict_equals_the_utc_run(tz):
    plain = cases.run(CASE)
    zoned = cases.run(CASE, tz=tz)
    assert [(s.start, s.end, s.verdict, s.approaching_limits, display(s)) for s in segments(zoned)] == [(s.start, s.end, s.verdict, s.approaching_limits, display(s)) for s in segments(plain)]
    assert (zoned.overall_verdict, zoned.overall_tier, zoned.flyable_windows) == (plain.overall_verdict, plain.overall_tier, plain.flyable_windows)
    assert masked(zoned) == masked(plain)


def test_none_and_utc_are_byte_identical_to_the_run_with_no_tz():
    plain = cases.run(CASE).model_dump()
    assert cases.run(CASE, tz=None).model_dump() == plain
    assert cases.run(CASE, tz="UTC").model_dump() == plain


def test_london_on_31_oct_labels_the_condition_gmt_and_the_clock_matches_utc():
    plain = cases.run(CASE)
    zoned = cases.run(CASE, tz=LONDON)
    danger_plain = {c for s in segments(plain) for c in s.conditions if c.startswith("Danger area EGD496A")}
    danger_zoned = {c for s in segments(zoned) for c in s.conditions if c.startswith("Danger area EGD496A")}
    assert len(danger_plain) == len(danger_zoned) == 1
    [utc_text], [gmt_text] = sorted(danger_plain), sorted(danger_zoned)
    assert utc_text.startswith("Danger area EGD496A active 08:00\u201318:00 (")
    assert gmt_text.startswith("Danger area EGD496A active 08:00\u201318:00 GMT (")
    assert gmt_text.replace(" GMT", "", 1) == utc_text  # the label is the only difference
    assert all(m.utcoffset().total_seconds() == 0 for m in (zoned.window_start, zoned.window_end))  # type: ignore[union-attr]
