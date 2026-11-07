"""G10: a NOTAM the reader was asked about and got no reading for (reader_status) goes down the no-reading path (Q8).

{read, failed, unavailable} x {over the LZ from SFC, within 2 NM, outside} -> action, verdict and assumption line, run through
the real assess(). Plus E14 (a UAS_ONLY reading is IGNORE, only in the debug list) and the programming-error checks.
"""

import pytest
from factories import circle_at, notam, pib_result, reading, run_assess, utc

from flysafelz.contracts import AppliesTo, NotamAction, Verdict

NOTAM_ID = "J1/26"
WINDOW = (utc(9), utc(10))
UNAVAILABLE_LINE = f"NOTAM reader unavailable for {NOTAM_ID}: treated as unread"

# Where the NOTAM's circle is, as the edge distance from the LZ; the reading (when there is one) says the same circle.
PLACES = {"over the LZ": 0, "within 2 NM": 1.0, "outside": 5.0}


def segment_of(assessment):
    segments = [s for h in assessment.hours for s in h.segments]
    assert len(segments) == 1
    return segments[0]


def run(place: str, how: str):
    """(assessment, its one segment) for a NOTAM at `place` that is `how`: read, failed or unavailable."""
    circle = circle_at(PLACES[place])
    pib = pib_result(notam(NOTAM_ID, circle=circle))
    if how == "read":
        assessment = run_assess(window=WINDOW, pib=pib, readings={NOTAM_ID: reading(notam_id=NOTAM_ID, geometry=circle)})
    else:
        assessment = run_assess(window=WINDOW, pib=pib, reader_status={NOTAM_ID: how})
    return assessment, segment_of(assessment)


def actions(segment):
    return {a.action for a in segment.notam_actions}


# (place, how) -> (the NOTAM's action in the segment, or None when it is not listed; the segment's verdict)
TABLE = [
    ("over the LZ", "read", NotamAction.CONTACT_FIRST, Verdict.PILOT_DISCRETION),
    ("over the LZ", "failed", NotamAction.UNREAD_OVER_LZ, Verdict.UNVERIFIED),
    ("over the LZ", "unavailable", NotamAction.UNREAD_OVER_LZ, Verdict.UNVERIFIED),
    ("within 2 NM", "read", NotamAction.AVOID_SECTOR, Verdict.PILOT_DISCRETION),
    ("within 2 NM", "failed", NotamAction.AVOID_SECTOR, Verdict.PILOT_DISCRETION),  # the code-only action (Q-circle edge), no wording from a reading
    ("within 2 NM", "unavailable", NotamAction.AVOID_SECTOR, Verdict.PILOT_DISCRETION),
    ("outside", "read", None, None),
    ("outside", "failed", None, None),
    ("outside", "unavailable", None, None),
]


@pytest.mark.parametrize(("place", "how", "action", "verdict"), TABLE, ids=[f"{p}-{h}" for p, h, _a, _v in TABLE])
def test_the_action_and_the_verdict_for_each_place_and_reader_outcome(place, how, action, verdict):
    _assessment, segment = run(place, how)
    assert actions(segment) == ({action} if action else set())
    if verdict is not None:
        assert segment.verdict is verdict
    assert segment.verdict is not Verdict.NO_GO  # an unread NOTAM is never NO_FLY or CONTACT_FIRST by itself
    assert NotamAction.NO_FLY not in actions(segment)
    if how != "read":
        assert NotamAction.CONTACT_FIRST not in actions(segment)


@pytest.mark.parametrize("place", PLACES)
@pytest.mark.parametrize("how", ["read", "failed", "unavailable"])
def test_only_an_unavailable_notam_adds_the_assumption_line(place, how):
    assessment, _segment = run(place, how)
    assert (UNAVAILABLE_LINE in assessment.assumptions) is (how == "unavailable")
    assert not [a for a in assessment.assumptions if "reader" in a and how != "unavailable"]


def test_the_failed_notam_over_the_lz_says_it_has_not_been_read_with_its_redacted_e_text():
    pib = pib_result(notam(NOTAM_ID, circle=circle_at(0), item_e="UAS OPR WI 0.5NM. FOR INFO 07700 900123."))
    segment = segment_of(run_assess(window=WINDOW, pib=pib, reader_status={NOTAM_ID: "failed"}))
    [condition] = [c for c in segment.conditions if "has not been read" in c]
    assert "FOR INFO <PHONE_1>" in condition
    assert "07700" not in condition and "900123" not in condition


def test_the_unavailable_line_names_every_id_sorted_once():
    one, two = notam("J9/26", circle=circle_at(0)), notam("J2/26", circle=circle_at(0))
    third = notam("J5/26", circle=circle_at(0))
    assessment = run_assess(window=WINDOW, pib=pib_result(one, two, third), reader_status={"J9/26": "unavailable", "J2/26": "unavailable", "J5/26": "failed"})
    [line] = [a for a in assessment.assumptions if a.startswith("NOTAM reader unavailable")]
    assert line == "NOTAM reader unavailable for J2/26, J9/26: treated as unread"


def test_a_uas_only_reading_is_ignored_and_only_in_the_debug_list():
    """E14: applies_to UAS_ONLY over the LZ comes out of assess as IGNORE: never in a segment, reason 'applies to drones only'."""
    circle = circle_at(0)
    drones = reading(notam_id=NOTAM_ID, geometry=circle, applies_to=AppliesTo.UAS_ONLY)
    assessment = run_assess(window=WINDOW, pib=pib_result(notam(NOTAM_ID, circle=circle)), readings={NOTAM_ID: drones})
    segment = segment_of(assessment)
    assert segment.notam_actions == [] and segment.verdict is not Verdict.UNVERIFIED
    assert [(i.notam_id, i.reason) for i in assessment.ignored_notams] == [(NOTAM_ID, "applies to drones only")]


def test_a_notam_in_both_readings_and_reader_status_is_a_programming_error_naming_the_id():
    circle = circle_at(0)
    with pytest.raises(ValueError, match=NOTAM_ID):
        run_assess(window=WINDOW, pib=pib_result(notam(NOTAM_ID, circle=circle)), readings={NOTAM_ID: reading(notam_id=NOTAM_ID, geometry=circle)}, reader_status={NOTAM_ID: "failed"})


def test_a_status_other_than_failed_or_unavailable_is_refused():
    with pytest.raises(ValueError, match="'read'"):
        run_assess(window=WINDOW, pib=pib_result(notam(NOTAM_ID, circle=circle_at(0))), reader_status={NOTAM_ID: "read"})


def test_both_defaults_are_empty_so_existing_callers_are_unchanged():
    assert run_assess(window=WINDOW).assumptions == run_assess(window=WINDOW, readings={}, reader_status={}).assumptions
