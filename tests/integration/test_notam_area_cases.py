"""G16 on the two captured cases Dan labelled: the area item E draws decides where a large NOTAM is.

druridge_bay_2026-07-22: Storm Warrior H2483/26 (Q-circle 152 NM, item E polygon round the LZ, SFC to UNL, a phone contact).
bamburgh_2026-10-06:     Strike Warrior H5874/26 (Q-circle 194 NM, item E polygon far to the north of the LZ).

Both folders are byte-for-byte copies of the labelled files, except druridge_bay's readings.yaml, where by design the labelled
`[]` line was replaced by the H2483/26 entry. Offline: nothing here fetches anything. No test holds the real phone number, only the action.
"""

import math
from datetime import UTC, datetime

import cases
import yaml
from factories import LZ, segment

from flysafelz.contracts import NotamAction, NotamReading
from flysafelz.notam_action import merged_limits
from flysafelz.notam_area import Shape, area_from_item_e

DRURIDGE, BAMBURGH = cases.DRURIDGE_BAY, cases.BAMBURGH
H2483, H5874 = "H2483/26", "H5874/26"
SCHEDULE_END = datetime(2026, 7, 22, 12, 0, tzinfo=UTC)  # H2483's item D is 0800-1200 (UTC)


def segments(assessment):
    return [s for h in assessment.hours for s in h.segments]


def notam_of(name: str, notam_id: str):
    _case, _forecast, _airspace, pib, _readings = cases.load(name)
    return next(n for n in pib.notams if n.id == notam_id)


# --- Druridge Bay: H2483/26 -------------------------------------------------------------------------------------------


def test_h2483_is_a_polygon_in_item_e_although_its_q_circle_is_area_wide():
    parsed = notam_of(DRURIDGE, H2483)
    assert parsed.radius_nm >= 50  # the area-wide rule would have said AWARENESS
    area = area_from_item_e(parsed)
    assert isinstance(area, Shape) and area.kind == "polygon"


def test_h2483_is_contact_first_in_every_segment_where_it_is_active_and_never_awareness():
    found = {}
    for segment_ in segments(cases.run(DRURIDGE)):
        actions = {a.notam_id: a.action for a in segment_.notam_actions}
        found[segment_.start] = actions.get(H2483)
    active = {start: action for start, action in found.items() if action is not None}
    assert active, "H2483/26 is in no segment"
    assert set(active.values()) == {NotamAction.CONTACT_FIRST}
    assert all(start < SCHEDULE_END for start in active)  # 0800-1200: it is not listed once it has ended
    assert all(action is not None for start, action in found.items() if start < SCHEDULE_END)  # and it is listed in every earlier segment


def test_h2483_reading_validates_as_a_notam_reading_and_is_the_hand_transcription():
    _case, _forecast, _airspace, _pib, readings = cases.load(DRURIDGE)
    reading = readings[H2483]
    assert NotamReading.model_validate(reading.model_dump()) == reading
    raw = yaml.safe_load((cases.case_dir(DRURIDGE) / "readings.yaml").read_text(encoding="utf-8"))
    assert [entry["id"] for entry in raw] == [H2483]
    assert raw[0]["schedule"] == "0800-1200" and raw[0]["activates_designators"] == []


def test_h2483_ceiling_is_no_upper_limit_and_does_not_disagree_with_item_g_unl():
    """The reading says FL999 and item G says UNL: the same thing, with no 'disagree' note anywhere."""
    _case, _forecast, _airspace, pib, readings = cases.load(DRURIDGE)
    parsed = next(n for n in pib.notams if n.id == H2483)
    _floor, ceiling = merged_limits(parsed, readings[H2483], LZ, segment())
    assert math.isinf(ceiling)
    for segment_ in segments(cases.run(DRURIDGE)):
        assert not [c for c in segment_.conditions if "disagree" in c and "ceiling" in c], segment_.start


# --- Bamburgh: H5874/26 -----------------------------------------------------------------------------------------------


def test_h5874_is_a_polygon_far_to_the_north_so_it_is_ignored_with_the_distance_in_the_reason():
    parsed = notam_of(BAMBURGH, H5874)
    assert parsed.radius_nm >= 50 and isinstance(area_from_item_e(parsed), Shape)
    assessment = cases.run(BAMBURGH)
    reasons = {i.notam_id: i.reason for i in assessment.ignored_notams}
    assert H5874 in reasons
    assert reasons[H5874].endswith("NM from the LZ, beyond the 2 NM local area")
    distance = float(reasons[H5874].split(" NM from the LZ")[0])
    assert distance > 2
    assert all(a.notam_id != H5874 for s in segments(assessment) for a in s.notam_actions)
