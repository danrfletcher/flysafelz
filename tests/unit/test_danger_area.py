"""G10: danger areas. Whether one is active in a segment, and what an active one means over the LZ.

Table-driven over the activation kind (byNotam, published hours, H24, unparseable, empty), the
reading (none, unread, phone, email, radio-only, prohibitive) and in or out of the schedule. Includes
the Blue Mill shape: TDA EGD496A activated 0800-1800 by J3857/26 with a hand-written reading.
"""

import re
from datetime import UTC, datetime

import pytest
from airspace_helpers import E_TEXT, LZ, SEGMENT, activation, amsl, assess, blue_mill_tda, only, reading, utc

from flysafelz import airspace as airspace_module
from flysafelz.airspace import Activation, Outcome, TimeSpan, danger_area_active
from flysafelz.clients.openaip import parse_airspace
from flysafelz.contracts import Category, Circle, Contact, ContactKind, ProhibitiveReason
from flysafelz.geo import distance_nm, shape_over
from flysafelz.verdict_sheet_draft import SEVERITY

PHONE_CONDITION = "Danger area EGD496A active 08:00\u201318:00 (UAS BVLOS ops): phone Flylogix 07700 900123 and fly only if they agree"


def test_blue_mill_is_inside_the_tda_and_about_95_m_from_its_south_west_edge():
    area = blue_mill_tda()
    assert shape_over(*LZ, area.shape)
    corner = area.shape[0][0][0]
    assert 80 < distance_nm(*LZ, *corner) * 1852 < 200  # the nearest corner is a few hundred metres at most


# --- the Blue Mill shape -----------------------------------------------------------------


def test_a_read_activation_with_a_phone_contact_is_contact_first_with_the_permission_wording():
    result = assess([blue_mill_tda()], activations=[activation(read=reading())])
    finding = only(result)
    assert finding.outcome is Outcome.CONTACT_FIRST
    assert finding.condition == PHONE_CONDITION
    assert "J3857/26" in finding.reason and finding.bearing_deg is None


def test_the_same_activation_without_a_reading_is_unverified_with_the_e_text():
    finding = only(assess([blue_mill_tda()], activations=[activation(read=None)]))
    assert finding.outcome is Outcome.UNVERIFIED
    assert E_TEXT in finding.reason and "J3857/26" in finding.reason and finding.condition is None


def test_an_email_contact_names_the_email():
    read = reading(contact=ContactKind.EMAIL, value="ops@flylogix.example")
    finding = only(assess([blue_mill_tda()], activations=[activation(read=read)]))
    assert finding.outcome is Outcome.CONTACT_FIRST
    assert finding.condition == "Danger area EGD496A active 08:00\u201318:00 (UAS BVLOS ops): email ops@flylogix.example and fly only if they agree"


def test_a_blank_activity_leaves_out_the_brackets():
    finding = only(assess([blue_mill_tda()], activations=[activation(read=reading(activity="  "))]))
    assert finding.condition == "Danger area EGD496A active 08:00\u201318:00: phone Flylogix 07700 900123 and fly only if they agree"


def test_the_window_in_the_wording_is_the_activation_in_utc():
    late = activation(read=reading(), start=utc(8, 30), end=utc(17, 45))
    assert "active 08:30\u201317:45" in (only(assess([blue_mill_tda()], activations=[late])).condition or "")


# --- the table: kind x reading x schedule ----------------------------------------------------

READINGS = {
    "unread": (None, Outcome.UNVERIFIED),
    "phone": (reading(), Outcome.CONTACT_FIRST),
    "email": (reading(contact=ContactKind.EMAIL, value="ops@example.com"), Outcome.CONTACT_FIRST),
    "radio-only": (reading(contact=ContactKind.RADIO_ONLY, value="Anglia Radar 120.620"), Outcome.CHECK_SUP),
    "no contact": (reading(contact=ContactKind.NONE, value=None), Outcome.CHECK_SUP),
    "phone without a number": (reading(value="  "), Outcome.CHECK_SUP),
    "byelaws + phone": (reading(reason=ProhibitiveReason.BYELAWS), Outcome.NO_FLY),
    "entry prohibited + phone": (reading(reason=ProhibitiveReason.ENTRY_PROHIBITED), Outcome.NO_FLY),
    "live firing + phone": (reading(reason=ProhibitiveReason.LIVE_FIRING), Outcome.NO_FLY),
    "prohibitive category, reason none + phone": (reading(category=Category.PROHIBITIVE), Outcome.CONTACT_FIRST),
    "informational category, reason none + phone": (reading(category=Category.INFORMATIONAL), Outcome.CONTACT_FIRST),
}

# (activation kind, area arguments, active with no activation?)
KINDS = {
    "byNotam": ({"by_notam": True}, False),
    "hours, inside": ({"hours": "DAILY 0800-1800"}, True),
    "hours, outside": ({"hours": "DAILY 1000-1100"}, False),
    "H24": ({"hours": "H24"}, True),
    "unparseable": ({"hours": "WHENEVER THE RANGE IS HOT"}, True),
    "unreadable": ({"hours_readable": False}, True),
    "empty": ({"hours": None}, True),
}


@pytest.mark.parametrize("reading_name", READINGS)
@pytest.mark.parametrize("kind", KINDS)
def test_an_activation_decides_by_its_reading_whatever_the_kind_of_area(kind, reading_name):
    read, expected = READINGS[reading_name]
    if read is None and KINDS[kind][1]:
        expected = Outcome.NO_FLY  # active on its own: an unread NOTAM naming it must not relax NO_FLY to UNVERIFIED
    area = blue_mill_tda(**{"by_notam": False, **KINDS[kind][0]})
    assert only(assess([area], activations=[activation(read=read)])).outcome is expected


@pytest.mark.parametrize("kind", KINDS)
def test_with_no_activation_the_kind_decides(kind):
    args, active = KINDS[kind]
    area = blue_mill_tda(**{"by_notam": False, **args})
    finding = only(assess([area]))
    assert finding.outcome is (Outcome.NO_FLY if active else Outcome.IGNORE)
    if active:
        assert "no NOTAM" in finding.reason and "do not fly" in finding.reason
    else:
        assert "not active" in finding.reason


@pytest.mark.parametrize("kind", KINDS)
def test_danger_area_active_for_each_kind(kind):
    args, active = KINDS[kind]
    area = blue_mill_tda(**{"by_notam": False, **args})
    result = danger_area_active(area, SEGMENT, [])
    assert result.active is active and result.activations == []
    if kind in ("H24", "unparseable", "unreadable", "empty"):
        assert result.notes  # noted


def _weekday_segment(hour: int, day: int = 4) -> TimeSpan:
    """2026-11-04 is a Wednesday."""
    return TimeSpan(start=datetime(2026, 11, day, hour, tzinfo=UTC), end=datetime(2026, 11, day, hour + 1, tzinfo=UTC))


@pytest.mark.parametrize("top_level_only", [False, True])
def test_r1_published_hours_and_by_notam_is_active_inside_the_hours_with_no_notam(top_level_only):
    """'MON-FRI 0800-1700, other times by NOTAM' (openAIP: hours plus a byNotam entry, or a top-level byNotam)."""
    geometry = {
        "type": "Polygon",
        "coordinates": [[[LZ[1] - 0.01, LZ[0] - 0.01], [LZ[1] + 0.01, LZ[0] - 0.01], [LZ[1] + 0.01, LZ[0] + 0.01], [LZ[1] - 0.01, LZ[0] + 0.01], [LZ[1] - 0.01, LZ[0] - 0.01]]],
    }
    hours = [{"dayOfWeek": d, "startTime": "08:00", "endTime": "17:00"} for d in range(1, 6)]
    raw = {"_id": "x", "name": "EGD496A", "type": 2, "geometry": geometry, "hoursOfOperation": {"operatingHours": hours}}
    if top_level_only:
        raw["byNotam"] = True
    else:
        raw["hoursOfOperation"]["operatingHours"].append({"dayOfWeek": 0, "byNotam": True})
    area = parse_airspace(raw)
    assert area.by_notam and area.hours == "MON 0800-1700, TUE 0800-1700, WED 0800-1700, THU 0800-1700, FRI 0800-1700"
    inside = _weekday_segment(10)
    finding = only(assess([area], segment=inside))
    assert finding.outcome is Outcome.NO_FLY and "published hours" in finding.reason
    # outside the hours, with no NOTAM: not active
    assert only(assess([area], segment=_weekday_segment(20))).outcome is Outcome.IGNORE
    # outside the hours, a NOTAM activates it (byNotam is one more way in)
    evening = _weekday_segment(20)
    note = activation(read=reading(), start=evening.start, end=evening.end)
    assert only(assess([area], segment=evening, activations=[note])).outcome is Outcome.CONTACT_FIRST


def test_r1_by_notam_with_hours_is_active_inside_them_in_danger_area_active():
    area = blue_mill_tda(hours="DAILY 0800-1800")  # by_notam=True as well
    inside = danger_area_active(area, SEGMENT, [])
    assert inside.active and inside.active_alone
    outside = danger_area_active(area, TimeSpan(start=utc(19), end=utc(20)), [])
    assert not outside.active and "outside published hours" in outside.reason
    assert danger_area_active(area, TimeSpan(start=utc(19), end=utc(20)), [activation(start=utc(19), end=utc(20))]).active


def test_r1_by_notam_with_unreadable_hours_is_active_and_noted():
    result = danger_area_active(blue_mill_tda(hours_readable=False), SEGMENT, [])
    assert result.active and result.active_alone and result.notes


def test_r1_by_notam_with_hours_and_no_designator_is_active_outside_the_hours_and_noted():
    area = blue_mill_tda(hours="DAILY 1000-1100", designator=None, name="MYSTERY TDA")
    result = danger_area_active(area, SEGMENT, [])
    assert result.active and not result.active_alone and any("no designator" in n for n in result.notes)


def test_r2_an_unread_notam_never_relaxes_an_area_that_is_active_on_its_own():
    unread = activation(read=None)
    for args in ({"hours": "H24"}, {"hours": None}, {"hours": "DAILY 0000-2400"}, {"hours": "DAILY 0800-1800"}):
        area = blue_mill_tda(by_notam=False, **args)
        finding = only(assess([area], activations=[unread]))
        assert finding.outcome is Outcome.NO_FLY and "J3857/26" in finding.reason and "has not been read" in finding.reason
        assert only(assess([area], activations=[activation(read=reading()), unread])).outcome is Outcome.NO_FLY
    # a by-NOTAM-only area (not active alone) still goes to UNVERIFIED for an unread NOTAM
    assert only(assess([blue_mill_tda()], activations=[unread])).outcome is Outcome.UNVERIFIED
    # outside its published hours an unread NOTAM is the only thing activating it: UNVERIFIED
    assert only(assess([blue_mill_tda(by_notam=False, hours="DAILY 1000-1100")], activations=[unread])).outcome is Outcome.UNVERIFIED


def test_an_activation_makes_a_by_notam_area_active_and_names_the_notam():
    result = danger_area_active(blue_mill_tda(), SEGMENT, [activation()])
    assert result.active and "J3857/26" in result.reason and len(result.activations) == 1


def test_published_hours_use_the_segment_to_the_minute():
    area = blue_mill_tda(by_notam=False, hours="DAILY 0800-1800")
    assert danger_area_active(area, TimeSpan(start=utc(7), end=utc(8)), []).active is False
    assert danger_area_active(area, TimeSpan(start=utc(7, 30), end=utc(8, 1)), []).active is True
    assert danger_area_active(area, TimeSpan(start=utc(17, 59), end=utc(19)), []).active is True
    assert danger_area_active(area, TimeSpan(start=utc(18), end=utc(19)), []).active is False


def test_sunrise_to_sunset_hours_are_worked_out_at_the_area():
    area = blue_mill_tda(by_notam=False, hours="DAILY SR-SS")
    assert danger_area_active(area, SEGMENT, []).active is True  # 09:00 on 31 Oct
    assert danger_area_active(area, TimeSpan(start=utc(2), end=utc(3)), []).active is False
    assert danger_area_active(area, TimeSpan(start=utc(20), end=utc(21)), []).active is False


def test_an_activation_must_cover_the_segment_and_name_the_area():
    area = blue_mill_tda()
    before = activation(start=utc(1), end=utc(9))  # ends as the segment starts
    after = activation(start=utc(10), end=utc(12))  # starts as the segment ends
    assert danger_area_active(area, SEGMENT, [before]).active is False
    assert danger_area_active(area, SEGMENT, [after]).active is False
    assert danger_area_active(area, SEGMENT, [activation("EGD497A")]).active is False
    assert danger_area_active(area, SEGMENT, [activation(start=utc(9, 59), end=utc(12))]).active is True  # any overlap fails closed


def test_an_activation_of_one_area_does_not_activate_another_with_a_similar_designator():
    assert danger_area_active(blue_mill_tda(), SEGMENT, [activation("EGD496")]).active is False
    assert danger_area_active(blue_mill_tda(), SEGMENT, [activation("EGD496B")]).active is False


def test_a_by_notam_area_with_no_designator_cannot_be_matched_so_it_is_active_and_noted():
    area = blue_mill_tda(designator=None, name="MYSTERY TDA")
    result = danger_area_active(area, SEGMENT, [activation()])
    assert result.active and result.notes and "no designator" in result.notes[0]


# --- F11: a contact never undoes a prohibition -------------------------------------------------


@pytest.mark.parametrize("why", [ProhibitiveReason.BYELAWS, ProhibitiveReason.ENTRY_PROHIBITED, ProhibitiveReason.LIVE_FIRING])
@pytest.mark.parametrize("kind", [ContactKind.PHONE, ContactKind.EMAIL])
def test_f11_a_prohibitive_reason_with_a_contact_is_no_fly_never_contact_first(why, kind):
    read = reading(contact=kind, value="someone@example.com or 01234 567890", reason=why)
    finding = only(assess([blue_mill_tda()], activations=[activation(read=read)]))
    assert finding.outcome is Outcome.NO_FLY and finding.condition is None
    assert why.value in finding.reason


def test_f11_a_prohibitive_reason_gives_no_fly_with_the_quote_in_the_reason():
    read = reading(reason=ProhibitiveReason.BYELAWS, quote="ENTRY PROHIBITED UNDER BYELAWS")
    finding = only(assess([blue_mill_tda()], activations=[activation(read=read)]))
    assert finding.outcome is Outcome.NO_FLY and 'prohibited (byelaws): "ENTRY PROHIBITED UNDER BYELAWS"' in finding.reason
    assert "J3857/26" in finding.reason


def test_f11_a_quote_never_carries_a_number_into_the_reason():
    read = reading(reason=ProhibitiveReason.ENTRY_PROHIBITED, quote="MUST NOT ENTER, ASK 07700 900123")
    finding = only(assess([blue_mill_tda()], activations=[activation(read=read)]))
    assert "07700" not in finding.reason and "<PHONE_1>" in finding.reason


def test_cr2_a_prohibitive_category_with_reason_none_and_a_phone_is_contact_first_not_no_fly():
    """E8: the airspace path follows the NOTAM table, which keys on prohibitive_reason and never on the category."""
    read = reading(category=Category.PROHIBITIVE)
    finding = only(assess([blue_mill_tda()], activations=[activation(read=read, notam_id="J3857/26")]))
    assert finding.outcome is Outcome.CONTACT_FIRST and finding.condition == PHONE_CONDITION
    assert "J3857/26" in finding.reason


def test_cr2_a_prohibitive_category_with_reason_none_and_no_contact_is_check_sup():
    read = reading(category=Category.PROHIBITIVE, contact=ContactKind.NONE, value=None)
    finding = only(assess([blue_mill_tda()], activations=[activation(read=read)]))
    assert finding.outcome is Outcome.CHECK_SUP and "J3857/26" in finding.reason


def test_f11_radio_only_is_check_sup_and_says_a_radio_frequency_is_not_a_contact():
    read = reading(contact=ContactKind.RADIO_ONLY, value="Anglia Radar 120.620")
    finding = only(assess([blue_mill_tda()], activations=[activation(read=read)]))
    assert finding.outcome is Outcome.CHECK_SUP and "radio frequency is not a contact" in finding.reason
    assert finding.condition == "read the SUP / AIP ENR 5.1 entry for EGD496A before flight"


def test_a_contact_object_that_is_phone_but_empty_is_check_sup():
    read = reading().model_copy(update={"contact": Contact(kind=ContactKind.PHONE, value=None)})
    assert only(assess([blue_mill_tda()], activations=[activation(read=read)])).outcome is Outcome.CHECK_SUP


# --- CHECK_SUP and the SUP suffix ------------------------------------------------------------------------


SUP = "AIP SUP 108/2026"


def test_check_sup_names_the_designator_and_the_sup():
    read = reading(contact=ContactKind.NONE, value=None)
    result = assess([blue_mill_tda()], activations=[activation(read=read, sup=[SUP])])
    finding = only(result)
    assert finding.outcome is Outcome.CHECK_SUP
    assert finding.condition == "read the SUP / AIP ENR 5.1 entry for EGD496A before flight; details in AIP SUP 108/2026"


def test_check_sup_names_every_sup_in_order():
    read = reading(contact=ContactKind.NONE, value=None)
    finding = only(assess([blue_mill_tda()], activations=[activation(read=read, sup=[SUP, "AIP SUP 7/2026"])]))
    assert finding.condition is not None and finding.condition.endswith("; details in AIP SUP 108/2026, AIP SUP 7/2026")


def test_contact_first_appends_the_sup_to_the_unchanged_call_wording():
    finding = only(assess([blue_mill_tda()], activations=[activation(read=reading(), sup=[SUP])]))
    assert finding.outcome is Outcome.CONTACT_FIRST
    assert finding.condition == PHONE_CONDITION + "; details in AIP SUP 108/2026"


@pytest.mark.parametrize("read", [reading(), reading(contact=ContactKind.NONE, value=None)])
def test_no_sup_reference_means_no_suffix_no_empty_brackets_and_no_trailing_separator(read):
    condition = only(assess([blue_mill_tda()], activations=[activation(read=read)])).condition or ""
    assert "details" not in condition and "[" not in condition and "()" not in condition and not condition.rstrip().endswith(";")


def test_the_reasons_of_contact_first_and_check_sup_hold_no_number_and_no_item_e():
    for read in (reading(), reading(contact=ContactKind.NONE, value=None)):
        finding = only(assess([blue_mill_tda()], activations=[activation(read=read, e_text="REDACTED <PHONE_1>")]))
        assert "<PHONE_1>" not in finding.reason and "07700" not in finding.reason


# --- the code decides the contact; a disagreement is noted -------------------------------------------------


def test_code_finds_no_contact_but_the_reading_has_a_phone_code_decides_and_a_note_says_so():
    result = assess([blue_mill_tda()], activations=[activation(read=reading(), contact=ContactKind.NONE)])
    assert only(result).outcome is Outcome.CHECK_SUP
    assert any("J3857/26" in n and "code finds none in item E" in n and "CHECK_SUP" in n for n in result.notes)


def test_code_finds_a_contact_but_the_reading_has_none_code_decides_and_points_at_the_notam():
    read = reading(contact=ContactKind.RADIO_ONLY, value="120.620")
    result = assess([blue_mill_tda()], activations=[activation(read=read, contact=ContactKind.PHONE)])
    finding = only(result)
    assert finding.outcome is Outcome.CONTACT_FIRST
    assert finding.condition == "Danger area EGD496A active 08:00\u201318:00 (UAS BVLOS ops): phone given in NOTAM J3857/26 and fly only if they agree"
    assert any("code finds a phone in item E" in n and "CONTACT_FIRST" in n for n in result.notes)


def test_no_note_when_the_code_and_the_reading_agree():
    assert assess([blue_mill_tda()], activations=[activation(read=reading())]).notes == []
    assert assess([blue_mill_tda()], activations=[activation(read=reading(contact=ContactKind.NONE, value=None))]).notes == []


def test_an_activation_cannot_claim_a_radio_frequency_as_its_contact():
    with pytest.raises(ValueError, match="radio frequency"):
        activation(contact=ContactKind.RADIO_ONLY)


# --- several activations: NO_FLY, then UNVERIFIED, then CHECK_SUP, then CONTACT_FIRST ------------------------


def test_the_worst_covering_activation_wins():
    good, unread = activation(read=reading()), activation(notam_id="J3999/26", read=None)
    no_contact = activation(notam_id="J3997/26", read=reading(contact=ContactKind.RADIO_ONLY, value="120.620"))
    bad = activation(notam_id="J3998/26", read=reading(reason=ProhibitiveReason.LIVE_FIRING, quote="LIVE FIRING"))
    area = blue_mill_tda()
    assert only(assess([area], activations=[good])).outcome is Outcome.CONTACT_FIRST
    assert only(assess([area], activations=[good, no_contact])).outcome is Outcome.CHECK_SUP
    assert only(assess([area], activations=[no_contact, good])).outcome is Outcome.CHECK_SUP
    assert only(assess([area], activations=[no_contact, unread])).outcome is Outcome.UNVERIFIED
    assert only(assess([area], activations=[unread, no_contact])).outcome is Outcome.UNVERIFIED
    assert only(assess([area], activations=[good, unread])).outcome is Outcome.UNVERIFIED
    assert only(assess([area], activations=[unread, good])).outcome is Outcome.UNVERIFIED
    assert only(assess([area], activations=[good, unread, bad])).outcome is Outcome.NO_FLY
    assert only(assess([area], activations=[bad, good])).outcome is Outcome.NO_FLY
    assert only(assess([area], activations=[no_contact, bad])).outcome is Outcome.NO_FLY


def test_a_no_fly_reading_plus_a_contact_reading_gives_no_fly_and_names_the_prohibiting_notam():
    bad = activation(notam_id="J3998/26", read=reading(reason=ProhibitiveReason.BYELAWS, quote="BYELAWS APPLY"))
    finding = only(assess([blue_mill_tda()], activations=[activation(read=reading()), bad]))
    assert finding.outcome is Outcome.NO_FLY and "J3998/26" in finding.reason


def test_check_sup_plus_contact_first_gives_check_sup_and_names_the_check_sup_notam():
    no_contact = activation(notam_id="J3997/26", read=reading(contact=ContactKind.NONE, value=None))
    finding = only(assess([blue_mill_tda()], activations=[activation(read=reading()), no_contact]))
    assert finding.outcome is Outcome.CHECK_SUP and "J3997/26" in finding.reason


def test_an_activation_outside_the_segment_is_not_counted_among_them():
    late_bad = activation(notam_id="J3998/26", start=utc(12), end=utc(14), read=reading(contact=ContactKind.NONE, value=None))
    finding = only(assess([blue_mill_tda()], activations=[activation(read=reading()), late_bad]))
    assert finding.outcome is Outcome.CONTACT_FIRST


def test_there_is_one_finding_per_danger_area_however_many_activations():
    result = assess(
        [blue_mill_tda()],
        activations=[activation(read=reading()), activation(notam_id="J3858/26", read=reading()), activation(notam_id="J3859/26", read=reading(contact=ContactKind.NONE, value=None))],
    )
    assert len(result.findings) == 1


# --- G13: no activation, or unread on an area that is active alone, stays NO_FLY ------------------------------


@pytest.mark.parametrize("hours", ["H24", None, "WHENEVER THE RANGE IS HOT", "DAILY 0800-1800"])
def test_g13_active_alone_with_no_activation_stays_no_fly(hours):
    finding = only(assess([blue_mill_tda(by_notam=False, hours=hours)]))
    assert finding.outcome is Outcome.NO_FLY and "no NOTAM gives a way in" in finding.reason


def test_g13_an_unread_activation_of_an_area_active_alone_stays_no_fly_and_of_a_by_notam_area_stays_unverified():
    unread = activation(read=None)
    assert only(assess([blue_mill_tda(by_notam=False, hours="H24")], activations=[unread])).outcome is Outcome.NO_FLY
    finding = only(assess([blue_mill_tda()], activations=[unread]))
    assert finding.outcome is Outcome.UNVERIFIED and E_TEXT in finding.reason


# --- the docstring says what the code does ------------------------------------------------------------------


def test_the_module_docstring_table_matches_the_rules():
    doc = airspace_module.__doc__ or ""
    # the stale rule is gone
    assert "has no phone or email (a radio frequency" not in doc
    assert "not a prohibitive category" not in doc
    # every outcome a danger area over the LZ can give is in the table, with the rule that gives it
    for outcome in ("NO_FLY", "UNVERIFIED", "CHECK_SUP", "CONTACT_FIRST"):
        assert outcome in doc
    assert "NO_FLY beats UNVERIFIED beats CHECK_SUP beats CONTACT_FIRST" in doc
    assert "category never decides" in " ".join(doc.split())
    # the same order as the verdict sheet's severity list and as airspace.Outcome
    order = [a for a in SEVERITY if a.value in ("NO_FLY", "UNREAD_OVER_LZ", "CHECK_SUP", "CONTACT_FIRST")]
    assert [a.value for a in order] == ["NO_FLY", "UNREAD_OVER_LZ", "CHECK_SUP", "CONTACT_FIRST"]
    for outcome in (Outcome.NO_FLY, Outcome.UNVERIFIED, Outcome.CHECK_SUP, Outcome.CONTACT_FIRST):
        assert outcome.value in doc
    # each row of the table names the behaviour the tests above pin
    assert re.search(r"no phone or email in its original item E\s+CHECK_SUP", doc)
    assert re.search(r"a phone or email in its original item E\s+CONTACT_FIRST", doc)
    assert re.search(r"prohibitive reason \(quoted\)\s+NO_FLY", doc)


# --- designators: any way of writing one matches --------------------------------------------------


@pytest.mark.parametrize("area_name", ["D496A", "EGD 496A", "EGD496A", "EG D496A"])
@pytest.mark.parametrize("notam_text", ["D496A", "EGD 496A", "EGD496A", "egd496a"])
def test_a_designator_written_any_way_in_openaip_or_the_notam_is_the_same_area(area_name, notam_text):
    geometry = {
        "type": "Polygon",
        "coordinates": [[[LZ[1] - 0.01, LZ[0] - 0.01], [LZ[1] + 0.01, LZ[0] - 0.01], [LZ[1] + 0.01, LZ[0] + 0.01], [LZ[1] - 0.01, LZ[0] + 0.01], [LZ[1] - 0.01, LZ[0] - 0.01]]],
    }
    area = parse_airspace({"_id": "x", "name": area_name, "type": 2, "byNotam": True, "geometry": geometry})
    finding = only(assess([area], activations=[activation(notam_text, read=reading())]))
    assert finding.outcome is Outcome.CONTACT_FIRST


def test_an_activation_with_something_that_is_not_a_designator_is_refused():
    with pytest.raises(ValueError):
        Activation(designator="CAMBRIDGE", notam_id="A1/26", e_text="x", start=utc(8), end=utc(9))


# --- the polygon: openAIP first, the NOTAM's Q-circle when openAIP lacks the area ---------------


QCIRCLE = Circle(lat=53 + 56 / 60, lon=1 / 60, radius_nm=10.0)  # EGD496A: 10 NM at 5356N00001E


def test_the_q_circle_of_the_notam_covers_blue_mill():
    assert distance_nm(*LZ, QCIRCLE.lat, QCIRCLE.lon) < QCIRCLE.radius_nm  # still over-cautious, still safe


def test_when_openaip_lacks_the_area_the_q_circle_stands_in():
    result = assess([], activations=[activation(read=reading(), circle=QCIRCLE)])
    finding = only(result)
    assert finding.outcome is Outcome.CONTACT_FIRST and finding.condition == PHONE_CONDITION
    assert any("Q-circle of NOTAM J3857/26" in n for n in result.notes)


def test_the_q_circle_without_a_reading_is_unverified():
    finding = only(assess([], activations=[activation(read=None, circle=QCIRCLE)]))
    assert finding.outcome is Outcome.UNVERIFIED and E_TEXT in finding.reason


def test_the_q_circle_is_not_used_when_openaip_has_the_area():
    result = assess([blue_mill_tda()], activations=[activation(read=reading(), circle=Circle(lat=10.0, lon=10.0, radius_nm=1.0))])
    assert only(result).outcome is Outcome.CONTACT_FIRST  # the polygon placed it; the far-off circle was never drawn


def test_the_q_circle_is_never_smaller_than_the_circle():
    for bearing_east in (0.0, 0.7, -0.7):
        for bearing_north in (0.0, 0.7, -0.7):
            if bearing_east == bearing_north == 0.0:
                continue
            # a point a hair inside the circle's edge, in 8 directions, is over the stand-in polygon
            lat = QCIRCLE.lat + bearing_north * 0.9998 * QCIRCLE.radius_nm / 60 / (bearing_east**2 + bearing_north**2) ** 0.5
            lon = QCIRCLE.lon + bearing_east * 0.9998 * QCIRCLE.radius_nm / 60 / (bearing_east**2 + bearing_north**2) ** 0.5 / 0.5913
            assert assess([], lz=(lat, lon), activations=[activation(read=reading(), circle=QCIRCLE)]).findings


def test_an_activation_openaip_lacks_and_with_no_circle_is_unverified_not_dropped():
    finding = only(assess([], activations=[activation(read=reading())]))
    assert finding.outcome is Outcome.UNVERIFIED
    assert "openAIP does not have the area" in finding.reason and E_TEXT in finding.reason


def test_an_unplaceable_activation_outside_the_segment_says_nothing():
    late = activation(start=utc(12), end=utc(14))
    assert assess([], activations=[late]).findings == []


def test_a_q_circle_stand_in_takes_its_floor_from_the_reading():
    high_reading = reading().model_copy(update={"floor": amsl(2000)})
    finding = only(assess([], activations=[activation(read=high_reading, circle=QCIRCLE)]))
    assert finding.outcome is Outcome.CEILING and finding.condition == "stay below 1,800 ft QNH"



def test_a_q_circle_stand_in_whose_reading_states_no_floor_comes_down_to_the_surface():
    no_floor = reading().model_copy(update={"floor": None})
    finding = only(assess([], activations=[activation(read=no_floor, circle=QCIRCLE)]))
    assert finding.outcome is Outcome.CONTACT_FIRST and finding.condition == PHONE_CONDITION

# --- above SFC, and inactive ------------------------------------------------------------------------


def test_an_active_danger_area_with_a_floor_above_sfc_follows_the_height_bands():
    area = blue_mill_tda(by_notam=False, hours="H24", lower=amsl(2000))
    assert only(assess([area])).condition == "stay below 1,800 ft QNH"
    assert only(assess([blue_mill_tda(by_notam=False, hours="H24", lower=amsl(7000))])).outcome is Outcome.WARNING
    assert only(assess([blue_mill_tda(by_notam=False, hours="H24", lower=amsl(12000))])).outcome is Outcome.AWARENESS


def test_an_inactive_danger_area_over_the_lz_is_ignored_with_a_debug_reason():
    finding = only(assess([blue_mill_tda()]))
    assert finding.outcome is Outcome.IGNORE and "EGD496A" in finding.reason and "not active" in finding.reason
    assert finding.condition is None


def test_the_by_notam_area_is_only_active_in_the_segment_the_notam_covers():
    area = blue_mill_tda()
    note = activation(read=reading(), start=utc(8), end=utc(18))
    assert only(assess([area], activations=[note])).outcome is Outcome.CONTACT_FIRST
    after = TimeSpan(start=utc(18), end=utc(19))
    assert only(assess([area], segment=after, activations=[note])).outcome is Outcome.IGNORE
