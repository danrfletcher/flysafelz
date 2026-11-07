"""R3.5: the print-time helper that puts real phone numbers back. Placeholders are numbered per NOTAM, so each line uses its own NOTAM's map.

The numbers are Ofcom's drama-range numbers (07700 900xxx / 01632 960xxx).
"""

from airspace_helpers import volume
from factories import NOW, airspace_fetch, circle_at, notam, pib_result, reading, run_assess, utc

from flysafelz import cli
from flysafelz.contracts import AppliesTo, NotamAction, NotamActionResult
from flysafelz.redact import redact_phones

ONE = "07700 900123"
TWO = "01632 960001"


def parsed(notam_id: str, number: str, extra: str = ""):
    return notam(notam_id, circle=circle_at(0), item_e=f"UAS OPS {extra}. BOOKING PHONE {number}.")


def result(notam_id: str, reason: str, action=NotamAction.CONTACT_FIRST):
    return NotamActionResult(notam_id=notam_id, action=action, reason=reason)


def book() -> cli.PhoneBook:
    return cli.PhoneBook.of([parsed("J1/26", ONE, "EGD111A"), parsed("J2/26", TWO, "EGD222B")])


def test_both_notams_have_a_phone_1_that_is_a_different_number():
    maps = book().maps
    assert maps["J1/26"] == {"<PHONE_1>": ONE} and maps["J2/26"] == {"<PHONE_1>": TWO}


def test_an_action_line_is_put_back_with_its_own_notams_map():
    phones = book()
    first = phones.restore_for("J1/26", "NOTAM J1/26 CONTACT_FIRST: phone <PHONE_1> first")
    second = phones.restore_for("J2/26", "NOTAM J2/26 CONTACT_FIRST: phone <PHONE_1> first")
    assert first == f"NOTAM J1/26 CONTACT_FIRST: phone {ONE} first"
    assert second == f"NOTAM J2/26 CONTACT_FIRST: phone {TWO} first"


def test_an_unknown_placeholder_is_left_as_it_is():
    phones = book()
    assert phones.restore_for("J1/26", "ring <PHONE_1> or <PHONE_2>") == f"ring {ONE} or <PHONE_2>"
    assert phones.restore_for("J9/26", "ring <PHONE_1>") == "ring <PHONE_1>"  # a NOTAM with no map


def test_a_line_without_a_placeholder_is_untouched():
    phones = book()
    for line in ("    condition: Danger area EGD111A active 08:00-18:00", "", "NOTAM J1/26 AWARENESS: nothing to ring"):
        assert phones.restore_for("J1/26", line) == line
        assert phones.restore(line, [result("J1/26", "x"), result("J2/26", "y")]) == line


def test_a_condition_is_put_back_with_the_notam_it_names_by_designator():
    phones = book()
    actions = [result("J1/26", "first notam's own words"), result("J2/26", "second notam's own words")]
    assert phones.restore("condition: Danger area EGD222B active: phone <PHONE_1> first", actions) == f"condition: Danger area EGD222B active: phone {TWO} first"
    assert phones.restore("condition: Danger area EGD111A active: phone <PHONE_1> first", actions) == f"condition: Danger area EGD111A active: phone {ONE} first"


def test_a_condition_that_holds_a_notams_reason_is_put_back_with_that_notam():
    phones = book()
    actions = [result("J1/26", "over the LZ: contact <PHONE_1> first"), result("J2/26", "from the surface: ring <PHONE_1> before flying")]
    assert phones.restore("condition: from the surface: ring <PHONE_1> before flying", actions) == f"condition: from the surface: ring {TWO} before flying"


def test_a_condition_is_put_back_with_the_notam_it_names_by_id():
    phones = book()
    actions = [result("J1/26", "first notam's own words"), result("J2/26", "second notam's own words")]
    assert phones.restore("unverified: NOTAM J2/26 has not been read: BOOKING PHONE <PHONE_1>", actions) == f"unverified: NOTAM J2/26 has not been read: BOOKING PHONE {TWO}"


def test_a_line_whose_notam_cannot_be_told_keeps_its_placeholder_when_the_notams_disagree():
    phones = book()
    actions = [result("J1/26", "first notam's own words"), result("J2/26", "second notam's own words")]
    assert phones.restore("condition: ring <PHONE_1>", actions) == "condition: ring <PHONE_1>"  # never another NOTAM's number


def test_a_line_whose_notam_cannot_be_told_is_put_back_when_every_notam_agrees():
    phones = cli.PhoneBook.of([parsed("J1/26", ONE), parsed("J2/26", ONE)])  # both Blue Mill NOTAMs carry the same contact
    actions = [result("J1/26", "first notam's own words"), result("J2/26", "second notam's own words")]
    assert phones.restore("condition: ring <PHONE_1>", actions) == f"condition: ring {ONE}"


def test_a_notam_that_is_not_in_the_segments_actions_is_never_a_guessed_candidate():
    phones = book()
    assert phones.restore("condition: ring <PHONE_1>", [result("J1/26", "a")]) == "condition: ring <PHONE_1>"  # J2 has a number too, so J1 is not certain
    assert phones.restore("condition: ring <PHONE_1>", []) == "condition: ring <PHONE_1>"


def test_a_line_that_names_a_notam_with_no_action_gets_that_notams_number_not_the_actions():
    phones = book()
    only_j2 = [result("J2/26", "second notam's own words")]
    assert phones.restore("condition: Danger area EGD111A active: phone <PHONE_1> first", only_j2) == f"condition: Danger area EGD111A active: phone {ONE} first"
    assert phones.restore("unverified: activated by NOTAM J1/26, not read: <PHONE_1>", only_j2) == f"unverified: activated by NOTAM J1/26, not read: {ONE}"


def test_a_line_that_names_a_notam_with_no_number_keeps_its_placeholder():
    phones = cli.PhoneBook.of([parsed("J1/26", ONE, "EGD111A"), notam("J3/26", circle=circle_at(0), item_e="DANGER AREA EGD333C ACTIVE.")])
    line = "condition: Danger area EGD333C active: phone <PHONE_1> first"
    assert phones.restore(line, [result("J1/26", "x")]) == line


def test_a_designator_the_reading_names_ties_a_line_to_its_notam():
    phones = book()
    phones.redact_contacts({"J2/26": reading(notam_id="J2/26", value="BOOKING <PHONE_1>", designators=["EGD777Z"])})
    assert phones.restore("condition: Danger area EGD777Z active: phone <PHONE_1> first", []) == f"condition: Danger area EGD777Z active: phone {TWO} first"


def test_a_danger_area_activated_by_an_ignored_notam_prints_that_notams_number_not_another_segment_notams():
    """R1, through the real assess() and report(): J1 is drone-only (ignored, so no action) and activates EGD999X over the LZ; J2 is in the segment's actions."""
    notams = [parsed("J1/26", ONE, "EGD999X"), notam("J2/26", circle=circle_at(0), item_e=f"UAS OPS EGD222B. BOOKING PHONE {TWO}.")]
    readings = {
        "J1/26": reading(notam_id="J1/26", geometry=circle_at(0), applies_to=AppliesTo.UAS_ONLY, designators=["EGD999X"], value="BOOKING <PHONE_1>"),
        "J2/26": reading(notam_id="J2/26", geometry=circle_at(0), activity="bravo contact", value="BOOKING <PHONE_1>"),
    }
    danger = volume("danger", name="EGD999X", designator="EGD999X", by_notam=True)
    assessment = run_assess(window=(utc(9), utc(10)), pib=pib_result(*notams), readings=readings, airspace=airspace_fetch(danger))
    segment = assessment.hours[0].segments[0]
    assert "J1/26" not in {a.notam_id for a in segment.notam_actions} and "J2/26" in {a.notam_id for a in segment.notam_actions}
    assert any("EGD999X" in c and "<PHONE_1>" in c for c in segment.conditions)
    text = cli.report(assessment, title="r1", lat=53.8, lon=-0.08, now=NOW, phones=cli.PhoneBook.of(notams))
    danger_lines = [line for line in text.splitlines() if "EGD999X" in line]
    assert danger_lines and all(ONE in line and TWO not in line for line in danger_lines)
    assert any(TWO in line and "bravo contact" in line and ONE not in line for line in text.splitlines())  # J2's own condition keeps J2's number


def test_a_fixtures_hand_written_contact_is_redacted_with_the_placeholder_of_its_notams_e_text():
    phones = book()
    readings = {"J1/26": reading(notam_id="J1/26", value=f"Flylogix {ONE}"), "J2/26": reading(notam_id="J2/26", value=f"Flylogix {TWO}")}
    redacted = phones.redact_contacts(readings)
    assert redacted["J1/26"].contact.value == "Flylogix <PHONE_1>" and redacted["J2/26"].contact.value == "Flylogix <PHONE_1>"
    assert readings["J1/26"].contact.value == f"Flylogix {ONE}"  # the input is not changed
    assert phones.restore_for("J2/26", redacted["J2/26"].contact.value) == f"Flylogix {TWO}"


def test_a_contact_number_that_is_not_in_the_e_text_gets_the_next_placeholder_and_the_map_learns_it():
    phones = book()
    other = "01632 960777"
    redacted = phones.redact_contacts({"J1/26": reading(notam_id="J1/26", value=f"{other} or {ONE}")})
    assert redacted["J1/26"].contact.value == "<PHONE_2> or <PHONE_1>"
    assert phones.restore_for("J1/26", redacted["J1/26"].contact.value) == f"{other} or {ONE}"


def test_a_contact_with_no_number_or_an_email_is_unchanged():
    phones = book()
    mail = reading(notam_id="J1/26", value="ops@example.invalid")
    none = reading(notam_id="J2/26", value=None)
    assert phones.redact_contacts({"J1/26": mail, "J2/26": none}) == {"J1/26": mail, "J2/26": none}


def test_the_book_is_built_from_the_raw_e_text_the_reader_redacted():
    notam_ = parsed("J1/26", ONE)
    assert cli.PhoneBook.of([notam_]).maps["J1/26"] == redact_phones(notam_.item_e)[1]
    assert cli.PhoneBook.of([notam("J3/26", circle=circle_at(0), item_e="NO NUMBER HERE")]).maps == {}


def test_two_notams_in_one_assessment_each_print_their_own_number_in_every_line():
    """G1, R3.5: two NOTAMs, a different number at each one's <PHONE_1>, through the real assess() and the real report()."""
    notams = [parsed("J1/26", ONE, "EGD111A"), parsed("J2/26", TWO, "EGD222B")]
    readings = {
        "J1/26": reading(notam_id="J1/26", geometry=circle_at(0), activity="alpha contact", value="BOOKING <PHONE_1>"),
        "J2/26": reading(notam_id="J2/26", geometry=circle_at(0), activity="bravo contact", value="BOOKING <PHONE_1>"),
    }
    assessment = run_assess(window=(utc(9), utc(10)), pib=pib_result(*notams), readings=readings)
    assert "<PHONE_1>" in assessment.model_dump_json() and ONE not in assessment.model_dump_json() and TWO not in assessment.model_dump_json()
    text = cli.report(assessment, title="two", lat=53.8, lon=-0.08, now=NOW, phones=cli.PhoneBook.of(notams))
    lines = text.splitlines()
    for notam_id, own, other in (("J1/26", ONE, TWO), ("J2/26", TWO, ONE)):
        mine = [line for line in lines if notam_id in line or ("alpha" if notam_id == "J1/26" else "bravo") in line]
        assert mine and all(own in line and other not in line for line in mine if "PHONE" in line or own in line)
        assert any(own in line for line in mine)
    assert "<PHONE_" not in text
    assert cli.report(assessment, title="two", lat=53.8, lon=-0.08, now=NOW).count("<PHONE_1>") >= 2  # without a book nothing is put back
