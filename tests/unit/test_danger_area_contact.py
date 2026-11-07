"""The contact rule of the danger-area path (G13, CR1): code finds a phone or email in the NOTAM's ORIGINAL item E.

Never the model. A phone is what the redact detector finds, or the <PHONE_n> placeholder it leaves; an email is x@y.z or
the NOTAM form x(AT)y.z. A radio frequency, an AR reference and coordinates are not a contact (E3).

Every constructed text uses Ofcom fiction-range digits (07700 900xxx, 01632 960xxx). The J3857/26 item E is read from the
committed Blue Mill PIB in memory and is never printed or asserted on as text.
"""

import re

import cases
import pytest
from airspace_helpers import activation, assess, blue_mill_tda, only, reading, utc

from flysafelz.airspace import Outcome, find_item_e_contact
from flysafelz.assess import _activations, display
from flysafelz.assess import assess as assess_all
from flysafelz.contracts import Contact, ContactKind
from flysafelz.notam_action import LandingZone
from flysafelz.redact import redact_phones

PHONE, EMAIL, NONE = ContactKind.PHONE, ContactKind.EMAIL, ContactKind.NONE

# (id, constructed item E, the contact code finds)
RULE = [
    # phones, in the shapes the archive shows
    ("sentence-final", "BOOKING PHONE 07700 900123.", PHONE),
    ("run-together", "CALL 07700900456 BEFORE FLIGHT", PHONE),
    ("odd grouping", "TEL 077009 00789", PHONE),
    ("+44(0)", "PHONE +44(0)1632 960001 FOR INFO", PHONE),
    ("+44 spaced", "PHONE +44 1632 960002", PHONE),
    ("0044", "PHONE 0044 1632 960003", PHONE),
    ("landline", "CONTACT 01632 960004 OR RADIO 128.925MHZ", PHONE),
    ("line break", "CONTACT 01632 \n960005", PHONE),
    ("placeholder", "BOOKING PHONE <PHONE_1>.", PHONE),
    ("second placeholder", "FOR INFO <PHONE_2>", PHONE),
    ("phone and email", "OPS(AT)FLYLOGIX.COM OR 07700 900123", PHONE),
    # emails
    ("@ form", "EMAIL OPS@FLYLOGIX.EXAMPLE FOR BOOKING", EMAIL),
    ("(AT) form", "EMAIL OPS(AT)FLYLOGIX.COM FOR BOOKING", EMAIL),
    ("(AT) form, lower case", "email ops(at)flylogix.example", EMAIL),
    ("(AT) form, sentence-final", "BOOK BY EMAIL TO OPS(AT)FLYLOGIX.COM.", EMAIL),
    ("(AT) form, sub-domain", "OPS(AT)MAIL.FLYLOGIX.CO.UK", EMAIL),
    # not a contact
    ("empty", "", NONE),
    ("radio frequency", "CONTACT ON 128.925MHZ", NONE),
    ("two frequencies", "EGD496A. CALL 128.925MHZ OR 119.130 FOR INFO", NONE),
    ("AR reference", "AR-2026-7404/05", NONE),
    ("coordinates", "WI 5NM OF 534732N 0000459W", NONE),
    ("date-time group", "ACTIVE 2610310800 TO 2610311800", NONE),
    ("times and days", "OCT 01 02 03 06 07 08 0900-1700", NONE),
    ("SUP reference", "SEE AIP SUP 108/2026", NONE),
    ("E3 shape", "TDA EGD496A ACTIVE 0800-1800. RADIO 128.925MHZ. AR-2026-7404/05. 534732N 0000459W. SEE AIP SUP 108/2026", NONE),
    ("AT as a word", "ATTN AT FLYLOGIX.COM", NONE),
    ("no domain dot", "OPS@FLYLOGIX", NONE),
    ("(AT) with no domain", "OPS(AT)", NONE),
]


@pytest.mark.parametrize(("text", "expected"), [(t, e) for _id, t, e in RULE], ids=[i for i, _t, _e in RULE])
def test_the_contact_rule(text, expected):
    assert find_item_e_contact(text) is expected


@pytest.mark.parametrize(("text", "expected"), [(t, e) for _id, t, e in RULE if e is not NONE], ids=[i for i, _t, e in RULE if e is not NONE])
def test_redacting_a_contact_first_never_turns_it_into_no_contact(text, expected):
    redacted = redact_phones(text)[0]
    assert find_item_e_contact(redacted) is expected


# --- the rule drives the outcome, through assess_airspace --------------------------------------------------


@pytest.mark.parametrize(("text", "expected"), [(t, e) for _id, t, e in RULE], ids=[i for i, _t, _e in RULE])
def test_the_outcome_follows_what_code_finds_not_what_the_reading_says(text, expected):
    """A reading that always claims a phone: the outcome still follows item E."""
    found = find_item_e_contact(text)
    activations = [activation(read=reading(), contact=found)]
    outcome = only(assess([blue_mill_tda()], activations=activations)).outcome
    assert outcome is (Outcome.CHECK_SUP if expected is NONE else Outcome.CONTACT_FIRST)


# --- J3857/26 from the committed Blue Mill PIB ---------------------------------------------------------------


def _pib_notam(notam_id="J3857/26"):
    _case, _forecast, _airspace, pib, _readings = cases.load(cases.BLUE_MILL)
    return next(n for n in pib.notams if n.id == notam_id)


def test_the_fixture_j3857_item_e_is_a_contact_through_the_raw_text_and_through_redact_phones():
    item_e = _pib_notam().item_e
    assert find_item_e_contact(item_e) is PHONE
    redacted, phones = redact_phones(item_e)
    assert phones  # the detector found its number
    assert find_item_e_contact(redacted) is PHONE


def test_the_fixture_j3857_email_alone_is_a_contact_and_so_is_its_number_alone():
    item_e = _pib_notam().item_e
    no_phone = redact_phones(item_e)[0]
    for number in re.findall(r"<PHONE_\d+>", no_phone):
        no_phone = no_phone.replace(number, "")
    assert find_item_e_contact(no_phone) is EMAIL  # the (AT) form
    no_email = re.sub(r"\S*\(AT\)\S*", "", item_e)
    assert find_item_e_contact(no_email) is PHONE


def test_the_fixture_j3857_frequencies_ar_reference_and_coordinates_alone_are_no_contact():
    item_e = redact_phones(_pib_notam().item_e)[0]
    stripped = re.sub(r"<PHONE_\d+>", "", re.sub(r"\S*\(AT\)\S*", "", item_e))
    assert find_item_e_contact(stripped) is NONE


def _run_with_item_e(item_e, *, reading_contact=None):
    """assess() over Blue Mill with J3857/26's item E replaced (and optionally its reading's contact). Returns the 11:00 segment."""
    case, forecast, airspace, pib, readings = cases.load(cases.BLUE_MILL)
    notams = [n.model_copy(update={"item_e": item_e}) if n.id == "J3857/26" else n for n in pib.notams]
    if reading_contact is not None:
        readings = {**readings, "J3857/26": readings["J3857/26"].model_copy(update={"contact": reading_contact})}
    assessment = assess_all(
        case.lat, case.lon, now=case.now, forecast=forecast, airspace=airspace, pib=pib.model_copy(update={"notams": notams}), readings=readings, window=(case.window.start, case.window.end)
    )
    return next(s for h in assessment.hours for s in h.segments if s.start == utc(11))


SAFE_SUP_CONTACT = Contact(kind=ContactKind.NONE, value=None)


def test_j3857_gives_contact_first_with_the_sup_through_the_raw_item_e():
    item_e = _pib_notam().item_e
    segment = _run_with_item_e(item_e)
    danger = [c for c in segment.conditions if "EGD496A" in c]
    assert len(danger) == 1 and "fly only if they agree" in danger[0] and danger[0].endswith("; details in AIP SUP 108/2026")
    assert display(segment).startswith("PILOT DISCRETION") and display(segment).endswith("WITH CONDITIONS")


def test_j3857_gives_contact_first_through_redact_phones_of_the_item_e():
    segment = _run_with_item_e(redact_phones(_pib_notam().item_e)[0])
    danger = [c for c in segment.conditions if "EGD496A" in c]
    assert len(danger) == 1 and "fly only if they agree" in danger[0] and "read the SUP" not in danger[0]


def test_j3857_with_its_email_removed_and_its_number_a_placeholder_is_still_contact_first_never_check_sup():
    redacted = redact_phones(_pib_notam().item_e)[0]
    only_placeholder = re.sub(r"\S*\(AT\)\S*", "", redacted)
    assert "<PHONE_1>" in only_placeholder
    segment = _run_with_item_e(only_placeholder)
    danger = [c for c in segment.conditions if "EGD496A" in c]
    assert len(danger) == 1 and "fly only if they agree" in danger[0] and "read the SUP" not in danger[0]
    assert danger[0].endswith("; details in AIP SUP 108/2026")


def test_j3857_with_a_reading_that_gives_no_contact_the_code_found_contact_still_decides():
    segment = _run_with_item_e(_pib_notam().item_e, reading_contact=SAFE_SUP_CONTACT)
    danger = [c for c in segment.conditions if "EGD496A" in c]
    assert len(danger) == 1 and "phone given in NOTAM J3857/26 and fly only if they agree" in danger[0]


def test_j3857_with_no_contact_in_item_e_is_check_sup_and_the_segment_is_with_conditions():
    item_e = redact_phones(_pib_notam().item_e)[0]
    item_e = re.sub(r"<PHONE_\d+>", "", re.sub(r"\S*\(AT\)\S*", "", item_e))
    segment = _run_with_item_e(item_e)
    danger = [c for c in segment.conditions if "EGD496A" in c]
    assert danger == ["read the SUP / AIP ENR 5.1 entry for EGD496A before flight; details in AIP SUP 108/2026"]
    assert display(segment).startswith("PILOT DISCRETION") and display(segment).endswith("WITH CONDITIONS")


def test_an_activation_holds_the_contact_kind_and_the_sup_and_no_number_or_raw_item_e():
    case, _forecast, _airspace, _pib, readings = cases.load(cases.BLUE_MILL)
    notam = _pib_notam()
    numbers = [re.sub(r"\D", "", n) for n in redact_phones(notam.item_e)[1].values()]
    assert numbers
    lz = LandingZone(lat=case.lat, lon=case.lon, elevation_ft=30.0)
    found = _activations([notam], readings, lz, utc(8), utc(18))
    assert found and all(a.item_e_contact is PHONE and a.sup_references == ["AIP SUP 108/2026"] for a in found)
    # the reading is the fixture's own (its contact value is a hand-written fixture field); everything the code added is checked
    dumped = re.sub(r"\D", "", " ".join(a.model_dump_json(exclude={"reading"}) for a in found))
    assert all(a.e_text == " ".join(redact_phones(notam.item_e)[0].split()) for a in found)
    assert not any(digits in dumped for digits in numbers)


def test_the_unread_assessment_holds_no_number_from_item_e():
    """With no reading, the reason carries the redacted E text: no number from the original item E reaches the JSON."""
    numbers = [re.sub(r"\D", "", n) for n in redact_phones(_pib_notam().item_e)[1].values()]
    assert numbers
    flat = re.sub(r"\D", "", cases.run(cases.BLUE_MILL, readings=False).model_dump_json())
    assert not any(digits in flat for digits in numbers)
