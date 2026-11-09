"""G5: check_reading, one table per check (a) to (f).

check_reading is the real function, called with explicit parsed NOTAMs, readings and redaction maps. Fixtures come from
tests/factories.py. The phone numbers in the E text are Ofcom drama-range numbers.
"""

import copy
import logging
import math

import pytest
from factories import agl, amsl, circle_at, exemption, fl, notam, reading
from pydantic import ValidationError

from flysafelz.contracts import (
    AppliesTo,
    Category,
    Contact,
    ContactKind,
    Datum,
    Exemption,
    ExemptionCovers,
    Limit,
    LimitUnit,
    ProhibitiveReason,
)
from flysafelz.reader.checks import check_reading, schema_failures
from flysafelz.reader.contract import ReaderOutput
from flysafelz.redact import redact_phones

E_TEXT = "UAS OPR NEAR GARTON. PARAGLIDERS EXEMPT. CALL 07700 900123 FIRST. EGD 496A ACTIVE. NO AIRCRAFT BELOW 500FT."
PARSED = notam(item_e=E_TEXT)
_REDACTED, MAP = redact_phones(E_TEXT)
REAL = "07700 900123"
SFC = Limit(value=0, unit=LimitUnit.FT, datum=Datum.SFC)


def check(**fields):
    """check_reading over E_TEXT's NOTAM, with a clean reading changed by `fields` (the factory's keywords)."""
    clean = {"contact": ContactKind.PHONE, "value": "<PHONE_1>"}
    return check_reading(PARSED, reading(**{**clean, **fields}), MAP)


def broken(**changes):
    """A reading around Pydantic's validation (model_copy and model_construct do not validate), as a bug upstream might hand over."""
    return reading(contact=ContactKind.PHONE, value="<PHONE_1>").model_copy(update=changes)


def test_the_fixture_is_what_the_tables_assume():
    assert MAP == {"<PHONE_1>": REAL}
    assert "CALL <PHONE_1> FIRST" in _REDACTED
    assert check() == []


# --- clean and pure ------------------------------------------------------------------------------------------------


def test_a_clean_reading_gives_an_empty_list():
    good = reading(exemptions=[exemption("PARAGLIDERS EXEMPT")], designators=["EGD496A"], value="<PHONE_1>", ceiling=amsl(800))
    assert check_reading(PARSED, good, MAP) == []


def test_check_reading_changes_nothing_and_gives_the_same_list_twice():
    """Every check fails here; the reading and the NOTAM come back byte for byte as they went in."""
    bad = reading(
        geometry=circle_at(0.4),
        schedule="0800-1800",
        floor=amsl(900),
        ceiling=amsl(300),
        value="<PHONE_3>",
        exemptions=[exemption("EXCEPT FOR PARAGLIDING")],
        designators=["EGD496"],
    )
    before, parsed_before = copy.deepcopy(bad), copy.deepcopy(PARSED)
    first = check_reading(PARSED, bad, MAP)
    assert len(first) == 4  # b, c, d, e
    assert bad == before and bad.model_dump() == before.model_dump()  # limits, id, area and schedule untouched
    assert PARSED == parsed_before
    assert check_reading(PARSED, bad, MAP) == first


def test_the_sentences_come_in_the_order_b_c_d_e_prohibitive_quote_f():
    bad = reading(floor=amsl(900), ceiling=amsl(300), value="<PHONE_3>", exemptions=[exemption("EXCEPT FOR PARAGLIDING")], designators=["EGD496"]).model_copy(
        update={"applies_to": "PARAMOTORS", "prohibitive_reason": ProhibitiveReason.LIVE_FIRING, "prohibitive_quote": "LIVE FIRING IN PROGRESS"}
    )
    sentences = check_reading(PARSED, bad, MAP)
    openings = ["your floor", "your contact value", "your activates_designators entry", "your exemption quote", "your prohibitive_quote", "your applies_to"]
    assert [s.startswith(o) for s, o in zip(sentences, openings, strict=True)] == [True] * len(openings)


# --- (a) area ------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("area", [None, circle_at(0), circle_at(0.4, radius_nm=0.3), circle_at(5.0, radius_nm=3.0)], ids=["q-circle", "over the LZ", "near", "far outside the Q-circle"])
def test_a_the_area_is_codes_so_check_reading_never_fails_on_it(area):
    """Code's build_area swaps an area too far outside the Q-circle (test_reader_repair.py); no area ever fails a check."""
    assert check(geometry=area) == []


# --- (b) floor <= ceiling ------------------------------------------------------------------------------------------

OK, FAIL, SKIP = "ok", "fail", "skipped (not comparable)"
HEIGHT_ROWS = [
    # floor SFC (0 on any datum)
    pytest.param(SFC, SFC, OK, id="SFC-SFC"),
    pytest.param(SFC, agl(500), OK, id="SFC-AGL"),
    pytest.param(SFC, amsl(500), OK, id="SFC-AMSL"),
    pytest.param(SFC, fl(5), OK, id="SFC-FL"),
    # floor AGL
    pytest.param(agl(500), SFC, FAIL, id="AGL-SFC"),
    pytest.param(agl(500), agl(800), OK, id="AGL-AGL-above"),
    pytest.param(agl(500), agl(500), OK, id="AGL-AGL-equal"),
    pytest.param(agl(500), agl(400), FAIL, id="AGL-AGL-below"),
    pytest.param(agl(800), amsl(100), SKIP, id="AGL-AMSL-inverted"),
    pytest.param(agl(100), amsl(800), SKIP, id="AGL-AMSL-ordered"),
    pytest.param(agl(800), fl(1), SKIP, id="AGL-FL-inverted"),
    # floor AMSL
    pytest.param(amsl(800), SFC, FAIL, id="AMSL-SFC"),
    pytest.param(amsl(800), amsl(1000), OK, id="AMSL-AMSL-above"),
    pytest.param(amsl(800), amsl(800), OK, id="AMSL-AMSL-equal"),
    pytest.param(amsl(800), amsl(500), FAIL, id="AMSL-AMSL-below"),
    pytest.param(amsl(800), agl(2000), SKIP, id="AMSL-AGL"),
    pytest.param(amsl(800), fl(10), OK, id="AMSL-FL-above"),
    pytest.param(amsl(800), fl(8), OK, id="AMSL-FL-equal"),
    pytest.param(amsl(800), fl(5), FAIL, id="AMSL-FL-below"),
    # floor FL (FL x 100 ft AMSL)
    pytest.param(fl(10), SFC, FAIL, id="FL-SFC"),
    pytest.param(fl(10), amsl(1000), OK, id="FL-AMSL-equal"),
    pytest.param(fl(10), amsl(900), FAIL, id="FL-AMSL-below"),
    pytest.param(fl(10), agl(5000), SKIP, id="FL-AGL"),
    pytest.param(fl(10), fl(65), OK, id="FL-FL-above"),
    pytest.param(fl(65), fl(10), FAIL, id="FL-FL-below"),
    # a flight level written with a ft unit is still compared as the unit says
    pytest.param(Limit(value=5000, unit=LimitUnit.FT, datum=Datum.AMSL), fl(65), OK, id="ft-AMSL-FL"),
]


@pytest.mark.parametrize(("floor", "ceiling", "outcome"), HEIGHT_ROWS)
def test_b_floor_and_ceiling_when_comparable(floor, ceiling, outcome, caplog):
    caplog.set_level(logging.DEBUG)
    sentences = check_reading(PARSED, reading(floor=floor, ceiling=ceiling, value="<PHONE_1>"), MAP)
    assert len(sentences) == (1 if outcome == FAIL else 0)
    assert any("cannot be compared" in r.getMessage() for r in caplog.records) == (outcome == SKIP)  # the skip is logged
    if outcome == FAIL:
        assert "floor" in sentences[0] and "above your ceiling" in sentences[0]


def test_b_the_sentence_names_both_heights():
    (sentence,) = check(floor=amsl(900), ceiling=amsl(300))
    assert "900 ft AMSL" in sentence and "300 ft AMSL" in sentence
    (sentence,) = check(floor=fl(65), ceiling=fl(10))
    assert "FL065" in sentence and "FL010" in sentence
    (sentence,) = check(floor=agl(500), ceiling=SFC)
    assert "500 ft AGL" in sentence and "SFC" in sentence


@pytest.mark.parametrize("changes", [{"floor": None}, {"ceiling": None}, {"floor": None, "ceiling": None}], ids=["no floor", "no ceiling", "neither"])
def test_b_a_missing_floor_or_ceiling_is_skipped_not_failed(changes, caplog):
    """E states none (G4). reading() cannot make a None floor or ceiling, so the change is applied to the built reading."""
    bad_order = reading(floor=amsl(900), ceiling=amsl(300), value="<PHONE_1>").model_copy(update=changes)
    caplog.set_level(logging.DEBUG)
    assert check_reading(PARSED, bad_order, MAP) == []
    assert not [r for r in caplog.records if "cannot be compared" in r.getMessage()]


NOT_FINITE = [pytest.param(math.nan, id="NaN"), pytest.param(math.inf, id="Infinity"), pytest.param(-math.inf, id="-Infinity")]


@pytest.mark.parametrize("bad", NOT_FINITE)
@pytest.mark.parametrize("unit", [LimitUnit.FT, LimitUnit.FL], ids=["ft", "FL"])
@pytest.mark.parametrize("which", ["floor", "ceiling"])
def test_b_a_height_that_is_not_a_finite_number_fails_and_is_named(which, unit, bad):
    """Every comparison with NaN is False, so without this row a NaN would pass (b) and break notam_action() later."""
    height = Limit(value=bad, unit=unit, datum=Datum.AMSL)
    other = {"ceiling": amsl(800), "floor": amsl(100)}[which]
    changes = {which: height, ("floor" if which == "ceiling" else "ceiling"): other}
    (sentence,) = check_reading(PARSED, reading(value="<PHONE_1>").model_copy(update=changes), MAP)
    assert sentence.startswith(f"your {which} (") and "is not a finite number" in sentence and str(bad).lstrip("-")[:3].lower() in sentence.lower()


@pytest.mark.parametrize("bad", NOT_FINITE)
def test_b_a_height_that_is_not_finite_fails_whatever_the_other_one_is(bad):
    nan = Limit(value=bad, unit=LimitUnit.FT, datum=Datum.AMSL)
    for other in (None, SFC, agl(500), fl(5)):
        assert len(check_reading(PARSED, reading(value="<PHONE_1>").model_copy(update={"floor": nan, "ceiling": other}), MAP)) == 1
        assert len(check_reading(PARSED, reading(value="<PHONE_1>").model_copy(update={"floor": other, "ceiling": nan}), MAP)) == 1


@pytest.mark.parametrize("bad", NOT_FINITE)
@pytest.mark.parametrize("which", ["floor", "ceiling"])
def test_b_an_sfc_height_that_is_not_finite_quotes_the_value_not_sfc_alone(which, bad):
    """An SFC height is shown as plain 'SFC' elsewhere; for a NaN or Infinity the sentence has to give the offending value."""
    height = Limit(value=bad, unit=LimitUnit.FT, datum=Datum.SFC)
    changes = {which: height, ("floor" if which == "ceiling" else "ceiling"): amsl(800)}
    (sentence,) = check_reading(PARSED, reading(value="<PHONE_1>").model_copy(update=changes), MAP)
    shown = f"{bad:g}"
    assert sentence.startswith(f"your {which} ({shown} ft SFC) is not a finite number")


def test_b_both_heights_not_finite_gives_one_sentence_each():
    both = {"floor": Limit(value=math.nan, unit=LimitUnit.FT, datum=Datum.AMSL), "ceiling": Limit(value=math.inf, unit=LimitUnit.FT, datum=Datum.AMSL)}
    sentences = check_reading(PARSED, reading(value="<PHONE_1>").model_copy(update=both), MAP)
    assert [s.split(" ")[1] for s in sentences] == ["floor", "ceiling"]


# --- (c) placeholders ----------------------------------------------------------------------------------------------


def test_c_a_placeholder_in_the_map_passes_anywhere_in_the_reading():
    assert check(activity="call <PHONE_1> first", exemptions=[exemption("CALL <PHONE_1> FIRST")]) == []


@pytest.mark.parametrize(
    ("field", "changes"),
    [
        pytest.param("contact value", {"value": "<PHONE_3>"}, id="contact value"),
        pytest.param("exemption quote", {"exemptions": [exemption("CALL <PHONE_3> FIRST")]}, id="exemption quote"),
        pytest.param("activity", {"activity": "call <PHONE_3>"}, id="activity"),
        pytest.param("activates_designators entry", {"designators": ["<PHONE_3>"]}, id="designator"),
    ],
)
def test_c_a_placeholder_the_map_does_not_hold_fails_and_is_named(field, changes):
    sentences = [s for s in check(**changes) if "<PHONE_3>" in s and s.startswith(f"your {field} mentions")]
    assert len(sentences) == 1
    assert "<PHONE_1>" in sentences[0]  # it says which placeholders the NOTAM text has
    assert REAL not in sentences[0] and "900123" not in sentences[0]


def test_c_a_placeholder_repeated_in_one_field_is_one_sentence():
    sentences = [s for s in check(activity="<PHONE_3> or <PHONE_3>") if s.startswith("your activity")]
    assert len(sentences) == 1


def test_c_a_text_with_no_phone_numbers_says_so():
    other = notam(item_e="CRANE ERECTED.")
    (sentence,) = check_reading(other, reading(value="<PHONE_1>", ceiling=amsl(800)), {})
    assert "contact value mentions <PHONE_1>" in sentence and "no phone numbers" in sentence


@pytest.mark.parametrize(
    ("field", "changes"),
    [
        pytest.param("contact value", {"value": REAL}, id="contact value"),
        pytest.param("contact value", {"value": "+44 7700 900123"}, id="contact value, international"),
        pytest.param("activity", {"activity": f"call {REAL}"}, id="activity"),
        pytest.param("activity", {"activity": "call 07700900123"}, id="activity, no spaces"),
        pytest.param("exemption quote", {"exemptions": [exemption(f"CALL {REAL} FIRST")]}, id="exemption quote"),
        pytest.param("activates_designators entry", {"designators": [REAL]}, id="designator"),
    ],
)
def test_c_a_phone_number_written_out_fails_and_the_sentence_holds_no_number(field, changes):
    sentences = [s for s in check(**changes) if s.startswith(f"your {field} has a phone number written out")]
    assert len(sentences) == 1
    assert "<PHONE_1>" in sentences[0]  # it says which placeholder to use
    assert "900123" not in sentences[0] and "7700" not in sentences[0]


def test_c_a_phone_number_written_out_says_so_when_the_text_has_no_phone_numbers():
    other = notam(item_e="CRANE ERECTED.")
    (sentence,) = check_reading(other, reading(value=REAL, ceiling=amsl(800)), {})
    assert sentence.startswith("your contact value has a phone number written out") and "no phone numbers" in sentence
    assert "900123" not in sentence


def test_c_ordinary_numbers_in_the_reading_are_not_phone_numbers():
    assert check(activity="UAS OPS 0800-1800 MAX 700FT AGL", exemptions=[exemption("NO AIRCRAFT BELOW 500FT")]) == []


def test_c_a_contact_with_no_value_is_fine():
    assert check(contact=ContactKind.NONE, value=None) == []


# --- (d) designators -----------------------------------------------------------------------------------------------


@pytest.mark.parametrize("entry", ["D496A", "EGD496A", "EGD 496A", "egd496a", " D 496A "])
def test_d_an_entry_that_normalises_to_a_designator_in_the_text_passes(entry):
    assert check(designators=[entry]) == []


def test_d_no_entries_passes_even_though_the_text_names_a_designator():
    assert check(designators=[]) == []


def test_d_exact_match_only_egd496_is_not_egd496a():
    parsed = notam(item_e="DANGER AREA EGD496A ACTIVE.")
    (sentence,) = check_reading(parsed, reading(designators=["EGD496"], value=None, contact=ContactKind.NONE), {})
    assert "'EGD496'" in sentence and "does not appear in the NOTAM text" in sentence


def test_d_an_entry_the_text_does_not_name_fails_with_its_normal_form():
    (sentence,) = check(designators=["D323"])
    assert sentence.startswith("your activates_designators entry 'D323' (EGD323) does not appear")


def test_d_an_entry_that_is_not_a_designator_fails():
    (sentence,) = check(designators=["DANGER AREA"])
    assert "'DANGER AREA' is not an airspace designator" in sentence


def test_d_each_bad_entry_gets_its_own_sentence_in_order():
    sentences = check(designators=["EGD496A", "NONSENSE", "EGR999"])
    assert [("NONSENSE" in s, "EGR999" in s) for s in sentences] == [(True, False), (False, True)]


# --- (e) exemption quotes ------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "quote",
    [
        "PARAGLIDERS EXEMPT",
        "paragliders exempt",
        "Paragliders   Exempt",
        "PARAGLIDERS\nEXEMPT",
        "PARAGLIDERS\u00a0EXEMPT",  # non-breaking space
        "  PARAGLIDERS EXEMPT.  ",
        "CALL <PHONE_1> FIRST",  # the redacted text is the reference, placeholder and all
        "UAS OPR NEAR GARTON. PARAGLIDERS EXEMPT. CALL <PHONE_1> FIRST. EGD 496A ACTIVE. NO AIRCRAFT BELOW 500FT.",
    ],
    ids=["verbatim", "lower case", "spaces", "newline", "nbsp", "padded", "placeholder", "whole text"],
)
def test_e_a_quote_in_the_redacted_text_passes(quote):
    assert check(exemptions=[exemption(quote)]) == []


def test_e_the_text_with_odd_whitespace_matches_too():
    wrapped = notam(item_e="PARAGLIDERS\u00a0\u00a0EXEMPT\r\nFROM THIS.")
    assert check_reading(wrapped, reading(exemptions=[exemption("paragliders exempt from this")], value=None, contact=ContactKind.NONE), {}) == []


@pytest.mark.parametrize("quote", ["EXCEPT FOR PARAGLIDING", "PARAGLIDERS ARE EXEMPT", "THIS NOTAM DOES NOT APPLY", "PARAGLIDERS EXEMPT FROM THIS"])
def test_e_a_quote_not_in_the_text_fails_with_the_word_for_word_instruction(quote):
    (sentence,) = check(exemptions=[exemption(quote)])
    assert sentence == f"your exemption quote '{quote}' does not appear in the NOTAM text; quote it word for word or leave exemptions empty"


@pytest.mark.parametrize("quote", ["", "   ", "\n\t", "\u00a0"], ids=["empty", "spaces", "newline and tab", "nbsp"])
def test_e_an_empty_quote_is_a_failure_not_a_match(quote):
    (sentence,) = check(exemptions=[exemption(quote)])
    assert "is empty" in sentence and "leave exemptions empty" in sentence


def test_e_a_quote_with_the_real_number_in_it_does_not_match_the_redacted_text_and_never_shows_the_number():
    # (c) also fails it, for the number written out; this test is about the (e) sentence.
    (sentence,) = [s for s in check(exemptions=[exemption(f"CALL {REAL} FIRST")]) if "does not appear" in s]
    assert "does not appear in the NOTAM text" in sentence
    assert "900123" not in sentence and "<PHONE_1>" in sentence  # the number it wrote is shown as a placeholder


def test_e_every_covers_value_is_checked():
    for covers in ExemptionCovers:
        assert len(check(exemptions=[exemption("NOT THERE", covers)])) == 1, covers


def test_e_each_bad_quote_gets_its_own_sentence_and_good_ones_none():
    sentences = check(exemptions=[exemption("PARAGLIDERS EXEMPT"), exemption("ALPHA"), exemption("BRAVO")])
    assert len(sentences) == 2 and "'ALPHA'" in sentences[0] and "'BRAVO'" in sentences[1]


def test_an_empty_e_text_with_no_exemptions_and_no_designators_passes():
    empty = notam(item_e="")
    assert check_reading(empty, reading(value=None, contact=ContactKind.NONE, exemptions=[], designators=[]), {}) == []


def test_e_with_an_empty_e_text_any_quote_fails():
    empty = notam(item_e="")
    assert len(check_reading(empty, reading(value=None, contact=ContactKind.NONE, exemptions=[exemption("X")]), {})) == 1


# --- (e2) prohibitive quote ----------------------------------------------------------------------------------------

REASONS = list(ProhibitiveReason)
GIVING_A_REASON = [r for r in ProhibitiveReason if r is not ProhibitiveReason.NONE]
QUOTES = {
    "none": (None, False),
    "empty": ("", False),
    "whitespace": ("  \n\t\u00a0", False),
    "in the text": ("PARAGLIDERS EXEMPT", True),
    "in the text, other case and whitespace": ("  uas   opr\nnear\u00a0GARTON.  ", True),
    "not in the text": ("ENTRY IS PROHIBITED", False),
    "a real number": (f"CALL {REAL} FIRST", False),
    "placeholder in the map": ("CALL <PHONE_1> FIRST", True),
    "placeholder not in the map": ("CALL <PHONE_2> FIRST", False),
}


def quote_sentences(sentences):
    return [s for s in sentences if s.startswith("your prohibitive_quote")]


@pytest.mark.parametrize("reason", REASONS, ids=lambda r: r.value)
@pytest.mark.parametrize("name", QUOTES)
def test_g9_the_quote_check_over_every_reason_and_quote(reason, name):
    quote, in_text = QUOTES[name]
    sentences = check_reading(PARSED, broken(prohibitive_reason=reason, prohibitive_quote=quote), MAP)
    failed = quote_sentences(sentences)
    if reason is ProhibitiveReason.NONE or in_text:
        assert failed == []
    else:
        assert len(failed) == 1
        assert "prohibitive_quote" in failed[0] and "or answer prohibitive_reason none" in failed[0] and "word for word" in failed[0]
    for sentence in sentences:
        assert "900123" not in sentence and "07700" not in sentence  # no sentence ever shows the number


def test_g9_a_clean_reading_with_a_reason_and_its_quote_passes_every_check():
    assert check_reading(PARSED, broken(prohibitive_reason=ProhibitiveReason.ENTRY_PROHIBITED, prohibitive_quote="uas opr near garton"), MAP) == []


def test_g9_an_empty_quote_says_so_and_names_the_reason():
    (sentence,) = broken_sentences(prohibitive_reason=ProhibitiveReason.LIVE_FIRING, prohibitive_quote="   ")
    assert (
        sentence
        == "your prohibitive_quote is empty but your prohibitive_reason is 'live firing'; quote the words of the NOTAM text the reason rests on, word for word, or answer prohibitive_reason none"
    )
    assert broken_sentences(prohibitive_reason=ProhibitiveReason.BYELAWS, prohibitive_quote=None) == [sentence.replace("live firing", "byelaws")]


def test_g9_a_quote_not_in_the_text_is_shown_back_trimmed_with_the_instruction():
    (sentence,) = broken_sentences(prohibitive_reason=ProhibitiveReason.BYELAWS, prohibitive_quote="  BYELAWS APPLY  ")
    assert sentence == "your prohibitive_quote 'BYELAWS APPLY' does not appear in the NOTAM text; quote the words the reason rests on, word for word, or answer prohibitive_reason none"


def broken_sentences(**changes):
    return quote_sentences(check_reading(PARSED, broken(**changes), MAP))


def test_g9_a_real_number_in_the_quote_fails_check_c_and_does_not_match_and_is_never_shown():
    sentences = check_reading(PARSED, broken(prohibitive_reason=ProhibitiveReason.LIVE_FIRING, prohibitive_quote=f"CALL {REAL} FIRST"), MAP)
    assert [s.split(" ")[:3] for s in sentences] == [["your", "prohibitive", "quote"], ["your", "prohibitive_quote", "'CALL"]]
    assert "has a phone number written out" in sentences[0]
    assert "<PHONE_1>" in sentences[1]
    assert all("900123" not in s and "07700" not in s for s in sentences)


def test_g9_check_c_also_covers_a_quote_with_a_placeholder_that_is_not_in_the_map():
    sentences = check_reading(PARSED, broken(prohibitive_reason=ProhibitiveReason.LIVE_FIRING, prohibitive_quote="CALL <PHONE_2> FIRST"), MAP)
    assert sentences[0].startswith("your prohibitive quote mentions <PHONE_2>, which is not in the NOTAM text")
    assert len(quote_sentences(sentences)) == 1


def test_g9_a_quote_across_two_sentences_passes_when_it_is_contiguous_and_fails_with_words_from_two_places():
    contiguous = "PARAGLIDERS EXEMPT. CALL <PHONE_1> FIRST"
    assert broken_sentences(prohibitive_reason=ProhibitiveReason.ENTRY_PROHIBITED, prohibitive_quote=contiguous) == []
    assert len(broken_sentences(prohibitive_reason=ProhibitiveReason.ENTRY_PROHIBITED, prohibitive_quote="UAS OPR NEAR GARTON. EGD 496A ACTIVE")) == 1
    assert len(broken_sentences(prohibitive_reason=ProhibitiveReason.ENTRY_PROHIBITED, prohibitive_quote="GARTON EGD 496A")) == 1


def test_g9_with_an_empty_e_text_any_reason_other_than_none_fails():
    empty = notam(item_e="")
    for quote in ("", "X", "   "):
        for reason in GIVING_A_REASON:
            changed = reading(value=None, contact=ContactKind.NONE).model_copy(update={"prohibitive_reason": reason, "prohibitive_quote": quote})
            assert len(quote_sentences(check_reading(empty, changed, {}))) == 1, (reason, quote)
    none = reading(value=None, contact=ContactKind.NONE).model_copy(update={"prohibitive_quote": None})
    assert check_reading(empty, none, {}) == []


def test_g9_unicode_and_quote_characters_pass_through_unchanged():
    text = "ENTRY \u2013 \u201cPILOTS ARE TO AVOID\u201d \u2019S \u00c9 \u4e2d THE AREA."
    quote = "\u201cpilots are to avoid\u201d \u2019s \u00e9 \u4e2d"  # lower case, \u00e9 casefolds from \u00c9
    unicode_notam = notam(item_e=text)
    changed = reading(value=None, contact=ContactKind.NONE).model_copy(update={"prohibitive_reason": ProhibitiveReason.ENTRY_PROHIBITED, "prohibitive_quote": quote})
    before = copy.deepcopy(changed)
    assert check_reading(unicode_notam, changed, {}) == []
    assert changed == before and changed.prohibitive_quote == quote and unicode_notam.item_e == text
    wrong_dash = changed.model_copy(update={"prohibitive_quote": "\u2014 \u201cpilots are to avoid\u201d"})  # an em dash is not the text's en dash
    assert len(quote_sentences(check_reading(unicode_notam, wrong_dash, {}))) == 1


def test_e2_no_check_forces_a_reason_when_the_text_says_pilots_are_to_avoid():
    avoid = notam(item_e="PILOTS ARE TO AVOID THE AREA")
    none = reading(value=None, contact=ContactKind.NONE).model_copy(update={"prohibitive_reason": ProhibitiveReason.NONE, "prohibitive_quote": None})
    assert check_reading(avoid, none, {}) == []


# --- (f) enums and types -------------------------------------------------------------------------------------------

ALLOWED = {
    "category": [c.value for c in Category],
    "applies_to": [a.value for a in AppliesTo],
    "prohibitive_reason": [p.value for p in ProhibitiveReason],
    "contact.kind": [k.value for k in ContactKind],
    "floor.unit": [u.value for u in LimitUnit],
    "floor.datum": [d.value for d in Datum],
    "exemptions[0].covers": [c.value for c in ExemptionCovers],
}
BAD = {
    "category": {"category": "restricted"},
    "applies_to": {"applies_to": "PARAMOTORS"},
    "prohibitive_reason": {"prohibitive_reason": "danger"},
    "contact.kind": {"contact": Contact.model_construct(kind="fax", value="<PHONE_1>")},
    "floor.unit": {"floor": Limit.model_construct(value=0, unit="m", datum=Datum.SFC)},
    "floor.datum": {"floor": Limit.model_construct(value=0, unit=LimitUnit.FT, datum="MSL")},
    "exemptions[0].covers": {"exemptions": [Exemption.model_construct(quote="PARAGLIDERS EXEMPT", covers="EVERYONE")]},
}


@pytest.mark.parametrize("path", sorted(BAD))
def test_f_a_bad_enum_fails_names_the_field_and_lists_the_allowed_values(path):
    sentences = check_reading(PARSED, broken(**BAD[path]), MAP)
    assert len(sentences) == 1, sentences
    assert sentences[0].startswith(f"your {path} is '")
    assert all(f"'{value}'" in sentences[0] for value in ALLOWED[path])


def test_f_a_valid_reading_passes_every_enum():
    assert check_reading(PARSED, broken(), MAP) == []


def reply_error(**overrides) -> ValidationError:
    fields = {
        "category": "cooperative",
        "activity": "drone operations",
        "floor": None,
        "ceiling": None,
        "applies_to": "ALL_AIRCRAFT",
        "exemptions": [],
        "contact": {"kind": "none", "value": None},
        "activates_designators": [],
        "prohibitive_reason": "none",
        "prohibitive_quote": None,
    }
    fields.update(overrides)
    fields = {k: v for k, v in fields.items() if v != "<missing>"}
    with pytest.raises(ValidationError) as raised:
        ReaderOutput.model_validate(fields)
    return raised.value


def test_f_schema_failures_for_a_bad_enum_lists_the_allowed_values():
    (sentence,) = schema_failures(reply_error(applies_to="PARAMOTORS"))
    assert sentence == "your applies_to is 'PARAMOTORS', which is not allowed; use 'ALL_AIRCRAFT', 'UAS_ONLY', 'AERODROME_USERS' or 'NAMED_CLASSES'"


def test_f_schema_failures_for_a_missing_key_a_forbidden_key_and_a_wrong_type():
    assert schema_failures(reply_error(ceiling="<missing>")) == ["your answer has no ceiling; give every field (null where the NOTAM text states none)"]
    assert schema_failures(reply_error(geometry={"kind": "circle"})) == ["your answer has a field geometry that is not on the form; leave it out"]
    (sentence,) = schema_failures(reply_error(activity=7))
    assert sentence.startswith("your activity is '7', which is not valid:")


def test_f_schema_failures_name_a_nested_field_by_its_path():
    (sentence,) = schema_failures(reply_error(exemptions=[{"quote": "X"}]))
    assert sentence.startswith("your answer has no exemptions[0].covers")
    (sentence,) = schema_failures(reply_error(floor={"value": 0, "unit": "m", "datum": "SFC"}))
    assert sentence.startswith("your floor.unit is 'm'")


@pytest.mark.parametrize(("raw", "shown"), [([], "a list"), (None, "null"), ("text", "'text'"), (7, "'7'")])
def test_f_schema_failures_show_a_rejected_value_by_its_type_unless_it_is_a_scalar(raw, shown):
    with pytest.raises(ValidationError) as raised:
        ReaderOutput.model_validate(raw)
    (sentence,) = schema_failures(raised.value)
    assert sentence.startswith(f"your answer is {shown}, which is not valid")


@pytest.mark.parametrize("key", [REAL, "07700900123", "+44 7700 900123"], ids=["spaced", "bare", "international"])
def test_f_a_real_number_written_as_an_extra_key_is_shown_as_a_placeholder_at_the_top_and_nested(key):
    top = schema_failures(reply_error(**{key: "x"}))
    nested = schema_failures(reply_error(contact={"kind": "none", "value": None, key: "x"}))
    for (sentence,) in (top, nested):
        assert "900123" not in sentence and "<PHONE_1>" in sentence and "not on the form" in sentence
    assert nested[0].startswith("your answer has a field contact.<PHONE_1>")


def test_f_a_real_number_in_a_rejected_value_is_shown_as_a_placeholder():
    (sentence,) = schema_failures(reply_error(applies_to=f"CALL {REAL}"))
    assert "900123" not in sentence and "<PHONE_1>" in sentence


def test_f_every_problem_gets_a_sentence_in_the_form_order():
    sentences = schema_failures(reply_error(category="x", ceiling="<missing>", prohibitive_reason="y"))
    assert [s.split(" ")[1] for s in sentences] == ["category", "answer", "prohibitive_reason"]


@pytest.mark.parametrize("value", ["ordnance", "demolition", "explosives", "rockets"])
def test_f3_a_new_reason_value_is_a_schema_failure_listing_the_four_allowed_values(value):
    (sentence,) = schema_failures(reply_error(prohibitive_reason=value))
    assert sentence == f"your prohibitive_reason is '{value}', which is not allowed; use 'byelaws', 'entry prohibited', 'live firing' or 'none'"


def test_f3_a_reply_without_the_quote_key_is_a_missing_field_sentence():
    assert schema_failures(reply_error(prohibitive_quote="<missing>")) == ["your answer has no prohibitive_quote; give every field (null where the NOTAM text states none)"]
