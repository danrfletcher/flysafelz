"""designators.py: normalising and finding EG<D|R|P><nnn><X> (E18)."""

import pytest

from flysafelz.designators import find_designators, normalise_designator

# Item E of J3857/26 and J3864/26 (31 Oct PIB archive): the same text, real line breaks included.
J3857_E = (
    "TEMPO DANGER AREA (TDA) EGD496A ACTIVATED. AIP SUP 108/2026\n"
    "REFERS. BEYOND VISUAL LINE OF SIGHT UAS OPS CONTAINED WHOLLY WI\n"
    "THE TDA. A SPECIAL USE AIRSPACE ACTIVITY INFORMATION SERVICE\n"
    "(SUAAIS) WILL BE AVBL FM ANGLIA RADAR 128.925MHZ OR 125.275MHZ OR\n"
    "LONDON INFO 125.475MHZ. FOR INFO AND BOOKING CTC\n"
    "OPS(AT)FLYLOGIX.COM OR 07784 546062. SURVEYING CYGNUS.\n"
    "AR-2026-7404/05."
)
J3864_E = J3857_E.replace("AR-2026-7404/05.", "AR-2026-7404/09.")
H7171_E = "UAS OPR WI 0.5NM RADIUS OF 534732N 0000459W (GARTON). MAX HGT \n700FT AGL. FOR INFO 07784 546062. AR-2026-10264/01."


@pytest.mark.parametrize("text", ["D323A", "EGD 323A", "EG D323A", "egd323a", "EGD323A", "d323a", "Egd 323a", "EG D 323A", " EGD323A ", "D 323A"])
def test_every_written_form_normalises_to_the_same_designator(text):
    assert normalise_designator(text) == "EGD323A"


@pytest.mark.parametrize(
    ("text", "expected"),
    [("R106", "EGR106"), ("EGR 106", "EGR106"), ("P108", "EGP108"), ("egp108", "EGP108"), ("EGD496", "EGD496"), ("D323B", "EGD323B")],
)
def test_the_kind_digits_and_optional_suffix_are_kept(text, expected):
    assert normalise_designator(text) == expected


@pytest.mark.parametrize("text", ["", "   ", "EGD", "EGD32", "EGD3234", "EGD323AB", "EGX323A", "ABC", "EGD323A EGD323B", "SUP 108", "EGD 323 A", "108/2026"])
def test_a_string_that_is_not_exactly_one_designator_gives_none(text):
    assert normalise_designator(text) is None


def test_matching_is_exact_egd496_is_not_egd496a():
    assert normalise_designator("EGD496") != normalise_designator("EGD496A")
    assert find_designators("EGD496 AND EGD496A") == ["EGD496", "EGD496A"]
    assert find_designators("EGD496A ONLY") == ["EGD496A"]
    assert find_designators("EGD496 ONLY") == ["EGD496"]


def test_j3857_and_j3864_e_text_give_only_egd496a():
    assert find_designators(J3857_E) == ["EGD496A"]
    assert find_designators(J3864_E) == ["EGD496A"]


def test_e_text_with_no_designator_gives_an_empty_list():
    assert find_designators(H7171_E) == []
    assert find_designators("") == []


@pytest.mark.parametrize(
    "text",
    [
        "AIP SUP 108/2026",
        "AIP SUP 108/2026 REFERS",
        "SUP 108",
        "D4960",
        "EGD4960",
        "XD323A",
        "ABCD323A",
        "RP108/2026",
        "EGD323AB",
        "WORD323",
        "ED12",
    ],
)
def test_no_designator_in_the_middle_of_a_word_or_reference(text):
    assert find_designators(text) == []


def test_aip_sup_108_2026_must_not_produce_egp108():
    assert "EGP108" not in find_designators("AIP SUP 108/2026")
    assert "EGP108" not in find_designators("TEMPO DANGER AREA (TDA) EGD496A ACTIVATED. AIP SUP 108/2026 REFERS.")


def test_designators_are_found_next_to_punctuation_and_line_breaks():
    assert find_designators("(EGD323A).") == ["EGD323A"]
    assert find_designators("ACTIVE:D323A,D323B/EGR106;") == ["EGD323A", "EGD323B", "EGR106"]
    assert find_designators("AREA EGD\n323A ACTIVE") == ["EGD323A"]
    assert find_designators("TDA EG D323A ACTIVE") == ["EGD323A"]


def test_lower_case_text_is_found():
    assert find_designators("tempo danger area egd496a activated") == ["EGD496A"]
    assert find_designators("area d323a and r106") == ["EGD323A", "EGR106"]


def test_results_keep_the_order_of_first_appearance_and_drop_repeats():
    text = "EGR106 THEN D323B THEN EGD323A THEN D323B AGAIN AND EGR 106"
    assert find_designators(text) == ["EGR106", "EGD323B", "EGD323A"]
