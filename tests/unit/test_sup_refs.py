"""sup_refs.py: AIP SUP references from item E by regex (G10, E4), and the one wording that names them."""

import pytest

from flysafelz.sup_refs import find_sup_references, sup_detail


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("TEMPO DANGER AREA (TDA) EGD496A ACTIVATED. AIP SUP 108/2026\nREFERS.", ["AIP SUP 108/2026"]),
        ("SEE AIP SUP 108/2026", ["AIP SUP 108/2026"]),
        ("SEE SUP 108/2026", ["AIP SUP 108/2026"]),  # AIP added when the text has none
        ("SEE SUPPLEMENT 108/2026", ["AIP SUP 108/2026"]),  # SUPPLEMENT shortened to SUP
        ("SEE AIP SUPPLEMENT 108/2026", ["AIP SUP 108/2026"]),
        ("(AIP SUP 108/2026).", ["AIP SUP 108/2026"]),
        ("REF:AIP SUP 108/2026,", ["AIP SUP 108/2026"]),
    ],
)
def test_a_single_reference_is_found_and_canonicalised(text, expected):
    assert find_sup_references(text) == expected


def test_two_references_come_back_in_order_of_appearance_not_sorted():
    text = "TDA ACTIVATED. AIP SUP 112/2026 REFERS. SEE AIP SUP 108/2026"
    assert find_sup_references(text) == ["AIP SUP 112/2026", "AIP SUP 108/2026"]


def test_a_line_break_inside_the_reference_is_collapsed():
    assert find_sup_references("REF AIP SUP\n35/26 FOR DETAILS") == ["AIP SUP 35/26"]
    assert find_sup_references("REF AIP\nSUP\n 35/26") == ["AIP SUP 35/26"]
    assert find_sup_references("REF AIP  \t SUP   35/26") == ["AIP SUP 35/26"]


@pytest.mark.parametrize("text", ["aip sup 108/2026", "Aip Sup 108/2026", "AIP sup 108/2026", "aip supplement 108/2026", "sup 108/2026"])
def test_case_does_not_matter_and_the_result_is_upper_case(text):
    assert find_sup_references(text) == ["AIP SUP 108/2026"]


def test_digits_are_kept_as_written():
    assert find_sup_references("AIP SUP 008/2026") == ["AIP SUP 008/2026"]  # leading zeros stay
    assert find_sup_references("AIP SUP 8/2026") == ["AIP SUP 8/2026"]
    assert find_sup_references("AIP SUP 35/26") == ["AIP SUP 35/26"]  # a two-digit year stays two digits
    assert find_sup_references("AIP SUP 35/26 AND AIP SUP 035/2026") == ["AIP SUP 35/26", "AIP SUP 035/2026"]  # not the same text, so not a repeat


def test_a_repeat_after_canonicalising_is_dropped_keeping_the_first_place():
    text = "SUP 108/2026 REFERS. AIP SUP\n108/2026. SUPPLEMENT 108/2026. AIP SUP 112/2026. aip sup 108/2026"
    assert find_sup_references(text) == ["AIP SUP 108/2026", "AIP SUP 112/2026"]


def test_the_non_breaking_space_is_whitespace_and_canonicalises_the_same_way():
    """Decided: a non-breaking space between SUP and the number is whitespace (regex \\s is Unicode), so it is a hit, not a pinned miss."""
    assert find_sup_references("AIP SUP\u00a0108/2026") == ["AIP SUP 108/2026"]
    assert find_sup_references("AIP\u00a0SUP\u00a0108/2026") == ["AIP SUP 108/2026"]


def test_only_ascii_digits_count_so_the_result_always_has_the_aip_sup_n_slash_yyyy_shape():
    assert find_sup_references("AIP SUP \u0661\u0660\u0668/\u0662\u0660\u0662\u0666") == []  # Arabic-Indic digits


@pytest.mark.parametrize(
    "text",
    [
        "SUPERSEDED BY AIP SUP WEF 12 NOV",
        "THIS AIP SUP IS AVBL AT WWW.EXAMPLE.INVALID",
        "MIL SUP CHECKLIST 001/2022",
        "SUPERVISOR",
        "SUPPORT",
        "SUP 108",
        "AIP SUP 108/",
        "AIP SUP /2026",
        "XSUP 108/2026",
        "SUP 108/20261",
        "SUP 1080/2026",  # a four-digit number is not a supplement number
        "SUP 108/2",  # a one-digit year is not a year
        "SUP108/2026",  # no gap between SUP and the number
        "",
        "   ",
    ],
)
def test_text_that_is_not_a_reference_gives_nothing(text):
    assert find_sup_references(text) == []


def test_known_miss_the_misspelt_supplement_is_not_matched():
    """Recorded, not fixed: item E of I6367/26 in the committed PIBs writes 'AIP SUPPELEMENT 022/2026'. Changing this must be deliberate."""
    assert find_sup_references("ED. AIP SUPPELEMENT 022/2026 ED") == []
    assert find_sup_references("AIP SUPPELEMENT 022/2026 REFERS. NOTAM WILL BE SUPERSEDED BY AIP SUP WEF 12 NOV") == []


def test_no_reference_gives_an_empty_list():
    assert find_sup_references("UAS OPR WI 0.5NM RADIUS OF 534732N 0000459W (GARTON). AR-2026-10264/01.") == []


# --- sup_detail: the one wording ------------------------------------------------


def test_sup_detail_wording_for_one_reference_is_pinned():
    """The exact string GP4/CR1 expect in the answer. PR-3 and PR-4 take their wording from here."""
    assert sup_detail(["AIP SUP 108/2026"]) == "details in AIP SUP 108/2026"


def test_sup_detail_wording_for_two_references_is_pinned():
    assert sup_detail(["AIP SUP 108/2026", "AIP SUP 112/2026"]) == "details in AIP SUP 108/2026, AIP SUP 112/2026"


def test_sup_detail_for_none_is_none_so_the_bracketed_part_is_dropped():
    assert sup_detail([]) is None
    assert sup_detail(find_sup_references("UAS OPR WI 0.5NM RADIUS OF 534732N 0000459W")) is None


def test_sup_detail_from_found_references_end_to_end():
    assert sup_detail(find_sup_references("TDA EGD496A ACTIVATED. SEE sup\n108/2026.")) == "details in AIP SUP 108/2026"
    assert sup_detail(find_sup_references("… AIP SUP 112/2026 … SEE AIP SUP 108/2026")) == "details in AIP SUP 112/2026, AIP SUP 108/2026"
