"""ParsedNotam.sup_references and .designators: read-only properties over item E (G10, G11), never fields.

Read through parse_pib on the committed Blue Mill PIB, plus constructed items E. The serialised shape does not change.
"""

from pathlib import Path

import pytest
from factories import notam
from test_designators import H7171_E, J3857_E

from flysafelz.contracts import NotamReading, ParsedNotam
from flysafelz.pib import parse_pib

ROOT = Path(__file__).resolve().parents[2]
BLUE_MILL_PIB = ROOT / "fixtures" / "blue_mill_2026-10-31" / "pib.xml"


@pytest.fixture(scope="module")
def blue_mill() -> dict[str, ParsedNotam]:
    _issued, notams = parse_pib(BLUE_MILL_PIB.read_bytes())
    return {n.id: n for n in notams}


# The ticket text names J3853/26; the committed Blue Mill PIB holds the same TDA text as J3857/26 and J3864/26 (EGD496A),
# and J3858/26 and J3865/26 (EGD496B). All QRD NOTAMs that name AIP SUP 108/2026 are listed.
EGD496A_IDS = ["J3857/26", "J3864/26"]
EGD496B_IDS = ["J3858/26", "J3865/26"]


@pytest.mark.parametrize("notam_id", EGD496A_IDS + EGD496B_IDS)
def test_the_blue_mill_tda_notams_name_aip_sup_108_2026_through_parse_pib(blue_mill, notam_id):
    assert blue_mill[notam_id].sup_references == ["AIP SUP 108/2026"]


@pytest.mark.parametrize("notam_id", EGD496A_IDS)
def test_the_egd496a_notams_give_egd496a(blue_mill, notam_id):
    assert blue_mill[notam_id].designators == ["EGD496A"]


@pytest.mark.parametrize("notam_id", EGD496B_IDS)
def test_the_egd496b_notams_give_egd496b(blue_mill, notam_id):
    assert blue_mill[notam_id].designators == ["EGD496B"]


def test_h7171_has_neither_a_sup_reference_nor_a_designator(blue_mill):
    assert blue_mill["H7171/26"].sup_references == []
    assert blue_mill["H7171/26"].designators == []


def test_a_qrd_with_no_sup_reference_gives_an_empty_list(blue_mill):
    """FR5: later condition text drops the bracketed 'details in ...' part. The EGD323 NOTAMs in the same PIB name none."""
    no_sup = [n for n in blue_mill.values() if n.subject == "RD" and not n.sup_references]
    assert no_sup, "the fixture should hold a QRD with no SUP reference"
    assert {n.id for n in no_sup} >= {"M7365/26", "D4278/26"}
    assert all(n.sup_references == [] for n in no_sup)


def test_a_constructed_item_e_is_read_through_the_properties():
    constructed = notam("J9999/26", item_e="TDA D323A AND EGD323AB ACTIVATED. AIP SUP 112/2026 … SEE AIP SUP 108/2026")
    assert constructed.designators == ["EGD323A", "EGD323AB"]
    assert constructed.sup_references == ["AIP SUP 112/2026", "AIP SUP 108/2026"]


def test_the_existing_item_e_constants_give_the_same_answers_as_the_fixture():
    assert notam("J3857/26", item_e=J3857_E).sup_references == ["AIP SUP 108/2026"]
    assert notam("J3857/26", item_e=J3857_E).designators == ["EGD496A"]
    assert notam("H7171/26", item_e=H7171_E).sup_references == []
    assert notam("H7171/26", item_e=H7171_E).designators == []


def test_the_properties_follow_item_e_and_are_read_only():
    a = notam("J9999/26", item_e="TDA EGD496A. AIP SUP 108/2026")
    b = a.model_copy(update={"item_e": "TDA EGD496B. AIP SUP 112/2026"})
    assert (b.designators, b.sup_references) == (["EGD496B"], ["AIP SUP 112/2026"])
    assert (a.designators, a.sup_references) == (["EGD496A"], ["AIP SUP 108/2026"])
    for name in ("sup_references", "designators"):  # a property with no setter: assignment is refused
        with pytest.raises(AttributeError):
            setattr(a, name, ["AIP SUP 1/26"])


# --- the serialised contract does not change ------------------------------------------------------------


def test_parsed_notam_has_no_sup_references_or_designators_field():
    assert "sup_references" not in ParsedNotam.model_fields
    assert "designators" not in ParsedNotam.model_fields
    assert "sup_references" not in ParsedNotam.model_computed_fields
    assert "designators" not in ParsedNotam.model_computed_fields


def test_the_serialised_shape_carries_neither_property_and_round_trips(blue_mill):
    original = blue_mill["J3857/26"]
    dumped = original.model_dump()
    assert "sup_references" not in dumped and "designators" not in dumped
    assert "sup_references" not in original.model_dump_json() and "designators" not in original.model_dump_json()
    assert ParsedNotam.model_validate_json(original.model_dump_json()) == original
    for extra in ("sup_references", "designators"):  # a model cannot fill them: extra fields are refused
        with pytest.raises(ValueError):
            ParsedNotam.model_validate({**dumped, extra: ["x"]})


def test_the_parsed_notam_field_set_is_the_one_before_this_pr():
    assert set(ParsedNotam.model_fields) == {
        "id", "series", "number", "year", "type", "replaces", "nof", "fir", "q_code", "traffic", "purpose", "scope", "lower_fl", "upper_fl",
        "centre_lat", "centre_lon", "radius_nm", "item_a", "valid_from", "valid_to", "permanent", "est", "item_d", "item_e", "item_f", "item_g",
        "section", "sections",
    }  # fmt: skip


def test_notam_reading_keeps_activates_designators_and_gains_no_sup_field():
    assert "activates_designators" in NotamReading.model_fields
    assert not [name for name in NotamReading.model_fields if "sup" in name.lower()]
    assert set(NotamReading.model_fields) == {
        "id", "category", "activity", "geometry", "floor", "ceiling", "schedule", "applies_to", "exemptions", "contact",
        "activates_designators", "prohibitive_reason", "prohibitive_quote",
    }  # fmt: skip
