"""redact_text / redact_artifact: the one boundary where phone numbers become <PHONE_n> (P1, FR13).

Phone numbers here are Ofcom's drama ranges only (07700 900xxx, 01632 960xxx).
"""

import dataclasses
import json
from datetime import UTC, datetime
from enum import Enum, StrEnum

import pytest
from pydantic import BaseModel, ConfigDict, PrivateAttr

from flysafelz.agent.redaction import redact_artifact, redact_text
from flysafelz.agent.state import OverlayFeature, OverlayProperties, PointGeometry

MOBILE = "07700 900123"
LANDLINE = "01632 960001"
OTHER_MOBILE = "07700 900456"


# --- the rule: numbering by first appearance, stable across turns ---------------------------------------------


def test_two_numbers_are_numbered_in_order_of_first_appearance():
    text, phone_map = redact_text(f"Ring {MOBILE} first, then {LANDLINE}.", {})
    assert text == "Ring <PHONE_1> first, then <PHONE_2>."
    assert phone_map == {"<PHONE_1>": MOBILE, "<PHONE_2>": LANDLINE}


def test_the_same_number_again_keeps_its_placeholder_whatever_the_spacing():
    text, phone_map = redact_text(f"{MOBILE}, {LANDLINE}, {MOBILE}, 07700900123, 01632960001", {})
    assert text == "<PHONE_1>, <PHONE_2>, <PHONE_1>, <PHONE_1>, <PHONE_2>"
    assert list(phone_map) == ["<PHONE_1>", "<PHONE_2>"]


def test_a_number_in_international_form_is_redacted_even_though_ticket_3_numbers_it_separately():
    """Ticket 3's normaliser keeps a leading + (`+447700900123`), so 07700 900123 and +44 7700 900123 are two numbers to it and get two
    placeholders. Both are hidden; the model just sees two placeholders for what is one phone."""
    text, phone_map = redact_text(f"{MOBILE} or +44 7700 900123", {})
    assert text == "<PHONE_1> or <PHONE_2>"
    assert phone_map == {"<PHONE_1>": MOBILE, "<PHONE_2>": "+44 7700 900123"}


def test_a_later_turns_map_continues_the_numbering_and_reuses_what_it_has():
    turn_one_text, after_turn_one = redact_text(f"{MOBILE} and {LANDLINE}", {})
    assert turn_one_text == "<PHONE_1> and <PHONE_2>"
    turn_two_text, after_turn_two = redact_text(f"{OTHER_MOBILE}, then {LANDLINE} again, then {MOBILE}", after_turn_one)
    assert turn_two_text == "<PHONE_3>, then <PHONE_2> again, then <PHONE_1>"
    assert after_turn_two == {"<PHONE_1>": MOBILE, "<PHONE_2>": LANDLINE, "<PHONE_3>": OTHER_MOBILE}


def test_the_map_given_is_never_changed():
    given = {"<PHONE_1>": MOBILE}
    _text, returned = redact_text(f"{LANDLINE}", given)
    assert given == {"<PHONE_1>": MOBILE}
    assert returned is not given and returned == {"<PHONE_1>": MOBILE, "<PHONE_2>": LANDLINE}
    _artifact, returned = redact_artifact({"a": LANDLINE}, given)
    assert given == {"<PHONE_1>": MOBILE} and returned == {"<PHONE_1>": MOBILE, "<PHONE_2>": LANDLINE}


def test_redacting_redacted_text_changes_nothing():
    once, phone_map = redact_text(f"{MOBILE} / {LANDLINE}", {})
    again, same_map = redact_text(once, phone_map)
    assert again == once and same_map == phone_map


# --- what is left alone ---------------------------------------------------------------------------------------

LEFT_ALONE = [
    "",
    "no number in this at all",
    "J3857/26 H7171/26 A1234/25",
    "534732N 0000459W",
    "centre 5348N00005W radius 5",
    "2026-10-31T07:30:00Z",
    "2026-10-31 07:30 GMT",
    "2610311200",
    "frequency 119.130 and 121.500",
    "OCT 01 02 03 06 07 08 0900-1700",
    "AR-2026-10264/01",
    "SFC TO 1300FT AMSL 0800-1800",
]


@pytest.mark.parametrize("text", LEFT_ALONE)
def test_text_that_is_not_a_phone_number_comes_back_byte_for_byte(text):
    redacted, phone_map = redact_text(text, {})
    assert redacted == text and phone_map == {}


def test_a_number_inside_surrounding_notam_text_leaves_the_rest_alone():
    text = "J3857/26 ACTIVATES EGD496A 534732N 0000459W. CTC FLYLOGIX 07700 900123 BEFORE 2026-10-31T07:30:00Z."
    redacted, _ = redact_text(text, {})
    assert redacted == "J3857/26 ACTIVATES EGD496A 534732N 0000459W. CTC FLYLOGIX <PHONE_1> BEFORE 2026-10-31T07:30:00Z."


# --- artifacts ------------------------------------------------------------------------------------------------


class Contact(BaseModel):
    model_config = ConfigDict(frozen=True)
    name: str
    number: str | None = None


class Notam(BaseModel):
    model_config = ConfigDict(frozen=True)
    id: str
    text: str
    contacts: list[Contact]
    extras: dict[str, list[str]]


class Briefing(BaseModel):
    """A list of dicts inside a model, inside a model: the nesting redaction has to reach."""

    model_config = ConfigDict(frozen=True)
    notams: list[Notam]
    lines: list[dict[str, str | int | None]]
    count: int
    when: datetime


def briefing() -> Briefing:
    return Briefing(
        notams=[
            Notam(
                id="J3857/26",
                text=f"CTC FLYLOGIX {MOBILE} OR {LANDLINE}",
                contacts=[Contact(name="Flylogix", number=MOBILE), Contact(name="Other", number=None)],
                extras={"alt": [OTHER_MOBILE, "no number"], "again": [LANDLINE]},
            ),
            Notam(id="H7171/26", text="nothing here", contacts=[], extras={}),
        ],
        lines=[{"text": f"call {OTHER_MOBILE}", "n": 2, "none": None}, {"text": MOBILE}],
        count=2,
        when=datetime(2026, 10, 31, 7, 30, tzinfo=UTC),
    )


def test_a_nested_artifact_has_no_real_number_left_anywhere():
    redacted, phone_map = redact_artifact(briefing(), {})
    dumped = json.dumps(redacted.model_dump(mode="json"))
    for number in (MOBILE, LANDLINE, OTHER_MOBILE):
        assert number not in dumped
        assert number.replace(" ", "") not in dumped.replace(" ", "")
    assert sorted(phone_map.values()) == sorted([MOBILE, LANDLINE, OTHER_MOBILE])


def test_artifact_numbers_are_numbered_in_the_order_the_walk_meets_them():
    redacted, phone_map = redact_artifact(briefing(), {})
    assert redacted.notams[0].text == "CTC FLYLOGIX <PHONE_1> OR <PHONE_2>"
    assert redacted.notams[0].contacts[0].number == "<PHONE_1>"
    assert redacted.notams[0].extras == {"alt": ["<PHONE_3>", "no number"], "again": ["<PHONE_2>"]}
    assert redacted.lines == [{"text": "call <PHONE_3>", "n": 2, "none": None}, {"text": "<PHONE_1>"}]
    assert phone_map == {"<PHONE_1>": MOBILE, "<PHONE_2>": LANDLINE, "<PHONE_3>": OTHER_MOBILE}


def test_an_artifact_walk_continues_the_threads_numbering():
    _, turn_one = redact_text(f"{OTHER_MOBILE}", {})
    redacted, phone_map = redact_artifact(briefing(), turn_one)
    assert redacted.notams[0].extras["alt"][0] == "<PHONE_1>"
    assert redacted.notams[0].text == "CTC FLYLOGIX <PHONE_2> OR <PHONE_3>"
    assert phone_map == {"<PHONE_1>": OTHER_MOBILE, "<PHONE_2>": MOBILE, "<PHONE_3>": LANDLINE}


def test_the_walk_keeps_the_types_and_every_value_that_is_not_text():
    original = briefing()
    redacted, _ = redact_artifact(original, {})
    assert type(redacted) is Briefing and type(redacted.notams[0]) is Notam and type(redacted.notams[0].contacts[0]) is Contact
    assert redacted.count == 2 and redacted.when == original.when
    assert redacted.notams[1] == original.notams[1]  # nothing to redact: equal
    assert original.notams[0].text.startswith("CTC FLYLOGIX 07700")  # the input is not changed
    with pytest.raises(ValueError, match="frozen"):
        redacted.count = 3


def test_an_artifact_with_nothing_to_redact_comes_back_equal_and_the_map_is_unchanged():
    clean = {"id": "J3857/26", "where": "534732N 0000459W", "n": 2}
    redacted, phone_map = redact_artifact(clean, {"<PHONE_1>": MOBILE})
    assert redacted == clean and phone_map == {"<PHONE_1>": MOBILE}


def test_dict_keys_tuples_sets_and_dataclasses_are_walked_too():
    @dataclasses.dataclass(frozen=True)
    class Row:
        label: str
        numbers: tuple[str, ...]

    artifact = {MOBILE: Row(label=f"ring {LANDLINE}", numbers=(OTHER_MOBILE, "x")), "set": frozenset({LANDLINE})}
    redacted, phone_map = redact_artifact(artifact, {})
    assert redacted == {"<PHONE_1>": Row(label="ring <PHONE_2>", numbers=("<PHONE_3>", "x")), "set": frozenset({"<PHONE_2>"})}
    assert phone_map == {"<PHONE_1>": MOBILE, "<PHONE_2>": LANDLINE, "<PHONE_3>": OTHER_MOBILE}


def test_a_named_tuple_keeps_its_type():
    from typing import NamedTuple

    class Pair(NamedTuple):
        a: str
        b: str

    redacted, _ = redact_artifact(Pair(MOBILE, "x"), {})
    assert redacted == Pair("<PHONE_1>", "x") and type(redacted) is Pair


def test_a_model_that_allows_extra_fields_has_them_redacted_too():
    class Loose(BaseModel):
        model_config = ConfigDict(extra="allow")
        name: str

    redacted, _ = redact_artifact(Loose.model_validate({"name": "n", "hidden": f"call {MOBILE}"}), {})
    assert redacted.model_extra == {"hidden": "call <PHONE_1>"}


def test_a_private_attribute_of_a_model_is_redacted_too():
    class Hidden(BaseModel):
        model_config = ConfigDict(frozen=True)
        name: str
        _secret: str = PrivateAttr(default="")
        _more: list[str] = PrivateAttr(default_factory=list)

    original = Hidden(name="n")
    original._secret = f"call {MOBILE}"
    original._more = [LANDLINE]
    redacted, phone_map = redact_artifact(original, {})
    assert redacted._secret == "call <PHONE_1>" and redacted._more == ["<PHONE_2>"]
    assert phone_map == {"<PHONE_1>": MOBILE, "<PHONE_2>": LANDLINE}
    assert original._secret == f"call {MOBILE}"  # the artifact given is not changed
    assert MOBILE not in repr(redacted.__pydantic_private__) and LANDLINE not in repr(redacted.__pydantic_private__)


def test_a_dataclass_field_that_is_not_an_init_argument_is_redacted_not_reset():
    @dataclasses.dataclass(frozen=True)
    class Row:
        label: str
        note: str = dataclasses.field(init=False, default="")
        unset: str = dataclasses.field(init=False)  # never assigned: nothing to carry over

        def __post_init__(self):
            object.__setattr__(self, "note", f"call {MOBILE}")

    redacted, phone_map = redact_artifact(Row(label="x"), {})
    assert redacted.note == "call <PHONE_1>" and not hasattr(redacted, "unset")
    assert phone_map == {"<PHONE_1>": MOBILE}


class Colour(Enum):
    RED = 1


class Word(StrEnum):
    HELLO = "hello"


def test_enums_numbers_none_and_dates_pass_through_as_they_are():
    artifact = [Colour.RED, Word.HELLO, 3, 2.5, True, None, datetime(2026, 10, 31, tzinfo=UTC)]
    redacted, phone_map = redact_artifact(artifact, {})
    assert redacted == artifact and phone_map == {}
    assert redacted[0] is Colour.RED and redacted[1] is Word.HELLO


def test_a_value_the_walk_does_not_know_raises_rather_than_pass_unread():
    class Opaque:
        number = MOBILE

    with pytest.raises(TypeError, match="Opaque"):
        redact_artifact({"x": Opaque()}, {})
    with pytest.raises(TypeError, match="bytes"):
        redact_artifact([MOBILE.encode()], {})


def test_the_real_overlay_shape_is_redacted():
    """A NOTAM overlay carries the raw E text; the model and the trace see the placeholder."""
    overlay = OverlayFeature(
        geometry=PointGeometry(coordinates=[-0.0833, 53.7922]),
        properties=OverlayProperties(id="J3857/26", kind="notam", action="CONTACT_FIRST", text=f"CTC {MOBILE}"),
    )
    redacted, phone_map = redact_artifact([overlay], {})
    assert redacted[0].properties.text == "CTC <PHONE_1>"
    assert redacted[0].properties.id == "J3857/26" and redacted[0].geometry == overlay.geometry
    assert phone_map == {"<PHONE_1>": MOBILE}
