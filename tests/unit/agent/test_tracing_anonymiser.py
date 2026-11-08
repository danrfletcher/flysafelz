"""G20 (PR-12.F1): what the LangSmith anonymiser masks, and what it leaves alone.

The real `tracing.ANONYMISER` over plain values and payloads. Numbers are Ofcom's drama-range test numbers (07700 900xxx, 01632 960xxx), never a person's.
The phone rows come from ticket 3's own table (`test_redact.PHONE_SHAPES`): every shape `redact_phones` finds, the anonymiser masks.
"""

import re
from pathlib import Path

import pytest
from test_redact import NOT_PHONES, PHONE_SHAPES

from flysafelz import redact
from flysafelz.agent import tracing
from flysafelz.agent.tracing import ANONYMISER, EMAIL_MASK, PHONE_MASK, RULES
from flysafelz.pib import read_notams

PLACEHOLDER = re.compile(r"<PHONE_\d+>")


def masked(text: str) -> str:
    return ANONYMISER(text)


# --- every phone shape of ticket 3's table ---------------------------------------------------------------------------


@pytest.mark.parametrize(("name", "text", "redacted", "mapping"), PHONE_SHAPES, ids=[row[0] for row in PHONE_SHAPES])
def test_every_phone_shape_redact_phones_finds_is_masked_in_the_same_place(name, text, redacted, mapping):
    assert masked(text) == PLACEHOLDER.sub(PHONE_MASK, redacted)  # the same text redact_phones leaves, with its placeholder spelt as the mask
    assert not any(number in masked(text) for number in mapping.values())


SAMPLES = [
    "07700 900123",
    "+44 7700 900123",
    "01632 960001",
    "+44 (0) 1632 960001",
    "0044 7700 900123",
    "07700900123",
    "01632 960 001",
]


@pytest.mark.parametrize("number", SAMPLES)
def test_a_drama_range_number_is_masked_wherever_it_stands(number):
    expected = redact.redact_phones(f"Call {number} now")[0]
    assert expected != f"Call {number} now", "this shape is not one ticket 3 redacts, so it does not belong in this table"
    assert masked(f"Call {number} now") == "Call <PHONE> now"
    assert masked(number) == PHONE_MASK
    assert masked(f"({number})") == f"({PHONE_MASK})"


# Numbers written with a hyphen or a bracket. `redact_phones` leaves these alone (a hyphen ends a number there), so the anonymiser has one more pattern for them,
# `redact.PHONE_PATTERNS[1]`, that `redact_phones` does not use. Each is a drama-range number (Ofcom: 07700 900xxx, 01632 960xxx, 020 7946 0xxx, 0117 496 0xxx).
PUNCTUATED = [
    "07700-900123",
    "07700-900-123",
    "01632-960-001",
    "01632-960001",
    "(07700) 900123",
    "(07700)900123",
    "(01632) 960001",
    "[01632] 960 001",
    "[01632]-960-001",
    "(020) 7946 0123",
    "020-7946-0123",
    "0117-496-0123",
    "+44-7700-900123",
    "+44 (0)7700-900-123",
    "(+44) 7700 900123",
    "(+44) 7700-900-123",
    "+44 (7700) 900123",
    "+44 (1632) 960001",
    "07700 - 900123",
    "01632 - 960 - 001",
    "020 - 7946 - 0123",
]


@pytest.mark.parametrize("number", PUNCTUATED)
def test_a_drama_range_number_with_dashes_or_brackets_is_masked_wherever_it_stands(number):
    assert masked(number) == PHONE_MASK
    assert masked(f"tel: {number}.") == "tel: <PHONE>."
    assert masked(f"Call {number} now") == "Call <PHONE> now"
    assert masked(f"A {number}, B 07700 900123.") == "A <PHONE>, B <PHONE>."


def test_the_dash_and_bracket_numbers_are_ones_redact_phones_does_not_find():
    """The reason the anonymiser has the extra pattern: ticket 3's `redact_phones` would let these through. If it ever finds them, this row can go."""
    for number in ("07700-900123", "(01632) 960001", "[01632] 960 001", "+44-7700-900123"):
        assert redact.redact_phones(number)[0] == number


def test_the_extra_pattern_finds_nothing_in_items_d_f_and_g_of_the_four_archive_pibs():
    """The same corpus ticket 3's redactor is held to: no schedule, date, frequency or reference becomes a phone."""
    pibs = sorted((Path(__file__).resolve().parents[3] / "evals" / "reader" / "pibs").glob("*.xml"))
    assert len(pibs) == 4
    checked = 0
    for pib in pibs:
        for parsed in read_notams(pib.read_bytes())[1]:
            for text in (parsed.item_d, parsed.item_f, parsed.item_g):
                assert ANONYMISER(text or "") == (text or ""), f"{pib.stem[:8]} {parsed.id}"
                checked += bool(text)
    assert checked > 0


# --- the captured PIBs: real data decides what "done" is -------------------------------------------------------------------
#
# Item E of every PIB under evals/ and fixtures/, read at run time. The numbers and schedules are found by shape, so no digit of the corpus is written in this
# file, and a failure message names the file and the NOTAM id, never the text (the assertions below are on plain booleans for the same reason).

_REPO = Path(__file__).resolve().parents[3]
CORPUS_PIBS = sorted((_REPO / "evals").rglob("*.xml")) + sorted((_REPO / "fixtures").rglob("pib.xml"))
_DAY = r"(?:MON|TUE|WED|THU|FRI|SAT|SUN)"
# The two real facility numbers the corpus writes with punctuation: `NNNNN-NNNNNN` (an ATIS number) and `(NNNNN) NNNNNN` (a MOD number).
CORPUS_PUNCTUATED = {
    "NNNNN-NNNNNN": re.compile(r"(?<![\w.\-])0\d{4}-\d{6}(?![\d-])"),
    "(NNNNN) NNNNNN": re.compile(r"(?<![\w.\-])\(0\d{4}\) \d{6}(?!\d)"),
}
# Schedule shapes, none of which may become a phone mask.
CORPUS_SCHEDULES = {
    "DD HHMM-HHMM": re.compile(r"(?<![\d-])\d{2} \d{4}-\d{4}(?![\d-])"),
    "DD-DD HHMM-HHMM": re.compile(r"(?<![\d-])\d{2}-\d{2} \d{4}-\d{4}(?![\d-])"),
    "HHMM-HHMM HHMM": re.compile(r"(?<![\d-])\d{4}-\d{4} \d{4}(?![\d-])"),
    "HHMM-HHMM": re.compile(r"(?<![\d-])\d{4}-\d{4}(?![\d-])"),
    "DAY-DAY": re.compile(rf"\b{_DAY}-{_DAY}\b"),
    "DAY DAY DAY": re.compile(rf"\b{_DAY}(?: {_DAY})+\b"),
}


def corpus_item_es() -> list[tuple[str, str, str]]:
    """(file, NOTAM id, item E text) for every NOTAM of every captured PIB."""
    rows = []
    for pib in CORPUS_PIBS:
        label = f"{pib.parent.name}/{pib.name}" if pib.parent.name != "pibs" else f"pibs/{pib.stem[:8]}"
        rows.extend((label, parsed.id, parsed.item_e) for parsed in read_notams(pib.read_bytes())[1])
    return rows


@pytest.fixture(scope="module")
def corpus_es():
    return corpus_item_es()


def test_the_corpus_is_all_the_captured_pibs(corpus_es):
    labels = {label for label, _, _ in corpus_es}
    assert len(labels) == len(CORPUS_PIBS) >= 9, f"read {len(labels)} of {len(CORPUS_PIBS)} PIBs"
    print(f"corpus: {len(corpus_es)} NOTAMs in {len(labels)} PIBs")


def test_the_corpus_numbers_with_a_hyphen_or_a_bracket_are_masked(corpus_es):
    found = dict.fromkeys(CORPUS_PUNCTUATED, 0)
    for label, notam_id, text in corpus_es:
        out = ANONYMISER(text)
        for shape, pattern in CORPUS_PUNCTUATED.items():
            for match in pattern.finditer(text):
                found[shape] += 1
                gone = match.group(0) not in out
                assert gone, f"{label} {notam_id}: a {shape} number was left in the trace text"
        masks_in, masks_out = text.count(PHONE_MASK), out.count(PHONE_MASK)
        assert masks_out >= masks_in  # a mask already in the text stays
    print("punctuated numbers found:", found)
    assert all(count >= 1 for count in found.values()), f"the corpus no longer holds one of these shapes: {found}"


def test_the_corpus_numbers_with_a_hyphen_or_a_bracket_are_each_one_mask_on_their_own(corpus_es):
    seen = 0
    for label, notam_id, text in corpus_es:
        for pattern in CORPUS_PUNCTUATED.values():
            for match in pattern.finditer(text):
                seen += 1
                ok = ANONYMISER(match.group(0)) == PHONE_MASK
                assert ok, f"{label} {notam_id}: a punctuated facility number alone is not one mask"
    assert seen >= 2


def test_no_schedule_shape_in_the_corpus_becomes_a_phone_mask(corpus_es):
    found = dict.fromkeys(CORPUS_SCHEDULES, 0)
    for label, notam_id, text in corpus_es:
        out = ANONYMISER(text)
        for shape, pattern in CORPUS_SCHEDULES.items():
            in_text = pattern.findall(text)
            found[shape] += len(in_text)
            for substring in in_text:
                alone = ANONYMISER(substring) == substring
                assert alone, f"{label} {notam_id}: a {shape} schedule changed on its own"
            same = sorted(pattern.findall(out)) == sorted(in_text)
            assert same, f"{label} {notam_id}: {shape} schedules in the NOTAM changed in context"
    print("schedule substrings checked:", found)
    # Every shape but the spaced day list occurs in item E of the corpus today (the list is covered by the table below); an empty row would make the check above prove nothing.
    assert all(count >= 1 for shape, count in found.items() if shape != "DAY DAY DAY"), f"the corpus no longer holds a schedule shape: {found}"


@pytest.mark.parametrize("schedule", ["MON-FRI", "MON WED FRI", "SAT SUN", "0800-1700 MON-FRI", "OCT 01 02 03 06 07 08 0900-1700", "27 0800-1600", "03-05 0700-1500", "0800-1800 2000"])
def test_day_lists_and_schedule_forms_are_left_alone(schedule):
    assert masked(schedule) == schedule


def test_placeholders_in_the_corpus_text_are_never_touched(corpus_es):
    """Each item E, redacted the way the model boundary does it, holds `<PHONE_n>`; the anonymiser leaves every one exactly where it is."""
    checked = 0
    for label, notam_id, text in corpus_es:
        redacted = redact.redact_phones(text)[0]
        wanted = PLACEHOLDER.findall(redacted)
        if not wanted:
            continue
        checked += len(wanted)
        kept = PLACEHOLDER.findall(ANONYMISER(redacted)) == wanted
        assert kept, f"{label} {notam_id}: a <PHONE_n> placeholder was changed or double-masked"
    print("placeholders checked:", checked)
    assert checked > 0


def test_two_numbers_in_one_string_are_both_masked():
    assert masked("A 07700 900123, B 01632 960001.") == "A <PHONE>, B <PHONE>."


# --- emails -------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("ops@example.org", EMAIL_MASK),
        ("Email ops@example.org for booking.", f"Email {EMAIL_MASK} for booking."),
        ("FOR INFO CTC OPS@EXAMPLE.COM OR 01632 960001", f"FOR INFO CTC {EMAIL_MASK} OR {PHONE_MASK}"),
        ("first.last+tag@mail.example.co.uk", EMAIL_MASK),
        ("a@b.io and c@d.io", f"{EMAIL_MASK} and {EMAIL_MASK}"),
    ],
)
def test_an_email_is_masked(text, expected):
    assert masked(text) == expected


# --- what must come through unchanged -----------------------------------------------------------------------------------

UNCHANGED = [
    "534732N 0000459W",
    "WI 0.5NM RADIUS OF 534732N 0000459W (GARTON)",
    "2026-10-31T07:30:00Z",
    "2026-10-31T10:00:00+00:00",
    "2026-10-31 10:00",
    "H7171/26",
    "J3857/26",
    "NOTAM H7171/26 AND J3857/26",
    "53.802797",
    "53.802797, -0.086644",
    "10:00\u201311:00 GMT",
    "800 ft AMSL",
    "2 NOTAMs within 2 NM. 0 awareness-only. 34 ignored.",
    "<PHONE_1>",
    "<PHONE_3>",
    "Phone Flylogix <PHONE_1> first (J3857/26)",
    "<PHONE_1> OR <PHONE_12>",
    "OPS(AT)FLYLOGIX.COM",
    "AR-2026-10264/01",
    "EGD496A",
    "",
    " ",
]


@pytest.mark.parametrize("text", UNCHANGED)
def test_text_that_is_not_a_phone_or_an_email_is_left_exactly_as_it_is(text):
    assert masked(text) == text


@pytest.mark.parametrize(("name", "text"), NOT_PHONES, ids=[row[0] for row in NOT_PHONES])
def test_everything_ticket_3_says_is_not_a_phone_is_left_alone_unless_it_is_an_email(name, text):
    expected = re.sub(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+", EMAIL_MASK, text) if "@" in text else text
    assert masked(text) == expected


def test_a_string_that_is_only_a_placeholder_is_never_masked_twice():
    once = masked("<PHONE_3>")
    assert once == "<PHONE_3>" and masked(once) == "<PHONE_3>"
    assert masked(masked("call 01632 960001")) == "call <PHONE>"  # the mask itself is no phone shape either


# --- the parity with ticket 3 ----------------------------------------------------------------------------------------------


def test_the_phone_rules_are_the_very_patterns_redact_phones_uses():
    phone_rules = [rule for rule in RULES if rule["replace"] == PHONE_MASK]
    assert [rule["pattern"] for rule in phone_rules] == list(redact.PHONE_PATTERNS)
    assert all(rule["pattern"] is pattern for rule, pattern in zip(phone_rules, redact.PHONE_PATTERNS, strict=True))  # the same objects, not equal copies
    assert redact.PHONE_PATTERNS[0] is redact._PHONE  # and redact_phones itself reads `_PHONE`


def test_tracing_has_no_phone_pattern_of_its_own():
    source = Path(tracing.__file__).read_text(encoding="utf-8")
    assert "_PHONE" not in source and not re.search(r"re\.compile\([^)]*\\d", source.replace(tracing.EMAIL.pattern, ""))  # F7: no copied or invented digit pattern


def test_rules_are_one_per_phone_pattern_and_one_email():
    assert len(RULES) == len(redact.PHONE_PATTERNS) + 1
    assert RULES[-1]["pattern"] is tracing.EMAIL and RULES[-1]["replace"] == EMAIL_MASK


# --- payloads --------------------------------------------------------------------------------------------------------------


def payload() -> dict:
    return {
        "messages": [
            {"type": "human", "content": "Can I fly here? Mail me at pilot@example.org"},
            {"type": "tool", "name": "get_notams", "content": "FOR INFO 01632 960001. AR-2026-10264/01.", "artifact": {"contacts": ["07700 900123"], "n": 2}},
        ],
        "overlays": [
            {
                "type": "Feature",
                "geometry": {"type": "Polygon", "coordinates": [[[-0.086644, 53.83607602899523], [-0.08, 53.8], [-0.086644, 53.83607602899523]]]},
                "properties": {"text": "CTC 01632 960001 OR ops@example.org", "id": "J3857/26"},
            }
        ],
        "phone_map": {"<PHONE_1>": "07700 900123"},
        "lat": 53.802797,
        "lon": -0.086644,
        "flag": True,
        "nothing": None,
        "count": 3,
    }


def test_a_phone_or_email_deep_inside_a_nested_payload_is_masked():
    out = ANONYMISER(payload())
    assert out["messages"][0]["content"] == f"Can I fly here? Mail me at {EMAIL_MASK}"
    assert out["messages"][1]["content"] == f"FOR INFO {PHONE_MASK}. AR-2026-10264/01."
    assert out["messages"][1]["artifact"]["contacts"] == [PHONE_MASK]
    assert out["overlays"][0]["properties"]["text"] == f"CTC {PHONE_MASK} OR {EMAIL_MASK}"
    assert out["phone_map"] == {"<PHONE_1>": PHONE_MASK}  # the key is the placeholder, the value was the real number


def test_geojson_coordinates_stay_numbers_and_keep_every_digit():
    out = ANONYMISER(payload())
    assert out["overlays"][0]["geometry"]["coordinates"] == [[[-0.086644, 53.83607602899523], [-0.08, 53.8], [-0.086644, 53.83607602899523]]]
    assert out["lat"] == 53.802797 and out["lon"] == -0.086644
    assert all(isinstance(x, float) for ring in out["overlays"][0]["geometry"]["coordinates"] for point in ring for x in point)
    assert out["overlays"][0]["properties"]["id"] == "J3857/26"


@pytest.mark.parametrize("value", [None, True, False, 0, 3, 53.802797, -0.086644, 1e-9, [], {}, [None, 1, 2.5], {"a": None, "b": 1}], ids=repr)
def test_a_value_that_is_not_a_string_passes_through_untouched(value):
    expected = repr(value)
    assert repr(ANONYMISER(value)) == expected


def test_an_empty_string_is_untouched_inside_a_payload():
    assert ANONYMISER({"a": "", "b": ["", None]}) == {"a": "", "b": ["", None]}


def test_an_unmasked_payload_is_the_same_payload():
    plain = {"a": "no number here 12:00\u201313:00", "b": [1, 2, {"c": "H7171/26"}]}
    assert ANONYMISER(plain) == {"a": "no number here 12:00\u201313:00", "b": [1, 2, {"c": "H7171/26"}]}


def test_the_walk_reaches_deeper_than_langsmiths_own_default_of_ten_levels():
    value: object = "CTC 01632 960001"
    for _ in range(25):
        value = {"inner": [value]}
    out = ANONYMISER(value)
    for _ in range(25):
        out = out["inner"][0]
    assert out == f"CTC {PHONE_MASK}"
