"""G1: phone redaction. One row per shape rule, not per NOTAM. Pure: no fixtures, no files.

Numbers here are Ofcom's drama-range test numbers (07700 900xxx, 01632 960xxx): never a real person's.
"""

import re

import pytest

from flysafelz.redact import e_hash, redact_phones, rehydrate

# (name, text, redacted, {placeholder: number as written})
PHONE_SHAPES = [
    ("uk mobile with a space", "CTC 07700 900123.", "CTC <PHONE_1>.", {"<PHONE_1>": "07700 900123"}),
    ("uk landline unspaced", "CTC 01632960001.", "CTC <PHONE_1>.", {"<PHONE_1>": "01632960001"}),
    ("uk number in an odd split", "TEL 077009 00124 NOW", "TEL <PHONE_1> NOW", {"<PHONE_1>": "077009 00124"}),
    ("uk number in three groups", "TEL 01632 960 003.", "TEL <PHONE_1>.", {"<PHONE_1>": "01632 960 003"}),
    ("uk london number, three groups", "CTC 020 7946 0958.", "CTC <PHONE_1>.", {"<PHONE_1>": "020 7946 0958"}),
    ("uk number across a line break", "CTC 01632 \n960003 NOW", "CTC <PHONE_1> NOW", {"<PHONE_1>": "01632 \n960003"}),
    ("uk number across a crlf line break", "CTC 01632\r\n960003 NOW", "CTC <PHONE_1> NOW", {"<PHONE_1>": "01632\r\n960003"}),
    (
        "two numbers split by a slash",
        "077009 00125 / \n01632 960004",
        "<PHONE_1> / \n<PHONE_2>",
        {"<PHONE_1>": "077009 00125", "<PHONE_2>": "01632 960004"},
    ),
    ("two numbers split by OR", "07700 900123 OR 01632 960001", "<PHONE_1> OR <PHONE_2>", {"<PHONE_1>": "07700 900123", "<PHONE_2>": "01632 960001"}),
    ("uk number with a twelfth digit typo", "FOR INFO 07700 9001234.", "FOR INFO <PHONE_1>.", {"<PHONE_1>": "07700 9001234"}),
    ("international plus", "CTC +44 1632 960002.", "CTC <PHONE_1>.", {"<PHONE_1>": "+44 1632 960002"}),
    ("international plus, short", "CTC +47 000 00 000.", "CTC <PHONE_1>.", {"<PHONE_1>": "+47 000 00 000"}),
    ("international plus with (0)", "CTC +44 (0)7700 900123.", "CTC <PHONE_1>.", {"<PHONE_1>": "+44 (0)7700 900123"}),
    ("international 00", "CTC 0044 7700 900123.", "CTC <PHONE_1>.", {"<PHONE_1>": "0044 7700 900123"}),
    ("M tag kept, its digits redacted", "CTC +44 M01632 960005 PLEASE", "CTC +44 M<PHONE_1> PLEASE", {"<PHONE_1>": "01632 960005"}),
    ("number at the very start and end", "07700 900123", "<PHONE_1>", {"<PHONE_1>": "07700 900123"}),
    ("number inside brackets", "(CTC 01632 960001)", "(CTC <PHONE_1>)", {"<PHONE_1>": "01632 960001"}),
]

# Phone-like shapes that must come through unchanged.
NOT_PHONES = [
    ("frequency", "CTC 119.130 MHZ"),
    ("frequency with channel", "CH 121.605"),
    ("frequency in a list", "ON 118.250 OR 121.500."),
    ("dms coordinate, seconds", "WI 0.5NM RADIUS OF 534732N 0000459W."),
    ("dms coordinate, minutes", "PSN 5348N 00005W."),
    ("dms coordinate across a line break", "AREA 534732N\n0000459W - 534800N 0000500W"),
    ("two coordinates in a row", "534732N 0000459W 534800N 0000500W 534900N 0000600W"),
    ("height agl", "MAX 700FT AGL."),
    ("height amsl range", "SFC TO 4500FT AMSL"),
    ("flight level", "FL195 TO FL660"),
    ("notam id", "REPLACES H7171/26 AND A0123/26."),
    ("notam id with a leading zero", "NOTAM B0012/26 REFERS"),
    ("ar reference", "AR-2026-10264/01."),
    ("ar reference, short", "AR-2026-0110/01."),
    ("date-time group", "FROM 2610311200 TO 2610311800"),
    ("time range", "0800-1800"),
    ("time ranges in a schedule", "MON-FRI 0800-1800, SAT 0900-1300"),
    ("two time ranges with spaces", "06 1630-2040, 07 1630-2040"),
    ("day-of-month list", "OCT 01 02 03 06 07 08 0900-1700"),
    ("day-of-month list with a time", "NOV 03 04 05 0900-1400"),
    ("d item schedule, days then times", "OCT 01 02 09 10 17 0730-1600, NOV 03 04 10 13 14 0800-1700"),
    ("d item schedule, day list across a line break", "MAR 05 19\n26  0800-1600"),
    ("d item schedule, days of two months", "JAN 01 03 04 09 11 1000-1500, FEB 07 08 09 0900-1200"),
    ("aic reference", "AIC M057/2026 WITH CHART, REFERS."),
    ("email", "EMAIL OPS(AT)EXAMPLE.COM OR ops@example.com"),
    ("short digits", "RWY 09/27 TORA 1500M"),
    ("a number of nine digits", "REF 012345678 ENDS"),
    ("a digit run inside a word", "CRANE-ID-0123456789012"),
]


@pytest.mark.parametrize(("name", "text", "redacted", "mapping"), PHONE_SHAPES, ids=[row[0] for row in PHONE_SHAPES])
def test_a_phone_shape_is_redacted_and_round_trips(name, text, redacted, mapping):
    out, found = redact_phones(text)
    assert out == redacted
    assert found == mapping
    assert rehydrate(out, found) == text


@pytest.mark.parametrize(("name", "text"), NOT_PHONES, ids=[row[0] for row in NOT_PHONES])
def test_a_phone_like_non_phone_is_untouched(name, text):
    assert redact_phones(text) == (text, {})


def test_placeholders_run_in_order_of_appearance():
    text = "A 01632 960003, B 07700 900123, C +44 1632 960002."
    out, found = redact_phones(text)
    assert out == "A <PHONE_1>, B <PHONE_2>, C <PHONE_3>."
    assert list(found) == ["<PHONE_1>", "<PHONE_2>", "<PHONE_3>"]
    assert list(found.values()) == ["01632 960003", "07700 900123", "+44 1632 960002"]


def test_text_outside_a_match_is_unchanged_byte_for_byte():
    before = "  LINE ONE\t(\u00e9\u2013 \u4e2d)\r\nCTC 07700 900123\r\n\r\n  AFTER  \x0b"
    after, _ = redact_phones(before)
    assert after == before.replace("07700 900123", "<PHONE_1>")


def test_a_phone_in_a_mixed_notam_text_is_the_only_thing_that_changes():
    text = "UAS OPR WI 0.5NM RADIUS OF 534732N 0000459W, SFC-700FT AGL. 0800-1800 DAILY.\nFOR INFO CTC 01632 960001 ON 119.130. AR-2026-10264/01."
    out, found = redact_phones(text)
    assert out == text.replace("01632 960001", "<PHONE_1>")
    assert found == {"<PHONE_1>": "01632 960001"}


# --- E4: the same number twice, two different numbers ---------------------------------------------


def test_the_same_number_twice_gets_one_placeholder():
    out, found = redact_phones("CTC 07700 900123 OR AGAIN 07700 900123.")
    assert out == "CTC <PHONE_1> OR AGAIN <PHONE_1>."
    assert found == {"<PHONE_1>": "07700 900123"}


def test_the_same_number_written_two_ways_gets_one_placeholder():
    out, found = redact_phones("CTC 07700 900123 OR 07700900123.")
    assert out == "CTC <PHONE_1> OR <PHONE_1>."
    assert found == {"<PHONE_1>": "07700 900123"}  # the first spelling is kept


def test_two_different_numbers_get_two_placeholders_and_the_map_round_trips():
    text = "CTC 07700 900123 OR 01632 960001. THEN 07700 900123 AGAIN"
    out, found = redact_phones(text)
    assert out == "CTC <PHONE_1> OR <PHONE_2>. THEN <PHONE_1> AGAIN"
    assert found == {"<PHONE_1>": "07700 900123", "<PHONE_2>": "01632 960001"}
    assert rehydrate(out, found) == text


def test_the_same_number_in_a_different_spelling_round_trips_to_the_first_spelling():
    out, found = redact_phones("07700 900123 / 07700900123")
    assert rehydrate(out, found) == "07700 900123 / 07700 900123"


# --- boundaries ---------------------------------------------------------------------------------


def test_empty_string():
    assert redact_phones("") == ("", {})
    assert rehydrate("", {}) == ""


def test_text_with_no_phones_comes_back_unchanged_with_an_empty_map():
    text = "RESTRICTED AREA ACTIVATED. WI 2NM RADIUS OF 520528N 0000757E.\n"
    out, found = redact_phones(text)
    assert out == text
    assert found == {}


def test_redacting_redacted_text_changes_nothing():
    once, found = redact_phones("CTC 07700 900123 OR 01632 960001")
    again, second = redact_phones(once)
    assert again == once
    assert second == {}
    assert found  # the first pass did find something


def test_unicode_and_crlf_are_kept():
    text = "Zürich \u2013 straße \U0001f6e9\r\nCTC 07700 900123\r\n"
    out, found = redact_phones(text)
    assert out == "Zürich \u2013 straße \U0001f6e9\r\nCTC <PHONE_1>\r\n"
    assert rehydrate(out, found) == text


def test_new_placeholders_continue_after_ones_already_in_the_text():
    out, found = redact_phones("<PHONE_1> OR 01632 960001")
    assert out == "<PHONE_1> OR <PHONE_2>"
    assert found == {"<PHONE_2>": "01632 960001"}


@pytest.mark.parametrize("text", ["07700 900123", "+44 1632 960002", "0044 7700 900123", "01632\n960003"])
def test_a_number_never_survives_in_the_redacted_text(text):
    out, _ = redact_phones(f"CTC {text} NOW")
    assert not re.search(r"\d{5}", out)


# --- rehydrate ----------------------------------------------------------------------------------


def test_each_notam_rehydrates_with_its_own_map():
    """Two NOTAMs whose texts both hold <PHONE_1>, with different numbers: never the other's."""
    first, first_map = redact_phones("FOR INFO 07700 900123.")
    second, second_map = redact_phones("FOR INFO 01632 960001.")
    assert first == second == "FOR INFO <PHONE_1>."
    assert first_map != second_map
    assert rehydrate(first, first_map) == "FOR INFO 07700 900123."
    assert rehydrate(second, second_map) == "FOR INFO 01632 960001."
    assert "960001" not in rehydrate(first, first_map)
    assert "900123" not in rehydrate(second, second_map)


def test_rehydrate_leaves_an_unknown_placeholder_alone():
    assert rehydrate("CTC <PHONE_1> OR <PHONE_2>", {"<PHONE_1>": "07700 900123"}) == "CTC 07700 900123 OR <PHONE_2>"


def test_rehydrate_makes_one_pass_so_a_restored_number_is_not_searched_again():
    assert rehydrate("<PHONE_1>", {"<PHONE_1>": "<PHONE_2>", "<PHONE_2>": "X"}) == "<PHONE_2>"


def test_rehydrate_tells_phone_1_from_phone_10():
    mapping = {f"<PHONE_{n}>": f"number {n}" for n in range(1, 11)}
    assert rehydrate("<PHONE_10> <PHONE_1>", mapping) == "number 10 number 1"


# --- e_hash -------------------------------------------------------------------------------------


def test_e_hash_ignores_line_breaks_and_spacing():
    one = e_hash("WI 2NM RADIUS OF\n520528N 0000757E.  APPLIES TO ALL ACFT.")
    two = e_hash("  WI 2NM RADIUS OF 520528N 0000757E. APPLIES\r\n\tTO ALL ACFT.\n")
    assert one == two
    assert re.fullmatch(r"[0-9a-f]{64}", one)


def test_e_hash_changes_with_one_character():
    assert e_hash("WI 2NM RADIUS OF 520528N 0000757E.") != e_hash("WI 3NM RADIUS OF 520528N 0000757E.")
    assert e_hash("A B") != e_hash("AB")  # a space between words is part of the text


def test_e_hash_is_taken_from_the_raw_text_so_it_differs_from_the_redacted_text():
    raw = "CTC 07700 900123"
    redacted, _ = redact_phones(raw)
    assert e_hash(raw) != e_hash(redacted)
