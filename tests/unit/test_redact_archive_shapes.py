"""E7/F5: the phone shapes the real NOTAM archive holds, kept as tests with fiction-range digits. Pure: no fixtures, no files.

scripts/phone_archive_scan.py ran the detector over the distinct NOTAMs of 123 daily archive PIBs (2026-07-01 to 2026-10-31).
Each row below is a shape that run found (spacing, `+44(0)`, `0044`, a full stop after the number, run-together digits, odd
groupings, a line break inside the number), with every digit swapped for one of Ofcom's drama-range numbers: 07700 900xxx,
01632 960xxx, 0113 496 0xxx, 020 7946 0xxx. No real number is in this file (F5). The non-match rows may be real because
coordinates, frequencies and AR references are not PII.

The shapes the detector gets wrong are the last two tables, each marked xfail(strict=True): they document the miss (a number it
leaves in) and the over-match (a coordinate it hides) without changing redact.py, and fail loudly the day the detector learns the
shape, so the row can move to the first or second table. Both are reported in the PR with the NOTAM ids the archive run found.
"""

import re

import pytest

from flysafelz.redact import redact_phones, rehydrate

# (name, text, redacted, {placeholder: number as written})
ARCHIVE_PHONE_SHAPES = [
    # sentence-final full stop, the commonest archive shape
    ("mobile, 5+6, full stop", "FOR INFO CTC 07700 900123.", "FOR INFO CTC <PHONE_1>.", {"<PHONE_1>": "07700 900123"}),
    ("landline 01632, 5+6, full stop", "CTC 01632 960124.", "CTC <PHONE_1>.", {"<PHONE_1>": "01632 960124"}),
    ("landline 0113, 4+3+4, full stop", "CTC 0113 496 0125.", "CTC <PHONE_1>.", {"<PHONE_1>": "0113 496 0125"}),
    ("landline 0113, 4+3+4, no full stop", "CTC 0113 496 0126 DURING OPS", "CTC <PHONE_1> DURING OPS", {"<PHONE_1>": "0113 496 0126"}),
    ("landline 020, 3+4+4, full stop", "CTC 020 7946 0127.", "CTC <PHONE_1>.", {"<PHONE_1>": "020 7946 0127"}),
    ("landline 020, 3+4+4, no full stop", "CTC 020 7946 0128 DURING OPS", "CTC <PHONE_1> DURING OPS", {"<PHONE_1>": "020 7946 0128"}),
    # run-together
    ("mobile, run-together", "CTC 07700900456", "CTC <PHONE_1>", {"<PHONE_1>": "07700900456"}),
    ("mobile, run-together, full stop", "CTC 07700900457.", "CTC <PHONE_1>.", {"<PHONE_1>": "07700900457"}),
    ("landline 01632, run-together, full stop", "CTC 01632960129.", "CTC <PHONE_1>.", {"<PHONE_1>": "01632960129"}),
    ("landline 0113, run-together", "CTC 01134960130 NOW", "CTC <PHONE_1> NOW", {"<PHONE_1>": "01134960130"}),
    ("landline 020, run-together", "CTC 02079460131 NOW", "CTC <PHONE_1> NOW", {"<PHONE_1>": "02079460131"}),
    # odd groupings
    ("mobile, odd grouping 6+5", "TEL 077009 00789 NOW", "TEL <PHONE_1> NOW", {"<PHONE_1>": "077009 00789"}),
    ("mobile, odd grouping 6+5, full stop", "TEL 077009 00790.", "TEL <PHONE_1>.", {"<PHONE_1>": "077009 00790"}),
    ("landline 01632, odd grouping 4+7, full stop", "TEL 0163 2960132.", "TEL <PHONE_1>.", {"<PHONE_1>": "0163 2960132"}),
    ("landline 01632, 5+3+3, full stop", "TEL 01632 960 133.", "TEL <PHONE_1>.", {"<PHONE_1>": "01632 960 133"}),
    # a line break inside the number
    ("landline 01632, line break, full stop", "TEL 01632\n960134.", "TEL <PHONE_1>.", {"<PHONE_1>": "01632\n960134"}),
    ("landline 01632, space and line break", "TEL 01632 \n960135 NOW", "TEL <PHONE_1> NOW", {"<PHONE_1>": "01632 \n960135"}),
    ("landline 01632, 5+3 line break 3, full stop", "TEL 01632 960\n136.", "TEL <PHONE_1>.", {"<PHONE_1>": "01632 960\n136"}),
    # +44 and 0044
    ("+44(0) with no space before (0)", "CTC +44(0)7700 900456.", "CTC <PHONE_1>.", {"<PHONE_1>": "+44(0)7700 900456"}),
    ("+44 (0) with a space before (0)", "CTC +44 (0)7700 900137", "CTC <PHONE_1>", {"<PHONE_1>": "+44 (0)7700 900137"}),
    ("0044, spaced", "CTC 0044 1632 960789", "CTC <PHONE_1>", {"<PHONE_1>": "0044 1632 960789"}),
    ("0044, spaced, full stop", "CTC 0044 1632 960138.", "CTC <PHONE_1>.", {"<PHONE_1>": "0044 1632 960138"}),
    ("+44 landline 01632", "CTC +44 1632 960139", "CTC <PHONE_1>", {"<PHONE_1>": "+44 1632 960139"}),
    ("+44 landline 01632, line break after the country code", "CTC +44\n1632 960140", "CTC <PHONE_1>", {"<PHONE_1>": "+44\n1632 960140"}),
    ("+44 landline 01632, line break inside", "CTC +44 1632 \n960141", "CTC <PHONE_1>", {"<PHONE_1>": "+44 1632 \n960141"}),
    ("+44 landline 020, 2+4+4", "CTC +44 20 7946 0142", "CTC <PHONE_1>", {"<PHONE_1>": "+44 20 7946 0142"}),
    ("+44 landline 020, 2+4+4, full stop", "CTC +44 20 7946 0143.", "CTC <PHONE_1>.", {"<PHONE_1>": "+44 20 7946 0143"}),
    ("+44 landline 0113, 3+3+4", "CTC +44 113 496 0144", "CTC <PHONE_1>", {"<PHONE_1>": "+44 113 496 0144"}),
    # two numbers, one NOTAM
    (
        "two numbers, a mobile then a landline",
        "INFO 07700 900145 BOOKING 01632 960146.",
        "INFO <PHONE_1> BOOKING <PHONE_2>.",
        {"<PHONE_1>": "07700 900145", "<PHONE_2>": "01632 960146"},
    ),
]

# Shapes that must come through unchanged. Coordinates, frequencies and AR references are not PII, so these may be real.
ARCHIVE_NON_PHONES = [
    ("dms coordinate", "534732N"),
    ("dms coordinate pair", "PSN 534732N 0000459W."),
    ("coordinate list, a 4-digit number then a coordinate on the next line", "WADDINGTON 1123 \n534732N 0000459W METHRINGHAM"),
    ("coordinate with decimal seconds", "ERECTED AT 512345.6N 0001234.5E."),
    ("frequency", "128.925"),
    ("frequency in a sentence", "CTC TOWER ON 128.925 OR 121.500."),
    ("ar reference", "AR-2026-10264/01"),
    ("ar reference in a sentence", "FOR INFO SEE AR-2026-10264/01."),
    ("obstacle id", "OBSTACLE ID: 20260000123 \nTYPE: CRANE"),
    ("hours table, day then times", "AD OPR HR:\nMON      0600-2200\n05      0800-1700\n"),
    ("hours table, times across a line break", "RADAR SERVICE HOURS\n05 0600-1900\n06 0800-1700\n"),
]

# Shapes the detector misses today. The digits are fiction-range (or all zeros for the non-UK one). Reported in the PR.
MISS_REASON = "detector miss, reported in PR"
ARCHIVE_MISSES = [
    ("0044 then (0) in brackets", "0044 (0)1632 960147"),
    ("00 44 then (0) in brackets", "00 44 (0)1632 960148"),
    ("+44 in brackets", "(+44)1632 960149"),
    ("area code in brackets", "(01632) 960150"),
    ("no leading 0, + or 00 (an Irish number)", "353 00 000 0000"),
    ("mobile with a thirteenth digit", "07700 90012345"),
]

# A shape the detector changes that holds no number: a 4-digit figure ending a line, then a coordinate on the next (H5914/26 has 11).
OVER_MATCH_REASON = "detector over-match, reported in PR"
ARCHIVE_OVER_MATCHES = [
    ("figure, line break, latitude", "WHISBY       0910 \n534500N 0001234W  VCY OF SCUNTHORPE"),
]

# What a fiction-range number looks like once written plainly: 0 then the national number, no spaces, no (0).
FICTION_PREFIXES = ("07700900", "01632960", "01134960", "02079460")
ALL_ZEROS = re.compile(r"\+?\d{2,3}(?: ?0+)+")  # a placeholder such as `353 00 000 0000`: a country code, then nothing but zeros


def plain(number: str) -> str:
    """The number as bare digits: +44 and 0044 become 0, (0) is dropped, spaces, line breaks and brackets are removed."""
    digits = re.sub(r"[\s()]+", "", number.replace("(0)", ""))
    if digits.startswith("+44"):
        return "0" + digits[3:]
    if digits.startswith("0044"):
        return "0" + digits[4:]
    return digits


def is_fiction(number: str) -> bool:
    return plain(number).startswith(FICTION_PREFIXES) or ALL_ZEROS.fullmatch(number) is not None


@pytest.mark.parametrize(("name", "text", "redacted", "mapping"), ARCHIVE_PHONE_SHAPES, ids=[row[0] for row in ARCHIVE_PHONE_SHAPES])
def test_an_archive_phone_shape_is_redacted_and_round_trips(name, text, redacted, mapping):
    out, found = redact_phones(text)
    assert out == redacted
    assert found == mapping
    assert rehydrate(out, found) == text


@pytest.mark.parametrize(("name", "text"), ARCHIVE_NON_PHONES, ids=[row[0] for row in ARCHIVE_NON_PHONES])
def test_an_archive_coordinate_frequency_or_reference_is_untouched(name, text):
    assert redact_phones(text) == (text, {})


@pytest.mark.parametrize(
    ("name", "number"),
    [pytest.param(name, number, id=name, marks=pytest.mark.xfail(strict=True, reason=MISS_REASON)) for name, number in ARCHIVE_MISSES],
)
def test_an_archive_phone_shape_the_detector_misses(name, number):
    text = f"CTC {number}."
    out, found = redact_phones(text)
    assert found, "the detector found no number"
    assert out == "CTC <PHONE_1>."
    assert rehydrate(out, found) == text


@pytest.mark.parametrize(
    ("name", "text"),
    [pytest.param(name, text, id=name, marks=pytest.mark.xfail(strict=True, reason=OVER_MATCH_REASON)) for name, text in ARCHIVE_OVER_MATCHES],
)
def test_an_archive_coordinate_the_detector_wrongly_redacts(name, text):
    assert redact_phones(text) == (text, {})


# --- F5: every number in this file is fiction-range ------------------------------------------------------------


def test_every_sample_in_this_file_is_fiction_range():
    """Runs the detector over each sample and normalises each match: +44 and 0044 become 0, (0) goes, spaces go."""
    samples = [row[1] for row in ARCHIVE_PHONE_SHAPES] + [row[1] for row in ARCHIVE_NON_PHONES] + [f"CTC {number}." for _name, number in ARCHIVE_MISSES]
    matched = [number for text in samples for number in redact_phones(text)[1].values()]
    assert len(matched) >= len(ARCHIVE_PHONE_SHAPES)  # the guard looked at real matches, not at nothing
    assert [number for number in matched if not is_fiction(number)] == []


def test_the_numbers_the_detector_misses_are_fiction_range_too():
    """The detector cannot guard these (it misses them), so the guard reads the number itself."""
    assert [number for _name, number in ARCHIVE_MISSES if not is_fiction(number)] == []


# A fiction number with one digit of its range moved (900 to 901, 960 to 961, 496 to 497, 7946 to 7947): outside Ofcom's ranges, so the
# guard must reject it. Built here by a replace, so that no such number is written out in this file.
OUTSIDE_THE_RANGES = [
    ("07700 900123", "900", "901"),
    ("+44(0)7700 900456", "900", "901"),
    ("0044 1632 960789", "960", "961"),
    ("01632960129", "960", "961"),
    ("0113 496 0125", "496", "497"),
    ("020 7946 0127", "7946", "7947"),
    ("+44 20 7946 0142", "7946", "7947"),
]


@pytest.mark.parametrize(("fiction", "part", "moved"), OUTSIDE_THE_RANGES, ids=[row[0] for row in OUTSIDE_THE_RANGES])
def test_the_guard_rejects_a_number_outside_the_fiction_ranges(fiction, part, moved):
    """Proves the guard can fail: the detector finds the moved number and the guard says it is not fiction-range."""
    outside = fiction.replace(part, moved)
    assert outside != fiction
    matched = list(redact_phones(f"CTC {outside}.")[1].values())
    assert matched == [outside]
    assert not is_fiction(outside)
    assert is_fiction(fiction)


@pytest.mark.parametrize("number", ["07700 900123", "+44(0)7700 900456", "0044 1632 960789", "01134960130", "020 7946 0127", "+44 20 7946 0142", "353 00 000 0000"])
def test_the_guard_accepts_the_fiction_ranges_in_every_written_form(number):
    assert is_fiction(number)
