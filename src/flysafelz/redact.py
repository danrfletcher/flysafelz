"""Phone redaction for NOTAM text (G1), and the hash that identifies an E text.

A real phone number never goes to a model, a store, a log or a report. `redact_phones` swaps each one for
`<PHONE_1>`, `<PHONE_2>` and so on, in order of appearance, and returns the map back to the real numbers.
`rehydrate` puts them back with the map of the same NOTAM. Placeholders are numbered per NOTAM, so every
NOTAM has its own `<PHONE_1>`: rehydrate a line with its own NOTAM's map and never another's.

What counts as a phone number (found by shape, not by a word like "PHONE"):

- a UK number: a leading 0 and ten to twelve digits in all (`07700 900123`, `01632960001`, `077009 00124`).
  The first group is at least three digits (`0` and two more, written together), so a list of two-digit days
  (`OCT 01 02 03 06 07 08 0900-1700`, 28 of them in item D of the four PIBs) is not read as a number. Real NOTAMs
  sometimes carry a twelfth digit by typo (`07700 9001234`), so that is redacted too;
- an international number: `+` then a country code and the number, nine to fifteen digits in all
  (`+44 1632 960002`, `+47 000 00 000`, `+44 (0)7700 900123`), or `00` then twelve to fifteen
  (`0044 7700 900123`). No date starts with `00`, so this one takes any grouping;
- in `+44 M01632 960005` the `M` is a mobile tag: `+44 M` stays and the digits after it are the number.

Digit groups may be split by spaces or by one line break (`01632 \\n960003`). A slash or a hyphen ends a number,
so `077009 00125 / \\n01632 960004` is two. Frequencies (`119.130`), coordinates (`534732N 0000459W`), heights,
NOTAM ids, `AR-2026-10264/01`, date-time groups (`2610311200`), times (`0800-1800`) and day-of-month lists are not
matched. Emails are kept. Run over items D, F and G of the 4,608 NOTAMs of the four archive PIBs, the redactor changes
none of them (tests/unit/test_redact_corpus.py).

Not covered: a number with no leading 0, + or 00 (`353 00 000 0000`, written that way once in the four PIBs). Without a
prefix it cannot be told from a list of times or dates. The one known over-match is a run of space-separated
4-digit times that starts with 0 and makes ten or more digits (`0900 1700 1800`): it hides a schedule from
the redacted text, never a number. None is in an item D, F or G of the four PIBs. The leak check in
scripts/build_reader_items.py looks for ten- and eleven-digit runs that survive in the sheet. It is blunter than the
redactor on purpose, so a day list that is (rightly) left alone would trip it: it fails loud, in the safe direction.
None of the 40 selected items has one.

The digit counts are written in the patterns below, not as Python numbers.
"""

import hashlib
import re

_PLACEHOLDER = re.compile(r"<PHONE_(\d+)>")

# Between two digit groups: spaces, or spaces around a single line break.
_GAP = r"(?:[^\S\r\n]+|[^\S\r\n]*\r?\n[^\S\r\n]*)"
_PHONE = re.compile(
    r"(?<![\d.])"
    rf"(?:\+\d{{1,3}}(?:[^\S\r\n]*\(0\))?(?:{_GAP}?\d){{8,12}}?"  # +CC, an optional (0), then the number: 9 to 15 digits in all
    rf"|00(?:{_GAP}?\d){{10,13}}?"  # 00 for +: 12 to 15 digits in all
    rf"|0\d{{2}}(?:{_GAP}?\d){{7,9}}?)"  # UK: 10 to 12 digits in all (a twelfth digit is a typo seen in real NOTAMs), the first group at least 3 digits
    r"(?!\d)"
)
_NOT_DIGIT = re.compile(r"\D")


def _same_number(written: str) -> str:
    """The digits of a number as written, with a leading + kept: `01632 960003` and `01632960003` are one number."""
    return ("+" if written.startswith("+") else "") + _NOT_DIGIT.sub("", written)


def redact_phones(text: str) -> tuple[str, dict[str, str]]:
    """(text with every phone number swapped for <PHONE_n>, {placeholder: number as written}).

    The same number twice gets one placeholder. Text outside a match is returned byte for byte. Text with no
    number comes back unchanged with an empty map, so redacting redacted text changes nothing. If the text already
    holds placeholders, new ones are numbered after the highest.
    """
    taken = max((int(n) for n in _PLACEHOLDER.findall(text)), default=0)
    placeholder_of: dict[str, str] = {}
    mapping: dict[str, str] = {}

    def swap(match: re.Match[str]) -> str:
        written = match.group(0)
        key = _same_number(written)
        if key not in placeholder_of:
            placeholder = f"<PHONE_{taken + len(placeholder_of) + 1}>"
            placeholder_of[key] = placeholder
            mapping[placeholder] = written
        return placeholder_of[key]

    return _PHONE.sub(swap, text), mapping


def rehydrate(text: str, mapping: dict[str, str]) -> str:
    """Put the real numbers back. A placeholder that is not in the map is left as it is.

    One pass over the text, so a restored number is never searched again.
    """
    return _PLACEHOLDER.sub(lambda match: mapping.get(match.group(0), match.group(0)), text)


def e_hash(text: str) -> str:
    """sha256 of an E text as published: runs of whitespace become one space and the ends are stripped.

    Two texts that differ only in line breaks or spacing hash the same; any other change gives another hash.
    Always taken from the raw text, never the redacted one, so it identifies the NOTAM text a label was written for.
    """
    return hashlib.sha256(" ".join(text.split()).encode("utf-8")).hexdigest()
