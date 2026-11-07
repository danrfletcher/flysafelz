"""AIP supplement (SUP) references from free text, found by code and never by a model (F4).

A NOTAM names the supplement that holds the details of a temporary danger area, for example
"AIP SUP 108/2026". NOTAM text writes it several ways (SUP 108/2026, aip sup 108/26, SUPPLEMENT 108/2026,
a line break between "AIP SUP" and the number), and all of them become "AIP SUP <n>/<yyyy>" with the digits
kept as written ("008/2026" stays "008/2026", "35/26" stays "35/26").

A word boundary sits before the reference and after its year, and the number needs a year, so "SUP 108",
"XSUP 108/2026", "SUP 108/20261", "SUPERVISOR" and "MIL SUP CHECKLIST 001/2022" give nothing. A misspelling such
as "SUPPELEMENT" is not matched either: that is a known miss, not a bug. Only the digits 0-9 count, so a
reference in other scripts' digits is not matched. A non-breaking space counts as whitespace.

Standard library only: this module imports no reader and no model client.
"""

import re

_SUP = re.compile(r"\b(?:AIP\s+)?SUP(?:PLEMENT)?\s+([0-9]{1,3})/([0-9]{2,4})\b", re.IGNORECASE)

_PREFIX = "details in "


def _canonical(number: str, year: str) -> str:
    return f"AIP SUP {number}/{year}"


def find_sup_references(text: str) -> list[str]:
    """Every SUP reference in the text as "AIP SUP <n>/<yyyy>", in order of first appearance, without repeats."""
    found = (_canonical(*m.groups()) for m in _SUP.finditer(text))
    return list(dict.fromkeys(found))


def sup_detail(refs: list[str]) -> str | None:
    """The wording that names the supplements: "details in AIP SUP 108/2026" (", " between several); None for none.

    The one place this wording is written. Conditions that name the SUP take it from here.
    """
    return f"{_PREFIX}{', '.join(refs)}" if refs else None
