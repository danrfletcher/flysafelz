"""UK airspace designators (EGD323A, EGR106, EGP108) from free text.

The designator is EG, one of D (danger), R (restricted) or P (prohibited),
three digits and an optional letter. NOTAM text writes it many ways (D323A,
EGD 323A, EG D323A, egd323a), and all of them become EGD323A.

Matching is exact: EGD496 and EGD496A are different areas. A word boundary
sits before the designator and after its suffix, so a reference such as
"AIP SUP 108/2026" or "D4960" never produces a designator.
"""

import re

_DESIGNATOR = r"(?:E?G?\s?)?([DRP])\s?(\d{3})([A-Z]?)"
_FIND = re.compile(rf"\b{_DESIGNATOR}\b", re.IGNORECASE)
_WHOLE = re.compile(rf"\s*{_DESIGNATOR}\s*", re.IGNORECASE)


def _canonical(kind: str, number: str, suffix: str) -> str:
    return f"EG{kind}{number}{suffix}".upper()


def normalise_designator(s: str) -> str | None:
    """EG<D|R|P><nnn><X> for a string that is exactly one designator, else None."""
    match = _WHOLE.fullmatch(s)
    return _canonical(*match.groups()) if match else None


def find_designators(text: str) -> list[str]:
    """Every designator in the text, normalised, in order of first appearance, without repeats."""
    found = (_canonical(*m.groups()) for m in _FIND.finditer(text))
    return list(dict.fromkeys(found))
