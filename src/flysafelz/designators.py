"""UK airspace designators (EGD323A, EGR106, EGP108) from free text.

The designator is EG, one of D (danger), R (restricted) or P (prohibited),
three digits and a suffix of up to two letters (none, A, or AB). NOTAM text
writes it many ways (D323A, EGD 323A, EG D323A, egd323a), and all of them
become EGD323A.

Matching is exact: EGD496 and EGD496A are different areas, and so are EGD323A
and EGD323AB. A word boundary sits before the designator and after its suffix,
so a reference such as "AIP SUP 108/2026" or "D4960" never produces a
designator, and neither does a three-letter suffix (EGD323ABC).
"""

import re

_DESIGNATOR = r"(?:E?G?\s?)?([DRP])\s?(\d{3})([A-Z]{0,2})"
_FIND = re.compile(
    r"\b(?:EG\s?(?P<k1>[DRP])\s?|(?P<k2>[DRP])(?:\s(?=\d{3}[A-Z]))?)(?P<n>\d{3})(?P<s>[A-Z]{0,2})\b(?P<more>(?:/[A-Z])+(?![A-Z0-9]))?",
    re.IGNORECASE,
)
_WHOLE = re.compile(rf"\s*{_DESIGNATOR}\s*", re.IGNORECASE)


def _canonical(kind: str, number: str, suffix: str) -> str:
    return f"EG{kind}{number}{suffix}".upper()


def normalise_designator(s: str) -> str | None:
    """EG<D|R|P><nnn><suffix of 0-2 letters> for a string that is exactly one designator, else None."""
    match = _WHOLE.fullmatch(s)
    return _canonical(*match.groups()) if match else None


def find_designators(text: str) -> list[str]:
    """Every designator in the text, normalised, in order of first appearance, without repeats."""
    found: list[str] = []
    for m in _FIND.finditer(text):
        kind, number, suffix = m.group("k1") or m.group("k2"), m.group("n"), m.group("s")
        found.append(_canonical(kind, number, suffix))
        if m.group("more") and len(suffix) == 1:  # "D323A/B": the letters after each slash are more suffixes
            found.extend(_canonical(kind, number, letter) for letter in m.group("more").split("/")[1:])
    return list(dict.fromkeys(found))
