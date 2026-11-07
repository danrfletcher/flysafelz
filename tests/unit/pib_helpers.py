"""Shared builders for the PIB tests: Parsed NOTAMs, PIB XML text, and a changed copy of the rules."""

from datetime import UTC, datetime
from pathlib import Path

from flysafelz.contracts import NotamType, ParsedNotam
from flysafelz.rules import RULES, Rules

NOW = datetime(2026, 10, 31, 7, 30, tzinfo=UTC)  # frozen: the same instant as the conftest clock
FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "pib"
TRIMMED = FIXTURES / "pib_trimmed.xml"

LZ = (53.802797, -0.086644)  # Blue Mill
SPAN = (datetime(2026, 10, 31, 7, 0, tzinfo=UTC), datetime(2026, 10, 31, 18, 0, tzinfo=UTC))
ISSUED = "2026-10-31T07:00:37+00:00"


def utc(day: int, hour: int, minute: int = 0, month: int = 10) -> datetime:
    return datetime(2026, month, day, hour, minute, tzinfo=UTC)


def parsed(notam_id: str = "H7171/26", **overrides: object) -> ParsedNotam:
    """A Parsed NOTAM at Blue Mill, active all day on 31 Oct, with a W subject. Override any field.

    `subject` is shorthand for the two subject letters of q_code.
    """
    series, rest = notam_id[0], notam_id[1:]
    number, year = rest.split("/")
    subject = overrides.pop("subject", "WU")
    fields: dict[str, object] = {
        "id": notam_id,
        "series": series,
        "number": int(number),
        "year": int(year),
        "type": NotamType.NEW,
        "nof": "EGGN",
        "fir": "EGTT",
        "q_code": f"Q{subject}LW",
        "traffic": "IV",
        "purpose": "BO",
        "scope": "W",
        "lower_fl": 0,
        "upper_fl": 8,
        "centre_lat": 53.8,
        "centre_lon": -0.0833,
        "radius_nm": 1.0,
        "item_a": "EGTT",
        "valid_from": utc(29, 8),
        "valid_to": utc(10, 18, month=11),
        "item_e": "UAS OPR WI 0.5NM. SFC-700FT AGL.",  # no coordinates: item E draws no area (G16), so the Q-circle is where it is
        "section": "WAR",
        "sections": ["WAR"],
    }
    fields.update(overrides)
    return ParsedNotam(**fields)


def with_rules(**values: object) -> Rules:
    """A copy of the shipped rules with the named rules' values replaced."""
    changed = {name: getattr(RULES, name).model_copy(update={"value": value}) for name, value in values.items()}
    return RULES.model_copy(update=changed)


# --- PIB XML text -------------------------------------------------------------------------------


def notam_xml(
    *,
    series: str = "H",
    number: int = 7171,
    year: int = 26,
    kind: str = "N",
    referred: str | None = None,
    section: str = "WAR",
    code23: str = "WU",
    code45: str = "LW",
    traffic: str = "IV",
    purpose: str = "BO",
    scope: str = "W",
    lower: str = "0",
    upper: str = "8",
    coordinates: str = "5348N00005W",
    radius: str | None = "1",
    item_a: tuple[str, ...] = ("EGTT",),
    start: str = "2610290800",
    end: str | None = "2611101800",
    estimation: bool = False,
    d: str | None = None,
    e: str | None = "UAS OPR WI 0.5NM RADIUS OF 534732N 0000459W (GARTON).",
    f: str | None = "SFC",
    g: str | None = "800FT AMSL",
    omit: tuple[str, ...] = (),
) -> str:
    """One Notam element in the real PIB's shape. `omit` leaves out the named elements."""

    def part(tag: str, value: str | None) -> str:
        return "" if value is None or tag in omit else f"<{tag}>{value}</{tag}>"

    referred_xml = ""
    if referred:  # "J2265/26"
        ref_series, ref_rest = referred[0], referred[1:]
        ref_number, ref_year = ref_rest.split("/")
        referred_xml = part("ReferredSeries", ref_series) + part("ReferredNumber", ref_number) + part("ReferredYear", ref_year)
    q_children = "".join(
        part(tag, value)
        for tag, value in (
            ("FIR", "EGTT"),
            ("Code23", code23),
            ("Code45", code45),
            ("Traffic", traffic),
            ("Purpose", purpose),
            ("Scope", scope),
            ("Lower", lower),
            ("Upper", upper),
        )
    )
    q_line = "" if "QLine" in omit else f"<QLine>{q_children}</QLine>"
    items_a = "".join(part("ItemA", a) for a in item_a)
    return (
        f'<Notam OriginalMessage="false" PIBSection="{section}">'
        + part("NOF", "EGGN")
        + part("Series", series)
        + part("Number", str(number))
        + part("Year", str(year))
        + part("Type", kind)
        + referred_xml
        + q_line
        + part("Coordinates", coordinates)
        + part("Radius", radius)
        + items_a
        + part("StartValidity", start)
        + part("EndValidity", end)
        + part("Estimation", "EST" if estimation else None)
        + part("ItemD", d)
        + part("ItemE", e)
        + part("ItemF", f)
        + part("ItemG", g)
        + part("Marker", "+")
        + "</Notam>"
    )


def pib_xml(*notams: str, issued: str | None = ISSUED, obsolete: tuple[str, ...] = ()) -> str:
    """A PIB with the given Notam elements in one Warnings section. issued=None leaves the Issued time out."""
    issued_xml = "" if issued is None else f"<Issued>{issued}</Issued>"
    return (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<Pib PIBType="F"><AreaPIBHeader><ShortSystemTitle>EAD</ShortSystemTitle>'
        f"{issued_xml}</AreaPIBHeader>"
        "<FIRSection><ICAO>EGTT</ICAO><Warnings>"
        f"<NotamList>{''.join(notams)}</NotamList>"
        f"<ObsoleteNotamList>{''.join(obsolete)}</ObsoleteNotamList>"
        "</Warnings></FIRSection></Pib>"
    )
