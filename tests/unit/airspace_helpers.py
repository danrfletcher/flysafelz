"""Shared builders for the airspace tests: small squares around a fixed LZ, limits, readings, activations."""

import math
from datetime import UTC, datetime
from typing import Any

from flysafelz.airspace import Activation, AirspaceAssessment, Finding, TimeSpan, assess_airspace
from flysafelz.clients.openaip import Airspace
from flysafelz.contracts import (
    Category,
    Circle,
    Contact,
    ContactKind,
    Datum,
    Limit,
    LimitUnit,
    NotamReading,
    ProhibitiveReason,
)
from flysafelz.geo import Shape, distance_nm
from flysafelz.rules import RULES, Rules

LZ = (53.802797, -0.086644)  # Blue Mill
ELEVATION_FT = 30.0
STANDARD_HPA = 1013.25
ARC_MINUTES = 60  # minutes of latitude in a degree: one per nautical mile


def utc(hour: int, minute: int = 0, day: int = 31, month: int = 10) -> datetime:
    return datetime(2026, month, day, hour, minute, tzinfo=UTC)


SEGMENT = TimeSpan(start=utc(9), end=utc(10))  # a Saturday morning; Blue Mill 31 Oct 2026

SFC = Limit(value=0, unit=LimitUnit.FT, datum=Datum.SFC)


def amsl(feet: float) -> Limit:
    return Limit(value=feet, unit=LimitUnit.FT, datum=Datum.AMSL)


def agl(feet: float) -> Limit:
    return Limit(value=feet, unit=LimitUnit.FT, datum=Datum.AGL)


def fl(level: float) -> Limit:
    return Limit(value=level, unit=LimitUnit.FL, datum=Datum.AMSL)


def offset(lat: float, lon: float, bearing: float, nm: float) -> tuple[float, float]:
    """The point nm nautical miles from (lat, lon) on a bearing, by the same flat map geo.py uses (checked below)."""
    theta = math.radians(bearing)
    dlat = nm * math.cos(theta) / ARC_MINUTES
    dlon = nm * math.sin(theta) / ARC_MINUTES / math.cos(math.radians(lat))
    point = (lat + dlat, lon + dlon)
    assert abs(distance_nm(lat, lon, *point) - nm) < 0.01 * nm  # the helper itself must be right to 1 %
    return point


def square(lat: float, lon: float, half_nm: float) -> Shape:
    """A square, half_nm from its centre to each side, as a (lat, lon) Shape."""
    dlat = half_nm / ARC_MINUTES
    dlon = dlat / math.cos(math.radians(lat))
    ring = [(lat - dlat, lon - dlon), (lat - dlat, lon + dlon), (lat + dlat, lon + dlon), (lat + dlat, lon - dlon)]
    return [[[*ring, ring[0]]]]


def around_lz(half_nm: float = 0.5) -> Shape:
    return square(*LZ, half_nm)


def volume(
    type_: str = "ctr",
    *,
    lower: Limit = SFC,
    cls: str | None = None,
    name: str | None = None,
    shape: Shape | None = None,
    by_notam: bool = False,
    hours: str | None = None,
    hours_readable: bool = True,
    designator: str | None = None,
) -> Airspace:
    return Airspace(
        id=f"id-{name or type_}",
        name=name or f"TEST {type_.upper()}",
        type=type_,
        icao_class=cls,
        lower=lower,
        by_notam=by_notam,
        hours=hours,
        hours_readable=hours_readable,
        designator=designator,
        shape=shape if shape is not None else around_lz(),
    )


def blue_mill_tda(**kwargs):
    """EGD496A as a polygon around Blue Mill whose south-west edge is about 95 m from the LZ."""
    lat, lon = LZ
    ring = [(lat - 0.00085, lon - 0.00145), (lat - 0.00085, lon + 0.08), (lat + 0.08, lon + 0.08), (lat + 0.08, lon - 0.00145)]
    shape = [[[*ring, ring[0]]]]
    args: dict[str, Any] = {"type_": "danger", "name": "EGD496A", "designator": "EGD496A", "by_notam": True, "shape": shape}
    args.update(kwargs)
    return volume(**args)


def assess(
    areas: list[Airspace],
    *,
    segment: TimeSpan = SEGMENT,
    elevation: float = ELEVATION_FT,
    pressure: float | None = STANDARD_HPA,
    activations: list[Activation] | None = None,
    lz: tuple[float, float] = LZ,
    rules: Rules = RULES,
) -> AirspaceAssessment:
    return assess_airspace(areas, lz, segment, site_elevation_ft=elevation, pressure_msl=pressure, activations=activations or [], rules=rules)


def only(result: AirspaceAssessment) -> Finding:
    assert len(result.findings) == 1, [(f.area, f.outcome) for f in result.findings]
    return result.findings[0]


def with_rules(**values: object) -> Rules:
    """A copy of the shipped rules with the named rules' values replaced."""
    changed = {name: getattr(RULES, name).model_copy(update={"value": value}) for name, value in values.items()}
    return RULES.model_copy(update=changed)


def reading(
    *,
    contact: ContactKind = ContactKind.PHONE,
    value: str | None = "Flylogix 07700 900123",
    category: Category = Category.COOPERATIVE,
    reason: ProhibitiveReason = ProhibitiveReason.NONE,
    quote: str | None = None,
    activity: str = "UAS BVLOS ops",
    notam_id: str = "J3857/26",
) -> NotamReading:
    """A hand-written reading of J3857/26 (TDA EGD496A, SFC to 1,300 ft AMSL, 0800-1800). The number is fiction-range."""
    return NotamReading(
        id=notam_id,
        category=category,
        activity=activity,
        floor=SFC,
        ceiling=amsl(1300),
        schedule="0800-1800",
        contact=Contact(kind=contact, value=value),
        activates_designators=["EGD496A"],
        prohibitive_reason=reason,
        prohibitive_quote=quote,
    )


E_TEXT = "TEMPORARY DANGER AREA EGD496A ACTIVATED. UAS BVLOS OPS. SFC-1300FT AMSL. 0800-1800."


def activation(
    designator: str = "EGD496A",
    *,
    read: NotamReading | None = None,
    start: datetime | None = None,
    end: datetime | None = None,
    notam_id: str = "J3857/26",
    circle: Circle | None = None,
    e_text: str = E_TEXT,
    contact: ContactKind | None = None,
    sup: list[str] | None = None,
) -> Activation:
    """An Activation. `contact` is what code found in the original item E (PHONE, EMAIL or NONE).

    Left out, it follows the reading's own contact when that is a phone or email with a value, else NONE, so a
    hand-written reading and its NOTAM agree unless a test says otherwise. `sup` is the SUP references code found.
    """
    if contact is None:
        has = read is not None and read.contact.kind in (ContactKind.PHONE, ContactKind.EMAIL) and bool((read.contact.value or "").strip())
        contact = read.contact.kind if has and read is not None else ContactKind.NONE
    return Activation(
        designator=designator,
        notam_id=notam_id,
        reading=read,
        e_text=e_text,
        start=start or utc(8),
        end=end or utc(18),
        circle=circle,
        item_e_contact=contact,
        sup_references=sup or [],
    )
