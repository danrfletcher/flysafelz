"""Builders for the NOTAM action tests: LZs, segments, parsed NOTAMs, readings, shapes.

Parsed NOTAMs come from PR-4's `parsed()` (tests/unit/pib_helpers.py); readings are written by
hand. No fixture file, no network, no .env.
"""

import math
from datetime import UTC, datetime

from pib_helpers import parsed as pib_parsed

from flysafelz.contracts import (
    AppliesTo,
    Category,
    Circle,
    Contact,
    ContactKind,
    Datum,
    Exemption,
    ExemptionCovers,
    Limit,
    LimitUnit,
    NotamReading,
    ParsedNotam,
    Polygon,
    ProhibitiveReason,
)
from flysafelz.geo import distance_nm
from flysafelz.notam_action import ActionSegment, LandingZone
from flysafelz.rules import RULES, Rules

BLUE_MILL = LandingZone(lat=53.802797, lon=-0.086644, elevation_ft=30.0)
LZ = BLUE_MILL
STANDARD_HPA = 1013.25
ARC_MINUTES = 60  # minutes of latitude in a degree: one per nautical mile


def utc(hour: int, minute: int = 0, day: int = 31, month: int = 10) -> datetime:
    return datetime(2026, month, day, hour, minute, tzinfo=UTC)


def segment(start: datetime | None = None, end: datetime | None = None, pressure: float | None = STANDARD_HPA) -> ActionSegment:
    """A Saturday morning at Blue Mill by default: 09:00-10:00Z on 31 Oct 2026."""
    return ActionSegment(start=start or utc(9), end=end or utc(10), pressure_msl=pressure)


def with_rules(**values: object) -> Rules:
    """A copy of the shipped rules with the named rules' values replaced."""
    changed = {name: getattr(RULES, name).model_copy(update={"value": value}) for name, value in values.items()}
    return RULES.model_copy(update=changed)


# --- limits and places ------------------------------------------------------------


SFC = Limit(value=0, unit=LimitUnit.FT, datum=Datum.SFC)


def amsl(feet: float) -> Limit:
    return Limit(value=feet, unit=LimitUnit.FT, datum=Datum.AMSL)


def agl(feet: float) -> Limit:
    return Limit(value=feet, unit=LimitUnit.FT, datum=Datum.AGL)


def fl(level: float) -> Limit:
    return Limit(value=level, unit=LimitUnit.FL, datum=Datum.AMSL)


def offset(lat: float, lon: float, bearing: float, nm: float) -> tuple[float, float]:
    """The point nm nautical miles from (lat, lon) on a bearing, on the same flat map geo.py uses (checked to 1 %)."""
    theta = math.radians(bearing)
    point = (lat + nm * math.cos(theta) / ARC_MINUTES, lon + nm * math.sin(theta) / ARC_MINUTES / math.cos(math.radians(lat)))
    assert abs(distance_nm(lat, lon, *point) - nm) <= 0.01 * nm
    return point


def circle_at(edge_nm: float, bearing: float = 180.0, radius_nm: float = 0.5) -> Circle:
    """A circle whose edge is edge_nm from the LZ on the bearing. 0 puts the LZ at its centre (over it)."""
    if edge_nm == 0:
        return Circle(lat=LZ.lat, lon=LZ.lon, radius_nm=radius_nm)
    lat, lon = offset(LZ.lat, LZ.lon, bearing, edge_nm + radius_nm)
    return Circle(lat=lat, lon=lon, radius_nm=radius_nm)


def square_at(edge_nm: float, bearing: float = 180.0, half_nm: float = 0.5) -> Polygon:
    """A square whose nearest side is edge_nm from the LZ on the bearing (0: the LZ is in its middle)."""
    centre = (LZ.lat, LZ.lon) if edge_nm == 0 else offset(LZ.lat, LZ.lon, bearing, edge_nm + half_nm)
    dlat = half_nm / ARC_MINUTES
    dlon = dlat / math.cos(math.radians(centre[0]))
    ring = [(centre[0] - dlat, centre[1] - dlon), (centre[0] - dlat, centre[1] + dlon), (centre[0] + dlat, centre[1] + dlon), (centre[0] + dlat, centre[1] - dlon)]
    return Polygon(points=ring)


# --- NOTAMs -----------------------------------------------------------------------


def notam(notam_id: str = "H7171/26", *, circle: Circle | None = None, **overrides: object) -> ParsedNotam:
    """PR-4's parsed NOTAM, active all day 31 Oct. circle sets its Q-circle (centre and radius)."""
    if circle is not None:
        overrides.update(centre_lat=circle.lat, centre_lon=circle.lon, radius_nm=circle.radius_nm)
    return pib_parsed(notam_id, **overrides)


def reading(
    *,
    notam_id: str = "H7171/26",
    category: Category = Category.COOPERATIVE,
    activity: str = "drone ops",
    geometry: Circle | Polygon | None = None,
    floor: Limit = SFC,
    ceiling: Limit | None = None,
    schedule: str = "",
    contact: ContactKind = ContactKind.PHONE,
    value: str | None = "07784 546062",
    applies_to: AppliesTo = AppliesTo.ALL_AIRCRAFT,
    exemptions: list[Exemption] | None = None,
    designators: list[str] | None = None,
    reason: ProhibitiveReason = ProhibitiveReason.NONE,
) -> NotamReading:
    return NotamReading(
        id=notam_id,
        category=category,
        activity=activity,
        geometry=geometry,
        floor=floor,
        ceiling=ceiling or amsl(800),
        schedule=schedule,
        applies_to=applies_to,
        exemptions=exemptions or [],
        contact=Contact(kind=contact, value=value),
        activates_designators=designators or [],
        prohibitive_reason=reason,
    )


def exemption(quote: str, covers: ExemptionCovers = ExemptionCovers.PARAMOTOR_CLASS) -> Exemption:
    return Exemption(quote=quote, covers=covers)
