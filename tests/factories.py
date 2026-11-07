"""Builders for the NOTAM action tests: LZs, segments, parsed NOTAMs, readings, shapes.

Parsed NOTAMs come from PR-4's `parsed()` (tests/unit/pib_helpers.py); readings are written by
hand. No fixture file, no network, no .env.
"""

import math
from collections.abc import Mapping
from datetime import UTC, datetime, timedelta

from pib_helpers import parsed as pib_parsed
from weather_helpers import calm

from flysafelz.assess import assess, default_window
from flysafelz.clients.open_meteo import Forecast
from flysafelz.clients.openaip import Airspace, AirspaceFetch
from flysafelz.contracts import (
    Assessment,
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
    SourceName,
    SourceState,
    SourceStatus,
)
from flysafelz.geo import distance_nm
from flysafelz.notam_action import ActionSegment, LandingZone
from flysafelz.pib import PibResult
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


# --- assess(): sources and a calm default run ---------------------------------------------


NOW = utc(7, 30)  # the conftest clock: 2026-10-31T07:30Z, a Saturday; London is on GMT (the clocks went back on 25 Oct)
ISSUED = utc(7, 0)  # a PIB issued half an hour before NOW
ONE_HOUR = timedelta(hours=1)


def status(source: SourceName, state: SourceState = SourceState.OK, reason: str = "", issued_at: datetime | None = None) -> SourceStatus:
    """A hand-built SourceStatus. Failed and stale ones get a reason, as the clients give them."""
    if state is not SourceState.OK and not reason:
        reason = f"{source.value} {state.value} in this test"
    return SourceStatus(source=source, state=state, reason=reason, fetched_at=NOW, issued_at=issued_at)


def weather_forecast(
    first: datetime,
    last: datetime,
    *,
    changes: Mapping[datetime, Mapping[str, float | None]] | None = None,
    after: Mapping[datetime, Mapping[str, float | None]] | None = None,
    elevation_m: float | None = 9.0,
    state: SourceState = SourceState.OK,
) -> Forecast:
    """A calm forecast with a sample at every whole hour from first to last, both included.

    changes maps a timestamp to values that replace the calm ones (None removes one).
    after maps a timestamp to values applied to that timestamp and every later one.
    A failed state gives a Forecast with no samples, as the client does.
    """
    if state is not SourceState.OK:
        return Forecast(status=status(SourceName.WEATHER, state))
    samples = {}
    t = first
    while t <= last:
        values = {}
        for since, over in (after or {}).items():
            if t >= since:
                values.update(over)
        values.update((changes or {}).get(t, {}))
        samples[t] = calm(**values)
        t += ONE_HOUR
    return Forecast(status=status(SourceName.WEATHER), elevation_m=elevation_m, samples=samples)


def forecast_failed(state: SourceState = SourceState.FAILED) -> Forecast:
    """What the weather client returns when it could not get a forecast."""
    return Forecast(status=status(SourceName.WEATHER, state))


def airspace_fetch(*areas: Airspace, state: SourceState = SourceState.OK) -> AirspaceFetch:
    return AirspaceFetch(areas=[] if state is not SourceState.OK else list(areas), status=status(SourceName.AIRSPACE, state))


def pib_result(*notams: ParsedNotam, state: SourceState = SourceState.OK, issued: datetime = ISSUED) -> PibResult:
    """A PIB result. Failed has no NOTAMs. Issued sets the PIB's Issued time, which the stale rule reads."""
    return PibResult(status=status(SourceName.NOTAMS, state, issued_at=issued), notams=[] if state is SourceState.FAILED else list(notams))


def run_assess(
    *,
    window: tuple[datetime, datetime] | None = (utc(9), utc(12)),
    now: datetime = NOW,
    forecast: Forecast | None = None,
    airspace: AirspaceFetch | None = None,
    pib: PibResult | None = None,
    readings: Mapping[str, NotamReading] | None = None,
    lat: float = LZ.lat,
    lon: float = LZ.lon,
    live: bool = False,
) -> Assessment:
    """assess() at Blue Mill on a calm day. By default: 09:00-12:00Z, calm weather, no airspace, no NOTAMs, all sources ok.

    window=None asks for the default window; the calm forecast then covers it.
    """
    if forecast is None:
        first, last = window or default_window(now)
        forecast = weather_forecast(first - ONE_HOUR, last + ONE_HOUR)
    return assess(
        lat,
        lon,
        now=now,
        forecast=forecast,
        airspace=airspace or airspace_fetch(),
        pib=pib or pib_result(),
        readings=readings,
        window=window,
        live=live,
    )


def wind(mph: float) -> dict[str, float]:
    """Calm weather with the 10 m wind at mph and a gust 2 mph above it (spread 2), and 80 m / 120 m wind kept near it."""
    return {"wind_10m": mph, "gust_10m": mph + 2, "wind_80m": mph + 1, "wind_120m": mph + 2}

