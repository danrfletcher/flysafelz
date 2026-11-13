"""openAIP airspace: fetch every airspace that touches a box around the LZ, and read it into plain models.

    fetch_airspace(lat, lon, *, api_key, client)   GET /api/airspaces, every page -> AirspaceFetch
    parse_airspace(item)                           one openAIP item -> Airspace

openAIP's firewall answers an HTML 403 to any airspaces request that carries `bbox` or `pos`,
with or without a key, so the box cannot be asked for. The request is `country=GB` instead,
PAGE_LIMIT items per page, following `nextPage` until it is empty. The box is then cut out
here: an airspace is kept when its polygon touches the box that reaches the airspace_box_nm
rule either side of the LZ (that rule is a half-width, so the box is a square twice that wide).
The key goes in the `x-openaip-api-key` header. This module never reads the
environment: the caller reads OPENAIP_API_KEY and passes it in. A missing key is
a failed status with a one-line reason that names the variable, and no request
is made. The key is never put in a reason.

Nothing is raised to the caller. An HTTP error status (an auth refusal, a server error), a timeout, a body
that is not JSON or not an openAIP page, or an item inside the box that cannot be read at all
gives a status of `failed` with a one-line reason and no airspaces. An empty
result is `ok` with zero airspaces. A 401/403 whose body is JSON is the key refused
("openAIP: HTTP 403"). A 401/403 whose body is HTML is the firewall, not the key, and is worded
as such ("openAIP blocked the request (HTML 403, not a key error)").

Fail closed when reading an item (each is noted on the Airspace):
- an unknown type code is "other" (awareness only);
- a missing or unreadable lower limit is SFC, and a missing or unreadable upper limit is unlimited;
- a polygon that is invalid is repaired with buffer(0); if that is still invalid, or empty,
  the polygon's bounding box is used instead.

This module only fetches and reads. airspace.py decides what an airspace means.
"""

import json
import math
import re
from datetime import UTC, datetime
from typing import Any

import httpx
from pydantic import BaseModel
from shapely.geometry import Polygon, box
from shapely.geometry.base import BaseGeometry

from flysafelz.contracts import STRICT, Datum, Limit, LimitUnit, SourceName, SourceState, SourceStatus
from flysafelz.designators import find_designators
from flysafelz.geo import GeometryError, Shape, shape_from_geojson
from flysafelz.rules import RULES, Rules

URL = "https://api.core.openaip.net/api/airspaces"
KEY_HEADER = "x-openaip-api-key"
KEY_VARIABLE = "OPENAIP_API_KEY"

TIMEOUT_SECONDS = 30
COUNTRY = "GB"  # the only airspace filter openAIP's firewall lets through; the box is cut out locally
PAGE_LIMIT = "1000"  # items asked for per page, as the query-string text it is sent as (GB is about 1,200 airspaces: two pages)
MAX_PAGES = 40  # a stop for a server that never says it is done; far more pages than the country holds

ARC_MINUTES_PER_DEGREE = 60  # one nautical mile is one minute of latitude, by definition
FEET_PER_METRE = 3.280839895

# The openAIP numeric codes, in one place. Each table lists the names in code
# order, so a name's position is its openAIP code (None marks a code openAIP
# does not use). They are the codes of the openAIP Core API schema (airspace
# `type` and `icaoClass`, a limit's `unit` and `referenceDatum`, and an operating-hours
# entry's `dayOfWeek`), written from
# the published schema and not from a live capture: the real capture (PR-7)
# is what confirms them. A code missing from a table is handled: type -> "other",
# class -> unknown, unit or datum -> the limit is not read, day_of_week -> the hours are not read.
CODES: dict[str, dict[int, str | None]] = {
    "type": dict(
        enumerate(
            (
                "other", "restricted", "danger", "prohibited", "ctr", "tmz", "rmz", "tma", "tra", "tsa",
                "fir", "uir", "adiz", "atz", "matz", "airway", "mtr", "alert", "warning", "protected",
                "htz", "gliding", "trp", "tiz", "tia", "mta", "cta", "acc", "sporting", "overflight_restriction",
                "mrt", "tfr", "vfr_sector", "fis_sector", "lta", "uta", "mctr",
            )
        )
    ),
    "icao_class": dict(enumerate(("A", "B", "C", "D", "E", "F", "G", None, "SUA"))),
    "unit": dict(enumerate(("m", "ft", None, None, None, None, "FL"))),
    "datum": dict(enumerate(("GND", "AMSL", "STD"))),
    # operatingHours dayOfWeek: 0 is every day and 1..7 are Monday..Sunday. Like the tables above this is from the
    # published schema and is NOT yet confirmed by a live capture. If openAIP counts the days 0-6 from Monday, a single-day
    # entry lands a day late (fail-open on that day), so check one real single-day entry before trusting it.
    "day_of_week": dict(enumerate(("DAILY", "MON", "TUE", "WED", "THU", "FRI", "SAT", "SUN"))),
}
OTHER = "other"


class OpenAipError(ValueError):
    """An openAIP item that cannot be used at all (no readable polygon, wrong shape)."""


class Airspace(BaseModel):
    """One airspace volume, as the rest of the code sees it.

    lower is SFC when openAIP gave none. upper is None for unlimited. hours is
    schedule text in the form schedule.py reads ("DAILY 0800-1800, SAT SR-SS", or
    "H24"), None when openAIP published none; hours_readable is False when it
    published some that could not be turned into that text. notes record every
    fail-closed choice made while reading the item.
    """

    model_config = STRICT

    id: str
    name: str
    type: str  # a value of CODES["type"]
    type_code: int | None = None
    icao_class: str | None = None  # "A".."G", "SUA", or None when unknown
    lower: Limit
    upper: Limit | None = None
    by_notam: bool = False
    hours: str | None = None
    hours_readable: bool = True
    designator: str | None = None  # EG<D|R|P><nnn><X> from the name, e.g. EGD496A
    shape: Shape
    notes: list[str] = []


class AirspaceFetch(BaseModel):
    """What one fetch returned: the airspaces (empty on failure) and the status."""

    model_config = STRICT

    areas: list[Airspace] = []
    status: SourceStatus


# --- limits -------------------------------------------------------------------


def metres_to_feet(metres: float) -> float:
    return metres * FEET_PER_METRE


def _read_limit(raw: object) -> tuple[Limit | None, str | None]:
    """A limit from openAIP's {value, unit, referenceDatum}, or (None, why not)."""
    if not isinstance(raw, dict):
        return None, "missing"
    value, unit_code, datum_code = raw.get("value"), raw.get("unit"), raw.get("referenceDatum")
    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value):
        return None, "has no numeric value"
    unit = CODES["unit"].get(unit_code) if isinstance(unit_code, int) else None
    datum = CODES["datum"].get(datum_code) if isinstance(datum_code, int) else None
    if unit is None:
        return None, f"has unknown unit code {unit_code!r}"
    if unit == "FL":
        return Limit(value=value, unit=LimitUnit.FL, datum=Datum.AMSL), None  # a flight level has no other datum
    if datum is None:
        return None, f"has unknown datum code {datum_code!r}"
    feet = metres_to_feet(value) if unit == "m" else float(value)
    if datum == "GND":
        return (Limit(value=0, unit=LimitUnit.FT, datum=Datum.SFC) if feet <= 0 else Limit(value=feet, unit=LimitUnit.FT, datum=Datum.AGL)), None
    return Limit(value=feet, unit=LimitUnit.FT, datum=Datum.AMSL), None  # AMSL, or STD read as AMSL


# --- hours --------------------------------------------------------------------

END_OF_DAY = "2400"
LAST_MINUTE = "2359"


CLOCK = re.compile(r"(\d\d):(\d\d)")


def _clock(text: object) -> str | None:
    """'08:00' -> '0800'. openAIP writes the end of a 24 h day as 23:59, which is read as 2400."""
    match = CLOCK.fullmatch(text) if isinstance(text, str) else None
    if match is None:
        return None
    digits = "".join(match.groups())
    return END_OF_DAY if digits == LAST_MINUTE else digits


def _hours_clause(entry: object) -> str | None:
    """One operatingHours entry as a schedule clause, or None when it cannot be read."""
    if not isinstance(entry, dict):
        return None
    day = entry.get("dayOfWeek", 0)
    day_word = CODES["day_of_week"].get(day) if isinstance(day, int) else None
    start = "SR" if entry.get("sunrise") else _clock(entry.get("startTime"))
    end = "SS" if entry.get("sunset") else _clock(entry.get("endTime"))
    if day_word is None or start is None or end is None:
        return None
    return f"{day_word} {start}-{end}"


def _read_hours(raw: object) -> tuple[str | None, bool, bool, list[str]]:
    """(schedule text, readable, by NOTAM, notes) from openAIP's hoursOfOperation."""
    entries = raw.get("operatingHours") if isinstance(raw, dict) else None
    if not entries:
        return None, True, False, []
    if not isinstance(entries, list):
        return None, False, False, ["hoursOfOperation could not be read"]
    by_notam = any(isinstance(e, dict) and e.get("byNotam") for e in entries)
    timed = [e for e in entries if not (isinstance(e, dict) and e.get("byNotam") and not e.get("startTime") and not e.get("sunrise"))]
    clauses = [_hours_clause(e) for e in timed]
    if any(c is None for c in clauses):
        return None, False, by_notam, ["hoursOfOperation could not be read"]
    if not clauses:
        return None, True, by_notam, []
    if clauses == [f"DAILY 0000-{END_OF_DAY}"]:
        return "H24", True, by_notam, []
    return ", ".join(c for c in clauses if c), True, by_notam, []


# --- polygons -----------------------------------------------------------------


def _polygons(geometry: BaseGeometry) -> list[Polygon]:
    if isinstance(geometry, Polygon):
        return [] if geometry.is_empty else [geometry]
    parts = getattr(geometry, "geoms", [])  # a MultiPolygon, or a GeometryCollection buffer(0) can return
    return [p for part in parts for p in _polygons(part)]


def _to_shape(polygons: list[Polygon]) -> Shape:
    """Polygons back to the (lat, lon) Shape geo.py reads. Shapely's x is lon and y is lat."""
    return [[[(y, x) for x, y in ring.coords] for ring in [p.exterior, *p.interiors]] for p in polygons]


def _repair(shape: Shape, name: str) -> tuple[Shape, list[str]]:
    """Valid polygons from the item's rings: buffer(0) on an invalid one, its bounding box if that fails too."""
    notes: list[str] = []
    fixed: list[Polygon] = []
    for poly in shape:
        rings = [[(lon, lat) for lat, lon in ring] for ring in poly]
        candidate = Polygon(rings[0], rings[1:])
        if candidate.is_valid:
            fixed.append(candidate)
            continue
        repaired = candidate.buffer(0)
        parts = _polygons(repaired)
        if repaired.is_valid and parts:
            fixed.extend(parts)
            notes.append(f"{name}: invalid polygon repaired with buffer(0)")
        else:
            fixed.append(box(*candidate.bounds))
            notes.append(f"{name}: invalid polygon could not be repaired; its bounding box is used (fail closed)")
    return _to_shape(fixed), notes


# --- one item -------------------------------------------------------------------


def _name(item: dict[str, Any]) -> str:
    return str(item.get("name") or "").strip() or "(unnamed airspace)"


def item_id(item: dict[str, Any]) -> str:
    """The id an item gets as an Airspace: openAIP's `_id`, else its name."""
    return str(item.get("_id") or _name(item))


def parse_airspace(item: dict[str, Any]) -> Airspace:
    """One entry of an openAIP page's `items` as an Airspace.

    Raises OpenAipError when the item has no usable polygon. Everything else
    that is missing is read the fail-closed way and noted on the Airspace.
    """
    if not isinstance(item, dict):
        raise OpenAipError("airspace item is not an object")
    name = _name(item)
    notes: list[str] = []

    type_code = item.get("type")
    type_name = CODES["type"].get(type_code) if isinstance(type_code, int) else None
    if type_name is None:
        notes.append(f"{name}: unknown airspace type code {type_code!r}; treated as other (awareness only)")
        type_name = OTHER

    class_code = item.get("icaoClass")
    icao_class = CODES["icao_class"].get(class_code) if isinstance(class_code, int) else None

    lower, why = _read_limit(item.get("lowerLimit"))
    if lower is None:
        lower = Limit(value=0, unit=LimitUnit.FT, datum=Datum.SFC)
        notes.append(f"{name}: lower limit {why}; treated as SFC")
    upper, why = _read_limit(item.get("upperLimit"))
    if upper is None:
        notes.append(f"{name}: upper limit {why}; treated as unlimited")

    hours, readable, hours_by_notam, hours_notes = _read_hours(item.get("hoursOfOperation"))
    notes += [f"{name}: {n}" for n in hours_notes]

    try:
        shape = shape_from_geojson(item.get("geometry") or {})
    except (GeometryError, TypeError, IndexError, KeyError) as exc:
        raise OpenAipError(f"{name}: no usable polygon ({exc})") from exc
    shape, repairs = _repair(shape, name)
    designators = find_designators(name)
    return Airspace(
        id=item_id(item),
        name=name,
        type=type_name,
        type_code=type_code if isinstance(type_code, int) else None,
        icao_class=icao_class,
        lower=lower,
        upper=upper,
        by_notam=bool(item.get("byNotam")) or hours_by_notam,
        hours=hours,
        hours_readable=readable,
        designator=designators[0] if designators else None,
        shape=shape,
        notes=notes + repairs,
    )


# --- fetching -------------------------------------------------------------------


def box_bounds(lat: float, lon: float, half_width_nm: float) -> tuple[float, float, float, float]:
    """(min lon, min lat, max lon, max lat) of the box half_width_nm either side of the point.

    The longitude width is taken at the box's far edge from the equator, where a
    degree of longitude is shortest, so the box always covers at least half_width_nm.
    """
    dlat = half_width_nm / ARC_MINUTES_PER_DEGREE
    dlon = dlat / math.cos(math.radians(abs(lat) + dlat))
    return lon - dlon, lat - dlat, lon + dlon, lat + dlat


def _failed(reason: str, now: datetime) -> AirspaceFetch:
    return AirspaceFetch(status=SourceStatus(source=SourceName.AIRSPACE, state=SourceState.FAILED, reason=reason, fetched_at=now))


def _is_html(response: httpx.Response) -> bool:
    return "html" in response.headers.get("content-type", "").lower() or response.text.lstrip().startswith("<")


def _get_page(client: httpx.Client, params: dict[str, str | int], key: str) -> httpx.Response | str:
    """The response, or a one-line failure reason (status code or exception class)."""
    try:
        response = client.get(URL, params=params, headers={KEY_HEADER: key})
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        status = exc.response.status_code
        if status in (401, 403) and _is_html(exc.response):  # the firewall's page, not openAIP's answer about the key
            return f"openAIP blocked the request (HTML {status}, not a key error)"
        return f"openAIP: HTTP {status}"
    except httpx.HTTPError as exc:
        return f"openAIP: {type(exc).__name__}"
    return response


def _coordinates(node: object) -> list[tuple[float, float]] | None:
    """Every [lon, lat] pair under a GeoJSON `coordinates` value, or None when it is not made of numbers."""
    if not isinstance(node, list) or not node:
        return None
    if all(isinstance(v, int | float) and not isinstance(v, bool) for v in node):
        return [(float(node[0]), float(node[1]))] if len(node) > 1 else None
    found: list[tuple[float, float]] = []
    for child in node:
        part = _coordinates(child)
        if part is None:
            return None
        found += part
    return found


def _raw_bounds(item: object) -> tuple[float, float, float, float] | None:
    """(min lon, min lat, max lon, max lat) of an item's raw geometry, or None when that cannot be worked out."""
    geometry = item.get("geometry") if isinstance(item, dict) else None
    points = _coordinates(geometry.get("coordinates")) if isinstance(geometry, dict) else None
    if not points:
        return None
    lons, lats = [p[0] for p in points], [p[1] for p in points]
    return min(lons), min(lats), max(lons), max(lats)


def _outside(bounds: tuple[float, float, float, float], area_box: tuple[float, float, float, float]) -> bool:
    lon0, lat0, lon1, lat1 = bounds
    box_lon0, box_lat0, box_lon1, box_lat1 = area_box
    return lon0 > box_lon1 or lon1 < box_lon0 or lat0 > box_lat1 or lat1 < box_lat0


def _touches(area: Airspace, area_box: tuple[float, float, float, float]) -> bool:
    """True when any polygon of the airspace overlaps or touches the box (shapely: x is lon, y is lat)."""
    window = box(*area_box)
    for poly in area.shape:
        rings = [[(lon, lat) for lat, lon in ring] for ring in poly]
        if Polygon(rings[0], rings[1:]).intersects(window):
            return True
    return False


def _next_page(payload: dict[str, Any], page: int) -> int | None | str:
    """The next page number, None when this was the last, or a one-line reason when the page cannot say.

    openAIP gives `nextPage` (empty on the last page). If a page has no `nextPage` at all, `totalPages` decides,
    so a short read cannot pass for the whole country. How the last page spells "none" has not been seen on the live
    API (the keyed capture will show it): null is the documented form, and a number that is not a later page (0, or
    this page again) is accepted as "none" only when `totalPages` says this page is the last. Anywhere else it fails.
    """
    if "nextPage" in payload:
        following = payload["nextPage"]
        if following is None:
            return None
        if isinstance(following, bool) or not isinstance(following, int):
            return "openAIP: nextPage is not a later page number"
        if following <= page:
            total = payload.get("totalPages")
            if isinstance(total, int) and not isinstance(total, bool) and page >= total:
                return None
            return "openAIP: nextPage is not a later page number"
        return following
    total = payload.get("totalPages", 1)
    if isinstance(total, bool) or not isinstance(total, int):
        return "openAIP: totalPages is not a number"
    return page + 1 if page < total else None


def _fetch_pages(client: httpx.Client, lat: float, lon: float, key: str, now: datetime, rules: Rules) -> AirspaceFetch:
    area_box = box_bounds(lat, lon, float(rules.airspace_box_nm.value))  # type: ignore[arg-type]
    base: dict[str, str | int] = {"country": COUNTRY, "limit": PAGE_LIMIT}
    areas: dict[str, Airspace] = {}
    page = 1
    for _ in range(MAX_PAGES):
        got = _get_page(client, {**base, "page": page}, key)
        if isinstance(got, str):
            return _failed(got, now)
        try:
            payload = got.json()
        except ValueError:
            return _failed("openAIP: response is not JSON", now)
        items = payload.get("items") if isinstance(payload, dict) else None
        if not isinstance(items, list):
            return _failed("openAIP: response is not an airspace page", now)
        for item in items:
            bounds = _raw_bounds(item)
            if bounds is not None and _outside(bounds, area_box):
                continue  # clearly elsewhere in the country: not read, so one odd item far away cannot fail the fetch
            try:
                area = parse_airspace(item)
            except OpenAipError as exc:
                return _failed(f"openAIP: {exc}", now)
            if _touches(area, area_box):
                areas.setdefault(area.id, area)
        following = _next_page(payload, page)
        if isinstance(following, str):
            return _failed(following, now)
        if following is None:
            status = SourceStatus(source=SourceName.AIRSPACE, state=SourceState.OK, fetched_at=now)
            return AirspaceFetch(areas=list(areas.values()), status=status)
        page = following
    return _failed("openAIP: more pages than expected", now)


def fetch_airspace(
    lat: float,
    lon: float,
    *,
    api_key: str | None,
    client: httpx.Client | None = None,
    now: datetime | None = None,
    rules: Rules = RULES,
) -> AirspaceFetch:
    """Every airspace touching the box around (lat, lon): all of country=GB, page by page, cut down to the box here.

    `now` is the fetched-at time; leave it out and the clock is read. Pass a client
    to control the transport (tests use httpx.MockTransport). Without one, a client
    with a timeout is made and closed here.
    """
    fetched_at = now if now is not None else datetime.now(UTC)
    if not api_key or not api_key.strip():
        return _failed(f"{KEY_VARIABLE} is not set", fetched_at)
    try:
        if client is not None:
            return _fetch_pages(client, lat, lon, api_key, fetched_at, rules)
        with httpx.Client(timeout=TIMEOUT_SECONDS, follow_redirects=True) as own_client:
            return _fetch_pages(own_client, lat, lon, api_key, fetched_at, rules)
    except (json.JSONDecodeError, ValueError, TypeError, KeyError) as exc:  # a payload shape nothing above expected
        return _failed(f"openAIP: unreadable response ({type(exc).__name__})", fetched_at)
