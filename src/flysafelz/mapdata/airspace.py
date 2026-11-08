"""Every GB airspace from openAIP as GeoJSON features, for GET /map/airspace.

    airspace_features(client, key)   -> the features, or raises RefreshFailed
    limit_text(limit), limit_feet(limit)

The route needs all of GB, not the box round one landing zone, so it does not call `fetch_airspace` (which cuts the country down to a box).
It reads every page itself with the client module's own page request, paging rule and reader: `_get_page`, `_next_page`, `parse_airspace`,
PAGE_LIMIT and MAX_PAGES. Any page that fails, an HTML 403, or a body that is not an openAIP page fails the whole refresh.

An item `parse_airspace` repairs (buffer(0)) or falls back on (a bounding box, SFC, unlimited, type "other") keeps its feature as the parser
gives it. An item with no polygon at all cannot be drawn and is left out; the count is logged, never the item.

A feature's properties: id, name, type, class (the ICAO class or null), lower and upper (text), lower_ft and upper_ft (feet, null when
unlimited), hours, action. The base layer is not about one landing zone or one time, so no airspace maps to an action here: `action` is null
(the answer's own overlays carry the action, from the agent). `hours` is openAIP's schedule text, "by NOTAM" when that is all it says, else null.
"""

import logging

import httpx

from flysafelz.airspace import FEET_PER_FLIGHT_LEVEL
from flysafelz.clients.openaip import (
    COUNTRY,
    KEY_VARIABLE,
    MAX_PAGES,
    PAGE_LIMIT,
    Airspace,
    OpenAipError,
    _get_page,
    _next_page,
    parse_airspace,
)
from flysafelz.contracts import Datum, Limit, LimitUnit
from flysafelz.mapdata.cache import RefreshFailed
from flysafelz.shapes import airspace_geometry

log = logging.getLogger("flysafelz.mapdata")

UNLIMITED = "UNL"
BY_NOTAM = "by NOTAM"


def limit_text(limit: Limit | None) -> str:
    """SFC, FL65, 1500 ft AGL (above the ground), 4500 ft AMSL (altitude), or UNL when there is no limit."""
    if limit is None:
        return UNLIMITED
    if limit.datum is Datum.SFC:
        return Datum.SFC.value
    if limit.unit is LimitUnit.FL:
        return f"FL{limit.value:.0f}"
    return f"{limit.value:.0f} ft {limit.datum.value}"


def limit_feet(limit: Limit | None) -> int | None:
    """The limit in feet as stated (SFC is 0, a flight level is hundreds of feet), or None when there is no limit."""
    if limit is None:
        return None
    if limit.datum is Datum.SFC:
        return 0
    if limit.unit is LimitUnit.FL:
        return round(limit.value * FEET_PER_FLIGHT_LEVEL)
    return round(limit.value)


def _hours(area: Airspace) -> str | None:
    if area.hours is not None:
        return area.hours
    return BY_NOTAM if area.by_notam else None


def airspace_feature(area: Airspace) -> dict:
    properties = {
        "id": area.id,
        "name": area.name,
        "type": area.type,
        "class": area.icao_class,
        "lower": limit_text(area.lower),
        "upper": limit_text(area.upper),
        "lower_ft": limit_feet(area.lower),
        "upper_ft": limit_feet(area.upper),
        "hours": _hours(area),
        "action": None,
    }
    return {"type": "Feature", "geometry": airspace_geometry(area.shape), "properties": properties}


def airspace_features(client: httpx.Client, key: str | None) -> list[dict]:
    """Every page of country=GB, in order, an airspace once however many pages repeat it."""
    if not key or not key.strip():
        raise RefreshFailed(f"{KEY_VARIABLE} is not set")
    areas: dict[str, Airspace] = {}
    unusable = 0
    page = 1
    for _ in range(MAX_PAGES):
        got = _get_page(client, {"country": COUNTRY, "limit": PAGE_LIMIT, "page": page}, key)
        if isinstance(got, str):
            raise RefreshFailed(got.replace(key, "***"))
        try:
            payload = got.json()
        except ValueError as error:
            raise RefreshFailed("openAIP: response is not JSON") from error
        items = payload.get("items") if isinstance(payload, dict) else None
        if not isinstance(items, list):
            raise RefreshFailed("openAIP: response is not an airspace page")
        for item in items:
            try:
                area = parse_airspace(item)
            except OpenAipError:
                unusable += 1
                continue
            areas.setdefault(area.id, area)
        following = _next_page(payload, page)
        if isinstance(following, str):
            raise RefreshFailed(following)
        if following is None:
            if unusable:
                log.warning("openAIP: %d airspace items have no usable polygon and are not on the map", unusable)
            return [airspace_feature(area) for area in areas.values()]
        page = following
    raise RefreshFailed("openAIP: more pages than expected")
