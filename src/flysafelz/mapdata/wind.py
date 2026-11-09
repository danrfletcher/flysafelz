"""The wind grid for GET /map/wind: Open-Meteo's current 10 m wind over the UK, as the two records leaflet-velocity draws.

    wind_grid(client, now)   -> [U record, V record], or raises RefreshFailed
    grid_points()            the 143 (latitude, longitude) points, north to south and, in each row, west to east
    location_params()        the multi-location `latitude` and `longitude` of those points (shared with rain_grid.py)
    request_params()         the query of the one multi-location request
    wind_uv(speed, direction) -> (u, v) in m/s

The grid is 49 to 61 N and 8 W to 2 E at 1 degree: 13 rows by 11 columns. One request carries all 143 points as comma-separated `latitude` and
`longitude`, asks for `current=wind_speed_10m,wind_direction_10m` in m/s, and Open-Meteo answers with a list of 143 locations in the same order.
This is its own request beside clients/open_meteo.py (per point, hourly, mph), which it does not touch.

leaflet-velocity wants a U record (parameterCategory 2, parameterNumber 2) then a V record (2, 3), each with the grid header and `data` in the same
order as the points: rows from the north, each row from the west. The wind direction is where the wind comes from, so
u (towards the east) = -speed * sin(direction) and v (towards the north) = -speed * cos(direction).

Any refresh that cannot give all 143 points whole is a failed refresh: an HTTP error, a timeout, a body that is not JSON or not a list of 143
locations, or a location with no `current` or a null speed (a null direction is accepted only for a calm point, speed 0). The route never serves a
short or misaligned grid. A reason names the kind of failure and never holds the response text.
"""

import math
from datetime import datetime

import httpx

from flysafelz.clients.open_meteo import FORECAST_URL
from flysafelz.clients.pib_source import utc_text
from flysafelz.mapdata.cache import RefreshFailed

NORTH = 61
SOUTH = 49
WEST = -8
EAST = 2
STEP = 1  # degrees
LATITUDES = tuple(range(NORTH, SOUTH - STEP, -STEP))  # north to south
LONGITUDES = tuple(range(WEST, EAST + STEP, STEP))  # west to east
ROWS, COLUMNS = len(LATITUDES), len(LONGITUDES)  # 13 and 11
POINTS = ROWS * COLUMNS  # 143

CURRENT = ("wind_speed_10m", "wind_direction_10m")
SPEED_UNIT = "ms"
DECIMALS = 3  # decimals kept of a component: a component is rounded so that sin(180 degrees) is 0 and not 1.2e-16

# leaflet-velocity's record kinds: GRIB2 table 4.2, category 2 (momentum), number 2 (u) and 3 (v)
PARAMETER_CATEGORY = 2
U_NUMBER = 2
V_NUMBER = 3
U_PARAMETER = (PARAMETER_CATEGORY, U_NUMBER)
V_PARAMETER = (PARAMETER_CATEGORY, V_NUMBER)


def grid_points() -> list[tuple[int, int]]:
    """Every (latitude, longitude) of the grid in leaflet-velocity's order: the northern row first, each row from the west."""
    return [(latitude, longitude) for latitude in LATITUDES for longitude in LONGITUDES]


def location_params() -> dict[str, str]:
    """The `latitude` and `longitude` of an Open-Meteo multi-location request for the whole grid, in grid_points() order. The rain grid asks for the same points."""
    points = grid_points()
    return {
        "latitude": ",".join(str(latitude) for latitude, _ in points),
        "longitude": ",".join(str(longitude) for _, longitude in points),
    }


def request_params() -> dict[str, str]:
    return {**location_params(), "current": ",".join(CURRENT), "wind_speed_unit": SPEED_UNIT}


def _number(value: object) -> float | None:
    """The value as a finite, non-boolean number, else None."""
    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value):
        return None
    return float(value)


def wind_uv(speed: float | None, direction: float | None) -> tuple[float, float]:
    """(u, v) in m/s from a speed and the direction it blows from, in degrees. Calm air (speed 0) is (0, 0) whatever the direction, even a missing one.

    Raises ValueError for a missing, negative or non-finite speed, and for a missing or non-finite direction when the air is not calm.
    """
    known_speed = _number(speed)
    if known_speed is None or known_speed < 0:
        raise ValueError("no usable wind speed")
    if known_speed == 0:
        return 0.0, 0.0
    known_direction = _number(direction)
    if known_direction is None:
        raise ValueError("no usable wind direction")
    radians = math.radians(known_direction)
    # + 0.0 turns a rounded -0.0 into 0.0
    return round(-known_speed * math.sin(radians), DECIMALS) + 0.0, round(-known_speed * math.cos(radians), DECIMALS) + 0.0


def _header(parameter: tuple[int, int], ref_time: str, latitudes: tuple[int, ...], longitudes: tuple[int, ...], step: int) -> dict:
    return {
        "parameterCategory": parameter[0],
        "parameterNumber": parameter[1],
        "nx": len(longitudes),
        "ny": len(latitudes),
        "lo1": longitudes[0],
        "la1": latitudes[0],
        "lo2": longitudes[-1],
        "la2": latitudes[-1],
        "dx": step,
        "dy": step,
        "refTime": ref_time,
    }


def grid_records(
    components: list[tuple[float, float]],
    ref_time: str,
    latitudes: tuple[int, ...] = LATITUDES,
    longitudes: tuple[int, ...] = LONGITUDES,
    step: int = STEP,
) -> list[dict]:
    """The U then V record for the (u, v) pair of each point, rows from the first latitude (the north) and each row from the first longitude (the west).

    The axes default to the UK grid; the headers are made from them. Raises ValueError when the pairs are not one per point.
    """
    if len(components) != len(latitudes) * len(longitudes):
        raise ValueError(f"expected {len(latitudes) * len(longitudes)} points, got {len(components)}")
    return [
        {"header": _header(U_PARAMETER, ref_time, latitudes, longitudes, step), "data": [u for u, _ in components]},
        {"header": _header(V_PARAMETER, ref_time, latitudes, longitudes, step), "data": [v for _, v in components]},
    ]


def _components(body: object) -> list[tuple[float, float]]:
    """(u, v) for each of the 143 locations of an Open-Meteo multi-location answer. Raises ValueError for anything else."""
    if not isinstance(body, list) or len(body) != POINTS:
        raise ValueError("not a list of the 143 grid locations")
    found = []
    for location in body:
        current = location.get("current") if isinstance(location, dict) else None
        if not isinstance(current, dict):
            raise ValueError("a location has no current block")  # noqa: TRY004 - every refusal is a ValueError; wind_grid reads them as one
        found.append(wind_uv(current.get(CURRENT[0]), current.get(CURRENT[1])))
    return found


def wind_grid(client: httpx.Client, now: datetime) -> list[dict]:
    """One request for the whole grid, as the U and V records with `now` as their refTime. Raises RefreshFailed."""
    try:
        response = client.get(FORECAST_URL, params=request_params())
        response.raise_for_status()
        body = response.json()
    except httpx.HTTPStatusError as error:
        raise RefreshFailed(f"Open-Meteo answered HTTP {error.response.status_code}") from None
    except httpx.HTTPError as error:
        raise RefreshFailed(f"Open-Meteo not reachable ({type(error).__name__})") from None
    except ValueError:
        raise RefreshFailed("Open-Meteo answered with something that is not JSON") from None
    try:
        return grid_records(_components(body), utc_text(now))
    except ValueError as error:
        raise RefreshFailed(f"Open-Meteo wind grid unusable: {error}") from None
