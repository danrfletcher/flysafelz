"""The rain grid for GET /map/rain-grid: Open-Meteo's current precipitation on the wind route's 1 degree UK grid. The UI uses it only when RainViewer's index fails.

    rain_grid(client)        -> the list of cells {"lat", "lon", "precipitation"} (mm), or raises RefreshFailed
    request_params()         the query of the one multi-location request

The points are wind.py's own (49 to 61 N and 8 W to 2 E at 1 degree, 143 of them, from the north-west corner row by row), taken from the same
`location_params()` helper, so the two grids can never drift apart. One request carries all 143 points and asks for `current=precipitation` in mm;
Open-Meteo answers with a list of 143 locations in the same order, and a cell takes its place in that list, not the coordinates the answer echoes.

A location with no `current` block, or a missing, null, negative or non-number `precipitation`, is left out: a cell with no value is never drawn
as dry. Zero is a value and is kept. A refresh that cannot give a list of 143 locations (an HTTP error, a timeout, a body that is not JSON or not
that list) or that leaves no cell at all is a failed refresh. A reason names the kind of failure and never holds the response text.
"""

import math

import httpx

from flysafelz.clients.open_meteo import FORECAST_URL
from flysafelz.mapdata.cache import RefreshFailed
from flysafelz.mapdata.wind import POINTS, grid_points, location_params

CURRENT = "precipitation"


def request_params() -> dict[str, str]:
    return {**location_params(), "current": CURRENT}


def _precipitation(location: object) -> float | None:
    """The location's precipitation in mm, or None when it has none that can be drawn (missing, null, not a number, not finite or negative)."""
    current = location.get("current") if isinstance(location, dict) else None
    value = current.get(CURRENT) if isinstance(current, dict) else None
    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value) or value < 0:
        return None
    return float(value)


def _cells(body: object) -> list[dict]:
    """One cell per location that has a precipitation value, in grid order. Raises ValueError for anything that is not 143 locations or has no value at all."""
    if not isinstance(body, list) or len(body) != POINTS:
        raise ValueError("not a list of the 143 grid locations")
    cells = [{"lat": latitude, "lon": longitude, CURRENT: value} for (latitude, longitude), location in zip(grid_points(), body, strict=True) if (value := _precipitation(location)) is not None]
    if not cells:
        raise ValueError("no location has a precipitation value")
    return cells


def rain_grid(client: httpx.Client) -> list[dict]:
    """One request for the whole grid, as the cells that have a value. Raises RefreshFailed."""
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
        return _cells(body)
    except ValueError as error:
        raise RefreshFailed(f"Open-Meteo rain grid unusable: {error}") from None
