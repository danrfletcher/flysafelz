"""Open-Meteo hourly weather (model ukmo_seamless, mph, UTC). No API key.

Two plain fetch functions share one request:

    fetch_forecast             the live forecast API
    fetch_historical_forecast  the historical-forecast API, used for any --case and for capture

fetch_weather picks between them with one `if` on whether a case is set.

Nothing is raised to the caller. An HTTP error, a timeout, a body that is not
JSON or has no `hourly` block gives a Forecast whose status is `failed`, with a
one-line reason and the time it was tried. There is no staleness rule for
weather: a successful fetch is `ok` with its fetched-at time.

The request always runs one hour past the window end, so the last row (23:00)
has the next day's 00:00 values (see weather.py for the row mapping).
"""

from datetime import UTC, datetime, timedelta

import httpx
from pydantic import AwareDatetime, BaseModel, ConfigDict, Field

from flysafelz.contracts import SourceName, SourceState, SourceStatus, WeatherValues

FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
HISTORICAL_URL = "https://historical-forecast-api.open-meteo.com/v1/forecast"

UKMO_SEAMLESS = "ukmo_seamless"
ONE_HOUR = timedelta(hours=1)

# Open-Meteo hourly variable -> WeatherValues field. The keys are exactly what we request.
VARIABLES = {
    "wind_speed_10m": "wind_10m",
    "wind_speed_80m": "wind_80m",
    "wind_speed_120m": "wind_120m",
    "wind_gusts_10m": "gust_10m",
    "wind_direction_10m": "direction_10m",
    "precipitation": "precipitation",
    "precipitation_probability": "precipitation_probability",
    "cape": "cape",
    "cloud_cover": "cloud_cover",
    "cloud_cover_low": "cloud_cover_low",
    "visibility": "visibility",
    "shortwave_radiation": "shortwave_radiation",
    "pressure_msl": "pressure_msl",
}


class Forecast(BaseModel):
    """What one fetch returned.

    samples maps each UTC timestamp the response had to its values. A value is
    None when the response said null or left the variable out. A timestamp the
    response did not have is simply absent: nothing is interpolated or filled.
    """

    model_config = ConfigDict(extra="forbid")

    status: SourceStatus
    elevation_m: float | None = None
    samples: dict[AwareDatetime, WeatherValues] = Field(default_factory=dict)


def request_params(lat: float, lon: float, start: datetime, end: datetime) -> dict[str, str | float]:
    """Query parameters for one fetch. The end date is taken one hour past the window end."""
    start, end = start.astimezone(UTC), end.astimezone(UTC)
    return {
        "latitude": lat,
        "longitude": lon,
        "hourly": ",".join(VARIABLES),
        "models": UKMO_SEAMLESS,
        "wind_speed_unit": "mph",
        "timezone": "UTC",
        "start_date": start.date().isoformat(),
        "end_date": (end + ONE_HOUR).date().isoformat(),
    }


def _number(x: object) -> float | None:
    """A JSON number as a float. null, a missing entry or anything else is None."""
    if isinstance(x, bool) or not isinstance(x, (int, float)):
        return None
    return float(x)


def _failed(reason: str, now: datetime) -> Forecast:
    status = SourceStatus(source=SourceName.WEATHER, state=SourceState.FAILED, reason=reason, fetched_at=now)
    return Forecast(status=status)


def parse_response(data: object, now: datetime) -> Forecast:
    """Turn the decoded JSON body into a Forecast, or a failed one if it has no usable `hourly`."""
    if not isinstance(data, dict) or not isinstance(data.get("hourly"), dict):
        return _failed("response has no hourly data", now)
    hourly = data["hourly"]
    times = hourly.get("time")
    if not isinstance(times, list):
        return _failed("response has no hourly data", now)
    try:
        stamps = [datetime.fromisoformat(t).replace(tzinfo=UTC) for t in times]
    except (TypeError, ValueError):
        return _failed("response has unreadable hourly times", now)

    columns = {}
    for api_name, field in VARIABLES.items():
        column = hourly.get(api_name)
        if column is not None and (not isinstance(column, list) or len(column) != len(stamps)):
            return _failed(f"hourly {api_name} does not match the times", now)
        columns[field] = column or [None] * len(stamps)

    samples = {}
    for i, stamp in enumerate(stamps):
        samples[stamp] = WeatherValues(**{field: _number(column[i]) for field, column in columns.items()})
    status = SourceStatus(source=SourceName.WEATHER, state=SourceState.OK, fetched_at=now)
    return Forecast(status=status, elevation_m=_number(data.get("elevation")), samples=samples)


def _fetch(url: str, lat: float, lon: float, start: datetime, end: datetime, client: httpx.Client, now: datetime | None) -> Forecast:
    now = now or datetime.now(UTC)
    try:
        response = client.get(url, params=request_params(lat, lon, start, end))
        response.raise_for_status()
        body = response.json()
    except httpx.TimeoutException:
        return _failed("request timed out", now)
    except httpx.HTTPStatusError as exc:
        return _failed(f"HTTP {exc.response.status_code}", now)
    except httpx.HTTPError as exc:
        return _failed(f"request failed: {type(exc).__name__}", now)
    except ValueError:
        return _failed("response is not JSON", now)
    return parse_response(body, now)


def fetch_forecast(
    lat: float, lon: float, start: datetime, end: datetime, *, client: httpx.Client, now: datetime | None = None
) -> Forecast:
    """Live forecast. `now` is the fetched-at time; it defaults to the clock."""
    return _fetch(FORECAST_URL, lat, lon, start, end, client, now)


def fetch_historical_forecast(
    lat: float, lon: float, start: datetime, end: datetime, *, client: httpx.Client, now: datetime | None = None
) -> Forecast:
    """Past forecast for a --case replay and for `make fixtures`."""
    return _fetch(HISTORICAL_URL, lat, lon, start, end, client, now)


def fetch_weather(
    lat: float, lon: float, start: datetime, end: datetime, *, client: httpx.Client, case: str | None = None, now: datetime | None = None
) -> Forecast:
    """The historical API when a case is set (replay or capture), the live forecast API otherwise."""
    if case is None:
        return fetch_forecast(lat, lon, start, end, client=client, now=now)
    return fetch_historical_forecast(lat, lon, start, end, client=client, now=now)
