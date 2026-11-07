"""Open-Meteo client (G1, E3, failure statuses). httpx.MockTransport serves hand-made JSON; no socket opens."""

import json
from datetime import UTC, datetime, timedelta, timezone

import httpx
import pytest
from weather_helpers import FIXTURES, NOW

from flysafelz.clients import open_meteo
from flysafelz.clients.open_meteo import (
    FORECAST_URL,
    HISTORICAL_URL,
    fetch_forecast,
    fetch_historical_forecast,
    fetch_weather,
)
from flysafelz.contracts import SourceName, SourceState

START = datetime(2026, 10, 31, 7, 0, tzinfo=UTC)
END = datetime(2026, 10, 31, 18, 0, tzinfo=UTC)
LAT, LON = 53.802797, -0.086644

EXPECTED_VARIABLES = {
    "wind_speed_10m",
    "wind_speed_80m",
    "wind_speed_120m",
    "wind_gusts_10m",
    "wind_direction_10m",
    "precipitation",
    "precipitation_probability",
    "cape",
    "cloud_cover",
    "cloud_cover_low",
    "visibility",
    "shortwave_radiation",
    "pressure_msl",
}


def fixture_body(name: str) -> str:
    return (FIXTURES / f"{name}.json").read_text(encoding="utf-8")


def client_serving(body: str = "", status: int = 200, seen: list[httpx.Request] | None = None) -> httpx.Client:
    """An httpx client whose transport answers every request with the given body and records it."""

    def handler(request: httpx.Request) -> httpx.Response:
        if seen is not None:
            seen.append(request)
        return httpx.Response(status, text=body or fixture_body("full_48h"), headers={"content-type": "application/json"})

    return httpx.Client(transport=httpx.MockTransport(handler))


def client_raising(exc: Exception) -> httpx.Client:
    def handler(request: httpx.Request) -> httpx.Response:
        raise exc

    return httpx.Client(transport=httpx.MockTransport(handler))


# --- G1: URL choice and request ---------------------------------------------------


def test_the_forecast_function_calls_the_forecast_api():
    seen = []
    fetch_forecast(LAT, LON, START, END, client=client_serving(seen=seen), now=NOW)
    assert len(seen) == 1
    assert str(seen[0].url).split("?")[0] == "https://api.open-meteo.com/v1/forecast"
    assert FORECAST_URL == "https://api.open-meteo.com/v1/forecast"


def test_the_historical_function_calls_the_historical_forecast_api():
    seen = []
    fetch_historical_forecast(LAT, LON, START, END, client=client_serving(seen=seen), now=NOW)
    assert str(seen[0].url).split("?")[0] == "https://historical-forecast-api.open-meteo.com/v1/forecast"
    assert HISTORICAL_URL == "https://historical-forecast-api.open-meteo.com/v1/forecast"


def test_no_case_picks_the_forecast_api_and_a_case_picks_the_historical_api():
    live, replay = [], []
    fetch_weather(LAT, LON, START, END, client=client_serving(seen=live), case=None, now=NOW)
    fetch_weather(LAT, LON, START, END, client=client_serving(seen=replay), case="blue_mill_2026-10-31", now=NOW)
    assert str(live[0].url).startswith(FORECAST_URL + "?")
    assert str(replay[0].url).startswith(HISTORICAL_URL + "?")


def test_request_params_are_the_model_unit_zone_and_exact_variables():
    seen = []
    fetch_forecast(LAT, LON, START, END, client=client_serving(seen=seen), now=NOW)
    params = dict(seen[0].url.params)
    assert params["models"] == "ukmo_seamless"
    assert params["wind_speed_unit"] == "mph"
    assert params["timezone"] == "UTC"
    assert float(params["latitude"]) == LAT and float(params["longitude"]) == LON
    requested = params["hourly"].split(",")
    assert len(requested) == len(EXPECTED_VARIABLES)
    assert set(requested) == EXPECTED_VARIABLES
    assert set(open_meteo.VARIABLES) == EXPECTED_VARIABLES


def test_every_fetch_asks_for_one_hour_past_the_window_end():
    # A window ending 23:00 UTC on 31 Oct: the extra hour moves the end date to 1 Nov, so 00:00 is in the data.
    seen = []
    fetch_forecast(LAT, LON, START, datetime(2026, 10, 31, 23, 0, tzinfo=UTC), client=client_serving(seen=seen), now=NOW)
    assert dict(seen[0].url.params)["start_date"] == "2026-10-31"
    assert dict(seen[0].url.params)["end_date"] == "2026-11-01"
    # The same for the historical API, and an end earlier in the day keeps the same date.
    fetch_historical_forecast(LAT, LON, START, END, client=client_serving(seen=seen), now=NOW)
    assert dict(seen[1].url.params)["end_date"] == "2026-10-31"
    # The window end is read in UTC even when given in another zone: 01:00 +01:00 on 1 Nov is 00:00 UTC.
    plus_one = datetime(2026, 11, 1, 1, 0, tzinfo=timezone(timedelta(hours=1)))
    fetch_forecast(LAT, LON, START, plus_one, client=client_serving(seen=seen), now=NOW)
    assert dict(seen[2].url.params)["end_date"] == "2026-11-01"


def test_no_key_or_auth_is_sent():
    seen = []
    fetch_forecast(LAT, LON, START, END, client=client_serving(seen=seen), now=NOW)
    assert "authorization" not in seen[0].headers
    assert not [k for k in dict(seen[0].url.params) if "key" in k.lower()]


# --- parsing -----------------------------------------------------------------------


def test_a_full_response_is_ok_with_fetched_at_elevation_and_every_timestamp():
    forecast = fetch_forecast(LAT, LON, START, END, client=client_serving(), now=NOW)
    assert forecast.status.source is SourceName.WEATHER
    assert forecast.status.state is SourceState.OK
    assert forecast.status.fetched_at == NOW
    assert forecast.status.reason == ""
    assert forecast.elevation_m == 8.0
    assert len(forecast.samples) == 49
    first = forecast.samples[datetime(2026, 10, 31, 0, 0, tzinfo=UTC)]
    raw = json.loads(fixture_body("full_48h"))["hourly"]
    assert first.wind_10m == raw["wind_speed_10m"][0]
    assert first.gust_10m == raw["wind_gusts_10m"][0]
    assert first.precipitation_probability == raw["precipitation_probability"][0]
    assert first.pressure_msl == raw["pressure_msl"][0]
    assert all(t.tzinfo is not None and t.utcoffset().total_seconds() == 0 for t in forecast.samples)


def test_a_null_stays_none_at_that_timestamp_only_and_nothing_is_filled():
    forecast = fetch_forecast(LAT, LON, START, END, client=client_serving(fixture_body("hole_in_cape")), now=NOW)
    assert forecast.status.state is SourceState.OK
    hole = datetime(2026, 10, 31, 12, 0, tzinfo=UTC)
    assert forecast.samples[hole].cape is None
    assert forecast.samples[hole].wind_10m is not None
    assert forecast.samples[datetime(2026, 10, 31, 11, 0, tzinfo=UTC)].cape is not None
    assert forecast.samples[datetime(2026, 10, 31, 13, 0, tzinfo=UTC)].cape is not None


def test_nulls_at_the_tail_stay_none_and_the_response_is_still_ok():
    forecast = fetch_forecast(LAT, LON, START, END, client=client_serving(fixture_body("tail_missing_80m")), now=NOW)
    assert forecast.status.state is SourceState.OK
    last = forecast.samples[datetime(2026, 11, 2, 0, 0, tzinfo=UTC)]
    assert last.wind_80m is None and last.wind_120m is None
    assert last.wind_10m is not None
    assert forecast.samples[datetime(2026, 11, 1, 20, 0, tzinfo=UTC)].wind_80m is not None


def test_a_variable_the_response_leaves_out_is_none_everywhere():
    body = json.loads(fixture_body("full_48h"))
    del body["hourly"]["wind_speed_120m"]
    del body["elevation"]
    forecast = fetch_forecast(LAT, LON, START, END, client=client_serving(json.dumps(body)), now=NOW)
    assert forecast.status.state is SourceState.OK
    assert forecast.elevation_m is None
    assert all(s.wind_120m is None for s in forecast.samples.values())
    assert all(s.wind_10m is not None for s in forecast.samples.values())


def test_odd_values_are_none_not_a_crash():
    body = {"elevation": "high", "hourly": {"time": ["2026-10-31T00:00"], "wind_speed_10m": ["fast"], "cape": [True]}}
    forecast = fetch_forecast(LAT, LON, START, END, client=client_serving(json.dumps(body)), now=NOW)
    assert forecast.status.state is SourceState.OK
    only = forecast.samples[datetime(2026, 10, 31, 0, 0, tzinfo=UTC)]
    assert only.wind_10m is None and only.cape is None
    assert forecast.elevation_m is None


def test_an_empty_hourly_block_is_ok_with_no_samples():
    forecast = fetch_forecast(LAT, LON, START, END, client=client_serving(json.dumps({"hourly": {"time": []}})), now=NOW)
    assert forecast.status.state is SourceState.OK
    assert forecast.samples == {}


# --- failures: nothing is raised, a one-line reason and the time tried --------------


@pytest.mark.parametrize("status", [400, 404, 429, 500, 503])
def test_an_http_error_is_a_failed_status_with_the_code(status):
    forecast = fetch_forecast(LAT, LON, START, END, client=client_serving("{}", status=status), now=NOW)
    assert forecast.status.state is SourceState.FAILED
    assert forecast.status.reason == f"HTTP {status}"
    assert forecast.status.fetched_at == NOW
    assert forecast.status.source is SourceName.WEATHER
    assert forecast.samples == {} and forecast.elevation_m is None


def test_a_timeout_is_a_failed_status():
    forecast = fetch_forecast(LAT, LON, START, END, client=client_raising(httpx.ReadTimeout("slow")), now=NOW)
    assert forecast.status.state is SourceState.FAILED
    assert forecast.status.reason == "request timed out"
    assert forecast.status.fetched_at == NOW


def test_a_connection_error_is_a_failed_status():
    forecast = fetch_historical_forecast(LAT, LON, START, END, client=client_raising(httpx.ConnectError("down")), now=NOW)
    assert forecast.status.state is SourceState.FAILED
    assert forecast.status.reason == "request failed: ConnectError"


@pytest.mark.parametrize(
    ("body", "reason"),
    [
        ("<html>oops</html>", "response is not JSON"),
        ("", "response is not JSON"),
        ('{"error": true}', "response has no hourly data"),
        ("[1, 2]", "response has no hourly data"),
        ('{"hourly": []}', "response has no hourly data"),
        ('{"hourly": {"wind_speed_10m": [1]}}', "response has no hourly data"),
        ('{"hourly": {"time": ["not a time"]}}', "response has unreadable hourly times"),
        ('{"hourly": {"time": [5]}}', "response has unreadable hourly times"),
        ('{"hourly": {"time": ["2026-10-31T00:00"], "cape": [1, 2]}}', "hourly cape does not match the times"),
        ('{"hourly": {"time": ["2026-10-31T00:00"], "cape": 5}}', "hourly cape does not match the times"),
    ],
)
def test_a_body_that_is_not_usable_is_a_failed_status(body, reason):
    # An empty body goes through the same code path as a non-JSON one.
    seen = []
    forecast = fetch_forecast(LAT, LON, START, END, client=client_serving(body or " ", seen=seen), now=NOW)
    assert len(seen) == 1
    assert forecast.status.state is SourceState.FAILED
    assert forecast.status.reason == reason
    assert "\n" not in forecast.status.reason
    assert forecast.status.fetched_at == NOW


def test_a_failed_fetch_never_raises_to_the_caller():
    for exc in (httpx.ReadTimeout("t"), httpx.ConnectError("c"), httpx.RemoteProtocolError("p"), httpx.DecodingError("d")):
        assert fetch_weather(LAT, LON, START, END, client=client_raising(exc), case=None, now=NOW).status.state is SourceState.FAILED


def test_there_is_no_staleness_rule_for_weather():
    # An old fetched-at time is simply ok: the status is decided by the fetch, not by its age.
    long_ago = datetime(2020, 1, 1, tzinfo=UTC)
    forecast = fetch_forecast(LAT, LON, START, END, client=client_serving(), now=long_ago)
    assert forecast.status.state is SourceState.OK and forecast.status.fetched_at == long_ago
