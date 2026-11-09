"""G21: GET /map/wind's data. The request Open-Meteo is sent, the u/v conversion, the leaflet-velocity records, and every way a refresh fails.

The request is read from the values the module builds; the answer comes through httpx.MockTransport (no socket opens, no live Open-Meteo).
"""

import copy
import json
import math
from datetime import UTC, datetime
from pathlib import Path

import httpx
import pytest

from flysafelz.clients.open_meteo import FORECAST_URL
from flysafelz.mapdata import wind as wind_module
from flysafelz.mapdata.cache import RefreshFailed
from flysafelz.mapdata.wind import POINTS, grid_points, grid_records, request_params, wind_grid, wind_uv

NOW = datetime(2026, 11, 8, 12, 0, tzinfo=UTC)
GRID = json.loads((Path(__file__).resolve().parents[1] / "data" / "mapdata" / "open_meteo_wind_grid.json").read_text(encoding="utf-8"))


def client(handler) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


def answering(body, status: int = 200):
    return client(lambda request: httpx.Response(status, json=body))


# --- the request --------------------------------------------------------------------------------------------------------


def test_the_grid_is_143_points_from_the_north_west_corner_row_by_row():
    points = grid_points()
    assert len(points) == POINTS == 143
    assert points[0] == (61, -8) and points[10] == (61, 2)  # the first row runs west to east
    assert points[11] == (60, -8)  # the next row is one degree south
    assert points[-1] == (49, 2)
    assert [lat for lat, _ in points[::11]] == list(range(61, 48, -1))
    assert len(set(points)) == 143


def test_the_request_is_one_multi_location_query_with_the_exact_strings():
    params = request_params()
    assert set(params) == {"latitude", "longitude", "current", "wind_speed_unit"}
    assert params["latitude"] == ",".join(str(lat) for lat in range(61, 48, -1) for _ in range(11))
    assert params["longitude"] == ",".join(str(lon) for _ in range(13) for lon in range(-8, 3))
    assert params["latitude"].startswith("61,61,61,") and params["latitude"].endswith(",49,49,49")
    assert params["longitude"].startswith("-8,-7,-6,") and params["longitude"].endswith(",0,1,2")
    assert params["current"] == "wind_speed_10m,wind_direction_10m"
    assert params["wind_speed_unit"] == "ms"
    assert len(params["latitude"].split(",")) == len(params["longitude"].split(",")) == 143


def test_wind_grid_sends_that_query_once_to_the_forecast_url():
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=GRID)

    wind_grid(client(handler), NOW)
    assert len(seen) == 1 and seen[0].method == "GET"
    assert str(seen[0].url.copy_with(query=None)) == FORECAST_URL
    assert dict(seen[0].url.params) == request_params()


# --- the conversion: one rule, u = -s sin(d), v = -s cos(d) ----------------------------------------------------------------

# (label, speed, direction from, u, v)
CONVERSION = [
    ("from the north blows south", 10, 0, 0, -10),
    ("from the east blows west", 10, 90, -10, 0),
    ("from the south blows north", 10, 180, 0, 10),
    ("from the west blows east", 10, 270, 10, 0),
    ("360 is the same as 0", 10, 360, 0, -10),
    ("from the south-west blows north-east", 10, 225, 7.071, 7.071),
    ("calm with a direction", 0, 135, 0, 0),
    ("calm with no direction", 0, None, 0, 0),
    ("a fraction of a metre a second", 0.5, 90, -0.5, 0),
]


@pytest.mark.parametrize("label,speed,direction,u,v", CONVERSION, ids=[row[0] for row in CONVERSION])
def test_u_and_v_follow_the_one_rule(label, speed, direction, u, v):
    found = wind_uv(speed, direction)
    assert found == pytest.approx((u, v), abs=1e-3)
    assert all(math.isfinite(part) for part in found)
    assert all(math.copysign(1, part) > 0 for part in found if part == 0)  # no -0.0 in the JSON


@pytest.mark.parametrize(
    "speed,direction",
    [(None, 90), (None, None), (5, None), (-1, 90), (float("nan"), 90), (5, float("nan")), (float("inf"), 0), ("5", 90), (True, 90)],
    ids=["null speed", "no speed or direction", "moving air with no direction", "negative speed", "nan speed", "nan direction", "infinite speed", "text speed", "boolean speed"],
)
def test_a_value_that_cannot_make_a_wind_is_refused(speed, direction):
    with pytest.raises(ValueError):
        wind_uv(speed, direction)


# --- the records ----------------------------------------------------------------------------------------------------------


def test_the_records_are_u_then_v_with_the_grid_header():
    records = wind_grid(answering(GRID), NOW)
    assert [(r["header"]["parameterCategory"], r["header"]["parameterNumber"]) for r in records] == [(2, 2), (2, 3)]
    for record in records:
        header = record["header"]
        assert {k: header[k] for k in ("nx", "ny", "lo1", "la1", "lo2", "la2", "dx", "dy")} == {"nx": 11, "ny": 13, "lo1": -8, "la1": 61, "lo2": 2, "la2": 49, "dx": 1, "dy": 1}
        assert header["refTime"] == "2026-11-08T12:00:00Z"
        assert len(record["data"]) == 143 and all(math.isfinite(x) for x in record["data"])


def test_the_data_follows_the_fixture_rows_north_to_south_and_each_row_west_to_east():
    u, v = (record["data"] for record in wind_grid(answering(GRID), NOW))
    for index, location in enumerate(GRID):
        speed, direction = location["current"]["wind_speed_10m"], location["current"]["wind_direction_10m"]
        assert (u[index], v[index]) == pytest.approx(wind_uv(speed, direction))
        row, column = divmod(index, 11)
        assert (location["latitude"], location["longitude"]) == (61 - row, -8 + column)  # the fixture is in the request's order


def test_a_hand_built_two_by_two_grid_has_its_header_and_its_north_to_south_west_to_east_order():
    # (lat 51, lon -1) (lat 51, lon 0) / (lat 50, lon -1) (lat 50, lon 0): the north row first, each row from the west.
    pairs = [wind_uv(10, 0), wind_uv(10, 90), wind_uv(10, 180), wind_uv(10, 270)]
    u, v = grid_records(pairs, "2026-11-08T12:00:00Z", latitudes=(51, 50), longitudes=(-1, 0))
    for record, parameter in ((u, (2, 2)), (v, (2, 3))):
        assert record["header"] == {
            "parameterCategory": parameter[0], "parameterNumber": parameter[1],
            "nx": 2, "ny": 2, "lo1": -1, "la1": 51, "lo2": 0, "la2": 50, "dx": 1, "dy": 1, "refTime": "2026-11-08T12:00:00Z",
        }  # fmt: skip
    assert u["data"] == pytest.approx([0, -10, 0, 10], abs=1e-3)
    assert v["data"] == pytest.approx([-10, 0, 10, 0], abs=1e-3)


@pytest.mark.parametrize("count", [0, 4, 142, 144])
def test_records_are_made_only_from_one_pair_per_point(count):
    with pytest.raises(ValueError):
        grid_records([(0.0, 0.0)] * count, "2026-11-08T12:00:00Z")
    assert len(grid_records([(0.0, 0.0)] * 143, "2026-11-08T12:00:00Z")[0]["data"]) == 143


def test_a_calm_point_with_no_direction_gives_zero_not_nan():
    body = copy.deepcopy(GRID)
    body[5]["current"].update(wind_speed_10m=0, wind_direction_10m=None)
    u, v = (record["data"] for record in wind_grid(answering(body), NOW))
    assert (u[5], v[5]) == (0.0, 0.0)
    assert all(math.isfinite(x) for x in u + v)
    assert "NaN" not in json.dumps([u, v])


# --- every way the refresh fails ------------------------------------------------------------------------------------------


def broken(change) -> list[dict]:
    body = copy.deepcopy(GRID)
    change(body)
    return body


def without_current(body):
    del body[40]["current"]


def null_speed(body):
    body[40]["current"]["wind_speed_10m"] = None


def null_direction_in_moving_air(body):
    body[40]["current"]["wind_direction_10m"] = None


def missing_speed_key(body):
    del body[40]["current"]["wind_speed_10m"]


def not_a_location(body):
    body[40] = "oops"


BAD_BODIES = {
    "a short list": GRID[:142],
    "a long list": GRID + GRID[:1],
    "an empty list": [],
    "one object, as a single location answers": GRID[0],
    "Open-Meteo's own error object": {"error": True, "reason": "Cannot initialize WeatherVariable"},
    "a missing current block": broken(without_current),
    "a null speed": broken(null_speed),
    "a missing speed": broken(missing_speed_key),
    "no direction in moving air": broken(null_direction_in_moving_air),
    "a location that is not an object": broken(not_a_location),
    "null": None,
}


@pytest.mark.parametrize("name", list(BAD_BODIES))
def test_a_body_that_is_not_the_whole_grid_is_a_failed_refresh(name):
    with pytest.raises(RefreshFailed) as failed:
        wind_grid(answering(BAD_BODIES[name]), NOW)
    assert "\n" not in failed.value.reason and failed.value.reason


def test_a_body_that_is_not_json_is_a_failed_refresh():
    with pytest.raises(RefreshFailed, match="not JSON"):
        wind_grid(client(lambda request: httpx.Response(200, text="<html>maintenance</html>")), NOW)


@pytest.mark.parametrize("status", [400, 429, 500, 503])
def test_an_http_error_is_a_failed_refresh_that_names_the_status(status):
    with pytest.raises(RefreshFailed, match=str(status)):
        wind_grid(answering(GRID, status), NOW)


@pytest.mark.parametrize("error", [httpx.ReadTimeout("slow"), httpx.ConnectTimeout("slow"), httpx.ConnectError("down")])
def test_a_timeout_or_a_connection_error_is_a_failed_refresh(error):
    def handler(request: httpx.Request) -> httpx.Response:
        raise error

    with pytest.raises(RefreshFailed, match=type(error).__name__):
        wind_grid(client(handler), NOW)


def test_a_reason_never_holds_the_response_text():
    with pytest.raises(RefreshFailed) as failed:
        wind_grid(answering({"error": True, "reason": "SECRET-UPSTREAM-TEXT"}), NOW)
    assert "SECRET-UPSTREAM-TEXT" not in failed.value.reason


# --- the rules.yaml fence ------------------------------------------------------------------------------------------------


def test_the_wind_modules_grid_constants_are_reviewed_named_constants_and_nothing_else_there_equals_a_rules_value(tmp_path):
    import ast

    from fence_scan import hard_coded_rule_values

    rules = {61.0, 2.0, 3.0}
    real = Path(wind_module.__file__)
    assert hard_coded_rule_values(real, ast.parse(real.read_text(encoding="utf-8")), rules) == []
    planted = tmp_path / "other.py"  # the same constants anywhere else are caught
    planted.write_text("NORTH = 61\n", encoding="utf-8")
    assert hard_coded_rule_values(planted, ast.parse(planted.read_text(encoding="utf-8")), rules) == [(1, 61)]
    unreviewed = tmp_path / "wind.py"  # a new literal in a file of that name is caught
    unreviewed.write_text("NORTH = 61\nTHRESHOLD = 3\n", encoding="utf-8")
    assert hard_coded_rule_values(unreviewed, ast.parse(unreviewed.read_text(encoding="utf-8")), rules) == [(2, 3)]
