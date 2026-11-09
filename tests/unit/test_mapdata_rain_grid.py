"""G21, G22, E3, F8: GET /map/rain-grid, the UI's fallback when RainViewer's index fails.

Two layers: the module (the request it sends, which cells it keeps, how it fails) and the real Starlette app driven by a TestClient (payload shape, the 3 h cache,
stale, 503, CORS, GET only, the files written). The app gets a temporary root, the clock the test moves and an Open-Meteo that answers through
httpx.MockTransport from the committed fixture, so no socket opens and nothing sleeps.
"""

import asyncio
import copy
import json
import subprocess
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx
import pytest
from starlette.testclient import TestClient

from flysafelz.clients.open_meteo import FORECAST_URL
from flysafelz.mapdata import rain_grid as rain_module
from flysafelz.mapdata.app import RAIN_TTL, create_app
from flysafelz.mapdata.cache import RefreshFailed
from flysafelz.mapdata.config import cache_dir
from flysafelz.mapdata.rain_grid import rain_grid, request_params
from flysafelz.mapdata.wind import POINTS, grid_points, location_params
from flysafelz.mapdata.wind import request_params as wind_request_params

ROOT = Path(__file__).resolve().parents[2]
GRID = json.loads((ROOT / "tests" / "data" / "mapdata" / "open_meteo_rain_grid.json").read_text(encoding="utf-8"))
DEV, PREVIEW, EXTRA, EVIL = "http://localhost:5173", "http://localhost:4173", "http://ui.example:8080", "http://evil.example"
ALLOW = "access-control-allow-origin"
SECOND = timedelta(seconds=1)


def client(handler) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


def answering(body, status: int = 200):
    return client(lambda request: httpx.Response(status, json=body))


def with_precipitation(position: int, value: object) -> list[dict]:
    """The fixture with one location's precipitation replaced (the key is removed when value is the string 'missing')."""
    body = copy.deepcopy(GRID)
    if value == "missing":
        del body[position]["current"]["precipitation"]
    else:
        body[position]["current"]["precipitation"] = value
    return body


# --- the request: the same points as the wind grid -----------------------------------------------------------------------


def test_the_rain_request_uses_the_wind_grids_points_through_one_shared_helper():
    rain, wind = request_params(), wind_request_params()
    assert rain["latitude"] == wind["latitude"] and rain["longitude"] == wind["longitude"]
    assert {"latitude": rain["latitude"], "longitude": rain["longitude"]} == location_params()
    assert rain_module.location_params is location_params  # imported, not rebuilt
    assert rain_module.grid_points is grid_points


def test_the_request_is_one_multi_location_query_for_current_precipitation():
    params = request_params()
    assert set(params) == {"latitude", "longitude", "current"}
    assert params["current"] == "precipitation"
    assert len(params["latitude"].split(",")) == len(params["longitude"].split(",")) == POINTS == 143
    assert params["latitude"].startswith("61,61,61,") and params["longitude"].startswith("-8,-7,-6,")
    assert params["latitude"].endswith(",49,49,49") and params["longitude"].endswith(",0,1,2")


def test_rain_grid_sends_that_query_once_to_the_forecast_url():
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=GRID)

    rain_grid(client(handler))
    assert len(seen) == 1 and seen[0].method == "GET"
    assert str(seen[0].url.copy_with(query=None)) == FORECAST_URL
    assert dict(seen[0].url.params) == request_params()


# --- the cells -----------------------------------------------------------------------------------------------------------


def test_the_cells_are_the_143_grid_points_in_order_with_their_precipitation_in_mm():
    cells = rain_grid(answering(GRID))
    assert len(cells) == 143
    assert [(c["lat"], c["lon"]) for c in cells] == grid_points()
    assert [c["precipitation"] for c in cells] == [g["current"]["precipitation"] for g in GRID]
    assert all(set(c) == {"lat", "lon", "precipitation"} for c in cells)
    assert any(c["precipitation"] == 0 for c in cells) and any(c["precipitation"] > 0 for c in cells)  # dry cells are kept, as zero


@pytest.mark.parametrize("bad", [None, "missing", "wet", -0.1, True, [], {}], ids=["null", "missing", "text", "negative", "boolean", "list", "dict"])
def test_a_location_with_no_usable_precipitation_is_left_out_and_not_drawn_as_zero(bad):
    cells = rain_grid(answering(with_precipitation(5, bad)))
    assert len(cells) == 142
    assert grid_points()[5] not in [(c["lat"], c["lon"]) for c in cells]
    assert [c["precipitation"] for c in cells] == [g["current"]["precipitation"] for i, g in enumerate(GRID) if i != 5]


def test_an_infinite_precipitation_is_left_out_too():
    text = json.dumps(with_precipitation(5, float("inf")))  # json.dumps writes Infinity, which json.loads reads back
    cells = rain_grid(client(lambda request: httpx.Response(200, content=text.encode(), headers={"content-type": "application/json"})))
    assert len(cells) == 142 and grid_points()[5] not in [(c["lat"], c["lon"]) for c in cells]


def test_a_location_with_no_current_block_or_that_is_not_an_object_is_left_out():
    body = copy.deepcopy(GRID)
    del body[0]["current"]
    body[1] = "nonsense"
    body[2]["current"] = None
    cells = rain_grid(answering(body))
    assert len(cells) == 140 and (cells[0]["lat"], cells[0]["lon"]) == grid_points()[3]


def test_a_cell_takes_its_place_from_the_list_not_from_the_coordinates_the_answer_echoes():
    body = copy.deepcopy(GRID)
    body[0]["latitude"], body[0]["longitude"] = 0.0, 0.0
    assert (rain_grid(answering(body))[0]["lat"], rain_grid(answering(body))[0]["lon"]) == (61, -8)


# --- every way a refresh fails ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "make",
    [
        lambda: answering(GRID, 500),
        lambda: answering(GRID, 429),
        lambda: client(lambda request: (_ for _ in ()).throw(httpx.ReadTimeout("slow"))),
        lambda: client(lambda request: httpx.Response(200, text="<html>not json</html>")),
        lambda: answering({"error": True, "reason": "Latitude must be in range"}),
        lambda: answering(GRID[:142]),
        lambda: answering([*GRID, GRID[0]]),
        lambda: answering(["x"] * 143),
        lambda: answering([{"current": {"precipitation": None}}] * 143),
    ],
    ids=["HTTP 500", "HTTP 429", "timeout", "not JSON", "error object", "short list", "long list", "no objects", "no value anywhere"],
)
def test_a_refresh_that_cannot_give_the_grid_is_refused_with_a_one_line_reason(make):
    with pytest.raises(RefreshFailed) as failure:
        rain_grid(make())
    reason = failure.value.reason
    assert reason.startswith("Open-Meteo") and "\n" not in reason
    assert "<html>" not in reason and "Latitude must be" not in reason  # a reason never holds the response text


# --- the route -------------------------------------------------------------------------------------------------------------

NOW = datetime(2026, 11, 8, 12, 0, tzinfo=UTC)


class Clock:
    def __init__(self) -> None:
        self.now = NOW

    def __call__(self) -> datetime:
        return self.now

    def advance(self, by: timedelta) -> None:
        self.now += by


class OpenMeteo:
    """A fake Open-Meteo: the fixture, or an error, a timeout or a body, and a count of the requests it got."""

    def __init__(self) -> None:
        self.requests: list[httpx.Request] = []
        self.status = 200
        self.body: object = GRID
        self.error: Exception | None = None
        self.delay = 0.0

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if self.delay:
            time.sleep(self.delay)
        if self.error is not None:
            raise self.error
        return httpx.Response(self.status, json=self.body)

    def factory(self) -> httpx.Client:
        return httpx.Client(transport=httpx.MockTransport(self))


class Setup:
    def __init__(self, tmp_path: Path, settings: dict[str, str] | None = None) -> None:
        self.root = tmp_path
        self.clock = Clock()
        self.upstream = OpenMeteo()
        self.client = TestClient(create_app(root=tmp_path, clock=self.clock, settings=settings or {}, client_factory=self.upstream.factory))

    def get(self, **headers: str) -> httpx.Response:
        return self.client.get("/map/rain-grid", headers=headers)

    @property
    def files(self) -> list[str]:
        return sorted(p.relative_to(self.root).as_posix() for p in self.root.rglob("*") if p.is_file())


@pytest.fixture
def setup(tmp_path) -> Setup:
    return Setup(tmp_path)


def test_the_route_answers_200_with_the_route_payload(setup):
    response = setup.get()
    body = response.json()
    assert response.status_code == 200 and response.headers["content-type"].startswith("application/json")
    assert set(body) == {"grid", "updated_at", "stale"}  # the Route payload, `grid` where the feature routes have `features`
    assert (body["updated_at"], body["stale"]) == ("2026-11-08T12:00:00Z", False)
    assert len(body["grid"]) == 143 and set(body["grid"][0]) == {"lat", "lon", "precipitation"}
    assert [(c["lat"], c["lon"]) for c in body["grid"]] == grid_points()


def test_the_route_makes_one_multi_location_request_with_the_wind_grids_points(setup):
    setup.get()
    assert len(setup.upstream.requests) == 1
    params = setup.upstream.requests[0].url.params
    assert (params["latitude"], params["longitude"]) == (wind_request_params()["latitude"], wind_request_params()["longitude"])
    assert params["current"] == "precipitation"


def test_cells_with_no_value_are_missing_from_the_payload_not_zero(setup):
    setup.upstream.body = with_precipitation(7, None)
    cells = setup.get().json()["grid"]
    assert len(cells) == 142 and grid_points()[7] not in [(c["lat"], c["lon"]) for c in cells]


# --- the 3 h cache (frozen clock) -----------------------------------------------------------------------------------------


def test_the_cache_lasts_exactly_three_hours(setup, now):
    assert RAIN_TTL == timedelta(hours=3)
    first = setup.get().json()
    setup.clock.advance(RAIN_TTL - SECOND)
    repeats = [setup.get().json() for _ in range(4)]
    assert len(setup.upstream.requests) == 1 and all(again == first for again in repeats)
    setup.clock.advance(SECOND)  # the age is now exactly 3 h: expired
    refreshed = setup.get().json()
    assert len(setup.upstream.requests) == 2
    assert refreshed["updated_at"] == "2026-11-08T15:00:00Z" and refreshed["stale"] is False
    assert now.tzinfo is not None  # the shared frozen clock from tests/conftest.py is the one this module's NOW sits beside


def test_the_route_runs_on_the_shared_frozen_clock(tmp_path, now):
    clock = Clock()
    clock.now = now
    app = TestClient(create_app(root=tmp_path, clock=clock, settings={}, client_factory=OpenMeteo().factory))
    assert app.get("/map/rain-grid").json()["updated_at"] == now.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def test_requests_that_arrive_during_a_refresh_share_one_upstream_call(tmp_path):
    slow = Setup(tmp_path)
    slow.upstream.delay = 0.2

    async def go() -> list[httpx.Response]:
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=slow.client.app), base_url="http://test") as http:
            return await asyncio.gather(*(http.get("/map/rain-grid") for _ in range(6)))

    responses = asyncio.run(go())
    assert len(slow.upstream.requests) == 1 and {r.status_code for r in responses} == {200}


def test_the_cache_file_is_the_rain_grid_file_under_data_map_and_that_folder_is_ignored(setup):
    setup.get()
    assert setup.files == ["data/map/rain-grid.json"]
    assert (cache_dir(setup.root) / "rain-grid.json").is_file()
    assert set(json.loads((cache_dir(setup.root) / "rain-grid.json").read_text(encoding="utf-8"))) == {"updated_at", "grid"}
    assert subprocess.run(["git", "check-ignore", "-q", "data/map/rain-grid.json"], cwd=ROOT, check=False).returncode == 0


def test_the_rain_cache_is_not_the_wind_cache(setup):
    setup.get()
    setup.client.get("/map/wind")  # the same fake Open-Meteo answers with rain-shaped locations: wind fails, rain's cache stays
    assert (cache_dir(setup.root) / "rain-grid.json").is_file() and not (cache_dir(setup.root) / "wind.json").exists()


def test_a_fresh_cache_file_on_disk_is_served_without_an_upstream_call(tmp_path):
    original = Setup(tmp_path).get().json()
    restarted = Setup(tmp_path)  # a new process, the same data/map/ folder
    restarted.clock.advance(timedelta(hours=1))
    assert restarted.get().json() == original and restarted.upstream.requests == []


# --- failure: stale with a cache, 503 without --------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "fail",
    [
        lambda up: setattr(up, "status", 500),
        lambda up: setattr(up, "error", httpx.ReadTimeout("slow")),
        lambda up: setattr(up, "body", GRID[:50]),
        lambda up: setattr(up, "body", {"error": True}),
        lambda up: setattr(up, "body", [{"current": {}}] * 143),
    ],
    ids=["HTTP 500", "timeout", "short list", "malformed", "no value anywhere"],
)
def test_a_failed_refresh_with_a_cache_serves_the_last_grid_stale(setup, fail):
    good = setup.get().json()
    setup.clock.advance(RAIN_TTL + SECOND)
    fail(setup.upstream)
    served = setup.get()
    body = served.json()
    assert served.status_code == 200 and body["stale"] is True
    assert body["grid"] == good["grid"] and body["updated_at"] == good["updated_at"]
    assert len(setup.upstream.requests) == 2
    assert json.loads((cache_dir(setup.root) / "rain-grid.json").read_text(encoding="utf-8"))["updated_at"] == good["updated_at"]  # the file is untouched


@pytest.mark.parametrize(
    "fail",
    [
        lambda up: setattr(up, "status", 500),
        lambda up: setattr(up, "error", httpx.ConnectError("down")),
        lambda up: setattr(up, "body", GRID[:142]),
        lambda up: setattr(up, "body", "garbage"),
    ],
    ids=["HTTP 500", "network error", "short list", "malformed"],
)
def test_a_failed_refresh_with_no_cache_is_a_503_with_a_one_line_reason_and_no_file(setup, fail):
    fail(setup.upstream)
    response = setup.get()
    assert response.status_code == 503
    body = response.json()
    assert set(body) == {"reason"} and body["reason"] and "\n" not in body["reason"]
    assert setup.files == []


def test_the_route_recovers_on_the_next_request_after_a_failure(setup):
    setup.upstream.status = 503
    assert setup.get().status_code == 503
    setup.upstream.status = 200
    assert setup.get().status_code == 200


# --- GET only (F8) -------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("method", ["POST", "PUT", "PATCH", "DELETE"])
def test_anything_but_get_is_a_405_with_no_upstream_call_and_no_file(setup, method):
    response = setup.client.request(method, "/map/rain-grid")
    assert response.status_code == 405 and "GET" in response.headers["allow"]
    assert setup.upstream.requests == [] and setup.files == []


def test_head_is_answered_like_get_without_a_second_class_of_write(setup):
    assert setup.client.head("/map/rain-grid").status_code == 200
    assert setup.files == ["data/map/rain-grid.json"]


# --- CORS (G22): the two dev origins plus FLYSAFELZ_UI_ORIGIN, nothing else -------------------------------------------------------


@pytest.mark.parametrize("origin", [DEV, PREVIEW])
def test_the_two_dev_ui_origins_are_allowed(setup, origin):
    assert setup.get(Origin=origin).headers[ALLOW] == origin
    preflight = setup.client.options("/map/rain-grid", headers={"Origin": origin, "Access-Control-Request-Method": "GET"})
    assert preflight.status_code == 200 and preflight.headers[ALLOW] == origin


def test_any_other_origin_gets_no_allow_header_and_there_is_no_wildcard(setup):
    assert ALLOW not in setup.get(Origin=EVIL).headers
    assert ALLOW not in setup.get().headers
    assert ALLOW not in setup.client.options("/map/rain-grid", headers={"Origin": EVIL, "Access-Control-Request-Method": "GET"}).headers
    assert setup.get(Origin=DEV).headers[ALLOW] != "*"


def test_flysafelz_ui_origin_adds_one_origin_and_the_others_stay_refused(tmp_path):
    configured = Setup(tmp_path, {"FLYSAFELZ_UI_ORIGIN": EXTRA})
    assert configured.get(Origin=EXTRA).headers[ALLOW] == EXTRA
    assert configured.get(Origin=DEV).headers[ALLOW] == DEV
    assert ALLOW not in configured.get(Origin=EVIL).headers
