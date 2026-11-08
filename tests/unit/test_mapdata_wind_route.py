"""G21, G22, E1, F8: GET /map/wind through the real Starlette app, driven by a Starlette TestClient.

The app is create_app with a temporary root (every file it writes lands in tmp_path), a clock the test moves and an Open-Meteo that answers
through httpx.MockTransport from the committed 143-location fixture. Nothing opens a socket and nothing sleeps.
"""

import asyncio
import json
import subprocess
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx
import pytest
from starlette.testclient import TestClient

from flysafelz.mapdata.app import WIND_TTL, create_app
from flysafelz.mapdata.config import cache_dir

ROOT = Path(__file__).resolve().parents[2]
GRID = json.loads((ROOT / "tests" / "data" / "mapdata" / "open_meteo_wind_grid.json").read_text(encoding="utf-8"))
NOW = datetime(2026, 11, 8, 12, 0, tzinfo=UTC)
DEV, PREVIEW, EVIL = "http://localhost:5173", "http://localhost:4173", "http://evil.example"
ALLOW = "access-control-allow-origin"
SECOND = timedelta(seconds=1)


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
    def __init__(self, tmp_path: Path) -> None:
        self.root = tmp_path
        self.clock = Clock()
        self.upstream = OpenMeteo()
        self.client = TestClient(create_app(root=tmp_path, clock=self.clock, settings={}, client_factory=self.upstream.factory))

    def get(self, **headers: str) -> httpx.Response:
        return self.client.get("/map/wind", headers=headers)

    @property
    def files(self) -> list[str]:
        return sorted(p.relative_to(self.root).as_posix() for p in self.root.rglob("*") if p.is_file())


@pytest.fixture
def setup(tmp_path) -> Setup:
    return Setup(tmp_path)


# --- 200 ------------------------------------------------------------------------------------------------------------------


def test_the_route_answers_200_with_the_grid_updated_at_and_stale(setup):
    response = setup.get()
    body = response.json()
    assert response.status_code == 200 and response.headers["content-type"].startswith("application/json")
    assert set(body) == {"grid", "updated_at", "stale"}  # the Route payload with `grid` where the others have `features`
    assert (body["updated_at"], body["stale"]) == ("2026-11-08T12:00:00Z", False)
    u, v = body["grid"]
    assert [(r["header"]["parameterCategory"], r["header"]["parameterNumber"]) for r in body["grid"]] == [(2, 2), (2, 3)]
    assert u["header"]["nx"] == 11 and u["header"]["ny"] == 13 and len(u["data"]) == len(v["data"]) == 143
    assert u["header"]["refTime"] == "2026-11-08T12:00:00Z"


def test_the_route_makes_one_multi_location_request_to_open_meteo(setup):
    setup.get()
    assert len(setup.upstream.requests) == 1
    params = setup.upstream.requests[0].url.params
    assert len(params["latitude"].split(",")) == len(params["longitude"].split(",")) == 143
    assert (params["current"], params["wind_speed_unit"]) == ("wind_speed_10m,wind_direction_10m", "ms")


def test_only_the_wind_route_is_added_and_there_is_no_rain_grid_route(setup):
    assert setup.client.get("/map/rain-grid").status_code == 404
    assert setup.upstream.requests == []


# --- the 3 h cache ------------------------------------------------------------------------------------------------------


def test_many_requests_inside_three_hours_make_one_upstream_call(setup):
    first = setup.get().json()
    setup.clock.advance(WIND_TTL - SECOND)
    repeats = [setup.get().json() for _ in range(5)]
    assert len(setup.upstream.requests) == 1
    assert all(again == first for again in repeats) and first["updated_at"] == "2026-11-08T12:00:00Z"


def test_the_cache_expires_at_exactly_three_hours(setup):
    assert WIND_TTL == timedelta(hours=3)
    setup.get()
    setup.clock.advance(WIND_TTL - SECOND)
    setup.get()
    assert len(setup.upstream.requests) == 1
    setup.clock.advance(SECOND)  # age is now exactly 3 h: expired
    refreshed = setup.get().json()
    assert len(setup.upstream.requests) == 2
    assert refreshed["updated_at"] == "2026-11-08T15:00:00Z" and refreshed["stale"] is False


def test_requests_that_arrive_during_a_refresh_share_one_upstream_call(tmp_path):
    slow = Setup(tmp_path)
    slow.upstream.delay = 0.2

    async def go() -> list[httpx.Response]:
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=slow.client.app), base_url="http://test") as client:
            return await asyncio.gather(*(client.get("/map/wind") for _ in range(8)))

    responses = asyncio.run(go())
    assert len(slow.upstream.requests) == 1
    assert {r.status_code for r in responses} == {200} and len({r.text for r in responses}) == 1


def test_the_cache_file_is_the_wind_file_under_data_map_and_that_folder_is_ignored(setup):
    setup.get()
    assert setup.files == ["data/map/wind.json"]
    assert (cache_dir(setup.root) / "wind.json").is_file()
    cached = json.loads((cache_dir(setup.root) / "wind.json").read_text(encoding="utf-8"))
    assert set(cached) == {"updated_at", "grid"}
    ignored = subprocess.run(["git", "check-ignore", "-q", "data/map/wind.json"], cwd=ROOT, check=False)
    assert ignored.returncode == 0  # data/map/ is gitignored: the wind file is never committed


def test_a_fresh_cache_file_on_disk_is_served_without_an_upstream_call(tmp_path):
    first = Setup(tmp_path)
    original = first.get().json()
    restarted = Setup(tmp_path)  # a new process, the same data/map/ folder
    restarted.clock.advance(timedelta(hours=1))
    assert restarted.get().json() == original and restarted.upstream.requests == []


# --- failure: stale, then 503 -----------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "fail",
    [
        lambda up: setattr(up, "status", 500),
        lambda up: setattr(up, "error", httpx.ReadTimeout("slow")),
        lambda up: setattr(up, "body", GRID[:50]),
        lambda up: setattr(up, "body", [{**g, "current": {**g["current"], "wind_speed_10m": None}} for g in GRID]),
    ],
    ids=["HTTP 500", "timeout", "short grid", "null speeds"],
)
def test_a_failed_refresh_with_a_cache_serves_the_last_grid_stale(setup, fail):
    good = setup.get().json()
    setup.clock.advance(WIND_TTL + SECOND)
    fail(setup.upstream)
    served = setup.get()
    body = served.json()
    assert served.status_code == 200 and body["stale"] is True
    assert body["grid"] == good["grid"] and body["updated_at"] == good["updated_at"]
    assert len(setup.upstream.requests) == 2


@pytest.mark.parametrize("fail", [lambda up: setattr(up, "status", 500), lambda up: setattr(up, "error", httpx.ReadTimeout("slow")), lambda up: setattr(up, "body", GRID[:142])])
def test_a_failed_refresh_with_no_cache_is_a_503_with_a_one_line_reason_and_no_file(setup, fail):
    fail(setup.upstream)
    response = setup.get()
    assert response.status_code == 503
    body = response.json()
    assert set(body) == {"reason"} and body["reason"] and "\n" not in body["reason"]
    assert setup.files == []


def test_a_wind_failure_does_not_touch_the_other_routes_and_recovers_on_the_next_request(setup):
    setup.upstream.status = 503
    assert setup.get().status_code == 503
    setup.upstream.status = 200
    assert setup.get().status_code == 200


# --- GET only (F8) ----------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("method", ["POST", "PUT", "PATCH", "DELETE"])
def test_anything_but_get_is_a_405_with_no_upstream_call_and_no_file(setup, method):
    response = setup.client.request(method, "/map/wind")
    assert response.status_code == 405 and "GET" in response.headers["allow"]
    assert setup.upstream.requests == [] and setup.files == []


# --- CORS (G22) ---------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("origin", [DEV, PREVIEW])
def test_the_two_dev_ui_origins_are_allowed(setup, origin):
    assert setup.get(Origin=origin).headers[ALLOW] == origin
    preflight = setup.client.options("/map/wind", headers={"Origin": origin, "Access-Control-Request-Method": "GET"})
    assert preflight.status_code == 200 and preflight.headers[ALLOW] == origin


def test_any_other_origin_gets_no_allow_header_and_there_is_no_wildcard(setup):
    assert ALLOW not in setup.get(Origin=EVIL).headers
    assert ALLOW not in setup.get().headers
    preflight = setup.client.options("/map/wind", headers={"Origin": EVIL, "Access-Control-Request-Method": "GET"})
    assert ALLOW not in preflight.headers
    assert setup.get(Origin=DEV).headers[ALLOW] != "*"
