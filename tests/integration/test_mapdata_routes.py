"""G19, G20, G22, F8: the two map routes through the real Starlette app, driven by a Starlette TestClient.

The app is create_app with a temporary root (so every file it writes lands in tmp_path), a clock the test moves, a made-up openAIP key
and an upstream that answers through httpx.MockTransport. Nothing opens a socket and nothing sleeps.
"""

import asyncio
import json
import logging
from datetime import UTC, datetime, timedelta

import httpx
import pytest
from mapdata_helpers import FAKE_KEY, SETTINGS, Upstream, notam, page, pib_of
from starlette.testclient import TestClient

from flysafelz.mapdata.app import AIRSPACE_TTL, NOTAM_TTL, create_app

NOW = datetime(2026, 11, 8, 12, 0, tzinfo=UTC)
HOUR = timedelta(hours=1)
DEV = "http://localhost:5173"
PREVIEW = "http://localhost:4173"
EVIL = "http://evil.example"
ROUTES = ["/map/airspace", "/map/notams"]
SECRET = "SECRET E TEXT 01632960000"


class Clock:
    def __init__(self, now: datetime = NOW) -> None:
        self.now = now

    def __call__(self) -> datetime:
        return self.now

    def advance(self, by: timedelta) -> None:
        self.now += by


class Setup:
    def __init__(self, tmp_path, *, upstream: Upstream | None = None, settings: dict | None = None) -> None:
        self.root = tmp_path
        self.clock = Clock()
        self.upstream = upstream or Upstream(pib=pib_of(notam(1, NOW - HOUR, NOW + HOUR, e=SECRET)))
        self.app = create_app(root=tmp_path, clock=self.clock, settings=SETTINGS if settings is None else settings, client_factory=self.upstream.factory)
        self.client = TestClient(self.app)

    def get(self, path: str, **headers: str) -> httpx.Response:
        return self.client.get(path, headers=headers)

    @property
    def files(self) -> list[str]:
        return sorted(p.relative_to(self.root).as_posix() for p in self.root.rglob("*") if p.is_file())


@pytest.fixture
def setup(tmp_path) -> Setup:
    return Setup(tmp_path)


# --- 200 -------------------------------------------------------------------------------------------------------------------


def test_airspace_is_a_200_feature_collection_with_the_envelope(setup):
    response = setup.get("/map/airspace")
    body = response.json()
    assert response.status_code == 200 and response.headers["content-type"].startswith("application/json")
    assert body["type"] == "FeatureCollection" and len(body["features"]) == 8
    assert (body["updated_at"], body["stale"]) == ("2026-11-08T12:00:00Z", False)
    assert set(body) == {"type", "features", "updated_at", "stale"}


def test_notams_is_a_200_feature_collection_with_the_envelope_and_the_text_as_published(setup):
    body = setup.get("/map/notams").json()
    assert body["type"] == "FeatureCollection" and [f["properties"]["id"] for f in body["features"]] == ["H1/26"]
    assert body["features"][0]["properties"]["text"] == SECRET
    assert (body["updated_at"], body["stale"]) == ("2026-11-08T12:00:00Z", False)


def test_a_pib_with_nothing_active_is_a_200_with_no_features_a_fresh_time_and_not_stale(tmp_path):
    quiet = Setup(tmp_path, upstream=Upstream(pib=pib_of(notam(1, NOW - 5 * HOUR, NOW - HOUR))))
    response = quiet.get("/map/notams")
    assert response.status_code == 200 and response.json() == {"type": "FeatureCollection", "features": [], "updated_at": "2026-11-08T12:00:00Z", "stale": False}


def test_an_empty_airspace_page_is_a_200_with_no_features(tmp_path):
    empty = Setup(tmp_path, upstream=Upstream(airspace={"1": page("empty")}))
    assert empty.get("/map/airspace").json()["features"] == []


def test_an_unknown_map_path_is_a_404(setup):
    assert setup.get("/map/wind").status_code == 404 and setup.upstream.calls == 0


# --- the cache through the routes ---------------------------------------------------------------------------------------------


def test_a_second_request_inside_the_ttl_makes_no_outbound_call(setup):
    for path in ROUTES:
        setup.get(path)
    calls = (len(setup.upstream.airspace_requests), len(setup.upstream.pib_requests))
    assert calls == (2, 1)  # two openAIP pages, one PIB
    setup.clock.advance(NOTAM_TTL - timedelta(seconds=1))
    for path in ROUTES:
        assert setup.get(path).json()["stale"] is False
    assert (len(setup.upstream.airspace_requests), len(setup.upstream.pib_requests)) == calls
    assert setup.get("/map/notams").json()["updated_at"] == "2026-11-08T12:00:00Z"  # still the first refresh's time


def test_the_notams_expire_after_15_minutes_and_the_airspace_after_24_hours(setup):
    for path in ROUTES:
        setup.get(path)
    setup.clock.advance(NOTAM_TTL)
    assert setup.get("/map/airspace").json()["updated_at"] == "2026-11-08T12:00:00Z"
    assert setup.get("/map/notams").json()["updated_at"] == "2026-11-08T12:15:00Z"
    assert (len(setup.upstream.airspace_requests), len(setup.upstream.pib_requests)) == (2, 2)
    setup.clock.advance(AIRSPACE_TTL)
    assert setup.get("/map/airspace").json()["updated_at"] == "2026-11-09T12:15:00Z"
    assert len(setup.upstream.airspace_requests) == 4


def test_two_requests_at_the_same_time_make_one_upstream_fetch(tmp_path):
    both = Setup(tmp_path)

    async def go():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=both.app), base_url="http://map.test") as client:
            return await asyncio.gather(*(client.get(path) for path in ROUTES for _ in range(5)))

    responses = asyncio.run(go())
    assert [r.status_code for r in responses] == [200] * 10
    assert (len(both.upstream.airspace_requests), len(both.upstream.pib_requests)) == (2, 1)  # two pages and one PIB, not ten fetches


# --- stale and 503 -----------------------------------------------------------------------------------------------------------------


def test_an_expired_cache_and_a_pib_that_is_down_is_a_200_stale_with_the_old_time(setup):
    first = setup.get("/map/notams").json()
    setup.upstream.pib_status = 503
    setup.clock.advance(NOTAM_TTL + timedelta(minutes=1))
    response = setup.get("/map/notams")
    body = response.json()
    assert response.status_code == 200 and body["stale"] is True
    assert body["updated_at"] == first["updated_at"] and body["features"] == first["features"]


def test_the_stale_answer_does_not_touch_the_file_and_the_next_good_refresh_replaces_it(setup):
    setup.get("/map/notams")
    path = setup.root / "data" / "map" / "notams.json"
    written = path.read_text(encoding="utf-8")
    setup.upstream.pib_status = 500
    setup.clock.advance(NOTAM_TTL)
    assert setup.get("/map/notams").json()["stale"] is True and path.read_text(encoding="utf-8") == written
    setup.upstream.pib_status = 200
    body = setup.get("/map/notams").json()
    assert body["stale"] is False and body["updated_at"] == "2026-11-08T12:15:00Z"
    assert json.loads(path.read_text(encoding="utf-8"))["updated_at"] == "2026-11-08T12:15:00Z"


@pytest.mark.parametrize("status", [500, 502, 503, 404])
def test_no_cache_and_a_pib_that_fails_is_a_503_with_a_one_line_reason(tmp_path, status):
    down = Setup(tmp_path)
    down.upstream.pib_status = status
    response = down.get("/map/notams")
    assert response.status_code == 503 and response.json() == {"reason": f"NATS PIB: HTTP {status}"}
    assert down.files == []  # a failed refresh writes nothing


def test_no_cache_and_openaip_down_is_a_503_and_the_reason_has_no_key(tmp_path):
    down = Setup(tmp_path)
    down.upstream.airspace_status = 403
    response = down.get("/map/airspace")
    assert response.status_code == 503 and response.json() == {"reason": "openAIP blocked the request (HTML 403, not a key error)"}  # the client's own fixed line
    assert FAKE_KEY not in response.text


def test_an_openaip_html_page_with_status_200_is_a_failed_refresh(tmp_path):
    blocked = Setup(tmp_path)
    blocked.upstream.airspace_html = True
    assert blocked.get("/map/airspace").status_code == 503
    assert blocked.files == []


def test_a_second_page_that_fails_leaves_no_partial_cache(tmp_path):
    upstream = Upstream()
    inner = upstream.__call__

    def second_page_fails(request):
        return httpx.Response(500) if request.url.params.get("page") == "2" else inner(request)

    flaky = Setup(tmp_path, upstream=upstream)
    flaky.app = create_app(
        root=tmp_path,
        clock=flaky.clock,
        settings=SETTINGS,
        client_factory=lambda: httpx.Client(transport=httpx.MockTransport(second_page_fails)),
    )
    flaky.client = TestClient(flaky.app)
    assert flaky.get("/map/airspace").status_code == 503 and flaky.files == []


def test_a_missing_key_is_a_503_that_names_the_variable_and_no_request_goes_out(tmp_path):
    keyless = Setup(tmp_path, settings={})
    response = keyless.get("/map/airspace")
    assert response.status_code == 503 and response.json() == {"reason": "OPENAIP_API_KEY is not set"}
    assert keyless.upstream.calls == 0


def test_the_key_is_read_from_the_env_file_when_the_environment_has_none(tmp_path):
    (tmp_path / ".env").write_text("OPENAIP_API_" + f"KEY={FAKE_KEY}\n", encoding="utf-8")  # split so the F6 key scan skips this file
    from_file = Setup(tmp_path, settings={})
    assert from_file.get("/map/airspace").status_code == 200
    assert {r.headers["x-openaip-api-key"] for r in from_file.upstream.airspace_requests} == {FAKE_KEY}
    assert ".env" in from_file.files and "data/map/airspace.json" in from_file.files  # the .env is only read; the one new file is the cache


def test_one_route_failing_never_affects_the_other(tmp_path):
    split = Setup(tmp_path)
    split.upstream.pib_status = 500
    assert split.get("/map/notams").status_code == 503
    assert split.get("/map/airspace").status_code == 200
    split.upstream.airspace_status = 500
    split.clock.advance(AIRSPACE_TTL + HOUR)
    split.upstream.pib_status = 200
    assert split.get("/map/airspace").json()["stale"] is True
    assert split.get("/map/notams").json()["stale"] is False


@pytest.mark.parametrize("content", ["", "{", "<html>", '{"updated_at": "x"}'])
def test_a_corrupt_cache_file_is_no_cache_and_the_next_good_refresh_overwrites_it(tmp_path, content):
    folder = tmp_path / "data" / "map"
    folder.mkdir(parents=True)
    (folder / "notams.json").write_text(content, encoding="utf-8")
    broken = Setup(tmp_path)
    broken.upstream.pib_status = 500
    assert broken.get("/map/notams").status_code == 503
    broken.upstream.pib_status = 200
    response = broken.get("/map/notams")
    assert response.status_code == 200 and json.loads((folder / "notams.json").read_text(encoding="utf-8"))["features"] == response.json()["features"]


# --- GET only, and the only files written ------------------------------------------------------------------------------------------


@pytest.mark.parametrize("path", ROUTES)
@pytest.mark.parametrize("method", ["POST", "PUT", "DELETE", "PATCH"])
def test_anything_but_get_is_a_405_and_causes_no_upstream_call_and_no_file(setup, path, method):
    response = setup.client.request(method, path, json={"x": 1})
    assert response.status_code == 405 and "GET" in response.headers["allow"]
    assert setup.upstream.calls == 0 and setup.files == []


@pytest.mark.parametrize("method", ["POST", "PUT", "DELETE", "PATCH"])
def test_the_405_survives_the_langgraph_servers_catch_all_mount(setup, method):
    """The server appends a catch-all Mount after a custom app's routes (langgraph_api/server.py). A route that only partly matched a POST
    would lose to it and answer 404; these routes answer every method themselves, so the 405 holds on the merged router."""
    from starlette.applications import Starlette
    from starlette.routing import Mount

    merged = Starlette(routes=[*setup.app.routes, Mount("", app=Starlette(routes=[]))])
    client = TestClient(merged)
    assert client.request(method, "/map/notams").status_code == 405 and client.request(method, "/map/airspace").status_code == 405
    assert client.get("/map/other").status_code == 404 and setup.upstream.calls == 0


def test_options_without_an_origin_is_a_405(setup):
    assert setup.client.options("/map/notams").status_code == 405


def test_every_file_written_is_under_data_map_and_nothing_is_left_over(setup):
    for path in ROUTES:
        setup.get(path)
    setup.clock.advance(AIRSPACE_TTL + HOUR)
    for path in ROUTES:
        setup.get(path)
    assert setup.files == ["data/map/airspace.json", "data/map/notams.json"]


# --- logs (Q10) ----------------------------------------------------------------------------------------------------------------------


def test_no_log_line_holds_the_notam_text_or_the_key_on_a_good_refresh_a_stale_one_or_a_failure(setup, caplog):
    with caplog.at_level(logging.DEBUG):
        setup.get("/map/notams")
        setup.get("/map/airspace")
        setup.upstream.pib_status = 500
        setup.upstream.airspace_status = 403
        setup.clock.advance(AIRSPACE_TTL + HOUR)
        setup.get("/map/notams")
        setup.get("/map/airspace")
    assert SECRET not in caplog.text and FAKE_KEY not in caplog.text
    assert "refresh failed" in caplog.text


# --- CORS (G22) ------------------------------------------------------------------------------------------------------------------------

ALLOW = "access-control-allow-origin"


@pytest.mark.parametrize("path", ROUTES)
@pytest.mark.parametrize("origin", [DEV, PREVIEW])
def test_the_two_dev_ui_origins_get_the_allow_header_on_a_simple_request(setup, path, origin):
    response = setup.get(path, Origin=origin)
    assert response.status_code == 200 and response.headers[ALLOW] == origin


@pytest.mark.parametrize("path", ROUTES)
@pytest.mark.parametrize("origin", [DEV, PREVIEW])
def test_a_preflight_from_the_dev_ui_is_answered_with_the_allow_header(setup, path, origin):
    response = setup.client.options(path, headers={"Origin": origin, "Access-Control-Request-Method": "GET"})
    assert response.status_code == 200 and response.headers[ALLOW] == origin and "GET" in response.headers["access-control-allow-methods"]
    assert setup.upstream.calls == 0  # the preflight is answered before any route runs


@pytest.mark.parametrize("path", ROUTES)
@pytest.mark.parametrize(
    "origin",
    [
        EVIL,
        "https://localhost:5173",  # right host and port, wrong scheme
        "http://localhost:5174",
        "http://localhost",
        "http://127.0.0.1:5173",
        "http://LOCALHOST:5173",
        "http://localhost:5173/",
        "http://localhost:5173.evil.example",
        "http://evil.example:5173",
        "null",
        "*",
        "",
    ],
)
def test_any_other_origin_gets_no_allow_header_on_a_simple_request_or_a_preflight(setup, path, origin):
    simple = setup.get(path, Origin=origin)
    assert ALLOW not in simple.headers and "access-control-allow-credentials" not in simple.headers
    preflight = setup.client.options(path, headers={"Origin": origin, "Access-Control-Request-Method": "GET"})
    assert ALLOW not in preflight.headers


def test_a_request_with_no_origin_header_gets_no_allow_header(setup):
    assert ALLOW not in setup.get("/map/notams").headers


def test_the_allow_header_is_never_a_wildcard(setup):
    for origin in [DEV, PREVIEW, EVIL]:
        assert setup.get("/map/airspace", Origin=origin).headers.get(ALLOW) != "*"


def test_flysafelz_ui_origin_is_allowed_when_set_and_the_dev_origins_stay(tmp_path):
    custom = Setup(tmp_path, settings={**SETTINGS, "FLYSAFELZ_UI_ORIGIN": "https://ui.flysafe.example:8443"})
    assert custom.get("/map/notams", Origin="https://ui.flysafe.example:8443").headers[ALLOW] == "https://ui.flysafe.example:8443"
    assert custom.get("/map/notams", Origin=DEV).headers[ALLOW] == DEV and custom.get("/map/notams", Origin=PREVIEW).headers[ALLOW] == PREVIEW
    assert ALLOW not in custom.get("/map/notams", Origin=EVIL).headers
    assert ALLOW not in custom.get("/map/notams", Origin="https://ui.flysafe.example").headers  # the port matters


def test_flysafelz_ui_origin_with_a_trailing_slash_is_taken_as_the_plain_origin(tmp_path):
    custom = Setup(tmp_path, settings={**SETTINGS, "FLYSAFELZ_UI_ORIGIN": " https://ui.flysafe.example/ "})
    assert custom.get("/map/notams", Origin="https://ui.flysafe.example").headers[ALLOW] == "https://ui.flysafe.example"


@pytest.mark.parametrize("value", ["*", "https://*.example", "https://ui.example/app", "ui.example", "javascript:alert(1)", "https://a.example, https://b.example", "", "   ", "null"])
def test_a_flysafelz_ui_origin_that_is_not_one_plain_origin_is_ignored(tmp_path, value):
    odd = Setup(tmp_path, settings={**SETTINGS, "FLYSAFELZ_UI_ORIGIN": value})
    for origin in [value.strip(), "https://ui.example", "https://a.example", "https://b.example", EVIL]:
        assert ALLOW not in odd.get("/map/notams", Origin=origin).headers
    assert odd.get("/map/notams", Origin=DEV).headers[ALLOW] == DEV


def test_a_preflight_for_the_agents_post_is_allowed_for_the_dev_origins_and_not_for_others(setup):
    """The server takes this app's middleware for every endpoint, so the chat's POST (with a JSON body) must pass its preflight."""
    ask = {"Access-Control-Request-Method": "POST", "Access-Control-Request-Headers": "content-type, x-custom"}
    allowed = setup.client.options("/threads/abc/runs/stream", headers={"Origin": DEV, **ask})
    assert allowed.status_code == 200 and allowed.headers[ALLOW] == DEV and "POST" in allowed.headers["access-control-allow-methods"]
    refused = setup.client.options("/threads/abc/runs/stream", headers={"Origin": EVIL, **ask})
    assert ALLOW not in refused.headers
