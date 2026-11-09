"""The map data app: GET /map/airspace, GET /map/notams, GET /map/wind and GET /map/rain-grid, mounted on the LangGraph server as `http.app` in langgraph.json.

    create_app(root=, clock=, settings=, client_factory=)   a Starlette app with the four routes and their CORS
    app                                                    create_app() with the real clock, settings and network

The airspace and NOTAM routes answer with one envelope: a GeoJSON FeatureCollection (`features`) with `updated_at` (ISO UTC, the last good refresh)
and `stale`. The wind route's envelope is the same with `grid` (leaflet-velocity's U and V records) in place of the features, and so is the rain-grid route's (its `grid` is a list of cells, the UI's fallback when RainViewer is down). A failed refresh serves
the last cache with `stale: true`; with no cache it is HTTP 503 and {"reason": one line}. One route failing does not touch the others. Only GET (and the HEAD that goes with it) is answered; any other method is 405 with `Allow`, and the only files written are under <root>/data/map/.

CORS allows the dev UI's two origins and FLYSAFELZ_UI_ORIGIN when it is set, and nothing else; there is no wildcard. The LangGraph server takes
this app's middleware as its own for every endpoint, the agent's included (the chat stream is a POST with a JSON body from the same origins), so
the middleware answers a preflight for every method and header; the routes themselves are GET only. langgraph.json (`http.cors`) lists the two
dev origins for the server's own layer, so the agent endpoints are not left on the server's default (any origin).

FLYSAFELZ_UI_ORIGIN reaches the real server only because langgraph-api (0.17.1, uv.lock) puts this app's middleware outside the server's own CORS layer
(langgraph_api/server.py: `app.user_middleware = custom_middleware + global_middleware + ...`). langgraph.json's `http.cors` lists only the dev
origins, so if a langgraph-api upgrade changed that order the extra origin would be dropped without a failure. tests/unit/agent/test_langgraph_json.py
pins the method, header and expose lists below to langgraph.json's and reads that line of the installed server, so such an upgrade fails a test.

`root`, `clock`, `settings` and `client_factory` are for tests: the repository root, the time, the settings and the HTTP client that reaches
openAIP and the PIB. Left out, they are the real ones.
"""

from collections.abc import Callable, Mapping
from datetime import UTC, datetime, timedelta
from http import HTTPStatus
from pathlib import Path

import httpx
from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Route

from flysafelz.clients.pib_source import TIMEOUT_SECONDS
from flysafelz.mapdata.airspace import airspace_features
from flysafelz.mapdata.cache import Layer
from flysafelz.mapdata.config import cache_dir, openaip_key, process_settings, project_root, ui_origins
from flysafelz.mapdata.notams import notam_features
from flysafelz.mapdata.rain_grid import rain_grid
from flysafelz.mapdata.wind import wind_grid

AIRSPACE_TTL = timedelta(hours=24)
NOTAM_TTL = timedelta(minutes=15)
WIND_TTL = timedelta(minutes=180)  # 3 h
RAIN_TTL = timedelta(minutes=180)  # 3 h
GRID = "grid"  # the wind and rain-grid payloads' key, where the other two routes have "features"
READ_METHODS = ("GET", "HEAD")
ANY_METHOD = [*READ_METHODS, "POST", "PUT", "PATCH", "DELETE", "OPTIONS"]
NOT_ALLOWED = HTTPStatus.METHOD_NOT_ALLOWED.value
SERVER_METHODS = ["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"]  # the agent endpoints' methods; same as langgraph.json http.cors
SERVER_HEADERS = ["*"]
SERVER_EXPOSED = ["x-pagination-total", "x-pagination-next", "content-location"]  # the ones the SDK's useStream reads


def default_client() -> httpx.Client:
    return httpx.Client(timeout=TIMEOUT_SECONDS, follow_redirects=True)


def real_clock() -> datetime:
    return datetime.now(UTC)


def create_app(
    *,
    root: Path | None = None,
    clock: Callable[[], datetime] | None = None,
    settings: Mapping[str, str] | None = None,
    client_factory: Callable[[], httpx.Client] | None = None,
) -> Starlette:
    root = project_root() if root is None else root
    clock = real_clock if clock is None else clock
    values = process_settings() if settings is None else settings
    make_client = default_client if client_factory is None else client_factory
    folder = cache_dir(root)

    def refresh_airspace(now: datetime) -> list[dict]:
        key = openaip_key(values, root)  # read on each refresh, so a newly added key is used without a restart
        with make_client() as client:
            return airspace_features(client, key)

    def refresh_notams(now: datetime) -> list[dict]:
        with make_client() as client:
            return notam_features(client, now)

    def refresh_wind(now: datetime) -> list[dict]:
        with make_client() as client:
            return wind_grid(client, now)

    def refresh_rain(now: datetime) -> list[dict]:
        with make_client() as client:
            return rain_grid(client)

    airspace = Layer("airspace", AIRSPACE_TTL, refresh_airspace)
    notams = Layer("notams", NOTAM_TTL, refresh_notams)
    wind = Layer("wind", WIND_TTL, refresh_wind, key=GRID)
    rain = Layer("rain-grid", RAIN_TTL, refresh_rain, key=GRID)

    def route(layer: Layer) -> Callable:
        async def respond(request: Request) -> Response:
            if request.method not in READ_METHODS:
                return Response(status_code=NOT_ALLOWED, headers={"Allow": ", ".join(READ_METHODS)})
            served = await layer.resolve_once(folder, clock)
            return JSONResponse(served.body, status_code=served.status)

        return respond

    # The routes take every common method and answer all but GET and HEAD themselves. With methods=["GET"] Starlette would only partly
    # match a POST, and the LangGraph server's catch-all mount (added after this app's routes) would answer 404 instead of 405.
    routes = [
        Route("/map/airspace", route(airspace), methods=ANY_METHOD),
        Route("/map/notams", route(notams), methods=ANY_METHOD),
        Route("/map/wind", route(wind), methods=ANY_METHOD),
        Route("/map/rain-grid", route(rain), methods=ANY_METHOD),
    ]
    cors = Middleware(
        CORSMiddleware,
        allow_origins=ui_origins(values),
        allow_methods=SERVER_METHODS,
        allow_headers=SERVER_HEADERS,
        expose_headers=SERVER_EXPOSED,
    )
    return Starlette(routes=routes, middleware=[cors])


app = create_app()
