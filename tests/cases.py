"""Load a committed fixture case (fixtures/<case>/) into what assess() takes. Offline: a mock transport serves the files.

Used by the integration tests. No network, no .env, no key (the airspace key is a made-up string the mock ignores).
"""

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import yaml

from flysafelz.assess import assess
from flysafelz.capture import Case, load_case
from flysafelz.clients.open_meteo import Forecast, parse_response
from flysafelz.clients.openaip import AirspaceFetch, fetch_airspace
from flysafelz.contracts import Assessment, NotamReading
from flysafelz.pib import PibResult, load_pib

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
BLUE_MILL = "blue_mill_2026-10-31"
WEST_SLEEKBURN = "west_sleekburn_2026-10-31"
TODAY = datetime(2026, 11, 6, 12, 0, tzinfo=UTC)  # frozen: after every case day


def case_dir(name: str) -> Path:
    return FIXTURES / name


def _served(body: object) -> httpx.Client:
    """A client whose every request is answered with this body, so nothing leaves the machine."""
    content = body if isinstance(body, bytes) else json.dumps(body).encode()
    return httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(200, content=content)))


def load(name: str) -> tuple[Case, Forecast, AirspaceFetch, PibResult, dict[str, NotamReading]]:
    folder = case_dir(name)
    case, _raw = load_case(folder / "case.yaml", today=TODAY)
    forecast = parse_response(json.loads((folder / "weather.json").read_text(encoding="utf-8")), case.now)
    airspace = fetch_airspace(
        case.lat, case.lon, api_key="not-a-real-key", client=_served(json.loads((folder / "airspace.json").read_text(encoding="utf-8"))), now=case.now
    )
    pib = load_pib("live", case.now, client=_served((folder / "pib.xml").read_bytes()))
    entries = yaml.safe_load((folder / "readings.yaml").read_text(encoding="utf-8")) or []
    readings = {r.id: r for r in (NotamReading.model_validate(item) for item in entries)}
    return case, forecast, airspace, pib, readings


def run(name: str, *, readings: bool = True, **changes: Any) -> Assessment:
    """assess() over a committed case, with the case's own lat, lon, now and window. `changes` replace any argument."""
    case, forecast, airspace, pib, read = load(name)
    arguments: dict[str, Any] = {
        "now": case.now,
        "forecast": forecast,
        "airspace": airspace,
        "pib": pib,
        "readings": read if readings else {},
        "window": (case.window.start, case.window.end),
    }
    arguments.update(changes)
    return assess(case.lat, case.lon, **arguments)
