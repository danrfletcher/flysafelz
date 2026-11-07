"""Load a committed fixture case (fixtures/<case>/) into what assess() takes. Offline: flysafelz.cases replays the files.

Used by the integration tests. No network, no .env, no key.
"""

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from flysafelz.assess import assess
from flysafelz.capture import Case
from flysafelz.cases import load_case_inputs
from flysafelz.clients.open_meteo import Forecast
from flysafelz.clients.openaip import AirspaceFetch
from flysafelz.contracts import Assessment, NotamReading
from flysafelz.pib import PibResult

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
BLUE_MILL = "blue_mill_2026-10-31"
WEST_SLEEKBURN = "west_sleekburn_2026-10-31"
TODAY = datetime(2026, 11, 6, 12, 0, tzinfo=UTC)  # frozen: after every case day


def case_dir(name: str) -> Path:
    return FIXTURES / name


def load(name: str) -> tuple[Case, Forecast, AirspaceFetch, PibResult, dict[str, NotamReading]]:
    return load_case_inputs(case_dir(name))


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
