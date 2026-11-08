"""What the data-tool tests share: a fetcher that counts and answers with the sources a test gives it, and a SourceCache over it."""

from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from factories import airspace_fetch, pib_result, weather_forecast
from tool_helpers import ToolRun, run_tools

from flysafelz.agent.sources import SourceCache
from flysafelz.agent.state import RunContext
from flysafelz.clients.open_meteo import Forecast, parse_response
from flysafelz.clients.openaip import Airspace, AirspaceFetch
from flysafelz.pib import PibResult

NOW = datetime(2026, 10, 31, 7, 30, tzinfo=UTC)
FIXTURES = Path(__file__).resolve().parents[2] / "fixtures"
WINDOW = {"start": "2026-10-31 10:00", "end": "2026-10-31 16:00"}


@dataclass
class World:
    """The three sources a live fetch returns. `calls` counts the fetches (I2)."""

    forecast: Forecast | None = None
    airspace: AirspaceFetch = field(default_factory=airspace_fetch)
    pib: PibResult = field(default_factory=pib_result)
    calls: list[tuple] = field(default_factory=list)

    def __call__(self, lat, lon, window, now):
        self.calls.append((lat, lon, window, now))
        first, last = window
        return self.forecast or weather_forecast(first, last), self.airspace, self.pib


def forecast_fixture(name: str) -> Forecast:
    import json

    return parse_response(json.loads((FIXTURES / "open_meteo" / f"{name}.json").read_text(encoding="utf-8")), NOW)


def areas_world(*areas: Airspace, **kwargs) -> World:
    return World(airspace=airspace_fetch(*areas), **kwargs)


def go(world: World, steps, *, ctx: RunContext | None = None, tz: str | None = "Europe/London", reader_model=None, **kwargs) -> ToolRun:
    """One turn over a live fetch (no case): the model asks for `steps` in turn."""
    source_cache = SourceCache(fetch=world, reader_model=reader_model, **kwargs)
    return run_tools(source_cache, steps, ctx=ctx or RunContext(now=NOW), tz=tz)
