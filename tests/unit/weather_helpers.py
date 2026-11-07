"""Shared builders for the weather tests: calm values, forecasts and rows, and a changed copy of rules.yaml."""

from datetime import UTC, datetime, timedelta
from pathlib import Path

from flysafelz.clients.open_meteo import ONE_HOUR, Forecast
from flysafelz.contracts import SourceName, SourceState, SourceStatus, WeatherValues
from flysafelz.rules import DEFAULT_PATH, Rules, load_rules
from flysafelz.weather import WeatherRow

NOW = datetime(2026, 10, 31, 7, 30, tzinfo=UTC)  # frozen: the same instant as the conftest clock
FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "open_meteo"

# Every value is under every caution and NO_GO limit in rules.yaml.
CALM = {
    "wind_10m": 5.0,
    "wind_80m": 8.0,
    "wind_120m": 9.0,
    "gust_10m": 7.0,
    "direction_10m": 180.0,
    "precipitation": 0.0,
    "precipitation_probability": 0.0,
    "cape": 0.0,
    "cloud_cover": 10.0,
    "cloud_cover_low": 10.0,
    "visibility": 20000.0,
    "shortwave_radiation": 100.0,
    "pressure_msl": 1015.0,
}


def calm(**over: float | None) -> WeatherValues:
    return WeatherValues(**{**CALM, **over})


def ok_status() -> SourceStatus:
    return SourceStatus(source=SourceName.WEATHER, state=SourceState.OK, fetched_at=NOW)


def forecast_of(samples: dict[datetime, WeatherValues]) -> Forecast:
    return Forecast(status=ok_status(), elevation_m=8.0, samples=samples)


def calm_forecast(start: datetime, last: datetime) -> Forecast:
    """A calm sample at every whole UTC hour from start to last, both included."""
    samples = {}
    t = start
    while t <= last:
        samples[t] = calm()
        t += ONE_HOUR
    return forecast_of(samples)


def calm_row(hour: datetime | None = None, **over: float | None) -> WeatherRow:
    """A row built directly, with derived values set to calm numbers unless overridden."""
    gust_spread = over.pop("gust_spread", 2.0)
    gradient = over.pop("gradient", 4.0)
    return WeatherRow(hour=hour or NOW.replace(minute=0), values=calm(**over), gust_spread=gust_spread, gradient=gradient)


def rules_with(tmp_path: Path, rule_id: str, value: float) -> Rules:
    """A changed copy of rules.yaml loaded through the real loader: rule_id gets a new value."""
    lines = DEFAULT_PATH.read_text(encoding="utf-8").splitlines()
    start = lines.index(f"{rule_id}:")
    index = next(i for i in range(start, len(lines)) if lines[i].startswith("  value:"))
    lines[index] = f"  value: {value}"
    path = tmp_path / "rules.yaml"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return load_rules(path)


def hours(start: datetime, count: int) -> list[datetime]:
    return [start + timedelta(hours=i) for i in range(count)]
