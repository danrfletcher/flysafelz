"""get_weather (G9): the hourly forecast at the landing zone, each value beside the limits it is checked against.

Each hour is `weather.build_rows` over the window, so a value is the same number the assessment uses (the worse of the hour's two timestamps
for wind, CAPE, cloud and visibility; the hour's closing value for gusts, rain and shortwave; gust spread and gradient worked out per
timestamp). The limits are `RULES.<rule>.value`, `.unit` and `.comparison` of the rules `weather.check_row` applies, printed as they stand in
rules.yaml. A value the forecast does not have prints as `not forecast`, never as 0.

The tool gives facts, not a verdict: whether an hour is a NO-GO is `assess`'s job.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import NamedTuple

from langchain.tools import ToolRuntime, tool
from langchain_core.tools import BaseTool
from langgraph.types import Command

from flysafelz.agent import progress
from flysafelz.agent.sources import SourceCache
from flysafelz.agent.state import lz_radius_feature
from flysafelz.agent.tools.artifacts import RuleLimit, WeatherArtifact, WeatherHour, WeatherValue
from flysafelz.agent.tools.common import (
    NOT_FORECAST,
    SOURCE_LABEL,
    Prepared,
    Redactor,
    WindowArgs,
    fetch_line,
    g,
    hour_label,
    measured,
    prepare,
    reply,
    source_info,
    unusable_text,
    usable,
    window_line,
)
from flysafelz.contracts import SourceName
from flysafelz.rules import RULES
from flysafelz.weather import WeatherRow, build_rows

CAUTION = "caution"
NO_GO = "no-go"
THERMIC = "thermic caution"  # the caution that needs CAPE and shortwave together


class LimitSpec(NamedTuple):
    """One limit a value is checked against: its kind, the rule's name in RULES and, when the kind does not say it, its wording in the text.

    `partner` is for a limit that applies only together with another variable's (the thermic caution): that variable's label and rule name.
    The text then puts each limit beside its own variable, `thermic caution >= 300 J/kg together with shortwave >= 600 W/m²`.
    """

    kind: str
    rule: str
    wording: str = ""  # empty: the kind (`caution`, `no-go`)
    partner: tuple[str, str] | None = None


@dataclass(frozen=True)
class Reading:
    """One printed value: where it comes from in a row and the limits it is checked against."""

    name: str
    label: str
    get: Callable[[WeatherRow], float | None]
    limits: tuple[LimitSpec, ...]


def caution(rule: str) -> LimitSpec:
    return LimitSpec(CAUTION, rule)


def no_go(rule: str) -> LimitSpec:
    return LimitSpec(NO_GO, rule)


# One entry per rule weather.check_row applies, in the order it applies them.
READINGS = (
    Reading("wind_10m", "wind at 10 m", lambda r: r.values.wind_10m, (caution("wind_10m_caution"), no_go("wind_10m_no_go"))),
    Reading("gust_spread", "gust spread", lambda r: r.gust_spread, (caution("gust_spread_caution"), no_go("gust_spread_no_go"))),
    Reading("wind_80m", "wind at 80 m", lambda r: r.values.wind_80m, (caution("wind_aloft_caution"),)),
    Reading("wind_120m", "wind at 120 m", lambda r: r.values.wind_120m, (caution("wind_aloft_caution"),)),
    Reading("gradient", "gradient", lambda r: r.gradient, (caution("gradient_caution"), no_go("gradient_no_go"))),
    Reading("precipitation", "precipitation", lambda r: r.values.precipitation, (no_go("precipitation_no_go"),)),
    Reading("rain_chance", "rain chance", lambda r: r.values.precipitation_probability, (caution("rain_chance_caution"),)),
    Reading("visibility", "visibility", lambda r: r.values.visibility, (caution("visibility_caution"), no_go("visibility_no_go"))),
    Reading("low_cloud", "low cloud", lambda r: r.values.cloud_cover_low, (caution("low_cloud_caution"),)),
    Reading("cape", "CAPE", lambda r: r.values.cape, (LimitSpec(CAUTION, "thermic_cape_caution", THERMIC, ("shortwave", "thermic_shortwave_caution")), no_go("cape_no_go"))),
    Reading("shortwave", "shortwave", lambda r: r.values.shortwave_radiation, (LimitSpec(CAUTION, "thermic_shortwave_caution", THERMIC, ("CAPE", "thermic_cape_caution")),)),
)

DESCRIPTION = (
    "Get the hourly weather forecast at the pilot's landing zone for a window: wind at 10, 80 and 120 m, gust spread, gradient, rain, visibility, "
    "low cloud, CAPE and shortwave, each beside its caution and no-go limit. Times are in the pilot's time zone. "
    "Give start and end as local times written YYYY-MM-DD HH:MM, or leave both out for the default window. "
    "The place and time zone are the pilot's and cannot be changed here."
)


def _rule_limits(reading: Reading) -> list[tuple[RuleLimit, str]]:
    """The limits of one reading as the artifact has them, each with its words in the text (`caution >= 12 mph`). Value, unit and comparison come from RULES."""
    found = []
    for spec in reading.limits:
        rule = getattr(RULES, spec.rule)
        limit = RuleLimit(rule_id=rule.id, kind=spec.kind, comparison=rule.comparison.value, limit=float(rule.value), unit=rule.unit)
        found.append((limit, f"{spec.wording or spec.kind} {_stated(limit)}{_together_with(spec)}"))
    return found


def _stated(limit: RuleLimit) -> str:
    return f"{limit.comparison} {g(limit.limit)} {limit.unit}"


def _together_with(spec: LimitSpec) -> str:
    """` together with CAPE >= 300 J/kg`: the other variable's own limit, beside that variable, for a limit that needs both."""
    if spec.partner is None:
        return ""
    label, rule_name = spec.partner
    rule = getattr(RULES, rule_name)
    return f" together with {label} {rule.comparison.value} {g(float(rule.value))} {rule.unit}"


def _value(reading: Reading, row: WeatherRow) -> tuple[WeatherValue, str]:
    """(the value for the artifact, its text): `wind at 10 m 8 mph (caution >= 12 mph, no-go >= 16 mph)`."""
    limits = _rule_limits(reading)
    unit = limits[0][0].unit
    shown = reading.get(row)
    limit_words = ", ".join(words for _, words in limits)
    value_words = NOT_FORECAST if shown is None else f"{measured(shown, [(limit.comparison, limit.limit) for limit, _ in limits])} {unit}"
    value = WeatherValue(name=reading.name, label=reading.label, value=shown, unit=unit, limits=[limit for limit, _ in limits])
    return value, f"{reading.label} {value_words} ({limit_words})"


def weather_answer(prepared: Prepared) -> tuple[str, WeatherArtifact]:
    """The text and the artifact for the window. Pure over the prepared sources."""
    forecast = prepared.sources.forecast
    status = forecast.status
    lines = ["Weather at the landing zone, hourly.", window_line(prepared), fetch_line(prepared)]
    hours: list[WeatherHour] = []
    if not usable(status):
        lines.append(unusable_text(SOURCE_LABEL[SourceName.WEATHER], status))
    else:
        rows = build_rows(forecast, prepared.window.start_utc, prepared.window.end_utc)
        lines.append("Each value is followed by the limits it is checked against (rules.yaml). `" + NOT_FORECAST + "` means the forecast has no value.")
        lines.append(f"{len(rows)} hours:" if rows else "No hours in this window.")
        for row in rows:
            label = hour_label(row.hour, prepared.tz)
            made = [_value(reading, row) for reading in READINGS]
            hours.append(WeatherHour(hour=row.hour, label=label, values=[value for value, _ in made]))
            everything_missing = all(value.value is None for value, _ in made)
            lines.append(f"{label}: {NOT_FORECAST} for every value." if everything_missing else f"{label}: " + "; ".join(text for _, text in made) + ".")
    artifact = WeatherArtifact(
        source=source_info(prepared, status),
        window_start=prepared.window.start_utc,
        window_end=prepared.window.end_utc,
        window_assumed=prepared.window.assumed,
        overlays=[lz_radius_feature(prepared.lat, prepared.lon)],
        hours=hours,
    )
    return "\n".join(lines), artifact


def build_get_weather(source_cache: SourceCache) -> BaseTool:
    """The get_weather tool over this source store."""

    @tool("get_weather", args_schema=WindowArgs, description=DESCRIPTION)
    def get_weather(runtime: ToolRuntime, start: str | None = None, end: str | None = None) -> Command:
        redactor = Redactor(runtime.state.get("phone_map"))
        prepared = prepare(runtime, source_cache, redactor, start, end, progress.CHECKING_WEATHER)
        if isinstance(prepared, str):
            return reply(runtime, prepared, None, redactor)
        text, artifact = weather_answer(prepared)
        return reply(runtime, text, artifact, redactor)

    return get_weather
