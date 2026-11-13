"""Weather rows, checks and data tier. Plain functions over Pydantic models, no model call.

Row mapping (G4). Row HH covers HH:00 to HH+1:00 and is built from the
forecast's HH and HH+1 timestamps:

    instantaneous  wind at every height, CAPE, cloud, visibility, pressure
                   -> the worse of HH and HH+1 (highest, but lowest for
                   visibility and pressure). Direction is not checked
                   and just reads HH.
    preceding hour gusts, precipitation, rain chance, shortwave
                   -> the HH+1 value only (it describes HH..HH+1)
    derived        gust spread (gust - 10 m wind) and gradient
                   (|120 m wind - 10 m wind|)
                   -> worked out at each timestamp first, then the worse of
                   the two. max(gust) - max(wind) would be a different number.
                   Missing when either timestamp lacks an input.

Nothing is interpolated or filled forward. A value that is missing at HH or at
HH+1 (including a timestamp the response lacks) is missing for the row.

Checks (G2, G3) compare against rules.yaml with each rule's own comparison, so
a value exactly at a limit is decided by the rule. A value exactly on a NO_GO
limit is a caution, never a blocker (G14): NO_GO needs a value strictly past
its limit. They return blockers first, then cautions, each in a fixed order. They never add up or weigh anything: the
verdict is PR-8's job.

Tier (G5, E3). The VERDICT tier needs every required variable and stops after
the verdict horizon day (day 0 is today in Europe/London). An hour after the
last complete hour is OUTLOOK. A required variable missing in an hour before it
makes that hour UNVERIFIED (source weather), not OUTLOOK.
"""

from datetime import datetime
from typing import Literal

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field

from flysafelz.clients.open_meteo import ONE_HOUR, Forecast
from flysafelz.contracts import Tier, WeatherValues
from flysafelz.daylight import london_day_offset, utc_hours
from flysafelz.rules import RULES, Rule

# Variables an hour must have to carry a verdict, as WeatherValues fields.
REQUIRED = (
    "wind_10m",
    "wind_80m",
    "wind_120m",
    "gust_10m",
    "precipitation",
    "cape",
    "visibility",
    "shortwave_radiation",
    "cloud_cover_low",
)

# Optional variables: when one is missing its check is skipped and a note is added.
OPTIONAL_NOTES = {
    "precipitation_probability": "rain chance not available",
    "cloud_cover": "total cloud not available",
    "direction_10m": "wind direction not available",
}

# Instantaneous values: which of the HH and HH+1 values is the worse one.
WORSE_OF_TWO = {
    "wind_10m": max,
    "wind_80m": max,
    "wind_120m": max,
    "cape": max,
    "cloud_cover": max,
    "cloud_cover_low": max,
    "visibility": min,
    "pressure_msl": min,
}

# Preceding-hour values: the HH+1 value describes the hour.
FROM_CLOSING_HOUR = ("gust_10m", "precipitation", "precipitation_probability", "shortwave_radiation")


class WeatherRow(BaseModel):
    """One UTC hour of weather. values keeps the raw numbers, which OUTLOOK hours still print."""

    model_config = ConfigDict(extra="forbid")

    hour: AwareDatetime
    values: WeatherValues
    gust_spread: float | None = None  # mph, worse of the two timestamps; None if either is missing
    gradient: float | None = None  # mph, worse of the two timestamps; None if either is missing
    missing_required: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
    # Set by build_rows from the whole response: this hour is after its last complete hour.
    # False (a hole counts as UNVERIFIED) is the safe default for a row built by hand.
    after_last_complete: bool = False


class Finding(BaseModel):
    """One weather result: a blocker or a caution, with its rule, value and limit."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["blocker", "caution"]
    rule_id: str
    text: str  # e.g. "gust spread 5 mph, limit 6"
    value: float
    limit: float


class HourTier(BaseModel):
    """The data tier of one hour. unverified is set for a hole inside the run (E3)."""

    model_config = ConfigDict(extra="forbid")

    tier: Tier
    unverified: bool = False
    missing: list[str] = Field(default_factory=list)
    reason: str = ""


# --- rows (G4, G5) ---------------------------------------------------------


def _get(sample: WeatherValues | None, field: str) -> float | None:
    return None if sample is None else getattr(sample, field)


FLOAT_NOISE_PLACES = 8


def _spread(sample: WeatherValues | None) -> float | None:
    """Gust minus 10 m wind at one timestamp, or None when the timestamp or either number is missing."""
    if sample is None or sample.gust_10m is None or sample.wind_10m is None:
        return None
    return round(sample.gust_10m - sample.wind_10m, FLOAT_NOISE_PLACES)


def _gradient(sample: WeatherValues | None) -> float | None:
    """|120 m wind - 10 m wind| at one timestamp, or None when the timestamp or either number is missing."""
    if sample is None or sample.wind_120m is None or sample.wind_10m is None:
        return None
    return round(abs(sample.wind_120m - sample.wind_10m), FLOAT_NOISE_PLACES)


def _row(hour: datetime, forecast: Forecast) -> WeatherRow:
    here = forecast.samples.get(hour)
    closing = forecast.samples.get(hour + ONE_HOUR)

    values = {}
    for field, worse in WORSE_OF_TWO.items():
        a, b = _get(here, field), _get(closing, field)
        values[field] = None if a is None or b is None else worse(a, b)
    for field in FROM_CLOSING_HOUR:
        values[field] = _get(closing, field)
    values["direction_10m"] = _get(here, "direction_10m")

    # Derived values come from each timestamp's own numbers and, like every other value, are missing for the row when
    # either timestamp lacks an input. Rounded to FLOAT_NOISE_PLACES to drop float noise (12.3 - 6.3 is
    # 6.000000000000001, which would break "> 6") without hiding a real fraction: a 6.04 mph spread stays 6.04.
    spreads, gradients = [], []
    for sample in (here, closing):
        spreads.append(_spread(sample))
        gradients.append(_gradient(sample))

    missing = [name for name in REQUIRED if values[name] is None]
    notes = [text for name, text in OPTIONAL_NOTES.items() if values[name] is None]
    return WeatherRow(
        hour=hour,
        values=WeatherValues(**values),
        gust_spread=None if None in spreads else max(spreads),
        gradient=None if None in gradients else max(gradients),
        missing_required=missing,
        notes=notes,
    )


def _last_complete_hour(forecast: Forecast) -> datetime | None:
    """The last hour of the whole response that has every required variable (not just the window)."""
    if not forecast.samples:
        return None
    first, last = min(forecast.samples), max(forecast.samples)
    complete = [row.hour for row in (_row(hour, forecast) for hour in utc_hours(first, last)) if not row.missing_required]
    return max(complete, default=None)


def build_rows(forecast: Forecast, start: datetime, end: datetime) -> list[WeatherRow]:
    """One row per UTC hour from start up to end (end excluded). An empty window gives [].

    Each row is also told whether it lies after the last complete hour of the whole
    response, so a hole in the last hours of the window is not mistaken for a tail.
    """
    last_complete = _last_complete_hour(forecast)
    rows = []
    for hour in utc_hours(start, end):
        row = _row(hour, forecast)
        row.after_last_complete = last_complete is None or hour > last_complete
        rows.append(row)
    return rows


# --- checks (G2, G3) -------------------------------------------------------

# Added to the precipitation caution for a value exactly on the limit (0.1 mm/h, trace forecast rain).
TRACE_RAIN_HINT = "trace rain forecast: check live radar for active showers before launch; a wet wing must not be flown"


def _number(x: float) -> str:
    return f"{x:.1f}".rstrip("0").rstrip(".")


def _limit(rule: Rule) -> float:
    """The number a limit rule holds (a list value belongs to a membership rule, not a limit)."""
    if isinstance(rule.value, list):
        raise TypeError(f"rule {rule.id} is not a numeric limit")
    return rule.value


def _finding(kind: Literal["blocker", "caution"], rule: Rule, label: str, value: float) -> Finding:
    shown, limit = _number(value), _number(_limit(rule))
    if kind == "blocker" and shown == limit:
        # A value past the limit must never print as "16 mph, limit 16" (float noise, or 16.04 rounded to 16).
        shown = f"{value:.2f}"
    text = f"{label} {shown} {rule.unit}, limit {limit}"
    return Finding(kind=kind, rule_id=rule.id, text=text, value=value, limit=_limit(rule))


def _on_the_limit(rule: Rule, label: str, value: float, hint: str | None) -> Finding:
    """A caution for a value exactly on a NO_GO limit that has no caution band of its own. Cites the NO_GO rule."""
    text = f"{label} {_number(value)} {rule.unit}, on the limit {_number(_limit(rule))}"
    if hint:
        text = f"{text}; {hint}"
    return Finding(kind="caution", rule_id=rule.id, text=text, value=value, limit=_limit(rule))


def _band(
    value: float | None, label: str, no_go: Rule | None, caution: Rule | None, on_limit_hint: str | None = None
) -> Finding | None:
    """NO_GO if the value breaches the NO_GO rule, else a caution if it breaches the caution rule.

    NO_GO needs a value strictly past its limit. A value exactly on the NO_GO limit is a caution: the caution
    rule gives it where there is one that is breached there, else (precipitation, CAPE) the NO_GO rule itself
    gives an "on the limit" caution. Equality is exact, on the same number `breached` compares.
    """
    if value is None:
        return None
    if no_go is not None and no_go.breached(value):
        return _finding("blocker", no_go, label, value)
    if caution is not None and caution.breached(value):
        return _finding("caution", caution, label, value)
    if no_go is not None and value == _limit(no_go):
        return _on_the_limit(no_go, label, value, on_limit_hint)
    return None


def check_row(row: WeatherRow) -> list[Finding]:
    """Blockers first, then cautions; each group in the order of the table in rules.yaml.

    A check whose value is missing is skipped. The row's missing variables are
    reported by weather_tier, so a NO_GO from the values that are there still shows.
    """
    v = row.values
    cape_no_go = v.cape is not None and RULES.cape_no_go.breached(v.cape)  # strictly past the limit

    found = [
        _band(v.wind_10m, "wind at 10 m", RULES.wind_10m_no_go, RULES.wind_10m_caution),
        _band(row.gust_spread, "gust spread", RULES.gust_spread_no_go, RULES.gust_spread_caution),
        _band(v.wind_80m, "wind at 80 m", None, RULES.wind_aloft_caution),  # aloft: caution only
        _band(v.wind_120m, "wind at 120 m", None, RULES.wind_aloft_caution),
        _band(row.gradient, "gradient", RULES.gradient_no_go, RULES.gradient_caution),
        _band(v.precipitation, "precipitation", RULES.precipitation_no_go, None, on_limit_hint=TRACE_RAIN_HINT),
        _band(v.precipitation_probability, "rain chance", None, RULES.rain_chance_caution),
        _band(v.visibility, "visibility", RULES.visibility_no_go, RULES.visibility_caution),
        _band(v.cloud_cover_low, "low cloud", None, RULES.low_cloud_caution),
        _band(v.cape, "CAPE", RULES.cape_no_go, None),
    ]
    # Thermic caution needs shortwave AND CAPE together. CAPE past its NO_GO limit is a blocker whatever the shortwave.
    # CAPE exactly on the limit is a caution, and the thermic caution may also apply.
    if (
        not cape_no_go
        and v.shortwave_radiation is not None
        and v.cape is not None
        and RULES.thermic_shortwave_caution.breached(v.shortwave_radiation)
        and RULES.thermic_cape_caution.breached(v.cape)
    ):
        label = f"thermic: shortwave {_number(v.shortwave_radiation)} {RULES.thermic_shortwave_caution.unit} (limit {_number(_limit(RULES.thermic_shortwave_caution))}) and CAPE"
        found.append(_finding("caution", RULES.thermic_cape_caution, label, v.cape))

    found = [f for f in found if f is not None]
    return [f for f in found if f.kind == "blocker"] + [f for f in found if f.kind == "caution"]


# --- tier (G5, E3) -----------------------------------------------------------


def weather_tier(hour: datetime, rows: list[WeatherRow], now: datetime) -> HourTier:
    """The data tier of one hour. Only VERDICT and OUTLOOK come from the weather; TOO_FAR is PR-9's.

    OUTLOOK: the hour is after the last hour of the whole response that has every required variable
    (build_rows works that out from the full response, so the window edge does not matter), or it is
    past the verdict horizon day. Otherwise VERDICT, and UNVERIFIED if this hour is missing a required
    variable (a hole inside the run). An hour with no row at all is judged against the last complete row given.
    """
    row = next((r for r in rows if r.hour == hour), None)
    if row is not None:
        after_data = row.after_last_complete
    else:
        last_complete = max((r.hour for r in rows if not r.missing_required), default=None)
        after_data = last_complete is None or hour > last_complete
    if after_data or london_day_offset(hour, now) > _limit(RULES.verdict_horizon_day):
        return HourTier(tier=Tier.OUTLOOK)

    missing = list(REQUIRED) if row is None else list(row.missing_required)
    if missing:
        return HourTier(tier=Tier.VERDICT, unverified=True, missing=missing, reason=f"weather data missing: {', '.join(missing)}")
    return HourTier(tier=Tier.VERDICT)
