"""What the guard tests share: `TurnSources` built from real tool results, never fetched.

blue_mill_turn()                the four data tools run in a real graph on the Blue Mill case (frozen now 2026-10-31T07:30Z), as ticket 4's tool tests run them
outlook_turn()                  an `assess` result whose three hours are all OUTLOOK, so there is no overall verdict
text_turn(*texts)               tool results that are only text (no artifact), for the tests of words and quotations
weather_turn(value, unit, ...)  one weather value with its limit and comparison in the artifact and nothing in the text
findings(answer, turn)          guard_answer as plain strings
"""

from datetime import UTC, date, datetime, timedelta
from functools import cache

from assess_tool_helpers import render
from factories import run_assess, utc
from factories import weather_forecast as forecast_between
from tool_helpers import ALL_THREE, BLUE_MILL_LZ, BLUE_MILL_NOW, BLUE_MILL_WINDOW, calls, run_tools

from flysafelz.agent.guard import ToolResult, TurnSources, guard_answer
from flysafelz.agent.sources import SourceCache
from flysafelz.agent.state import OverlayFeature, OverlayProperties, PointGeometry, RunContext
from flysafelz.agent.tools.artifacts import RuleLimit, SourceInfo, WeatherArtifact, WeatherHour, WeatherValue

LONDON = "Europe/London"
ONE_DAY = timedelta(days=1)
WEEK = tuple(BLUE_MILL_NOW.date() + n * ONE_DAY for n in range(8))  # the prompt's calendar: today and seven days
BLUE_MILL_CTX = RunContext(now=BLUE_MILL_NOW, case="blue_mill_2026-10-31", reader="fixture")
FRESH = datetime(2026, 10, 31, 7, 30, tzinfo=UTC)


@cache
def blue_mill_turn() -> TurnSources:
    run = run_tools(SourceCache(), [calls(*ALL_THREE, **BLUE_MILL_WINDOW), calls("assess", **BLUE_MILL_WINDOW)], ctx=BLUE_MILL_CTX)
    results = tuple(ToolResult(str(m.name), str(m.content), m.artifact) for m in run.messages)
    return TurnSources(results=results, tz=LONDON, phone_keys=frozenset(run.state["phone_map"]), calendar=WEEK, lz=(BLUE_MILL_LZ["lat"], BLUE_MILL_LZ["lon"]))


def outlook_turn() -> TurnSources:
    first = datetime(2026, 11, 5, 10, tzinfo=UTC)
    window = (first, first + 3 * timedelta(hours=1))
    forecast = forecast_between(utc(6), datetime(2026, 11, 8, 23, tzinfo=UTC))
    text, artifact = render(run_assess(window=window, forecast=forecast), window)
    return turn_of(ToolResult("assess", text, artifact))


def turn_of(*results: ToolResult, phones: tuple[str, ...] = (), calendar: tuple[date, ...] = WEEK, lz: tuple[float, float] | None = None) -> TurnSources:
    return TurnSources(results=results, tz=LONDON, phone_keys=frozenset(phones), calendar=calendar, lz=lz)


def text_turn(*texts: str, **kwargs) -> TurnSources:
    return turn_of(*(ToolResult("get_notams", text) for text in texts), **kwargs)


def weather_turn(value: float, unit: str, limit: float | None = None, comparison: str = ">=", text: str = "", *, limit_unit: str | None = None, overlays: list | None = None) -> TurnSources:
    """One weather value, and one limit it is checked against when `limit` is given. The text is empty, so the artifact is the only source."""
    limits = [] if limit is None else [RuleLimit(rule_id="gust_spread_caution", kind="caution", comparison=comparison, limit=limit, unit=limit_unit or unit)]
    source = SourceInfo(source="weather", state="ok", reason="", fetched_at=FRESH, check_by_hand=None)
    hour = WeatherHour(hour=utc(10), label="Sat 31 Oct 10:00 GMT", values=[WeatherValue(name="gust_spread", label="gust spread", value=value, unit=unit, limits=limits)])
    artifact = WeatherArtifact(source=source, window_start=utc(10), window_end=utc(11), window_assumed=False, overlays=overlays or [], hours=[hour])
    return turn_of(ToolResult("get_weather", text, artifact))


def point_feature(lat: float, lon: float, text: str | None = None) -> OverlayFeature:
    """A NOTAM's point on the map, as GeoJSON holds it (longitude first)."""
    return OverlayFeature(geometry=PointGeometry(coordinates=[lon, lat]), properties=OverlayProperties(id="H7171/26", kind="notam", text=text))


def findings(answer: str, turn: TurnSources) -> list[str]:
    return [str(f) for f in guard_answer(answer, turn)]


def mixed_turn() -> TurnSources:
    """Eighteen hours with a verdict (overall NO-GO, PILOT DISCRETION hours among them), then OUTLOOK hours that have none."""
    first = datetime(2026, 11, 3, 6, tzinfo=UTC)
    window = (first, first + timedelta(hours=40))
    forecast = forecast_between(utc(6), datetime(2026, 11, 8, 23, tzinfo=UTC))
    text, artifact = render(run_assess(window=window, forecast=forecast), window)
    return turn_of(ToolResult("assess", text, artifact))


TURN_TEXT = "Window Saturday 31 Oct 10:00-16:00 GMT. Closing 17:02. H7171/26 and J3857/26 within 2 NM. Phone Flylogix <PHONE_1>."


def turn(**kwargs) -> TurnSources:
    """A turn whose only tool result is TURN_TEXT, in the thread that has the placeholder <PHONE_1>."""
    return text_turn(TURN_TEXT, phones=("<PHONE_1>",), **kwargs)
