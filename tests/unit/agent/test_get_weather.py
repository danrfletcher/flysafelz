"""get_weather (G9): every value beside the limits it is checked against, "not forecast" for a hole, the zone labels, the artifact's pairs."""

import inspect
import re
from datetime import UTC, datetime

import pytest
from factories import weather_forecast
from tool_helpers import calls
from tool_world import WINDOW, World, forecast_fixture, go

from flysafelz import weather
from flysafelz.agent.state import RunContext
from flysafelz.agent.tools.common import measured
from flysafelz.agent.tools.weather_tool import READINGS
from flysafelz.contracts import SourceName, SourceState, WeatherValues
from flysafelz.rules import RULES
from flysafelz.sources import CHECK_BY_HAND

WEATHER = [calls("get_weather", **WINDOW)]


def full_48h_world() -> World:
    return World(forecast=forecast_fixture("full_48h"))


def number(value: float) -> str:
    return f"{value:g}"


# Every rule weather.check_row applies, read from check_row itself: a rule the tool forgot to print is a missing case here, not a lost one.
CHECK_ROW_RULES = sorted(set(re.findall(r"RULES\.(\w+)", inspect.getsource(weather.check_row))))


def readings_with(rule_name: str):
    return [(reading, spec) for reading in READINGS for spec in reading.limits if spec.rule == rule_name]


def test_the_tool_prints_exactly_the_rules_check_row_applies():
    assert CHECK_ROW_RULES, "check_row's rules were found"
    assert {spec.rule for reading in READINGS for spec in reading.limits} == set(CHECK_ROW_RULES)


@pytest.mark.parametrize("rule_name", CHECK_ROW_RULES)
def test_every_limit_the_tool_prints_and_pairs_is_the_rule_in_rules_yaml(rule_name):
    rule = getattr(RULES, rule_name)
    run = go(full_48h_world(), WEATHER)
    text = run.text("get_weather")
    hour = run.artifact("get_weather").hours[0]
    shown = readings_with(rule_name)
    assert shown, f"{rule_name} is printed beside at least one value"
    for reading, spec in shown:
        value = next(v for v in hour.values if v.name == reading.name)
        limit = next(lim for lim in value.limits if lim.rule_id == rule.id)
        assert (limit.limit, limit.unit, limit.comparison, limit.kind) == (float(rule.value), rule.unit, rule.comparison.value, spec.kind)
        assert value.unit == rule.unit
        row = next(line for line in text.splitlines() if line.startswith("Sat 31 Oct 10:00 GMT"))
        assert f"{reading.label} " in row and f"{rule.comparison.value} {number(float(rule.value))} {rule.unit}" in row


def test_the_thermic_caution_puts_each_limit_beside_its_own_variable():
    """CAPE 300 J/kg and shortwave 600 W/m² are the two limits of one caution: each is written next to the variable it is the limit of."""
    run = go(full_48h_world(), WEATHER)
    row = next(line for line in run.text("get_weather").splitlines() if line.startswith("Sat 31 Oct 10:00 GMT"))
    found = re.search(r"CAPE [\d.]+ J/kg \((.*?)\); shortwave [\d.]+ W/m² \((.*?)\)\.$", row)
    assert found, row
    cape_limits, shortwave_limits = found.groups()
    cape, shortwave = number(float(RULES.thermic_cape_caution.value)), number(float(RULES.thermic_shortwave_caution.value))
    assert cape_limits.startswith(f"thermic caution >= {cape} J/kg together with shortwave >= {shortwave} W/m²")
    assert shortwave_limits == f"thermic caution >= {shortwave} W/m² together with CAPE >= {cape} J/kg: under the {shortwave} W/m² limit"
    # and the shipped numbers, so a change to rules.yaml shows up here as a change to the briefing
    assert cape_limits == "thermic caution >= 300 J/kg together with shortwave >= 600 W/m²: under the 300 J/kg limit, no-go > 1000 J/kg: under the 1000 J/kg limit"
    assert shortwave_limits == "thermic caution >= 600 W/m² together with CAPE >= 300 J/kg: under the 600 W/m² limit"


@pytest.mark.parametrize(
    ("value", "limits", "shown"),
    [
        (11.96, [(">=", 12.0)], "11.96"),  # 12 would read as at the caution limit when it is under it
        (12.04, [(">=", 12.0)], "12"),  # both sides of the limit agree: one decimal is enough
        (0.14, [(">", 0.1)], "0.14"),  # 0.1 would read as not over the no-go when it is over
        (0.05, [(">", 0.1)], "0.1"),  # rounds up to the limit, and 0.1 is not > 0.1, as 0.05 is not
        (0.1, [(">", 0.1)], "0.1"),
        (11.99999, [(">=", 12.0)], "11.99999"),
        (8.4, [(">=", 12.0), (">", 16.0)], "8.4"),
        (8.0, [], "8"),
    ],
)
def test_a_printed_value_is_on_the_same_side_of_each_limit_as_the_true_one(value, limits, shown):
    assert measured(value, limits) == shown
    for comparison, limit in limits:
        printed = float(measured(value, limits))
        assert {">=": printed >= limit, ">": printed > limit}[comparison] == {">=": value >= limit, ">": value > limit}[comparison]


@pytest.mark.parametrize(
    ("changes", "wind", "rain"),
    [
        ({"wind_10m": 11.96, "precipitation": 0.14}, "wind at 10 m 11.96 mph (caution >= 12 mph: under the 12 mph limit", "precipitation 0.14 mm/h (no-go > 0.1 mm/h: over the 0.1 mm/h limit)"),
        ({"wind_10m": 12.04, "precipitation": 0.05}, "wind at 10 m 12 mph (caution >= 12 mph: over the 12 mph limit", "precipitation 0.1 mm/h (no-go > 0.1 mm/h: under the 0.1 mm/h limit)"),
    ],
)
def test_the_hour_text_never_prints_a_value_that_crosses_its_own_limit_when_rounded(changes, wind, rain):
    """Through the tool: the 10:00 hour uses the worse of 09:00 and 10:00 for wind and the closing value for rain, so both timestamps carry the value."""
    first, last = datetime(2026, 10, 31, 8, tzinfo=UTC), datetime(2026, 10, 31, 16, tzinfo=UTC)
    run = go(World(forecast=weather_forecast(first, last, after={first: changes})), WEATHER)
    row = next(line for line in run.text("get_weather").splitlines() if line.startswith("Sat 31 Oct 10:00 GMT"))
    assert wind in row and rain in row


def test_each_row_has_every_value_in_the_order_check_row_applies_them():
    run = go(full_48h_world(), WEATHER)
    hours = run.artifact("get_weather").hours
    assert len(hours) == 6
    assert [v.name for v in hours[0].values] == [r.name for r in READINGS]
    names = [r.label for r in READINGS]
    first_row = next(line for line in run.text("get_weather").splitlines() if line.startswith("Sat 31 Oct 10:00 GMT"))
    positions = [first_row.index(name) for name in names]
    assert positions == sorted(positions)


def test_a_value_the_forecast_does_not_have_prints_as_not_forecast_never_as_0():
    run = go(World(forecast=forecast_fixture("hole_in_cape")), [calls("get_weather", start="2026-10-31 00:00", end="2026-11-01 00:00")])
    text, artifact = run.text("get_weather"), run.artifact("get_weather")
    holes = [(h.label, v) for h in artifact.hours for v in h.values if v.value is None]
    assert holes, "the fixture has a hole in CAPE"
    assert all(v.name == "cape" for _, v in holes)
    for label, value in holes:
        row = next(line for line in text.splitlines() if line.startswith(label))
        assert re.search(r"CAPE not forecast \(", row)
    assert "CAPE 0 J/kg" not in "".join(line for line in text.splitlines() if any(line.startswith(label) for label, _ in holes))


def test_an_hour_with_nothing_forecast_says_so_on_one_line():
    first, last = datetime(2026, 10, 31, 10, tzinfo=UTC), datetime(2026, 10, 31, 16, tzinfo=UTC)
    every_value = dict.fromkeys(WeatherValues.model_fields)
    run = go(World(forecast=weather_forecast(first, last, after={first: every_value})), WEATHER)
    assert "Sat 31 Oct 10:00 GMT: not forecast for every value." in run.text("get_weather")


def test_the_window_header_fetch_time_and_zone_labels():
    run = go(full_48h_world(), WEATHER)
    text = run.text("get_weather")
    assert "Window: Saturday 31 Oct 10:00 – 16:00 GMT." in text
    assert "Fetched: 2026-10-31 07:30 GMT." in text


def test_an_assumed_window_is_said_to_be_assumed():
    run = go(full_48h_world(), [calls("get_weather")])
    assert "assumed window: " in run.text("get_weather")
    assert run.artifact("get_weather").window_assumed is True


def test_hours_across_the_clock_change_are_labelled_bst_then_gmt():
    """The clocks went back at 01:00Z on 25 Oct 2026: 00:00Z is 01:00 BST, 01:00Z is 01:00 GMT."""
    ctx = RunContext(now=datetime(2026, 10, 24, 20, 0, tzinfo=UTC))
    world = World(forecast=weather_forecast(datetime(2026, 10, 24, 20, tzinfo=UTC), datetime(2026, 10, 25, 6, tzinfo=UTC)))
    run = go(world, [calls("get_weather", start="2026-10-25 00:30", end="2026-10-25 03:00")], ctx=ctx)
    labels = [h.label for h in run.artifact("get_weather").hours]
    assert labels[0].endswith("BST") and labels[-1].endswith("GMT")
    assert {label.rsplit(" ", 1)[1] for label in labels} == {"BST", "GMT"}


def test_the_artifact_carries_the_lz_circle_and_the_source():
    run = go(full_48h_world(), WEATHER)
    artifact = run.artifact("get_weather")
    assert [f.properties.kind for f in artifact.overlays] == ["lz_radius"]
    assert artifact.source.state == SourceState.OK and artifact.source.check_by_hand is None
    assert artifact.window_end > artifact.window_start


def test_a_failed_source_is_named_with_the_manual_check_and_gives_no_values():
    first, last = datetime(2026, 10, 31, 7, tzinfo=UTC), datetime(2026, 11, 8, tzinfo=UTC)
    world = World(forecast=weather_forecast(first, last, state=SourceState.FAILED))
    run = go(world, WEATHER)
    text = run.text("get_weather")
    assert "FAILED" in text and CHECK_BY_HAND[SourceName.WEATHER] in text
    assert "mph" not in text
    artifact = run.artifact("get_weather")
    assert artifact.hours == [] and artifact.source.state == SourceState.FAILED
    assert artifact.source.check_by_hand == CHECK_BY_HAND[SourceName.WEATHER]


def test_a_window_that_has_passed_is_relayed_and_nothing_is_fetched():
    world = full_48h_world()
    run = go(world, [calls("get_weather", start="2026-10-30 10:00", end="2026-10-30 16:00")])
    assert world.calls == []
    assert run.progress == []
    assert run.of("get_weather")[0].content and run.artifact("get_weather") is None


def test_it_writes_its_progress_line_once_the_window_is_good():
    run = go(full_48h_world(), WEATHER)
    assert run.progress == ["Checking weather…"]
