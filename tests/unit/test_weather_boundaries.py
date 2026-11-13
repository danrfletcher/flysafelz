"""G14: a value exactly on a limit is a caution (APPROACHING LIMITS); NO-GO needs a value strictly past it.

One table row per rule. Every probe value is built from RULES, so a later change to a limit moves the
probes with it and cannot silently skip them. Checks go through check_row and assess with constructed
rows and forecasts; nothing here touches the network.
"""

import math

import pytest
from factories import run_assess, utc, weather_forecast
from weather_helpers import calm_row

from flysafelz.assess import display
from flysafelz.contracts import Verdict
from flysafelz.rules import RULES
from flysafelz.weather import TRACE_RAIN_HINT, check_row

R = RULES

# The comparison of every rule in rules.yaml. A change to any of them has to be made here on purpose.
EXPECTED_COMPARISONS = {
    "wind_10m_caution": ">=",
    "wind_10m_no_go": ">",  # G14
    "gust_spread_caution": ">=",
    "gust_spread_no_go": ">",
    "wind_aloft_caution": ">=",
    "gradient_caution": ">=",
    "gradient_no_go": ">",  # G14
    "precipitation_no_go": ">",  # G14
    "rain_chance_caution": ">=",
    "visibility_caution": "<=",  # G14
    "visibility_no_go": "<",
    "low_cloud_caution": ">=",
    "thermic_shortwave_caution": ">=",
    "thermic_cape_caution": ">=",
    "cape_no_go": ">",  # G14
    "ceiling_left_no_go": "<",
    "ceiling_buffer": "=",
    "fl_ft_per_hpa": "=",
    "fl_reference_hpa": "=",
    "local_area": "=",
    "airspace_box_nm": "=",
    "band_low": "=",
    "band_high": "=",
    "verdict_horizon_day": "=",
    "outlook_horizon_day": "=",
    "pib_stale_hours": "=",
    "uk_bounds_lat_min": "<",
    "uk_bounds_lat_max": ">",
    "uk_bounds_lon_min": "<",
    "uk_bounds_lon_max": ">",
    "civil_twilight_depression": "=",
    "notam_max_lower_fl": ">",
    "notam_ignore_subject_letters": "in",
    "notam_keep_subjects": "in",
    "notam_wide_ignore_subjects": "in",
    "notam_ignore_traffic": "in",
    "notam_ignore_purpose": "in",
    "notam_security_series": "in",
    "notam_ignore_security_subjects": "in",
    "notam_restriction_subjects": "in",
    "notam_awareness_subjects": "in",
    "notam_area_wide_nm": ">=",
    "capture_trim_nm": ">",
    "reader_area_slack_nm": ">",
}


def test_every_rules_comparison_is_the_expected_one():
    assert {rule.id: rule.comparison.value for rule in R.all()} == EXPECTED_COMPARISONS


def test_the_five_g14_comparisons():
    assert [R.wind_10m_no_go.comparison.value, R.gradient_no_go.comparison.value] == [">", ">"]
    assert [R.cape_no_go.comparison.value, R.precipitation_no_go.comparison.value] == [">", ">"]
    assert R.visibility_caution.comparison.value == "<="


def test_no_note_says_a_value_exactly_on_a_limit_is_no_go():
    assert "16.0 mph is NO_GO" not in R.wind_10m_no_go.note
    assert "16.0 mph is APPROACHING LIMITS; NO_GO needs more than 16.0 mph." in R.wind_10m_no_go.note
    for rule in (R.gradient_no_go, R.cape_no_go, R.visibility_caution):
        assert "APPROACHING LIMITS" in rule.note, rule.id
    assert "0.1 mm/h (trace forecast rain) is APPROACHING LIMITS" in R.precipitation_no_go.note
    assert "0.2 mm/h or more (sustained rain) is NO_GO because of wet-wing deep-stall risk and prop erosion." in R.precipitation_no_go.note


def _row(field: str, x: float):
    """A calm row with one probed value set (the derived values are row fields, the rest are weather values)."""
    return calm_row(**{field: x})


def _result(field: str, x: float) -> list[tuple[str, str]]:
    return [(f.kind, f.rule_id) for f in check_row(_row(field, x))]


def _past(rule, by: float) -> float:
    return rule.value + by


# (name, field on the row, rule, a value just past the limit, what is found exactly on the limit, what is found just past it)
BOUNDARIES = [
    ("wind_10m", "wind_10m", R.wind_10m_no_go, 0.1, [("caution", "wind_10m_caution")], [("blocker", "wind_10m_no_go")]),
    ("gradient", "gradient", R.gradient_no_go, 0.1, [("caution", "gradient_caution")], [("blocker", "gradient_no_go")]),
    ("cape", "cape", R.cape_no_go, 1.0, [("caution", "cape_no_go")], [("blocker", "cape_no_go")]),
    ("precipitation", "precipitation", R.precipitation_no_go, 0.1, [("caution", "precipitation_no_go")], [("blocker", "precipitation_no_go")]),
]


@pytest.mark.parametrize(("name", "field", "rule", "past_by", "on_limit", "past"), BOUNDARIES, ids=[b[0] for b in BOUNDARIES])
def test_exactly_on_a_no_go_limit_is_a_caution_and_past_it_is_a_blocker(name, field, rule, past_by, on_limit, past):
    assert _result(field, rule.value) == on_limit
    assert _result(field, _past(rule, past_by)) == past


def test_wind_at_10_m_exactly_16_0_is_a_caution_and_16_1_is_no_go():
    assert R.wind_10m_no_go.value == 16
    assert _result("wind_10m", 16.0) == [("caution", "wind_10m_caution")]
    assert _result("wind_10m", 16.1) == [("blocker", "wind_10m_no_go")]


def test_gradient_exactly_16_is_a_caution_and_16_1_is_no_go():
    assert _result("gradient", 16.0) == [("caution", "gradient_caution")]
    assert _result("gradient", 16.1) == [("blocker", "gradient_no_go")]


def test_cape_exactly_on_the_limit_cites_cape_no_go_as_a_caution_and_one_more_is_no_go():
    [found] = check_row(_row("cape", R.cape_no_go.value))
    assert (found.kind, found.rule_id) == ("caution", "cape_no_go")
    assert found.text == "CAPE 1000 J/kg, on the limit 1000"
    [found] = check_row(_row("cape", R.cape_no_go.value + 1))
    assert (found.kind, found.rule_id) == ("blocker", "cape_no_go")


def test_precipitation_zero_is_no_finding_point_one_is_a_caution_and_point_two_is_no_go():
    assert R.precipitation_no_go.value == 0.1
    assert _result("precipitation", 0.0) == []
    assert _result("precipitation", 0.1) == [("caution", "precipitation_no_go")]
    assert _result("precipitation", 0.2) == [("blocker", "precipitation_no_go")]
    assert _result("precipitation", 5.0) == [("blocker", "precipitation_no_go")]


def test_the_trace_rain_caution_says_on_the_limit_and_how_to_check_for_showers():
    [found] = check_row(_row("precipitation", R.precipitation_no_go.value))
    assert found.rule_id == "precipitation_no_go"
    assert "on the limit" in found.text
    assert found.text == ("precipitation 0.1 mm/h, on the limit 0.1; trace rain forecast: check live radar for active showers before launch; a wet wing must not be flown")
    assert TRACE_RAIN_HINT in found.text
    assert (found.value, found.limit) == (0.1, 0.1)


def test_only_the_precipitation_caution_carries_the_trace_rain_text():
    [cape] = check_row(_row("cape", R.cape_no_go.value))
    assert "radar" not in cape.text
    [blocker] = check_row(_row("precipitation", 0.2))
    assert "radar" not in blocker.text


def test_visibility_exactly_5000_is_a_caution_and_5001_has_no_finding():
    assert _result("visibility", R.visibility_caution.value) == [("caution", "visibility_caution")]
    assert _result("visibility", R.visibility_caution.value + 1) == []


def test_visibility_exactly_1500_is_a_caution_not_no_go_and_1499_is_no_go():
    assert R.visibility_no_go.comparison.value == "<"
    assert _result("visibility", R.visibility_no_go.value) == [("caution", "visibility_caution")]
    assert _result("visibility", R.visibility_no_go.value - 1) == [("blocker", "visibility_no_go")]


def test_gust_spread_exactly_6_is_not_no_go_regression():
    assert _result("gust_spread", R.gust_spread_no_go.value) == [("caution", "gust_spread_caution")]
    assert _result("gust_spread", R.gust_spread_no_go.value + 0.1) == [("blocker", "gust_spread_no_go")]


# --- edge cases -----------------------------------------------------------------------------


@pytest.mark.parametrize("noisy", [16.000000001, math.nextafter(16.0, 17.0)], ids=["1e-9", "one-ulp"])
def test_float_noise_past_the_limit_is_no_go_and_never_prints_as_16_mph_limit_16(noisy):
    [found] = check_row(_row("wind_10m", noisy))
    assert found.kind == "blocker"
    assert found.rule_id == "wind_10m_no_go"
    assert found.text == "wind at 10 m 16.00 mph, limit 16"
    assert found.text != "wind at 10 m 16 mph, limit 16"


def test_a_no_go_value_that_rounds_to_its_limit_prints_two_decimals():
    [found] = check_row(_row("wind_10m", 16.04))
    assert (found.kind, found.text) == ("blocker", "wind at 10 m 16.04 mph, limit 16")
    [found] = check_row(_row("visibility", 1499.96))  # "<": 1499.96 is below 1500 but prints as 1500
    assert (found.kind, found.text) == ("blocker", "visibility 1499.96 m, limit 1500")


def test_a_no_go_value_that_prints_differently_from_its_limit_keeps_the_short_form():
    [found] = check_row(_row("wind_10m", 16.1))
    assert found.text == "wind at 10 m 16.1 mph, limit 16"


def test_on_one_limit_and_past_another_the_blocker_comes_first_and_the_on_limit_caution_is_still_listed():
    row = calm_row(wind_10m=16.0, gradient=16.1)
    assert [(f.kind, f.rule_id) for f in check_row(row)] == [("blocker", "gradient_no_go"), ("caution", "wind_10m_caution")]
    row = calm_row(precipitation=0.1, cape=R.cape_no_go.value + 1)
    assert [(f.kind, f.rule_id) for f in check_row(row)] == [("blocker", "cape_no_go"), ("caution", "precipitation_no_go")]


def test_a_missing_value_gives_no_finding():
    row = calm_row(wind_10m=None, precipitation=None, cape=None, visibility=None, gradient=None)
    assert check_row(row) == []


def test_cape_on_the_limit_with_strong_shortwave_gives_both_cautions_and_no_blocker():
    row = calm_row(cape=R.cape_no_go.value, shortwave_radiation=R.thermic_shortwave_caution.value)
    assert [(f.kind, f.rule_id) for f in check_row(row)] == [("caution", "cape_no_go"), ("caution", "thermic_cape_caution")]


# --- through assess ------------------------------------------------------------------------------


def _assessed(**weather):
    """assess() at Blue Mill 09:00-12:00Z on a calm day, with these weather values from 08:00 on (so every hour has them)."""
    forecast = weather_forecast(utc(8), utc(13), after={utc(8): weather})
    return run_assess(forecast=forecast)


def _segments(assessment):
    return [s for hour in assessment.hours for s in hour.segments]


def test_a_row_whose_only_finding_is_an_on_the_limit_caution_is_approaching_limits():
    assessment = _assessed(wind_10m=16.0, gust_10m=17.0)
    segments = _segments(assessment)
    assert segments
    for segment in segments:
        assert segment.verdict is Verdict.PILOT_DISCRETION
        assert segment.approaching_limits is True
        assert segment.blockers == []
        assert [c.rule_id for c in segment.cautions] == ["wind_10m_caution"]
        assert display(segment) == "PILOT DISCRETION \u2013 APPROACHING LIMITS"
    assert assessment.overall_verdict is Verdict.PILOT_DISCRETION


def test_wind_at_10_m_16_1_is_no_go_through_assess():
    segments = _segments(_assessed(wind_10m=16.1, gust_10m=17.0))
    assert {s.verdict for s in segments} == {Verdict.NO_GO}
    assert {display(s) for s in segments} == {"NO-GO"}
    assert all(b.text == "wind at 10 m 16.1 mph, limit 16" for s in segments for b in s.blockers)


def test_trace_rain_is_approaching_limits_through_assess_with_the_radar_text():
    segments = _segments(_assessed(precipitation=0.1))
    assert segments
    for segment in segments:
        assert segment.verdict is Verdict.PILOT_DISCRETION
        assert segment.approaching_limits is True
        [caution] = segment.cautions
        assert caution.rule_id == "precipitation_no_go"
        assert "on the limit" in caution.text
        assert "check live radar for active showers before launch; a wet wing must not be flown" in caution.text


@pytest.mark.parametrize(
    ("weather", "rule_id"),
    [
        ({"cape": 1000.0}, "cape_no_go"),
        ({"visibility": 5000.0}, "visibility_caution"),
        ({"visibility": 1500.0}, "visibility_caution"),
        ({"wind_120m": 25.0, "wind_10m": 9.0}, "gradient_caution"),
    ],
    ids=["cape", "visibility-5000", "visibility-1500", "gradient-16"],
)
def test_each_other_on_the_limit_value_is_approaching_limits_through_assess(weather, rule_id):
    segments = _segments(_assessed(**weather))
    for segment in segments:
        assert segment.verdict is Verdict.PILOT_DISCRETION, weather
        assert segment.approaching_limits is True
        assert rule_id in [c.rule_id for c in segment.cautions]
