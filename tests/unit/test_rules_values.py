"""G22: the initial values are in rules.yaml exactly as Plan 2 gives them."""

import pytest

from flysafelz.rules import RULES

# (rule id, value, unit, comparison): one row per G22 table cell and per other setting.
EXPECTED = [
    ("wind_10m_caution", 12, "mph", ">="),
    ("wind_10m_no_go", 16, "mph", ">="),
    ("gust_spread_caution", 4, "mph", ">="),
    ("gust_spread_no_go", 6, "mph", ">"),
    ("wind_aloft_caution", 16, "mph", ">="),
    ("gradient_caution", 9, "mph", ">="),
    ("gradient_no_go", 16, "mph", ">="),
    ("precipitation_no_go", 0.1, "mm/h", ">="),
    ("rain_chance_caution", 50, "%", ">="),
    ("visibility_caution", 5000, "m", "<"),
    ("visibility_no_go", 1500, "m", "<"),
    ("low_cloud_caution", 90, "%", ">="),
    ("thermic_shortwave_caution", 600, "W/m²", ">="),
    ("thermic_cape_caution", 300, "J/kg", ">="),
    ("cape_no_go", 1000, "J/kg", ">="),
    ("ceiling_left_no_go", 300, "ft", "<"),
    ("ceiling_buffer", 200, "ft", "="),
    ("fl_ft_per_hpa", 27, "ft/hPa", "="),
    ("fl_reference_hpa", 1013.25, "hPa", "="),
    ("local_area", 2, "NM", "="),
    ("airspace_box_nm", 10, "NM", "="),
    ("band_low", 5000, "ft", "="),
    ("band_high", 10000, "ft", "="),
    ("verdict_horizon_day", 3, "day", "="),
    ("outlook_horizon_day", 7, "day", "="),
    ("pib_stale_hours", 3, "h", "="),
    ("uk_bounds_lat_min", 49.8, "deg", "<"),
    ("uk_bounds_lat_max", 61.0, "deg", ">"),
    ("uk_bounds_lon_min", -8.7, "deg", "<"),
    ("uk_bounds_lon_max", 2.0, "deg", ">"),
    ("civil_twilight_depression", 6, "deg", "="),
    ("notam_max_lower_fl", 195, "FL", ">"),
    ("notam_ignore_subject_letters", ["L", "F", "M", "N", "I", "C", "S", "O"], "subject first letter", "in"),
    ("notam_keep_subjects", ["FA", "OB", "OL"], "subject", "in"),
    ("notam_wide_ignore_subjects", ["AF"], "subject", "in"),
    ("notam_ignore_traffic", ["I"], "Q-line traffic", "in"),
    ("notam_ignore_purpose", ["K"], "Q-line purpose", "in"),
    ("notam_ignore_security_subjects", ["RO"], "subject", "in"),
    ("notam_restriction_subjects", ["RD", "RR", "RP", "RT"], "subject", "in"),
    ("notam_awareness_subjects", ["GW"], "subject", "in"),
    ("notam_area_wide_nm", 50, "NM", ">="),
    ("capture_trim_nm", 25, "NM", ">"),
]


@pytest.mark.parametrize(("rule_id", "value", "unit", "comparison"), EXPECTED, ids=[e[0] for e in EXPECTED])
def test_rule_value_unit_comparison_and_source(rule_id, value, unit, comparison):
    rule = getattr(RULES, rule_id)
    assert rule.id == rule_id
    assert rule.value == value
    assert rule.unit == unit
    assert rule.comparison.value == comparison
    assert rule.source.strip()
    assert rule.confidence.value in {"sourced", "best practice", "educated guess"}
    assert rule.note.strip()


def test_no_rule_is_missing_from_the_table_and_none_is_extra():
    assert sorted(e[0] for e in EXPECTED) == sorted(r.id for r in RULES.all())
