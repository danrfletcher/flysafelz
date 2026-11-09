"""G2 / G3 checks: at and around every caution and NO_GO limit, built from the loaded rules.

Each case probes a rule just below, exactly at, and just above its limit. The expected
rule ids are written out by hand per probe, because the comparison differs: NO_GO needs a value
strictly past its limit, so 16.0 mph is a caution under "> 16", as is a gust spread of 6.0 under "> 6".
The kind of each finding (caution or blocker) at the limit is tested in test_weather_boundaries.py.
"""

import pytest
from weather_helpers import calm_row, rules_with

from flysafelz import weather
from flysafelz.rules import RULES
from flysafelz.weather import Finding, check_row

R = RULES


def step(rule) -> float:
    return abs(rule.value) / 100


# (id, rule, how to set the probed value on a calm row, ids found [below, at, above])
CASES = [
    ("wind_10m_caution", R.wind_10m_caution, lambda x: {"wind_10m": x}, [[], ["wind_10m_caution"], ["wind_10m_caution"]]),
    ("wind_10m_no_go", R.wind_10m_no_go, lambda x: {"wind_10m": x}, [["wind_10m_caution"], ["wind_10m_caution"], ["wind_10m_no_go"]]),  # exactly at ">" is a caution
    ("gust_spread_caution", R.gust_spread_caution, lambda x: {"gust_spread": x}, [[], ["gust_spread_caution"], ["gust_spread_caution"]]),
    (
        "gust_spread_no_go",
        R.gust_spread_no_go,
        lambda x: {"gust_spread": x},
        [["gust_spread_caution"], ["gust_spread_caution"], ["gust_spread_no_go"]],  # exactly at ">" is not NO_GO
    ),
    ("wind_aloft_80m", R.wind_aloft_caution, lambda x: {"wind_80m": x}, [[], ["wind_aloft_caution"], ["wind_aloft_caution"]]),
    ("wind_aloft_120m", R.wind_aloft_caution, lambda x: {"wind_120m": x}, [[], ["wind_aloft_caution"], ["wind_aloft_caution"]]),
    ("gradient_caution", R.gradient_caution, lambda x: {"gradient": x}, [[], ["gradient_caution"], ["gradient_caution"]]),
    ("gradient_no_go", R.gradient_no_go, lambda x: {"gradient": x}, [["gradient_caution"], ["gradient_caution"], ["gradient_no_go"]]),
    ("precipitation_no_go", R.precipitation_no_go, lambda x: {"precipitation": x}, [[], ["precipitation_no_go"], ["precipitation_no_go"]]),  # at: on-the-limit caution; above: blocker
    (
        "rain_chance_caution",
        R.rain_chance_caution,
        lambda x: {"precipitation_probability": x},
        [[], ["rain_chance_caution"], ["rain_chance_caution"]],
    ),
    (
        "visibility_caution",
        R.visibility_caution,
        lambda x: {"visibility": x},
        [["visibility_caution"], ["visibility_caution"], []],  # "<=": at the limit is a caution
    ),
    (
        "visibility_no_go",
        R.visibility_no_go,
        lambda x: {"visibility": x},
        [["visibility_no_go"], ["visibility_caution"], ["visibility_caution"]],
    ),
    ("low_cloud_caution", R.low_cloud_caution, lambda x: {"cloud_cover_low": x}, [[], ["low_cloud_caution"], ["low_cloud_caution"]]),
    ("cape_no_go", R.cape_no_go, lambda x: {"cape": x}, [[], ["cape_no_go"], ["cape_no_go"]]),  # shortwave is calm: no thermic caution
]


def _probe(rule, setter, x):
    over = setter(x)
    derived = {k: over.pop(k) for k in ("gust_spread", "gradient") if k in over}
    return check_row(calm_row(**over, **derived))


@pytest.mark.parametrize(("name", "rule", "setter", "expected"), CASES, ids=[c[0] for c in CASES])
def test_every_limit_below_at_and_above(name, rule, setter, expected):
    below, at, above = rule.value - step(rule), rule.value, rule.value + step(rule)
    found = [[f.rule_id for f in _probe(rule, setter, x)] for x in (below, at, above)]
    assert found == expected


def test_a_calm_row_has_no_blockers_and_no_cautions():
    assert check_row(calm_row()) == []


def test_a_row_with_every_value_just_inside_its_limit_is_also_clean():
    row = calm_row(
        wind_10m=R.wind_10m_caution.value - 0.1,
        gust_spread=R.gust_spread_caution.value - 0.1,
        wind_80m=R.wind_aloft_caution.value - 0.1,
        wind_120m=R.wind_aloft_caution.value - 0.1,
        gradient=R.gradient_caution.value - 0.1,
        precipitation=R.precipitation_no_go.value - 0.01,
        precipitation_probability=R.rain_chance_caution.value - 1,
        visibility=R.visibility_caution.value + 1,
        cloud_cover_low=R.low_cloud_caution.value - 1,
        cape=R.thermic_cape_caution.value - 1,
        shortwave_radiation=R.thermic_shortwave_caution.value,
    )
    assert check_row(row) == []


# --- thermic: shortwave AND CAPE (G3) -------------------------------------------------

SW = R.thermic_shortwave_caution.value
CAPE = R.thermic_cape_caution.value


@pytest.mark.parametrize(
    ("shortwave", "cape", "expected"),
    [
        (SW, CAPE, ["thermic_cape_caution"]),  # both exactly at their limits
        (SW + 100, CAPE + 100, ["thermic_cape_caution"]),
        (SW - 1, CAPE + 100, []),  # CAPE alone is not enough
        (SW + 100, CAPE - 1, []),  # shortwave alone is not enough
        (0, R.cape_no_go.value - 1, []),
        (SW + 100, R.cape_no_go.value - 1, ["thermic_cape_caution"]),  # just under CAPE NO_GO
        (SW + 100, R.cape_no_go.value, ["cape_no_go", "thermic_cape_caution"]),  # exactly on the limit: two cautions
        (0, R.cape_no_go.value, ["cape_no_go"]),  # ... and the on-the-limit caution needs no shortwave
        (SW + 100, R.cape_no_go.value + 1, ["cape_no_go"]),  # past the limit: the blocker wins over the thermic caution
        (0, R.cape_no_go.value + 1, ["cape_no_go"]),  # ... whatever the shortwave
        (SW - 1, R.cape_no_go.value + 500, ["cape_no_go"]),
    ],
)
def test_thermic_caution_needs_both_and_cape_past_no_go_needs_neither(shortwave, cape, expected):
    found = check_row(calm_row(shortwave_radiation=shortwave, cape=cape))
    assert [f.rule_id for f in found] == expected


# --- what a result says ----------------------------------------------------------------


def test_a_result_says_its_kind_rule_value_and_limit_in_plain_words():
    [caution] = check_row(calm_row(gust_spread=5))
    assert caution == Finding(kind="caution", rule_id="gust_spread_caution", text="gust spread 5 mph, limit 4", value=5, limit=4)
    [blocker] = check_row(calm_row(gust_spread=7.5))
    assert blocker.kind == "blocker"
    assert blocker.text == "gust spread 7.5 mph, limit 6"
    assert (blocker.value, blocker.limit) == (7.5, 6)


def test_text_for_other_units():
    assert check_row(calm_row(visibility=4000))[0].text == "visibility 4000 m, limit 5000"
    assert check_row(calm_row(precipitation_probability=60))[0].text == "rain chance 60 %, limit 50"
    assert check_row(calm_row(precipitation=0.3))[0].text == "precipitation 0.3 mm/h, limit 0.1"
    assert check_row(calm_row(wind_120m=17))[0].text == "wind at 120 m 17 mph, limit 16"
    assert check_row(calm_row(shortwave_radiation=650, cape=350))[0].text == ("thermic: shortwave 650 W/m² (limit 600) and CAPE 350 J/kg, limit 300")


def test_results_come_blockers_first_then_cautions_in_a_fixed_order():
    row = calm_row(
        wind_10m=13,  # caution
        wind_80m=17,  # caution
        precipitation=1.0,  # blocker
        visibility=1000,  # blocker
        precipitation_probability=80,  # caution
        cloud_cover_low=95,  # caution
        gust_spread=5,  # caution
        gradient=10,  # caution
        cape=1200,  # blocker
    )
    found = check_row(row)
    assert [(f.kind, f.rule_id) for f in found] == [
        ("blocker", "precipitation_no_go"),
        ("blocker", "visibility_no_go"),
        ("blocker", "cape_no_go"),
        ("caution", "wind_10m_caution"),
        ("caution", "gust_spread_caution"),
        ("caution", "wind_aloft_caution"),
        ("caution", "gradient_caution"),
        ("caution", "rain_chance_caution"),
        ("caution", "low_cloud_caution"),
    ]
    assert check_row(row) == found  # the same input gives the same output


def test_wind_aloft_gives_a_caution_never_a_blocker_even_far_above_the_limit():
    found = check_row(calm_row(wind_80m=60, wind_120m=70))
    assert [(f.kind, f.rule_id) for f in found] == [("caution", "wind_aloft_caution")] * 2


def test_a_value_that_is_missing_is_skipped_and_the_rest_still_count():
    row = calm_row(cape=None, wind_10m=20, precipitation_probability=None)
    assert [f.rule_id for f in check_row(row)] == ["wind_10m_no_go"]


def test_an_optional_missing_variable_skips_only_its_check():
    row = calm_row(precipitation_probability=None, cloud_cover=None, direction_10m=None)
    assert check_row(row) == []


def test_no_summed_score_and_no_go_string_in_a_result():
    found = check_row(calm_row(wind_10m=13, gust_spread=7, precipitation=1))
    assert {f.kind for f in found} <= {"blocker", "caution"}
    for f in found:
        assert not any(word in f.text.split() for word in ("GO", "score"))
        assert set(Finding.model_fields) == {"kind", "rule_id", "text", "value", "limit"}


# --- F3: the limits come from rules.yaml ------------------------------------------------


def test_a_changed_rules_file_changes_the_result(tmp_path, monkeypatch):
    row = calm_row(wind_10m=17)
    assert [f.rule_id for f in check_row(row)] == ["wind_10m_no_go"]
    monkeypatch.setattr(weather, "RULES", rules_with(tmp_path, "wind_10m_no_go", 20))
    assert [f.rule_id for f in check_row(row)] == ["wind_10m_caution"]
    [blocker] = check_row(calm_row(wind_10m=21))
    assert (blocker.kind, blocker.text) == ("blocker", "wind at 10 m 21 mph, limit 20")


def test_a_changed_thermic_limit_changes_the_result(tmp_path, monkeypatch):
    row = calm_row(shortwave_radiation=SW, cape=CAPE)
    assert [f.rule_id for f in check_row(row)] == ["thermic_cape_caution"]
    monkeypatch.setattr(weather, "RULES", rules_with(tmp_path, "thermic_shortwave_caution", SW + 50))
    assert check_row(row) == []


def test_a_list_rule_is_not_a_numeric_limit():
    with pytest.raises(TypeError, match="not a numeric limit"):
        weather._limit(R.wind_10m_no_go.model_copy(update={"value": ["a"]}))
