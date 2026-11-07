"""E9, E14, E15: the ceiling. Floors in QNH feet, one ceiling over the LZ, and NO_GO when too little is left."""

import pytest
from airspace_helpers import ELEVATION_FT, LZ, SEGMENT, agl, amsl, assess, fl, only, square, volume, with_rules

from flysafelz.airspace import Outcome, limit_qnh_ft
from flysafelz.rules import RULES, Comparison

WEST_SLEEKBURN = (55.155, -1.565)


# --- E9: how a floor becomes QNH feet -----------------------------------------------------------------


def test_e9_the_same_flight_level_is_lower_at_990_hpa_than_at_1030_hpa():
    low_day = only(assess([volume("ctr", lower=fl(30))], pressure=990.0))
    high_day = only(assess([volume("ctr", lower=fl(30))], pressure=1030.0))
    assert low_day.ceiling_qnh_ft is not None and high_day.ceiling_qnh_ft is not None
    assert low_day.ceiling_qnh_ft < high_day.ceiling_qnh_ft
    assert low_day.floor_qnh_ft == pytest.approx(3000 - (1013.25 - 990.0) * 27)
    assert high_day.floor_qnh_ft == pytest.approx(3000 - (1013.25 - 1030.0) * 27)
    assert low_day.condition == "stay below 2,172 ft QNH" and high_day.condition == "stay below 3,252 ft QNH"


def test_e9_a_flight_level_at_the_reference_pressure_is_hundreds_of_feet():
    assert only(assess([volume("ctr", lower=fl(30))], pressure=1013.25)).floor_qnh_ft == pytest.approx(3000.0)


def test_e9_the_conversion_comes_from_the_rules():
    steeper = with_rules(fl_ft_per_hpa=30)
    assert only(assess([volume("ctr", lower=fl(30))], pressure=990.0, rules=steeper)).floor_qnh_ft == pytest.approx(3000 - 23.25 * 30)
    elsewhere = with_rules(fl_reference_hpa=1000.0)
    assert only(assess([volume("ctr", lower=fl(30))], pressure=990.0, rules=elsewhere)).floor_qnh_ft == pytest.approx(3000 - 10.0 * 27)


def test_e9_an_agl_floor_adds_the_site_elevation_and_an_amsl_floor_does_not():
    on_the_hill = assess([volume("ctr", lower=agl(1000))], elevation=250.0)
    assert only(on_the_hill).floor_qnh_ft == pytest.approx(1250.0) and only(on_the_hill).condition == "stay below 1,050 ft QNH"
    assert only(assess([volume("ctr", lower=amsl(1000))], elevation=250.0)).floor_qnh_ft == pytest.approx(1000.0)


def test_e9_limit_qnh_ft_for_sfc_is_the_ground():
    sfc = volume("ctr").lower
    assert limit_qnh_ft(sfc, 123.0, 1013.25) == 123.0


# --- E14: stacked volumes over the LZ ------------------------------------------------------------------


def test_e14_an_atz_from_sfc_under_a_cta_at_1500_gives_both_the_permission_and_one_ceiling():
    result = assess([volume("atz", name="ATZ"), volume("cta", lower=amsl(1500), cls="D", name="CTA")])
    by_area = {f.area: f for f in result.findings}
    assert by_area["ATZ"].outcome is Outcome.PERMISSION and by_area["ATZ"].condition == "permission from ATZ first"
    assert by_area["CTA"].outcome is Outcome.CEILING and by_area["CTA"].condition == "stay below 1,300 ft QNH"
    assert result.ceiling_qnh_ft == 1300.0 and len(result.findings) == 2


@pytest.mark.parametrize("order", [(0, 1, 2), (2, 1, 0), (1, 0, 2)])
def test_e14_the_lowest_floor_above_sfc_sets_the_single_ceiling_whatever_the_order(order):
    stack = [volume("cta", lower=amsl(3000), cls="C", name="HIGH"), volume("tma", lower=amsl(1500), cls="D", name="LOW"), volume("airway", lower=amsl(2500), cls="E", name="MID")]
    result = assess([stack[i] for i in order])
    ceilings = [f for f in result.findings if f.outcome is Outcome.CEILING]
    assert [(f.area, f.condition) for f in ceilings] == [("LOW", "stay below 1,300 ft QNH")]
    assert result.ceiling_qnh_ft == 1300.0
    ignored = {f.area: f for f in result.findings if f.outcome is Outcome.IGNORE}
    assert set(ignored) == {"HIGH", "MID"} and all("LOW" in f.reason and "sets the ceiling" in f.reason for f in ignored.values())


def test_e14_a_ctr_with_a_floor_in_the_stack_sets_the_ceiling_too():
    result = assess([volume("ctr", lower=amsl(2500), name="CTR STEP"), volume("cta", lower=amsl(1500), cls="D", name="CTA")])
    assert result.ceiling_qnh_ft == 1300.0


def test_e14_a_mid_band_floor_does_not_compete_with_a_lower_ceiling():
    result = assess([volume("cta", lower=amsl(7000), cls="A", name="HIGH"), volume("cta", lower=amsl(1500), cls="D", name="LOW")])
    assert {(f.area, f.outcome) for f in result.findings} == {("HIGH", Outcome.WARNING), ("LOW", Outcome.CEILING)}


def test_e14_each_volumes_own_sfc_outcome_still_applies_under_a_ceiling():
    result = assess([volume("rmz", name="RMZ"), volume("ctr", lower=amsl(2000), name="CTR STEP")])
    assert {(f.area, f.outcome) for f in result.findings} == {("RMZ", Outcome.RADIO), ("CTR STEP", Outcome.CEILING)}


# --- E15: too little room ---------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("floor", "left", "outcome"),
    [
        (800, 570, Outcome.CEILING),
        (531, 301, Outcome.CEILING),
        (530, 300, Outcome.CEILING),  # exactly 300: the rule is "< 300", so 300 is enough
        (529, 299, Outcome.NO_GO),
        (230, 0, Outcome.NO_GO),
        (100, -130, Outcome.NO_GO),
    ],
)
def test_e15_the_ceiling_left_above_the_site_decides(floor, left, outcome):
    finding = only(assess([volume("cta", lower=amsl(floor), cls="D", name="LOW CTA")]))
    assert finding.outcome is outcome
    assert finding.ceiling_qnh_ft == pytest.approx(floor - 200)
    if outcome is Outcome.NO_GO:
        assert f"{floor - 200:,} ft QNH" in finding.reason and "LOW CTA" in finding.reason and finding.condition is None
        assert f"only {left:,} ft above the site" in finding.reason
    else:
        assert finding.condition == f"stay below {floor - 200:,} ft QNH"


def test_e15_at_exactly_the_limit_the_rules_own_comparison_decides():
    at_limit = [volume("cta", lower=amsl(530), cls="D")]  # 300 ft left at a 30 ft site
    assert only(assess(at_limit)).outcome is Outcome.CEILING
    strict = RULES.ceiling_left_no_go.model_copy(update={"comparison": Comparison.LE})
    assert only(assess(at_limit, rules=RULES.model_copy(update={"ceiling_left_no_go": strict}))).outcome is Outcome.NO_GO


def test_e15_the_minimum_comes_from_the_rules():
    cta = [volume("cta", lower=amsl(800), cls="D")]
    assert only(assess(cta)).outcome is Outcome.CEILING
    assert only(assess(cta, rules=with_rules(ceiling_left_no_go=600))).outcome is Outcome.NO_GO
    assert only(assess(cta, rules=with_rules(ceiling_buffer=600))).outcome is Outcome.NO_GO


def test_e15_it_is_the_ceiling_above_the_site_not_above_the_sea():
    # The same 1,500 ft floor leaves 500 ft at a 800 ft site and nothing at a 1,300 ft site.
    cta = [volume("cta", lower=amsl(1500), cls="D")]
    assert only(assess(cta, elevation=800.0)).outcome is Outcome.CEILING
    assert only(assess(cta, elevation=1000.0)).outcome is Outcome.CEILING  # exactly 300 left
    assert only(assess(cta, elevation=1001.0)).outcome is Outcome.NO_GO
    assert only(assess(cta, elevation=1300.0)).outcome is Outcome.NO_GO


def test_e15_when_the_lowest_ceiling_is_no_go_the_findings_say_so_once():
    result = assess([volume("cta", lower=amsl(400), cls="D", name="LOW"), volume("cta", lower=amsl(3000), cls="D", name="HIGH")])
    assert [(f.area, f.outcome) for f in result.findings] == [("LOW", Outcome.NO_GO), ("HIGH", Outcome.IGNORE)]
    assert result.ceiling_qnh_ft == 200.0


# --- West Sleekburn ------------------------------------------------------------------------------------------------


def test_west_sleekburn_a_cta_at_1500_ft_alt_over_a_site_at_30_ft_is_stay_below_1300_and_not_no_go():
    cta = volume("cta", lower=amsl(1500), cls="D", name="NEWCASTLE CTA", shape=square(*WEST_SLEEKBURN, 5.0))
    result = assess([cta], lz=WEST_SLEEKBURN, elevation=30.0)
    finding = only(result)
    assert finding.outcome is Outcome.CEILING
    assert finding.condition == "stay below 1,300 ft QNH"
    assert result.ceiling_qnh_ft == 1300.0


def test_west_sleekburn_the_ceiling_is_the_same_by_segment():
    cta = volume("cta", lower=amsl(1500), cls="D", name="NEWCASTLE CTA", shape=square(*WEST_SLEEKBURN, 5.0))
    for hour in (9, 12, 15):
        from airspace_helpers import utc
        from flysafelz.airspace import TimeSpan

        segment = TimeSpan(start=utc(hour), end=utc(hour + 1))
        assert only(assess([cta], lz=WEST_SLEEKBURN, elevation=30.0, segment=segment)).condition == "stay below 1,300 ft QNH"
    assert ELEVATION_FT == 30.0 and LZ and SEGMENT  # the shared defaults match the case
