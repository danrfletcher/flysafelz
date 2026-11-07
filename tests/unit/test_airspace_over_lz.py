"""G8: what each volume over the LZ means. Table-driven over type x floor band x datum.

Every case builds a small square around the fixed LZ (Blue Mill, site elevation 30 ft, standard
pressure) and runs assess_airspace, so the real shape test and the real rules are used.
"""

import pytest
from airspace_helpers import LZ, agl, amsl, around_lz, assess, fl, only, square, volume, with_rules

from flysafelz.airspace import Outcome
from flysafelz.contracts import Datum, Limit, LimitUnit

NO_GO_TYPES = ["ctr", "prohibited", "restricted", "tmz"]
PERMISSION_TYPES = ["atz", "matz"]
CONTROLLED = ["cta", "tma", "airway"]
OTHER_TYPES = ["gliding", "other", "tra", "tsa", "mtr", "fir", "uir", "adiz", "alert", "warning", "protected", "htz", "mctr", "lta", "uta", "sporting"]

# One floor in each band, written in each datum. At 30 ft elevation and 1013.25 hPa every row of
# a band is the same height above sea level, so the same outcome is expected.
LOW = [("amsl", amsl(2000), 2000), ("agl", agl(1970), 2000), ("fl", fl(20), 2000)]  # 0-5,000 ft: a ceiling
MID = [("amsl", amsl(7000), 7000), ("agl", agl(6970), 7000), ("fl", fl(70), 7000)]  # 5,000-10,000 ft: a warning
HIGH = [("amsl", amsl(12000), 12000), ("agl", agl(11970), 12000), ("fl", fl(120), 12000)]  # above 10,000 ft: awareness
SFC_ROWS = [
    ("sfc", Limit(value=0, unit=LimitUnit.FT, datum=Datum.SFC), None),
    ("agl0", agl(0), None),
    ("amsl at the site elevation", amsl(30), None),
    ("amsl below the site elevation", amsl(10), None),
    ("fl below the site elevation", fl(0), None),
]


def ids(rows):
    return [r[0] for r in rows]


@pytest.mark.parametrize(("type_", "outcome"), [(t, Outcome.NO_GO) for t in NO_GO_TYPES] + [(t, Outcome.PERMISSION) for t in PERMISSION_TYPES] + [("rmz", Outcome.RADIO)])
@pytest.mark.parametrize(("label", "lower", "_height"), SFC_ROWS, ids=ids(SFC_ROWS))
def test_a_g8_type_from_sfc_gives_its_sfc_outcome(type_, outcome, label, lower, _height):
    finding = only(assess([volume(type_, lower=lower, name="TEST AREA")]))
    assert finding.outcome is outcome
    assert "TEST AREA" in finding.reason
    assert finding.bearing_deg is None  # over the LZ, not beside it


def test_the_sfc_wording():
    assert only(assess([volume("ctr", name="HUMBERSIDE")])).reason == "over HUMBERSIDE (CTR) from the surface"
    assert only(assess([volume("atz", name="LEEMING")])).condition == "permission from LEEMING first"
    assert only(assess([volume("matz", name="LEEMING")])).condition == "permission from LEEMING first"
    assert only(assess([volume("rmz", name="TEESSIDE")])).condition == "radio contact with TEESSIDE required"
    assert only(assess([volume("ctr")])).condition is None


@pytest.mark.parametrize("type_", NO_GO_TYPES + PERMISSION_TYPES + ["rmz"])
@pytest.mark.parametrize(("label", "lower", "height"), LOW, ids=ids(LOW))
def test_a_g8_type_with_a_low_floor_gives_a_ceiling(type_, label, lower, height):
    finding = only(assess([volume(type_, lower=lower)]))
    assert finding.outcome is Outcome.CEILING
    assert finding.condition == f"stay below {height - 200:,} ft QNH"
    assert finding.floor_qnh_ft == pytest.approx(height) and finding.ceiling_qnh_ft == pytest.approx(height - 200)


@pytest.mark.parametrize("type_", NO_GO_TYPES + PERMISSION_TYPES + ["rmz"])
@pytest.mark.parametrize(("label", "lower", "height"), MID, ids=ids(MID))
def test_a_g8_type_with_a_floor_between_the_bands_gives_a_warning(type_, label, lower, height):
    finding = only(assess([volume(type_, lower=lower, name="TEST AREA")]))
    assert finding.outcome is Outcome.WARNING
    assert finding.condition is not None and "TEST AREA" in finding.condition and f"{height:,} ft QNH" in finding.condition


@pytest.mark.parametrize("type_", NO_GO_TYPES + PERMISSION_TYPES + ["rmz"])
@pytest.mark.parametrize(("label", "lower", "height"), HIGH, ids=ids(HIGH))
def test_a_g8_type_with_a_floor_above_the_bands_is_awareness_only(type_, label, lower, height):
    finding = only(assess([volume(type_, lower=lower)]))
    assert finding.outcome is Outcome.AWARENESS and finding.condition is None


@pytest.mark.parametrize("type_", CONTROLLED)
@pytest.mark.parametrize("cls", ["A", "B", "C", "D", "E"])
@pytest.mark.parametrize(("label", "lower", "height"), LOW + MID + HIGH, ids=ids(LOW + MID + HIGH))
def test_a_cta_tma_or_airway_of_class_a_to_e_follows_the_floor_bands(type_, cls, label, lower, height):
    expected = Outcome.CEILING if height < 5000 else Outcome.WARNING if height < 10000 else Outcome.AWARENESS
    assert only(assess([volume(type_, lower=lower, cls=cls)])).outcome is expected


@pytest.mark.parametrize("type_", CONTROLLED)
@pytest.mark.parametrize("cls", ["F", "G", "SUA"])
@pytest.mark.parametrize(("label", "lower", "height"), LOW + MID + HIGH, ids=ids(LOW + MID + HIGH))
def test_a_cta_tma_or_airway_of_class_f_or_g_is_awareness_at_any_floor(type_, cls, label, lower, height):
    assert only(assess([volume(type_, lower=lower, cls=cls)])).outcome is Outcome.AWARENESS


@pytest.mark.parametrize("type_", CONTROLLED)
def test_a_controlled_type_with_no_class_is_treated_as_controlled_and_noted(type_):
    result = assess([volume(type_, lower=amsl(2000), cls=None, name="NO CLASS")])
    assert only(result).outcome is Outcome.CEILING
    assert any("NO CLASS: ICAO class unknown" in n for n in result.notes)


@pytest.mark.parametrize("type_", OTHER_TYPES)
@pytest.mark.parametrize("lower", [Limit(value=0, unit=LimitUnit.FT, datum=Datum.SFC), amsl(2000), amsl(7000), amsl(12000), fl(60)], ids=["sfc", "low", "mid", "high", "fl"])
def test_gliding_and_every_other_type_is_awareness_at_any_floor(type_, lower):
    finding = only(assess([volume(type_, lower=lower)]))
    assert finding.outcome is Outcome.AWARENESS and finding.condition is None


@pytest.mark.parametrize(
    ("lower", "outcome"),
    [
        (amsl(5000), Outcome.CEILING),  # exactly at the low band edge is still a ceiling
        (amsl(5001), Outcome.WARNING),
        (amsl(10000), Outcome.WARNING),  # exactly at the high band edge is still a warning
        (amsl(10001), Outcome.AWARENESS),
    ],
)
def test_the_band_edges(lower, outcome):
    assert only(assess([volume("ctr", lower=lower)])).outcome is outcome


def test_a_cta_that_reaches_the_ground_is_no_go_through_the_ceiling_rule():
    finding = only(assess([volume("cta", cls="D", name="GROUND CTA")]))
    assert finding.outcome is Outcome.NO_GO and "GROUND CTA" in finding.reason


def test_the_band_edges_follow_the_rules_not_the_code():
    mid = amsl(7000)
    assert only(assess([volume("ctr", lower=mid)])).outcome is Outcome.WARNING
    assert only(assess([volume("ctr", lower=mid)], rules=with_rules(band_low=8000))).outcome is Outcome.CEILING
    assert only(assess([volume("ctr", lower=mid)], rules=with_rules(band_high=6000))).outcome is Outcome.AWARENESS


def test_a_floor_at_or_below_the_site_elevation_counts_as_sfc_for_every_datum():
    high_site = 800.0
    for lower in (amsl(800), amsl(500), agl(0), fl(5)):
        assert only(assess([volume("atz", lower=lower)], elevation=high_site)).outcome is Outcome.PERMISSION
    assert only(assess([volume("atz", lower=amsl(801))], elevation=high_site)).outcome is Outcome.NO_GO  # above the site, but no room left
    assert only(assess([volume("atz", lower=amsl(1500))], elevation=high_site)).outcome is Outcome.CEILING


# --- where the LZ is -------------------------------------------------------------------


def test_an_lz_exactly_on_the_boundary_is_over_the_volume():
    lat, lon = LZ
    on_vertex = [[[(lat, lon), (lat, lon + 0.02), (lat + 0.02, lon + 0.02), (lat + 0.02, lon), (lat, lon)]]]
    assert only(assess([volume("ctr", shape=on_vertex)])).outcome is Outcome.NO_GO
    on_edge = [[[(lat - 0.01, lon), (lat - 0.01, lon + 0.02), (lat + 0.01, lon + 0.02), (lat + 0.01, lon), (lat - 0.01, lon)]]]
    assert only(assess([volume("ctr", shape=on_edge)])).outcome is Outcome.NO_GO


def test_an_lz_inside_a_hole_is_not_over_the_volume():
    outer = around_lz(2.0)[0][0]
    hole = around_lz(0.5)[0][0]
    donut = [[outer, hole]]
    finding = only(assess([volume("ctr", shape=donut)]))
    assert finding.outcome is Outcome.AVOID_SECTOR  # inside the hole: not over, but its edge is within the local area


def test_a_volume_far_away_gives_nothing():
    far = square(LZ[0] + 0.5, LZ[1], 1.0)
    result = assess([volume("ctr", shape=far)])
    assert result.findings == [] and result.ceiling_qnh_ft is None


def test_two_volumes_each_get_their_own_finding():
    result = assess([volume("ctr", name="ONE"), volume("rmz", name="TWO"), volume("gliding", name="THREE")])
    assert {(f.area, f.outcome) for f in result.findings} == {("ONE", Outcome.NO_GO), ("TWO", Outcome.RADIO), ("THREE", Outcome.AWARENESS)}


def test_an_item_noted_while_reading_carries_its_notes_into_the_assessment():
    area = volume("ctr")
    area = area.model_copy(update={"notes": ["TEST CTR: upper limit missing; treated as unlimited"]})
    assert assess([area]).notes == ["TEST CTR: upper limit missing; treated as unlimited"]


def test_a_missing_pressure_uses_the_reference_pressure_and_says_so():
    result = assess([volume("ctr", lower=fl(20))], pressure=None)
    assert only(result).condition == "stay below 1,800 ft QNH"
    assert any("no forecast pressure" in n for n in result.notes)
