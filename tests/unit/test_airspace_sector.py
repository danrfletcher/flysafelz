"""G9 and E22: volumes within the local area but not over the LZ.

The rule: every non-AWARENESS G8 outcome, moved 1 NM away at each of the 16 sector centres, gives
the compass16 label for that centre and a bearing within 0.5 degrees. The same volume 2.5 NM away gives nothing.
"""

import math

import pytest
from airspace_helpers import LZ, activation, amsl, assess, offset, only, reading, square, utc, volume, with_rules

from flysafelz.contracts import Circle, ContactKind
from flysafelz.airspace import Outcome, TimeSpan
from flysafelz.geo import COMPASS_POINTS, compass16, shape_edge_bearing_deg, shape_edge_distance_nm

TINY = 0.003  # half-width in NM: small enough that the nearest point is in the direction of the centre
CENTRES = [(k * 22.5, COMPASS_POINTS[k]) for k in range(len(COMPASS_POINTS))]

# The non-AWARENESS outcomes of G8: what the same volume over the LZ would be, and what it is beside the LZ.
VOLUMES = [
    ("ctr from the surface", dict(type_="ctr"), Outcome.AVOID_SECTOR, "avoid the sector to the {compass}"),
    ("prohibited from the surface", dict(type_="prohibited"), Outcome.AVOID_SECTOR, "avoid the sector to the {compass}"),
    ("restricted from the surface", dict(type_="restricted"), Outcome.AVOID_SECTOR, "avoid the sector to the {compass}"),
    ("tmz from the surface", dict(type_="tmz"), Outcome.AVOID_SECTOR, "avoid the sector to the {compass}"),
    ("atz from the surface", dict(type_="atz"), Outcome.AVOID_SECTOR, "avoid the sector to the {compass}"),
    ("matz from the surface", dict(type_="matz"), Outcome.AVOID_SECTOR, "avoid the sector to the {compass}"),
    ("rmz from the surface", dict(type_="rmz"), Outcome.AVOID_SECTOR, "avoid the sector to the {compass}"),
    ("cta from the surface", dict(type_="cta", cls="D"), Outcome.AVOID_SECTOR, "avoid the sector to the {compass}"),
    ("ctr with a low floor", dict(type_="ctr", lower=amsl(2000)), Outcome.CEILING, "stay below 1,800 ft QNH"),
    ("cta with a low floor", dict(type_="cta", cls="D", lower=amsl(1500)), Outcome.CEILING, "stay below 1,300 ft QNH"),
    ("atz with a mid floor", dict(type_="atz", lower=amsl(7000)), Outcome.WARNING, "{name}"),
]


def angle_between(a: float, b: float) -> float:
    """The smaller angle between two bearings, so 359.9 and 0.1 are 0.2 apart."""
    return abs((a - b + 180.0) % 360.0 - 180.0)


def beside(bearing: float, nm: float, **volume_args):
    lat, lon = offset(*LZ, bearing, nm)
    shape = square(lat, lon, TINY)
    volume_args.setdefault("name", "NEARBY")
    return volume(shape=shape, **volume_args)


@pytest.mark.parametrize(("label", "args", "outcome", "condition"), VOLUMES, ids=[v[0] for v in VOLUMES])
@pytest.mark.parametrize(("bearing", "word"), CENTRES, ids=[c[1] for c in CENTRES])
def test_a_volume_1_nm_away_is_labelled_with_the_sector_it_is_in(label, args, outcome, condition, bearing, word):
    finding = only(assess([beside(bearing, 1.0, **args)]))
    assert finding.outcome is outcome
    assert finding.compass == compass16(bearing) == word
    assert finding.bearing_deg is not None and angle_between(finding.bearing_deg, bearing) <= 0.5
    assert finding.condition is not None and condition.format(compass=word, name="NEARBY") in finding.condition


@pytest.mark.parametrize(("label", "args", "outcome", "condition"), VOLUMES, ids=[v[0] for v in VOLUMES])
@pytest.mark.parametrize(("bearing", "word"), CENTRES[::5], ids=[c[1] for c in CENTRES[::5]])
def test_the_same_volume_2_5_nm_away_gives_nothing(label, args, outcome, condition, bearing, word):
    result = assess([beside(bearing, 2.5, **args)])
    assert result.findings == [] and result.ceiling_qnh_ft is None


def test_the_bearing_runs_from_the_lz_to_the_nearest_boundary_point_not_to_the_centre():
    # A wide box to the east whose nearest point is the middle of its west side, due east of the LZ,
    # even though its centre is a long way north-east.
    lat, lon = offset(*LZ, 90.0, 1.0)
    wide = [[[(lat - 0.001, lon), (lat - 0.001, lon + 0.1), (lat + 0.2, lon + 0.1), (lat + 0.2, lon), (lat - 0.001, lon)]]]
    finding = only(assess([volume("ctr", shape=wide)]))
    assert finding.compass == "E"
    assert finding.bearing_deg is not None and angle_between(finding.bearing_deg, 90.0) <= 0.5
    assert finding.bearing_deg == pytest.approx(shape_edge_bearing_deg(*LZ, wide))


def test_the_distance_in_the_reason_is_to_the_nearest_boundary_point():
    finding = only(assess([beside(0.0, 1.5, type_="ctr")]))
    assert "1.50 NM to the N" in finding.reason


def test_a_floor_above_the_low_band_is_never_an_avoid_and_keeps_its_bearing():
    mid = only(assess([beside(135.0, 1.0, type_="ctr", lower=amsl(7000))]))
    high = only(assess([beside(135.0, 1.0, type_="ctr", lower=amsl(12000))]))
    gliding = only(assess([beside(135.0, 1.0, type_="gliding")]))
    assert (mid.outcome, high.outcome, gliding.outcome) == (Outcome.WARNING, Outcome.AWARENESS, Outcome.AWARENESS)
    for finding in (mid, high, gliding):
        assert finding.compass == "SE" and finding.bearing_deg is not None


def test_a_class_g_cta_beside_the_lz_is_awareness_not_a_ceiling():
    assert only(assess([beside(0.0, 1.0, type_="cta", cls="G", lower=amsl(1500))])).outcome is Outcome.AWARENESS


def test_a_ceiling_beside_the_lz_does_not_make_the_lz_no_go():
    # A low floor beside the LZ is a condition for the sector; only a ceiling over the LZ can be NO_GO (E15).
    finding = only(assess([beside(0.0, 1.0, type_="ctr", lower=amsl(300))]))
    assert finding.outcome is Outcome.CEILING and finding.condition == "stay below 100 ft QNH"


def test_a_volume_exactly_at_the_local_area_is_within_and_a_hair_beyond_is_not():
    area = beside(45.0, 2.0, type_="ctr")
    edge = shape_edge_distance_nm(*LZ, area.shape)
    exact = with_rules(local_area=edge)
    assert only(assess([area], rules=exact)).outcome is Outcome.AVOID_SECTOR
    assert assess([area], rules=with_rules(local_area=math.nextafter(edge, 0.0))).findings == []
    assert only(assess([area], rules=with_rules(local_area=3))).outcome is Outcome.AVOID_SECTOR


def test_the_shipped_local_area_decides_between_1_9_and_2_1_nm():
    assert only(assess([beside(0.0, 1.9, type_="ctr")])).outcome is Outcome.AVOID_SECTOR
    assert assess([beside(0.0, 2.1, type_="ctr")]).findings == []


# --- E22: danger areas beside the LZ -------------------------------------------------------


def danger_beside(**kwargs):
    return beside(200.0, 1.0, type_="danger", name="EGD323A", designator="EGD323A", **kwargs)


def test_a_by_notam_danger_area_beside_the_lz_is_avoid_sector_while_activated_and_ignore_otherwise():
    area = danger_beside(by_notam=True)
    active = only(assess([area], activations=[activation("EGD323A")]))
    assert active.outcome is Outcome.AVOID_SECTOR and active.compass == "SSW" and active.bearing_deg is not None
    assert "avoid the sector to the SSW" in (active.condition or "")
    inactive = only(assess([area]))
    assert inactive.outcome is Outcome.IGNORE and "not active" in inactive.reason and inactive.bearing_deg is None
    elsewhere = assess([area], activations=[activation("EGD999A", circle=Circle(lat=LZ[0] + 1.0, lon=LZ[1], radius_nm=1.0))])
    assert [(f.area, f.outcome) for f in elsewhere.findings] == [("EGD323A", Outcome.IGNORE)]


def test_an_activation_that_has_ended_does_not_make_it_active():
    area = danger_beside(by_notam=True)
    ended = activation("EGD323A", start=utc(5), end=utc(9))
    assert only(assess([area], activations=[ended])).outcome is Outcome.IGNORE
    starts_late = activation("EGD323A", start=utc(10), end=utc(12))
    assert only(assess([area], activations=[starts_late])).outcome is Outcome.IGNORE


def test_a_danger_area_with_published_hours_is_avoid_sector_inside_them_and_ignore_outside():
    inside = only(assess([danger_beside(hours="DAILY 0800-1800")]))
    assert inside.outcome is Outcome.AVOID_SECTOR and inside.compass == "SSW"
    outside = only(assess([danger_beside(hours="DAILY 1000-1100")]))
    assert outside.outcome is Outcome.IGNORE and "outside published hours" in outside.reason


def test_a_danger_area_with_published_hours_is_avoid_sector_outside_them_when_a_notam_activates_it():
    area = danger_beside(hours="DAILY 1000-1100")
    finding = only(assess([area], activations=[activation("EGD323A")]))
    assert finding.outcome is Outcome.AVOID_SECTOR


def test_an_h24_danger_area_beside_the_lz_is_always_avoid_sector():
    for hours in ("H24", "DAILY 0000-2400"):
        finding = only(assess([danger_beside(hours=hours)]))
        assert finding.outcome is Outcome.AVOID_SECTOR and finding.compass == "SSW"


def test_an_h24_danger_area_has_no_inactive_segment():
    night = TimeSpan(start=utc(2), end=utc(3))
    assert only(assess([danger_beside(hours="H24")], segment=night)).outcome is Outcome.AVOID_SECTOR


@pytest.mark.parametrize("area_args", [dict(hours=None), dict(hours="WHENEVER WE FEEL LIKE IT"), dict(hours="oops", hours_readable=False)])
def test_a_danger_area_with_empty_or_unreadable_hours_beside_the_lz_is_active_and_noted(area_args):
    result = assess([danger_beside(**area_args)])
    assert only(result).outcome is Outcome.AVOID_SECTOR
    assert any("treated as active" in n or "fail closed" in n or "could not read" in n for n in result.notes)


def test_a_danger_area_beside_the_lz_is_avoid_sector_whatever_the_reading_says():
    area = danger_beside(by_notam=True)
    for read in (None, reading(), reading(contact=ContactKind.RADIO_ONLY)):
        assert only(assess([area], activations=[activation("EGD323A", read=read)])).outcome is Outcome.AVOID_SECTOR


def test_a_danger_area_with_a_high_floor_beside_the_lz_follows_the_bands():
    area = beside(200.0, 1.0, type_="danger", name="EGD323A", designator="EGD323A", hours="H24", lower=amsl(2000))
    finding = only(assess([area]))
    assert finding.outcome is Outcome.CEILING and finding.condition == "stay below 1,800 ft QNH" and finding.compass == "SSW"


def test_a_danger_area_2_5_nm_away_gives_nothing_even_when_active():
    area = beside(200.0, 2.5, type_="danger", name="EGD323A", designator="EGD323A", hours="H24")
    assert assess([area]).findings == []


def test_a_segment_is_all_that_matters_not_the_date():
    next_day = TimeSpan(start=utc(9, day=1, month=11), end=utc(10, day=1, month=11))
    area = danger_beside(by_notam=True)
    assert only(assess([area], segment=next_day, activations=[activation("EGD323A")])).outcome is Outcome.IGNORE
