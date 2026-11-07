"""G14: notam_action() against the Spec section 2 table, first match wins.

Every test calls the real function with hand-written readings and parsed NOTAMs placed round
the Blue Mill LZ (site elevation 30 ft, 1013.25 hPa, segment 09:00-10:00Z on 31 Oct 2026).
Geometry, schedules and rules are the real ones; nothing is mocked.
"""

import copy
import itertools
import math
from datetime import timedelta

import pytest
from factories import (
    LZ,
    SFC,
    STANDARD_HPA,
    agl,
    amsl,
    circle_at,
    exemption,
    fl,
    notam,
    reading,
    segment,
    square_at,
    utc,
    with_rules,
)

from flysafelz import notam_action as module
from flysafelz.contracts import (
    AppliesTo,
    Category,
    Circle,
    ContactKind,
    ExemptionCovers,
    NotamAction,
    Polygon,
    ProhibitiveReason,
    Verdict,
)
from flysafelz.contracts import Segment as ContractSegment
from flysafelz.daylight import round_down_minute, round_up_minute, sunrise_sunset
from flysafelz.geo import circle_edge_bearing_deg, circle_edge_distance_nm, compass16
from flysafelz.notam_action import ActionResult, active_intervals, notam_action

A = NotamAction

E_TEXT = "AIRSPACE RESTRICTED. PARAMOTORS EXEMPT. FLY ONLY WITH ATC PERMISSION."
QUOTED = "Paramotors exempt."  # in the E text, in other case
UNQUOTED = "Hang gliders are exempt from this notice."  # not in the E text
CONDITION = "fly only with   ATC permission."  # in the E text, with odd spacing

BANDS = {
    "sfc": (SFC, "SFC"),
    "0-5000": (amsl(2000), "2000FT AMSL"),
    "5000-10000": (amsl(7000), "7000FT AMSL"),
    ">10000": (amsl(12000), "12000FT AMSL"),
}
EDGE_NM = {"over": 0.0, "within": 1.0, "beyond": 3.0}
ACTIVE_D = {"active": "0800-1800", "inactive": "1200-1800"}  # the segment is 09:00-10:00Z
CONTACTS = {
    "phone": (ContactKind.PHONE, "07784 546062"),
    "email": (ContactKind.EMAIL, "ops@example.org"),
    "radio-only": (ContactKind.RADIO_ONLY, "123.450"),
    "none": (ContactKind.NONE, None),
}
EXEMPTIONS = {
    "no-exemption": [],
    "paramotor-quoted": [exemption(QUOTED)],
    "paramotor-unquoted": [exemption(UNQUOTED)],
    "conditional": [exemption(CONDITION, ExemptionCovers.CONDITIONAL)],
}


def act(parsed, read=None, seg=None, **kw) -> ActionResult:
    return notam_action(parsed, read, LZ, seg or segment(), **kw)


def build(*, edge, band, d, read, category=Category.COOPERATIVE, contact="phone", applies=AppliesTo.ALL_AIRCRAFT, exemptions="no-exemption"):
    """A parsed NOTAM and (if read) its reading, with the same circle, floor and schedule on both."""
    limit, text = BANDS[band]
    circle = circle_at(EDGE_NM[edge])
    parsed = notam(circle=circle, item_d=ACTIVE_D[d], item_f=text, item_e=E_TEXT)
    if not read:
        return parsed, None
    kind, value = CONTACTS[contact]
    return parsed, reading(
        category=category,
        geometry=circle,
        floor=limit,
        ceiling=amsl(20000),
        schedule=ACTIVE_D[d],
        contact=kind,
        value=value,
        applies_to=applies,
        exemptions=EXEMPTIONS[exemptions],
    )


def expected(*, edge, band, d, read, category=Category.COOPERATIVE, contact="phone", applies=AppliesTo.ALL_AIRCRAFT, exemptions="no-exemption") -> NotamAction:
    """The table, written out rule by rule (first match wins). The oracle for the matrix."""
    if read and (applies in (AppliesTo.UAS_ONLY, AppliesTo.AERODROME_USERS) or exemptions == "paramotor-quoted"):
        return A.IGNORE  # row 1: does not apply to a paramotor
    if d == "inactive" or edge == "beyond":
        return A.IGNORE  # row 1: not active, or beyond the local area
    over = edge == "over"
    if over and band == "sfc":
        if not read:
            return A.UNREAD_OVER_LZ  # Q8
        if category is Category.PROHIBITIVE:
            return A.NO_FLY  # row 2
        if contact in ("phone", "email") and (category is Category.COOPERATIVE or exemptions == "conditional"):
            return A.CONTACT_FIRST  # row 3
        if category is Category.COOPERATIVE:
            return A.NO_FLY  # E12: cooperative, but nobody to ask
        return A.AWARENESS  # Q13: informational is a warning
    if not over or band in ("sfc", "0-5000"):
        return A.AVOID_SECTOR  # row 4
    return A.AWARENESS if band == "5000-10000" else A.AWARENESS_MAP  # rows 5 and 6


def case(**kw):
    ident = "-".join(
        [
            "read" if kw["read"] else "unread",
            kw["edge"],
            kw["band"],
            kw["d"],
            *([kw["category"].value, kw["contact"]] if kw["read"] else []),
            *([kw["applies"].value.lower(), kw["exemptions"]] if kw["read"] else []),
        ]
    )
    return pytest.param(kw, id=ident)


def matrix():
    for edge, band, d in itertools.product(EDGE_NM, BANDS, ACTIVE_D):
        yield case(edge=edge, band=band, d=d, read=False)
        for category, contact in itertools.product(Category, CONTACTS):
            yield case(edge=edge, band=band, d=d, read=True, category=category, contact=contact, applies=AppliesTo.ALL_AIRCRAFT, exemptions="no-exemption")
        for applies, exemptions in itertools.product(AppliesTo, EXEMPTIONS):
            yield case(edge=edge, band=band, d=d, read=True, category=Category.COOPERATIVE, contact="phone", applies=applies, exemptions=exemptions)
        for exemptions in EXEMPTIONS:
            yield case(edge=edge, band=band, d=d, read=True, category=Category.PROHIBITIVE, contact="phone", applies=AppliesTo.ALL_AIRCRAFT, exemptions=exemptions)


# Some cases are reached by two of the loops above; keyed by id they are run once.
CASES = list({p.id: p for p in matrix()}.values())


@pytest.mark.parametrize("kw", [p.values[0] for p in CASES], ids=[p.id for p in CASES])
def test_the_table_first_match_wins(kw):
    parsed, read = build(**kw)
    assert act(parsed, read).action is expected(**kw)


# --- reasons ------------------------------------------------------------------------------


def test_sector_reason_gives_distance_bearing_and_the_ceiling_as_published():
    circle = circle_at(0.40, bearing=90)
    parsed = notam(circle=circle_at(0, radius_nm=1.0), item_f="SFC", item_g="700FT AGL")
    result = act(parsed, reading(geometry=circle, ceiling=agl(700)))
    edge = circle_edge_distance_nm(LZ.lat, LZ.lon, circle.lat, circle.lon, circle.radius_nm)
    bearing = circle_edge_bearing_deg(LZ.lat, LZ.lon, circle.lat, circle.lon, circle.radius_nm)
    assert result.action is A.AVOID_SECTOR
    assert result.reason == f"drone ops {edge:.2f} NM to the {compass16(bearing)} (circle edge), up to 700 ft AGL: avoid that sector"
    assert result.bearing_deg == pytest.approx(bearing)
    assert compass16(bearing) == "E"
    assert "\n" not in result.reason


def test_a_polygon_sector_says_polygon_edge_and_a_sector_above_the_ground_says_where_it_starts():
    parsed = notam(circle=circle_at(0, radius_nm=1.0), item_f="1500FT AMSL")
    result = act(parsed, reading(geometry=square_at(1.0, bearing=0), floor=amsl(1500), ceiling=amsl(3000)))
    assert result.action is A.AVOID_SECTOR
    assert "NM to the N (polygon edge), from 1,500 ft AMSL, up to 3,000 ft AMSL: avoid that sector" in result.reason
    assert result.bearing_deg == pytest.approx(0, abs=0.5)


def test_an_unread_sector_names_the_notam_and_says_q_circle():
    result = act(notam(circle=circle_at(1.0, bearing=225), item_f="SFC", item_g="700FT AGL"))
    assert result.action is A.AVOID_SECTOR
    assert result.reason.startswith("NOTAM H7171/26 1.00 NM to the SW (Q-circle edge), up to 700 ft AGL")
    assert result.bearing_deg == pytest.approx(225, abs=1)


def test_a_ceiling_over_the_lz_says_stay_below_the_floor_less_the_buffer():
    parsed = notam(circle=circle_at(0), item_f="2000FT AMSL")
    result = act(parsed, reading(geometry=circle_at(0), floor=amsl(2000)))
    assert result.action is A.AVOID_SECTOR
    assert "stay below 1,800 ft QNH" in result.reason
    assert result.bearing_deg is None


def test_unread_over_the_lz_prints_the_whole_e_text_on_one_line():
    e_text = "TEMPORARY DANGER AREA EGD496A ACTIVATED.\nUAS BVLOS OPS.\nSFC-1300FT AMSL."
    result = act(notam("J3857/26", circle=circle_at(0, radius_nm=10), item_e=e_text, item_f="SFC", item_g="1300FT AMSL"))
    assert result.action is A.UNREAD_OVER_LZ
    assert "TEMPORARY DANGER AREA EGD496A ACTIVATED. UAS BVLOS OPS. SFC-1300FT AMSL." in result.reason
    assert "\n" not in result.reason


def test_a_reading_that_activates_a_designator_gets_its_table_action_and_names_it():
    circle = circle_at(0, radius_nm=10)
    result = act(notam("J3857/26", circle=circle, item_f="SFC"), reading(notam_id="J3857/26", activity="UAS BVLOS ops", designators=["D496A"]))
    assert result.action is A.CONTACT_FIRST
    assert result.reason.endswith("activates EGD496A")
    assert "07784 546062" in result.reason


def test_a_designator_written_loosely_is_normalised_and_a_non_designator_is_left_out():
    result = act(notam(circle=circle_at(0), item_f="SFC"), reading(designators=["EGD 323A", "not one"]))
    assert result.reason.endswith("activates EGD323A")


# --- E10: limits ---------------------------------------------------------------------------


def test_reading_and_fg_that_differ_take_the_lower_floor_and_the_higher_ceiling_and_say_so():
    parsed = notam(circle=circle_at(0), item_f="1500FT AMSL", item_g="4000FT AMSL")
    over = act(parsed, reading(geometry=circle_at(0), floor=amsl(2000), ceiling=amsl(3000)))
    assert over.action is A.AVOID_SECTOR
    assert "stay below 1,300 ft QNH" in over.reason  # the F floor, 1,500, not the reading's 2,000
    assert "disagree on the floor (2,000 ft AMSL against 1,500 ft AMSL)" in over.reason
    parsed = notam(circle=circle_at(1.0), item_f="SFC", item_g="4000FT AMSL")
    within = act(parsed, reading(geometry=circle_at(1.0), ceiling=amsl(3000)))
    assert "up to 4,000 ft AMSL" in within.reason
    assert "disagree on the ceiling (3,000 ft AMSL against 4,000 ft AMSL)" in within.reason


def test_a_lower_reading_ceiling_loses_to_a_higher_fg_ceiling_and_a_lower_fg_floor_loses_to_a_lower_reading_floor():
    parsed = notam(circle=circle_at(1.0), item_f="2000FT AMSL", item_g="500FT AMSL")
    result = act(parsed, reading(geometry=circle_at(1.0), floor=amsl(1000), ceiling=amsl(3000)))
    assert "from 1,000 ft AMSL, up to 3,000 ft AMSL" in result.reason


def test_reading_and_fg_that_agree_say_nothing_about_a_conflict():
    parsed = notam(circle=circle_at(1.0), item_f="SFC", item_g="700FT AGL")
    # 700 ft AGL at 30 ft elevation is 730 ft QNH; the same height written as AMSL agrees.
    result = act(parsed, reading(geometry=circle_at(1.0), ceiling=amsl(730)))
    assert "disagree" not in result.reason


def test_without_a_reading_fg_is_used_before_the_q_line():
    parsed = notam(circle=circle_at(0), item_f="2000FT AMSL", item_g="3000FT AMSL", lower_fl=5, upper_fl=20)
    result = act(parsed)
    assert result.action is A.AVOID_SECTOR
    assert "stay below 1,800 ft QNH" in result.reason  # F, not the Q-line's FL005


def test_without_a_reading_or_fg_the_q_line_gives_both_limits():
    parsed = notam(circle=circle_at(0), item_f=None, item_g=None, lower_fl=20, upper_fl=40)
    assert "stay below 1,800 ft QNH" in act(parsed).reason  # FL020 at the reference pressure
    within = act(notam(circle=circle_at(1.0), item_f=None, item_g=None, lower_fl=0, upper_fl=40))
    assert within.reason.endswith("up to FL040: avoid that sector")  # lower FL000 is the surface
    assert "from" not in within.reason


def test_a_q_line_upper_limit_of_fl999_has_no_upper_limit():
    parsed = notam(circle=circle_at(1.0), item_f=None, item_g=None, lower_fl=0, upper_fl=999)
    assert ", no upper limit: avoid that sector" in act(parsed).reason


def test_unreadable_fg_falls_back_to_the_q_line_and_unlimited_is_understood():
    parsed = notam(circle=circle_at(1.0), item_f="SEE E", item_g="UNL", lower_fl=0, upper_fl=20)
    assert ", no upper limit" in act(parsed).reason


def test_an_e_text_that_contradicts_fg_is_never_read_for_limits():
    parsed = notam(circle=circle_at(1.0), item_f="SFC", item_g="700FT AGL", item_e="UAS OPR WI 0.5NM. MAX HGT 100FT AGL. FROM 2000FT AMSL.")
    result = act(parsed)
    assert result.action is A.AVOID_SECTOR
    assert "up to 700 ft AGL" in result.reason
    assert "100" not in result.reason


# --- E10 with a reading that states no height (ticket 3: floor and ceiling may be None) ----------------------------------

READING_FLOOR, FG_FLOOR, Q_FLOOR = "1,000 ft AMSL", "1,500 ft AMSL", "FL020"
READING_CEILING, FG_CEILING, Q_CEILING = "5,000 ft AMSL", "4,000 ft AMSL", "FL040"


@pytest.mark.parametrize("fg_ceiling", [True, False], ids=["G present", "G absent"])
@pytest.mark.parametrize("fg_floor", [True, False], ids=["F present", "F absent"])
@pytest.mark.parametrize("read_ceiling", [True, False], ids=["reading ceiling", "ceiling None"])
@pytest.mark.parametrize("read_floor", [True, False], ids=["reading floor", "floor None"])
def test_a_none_limit_goes_on_to_fg_and_then_the_q_line_and_a_disagree_note_needs_both(read_floor, read_ceiling, fg_floor, fg_ceiling):
    parsed = notam(
        circle=circle_at(1.0),
        item_f="1500FT AMSL" if fg_floor else None,
        item_g="4000FT AMSL" if fg_ceiling else None,
        lower_fl=20,
        upper_fl=40,
    )
    read = reading(geometry=circle_at(1.0), floor=amsl(1000), ceiling=amsl(5000)).model_copy(
        update={"floor": amsl(1000) if read_floor else None, "ceiling": amsl(5000) if read_ceiling else None}
    )
    result = act(parsed, read)
    assert result.action is A.AVOID_SECTOR
    floor = READING_FLOOR if read_floor else FG_FLOOR if fg_floor else Q_FLOOR  # the reading's, else F, else the Q-line
    ceiling = READING_CEILING if read_ceiling else FG_CEILING if fg_ceiling else Q_CEILING
    assert f"from {floor}, up to {ceiling}: avoid that sector" in result.reason
    assert ("disagree on the floor" in result.reason) == (read_floor and fg_floor)  # the reading's 1,000 is below F's 1,500: both given and different
    assert ("disagree on the ceiling" in result.reason) == (read_ceiling and fg_ceiling)


def test_a_none_floor_with_no_item_f_and_a_surface_q_line_is_the_surface():
    parsed = notam(circle=circle_at(0), item_f=None, item_g=None, lower_fl=0, upper_fl=40)
    read = reading(geometry=circle_at(0)).model_copy(update={"floor": None, "ceiling": None})
    result = act(parsed, read)
    assert result.action is A.CONTACT_FIRST and "over the LZ from the surface" in result.reason


def test_a_none_floor_with_a_raised_q_line_floor_is_not_the_surface():
    parsed = notam(circle=circle_at(0), item_f=None, item_g=None, lower_fl=20, upper_fl=40)
    read = reading(geometry=circle_at(0)).model_copy(update={"floor": None, "ceiling": None})
    result = act(parsed, read)
    assert result.action is A.AVOID_SECTOR and "stay below 1,800 ft QNH" in result.reason


def test_a_none_ceiling_with_a_q_line_upper_limit_of_fl999_and_no_item_g_has_no_upper_limit():
    parsed = notam(circle=circle_at(1.0), item_f=None, item_g=None, lower_fl=0, upper_fl=999)
    read = reading(geometry=circle_at(1.0)).model_copy(update={"floor": None, "ceiling": None})
    assert ", no upper limit: avoid that sector" in act(parsed, read).reason


# --- AGL and FL floors ---------------------------------------------------------------------


@pytest.mark.parametrize("floor", [SFC, agl(0), amsl(30), amsl(0), fl(0)], ids=["sfc", "0-ft-agl", "at-elevation", "below-elevation", "fl000"])
def test_sfc_gnd_and_zero_agl_all_count_as_the_surface(floor):
    result = act(notam(circle=circle_at(0)), reading(geometry=circle_at(0), floor=floor, category=Category.PROHIBITIVE))
    assert result.action is A.NO_FLY


def test_gnd_in_item_f_is_the_surface():
    assert act(notam(circle=circle_at(0), item_f="GND")).action is A.UNREAD_OVER_LZ


def test_an_agl_floor_adds_the_site_elevation():
    result = act(notam(circle=circle_at(0)), reading(geometry=circle_at(0), floor=agl(970)))
    assert result.action is A.AVOID_SECTOR
    assert "stay below 800 ft QNH" in result.reason  # 970 + 30 - 200
    assert "from 970 ft AGL (1,000 ft QNH)" in result.reason


def test_an_agl_floor_follows_the_lzs_own_elevation():
    high = LZ.model_copy(update={"elevation_ft": 530.0})
    result = notam_action(notam(circle=circle_at(0)), reading(geometry=circle_at(0), floor=agl(970)), high, segment())
    assert "stay below 1,300 ft QNH" in result.reason  # 970 + 530 - 200


def test_an_fl_floor_on_a_low_pressure_segment_is_a_lower_qnh_floor():
    parsed = notam(circle=circle_at(0))
    read = reading(geometry=circle_at(0), floor=fl(40))
    standard = act(parsed, read, segment(pressure=STANDARD_HPA))
    low = act(parsed, read, segment(pressure=1003.25))  # ten hPa low: 270 ft lower
    assert "stay below 3,800 ft QNH" in standard.reason
    assert "stay below 3,530 ft QNH" in low.reason


def test_a_low_pressure_flight_level_can_change_the_band():
    parsed = notam(circle=circle_at(0))
    read = reading(geometry=circle_at(0), floor=fl(52))
    assert act(parsed, read, segment(pressure=STANDARD_HPA)).action is A.AWARENESS  # 5,200 ft
    low = act(parsed, read, segment(pressure=980.0))  # 4,302 ft
    assert low.action is A.AVOID_SECTOR
    assert "stay below 4,102 ft QNH" in low.reason


def test_a_segment_without_pressure_takes_flight_levels_at_the_reference_pressure():
    parsed = notam(circle=circle_at(0))
    read = reading(geometry=circle_at(0), floor=fl(40))
    plain = ContractSegment(start=utc(9), end=utc(10), verdict=Verdict.PILOT_DISCRETION)
    assert act(parsed, read, plain).reason == act(parsed, read, segment(pressure=None)).reason
    assert "stay below 3,800 ft QNH" in act(parsed, read, plain).reason


# --- the edges: 2 NM, 5,000 ft, 10,000 ft -------------------------------------------------


def edge_distance(circle) -> float:
    return circle_edge_distance_nm(LZ.lat, LZ.lon, circle.lat, circle.lon, circle.radius_nm)


def test_a_distance_exactly_at_the_local_area_is_within_and_just_beyond_is_not():
    circle = circle_at(1.5, bearing=45)
    parsed = notam(circle=circle)
    read = reading(geometry=circle)
    distance = edge_distance(circle)
    assert act(parsed, read, rules=with_rules(local_area=distance)).action is A.AVOID_SECTOR  # exactly at the edge
    assert act(parsed, read, rules=with_rules(local_area=distance + 1e-6)).action is A.AVOID_SECTOR
    assert act(parsed, read, rules=with_rules(local_area=distance - 1e-6)).action is A.IGNORE
    assert "beyond the" in act(parsed, read, rules=with_rules(local_area=distance - 1e-6)).reason


def test_the_default_local_area_is_the_rules_value_one_value_each_side():
    local = float(module.RULES.local_area.value)  # type: ignore[arg-type]
    for bearing in (0, 90, 180, 270):
        near, far = circle_at(local - 0.05, bearing), circle_at(local + 0.05, bearing)
        assert act(notam(circle=near), reading(geometry=near)).action is A.AVOID_SECTOR
        assert act(notam(circle=far), reading(geometry=far)).action is A.IGNORE


@pytest.mark.parametrize(
    ("feet", "action"),
    [
        (4999, A.AVOID_SECTOR),
        (5000, A.AVOID_SECTOR),
        (5001, A.AWARENESS),
        (9999, A.AWARENESS),
        (10000, A.AWARENESS),
        (10001, A.AWARENESS_MAP),
    ],
)
def test_floors_either_side_of_5000_and_10000_ft_qnh(feet, action):
    result = act(notam(circle=circle_at(0)), reading(geometry=circle_at(0), floor=amsl(feet), ceiling=amsl(feet + 1000)))
    assert result.action is action


def test_the_band_edges_are_the_rules_values():
    parsed, read = notam(circle=circle_at(0)), reading(geometry=circle_at(0), floor=amsl(2000))
    assert act(parsed, read).action is A.AVOID_SECTOR
    assert act(parsed, read, rules=with_rules(band_low=1500)).action is A.AWARENESS
    assert act(parsed, read, rules=with_rules(band_low=1500, band_high=1900)).action is A.AWARENESS_MAP
    assert "stay below 1,700 ft QNH" in act(parsed, read, rules=with_rules(ceiling_buffer=300)).reason


def test_patching_the_loaded_rules_changes_the_outcome(monkeypatch):
    circle = circle_at(1.5)
    parsed, read = notam(circle=circle), reading(geometry=circle)
    assert act(parsed, read).action is A.AVOID_SECTOR
    monkeypatch.setattr(module, "RULES", with_rules(local_area=1))
    assert act(parsed, read).action is A.IGNORE


# --- geometry ------------------------------------------------------------------------------


@pytest.mark.parametrize("shape", [square_at, circle_at], ids=["polygon", "circle"])
def test_e6_the_q_circle_covers_the_lz_but_the_readings_shape_misses_it(shape):
    q_circle = circle_at(0, radius_nm=1.0)
    parsed = notam(circle=q_circle)
    assert act(parsed).action is A.UNREAD_OVER_LZ  # unread, the Q-circle is what there is
    near = act(parsed, reading(geometry=shape(1.0), category=Category.PROHIBITIVE))
    far = act(parsed, reading(geometry=shape(3.0), category=Category.PROHIBITIVE))
    assert near.action is A.AVOID_SECTOR
    assert far.action is A.IGNORE
    inside = act(parsed, reading(geometry=shape(0), category=Category.PROHIBITIVE))
    assert inside.action is A.NO_FLY


def test_reading_geometry_none_falls_back_to_the_q_circle():
    parsed = notam(circle=circle_at(0, radius_nm=1.0), item_f="SFC")
    assert act(parsed, reading(geometry=None, category=Category.PROHIBITIVE)).action is A.NO_FLY
    far = notam(circle=circle_at(1.0))
    assert act(far, reading(geometry=None)).action is A.AVOID_SECTOR
    assert "(Q-circle edge)" in act(far, reading(geometry=None)).reason


def test_a_polygon_that_cannot_be_used_falls_back_to_the_q_circle_and_logs_it():
    broken = Polygon(points=[(53.8, -0.08), (53.81, -0.08)])
    result = act(notam(circle=circle_at(1.0)), reading(geometry=broken))
    assert result.action is A.AVOID_SECTOR
    assert "(Q-circle edge)" in result.reason
    assert "reading geometry not usable; the Q-circle is used" in result.debug


def test_with_no_usable_geometry_the_notam_is_treated_as_over_the_lz_and_the_reason_says_so():
    parsed = notam(radius_nm=0.0, item_f="SFC")
    unread = act(parsed)
    assert unread.action is A.UNREAD_OVER_LZ
    assert "no usable geometry, so treated as over the LZ" in unread.reason
    read = act(parsed, reading(geometry=Polygon(points=[]), category=Category.PROHIBITIVE))
    assert read.action is A.NO_FLY
    assert "no usable geometry, so treated as over the LZ" in read.reason
    not_a_number = notam(centre_lat=math.nan, item_f="SFC")
    assert act(not_a_number).action is A.UNREAD_OVER_LZ


def test_an_lz_exactly_on_a_polygon_boundary_is_over_it():
    # The bottom side of this square runs through the LZ.
    ring = [(LZ.lat, LZ.lon - 0.01), (LZ.lat, LZ.lon + 0.01), (LZ.lat + 0.01, LZ.lon + 0.01), (LZ.lat + 0.01, LZ.lon - 0.01)]
    result = act(notam(circle=circle_at(3.0)), reading(geometry=Polygon(points=ring), category=Category.PROHIBITIVE))
    assert result.action is A.NO_FLY


def test_an_lz_exactly_on_a_circle_edge_is_over_it():
    circle = Circle(lat=LZ.lat, lon=LZ.lon, radius_nm=0.5)
    on_edge = circle_at(0, radius_nm=0.5).model_copy(update={"lat": LZ.lat})
    assert act(notam(circle=circle), reading(geometry=on_edge, category=Category.PROHIBITIVE)).action is A.NO_FLY


# --- E7: schedule --------------------------------------------------------------------------


def test_e7_inactive_for_the_whole_segment_is_ignore_for_that_segment_only():
    parsed = notam(circle=circle_at(0), item_d="1200-1800", item_f="SFC")
    read = reading(geometry=circle_at(0), schedule="1200-1800", category=Category.PROHIBITIVE)
    early = act(parsed, read, segment(utc(9), utc(10)))
    later = act(parsed, read, segment(utc(12), utc(13)))
    assert early.action is A.IGNORE
    assert "not active in this segment" in early.reason
    assert later.action is A.NO_FLY
    assert act(parsed, None, segment(utc(9), utc(10))).action is A.IGNORE
    assert act(parsed, None, segment(utc(12), utc(13))).action is A.UNREAD_OVER_LZ


def test_e7_a_segment_the_schedule_only_partly_covers_is_active():
    parsed = notam(circle=circle_at(0), item_d="0930-1800", item_f="SFC")
    assert act(parsed).action is A.UNREAD_OVER_LZ  # 09:30 starts inside 09:00-10:00
    ends = notam(circle=circle_at(0), item_d="0700-0930", item_f="SFC")
    assert act(ends).action is A.UNREAD_OVER_LZ


def test_e7_the_notams_own_start_and_end_bound_the_schedule():
    before = notam(circle=circle_at(0), item_f="SFC", valid_from=utc(10), valid_to=utc(18))
    assert act(before).action is A.IGNORE  # starts when the segment ends
    after = notam(circle=circle_at(0), item_f="SFC", valid_from=utc(1), valid_to=utc(9))
    assert act(after).action is A.IGNORE  # ended when the segment starts
    permanent = notam(circle=circle_at(0), item_f="SFC", valid_to=None, permanent=True)
    assert act(permanent).action is A.UNREAD_OVER_LZ


def test_e7_an_unparseable_schedule_is_active_for_all_of_b_to_c_and_says_so():
    parsed = notam(circle=circle_at(0), item_d="BY ARRANGEMENT WITH THE ESTATE", item_f="SFC")
    unread = act(parsed)
    assert unread.action is A.UNREAD_OVER_LZ
    assert unread.reason.endswith("schedule not parsed")
    assert any("BY ARRANGEMENT" in line for line in unread.debug)
    assert act(parsed, reading(geometry=circle_at(0), category=Category.PROHIBITIVE, schedule="")).action is A.NO_FLY


def test_the_reading_schedule_wins_over_the_raw_d_item_when_it_has_one():
    parsed = notam(circle=circle_at(0), item_d="BY ARRANGEMENT", item_f="SFC")
    read = reading(geometry=circle_at(0), schedule="1200-1800", category=Category.PROHIBITIVE)
    assert act(parsed, read).action is A.IGNORE


def test_sunrise_to_sunset_is_resolved_at_the_lz():
    parsed = notam(circle=circle_at(0), item_d="SR-SS", item_f="SFC")
    assert act(parsed, seg=segment(utc(4), utc(5))).action is A.IGNORE  # dark
    assert act(parsed, seg=segment(utc(10), utc(11))).action is A.UNREAD_OVER_LZ
    assert act(parsed, seg=segment(utc(18), utc(19))).action is A.IGNORE


def test_active_intervals_on_b_to_c_alone():
    parsed = notam(valid_from=utc(8), valid_to=utc(12), item_d=None)
    assert active_intervals(parsed, None, LZ, utc(7), utc(18)) == [(utc(8), utc(12))]
    assert active_intervals(parsed, None, LZ, utc(9), utc(10)) == [(utc(9), utc(10))]
    assert active_intervals(parsed, None, LZ, utc(13), utc(14)) == []
    permanent = notam(valid_from=utc(8), valid_to=None, permanent=True, item_d=None)
    assert active_intervals(permanent, None, LZ, utc(7), utc(18)) == [(utc(8), utc(18))]


def test_active_intervals_on_a_d_item_schedule():
    parsed = notam(item_d="1000-1100, 1400-1500")
    assert active_intervals(parsed, None, LZ, utc(9), utc(16)) == [(utc(10), utc(11)), (utc(14), utc(15))]
    assert active_intervals(parsed, None, LZ, utc(10, 30), utc(14, 30)) == [(utc(10, 30), utc(11)), (utc(14), utc(14, 30))]
    read = reading(schedule="1600-1700")
    assert active_intervals(parsed, read, LZ, utc(9), utc(18)) == [(utc(16), utc(17))]


def test_active_intervals_on_an_sr_ss_schedule_are_the_sun_times_at_the_lz():
    parsed = notam(item_d="SR-SS")
    rise, set_ = sunrise_sunset(LZ.lat, LZ.lon, utc(0).date())
    assert active_intervals(parsed, None, LZ, utc(0), utc(23)) == [(round_down_minute(rise), round_up_minute(set_))]
    offsets = notam(item_d="SR-30 TO SS+30")
    [(start, end)] = active_intervals(offsets, None, LZ, utc(0), utc(23))
    assert start == round_down_minute(rise - timedelta(minutes=30))
    assert end == round_up_minute(set_ + timedelta(minutes=30))


def test_active_intervals_on_an_interval_that_crosses_midnight_utc():
    parsed = notam(item_d="2300-0600")
    assert active_intervals(parsed, None, LZ, utc(22), utc(2, day=1, month=11)) == [(utc(23), utc(2, day=1, month=11))]
    assert active_intervals(parsed, None, LZ, utc(1, day=1, month=11), utc(8, day=1, month=11)) == [(utc(1, day=1, month=11), utc(6, day=1, month=11))]
    assert active_intervals(parsed, None, LZ, utc(7, day=1, month=11), utc(22, day=1, month=11)) == []
    whole = active_intervals(parsed, None, LZ, utc(12), utc(12, day=1, month=11))
    assert whole == [(utc(23), utc(6, day=1, month=11))]


def test_active_intervals_on_a_schedule_it_cannot_read_are_all_of_b_to_c():
    parsed = notam(item_d="BY ARRANGEMENT", valid_from=utc(8), valid_to=utc(12))
    assert active_intervals(parsed, None, LZ, utc(7), utc(18)) == [(utc(8), utc(12))]


# --- Q2: who the NOTAM applies to ---------------------------------------------------------


def test_applies_to_drones_only_is_ignored_whatever_the_geometry_floor_or_category():
    parsed = notam("J1812/26", circle=circle_at(0), item_e="RA(T) INSTALLED FOR UAS ONLY", item_f="SFC")
    read = reading(notam_id="J1812/26", geometry=circle_at(0), category=Category.PROHIBITIVE, applies_to=AppliesTo.UAS_ONLY, contact=ContactKind.NONE, value=None)
    result = act(parsed, read)
    assert result.action is A.IGNORE
    assert result.reason == "applies to drones only"


def test_applies_to_aerodrome_users_is_ignored_and_names_the_aerodrome():
    parsed = notam(circle=circle_at(0), item_a="EGSU", item_f="SFC")
    result = act(parsed, reading(geometry=circle_at(0), category=Category.PROHIBITIVE, applies_to=AppliesTo.AERODROME_USERS))
    assert result.action is A.IGNORE
    assert result.reason == "applies only to aircraft using EGSU"


def test_applies_to_named_classes_is_not_ignored_on_its_own():
    result = act(notam(circle=circle_at(0)), reading(geometry=circle_at(0), category=Category.PROHIBITIVE, applies_to=AppliesTo.NAMED_CLASSES))
    assert result.action is A.NO_FLY


def test_a_paramotor_exemption_whose_quote_is_in_the_e_text_ignores_the_notam_and_quotes_it():
    parsed = notam(circle=circle_at(0), item_e=E_TEXT, item_f="SFC")
    result = act(parsed, reading(geometry=circle_at(0), category=Category.PROHIBITIVE, exemptions=[exemption("  paramotors   EXEMPT. ")]))
    assert result.action is A.IGNORE
    assert 'exempts paramotors: "paramotors EXEMPT."' in result.reason


def test_a_paramotor_exemption_wins_over_geometry_floor_and_category():
    for edge, floor in itertools.product((0, 1.0, 3.0), (SFC, amsl(2000), amsl(7000), amsl(12000))):
        parsed = notam(circle=circle_at(edge), item_e=E_TEXT)
        read = reading(geometry=circle_at(edge), floor=floor, category=Category.PROHIBITIVE, exemptions=[exemption(QUOTED)])
        assert act(parsed, read).action is A.IGNORE


def test_a_paramotor_exemption_with_a_made_up_quote_is_dropped_and_logged():
    parsed = notam(circle=circle_at(0), item_e=E_TEXT, item_f="SFC")
    result = act(parsed, reading(geometry=circle_at(0), category=Category.PROHIBITIVE, exemptions=[exemption(UNQUOTED)]))
    assert result.action is A.NO_FLY  # as if there were no exemption
    assert f'exemption quote not found: "{UNQUOTED}"' in result.debug
    other = act(parsed, reading(geometry=circle_at(0), exemptions=[exemption(UNQUOTED), exemption(QUOTED)]))
    assert other.action is A.IGNORE  # a real quote beside a made-up one still counts
    assert f'exemption quote not found: "{UNQUOTED}"' in other.debug


def test_an_empty_quote_is_not_evidence():
    parsed = notam(circle=circle_at(0), item_e=E_TEXT, item_f="SFC")
    assert act(parsed, reading(geometry=circle_at(0), exemptions=[exemption("   ")])).action is A.CONTACT_FIRST


def test_an_exemption_for_some_other_class_changes_nothing():
    parsed = notam(circle=circle_at(0), item_e=E_TEXT, item_f="SFC")
    result = act(parsed, reading(geometry=circle_at(0), category=Category.PROHIBITIVE, exemptions=[exemption(QUOTED, ExemptionCovers.OTHER)]))
    assert result.action is A.NO_FLY
    assert result.debug == []


J1458_E = (
    "TEMPORARY RESTRICTED AREA WI 2.2NM RADIUS OF PSN 520528N 0000757E SFC-2000FT AMSL. "
    "APPLIES TO ALL ACFT INCLUDING MICROLIGHT AND PARAMOTOR. "
    "DOES NOT APPLY TO ANY ACFT FLYING IN ACCORDANCE WITH EXCEPTIONS IN THE AIC."
)
J1458_EXCEPTION = "DOES NOT APPLY TO ANY ACFT FLYING IN ACCORDANCE WITH EXCEPTIONS IN THE AIC."


def j1458(edge, *, conditional=False, contact=ContactKind.NONE, value=None):
    circle = circle_at(edge, radius_nm=2.2)
    parsed = notam("J1458/26", circle=circle, subject="RT", item_e=J1458_E, item_f="SFC", item_g="2000FT AMSL")
    read = reading(
        notam_id="J1458/26",
        activity="Duxford RA(T)",
        category=Category.PROHIBITIVE,
        geometry=circle,
        ceiling=amsl(2000),
        contact=contact,
        value=value,
        exemptions=[exemption(J1458_EXCEPTION, ExemptionCovers.CONDITIONAL)] if conditional else [],
    )
    return parsed, read


def test_q1_a_restricted_area_temporary_over_the_lz_is_no_fly_with_its_name():
    parsed, read = j1458(0)
    result = act(parsed, read)
    assert parsed.subject == "RT"
    assert result.action is A.NO_FLY
    assert "Duxford RA(T)" in result.reason


def test_q1_the_same_reading_one_nm_outside_the_circle_is_avoid_sector():
    parsed, read = j1458(1.0)
    result = act(parsed, read)
    assert result.action is A.AVOID_SECTOR
    assert f"Duxford RA(T) {edge_distance(circle_at(1.0, radius_nm=2.2)):.2f} NM to the S (circle edge)" in result.reason


def test_q1_with_no_reading_the_same_notam_over_the_lz_is_unverified_never_ignore():
    parsed, _ = j1458(0)
    result = act(parsed)
    assert result.action is A.UNREAD_OVER_LZ
    assert "APPLIES TO ALL ACFT INCLUDING MICROLIGHT AND PARAMOTOR" in result.reason


def test_a_conditional_exemption_never_drops_a_prohibitive_notam():
    parsed, read = j1458(0, conditional=True)
    result = act(parsed, read)
    assert result.action is A.NO_FLY
    assert f'condition: "{J1458_EXCEPTION}"' in result.reason


def test_a_conditional_exemption_with_a_phone_contact_never_makes_a_prohibitive_notam_contact_first():
    parsed, read = j1458(0, conditional=True, contact=ContactKind.PHONE, value="01223 000000")
    assert act(parsed, read).action is A.NO_FLY


def test_j1812_shape_drones_only_is_ignored_and_the_conditional_one_is_not():
    parsed = notam("J1812/26", circle=circle_at(0, radius_nm=2.2), subject="RT", item_e="RA(T) INSTALLED FOR UAS ONLY", item_f="SFC")
    read = reading(notam_id="J1812/26", activity="Duxford RA(T)", category=Category.PROHIBITIVE, geometry=circle_at(0, radius_nm=2.2), applies_to=AppliesTo.UAS_ONLY, contact=ContactKind.NONE, value=None)
    assert act(parsed, read).action is A.IGNORE
    assert act(parsed).action is A.UNREAD_OVER_LZ  # unread, nothing says it is for drones


def test_a_conditional_exemption_on_a_cooperative_notam_is_contact_first_with_the_condition_in_the_reason():
    parsed = notam(circle=circle_at(0), item_e=E_TEXT, item_f="SFC")
    with_phone = act(parsed, reading(geometry=circle_at(0), exemptions=[exemption(CONDITION, ExemptionCovers.CONDITIONAL)]))
    assert with_phone.action is A.CONTACT_FIRST
    assert 'condition: "fly only with   ATC permission."' in with_phone.reason
    no_contact = act(parsed, reading(geometry=circle_at(0), contact=ContactKind.NONE, value=None, exemptions=[exemption(CONDITION, ExemptionCovers.CONDITIONAL)]))
    assert no_contact.action is A.NO_FLY
    assert "condition:" in no_contact.reason


def test_a_conditional_exemption_on_an_informational_notam_asks_for_contact_when_there_is_one():
    parsed = notam(circle=circle_at(0), item_e=E_TEXT, item_f="SFC")
    exemptions = [exemption(CONDITION, ExemptionCovers.CONDITIONAL)]
    with_phone = act(parsed, reading(geometry=circle_at(0), category=Category.INFORMATIONAL, exemptions=exemptions))
    assert with_phone.action is A.CONTACT_FIRST
    without = act(parsed, reading(geometry=circle_at(0), category=Category.INFORMATIONAL, contact=ContactKind.NONE, value=None, exemptions=exemptions))
    assert without.action is A.AWARENESS


def test_a_conditional_exemption_does_not_change_the_other_rows():
    parsed = notam(circle=circle_at(1.0), item_e=E_TEXT)
    result = act(parsed, reading(geometry=circle_at(1.0), exemptions=[exemption(CONDITION, ExemptionCovers.CONDITIONAL)]))
    assert result.action is A.AVOID_SECTOR
    assert 'condition: "fly only with   ATC permission."' in result.reason


# --- E12, F11, Q13 -------------------------------------------------------------------------


def over_sfc(**kw) -> ActionResult:
    return act(notam(circle=circle_at(0), item_f="SFC"), reading(geometry=circle_at(0), **kw))


def test_e12_a_cooperative_reading_with_no_contact_is_no_fly_and_says_why():
    result = over_sfc(contact=ContactKind.NONE, value=None)
    assert result.action is A.NO_FLY
    assert "there is no phone or email contact" in result.reason


@pytest.mark.parametrize("kind", [ContactKind.PHONE, ContactKind.EMAIL])
def test_a_phone_or_email_with_no_value_is_not_a_contact(kind):
    assert over_sfc(contact=kind, value="  ").action is A.NO_FLY
    assert over_sfc(contact=kind, value=None).action is A.NO_FLY


@pytest.mark.parametrize("kind", [ContactKind.PHONE, ContactKind.EMAIL])
def test_f11_a_prohibitive_category_with_a_contact_is_no_fly_never_contact_first(kind):
    result = over_sfc(category=Category.PROHIBITIVE, contact=kind, value="07784 546062")
    assert result.action is A.NO_FLY
    assert result.reason.endswith("prohibitive")


@pytest.mark.parametrize("why", [ProhibitiveReason.BYELAWS, ProhibitiveReason.ENTRY_PROHIBITED, ProhibitiveReason.LIVE_FIRING])
def test_f11_a_prohibitive_reason_wins_over_a_cooperative_category(why):
    result = over_sfc(category=Category.COOPERATIVE, reason=why)
    assert result.action is A.NO_FLY
    assert f"prohibitive ({why.value})" in result.reason


def test_f11_a_prohibitive_reason_with_an_informational_category_is_no_fly():
    assert over_sfc(category=Category.INFORMATIONAL, reason=ProhibitiveReason.LIVE_FIRING).action is A.NO_FLY


def test_f11_a_radio_frequency_is_never_a_contact():
    result = over_sfc(contact=ContactKind.RADIO_ONLY, value="123.450")
    assert result.action is A.NO_FLY
    assert "a radio frequency alone is not a contact" in result.reason


def test_q13_an_informational_reading_over_the_lz_from_sfc_is_a_warning_never_no_fly():
    result = over_sfc(category=Category.INFORMATIONAL, activity="Fly-in at Garton", schedule="SR-SS", contact=ContactKind.NONE, value=None)
    assert result.action is A.AWARENESS  # shown in the answer, so PILOT DISCRETION - WITH CONDITIONS
    assert result.action not in (A.NO_FLY, A.IGNORE)
    assert result.reason.startswith("Fly-in at Garton (SR-SS): informational over the LZ from the surface")


def test_q13_an_informational_reading_within_or_above_the_lz_follows_the_rest_of_the_table():
    near = act(notam(circle=circle_at(1.0)), reading(geometry=circle_at(1.0), category=Category.INFORMATIONAL))
    assert near.action is A.AVOID_SECTOR
    above = act(notam(circle=circle_at(0)), reading(geometry=circle_at(0), category=Category.INFORMATIONAL, floor=amsl(2000)))
    assert above.action is A.AVOID_SECTOR


# --- no reading (Q8) -----------------------------------------------------------------------


def test_an_unread_notam_is_never_no_fly_or_contact_first():
    seen = set()
    for edge, band, d in itertools.product(EDGE_NM, BANDS, ACTIVE_D):
        parsed, _ = build(edge=edge, band=band, d=d, read=False)
        seen.add(act(parsed).action)
    assert seen == {A.IGNORE, A.UNREAD_OVER_LZ, A.AVOID_SECTOR, A.AWARENESS, A.AWARENESS_MAP}


def test_an_unread_notam_beyond_the_local_area_is_ignored_with_its_distance():
    result = act(notam(circle=circle_at(3.0)))
    assert result.action is A.IGNORE
    assert result.reason == f"{edge_distance(circle_at(3.0)):.2f} NM from the LZ, beyond the 2 NM local area"


# --- purity --------------------------------------------------------------------------------


def test_same_inputs_give_equal_results_and_the_inputs_are_not_changed():
    circle = circle_at(1.0)
    parsed = notam(circle=circle, item_f="SFC", item_g="700FT AGL", item_d="SR-SS")
    read = reading(geometry=circle, ceiling=agl(700), exemptions=[exemption(UNQUOTED)])
    seg = segment()
    before = (parsed.model_dump(), read.model_dump(), seg.model_dump(), LZ.model_dump())
    first = notam_action(parsed, read, LZ, seg)
    second = notam_action(copy.deepcopy(parsed), copy.deepcopy(read), LZ, seg)
    assert first == second
    assert (parsed.model_dump(), read.model_dump(), seg.model_dump(), LZ.model_dump()) == before
    assert first.debug == [f'exemption quote not found: "{UNQUOTED}"']
    unread = notam_action(parsed, None, LZ, seg)
    assert unread == notam_action(parsed, None, LZ, seg)
    assert parsed.model_dump() == before[0]


def test_the_result_is_a_pydantic_model_with_the_notam_id_action_and_one_line_reason():
    result = act(notam(circle=circle_at(1.0)))
    assert isinstance(result, ActionResult)
    assert result.notam_id == "H7171/26"
    assert ActionResult.model_validate_json(result.model_dump_json()) == result
