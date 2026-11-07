"""Blue Mill, 31 Oct 2026: J3857/26 (TDA EGD496A) and H7171/26 (Garton drone ops), built in code from the plan values.

LZ 53.802797, -0.086644; frozen now 2026-10-31T07:30Z; segment 08:00-09:00Z. Nothing is read from
a fixture file. The readings are written by hand.
"""

import pytest
from factories import LZ, SFC, agl, amsl, notam, reading, segment, utc

from flysafelz.contracts import Category, Circle, ContactKind, NotamAction
from flysafelz.geo import circle_edge_distance_nm, circle_over, compass16, distance_nm
from flysafelz.notam_action import notam_action
from flysafelz.redact import redact_phones

A = NotamAction
SEGMENT = segment(utc(8), utc(9))
EARLIER = segment(utc(7), utc(8))

J3857_E = "TEMPORARY DANGER AREA EGD496A ACTIVATED. UAS BVLOS OPS. SFC-1300FT AMSL. 0800-1800. BOOKING PHONE 07784 546062."
H7171_E = "UAS OPR WI 0.5NM RADIUS OF 534732N 0000459W (GARTON). MAX HGT 700FT AGL. FOR INFO 07784 546062. AR-2026-10264/01."

# J3857/26: Q-circle 10 NM centred 5356N00001E.
J3857_Q = Circle(lat=53 + 56 / 60, lon=1 / 60, radius_nm=10)
# H7171/26: Q-circle 1 NM centred 5348N00005W; the E text's own circle is 0.5 NM round 534732N 0000459W.
H7171_Q = Circle(lat=53 + 48 / 60, lon=-5 / 60, radius_nm=1)
H7171_CIRCLE = Circle(lat=53 + 47 / 60 + 32 / 3600, lon=-(4 / 60 + 59 / 3600), radius_nm=0.5)


@pytest.fixture
def j3857():
    return notam("J3857/26", circle=J3857_Q, subject="RD", item_d="0800-1800", item_e=J3857_E, item_f="SFC", item_g="1300FT AMSL", valid_from=utc(8), valid_to=utc(18))


@pytest.fixture
def j3857_reading():
    return reading(
        notam_id="J3857/26",
        category=Category.COOPERATIVE,
        activity="UAS BVLOS ops",
        geometry=None,
        floor=SFC,
        ceiling=amsl(1300),
        schedule="0800-1800",
        contact=ContactKind.PHONE,
        value="Flylogix 07784 546062",
        designators=["EGD496A"],
    )


@pytest.fixture
def h7171():
    # G is written as the 700 ft AGL the E text and the reading give, so there is nothing to merge.
    return notam("H7171/26", circle=H7171_Q, item_d="0800-1800", item_e=H7171_E, item_f="SFC", item_g="700FT AGL", valid_from=utc(8, day=29), valid_to=utc(18, day=10, month=11))


@pytest.fixture
def h7171_reading():
    return reading(notam_id="H7171/26", category=Category.COOPERATIVE, activity="drone ops", geometry=H7171_CIRCLE, floor=SFC, ceiling=agl(700), schedule="0800-1800")


def test_the_clock_is_frozen_at_0730(now):
    assert now == utc(7, 30)


def test_j3857_read_is_contact_first_and_activates_egd496a(j3857, j3857_reading):
    result = notam_action(j3857, j3857_reading, LZ, SEGMENT)
    assert result.action is A.CONTACT_FIRST
    assert "activates EGD496A" in result.reason
    assert "07784 546062" in result.reason
    assert "\n" not in result.reason


def test_j3857_unread_is_unverified_with_the_redacted_e_text(j3857):
    """The E text in the reason has its phone number swapped for <PHONE_n>; the CLI puts the number back when it prints (R3.5)."""
    result = notam_action(j3857, None, LZ, SEGMENT)
    assert result.action is A.UNREAD_OVER_LZ
    assert redact_phones(J3857_E)[0] in result.reason
    assert redact_phones(J3857_E)[1]["<PHONE_1>"] not in result.reason and "<PHONE_1>" in result.reason


def test_j3857_in_the_0700_to_0800_segment_is_ignored_as_inactive(j3857, j3857_reading):
    assert notam_action(j3857, j3857_reading, LZ, EARLIER).action is A.IGNORE
    assert notam_action(j3857, None, LZ, EARLIER).action is A.IGNORE
    assert "not active in this segment" in notam_action(j3857, None, LZ, EARLIER).reason


def test_h7171_geometry_the_lz_is_inside_the_q_circle_and_0_15_nm_outside_the_reading_circle():
    assert circle_over(LZ.lat, LZ.lon, H7171_Q.lat, H7171_Q.lon, H7171_Q.radius_nm)
    assert not circle_over(LZ.lat, LZ.lon, H7171_CIRCLE.lat, H7171_CIRCLE.lon, H7171_CIRCLE.radius_nm)
    assert distance_nm(LZ.lat, LZ.lon, H7171_CIRCLE.lat, H7171_CIRCLE.lon) == pytest.approx(0.65, abs=0.01)  # centre
    assert circle_edge_distance_nm(LZ.lat, LZ.lon, H7171_CIRCLE.lat, H7171_CIRCLE.lon, H7171_CIRCLE.radius_nm) == pytest.approx(0.15, abs=0.01)
    assert circle_edge_distance_nm(LZ.lat, LZ.lon, H7171_Q.lat, H7171_Q.lon, H7171_Q.radius_nm) == 0.0


def test_h7171_read_is_avoid_sector_0_15_nm_to_the_sse_up_to_700_ft_agl(h7171, h7171_reading):
    result = notam_action(h7171, h7171_reading, LZ, SEGMENT)
    assert result.action is A.AVOID_SECTOR
    assert result.bearing_deg == pytest.approx(168.6, abs=0.5)  # golden: degrees, never the word (0.15 deg from S/SSE)
    assert result.reason == f"drone ops 0.15 NM to the {compass16(result.bearing_deg)} (circle edge), up to 700 ft AGL: avoid that sector"
    assert "0.15 NM" in result.reason
    assert "up to 700 ft AGL" in result.reason


def test_h7171_unread_is_a_sector_because_its_e_text_draws_a_circle_that_misses_the_lz(h7171):
    """Code reads the E text's circle (G16), so no reading is needed to know it is 0.15 NM from the LZ."""
    result = notam_action(h7171, None, LZ, SEGMENT)
    assert result.action is A.AVOID_SECTOR
    assert result.bearing_deg == pytest.approx(168.6, abs=0.5)
    assert "(circle edge)" in result.reason


def test_h7171_unread_is_unverified_when_its_e_text_draws_nothing_because_the_q_circle_covers_the_lz_from_the_surface(h7171):
    no_area = h7171.model_copy(update={"item_e": H7171_E.replace("WI 0.5NM RADIUS OF 534732N 0000459W", "WI 0.5NM")})
    result = notam_action(no_area, None, LZ, SEGMENT)
    assert result.action is A.UNREAD_OVER_LZ
    assert redact_phones(no_area.item_e)[0] in result.reason  # redacted: no real number in the reason (R3.5)
    assert redact_phones(no_area.item_e)[1]["<PHONE_1>"] not in result.reason


def test_h7171_with_the_real_g_item_the_merge_rule_gives_the_higher_ceiling(h7171_reading):
    """The PIB's own G is 800FT AMSL; the reading says 700 ft AGL (730 ft AMSL here). E10: the higher ceiling wins, and says so."""
    real = notam("H7171/26", circle=H7171_Q, item_d="0800-1800", item_e=H7171_E, item_f="SFC", item_g="800FT AMSL", valid_from=utc(8, day=29), valid_to=utc(18, day=10, month=11))
    result = notam_action(real, h7171_reading, LZ, SEGMENT)
    assert result.action is A.AVOID_SECTOR
    assert "up to 800 ft AMSL" in result.reason
    assert "disagree on the ceiling (700 ft AGL against 800 ft AMSL)" in result.reason


def test_h7171_is_ignored_before_0800(h7171, h7171_reading):
    assert notam_action(h7171, h7171_reading, LZ, EARLIER).action is A.IGNORE
