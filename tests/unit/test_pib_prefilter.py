"""G13: the prefilter keeps a NOTAM only when the Q-circle edge, the lower FL and B-C against the window all pass."""

import math
from datetime import timedelta

import pytest
from pib_helpers import LZ, SPAN, TRIMMED, parsed, utc, with_rules

from flysafelz.geo import circle_edge_distance_nm, distance_nm
from flysafelz.pib import parse_pib, prefilter
from flysafelz.rules import RULES

WINDOW = SPAN  # 2026-10-31 07:00Z to 18:00Z, half-open


def circle_at_edge_distance(edge_nm: float, **overrides: object):
    """A NOTAM whose Q-circle edge is `edge_nm` from the LZ, measured by geo.py (due north, 4 NM centre)."""
    centre_lat = LZ[0] + 4 / 60  # roughly 4 NM north; the exact distance comes from geo.distance_nm
    centre_distance = distance_nm(LZ[0], LZ[1], centre_lat, LZ[1])
    return parsed(centre_lat=centre_lat, centre_lon=LZ[1], radius_nm=centre_distance - edge_nm, **overrides)


def verdict(notam, window=WINDOW, rules=RULES):
    kept, dropped = prefilter([notam], LZ, window, rules=rules)
    assert len(kept) + len(dropped) == 1
    return (True, "") if kept else (False, dropped[0].reason)


# --- the Q-circle edge ---------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("edge_nm", "keep"),
    [(0.0, True), (1.0, True), (1.99, True), (2.01, False), (2.5, False), (10.0, False)],
)
def test_the_distance_to_the_circle_edge_decides(edge_nm, keep):
    notam = circle_at_edge_distance(edge_nm)
    measured = circle_edge_distance_nm(LZ[0], LZ[1], notam.centre_lat, notam.centre_lon, notam.radius_nm)
    assert measured == pytest.approx(edge_nm, abs=1e-9)  # the test's own geometry agrees with geo.py
    assert verdict(notam)[0] is keep


def test_an_edge_at_exactly_the_local_area_is_kept_and_a_hair_beyond_is_dropped():
    notam = circle_at_edge_distance(2.0)
    edge = circle_edge_distance_nm(LZ[0], LZ[1], notam.centre_lat, notam.centre_lon, notam.radius_nm)
    exact = with_rules(local_area=edge)
    assert verdict(notam, rules=exact)[0] is True
    just_inside = with_rules(local_area=math.nextafter(edge, 0.0))
    assert verdict(notam, rules=just_inside)[0] is False


def test_the_shipped_local_area_is_two_nm():
    assert RULES.local_area.value == 2
    assert verdict(circle_at_edge_distance(2.0 - 1e-6))[0] is True
    assert verdict(circle_at_edge_distance(2.0 + 1e-6))[0] is False


def test_inside_the_circle_counts_as_zero_distance():
    notam = parsed(centre_lat=LZ[0], centre_lon=LZ[1], radius_nm=0.5)
    assert circle_edge_distance_nm(LZ[0], LZ[1], notam.centre_lat, notam.centre_lon, notam.radius_nm) == 0.0
    assert verdict(notam)[0] is True


def test_the_distance_is_to_the_edge_not_the_centre():
    # The centre is about 4 NM away, beyond the local area, but the circle reaches to 1 NM of the LZ.
    notam = circle_at_edge_distance(1.0)
    centre = distance_nm(LZ[0], LZ[1], notam.centre_lat, notam.centre_lon)
    assert centre > RULES.local_area.value
    assert verdict(notam)[0] is True


def test_a_999_nm_circle_around_the_uk_passes_the_distance_check():
    notam = parsed(centre_lat=55 + 4 / 60, centre_lon=-5.0, radius_nm=999.0)
    assert verdict(notam)[0] is True


def test_a_far_circle_is_dropped_with_the_distance_in_the_reason():
    kept, dropped = prefilter([parsed("A3857/26", centre_lat=51.48, centre_lon=-0.38, radius_nm=1.0)], LZ, WINDOW)
    assert kept == []
    assert dropped[0].notam_id == "A3857/26"
    assert "NM from the LZ" in dropped[0].reason
    assert "2 NM local area" in dropped[0].reason


# --- the lower flight level -------------------------------------------------------------------------------


@pytest.mark.parametrize(("lower", "keep"), [(0, True), (194, True), (195, True), (196, False), (999, False)])
def test_lower_fl_at_the_cap_is_kept_and_one_above_is_dropped(lower, keep):
    ok, reason = verdict(parsed(lower_fl=lower))
    assert ok is keep
    if not keep:
        assert f"FL{lower:03d}" in reason and "FL195" in reason


# --- B-C against the half-open window [start, end) ----------------------------------------------------------


START, END = WINDOW


@pytest.mark.parametrize(
    ("name", "valid_from", "valid_to", "keep"),
    [
        ("inside", START + timedelta(hours=1), END - timedelta(hours=1), True),
        ("covers the whole window", START - timedelta(days=1), END + timedelta(days=1), True),
        ("C one minute after the window start", START - timedelta(days=1), START + timedelta(minutes=1), True),
        ("C exactly at the window start", START - timedelta(days=1), START, False),
        ("C before the window start", START - timedelta(days=2), START - timedelta(minutes=1), False),
        ("B one minute before the window end", START, END - timedelta(minutes=1), True),
        ("B exactly at the window end", END, END + timedelta(days=1), False),
        ("B after the window end", END + timedelta(hours=1), END + timedelta(days=1), False),
        ("B at the window start", START, START + timedelta(hours=1), True),
    ],
)
def test_b_to_c_must_overlap_the_half_open_window(name, valid_from, valid_to, keep):
    assert verdict(parsed(valid_from=valid_from, valid_to=valid_to))[0] is keep, name


@pytest.mark.parametrize(
    ("valid_from", "keep"),
    [
        (START - timedelta(days=30), True),  # PERM from long ago overlaps any window after B
        (START + timedelta(hours=2), True),  # B inside the window
        (END - timedelta(minutes=1), True),
        (END, False),  # B at the window end: not yet
        (END + timedelta(days=1), False),
    ],
)
def test_a_perm_c_overlaps_any_window_after_b(valid_from, keep):
    assert verdict(parsed(valid_from=valid_from, valid_to=None, permanent=True))[0] is keep


def test_dropped_window_reasons_say_which_edge():
    _ok, late = verdict(parsed(valid_from=END, valid_to=None, permanent=True))
    _ok, early = verdict(parsed(valid_from=START - timedelta(days=1), valid_to=START))
    assert "window end" in late
    assert "window start" in early


# --- order, bookkeeping, and the real fixture --------------------------------------------------------------------


def test_the_first_failing_rule_gives_the_reason_and_every_notam_is_accounted_for():
    notams = [
        parsed("H1/26"),
        parsed("H2/26", centre_lat=51.0),
        parsed("H3/26", lower_fl=300),
        parsed("H4/26", valid_from=END),
        parsed("H5/26", centre_lat=51.0, lower_fl=300, valid_from=END),
    ]
    kept, dropped = prefilter(notams, LZ, WINDOW)
    assert [n.id for n in kept] == ["H1/26"]
    assert [d.notam_id for d in dropped] == ["H2/26", "H3/26", "H4/26", "H5/26"]
    assert "NM from the LZ" in dropped[3].reason  # distance is checked first


def test_an_empty_list_gives_two_empty_lists():
    assert prefilter([], LZ, WINDOW) == ([], [])


def test_the_trimmed_pib_keeps_only_what_is_near_blue_mill_in_the_window():
    _issued, notams = parse_pib(TRIMMED.read_bytes())
    week = (utc(31, 7), utc(7, 7, month=11))
    kept, dropped = prefilter(notams, LZ, week)
    assert sorted(n.id for n in kept) == ["H7171/26", "H7297/26", "J3857/26", "J3864/26", "Q172/26", "V34/26"]
    assert {d.notam_id for d in dropped} == {"A3857/26", "C7072/26", "J3377/26"}  # Heathrow, Teesside and Birmingham: too far


# --- F3: the numbers come from rules.yaml ---------------------------------------------------------------------------


def test_changing_the_local_area_in_a_rules_copy_changes_the_result():
    notam = circle_at_edge_distance(2.5)
    assert verdict(notam)[0] is False
    assert verdict(notam, rules=with_rules(local_area=3))[0] is True
    assert verdict(circle_at_edge_distance(1.5), rules=with_rules(local_area=1))[0] is False


def test_changing_the_fl_cap_in_a_rules_copy_changes_the_result():
    notam = parsed(lower_fl=200)
    assert verdict(notam)[0] is False
    assert verdict(notam, rules=with_rules(notam_max_lower_fl=250))[0] is True
    assert verdict(parsed(lower_fl=100), rules=with_rules(notam_max_lower_fl=50))[0] is False


def test_a_list_rule_read_as_a_number_is_a_programming_error():
    from flysafelz.pib import _number

    assert _number(RULES.local_area) == 2
    with pytest.raises(TypeError, match="notam_keep_subjects"):
        _number(RULES.notam_keep_subjects)
