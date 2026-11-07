"""G3: the area code builds from a NOTAM's E text. Cases, the slack boundary, and DMS limits."""

import ast
import logging
from pathlib import Path

import pytest
from factories import LZ, circle_at, notam, segment, utc

from flysafelz.contracts import Circle, NotamAction, Polygon
from flysafelz.geo import METRES_PER_KM, METRES_PER_NM, distance_nm
from flysafelz.notam_action import notam_action
from flysafelz.notam_area import Broken, NoArea, Route, Shape, area_from_item_e, build_area
from flysafelz.pib import SortAction, sort_notam
from flysafelz.rules import RULES

# A Q-circle that holds every shape below (they are all within a mile of Garton), so the slack rule is not in play.
BIG_Q = Circle(lat=53.8, lon=-0.0833, radius_nm=5.0)


def area_of(text: str, q: Circle = BIG_Q):
    return build_area(notam(item_e=text, circle=q))


# --- one point and a radius: a circle ---------------------------------------------------------------


def test_one_coordinate_and_one_radius_is_a_circle():
    area, note = area_of("UAS OPR WI 0.5NM RADIUS OF 534732N 0000459W, SFC-700FT AGL.")
    assert isinstance(area, Circle)
    assert (area.lat, area.lon, area.radius_nm) == pytest.approx((53 + 47 / 60 + 32 / 3600, -(4 / 60 + 59 / 3600), 0.5), abs=1e-9)
    assert note.startswith("E text")


def test_within_of_psn_with_no_radius_word_is_a_circle():
    area, _ = area_of("DRONE OPS WI 0.5NM OF PSN 534732N 0000459W.")
    assert isinstance(area, Circle)
    assert area.radius_nm == 0.5


def test_within_of_without_psn_is_a_circle():
    area, _ = area_of("WI 0.5NM OF 534732N 0000459W.")
    assert isinstance(area, Circle)
    assert area.radius_nm == 0.5


def test_radius_before_the_word_and_after_it_both_count():
    before, _ = area_of("PARACHUTING 2NM RADIUS 534732N 0000459W.")
    after, _ = area_of("PARACHUTING RADIUS 2NM CENTRED ON 534732N 0000459W.")
    assert isinstance(before, Circle) and isinstance(after, Circle)
    assert before.radius_nm == after.radius_nm == 2.0


def test_a_decimal_radius_is_kept():
    area, _ = area_of("WI 1.61NM RADIUS OF 534732N 0000459W.")
    assert isinstance(area, Circle)
    assert area.radius_nm == 1.61


def test_the_same_radius_written_twice_is_still_one_radius():
    area, _ = area_of("WI 1NM RADIUS OF 534732N 0000459W. AREA IS A CIRCLE RADIUS 1NM.")
    assert isinstance(area, Circle)
    assert area.radius_nm == 1.0


# --- E7: metres and km are converted by code --------------------------------------------------------


def test_a_radius_in_metres_is_converted_with_metres_per_nm():
    area, _ = area_of("AREA BOUNDED BY A CIRCLE RADIUS 2000M CENTRED ON PSN 534732N 0000459W.")
    assert isinstance(area, Circle)
    assert area.radius_nm == pytest.approx(2000 / METRES_PER_NM)


def test_a_radius_in_km_is_converted_with_metres_per_km_and_metres_per_nm():
    area, _ = area_of("WI 2KM RADIUS OF 534732N 0000459W.")
    assert isinstance(area, Circle)
    assert area.radius_nm == pytest.approx(2 * METRES_PER_KM / METRES_PER_NM)


def test_metres_written_out_and_with_a_space():
    area, _ = area_of("RADIUS 185 METRES OF 534732N 0000459W.")
    assert isinstance(area, Circle)
    assert area.radius_nm == pytest.approx(185 / METRES_PER_NM)


def test_a_zero_radius_is_not_a_circle():
    area, note = area_of("WI 0NM RADIUS OF 534732N 0000459W.")
    assert area is None
    assert note == "Q-circle: the radius in the E text is zero or less, so there is no area"


def test_a_zero_radius_with_several_points_is_also_a_zero_radius_not_separate_areas():
    area, note = area_of("WI 0NM RADIUS OF 534732N 0000459W AND 534800N 0000500W.")
    assert area is None
    assert "zero or less" in note


def test_a_radius_with_more_than_one_point_is_separate_areas():
    area, note = area_of("WI 2NM RADIUS OF 534732N 0000459W AND 534800N 0000500W.")
    assert area is None
    assert note == "Q-circle: a radius with more than one point (separate areas)"


# --- one point, no radius; no point ----------------------------------------------------------------


def test_one_point_and_no_radius_is_the_q_circle():
    area, note = area_of("CRANE ERECTED AT 534732N 0000459W. HGT 150FT AGL.")
    assert area is None
    assert "1 point and no radius" in note


def test_two_points_and_no_radius_is_the_q_circle():
    area, note = area_of("ACT BETWEEN 534732N 0000459W AND 534800N 0000500W.")
    assert area is None
    assert "2 points and no radius" in note


def test_no_coordinates_is_the_q_circle():
    area, note = area_of("AERODROME USERS ONLY. CTC AERODROME ON 119.130 FOR DETAILS.")
    assert area is None
    assert "no coordinates" in note


def test_an_empty_e_text_is_the_q_circle():
    assert area_of("")[0] is None


# --- three or more points: a polygon ---------------------------------------------------------------


def test_three_points_and_no_radius_is_a_polygon_in_the_order_written():
    area, note = area_of("AREA BOUNDED BY 534700N 0000400W - 534800N 0000500W - 534900N 0000300W.")
    assert isinstance(area, Polygon)
    assert area.points == [
        pytest.approx((53 + 47 / 60, -(4 / 60))),
        pytest.approx((53 + 48 / 60, -(5 / 60))),
        pytest.approx((53 + 49 / 60, -(3 / 60))),
    ]
    assert "3 points" in note


def test_a_closing_repeat_of_the_first_point_is_dropped():
    area, _ = area_of("AREA 534700N 0000400W - 534800N 0000500W - 534900N 0000300W - 534700N 0000400W.")
    assert isinstance(area, Polygon)
    assert len(area.points) == 3


def test_a_polygon_that_repeats_a_point_in_the_middle_is_two_areas():
    area, note = area_of("AREA 534700N 0000400W - 534800N 0000500W - 534700N 0000400W - 534900N 0000300W - 535000N 0000200W.")
    assert area is None
    assert "repeats a point" in note


def test_a_polygon_list_split_over_lines_and_ending_in_a_full_stop():
    text = "AREA BOUNDED BY STRAIGHT LINES JOINING:\n534700N 0000400W - 534800N 0000500W -\n534900N 0000300W - 534700N 0000400W. DUE TO ACTIVITY."
    area, _ = area_of(text)
    assert isinstance(area, Polygon)
    assert len(area.points) == 3


# --- messy shapes: the Q-circle --------------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "text"),
    [
        ("corridor either side of a track", "CORRIDOR 1NM EITHER SIDE OF TRACK 534700N 0000400W - 534800N 0000500W - 534900N 0000300W."),
        ("either side, no corridor word", "WI 0.5NM EITHER SIDE OF THE LINE 534700N 0000400W - 534800N 0000500W."),
        ("buffer around a polygon", "BUFFER 0.5NM AROUND 534700N 0000400W - 534800N 0000500W - 534900N 0000300W."),
        ("an arc", "AREA BOUNDED BY ARC OF CIRCLE 534700N 0000400W - 534800N 0000500W - 534900N 0000300W."),
        ("along a line", "ALONG 534700N 0000400W - 534800N 0000500W - 534900N 0000300W."),
        ("two separate circles", "WI 1NM RADIUS OF 534732N 0000459W AND WI 1NM RADIUS OF 535000N 0001000W."),
        ("two separate points", "ACT AT 534732N 0000459W AND AT 535000N 0001000W."),
        ("two different radii", "WI 1NM RADIUS OF 534732N 0000459W AND 2NM RADIUS OF PSN."),
        ("the word radius with no figure", "WI THE RADIUS OF 534732N 0000459W."),
    ],
)
def test_a_messy_shape_is_the_q_circle(name, text):
    area, note = area_of(text)
    assert area is None
    assert note.startswith("Q-circle")


def test_a_polygon_whose_text_is_cut_off_is_the_q_circle():
    area, note = area_of("AREA BOUNDED BY STRAIGHT LINES JOINING: 534700N 0000400W - 534800N 0000500W - 534900N 0000300W -")
    assert area is None
    assert "stops in the middle" in note


@pytest.mark.parametrize("tail", ["- 5350", "- 535000N 0001", "-\n", "- 535000N 00010", "\u2013 5350"])
def test_a_list_cut_off_inside_a_coordinate_is_the_q_circle(tail):
    area, _ = area_of(f"AREA 534700N 0000400W - 534800N 0000500W - 534900N 0000300W {tail}")
    assert area is None


def test_a_list_followed_by_words_is_not_cut_off():
    area, _ = area_of("AREA 534700N 0000400W - 534800N 0000500W - 534900N 0000300W. DUE TO 1500FT ACTIVITY")
    assert isinstance(area, Polygon)


# --- DMS boundaries --------------------------------------------------------------------------------


def test_seconds_form_in_the_west():
    area, _ = area_of("WI 1NM RADIUS OF 534732N 0000459W.")
    assert isinstance(area, Circle)
    assert area.lon < 0 < area.lat


def test_degrees_and_minutes_form_in_the_west():
    area, _ = area_of("WI 1NM RADIUS OF 5348N 00005W.")
    assert isinstance(area, Circle)
    assert (area.lat, area.lon) == pytest.approx((53 + 48 / 60, -5 / 60))


def test_degrees_and_minutes_form_in_the_east():
    area, _ = area_of("WI 1NM RADIUS OF 5348N 00005E.", Circle(lat=53.8, lon=0.0833, radius_nm=5.0))
    assert isinstance(area, Circle)
    assert area.lon == pytest.approx(5 / 60)


def test_south_is_negative():
    area, _ = area_of("WI 1NM RADIUS OF 5348S 00005E.", Circle(lat=-53.8, lon=0.0833, radius_nm=5.0))
    assert isinstance(area, Circle)
    assert area.lat == pytest.approx(-(53 + 48 / 60))


def test_seconds_with_decimals():
    area, _ = area_of("WI 1NM RADIUS OF 534732.5N 0000459.5W.")
    assert isinstance(area, Circle)
    assert (area.lat, area.lon) == pytest.approx((53 + 47 / 60 + 32.5 / 3600, -(4 / 60 + 59.5 / 3600)))


@pytest.mark.parametrize(
    ("name", "coordinate"),
    [
        ("minutes 60 (lat)", "5360N 00005W"),
        ("minutes 99 (lat)", "5399N 00005W"),
        ("minutes 60 (lon)", "5348N 00060W"),
        ("seconds 60 (lat)", "534760N 0000459W"),
        ("seconds 75 (lon)", "534732N 0000475W"),
        ("latitude 91", "9100N 00005W"),
        ("latitude 99 degrees", "995959N 0000459W"),
        ("latitude 90 and a bit", "900001N 0000459W"),
        ("longitude 181", "5348N 18100W"),
        ("longitude 200", "5348N 20000W"),
    ],
)
def test_an_impossible_coordinate_is_the_q_circle_and_never_an_exception(name, coordinate):
    area, note = area_of(f"WI 1NM RADIUS OF {coordinate}.")
    assert area is None
    assert "not a possible place" in note


def test_the_limits_themselves_are_possible_places():
    pole, _ = area_of("WI 1NM RADIUS OF 9000N 00000E.", Circle(lat=90.0, lon=0.0, radius_nm=5.0))
    date_line, _ = area_of("WI 1NM RADIUS OF 0000N 18000E.", Circle(lat=0.0, lon=180.0, radius_nm=5.0))
    last_second, _ = area_of("WI 1NM RADIUS OF 535959N 0005959W.", Circle(lat=54.0, lon=-1.0, radius_nm=5.0))
    assert isinstance(pole, Circle) and isinstance(date_line, Circle) and isinstance(last_second, Circle)


def test_one_impossible_coordinate_in_a_polygon_makes_it_the_q_circle():
    area, _ = area_of("AREA 534700N 0000400W - 534860N 0000500W - 534900N 0000300W.")
    assert area is None


def test_a_coordinate_glued_to_other_digits_is_not_read_as_one():
    area, _ = area_of("REF 9534732N 0000459W9 WI 1NM RADIUS OF PSN")
    assert area is None


# --- G3 slack: more than reader_area_slack_nm outside the Q-circle is a parser miss ------------------

SLACK = RULES.reader_area_slack_nm.value
assert isinstance(SLACK, float)
CENTRE_DMS = "534800N 0000500W"
CENTRE = (53 + 48 / 60, -(5 / 60))


def _q(radius_nm: float) -> Circle:
    return Circle(lat=CENTRE[0], lon=CENTRE[1], radius_nm=radius_nm)


def test_a_circle_exactly_the_slack_outside_is_kept(caplog):
    assert distance_nm(*CENTRE, *CENTRE) == 0  # the E text's centre is the Q-circle's centre, so the reach is just the radius
    with caplog.at_level(logging.WARNING, logger="flysafelz.notam_area"):
        area, note = area_of(f"WI {2 + SLACK:g}NM RADIUS OF {CENTRE_DMS}.", _q(2.0))
    assert isinstance(area, Circle)
    assert area.radius_nm == 2 + SLACK
    assert note.startswith("E text")
    assert caplog.records == []


def test_a_circle_a_little_more_than_the_slack_outside_is_replaced_and_logged_once(caplog):
    with caplog.at_level(logging.WARNING, logger="flysafelz.notam_area"):
        area, note = area_of(f"WI {2 + SLACK + 0.01:g}NM RADIUS OF {CENTRE_DMS}.", _q(2.0))
    assert area is None
    assert "parser miss" in note
    assert len(caplog.records) == 1
    assert "H7171/26" in caplog.records[0].getMessage()
    assert "parser miss" in caplog.records[0].getMessage()


def test_a_circle_inside_the_q_circle_is_kept(caplog):
    with caplog.at_level(logging.WARNING, logger="flysafelz.notam_area"):
        area, _ = area_of(f"WI 0.5NM RADIUS OF {CENTRE_DMS}.", _q(2.0))
    assert isinstance(area, Circle)
    assert caplog.records == []


def test_a_circle_off_centre_counts_its_far_edge(caplog):
    # Centred 1 NM north of the Q-circle's centre (one minute of latitude) with a 1.5 NM radius, it reaches 2.5 NM out:
    # 0.5 NM beyond a 2 NM Q-circle, inside the slack, so kept. With a 2.6 NM radius it reaches 3.6: 1.6 beyond, so replaced.
    with caplog.at_level(logging.WARNING, logger="flysafelz.notam_area"):
        kept, _ = area_of("WI 1.5NM RADIUS OF 534900N 0000500W.", _q(2.0))
        replaced, note = area_of("WI 2.6NM RADIUS OF 534900N 0000500W.", _q(2.0))
    assert isinstance(kept, Circle)
    assert replaced is None
    assert "parser miss" in note
    assert len(caplog.records) == 1


def test_a_polygon_more_than_the_slack_outside_is_replaced_and_logged_once(caplog):
    with caplog.at_level(logging.WARNING, logger="flysafelz.notam_area"):
        area, note = area_of("AREA 534700N 0000400W - 534800N 0000500W - 544900N 0000300W.", _q(2.0))
    assert area is None
    assert "parser miss" in note
    assert len(caplog.records) == 1


def test_a_polygon_inside_the_q_circle_is_kept(caplog):
    with caplog.at_level(logging.WARNING, logger="flysafelz.notam_area"):
        area, _ = area_of("AREA 534700N 0000400W - 534800N 0000500W - 534900N 0000300W.", _q(2.0))
    assert isinstance(area, Polygon)
    assert caplog.records == []


def test_a_q_circle_answer_is_not_logged_as_a_miss(caplog):
    with caplog.at_level(logging.WARNING, logger="flysafelz.notam_area"):
        assert area_of("NO COORDINATES HERE")[0] is None
    assert caplog.records == []


def test_the_slack_is_read_from_the_rule_not_held_here():
    assert SLACK == 1.0
    assert RULES.reader_area_slack_nm.comparison == ">"


# --- E8: a cut-off or odd real text never raises ---------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "",
        "\n\n",
        "534732N",
        "534732N 0000459",
        "RADIUS",
        "WI NM RADIUS OF",
        "WI 99999999999999999999NM RADIUS OF 534732N 0000459W",
        "- - - -",
        "534700N 0000400W -",
        "\u00e9\u4e2d 534732N 0000459W \U0001f6e9",
        "STRAIGHT LINES JOINING: \n145300N 0362700E - 143000N 0390000E - 141200N 04",
    ],
)
def test_odd_text_never_raises(text):
    area, note = area_of(text)
    assert area is None or isinstance(area, Circle | Polygon)
    assert note


# --- G16: where a NOTAM is, sorted into Shape, Route, Broken or NoArea -----------------------------------------------
# Fiction coordinates round 55 10 N 001 30 W (open sea), inside a 100 NM Q-circle.

HUGE_Q = Circle(lat=55 + 10 / 60, lon=-1.5, radius_nm=100.0)
SQUARE = "551000N 0013000W - 551000N 0012000W - 550500N 0012000W - 550500N 0013000W"


def kind_of(text: str, q: Circle = HUGE_Q):
    return area_from_item_e(notam(item_e=text, circle=q))


@pytest.mark.parametrize(
    ("text", "corners"),
    [
        (f"MILITARY EXERCISE WI AREA BOUNDED BY {SQUARE} - 551000N 0013000W. SFC-UNL.", 4),  # closed on its first corner
        (f"MILITARY EXERCISE WI AREA BOUNDED BY {SQUARE}. SFC-UNL.", 4),  # closed by code
        ("AREA 551000N 0013000W -\n551000N 0012000W -\n  550500N 0012000W\n- 550500N 0013000W -\n551000N 0013000W", 4),  # line breaks inside the list
        ("AREA 551000N0013000W-551000N0012000W-550500N0012000W", 3),  # no spaces at all
    ],
    ids=["closed", "unclosed", "line breaks", "no spaces"],
)
def test_a_hyphen_joined_list_is_a_polygon(text, corners):
    area = kind_of(text)
    assert isinstance(area, Shape) and area.kind == "polygon"
    assert isinstance(area.area, Polygon) and len(area.area.points) == corners  # the closing repeat is dropped


@pytest.mark.parametrize("text", ["DRONES WI 5NM OF PSN 551000N 0013000W", "DRONES WI 2.5NM RADIUS OF 551000N0013000W", "drones wi 2.5nm radius of psn 551000n 0013000w"])
def test_a_radius_and_one_coordinate_is_a_circle(text):
    area = kind_of(text)
    assert isinstance(area, Shape) and area.kind == "circle"
    assert isinstance(area.area, Circle) and area.area.radius_nm in (5.0, 2.5)
    assert area.area.lat == pytest.approx(55 + 10 / 60) and area.area.lon == pytest.approx(-1.5)


@pytest.mark.parametrize(
    ("text", "reason"),
    [
        ("DRONES WI 5NM OF PSN 551061N 0013000W", "not a possible place"),  # minutes 61
        ("DRONES WI 5NM OF PSN 551000N 0016100W", "not a possible place"),
        ("DRONES WI 5NM OF PSN 551060N 0013000W", "not a possible place"),  # seconds 60
        ("DRONES WI 5NM OF PSN 911000N 0013000W", "not a possible place"),  # latitude past 90
        ("AREA 551000N 0013000W - 551000N 0012000W", "2 points and no radius"),
        ("AREA 551000N 0013000W", "1 point and no radius"),
        ("AREA 551000N 0013000W - 550500N 0012000W - 551000N 0012000W - 550500N 0013000W", "crosses itself"),  # a bow tie
        ("DRONES WI 5NM OF PSN 551000N", "has no partner"),  # a latitude with no longitude
        ("DRONES WI 5NM OF PSN 0013000W", "has no partner"),
        ("AREA 551000N 0013000W - 551000N 0012000W - 550500N 0012000W - 551000N 0012000W - 550500N 0013000W", "repeats a point"),
        (f"AREA {SQUARE} - 55", "stops in the middle"),
        (f"CORRIDOR EITHER SIDE OF {SQUARE}", "corridor"),
        ("DRONES WI 0NM RADIUS OF 551000N 0013000W", "zero"),
        ("DRONES WI 5NM RADIUS OF 551000N 0013000W AND 551000N 0012000W", "separate areas"),
    ],
)
def test_coordinate_looking_text_that_makes_no_valid_shape_is_broken_with_a_reason(text, reason):
    area = kind_of(text)
    assert isinstance(area, Broken), area
    assert reason in area.reason
    assert "5510" not in area.reason  # the reason never quotes the E text


def test_a_shape_reaching_far_outside_the_q_circle_is_broken_as_a_parser_miss():
    area = kind_of(f"AREA {SQUARE}", q=circle_at(0, radius_nm=1.0))  # the Q-circle is round the LZ, a long way from 55 10 N
    assert isinstance(area, Broken) and "parser miss" in area.reason


def test_coordinates_each_with_a_time_are_a_route():
    text = "RED ARROWS TRANSIT\n551000N 0013000W 0900\n551000N 0012000W 0910Z\n550500N 0012000W 0920\n"
    assert kind_of(text) == Route(waypoints=3)


def test_a_polygon_written_on_one_line_with_a_schedule_after_it_is_a_shape_not_a_route():
    """Every corner on one line, then '0800-1200 DAILY': the time is the schedule, not a waypoint time (the Storm Warrior case)."""
    one_line = f"EXERCISE WI AREA BOUNDED BY {SQUARE}. SFC-UNL. 0800-1200 DAILY."
    assert isinstance(kind_of(one_line), Shape)
    broken_up = f"EXERCISE WI AREA BOUNDED BY {SQUARE}.\nSFC-UNL. 0800-1200 DAILY."
    assert isinstance(kind_of(broken_up), Shape)  # the same text with a line break gave a Shape before, so it must now too
    with_places = "TRANSIT\n551000N 0013000W ALNWICK 0900\n551000N 0012000W AMBLE 0910Z\n"
    assert kind_of(with_places) == Route(waypoints=2)  # a place name between the coordinate and its time is still a route


def test_a_one_line_polygon_with_a_schedule_is_measured_not_sent_to_the_area_wide_rule():
    """Through the sort and the action table: a 152 NM Q-circle over the LZ and a small square 5 NM away, times in the text."""
    far_square = " - ".join(f"{lat} {lon}" for lat, lon in [("554000N", "0013000W"), ("554000N", "0012000W"), ("553500N", "0012000W"), ("553500N", "0013000W")])
    parsed = notam(circle=circle_at(0, radius_nm=152.0), item_e=f"EXERCISE WI AREA BOUNDED BY {far_square}. SFC-UNL. 0800-1200 DAILY.", item_f="SFC", item_g="UNL")
    assert isinstance(area_from_item_e(parsed), Shape)
    assert sort_notam(parsed, (utc(9), utc(10)), (LZ.lat, LZ.lon)).action is SortAction.TO_READER
    assert notam_action(parsed, None, LZ, segment()).action is NotamAction.IGNORE


def test_a_parser_miss_is_logged_once_however_many_segments_and_callers_ask(caplog):
    """A shape ~80 NM from the LZ on a 60 NM Q-circle over it is a parser miss. Five segments, the sort and the reader's
    build_area all ask for its area: one warning."""
    parsed = notam("P1/26", circle=circle_at(0, radius_nm=60.0), item_e=f"EXERCISE WI AREA BOUNDED BY {SQUARE}.", item_f="SFC", item_g="UNL")
    with caplog.at_level(logging.WARNING, logger="flysafelz.notam_area"):
        assert sort_notam(parsed, (utc(9), utc(14)), (LZ.lat, LZ.lon)).action is SortAction.TO_READER
        for hour in range(9, 14):
            result = notam_action(parsed, None, LZ, segment(utc(hour), utc(hour + 1)))
            assert result.action is NotamAction.UNREAD_OVER_LZ  # the Q-circle (60 NM, over the LZ) decides
        assert build_area(parsed)[0] is None
    misses = [r for r in caplog.records if "parser miss" in r.getMessage()]
    assert len(misses) == 1 and "P1/26" in misses[0].getMessage()


def test_a_remembered_area_follows_the_slack_rule_and_is_not_shared_between_callers():
    parsed = notam(circle=Circle(lat=53.8, lon=-0.0833, radius_nm=5.0), item_e="UAS OPR WI 0.5NM RADIUS OF 534732N 0000459W.")
    first, _ = build_area(parsed)
    assert isinstance(first, Circle)
    first.radius_nm = 99.0  # a caller changing its copy
    second, _ = build_area(parsed)
    assert isinstance(second, Circle) and second.radius_nm == 0.5


def test_a_polygon_whose_corners_have_no_times_is_not_a_route_and_one_with_a_time_missing_is_not_either():
    assert isinstance(kind_of(f"AREA {SQUARE}"), Shape)
    partly = "TRANSIT\n551000N 0013000W 0900\n551000N 0012000W\n550500N 0012000W 0920\n"
    assert not isinstance(kind_of(partly), Route)  # one line has no time: not a route (a polygon whose text stops mid-way is Broken)


@pytest.mark.parametrize(
    "text",
    [
        "",
        "FREQ 128.925MHZ OR 125.275MHZ. FL195. FL095-FL195.",
        "SEE AIP SUP 108/2026 AND AR-2026-7404/05. AIP ENR 5.1.",
        "CTC 07700 900123 OR 07700 900456 OR 01632 960001.",
        "SFC-2000FT AMSL. 0800-1800. DAILY 1200 1230Z.",
        "TEMPORARY DANGER AREA EGD496A ACTIVATED.",
    ],
)
def test_text_that_only_looks_like_numbers_is_no_area(text):
    assert kind_of(text) == NoArea()


def test_item_e_is_read_case_insensitively_and_never_raises():
    assert isinstance(kind_of(f"area {SQUARE.lower()}"), Shape)
    for text in ("", "\n", "-", "551000N 0013000W -", "\u00e9\u4e2d 551000N 0013000W", "9" * 400 + "N"):
        assert kind_of(text) is not None


def test_a_broken_text_falls_back_to_the_q_circle_never_to_a_place_it_might_have_been_outside():
    """Fail-closed order: the Q-circle covers the LZ, and the broken text (a latitude with no longitude, then a polygon
    drawn far away that cannot be used) would have put the NOTAM outside. It comes out over the LZ."""
    q_over_lz = circle_at(0, radius_nm=3.0)
    parsed = notam(circle=q_over_lz, item_e="PARACHUTING WI 1NM OF PSN 551000N, 550000N 0013000W - 550000N 0012000W. SFC-UNL", item_f="SFC")
    result = notam_action(parsed, None, LZ, segment())
    assert result.action is NotamAction.UNREAD_OVER_LZ
    assert any("coordinates not usable" in line and "the Q-circle is used" in line for line in result.debug)
    outside = notam(circle=circle_at(5.0), item_e="PARACHUTING WI 1NM OF PSN 551000N", item_f="SFC")
    assert notam_action(outside, None, LZ, segment()).action is NotamAction.IGNORE  # the Q-circle is what decides, both ways


def test_a_polygon_in_item_e_decides_however_large_the_q_circle():
    """A 152 NM Q-circle over the LZ, a small polygon 5 NM away drawn in item E: the polygon decides, not the Q-circle."""
    far_square = " - ".join(f"{lat} {lon}" for lat, lon in [("554000N", "0013000W"), ("554000N", "0012000W"), ("553500N", "0012000W"), ("553500N", "0013000W")])
    parsed = notam(circle=circle_at(0, radius_nm=152.0), item_e=f"EXERCISE WI AREA BOUNDED BY {far_square}", item_f="SFC", item_g="UNL")
    assert isinstance(area_from_item_e(parsed), Shape)
    assert notam_action(parsed, None, LZ, segment()).action is NotamAction.IGNORE


def test_notam_area_imports_only_the_standard_library_and_the_projects_pure_modules():
    """Fence: no model, no network, no reader and no client is reachable from the area code."""
    tree = ast.parse(Path(build_area.__code__.co_filename).read_text(encoding="utf-8"))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
        elif isinstance(node, ast.Import):
            imported |= {alias.name for alias in node.names}
    assert {m for m in imported if m.startswith("flysafelz")} <= {"flysafelz.contracts", "flysafelz.geo", "flysafelz.rules"}
    assert not {m.split(".")[0] for m in imported} & {"httpx", "requests", "openai", "langchain", "langgraph", "socket", "urllib"}
