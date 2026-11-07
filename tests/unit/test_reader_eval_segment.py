"""G13 eval segment: one hour centred on the middle of the NOTAM's first active interval, resolved at the area's centre."""

from datetime import timedelta

import pytest
from factories import LZ, reading, utc
from reader_eval_helpers import q_notam

from evals.reader_eval.items import load_items, parsed_notam
from evals.reader_eval.settings import SETTINGS
from evals.reader_eval.spots import area_centre, eval_segment
from flysafelz.contracts import Circle
from flysafelz.notam_action import LandingZone, active_intervals
from flysafelz.notam_area import build_area

HOUR = timedelta(hours=1)
ITEMS = load_items()


def centre_lz(parsed):
    lat, lon = area_centre(build_area(parsed)[0], parsed)
    return LandingZone(lat=lat, lon=lon, elevation_ft=SETTINGS.site_elevation_ft)


def test_with_no_d_the_segment_is_centred_on_the_middle_of_b_to_c():
    parsed = q_notam(valid_from=utc(8), valid_to=utc(16))
    segment = eval_segment(parsed, None)
    assert segment.start == utc(11, 30)
    assert segment.end - segment.start == HOUR
    assert segment.pressure_msl == 1013.25  # RULES.fl_reference_hpa, read by name in settings.py


def test_d_decides_the_first_interval():
    parsed = q_notam(valid_from=utc(0), valid_to=utc(23), item_d="1000-1200")
    segment = eval_segment(parsed, None)
    assert segment.start == utc(10, 30)  # the middle of 1000-1200 is 11:00
    assert segment.end == utc(11, 30)


def test_sr_ss_resolves_at_the_area_centre():
    parsed = q_notam(valid_from=utc(0), valid_to=utc(23, 59), item_d="SR-SS", item_f="SFC")
    segment = eval_segment(parsed, None)
    (first, last) = active_intervals(parsed, None, centre_lz(parsed), utc(0), utc(23, 59))[0]
    assert segment.start + HOUR / 2 == first + (last - first) / 2
    assert first.hour in range(5, 9) and last.hour in range(14, 18)  # a day-length interval in late October: sunrise and sunset in UTC


def test_the_segment_uses_the_area_centre_not_the_landing_zone():
    far_east = Circle(lat=LZ.lat, lon=LZ.lon + 8.0, radius_nm=3.0)  # sunrise is half an hour or more earlier there
    near = q_notam(valid_from=utc(0), valid_to=utc(23, 59), item_d="SR-SS", item_f="SFC")
    far = q_notam(valid_from=utc(0), valid_to=utc(23, 59), item_d="SR-SS", item_f="SFC", circle=far_east)
    assert eval_segment(far, None).start < eval_segment(near, None).start


def test_an_interval_shorter_than_an_hour_still_gets_a_segment_that_overlaps_it():
    parsed = q_notam(valid_from=utc(0), valid_to=utc(23), item_d="1000-1020")
    segment = eval_segment(parsed, None)
    ((first, last),) = active_intervals(parsed, None, centre_lz(parsed), utc(0), utc(23))
    assert last - first == timedelta(minutes=20)
    assert segment.start < last and first < segment.end


def test_no_d_and_no_end_time_is_b_to_b_plus_an_hour():
    open_ended = q_notam(valid_from=utc(8), valid_to=None, permanent=False)
    permanent = q_notam(valid_from=utc(8), valid_to=None, permanent=True)
    for parsed in (open_ended, permanent):
        segment = eval_segment(parsed, None)
        assert (segment.start, segment.end) == (parsed.valid_from, parsed.valid_from + HOUR)


def test_a_d_item_with_no_end_time_is_searched_from_b():
    parsed = q_notam(valid_from=utc(8), valid_to=None, permanent=True, item_d="1000-1200")
    segment = eval_segment(parsed, None)
    assert segment.start == utc(10, 30)


def test_the_segment_is_the_same_for_the_reader_and_for_the_label():
    """The reading's schedule is code's item D, so a reading and the label give one segment: the two sides differ only by their readings."""
    parsed = q_notam(valid_from=utc(0), valid_to=utc(23), item_d="1000-1200")
    assert eval_segment(parsed, reading(schedule="1000-1200")) == eval_segment(parsed, None)


@pytest.mark.parametrize("item", ITEMS, ids=[item["id"] for item in ITEMS])
def test_every_committed_item_gets_a_segment_that_overlaps_an_active_interval(item):
    parsed = parsed_notam(item)
    segment = eval_segment(parsed, None)
    assert segment.end - segment.start == HOUR
    lz = centre_lz(parsed)
    horizon = parsed.valid_to if parsed.valid_to is not None else segment.end + timedelta(days=7)
    intervals = active_intervals(parsed, None, lz, parsed.valid_from, horizon)
    assert any(start < segment.end and segment.start < end for start, end in intervals), (item["id"], parsed.item_d, segment)
