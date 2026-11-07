"""F6: assess() gives the same answer after the G2 refactor, and its cuts are the helpers' cuts.

tests/golden/assess_<case>.json are `model_dump_json()` of cases.run(<case>) taken once from
d6507a6, before notams_in_hour() and cut_points_with_reasons() existed. Like the text goldens they are
never rewritten by a test run.
"""

from pathlib import Path

import cases
import pytest
from factories import LZ

from flysafelz.assess import cut_points_with_reasons, notams_in_hour
from flysafelz.contracts import Tier
from flysafelz.daylight import ONE_HOUR, Phase, utc_hours
from flysafelz.pib import prefilter

GOLDEN = Path(__file__).resolve().parents[1] / "golden"
CASES = [cases.BLUE_MILL, cases.WEST_SLEEKBURN]


@pytest.mark.parametrize("name", CASES)
def test_assess_output_is_identical_to_the_snapshot_taken_before_the_refactor(name):
    snapshot = (GOLDEN / f"assess_{name}.json").read_text(encoding="utf-8").rstrip("\n")
    assert cases.run(name).model_dump_json() == snapshot


def helper_cuts(name, hour):
    """The helpers called the way assess calls them: prefiltered NOTAMs, then the hour's set, then its cuts."""
    case, _forecast, _airspace, pib, readings = cases.load(name)
    hours = utc_hours(case.window.start, case.window.end)
    near, _dropped = prefilter(pib.notams, (case.lat, case.lon), (hours[0], hours[-1] + ONE_HOUR))
    lz = LZ.model_copy(update={"lat": case.lat, "lon": case.lon})
    in_hour = notams_in_hour(hour, near, readings, lz)
    return cut_points_with_reasons(hour, case.lat, case.lon, in_hour, readings, lz)


@pytest.mark.parametrize("name", CASES)
def test_for_every_verdict_hour_the_helper_cuts_are_the_inner_segment_boundaries(name):
    assessment = cases.run(name)
    verdict_hours = [h for h in assessment.hours if h.tier is Tier.VERDICT]
    assert verdict_hours
    for hour in verdict_hours:
        inner = [s.start for s in hour.segments[1:]]
        assert [t for t, _ in helper_cuts(name, hour.hour)] == inner, hour.hour
        assert hour.segments[0].start == hour.hour and hour.segments[-1].end == hour.hour + ONE_HOUR


def test_blue_mill_17_hour_cuts_only_at_civil_dusk():
    # E1: the real NOTAMs add nothing to this hour.
    hour = cases.run(cases.BLUE_MILL).hours[-1].hour
    assert helper_cuts(cases.BLUE_MILL, hour) == [(hour.replace(minute=6), [(Phase.TWILIGHT, Phase.NIGHT)])]


def test_blue_mill_j3857_starts_on_the_hour_so_the_08_hour_is_one_segment():
    # E10 / E9: J3857/26 is valid from 08:00. Nothing is strictly inside the 08:00 hour, so it is one segment.
    assessment = cases.run(cases.BLUE_MILL)
    eight = next(h for h in assessment.hours if h.hour.hour == 8)
    assert helper_cuts(cases.BLUE_MILL, eight.hour) == []
    assert len(eight.segments) == 1
    assert {a.notam_id for a in eight.segments[0].notam_actions} >= {"J3857/26"}


def test_blue_mill_06_hour_has_dawn_and_sunrise_cuts_though_it_is_before_the_window():
    hour = cases.run(cases.BLUE_MILL).hours[0].hour.replace(hour=6)
    assert [(t.minute, r) for t, r in helper_cuts(cases.BLUE_MILL, hour)] == [(21, [(Phase.NIGHT, Phase.TWILIGHT)]), (59, [(Phase.TWILIGHT, Phase.DAY)])]
