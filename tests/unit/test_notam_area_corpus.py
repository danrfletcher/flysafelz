"""G3 over every NOTAM in the 4 committed archive PIBs: nothing raises, and each answer is a Circle, a Polygon or None."""

import logging
from collections import Counter
from pathlib import Path

import pytest
import yaml
from factories import notam

from flysafelz.contracts import Circle, Polygon
from flysafelz.geo import distance_nm
from flysafelz.notam_area import Broken, NoArea, Route, Shape, area_from_item_e, build_area
from flysafelz.pib import read_notams
from flysafelz.rules import RULES

ROOT = Path(__file__).resolve().parents[2]
PIBS = sorted((ROOT / "evals" / "reader" / "pibs").glob("*.xml"))
BLUE_MILL_SHA = "bdab14658f0270dc41460eebcd03332abfb230ec"
READINGS = ROOT / "fixtures" / "blue_mill_2026-10-31" / "readings.yaml"


@pytest.fixture(scope="module")
def corpus():
    return {path.stem: read_notams(path.read_bytes())[1] for path in PIBS}


def test_the_four_pibs_are_committed():
    assert len(PIBS) == 4
    assert BLUE_MILL_SHA in {path.stem for path in PIBS}


def test_build_area_over_every_notam_never_raises_and_always_answers_with_a_shape_or_none(corpus, caplog):
    counts = Counter()
    with caplog.at_level(logging.WARNING, logger="flysafelz.notam_area"):
        for sha, notams in corpus.items():
            assert notams, sha
            for parsed in notams:
                area, note = build_area(parsed)
                assert area is None or isinstance(area, Circle | Polygon), (sha, parsed.id)
                assert note, (sha, parsed.id)
                counts["q_circle" if area is None else area.kind] += 1
    total = sum(len(notams) for notams in corpus.values())
    print(f"build_area over {total} NOTAMs: {dict(sorted(counts.items()))}")
    assert sum(counts.values()) == total
    assert counts["circle"] > 0 and counts["polygon"] > 0 and counts["q_circle"] > 0
    assert len(caplog.records) <= counts["q_circle"]  # a parser miss is logged once, and only when it falls back


def test_every_area_built_from_the_corpus_is_within_the_slack_of_its_q_circle(corpus):
    for sha, notams in corpus.items():
        for parsed in notams:
            area, _ = build_area(parsed)
            if isinstance(area, Circle):
                reach = distance_nm(parsed.centre_lat, parsed.centre_lon, area.lat, area.lon) + area.radius_nm
            elif isinstance(area, Polygon):
                reach = max(distance_nm(parsed.centre_lat, parsed.centre_lon, lat, lon) for lat, lon in area.points)
            else:
                continue
            assert not RULES.reader_area_slack_nm.breached(reach - parsed.radius_nm), (sha[:8], parsed.id)


def test_h7171_builds_the_circle_in_blue_mills_readings(corpus):
    parsed = next(n for n in corpus[BLUE_MILL_SHA] if n.id == "H7171/26")
    area, _ = build_area(parsed)
    assert isinstance(area, Circle)
    assert area.lat == pytest.approx(53.79222, abs=1e-4)
    assert area.lon == pytest.approx(-0.08306, abs=1e-4)
    assert area.radius_nm == 0.5

    readings = {entry["id"]: entry for entry in yaml.safe_load(READINGS.read_text(encoding="utf-8"))}
    geometry = readings["H7171/26"]["geometry"]
    assert (geometry["kind"], geometry["radius_nm"]) == ("circle", 0.5)
    assert area.lat == pytest.approx(geometry["lat"], abs=1e-4)
    assert area.lon == pytest.approx(geometry["lon"], abs=1e-4)


def test_the_ethiopia_straight_lines_polygon_never_raises_and_is_the_q_circle(corpus, caplog):
    """A 10-point polygon over Ethiopia on a NOTAM whose Q-circle is in the UK: a parser miss, answered with the Q-circle."""
    found = [n for n in corpus[BLUE_MILL_SHA] if "STRAIGHT LINES" in n.item_e and "ETHIOPIA" in n.item_e]
    assert found
    with caplog.at_level(logging.WARNING, logger="flysafelz.notam_area"):
        for parsed in found:
            area, note = build_area(parsed)
            assert area is None
            assert "Q-circle" in note


def test_the_ethiopia_text_cut_off_at_every_length_never_raises(corpus):
    parsed = next(n for n in corpus[BLUE_MILL_SHA] if "STRAIGHT LINES" in n.item_e and "ETHIOPIA" in n.item_e)
    text = parsed.item_e
    for length in range(len(text) + 1):
        area, note = build_area(notam(item_e=text[:length], circle=Circle(lat=parsed.centre_lat, lon=parsed.centre_lon, radius_nm=parsed.radius_nm)))
        assert area is None or isinstance(area, Circle | Polygon)
        assert note


def test_area_from_item_e_over_every_notam_never_raises_and_every_shape_is_inside_its_q_circle_slack(corpus):
    counts = Counter()
    for sha, notams in corpus.items():
        for parsed in notams:
            area = area_from_item_e(parsed)
            assert isinstance(area, Shape | Route | Broken | NoArea), (sha[:8], parsed.id)
            counts[type(area).__name__] += 1
            if isinstance(area, Broken):
                assert area.reason and "\n" not in area.reason, (sha[:8], parsed.id)
            if isinstance(area, Shape):
                assert build_area(parsed)[0] == area.area  # the same answer build_area gives
    print(f"area_from_item_e over the corpus: {dict(sorted(counts.items()))}")
    assert counts["Shape"] > 0 and counts["NoArea"] > counts["Shape"]


def test_the_ethiopia_polygon_is_broken_not_a_shape_so_the_q_circle_decides(corpus):
    found = [n for n in corpus[BLUE_MILL_SHA] if "STRAIGHT LINES" in n.item_e and "ETHIOPIA" in n.item_e]
    assert found and all(isinstance(area_from_item_e(n), Broken) for n in found)


def test_h7171_is_a_shape_a_circle_of_half_a_mile(corpus):
    parsed = next(n for n in corpus[BLUE_MILL_SHA] if n.id == "H7171/26")
    area = area_from_item_e(parsed)
    assert isinstance(area, Shape) and area.kind == "circle" and area.area.radius_nm == 0.5
