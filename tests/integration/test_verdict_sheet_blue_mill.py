"""GP2 / GP3 / E1 / G1 / G5: the blind Blue Mill entry, built offline from the committed fixture case.

build_blind_entry() runs over fixtures/blue_mill_2026-10-31/ and its rows are compared with tests.cases.run(),
the assess() call every other Blue Mill test makes. No network, no key.
"""

from pathlib import Path

import cases
import yaml

from flysafelz.contracts import NotamAction, Tier
from flysafelz.geo import circle_edge_distance_nm
from flysafelz.verdict_sheet import build_blind_entry, read_examples

GP1 = Path(__file__).resolve().parents[1] / "data" / "verdict_sheet" / "examples_blue_mill.yaml"


def stamp(t):
    return t.strftime("%Y-%m-%dT%H:%M:%SZ")


def blue_mill_entry():
    (example,) = read_examples(GP1)
    return build_blind_entry(example, cases.case_dir(cases.BLUE_MILL))


def test_gp2_one_blind_entry_with_13_rows_and_no_label_anywhere():
    entry = blue_mill_entry()
    assert entry["id"] == "blue_mill_2026-10-31" and entry["mode"] == "blind" and entry["actions_done"] is False
    assert entry["window"] == {"start": "2026-10-31T07:00:00Z", "end": "2026-10-31T18:00:00Z"}
    assert set(entry) == {"id", "mode", "window", "actions_done", "notam_actions", "rows"}
    rows = entry["rows"]
    assert len(rows) == 13
    assert [(r["start"][11:16], r["end"][11:16]) for r in rows] == [
        *[(f"{h:02d}:00", f"{h + 1:02d}:00") for h in range(7, 16)],
        ("16:00", "16:28"),
        ("16:28", "17:00"),
        ("17:00", "17:06"),
        ("17:06", "18:00"),
    ]
    for row in rows:
        assert set(row) == {"start", "end", "edge", "verdict", "approaching_limits"}
        assert row["verdict"] is None and row["approaching_limits"] is None


def test_gp2_the_notam_list_is_every_notam_within_2_nm_with_a_null_value():
    entry = blue_mill_entry()
    _case, _forecast, _airspace, pib, _readings = cases.load(cases.BLUE_MILL)
    near = sorted(n.id for n in pib.notams if circle_edge_distance_nm(53.802797, -0.086644, n.centre_lat, n.centre_lon, n.radius_nm) <= 2)
    assert len(entry["notam_actions"]) == 20  # the committed fixture gives the 20 the plan expects
    assert list(entry["notam_actions"]) == near
    assert all(value is None for value in entry["notam_actions"].values())
    assert "J3864/26" in entry["notam_actions"]  # dropped by prefilter for starting after the window, but still within 2 NM
    assert len(pib.notams) > 20  # the 16 further away are not listed


def test_gp2_no_unread_tag_no_comment_no_reason_and_no_action_text_anywhere():
    text = yaml.safe_dump(blue_mill_entry(), sort_keys=False)
    assert "unread" not in text.lower() and "#" not in text and "why" not in text.lower()
    for action in NotamAction:
        assert action.value not in text and action.name not in text


def test_gp3_the_edge_list_from_the_committed_fixture():
    edges = [(r["start"][11:16], r["edge"]) for r in blue_mill_entry()["rows"]]
    assert edges == [
        ("07:00", "window start"),
        ("08:00", "H7171/26 starts + J3857/26 starts"),
        *[(f"{h:02d}:00", "hour") for h in range(9, 17)],
        ("16:28", "sunset"),
        ("17:00", "hour"),
        ("17:06", "civil dusk"),
    ]


def test_gp3_the_08_00_edge_is_the_fixtures_own_two_notams_and_no_other_notam_has_an_edge_in_the_window():
    _case, _forecast, _airspace, pib, _readings = cases.load(cases.BLUE_MILL)
    by_id = {n.id: n for n in pib.notams}
    assert by_id["J3857/26"].valid_from.strftime("%H:%M") == "08:00"  # B) 08:00 on 31 Oct
    assert by_id["H7171/26"].item_d == "0800-1800"  # D) schedule: active from 08:00 each day
    start, end = cases.load(cases.BLUE_MILL)[0].window.start, cases.load(cases.BLUE_MILL)[0].window.end
    listed = blue_mill_entry()["notam_actions"]  # J3858/26 also starts in the window, but it is further than 2 NM away
    inside = [n.id for n in pib.notams if n.id in listed and (start < n.valid_from < end or (n.valid_to is not None and start < n.valid_to < end))]
    assert inside == ["J3857/26"]  # the only B or C edge inside the window; the other 08:00 edge is H7171's schedule


def test_e1_the_17_00_hour_splits_at_civil_dusk_17_06():
    rows = blue_mill_entry()["rows"]
    assert [(r["start"][11:16], r["end"][11:16], r["edge"]) for r in rows if r["start"][11:13] == "17"] == [("17:00", "17:06", "hour"), ("17:06", "18:00", "civil dusk")]


def test_g1_the_rows_are_exactly_the_segments_of_assess_and_no_hour_is_missing():
    assessment = cases.run(cases.BLUE_MILL)
    rows = iter(blue_mill_entry()["rows"])
    count = 0
    for hour in assessment.hours:
        assert hour.tier is Tier.VERDICT
        for segment in hour.segments:
            row = next(rows)
            assert (row["start"], row["end"]) == (stamp(segment.start), stamp(segment.end))
            count += 1
    assert next(rows, None) is None and count == 13


def test_g1_the_rows_are_contiguous_and_cover_the_assess_hour_span():
    assessment = cases.run(cases.BLUE_MILL)
    rows = blue_mill_entry()["rows"]
    assert rows[0]["start"] == stamp(assessment.hours[0].hour)
    assert rows[-1]["end"] == stamp(assessment.hours[-1].hour.replace(hour=assessment.hours[-1].hour.hour + 1))
    assert all(a["end"] == b["start"] for a, b in zip(rows, rows[1:]))
