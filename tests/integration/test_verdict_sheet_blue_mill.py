"""GP2 / GP3 / E1 / G1 / G5: the blind Blue Mill entry, built offline from the committed fixture case.

build_blind_entry() runs over fixtures/blue_mill_2026-10-31/ and its rows are compared with tests.cases.run(),
the assess() call every other Blue Mill test makes. No network, no key.

The last section (PR-5.F1, E3) reruns the command over a tmp_path copy of the case: build, rerun, edit a fixture file,
`--check`. The examples file and the sheet live under tmp_path; the committed fixtures are never touched.
"""

import shutil
from datetime import UTC, datetime
from pathlib import Path

import cases
import pytest
import yaml
from verdict_sheet_helpers import example, examples_file

from flysafelz.contracts import NotamAction, Tier
from flysafelz.geo import circle_edge_distance_nm
from flysafelz.verdict_sheet import build_blind_entry, main, read_examples
from flysafelz.verdict_sheet_check import parse_sheet

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


# --- PR-5.F1, E3: a rerun after the fixture files change ------------------------------------------------------------------

NOW = datetime(2026, 10, 31, 7, 30, tzinfo=UTC)
WINDOW = (datetime(2026, 10, 31, 7, tzinfo=UTC), datetime(2026, 10, 31, 18, tzinfo=UTC))
READINGS = "readings.yaml"


class Run:
    """verdict_sheet.main() over a tmp_path copy of the committed Blue Mill case, one example."""

    def __init__(self, tmp_path, capsys, *, blind):
        self.root, self.capsys, self.blind = tmp_path, capsys, blind
        self.case = tmp_path / "fixtures" / cases.BLUE_MILL
        shutil.copytree(cases.case_dir(cases.BLUE_MILL), self.case)
        self.sheet = tmp_path / "sheet.yaml"

    def go(self, *flags, window=WINDOW):
        path = examples_file(self.root / "examples.yaml", example("bm", cases.BLUE_MILL, now=NOW, window=window, blind=self.blind))
        code = main([*flags, "--examples", str(path), "--out", str(self.sheet)], root=self.root, environ={})
        out, err = self.capsys.readouterr()
        return code, out, err

    def edit(self, name, change):
        path = self.case / name
        path.write_text(change(path.read_text(encoding="utf-8")), encoding="utf-8")

    def entry_span(self):
        (entry,) = parse_sheet(self.sheet.read_text(encoding="utf-8"), self.sheet)
        return entry.span.encode("utf-8")


def committed_bytes():
    return {p.name: p.read_bytes() for p in cases.case_dir(cases.BLUE_MILL).iterdir()}


def test_e3_blind_build_then_rerun_then_a_readings_edit_that_moves_no_row_edge_is_not_stale(tmp_path, capsys):
    run, committed = Run(tmp_path, capsys, blind=True), committed_bytes()
    assert run.go()[0] == 0
    built = run.sheet.read_bytes()
    code, out, _err = run.go()
    assert code == 0 and "0 new, 1 unchanged, 0 stale" in out and run.sheet.read_bytes() == built
    run.edit(READINGS, lambda t: t.replace("activity: UAS BVLOS ops", "activity: drone flights, BVLOS", 1))  # the same schedule: no edge moves
    assert run.go()[2] == ""
    code, out, _err = run.go("--check")
    assert code == 0 and out.splitlines()[0].endswith("stale no") and "0 stale" in out
    assert "0 new, 1 unchanged, 0 stale" in run.go()[1] and run.sheet.read_bytes() == built
    assert committed_bytes() == committed


def test_e3_blind_a_readings_edit_that_moves_a_row_edge_is_stale_rows_and_the_entry_is_untouched(tmp_path, capsys):
    run = Run(tmp_path, capsys, blind=True)
    assert run.go()[0] == 0
    built, span = run.sheet.read_bytes(), run.entry_span()
    run.edit(READINGS, lambda t: t.replace("schedule: 0800-1800", "schedule: 0800-1730", 1))  # J3857/26 now ends at 17:30: a new cut in the 17:00 hour
    code, out, _err = run.go("--check")
    assert code == 0 and out.splitlines()[0].endswith("stale (rows)") and out.splitlines()[-1].endswith("1 stale, 0 orphans")
    code, out, _err = run.go()
    assert code == 0 and "example bm: stale (rows); its entry is left as it is" in out and "0 new, 0 unchanged, 1 stale" in out
    assert run.sheet.read_bytes() == built and run.entry_span() == span


@pytest.mark.parametrize("name", ["weather.json", "airspace.json", "pib.xml"])
def test_e3_blind_an_edit_to_another_fixture_file_that_changes_no_row_is_not_stale(tmp_path, capsys, name):
    run = Run(tmp_path, capsys, blind=True)
    run.go()
    run.edit(name, lambda t: t + "\n")
    assert run.go("--check")[1].splitlines()[0].endswith("stale no")


def test_e3_blind_a_changed_window_is_stale_inputs_on_blue_mill(tmp_path, capsys):
    run = Run(tmp_path, capsys, blind=True)
    run.go()
    built = run.sheet.read_bytes()
    shorter = (WINDOW[0], WINDOW[1].replace(hour=16))
    code, out, _err = run.go("--check", window=shorter)
    assert code == 0 and out.splitlines()[0].endswith("stale (inputs)")
    assert "example bm: stale (inputs); its entry is left as it is" in run.go(window=shorter)[1] and run.sheet.read_bytes() == built


@pytest.mark.parametrize("name", ["weather.json", "airspace.json", "pib.xml", READINGS])
def test_e3_draft_an_edit_to_any_of_the_four_fixture_files_is_stale_and_names_that_file(tmp_path, capsys, name):
    run, committed = Run(tmp_path, capsys, blind=False), committed_bytes()
    assert run.go()[0] == 0
    built, span = run.sheet.read_bytes(), run.entry_span()
    assert run.go("--check")[1].splitlines()[0].endswith("stale no")
    run.edit(name, lambda t: t + ("# touched\n" if name == READINGS else "\n"))  # one byte more; a draft hashes the raw bytes
    code, out, _err = run.go("--check")
    assert code == 0 and out.splitlines()[0].endswith(f"stale ({name})"), out
    code, out, _err = run.go()
    assert code == 0 and f"example bm: stale ({name}); its entry is left as it is" in out
    assert run.sheet.read_bytes() == built and run.entry_span() == span
    assert committed_bytes() == committed


def test_e3_draft_two_files_changed_are_both_named_in_fingerprint_order(tmp_path, capsys):
    run = Run(tmp_path, capsys, blind=False)
    run.go()
    run.edit("pib.xml", lambda t: t + "\n")
    run.edit("weather.json", lambda t: t + "\n")
    assert run.go("--check")[1].splitlines()[0].endswith("stale (weather.json, pib.xml)")


def test_e3_draft_a_readings_edit_that_also_moves_an_edge_names_rows_and_the_file(tmp_path, capsys):
    run = Run(tmp_path, capsys, blind=False)
    run.go()
    run.edit(READINGS, lambda t: t.replace("schedule: 0800-1800", "schedule: 0800-1730", 1))
    assert run.go("--check")[1].splitlines()[0].endswith("stale (rows, readings.yaml)")
