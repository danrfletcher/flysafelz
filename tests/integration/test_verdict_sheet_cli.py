"""The command: GP2 output, E4, E5, E6, E10, E12, E14, E16, G5, G8, G9, the existing --out refusal, and F2.

Two ways in. A subprocess runs scripts/build_verdict_sheet.py itself over the committed Blue Mill case (the
script finds fixtures/ from its own location, G13). Everything that needs a different fixtures/ folder calls
verdict_sheet.main() in-process with root=tmp_path and synthetic cases written by tests/verdict_sheet_helpers.py.
Capture is never called for real: it is monkeypatched, and --disable-socket stays on.
"""

import itertools
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import cases
import pytest
import yaml
from factories import circle_at, notam
from verdict_sheet_helpers import example, examples_file, write_case

from flysafelz import capture
from flysafelz.capture import CaptureError
from flysafelz.verdict_sheet import main

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts" / "build_verdict_sheet.py"
GP1 = REPO / "tests" / "data" / "verdict_sheet" / "examples_blue_mill.yaml"
NOW = datetime(2026, 10, 31, 7, 30, tzinfo=UTC)
FAKE_KEY = "fake-key-do-not-print-0123456789"


def utc(day, hour, minute=0, month=10):
    return datetime(2026, month, day, hour, minute, tzinfo=UTC)


def run_script(*args, cwd=None):
    return subprocess.run([sys.executable, str(SCRIPT), *args], capture_output=True, text=True, cwd=cwd, check=False)


def sheet(path):
    return yaml.safe_load(path.read_text(encoding="utf-8"))


@pytest.fixture
def gp1(tmp_path):
    path = tmp_path / "examples.yaml"
    shutil.copy(GP1, path)
    return path


def blue_mill_root(tmp_path):
    """tmp_path/fixtures/blue_mill_2026-10-31 as a copy of the committed case."""
    shutil.copytree(cases.case_dir(cases.BLUE_MILL), tmp_path / "fixtures" / cases.BLUE_MILL)
    return tmp_path


def call(tmp_path, *examples, root=None, environ=None, extra=()):
    path = examples_file(tmp_path / "examples.yaml", *examples)
    return main(["--blind", "--examples", str(path), "--out", str(tmp_path / "sheet.yaml"), *extra], root=root or tmp_path, environ=environ or {})


# --- the script itself, over the committed Blue Mill case -------------------------------------------------------


def test_gp2_the_script_writes_one_blind_entry_with_13_rows_and_20_null_notam_actions(gp1, tmp_path):
    out = tmp_path / "verdict_sheet.yaml"
    result = run_script("--blind", "--examples", str(gp1), "--out", str(out), cwd=tmp_path)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == f"wrote {out}: 1 new, 0 unchanged, 0 stale, 0 skipped, 0 failed" and result.stderr == ""
    (entry,) = sheet(out)
    assert (entry["id"], entry["mode"], entry["actions_done"]) == ("blue_mill_2026-10-31", "blind", False)
    assert len(entry["rows"]) == 13
    assert all(r["verdict"] is None and r["approaching_limits"] is None for r in entry["rows"])
    assert len(entry["notam_actions"]) == 20 and all(v is None for v in entry["notam_actions"].values())
    text = out.read_text(encoding="utf-8")
    assert "unread" not in text and "#" not in text


def test_the_script_finds_fixtures_from_its_own_location_not_the_working_directory(gp1, tmp_path):
    elsewhere = tmp_path / "somewhere" / "else"
    elsewhere.mkdir(parents=True)
    result = run_script("--blind", "--examples", str(gp1), "--out", str(elsewhere / "out.yaml"), cwd=elsewhere)
    assert result.returncode == 0, result.stderr
    assert (elsewhere / "out.yaml").exists()


def test_f2_the_examples_file_keeps_its_bytes_and_mtime_after_a_build(gp1, tmp_path):
    before = (gp1.read_bytes(), gp1.stat().st_mtime_ns)
    assert run_script("--blind", "--examples", str(gp1), "--out", str(tmp_path / "out.yaml")).returncode == 0
    assert (gp1.read_bytes(), gp1.stat().st_mtime_ns) == before


def test_an_existing_out_gets_the_new_entry_appended_and_its_own_bytes_are_unchanged(gp1, tmp_path):
    out = tmp_path / "verdict_sheet.yaml"
    before = b"- id: Dan's labels\n  mode: blind\n  rows: []  # his own comment\n"
    out.write_bytes(before)
    result = run_script("--blind", "--examples", str(gp1), "--out", str(out))
    assert result.stdout.strip() == f"wrote {out}: 1 new, 0 unchanged, 0 stale, 0 skipped, 0 failed"
    assert out.read_bytes().startswith(before)
    assert [e["id"] for e in sheet(out)] == ["Dan's labels", "blue_mill_2026-10-31"]


def test_e4_two_examples_with_the_same_id_exit_1_naming_the_id_and_create_no_sheet(tmp_path):
    path = examples_file(tmp_path / "examples.yaml", example("twin", cases.BLUE_MILL, now=NOW, window=(utc(31, 7), utc(31, 18))), example("twin", cases.BLUE_MILL, now=NOW, window=None))
    out = tmp_path / "out.yaml"
    result = run_script("--blind", "--examples", str(path), "--out", str(out))
    assert result.returncode == 1 and "twin" in result.stderr and result.stderr.count("\n") == 1
    assert not out.exists()


def test_a_missing_id_and_an_unreadable_examples_file_exit_1_with_one_line_and_write_nothing(tmp_path):
    no_id = example("x", cases.BLUE_MILL, now=NOW, window=None)
    del no_id["id"]
    path = examples_file(tmp_path / "examples.yaml", no_id)
    out = tmp_path / "out.yaml"
    result = run_script("--blind", "--examples", str(path), "--out", str(out))
    assert (result.returncode, result.stderr) == (1, "example #1: missing id\n") and not out.exists()
    result = run_script("--blind", "--examples", str(tmp_path / "absent.yaml"), "--out", str(out))
    assert result.returncode == 1 and "absent.yaml" in result.stderr and result.stderr.count("\n") == 1 and not out.exists()


def test_an_unknown_drop_value_exits_1_naming_the_example_and_writes_nothing(tmp_path):
    path = examples_file(tmp_path / "examples.yaml", example("dropper", cases.BLUE_MILL, now=NOW, window=None, drop=["radar"]))
    out = tmp_path / "out.yaml"
    result = run_script("--blind", "--examples", str(path), "--out", str(out))
    assert result.returncode == 1 and "dropper" in result.stderr and "'radar'" in result.stderr and not out.exists()


def test_no_flag_builds_each_example_in_its_own_mode_and_the_flags_pick_one_mode(gp1, tmp_path):
    unmarked = examples_file(tmp_path / "unmarked.yaml", example("open", cases.BLUE_MILL, now=NOW, window=(utc(31, 7), utc(31, 18)), blind=False))
    out = tmp_path / "out.yaml"
    result = run_script("--examples", str(unmarked), "--out", str(out))
    assert result.returncode == 0, result.stderr
    assert [(e["id"], e["mode"]) for e in sheet(out)] == [("open", "draft")]
    blind_only = run_script("--blind", "--examples", str(unmarked), "--out", str(tmp_path / "b.yaml"))
    assert blind_only.returncode == 0 and blind_only.stdout.strip() == "nothing written: 0 new, 0 unchanged, 0 stale, 1 skipped, 0 failed" and not (tmp_path / "b.yaml").exists()
    draft_only = run_script("--draft", "--examples", str(gp1), "--out", str(tmp_path / "d.yaml"))
    assert draft_only.returncode == 0 and "0 new, 0 unchanged, 0 stale, 1 skipped" in draft_only.stdout and not (tmp_path / "d.yaml").exists()
    marked = run_script("--examples", str(gp1), "--out", str(tmp_path / "m.yaml"))
    assert marked.returncode == 0, marked.stderr
    assert [e["mode"] for e in sheet(tmp_path / "m.yaml")] == ["blind"]


def test_blind_and_draft_together_are_a_usage_error(gp1, tmp_path):
    assert run_script("--blind", "--draft", "--examples", str(gp1), "--out", str(tmp_path / "o.yaml")).returncode == 2


# --- skipped and failed examples, in-process --------------------------------------------------------------


def test_e5_an_explain_example_is_skipped_and_the_summary_counts_it(tmp_path, capsys):
    root = blue_mill_root(tmp_path)
    explain = {"id": "why", "reference": {"path": "explain"}}
    code = call(tmp_path, example("bm", cases.BLUE_MILL, now=NOW, window=(utc(31, 7), utc(31, 18))), explain, root=root)
    assert code == 0
    assert [e["id"] for e in sheet(tmp_path / "sheet.yaml")] == ["bm"]
    assert capsys.readouterr().out.strip() == f"wrote {tmp_path / 'sheet.yaml'}: 1 new, 0 unchanged, 0 stale, 1 skipped, 0 failed"


def test_e5_only_explain_examples_write_nothing_and_exit_0(tmp_path, capsys):
    assert call(tmp_path, {"id": "why", "reference": {"path": "explain"}}) == 0
    assert not (tmp_path / "sheet.yaml").exists()
    assert capsys.readouterr().out.strip() == "nothing written: 0 new, 0 unchanged, 0 stale, 1 skipped, 0 failed"


def test_e6_a_capture_failure_skips_that_example_with_one_line_and_the_rest_are_written(tmp_path, capsys, monkeypatch):
    root = blue_mill_root(tmp_path)
    attempts = []

    def fails(name, case_dir, key, client, *, today):
        attempts.append((name, key, (case_dir / "case.yaml").read_text(encoding="utf-8")))
        raise CaptureError(f"openAIP: the key was rejected (HTTP 403); check OPENAIP_API_KEY (sent {key})")

    monkeypatch.setattr(capture, "capture", fails)
    day = datetime(2026, 11, 1, 7, 30, tzinfo=UTC)
    code = call(
        tmp_path,
        example("lost", "never_captured", now=day, window=(day.replace(hour=7), day.replace(hour=12))),
        example("bm", cases.BLUE_MILL, now=NOW, window=(utc(31, 7), utc(31, 18))),
        root=root,
        environ={"OPENAIP_API_KEY": FAKE_KEY},
    )
    out, err = capsys.readouterr()
    assert code == 0
    assert [e["id"] for e in sheet(tmp_path / "sheet.yaml")] == ["bm"]
    assert out.strip().endswith("1 new, 0 unchanged, 0 stale, 0 skipped, 1 failed")
    assert err.count("\n") == 1 and err.startswith("example lost: openAIP: the key was rejected (HTTP 403)")
    assert FAKE_KEY not in out + err and (tmp_path / "sheet.yaml").read_text(encoding="utf-8").count(FAKE_KEY) == 0
    assert [a[0] for a in attempts] == ["never_captured"] and "lat: 53.802797" in attempts[0][2] and "days" not in attempts[0][2]
    assert not (root / "fixtures" / "never_captured").exists()  # the folder made for the attempt is taken back


def test_e6_with_no_key_the_example_is_skipped_and_nothing_is_captured(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(capture, "capture", lambda *a, **k: pytest.fail("capture was called without a key"))
    day = datetime(2026, 11, 1, 7, 30, tzinfo=UTC)
    code = call(tmp_path, example("lost", "never_captured", now=day, window=None))
    out, err = capsys.readouterr()
    assert code == 1 and out.strip() == "nothing written: 0 new, 0 unchanged, 0 stale, 0 skipped, 1 failed"
    assert err == "example lost: OPENAIP_API_KEY is not set (add it to .env)\n"
    assert not (tmp_path / "sheet.yaml").exists() and not (tmp_path / "fixtures" / "never_captured").exists()


def test_a_missing_case_gets_a_case_yaml_and_one_capture_and_then_builds(tmp_path, monkeypatch):
    """A fake capture writes the data files the real one would; the case.yaml it is handed is the one the script wrote."""
    root = tmp_path
    window = (utc(31, 7), utc(31, 12))
    seen = {}

    def fake_capture(name, case_dir, key, client, *, today):
        seen["case_yaml"] = yaml.safe_load((case_dir / "case.yaml").read_text(encoding="utf-8"))
        seen["key"] = key
        built = write_case(tmp_path / "donor", name, now=NOW, window=window)
        for file in built.iterdir():
            if file.name != "case.yaml":
                shutil.copy(file, case_dir / file.name)
        return []

    monkeypatch.setattr(capture, "capture", fake_capture)
    assert call(tmp_path, example("new", "new_case", now=NOW, window=window), root=root, environ={"OPENAIP_API_KEY": FAKE_KEY}) == 0
    assert seen["case_yaml"] == {"lat": 53.802797, "lon": -0.086644, "now": "2026-10-31T07:30:00Z", "window": {"start": "2026-10-31T07:00:00Z", "end": "2026-10-31T12:00:00Z"}}
    assert seen["key"] == FAKE_KEY
    (entry,) = sheet(tmp_path / "sheet.yaml")
    assert entry["id"] == "new" and entry["rows"][0]["edge"] == "window start"


def test_a_multi_day_example_with_no_case_is_captured_with_days_and_a_window_inside_them(tmp_path, monkeypatch):
    seen = {}

    def stop(name, case_dir, key, client, *, today):
        seen["case_yaml"] = yaml.safe_load((case_dir / "case.yaml").read_text(encoding="utf-8"))
        raise CaptureError("stop here")

    monkeypatch.setattr(capture, "capture", stop)
    assert call(tmp_path, example("long", "long_case", now=NOW, window=None), environ={"OPENAIP_API_KEY": FAKE_KEY}) == 1
    assert seen["case_yaml"]["days"] == 2  # the default window runs to the end of tomorrow in London
    assert seen["case_yaml"]["window"] == {"start": "2026-10-31T07:00:00Z", "end": "2026-11-02T00:00:00Z"}


def test_an_existing_case_is_never_captured_again(tmp_path, monkeypatch):
    root = blue_mill_root(tmp_path)
    monkeypatch.setattr(capture, "capture", lambda *a, **k: pytest.fail("an existing case was recaptured"))
    before = {p.name: p.read_bytes() for p in (root / "fixtures" / cases.BLUE_MILL).iterdir()}
    assert call(tmp_path, example("bm", cases.BLUE_MILL, now=NOW, window=(utc(31, 7), utc(31, 18))), root=root, environ={"OPENAIP_API_KEY": FAKE_KEY}) == 0
    assert {p.name: p.read_bytes() for p in (root / "fixtures" / cases.BLUE_MILL).iterdir()} == before


def test_e14_a_default_window_on_single_day_blue_mill_names_the_example_and_the_missing_day(tmp_path, capsys):
    root = blue_mill_root(tmp_path)
    code = call(tmp_path, example("bm_default", cases.BLUE_MILL, now=NOW, window=None), root=root)
    out, err = capsys.readouterr()
    assert code == 1 and not (tmp_path / "sheet.yaml").exists()
    assert err.count("\n") == 1 and err.startswith("example bm_default:") and "London day 2026-11-01" in err
    assert "blue_mill_2026-10-31_2day" in err and "days: 2" in err
    assert out.strip() == "nothing written: 0 new, 0 unchanged, 0 stale, 0 skipped, 1 failed"


def test_e14_e16_a_failing_example_gets_no_entry_and_the_good_one_is_still_written(tmp_path, capsys):
    root = blue_mill_root(tmp_path)
    code = call(
        tmp_path,
        example("late", cases.BLUE_MILL, now=utc(31, 9, 30), window=(utc(31, 7), utc(31, 18))),
        example("bm_default", cases.BLUE_MILL, now=NOW, window=None),
        example("bm", cases.BLUE_MILL, now=NOW, window=(utc(31, 7), utc(31, 18))),
        root=root,
    )
    out, err = capsys.readouterr()
    assert code == 0
    assert [e["id"] for e in sheet(tmp_path / "sheet.yaml")] == ["bm"]
    assert "example late: inputs.now is 2026-10-31T09:30:00Z but case blue_mill_2026-10-31 has 2026-10-31T07:30:00Z" in err
    assert "example bm_default:" in err and out.strip().endswith("1 new, 0 unchanged, 0 stale, 0 skipped, 2 failed")


# --- synthetic cases ---------------------------------------------------------------------------------------------


def rows_of(entry):
    return [(r["start"][5:16], r["end"][5:16], r["edge"]) for r in entry["rows"]]


def test_e10_a_factory_notam_starting_exactly_on_the_hour_adds_no_cut_and_names_that_row(tmp_path):
    x1 = notam("X1/26", circle=circle_at(0), valid_from=utc(31, 9), valid_to=utc(31, 13))
    write_case(tmp_path, "e10", now=NOW, window=(utc(31, 8), utc(31, 11)), notams=(x1,))
    assert call(tmp_path, example("e10", "e10", now=NOW, window=(utc(31, 8), utc(31, 11)))) == 0
    (entry,) = sheet(tmp_path / "sheet.yaml")
    assert rows_of(entry) == [("10-31T08:00", "10-31T09:00", "window start"), ("10-31T09:00", "10-31T10:00", "X1/26 starts"), ("10-31T10:00", "10-31T11:00", "hour")]
    assert entry["notam_actions"] == {"X1/26": None}


def test_e11_a_window_starting_at_07_30_gives_a_first_row_from_07_00_named_window_start(tmp_path):
    write_case(tmp_path, "e11", now=NOW, window=(utc(31, 7, 30), utc(31, 9)))
    assert call(tmp_path, example("e11", "e11", now=NOW, window=(utc(31, 7, 30), utc(31, 9)))) == 0
    (entry,) = sheet(tmp_path / "sheet.yaml")
    assert rows_of(entry)[0] == ("10-31T07:00", "10-31T08:00", "window start")
    assert entry["window"] == {"start": "2026-10-31T07:30:00Z", "end": "2026-10-31T09:00:00Z"}


def test_e12_g9_no_window_on_a_two_day_case_runs_from_nows_hour_to_the_end_of_tomorrow_and_says_default(tmp_path):
    write_case(tmp_path, "two_day", now=NOW, window=(utc(31, 7), utc(1, 0, month=11) + (utc(2, 0, month=11) - utc(1, 0, month=11))), days=2)
    assert call(tmp_path, example("two", "two_day", now=NOW, window=None)) == 0
    (entry,) = sheet(tmp_path / "sheet.yaml")
    assert entry["window"] == "default"
    rows = entry["rows"]
    assert rows[0]["start"] == "2026-10-31T07:00:00Z" and rows[0]["edge"] == "window start"
    assert rows[-1]["end"] == "2026-11-02T00:00:00Z"  # the end of 1 Nov in London (GMT)
    assert all(a["end"] == b["start"] for a, b in itertools.pairwise(rows))
    assert {r["start"][:13] for r in rows if r["start"].endswith(":00:00Z")} == {f"2026-10-31T{h:02d}" for h in range(7, 24)} | {f"2026-11-01T{h:02d}" for h in range(24)}
    assert all(set(r) == {"start", "end", "edge", "verdict", "approaching_limits"} for r in rows)


def test_g5_outlook_and_too_far_hours_are_one_tier_only_row_each_with_no_label_keys(tmp_path):
    window = (utc(3, 20, month=11), utc(12, 0, month=11))
    write_case(tmp_path, "far", now=NOW, window=(utc(31, 7), utc(4, 0, month=11)), days=4)
    assert call(tmp_path, example("far", "far", now=NOW, window=window)) == 0
    (entry,) = sheet(tmp_path / "sheet.yaml")
    rows = entry["rows"]
    labelled = [r for r in rows if "tier" not in r]
    tiered = [r for r in rows if "tier" in r]
    assert [r["start"][11:13] for r in labelled if r["start"].startswith("2026-11-03")][:1] == ["20"]
    assert all(r["start"] < "2026-11-04" for r in labelled) and all(r["start"] >= "2026-11-04" for r in tiered)
    for row in tiered:
        assert set(row) == {"start", "end", "edge", "tier"} and row["edge"] == "hour"
        assert row["tier"] == ("OUTLOOK" if row["start"] < "2026-11-08" else "TOO_FAR")
    assert len([r for r in tiered if r["tier"] == "OUTLOOK"]) == 96 and len([r for r in tiered if r["tier"] == "TOO_FAR"]) == 96
    assert all(r["verdict"] is None and r["approaching_limits"] is None for r in labelled)


def test_g5_a_window_that_starts_in_the_outlook_makes_its_first_row_window_start(tmp_path):
    write_case(tmp_path, "far2", now=NOW, window=(utc(31, 7), utc(4, 0, month=11)), days=4)
    window = (utc(5, 10, month=11), utc(5, 12, month=11))
    assert call(tmp_path, example("far2", "far2", now=NOW, window=window)) == 0
    (entry,) = sheet(tmp_path / "sheet.yaml")
    assert [(r["edge"], r["tier"]) for r in entry["rows"]] == [("window start", "OUTLOOK"), ("hour", "OUTLOOK")]


@pytest.mark.parametrize("source", ["weather", "airspace", "pib"])
def test_g8_a_dropped_source_still_gives_rows_from_assess_and_no_verdicts(tmp_path, source):
    x1 = notam("X1/26", circle=circle_at(0), valid_from=utc(31, 9), valid_to=utc(31, 13))
    write_case(tmp_path, "d", now=NOW, window=(utc(31, 8), utc(31, 11)), notams=(x1,))
    assert call(tmp_path, example("d", "d", now=NOW, window=(utc(31, 8), utc(31, 11)), drop=[source])) == 0
    (entry,) = sheet(tmp_path / "sheet.yaml")
    assert len(entry["rows"]) == 3 and all(r["verdict"] is None and r["approaching_limits"] is None for r in entry["rows"])
    assert entry["notam_actions"] == ({} if source == "pib" else {"X1/26": None})
    assert (entry["rows"][1]["edge"] == "hour") == (source == "pib")  # the dropped PIB has no NOTAM to start at 09:00


def test_g8_a_dropped_weather_needs_no_weather_coverage(tmp_path):
    write_case(tmp_path, "short", now=NOW, window=(utc(31, 8), utc(31, 11)))
    assert call(tmp_path, example("short", "short", now=NOW, window=None, drop=["weather"])) == 0
    assert call(tmp_path, example("short2", "short", now=NOW, window=None), extra=("--out", str(tmp_path / "second.yaml"))) == 1
