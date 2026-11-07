"""G10 / GP5: validation shared by `--check` and load_sheet, and the `--check` command itself (output columns, exit codes).

The rule table calls check_entry() on hand-built entries parsed from YAML text. The command tests build a real
sheet with verdict_sheet.main() over a synthetic case in tmp_path, edit its text like Dan would, and run `--check`
in-process (main(argv)) and as a subprocess (the script, over the committed Blue Mill case).
"""

import copy
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

from flysafelz.contracts import NotamAction, Verdict
from flysafelz.verdict_sheet import main
from flysafelz.verdict_sheet_check import SheetError, check_entries, check_entry, parse_sheet, report_line

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts" / "build_verdict_sheet.py"
GP1 = REPO / "tests" / "data" / "verdict_sheet" / "examples_blue_mill.yaml"
NOW = datetime(2026, 10, 31, 7, 30, tzinfo=UTC)


def utc(hour, minute=0, day=31):
    return datetime(2026, 10, day, hour, minute, tzinfo=UTC)


def row(start, end, **labels):
    return {"start": f"2026-10-31T{start}:00Z", "end": f"2026-10-31T{end}:00Z", "edge": "hour", "verdict": None, "approaching_limits": None, **labels}


def good_entry() -> dict:
    """A 07:30 window: assess starts at 07:00, so the rows run 07:00 to 09:00."""
    return {
        "id": "e1",
        "mode": "blind",
        "window": {"start": "2026-10-31T07:30:00Z", "end": "2026-10-31T09:00:00Z"},
        "actions_done": False,
        "notam_actions": {"A1/26": None},
        "rows": [row("07:00", "08:00"), row("08:00", "09:00")],
    }


def checked(entry: dict, default_span=None):
    (parsed,) = parse_sheet(yaml.safe_dump([entry], sort_keys=False), "sheet.yaml")
    return check_entry(parsed, default_span)


def edited(**changes) -> dict:
    entry = good_entry()
    entry.update(changes)
    return entry


def with_rows(*rows) -> dict:
    return edited(rows=list(rows))


def test_a_good_entry_has_no_error_and_a_07_30_window_starts_its_rows_at_07_00():
    report = checked(good_entry())
    assert report.errors == [] and report.rows == 2 and report.unlabelled == 2 and report.labelled == 0


# (what is wrong, the entry, words that every message about it must contain)
RULES = [
    ("verdict not in the enum", with_rows(row("07:00", "08:00", verdict="MAYBE", approaching_limits=False), row("08:00", "09:00")), ["example e1", "row 1", "2026-10-31T07:00:00Z", "'MAYBE'", "NO_GO"]),
    ("verdict in the wrong case", with_rows(row("07:00", "08:00"), row("08:00", "09:00", verdict="no_go", approaching_limits=False)), ["example e1", "row 2", "'no_go'"]),
    ("verdict is not text", with_rows(row("07:00", "08:00", verdict=["NO_GO"], approaching_limits=False), row("08:00", "09:00")), ["example e1", "row 1"]),
    ("approaching_limits is not a boolean", with_rows(row("07:00", "08:00", verdict="NO_GO", approaching_limits="yes"), row("08:00", "09:00")), ["example e1", "row 1", "approaching_limits", "'yes'"]),
    ("notam action not in the enum", edited(notam_actions={"A1/26": "AVOID"}), ["example e1", "NOTAM A1/26", "'AVOID'", "CONTACT_FIRST"]),
    ("a gap between rows", with_rows(row("07:00", "08:00"), row("08:10", "09:00")), ["example e1", "row 2", "2026-10-31T08:10:00Z", "gap"]),
    ("an overlap between rows", with_rows(row("07:00", "08:00"), row("07:50", "09:00")), ["example e1", "row 2", "overlap"]),
    ("first row starts after the window's hour", with_rows(row("07:30", "08:00"), row("08:00", "09:00")), ["example e1", "row 1", "2026-10-31T07:00:00Z"]),
    ("first row starts before the window's hour", with_rows(row("06:00", "08:00"), row("08:00", "09:00")), ["example e1", "row 1", "07:00"]),
    ("last row stops short of the window", with_rows(row("07:00", "08:00")), ["example e1", "row 1", "2026-10-31T09:00:00Z"]),
    ("last row runs past the window's hour", with_rows(row("07:00", "08:00"), row("08:00", "10:00")), ["example e1", "row 2", "2026-10-31T09:00:00Z"]),
    ("a row that ends before it starts", with_rows(row("07:00", "08:00"), row("09:00", "08:00")), ["example e1", "row 2"]),
    ("a time that is not a time", with_rows(row("07:00", "08:00"), {**row("08:00", "09:00"), "end": "soon"}), ["example e1", "row 2", "timezone"]),
    ("a row that is not a mapping", with_rows(row("07:00", "08:00"), "08:00-09:00"), ["example e1", "row 2"]),
    ("a tier that is neither OUTLOOK nor TOO_FAR", with_rows(row("07:00", "08:00"), {"start": "2026-10-31T08:00:00Z", "end": "2026-10-31T09:00:00Z", "tier": "VERDICT"}), ["example e1", "row 2", "tier"]),
    ("no rows", edited(rows=[]), ["example e1", "no rows"]),
    ("rows is not a list", edited(rows="none"), ["example e1", "rows"]),
    ("a window that is neither default nor a start and end", edited(window="today"), ["example e1", "window"]),
    ("actions_done that is not a boolean", edited(actions_done="yes"), ["example e1", "actions_done"]),
    ("notam_actions that is not a mapping", edited(notam_actions=["A1/26"]), ["example e1", "notam_actions"]),
]


@pytest.mark.parametrize("entry, words", [r[1:] for r in RULES], ids=[r[0] for r in RULES])
def test_g10_every_validation_rule_gives_an_error_that_names_the_example_and_the_row(entry, words):
    errors = checked(entry).errors
    assert errors, "no error"
    assert any(all(word in message for word in words) for message in errors), errors


def test_g10_the_enum_messages_list_exactly_the_values_of_contracts():
    verdict_error = checked(with_rows(row("07:00", "08:00", verdict="X", approaching_limits=False), row("08:00", "09:00"))).errors[0]
    assert all(v.value in verdict_error for v in Verdict)
    action_error = checked(edited(notam_actions={"A1/26": "X"})).errors[0]
    assert all(a.value in action_error for a in NotamAction)


def test_g10_a_null_verdict_and_every_enum_value_are_accepted():
    rows = [row("07:00", "08:00", verdict=v.value, approaching_limits=bool(i % 2)) for i, v in enumerate(Verdict)]
    rows = [{**r, "start": f"2026-10-31T{7 + i:02d}:00:00Z", "end": f"2026-10-31T{8 + i:02d}:00:00Z"} for i, r in enumerate(rows)]
    entry = edited(window={"start": "2026-10-31T07:00:00Z", "end": "2026-10-31T10:00:00Z"}, rows=rows, notam_actions={a.value: a.value for a in NotamAction} | {"N/26": None})
    report = checked(entry)
    assert report.errors == [] and report.labelled == 3


def test_check_sup_is_a_valid_notam_action_and_loads_as_itself():
    assert checked(edited(notam_actions={"A1/26": "CHECK_SUP"})).errors == []
    assert checked(edited(notam_actions={"A1/26": "CHECK_SUP"})).unread_left == 0
    assert "CHECK_SUP" in checked(edited(notam_actions={"A1/26": "X"})).errors[0]


@pytest.mark.parametrize("labels", [{"verdict": "NO_GO", "approaching_limits": None}, {"verdict": None, "approaching_limits": True}])
def test_g10_a_row_with_only_one_of_the_two_labels_is_incomplete_counted_and_not_an_error(labels):
    report = checked(with_rows(row("07:00", "08:00", **labels), row("08:00", "09:00")))
    assert report.errors == [] and (report.incomplete, report.labelled, report.unlabelled) == (1, 0, 2)


def test_gp5_the_counts_split_labelled_unlabelled_incomplete_and_tier_only_rows():
    far = {"start": "2026-10-31T09:00:00Z", "end": "2026-10-31T10:00:00Z", "edge": "hour", "tier": "OUTLOOK"}
    entry = edited(
        window={"start": "2026-10-31T07:30:00Z", "end": "2026-10-31T10:00:00Z"},
        rows=[row("07:00", "08:00", verdict="NO_GO", approaching_limits=False), row("08:00", "09:00", verdict="UNVERIFIED"), far],
    )
    report = checked(entry)
    assert report.errors == []
    assert (report.rows, report.tier_rows, report.labelled, report.unlabelled, report.incomplete) == (3, 1, 1, 1, 1)


def test_unread_left_counts_every_unread_over_lz_action_and_is_not_an_error():
    report = checked(edited(actions_done=True, notam_actions={"A1/26": "UNREAD_OVER_LZ", "B2/26": "UNREAD_OVER_LZ", "C3/26": "NO_FLY"}))
    assert (report.unread_left, report.errors, report.actions_done) == (2, [], True)


def test_a_default_window_is_checked_for_contiguity_alone_unless_the_caller_knows_the_default():
    entry = edited(window="default", rows=[row("07:00", "08:00"), row("08:00", "09:00")])
    assert checked(entry).errors == []
    assert checked(edited(window="default", rows=[row("07:00", "08:00"), row("08:30", "09:00")])).errors
    assert checked(entry, (utc(7), utc(9))).errors == []
    errors = checked(entry, (utc(7), utc(11))).errors
    assert len(errors) == 1 and "row 2" in errors[0] and "2026-10-31T11:00:00Z" in errors[0]


def test_times_a_yaml_editor_left_unquoted_are_read_as_times():
    text = "- id: e1\n  mode: blind\n  window: {start: 2026-10-31T07:30:00Z, end: 2026-10-31T09:00:00Z}\n  actions_done: false\n  rows:\n    - {start: 2026-10-31T07:00:00Z, end: 2026-10-31T08:00:00Z, verdict: null, approaching_limits: null}\n    - {start: 2026-10-31T08:00:00Z, end: 2026-10-31T09:00:00Z, verdict: null, approaching_limits: null}\n"
    (entry,) = parse_sheet(text, "sheet.yaml")
    assert check_entry(entry).errors == []


def test_an_entry_that_is_not_a_mapping_or_has_no_id_is_an_error_naming_its_place():
    entries = parse_sheet("- just text\n- mode: blind\n  rows: []\n", "sheet.yaml")
    reports = check_entries(entries)
    assert "entry #1 (line 1)" in reports[0].errors[0] and "must be a mapping" in reports[0].errors[0]
    assert any("entry #2 (line 2)" in e and "missing id" in e for e in reports[1].errors)


def test_two_entries_with_one_id_are_an_error_on_the_second():
    entries = parse_sheet(yaml.safe_dump([good_entry(), good_entry()], sort_keys=False), "sheet.yaml")
    first, second = check_entries(entries)
    assert first.errors == [] and "duplicate id" in second.errors[0] and "example e1" in second.errors[0]


def test_every_error_is_collected_not_just_the_first():
    bad = with_rows(row("07:00", "08:00", verdict="X", approaching_limits=False), row("08:20", "09:00", verdict="Y", approaching_limits=False))
    assert len(checked(bad).errors) >= 3


def test_the_check_does_not_change_the_entry_it_is_given():
    entry = with_rows(row("07:00", "08:00", verdict="X", approaching_limits=False), row("08:00", "09:00"))
    before = copy.deepcopy(entry)
    checked(entry)
    assert entry == before


# --- GP5: the `--check` command ---------------------------------------------------------------------------------


@pytest.fixture
def built(tmp_path):
    """A real sheet: one blind example with one NOTAM, built by main() over a synthetic case under tmp_path."""
    window = (utc(8), utc(11))
    x1 = notam("X1/26", circle=circle_at(0), valid_from=utc(9), valid_to=utc(13))
    write_case(tmp_path, "c", now=NOW, window=window, notams=(x1,))
    examples = examples_file(tmp_path / "examples.yaml", example("e1", "c", now=NOW, window=window))
    sheet = tmp_path / "sheet.yaml"
    assert main(["--examples", str(examples), "--out", str(sheet)], root=tmp_path, environ={}) == 0
    return tmp_path, examples, sheet


def run_check(built, capsys):
    root, examples, sheet = built
    code = main(["--check", "--examples", str(examples), "--out", str(sheet)], root=root, environ={})
    out, err = capsys.readouterr()
    return code, out, err


def test_gp5_check_prints_one_line_per_example_with_the_columns_and_exits_0(built, capsys):
    code, out, err = run_check(built, capsys)
    assert code == 0 and err == ""
    lines = out.splitlines()
    assert lines[0] == "e1 (blind): rows 3, labelled 0, unlabelled 3, unread left 0, actions done no (NOTAM actions not done), stale no"
    assert lines[-1] == "1 entries checked: 0 errors, 0 stale, 0 orphans" and len(lines) == 2


def test_gp5_the_counts_follow_the_labels_dan_types_and_actions_done(built, capsys):
    _root, _examples, sheet = built
    text = sheet.read_text(encoding="utf-8")
    text = text.replace("verdict: null\n    approaching_limits: null", "verdict: NO_GO\n    approaching_limits: true", 1).replace("actions_done: false", "actions_done: true")
    text = text.replace("X1/26: null", "X1/26: UNREAD_OVER_LZ")
    sheet.write_text(text, encoding="utf-8")
    code, out, _err = run_check(built, capsys)
    assert code == 0
    assert out.splitlines()[0] == "e1 (blind): rows 3, labelled 1, unlabelled 2, unread left 1, actions done yes, stale no"


def test_gp5_a_bad_verdict_prints_the_error_with_example_and_row_and_exits_1(built, capsys):
    _root, _examples, sheet = built
    sheet.write_text(sheet.read_text(encoding="utf-8").replace("verdict: null", "verdict: MAYBE", 1), encoding="utf-8")
    code, out, _err = run_check(built, capsys)
    assert code == 1
    (error,) = [line for line in out.splitlines() if line.startswith("error:")]
    assert "example e1, row 1 (2026-10-31T08:00:00Z)" in error and "'MAYBE'" in error
    assert out.splitlines()[-1].startswith("1 entries checked: 1 errors")


def test_check_writes_nothing(built, capsys):
    _root, examples, sheet = built
    before = [(p.read_bytes(), p.stat().st_mtime_ns) for p in (sheet, examples)]
    run_check(built, capsys)
    assert [(p.read_bytes(), p.stat().st_mtime_ns) for p in (sheet, examples)] == before


def test_check_on_a_missing_sheet_exits_1_with_one_line_naming_the_path(built, capsys):
    root, examples, _sheet = built
    gone = root / "nowhere" / "sheet.yaml"
    code = main(["--check", "--examples", str(examples), "--out", str(gone)], root=root, environ={})
    out, err = capsys.readouterr()
    assert code == 1 and out == "" and err.count("\n") == 1 and str(gone) in err
    assert not gone.exists() and not gone.parent.exists()


def test_check_cannot_be_combined_with_the_build_mode_flags(built):
    _root, examples, sheet = built
    with pytest.raises(SystemExit) as exit_info:
        main(["--check", "--blind", "--examples", str(examples), "--out", str(sheet)])
    assert exit_info.value.code == 2


def test_an_orphan_is_reported_not_an_error_and_a_stale_entry_is_reported_not_an_error(built, capsys):
    root, _examples, sheet = built
    other = examples_file(root / "other.yaml", example("somebody_else", "c", now=NOW, window=(utc(8), utc(11))))
    code = main(["--check", "--examples", str(other), "--out", str(sheet)], root=root, environ={})
    out, _err = capsys.readouterr()
    assert code == 0 and out.splitlines()[0].endswith("stale no, orphan (not in the examples file)")
    assert out.splitlines()[-1] == "1 entries checked: 0 errors, 0 stale, 1 orphans"


def test_report_line_formats_tier_only_and_incomplete_counts_and_a_named_change():
    entry = edited(
        window={"start": "2026-10-31T07:30:00Z", "end": "2026-10-31T10:00:00Z"},
        rows=[row("07:00", "08:00", verdict="NO_GO"), row("08:00", "09:00"), {"start": "2026-10-31T09:00:00Z", "end": "2026-10-31T10:00:00Z", "tier": "OUTLOOK"}],
        actions_done=True,
    )
    line = report_line(checked(entry), "(inputs, rows)")
    assert line == "e1 (blind): rows 3 (1 tier-only), labelled 0, unlabelled 2 (1 incomplete), unread left 0, actions done yes, stale (inputs, rows)"


def test_a_sheet_that_is_not_a_list_stops_with_its_line():
    with pytest.raises(SheetError, match=r"sheet.yaml: line 1: the sheet must be a YAML list"):
        parse_sheet("id: e1\nmode: blind\n", "sheet.yaml")


# --- the exit codes through the real script -----------------------------------------------------------------------


def run_script(*args, cwd=None):
    return subprocess.run([sys.executable, str(SCRIPT), *args], capture_output=True, text=True, cwd=cwd, check=False)


def test_the_script_exits_0_on_a_sheet_that_validates_and_1_on_one_that_does_not(tmp_path):
    examples = tmp_path / "examples.yaml"
    shutil.copy(GP1, examples)
    sheet = tmp_path / "sheet.yaml"
    assert run_script("--examples", str(examples), "--out", str(sheet)).returncode == 0
    ok = run_script("--check", "--examples", str(examples), "--out", str(sheet))
    assert ok.returncode == 0 and ok.stderr == ""
    assert ok.stdout.splitlines()[0].startswith(f"{cases.BLUE_MILL} (blind): rows 13, labelled 0, unlabelled 13, unread left 0, actions done no")
    sheet.write_text(sheet.read_text(encoding="utf-8").replace("verdict: null", "verdict: MAYBE", 1), encoding="utf-8")
    bad = run_script("--check", "--examples", str(examples), "--out", str(sheet))
    assert bad.returncode == 1 and f"error: example {cases.BLUE_MILL}, row 1" in bad.stdout and "'MAYBE'" in bad.stdout
