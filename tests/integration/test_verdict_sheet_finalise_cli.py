"""5a PR-2.F1 G1 / G2 / E2 / E15 / F1: `--finalise` and `--check` through verdict_sheet.main(argv, root=tmp), over a tmp sheet.

The sheet is built by main() itself (the real sheet writer) over synthetic cases, then edited as Dan would edit it in VS Code (text edits:
actions, comments), so the file the finaliser reads is the shape the real one has. The paths are the defaults: evals/agent/data/ under the
tmp root. Nothing here opens a socket, runs a reader or touches the repository's own files.
"""

import re
import shutil
from datetime import UTC, datetime
from pathlib import Path

import pytest
import yaml
from factories import circle_at, notam
from verdict_sheet_helpers import example, examples_file, write_case

from flysafelz import capture, verdict_sheet_finalise
from flysafelz.verdict_sheet import main
from flysafelz.verdict_sheet_check import parse_sheet

NOW = datetime(2026, 10, 31, 7, 30, tzinfo=UTC)
FINALISED_KEYS = ("verdict", "approaching_limits")


def utc(hour, minute=0):
    return datetime(2026, 10, 31, hour, minute, tzinfo=UTC)


WINDOW = (utc(8), utc(11))


@pytest.fixture
def root(tmp_path, monkeypatch):
    """A repository-shaped tmp root: fixtures/c, evals/agent/data/examples.yaml and a built verdict_sheet.yaml of three blind entries."""
    x1 = notam("X1/26", circle=circle_at(0), valid_from=utc(9), valid_to=utc(13))
    write_case(tmp_path, "c", now=NOW, window=WINDOW, notams=(x1,))
    data = tmp_path / "evals" / "agent" / "data"
    data.mkdir(parents=True)
    examples_file(data / "examples.yaml", *(example(i, "c", now=NOW, window=WINDOW) for i in ("a", "b", "c")))
    assert main(["--examples", str(data / "examples.yaml"), "--out", str(data / "verdict_sheet.yaml")], root=tmp_path, environ={}) == 0
    monkeypatch.setattr(capture, "capture", lambda *a, **k: pytest.fail("finalise must never capture"))
    return tmp_path


def sheet_path(root: Path) -> Path:
    return root / "evals" / "agent" / "data" / "verdict_sheet.yaml"


def read(root: Path) -> str:
    return sheet_path(root).read_text(encoding="utf-8")


def entry_span(root: Path, entry_id: str) -> str:
    return next(e.span for e in parse_sheet(read(root), "sheet.yaml") if e.id == entry_id)


def edit(root: Path, entry_id: str, old: str, new: str) -> None:
    """Dan's text edit inside one entry."""
    text = read(root)
    span = entry_span(root, entry_id)
    assert old in span
    sheet_path(root).write_text(text.replace(span, span.replace(old, new, 1), 1), encoding="utf-8")


def done(root: Path, entry_id: str, action: str = "NO_FLY") -> None:
    edit(root, entry_id, "actions_done: false", "actions_done: true")
    edit(root, entry_id, "X1/26: null", f"X1/26: {action}")


def finalise(root: Path, capsys, *extra: str, code_root: Path | None = None):
    code = main(["--finalise", *extra], root=root, environ={}, code_root=code_root)
    out, err = capsys.readouterr()
    return code, out, err


def check(root: Path, capsys, code_root: Path | None = None):
    code = main(["--check"], root=root, environ={}, code_root=code_root)
    out, err = capsys.readouterr()
    return code, out, err


def snapshot(root: Path) -> dict[Path, tuple[bytes, int]]:
    return {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in sorted(root.rglob("*")) if p.is_file()}


def rows(root: Path, entry_id: str) -> list[dict]:
    return next(e for e in yaml.safe_load(read(root)) if e["id"] == entry_id)["rows"]


def without_finalised(data):
    out = []
    for item in data:
        item = {k: v for k, v in item.items() if k != "finalised"}
        item["rows"] = [{k: v for k, v in r.items() if k not in FINALISED_KEYS} for r in item["rows"]]
        out.append(item)
    return out


# --- G1: the command ---------------------------------------------------------------------------------------------------------------------


def test_finalise_labels_the_rows_of_a_done_entry_from_dans_actions_and_skips_the_others(root, capsys):
    done(root, "a", "NO_FLY")
    done(root, "c", "IGNORE")
    code, _out, err = finalise(root, capsys)
    assert code == 0 and err == ""
    assert all(isinstance(r["approaching_limits"], bool) for r in rows(root, "a") + rows(root, "c"))
    assert [r["verdict"] for r in rows(root, "a")] == ["PILOT_DISCRETION", "NO_GO", "NO_GO"]  # X1 starts at 09:00
    assert {r["verdict"] for r in rows(root, "c")} == {"PILOT_DISCRETION"}
    assert all(r["verdict"] is None and r["approaching_limits"] is None for r in rows(root, "b"))  # skipped: nothing written
    assert "finalised" in yaml.safe_load(entry_span(root, "a"))[0] and "finalised" not in yaml.safe_load(entry_span(root, "b"))[0]


def test_the_summary_has_counts_by_verdict_and_one_line_per_moved_row_never_every_row(root, capsys):
    done(root, "a", "NO_FLY")
    done(root, "c", "IGNORE")
    _code, out, _err = finalise(root, capsys)
    lines = out.splitlines()
    assert lines[0] == "finalised 2 entries, 1 skipped (actions not done), 0 failed"
    assert lines[1] == "rows by verdict, before -> after:"
    counts = [line for line in lines if re.fullmatch(r"  [A-Za-z_]+: \d+ -> \d+", line)]
    assert counts == ["  NO_GO: 0 -> 2", "  PILOT_DISCRETION: 0 -> 4", "  unlabelled: 6 -> 0"]
    assert "6 rows moved:" in lines
    moved = [line for line in lines if re.fullmatch(r"  [ac] \S+Z–\S+Z: .+ → .+", line)]
    assert len(moved) == 6  # every one of the six rows changed from unlabelled; b's three rows are not listed
    assert any(line.startswith("  a 2026-10-31T09:00:00Z–2026-10-31T10:00:00Z: unlabelled → NO_GO") for line in lines)
    assert any(line.startswith("  a 2026-10-31T08:00:00Z–2026-10-31T09:00:00Z: unlabelled → PILOT_DISCRETION") for line in lines)
    assert not any(line.startswith("  b ") for line in lines)


def test_a_row_that_does_not_move_is_not_listed(root, capsys):
    done(root, "a")
    finalise(root, capsys)
    _code, out, _err = finalise(root, capsys)
    assert "0 rows moved" in out.splitlines() and not [line for line in out.splitlines() if " → " in line]


def test_only_the_row_labels_and_the_stamp_are_written_and_comments_and_the_fingerprint_stay(root, capsys):
    done(root, "a")
    text = read(root)
    text = "# Dan: top-of-file comment\n" + text.replace("  notam_actions:\n    X1/26: NO_FLY", "  notam_actions:\n    # the SUP says this is a prohibited area\n    X1/26: NO_FLY  # my reading", 1)
    text = text.replace("    edge: window start\n", "    edge: window start  # first row\n    # a comment line inside a row\n", 1)
    sheet_path(root).write_text(text, encoding="utf-8")
    before_data = yaml.safe_load(read(root))
    before_fingerprints = [e["fingerprint"] for e in before_data]
    before_lines = read(root).splitlines()
    code, _out, _err = finalise(root, capsys)
    after_text = read(root)
    assert code == 0
    comment_lines = [line for line in before_lines if "#" in line]
    assert len(comment_lines) == 5 and all(line in after_text.splitlines() for line in comment_lines)
    after_data = yaml.safe_load(after_text)
    assert without_finalised(after_data) == without_finalised(before_data)  # parsed-data equality minus the written keys
    assert [e["fingerprint"] for e in after_data] == before_fingerprints
    for e in before_data:
        assert re.search(rf"^    seal: {e['fingerprint']['seal']}$", after_text, re.MULTILINE)
    removed = [line for line in before_lines if line not in after_text.splitlines()]
    assert all(re.match(r"\s+(verdict|approaching_limits): ", line) for line in removed)  # the only old lines to go are the rows' labels
    assert entry_span(root, "b") in text  # the skipped entry's text is the same bytes as before


def test_a_stored_tier_that_code_does_not_cut_is_an_error_naming_the_row_and_the_tier(root, capsys):
    """A row whose tier code now cuts differently (here a VERDICT row stored as OUTLOOK) is refused whole, like a moved edge."""
    done(root, "a")
    edit(root, "a", "    edge: hour\n    verdict: null\n    approaching_limits: null", "    edge: hour\n    tier: OUTLOOK")
    before = snapshot(root)
    code, _out, err = finalise(root, capsys)
    assert code == 1 and err.splitlines() == [
        "example a: code's rows no longer match the stored rows: row 3 is 2026-10-31T10:00:00Z to 2026-10-31T11:00:00Z (tier OUTLOOK) on the sheet but code cuts 2026-10-31T10:00:00Z to 2026-10-31T11:00:00Z"
    ]
    assert snapshot(root) == before


def test_finalise_is_idempotent_and_the_second_run_leaves_the_file_byte_identical(root, capsys):
    done(root, "a")
    done(root, "c", "AWARENESS")
    finalise(root, capsys)
    first = sheet_path(root).read_bytes()
    stat = sheet_path(root).stat().st_mtime_ns
    code, out, _err = finalise(root, capsys)
    assert code == 0 and sheet_path(root).read_bytes() == first
    assert sheet_path(root).stat().st_mtime_ns == stat  # not even rewritten
    assert out.splitlines()[0] == "finalised 2 entries, 1 skipped (actions not done), 0 failed" and "0 rows moved" in out.splitlines()


def test_a_second_finalise_after_dan_changes_an_action_moves_only_that_entrys_rows(root, capsys):
    done(root, "a")
    finalise(root, capsys)
    edit(root, "a", "X1/26: NO_FLY", "X1/26: IGNORE")
    code, out, _err = finalise(root, capsys)
    assert code == 0
    moved = [line for line in out.splitlines() if " → " in line]
    assert len(moved) == 2 and all(line.startswith("  a ") and "NO_GO" in line and "→ PILOT_DISCRETION" in line for line in moved)
    assert {r["verdict"] for r in rows(root, "a")} == {"PILOT_DISCRETION"}


def test_null_notam_actions_have_no_notam_effects(root, capsys):
    edit(root, "a", "actions_done: false", "actions_done: true")
    edit(root, "a", "  notam_actions:\n    X1/26: null\n", "  notam_actions: null\n")
    code, _out, err = finalise(root, capsys)
    assert code == 0 and err == "" and {r["verdict"] for r in rows(root, "a")} == {"PILOT_DISCRETION"}


def test_an_unread_notam_left_in_a_done_entry_stops_the_run_and_writes_nothing(root, capsys):
    done(root, "a", "UNREAD_OVER_LZ")
    before = snapshot(root)
    code, _out, err = finalise(root, capsys)
    assert code == 1 and "a" in err and "UNREAD_OVER_LZ" in err and snapshot(root) == before


# --- G1: usage ---------------------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("other", ["--blind", "--draft", "--check"])
def test_finalise_cannot_be_combined_with_the_build_flags(root, other):
    before = snapshot(root)
    with pytest.raises(SystemExit) as info:
        main(["--finalise", other], root=root, environ={})
    assert info.value.code == 2 and snapshot(root) == before


def test_without_finalise_or_check_the_examples_and_out_are_still_required(root):
    with pytest.raises(SystemExit) as info:
        main([], root=root, environ={})
    assert info.value.code == 2


def test_explicit_paths_beat_the_defaults(root, capsys):
    elsewhere = root / "elsewhere.yaml"
    shutil.copyfile(sheet_path(root), elsewhere)
    done(root, "a")  # the default sheet changes; the other copy does not
    edit_text = elsewhere.read_text(encoding="utf-8")
    code, out, _err = finalise(root, capsys, "--out", str(elsewhere), "--examples", str(root / "evals" / "agent" / "data" / "examples.yaml"))
    assert code == 0 and out.splitlines()[0] == "finalised 0 entries, 3 skipped (actions not done), 0 failed"
    assert elsewhere.read_text(encoding="utf-8") == edit_text


# --- F1: writes only the sheet ------------------------------------------------------------------------------------------------------------


def test_finalise_writes_only_the_sheet_and_leaves_no_temporary_file(root, capsys):
    done(root, "a")
    sheet_path(root).chmod(0o640)
    mode = 0o640
    before = snapshot(root)
    code, _out, _err = finalise(root, capsys)
    after = snapshot(root)
    assert code == 0
    changed = {p for p in after if after[p][0] != before.get(p, (None,))[0]}
    assert changed == {sheet_path(root)}
    assert {p for p in after if p != sheet_path(root) and after[p][1] != before[p][1]} == set()  # no other file's mtime moved
    assert set(after) == set(before)  # nothing created beside it
    assert sheet_path(root).stat().st_mode & 0o777 == mode


def test_check_writes_nothing(root, capsys):
    done(root, "a")
    finalise(root, capsys)
    before = snapshot(root)
    check(root, capsys)
    assert snapshot(root) == before


# --- G2 / E2: --check -------------------------------------------------------------------------------------------------------------------


def test_after_finalise_check_passes_and_reports_the_unfinalised_entry_as_a_note_not_an_error(root, capsys):
    done(root, "a")
    done(root, "c", "CONTACT_FIRST")
    finalise(root, capsys)
    code, out, err = check(root, capsys)
    assert code == 0 and err == ""
    lines = out.splitlines()
    assert "note: example b is not finalised (NOTAM actions not done)" in lines
    assert lines[-1] == "3 entries checked: 0 errors, 0 stale, 0 orphans"


def test_check_refuses_a_done_entry_that_was_never_finalised(root, capsys):
    done(root, "a")
    code, out, _err = check(root, capsys)
    assert code == 1
    assert "error: example a: verdict sheet is stale (finalised stamp missing): run make dataset" in out.splitlines()


def test_check_refuses_a_stamp_edited_by_hand(root, capsys):
    done(root, "a")
    finalise(root, capsys)
    text = read(root)
    stamp = yaml.safe_load(text)[0]["finalised"]
    sheet_path(root).write_text(text.replace(stamp["rules"], "0" * 64, 1), encoding="utf-8")
    code, out, _err = check(root, capsys)
    assert code == 1 and "error: example a: verdict sheet is stale (finalised stamp unreadable): run make dataset" in out.splitlines()


def test_check_refuses_after_dans_label_is_edited_after_finalise(root, capsys):
    done(root, "a")
    finalise(root, capsys)
    edit(root, "a", "X1/26: NO_FLY", "X1/26: IGNORE")
    code, out, _err = check(root, capsys)
    assert code == 1 and "error: example a: verdict sheet is stale (labels changed): run make dataset" in out.splitlines()


@pytest.mark.parametrize("name", ["weather.json", "airspace.json", "pib.xml", "readings.yaml"])
def test_check_refuses_after_a_fixture_file_changes_naming_it(root, capsys, name):
    done(root, "a")
    finalise(root, capsys)
    with (root / "fixtures" / "c" / name).open("ab") as out:
        out.write(b"\n")
    code, out, _err = check(root, capsys)
    assert code == 1 and f"error: example a: verdict sheet is stale ({name} changed): run make dataset" in out.splitlines()


def test_check_refuses_after_rules_yaml_or_a_module_changes_in_the_code_it_is_pointed_at(root, capsys, tmp_path_factory):
    """The finalise ran over the repository's code; --check is told the code is another copy with one byte more in rules.yaml."""
    done(root, "a")
    finalise(root, capsys)
    src = Path(verdict_sheet_finalise.__file__).parent.parent
    copy = tmp_path_factory.mktemp("code")
    for relative in [*verdict_sheet_finalise.module_files(), "flysafelz/rules.yaml"]:
        (copy / relative).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src / relative, copy / relative)
    assert check(root, capsys, code_root=copy)[0] == 0
    with (copy / "flysafelz" / "rules.yaml").open("ab") as out:
        out.write(b"\n")
    code, out, _err = check(root, capsys, code_root=copy)
    assert code == 1 and "error: example a: verdict sheet is stale (rules.yaml changed): run make dataset" in out.splitlines()
    with (copy / "flysafelz" / "pib.py").open("ab") as out:
        out.write(b"\n")
    code, out, _err = check(root, capsys, code_root=copy)
    assert "error: example a: verdict sheet is stale (rules.yaml, src/flysafelz/pib.py changed): run make dataset" in out.splitlines()


def test_a_finalise_run_from_the_other_copy_of_the_code_makes_check_pass_again(root, capsys, tmp_path_factory):
    done(root, "a")
    copy = tmp_path_factory.mktemp("code2")
    src = Path(verdict_sheet_finalise.__file__).parent.parent
    for relative in [*verdict_sheet_finalise.module_files(), "flysafelz/rules.yaml"]:
        (copy / relative).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src / relative, copy / relative)
    with (copy / "flysafelz" / "rules.yaml").open("ab") as out:
        out.write(b"\n")
    assert finalise(root, capsys, code_root=copy)[0] == 0
    assert check(root, capsys, code_root=copy)[0] == 0
    assert check(root, capsys)[0] == 1  # but not against the repository's own rules.yaml


def test_finalised_problems_is_the_one_function_check_and_the_loader_use(root, capsys):
    done(root, "a")
    done(root, "c")
    finalise(root, capsys)
    edit(root, "c", "X1/26: NO_FLY", "X1/26: AWARENESS")
    problems = verdict_sheet_finalise.finalised_problems(sheet_path(root), root)
    assert problems == {"c": "verdict sheet is stale (labels changed): run make dataset"}
    _code, out, _err = check(root, capsys)
    assert [f"error: example {k}: {v}" for k, v in problems.items()] == [line for line in out.splitlines() if line.startswith("error:")]


def test_check_gets_its_problems_from_finalised_problems_called_with_the_sheet_root_and_examples(root, capsys, monkeypatch):
    """--check and the loader share the one named function: whatever it answers is what --check refuses."""
    calls = []

    def fake(sheet, where, examples=None, *, code_root=None):
        calls.append((sheet, where, examples, code_root))
        return {"b": "verdict sheet is stale (made up): run make dataset"}

    monkeypatch.setattr(verdict_sheet_finalise, "finalised_problems", fake)
    code, out, _err = check(root, capsys)
    data = root / "evals" / "agent" / "data"
    assert calls == [(data / "verdict_sheet.yaml", root, data / "examples.yaml", None)]
    assert code == 1 and "error: example b: verdict sheet is stale (made up): run make dataset" in out.splitlines()


def test_finalised_problems_is_empty_for_a_finalised_sheet_and_never_lists_a_not_done_entry(root, capsys):
    assert verdict_sheet_finalise.finalised_problems(sheet_path(root), root) == {}  # nothing done, so nothing refused
    done(root, "a")
    assert list(verdict_sheet_finalise.finalised_problems(sheet_path(root), root)) == ["a"]
    finalise(root, capsys)
    assert verdict_sheet_finalise.finalised_problems(sheet_path(root), root) == {}


def test_finalised_problems_takes_the_examples_beside_the_sheet_or_a_named_file(root, capsys):
    done(root, "a")
    named = root / "evals" / "agent" / "data" / "examples.yaml"
    assert verdict_sheet_finalise.finalised_problems(sheet_path(root), root, named) == verdict_sheet_finalise.finalised_problems(sheet_path(root), root)
    assert verdict_sheet_finalise.finalised_problems(sheet_path(root), root, examples_file(root / "none.yaml")) == {}  # an entry with no example is not checked


# --- E15: the rows code cuts differ from the stored rows -----------------------------------------------------------------------------------------


def replace_rows(root: Path, entry_id: str, from_start: str, new_rows: str) -> None:
    """Dan's edit of an entry's rows: everything from the row starting at `from_start` is replaced by `new_rows`."""
    span = entry_span(root, entry_id)
    head = span.split(f"  - start: '{from_start}'")[0]
    sheet_path(root).write_text(read(root).replace(span, head + new_rows, 1), encoding="utf-8")


def row_text(start: str, end: str, edge: str) -> str:
    return f"  - start: '2026-10-31T{start}:00Z'\n    end: '2026-10-31T{end}:00Z'\n    edge: {edge}\n    verdict: null\n    approaching_limits: null\n"


def test_a_boundary_moved_by_a_minute_is_an_error_naming_the_entry_and_row_and_nothing_is_written_for_it(root, capsys):
    done(root, "a")
    done(root, "c")
    edit(root, "a", "end: '2026-10-31T09:00:00Z'\n    edge: window start", "end: '2026-10-31T09:01:00Z'\n    edge: window start")
    edit(root, "a", "start: '2026-10-31T09:00:00Z'", "start: '2026-10-31T09:01:00Z'")
    a_before = entry_span(root, "a")
    code, out, err = finalise(root, capsys)
    assert code == 1
    (line,) = err.splitlines()
    assert line == (
        "example a: code's rows no longer match the stored rows: row 1 is 2026-10-31T08:00:00Z to 2026-10-31T09:01:00Z on the sheet but code cuts 2026-10-31T08:00:00Z to 2026-10-31T09:00:00Z"
    )
    assert out.splitlines()[0] == "finalised 1 entries, 1 skipped (actions not done), 1 failed"  # c is still finalised and reported
    assert entry_span(root, "a") == a_before  # not a byte of the failing entry
    assert [r["verdict"] for r in rows(root, "c")] == ["PILOT_DISCRETION", "NO_GO", "NO_GO"]


def test_a_row_added_is_an_error_naming_the_first_row_that_differs(root, capsys):
    done(root, "a")
    replace_rows(root, "a", "2026-10-31T10:00:00Z", row_text("10:00", "10:30", "hour") + row_text("10:30", "11:00", "hour"))
    assert len(rows(root, "a")) == 4
    before = snapshot(root)
    code, _out, err = finalise(root, capsys)
    assert code == 1
    assert err.splitlines() == [
        "example a: code's rows no longer match the stored rows: row 3 is 2026-10-31T10:00:00Z to 2026-10-31T10:30:00Z on the sheet but code cuts 2026-10-31T10:00:00Z to 2026-10-31T11:00:00Z"
    ]
    assert snapshot(root) == before


def test_a_row_dropped_is_an_error_naming_the_first_row_that_differs(root, capsys):
    done(root, "a")
    replace_rows(root, "a", "2026-10-31T09:00:00Z", row_text("09:00", "11:00", "X1/26 starts"))
    assert len(rows(root, "a")) == 2
    before = snapshot(root)
    code, _out, err = finalise(root, capsys)
    assert code == 1
    assert err.splitlines() == [
        "example a: code's rows no longer match the stored rows: row 2 is 2026-10-31T09:00:00Z to 2026-10-31T11:00:00Z on the sheet but code cuts 2026-10-31T09:00:00Z to 2026-10-31T10:00:00Z"
    ]
    assert snapshot(root) == before


def test_one_boundary_moved_so_two_rows_overlap_fails_that_entry_alone_and_the_others_are_still_finalised(root, capsys):
    """The sheet's own row check calls this an overlap; finalise reports it as E15 for the entry, not as a sheet it cannot read."""
    done(root, "a")
    done(root, "c")
    edit(root, "a", "end: '2026-10-31T09:00:00Z'\n    edge: window start", "end: '2026-10-31T09:01:00Z'\n    edge: window start")
    a_before = entry_span(root, "a")
    code, out, err = finalise(root, capsys)
    assert code == 1
    assert err.splitlines() == [
        "example a: code's rows no longer match the stored rows: row 1 is 2026-10-31T08:00:00Z to 2026-10-31T09:01:00Z on the sheet but code cuts 2026-10-31T08:00:00Z to 2026-10-31T09:00:00Z"
    ]
    assert out.splitlines()[0] == "finalised 1 entries, 1 skipped (actions not done), 1 failed"
    assert entry_span(root, "a") == a_before
    assert [r["verdict"] for r in rows(root, "c")] == ["PILOT_DISCRETION", "NO_GO", "NO_GO"]
    assert "finalised" in yaml.safe_load(entry_span(root, "c"))[0] and "finalised" not in yaml.safe_load(entry_span(root, "a"))[0]


def test_one_boundary_moved_so_two_rows_leave_a_gap_fails_that_entry_alone(root, capsys):
    done(root, "a")
    done(root, "c")
    edit(root, "a", "start: '2026-10-31T09:00:00Z'", "start: '2026-10-31T08:59:00Z'")  # row 1 now ends after row 2 starts: an overlap the other way
    edit(root, "c", "start: '2026-10-31T10:00:00Z'", "start: '2026-10-31T10:01:00Z'")  # row 2 ends at 10:00, row 3 starts at 10:01: a gap
    code, out, err = finalise(root, capsys)
    assert code == 1 and out.splitlines()[0] == "finalised 0 entries, 1 skipped (actions not done), 2 failed"
    assert sorted(line.split(":")[0] for line in err.splitlines()) == ["example a", "example c"]
    assert all("code's rows no longer match the stored rows: row " in line for line in err.splitlines())


def test_the_last_row_dropped_so_the_rows_stop_short_of_the_window_fails_that_entry_alone(root, capsys):
    done(root, "a")
    done(root, "c")
    replace_rows(root, "a", "2026-10-31T10:00:00Z", "")
    assert len(rows(root, "a")) == 2
    code, out, err = finalise(root, capsys)
    assert code == 1
    assert err.splitlines() == ["example a: code's rows no longer match the stored rows: row 3 is not on the sheet but code cuts 2026-10-31T10:00:00Z to 2026-10-31T11:00:00Z"]
    assert out.splitlines()[0] == "finalised 1 entries, 1 skipped (actions not done), 1 failed"
    assert [r["verdict"] for r in rows(root, "c")] == ["PILOT_DISCRETION", "NO_GO", "NO_GO"]


def test_rows_equal_to_codes_under_a_window_that_asks_for_more_are_still_refused_for_that_entry_alone(root, capsys):
    done(root, "a")
    done(root, "c")
    edit(root, "a", "  window:\n    start: '2026-10-31T08:00:00Z'\n    end: '2026-10-31T11:00:00Z'", "  window:\n    start: '2026-10-31T08:00:00Z'\n    end: '2026-10-31T12:00:00Z'")
    before = entry_span(root, "a")
    code, out, err = finalise(root, capsys)
    assert code == 1 and len(err.splitlines()) == 1 and err.startswith("example a: ") and "the window needs the rows to end at 2026-10-31T12:00:00Z" in err
    assert out.splitlines()[0] == "finalised 1 entries, 1 skipped (actions not done), 1 failed"
    assert entry_span(root, "a") == before


def test_an_overlap_in_an_entry_whose_actions_are_not_done_still_stops_the_run(root, capsys):
    """Only an entry being finalised is judged by E15; a malformed entry Dan has not got to is an invalid sheet."""
    done(root, "c")
    edit(root, "a", "end: '2026-10-31T09:00:00Z'\n    edge: window start", "end: '2026-10-31T09:01:00Z'\n    edge: window start")
    before = snapshot(root)
    code, out, err = finalise(root, capsys)
    assert code == 1 and out == "" and "overlap" in err and snapshot(root) == before


# --- failures that stop the run with one line ---------------------------------------------------------------------------------------------


def test_a_missing_case_folder_is_a_one_line_error_and_the_sheet_is_untouched(root, capsys):
    done(root, "a")
    shutil.rmtree(root / "fixtures" / "c")
    before = snapshot(root)
    code, _out, err = finalise(root, capsys)
    assert code == 1 and len(err.splitlines()) == 1 and err.startswith("example a: ") and "c" in err
    assert snapshot(root) == before


def test_a_missing_case_folder_of_one_done_entry_leaves_the_whole_sheet_untouched_though_the_other_is_fine(root, capsys):
    """Spec C20: one line, sheet untouched. Entry a (case c, whole) must not be written while entry b (case d, gone) cannot be."""
    x1 = notam("X1/26", circle=circle_at(0), valid_from=utc(9), valid_to=utc(13))
    write_case(root, "d", now=NOW, window=WINDOW, notams=(x1,))
    data = root / "evals" / "agent" / "data"
    examples_file(data / "examples.yaml", example("a", "c", now=NOW, window=WINDOW), example("b", "d", now=NOW, window=WINDOW))
    sheet_path(root).unlink()
    assert main(["--examples", str(data / "examples.yaml"), "--out", str(sheet_path(root))], root=root, environ={}) == 0
    capsys.readouterr()
    done(root, "a")
    done(root, "b")
    shutil.rmtree(root / "fixtures" / "d")
    before = snapshot(root)
    code, out, err = finalise(root, capsys)
    assert code == 1 and out == ""
    assert err.splitlines() == ["example b: no case folder fixtures/d/ to finalise its entry from; nothing written"]
    assert snapshot(root) == before
    assert all(r["verdict"] is None for r in rows(root, "a"))


def test_an_invalid_sheet_is_a_one_line_error_and_the_sheet_is_untouched(root, capsys):
    done(root, "a")
    edit(root, "a", "verdict: null", "verdict: MAYBE")
    before = snapshot(root)
    code, out, err = finalise(root, capsys)
    assert code == 1 and out == "" and len(err.splitlines()) == 1 and "MAYBE" in err
    assert snapshot(root) == before


def test_a_missing_sheet_is_a_one_line_error(root, capsys):
    sheet_path(root).unlink()
    code, out, err = finalise(root, capsys)
    assert code == 1 and out == "" and len(err.splitlines()) == 1 and "verdict_sheet.yaml" in err
    assert not sheet_path(root).exists()
