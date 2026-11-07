"""G6 / E3 / E7 / F2 / F3: the rerun. verdict_sheet.main() over synthetic cases in tmp_path, called through its real command line.

Each test builds a sheet with main(), edits it as Dan would (text edits in VS Code), reruns, and compares the bytes: the
file as a whole, and the exact text span of each entry (parse_sheet gives the spans). Nothing touches the network or
the committed fixtures.
"""

import ast
import hashlib
import re
from datetime import UTC, datetime
from pathlib import Path

import pytest
import yaml
from factories import circle_at, notam
from verdict_sheet_helpers import example, examples_file, write_case

from flysafelz import verdict_sheet
from flysafelz.verdict_sheet import main
from flysafelz.verdict_sheet_check import parse_sheet

SRC = Path(verdict_sheet.__file__).parent
NOW = datetime(2026, 10, 31, 7, 30, tzinfo=UTC)


def utc(hour, minute=0, day=31):
    return datetime(2026, 10, day, hour, minute, tzinfo=UTC)


WINDOW = (utc(8), utc(11))


@pytest.fixture
def root(tmp_path):
    """fixtures/c/ under tmp_path: one calm case with a NOTAM that starts at 09:00."""
    x1 = notam("X1/26", circle=circle_at(0), valid_from=utc(9), valid_to=utc(13))
    write_case(tmp_path, "c", now=NOW, window=WINDOW, notams=(x1,))
    return tmp_path


def run(root, *examples, capsys, check=False, name="examples.yaml"):
    """main() on the examples given; (exit code, stdout, stderr)."""
    path = examples_file(root / name, *examples)
    argv = ["--examples", str(path), "--out", str(root / "sheet.yaml")]
    code = main([*(["--check"] if check else []), *argv], root=root, environ={})
    out, err = capsys.readouterr()
    return code, out, err


def sheet_bytes(root) -> bytes:
    return (root / "sheet.yaml").read_bytes()


def spans(root) -> dict[str, bytes]:
    text = sheet_bytes(root).decode("utf-8")
    return {str(e.id): e.span.encode("utf-8") for e in parse_sheet(text, "sheet.yaml")}


def ex(example_id, **kw):
    return example(example_id, "c", now=NOW, window=kw.pop("window", WINDOW), **kw)


def label_like_dan(root, first="NO_GO"):
    """What Dan does in VS Code: type a verdict, a flag, a comment of his own, odd spacing; the file's other bytes stay."""
    path = root / "sheet.yaml"
    text = path.read_text(encoding="utf-8")
    text = text.replace("verdict: null", f"verdict:   {first}   # my call", 1).replace("approaching_limits: null", "approaching_limits: true", 1)
    path.write_text("# Dan's sheet, labelled on the train\n\n" + text.rstrip("\n") + "\n\n# end of the first batch\n", encoding="utf-8")


# --- G6: append-only --------------------------------------------------------------------------------------------


def test_g6_a_rerun_appends_only_the_new_entry_as_text_and_the_old_bytes_are_unchanged(root, capsys):
    assert run(root, ex("a"), capsys=capsys)[0] == 0
    label_like_dan(root)
    before = sheet_bytes(root)
    a_span = spans(root)["a"]
    code, out, err = run(root, ex("a"), ex("b"), capsys=capsys)
    after = sheet_bytes(root)
    assert code == 0 and err == ""
    assert out.strip() == f"wrote {root / 'sheet.yaml'}: 1 new, 1 unchanged, 0 stale, 0 skipped, 0 failed"
    assert after.startswith(before) and len(after) > len(before)  # the prefix, byte for byte
    assert spans(root)["a"] == a_span  # entry a's own span (with Dan's comments) is exactly what it was
    assert list(spans(root)) == ["a", "b"]  # the new entry is at the end
    assert b"verdict:   NO_GO   # my call" in after and after.startswith(b"# Dan's sheet")
    assert after[len(before) :].decode("utf-8").startswith("- id: b\n")


def test_g6_a_file_that_does_not_end_in_a_newline_still_keeps_its_bytes_and_gets_the_new_entry_after_one(root, capsys):
    run(root, ex("a"), capsys=capsys)
    path = root / "sheet.yaml"
    path.write_bytes(path.read_bytes().rstrip(b"\n"))
    before = sheet_bytes(root)
    assert run(root, ex("a"), ex("b"), capsys=capsys)[0] == 0
    assert sheet_bytes(root).startswith(before) and list(spans(root)) == ["a", "b"]
    assert yaml.safe_load(sheet_bytes(root))[1]["id"] == "b"


def test_g6_the_whole_file_is_parsed_again_after_the_append_and_holds_both_entries(root, capsys):
    run(root, ex("a"), capsys=capsys)
    run(root, ex("a"), ex("b", blind=False), capsys=capsys)
    entries = yaml.safe_load(sheet_bytes(root))
    assert [(e["id"], e["mode"]) for e in entries] == [("a", "blind"), ("b", "draft")]
    assert all(set(e["fingerprint"]) >= {"inputs", "rows"} for e in entries)


def test_g6_an_append_that_would_not_parse_leaves_the_file_alone(root, capsys):
    """A sheet written as a flow list cannot take block entries after it; the check before writing refuses and nothing changes."""
    (root / "sheet.yaml").write_text("[]\n", encoding="utf-8")
    before = (sheet_bytes(root), (root / "sheet.yaml").stat().st_mtime_ns)
    code, out, err = run(root, ex("a"), capsys=capsys)
    assert code == 1 and "nothing written" in err and err.count("\n") == 1
    assert (sheet_bytes(root), (root / "sheet.yaml").stat().st_mtime_ns) == before


def test_the_same_run_twice_appends_nothing_reports_0_new_and_0_stale_and_the_file_is_identical(root, capsys):
    run(root, ex("a"), ex("b", blind=False), capsys=capsys)
    path = root / "sheet.yaml"
    before = (path.read_bytes(), path.stat().st_mtime_ns)
    code, out, err = run(root, ex("a"), ex("b", blind=False), capsys=capsys)
    assert code == 0 and err == ""
    assert out.strip() == "nothing written: 0 new, 2 unchanged, 0 stale, 0 skipped, 0 failed"
    assert (path.read_bytes(), path.stat().st_mtime_ns) == before


def test_a_sheet_file_that_does_not_exist_is_created_by_the_first_run(root, capsys):
    assert not (root / "sheet.yaml").exists()
    code, out, _err = run(root, ex("a"), capsys=capsys)
    assert code == 0 and (root / "sheet.yaml").exists() and "1 new" in out


def test_an_empty_or_comment_only_sheet_takes_the_first_entry_after_its_own_text(root, capsys):
    (root / "sheet.yaml").write_text("# start of my sheet\n", encoding="utf-8")
    assert run(root, ex("a"), capsys=capsys)[0] == 0
    assert sheet_bytes(root).startswith(b"# start of my sheet\n- id: a\n")


def test_nothing_new_and_no_sheet_creates_no_file(root, capsys):
    code, out, _err = run(root, {"id": "why", "reference": {"path": "explain"}}, capsys=capsys)
    assert code == 0 and not (root / "sheet.yaml").exists() and out.strip() == "nothing written: 0 new, 0 unchanged, 0 stale, 1 skipped, 0 failed"


# --- E3: stale ---------------------------------------------------------------------------------------------------


def test_e3_a_changed_window_is_reported_stale_with_inputs_and_the_entry_is_left_byte_for_byte(root, capsys):
    run(root, ex("a"), capsys=capsys)
    label_like_dan(root)
    before = sheet_bytes(root)
    moved = ex("a", window=(utc(8), utc(10)))
    code, out, err = run(root, moved, capsys=capsys)
    assert code == 0 and err == ""
    assert "example a: stale (inputs); its entry is left as it is" in out
    assert out.strip().endswith("nothing written: 0 new, 0 unchanged, 1 stale, 0 skipped, 0 failed")
    assert sheet_bytes(root) == before
    code, out, _err = run(root, moved, check=True, capsys=capsys)
    assert code == 0 and out.splitlines()[0].endswith("stale (inputs)")
    assert sheet_bytes(root) == before


def test_e3_a_stale_entry_does_not_stop_the_new_entry_and_stays_exactly_as_it_was(root, capsys):
    run(root, ex("a"), capsys=capsys)
    label_like_dan(root)
    a_span, before = spans(root)["a"], sheet_bytes(root)
    code, out, _err = run(root, ex("a", window=(utc(8), utc(10))), ex("b"), capsys=capsys)
    assert code == 0 and "1 new, 0 unchanged, 1 stale" in out
    assert sheet_bytes(root).startswith(before) and spans(root)["a"] == a_span and list(spans(root)) == ["a", "b"]


def test_e3_the_pilots_lat_lon_now_case_and_drop_each_make_an_entry_stale_too(root, capsys):
    run(root, ex("a"), capsys=capsys)
    before = sheet_bytes(root)
    for changed in (ex("a", lat=53.9), ex("a", lon=-0.1), ex("a", drop=["pib"]), example("a", "c", now=utc(7, 45), window=WINDOW)):
        code, out, _err = run(root, changed, capsys=capsys)
        assert "example a: stale (inputs)" in out, changed
        assert sheet_bytes(root) == before


def test_f2_a_blind_entry_whose_example_is_now_a_draft_is_stale_and_never_refilled(root, capsys):
    run(root, ex("a"), capsys=capsys)
    before = sheet_bytes(root)
    code, out, _err = run(root, ex("a", blind=False), capsys=capsys)
    assert code == 0 and "example a: stale (inputs: mode was blind, now draft)" in out
    assert sheet_bytes(root) == before
    text = before.decode("utf-8")
    assert "# why" not in text and "mode: draft" not in text
    entry = yaml.safe_load(text)[0]
    assert all(r["verdict"] is None and r["approaching_limits"] is None for r in entry["rows"])
    assert all(v is None for v in entry["notam_actions"].values())
    code, out, _err = run(root, ex("a", blind=False), check=True, capsys=capsys)
    assert "stale (inputs: mode was blind, now draft)" in out.splitlines()[0]


def test_f2_a_draft_entry_whose_example_is_now_blind_is_stale_and_unchanged(root, capsys):
    run(root, ex("a", blind=False), capsys=capsys)
    before = sheet_bytes(root)
    code, out, _err = run(root, ex("a"), capsys=capsys)
    assert code == 0 and "stale (inputs: mode was draft, now blind)" in out and sheet_bytes(root) == before


@pytest.mark.parametrize(
    "mangle",
    [
        pytest.param(lambda t: re.sub(r"  fingerprint:\n(    .+\n)+", "", t), id="missing"),
        pytest.param(lambda t: re.sub(r"  fingerprint:\n(    .+\n)+", "  fingerprint: abc\n", t), id="a bare word"),
        pytest.param(lambda t: re.sub(r"(    inputs: )[0-9a-f]+", r"\1not-a-hash", t), id="inputs edited by hand to text"),
        pytest.param(lambda t: re.sub(r"    rows: .+\n", "", t), id="one part deleted"),
        pytest.param(lambda t: re.sub(r"(    inputs: )[0-9a-f]", lambda m: m.group(1) + ("0" if m.group(0)[-1] != "0" else "1"), t), id="inputs edited by hand to another valid hash"),
        pytest.param(lambda t: re.sub(r"(    rows: )[0-9a-f]", lambda m: m.group(1) + ("0" if m.group(0)[-1] != "0" else "1"), t), id="rows edited by hand to another valid hash"),
        pytest.param(lambda t: re.sub(r"    seal: .+\n", "", t), id="seal deleted"),
        pytest.param(lambda t: t.replace("  fingerprint:\n", "  fingerprint: null\n  fingerprint_old:\n"), id="emptied"),
    ],
)
def test_added_a_missing_or_hand_edited_fingerprint_is_stale_unreadable_and_never_rewritten(root, capsys, mangle):
    run(root, ex("a"), capsys=capsys)
    path = root / "sheet.yaml"
    path.write_text(mangle(path.read_text(encoding="utf-8")), encoding="utf-8")
    before = sheet_bytes(root)
    code, out, _err = run(root, ex("a"), capsys=capsys)
    assert code == 0 and "example a: stale (fingerprint unreadable)" in out
    assert sheet_bytes(root) == before
    code, out, _err = run(root, ex("a"), check=True, capsys=capsys)
    assert "stale (fingerprint unreadable)" in out.splitlines()[0] and sheet_bytes(root) == before


def test_added_a_sheet_entry_whose_id_is_not_in_the_examples_is_an_orphan_left_untouched(root, capsys):
    run(root, ex("a"), ex("gone"), capsys=capsys)
    label_like_dan(root)
    before, gone_span = sheet_bytes(root), spans(root)["gone"]
    code, out, _err = run(root, ex("a"), check=True, capsys=capsys)
    lines = out.splitlines()
    assert code == 0 and lines[1].startswith("gone (blind):") and lines[1].endswith("orphan (not in the examples file)")
    assert lines[-1] == "2 entries checked: 0 errors, 0 stale, 1 orphans"
    code, out, _err = run(root, ex("a"), capsys=capsys)  # a rerun leaves it, never deletes it
    assert code == 0 and sheet_bytes(root) == before and spans(root)["gone"] == gone_span


# --- F2: the entries' own bytes, whatever state they are in -------------------------------------------------------


def test_f2_stale_label_error_and_partly_labelled_entries_keep_their_exact_span_when_new_entries_are_added(root, capsys):
    run(root, ex("stale"), ex("broken"), ex("partial"), capsys=capsys)
    path = root / "sheet.yaml"
    text = path.read_text(encoding="utf-8")
    parts = re.split(r"(?m)^(?=- id: )", text)
    by_id = {p.split("\n", 1)[0].removeprefix("- id: "): p for p in parts if p.startswith("- id: ")}
    by_id["broken"] = by_id["broken"].replace("verdict: null", "verdict: MAYBE  # typo", 1)
    by_id["partial"] = by_id["partial"].replace("verdict: null", "verdict: UNVERIFIED", 1)  # a verdict with no flag yet: incomplete
    path.write_text("".join(by_id[i] for i in ("stale", "broken", "partial")), encoding="utf-8")
    before, old_spans = sheet_bytes(root), spans(root)
    code, out, err = run(root, ex("stale", window=(utc(8), utc(10))), ex("broken"), ex("partial"), ex("fresh"), capsys=capsys)
    assert code == 1  # the label error is an error; the new entry is still written
    assert "example broken, row 1" in err and "'MAYBE'" in err
    assert "example stale: stale (inputs)" in out and "1 new, 2 unchanged, 1 stale" in out
    assert sheet_bytes(root).startswith(before)
    now = spans(root)
    assert list(now) == ["stale", "broken", "partial", "fresh"]
    assert all(now[i] == old_spans[i] for i in old_spans)


def test_e7_label_errors_in_existing_entries_are_reported_and_new_entries_are_still_appended(root, capsys):
    run(root, ex("a"), capsys=capsys)
    path = root / "sheet.yaml"
    path.write_text(path.read_text(encoding="utf-8").replace("verdict: null", "verdict: MAYBE", 1), encoding="utf-8")
    before = sheet_bytes(root)
    code, out, err = run(root, ex("a"), ex("b"), capsys=capsys)
    assert code == 1
    assert err.count("\n") == 1 and err.startswith("example a, row 1 (2026-10-31T08:00:00Z): verdict 'MAYBE'")
    assert out.strip() == f"wrote {path}: 1 new, 1 unchanged, 0 stale, 0 skipped, 0 failed"
    assert sheet_bytes(root).startswith(before) and list(spans(root)) == ["a", "b"]


def test_e7_a_gap_between_existing_rows_is_reported_and_the_rerun_still_appends(root, capsys):
    run(root, ex("a"), capsys=capsys)
    path = root / "sheet.yaml"
    text = path.read_text(encoding="utf-8")
    path.write_text(text.replace("start: '2026-10-31T09:00:00Z'", "start: '2026-10-31T09:10:00Z'", 1), encoding="utf-8")
    code, _out, err = run(root, ex("a"), ex("b"), capsys=capsys)
    assert code == 1 and "gap" in err and list(spans(root)) == ["a", "b"]


# --- E7: a sheet that is not YAML ---------------------------------------------------------------------------------

BROKEN = "- id: a\n  mode: blind\n\tindent: tab\n"


@pytest.mark.parametrize("check", [False, True], ids=["rerun", "check"])
def test_e7_invalid_yaml_stops_the_command_with_the_line_number_and_leaves_the_file_untouched(root, capsys, check):
    path = root / "sheet.yaml"
    path.write_text(BROKEN, encoding="utf-8")
    before = (path.read_bytes(), path.stat().st_mtime_ns)
    code, out, err = run(root, ex("a"), ex("b"), check=check, capsys=capsys)
    assert code == 1 and out == ""
    assert err.count("\n") == 1 and err.startswith(f"{path}: invalid YAML at line 3:")
    assert (path.read_bytes(), path.stat().st_mtime_ns) == before


def test_e7_a_sheet_whose_indentation_was_broken_by_hand_names_its_line(root, capsys):
    run(root, ex("a"), capsys=capsys)
    path = root / "sheet.yaml"
    text = path.read_text(encoding="utf-8").replace("  mode: blind\n", " mode: blind\n", 1)  # one space short
    path.write_text(text, encoding="utf-8")
    before = path.read_bytes()
    for check in (False, True):
        code, _out, err = run(root, ex("a"), check=check, capsys=capsys)
        assert code == 1 and re.search(r"invalid YAML at line \d+", err)
    assert path.read_bytes() == before


# --- F2: examples.yaml is read only -------------------------------------------------------------------------------


def test_f2_the_examples_file_keeps_its_hash_and_mtime_through_a_build_a_rerun_and_a_check(root, capsys):
    path = examples_file(root / "examples.yaml", ex("a"), ex("b", blind=False))
    stamp = lambda: (hashlib.sha256(path.read_bytes()).hexdigest(), path.stat().st_mtime_ns)  # noqa: E731
    before = stamp()
    argv = ["--examples", str(path), "--out", str(root / "sheet.yaml")]
    for extra in ([], [], ["--check"]):
        main([*extra, *argv], root=root, environ={})
        capsys.readouterr()
        assert stamp() == before


def test_a_failing_example_leaves_an_existing_entry_for_it_alone_and_is_counted(root, capsys):
    run(root, ex("a"), capsys=capsys)
    before = sheet_bytes(root)
    moved = example("a", "c", now=NOW, window=(utc(8), utc(11)))
    (root / "fixtures" / "c" / "weather.json").unlink()  # the case can no longer be built
    code, out, err = run(root, moved, capsys=capsys)
    assert code == 1 and "example a" in err and "1 failed" in out and sheet_bytes(root) == before


def test_an_existing_entrys_case_folder_gone_is_never_captured_again(root, capsys, monkeypatch):
    from flysafelz import capture

    run(root, ex("a"), capsys=capsys)
    before = sheet_bytes(root)
    monkeypatch.setattr(capture, "capture", lambda *a, **k: pytest.fail("a case was captured for an existing entry"))
    moved = example("a", "new_case_name", now=NOW, window=WINDOW)
    code, out, err = run(root, moved, capsys=capsys)
    assert code == 0 and "example a: stale (inputs)" in out and sheet_bytes(root) == before


# --- F3: nothing of 5a in the sheet code ---------------------------------------------------------------------------


SHEET_CODE = [
    SRC / "verdict_sheet.py",
    SRC / "verdict_sheet_check.py",
    SRC / "verdict_sheet_draft.py",
    SRC / "verdict_sheet_fingerprint.py",
    Path(__file__).resolve().parents[2] / "scripts" / "build_verdict_sheet.py",
]


def test_f3_the_sheet_code_names_no_langsmith_model_client_or_eval_metric():
    text_pattern = re.compile(r"langsmith|openai(?!p)|anthropic", re.IGNORECASE)  # openaip, the airspace source, is not a model client
    code_pattern = re.compile(r"evaluat|upload|push|create_dataset|create_example|accuracy|f1_|precision|recall|metric", re.IGNORECASE)
    for path in SHEET_CODE:
        text = path.read_text(encoding="utf-8")
        assert text_pattern.search(text) is None, path.name
        names = []
        for node in ast.walk(ast.parse(text)):
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                names.append(node.name)
            elif isinstance(node, ast.Name):
                names.append(node.id)
            elif isinstance(node, ast.Attribute):
                names.append(node.attr)
            elif isinstance(node, ast.alias):
                names.append(node.name)
        assert not [n for n in names if code_pattern.search(n)], path.name


def test_f3_load_sheet_returns_data_only(tmp_path):
    from flysafelz.verdict_sheet_check import load_sheet

    path = tmp_path / "s.yaml"
    path.write_text("- id: x\n  mode: blind\n  window: default\n  actions_done: false\n  rows:\n    - {start: '2026-10-31T07:00:00Z', end: '2026-10-31T08:00:00Z'}\n", encoding="utf-8")
    loaded = load_sheet(path)
    assert type(loaded) is dict and all(type(v) is dict for v in loaded.values())
    assert set(loaded["x"]) == {"verdict_by_segment", "tier"}
