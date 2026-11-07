"""GP4 / E8 / F2 / F4 / F7: the draft entry for Blue Mill, built offline from the committed fixture case.

The draft is built and written the way the command does it: verdict_sheet.main() over fixtures/ under tmp_path (a copy
of the committed Blue Mill case, or a synthetic one), then the written file is read back with yaml.safe_load and as
text. No network, no key. Inputs are loaded through flysafelz.cases.load_case_inputs.
"""

import shutil
from datetime import UTC, datetime
from pathlib import Path

import cases
import pytest
import yaml
from factories import circle_at, notam, reading, square_at
from verdict_sheet_helpers import example, examples_file, write_case

from flysafelz.cases import load_case_inputs
from flysafelz.contracts import NotamAction, Tier, Verdict
from flysafelz.geo import circle_edge_distance_nm
from flysafelz.verdict_sheet import SheetError, build_draft_entry, local_notam_ids, main, read_examples
from flysafelz.verdict_sheet_draft import IGNORED_HEADER, UNREAD_NOTE

GP1 = Path(__file__).resolve().parents[1] / "data" / "verdict_sheet" / "examples_blue_mill.yaml"
NOW = datetime(2026, 10, 31, 7, 30, tzinfo=UTC)
LAT, LON = 53.802797, -0.086644
ENTRY_KEYS = {"id", "mode", "window", "actions_done", "notam_actions", "rows"}


def utc(day, hour, minute=0, month=10):
    return datetime(2026, month, day, hour, minute, tzinfo=UTC)


def draft_examples(tmp_path: Path, name: str = "examples.yaml") -> Path:
    """A copy of the GP1 example with `metadata.blind` removed."""
    (raw,) = yaml.safe_load(GP1.read_text(encoding="utf-8"))
    del raw["metadata"]["blind"]
    path = tmp_path / name
    path.write_text(yaml.safe_dump([raw], sort_keys=False), encoding="utf-8")
    return path


def blue_mill_root(tmp_path: Path) -> Path:
    shutil.copytree(cases.case_dir(cases.BLUE_MILL), tmp_path / "fixtures" / cases.BLUE_MILL)
    return tmp_path


def build(tmp_path: Path, examples: Path, *args: str, capsys=None) -> tuple[str, list[dict], str]:
    """Run main() in-process over tmp_path/fixtures: (the written text, its parse, stdout)."""
    out = tmp_path / "sheet.yaml"
    assert main([*args, "--examples", str(examples), "--out", str(out)], root=tmp_path, environ={}) == 0
    text = out.read_text(encoding="utf-8")
    return text, yaml.safe_load(text), capsys.readouterr().out if capsys else ""


@pytest.fixture
def draft(tmp_path, capsys):
    text, sheet, out = build(blue_mill_root(tmp_path), draft_examples(tmp_path), capsys=capsys)
    (entry,) = sheet
    return text, entry, out


def block_lines(text: str) -> list[str]:
    lines = text.splitlines()
    start = lines.index(f"    {IGNORED_HEADER}")
    block = []
    for line in lines[start + 1 :]:
        if not line.startswith("    # "):
            break
        block.append(line[6:])
    return block


# --- GP4 -------------------------------------------------------------------------------------------------------


def test_gp4_blue_mill_draft_has_mode_draft_13_rows_with_pr3_boundaries_and_actions_done_false(draft):
    _text, entry, _out = draft
    assert set(entry) == ENTRY_KEYS
    assert (entry["id"], entry["mode"], entry["actions_done"]) == ("blue_mill_2026-10-31", "draft", False)
    assert entry["window"] == {"start": "2026-10-31T07:00:00Z", "end": "2026-10-31T18:00:00Z"}
    rows = entry["rows"]
    assert [(r["start"][11:16], r["end"][11:16]) for r in rows] == [
        *[(f"{h:02d}:00", f"{h + 1:02d}:00") for h in range(7, 16)],
        ("16:00", "16:28"),
        ("16:28", "17:00"),
        ("17:00", "17:06"),
        ("17:06", "18:00"),
    ]
    assert [r["edge"] for r in rows][1] == "H7171/26 starts + J3857/26 starts" and rows[-1]["edge"] == "civil dusk"


def test_gp4_notam_actions_are_contact_first_then_avoid_sector(draft):
    _text, entry, _out = draft
    assert list(entry["notam_actions"].items()) == [("J3857/26", "CONTACT_FIRST"), ("H7171/26", "AVOID_SECTOR")]


def test_gp4_the_rows_values_are_those_of_the_same_assess_call_the_tests_make_everywhere_else(draft):
    _text, entry, _out = draft
    segments = [s for h in cases.run(cases.BLUE_MILL).hours for s in h.segments]
    assert len(segments) == len(entry["rows"]) == 13
    for segment, row in zip(segments, entry["rows"], strict=True):
        assert (row["verdict"], row["approaching_limits"]) == (segment.verdict.value, segment.approaching_limits)
        assert "unread" not in row


def test_gp4_the_17_00_and_17_06_rows_have_verdicts_and_why_lines_in_the_text(draft):
    text, entry, _out = draft
    rows = {r["start"][11:16]: r for r in entry["rows"]}
    assert (rows["17:00"]["verdict"], rows["17:06"]["verdict"]) == ("PILOT_DISCRETION", "NO_GO")
    lines = text.splitlines()
    for start in ("17:00", "17:06"):
        at = lines.index(f"  - start: '2026-10-31T{start}:00Z'")
        assert lines[at + 5].startswith("    # why: ")
    assert "    # why: night: outside civil twilight; " in text


def test_gp4_every_labelable_row_has_exactly_one_why_comment_and_none_is_a_key(draft):
    text, entry, _out = draft
    assert text.count("# why: ") == len(entry["rows"]) == 13
    assert all(set(r) == {"start", "end", "edge", "verdict", "approaching_limits"} for r in entry["rows"])


def test_gp4_the_block_has_j3864_and_v32_with_code_s_reasons_and_its_count_matches_the_fixture(draft):
    text, entry, _out = draft
    lines = block_lines(text)
    assert "J3864/26: starts 2026-11-01T08:00:00Z, at or after the window end 2026-10-31T18:00:00Z" in lines
    assert any(line.startswith("V32/26: ") and "series V subject RO" in line for line in lines)
    _case, _forecast, _airspace, pib, _readings = cases.load(cases.BLUE_MILL)
    near = {n.id for n in pib.notams if circle_edge_distance_nm(LAT, LON, n.centre_lat, n.centre_lon, n.radius_nm) <= 2}
    expected = near - set(entry["notam_actions"]) - {s.notam_id for s in pib.superseded}
    print(f"ignored-by-code block: {len(lines)} lines (the plan estimates 18); {len(near)} NOTAMs within 2 NM, {len(entry['notam_actions'])} with an action")
    assert len(lines) == len(expected)
    assert [line.split(":")[0] for line in lines] == sorted(expected)
    assert not set(entry["notam_actions"]) & {line.split(":")[0] for line in lines}


def test_gp4_the_block_is_below_notam_actions_and_above_rows_and_actions_done_is_false(draft):
    text, _entry, _out = draft
    assert text.index("    J3857/26: CONTACT_FIRST\n    H7171/26: AVOID_SECTOR\n") < text.index(IGNORED_HEADER) < text.index("  rows:\n")
    assert "  actions_done: false\n" in text


def test_gp4_reparsing_gives_the_expected_keys_and_only_enum_values(draft):
    _text, entry, _out = draft
    assert set(entry) == ENTRY_KEYS
    assert all(Verdict(r["verdict"]) and isinstance(r["approaching_limits"], bool) for r in entry["rows"])
    assert all(NotamAction(v) for v in entry["notam_actions"].values())


def test_f4_after_loading_no_row_or_entry_has_a_key_with_why_reason_or_ignored(draft):
    _text, entry, _out = draft
    keys = set(entry) | {k for r in entry["rows"] for k in r}
    assert not [k for k in keys if any(w in k for w in ("why", "reason", "ignored"))]


def test_f7_the_written_sheet_and_stdout_hold_no_recheck_text(draft):
    text, _entry, out = draft
    for shown in (text, out):
        assert "re-check" not in shown.lower() and "recheck" not in shown.lower()


def test_the_command_prints_the_one_summary_line_only(draft, tmp_path):
    _text, _entry, out = draft
    assert out.strip() == f"wrote {tmp_path / 'sheet.yaml'}: 1 built, 0 skipped, 0 failed"


def test_the_written_draft_equals_the_built_entry(tmp_path):
    root = blue_mill_root(tmp_path)
    (ex,) = read_examples(draft_examples(tmp_path))
    built = build_draft_entry(ex, root / "fixtures" / cases.BLUE_MILL)
    text, (parsed,), _out = build(root, draft_examples(tmp_path, "again.yaml"))
    assert parsed == dict(built) and list(parsed) == list(built)


# --- blind entries stay as they were (critique 3) -------------------------------------------------------------------


def test_g3_a_blind_blue_mill_entry_has_no_unread_key_and_no_comment_derived_from_code(tmp_path):
    text, (entry,), _out = build(blue_mill_root(tmp_path), GP1)
    assert entry["mode"] == "blind" and "unread" not in entry and all("unread" not in r for r in entry["rows"])
    assert "# why" not in text and "# ignored by code" not in text and "#" not in text
    assert entry["actions_done"] is False


def test_cq2_the_blind_and_the_draft_entry_both_carry_actions_done_false(tmp_path):
    root = blue_mill_root(tmp_path)
    (raw,) = yaml.safe_load(GP1.read_text(encoding="utf-8"))
    twin = {**raw, "id": "blue_mill_draft", "metadata": {}}
    _text, sheet, _out = build(root, examples_file(tmp_path / "mixed.yaml", raw, twin))
    assert [(e["mode"], e["actions_done"]) for e in sheet] == [("blind", False), ("draft", False)]


def test_a_mixed_file_writes_one_entry_of_each_mode_and_neither_has_the_others_keys(tmp_path):
    root = blue_mill_root(tmp_path)
    (raw,) = yaml.safe_load(GP1.read_text(encoding="utf-8"))
    twin = {**raw, "id": "blue_mill_draft", "metadata": {}}
    text, (blind, drafted), _out = build(root, examples_file(tmp_path / "mixed.yaml", raw, twin))
    assert (blind["id"], blind["mode"], drafted["id"], drafted["mode"]) == ("blue_mill_2026-10-31", "blind", "blue_mill_draft", "draft")
    assert set(blind) == set(drafted) == ENTRY_KEYS
    assert len(blind["notam_actions"]) == 20 and all(v is None for v in blind["notam_actions"].values())
    assert all(set(r) == {"start", "end", "edge", "verdict", "approaching_limits"} and r["verdict"] is None and r["approaching_limits"] is None for r in blind["rows"])
    assert all(r["verdict"] is not None for r in drafted["rows"]) and len(drafted["notam_actions"]) == 2
    head, tail = text.split("- id: blue_mill_draft")
    assert "#" not in head and "# why" in tail and IGNORED_HEADER in tail


def test_f2_the_order_of_the_examples_does_not_change_which_entry_is_blind(tmp_path):
    root = blue_mill_root(tmp_path)
    (raw,) = yaml.safe_load(GP1.read_text(encoding="utf-8"))
    twin = {**raw, "id": "blue_mill_draft", "metadata": {}}
    for position, order in enumerate(((raw, twin), (twin, raw))):
        out = tmp_path / f"order{position}.yaml"
        examples = examples_file(tmp_path / "e.yaml", *order)
        assert main(["--examples", str(examples), "--out", str(out)], root=root, environ={}) == 0
        modes = {e["id"]: e["mode"] for e in yaml.safe_load(out.read_text(encoding="utf-8"))}
        assert modes == {"blue_mill_2026-10-31": "blind", "blue_mill_draft": "draft"}
        blind = next(e for e in yaml.safe_load(out.read_text(encoding="utf-8")) if e["mode"] == "blind")
        assert all(r["verdict"] is None for r in blind["rows"]) and all(v is None for v in blind["notam_actions"].values())


def test_f2_the_examples_file_keeps_its_bytes_after_a_draft_build(tmp_path):
    root = blue_mill_root(tmp_path)
    examples = draft_examples(tmp_path)
    before = (examples.read_bytes(), examples.stat().st_mtime_ns)
    build(root, examples)
    assert (examples.read_bytes(), examples.stat().st_mtime_ns) == before


def test_flags_build_only_their_own_mode_and_count_the_rest_as_skipped(tmp_path, capsys):
    root = blue_mill_root(tmp_path)
    (raw,) = yaml.safe_load(GP1.read_text(encoding="utf-8"))
    twin = {**raw, "id": "blue_mill_draft", "metadata": {}}
    examples = examples_file(tmp_path / "mixed.yaml", raw, twin)
    _text, sheet, out = build(root, examples, "--draft", capsys=capsys)
    assert [e["mode"] for e in sheet] == ["draft"] and out.strip().endswith("1 built, 1 skipped, 0 failed")
    (tmp_path / "sheet.yaml").unlink()
    _text, sheet, out = build(root, examples, "--blind", capsys=capsys)
    assert [e["mode"] for e in sheet] == ["blind"] and out.strip().endswith("1 built, 1 skipped, 0 failed")


# --- E8: a reading is missing ------------------------------------------------------------------------------------------


def blue_mill_without_j3857(tmp_path: Path) -> Path:
    root = blue_mill_root(tmp_path)
    readings = root / "fixtures" / cases.BLUE_MILL / "readings.yaml"
    kept = [r for r in yaml.safe_load(readings.read_text(encoding="utf-8")) if r["id"] != "J3857/26"]
    assert len(kept) == 1
    readings.write_text(yaml.safe_dump(kept), encoding="utf-8")
    return root


def test_e8_without_its_reading_j3857_is_unread_over_the_lz_with_the_comment_and_its_rows_are_tagged(tmp_path):
    root = blue_mill_without_j3857(tmp_path)
    text, (entry,), _out = build(root, draft_examples(tmp_path))
    assert entry["notam_actions"]["J3857/26"] == "UNREAD_OVER_LZ" and entry["notam_actions"]["H7171/26"] == "AVOID_SECTOR"
    assert list(entry["notam_actions"])[:2] == ["J3857/26", "H7171/26"]
    assert f"    J3857/26: UNREAD_OVER_LZ  {UNREAD_NOTE}\n" in text and text.count(UNREAD_NOTE) == 1
    unverified = [r for r in entry["rows"] if r["verdict"] == "UNVERIFIED"]
    assert len(unverified) == 9
    assert all(r["unread"] == ["J3857/26"] for r in unverified)
    assert all("unread" not in r for r in entry["rows"] if r["verdict"] != "UNVERIFIED")
    assert "    unread: [J3857/26]\n" in text and text.count("unread: [") == 9


def test_e8_the_blind_entry_of_the_same_case_has_no_tag_and_lists_j3857_with_a_null(tmp_path):
    root = blue_mill_without_j3857(tmp_path)
    text, (entry,), _out = build(root, GP1)
    assert "J3857/26" in entry["notam_actions"] and entry["notam_actions"]["J3857/26"] is None
    assert "unread" not in text and "UNREAD" not in text and all("unread" not in r for r in entry["rows"])
    assert all(v is None for v in entry["notam_actions"].values())


def test_q3_a_case_with_no_readings_yaml_drafts_with_no_readings_and_tags_every_unread_row(tmp_path):
    root = blue_mill_root(tmp_path)
    (root / "fixtures" / cases.BLUE_MILL / "readings.yaml").unlink()
    text, (entry,), _out = build(root, draft_examples(tmp_path))
    _case, _forecast, _airspace, pib, _readings = load_case_inputs(root / "fixtures" / cases.BLUE_MILL, readings_optional=True)
    assert _readings == {}
    assert entry["notam_actions"] == {"H7171/26": "UNREAD_OVER_LZ", "J3857/26": "UNREAD_OVER_LZ"}
    assert text.count(UNREAD_NOTE) == 2
    unverified = [r for r in entry["rows"] if r["verdict"] == "UNVERIFIED"]
    assert unverified and all(r["unread"] == ["H7171/26", "J3857/26"] for r in unverified)
    assert all("unread" not in r for r in entry["rows"] if r["verdict"] != "UNVERIFIED")
    assert local_notam_ids(pib, LAT, LON)


def test_q3_a_case_with_no_readings_yaml_is_still_an_error_for_a_blind_entry(tmp_path, capsys):
    root = blue_mill_root(tmp_path)
    (root / "fixtures" / cases.BLUE_MILL / "readings.yaml").unlink()
    assert main(["--examples", str(GP1), "--out", str(tmp_path / "s.yaml")], root=root, environ={}) == 1
    assert "readings.yaml" in capsys.readouterr().err and not (tmp_path / "s.yaml").exists()


def test_a_readings_yaml_that_is_there_but_invalid_is_an_error_for_a_draft(tmp_path):
    root = blue_mill_root(tmp_path)
    (root / "fixtures" / cases.BLUE_MILL / "readings.yaml").write_text("- not: a reading\n", encoding="utf-8")
    (ex,) = read_examples(draft_examples(tmp_path))
    with pytest.raises(SheetError, match="readings.yaml"):
        build_draft_entry(ex, root / "fixtures" / cases.BLUE_MILL)


# --- G5: tier-only rows in a draft ----------------------------------------------------------------------------------------------------


def test_g5_outlook_and_too_far_rows_in_a_draft_stay_tier_only(tmp_path):
    window = (utc(3, 20, month=11), utc(12, 0, month=11))
    write_case(tmp_path, "far", now=NOW, window=(utc(31, 7), utc(4, 0, month=11)), days=4)
    raw = example("far", "far", now=NOW, window=window, blind=False)
    text, (entry,), _out = build(tmp_path, examples_file(tmp_path / "far.yaml", raw))
    rows = entry["rows"]
    labelled = [r for r in rows if "tier" not in r]
    tiered = [r for r in rows if "tier" in r]
    assert labelled and len(tiered) == 192
    assert {r["tier"] for r in tiered} == {Tier.OUTLOOK.value, Tier.TOO_FAR.value}
    assert all(set(r) == {"start", "end", "edge", "tier"} for r in tiered)
    assert all({"verdict", "approaching_limits"} <= set(r) and "tier" not in r for r in labelled)
    assert text.count("# why: ") == len(labelled)  # none for a tier-only row
    assert entry["notam_actions"] == {}


# --- G4: which NOTAMs the block lists, through the real code ---------------------------------------------------------------------------


def test_g4_a_far_notam_is_in_neither_the_actions_nor_the_block_and_a_reading_geometry_ignore_is_in_the_block(tmp_path):
    near = notam("R1/26", circle=circle_at(1.0))
    far = notam("F1/26", circle=circle_at(5.0))
    shown = notam("S1/26", circle=circle_at(0.0))
    write_case(tmp_path, "g4", now=NOW, window=(utc(31, 8), utc(31, 11)), notams=(near, far, shown))
    readings = [reading(notam_id="R1/26", geometry=square_at(5.0)), reading(notam_id="S1/26")]
    (tmp_path / "fixtures" / "g4" / "readings.yaml").write_text(yaml.safe_dump([r.model_dump(mode="json") for r in readings]), encoding="utf-8")
    raw = example("g4", "g4", now=NOW, window=(utc(31, 8), utc(31, 11)), blind=False)
    text, (entry,), _out = build(tmp_path, examples_file(tmp_path / "g4.yaml", raw))
    assert list(entry["notam_actions"]) == ["S1/26"]
    lines = block_lines(text)
    assert [line.split(":")[0] for line in lines] == ["R1/26"] and "local area" in lines[0]
    assert "F1/26" not in text.replace("id: g4", "")
