"""GP6 / E13 / G4: load_sheet() against a hand-written sheet in tmp_path.

Three entries: `ready` (fully labelled, actions done), `untouched` (blind, actions_done: false, some rows labelled), and
`unread` (actions_done: true with an UNREAD_OVER_LZ left). Plus the shape of what comes back and what stops the load.
"""

import json
import sys
from pathlib import Path

import pytest

from flysafelz.verdict_sheet import load_sheet as load_sheet_from_the_sheet_module
from flysafelz.verdict_sheet_check import NotamActions, SheetError, load_sheet, load_sheet_skipping

SHEET = """\
# Dan's sheet
- id: ready
  mode: blind
  window: {start: '2026-10-31T07:30:00Z', end: '2026-10-31T10:00:00Z'}
  actions_done: true
  notam_actions:
    A1/26: CONTACT_FIRST
    B2/26: null
    C3/26: NO_FLY
  rows:
    - {start: '2026-10-31T07:00:00Z', end: '2026-10-31T08:00:00Z', edge: window start, verdict: NO_GO, approaching_limits: false}
    - {start: '2026-10-31T08:00:00Z', end: '2026-10-31T08:30:00Z', edge: hour, verdict: PILOT_DISCRETION, approaching_limits: true}
    - {start: '2026-10-31T08:30:00Z', end: '2026-10-31T09:00:00Z', edge: A1/26 starts, verdict: UNVERIFIED, approaching_limits: false}
    - {start: '2026-10-31T09:00:00Z', end: '2026-10-31T10:00:00Z', edge: hour, tier: OUTLOOK}

- id: untouched
  mode: blind
  window: {start: '2026-10-31T07:00:00Z', end: '2026-10-31T09:00:00Z'}
  actions_done: false
  notam_actions:
    A1/26: AVOID_SECTOR
    B2/26: null
  rows:
    - {start: '2026-10-31T07:00:00Z', end: '2026-10-31T08:00:00Z', edge: window start, verdict: NO_GO, approaching_limits: true}
    - {start: '2026-10-31T08:00:00Z', end: '2026-10-31T09:00:00Z', edge: hour, verdict: null, approaching_limits: null}

- id: unread
  mode: draft
  window: {start: '2026-10-31T07:00:00Z', end: '2026-10-31T08:00:00Z'}
  actions_done: true
  notam_actions:
    A1/26: UNREAD_OVER_LZ  # unread: read the NOTAM and set its action
    B2/26: AWARENESS
  rows:
    - {start: '2026-10-31T07:00:00Z', end: '2026-10-31T08:00:00Z', edge: window start, verdict: UNVERIFIED, approaching_limits: false}
"""


@pytest.fixture
def sheet(tmp_path) -> Path:
    path = tmp_path / "sheet.yaml"
    path.write_text(SHEET, encoding="utf-8")
    return path


def test_gp6_one_item_per_example_id_with_the_labelled_rows_in_the_section_4_shape(sheet):
    loaded = load_sheet(sheet)
    assert list(loaded) == ["ready", "untouched", "unread"]
    assert loaded["ready"]["verdict_by_segment"] == [
        {"start": "2026-10-31T07:00:00Z", "end": "2026-10-31T08:00:00Z", "verdict": "NO_GO", "approaching_limits": False},
        {"start": "2026-10-31T08:00:00Z", "end": "2026-10-31T08:30:00Z", "verdict": "PILOT_DISCRETION", "approaching_limits": True},
        {"start": "2026-10-31T08:30:00Z", "end": "2026-10-31T09:00:00Z", "verdict": "UNVERIFIED", "approaching_limits": False},
    ]
    assert all(set(r) == {"start", "end", "verdict", "approaching_limits"} for r in loaded["ready"]["verdict_by_segment"])


def test_gp6_tier_only_rows_load_as_tier_expectations_with_start_and_end(sheet):
    assert load_sheet(sheet)["ready"]["tier"] == [{"start": "2026-10-31T09:00:00Z", "end": "2026-10-31T10:00:00Z", "tier": "OUTLOOK"}]
    assert load_sheet(sheet)["untouched"]["tier"] == []
    assert all("tier" not in r for r in load_sheet(sheet)["ready"]["verdict_by_segment"])


def test_gp6_null_actions_load_as_ignore_and_a_notam_that_is_not_listed_is_ignore_too(sheet):
    actions = load_sheet(sheet)["ready"]["notam_actions"]
    assert actions == {"A1/26": "CONTACT_FIRST", "B2/26": "IGNORE", "C3/26": "NO_FLY"}
    assert isinstance(actions, NotamActions)
    assert actions["Z9/26"] == "IGNORE" and "Z9/26" not in actions  # asking never adds it
    assert actions.get("Z9/26") is None and len(actions) == 3


def test_added_actions_done_true_with_only_null_values_loads_every_listed_notam_as_ignore(tmp_path):
    path = tmp_path / "s.yaml"
    path.write_text(
        "- id: quiet\n  mode: blind\n  window: {start: '2026-10-31T07:00:00Z', end: '2026-10-31T08:00:00Z'}\n  actions_done: true\n"
        "  notam_actions: {A1/26: null, B2/26: null}\n  rows:\n    - {start: '2026-10-31T07:00:00Z', end: '2026-10-31T08:00:00Z', verdict: null, approaching_limits: null}\n",
        encoding="utf-8",
    )
    actions = load_sheet(path)["quiet"]["notam_actions"]
    assert actions == {"A1/26": "IGNORE", "B2/26": "IGNORE"} and actions["not/listed"] == "IGNORE"
    assert load_sheet(path)["quiet"]["verdict_by_segment"] == []


def test_added_actions_done_true_with_an_empty_notam_actions_loads_an_empty_mapping_that_still_ignores(tmp_path):
    path = tmp_path / "s.yaml"
    path.write_text(
        "- id: none\n  mode: blind\n  window: {start: '2026-10-31T07:00:00Z', end: '2026-10-31T08:00:00Z'}\n  actions_done: true\n  notam_actions: {}\n  rows:\n    - {start: '2026-10-31T07:00:00Z', end: '2026-10-31T08:00:00Z'}\n",
        encoding="utf-8",
    )
    assert load_sheet(path)["none"]["notam_actions"]["any/26"] == "IGNORE"


def test_e13_an_untouched_entry_returns_no_notam_actions_but_its_labelled_rows_still_load(sheet):
    entry = load_sheet(sheet)["untouched"]
    assert "notam_actions" not in entry
    assert entry["verdict_by_segment"] == [{"start": "2026-10-31T07:00:00Z", "end": "2026-10-31T08:00:00Z", "verdict": "NO_GO", "approaching_limits": True}]


def test_a_missing_actions_done_counts_as_not_done(tmp_path):
    path = tmp_path / "s.yaml"
    path.write_text("- id: x\n  mode: blind\n  window: default\n  notam_actions: {A1/26: NO_FLY}\n  rows:\n    - {start: '2026-10-31T07:00:00Z', end: '2026-10-31T08:00:00Z'}\n", encoding="utf-8")
    assert "notam_actions" not in load_sheet(path)["x"]


def test_added_an_unread_over_lz_left_with_actions_done_true_skips_that_entrys_actions(sheet):
    entry = load_sheet(sheet)["unread"]
    assert "notam_actions" not in entry
    assert entry["verdict_by_segment"][0]["verdict"] == "UNVERIFIED"  # its labelled row still loads
    assert "notam_actions" in load_sheet(sheet)["ready"]  # another entry is not affected


def test_an_incomplete_row_is_skipped(tmp_path):
    path = tmp_path / "s.yaml"
    path.write_text(
        "- id: x\n  mode: blind\n  window: {start: '2026-10-31T07:00:00Z', end: '2026-10-31T09:00:00Z'}\n  actions_done: false\n  rows:\n"
        "    - {start: '2026-10-31T07:00:00Z', end: '2026-10-31T08:00:00Z', verdict: NO_GO, approaching_limits: null}\n"
        "    - {start: '2026-10-31T08:00:00Z', end: '2026-10-31T09:00:00Z', verdict: null, approaching_limits: false}\n",
        encoding="utf-8",
    )
    assert load_sheet(path)["x"]["verdict_by_segment"] == []


def test_an_orphan_entry_still_loads(sheet):
    """load_sheet knows nothing of the examples file: an entry whose example has gone is data like any other."""
    assert "ready" in load_sheet(sheet)


def test_the_result_is_plain_data_that_json_can_carry(sheet):
    loaded = load_sheet(sheet)
    assert json.loads(json.dumps(loaded))["ready"]["notam_actions"] == {"A1/26": "CONTACT_FIRST", "B2/26": "IGNORE", "C3/26": "NO_FLY"}


def test_load_sheet_is_importable_from_the_sheet_module_without_running_the_script():
    assert load_sheet_from_the_sheet_module is load_sheet
    assert "build_verdict_sheet" not in sys.modules


def test_a_sheet_with_a_label_error_is_not_loaded_and_the_error_names_example_and_row(tmp_path):
    path = tmp_path / "s.yaml"
    path.write_text(SHEET.replace("verdict: NO_GO, approaching_limits: false", "verdict: MAYBE, approaching_limits: false"), encoding="utf-8")
    with pytest.raises(SheetError) as raised:
        load_sheet(path)
    assert "example ready, row 1 (2026-10-31T07:00:00Z)" in str(raised.value) and "'MAYBE'" in str(raised.value)


def test_a_gap_between_rows_stops_the_load(tmp_path):
    path = tmp_path / "s.yaml"
    path.write_text(SHEET.replace("start: '2026-10-31T08:30:00Z', end: '2026-10-31T09:00:00Z'", "start: '2026-10-31T08:40:00Z', end: '2026-10-31T09:00:00Z'"), encoding="utf-8")
    with pytest.raises(SheetError, match=r"example ready, row 3 \(2026-10-31T08:40:00Z\).*gap"):
        load_sheet(path)


def test_a_missing_file_and_invalid_yaml_raise_one_line_errors(tmp_path):
    with pytest.raises(SheetError) as missing:
        load_sheet(tmp_path / "absent.yaml")
    assert str(missing.value) == f"{tmp_path / 'absent.yaml'}: no such file"
    bad = tmp_path / "bad.yaml"
    bad.write_text("- id: x\n  rows: [\n", encoding="utf-8")
    with pytest.raises(SheetError, match=r"invalid YAML at line \d+") as invalid:
        load_sheet(bad)
    assert "\n" not in str(invalid.value)


def test_an_empty_sheet_loads_as_nothing(tmp_path):
    path = tmp_path / "empty.yaml"
    path.write_text("# nothing yet\n", encoding="utf-8")
    assert load_sheet(path) == {}


def test_load_sheet_skipping_loads_the_entries_that_validate_and_returns_the_errors_of_the_rest(sheet):
    text = sheet.read_text(encoding="utf-8").replace("edge: A1/26 starts, verdict: UNVERIFIED", "edge: A1/26 starts, verdict: MAYBE", 1)
    sheet.write_text(text, encoding="utf-8")
    with pytest.raises(SheetError):
        load_sheet(sheet)  # one typo stops the whole load
    loaded, errors = load_sheet_skipping(sheet)
    assert sorted(loaded) == ["unread", "untouched"]
    assert loaded["untouched"]["verdict_by_segment"][0]["verdict"] == "NO_GO"  # the same data load_sheet gives on a clean sheet
    assert len(errors) == 1 and "example ready, row 3" in errors[0] and "'MAYBE'" in errors[0]


def test_load_sheet_skipping_on_a_valid_sheet_is_load_sheet_with_no_errors(sheet):
    loaded, errors = load_sheet_skipping(sheet)
    assert errors == [] and loaded == load_sheet(sheet)


def test_load_sheet_skipping_still_refuses_a_missing_sheet(tmp_path):
    with pytest.raises(SheetError, match="no such file"):
        load_sheet_skipping(tmp_path / "nowhere.yaml")


def test_a_label_on_a_tier_only_row_does_not_stop_load_sheet_and_is_not_loaded(tmp_path):
    path = tmp_path / "s.yaml"
    path.write_text(
        "- id: x\n  mode: blind\n  window: {start: '2026-10-31T07:00:00Z', end: '2026-10-31T09:00:00Z'}\n  actions_done: false\n  rows:\n"
        "    - {start: '2026-10-31T07:00:00Z', end: '2026-10-31T08:00:00Z', verdict: NO_GO, approaching_limits: false}\n"
        "    - {start: '2026-10-31T08:00:00Z', end: '2026-10-31T09:00:00Z', tier: OUTLOOK, verdict: GO, approaching_limits: false}\n",
        encoding="utf-8",
    )
    entry = load_sheet(path)["x"]
    assert [r["verdict"] for r in entry["verdict_by_segment"]] == ["NO_GO"]
    assert entry["tier"] == [{"start": "2026-10-31T08:00:00Z", "end": "2026-10-31T09:00:00Z", "tier": "OUTLOOK"}]
