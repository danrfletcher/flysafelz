"""The loader on the real committed dataset (5a PR-3.F1): counts, fixture folders, the stale check, full dev metadata, and nothing written.

Held-out is checked by count and case folder names only (F7): no held-out question, reference, judge note or example id is read into an assertion or
printed. Dev records are checked on shape, and no LangSmith call is made (the one upload runs against a RecordingClient).
"""

import hashlib
import os
from pathlib import Path

import pytest
import yaml

from evals.agent import loader
from evals.agent.loader import EXPECTED, REPO, check_dataset, load_dataset, load_examples, upload_dataset
from evals.agent.recording_client import RecordingClient

DATA = REPO / "evals" / "agent" / "data"
FIXTURES = REPO / "fixtures"
NINE = ["split", "case_type", "path", "failure_source", "notam_category", "airspace_context", "constructed", "compound", "parts"]


@pytest.fixture(autouse=True)
def no_langsmith_key(monkeypatch):
    monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)


@pytest.fixture(scope="module")
def raw():
    return yaml.safe_load((DATA / "examples.yaml").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def dev():
    return load_examples("dev")


def snapshot():
    """{path: (mtime_ns, sha256)} of the data files and every file under fixtures/."""
    files = [*sorted(DATA.glob("*.yaml")), *sorted(p for p in FIXTURES.rglob("*") if p.is_file())]
    return {str(p): (os.stat(p).st_mtime_ns, hashlib.sha256(p.read_bytes()).hexdigest()) for p in files}


def test_the_counts_are_29_15_and_24(raw):
    assert EXPECTED == {"dev": 29, "held_out": 15, "sheet": 24}
    assert sum(e["metadata"]["split"] == "dev" for e in raw) == 29
    assert sum(e["metadata"]["split"] == "held_out" for e in raw) == 15
    sheet = yaml.safe_load((DATA / "verdict_sheet.yaml").read_text(encoding="utf-8"))
    assert len(sheet) == 24 and len(sheet) == sum(e["reference"]["path"] == "assess" for e in raw)


def test_all_25_case_folders_are_present_with_a_case_yaml(raw):
    cases = {e["fixtures"]["case"] for e in raw}
    assert len(cases) == 25
    assert [c for c in sorted(cases) if not (FIXTURES / c / "case.yaml").is_file()] == []


def test_the_check_prints_the_counts_line_and_the_sheet_is_finalised():
    assert check_dataset() == "dev 29 · held_out 15 · sheet 24 · fixtures 25/25 present · sheet finalised"


def test_dev_records_load_with_the_full_metadata(dev):
    assert len(dev) == 29
    for record in dev:
        assert set(record) == {"id", "inputs", "fixtures", "reference", "metadata"}
        assert list(record["metadata"]) == NINE
        assert record["metadata"]["split"] == "dev"
        assert record["metadata"]["path"] == record["reference"]["path"]
        assert isinstance(record["metadata"]["compound"], bool) and isinstance(record["metadata"]["parts"], list)
        assert set(record["inputs"]) == {"lat", "lon", "question", "now"} and isinstance(record["inputs"]["now"], str)
        assert record["fixtures"]["case"] and isinstance(record["fixtures"]["drop"], list)
        assert (FIXTURES / record["fixtures"]["case"] / "case.yaml").is_file()


def test_every_assess_dev_record_has_the_sheets_labels_and_the_others_do_not(dev):
    assess = [r for r in dev if r["metadata"]["path"] == "assess"]
    assert assess and len(assess) < len(dev)
    for record in assess:
        for key in ("verdict_by_segment", "tier", "notam_actions"):
            assert key in record["reference"], (record["id"], key)
        assert isinstance(record["reference"]["verdict_by_segment"], list)  # may be empty for a window with no VERDICT-tier hour
    for record in dev:
        if record["metadata"]["path"] != "assess":
            assert not {"verdict_by_segment", "tier", "notam_actions"} & set(record["reference"])


def test_blue_mill_today_has_no_compound_in_the_file_and_loads_as_false(raw, dev):
    in_file = [e for e in raw if e["id"] == "blue_mill_today"]
    assert len(in_file) == 1 and "compound" not in in_file[0]["metadata"]
    loaded = next(r for r in dev if r["id"] == "blue_mill_today")
    assert loaded["metadata"]["compound"] is False and loaded["metadata"]["parts"] == []


def test_the_frozen_now_is_the_files_text_not_the_clock(raw, dev):
    in_file = {e["id"]: e["inputs"]["now"] for e in raw if e["metadata"]["split"] == "dev"}
    assert {r["id"]: r["inputs"]["now"] for r in dev} == in_file
    assert all(isinstance(v, str) for v in in_file.values())


def test_judge_notes_come_through_on_the_dev_examples_that_have_them(raw, dev):
    wanted = {e["id"]: e["reference"]["judge_notes"] for e in raw if e["metadata"]["split"] == "dev" and "judge_notes" in e["reference"]}
    assert wanted
    assert {r["id"]: r["reference"]["judge_notes"] for r in dev if "judge_notes" in r["reference"]} == wanted


def test_held_out_is_checked_by_count_and_case_folder_only(raw):
    held = load_dataset("held_out")
    assert len(held.records) + len(held.refused) == 15
    held_cases = {e["fixtures"]["case"] for e in raw if e["metadata"]["split"] == "held_out"}
    assert held_cases and all((FIXTURES / c / "case.yaml").is_file() for c in held_cases)


def test_a_load_and_a_dev_upload_change_nothing_on_disk_and_never_read_readings():
    before = snapshot()
    load_examples("dev")
    client = RecordingClient()
    summary = upload_dataset("dev", client)
    assert summary.total == 29 and summary.created == 29 and list(client.datasets) == ["flysafelz-dev"]
    assert snapshot() == before


def test_the_loader_does_not_read_readings_itself():
    assert "readings" not in Path(loader.__file__).read_text(encoding="utf-8")
