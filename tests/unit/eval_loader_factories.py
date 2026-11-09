"""Synthetic datasets for the eval loader's tests (5a PR-3.F1). Everything here is invented: no dataset question, LZ, NOTAM id or case name.

    make_examples(...)       a list of example dicts: `dev` dev examples then `held_out` held-out ones, the first `assess_*` of each an assess example
    write_dataset(tmp, ...)  write examples.yaml, verdict_sheet.yaml (finalised stamps computed by the real stamp code) and fixtures/<case>/ under tmp;
                             returns Built(data_dir, root, examples, sheet)

The default is the real dataset's shape: 29 dev + 15 held-out examples, 16 + 8 = 24 assess ones with a sheet entry each. A test that wants a
different shape passes counts and its own `expected` to the loader. The one LZ point is the invented one test_lz_allow_list.py already names.
"""

from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any

import yaml

from flysafelz import verdict_sheet_finalise as fin

LAT, LON = 51.9, -1.1
NOW = "2026-01-02T03:04Z"  # not the frozen test clock: a loader that reads the clock would show it
START, END = "2026-01-02T04:00:00Z", "2026-01-02T06:00:00Z"
CASES = 5
FIXTURE_BYTES = {"weather.json": b'{"invented": "weather"}', "airspace.json": b'{"invented": "airspace"}', "pib.xml": b"<invented/>", "readings.yaml": b"invented: readings\n"}
TRICKY_QUESTION = "Can\u2019t I fly at M\u00fcller\u2019s field \u2014 it\u2019s \u201cfine\u201d, isn't it? 5\u00b0C, caf\u00e9 \u00e9t\u00e9"  # curly quotes, an apostrophe, non-ASCII


@cache
def code_hashes() -> fin.CodeHashes:
    """The real code hashes, read once: every stamp in a test run is over the same committed code."""
    return fin.CodeHashes.read()


def example_id(split: str, number: int) -> str:
    return f"inv_{'dev' if split == 'dev' else 'ho'}_{number:02d}"


def make_example(split: str, number: int, path: str = "assess", **changes: Any) -> dict[str, Any]:
    """One example in the file's shape. `changes` replace top-level sections whole (inputs=..., metadata=...)."""
    item: dict[str, Any] = {
        "id": example_id(split, number),
        "inputs": {"lat": LAT, "lon": LON, "now": NOW, "question": f"Invented question {split} {number}: is it flyable?"},
        "fixtures": {"case": f"inv_case_{number % CASES}"},
        "reference": {"path": path, "window": [START, END], "required_tools": ["assess"]} if path == "assess" else {"path": path},
        "metadata": {
            "split": split,
            "case_type": "plain",
            "failure_source": "none",
            "notam_category": None,
            "airspace_context": "none",
            "constructed": False,
            "blind": False,
            "compound": False,
        },
    }
    item.update(changes)
    return item


def make_examples(dev: int = 29, held_out: int = 15, assess_dev: int = 16, assess_held_out: int = 8) -> list[dict[str, Any]]:
    out = []
    for split, count, assess in (("dev", dev, assess_dev), ("held_out", held_out, assess_held_out)):
        for number in range(1, count + 1):
            out.append(make_example(split, number, "assess" if number <= assess else ("explain", "decline", "clarify")[number % 3]))
    return out


def write_case(root: Path, case: str, *, case_yaml: bool = True) -> Path:
    folder = root / "fixtures" / case
    folder.mkdir(parents=True, exist_ok=True)
    if case_yaml:
        (folder / "case.yaml").write_text(f"lat: {LAT}\nlon: {LON}\nnow: '{NOW}'\n", encoding="utf-8")
    for name, body in FIXTURE_BYTES.items():
        (folder / name).write_bytes(body)
    return folder


def sheet_entry(example: dict[str, Any], *, actions_done: bool = True, stamped_at: Path | None = None, notam_actions: dict[str, Any] | None = None) -> dict[str, Any]:
    """A verdict sheet entry for an assess example: two labelled rows, two NOTAM actions, and (when `stamped_at` is its case folder) a real stamp."""
    entry: dict[str, Any] = {
        "id": example["id"],
        "mode": "blind",
        "window": {"start": START, "end": END},
        "actions_done": actions_done,
        "notam_actions": notam_actions if notam_actions is not None else {"INV1/26": "NO_FLY", "INV2/26": "AWARENESS"},
        "rows": [
            {"start": START, "end": "2026-01-02T05:00:00Z", "verdict": "NO_GO", "approaching_limits": False},
            {"start": "2026-01-02T05:00:00Z", "end": END, "verdict": "PILOT_DISCRETION", "approaching_limits": True},
        ],
    }
    if stamped_at is not None and actions_done:
        entry["finalised"] = fin.stamp_for(entry, stamped_at, code_hashes())
    return entry


@dataclass
class Built:
    data_dir: Path
    root: Path
    examples: list[dict[str, Any]]
    sheet: list[dict[str, Any]]


def write_dataset(
    tmp_path: Path,
    examples: list[dict[str, Any]] | None = None,
    *,
    sheet: list[dict[str, Any]] | None = None,
    not_done: frozenset[str] = frozenset(),
    no_entry: frozenset[str] = frozenset(),
    no_case_yaml: frozenset[str] = frozenset(),
    no_folder: frozenset[str] = frozenset(),
    examples_text: str | None = None,
) -> Built:
    """Write the dataset under tmp_path (data in tmp_path/data, fixtures in tmp_path/fixtures).

    `sheet` replaces the generated entries; `not_done` ids get actions_done false; `no_entry` ids get no entry; `no_case_yaml` and `no_folder` are case names.
    `examples_text` replaces the examples file's text (for malformed input); the sheet is still made from `examples`.
    """
    examples = make_examples() if examples is None else examples
    data = tmp_path / "data"
    data.mkdir(parents=True, exist_ok=True)
    for case in sorted({e["fixtures"]["case"] for e in examples if isinstance(e.get("fixtures"), dict) and isinstance(e["fixtures"].get("case"), str)}):
        if case not in no_folder:
            write_case(tmp_path, case, case_yaml=case not in no_case_yaml)
    if sheet is None:
        sheet = [
            sheet_entry(e, actions_done=e["id"] not in not_done, stamped_at=tmp_path / "fixtures" / e["fixtures"].get("case", "inv_case_0"))
            for e in examples
            if e["reference"].get("path", "assess") == "assess" and e["id"] not in no_entry
        ]
    (data / "examples.yaml").write_text(examples_text if examples_text is not None else yaml.safe_dump(examples, sort_keys=False, allow_unicode=True), encoding="utf-8")
    (data / "verdict_sheet.yaml").write_text(yaml.safe_dump(sheet, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return Built(data, tmp_path, examples, sheet)
