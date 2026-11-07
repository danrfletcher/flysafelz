"""G6: the per-entry fingerprint. One table over every input in the field list, for both modes, plus what is not an input.

fingerprint() is called with explicit Example, rows and a case folder; each case changes exactly one thing and the
names of the parts that move are compared with what the rule says. Nothing here builds a sheet or opens a socket.
"""

import ast
import dataclasses
from datetime import UTC, datetime
from pathlib import Path

import pytest

from flysafelz import verdict_sheet_fingerprint as fp
from flysafelz.verdict_sheet import Example

SRC = Path(fp.__file__)
NOW = datetime(2026, 10, 31, 7, 30, tzinfo=UTC)
WINDOW = (datetime(2026, 10, 31, 7, tzinfo=UTC), datetime(2026, 10, 31, 18, tzinfo=UTC))
ROWS = [
    {"start": "2026-10-31T07:00:00Z", "end": "2026-10-31T08:00:00Z", "edge": "window start", "verdict": None, "approaching_limits": None},
    {"start": "2026-10-31T08:00:00Z", "end": "2026-10-31T09:00:00Z", "edge": "hour", "verdict": None, "approaching_limits": None},
    {"start": "2026-10-31T09:00:00Z", "end": "2026-10-31T10:00:00Z", "edge": "hour", "tier": "OUTLOOK"},
]


def example(**changes) -> Example:
    base = Example(id="x", path="assess", blind=True, lat=53.8, lon=-0.08, now=NOW, window=WINDOW, case="a_case", drop=())
    return dataclasses.replace(base, **changes)


def with_row(index: int, **changes) -> list[dict]:
    rows = [dict(r) for r in ROWS]
    rows[index].update(changes)
    return rows


@pytest.fixture
def case_dir(tmp_path) -> Path:
    folder = tmp_path / "a_case"
    folder.mkdir()
    for name in fp.FIXTURE_FILES:
        (folder / name).write_bytes(f"original {name}\n".encode())
    return folder


def parts(case_dir, mode, *, ex=None, rows=None) -> dict[str, str]:
    return fp.fingerprint(ex or example(), rows or ROWS, case_dir, mode)


def moved(case_dir, mode, **how) -> list[str]:
    """Names of the parts that differ from the base fingerprint after the one change in `how` (ex=..., rows=..., or a file edit)."""
    base = parts(case_dir, mode)
    if "edit" in how:
        name, data = how["edit"]
        (case_dir / name).write_bytes(data)
    current = parts(case_dir, mode, ex=how.get("ex"), rows=how.get("rows"))
    return fp.changed_parts(base, current)


BOTH_INPUTS = ([fp.PART_INPUTS], [fp.PART_INPUTS])
BOTH_ROWS = ([fp.PART_ROWS], [fp.PART_ROWS])
DRAFT_ONLY = lambda name: ([], [name])  # noqa: E731

# field changed alone -> (parts that move in blind mode, parts that move in draft mode)
TABLE = {
    "lat": (dict(ex=example(lat=53.9)), *BOTH_INPUTS),
    "lon": (dict(ex=example(lon=-0.09)), *BOTH_INPUTS),
    "now": (dict(ex=example(now=NOW.replace(minute=45))), *BOTH_INPUTS),
    "window end": (dict(ex=example(window=(WINDOW[0], WINDOW[1].replace(hour=17)))), *BOTH_INPUTS),
    "window start (07:30 as written, not 07:00)": (dict(ex=example(window=(WINDOW[0].replace(minute=30), WINDOW[1]))), *BOTH_INPUTS),
    "window null instead of set": (dict(ex=example(window=None)), *BOTH_INPUTS),
    "reference.path": (dict(ex=example(path="explain")), *BOTH_INPUTS),
    "fixtures.drop": (dict(ex=example(drop=("weather",))), *BOTH_INPUTS),
    "case name": (dict(ex=example(case="b_case")), *BOTH_INPUTS),
    "row start": (dict(rows=with_row(1, start="2026-10-31T08:30:00Z")), *BOTH_ROWS),
    "row end": (dict(rows=with_row(1, end="2026-10-31T09:30:00Z")), *BOTH_ROWS),
    "row tier": (dict(rows=with_row(2, tier="TOO_FAR")), *BOTH_ROWS),
    "an extra row": (dict(rows=[*ROWS, {"start": "2026-10-31T10:00:00Z", "end": "2026-10-31T11:00:00Z", "edge": "hour"}]), *BOTH_ROWS),
    "weather.json bytes": (dict(edit=("weather.json", b"changed\n")), *DRAFT_ONLY("weather.json")),
    "airspace.json bytes": (dict(edit=("airspace.json", b"changed\n")), *DRAFT_ONLY("airspace.json")),
    "pib.xml bytes": (dict(edit=("pib.xml", b"changed\n")), *DRAFT_ONLY("pib.xml")),
    "readings.yaml bytes": (dict(edit=("readings.yaml", b"changed\n")), *DRAFT_ONLY("readings.yaml")),
    # not inputs: nothing may move
    "id": (dict(ex=example(id="renamed")), [], []),
    "blind flag alone (the stored mode carries it)": (dict(ex=example(blind=False)), [], []),
    "row edge name": (dict(rows=with_row(1, edge="J1/26 starts")), [], []),
    "a label typed into a row": (dict(rows=with_row(0, verdict="NO_GO", approaching_limits=True)), [], []),
    "key order within a row": (dict(rows=[dict(reversed(list(r.items()))) for r in ROWS]), [], []),
}


@pytest.mark.parametrize("name", TABLE)
def test_g6_each_input_changed_alone_moves_only_the_parts_it_belongs_to(name, tmp_path):
    change, in_blind, in_draft = TABLE[name]
    for mode, expected in (("blind", in_blind), ("draft", in_draft)):
        folder = tmp_path / mode / "a_case"
        folder.mkdir(parents=True)
        for file in fp.FIXTURE_FILES:
            (folder / file).write_bytes(f"original {file}\n".encode())
        assert moved(folder, mode, **change) == expected, (name, mode)


def test_the_table_covers_every_field_of_the_g6_list():
    named = " ".join(TABLE)
    for field in ("lat", "lon", "now", "window", "reference.path", "fixtures.drop", "case name", "row start", "row end", "row tier"):
        assert field in named
    for file in fp.FIXTURE_FILES:
        assert file in named


def test_a_blind_fingerprint_has_two_parts_and_a_draft_one_adds_the_four_fixture_files(case_dir):
    assert list(parts(case_dir, "blind")) == [fp.PART_INPUTS, fp.PART_ROWS]
    assert list(parts(case_dir, "draft")) == [fp.PART_INPUTS, fp.PART_ROWS, *fp.FIXTURE_FILES]
    assert fp.FIXTURE_FILES == ("weather.json", "airspace.json", "pib.xml", "readings.yaml")


def test_blind_ignores_fixture_bytes_and_draft_hashes_them_raw(case_dir):
    blind = parts(case_dir, "blind")
    (case_dir / "readings.yaml").write_bytes(b"completely different\n")
    (case_dir / "weather.json").write_bytes(b"{}")
    assert parts(case_dir, "blind") == blind
    draft = parts(case_dir, "draft")
    (case_dir / "readings.yaml").write_bytes(b"completely different \n")  # one trailing space: still a different hash
    assert fp.changed_parts(draft, parts(case_dir, "draft")) == ["readings.yaml"]


def test_the_shared_parts_are_the_same_in_both_modes(case_dir):
    blind, draft = parts(case_dir, "blind"), parts(case_dir, "draft")
    assert {k: draft[k] for k in blind} == blind


def test_a_fixture_file_the_case_does_not_have_is_absent_and_not_the_hash_of_an_empty_file(tmp_path):
    folder = tmp_path / "c"
    folder.mkdir()
    (folder / "readings.yaml").write_bytes(b"")
    found = fp.file_parts(folder)
    assert found["weather.json"] == fp.ABSENT and found["readings.yaml"] != fp.ABSENT
    assert fp.is_readable(parts(folder, "draft"), "draft")


def test_the_rules_yaml_and_the_code_version_are_not_inputs(case_dir):
    before = parts(case_dir, "draft")
    (case_dir / "rules.yaml").write_text("wind_10m_no_go: {value: 99}\n")  # beside the fixtures: not one of the four files
    assert parts(case_dir, "draft") == before
    tree = ast.parse(SRC.read_text(encoding="utf-8"))
    imported = [n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)] + [a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names]
    assert not any("rules" in (m or "") or "version" in (m or "") for m in imported)


@pytest.mark.parametrize(
    "stored, readable",
    [
        ({fp.PART_INPUTS: "0" * 64, fp.PART_ROWS: "f" * 64}, True),
        ({fp.PART_INPUTS: "0" * 64}, False),  # a part missing
        ({fp.PART_INPUTS: "0" * 64, fp.PART_ROWS: "f" * 64, "extra": "0" * 64}, False),
        ({fp.PART_INPUTS: "0" * 64, fp.PART_ROWS: "xyz"}, False),
        ({fp.PART_INPUTS: "0" * 64, fp.PART_ROWS: 5}, False),
        ({fp.PART_INPUTS: "0" * 64, fp.PART_ROWS: "F" * 64}, False),  # not the lower-case hex the script writes
        ("abc", False),
        (None, False),
    ],
)
def test_is_readable_for_a_blind_entry(stored, readable):
    assert fp.is_readable(stored, "blind") is readable


def test_is_readable_wants_the_draft_parts_for_a_draft_entry_and_knows_no_other_mode(case_dir):
    assert fp.is_readable(parts(case_dir, "draft"), "draft")
    assert not fp.is_readable(parts(case_dir, "blind"), "draft")
    assert not fp.is_readable(parts(case_dir, "draft"), "blind")
    assert not fp.is_readable(parts(case_dir, "blind"), "maybe") and not fp.is_readable(parts(case_dir, "blind"), None)


def test_unseal_gives_the_parts_back_and_refuses_a_part_edited_to_another_valid_hash(case_dir):
    for mode in ("blind", "draft"):
        original = parts(case_dir, mode)
        stored = fp.sealed(original)
        assert list(stored) == [*original, fp.SEAL] and fp.unseal(stored, mode) == original
        for name in original:
            edited = {**stored, name: ("0" if stored[name][0] != "0" else "1") + stored[name][1:]}
            assert fp.is_readable({k: v for k, v in edited.items() if k != fp.SEAL}, mode)  # still the right shape
            assert fp.unseal(edited, mode) is None
    blind = fp.sealed(parts(case_dir, "blind"))
    assert fp.unseal({k: v for k, v in blind.items() if k != fp.SEAL}, "blind") is None  # no seal
    assert fp.unseal(blind, "draft") is None and fp.unseal("abc", "blind") is None and fp.unseal(None, "blind") is None
