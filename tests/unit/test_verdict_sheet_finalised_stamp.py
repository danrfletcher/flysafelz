"""5a PR-2.F1 G2 / E2: the `finalised` stamp of the verdict sheet: its parts, its seal, the module walk, and what makes it stale.

Staleness is tested through entry_problem() and finalised_problems()' own hashing, over a copy of the code (rules.yaml and the
modules the walk finds) in a tmp `code_root` and a copy of a case folder in a tmp root: the repository's own files are never touched.
"""

import ast
import copy
import hashlib
import shutil
from pathlib import Path

import pytest
import yaml

from flysafelz import verdict_sheet_finalise as fin
from flysafelz.verdict_sheet_fingerprint import FIXTURE_FILES, SEAL, sealed

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "src"
CASE = REPO / "fixtures" / "blue_mill_2026-10-31"
NOTAM_ACTIONS = {"J3857/26": "CONTACT_FIRST", "H7171/26": "IGNORE"}
ROWS = [{"start": "2026-10-31T07:00:00Z", "end": "2026-10-31T08:00:00Z", "verdict": None}, {"start": "2026-10-31T08:00:00Z", "end": "2026-10-31T09:00:00Z", "verdict": None}]


def entry() -> dict:
    return {"id": "e1", "mode": "blind", "actions_done": True, "notam_actions": dict(NOTAM_ACTIONS), "rows": copy.deepcopy(ROWS)}


@pytest.fixture
def world(tmp_path):
    """(code_root, case_dir): copies of the code the stamp covers and of a committed case."""
    code_root = tmp_path / "code"
    for relative in [*fin.module_files(), "flysafelz/rules.yaml"]:
        target = code_root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(SRC / relative, target)
    case_dir = tmp_path / "fixtures" / "c"
    shutil.copytree(CASE, case_dir)
    return code_root, case_dir


def stamped(world, data=None):
    code_root, case_dir = world
    data = data or entry()
    data[fin.STAMP_KEY] = fin.stamp_for(data, case_dir, fin.CodeHashes.read(code_root))
    return data


def problem(world, data):
    code_root, case_dir = world
    return fin.entry_problem(data, case_dir, fin.CodeHashes.read(code_root))


def append_byte(path: Path) -> None:
    with path.open("ab") as out:
        out.write(b"\n")


# --- the module walk ---------------------------------------------------------------------------------------------------------------------


def test_the_walk_finds_the_modules_assess_depends_on():
    found = set(fin.imported_modules())
    for name in ("weather", "daylight", "airspace", "notam_action", "notam_area", "schedule", "pib", "rules", "contracts", "assess"):
        assert f"flysafelz/{name}.py" in found, name
    assert "flysafelz/__init__.py" in found and "flysafelz/clients/open_meteo.py" in found  # a parent package and a sub-package module
    assert fin.SELF_MODULE in fin.module_files() and fin.SELF_MODULE not in found  # assess does not import the finaliser: it is added


def test_the_walk_is_made_from_the_imports_so_it_leaves_out_what_assess_does_not_import():
    found = set(fin.module_files())
    assert not any(name.startswith(("flysafelz/reader", "flysafelz/agent", "flysafelz/capture")) for name in found)
    assert "flysafelz/verdict_sheet.py" not in found  # the sheet code is not an input to assess


def test_the_walk_follows_every_kind_of_import_in_a_tmp_package(tmp_path):
    pkg = tmp_path / "flysafelz"
    (pkg / "sub").mkdir(parents=True)
    files = {
        "__init__.py": "",
        "assess.py": "import os\nimport httpx\nfrom flysafelz import a\nfrom flysafelz.b import thing\nfrom .sub import c\n",
        "a.py": "def f():\n    import flysafelz.late\n",
        "b.py": "from . import d\nfrom typing import TYPE_CHECKING\nif TYPE_CHECKING:\n    from flysafelz.typed import T\n",
        "d.py": "",
        "late.py": "",
        "typed.py": "",
        "unused.py": "",
        "sub/__init__.py": "",
        "sub/c.py": "from .. import e\nfrom ..f import g\n",
        "e.py": "",
        "f.py": "g = 1\n",
    }
    for name, text in files.items():
        (pkg / name).write_text(text, encoding="utf-8")
    got = fin.imported_modules(tmp_path)
    assert got == sorted(
        [
            "flysafelz/__init__.py",
            "flysafelz/a.py",
            "flysafelz/assess.py",
            "flysafelz/b.py",
            "flysafelz/d.py",
            "flysafelz/late.py",
            "flysafelz/typed.py",
            "flysafelz/sub/__init__.py",
            "flysafelz/sub/c.py",
            "flysafelz/e.py",
            "flysafelz/f.py",
        ]
    )


def test_a_module_added_to_assess_later_is_covered_without_an_edit(tmp_path):
    pkg = tmp_path / "flysafelz"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    (pkg / "assess.py").write_text("from flysafelz import new_rule\n", encoding="utf-8")
    (pkg / "new_rule.py").write_text("", encoding="utf-8")
    assert "flysafelz/new_rule.py" in fin.imported_modules(tmp_path)


def test_the_finalise_module_itself_imports_nothing_from_the_reader_capture_or_a_model():
    """F1: the AST of verdict_sheet_finalise.py (this also covers `capture.capture` never being called by it)."""
    tree = ast.parse(Path(fin.__file__).read_text(encoding="utf-8"))
    names = {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
    names |= {n.module or "" for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
    forbidden = ("flysafelz.reader", "flysafelz.capture", "flysafelz.agent", "httpx", "openai", "anthropic", "langchain", "langgraph", "langsmith", "requests")
    assert not [n for n in names if n.startswith(forbidden)]
    assert "capture" not in {a.name for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module == "flysafelz" for a in n.names}


# --- the stamp: its parts and its seal ---------------------------------------------------------------------------------------------------


def test_the_stamp_holds_rules_every_module_the_four_fixture_files_labels_and_a_seal(world):
    code_root, case_dir = world
    stamp = fin.stamp_for(entry(), case_dir, fin.CodeHashes.read(code_root))
    assert set(stamp) == {"rules", "modules", *FIXTURE_FILES, "labels", SEAL}
    assert stamp["rules"] == hashlib.sha256((SRC / "flysafelz" / "rules.yaml").read_bytes()).hexdigest()
    assert set(stamp["modules"]) == {f"src/{f}" for f in fin.module_files()}
    for name in FIXTURE_FILES:
        assert stamp[name] == hashlib.sha256((case_dir / name).read_bytes()).hexdigest()
    assert stamp[SEAL] == sealed({k: v for k, v in stamp.items() if k != SEAL})[SEAL]
    assert fin.read_stamp(stamp) is not None


def test_the_stamp_is_what_the_repository_code_gives_when_no_code_root_is_named():
    code = fin.CodeHashes.read()
    assert code.rules == hashlib.sha256((SRC / "flysafelz" / "rules.yaml").read_bytes()).hexdigest()
    assert code.modules["src/flysafelz/weather.py"] == hashlib.sha256((SRC / "flysafelz" / "weather.py").read_bytes()).hexdigest()


def test_the_labels_part_follows_notam_actions_and_the_row_edges_but_not_the_verdicts():
    base = fin.labels_part(entry())
    changed_action = entry()
    changed_action["notam_actions"]["H7171/26"] = "NO_FLY"
    changed_edge = entry()
    changed_edge["rows"][0]["end"] = "2026-10-31T08:01:00Z"
    labelled = entry()
    labelled["rows"][0]["verdict"] = "NO_GO"
    null_map = entry()
    null_map["notam_actions"] = None
    assert fin.labels_part(changed_action) != base and fin.labels_part(changed_edge) != base and fin.labels_part(null_map) != base
    assert fin.labels_part(labelled) == base  # the verdicts are what finalise writes: they are not part of what it read
    assert fin.labels_part(entry()) == base


def test_a_stamp_is_readable_only_whole_and_sealed(world):
    good = stamped(world)[fin.STAMP_KEY]
    assert fin.read_stamp(good) is not None
    assert fin.read_stamp(None) is None and fin.read_stamp("text") is None and fin.read_stamp({}) is None
    no_seal = {k: v for k, v in good.items() if k != SEAL}
    assert fin.read_stamp(no_seal) is None
    missing_part = {k: v for k, v in good.items() if k != "labels"}
    assert fin.read_stamp(missing_part) is None
    extra = {**good, "extra": "0" * 64}
    assert fin.read_stamp(extra) is None
    bad_hash = copy.deepcopy(good)
    bad_hash["rules"] = "not a hash"
    assert fin.read_stamp(bad_hash) is None


# --- E2: every way a stamp is refused ------------------------------------------------------------------------------------------------------


def test_a_fresh_stamp_passes(world):
    assert problem(world, stamped(world)) is None


def test_an_entry_with_no_stamp_is_refused_with_the_rerun_message(world):
    assert problem(world, entry()) == "verdict sheet is stale (finalised stamp missing): run make dataset"


def test_a_stamp_with_a_hand_edited_hash_is_refused_as_unreadable(world):
    data = stamped(world)
    data[fin.STAMP_KEY]["rules"] = "0" * 64  # another valid hash, so only the seal can tell
    assert problem(world, data) == "verdict sheet is stale (finalised stamp unreadable): run make dataset"
    data = stamped(world)
    data[fin.STAMP_KEY]["modules"]["src/flysafelz/weather.py"] = "f" * 64
    assert problem(world, data) == "verdict sheet is stale (finalised stamp unreadable): run make dataset"
    data = stamped(world)
    data[fin.STAMP_KEY] = "gibberish"
    assert problem(world, data) == "verdict sheet is stale (finalised stamp unreadable): run make dataset"


def test_one_byte_more_in_rules_yaml_is_refused_naming_rules_yaml(world):
    data = stamped(world)
    append_byte(world[0] / "flysafelz" / "rules.yaml")
    assert problem(world, data) == "verdict sheet is stale (rules.yaml changed): run make dataset"


@pytest.mark.parametrize("module", ["weather", "daylight", "airspace", "notam_action", "notam_area", "schedule", "pib", "rules", "assess", "contracts"])
def test_one_byte_more_in_an_imported_module_is_refused_naming_its_path(world, module):
    data = stamped(world)
    append_byte(world[0] / "flysafelz" / f"{module}.py")
    assert problem(world, data) == f"verdict sheet is stale (src/flysafelz/{module}.py changed): run make dataset"


def test_one_byte_more_in_the_finalise_module_is_refused_naming_it(world):
    data = stamped(world)
    append_byte(world[0] / "flysafelz" / "verdict_sheet_finalise.py")
    assert problem(world, data) == "verdict sheet is stale (src/flysafelz/verdict_sheet_finalise.py changed): run make dataset"


@pytest.mark.parametrize("name", FIXTURE_FILES)
def test_one_byte_more_in_each_fixture_file_is_refused_naming_the_file(world, name):
    data = stamped(world)
    append_byte(world[1] / name)
    assert problem(world, data) == f"verdict sheet is stale ({name} changed): run make dataset"


def test_a_fixture_file_taken_away_is_refused_naming_it(world):
    data = stamped(world)
    (world[1] / "readings.yaml").unlink()
    assert problem(world, data) == "verdict sheet is stale (readings.yaml changed): run make dataset"


def test_a_hand_edit_of_dans_notam_action_after_finalise_is_refused_naming_labels(world):
    data = stamped(world)
    data["notam_actions"]["H7171/26"] = "NO_FLY"
    assert problem(world, data) == "verdict sheet is stale (labels changed): run make dataset"


def test_a_row_edge_moved_after_finalise_is_refused_naming_labels(world):
    data = stamped(world)
    data["rows"][1]["start"] = "2026-10-31T08:01:00Z"
    assert problem(world, data) == "verdict sheet is stale (labels changed): run make dataset"


def test_a_verdict_typed_into_a_row_after_finalise_is_not_a_stale_stamp(world):
    """Only what finalise read is sealed. (The row's own value is finalise's output; a rerun puts it back.)"""
    data = stamped(world)
    data["rows"][0]["verdict"] = "NO_GO"
    assert problem(world, data) is None


def test_several_parts_changed_are_all_named_in_the_order_rules_modules_files_labels(world):
    data = stamped(world)
    append_byte(world[0] / "flysafelz" / "rules.yaml")
    append_byte(world[0] / "flysafelz" / "weather.py")
    append_byte(world[1] / "pib.xml")
    data["notam_actions"]["H7171/26"] = "NO_FLY"
    assert problem(world, data) == "verdict sheet is stale (rules.yaml, src/flysafelz/weather.py, pib.xml, labels changed): run make dataset"


def test_the_stamp_is_stored_as_plain_yaml_and_comes_back_equal(world):
    data = stamped(world)
    again = yaml.safe_load(yaml.safe_dump({"finalised": data[fin.STAMP_KEY]}))["finalised"]
    assert again == data[fin.STAMP_KEY] and fin.read_stamp(again) is not None
