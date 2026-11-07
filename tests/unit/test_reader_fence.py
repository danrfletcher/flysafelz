"""Fence checks for the reader package (ticket 3 F2, F5, F6), and a planted violation for each ticket 2 test narrowed for it.

Ticket 2's fence tests now exempt `src/flysafelz/reader/` (and only it) from the LLM-import, `READER_`/`/models` and
`cache` scans. Each exemption is shown here to be exactly that wide: the same text planted in a copy of the source tree,
anywhere outside `reader/`, is still caught by the real scanning functions. Nothing is planted in the real tree.
"""

import ast
import re
import shutil
import tomllib
from pathlib import Path

import pytest
import yaml
from fence_scan import (
    OPENAIP_KEY_READERS,
    READER_CONFIG,
    banned_words,
    env_read_files,
    hard_coded_rule_values,
    llm_env_words,
    unreviewed_literals,
)
from test_fence import _agent_imports, _banned_dependencies, _declared_dependencies

from flysafelz.rules import DEFAULT_PATH

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src" / "flysafelz"
READER = SRC / "reader"
READER_FILES = sorted(READER.glob("*.py"))
PY_FILES = sorted(SRC.rglob("*.py"))
CONFIG_PY = READER / "config.py"
# Written in pieces, so this file never matches the scans it tests.
READER_NAME = "READER_" + "MODEL"
MODELS_PATH = "/mod" + "els"
ENV_READ = re.compile(r"environ|getenv|\.env\b|dotenv")
LLM_PACKAGES = {"langchain_openai", "langchain_core", "langchain", "langgraph", "openai", "anthropic"}


def tree_of(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def imports_of(path: Path) -> list[str]:
    found = []
    for node in ast.walk(tree_of(path)):
        if isinstance(node, ast.Import):
            found += [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module:
            found.append(node.module)
    return found


def rule_values() -> set[float]:
    data = yaml.safe_load(DEFAULT_PATH.read_text(encoding="utf-8"))
    return {e["value"] for e in data.values() if isinstance(e["value"], (int, float))}


@pytest.fixture
def source_copy(tmp_path) -> Path:
    """A copy of src/flysafelz under tmp_path (without bytecode), to plant violations in. Returns the copy's package folder."""
    copy = tmp_path / "src" / "flysafelz"
    shutil.copytree(SRC, copy, ignore=shutil.ignore_patterns("__pycache__"))
    return copy


def append(path: Path, text: str) -> None:
    path.write_text((path.read_text(encoding="utf-8") if path.exists() else "") + "\n" + text, encoding="utf-8")


# --- F2: dependencies --------------------------------------------------------------------------------------------------------------


def test_pyproject_adds_exactly_langchain_openai_and_nothing_else_of_the_banned_kinds():
    declared = _declared_dependencies()
    names = {re.split(r"[<>=!~\[ ;]", d, maxsplit=1)[0].lower().replace("_", "-") for d in declared}
    assert {n for n in names if n.startswith(("langchain", "langgraph", "langsmith", "openai", "anthropic"))} == {"langchain-openai"}
    assert _banned_dependencies(declared) == []


@pytest.mark.parametrize("planted", ["langgraph", "langgraph>=1", "openai>=1", "anthropic", "langchain-community", "langchain", "langchain-anthropic"])
def test_a_planted_dependency_in_a_copy_of_the_pyproject_is_caught(tmp_path, planted):
    copy = tmp_path / "pyproject.toml"
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert '"astral>=3.2",' in text
    copy.write_text(text.replace('"astral>=3.2",', f'"astral>=3.2",\n    "{planted}",', 1), encoding="utf-8")
    config = tomllib.loads(copy.read_text(encoding="utf-8"))

    assert _banned_dependencies(config["project"]["dependencies"]) == [re.split(r"[<>=]", planted)[0]]
    assert _banned_dependencies(_declared_dependencies()) == []  # the real file is clean, so the hit is the planted line


# --- F2: the three narrowed source scans, each against a planted violation outside reader/ ---------------------------------------------------


def scan_the_copy(copy: Path) -> dict[str, list[str]]:
    """Run the three real scans (imports, LLM strings, the word cache) over every file of a copy of the tree; keep the files with a hit."""
    reader = copy / "reader"
    hits: dict[str, list[str]] = {}
    for path in sorted(copy.rglob("*.py")):
        found = _agent_imports(path, reader) + llm_env_words(path, reader) + banned_words(path, reader)
        if found:
            hits[path.relative_to(copy).as_posix()] = found
    return hits


def test_a_clean_copy_of_the_tree_has_no_hit(source_copy):
    assert scan_the_copy(source_copy) == {}


@pytest.mark.parametrize(
    ("planted", "hit"),
    [
        ("import langchain_openai\n", "langchain_openai"),
        ("from langchain_core.messages import HumanMessage\n", "langchain_core.messages"),
        ("import langgraph\n", "langgraph"),
        ("import openai\n", "openai"),
        ("import anthropic\n", "anthropic"),
        (f'NAME = "{READER_NAME}"\n', "READER_"),
        (f'URL = "http://host/api/v1{MODELS_PATH}"\n', MODELS_PATH),
        ("cache = {}\n", "cache"),
    ],
    ids=["langchain_openai", "langchain_core", "langgraph", "openai", "anthropic", "a READER_ name", "a /models path", "the word cache"],
)
@pytest.mark.parametrize("where", ["newmodule.py", "clients/newclient.py", "readerish/other.py"])
def test_a_violation_planted_outside_reader_is_still_caught(source_copy, planted, hit, where):
    target = source_copy / where
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(planted, encoding="utf-8")

    hits = scan_the_copy(source_copy)

    assert list(hits) == [where]
    assert hits[where] == [hit]


@pytest.mark.parametrize(
    "planted",
    ["import langchain_openai\nfrom langchain_core.messages import HumanMessage\n", f'NAME = "{READER_NAME}"\nURL = "x{MODELS_PATH}"\n', "stored = cache\n"],
    ids=["the two allowed imports", "READER_ and /models", "the word cache"],
)
def test_the_same_text_inside_reader_is_exempt(source_copy, planted):
    (source_copy / "reader" / "newmodule.py").write_text(planted, encoding="utf-8")
    assert scan_the_copy(source_copy) == {}


def test_inside_reader_only_the_two_allowed_packages_may_be_imported(source_copy):
    """The exemption is for langchain_openai and langchain_core, not for the rest of the banned list."""
    (source_copy / "reader" / "newmodule.py").write_text("import langgraph\nimport openai\nimport anthropic\nimport langchain_community\n", encoding="utf-8")
    assert scan_the_copy(source_copy) == {"reader/newmodule.py": ["langgraph", "openai", "anthropic", "langchain_community"]}


def test_the_real_tree_uses_each_exemption_and_nothing_outside_reader_does():
    assert any(m.startswith("langchain_openai") for m in imports_of(CONFIG_PY))  # config.py builds ChatOpenAI
    assert any(re.search(r"\bcache\b", path.read_text(encoding="utf-8")) for path in READER_FILES)
    assert "READER_" in CONFIG_PY.read_text(encoding="utf-8")
    for path in PY_FILES:
        if READER not in path.parents:
            assert _agent_imports(path) + llm_env_words(path, READER) + banned_words(path, READER) == [], path


CLI_READER_IMPORTS = {
    "flysafelz.reader",
    "flysafelz.reader.config",
    "flysafelz.reader.cached_read",
    "flysafelz.reader.cached_read.model_calls",
    "flysafelz.reader.contract",
    "flysafelz.reader.contract.ReaderResult",
    "flysafelz.reader.contract.ReaderStatus",
    "flysafelz.reader.reader_set",
    "flysafelz.reader.reader_set.failed_before",
    "flysafelz.reader.reader_set.read_for_lz",
}


def test_cli_py_passes_every_narrowed_scan_unchanged():
    cli = SRC / "cli.py"
    assert banned_words(cli, READER) == [] and llm_env_words(cli, READER) == [] and _agent_imports(cli) == []
    assert not re.search(r"environ|getenv", cli.read_text(encoding="utf-8"))
    # PR-6: the command line is the one file outside reader/ that imports the reader, and only these modules of it.
    assert {m for m in imports_of(cli) if m.startswith("flysafelz.reader")} <= CLI_READER_IMPORTS


# --- F2: the environment is read in reader/config.py only (beside the reviewed openAIP key readers) -------------------------------------------


def test_the_environment_and_dotenv_are_read_in_reader_config_and_nowhere_else_new():
    assert CONFIG_PY.relative_to(SRC).as_posix() == READER_CONFIG and "environ" in CONFIG_PY.read_text(encoding="utf-8")
    assert env_read_files(SRC, PY_FILES) == []
    for path in READER_FILES:
        if path != CONFIG_PY:
            assert not re.search(r"environ|getenv|\.env\b|dotenv|READER_", path.read_text(encoding="utf-8")), path


def test_only_config_py_names_a_reader_variable():
    named = [path.relative_to(SRC).as_posix() for path in PY_FILES if re.search(r"READER_[A-Z]", path.read_text(encoding="utf-8"))]
    assert named == [READER_CONFIG]


def test_the_reviewed_openaip_key_readers_are_still_the_files_that_read_the_environment():
    """The key readers date from before the reader. A stale entry (a file that no longer matches) is dropped; a new one needs a reason."""
    for relative, reason in OPENAIP_KEY_READERS.items():
        text = (SRC / relative).read_text(encoding="utf-8")
        assert ENV_READ.search(text), f"{relative} no longer reads the environment: drop it from the list"
        assert reason
        assert "READER_" not in text  # none of them reads a reader setting


@pytest.mark.parametrize(
    ("where", "text"),
    [
        ("reader/prompt.py", "import os\nHOME = os.environ['HOME']\n"),
        ("reader/read.py", "import os\nx = os.getenv('X')\n"),
        ("reader/cache.py", "from pathlib import Path\nenv = Path('.env').read_text()\n"),
        ("reader/read.py", "import dotenv\n"),
        ("pib.py", "import os\nx = os.environ.get('Y')\n"),
        ("newmodule.py", "import os\nx = os.getenv('Z')\n"),
    ],
)
def test_an_environment_read_planted_in_any_other_file_is_caught(source_copy, where, text):
    append(source_copy / where, text)
    assert env_read_files(source_copy, sorted(source_copy.rglob("*.py"))) == [where]


def test_the_scan_leaves_config_py_and_the_reviewed_readers_alone(source_copy):
    assert env_read_files(source_copy, sorted(source_copy.rglob("*.py"))) == []
    append(source_copy / "reader" / "config.py", "import os\nx = os.getenv('MORE')\n")
    assert env_read_files(source_copy, sorted(source_copy.rglob("*.py"))) == []


# --- F2: the sort and assess path is model-free -----------------------------------------------------------------------------------------------


def flysafelz_imports(path: Path) -> set[Path]:
    """The source files of the flysafelz modules this file imports (`import flysafelz.x`, `from flysafelz.x import y`, `from flysafelz import x`)."""
    found = set()
    for node in ast.walk(tree_of(path)):
        names: list[str] = []
        if isinstance(node, ast.Import):
            names = [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module:
            names = [node.module] + [f"{node.module}.{a.name}" for a in node.names]
        for name in names:
            parts = name.split(".")
            if parts[0] != "flysafelz" or len(parts) == 1:
                continue
            relative = Path(*parts[1:])
            found.update(c for c in (SRC / relative.with_suffix(".py"), SRC / relative / "__init__.py") if c.is_file())
    return found


def closure(start: Path) -> set[Path]:
    """The file and every flysafelz file it imports, directly or not."""
    seen, todo = set(), [start]
    while todo:
        path = todo.pop()
        if path not in seen:
            seen.add(path)
            todo += sorted(flysafelz_imports(path))
    return seen


@pytest.mark.parametrize("name", ["notam_action.py", "assess.py", "pib.py", "airspace.py"])
def test_the_sort_assess_and_notam_action_paths_import_nothing_from_the_reader_and_no_llm_client(name):
    files = closure(SRC / name)
    assert SRC / name in files
    assert [p for p in files if READER in p.parents] == []
    for path in files:
        assert not [m for m in imports_of(path) if m.split(".")[0] in LLM_PACKAGES], path


def test_the_cli_reaches_the_reader_and_every_other_module_it_reaches_has_no_llm_client():
    """PR-6: cli.py imports the reader on purpose. Outside reader/, nothing it reaches imports a model client."""
    files = closure(SRC / "cli.py")
    assert READER / "reader_set.py" in files and READER / "config.py" in files
    for path in files:
        if READER not in path.parents:
            assert not [m for m in imports_of(path) if m.split(".")[0] in LLM_PACKAGES], path


def test_the_closure_walk_finds_the_reader_when_a_module_imports_it(tmp_path):
    """The walk is not blind: a file that imports the reader reaches the reader's modules, and through them the model client."""
    importer = tmp_path / "importer.py"
    importer.write_text("from flysafelz.reader.reader_set import read_for_lz\n", encoding="utf-8")
    reached = closure(importer)
    assert READER / "reader_set.py" in reached and READER / "read.py" in reached


# --- F5: a thread pool, no asyncio, no network ---------------------------------------------------------------------------------------------------

ASYNC = re.compile(r"\basyncio\b|\basync\s+def\b|\bawait\b|\bainvoke\b|\bastream\b|\babatch\b")


def test_the_reader_limits_calls_with_concurrent_futures_and_invoke_and_no_asyncio():
    text = {path.name: path.read_text(encoding="utf-8") for path in READER_FILES}
    for name, source in text.items():
        assert not ASYNC.search(source), name
        assert "asyncio" not in imports_of(READER / name)
    assert "from concurrent.futures import ThreadPoolExecutor" in text["reader_set.py"] and "ThreadPoolExecutor(max_workers=SETTINGS.concurrency)" in text["reader_set.py"]
    assert "llm.invoke(" in text["read.py"]


@pytest.mark.parametrize("planted", ["import asyncio\n", "from asyncio import gather\n", "async def read():\n    await go()\n", "result = await llm.ainvoke(x)\n"])
def test_a_planted_asyncio_in_the_reader_is_caught(source_copy, planted):
    target = source_copy / "reader" / "reader_set.py"
    append(target, planted)
    assert ASYNC.search(target.read_text(encoding="utf-8"))
    assert not ASYNC.search((READER / "reader_set.py").read_text(encoding="utf-8"))


def test_the_makefile_has_the_eval_target_and_only_test_runs_the_suite():
    """PR-8 adds `eval-reader`, the one target that calls the model. `make test` does not run it and mentions no READER_* setting."""
    makefile = (ROOT / "Makefile").read_text(encoding="utf-8")
    assert re.findall(r"^([a-z-]+):", makefile, re.MULTILINE) == ["setup", "test", "fixtures", "eval-reader"]
    test_recipe = re.search(r"^test:[^\n]*\n((?:\t[^\n]*\n?)+)", makefile, re.MULTILINE)
    assert test_recipe and "eval-reader" not in test_recipe[1] and "READER" not in test_recipe[1] and "reader_eval.run" not in test_recipe[1]
    assert "--cov=flysafelz.reader " in makefile and "--cov=evals.reader_eval " in makefile and "--cov-fail-under=90" in makefile


# --- F6: no threshold literal in the reader --------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("path", READER_FILES, ids=lambda p: p.name)
def test_no_reader_module_holds_a_numeric_literal_other_than_zero_and_one(path):
    tree = tree_of(path)
    assert unreviewed_literals(path, tree) == []
    assert hard_coded_rule_values(path, tree, rule_values()) == []


def test_the_settings_are_loaded_by_name_and_never_written_as_literals():
    reader_set = (READER / "reader_set.py").read_text(encoding="utf-8")
    read = (READER / "read.py").read_text(encoding="utf-8")
    assert "SETTINGS.concurrency" in reader_set and "SETTINGS.timeout_s" in read
    assert yaml.safe_load((READER / "settings.yaml").read_text(encoding="utf-8")) == {"timeout_s": 45, "concurrency": 3, "runs": 3}
    assert 3 in rule_values()  # pib_stale_hours: the number the reader must never write as a literal
    for path in READER_FILES:
        literals = {node.value for node in ast.walk(tree_of(path)) if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool)}
        assert not literals & {3, 45}, path


@pytest.mark.parametrize(
    ("planted", "found"),
    [("workers = 3\n", [3]), ("timeout = 45\n", [45]), ("limit = 12\n", [12]), ("pause = 0.5\n", [0.5]), ("if count > 10:\n    pass\n", [10])],
)
def test_a_threshold_planted_in_a_reader_module_is_caught(source_copy, planted, found):
    target = source_copy / "reader" / "reader_set.py"
    append(target, planted)
    assert [v for _, v in unreviewed_literals(target, tree_of(target))] == found


def test_the_equals_a_rules_value_scan_flags_3_and_12_and_exempts_structural_zero_and_one():
    values = rule_values()
    assert {3, 12} <= values and 1.0 in values
    planted = ast.parse("a = 3\nb = 12\nc = 1 - WGS84_E2\nd = 0\ne = 1.0\n")
    assert [v for _, v in hard_coded_rule_values(Path("reader/reader_set.py"), planted, values)] == [3, 12]
    assert hard_coded_rule_values(Path("geo.py"), ast.parse("e = 1 - WGS84_E2\nx = 1.0\n"), values) == []


def test_reader_area_slack_nm_is_a_flat_rule_with_its_metadata_and_a_readme_row():
    rules = yaml.safe_load(DEFAULT_PATH.read_text(encoding="utf-8"))
    assert list(rules).count("reader_area_slack_nm") == 1
    entry = rules["reader_area_slack_nm"]
    assert set(entry) >= {"value", "unit", "comparison", "source", "confidence", "note"} and entry["value"] == 1.0 and entry["unit"] == "NM"
    assert all(entry[key] for key in ("source", "confidence", "note"))
    readme_rows = [line for line in (ROOT / "README.md").read_text(encoding="utf-8").splitlines() if line.startswith("| reader_area_slack_nm |")]
    assert len(readme_rows) == 1
