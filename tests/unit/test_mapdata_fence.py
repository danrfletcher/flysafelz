"""F6 and F8 for src/flysafelz/mapdata/: the web framework and the word `cache` are allowed there and nowhere else; the only writes are under data/map/;
the key and NOTAM text reach no log line; and data/map/ is git-ignored.

Each exemption is shown to be exactly that wide with the real scanning functions on planted files in a temporary copy of the layout. Nothing is
planted in the real tree.
"""

import ast
import re
import subprocess
from pathlib import Path

import pytest
from fence_scan import banned_words, env_read_files, import_violations, llm_env_words

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src" / "flysafelz"
MAPDATA = SRC / "mapdata"
MAPDATA_FILES = sorted(MAPDATA.glob("*.py"))
PACKAGES = ["agent", "guidance", "reader", "clients", "mapdata"]
FRAMEWORKS = [
    "import starlette",
    "from starlette.applications import Starlette",
    "from starlette.middleware.cors import CORSMiddleware",
    "import fastapi",
    "from fastapi import FastAPI",
]


def planted_tree(tmp_path: Path, package: str, text: str) -> tuple[Path, Path, Path]:
    """(src, mapdata, planted file): a fake flysafelz source folder with `text` in a new file of `package` ("" is a top-level module)."""
    src = tmp_path / "flysafelz"
    folder = src / package if package else src
    folder.mkdir(parents=True)
    path = folder / "planted.py"
    path.write_text(text + "\n", encoding="utf-8")
    return src, src / "mapdata", path


# --- the import fence ---------------------------------------------------------------------------------------------------------------


def test_the_real_mapdata_package_imports_its_web_framework_and_nothing_else_the_fence_forbids():
    imported = {line.split()[1].split(".")[0] for path in MAPDATA_FILES for line in path.read_text(encoding="utf-8").splitlines() if re.match(r"(from|import) \w", line)}
    assert "starlette" in imported
    for path in MAPDATA_FILES:
        assert import_violations(path, SRC) == [], path


@pytest.mark.parametrize("statement", FRAMEWORKS)
def test_the_web_framework_is_allowed_in_mapdata(tmp_path, statement):
    src, _mapdata, planted = planted_tree(tmp_path, "mapdata", statement)
    assert import_violations(planted, src) == []


@pytest.mark.parametrize("package", [p for p in PACKAGES if p != "mapdata"] + [""])
@pytest.mark.parametrize("statement", FRAMEWORKS)
def test_the_import_scan_still_catches_a_planted_web_framework_import_outside_mapdata(tmp_path, package, statement):
    src, _mapdata, planted = planted_tree(tmp_path, package, statement)
    assert import_violations(planted, src) != [], f"{statement!r} in {package or 'a top-level module'} was not caught"


def test_the_agent_package_in_the_real_tree_imports_no_web_framework():
    for path in (SRC / "agent").rglob("*.py"):
        assert not re.search(r"^\s*(from|import) (starlette|fastapi)", path.read_text(encoding="utf-8"), re.MULTILINE), path


def test_mapdata_gets_no_other_import_exemption(tmp_path):
    for statement in ["import openai", "import anthropic", "from langchain_openai import ChatOpenAI", "import langgraph", "from fastembed import TextEmbedding", "import langchain_core"]:
        src, _mapdata, planted = planted_tree(tmp_path / statement.replace(" ", "_"), "mapdata", statement)
        assert import_violations(planted, src) != [], statement
    src, _mapdata, planted = planted_tree(tmp_path / "internal", "mapdata", "from flysafelz.agent.state import RunContext")
    assert import_violations(planted, src) != []  # mapdata does not reach into the agent package


# --- the word `cache` -----------------------------------------------------------------------------------------------------------------


def test_the_word_cache_is_in_the_real_mapdata_package_and_the_scan_passes_it():
    assert any(re.search(r"\bcache\b", path.read_text(encoding="utf-8")) for path in MAPDATA_FILES)
    for path in MAPDATA_FILES:
        assert banned_words(path, SRC / "reader", MAPDATA) == [], path


def test_cache_is_allowed_in_the_reader_and_in_mapdata_only(tmp_path):
    for package, allowed in [("mapdata", True), ("reader", True), ("agent", False), ("guidance", False), ("clients", False), ("", False)]:
        src, mapdata, planted = planted_tree(tmp_path / (package or "top"), package, "cache = {}")
        assert (banned_words(planted, src / "reader", mapdata) == []) is allowed, package


@pytest.mark.parametrize("word", ["@lru_cache(maxsize=1)", "@functools.cache", "registry = {}", "class ModelRegistry: pass", "def inject(x): pass", "class DataProvider: pass"])
def test_lru_cache_functools_cache_and_registry_stay_banned_in_mapdata_too(tmp_path, word):
    src, mapdata, planted = planted_tree(tmp_path, "mapdata", word)
    assert banned_words(planted, src / "reader", mapdata) != []


def test_the_scan_without_a_mapdata_folder_treats_it_like_any_other_package(tmp_path):
    src, _mapdata, planted = planted_tree(tmp_path, "mapdata", "cache = {}")
    assert banned_words(planted, src / "reader") != []  # callers that do not name the folder (ticket 3's tests) keep the old rule


# --- environment reads and LLM names ---------------------------------------------------------------------------------------------------


def test_only_config_py_in_mapdata_reads_the_environment_and_it_is_a_reviewed_reader():
    reading = [path.name for path in MAPDATA_FILES if re.search(r"environ|getenv|\.env\b|dotenv", path.read_text(encoding="utf-8"))]
    assert reading == ["config.py"]
    assert env_read_files(SRC, MAPDATA_FILES) == []  # config.py is on OPENAIP_KEY_READERS, with its reason


def test_a_second_mapdata_file_that_reads_the_environment_fails_the_scan(tmp_path):
    src, _mapdata, planted = planted_tree(tmp_path, "mapdata", "import os\nX = os.environ['A']")
    assert env_read_files(src, [planted]) == ["mapdata/planted.py"]


def test_the_key_is_read_through_capture_find_key_and_nothing_in_mapdata_names_the_variable():
    text = (MAPDATA / "config.py").read_text(encoding="utf-8")
    assert "find_key" in text
    for path in MAPDATA_FILES:
        source = path.read_text(encoding="utf-8")
        assert "os.getenv" not in source and ".environ[" not in source and "OPENAIP_API_KEY" not in source.replace("KEY_VARIABLE", ""), path


def test_no_llm_environment_name_appears_in_mapdata_and_a_planted_one_is_caught(tmp_path):
    for path in MAPDATA_FILES:
        assert llm_env_words(path, SRC / "reader") == [], path
    src, _mapdata, planted = planted_tree(tmp_path, "mapdata", "NAME = 'OPENAI_BASE_URL'")
    assert llm_env_words(planted, src / "reader") != []


# --- F8: writes only under data/map/ ---------------------------------------------------------------------------------------------------

WRITE_CALLS = {
    "write_text", "write_bytes", "mkdir", "mkstemp", "mkdtemp", "NamedTemporaryFile", "TemporaryFile", "unlink", "rmtree", "touch", "copyfile", "move", "open",
    "os.replace", "os.rename", "os.remove", "os.fdopen", "os.makedirs", "shutil.copy", "shutil.move",
}  # fmt: skip


def calls_in(path: Path) -> set[str]:
    """The names a file calls: `name`, `x.attr` for an attribute of a plain name, and bare `attr` for the method names that are only ever file writes."""
    names = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Attribute):
                names.add(func.attr)
                if isinstance(func.value, ast.Name):
                    names.add(f"{func.value.id}.{func.attr}")
            else:
                names.add(getattr(func, "id", ""))
    return names


def test_only_the_cache_module_writes_files():
    writers = {path.name for path in MAPDATA_FILES if calls_in(path) & WRITE_CALLS}
    assert writers == {"cache.py"}


def test_the_cache_modules_paths_all_come_from_the_folder_it_is_given():
    text = (MAPDATA / "cache.py").read_text(encoding="utf-8")
    assert '"data"' not in text and "'data'" not in text  # the folder name is config.py's alone; cache.py writes only where `resolve` is pointed
    assert "tempfile.mkstemp" in text and "dir=path.parent" in text  # the temp file is made beside the target, never in the system temp folder


def test_the_apps_cache_folder_is_data_map_under_the_root(tmp_path):
    from flysafelz.mapdata.config import cache_dir

    assert cache_dir(tmp_path) == tmp_path / "data" / "map"


# --- Q10: no key and no NOTAM text in a log line ------------------------------------------------------------------------------------------

LOG_METHODS = {"debug", "info", "warning", "error", "exception", "critical", "log"}
FORBIDDEN_IN_LOG_ARGUMENTS = {"key", "api_key", "text", "item_e", "e", "body", "content", "response", "request", "notam", "notams", "features", "payload", "error", "exc", "failure"}


def log_calls(path: Path) -> list[ast.Call]:
    return [
        node
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8")))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr in LOG_METHODS and isinstance(node.func.value, ast.Name) and node.func.value.id == "log"
    ]


def test_mapdata_logs_through_a_logger_and_never_prints():
    assert sum(len(log_calls(path)) for path in MAPDATA_FILES) >= 3
    for path in MAPDATA_FILES:
        assert "print" not in calls_in(path), path


def test_no_log_call_in_mapdata_is_given_a_name_that_could_hold_a_key_or_notam_text():
    """A log argument is a count, a layer name, a reason line, or a class name (type(error).__name__): never a whole error, body, feature or text."""
    for path in MAPDATA_FILES:
        for call in log_calls(path):
            for argument in call.args[1:] + [k.value for k in call.keywords]:
                reasons = {id(a.value) for a in ast.walk(argument) if isinstance(a, ast.Attribute) and a.attr == "reason"}  # `failure.reason` is the one-line reason
                names = {n.id for n in ast.walk(argument) if isinstance(n, ast.Name) and id(n) not in reasons}
                inside_type_name = {n.id for t in ast.walk(argument) if isinstance(t, ast.Call) and getattr(t.func, "id", "") == "type" for n in ast.walk(t) if isinstance(n, ast.Name)}
                assert not (names - inside_type_name) & FORBIDDEN_IN_LOG_ARGUMENTS, f"{path.name}:{call.lineno}"


# --- .gitignore ---------------------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["notams.json", "airspace.json", ".tmp-abc123.json"])
def test_the_cache_files_are_git_ignored(name):
    result = subprocess.run(["git", "check-ignore", "-q", f"data/map/{name}"], cwd=ROOT, capture_output=True, text=True, check=False)
    assert result.returncode == 0, f"data/map/{name} is not ignored"


def test_no_file_under_data_map_is_tracked():
    result = subprocess.run(["git", "ls-files", "data/map"], cwd=ROOT, capture_output=True, text=True, check=True)
    assert result.stdout == ""


def test_the_other_data_folders_are_not_swept_up_by_the_new_rule():
    result = subprocess.run(["git", "check-ignore", "-q", "data/guidance/sources.yaml"], cwd=ROOT, capture_output=True, text=True, check=False)
    assert result.returncode == 1  # data/guidance stays tracked
