"""agent/settings.py and settings.yaml (F7): the shipped file, every malformed key, no clash with rules.yaml, and the loader's own imports."""

import ast
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

import pytest
import yaml

from flysafelz.agent.settings import SETTINGS, SETTINGS_PATH, AgentSettings, SettingsError, embedding_folder, load_settings, repository_root
from flysafelz.rules import RULES

ROOT = Path(__file__).resolve().parents[2]
AGENT = ROOT / "src" / "flysafelz" / "agent"
KEYS = list(AgentSettings.model_fields)


def shipped() -> dict:
    return yaml.safe_load(SETTINGS_PATH.read_text(encoding="utf-8"))


def write(tmp_path: Path, mutate: Callable[[dict], None]) -> Path:
    data = shipped()
    mutate(data)
    path = tmp_path / "settings.yaml"
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return path


def set_key(key: str, value: object) -> Callable[[dict], None]:
    return lambda d: d.__setitem__(key, value)


def test_the_shipped_file_loads_with_the_values_the_plan_names():
    assert SETTINGS == load_settings()
    assert SETTINGS.model_dump() == {
        "top_k": 4,
        "chunk_tokens_min": 300,
        "chunk_tokens_max": 800,
        "guard_retries": 1,
        "model_max_retries": 1,
        "source_reuse_minutes": 15,
        "embedding_model": "BAAI/bge-small-en-v1.5",
        "embedding_cache_dir": ".cache/fastembed",
        "chunker_version": 1,
        "langsmith_project": "flysafelz",
    }
    assert set(shipped()) == set(KEYS)


def test_the_settings_are_frozen():
    with pytest.raises(ValueError, match="frozen"):
        SETTINGS.top_k = 5  # type: ignore[misc]


@pytest.mark.parametrize("key", KEYS)
def test_a_missing_key_is_named(tmp_path, key):
    with pytest.raises(SettingsError, match=key):
        load_settings(write(tmp_path, lambda d: d.pop(key)))


def test_every_missing_key_is_named_for_an_empty_file(tmp_path):
    path = tmp_path / "settings.yaml"
    path.write_text("", encoding="utf-8")
    with pytest.raises(SettingsError) as error:
        load_settings(path)
    assert all(key in str(error.value) for key in KEYS)


def test_an_unknown_key_is_named(tmp_path):
    with pytest.raises(SettingsError, match="top_n"):
        load_settings(write(tmp_path, set_key("top_n", 4)))


INTEGER_KEYS = ["top_k", "chunk_tokens_min", "chunk_tokens_max", "guard_retries", "model_max_retries", "source_reuse_minutes", "chunker_version"]
TEXT_KEYS = ["embedding_model", "embedding_cache_dir", "langsmith_project"]
WRONG_TYPES = [("a string", "4"), ("a float", 1.5), ("a whole-number float", 4.0), ("a boolean", True), ("a list", [4]), ("null", None)]


@pytest.mark.parametrize("key", INTEGER_KEYS)
@pytest.mark.parametrize(("what", "value"), WRONG_TYPES, ids=[w for w, _ in WRONG_TYPES])
def test_a_whole_number_key_with_the_wrong_type_is_refused_with_the_key_named(tmp_path, key, what, value):
    with pytest.raises(SettingsError, match=key):
        load_settings(write(tmp_path, set_key(key, value)))


@pytest.mark.parametrize("key", TEXT_KEYS)
@pytest.mark.parametrize("value", [4, 1.5, True, ["a"], None, ""], ids=["int", "float", "bool", "list", "null", "empty"])
def test_a_text_key_with_the_wrong_type_or_no_text_is_refused_with_the_key_named(tmp_path, key, value):
    with pytest.raises(SettingsError, match=key):
        load_settings(write(tmp_path, set_key(key, value)))


# One row per knob: the value that is refused and the key the message must name. Retries may be 0 (no retry); the rest must be positive.
REFUSED_VALUES = [
    ("top_k is zero", "top_k", 0),
    ("top_k is negative", "top_k", -4),
    ("chunk_tokens_min is zero", "chunk_tokens_min", 0),
    ("chunk_tokens_max is negative", "chunk_tokens_max", -800),
    ("guard_retries is negative", "guard_retries", -1),
    ("model_max_retries is negative", "model_max_retries", -1),
    ("source_reuse_minutes is zero", "source_reuse_minutes", 0),
    ("chunker_version is zero", "chunker_version", 0),
    ("guard_retries is not an integer", "guard_retries", 0.5),
    ("model_max_retries is not an integer", "model_max_retries", "one"),
]


@pytest.mark.parametrize(("what", "key", "value"), REFUSED_VALUES, ids=[r[0] for r in REFUSED_VALUES])
def test_a_knob_out_of_range_is_refused_with_the_key_named(tmp_path, what, key, value):
    with pytest.raises(SettingsError, match=key):
        load_settings(write(tmp_path, set_key(key, value)))


def test_an_inverted_chunk_band_is_refused_naming_both_ends(tmp_path):
    with pytest.raises(SettingsError) as error:
        load_settings(write(tmp_path, lambda d: d.update(chunk_tokens_min=800, chunk_tokens_max=300)))
    assert "chunk_tokens_min" in str(error.value) and "chunk_tokens_max" in str(error.value)


def test_a_band_of_one_size_and_zero_retries_are_allowed(tmp_path):
    loaded = load_settings(write(tmp_path, lambda d: d.update(chunk_tokens_min=500, chunk_tokens_max=500, guard_retries=0, model_max_retries=0)))
    assert (loaded.chunk_tokens_min, loaded.chunk_tokens_max, loaded.guard_retries, loaded.model_max_retries) == (500, 500, 0, 0)


def test_a_file_that_is_not_a_mapping_or_not_yaml_is_refused(tmp_path):
    path = tmp_path / "settings.yaml"
    path.write_text("- top_k\n- 4\n", encoding="utf-8")
    with pytest.raises(SettingsError, match="mapping"):
        load_settings(path)
    path.write_text("top_k: [4\n", encoding="utf-8")
    with pytest.raises(SettingsError, match="not valid YAML"):
        load_settings(path)


def test_no_settings_key_has_the_name_of_a_rules_yaml_id():
    rule_ids = {rule.id for rule in RULES.all()}
    assert rule_ids
    assert set(KEYS) & rule_ids == set()


def test_the_loader_imports_only_the_standard_library_yaml_and_pydantic():
    tree = ast.parse((AGENT / "settings.py").read_text(encoding="utf-8"))
    modules = {n.module.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module}
    modules |= {a.name.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
    assert modules, "the scan read no imports"
    assert modules <= set(sys.stdlib_module_names) | {"yaml", "pydantic"}


def test_importing_the_loader_loads_no_langchain_langgraph_or_langsmith():
    code = "import sys, flysafelz.agent.settings; sys.exit(1 if [m for m in sys.modules if m.split('.')[0] in ('langchain', 'langchain_core', 'langchain_openai', 'langgraph', 'langsmith', 'fastembed')] else 0)"
    assert subprocess.run([sys.executable, "-c", code], cwd=ROOT, check=False).returncode == 0


def test_the_agent_package_init_is_empty():
    assert (AGENT / "__init__.py").read_text(encoding="utf-8") == ""
    assert (ROOT / "src" / "flysafelz" / "guidance" / "__init__.py").read_text(encoding="utf-8") == ""


def test_the_embedding_folder_is_under_the_repository_and_git_ignored():
    folder = embedding_folder()
    assert folder == repository_root() / SETTINGS.embedding_cache_dir and repository_root() == ROOT
    # A path inside the folder, so the answer is the same before `make setup` has made the folder (a pattern ending in / matches a folder only).
    ignored = subprocess.run(["git", "check-ignore", "-q", f"{SETTINGS.embedding_cache_dir}/models--x/file"], cwd=ROOT, check=False)
    assert ignored.returncode == 0
