"""Fence scans (F2, F3, F4, F6, F7, F8, F10). Plain AST and text checks; later PRs inherit them."""

import ast
import enum
import re
import subprocess
import tomllib
from pathlib import Path

import pytest
import yaml
from pydantic import BaseModel, ValidationError

from flysafelz import cli, contracts
from flysafelz.rules import DEFAULT_PATH, RULES

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src" / "flysafelz"
PY_FILES = sorted(SRC.rglob("*.py"))


def _parse(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _number(node: ast.AST) -> float | None:
    """The value of a numeric literal (including a negated one), else None."""
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        inner = _number(node.operand)
        return None if inner is None else (-inner if isinstance(node.op, ast.USub) else inner)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
        return node.value
    return None


def _rule_values() -> set[float]:
    data = yaml.safe_load(DEFAULT_PATH.read_text(encoding="utf-8"))
    return {e["value"] for e in data.values() if isinstance(e["value"], (int, float))}


def test_source_files_are_found():
    assert SRC / "contracts.py" in PY_FILES and SRC / "rules.py" in PY_FILES


# F2: no agent, tool, LangGraph, LangChain, RAG or UI code.
def test_pyproject_has_no_agent_or_llm_dependency():
    config = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    declared = list(config["project"]["dependencies"])
    for group in config.get("dependency-groups", {}).values():
        declared += [d for d in group if isinstance(d, str)]
    names = [re.split(r"[<>=!~\[ ;]", d, maxsplit=1)[0].lower() for d in declared]
    assert names, "no dependencies read"
    for name in names:
        assert not name.startswith(("langgraph", "langchain", "openai", "anthropic")), name


def test_source_imports_no_agent_framework():
    for path in PY_FILES:
        for node in ast.walk(_parse(path)):
            modules = []
            if isinstance(node, ast.Import):
                modules = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                modules = [node.module]
            for module in modules:
                assert not module.startswith(("langgraph", "langchain", "openai", "anthropic")), (path, module)


# F3: no threshold literals in Python.
def test_no_numeric_literal_in_a_comparison_other_than_0_or_1():
    allowed = {0, 1}
    for path in PY_FILES:
        for node in ast.walk(_parse(path)):
            if isinstance(node, ast.Compare):
                for operand in [node.left, *node.comparators]:
                    value = _number(operand)
                    assert value is None or value in allowed, f"{path}:{node.lineno} compares against {value}"


def test_no_numeric_literal_equals_a_rules_yaml_value():
    values = _rule_values()
    assert values, "no rule values read"
    for path in PY_FILES:
        for node in ast.walk(_parse(path)):
            value = _number(node)
            if value is not None and abs(value) in values:
                pytest.fail(f"{path}:{node.lineno} hard-codes {value}, which is a rules.yaml value")


def test_f3_scan_catches_a_planted_literal():
    planted = ast.parse("if wind >= 16:\n    pass\nlimit = 12\n")
    compared = [_number(o) for n in ast.walk(planted) if isinstance(n, ast.Compare) for o in n.comparators]
    assert compared == [16]
    assert 12 in _rule_values()


# F4: no score, no bare GO verdict.
def test_verdict_enum_has_exactly_three_members():
    assert {v.name for v in contracts.Verdict} == {"NO_GO", "UNVERIFIED", "PILOT_DISCRETION"}


def test_no_string_literal_is_a_bare_go():
    bare_go = re.compile(r"(?<!NO-)(?<!NO_)\bGO\b")
    for path in PY_FILES:
        for node in ast.walk(_parse(path)):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                assert not bare_go.search(node.value), f"{path}:{node.lineno}: {node.value!r}"


def test_no_score_or_weight_identifiers():
    for path in PY_FILES:
        for node in ast.walk(_parse(path)):
            names = []
            if isinstance(node, ast.Name):
                names = [node.id]
            elif isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                names = [node.name]
            elif isinstance(node, ast.arg):
                names = [node.arg]
            elif isinstance(node, ast.Attribute):
                names = [node.attr]
            for name in names:
                assert not re.search(r"score|weight", name, re.IGNORECASE), f"{path}:{node.lineno}: {name}"


def test_readme_uses_the_three_verdicts_not_the_old_wording():
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "GO WITH CONDITIONS" not in text
    assert "MARGINAL" not in text
    for wording in ("NO-GO", "UNVERIFIED", "PILOT DISCRETION", "APPROACHING LIMITS", "WITH CONDITIONS"):
        assert wording in text


# F6: no secrets in tracked files.
def _tracked_files() -> list[str]:
    out = subprocess.run(["git", "ls-files"], cwd=ROOT, check=True, capture_output=True, text=True).stdout
    return out.splitlines()


def test_no_key_value_in_tracked_files():
    tracked = _tracked_files()
    assert ".env.example" in tracked
    assert ".env" not in tracked
    key_with_value = re.compile(r"OPENAIP_API_KEY[ \t]*=[ \t]*[^\s#]")
    for name in tracked:
        path = ROOT / name
        if not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        assert not key_with_value.search(text), f"{name} has an OPENAIP_API_KEY value"


def test_env_is_gitignored_and_example_is_empty():
    ignored = subprocess.run(["git", "check-ignore", ".env"], cwd=ROOT, capture_output=True, text=True)
    assert ignored.returncode == 0
    assert (ROOT / ".env.example").read_text(encoding="utf-8").strip() == "OPENAIP_API_" + "KEY="  # split so this file never matches the F6 scan


# F7: plain modules and functions.
def test_contract_classes_subclass_only_basemodel_or_enum():
    classes = [o for o in vars(contracts).values() if isinstance(o, type) and o.__module__ == contracts.__name__]
    assert classes
    for cls in classes:
        assert cls.__bases__ in {(BaseModel,), (enum.Enum,), (enum.StrEnum,)}, cls


def test_no_registry_di_or_caching_in_source():
    pattern = re.compile(r"lru_cache|functools\.cache|\bcache\b|registry|Registry|\binject\b|Provider\b")
    for path in PY_FILES:
        assert not pattern.search(path.read_text(encoding="utf-8")), path


# F8: limits and the local area cannot be adjusted by the user.
def test_rules_cannot_be_assigned_to():
    with pytest.raises(ValidationError):
        RULES.wind_10m_no_go = RULES.wind_10m_caution
    with pytest.raises(ValidationError):
        RULES.wind_10m_no_go.value = 99
    with pytest.raises(ValidationError):
        RULES.local_area.value = 99


@pytest.mark.parametrize("flag", ["--rules", "--rules-path", "--path", "--local-area", "--limit"])
def test_cli_has_no_flag_for_rules_or_limits(flag):
    with pytest.raises(SystemExit) as exit_info:
        cli.main(["rules", flag, "x"])
    assert exit_info.value.code == 2


def test_rules_module_and_cli_read_no_environment():
    for name in ("rules.py", "cli.py"):
        text = (SRC / name).read_text(encoding="utf-8")
        assert not re.search(r"environ|getenv", text), name


# F10: no LLM env vars, no /models dependency.
def test_no_llm_env_vars_or_models_endpoint_in_source():
    pattern = re.compile(r"OPENAI(?!P_API_KEY)|ANTHROPIC|LLM_|MODEL_|/models")
    for path in PY_FILES:
        match = pattern.search(path.read_text(encoding="utf-8"))
        assert match is None, f"{path}: {match.group(0) if match else ''}"
