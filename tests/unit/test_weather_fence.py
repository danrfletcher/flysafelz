"""Fence checks for the weather half (F3, F4, F5, F7, F10): open_meteo.py and weather.py."""

import ast
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
MODULES = [ROOT / "src" / "flysafelz" / "clients" / "open_meteo.py", ROOT / "src" / "flysafelz" / "weather.py"]
TESTS = [Path(__file__).parent / f"{name}.py" for name in ("test_open_meteo", "test_weather_checks", "test_weather_rows", "test_weather_tier")]


def tree(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def numbers_in_comparisons(module: ast.Module) -> list[float]:
    found = []
    for node in ast.walk(module):
        if isinstance(node, ast.Compare):
            for operand in [node.left, *node.comparators]:
                if isinstance(operand, ast.Constant) and isinstance(operand.value, (int, float)) and not isinstance(operand.value, bool):
                    found.append(operand.value)
    return found


@pytest.mark.parametrize("path", MODULES, ids=lambda p: p.name)
def test_f3_no_comparison_holds_a_number_other_than_zero_or_one(path):
    assert path.is_file()
    assert [n for n in numbers_in_comparisons(tree(path)) if n not in (0, 1)] == []


def test_f3_the_scan_catches_a_planted_comparison():
    assert numbers_in_comparisons(ast.parse("if wind >= 16:\n    pass")) == [16]


@pytest.mark.parametrize("path", MODULES, ids=lambda p: p.name)
def test_f4_no_score_no_weight_and_no_bare_go_string(path):
    module = tree(path)
    for node in ast.walk(module):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            assert not re.search(r"(?<!NO-)(?<!NO_)\bGO\b", node.value), node.value
    assert not re.search(r"score|weight", path.read_text(encoding="utf-8"), re.IGNORECASE)


@pytest.mark.parametrize("path", MODULES, ids=lambda p: p.name)
def test_f7_f10_plain_functions_and_models_and_no_key_or_llm_variable(path):
    text = path.read_text(encoding="utf-8")
    for node in ast.walk(tree(path)):
        if isinstance(node, ast.ClassDef):
            assert {ast.unparse(b) for b in node.bases} == {"BaseModel"}, node.name
            assert not [n for n in node.body if isinstance(n, ast.FunctionDef)], f"{node.name} has methods"
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            assert not node.decorator_list, node.name
    assert not re.search(r"environ|getenv|dotenv|(?i:api_?key)|OPENAI|ANTHROPIC|LLM_|MODEL_|/models", text)


@pytest.mark.parametrize("path", TESTS, ids=lambda p: p.name)
def test_f5_the_weather_tests_never_open_a_socket_read_env_or_ask_the_clock(path):
    text = path.read_text(encoding="utf-8")
    assert not re.search(r"\.now\(|utcnow|\.today\(|time\.time\(|freezegun|dotenv|environ|getenv|\.env\b|import socket|requests|urllib", text)
