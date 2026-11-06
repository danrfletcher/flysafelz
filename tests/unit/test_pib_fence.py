"""Fence checks for the NOTAM half (F1, F3, F5, F7, F10): pib_source.py and pib.py."""

import ast
import re
from pathlib import Path

import pytest
from pib_helpers import LZ, SPAN, parsed

from flysafelz.pib import prefilter, sort_notam
from flysafelz.rules import RULES

ROOT = Path(__file__).resolve().parents[2]
MODULES = [ROOT / "src" / "flysafelz" / "clients" / "pib_source.py", ROOT / "src" / "flysafelz" / "pib.py"]
TESTS = [Path(__file__).parent / f"{name}.py" for name in ("test_pib_parse", "test_pib_prefilter", "test_notam_sort", "test_pib_source")]
ALLOWED_BASES = {"BaseModel", "ValueError", "StrEnum"}  # models, one error type, one enum: no source base class


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
def test_f1_f10_no_llm_client_and_no_environment_read(path):
    text = path.read_text(encoding="utf-8")
    assert path.is_file()
    assert not re.search(r"openai|anthropic|langchain|os\.environ|getenv|dotenv", text, re.IGNORECASE)
    assert not re.search(r"LLM_|MODEL_", text)  # an LLM environment variable name (case matters: model_config is fine)


@pytest.mark.parametrize("path", MODULES, ids=lambda p: p.name)
def test_f3_no_comparison_holds_a_number_other_than_zero_or_one(path):
    assert [n for n in numbers_in_comparisons(tree(path)) if n not in (0, 1)] == []


@pytest.mark.parametrize("path", MODULES, ids=lambda p: p.name)
def test_f3_no_threshold_literal_anywhere(path):
    """2 NM, FL195, 50 NM, 3 h: none of them appears as a number in the code."""
    for node in ast.walk(tree(path)):
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
            assert node.value not in (2, 2.0, 195, 50, 50.0, 3, 3.0, 999), (path.name, node.lineno, node.value)


def test_f3_the_scan_catches_a_planted_comparison():
    assert numbers_in_comparisons(ast.parse("if edge > 2:\n    pass")) == [2]


def test_f3_the_numbers_are_in_rules_yaml():
    assert (RULES.local_area.value, RULES.notam_max_lower_fl.value, RULES.notam_area_wide_nm.value, RULES.pib_stale_hours.value) == (2, 195, 50, 3)


def test_f3_a_changed_local_area_changes_the_prefilter():
    far = parsed(centre_lat=LZ[0] + 2.5 / 60, centre_lon=LZ[1], radius_nm=0.0)  # about 2.5 NM north
    assert prefilter([far], LZ, SPAN)[0] == []
    wider = RULES.model_copy(update={"local_area": RULES.local_area.model_copy(update={"value": 3})})
    assert prefilter([far], LZ, SPAN, rules=wider)[0] == [far]


def test_f3_a_changed_area_wide_radius_changes_the_sort():
    notam = parsed(subject="WX", radius_nm=60.0)
    assert sort_notam(notam, SPAN, LZ).action.value == "AWARENESS"
    wider = RULES.model_copy(update={"notam_area_wide_nm": RULES.notam_area_wide_nm.model_copy(update={"value": 70})})
    assert sort_notam(notam, SPAN, LZ, rules=wider).action.value == "TO_READER"


@pytest.mark.parametrize("path", MODULES, ids=lambda p: p.name)
def test_f7_plain_functions_and_models_only(path):
    for node in ast.walk(tree(path)):
        if isinstance(node, ast.ClassDef):
            assert {ast.unparse(b) for b in node.bases} <= ALLOWED_BASES, node.name
            assert not [n for n in node.body if isinstance(n, ast.FunctionDef)], f"{node.name} has methods"
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            assert not node.decorator_list, node.name
    assert not re.search(r"\bcache\b|lru_cache|registry|abc\b|ABC", path.read_text(encoding="utf-8"))


@pytest.mark.parametrize("path", TESTS, ids=lambda p: p.name)
def test_f5_the_tests_never_open_a_socket_read_env_or_ask_the_clock(path):
    text = path.read_text(encoding="utf-8")
    assert not re.search(r"\.now\(|utcnow|\.today\(|time\.time\(|freezegun|dotenv|environ|getenv|\.env\b|import socket|requests|urllib", text)


def test_f5_pytest_runs_with_sockets_disabled():
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert '--disable-socket' in pyproject
