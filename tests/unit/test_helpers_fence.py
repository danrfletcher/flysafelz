"""Fence checks for the pure helper modules geo, daylight, schedule and designators (F1, F2, F3, F5, F7).

The wider scans in test_fence.py cover every source file. These add the checks that
are specific to the helpers: a reviewed list of the numeric literals they may hold,
the imports they may use, and the shape of their classes and functions.
"""

import ast
import re
from pathlib import Path

import pytest
import yaml

from flysafelz.rules import DEFAULT_PATH

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src" / "flysafelz"
MODULES = [SRC / f"{name}.py" for name in ("geo", "daylight", "schedule", "designators")]
NEW_TESTS = [Path(__file__).parent / f"{name}.py" for name in ("test_geo", "test_daylight", "test_schedule", "test_schedule_corpus", "test_designators")]

# Every numeric literal the helpers may hold besides 0 and 1, each with its reason.
# None of these is a threshold: they are units, calendar and format facts, or the WGS84 ellipsoid.
# A new literal fails the test until it is added here and reviewed.
REVIEWED_LITERALS = {
    1852: "metres per nautical mile (unit)",
    60: "minutes per hour (unit)",
    24: "hours per day (unit; also the 2400 end-of-day time)",
    360: "degrees in a circle (unit)",
    180: "degrees in a half turn (unit)",
    100: "HHMM to hours and minutes (format)",
    0.5: "half (a sector's half-width, half of pi)",
    6378137.0: "WGS84 equatorial radius in metres",
    0.00669437999014: "WGS84 first eccentricity squared",
    1e-6: "metres: float noise when asking whether the LZ is on an edge",
    2024: "a leap year, used only to check that a month and day exist",
}


def _tree(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _literals(tree: ast.AST) -> list[tuple[int, float]]:
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
            found.append((node.lineno, node.value))
    return found


def _rule_values() -> set[float]:
    data = yaml.safe_load(DEFAULT_PATH.read_text(encoding="utf-8"))
    return {abs(e["value"]) for e in data.values() if isinstance(e["value"], (int, float))}


def test_the_four_modules_exist():
    assert all(path.is_file() for path in MODULES)


@pytest.mark.parametrize("path", MODULES, ids=lambda p: p.name)
def test_f3_every_numeric_literal_is_zero_one_or_a_reviewed_unit_constant(path):
    for lineno, value in _literals(_tree(path)):
        assert value in (0, 1) or value in REVIEWED_LITERALS, f"{path.name}:{lineno} holds the literal {value}"


@pytest.mark.parametrize("path", MODULES, ids=lambda p: p.name)
def test_f3_no_literal_equals_a_rules_yaml_value_and_none_is_a_uk_bound_or_the_twilight_angle(path):
    values = _rule_values()
    assert {49.8, 61.0, 8.7, 2.0, 6} <= values  # the new rules are in the file, so the scan covers them
    for lineno, value in _literals(_tree(path)):
        assert abs(value) not in values, f"{path.name}:{lineno} hard-codes {value}, which is a rules.yaml value"


@pytest.mark.parametrize("path", MODULES, ids=lambda p: p.name)
def test_f3_the_threshold_numbers_do_not_appear_as_text_either(path):
    text = path.read_text(encoding="utf-8")
    assert not re.search(r"49\.8|61\.0|(?<!\d)8\.7(?!\d)|depression\s*=\s*6\b|\b6\s*°|\b6 degrees below", text)


def test_f3_the_scan_catches_a_planted_literal():
    planted = _tree_of("limit = 61.0\ndepression = 6\nradius = 2\nunit = 1852\n")
    flagged = [v for _, v in _literals(planted) if v not in (0, 1) and v not in REVIEWED_LITERALS or abs(v) in _rule_values()]
    assert flagged == [61.0, 6, 2]


def _tree_of(source: str) -> ast.Module:
    return ast.parse(source)


# F1 and F2: nothing from LLM, agent, LangGraph, network or UI code.
ALLOWED_IMPORTS = {
    "math", "re", "itertools", "dataclasses", "datetime", "enum", "zoneinfo", "collections.abc",
    "astral", "astral.sun",
    "flysafelz.rules", "flysafelz.daylight",
}  # fmt: skip


@pytest.mark.parametrize("path", MODULES, ids=lambda p: p.name)
def test_f1_f2_imports_are_stdlib_astral_and_the_rules_loader_only(path):
    modules = []
    for node in ast.walk(_tree(path)):
        if isinstance(node, ast.Import):
            modules += [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.append(node.module)
    assert modules
    for module in modules:
        assert module in ALLOWED_IMPORTS, f"{path.name} imports {module}"
        assert not module.startswith(("langgraph", "langchain", "openai", "anthropic", "httpx", "socket", "requests", "urllib"))


# F7: plain functions, small return types, no classes with behaviour, no registries, no caching decorators.
@pytest.mark.parametrize("path", MODULES, ids=lambda p: p.name)
def test_f7_classes_are_exceptions_enums_or_dataclasses_without_methods(path):
    for node in ast.walk(_tree(path)):
        if isinstance(node, ast.ClassDef):
            bases = {ast.unparse(b) for b in node.bases}
            decorators = {ast.unparse(d) for d in node.decorator_list}
            plain_data = decorators <= {"dataclass", "dataclass(frozen=True)"}
            assert bases <= {"ValueError", "StrEnum"}, f"{path.name}:{node.lineno} {node.name} has bases {bases}"
            assert plain_data, f"{path.name}:{node.lineno} {node.name} has decorators {decorators}"
            assert not [n for n in node.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))], f"{node.name} has methods"


@pytest.mark.parametrize("path", MODULES, ids=lambda p: p.name)
def test_f7_no_function_is_decorated(path):
    for node in ast.walk(_tree(path)):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            assert not node.decorator_list, f"{path.name}:{node.lineno} {node.name} is decorated"


@pytest.mark.parametrize("path", MODULES, ids=lambda p: p.name)
def test_the_modules_read_no_clock_environment_or_dotenv(path):
    text = path.read_text(encoding="utf-8")
    assert not re.search(r"\.now\(|utcnow|today\(|time\.time\(|environ|getenv|dotenv|\.env\b|open\(", text)


# F5: the new tests take explicit datetimes, open no sockets and read no .env.
@pytest.mark.parametrize("path", NEW_TESTS, ids=lambda p: p.name)
def test_f5_new_tests_never_ask_the_clock_or_touch_env_or_sockets(path):
    text = path.read_text(encoding="utf-8")
    assert not re.search(r"\.now\(|utcnow|\.today\(|time\.time\(|freezegun", text), path.name
    assert not re.search(r"dotenv|environ|getenv|\.env\b|\bsocket\b|httpx|urllib|requests", text), path.name
