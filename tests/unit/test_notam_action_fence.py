"""Fence checks for notam_action.py: F1 (no model call), F3 (no threshold literal), F7 (plain code)."""

import ast
import re
from pathlib import Path

from flysafelz import notam_action as module

PATH = Path(module.__file__)
FORBIDDEN_MODULES = ("openai", "anthropic", "langchain", "langgraph", "httpx")
THRESHOLDS = {2, 5000, 10000, 200}  # 2 NM, the 5,000 and 10,000 ft bands, the 200 ft buffer
TESTS = Path(__file__).parent


def tree() -> ast.Module:
    return ast.parse(PATH.read_text(encoding="utf-8"), filename=str(PATH))


def imported(module_tree: ast.Module) -> list[str]:
    found = []
    for node in ast.walk(module_tree):
        if isinstance(node, ast.Import):
            found += [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            found.append(node.module or "")
    return found


def test_f1_notam_action_imports_nothing_that_could_call_a_model_or_the_network():
    for name in imported(tree()):
        root = name.split(".")[0]
        assert root not in FORBIDDEN_MODULES, name
        assert not root.startswith("langchain"), name
    assert not re.search(r"openai|anthropic|langchain|langgraph|httpx|os\.environ|getenv|dotenv", PATH.read_text(encoding="utf-8"), re.IGNORECASE)


def test_f1_the_test_files_import_no_model_client_either():
    for name in ("test_notam_action", "test_notam_action_blue_mill"):
        path = TESTS / f"{name}.py"
        for found in imported(ast.parse(path.read_text(encoding="utf-8"))):
            assert found.split(".")[0] not in FORBIDDEN_MODULES, (name, found)


def test_f3_no_threshold_literal_in_the_code():
    numbers = [n.value for n in ast.walk(tree()) if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)) and not isinstance(n.value, bool)]
    assert [n for n in numbers if n in THRESHOLDS] == []


def test_f3_no_comparison_in_the_code_holds_a_number_other_than_zero_or_the_fl999_definition():
    allowed = {0, 1, 0.5, module.UNLIMITED_FL}
    for node in ast.walk(tree()):
        if isinstance(node, ast.Compare):
            for operand in [node.left, *node.comparators]:
                if isinstance(operand, ast.Constant) and isinstance(operand.value, (int, float)):
                    assert operand.value in allowed, ast.unparse(node)


def test_f7_no_class_hierarchy_beyond_models():
    bases = {ast.unparse(base) for node in ast.walk(tree()) if isinstance(node, ast.ClassDef) for base in node.bases}
    assert bases <= {"BaseModel", "TimeSpan", "NotamActionResult"}
