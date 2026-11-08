"""F6 and F7 for the new modules: LangChain only under agent/, no model client, and no number in them that is a rules.yaml value."""

import ast
from pathlib import Path

import pytest
from fence_scan import hard_coded_rule_values, import_violations, unreviewed_literals

from flysafelz.rules import RULES

SRC = Path(__file__).resolve().parents[3] / "src" / "flysafelz"
TOOLS = sorted((SRC / "agent" / "tools").glob("*.py"))
NEW_MODULES = [*TOOLS, SRC / "shapes.py"]
MODEL_CLIENTS = {"langchain_openai", "openai", "anthropic", "flysafelz.reader.read", "flysafelz.reader.config", "flysafelz.reader.settings"}


def tree_of(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def imported(path: Path) -> set[str]:
    found: set[str] = set()
    for node in ast.walk(tree_of(path)):
        if isinstance(node, ast.Import):
            found |= {alias.name for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            found.add(node.module)
            found |= {f"{node.module}.{alias.name}" for alias in node.names}
    return found


def rule_numbers() -> set[float]:
    return {abs(float(rule.value)) for rule in vars(RULES).values() if hasattr(rule, "value") and isinstance(rule.value, (int, float)) and not isinstance(rule.value, bool)}


def test_the_tool_modules_are_found():
    assert {p.name for p in TOOLS} >= {"weather_tool.py", "airspace_tool.py", "notams_tool.py", "assess_tool.py", "common.py", "artifacts.py"}


@pytest.mark.parametrize("path", NEW_MODULES, ids=lambda p: p.name)
def test_the_import_scan_passes(path):
    assert import_violations(path, SRC) == []


@pytest.mark.parametrize("path", NEW_MODULES, ids=lambda p: p.name)
def test_no_model_client_is_imported(path):
    assert not (imported(path) & MODEL_CLIENTS)


@pytest.mark.parametrize("path", NEW_MODULES, ids=lambda p: p.name)
def test_no_number_in_it_is_a_rules_yaml_value(path):
    assert hard_coded_rule_values(path, tree_of(path), rule_numbers()) == []


@pytest.mark.parametrize("path", TOOLS, ids=lambda p: p.name)
def test_the_tool_modules_hold_no_number_but_0_and_1(path):
    assert unreviewed_literals(path, tree_of(path)) == []


def test_the_scan_sees_a_planted_literal():
    planted = ast.parse("limit = 17.5\n")
    assert unreviewed_literals(Path("weather_tool.py"), planted) == [(1, 17.5)]


def test_the_assess_tool_never_builds_a_chat_model_or_reads_the_environment():
    """F2/F6 for G12: the verdict tool takes the reader's model from the source cache like get_notams, and builds none of its own."""
    path = SRC / "agent" / "tools" / "assess_tool.py"
    names = {node.id for node in ast.walk(tree_of(path)) if isinstance(node, ast.Name)} | {node.attr for node in ast.walk(tree_of(path)) if isinstance(node, ast.Attribute)}
    assert not names & {"ChatOpenAI", "init_chat_model", "reader_llm", "environ", "getenv"}
    assert "read_chosen" in names  # the NOTAMs are read through the one shared reading step


@pytest.mark.parametrize("module", ["assess.py", "notam_action.py"])
def test_ticket_2s_modules_import_nothing_from_the_agent(module):
    assert not [name for name in imported(SRC / module) if name == "flysafelz.agent" or name.startswith("flysafelz.agent.")]
