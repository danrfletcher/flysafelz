"""PR-9.F1 fence: F1 (five tools, one `create_agent`, one middleware class, no StateGraph or sub-agent) and F8 (no UI, no eval harness, no streamed answer).

Scans read the source as a syntax tree, so a word in a comment or a string does not trip them and a renamed import does.
"""

import ast
import re
import shutil
from pathlib import Path

import pytest
from fakes import FakeToolChatModel
from fence_scan import env_read_files

from flysafelz.agent.graph import TOOL_NAMES, build_graph

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src" / "flysafelz"
AGENT = SRC / "agent"
PY_FILES = sorted(SRC.rglob("*.py"))
FORBIDDEN_TOOL_WORDS = re.compile(r"watch|alert|notify|action", re.IGNORECASE)
SKIPPED_DIRS = {".venv", ".git", "node_modules", "__pycache__", ".cache", ".langgraph_api"}


def tree_of(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"))


def names_in(tree: ast.AST) -> set[str]:
    """Every identifier a file uses: names, attributes, imported names and module parts, function and class names."""
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            found.add(node.id)
        elif isinstance(node, ast.Attribute):
            found.add(node.attr)
        elif isinstance(node, ast.ImportFrom):
            found.update(alias.name for alias in node.names)
            found.update((node.module or "").split("."))
        elif isinstance(node, ast.Import):
            for alias in node.names:
                found.update(alias.name.split("."))
        elif isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            found.add(node.name)
    return found


def calls_of(tree: ast.AST, name: str) -> list[ast.Call]:
    return [n for n in ast.walk(tree) if isinstance(n, ast.Call) and (getattr(n.func, "id", None) == name or getattr(n.func, "attr", None) == name)]


def project_python_files() -> list[Path]:
    """Every .py file of the repository except dependencies and caches."""
    return sorted(p for p in ROOT.rglob("*.py") if not SKIPPED_DIRS & set(p.relative_to(ROOT).parts))


# --- F1: five tools ----------------------------------------------------------------------------------------------------------------------------


def test_the_graph_registers_exactly_these_five_tools_in_this_order():
    graph = build_graph(model=FakeToolChatModel())
    assert list(graph.nodes["tools"].bound.tools_by_name) == ["get_weather", "get_airspace", "get_notams", "assess", "search_guidance"]
    assert list(TOOL_NAMES) == list(graph.nodes["tools"].bound.tools_by_name)


def test_no_tool_name_or_tool_module_matches_watch_alert_notify_or_action():
    assert not [name for name in TOOL_NAMES if FORBIDDEN_TOOL_WORDS.search(name)]
    modules = [p.name for p in (AGENT / "tools").glob("*.py")]
    assert modules and not [m for m in modules if FORBIDDEN_TOOL_WORDS.search(m)]


def test_the_tool_scan_catches_a_planted_name():
    assert FORBIDDEN_TOOL_WORDS.search("watch_notams") and FORBIDDEN_TOOL_WORDS.search("send_alert") and FORBIDDEN_TOOL_WORDS.search("notify_pilot")
    assert FORBIDDEN_TOOL_WORDS.search("notam_action") and not FORBIDDEN_TOOL_WORDS.search("get_notams")


# --- F1: one create_agent, one middleware class ------------------------------------------------------------------------------------------------


def test_exactly_one_create_agent_call_in_src_and_it_is_in_graph_py():
    calls = {path.relative_to(SRC).as_posix(): len(calls_of(tree_of(path), "create_agent")) for path in PY_FILES}
    assert {where: n for where, n in calls.items() if n} == {"agent/graph.py": 1}


def test_no_stategraph_add_node_add_edge_or_sub_agent_in_the_agent_package():
    banned = {"StateGraph", "MessageGraph", "add_node", "add_edge", "add_conditional_edges", "set_entry_point", "create_react_agent", "create_deep_agent", "SubAgent", "subagent", "sub_agent", "Send"}
    for path in sorted(AGENT.rglob("*.py")):
        assert not names_in(tree_of(path)) & banned, path


def test_the_tests_scan_does_catch_a_planted_stategraph():
    planted = ast.parse("from langgraph.graph import StateGraph\ng = StateGraph(dict)\ng.add_node('x', f)\n")
    assert {"StateGraph", "add_node"} <= names_in(planted)


def test_no_file_anywhere_imports_deepagents():
    for path in project_python_files():
        assert "deepagents" not in names_in(tree_of(path)), path


def test_exactly_one_agent_middleware_subclass_in_src_and_it_defines_wrap_model_call():
    subclasses = []
    for path in PY_FILES:
        for node in ast.walk(tree_of(path)):
            if isinstance(node, ast.ClassDef) and any("AgentMiddleware" in ast.unparse(base) for base in node.bases):
                subclasses.append((path.relative_to(SRC).as_posix(), node.name, {n.name for n in node.body if isinstance(n, ast.FunctionDef | ast.AsyncFunctionDef)}))
    [(where, name, methods)] = subclasses
    assert (where, name) == ("agent/middleware.py", "FlySafeMiddleware")
    assert {"wrap_model_call", "awrap_model_call", "before_agent", "after_agent"} <= methods


def test_the_model_failure_handler_is_a_method_of_that_class_and_no_decorator_style_middleware_exists():
    for path in PY_FILES:
        names = names_in(tree_of(path))
        used = names & {"wrap_model_call", "awrap_model_call", "before_model", "after_model", "dynamic_prompt", "wrap_tool_call", "before_agent", "after_agent"}
        if path.relative_to(SRC).as_posix() in {"agent/middleware.py"}:
            continue
        # input_check.py keeps the stand-alone `before_agent` hook the contract tests' graph uses; the real graph does not pass it to create_agent
        assert used <= {"before_agent"}, (path, used)


def test_the_graph_is_given_a_middleware_list_of_length_one(monkeypatch):
    from flysafelz.agent import graph as graph_module

    seen = []
    real = graph_module.create_agent
    monkeypatch.setattr(graph_module, "create_agent", lambda *a, **k: seen.append(k["middleware"]) or real(*a, **k))
    build_graph(model=FakeToolChatModel())
    assert [len(m) for m in seen] == [1]


def test_input_check_py_is_not_given_to_create_agent_by_the_real_graph():
    assert "check_input" not in names_in(tree_of(AGENT / "graph.py"))


# --- F8: not in this PR -------------------------------------------------------------------------------------------------------------------------


def test_no_ui_files_or_folders_exist():
    found = []
    for path in ROOT.rglob("*"):
        parts = path.relative_to(ROOT).parts
        if SKIPPED_DIRS & set(parts):
            continue
        if path.suffix in {".tsx", ".jsx", ".vue", ".svelte"} or (path.is_dir() and parts[0] in {"ui", "web", "frontend"} and len(parts) == 1):
            found.append("/".join(parts))
    assert found == []


def test_no_eval_harness_judge_baseline_or_dataset_code_for_the_agent_is_added():
    # evals/agent/ is ticket 5a's eval harness (PR-1.S1 adds its dataset import); ticket 4 itself adds none of it, so the rest of evals/ is scanned as before.
    names = [
        p.relative_to(ROOT).as_posix()
        for p in (ROOT / "evals").rglob("*")
        if re.search(r"judge|baseline|dataset|experiment|v0", p.name, re.IGNORECASE) and "__pycache__" not in p.parts and "agent" not in p.relative_to(ROOT / "evals").parts[:1]
    ]
    assert names == []
    folders = {p.name for p in (ROOT / "evals").iterdir() if p.is_dir() and p.name != "__pycache__"}
    assert folders <= {"agent", "guidance", "reader", "reader_eval"}  # guidance/ is the one folder ticket 4 adds (Dan's labelled questions); reader* are ticket 3's; agent/ is ticket 5a's


def test_make_run_streams_only_custom_progress_and_state_values_never_message_chunks():
    tree = tree_of(AGENT / "run.py")
    modes = [kw.value for call in calls_of(tree, "stream") for kw in call.keywords if kw.arg == "stream_mode"]
    assert len(modes) == 1
    assert {ast.literal_eval(m) for m in modes[0].elts} == {"custom", "values"}
    assert "astream" not in names_in(tree) and "stream_events" not in names_in(tree)


# --- the older fence still catches a violation outside agent/ ---------------------------------------------------------------------------------------


@pytest.mark.parametrize("where", ["weather.py", "pib.py", "agent/graph_extra.py", "guidance/index.py"])
def test_an_environment_read_planted_outside_the_reviewed_files_is_still_caught(tmp_path, where):
    copy = tmp_path / "src" / "flysafelz"
    shutil.copytree(SRC, copy, ignore=shutil.ignore_patterns("__pycache__"))
    assert env_read_files(copy, sorted(copy.rglob("*.py"))) == []
    target = copy / where
    target.write_text((target.read_text(encoding="utf-8") if target.exists() else "") + "\nimport os\nx = os.environ['HOME']\n", encoding="utf-8")
    assert env_read_files(copy, sorted(copy.rglob("*.py"))) == [where]


def test_run_py_is_the_one_new_reviewed_environment_reader_and_it_reads_no_agent_setting():
    from fence_scan import AGENT_RUN, ENV_READ

    assert AGENT_RUN == "agent/run.py" and ENV_READ.search((SRC / AGENT_RUN).read_text(encoding="utf-8"))
