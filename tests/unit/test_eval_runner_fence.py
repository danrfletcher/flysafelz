"""The fence for the runner (5a PR-4.F1): F1 it imports the agent and changes nothing of it, F3 no key and no real experiment, F7 held-out never by default.

F6 (fiction-range phones, no new LZ coordinates) is test_eval_import_fence.py's scan, which covers evals/agent/*.py and the runner's tests. Every scan here has a
planted violation proving it fires.
"""

import ast
from pathlib import Path

import pytest

from evals.agent import loader

REPO = Path(__file__).resolve().parents[2]
RUNNER_MODULES = [REPO / "evals" / "agent" / "runner.py", REPO / "evals" / "agent" / "run_v1.py"]
NAMES = [p.name for p in RUNNER_MODULES]


def tree(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"))


# --- F1: the agent is imported, not patched, wrapped or subclassed ------------------------------------------------------


def patches_or_wraps(source: str) -> list[str]:
    """Findings: an assignment to an attribute of anything imported from flysafelz, setattr/delattr/patch/monkeypatch calls, and a class that subclasses something imported from flysafelz."""
    module = ast.parse(source)
    imported: set[str] = set()
    for node in ast.walk(module):
        if isinstance(node, ast.ImportFrom) and (node.module or "").split(".")[0] == "flysafelz":
            imported.update(alias.asname or alias.name for alias in node.names)
        elif isinstance(node, ast.Import):
            imported.update(alias.asname or alias.name.split(".")[0] for alias in node.names if alias.name.split(".")[0] == "flysafelz")
    found = []
    for node in ast.walk(module):
        if isinstance(node, (ast.Assign, ast.AugAssign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                root = target
                while isinstance(root, (ast.Attribute, ast.Subscript)):
                    root = root.value
                if isinstance(target, ast.Attribute) and isinstance(root, ast.Name) and root.id in imported:
                    found.append(f"assigns {ast.unparse(target)}")
        elif isinstance(node, ast.Call):
            name = node.func.attr if isinstance(node.func, ast.Attribute) else node.func.id if isinstance(node.func, ast.Name) else ""
            if name in ("setattr", "delattr", "patch", "monkeypatch"):
                found.append(f"calls {name}")
        elif isinstance(node, ast.ClassDef):
            found += [f"subclasses {ast.unparse(b)}" for b in node.bases if isinstance(b, ast.Name) and b.id in imported]
    return found


@pytest.mark.parametrize("path", RUNNER_MODULES, ids=NAMES)
def test_the_runner_does_not_patch_wrap_or_subclass_anything_of_the_agent(path):
    assert patches_or_wraps(path.read_text(encoding="utf-8")) == []


@pytest.mark.parametrize(
    "source",
    [
        "import flysafelz.agent.graph as g\ng.build_graph = lambda: 1",
        "from flysafelz.agent import graph\nsetattr(graph, 'build_graph', None)",
        "from unittest.mock import patch\npatch('flysafelz.agent.graph.build_graph')",
        "from flysafelz.agent.state import RunContext\nclass Mine(RunContext):\n    pass",
        "from flysafelz.agent.state import RunContext\nRunContext.now = 1",
    ],
)
def test_the_patch_scan_catches_a_planted_patch(source):
    assert patches_or_wraps(source)


def test_the_runner_only_reads_the_agent_through_the_names_it_imports():
    imported = set()
    for path in RUNNER_MODULES:
        for node in ast.walk(tree(path)):
            if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("flysafelz"):
                imported.add(node.module)
            if isinstance(node, ast.Import):
                imported.update(a.name for a in node.names if a.name.startswith("flysafelz"))
    assert imported <= {
        "flysafelz.agent.state",
        "flysafelz.agent.middleware",  # is_model_failure only: which model errors the agent's middleware hides behind its fallback answer (read, never patched)
        "flysafelz.agent.tools.notams_tool",
        "flysafelz.redact",
        "flysafelz.agent.config",
        "flysafelz.reader.config",
        "flysafelz.agent.graph",
        "flysafelz.agent.tracing",
    }


# --- the runner opens no file, so it cannot open readings.yaml ---------------------------------------------------------


FILE_CALLS = {"open", "read_text", "read_bytes", "write_text", "write_bytes", "mkdir", "touch", "unlink", "rename", "replace", "mkstemp", "copy", "copyfile", "rmtree", "connect"}


def file_calls(source: str) -> list[str]:
    found = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Call):
            name = node.func.attr if isinstance(node.func, ast.Attribute) else node.func.id if isinstance(node.func, ast.Name) else ""
            if name in FILE_CALLS:
                found.append(name)
    return found


@pytest.mark.parametrize("path", RUNNER_MODULES, ids=NAMES)
def test_the_runner_makes_no_file_call_of_its_own(path):
    assert file_calls(path.read_text(encoding="utf-8")) == []


def test_the_file_scan_catches_a_planted_open():
    assert file_calls("def f(p):\n    p.read_text()\n    open('readings.yaml')") == ["read_text", "open"]


def strings_outside_docstrings(source: str) -> list[str]:
    module = ast.parse(source)
    docstrings = {
        id(n.body[0].value)
        for n in ast.walk(module)
        if isinstance(n, (ast.Module, ast.FunctionDef, ast.ClassDef)) and n.body and isinstance(n.body[0], ast.Expr) and isinstance(n.body[0].value, ast.Constant)
    }
    return [n.value for n in ast.walk(module) if isinstance(n, ast.Constant) and isinstance(n.value, str) and id(n) not in docstrings]


@pytest.mark.parametrize("path", RUNNER_MODULES, ids=NAMES)
def test_the_runner_never_names_readings_yaml_in_code(path):
    assert [s for s in strings_outside_docstrings(path.read_text(encoding="utf-8")) if "readings" in s.lower()] == []


def test_the_reader_is_always_live_in_the_run_context():
    assert "live" in strings_outside_docstrings(RUNNER_MODULES[0].read_text(encoding="utf-8"))
    assert "fixture" not in strings_outside_docstrings(RUNNER_MODULES[0].read_text(encoding="utf-8"))


# --- F3: no key in the build, no real experiment -----------------------------------------------------------------------


def test_nothing_is_built_at_import_the_graph_and_the_client_come_in_through_arguments():
    for path in RUNNER_MODULES:
        top_level_calls = [n for n in tree(path).body if isinstance(n, ast.Expr) and isinstance(n.value, ast.Call)]
        assert top_level_calls == []
    source = (REPO / "evals" / "agent" / "runner.py").read_text(encoding="utf-8")
    assert "build_graph" not in source and "Client(" not in source and "os.environ" not in source


def test_the_command_reads_the_environment_only_in_main_and_only_when_none_is_given():
    source = (REPO / "evals" / "agent" / "run_v1.py").read_text(encoding="utf-8")
    assert source.count("os.environ") == 1  # `environ = os.environ if environ is None else environ`


def test_no_module_in_the_runner_imports_the_network():
    for path in RUNNER_MODULES:
        names = []
        for node in ast.walk(tree(path)):
            if isinstance(node, ast.Import):
                names += [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                names.append(node.module or "")
        assert not [n for n in names if n.split(".")[0] in {"httpx", "requests", "urllib", "socket", "aiohttp", "http", "openai"}]


# --- F7: held-out never by default -------------------------------------------------------------------------------------


def test_the_default_split_is_dev_and_nothing_in_the_runner_names_held_out_as_a_default():
    assert loader.DEFAULT_SPLIT == "dev"
    for path in RUNNER_MODULES:
        for node in ast.walk(tree(path)):
            if isinstance(node, ast.Constant) and node.value == "held_out":
                parent_line = path.read_text(encoding="utf-8").splitlines()[node.lineno - 1]
                assert "default" not in parent_line.lower() or "dev unless" in parent_line, parent_line


def test_make_test_measures_the_runner():
    makefile = (REPO / "Makefile").read_text(encoding="utf-8")
    assert "--cov=evals.agent.runner " in makefile and "--cov-fail-under=90" in makefile
