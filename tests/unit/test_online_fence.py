"""The online fence (7 PR-1.F1: F2, F4). Text and AST scans: `evals/online/` never reads experiments or the held-out project, and no online test builds a real LangSmith client."""

import ast
import re
import tomllib
from pathlib import Path

import pytest

from evals.agent.loader import SPLITS, experiment_target

ROOT = Path(__file__).resolve().parents[2]
ONLINE = ROOT / "evals" / "online"
ONLINE_FILES = sorted(ONLINE.glob("*.py"))
ONLINE_TESTS = sorted((ROOT / "tests" / "unit").glob("test_online_*.py")) + [ROOT / "tests" / "online_helpers.py"]
CLIENT_FACTORY = ONLINE / "job.py"  # the one place a real client may be built (PR-4.F1)
FORBIDDEN_CALLS = ("list_examples", "read_dataset", "list_projects")
FORBIDDEN_EVALUATE = re.compile(r"a?evaluate")  # evaluate, evaluate_existing, evaluate_comparative, aevaluate


def _names_called(source: str) -> list[str]:
    """Every name or attribute that is called or referenced as a function in the source: `x.list_examples(...)`, `evaluate(...)`, also `client.evaluate_existing` taken without a call."""
    found = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Attribute):
            found.append(node.attr)
        elif isinstance(node, ast.Name):
            found.append(node.id)
        elif isinstance(node, ast.alias):
            found.append(node.name.rsplit(".", 1)[-1])
    return found


def fence_violations(source: str) -> list[str]:
    return sorted({name for name in _names_called(source) if name in FORBIDDEN_CALLS or FORBIDDEN_EVALUATE.match(name)})


def client_constructions(source: str) -> list[int]:
    """The lines that call `Client(...)` or `langsmith.Client(...)`."""
    return [
        node.lineno
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Call) and ((isinstance(node.func, ast.Name) and node.func.id == "Client") or (isinstance(node.func, ast.Attribute) and node.func.attr == "Client"))
    ]


def string_constants(source: str) -> list[str]:
    return [node.value for node in ast.walk(ast.parse(source)) if isinstance(node, ast.Constant) and isinstance(node.value, str)]


def test_the_scan_finds_the_online_files():
    assert {path.name for path in ONLINE_FILES} >= {"pull.py", "reference.py", "runs.py"}
    assert len(ONLINE_TESTS) >= 5


# --- F2 -------------------------------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("path", ONLINE_FILES, ids=lambda p: p.name)
def test_f2_online_code_never_reads_datasets_examples_projects_or_runs_an_evaluation(path):
    assert fence_violations(path.read_text(encoding="utf-8")) == []


@pytest.mark.parametrize(
    "source",
    [
        "client.list_examples(dataset_id=1)",
        "client.read_dataset(dataset_name='x')",
        "client.list_projects()",
        "evaluate(target, data='x')",
        "client.evaluate_existing('exp')",
        "client.aevaluate(f)",
        "from langsmith import evaluate",
        "fn = client.list_projects",
    ],
)
def test_f2_the_scan_does_catch_each_forbidden_call(source):
    assert fence_violations(source) != []


def test_f2_the_scan_leaves_the_calls_the_pull_makes_alone():
    assert fence_violations("client.list_runs(project_name=p, is_root=True, start_time=t)\nclient.read_project(project_name=p)\nclient.create_feedback(run_id, key)") == []


@pytest.mark.parametrize("path", ONLINE_FILES, ids=lambda p: p.name)
def test_f2_online_code_does_not_import_the_offline_runners(path):
    text = path.read_text(encoding="utf-8")
    for module in ("evals.agent.runner", "evals.agent.run_v1", "evals.agent.v0", "evals.agent.annotate", "evals.agent.agreement"):
        assert f"import {module}" not in text and f"from {module} " not in text, module


def test_f2_the_protected_project_names_are_read_from_5as_config_and_written_nowhere_in_online_code():
    names = {experiment_target(split).project for split in SPLITS} | {experiment_target(split).dataset for split in SPLITS}
    assert len(names) == 2 and all(names)
    for path in ONLINE_FILES:
        literals = {text.strip().casefold() for text in string_constants(path.read_text(encoding="utf-8"))}
        assert not {name.casefold() for name in names} & literals, f"{path.name} writes a protected project name as a literal"


def test_f2_the_guard_imports_the_names_from_5as_config():
    text = (ONLINE / "pull.py").read_text(encoding="utf-8")
    assert "from evals.agent.loader import experiment_target" in text


# --- F4 -------------------------------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("path", [p for p in ONLINE_FILES if p != CLIENT_FACTORY] + ONLINE_TESTS, ids=lambda p: p.name)
def test_f4_no_real_langsmith_client_is_constructed_outside_the_jobs_factory(path):
    assert client_constructions(path.read_text(encoding="utf-8")) == []


def test_f4_the_scan_does_catch_a_client_construction():
    assert client_constructions("import langsmith\nc = langsmith.Client(api_key='k')") == [2]
    assert client_constructions("from langsmith import Client\nc = Client()") == [2]
    assert client_constructions("c = OnlineRecordingClient(runs)") == []


def test_f4_sockets_stay_off_for_the_whole_suite():
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert "--disable-socket" in pyproject["tool"]["pytest"]["ini_options"]["addopts"]


def test_f4_the_online_tests_use_5as_recording_client_and_no_langsmith_import_of_a_client():
    helpers = (ROOT / "tests" / "online_helpers.py").read_text(encoding="utf-8")
    assert "from evals.agent.recording_client import RecordingClient" in helpers and "class OnlineRecordingClient(RecordingClient)" in helpers


def test_the_existing_fence_still_keeps_langsmith_out_of_src_and_scripts():
    text = (ROOT / "tests" / "unit" / "test_fence.py").read_text(encoding="utf-8")
    assert 'FORBIDDEN_IMPORTS = ("langgraph", "langchain", "langsmith", "openai", "anthropic")' in text
    assert not list((ROOT / "src").rglob("online*")) and ONLINE.parent.name == "evals"
