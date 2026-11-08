"""The fence for the evaluators (5a PR-5.F1): F1 they read the agent and the reader eval and change nothing (the empty `git diff` of src/ and tests/golden is the tester's transcript, not a test: a later slice may change src/), F2 no model, LangSmith or network client, F7 no held-out id in these tests.

F6 (no hard-coded rules.yaml value) is test_fence.py's scan, extended to evals/agent/evaluators/. Every scan here has a planted violation proving it fires.
"""

import ast
from pathlib import Path

import pytest
import yaml
from test_eval_runner_fence import patches_or_wraps

REPO = Path(__file__).resolve().parents[3]
EVALUATORS = REPO / "evals" / "agent" / "evaluators"
MODULES = sorted(EVALUATORS.glob("*.py"))
TESTS = Path(__file__).resolve().parent
BANNED_PREFIXES = ("langsmith", "langchain", "langgraph", "httpx", "openai", "anthropic", "requests", "urllib3", "socket")


def imported_modules(source: str) -> list[str]:
    found = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            found += [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
            found.append(node.module)
    return found


def banned_imports(source: str) -> list[str]:
    return [m for m in imported_modules(source) if m.split(".")[0].startswith(BANNED_PREFIXES) or m.split(".")[0] in BANNED_PREFIXES]


def test_the_scan_finds_the_evaluator_modules():
    names = {p.name for p in MODULES}
    assert {"minutes.py", "answer.py", "labels.py", "blocker.py", "verdict.py", "paths.py", "safety.py", "aggregate.py", "__init__.py"} <= names


# --- F2 ----------------------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("path", MODULES, ids=lambda p: p.name)
def test_f2_an_evaluator_module_imports_no_model_langsmith_or_network_client(path):
    assert banned_imports(path.read_text(encoding="utf-8")) == []


@pytest.mark.parametrize(
    "source", ["import langsmith", "from langchain_core.messages import X", "import httpx", "from httpx import Client", "from langsmith.evaluation import evaluate", "import openai"]
)
def test_f2_the_import_scan_catches_a_planted_import(source):
    assert banned_imports(source)


def test_f2_the_modules_import_flysafelz_and_the_reader_eval_only_for_reading():
    imported = {m.split(".")[0] for path in MODULES for m in imported_modules(path.read_text(encoding="utf-8"))}
    assert {"flysafelz", "evals"} <= imported


# --- F1 ----------------------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("path", MODULES, ids=lambda p: p.name)
def test_f1_no_evaluator_module_patches_wraps_or_subclasses_anything_of_the_agent(path):
    assert patches_or_wraps(path.read_text(encoding="utf-8")) == []


# --- F7 ----------------------------------------------------------------------------------------------------------------------------


def held_out_ids() -> list[str]:
    examples = yaml.safe_load((REPO / "evals" / "agent" / "data" / "examples.yaml").read_text(encoding="utf-8"))
    return [e["id"] for e in examples if (e.get("metadata") or {}).get("split") == "held_out"]


def files_naming(ids: list[str], folder: Path) -> list[str]:
    return sorted(f"{p.name}: {i}" for p in folder.rglob("*.py") for i in ids if i in p.read_text(encoding="utf-8"))


def test_f7_there_are_held_out_ids_to_look_for():
    ids = held_out_ids()
    assert len(ids) == 15 and all(i.startswith("ho_") for i in ids)


def test_f7_no_held_out_example_id_appears_in_these_tests(tmp_path):
    assert files_naming(held_out_ids(), TESTS) == []


def test_f7_the_scan_catches_a_planted_id(tmp_path):
    (tmp_path / "test_x.py").write_text(f"ID = '{held_out_ids()[0]}'\n", encoding="utf-8")
    assert files_naming(held_out_ids(), tmp_path) == [f"test_x.py: {held_out_ids()[0]}"]
