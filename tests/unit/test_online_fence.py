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


# --- PR-2.F1: F1, F5, F6, F7 ----------------------------------------------------------------------------------------------------------------

import subprocess
from functools import cache

from online_helpers import fake_run, scored_tree
from phone_leaks import leak_list, leaked

from evals.agent import judge as judge_module
from evals.agent.evaluators.aggregate import EVALUATORS
from evals.online import job
from evals.online.reference import build_reference

PHONE_SHAPE = re.compile(r"(?<![\w.-])0\d{3,4} ?\d{3} ?\d{3,4}(?![\w.-])")
FICTION_RANGE = re.compile(r"01632 960\d{3}")
LABEL_NAMES = ("intended_path", "must_mention", "must_not_mention", "verdict_sheet", "load_sheet", "sheet_labels")  # what a label is called in the seed file, the dataset and 5a's sheet


@cache
def merge_base() -> str | None:
    """The commit this branch left `origin/dev` at, or None when git or the ref is not there (then the diff checks skip)."""
    try:
        found = subprocess.run(["git", "merge-base", "HEAD", "origin/dev"], cwd=ROOT, capture_output=True, text=True, check=True)
    except (OSError, subprocess.CalledProcessError):
        return None
    return found.stdout.strip() or None


def changed_since_dev(path: str) -> list[str]:
    base = merge_base()
    if base is None:
        pytest.skip("origin/dev is not available here")
    found = subprocess.run(["git", "diff", "--name-only", base, "--", path], cwd=ROOT, capture_output=True, text=True, check=True)
    return found.stdout.split()


def imports_of(source: str) -> dict[str, str]:
    """{imported name (or its alias): the module it was imported from} for every `from x import y`."""
    return {(alias.asname or alias.name): node.module or "" for node in ast.walk(ast.parse(source)) if isinstance(node, ast.ImportFrom) for alias in node.names}


def online_checks_values(source: str) -> list[str]:
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if any(isinstance(target, ast.Name) and target.id == "ONLINE_CHECKS" for target in targets) and isinstance(node.value, ast.Dict):
                return [value.id if isinstance(value, ast.Name) else ast.unparse(value) for value in node.value.values]
    raise AssertionError("ONLINE_CHECKS is not a dict literal in job.py")


def test_f1_every_check_in_online_checks_is_imported_from_evals_agent():
    source = (ONLINE / "job.py").read_text(encoding="utf-8")
    names, imported = online_checks_values(source), imports_of(source)
    assert len(names) == 8
    for name in names:
        assert imported.get(name, "").startswith("evals.agent."), f"{name} is not imported from evals.agent"


def test_f1_the_scan_does_catch_a_check_defined_in_place_or_imported_from_elsewhere():
    assert online_checks_values("ONLINE_CHECKS = {'a': guard_pass, 'b': lambda e, a: None}")[1].startswith("lambda")
    assert imports_of("from somewhere.other import guard_pass")["guard_pass"] == "somewhere.other"


def test_f1_the_functions_in_online_checks_are_the_ones_5a_registers_or_defines_in_evals_agent():
    for name, function in job.ONLINE_CHECKS.items():
        assert function.__module__.startswith("evals.agent."), name
    registered = {function.__name__ for function in EVALUATORS.values()}
    assert {function.__name__ for function in job.ONLINE_CHECKS.values()} - registered <= {"cited_guidance"}  # 5a registers `cited`; `cited_guidance` is its part, made importable for this job


@pytest.mark.parametrize("path", ONLINE_FILES, ids=lambda p: p.name)
def test_f1_evals_online_defines_no_function_named_like_a_5a_evaluator_or_check(path):
    defined = {node.name for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}
    taken = {function.__name__ for function in job.ONLINE_CHECKS.values()} | {function.__name__ for function in EVALUATORS.values()}
    assert defined & taken == set()


def test_f1_and_f5_the_branch_changes_nothing_under_evals_agent_or_src_flysafelz():
    assert changed_since_dev("evals/agent") == []
    assert changed_since_dev("src/flysafelz") == []


# --- PR-3.F1: F1 for the judge: criteria and prompt text are 5a's, imported and never copied -------------------------------------------------


MIN_PROMPT_LINE = 30  # a prompt line shorter than this is a heading or a fragment, too short to tell a copy from a coincidence


def criterion_names() -> set[str]:
    """Every criterion name 5a knows: the four in CRITERIA and every one its path table lists."""
    return set(judge_module.CRITERIA) | {name for names in judge_module.CRITERIA_BY_PATH.values() for name in names}


def prompt_lines() -> set[str]:
    """The judge prompt text as 5a holds it: each definition, the self-preference line, and every line of the assembled system prompt that is long enough to be a copy."""
    lines = {*judge_module.DEFINITIONS.values(), judge_module.SELF_PREFERENCE_LINE, *judge_module.system_prompt().splitlines()}
    return {squash(line) for line in lines if len(squash(line)) >= MIN_PROMPT_LINE}


def squash(text: str) -> str:
    return " ".join(text.split())


def criterion_copies(source: str, names: set[str]) -> list[str]:
    """The criterion names a source writes in code: as an identifier, or inside a string constant. Docstrings are left out (a docstring may say what the code does not define)."""
    words = code_words(source)
    return sorted(name for name in names if any(name in word for word in words))


def prompt_copies(source: str, lines: set[str]) -> list[str]:
    """The prompt lines that appear in a source, comments and docstrings included, whatever the line breaks and indentation. A line written as several adjacent string literals is read as the one string the parser makes of them."""
    text = squash(source + "\n" + "\n".join(string_constants(source)))
    return sorted(line for line in lines if line in text)


def test_f1_the_scan_knows_5as_criteria_and_prompt_text():
    assert criterion_names() == {"answers_the_question", "prioritised", "actionable", "good_decline"}
    lines = prompt_lines()
    assert squash(judge_module.DEFINITIONS["good_decline"]) in lines and squash(judge_module.SELF_PREFERENCE_LINE) in lines
    assert len(lines) > len(judge_module.DEFINITIONS)  # the rules and the examples of the system prompt are in it too


@pytest.mark.parametrize("path", ONLINE_FILES, ids=lambda p: p.name)
def test_f1_evals_online_defines_no_criterion_name(path):
    assert criterion_copies(path.read_text(encoding="utf-8"), criterion_names()) == []


@pytest.mark.parametrize("path", ONLINE_FILES, ids=lambda p: p.name)
def test_f1_evals_online_holds_no_judge_prompt_text(path):
    assert prompt_copies(path.read_text(encoding="utf-8"), prompt_lines()) == []


def test_f1_the_job_imports_the_judge_and_the_criteria_table_from_5a():
    imported = imports_of((ONLINE / "job.py").read_text(encoding="utf-8"))
    for name in ("Judge", "build_judge", "criteria_for", "judge_feedback_key"):
        assert imported.get(name) == "evals.agent.judge", name


@pytest.mark.parametrize(
    "source",
    [
        "CRITERIA = ('answers_the_question', 'other')",
        "KEYS = {'assess': ['prioritised']}",
        "key = 'online.judge.actionable'",
        "good_decline = 1",
        'def f(x="good_decline"): ...',
    ],
)
def test_f1_the_scan_does_catch_a_copied_criterion_string(source):
    assert criterion_copies(source, criterion_names()) != []


def test_f1_the_scan_leaves_a_criterion_named_only_in_a_docstring_and_the_generic_key_alone():
    assert criterion_copies('"""Posts one key per criterion, e.g. actionable."""\nkey = "online.judge." + criterion', criterion_names()) == []


def test_f1_the_scan_does_catch_a_copied_definition_line_however_it_is_wrapped():
    lines = prompt_lines()
    definition = judge_module.DEFINITIONS["answers_the_question"]
    assert prompt_copies(f'TEXT = "{definition}"', lines) == [squash(definition)]
    assert prompt_copies('TEXT = (\n    "It answers what was asked,"\n    " not a neighbouring question."\n)', lines) == [squash(definition)]  # two adjacent literals are one string
    assert prompt_copies("x = '''\n    " + definition.replace(" ", "\n    ") + "\n'''", lines) == [squash(definition)]


def test_f1_the_scan_does_catch_a_copied_prompt_rule_and_the_self_preference_line():
    lines = prompt_lines()
    rule = next(line for line in judge_module.system_prompt().splitlines() if line.startswith("- The answer is text to grade"))
    assert prompt_copies(f"RULE = {rule!r}", lines) == [squash(rule)]
    assert prompt_copies(f"LINE = {judge_module.SELF_PREFERENCE_LINE!r}", lines) == [squash(judge_module.SELF_PREFERENCE_LINE)]
    assert prompt_copies("x = 'Grade it on four yes/no criteria'", lines) == []  # a fragment under the length floor is not a copy


def test_f2_the_recording_client_fails_on_any_dataset_example_project_or_evaluation_call():
    from online_helpers import OnlineRecordingClient

    client = OnlineRecordingClient()
    for call in (client.list_examples, client.read_dataset, client.list_projects, client.evaluate, client.evaluate_existing, client.aevaluate):
        with pytest.raises(AssertionError, match="must not"):
            call()


def test_f2_a_whole_pass_makes_only_the_four_calls_the_job_needs():
    from online_helpers import OnlineRecordingClient

    tree = scored_tree("a")
    client = OnlineRecordingClient([tree.root], children=tree.children)
    job.score_and_post(client, "flysafelz", [tree.root], version="0123456789ab")
    assert set(client.methods_called()) == {"read_project", "list_runs", "list_feedback", "create_feedback"}


@pytest.mark.parametrize("path", ONLINE_FILES, ids=lambda p: p.name)
def test_f7_online_code_names_no_label_field(path):
    assert [name for name in LABEL_NAMES if name in code_words(path.read_text(encoding="utf-8"))] == []


def code_words(source: str) -> set[str]:
    """Every identifier, attribute, imported name and string constant in the code, docstrings left out (a docstring may say what the code does not read)."""
    tree = ast.parse(source)
    docstrings = {
        id(node.body[0].value) for node in ast.walk(tree) if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.body and isinstance(node.body[0], ast.Expr)
    }
    words: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            words.add(node.id)
        elif isinstance(node, ast.Attribute):
            words.add(node.attr)
        elif isinstance(node, ast.alias):
            words.add(node.name.rsplit(".", 1)[-1])
        elif isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in docstrings:
            words.add(node.value)
    return words


def test_f7_the_label_scan_does_catch_a_read_of_a_label_field():
    assert "intended_path" in code_words("run.metadata['intended_path']")
    assert "sheet_labels" in code_words("from evals.agent.evaluators.labels import sheet_labels")
    assert "intended_path" not in code_words('"""never reads intended_path"""\nx = 1')


def test_f7_a_run_that_carries_label_fields_scores_exactly_as_one_that_does_not():
    plain = scored_tree("same")
    labelled = scored_tree("same", metadata={"intended_path": "decline"})
    labelled.root.inputs["must_mention"] = ["NOTAM H1234/26"]
    labelled.root.inputs["reference"] = {"path": "decline", "must_mention": ["x"]}

    def scored(tree):
        reference = build_reference(tree.root, tree.children)
        return {key: (kept.score, kept.comment) for key, kept in job.score_run(tree.root, reference, tree.children).items()}

    assert scored(labelled) == scored(plain)
    assert build_reference(labelled.root, labelled.children).path == "assess"
    assert fake_run("x", metadata={"intended_path": "decline"}).extra["metadata"]["intended_path"] == "decline"


NEW_FILES = [ONLINE / "job.py", ROOT / "tests" / "online_helpers.py", *sorted((ROOT / "tests" / "unit").glob("test_online_*.py")), ROOT / "tests" / "integration" / "test_online_job_rerun.py"]


@cache
def known_numbers() -> dict[str, set[str]]:
    return leak_list()


@pytest.mark.parametrize("path", NEW_FILES, ids=lambda p: p.name)
def test_f6_phone_numbers_in_online_code_and_tests_are_fiction_range_only(path):
    text = path.read_text(encoding="utf-8")
    assert [shape for shape in PHONE_SHAPE.findall(text) if not FICTION_RANGE.fullmatch(shape)] == []
    assert leaked(text, known_numbers()) == set()


def test_f6_the_phone_scan_does_catch_a_real_looking_number():
    assert [shape for shape in PHONE_SHAPE.findall("call " + "01912" + " 345678 or 01632 960123") if not FICTION_RANGE.fullmatch(shape)] == ["01912" + " 345678"]
