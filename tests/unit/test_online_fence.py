"""The online fence (7 PR-1.F1: F2, F4). Text and AST scans: `evals/online/` never reads experiments or the held-out project, and no online test builds a real LangSmith client."""

import ast
import re
import sys
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


def changed_by_the_online_branch(path: str) -> list[str]:
    """What this branch changed under `path`, but only on the branch that built evals/online (it changes files there).

    The F1 fence says the online-eval work leaves 5a (evals/agent) and src/flysafelz alone. Once that work is in dev, any later
    branch (a polish of the deterministic core, say) changes those paths on purpose and has nothing to do with the fence, so on
    a branch that does not touch evals/online this returns [] and the fence is not about it.
    """
    return changed_since_dev(path) if changed_since_dev("evals/online") else []


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
    assert changed_by_the_online_branch("evals/agent") == []
    assert changed_by_the_online_branch("src/flysafelz") == []


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


SEED = ONLINE / "seed.py"  # reads `intended_path` from the seed file to check it (PR-5.F1), and nothing else of a label


@pytest.mark.parametrize("path", ONLINE_FILES, ids=lambda p: p.name)
def test_f7_online_code_names_no_label_field(path):
    labels = [name for name in LABEL_NAMES if name != "intended_path" or path != SEED]  # the seed is the one module that names `intended_path`, and it is held to its own test below
    assert [name for name in labels if name in code_words(path.read_text(encoding="utf-8"))] == []


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


NEW_FILES = [
    ONLINE / "job.py",
    ONLINE / "summary.py",
    ONLINE / "seed.py",
    ONLINE / "seed_questions.yaml",
    ROOT / "tests" / "unit" / "test_readme_production_evals.py",
    ROOT / "tests" / "online_helpers.py",
    ROOT / "tests" / "online_summary_helpers.py",
    ROOT / "tests" / "online_job_helpers.py",
    *sorted((ROOT / "tests" / "unit").glob("test_online_*.py")),
    ROOT / "tests" / "integration" / "test_online_job_rerun.py",
    ROOT / "tests" / "integration" / "test_online_eval_job.py",
]


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


# --- PR-4.F1: the summary's fence (F1, F3, the privacy scan) ----------------------------------------------------------------------------------------------

SUMMARY = ONLINE / "summary.py"
SCHEDULERS = re.compile(r"crontab|launchd|LaunchAgents|dagu", re.IGNORECASE)
LOCAL_PACKAGES = {"evals.agent", "evals.online"}


def modules_imported(source: str) -> set[str]:
    found = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            found.add(node.module)
    return found


def foreign_imports(source: str) -> set[str]:
    """The imported modules that are neither the standard library nor 5a's `evals.agent` nor this package."""
    import sys

    return {name for name in modules_imported(source) if name.split(".")[0] not in sys.stdlib_module_names and not any(name == pkg or name.startswith(pkg + ".") for pkg in LOCAL_PACKAGES)}


def test_f1_the_summary_imports_the_standard_library_5a_and_the_online_package_only():
    assert SUMMARY.exists() and foreign_imports(SUMMARY.read_text(encoding="utf-8")) == set()
    assert {"evals.agent.report", "evals.agent.judge"} <= modules_imported(SUMMARY.read_text(encoding="utf-8"))  # the table formatting and the criteria table come from 5a


def test_f1_the_scan_does_catch_an_import_from_outside_5a():
    assert foreign_imports("import requests\nfrom evals.reader_eval.report import render_markdown\nfrom evals.agent.report import write_report") == {"requests", "evals.reader_eval.report"}


def test_f1_the_summary_writes_through_5as_write_report_and_defines_no_metric_name():
    source = SUMMARY.read_text(encoding="utf-8")
    assert "write_report" in _names_called(source)
    assert not [name for name in string_constants(source) if name.startswith(("online.", "judge.")) and name not in {"online.", "online.judge."}]


def test_f3_nothing_schedules_the_online_eval():
    places = [*ONLINE_FILES, ROOT / "Makefile", ROOT / "README.md", *sorted((ROOT / "scripts").glob("*"))]
    hits = {path.name: SCHEDULERS.findall(path.read_text(encoding="utf-8")) for path in places if path.is_file()}
    assert {name: found for name, found in hits.items() if found} == {}
    assert SCHEDULERS.search("0 8 * * * crontab -l") and SCHEDULERS.search("~/Library/LaunchAgents/x.plist") and SCHEDULERS.search("dagu start online")


def test_f3_the_online_eval_target_is_in_phony_and_nothing_else_calls_it():
    makefile = (ROOT / "Makefile").read_text(encoding="utf-8")
    assert re.search(r"^\.PHONY:.*\bonline-eval\b", makefile, re.MULTILINE)
    recipes = [line for line in makefile.splitlines() if line.startswith("\t")]
    assert not [line for line in recipes if "online-eval" in line.replace("online-eval:", "") and "evals.online.job" not in line]  # no recipe calls the target: the one recipe line runs the module
    test_recipe = re.search(r"^test:[^\n]*\n((?:\t[^\n]*\n?)+)", makefile, re.MULTILINE)
    assert test_recipe and "online-eval" not in test_recipe[1] and "evals.online.job" not in test_recipe[1]


# --- PR-5.F1: the seed's fence (F1, F3, F7) -----------------------------------------------------------------------------------------------------------

SEED_SCHEDULERS = re.compile(r"crontab|launchctl|\.plist|dagu", re.IGNORECASE)
INTENDED_PATH_READ = re.compile(r"""["']intended_path["']|\.intended_path\b|\bintended_path\s*=""")


def scheduling_hits(texts: dict[str, str]) -> dict[str, list[str]]:
    return {name: found for name, text in texts.items() if (found := SEED_SCHEDULERS.findall(text))}


def online_texts() -> dict[str, str]:
    """Every file under evals/online (the Python, the seed file, anything else added later; not the bytecode cache) and the Makefile."""
    files = [path for path in sorted(ONLINE.rglob("*")) if path.is_file() and "__pycache__" not in path.parts]
    return {str(path.relative_to(ROOT)): path.read_text(encoding="utf-8") for path in [*files, ROOT / "Makefile"]}


def test_f3_the_seed_adds_no_cron_launchctl_plist_or_dagu_entry_to_evals_online_or_the_makefile():
    texts = online_texts()
    assert {"evals/online/seed.py", "evals/online/seed_questions.yaml", "Makefile"} <= set(texts)
    assert scheduling_hits(texts) == {}
    assert not [path for path in ROOT.rglob("*.plist") if ".venv" not in path.parts and "node_modules" not in path.parts]


@pytest.mark.parametrize("text", ["0 8 * * * crontab -l", "launchctl load ~/Library/LaunchAgents/x", "com.flysafelz.online.plist", "dagu start online-seed", "DAGU: nightly"])
def test_f3_the_scan_does_catch_each_scheduler(text):
    assert scheduling_hits({"somewhere": text}) == {"somewhere": [SEED_SCHEDULERS.search(text).group(0)]}


def test_f3_the_seed_target_is_in_phony_and_nothing_calls_it():
    makefile = (ROOT / "Makefile").read_text(encoding="utf-8")
    assert re.search(r"^\.PHONY:.*\bonline-seed\b", makefile, re.MULTILINE)
    recipes = [line for line in makefile.splitlines() if line.startswith("\t")]
    assert [line for line in recipes if "evals.online.seed" in line] == ["\t@uv run $(if $(wildcard .env),--env-file .env) python -m evals.online.seed"]
    assert not [line for line in recipes if "online-seed" in line]  # no recipe runs the target, `test` included
    test_recipe = re.search(r"^test:[^\n]*\n((?:\t[^\n]*\n?)+)", makefile, re.MULTILINE)
    assert test_recipe and "online-seed" not in test_recipe[1] and "evals.online.seed" not in test_recipe[1]
    comment = makefile[: makefile.index("\nonline-seed:")].rsplit("\n\n", 1)[-1]
    assert "needs keys" in comment and "only on Dan's Mac" in comment and "`make test` never runs it" in comment


def test_f7_no_module_in_evals_online_but_the_seed_reads_intended_path():
    for path in ONLINE_FILES:
        text = path.read_text(encoding="utf-8")
        if path == SEED:
            continue
        assert "intended_path" not in code_words(text), path.name
        assert INTENDED_PATH_READ.search(text) is None, path.name
    assert "intended_path" in code_words(SEED.read_text(encoding="utf-8"))  # the seed does, to check the file


@pytest.mark.parametrize("source", ["row['intended_path']", "row.intended_path", "intended_path = row", 'x = {"intended_path": 1}'])
def test_f7_the_scan_does_catch_a_read_of_the_label(source):
    assert INTENDED_PATH_READ.search(source) is not None or "intended_path" in code_words(source)


def test_f7_the_seed_sends_intended_path_nowhere():
    """Outside the file check, the seed's code does not hold the name: not in the request, the config or the metadata it builds, nor in the functions that ask."""
    tree = ast.parse(SEED.read_text(encoding="utf-8"))
    builders = {node.name: ast.unparse(node) for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name in {"request_for", "config_for", "ask_question", "main", "default_entry"}}
    assert set(builders) == {"request_for", "config_for", "ask_question", "main", "default_entry"}
    assert [name for name, text in builders.items() if "intended_path" in code_words(text)] == []  # a docstring may say what the code does not do


def test_f1_the_seed_imports_nothing_from_5as_evaluators_or_judge_and_defines_no_metric():
    source = SEED.read_text(encoding="utf-8")
    assert not [name for name in modules_imported(source) if name == "evals.agent" or name.startswith("evals.agent.")]
    assert not [name for name in string_constants(source) if name.startswith(("online.", "judge.")) or name in {"guard_pass", "decision_first", "step_cap"}]
    assert not {name for name in imports_of(source) if "feedback" in name.lower() or "evaluat" in name.lower()}


def test_f1_the_seed_posts_no_feedback_and_calls_nothing_but_the_run_url_and_the_flush():
    source = SEED.read_text(encoding="utf-8")
    called = set(_names_called(source))
    assert not called & {"create_feedback", "list_feedback", "list_runs", "read_project", "create_run", "update_run", *FORBIDDEN_CALLS}
    assert {"get_run_url", "flush", "collect_runs", "tracing_v2_enabled"} <= called
    assert not fence_violations(source) and client_constructions(source) == []


def test_f1_nothing_under_evals_agent_changed_in_this_branch():
    assert changed_by_the_online_branch("evals/agent") == []


def test_the_branch_scoped_fence_ignores_a_branch_that_does_not_touch_evals_online(monkeypatch):
    changed = {"evals/online": [], "evals/agent": ["evals/agent/data/verdict_sheet.yaml"]}
    monkeypatch.setattr(sys.modules[__name__], "changed_since_dev", lambda path: changed[path])
    assert changed_by_the_online_branch("evals/agent") == []  # a later branch that edits 5a on purpose
    changed["evals/online"] = ["evals/online/job.py"]
    assert changed_by_the_online_branch("evals/agent") == ["evals/agent/data/verdict_sheet.yaml"]  # the online branch must not
