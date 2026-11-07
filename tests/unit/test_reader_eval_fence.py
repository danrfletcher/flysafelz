"""Fence checks for the reader eval package (ticket 3 F2, F4, F6, F7). Each scan is shown to catch a planted violation in a tmp_path copy.

Nothing is planted in the real tree. The scans read evals/reader_eval/*.py as text and as an AST; they import nothing from it.

run.py is the one file that reads a model and writes anything: it may import the reader's model-calling modules (never an LLM
client package itself) and write only the report folder, in the two functions named in REPORT_WRITERS. Every other file
imports none of those modules and writes nothing.
"""

import ast
import re
import shutil
from pathlib import Path

import pytest
import yaml
from fence_scan import hard_coded_rule_values
from test_reader_fence import LLM_PACKAGES, imports_of, tree_of

from flysafelz.rules import RULES

ROOT = Path(__file__).resolve().parents[2]
EVAL = ROOT / "evals" / "reader_eval"
EVAL_FILES = sorted(EVAL.glob("*.py"))
SETTINGS_YAML = EVAL / "settings.yaml"
READER_SETTINGS_YAML = ROOT / "src" / "flysafelz" / "reader" / "settings.yaml"
RUNNER = EVAL / "run.py"
# The only functions of run.py that write: they make the run folder and put the two reports in it.
REPORT_WRITERS = {"make_run_folder", "write_report"}
# Identifiers the spec rules out of this ticket (F7): a few-shot prompt, a judge or a guard around the scoring, and putting a phone number back.
BANNED_IDENTIFIERS = re.compile(r"few_?shot|judge|guard|rehydrat|unredact", re.IGNORECASE)
# The reader modules that call a model or read its configuration. reader.contract (the shapes of a reading) is not one of them.
MODEL_CALLING_READER_MODULES = {"flysafelz.reader.read", "flysafelz.reader.config", "flysafelz.reader.cached_read", "flysafelz.reader.prompt", "flysafelz.reader.cache", "flysafelz.reader.reader_set"}
WRITE_MODES = set("wax+")
WRITE_METHODS = {
    "write_text",
    "write_bytes",
    "unlink",
    "touch",
    "rename",
    "replace",
    "rmdir",
    "mkdir",
    "dump",
    "safe_dump",
    "dump_all",
    "safe_dump_all",
    "copy",
    "copyfile",
    "move",
    "rmtree",
    "remove",
}
# What the planted violations are built from: written in pieces so that this file never matches its own scans.
LABELS = "lab" + "els.yaml"


def settings_values() -> set[float]:
    data = yaml.safe_load(SETTINGS_YAML.read_text(encoding="utf-8"))
    return {value for value in data.values() if isinstance(value, (int, float))}


def llm_imports(path: Path, *, reads_model: bool = False) -> list[str]:
    """The LLM client packages and the model-calling reader modules a file imports. A file that `reads_model` (run.py) may import the modules, never a package."""
    return [name for name in imports_of(path) if name.split(".")[0] in LLM_PACKAGES or (not reads_model and name in MODEL_CALLING_READER_MODULES)]


def write_calls(path: Path, *, outside: frozenset[str] | set[str] = frozenset()) -> list[tuple[int, str]]:
    """(line, what) for each call that could write: open() with a write mode, and any write, delete, move or dump method.

    Calls inside a function named in `outside` are not counted: that is where a file is allowed to write.
    """
    tree = tree_of(path)
    allowed = {id(node) for fn in ast.walk(tree) if isinstance(fn, ast.FunctionDef) and fn.name in outside for node in ast.walk(fn)}
    found = []
    for node in ast.walk(tree):
        if id(node) in allowed:
            continue
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = func.id if isinstance(func, ast.Name) else func.attr if isinstance(func, ast.Attribute) else ""
        if name == "open":
            # open(path, mode) / Path.open(mode) / open(path, mode="w"): the mode is the second argument of the function or the first of the method
            args = list(node.args[1:] if isinstance(func, ast.Name) else node.args)
            modes = [a.value for a in args[:1] if isinstance(a, ast.Constant)] + [k.value.value for k in node.keywords if k.arg == "mode" and isinstance(k.value, ast.Constant)]
            if any(isinstance(m, str) and WRITE_MODES & set(m) for m in modes) or any(not isinstance(a, ast.Constant) for a in args[:1]):
                found.append((node.lineno, "open() with a write or unknown mode"))
        elif name in WRITE_METHODS:
            found.append((node.lineno, name))
    return found


def plant(tmp_path: Path, source: str) -> Path:
    path = tmp_path / "planted.py"
    path.write_text(source, encoding="utf-8")
    return path


def test_the_package_is_where_the_scans_look():
    assert {path.name for path in EVAL_FILES} >= {"__init__.py", "blue_mill.py", "downgrade.py", "fields.py", "items.py", "labels.py", "report.py", "run.py", "settings.py", "spots.py"}
    assert (ROOT / "evals" / "__init__.py").exists() and SETTINGS_YAML.exists()


# --- F2: no model --------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("path", EVAL_FILES, ids=lambda p: p.name)
def test_f2_the_eval_imports_no_llm_client(path):
    assert llm_imports(path, reads_model=path == RUNNER) == []


def test_f2_only_the_runner_imports_the_reader_modules_that_call_a_model():
    """run.py builds the model and reads through flysafelz.reader; nothing else in the eval does."""
    importers = {path.name for path in EVAL_FILES if set(imports_of(path)) & MODEL_CALLING_READER_MODULES}
    assert importers == {"run.py"}
    assert "flysafelz.reader.config" in imports_of(RUNNER)  # it builds the model through the reader's own config


def test_f2_the_runner_takes_its_model_as_a_parameter():
    """Tests hand in a fake: evaluate() and main() both have an `llm` parameter."""
    functions = {node.name: [a.arg for a in node.args.args] for node in ast.walk(tree_of(RUNNER)) if isinstance(node, ast.FunctionDef)}
    assert "llm" in functions["evaluate"] and "llm" in functions["main"]


def test_f2_notam_action_stays_model_free():
    assert llm_imports(ROOT / "src" / "flysafelz" / "notam_action.py") == []
    assert not [name for name in imports_of(ROOT / "src" / "flysafelz" / "notam_action.py") if name.startswith("flysafelz.reader")]


@pytest.mark.parametrize(
    "source",
    [
        "import openai\n",
        "from langchain_openai import ChatOpenAI\n",
        "import anthropic.types\n",
        "from langgraph.graph import StateGraph\n",
        "from flysafelz.reader.read import read_notam\n",
        "from flysafelz.reader.config import load_config\n",
    ],
)
def test_f2_a_planted_llm_import_is_caught(tmp_path, source):
    assert llm_imports(plant(tmp_path, source))


@pytest.mark.parametrize("source", ["import openai\n", "from langchain_core.language_models.chat_models import BaseChatModel\n", "from langchain_openai import ChatOpenAI\n"])
def test_f2_the_runner_may_not_import_a_client_package_either(tmp_path, source):
    assert llm_imports(plant(tmp_path, source), reads_model=True)


def test_f2_the_runner_may_import_the_reader_modules(tmp_path):
    assert llm_imports(plant(tmp_path, "from flysafelz.reader.config import build_llm\n"), reads_model=True) == []


def test_f2_the_reader_contract_import_is_allowed():
    assert any(name == "flysafelz.reader.contract" for path in EVAL_FILES for name in imports_of(path))  # the eval does use it
    assert llm_imports(ROOT / "evals" / "reader_eval" / "labels.py") == []


# --- F4: labels.yaml is only ever read -------------------------------------------------------------------------------------------
# The build commits no labels.yaml: that is held by the scan below (nothing in the eval can write one) and by this slice adding none.
# It is deliberately not asserted as "no labels.yaml exists in the tree": Dan's own evals/reader/labels.yaml is added on this branch
# at Review and may sit untracked while he runs the eval, and make test must pass with it present.


@pytest.mark.parametrize("path", EVAL_FILES, ids=lambda p: p.name)
def test_f4_no_code_in_the_eval_writes_anything_but_the_report_writers(path):
    assert write_calls(path, outside=REPORT_WRITERS if path == RUNNER else set()) == []


def test_f4_the_runner_does_write_the_report_in_its_two_functions():
    """The exemption is real (the writers exist and do write) and narrow (they are the only writers)."""
    assert len(write_calls(RUNNER)) >= 2
    assert write_calls(RUNNER, outside=REPORT_WRITERS) == []
    names = {node.name for node in ast.walk(tree_of(RUNNER)) if isinstance(node, ast.FunctionDef)}
    assert REPORT_WRITERS <= names


def test_f4_a_write_outside_the_report_writers_is_caught_in_the_runner(tmp_path):
    source = f"from pathlib import Path\ndef write_report(folder):\n    (folder / 'report.md').write_text('')\ndef sneaky():\n    Path('{LABELS}').write_text('')\n"
    found = write_calls(plant(tmp_path, source), outside=REPORT_WRITERS)
    assert [what for _line, what in found] == ["write_text"]


def test_f4_the_runner_never_names_the_labels_file_to_a_writer():
    """labels.yaml reaches run.py only as paths.labels, handed to the loader and to `.is_file()`."""
    text = RUNNER.read_text(encoding="utf-8")
    uses = [line.strip() for line in text.splitlines() if "paths.labels" in line]
    assert uses and all("load_labels(" in line or "reject_malformed(" in line or "is_file()" in line or "shown(" in line for line in uses)


@pytest.mark.parametrize(
    "source",
    [
        f'from pathlib import Path\nPath("{LABELS}").write_text("")\n',
        f'open("{LABELS}", "w")\n',
        f'open("{LABELS}", mode="a")\n',
        f'open("{LABELS}", "r+")\n',
        f'from pathlib import Path\nPath("{LABELS}").open("w")\n',
        f'import yaml\nyaml.dump({{}}, open("{LABELS}", "rb"))\n',
        "import yaml\nyaml.safe_dump({}, stream)\n",
        f'from pathlib import Path\nPath("{LABELS}").unlink()\n',
        "def f(path, mode):\n    return open(path, mode)\n",
        f'from pathlib import Path\nPath("{LABELS}").write_bytes(b"")\n',
    ],
)
def test_f4_a_planted_write_is_caught(tmp_path, source):
    assert write_calls(plant(tmp_path, source))


@pytest.mark.parametrize("source", ['open("x.yaml")\n', 'open("x.yaml", "r")\n', 'from pathlib import Path\nPath("x").read_text()\n', "import yaml\nyaml.safe_load('a: 1')\n"])
def test_f4_reading_is_not_a_write(tmp_path, source):
    assert write_calls(plant(tmp_path, source)) == []


# --- F6: no threshold literal ----------------------------------------------------------------------------------------------------


def test_f6_the_settings_values_are_the_ones_the_spec_names():
    assert settings_values() >= {0.1, 1.9, 500, 100}


@pytest.mark.parametrize("path", EVAL_FILES, ids=lambda p: p.name)
def test_f6_no_number_in_the_eval_equals_a_settings_value(path):
    assert hard_coded_rule_values(path, tree_of(path), settings_values()) == []


@pytest.mark.parametrize("literal", ["1.9", "0.1", "500", "100", "-1.9", "500.0"])
def test_f6_a_planted_settings_value_is_caught(tmp_path, literal):
    path = plant(tmp_path, f"ring = {literal}\n")
    assert hard_coded_rule_values(path, tree_of(path), settings_values())


def test_f6_a_planted_copy_of_a_real_module_is_caught_where_it_hard_codes_the_outer_ring(tmp_path):
    copy = tmp_path / "spots.py"
    shutil.copy(EVAL / "spots.py", copy)
    clean = copy.read_text(encoding="utf-8")
    assert hard_coded_rule_values(copy, ast.parse(clean), settings_values()) == []
    planted = clean.replace("settings.ring_outer_nm", "1.9", 1)
    assert planted != clean
    assert hard_coded_rule_values(copy, ast.parse(planted), settings_values())


def test_f6_the_structural_zero_and_one_are_kept_and_the_scan_still_catches_the_neighbour(tmp_path):
    path = plant(tmp_path, "start = 0\nstep = 1\nratio = 0.0\nring = 1.9\n")
    found = hard_coded_rule_values(path, tree_of(path), settings_values() | {0, 1})
    assert [value for _line, value in found] == [1.9]  # 0, 1 and 0.0 are structural and pass; 1.9 does not


def test_f6_the_pressure_is_read_from_the_rules_not_written():
    pressure = RULES.fl_reference_hpa.value
    assert not isinstance(pressure, list)
    assert "RULES.fl_reference_hpa" in (EVAL / "settings.py").read_text(encoding="utf-8")
    for path in EVAL_FILES:
        assert hard_coded_rule_values(path, tree_of(path), {float(pressure)}) == [], path.name
    planted = ROOT / "planted_pressure.py"  # not written: a tree is parsed from text to show the scan would catch the literal
    tree = ast.parse(f"pressure = {float(pressure)}\n")
    assert hard_coded_rule_values(planted, tree, {float(pressure)})


# --- F6: runs and concurrency come from the reader's settings, by name ----------------------------------------------------------


def reader_settings_values() -> set[float]:
    data = yaml.safe_load(READER_SETTINGS_YAML.read_text(encoding="utf-8"))
    return {value for value in data.values() if isinstance(value, (int, float))}


def test_f6_the_runner_reads_runs_and_concurrency_by_name():
    text = RUNNER.read_text(encoding="utf-8")
    assert "READER_SETTINGS.runs" in text and "READER_SETTINGS.concurrency" in text
    assert reader_settings_values() >= {3, 45}


# The files of this slice: the older ones use a 3 for the sides of a triangle, which is not the number of runs.
RUN_FILES = [EVAL / "run.py", EVAL / "report.py", EVAL / "blue_mill.py"]


@pytest.mark.parametrize("path", RUN_FILES, ids=lambda p: p.name)
def test_f6_no_number_in_the_run_files_equals_a_reader_setting(path):
    assert hard_coded_rule_values(path, tree_of(path), reader_settings_values()) == []


@pytest.mark.parametrize("source", ["pool = ThreadPoolExecutor(max_workers=3)\n", "runs = 3\n", "timeout = 45\n"])
def test_f6_a_planted_reader_setting_is_caught(tmp_path, source):
    path = plant(tmp_path, source)
    assert hard_coded_rule_values(path, tree_of(path), reader_settings_values())


# --- F7: nothing outside this ticket ---------------------------------------------------------------------------------------------


def identifiers(path: Path) -> list[str]:
    """Every name the file defines or uses: variables, attributes, functions, arguments, imports."""
    found = []
    for node in ast.walk(tree_of(path)):
        if isinstance(node, ast.Name):
            found.append(node.id)
        elif isinstance(node, ast.Attribute):
            found.append(node.attr)
        elif isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            found.append(node.name)
        elif isinstance(node, ast.arg):
            found.append(node.arg)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            found += [alias.name for alias in node.names] + ([node.module] if isinstance(node, ast.ImportFrom) and node.module else [])
    return [name for name in found if BANNED_IDENTIFIERS.search(name)]


@pytest.mark.parametrize("path", EVAL_FILES, ids=lambda p: p.name)
def test_f7_no_few_shot_judge_guard_or_phone_rehydration_in_the_eval(path):
    assert identifiers(path) == []


@pytest.mark.parametrize(
    "source",
    ["few_shot_examples = []\n", "def judge(report):\n    pass\n", "from flysafelz.redact import rehydrate\n", "class Guard:\n    pass\n", "x = obj.unredact(text)\n", "def f(fewshot):\n    pass\n"],
)
def test_f7_a_planted_identifier_is_caught(tmp_path, source):
    assert identifiers(plant(tmp_path, source))


def test_f7_the_prompt_is_not_touched_by_the_eval():
    """The eval reads PROMPT_VERSION and nothing else of the prompt: it neither builds a prompt nor changes one."""
    uses = [name for path in EVAL_FILES for name in imports_of(path) if name == "flysafelz.reader.prompt"]
    assert uses == ["flysafelz.reader.prompt"]  # run.py, for PROMPT_VERSION
    assert "from flysafelz.reader.prompt import PROMPT_VERSION\n" in RUNNER.read_text(encoding="utf-8")


def test_f2_runs_are_gitignored():
    import subprocess

    done = subprocess.run(["git", "check-ignore", "-q", "evals/reader/runs/20261107T160000Z/report.md"], cwd=ROOT, check=False)
    assert done.returncode == 0
