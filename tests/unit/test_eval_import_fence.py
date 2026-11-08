"""F4, F6 and F8 for the eval harness's import code (5a PR-1.S1). The LZ coordinates are test_lz_allow_list.py's.

F4  nothing captures a labelled case: no module under evals/agent/ imports flysafelz.capture or flysafelz.clients (AST scan), the `dataset` target never
    calls `make fixtures`, and every committed labelled case has the bytes of the labelled source (the checksum list says so).
F6  every phone number written into a test or an added module is in a fiction range (01632 96xxxx, 07700 900xxx), and the test sums files use invented case names.
F8  the import code writes only where it was told to: the write calls sit in the named functions and nowhere else, no module names the vault, and the
    read-only checksum module has no write call at all. (test_import_dataset.py and test_import_fixtures.py make the sources read-only and compare them.)

Every scan has a planted violation proving it fires. The bad phone numbers are built at run time, so this file holds none.
"""

import ast
import re
from pathlib import Path

import pytest

from evals.agent.checksums import check_labelled, read_exceptions, read_sums

REPO = Path(__file__).resolve().parents[2]
AGENT = REPO / "evals" / "agent"
FIXTURES = REPO / "fixtures"
MODULES = sorted(AGENT.rglob("*.py"))
ADDED_TESTS = [
    REPO / "tests" / "unit" / "test_eval_checksums.py",
    REPO / "tests" / "unit" / "test_import_dataset.py",
    REPO / "tests" / "unit" / "test_import_fixtures.py",
    REPO / "tests" / "unit" / "test_eval_import_fence.py",
    REPO / "tests" / "unit" / "test_lz_allow_list.py",
    REPO / "tests" / "test_labelled_fixtures_committed.py",
    # the runner (5a PR-4.F1): its tests, their fakes and the fence test hold fiction-range phones and invented case names only
    REPO / "tests" / "unit" / "eval_runner_helpers.py",
    REPO / "tests" / "unit" / "test_eval_run_v1_cli.py",
    REPO / "tests" / "unit" / "test_eval_runner_context.py",
    REPO / "tests" / "unit" / "test_eval_runner_errors.py",
    REPO / "tests" / "unit" / "test_eval_runner_fence.py",
    REPO / "tests" / "unit" / "test_eval_runner_order.py",
    REPO / "tests" / "unit" / "test_eval_runner_usage.py",
    REPO / "tests" / "integration" / "test_eval_runner_dev.py",
    # the judge (5a PR-7.F1): its fakes and tests hold fiction-range phones only. Two of them read the dataset's case names on purpose (see NAMES_REAL_CASES)
    REPO / "tests" / "unit" / "judge_helpers.py",
    REPO / "tests" / "unit" / "test_judge_config.py",
    REPO / "tests" / "unit" / "test_judge_criteria.py",
    REPO / "tests" / "unit" / "test_judge_examples.py",
    REPO / "tests" / "unit" / "test_judge_message.py",
    REPO / "tests" / "integration" / "test_judge_evaluator.py",
]
# these two read the committed labelled cases by name on purpose; the rest of the added tests use invented names
NAMES_REAL_CASES = {"test_labelled_fixtures_committed.py", "test_lz_allow_list.py", "test_judge_message.py", "test_judge_examples.py"}  # the last two use real case names to test against them


# --- F4: nothing captures ---------------------------------------------------------------------------------------------------------

FORBIDDEN_IMPORTS = ("flysafelz.capture", "flysafelz.clients")


def capture_imports(source: str) -> list[str]:
    """Every import of flysafelz.capture or flysafelz.clients, in any form, and any string that names one for a dynamic import."""
    found = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            base = node.module or ""
            names = [base, *(f"{base}.{alias.name}" for alias in node.names)]
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            names = [node.value]
        else:
            continue
        found.extend(name for name in names if any(name == bad or name.startswith(f"{bad}.") for bad in FORBIDDEN_IMPORTS))
    return found


def network_imports(source: str) -> list[str]:
    found = []
    for node in ast.walk(ast.parse(source)):
        names = [alias.name for alias in node.names] if isinstance(node, ast.Import) else [node.module or ""] if isinstance(node, ast.ImportFrom) else []
        found.extend(name for name in names if name.split(".")[0] in {"httpx", "requests", "urllib", "socket", "aiohttp", "http"})
    return found


def test_there_are_added_modules_to_scan():
    assert {p.name for p in MODULES} >= {"__init__.py", "checksums.py", "import_dataset.py", "import_fixtures.py"}


@pytest.mark.parametrize("path", MODULES, ids=lambda p: p.name)
def test_no_added_module_imports_the_capture_code_or_the_source_clients(path):
    assert capture_imports(path.read_text(encoding="utf-8")) == []
    assert network_imports(path.read_text(encoding="utf-8")) == []


@pytest.mark.parametrize(
    "source",
    [
        "import flysafelz.capture",
        "from flysafelz.capture import main",
        "from flysafelz import capture",
        "from flysafelz.clients import openaip",
        "from flysafelz.clients.pib_source import fetch",
        "import flysafelz.clients.open_meteo as om",
        "import importlib\nimportlib.import_module('flysafelz.capture')",
    ],
)
def test_the_import_scan_catches_every_form(source):
    assert capture_imports(source)


def test_the_import_scan_leaves_ordinary_imports_alone():
    assert capture_imports("from flysafelz.verdict_sheet_fingerprint import unseal\nfrom flysafelz.pib import parse_pib\nimport flysafelz.captured_notes") == []
    assert network_imports("import httpx") == ["httpx"]


def dataset_recipe() -> str:
    makefile = (REPO / "Makefile").read_text(encoding="utf-8")
    recipe = re.search(r"^dataset:[^\n]*\n((?:\t[^\n]*\n?)+)", makefile, re.MULTILINE)
    assert recipe, "the Makefile has no dataset target"
    return recipe[1]


def test_the_dataset_target_runs_the_import_and_never_captures():
    recipe = dataset_recipe()
    assert "evals.agent.import_dataset" in recipe
    assert not re.search(r"fixtures|capture|\$\(MAKE\)|\bmake\b|openaip|curl|wget", recipe), recipe


def test_the_dataset_target_passes_the_source_only_through_the_environment():
    makefile = (REPO / "Makefile").read_text(encoding="utf-8")
    assert re.search(r"^export DATASET_SRC \?= \$\(HOME\)/\.local/share/flysafelz/dataset$", makefile, re.MULTILINE)
    assert "$(DATASET_SRC)" not in dataset_recipe() and "$$DATASET_SRC" not in dataset_recipe()  # not in the command line: the shell never parses it


def test_make_test_measures_the_eval_harness():
    makefile = (REPO / "Makefile").read_text(encoding="utf-8")
    assert "--cov=evals.agent " in makefile and "--cov-fail-under=90" in makefile


def test_every_committed_labelled_case_has_the_bytes_of_the_labelled_source():
    """The list is a verbatim copy of the source's; a capture run would have produced different bytes (openAIP and the archive are live)."""
    sums = read_sums(FIXTURES / "labelled-fixtures.sha256")
    exceptions = read_exceptions(FIXTURES / "labelled_exceptions.yaml")
    assert check_labelled(FIXTURES, sums, exceptions) == []
    assert len(exceptions) == 1  # the one named, reasoned exception; tests/test_labelled_fixtures_committed.py says which


# --- F6: fiction-range phone numbers only -----------------------------------------------------------------------------------------

PHONE = re.compile(r"(?<![\w.\-])(?:\+44[\s-]?\(?0?\)?[\s-]?|0)\d(?:[\s-]?\d){8,10}(?![\w.\-]|\s?\d)")
FICTION_PREFIXES = ("0163296", "07700900")


def digits_of(match: str) -> str:
    digits = re.sub(r"\D", "", match)
    if not match.lstrip().startswith("+44"):
        return digits
    national = digits[2:]  # `+44 (0) 1632 ...` keeps its 0, `+44 7700 ...` has none
    return national if national.startswith("0") else "0" + national


def real_looking_phones(source: str) -> list[str]:
    """The phone-shaped numbers in `source` that are outside 01632 96xxxx and 07700 900xxx. The count only, for the failure message."""
    return [m.group(0) for m in PHONE.finditer(source) if not digits_of(m.group(0)).startswith(FICTION_PREFIXES)]


@pytest.mark.parametrize("number", ["01632 960123", "07700 900123", "01632960123", "+44 7700 900123", "+44 (0) 1632 960123", "call 07700900456 now"])
def test_fiction_range_numbers_pass(number):
    assert real_looking_phones(f"phone: {number}") == []


@pytest.mark.parametrize("prefix, rest", [("0778", "4 546062"), ("0191", " 234 5678"), ("0207", "946 0000"), ("+44 7", "784 546062"), ("0163", "3 960123"), ("0770", "0 901123")])
def test_a_number_outside_the_fiction_ranges_is_caught(prefix, rest):
    assert real_looking_phones(f"phone: {prefix}{rest}")


def test_coordinates_dates_and_digests_are_not_phone_numbers():
    assert real_looking_phones("lat: 53.802797\nlon: -0.086644\nnow: '2026-10-31T07:30:00Z'\nsha: 037c6199a207a5ec14c0daad33ce3bba4281bbbc6f90d8bc96c5321254f2b167\nwindow: 0800-1800") == []


@pytest.mark.parametrize("path", [*ADDED_TESTS, *MODULES], ids=lambda p: p.name)
def test_every_phone_number_in_an_added_test_or_module_is_in_a_fiction_range(path):
    assert real_looking_phones(path.read_text(encoding="utf-8")) == [], f"{path.name} holds a phone number outside the fiction ranges"


def real_case_names() -> set[str]:
    return {p.name for p in FIXTURES.iterdir() if p.is_dir()}


@pytest.mark.parametrize("path", [p for p in ADDED_TESTS if p.name not in NAMES_REAL_CASES], ids=lambda p: p.name)
def test_the_added_tests_use_invented_case_names(path):
    text = path.read_text(encoding="utf-8")
    assert [name for name in sorted(real_case_names()) if name in text] == []


def test_the_scan_of_real_case_names_is_not_empty():
    assert len(real_case_names()) >= 26


# --- F8: the import code writes only where it is told to --------------------------------------------------------------------------

WRITE_ATTRIBUTES = {
    "write_text", "write_bytes", "mkdir", "unlink", "rename", "replace", "touch", "rmdir", "chmod", "symlink_to", "hardlink_to",
    "copyfile", "copy", "copy2", "copytree", "move", "rmtree", "remove", "removedirs", "mkstemp", "mkdtemp", "NamedTemporaryFile", "TemporaryDirectory", "makedirs", "fdopen", "chown", "utime",
}  # fmt: skip
# the functions that may write; every other function in the module, and every module not named here, may not
# agreement.py (5a PR-8.F1) writes the judge cache (write_cache, atomically) and the report (main), into the two folders it is given; annotate.py writes no file
MAY_WRITE = {"import_dataset.py": {"write_all"}, "import_fixtures.py": {"copy_folder", "import_fixtures"}, "agreement.py": {"write_cache", "main"}}


def write_calls(source: str) -> list[tuple[str, str]]:
    """(function name, call) for every call that can write: a write-named method, or open() with a mode that is not plain read."""
    found = []

    def visit(node: ast.AST, function: str) -> None:
        for child in ast.iter_child_nodes(node):
            inner = child.name if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)) else function
            if isinstance(child, ast.Call):
                name = child.func.attr if isinstance(child.func, ast.Attribute) else child.func.id if isinstance(child.func, ast.Name) else ""
                modes = [a.value for a in [*child.args[1:2], *(k.value for k in child.keywords if k.arg == "mode")] if isinstance(a, ast.Constant) and isinstance(a.value, str)]
                if name in WRITE_ATTRIBUTES or (name == "open" and any(set(mode) & set("wax+") for mode in modes)):
                    found.append((function, name))
            visit(child, inner)

    visit(ast.parse(source), "<module>")
    return found


@pytest.mark.parametrize("path", MODULES, ids=lambda p: p.name)
def test_write_calls_sit_only_in_the_functions_named_for_them(path):
    allowed = MAY_WRITE.get(path.name, set())
    stray = [(function, call) for function, call in write_calls(path.read_text(encoding="utf-8")) if function not in allowed]
    assert stray == [], f"{path.name} writes outside {sorted(allowed)}"


def test_the_checksum_module_has_no_write_call_at_all():
    assert write_calls((AGENT / "checksums.py").read_text(encoding="utf-8")) == []


def test_each_importer_has_its_write_calls_where_the_allow_list_says():
    for name, functions in MAY_WRITE.items():
        in_function = {function for function, _call in write_calls((AGENT / name).read_text(encoding="utf-8"))}
        assert in_function == functions, name


@pytest.mark.parametrize(
    "source",
    [
        "from pathlib import Path\ndef f():\n    Path('x').write_text('y')",
        "def f():\n    open('x', 'w')",
        "def f():\n    open('x', mode='ab')",
        "import shutil\ndef f():\n    shutil.copyfile('a', 'b')",
        "import os\ndef f():\n    os.replace('a', 'b')",
        "from pathlib import Path\ndef f(p):\n    p.mkdir()",
    ],
)
def test_the_write_scan_catches_a_planted_write(source):
    assert write_calls(source)


def test_the_write_scan_lets_a_read_through():
    assert write_calls("from pathlib import Path\ndef f(p):\n    open('x')\n    open('x', 'rb')\n    p.read_bytes()\n    p.is_file()") == []


VAULT = re.compile("|".join(["Documents" + "/Notes", "Documents", r"\bNotes\b", "obsidian", r"\bvault\b"]), re.IGNORECASE)


@pytest.mark.parametrize("path", MODULES, ids=lambda p: p.name)
def test_no_added_module_names_the_vault_or_a_notes_folder(path):
    strings = [n.value for n in ast.walk(ast.parse(path.read_text(encoding="utf-8"))) if isinstance(n, ast.Constant) and isinstance(n.value, str)]
    assert [s for s in strings if VAULT.search(s)] == []


def test_the_vault_scan_catches_a_planted_path():
    assert VAULT.search("~/Documents/" + "Notes/FDE Play")
