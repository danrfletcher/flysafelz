"""Fence scans (F2, F3, F4, F6, F7, F8, F10). Plain AST and text checks; later PRs inherit them."""

import ast
import enum
import re
import subprocess
import tomllib
from pathlib import Path

import pytest
import yaml
from pydantic import BaseModel, ValidationError

from fence_scan import (
    READER_REVIEWED_LITERALS,
    REVIEWED_FLOATS,
    REVIEWED_NAMED_CONSTANTS,
    hard_coded_rule_values,
    number,
    unreviewed_floats,
    unreviewed_literals,
)

from flysafelz import cli, contracts
from flysafelz.rules import DEFAULT_PATH, RULES

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src" / "flysafelz"
PY_FILES = sorted(SRC.rglob("*.py"))
SCRIPT_FILES = sorted((ROOT / "scripts").glob("*.py"))
FORBIDDEN_IMPORTS = ("langgraph", "langchain", "langsmith", "openai", "anthropic")


def _parse(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _number(node: ast.AST) -> float | None:
    """The value of a numeric literal (including a negated one), else None."""
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        inner = _number(node.operand)
        return None if inner is None else (-inner if isinstance(node.op, ast.USub) else inner)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
        return node.value
    return None


def _rule_values() -> set[float]:
    data = yaml.safe_load(DEFAULT_PATH.read_text(encoding="utf-8"))
    return {e["value"] for e in data.values() if isinstance(e["value"], (int, float))}


def test_source_files_are_found():
    assert SRC / "contracts.py" in PY_FILES and SRC / "rules.py" in PY_FILES


# F2: no agent, tool, LangGraph, LangChain, RAG or UI code. Narrowed for the reader (ticket 3 F2): the pyproject may declare
# langchain-openai (and its langchain-core), and only src/flysafelz/reader/ may import them. Everything else is still banned.
READER = SRC / "reader"
ALLOWED_LLM_DEPENDENCIES = {"langchain-openai", "langchain-core"}
ALLOWED_LLM_IMPORTS = ("langchain_openai", "langchain_core")
BANNED_DEPENDENCY_PREFIXES = ("langgraph", "langchain", "openai", "anthropic")


def _banned_dependencies(declared: list[str]) -> list[str]:
    """The declared dependencies that are an agent or LLM client and not one of the two the reader may use."""
    names = [re.split(r"[<>=!~\[ ;]", d, maxsplit=1)[0].lower().replace("_", "-") for d in declared]
    return [name for name in names if name.startswith(BANNED_DEPENDENCY_PREFIXES) and name not in ALLOWED_LLM_DEPENDENCIES]


def _declared_dependencies() -> list[str]:
    config = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    declared = list(config["project"]["dependencies"])
    for group in config.get("dependency-groups", {}).values():
        declared += [d for d in group if isinstance(d, str)]
    return declared


def test_pyproject_has_no_agent_or_llm_dependency():
    declared = _declared_dependencies()
    assert declared, "no dependencies read"
    assert _banned_dependencies(declared) == []


def test_the_dependency_scan_allows_exactly_the_two_reader_packages_and_still_catches_the_rest():
    assert _banned_dependencies(["langchain-openai>=1.7.4", "langchain-core", "langchain_core>=1"]) == []
    planted = ["langgraph", "langchain", "langchain-community>=0.3", "langchain-anthropic", "openai>=1", "anthropic", "openai-agents", "langchain-openai-extra"]
    assert _banned_dependencies(planted) == [d.split(">=")[0] for d in planted]
    assert "langchain-openai" in {re.split(r"[<>=!~\[ ;]", d, maxsplit=1)[0].lower() for d in _declared_dependencies()}  # the reader's one new dependency is declared


def _imported_modules(path: Path) -> list[str]:
    modules = []
    for node in ast.walk(_parse(path)):
        if isinstance(node, ast.Import):
            modules += [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.append(node.module)
    return modules


def _is_allowed_import(module: str, path: Path, reader: Path) -> bool:
    """langchain_openai and langchain_core (and their submodules) are allowed in the reader package only."""
    in_reader = reader in path.parents
    return in_reader and module.split(".")[0] in ALLOWED_LLM_IMPORTS


def _agent_imports(path: Path, reader: Path = READER) -> list[str]:
    """The modules the file imports that are an agent or LLM client, less what the reader package may import."""
    return [m for m in _imported_modules(path) if m.startswith(FORBIDDEN_IMPORTS) and not _is_allowed_import(m, path, reader)]


def test_source_imports_no_agent_framework():
    for path in PY_FILES + SCRIPT_FILES:
        assert _agent_imports(path) == [], path


def test_only_the_reader_package_imports_an_llm_client():
    """The reader really uses the exemption (so it is not dead), and nothing outside it does."""
    reader_files = sorted(READER.rglob("*.py"))
    assert reader_files
    assert {m.split(".")[0] for path in reader_files for m in _imported_modules(path)} >= {"langchain_core"}
    for path in PY_FILES + SCRIPT_FILES:
        if READER not in path.parents:
            assert not [m for m in _imported_modules(path) if m.split(".")[0] in ALLOWED_LLM_IMPORTS], path


def test_the_import_scan_still_catches_a_planted_llm_import_outside_the_reader(tmp_path):
    """Planted violations: langchain_core appended to a temp copy of cli.py, and the same line in the reader's place."""
    cli_copy = tmp_path / "cli.py"
    cli_copy.write_text((SRC / "cli.py").read_text(encoding="utf-8") + "\nimport langchain_core\n", encoding="utf-8")
    assert _agent_imports(cli_copy, reader=tmp_path / "reader") == ["langchain_core"]
    assert _agent_imports(SRC / "cli.py") == []  # the real file is clean, so the hit above is the planted line
    for module in ("langchain_openai", "langchain_core.messages", "langgraph", "openai", "anthropic", "langchain_community", "langsmith"):
        outside = tmp_path / "other.py"
        outside.write_text(f"import {module}\n", encoding="utf-8")
        assert _agent_imports(outside, reader=tmp_path / "reader") == [module], module
    (tmp_path / "reader").mkdir()
    inside = tmp_path / "reader" / "read.py"
    inside.write_text("import langchain_core.messages\nfrom langchain_openai import ChatOpenAI\nimport langgraph\nimport openai\nimport langchain_community\n", encoding="utf-8")
    assert _agent_imports(inside, reader=tmp_path / "reader") == ["langgraph", "openai", "langchain_community"]  # the two allowed packages pass; the rest do not


# F1 (PR-3.F1): the verdict sheet has no model, no reader, and writes no label on a blind row.
# VERDICT_SHEET_FILES is the blind path (the builder and its script); the draft module is the only code that writes
# a verdict or an action, and the builder reaches it only for an example that is not blind (PR-4.F1).
VERDICT_SHEET_FILES = [SRC / "verdict_sheet.py", SRC / "verdict_sheet_fingerprint.py", ROOT / "scripts" / "build_verdict_sheet.py"]
DRAFT_FILES = [SRC / "verdict_sheet_draft.py", SRC / "verdict_sheet_check.py"]  # the check module names the enums to validate and load labels (PR-5.F1); it writes none


def test_the_verdict_sheet_imports_no_model_client_and_no_reader():
    assert all(p.exists() for p in VERDICT_SHEET_FILES + DRAFT_FILES)
    for path in VERDICT_SHEET_FILES + DRAFT_FILES:
        for module in _imported_modules(path):
            assert not module.startswith(FORBIDDEN_IMPORTS), (path, module)
            assert "reader" not in module.lower(), (path, module)


def test_the_verdict_sheet_writes_no_label_value():
    labels = {m.value for m in contracts.Verdict} | {m.name for m in contracts.Verdict} | {m.value for m in contracts.NotamAction}
    for path in VERDICT_SHEET_FILES:
        tree = _parse(path)
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                assert node.value not in labels, f"{path}:{node.lineno}: {node.value!r}"
            if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
                assert node.value.id not in {"Verdict", "NotamAction"}, f"{path}:{node.lineno}: {node.value.id}.{node.attr}"
        for node in ast.walk(tree):
            if isinstance(node, ast.Dict):
                for key, value in zip(node.keys, node.values, strict=True):
                    if isinstance(key, ast.Constant) and key.value in {"verdict", "approaching_limits"}:
                        assert isinstance(value, ast.Constant) and value.value is None, f"{path}:{node.lineno}: {key.value} is not None"


CUT_POINT_NAMES = {"active_intervals", "daylight_edges"}


def _cut_point_uses(tree: ast.AST) -> list[tuple[int, str]]:
    """Calls to, and imports of, the cut-point primitives that the sheet must leave to assess."""
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            name = func.id if isinstance(func, ast.Name) else func.attr if isinstance(func, ast.Attribute) else None
            if name in CUT_POINT_NAMES:
                found.append((node.lineno, f"call {name}"))
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                if alias.name.rsplit(".", 1)[-1] in CUT_POINT_NAMES:
                    found.append((node.lineno, f"import {alias.name}"))
    return found


def test_the_verdict_sheet_defines_no_cut_point_logic_of_its_own():
    for path in VERDICT_SHEET_FILES:
        assert _cut_point_uses(_parse(path)) == [], path


def test_the_cut_point_scan_catches_planted_uses():
    planted = ast.parse(
        "from flysafelz.assess import active_intervals\n"
        "import flysafelz.daylight as d\n"
        "from flysafelz.daylight import daylight_edges as edges\n"
        "a = active_intervals(x)\n"
        "b = d.daylight_edges(y)\n"
    )
    uses = [kind for _, kind in _cut_point_uses(planted)]
    assert sorted(uses) == [
        "call active_intervals",
        "call daylight_edges",
        "import active_intervals",
        "import daylight_edges",
    ]


def test_the_label_scan_catches_a_planted_label():
    planted = ast.parse('row = {"verdict": "NO_GO"}\nother = Verdict.NO_GO\n')
    constants = [n.value for n in ast.walk(planted) if isinstance(n, ast.Constant)]
    assert "NO_GO" in constants
    assert any(isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name) and n.value.id == "Verdict" for n in ast.walk(planted))


# F3: no threshold literals in Python.
def test_no_numeric_literal_in_a_comparison_other_than_0_or_1():
    allowed = {0, 1}
    for path in PY_FILES:
        for node in ast.walk(_parse(path)):
            if isinstance(node, ast.Compare):
                for operand in [node.left, *node.comparators]:
                    value = _number(operand)
                    assert value is None or value in allowed, f"{path}:{node.lineno} compares against {value}"


def test_no_numeric_literal_equals_a_rules_yaml_value():
    """Structural 0 and 1 are exempt (reader_area_slack_nm is 1.0) and so are the reviewed named constants in geo.py (F6)."""
    values = _rule_values()
    assert values, "no rule values read"
    for path in PY_FILES:
        for lineno, value in hard_coded_rule_values(path, _parse(path), values):
            pytest.fail(f"{path}:{lineno} hard-codes {value}, which is a rules.yaml value")


def test_the_reader_area_slack_is_a_rule_and_the_area_module_holds_no_literal_for_it():
    assert 1.0 in _rule_values()
    text = (SRC / "notam_area.py").read_text(encoding="utf-8")
    assert "RULES.reader_area_slack_nm" in text
    floats = [n.value for n in ast.walk(ast.parse(text)) if isinstance(n, ast.Constant) and isinstance(n.value, float)]
    assert floats == [], "notam_area.py holds a float literal"


def _planted_hits(source: str, filename: str) -> list[float]:
    return [value for _, value in hard_coded_rule_values(Path(filename), ast.parse(source), _rule_values())]


def test_the_narrowed_scan_still_flags_a_planted_limit_outside_geo_py():
    """F6 planted check: 1000 is cape_no_go. The geo.py exemption is for METRES_PER_KM = 1000 in geo.py only."""
    assert _planted_hits("limit = 1000\n", "notam_area.py") == [1000]
    assert _planted_hits("METRES_PER_KM = 1000\n", "notam_area.py") == [1000]  # same name, wrong file
    assert _planted_hits("limit = 1000\n", "geo.py") == [1000]  # right file, not a reviewed name
    assert _planted_hits("METRES_PER_KM = 1000\n", "geo.py") == []
    assert _planted_hits("MIN_POLYGON_POINTS = 3\nMETRES_PER_KM = 1000\n", "geo.py") == []
    assert _planted_hits("MIN_POLYGON_POINTS = 4\n", "geo.py") == [4]  # a reviewed name holding another rules value
    assert _planted_hits("retries = 3\n", "pib.py") == [3]
    assert {"METRES_PER_KM": 1000, "MIN_POLYGON_POINTS": 3} == REVIEWED_NAMED_CONSTANTS


def test_the_narrowed_scan_exempts_structural_zero_and_one_only():
    """Slack is 1.0, so 1 would flag everywhere without the exemption; 1.5 is not a rule value, and 2 still is (local_area)."""
    assert _planted_hits("e = 1 - WGS84_E2\nzero = 0\nx = 1.0\n", "geo.py") == []
    assert _planted_hits("slack = 2.0\nlimit = -2\n", "notam_area.py") == [2.0, 2]
    assert 1.5 not in _rule_values()  # so the equals scan above cannot see a planted slack of 1.5; the reviewed-literal scan below does


def test_a_planted_slack_of_1_5_is_flagged_in_every_reader_module():
    """F6 planted check: the reviewed-literal scan flags `slack = 1.5` in notam_area.py, redact.py and the script (not only geo.py)."""
    planted = ast.parse("slack = 1.5\n")
    for name in READER_REVIEWED_LITERALS:
        assert [v for _, v in unreviewed_literals(Path(name), planted)] == [1.5], name
    assert [v for _, v in unreviewed_literals(Path("notam_area.py"), ast.parse("slack = 40\nunit = 60\n"))] == [40]  # 40 is reviewed for the script only


def _relative(path: Path) -> str:
    return path.relative_to(SRC).as_posix()


def test_a_planted_slack_of_1_5_is_flagged_in_every_module_under_src():
    """F6 planted check, whole tree: the float scan flags `slack = 1.5` in each real module, not only the reader's."""
    planted = ast.parse("slack = 1.5\n")
    assert PY_FILES
    for path in PY_FILES:
        assert [v for _, v in unreviewed_floats(_relative(path), planted)] == [1.5], path
    # Appended to the real source of a module that holds no float of its own, it is the only hit.
    pib = SRC / "pib.py"
    appended = ast.parse(pib.read_text(encoding="utf-8") + "\nslack = 1.5\n")
    assert [v for _, v in unreviewed_floats("pib.py", appended)] == [1.5]
    # Appended to a module with reviewed floats, the reviewed ones stay quiet and 1.5 is still caught.
    geo = ast.parse((SRC / "geo.py").read_text(encoding="utf-8") + "\nslack = 1.5\n")
    assert [v for _, v in unreviewed_floats("geo.py", geo)] == [1.5]
    # A reviewed value is reviewed for its own module only.
    assert [v for _, v in unreviewed_floats("pib.py", ast.parse("x = 3.280839895\n"))] == [3.280839895]


@pytest.mark.parametrize("path", PY_FILES, ids=_relative)
def test_every_float_literal_under_src_is_zero_one_or_reviewed(path):
    assert unreviewed_floats(_relative(path), _parse(path)) == []


def test_every_reviewed_float_is_still_used():
    """A reviewed entry that no longer matches a literal is stale; drop it."""
    for relative, reviewed in REVIEWED_FLOATS.items():
        path = SRC / relative
        assert path.is_file(), relative
        used = {abs(v) for v in (number(n) for n in ast.walk(_parse(path))) if isinstance(v, float)}
        assert set(reviewed) <= used, f"{relative}: stale reviewed floats {set(reviewed) - used}"
        assert all(reviewed.values()), f"{relative}: a reviewed float has no reason"


@pytest.mark.parametrize("name", sorted(READER_REVIEWED_LITERALS))
def test_every_numeric_literal_in_a_reader_module_is_zero_one_or_reviewed(name):
    path = ROOT / "scripts" / name if name.startswith("build_") else SRC / name
    assert path.is_file(), path
    assert unreviewed_literals(path, _parse(path)) == []
    assert hard_coded_rule_values(path, _parse(path), _rule_values()) == []  # the script is not under src/, so the equals scan has not seen it


def test_f3_scan_catches_a_planted_literal():
    planted = ast.parse("if wind >= 16:\n    pass\nlimit = 12\n")
    compared = [_number(o) for n in ast.walk(planted) if isinstance(n, ast.Compare) for o in n.comparators]
    assert compared == [16]
    assert 12 in _rule_values()


# F4: no score, no bare GO verdict.
def test_verdict_enum_has_exactly_three_members():
    assert {v.name for v in contracts.Verdict} == {"NO_GO", "UNVERIFIED", "PILOT_DISCRETION"}


def test_no_string_literal_is_a_bare_go():
    bare_go = re.compile(r"(?<!NO-)(?<!NO_)\bGO\b")
    for path in PY_FILES:
        for node in ast.walk(_parse(path)):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                assert not bare_go.search(node.value), f"{path}:{node.lineno}: {node.value!r}"


def test_no_score_or_weight_identifiers():
    for path in PY_FILES:
        for node in ast.walk(_parse(path)):
            names = []
            if isinstance(node, ast.Name):
                names = [node.id]
            elif isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                names = [node.name]
            elif isinstance(node, ast.arg):
                names = [node.arg]
            elif isinstance(node, ast.Attribute):
                names = [node.attr]
            for name in names:
                assert not re.search(r"score|weight", name, re.IGNORECASE), f"{path}:{node.lineno}: {name}"


def test_readme_uses_the_three_verdicts_not_the_old_wording():
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "GO WITH CONDITIONS" not in text
    assert "MARGINAL" not in text
    for wording in ("NO-GO", "UNVERIFIED", "PILOT DISCRETION", "APPROACHING LIMITS", "WITH CONDITIONS"):
        assert wording in text


# F6: no secrets in tracked files.
def _tracked_files() -> list[str]:
    out = subprocess.run(["git", "ls-files"], cwd=ROOT, check=True, capture_output=True, text=True).stdout
    return out.splitlines()


API_KEY_WITH_VALUE = re.compile(r"\b([A-Z0-9_]*API_KEY)[ \t]*=[ \t]*[^\s#]")  # OPENAIP_API_KEY, READER_API_KEY, any *_API_KEY


def _files_with_a_key_value(root: Path, names: list[str]) -> list[str]:
    """'<file>: <NAME>' for every file that holds a *_API_KEY set to something."""
    found = []
    for name in names:
        path = root / name
        if not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        match = API_KEY_WITH_VALUE.search(text)
        if match:
            found.append(f"{name}: {match.group(1)}")
    return found


def test_no_key_value_in_tracked_files():
    tracked = _tracked_files()
    assert ".env.example" in tracked
    assert ".env" not in tracked
    assert _files_with_a_key_value(ROOT, tracked) == []


def test_the_key_scan_catches_a_planted_reader_key_and_ignores_an_empty_one(tmp_path):
    # Split so this file never matches its own scan.
    (tmp_path / "planted.env").write_text("READER_API_" + "KEY=sk-not-a-real-key\n", encoding="utf-8")
    (tmp_path / "other.env").write_text("OPENAIP_API_" + "KEY=abc123\n", encoding="utf-8")
    (tmp_path / "empty.env").write_text("READER_API_" + "KEY=\nREADER_API_" + "KEY= # unset\n", encoding="utf-8")
    names = ["planted.env", "other.env", "empty.env", "missing.env"]
    assert _files_with_a_key_value(tmp_path, names) == ["planted.env: READER_API_KEY", "other.env: OPENAIP_API_KEY"]


def test_env_is_gitignored_and_example_is_empty():
    ignored = subprocess.run(["git", "check-ignore", ".env"], cwd=ROOT, capture_output=True, text=True)
    assert ignored.returncode == 0
    assert (ROOT / ".env.example").read_text(encoding="utf-8").strip() == "OPENAIP_API_" + "KEY="  # split so this file never matches the F6 scan


# F7: plain modules and functions.
def test_contract_classes_subclass_only_basemodel_or_enum():
    classes = [o for o in vars(contracts).values() if isinstance(o, type) and o.__module__ == contracts.__name__]
    assert classes
    for cls in classes:
        assert cls.__bases__ in {(BaseModel,), (enum.Enum,), (enum.StrEnum,)}, cls


def test_no_registry_di_or_caching_in_source():
    pattern = re.compile(r"lru_cache|functools\.cache|\bcache\b|registry|Registry|\binject\b|Provider\b")
    for path in PY_FILES:
        assert not pattern.search(path.read_text(encoding="utf-8")), path


# F8: limits and the local area cannot be adjusted by the user.
def test_rules_cannot_be_assigned_to():
    with pytest.raises(ValidationError):
        RULES.wind_10m_no_go = RULES.wind_10m_caution
    with pytest.raises(ValidationError):
        RULES.wind_10m_no_go.value = 99
    with pytest.raises(ValidationError):
        RULES.local_area.value = 99


@pytest.mark.parametrize("flag", ["--rules", "--rules-path", "--path", "--local-area", "--limit"])
def test_cli_has_no_flag_for_rules_or_limits(flag):
    with pytest.raises(SystemExit) as exit_info:
        cli.main(["rules", flag, "x"])
    assert exit_info.value.code == 2


def test_rules_module_and_cli_read_no_environment():
    for name in ("rules.py", "cli.py"):
        text = (SRC / name).read_text(encoding="utf-8")
        assert not re.search(r"environ|getenv", text), name


# F10: no LLM env vars, no /models dependency.
def test_no_llm_env_vars_or_models_endpoint_in_source():
    pattern = re.compile(r"OPENAI(?!P_API_KEY)|ANTHROPIC|LLM_|MODEL_|/models")
    for path in PY_FILES:
        match = pattern.search(path.read_text(encoding="utf-8"))
        assert match is None, f"{path}: {match.group(0) if match else ''}"


# F4 / F7 / F1 (PR-4.F1): `# why` and the ignored block are comments only; no answer text; the draft is model-free.
def _string_keys(tree: ast.AST) -> list[tuple[int, str]]:
    """Every string used as a mapping key: dict literal keys and subscripts (a call's keyword argument is not a sheet key)."""
    keys: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Dict):
            keys += [(k.lineno, k.value) for k in node.keys if isinstance(k, ast.Constant) and isinstance(k.value, str)]
        elif isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Constant) and isinstance(node.slice.value, str):
            keys.append((node.lineno, node.slice.value))
    return keys


def test_the_builder_writes_no_why_reason_or_ignored_key():
    for path in VERDICT_SHEET_FILES + DRAFT_FILES:
        for line, key in _string_keys(_parse(path)):
            assert not any(word in key.lower() for word in ("why", "reason", "ignored")), f"{path}:{line}: key {key!r}"
        for node in ast.walk(_parse(path)):  # a string that starts a `why:` line (the comment is written as '# why: ...')
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                assert not re.match(r"\s*why\s*:", node.value), f"{path}:{node.lineno}: writes a why: key"


def test_the_builder_imports_no_answer_formatting_from_the_cli():
    for path in VERDICT_SHEET_FILES + DRAFT_FILES:
        for node in ast.walk(_parse(path)):
            if isinstance(node, ast.ImportFrom) and node.module and node.module.endswith("cli"):
                names = {a.name for a in node.names}
                assert not names & {"overall_line", "footer_lines"}, f"{path}:{node.lineno}: {names}"
            if isinstance(node, ast.Attribute) and node.attr in {"overall_line", "footer_lines"}:
                raise AssertionError(f"{path}:{node.lineno}: {node.attr}")


def test_only_the_draft_module_may_name_a_verdict_or_an_action():
    names = {"Verdict", "NotamAction"}
    for path in VERDICT_SHEET_FILES:
        for node in ast.walk(_parse(path)):
            if isinstance(node, ast.ImportFrom):
                assert not names & {a.name for a in node.names}, f"{path}:{node.lineno}"
    assert any(isinstance(n, ast.ImportFrom) and n.module == "flysafelz.verdict_sheet_draft" for n in ast.walk(_parse(SRC / "verdict_sheet.py")))
