"""Fence scans (F2, F3, F4, F6, F7, F8, F10). Plain AST and text checks; later PRs inherit them."""

import ast
import enum
import re
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest
import yaml
from fence_scan import (
    AGENT_CONFIG,
    READER_REVIEWED_LITERALS,
    REVIEWED_FLOATS,
    REVIEWED_NAMED_CONSTANTS,
    banned_words,
    hard_coded_rule_values,
    import_violations,
    llm_env_words,
    number,
    tree_import_violations,
    unreviewed_floats,
    unreviewed_literals,
)
from pydantic import BaseModel, ValidationError

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


# F6: the dependency allowlist. The names declared in [project] dependencies and in each dependency group must equal these sets, so a new
# package (openai, anthropic, chromadb, ...) fails until it is added here on purpose. langgraph-cli sits in the dev group only.
ALLOWED_PROJECT_DEPENDENCIES = {
    "astral",
    "fastembed",
    "httpx",
    "langchain",
    "langchain-openai",
    "langgraph",
    "langsmith",
    "pydantic",
    "pypdf",
    "pyyaml",
    "shapely",
    "starlette",
    "tzdata",
}
ALLOWED_GROUP_DEPENDENCIES = {"dev": {"langgraph-cli", "pytest", "pytest-cov", "pytest-socket", "ruff"}}
READER = SRC / "reader"  # the one folder where READER_* names are allowed (ticket 3), and with MAPDATA one of two where the word `cache` is
MAPDATA = SRC / "mapdata"  # the map routes: the one package that may import starlette or fastapi, and the other where the word `cache` is allowed (ticket 8 F6)


def _dependency_name(requirement: str) -> str:
    """'langgraph-cli[inmem]>=0.4' -> 'langgraph-cli': lower case, `_` as `-`, without extras, version or marker."""
    return re.split(r"[<>=!~\[ ;]", requirement, maxsplit=1)[0].lower().replace("_", "-")


def _dependency_differences(config: dict) -> dict[str, dict[str, list[str]]]:
    """{where: {"unlisted": [...], "missing": [...]}} for every place that declares dependencies, less the places that match the allowlist exactly."""
    declared = {"project": {_dependency_name(d) for d in config["project"]["dependencies"]}}
    for group, requirements in config.get("dependency-groups", {}).items():
        declared[f"group {group}"] = {_dependency_name(d) for d in requirements if isinstance(d, str)}
    allowed = {"project": ALLOWED_PROJECT_DEPENDENCIES} | {f"group {g}": names for g, names in ALLOWED_GROUP_DEPENDENCIES.items()}
    differences = {}
    for where in sorted(declared.keys() | allowed.keys()):
        unlisted = sorted(declared.get(where, set()) - allowed.get(where, set()))
        missing = sorted(allowed.get(where, set()) - declared.get(where, set()))
        if unlisted or missing:
            differences[where] = {"unlisted": unlisted, "missing": missing}
    return differences


def _pyproject() -> dict:
    return tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))


def test_pyproject_declares_exactly_the_allowlisted_dependencies():
    assert _dependency_differences(_pyproject()) == {}


def test_no_other_table_can_declare_a_dependency_the_allowlist_does_not_see():
    config = _pyproject()
    assert "optional-dependencies" not in config["project"]
    assert "dev-dependencies" not in config.get("tool", {}).get("uv", {})
    assert set(config["dependency-groups"]) == set(ALLOWED_GROUP_DEPENDENCIES)


def test_langgraph_cli_is_in_the_dev_group_only():
    config = _pyproject()
    assert "langgraph-cli" not in {_dependency_name(d) for d in config["project"]["dependencies"]}
    assert "langgraph-cli" in {_dependency_name(d) for d in config["dependency-groups"]["dev"]}


@pytest.mark.parametrize("planted", ["openai", "openai>=1", "anthropic", "chromadb", "langchain-community", "langchain_anthropic>=1", "numpy"])
def test_an_unlisted_package_in_the_project_or_a_group_fails_the_allowlist(planted):
    for where, section in (("project", lambda c: c["project"]["dependencies"]), ("group dev", lambda c: c["dependency-groups"]["dev"])):
        config = _pyproject()
        section(config).append(planted)
        assert _dependency_differences(config) == {where: {"unlisted": [_dependency_name(planted)], "missing": []}}, where
    config = _pyproject()
    config["dependency-groups"]["extra"] = [planted]  # a new group is caught too
    assert _dependency_differences(config)["group extra"] == {"unlisted": [_dependency_name(planted)], "missing": []}


def test_a_dropped_or_moved_package_fails_the_allowlist_too():
    config = _pyproject()
    config["project"]["dependencies"] = [d for d in config["project"]["dependencies"] if _dependency_name(d) != "pypdf"]
    config["project"]["dependencies"].append("langgraph-cli[inmem]")  # the dev-only package moved into the project
    config["dependency-groups"]["dev"] = [d for d in config["dependency-groups"]["dev"] if _dependency_name(d) != "langgraph-cli"]
    assert _dependency_differences(config) == {
        "project": {"unlisted": ["langgraph-cli"], "missing": ["pypdf"]},
        "group dev": {"unlisted": [], "missing": ["langgraph-cli"]},
    }


def test_the_name_reader_strips_extras_versions_and_markers():
    requirements = ("langgraph-cli[inmem]>=0.4.42", "langchain_openai>=1", "PyYAML", "a ; python_version>'3'", "b~=1.0")
    assert [_dependency_name(d) for d in requirements] == ["langgraph-cli", "langchain-openai", "pyyaml", "a", "b"]


def _imported_modules(path: Path) -> list[str]:
    modules = []
    for node in ast.walk(_parse(path)):
        if isinstance(node, ast.Import):
            modules += [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.append(node.module)
    return modules


# F6: the per-package import scan (fence_scan.import_violations). Only agent/ imports langchain*, langgraph* and langsmith (the model only through
# langchain_openai); only guidance/ imports fastembed and pypdf, and may read flysafelz.agent.settings; reader/ keeps ticket 3's langchain_openai and
# langchain_core; openai and anthropic are banned everywhere; no other file under src/flysafelz imports any of them, or flysafelz.agent / flysafelz.guidance.
AGENT = SRC / "agent"
GUIDANCE = SRC / "guidance"


def test_the_real_tree_and_the_scripts_import_only_what_their_package_may():
    assert tree_import_violations(SRC, SCRIPT_FILES) == {}


def test_each_package_really_uses_what_it_is_allowed():
    """The scan reads real imports, and the allowances are not dead: agent/ imports langchain_openai, guidance/ imports fastembed and agent.settings, reader/ imports langchain_openai."""
    agent = {m.split(".")[0] for path in AGENT.rglob("*.py") for m in _imported_modules(path)}
    guidance = {m for path in GUIDANCE.rglob("*.py") for m in _imported_modules(path)}
    reader = {m.split(".")[0] for path in READER.rglob("*.py") for m in _imported_modules(path)}
    assert "langchain_openai" in agent
    assert {"fastembed", "flysafelz.agent.settings"} <= guidance
    assert {"langchain_openai", "langchain_core"} <= reader


def _plant(root: Path, files: dict[str, str]) -> Path:
    """A throwaway source tree: root/src/flysafelz/<relative path> for each file. Returns the flysafelz folder."""
    package = root / "src" / "flysafelz"
    for relative, text in files.items():
        (package / relative).parent.mkdir(parents=True, exist_ok=True)
        (package / relative).write_text(text, encoding="utf-8")
    return package


def test_planted_violations_are_caught_and_the_same_imports_in_their_own_package_pass(tmp_path):
    """The four cases of the brief: langgraph in a core module, fastembed in agent/, langchain_core in guidance/, and the same imports where they belong."""
    wrong = _plant(tmp_path / "wrong", {"weather.py": "import langgraph\n", "agent/graph.py": "import fastembed\n", "guidance/index.py": "from langchain_core import messages\n"})
    assert tree_import_violations(wrong) == {
        "flysafelz/agent/graph.py": ["fastembed"],
        "flysafelz/guidance/index.py": ["langchain_core"],
        "flysafelz/weather.py": ["langgraph"],
    }
    fitting = {"weather.py": "import re\n", "agent/graph.py": "import langgraph\nfrom langchain_core import messages\n", "guidance/index.py": "import fastembed\nimport pypdf\n"}
    right = _plant(tmp_path / "right", fitting)
    assert tree_import_violations(right) == {}


def test_the_tree_scan_catches_the_planted_relative_imports_of_the_review(tmp_path):
    """R1: relative imports are resolved, not skipped, when the whole tree is scanned."""
    wrong = _plant(
        tmp_path,
        {
            "weather.py": "from .agent import config\nfrom . import guidance\n",
            "guidance/index.py": "from ..agent.config import build_chat_model\n",
        },
    )
    assert tree_import_violations(wrong) == {
        "flysafelz/guidance/index.py": ["flysafelz.agent.config.build_chat_model"],
        "flysafelz/weather.py": ["flysafelz.agent.config", "flysafelz.guidance"],
    }


def test_a_relative_import_that_cannot_be_resolved_is_ignored_not_misread(tmp_path):
    """A script outside src, or an import climbing past the top package, names no flysafelz module: nothing to flag."""
    package = _plant(tmp_path, {"weather.py": "from ... import agent\n"})
    script = tmp_path / "scripts" / "tool.py"
    script.parent.mkdir()
    script.write_text("from .agent import config\n", encoding="utf-8")
    assert import_violations(package / "weather.py", package) == []
    assert import_violations(script, package) == []


@pytest.mark.parametrize(
    ("where", "text", "hit"),
    [
        ("weather.py", "import langchain\n", ["langchain"]),
        ("clients/open_meteo.py", "from langchain_openai import ChatOpenAI\n", ["langchain_openai"]),
        ("weather.py", "import langsmith\n", ["langsmith"]),
        ("weather.py", "from fastembed import TextEmbedding\n", ["fastembed"]),
        ("weather.py", "import pypdf\n", ["pypdf"]),
        ("weather.py", "import openai\n", ["openai"]),
        ("agent/graph.py", "import openai\n", ["openai"]),
        ("guidance/index.py", "import anthropic\n", ["anthropic"]),
        ("agent/graph.py", "import langchain_anthropic\n", ["langchain_anthropic"]),
        ("agent/graph.py", "from langchain_community.chat_models import ChatOllama\n", ["langchain_community.chat_models"]),
        ("agent/graph.py", "import pypdf\n", ["pypdf"]),
        ("guidance/index.py", "import langsmith\n", ["langsmith"]),
        ("guidance/index.py", "import langgraph.graph\n", ["langgraph.graph"]),
        ("reader/read.py", "import langgraph\n", ["langgraph"]),
        ("reader/read.py", "import fastembed\n", ["fastembed"]),
        ("agent/graph.py", "from langchain.chat_models import init_chat_model\n", ["init_chat_model"]),
        ("agent/graph.py", "import langchain\nmodel = langchain.chat_models.init_chat_model('x')\n", ["init_chat_model"]),
    ],
)
def test_each_planted_import_is_caught_where_it_may_not_be(tmp_path, where, text, hit):
    package = _plant(tmp_path, {where: text})
    found = import_violations(package / where, package)
    assert found and set(hit) <= set(found) and set(found) <= set(hit) | {"langchain.chat_models.init_chat_model"}


@pytest.mark.parametrize(
    ("where", "text"),
    [
        ("agent/graph.py", "import langchain\nfrom langchain.agents import create_agent\nfrom langchain_core.messages import AIMessage\nfrom langchain_openai import ChatOpenAI\n"),
        ("agent/graph.py", "import langgraph\nfrom langgraph.graph import END\nimport langgraph_sdk\nimport langsmith\nfrom langsmith import Client\n"),
        ("agent/graph.py", "from flysafelz.guidance import index\nfrom flysafelz.agent.settings import SETTINGS\n"),
        ("guidance/index.py", "import fastembed\nfrom fastembed import TextEmbedding\nimport pypdf\n"),
        ("guidance/index.py", "from flysafelz.agent.settings import SETTINGS\nfrom flysafelz.agent import settings\nfrom flysafelz.guidance import chunks\n"),
        ("reader/read.py", "from langchain_openai import ChatOpenAI\nfrom langchain_core.messages import HumanMessage\n"),
        ("weather.py", "from flysafelz.rules import RULES\nimport yaml\n"),
    ],
)
def test_the_allowed_imports_pass_in_their_own_package(tmp_path, where, text):
    package = _plant(tmp_path, {where: text})
    assert import_violations(package / where, package) == []


@pytest.mark.parametrize(
    ("where", "text", "hit"),
    [
        ("weather.py", "from flysafelz.agent.settings import SETTINGS\n", ["flysafelz.agent.settings.SETTINGS"]),
        ("weather.py", "from flysafelz.agent import config\n", ["flysafelz.agent.config"]),
        ("assess.py", "import flysafelz.guidance\n", ["flysafelz.guidance"]),
        ("cli.py", "from flysafelz import agent\n", ["flysafelz.agent"]),
        ("reader/read.py", "from flysafelz.guidance import index\n", ["flysafelz.guidance.index"]),
        ("guidance/index.py", "from flysafelz.agent.config import agent_config\n", ["flysafelz.agent.config.agent_config"]),
        ("guidance/index.py", "from flysafelz.agent import graph\n", ["flysafelz.agent.graph"]),
        # the same rules hold for relative imports, resolved against the file's own package
        ("weather.py", "from .agent import config\n", ["flysafelz.agent.config"]),
        ("weather.py", "from . import guidance\n", ["flysafelz.guidance"]),
        ("weather.py", "from .guidance.index import build\n", ["flysafelz.guidance.index.build"]),
        ("reader/read.py", "from ..agent.settings import SETTINGS\n", ["flysafelz.agent.settings.SETTINGS"]),
        ("reader/read.py", "from .. import guidance\n", ["flysafelz.guidance"]),
        ("clients/openaip.py", "from ..agent import config\n", ["flysafelz.agent.config"]),
        ("guidance/index.py", "from ..agent.config import build_chat_model\n", ["flysafelz.agent.config.build_chat_model"]),
        ("guidance/index.py", "from ..agent import graph\n", ["flysafelz.agent.graph"]),
        ("guidance/index.py", "from .. import agent\n", ["flysafelz.agent"]),
        ("guidance/deep/page.py", "from ...agent import config\n", ["flysafelz.agent.config"]),
        ("guidance/deep/page.py", "from ... import agent\n", ["flysafelz.agent"]),
        # and the allowed relative imports stay clean
        ("weather.py", "from .rules import RULES\nfrom . import geo\nfrom .clients import openaip\n", []),
        ("guidance/index.py", "from ..agent.settings import SETTINGS\nfrom ..agent import settings\nfrom . import chunks\nfrom .chunks import split\n", []),
        ("agent/graph.py", "from .config import build_chat_model\nfrom ..guidance import index\n", []),
    ],
)
def test_core_modules_do_not_import_agent_or_guidance_and_guidance_reads_only_the_settings(tmp_path, where, text, hit):
    package = _plant(tmp_path, {where: text})
    assert import_violations(package / where, package) == hit


def test_a_script_outside_src_is_held_to_the_core_rule(tmp_path):
    package = _plant(tmp_path, {})
    script = tmp_path / "scripts" / "tool.py"
    script.parent.mkdir()
    script.write_text("import langgraph\nimport fastembed\n", encoding="utf-8")
    assert import_violations(script, package) == ["langgraph", "fastembed"]
    assert tree_import_violations(package, [script]) == {"tool.py": ["langgraph", "fastembed"]}


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
def _compared_literals(path: Path) -> list[str]:
    """'<path>:<line> compares against <n>' for every comparison against a number other than 0 or 1."""
    allowed = {0, 1}
    found = []
    for node in ast.walk(_parse(path)):
        if isinstance(node, ast.Compare):
            for operand in [node.left, *node.comparators]:
                value = _number(operand)
                if value is not None and value not in allowed:
                    found.append(f"{path}:{node.lineno} compares against {value}")
    return found


def test_no_numeric_literal_in_a_comparison_other_than_0_or_1():
    for path in PY_FILES:
        assert _compared_literals(path) == []


def test_no_numeric_literal_equals_a_rules_yaml_value():
    """Structural 0 and 1 are exempt (reader_area_slack_nm is 1.0) and so are the reviewed named constants in geo.py (F6)."""
    values = _rule_values()
    assert values, "no rule values read"
    for path in PY_FILES:
        for lineno, value in hard_coded_rule_values(path, _parse(path), values):
            pytest.fail(f"{path}:{lineno} hard-codes {value}, which is a rules.yaml value")


# F7 (ticket 4): the two scans above cover agent/ and guidance/ too. A knob that equals a rules.yaml value (top_k 4, the 300 token minimum, the 3 of
# pib_stale_hours) cannot be written as a literal in code; it is read from agent/settings.yaml, which the scans do not read.
def test_the_numeric_scans_read_a_file_from_the_agent_and_guidance_packages():
    assert SRC / "agent" / "config.py" in PY_FILES and SRC / "agent" / "settings.py" in PY_FILES
    assert SRC / "guidance" / "embedding_model.py" in PY_FILES


@pytest.mark.parametrize("where", ["agent/search.py", "guidance/index.py"])
def test_a_literal_knob_in_agent_or_guidance_code_fails_the_scans(tmp_path, where):
    rule_values = _rule_values()
    assert {4, 300, 3} <= rule_values  # top_k's 4, the chunk band's 300 and pib_stale_hours' 3 are rules.yaml values
    planted = tmp_path / where
    planted.parent.mkdir()
    planted.write_text("def search(chunks, top_k=4):\n    if len(chunks) > 4:\n        return chunks[:4]\n    return [c for c in chunks if c.tokens >= 300]\n", encoding="utf-8")
    assert len(_compared_literals(planted)) == 2  # > 4 and >= 300
    assert sorted(v for _, v in hard_coded_rule_values(planted, _parse(planted), rule_values)) == [4, 4, 4, 300]
    shipped = SRC / "agent" / "settings.py"
    assert _compared_literals(shipped) == [] and hard_coded_rule_values(shipped, _parse(shipped), rule_values) == []


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
    assert {"METRES_PER_KM": 1000, "MIN_POLYGON_POINTS": 3, "QUARTER_TURN": 90} == REVIEWED_NAMED_CONSTANTS


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


@pytest.mark.parametrize("name", ["AGENT_API_" + "KEY", "LANGSMITH_API_" + "KEY"])  # split so this file never matches its own scan
def test_the_key_scan_covers_the_agent_and_langsmith_keys(tmp_path, name):
    (tmp_path / "planted.env").write_text(f"{name}=abc123\n", encoding="utf-8")
    (tmp_path / "spaced.env").write_text(f"{name} = abc123\n", encoding="utf-8")
    (tmp_path / "empty.env").write_text(f"{name}=\n{name}= # unset\n", encoding="utf-8")
    assert _files_with_a_key_value(tmp_path, ["planted.env", "spaced.env", "empty.env"]) == [f"planted.env: {name}", f"spaced.env: {name}"]


def test_env_is_gitignored_and_example_is_empty():
    ignored = subprocess.run(["git", "check-ignore", ".env"], cwd=ROOT, capture_output=True, text=True, check=False)
    assert ignored.returncode == 0
    lines = (ROOT / ".env.example").read_text(encoding="utf-8").splitlines()
    assert lines, "the example is empty"
    assert all(re.fullmatch(r"[A-Z][A-Z0-9_]*=", line) for line in lines), "every line must be NAME= with nothing after it"
    # Names split so this file never matches the F6 scan.
    listed = {"OPENAIP_API_" + "KEY=", "READER_PROVIDER=", "READER_BASE_URL=", "READER_API_" + "KEY=", "READER_MODEL="}
    listed |= {"AGENT_BASE_URL=", "AGENT_MODEL=", "AGENT_API_" + "KEY=", "LANGSMITH_API_" + "KEY=", "LANGSMITH_PROJECT="}
    assert set(lines) == listed and len(lines) == len(listed)


def test_the_example_scan_refuses_a_value_after_the_name():
    ok = re.compile(r"[A-Z][A-Z0-9_]*=")
    assert [bool(ok.fullmatch(line)) for line in ("AGENT_MODEL=", "AGENT_MODEL=x", "AGENT_MODEL= ", "AGENT_MODEL=#c", "AGENT_MODEL", "# AGENT_MODEL=")] == [True, False, False, False, False, False]


# F7: plain modules and functions.
def test_contract_classes_subclass_only_basemodel_or_enum():
    classes = [o for o in vars(contracts).values() if isinstance(o, type) and o.__module__ == contracts.__name__]
    assert classes
    for cls in classes:
        assert cls.__bases__ in {(BaseModel,), (enum.Enum,), (enum.StrEnum,)}, cls


def test_no_registry_di_or_caching_in_source():
    """Narrowed for the reader (ticket 3 F2) and for the map routes (ticket 8 F6): only src/flysafelz/reader/ and src/flysafelz/mapdata/ may use the word `cache`.
    test_reader_cache_fence.py and test_mapdata_fence.py show a planted one elsewhere is still caught."""
    for path in PY_FILES:
        assert banned_words(path, READER, MAPDATA) == [], path


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


def test_only_agent_config_py_names_an_agent_variable():
    """AGENT_* is read in one place, as READER_* is read in reader/config.py."""
    named = [path.relative_to(SRC).as_posix() for path in PY_FILES if re.search(r"AGENT_[A-Z]", path.read_text(encoding="utf-8"))]
    assert named == [AGENT_CONFIG]


def test_rules_module_and_cli_read_no_environment():
    for name in ("rules.py", "cli.py"):
        text = (SRC / name).read_text(encoding="utf-8")
        assert not re.search(r"environ|getenv", text), name


# F10: no LLM env vars, no /models dependency. Narrowed for the reader (ticket 3 F2): only src/flysafelz/reader/ may name READER_*. mapdata/ gets no exemption.
# test_reader_fence.py shows a planted READER_ or /models string outside it is still caught.
def test_no_llm_env_vars_or_models_endpoint_in_source():
    for path in PY_FILES:
        assert llm_env_words(path, READER) == [], path


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


# F4 (3b PR-2.F1): the SUP reference and the display designator list are found by code, never by a model. The two modules that find them
# import only the standard library: no flysafelz.reader, no LLM client, nothing in FORBIDDEN_IMPORTS.
CODE_FOUND_FILES = [SRC / "sup_refs.py", SRC / "designators.py"]


def _non_stdlib_imports(path: Path) -> list[str]:
    return [m for m in _imported_modules(path) if m.split(".")[0] not in sys.stdlib_module_names]


def test_the_code_found_field_modules_import_only_the_standard_library():
    assert all(p.exists() for p in CODE_FOUND_FILES)
    for path in CODE_FOUND_FILES:
        assert _imported_modules(path), path  # the scan reads real imports
        assert _non_stdlib_imports(path) == [], path
        assert not [m for m in _imported_modules(path) if m.startswith(FORBIDDEN_IMPORTS) or "reader" in m.lower()], path


def test_the_standard_library_scan_catches_planted_imports(tmp_path):
    for module in ("flysafelz.reader", "flysafelz.reader.read", "langchain_core", "openai", "anthropic", "yaml"):
        planted = tmp_path / "planted.py"
        planted.write_text(f"import re\nimport {module}\n", encoding="utf-8")
        assert _non_stdlib_imports(planted) == [module], module
    planted.write_text("import re\nfrom collections import abc\n", encoding="utf-8")
    assert _non_stdlib_imports(planted) == []


def test_the_sup_and_designator_properties_are_plain_properties_not_fields():
    for name in ("sup_references", "designators"):
        assert isinstance(vars(contracts.ParsedNotam)[name], property), name
        assert name not in contracts.ParsedNotam.model_fields, name
        assert name not in contracts.NotamReading.model_fields, name
    assert "activates_designators" in contracts.NotamReading.model_fields  # activation stays the reader's field (CR5)
