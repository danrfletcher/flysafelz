"""Fence checks for assess.py: F3 (no threshold literal), F4 (three verdicts, no score), F5/F1 (no model, no network), F7 (plain code)."""

import ast
import itertools
import re
from pathlib import Path

from factories import NOW, run_assess, utc

from flysafelz import assess as module
from flysafelz.assess import display
from flysafelz.contracts import Verdict, display_verdict

PATH = Path(module.__file__)
FORBIDDEN_MODULES = ("openai", "anthropic", "langchain", "langgraph", "httpx", "requests", "urllib", "socket", "dotenv", "os")
CLIENT_CALLS = {"fetch_forecast", "fetch_airspace", "fetch_pib", "load_pib", "fetch_pib_text", "post", "request", "stream", "Client"}
FROM_CLIENTS = {"Forecast", "AirspaceFetch", "metres_to_feet", "utc_text"}  # types and pure helpers, no fetch function
TESTS = Path(__file__).parent
SIX_STRINGS = {
    "NO-GO",
    "UNVERIFIED",
    "PILOT DISCRETION",
    "PILOT DISCRETION \u2013 APPROACHING LIMITS",
    "PILOT DISCRETION \u2013 WITH CONDITIONS",
    "PILOT DISCRETION \u2013 APPROACHING LIMITS \u00b7 WITH CONDITIONS",
}


def tree() -> ast.Module:
    return ast.parse(PATH.read_text(encoding="utf-8"), filename=str(PATH))


def imported(module_tree: ast.Module) -> list[str]:
    found = []
    for node in ast.walk(module_tree):
        if isinstance(node, ast.Import):
            found += [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            found.append(node.module or "")
    return found


# --- F3 ------------------------------------------------------------------------------------


def test_f3_no_comparison_holds_a_number_other_than_zero_or_one():
    for node in ast.walk(tree()):
        if isinstance(node, ast.Compare):
            for operand in [node.left, *node.comparators]:
                if isinstance(operand, ast.Constant) and isinstance(operand.value, (int, float)):
                    assert operand.value in (0, 1), ast.unparse(node)


def test_f3_no_number_in_the_code_at_all_except_zero_and_one():
    numbers = [n.value for n in ast.walk(tree()) if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)) and not isinstance(n.value, bool)]
    assert set(numbers) <= {0, 1}, sorted(set(numbers) - {0, 1})


def test_f3_the_caps_and_stale_hours_are_read_from_the_rules():
    source = PATH.read_text(encoding="utf-8")
    for rule in ("outlook_horizon_day", "verdict_horizon_day", "pib_stale_hours"):
        assert f"RULES.{rule}" in source, rule


# --- F4 ------------------------------------------------------------------------------------


def test_f4_the_verdict_enum_has_exactly_three_members():
    assert {v.name for v in Verdict} == {"NO_GO", "UNVERIFIED", "PILOT_DISCRETION"}
    assert len(Verdict) == 3


def test_f4_nothing_is_summed_or_weighted():
    called = {n.func.id for n in ast.walk(tree()) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
    assert not called & {"sum", "fsum", "mean"}
    names = {n.id for n in ast.walk(tree()) if isinstance(n, ast.Name)} | {n.attr for n in ast.walk(tree()) if isinstance(n, ast.Attribute)}
    assert not [n for n in names if re.search(r"weight|score", n, re.IGNORECASE)]
    assert not any(isinstance(n, ast.Constant) and n.value == "GO" for n in ast.walk(tree()))
    assert not [n for n in ast.walk(tree()) if isinstance(n, ast.BinOp) and isinstance(n.op, (ast.Mult, ast.Div))]


def test_f4_display_comes_only_from_the_six_strings_and_is_never_a_bare_go():
    seen = set()
    for verdict, approaching, conditions in itertools.product(Verdict, [False, True], [False, True]):
        seen.add(display_verdict(verdict, approaching, conditions))
    assert seen == SIX_STRINGS
    assert "GO" not in seen


def test_f4_every_segment_assess_returns_displays_as_one_of_the_six():
    assessment = run_assess(window=(utc(6), utc(18)))
    shown = {display(s) for h in assessment.hours for s in h.segments}
    assert shown and shown <= SIX_STRINGS


def test_f4_display_is_composed_in_one_function():
    uses = [n for n in ast.walk(tree()) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "display_verdict"]
    assert len(uses) == 1


# --- F5 / F1 -------------------------------------------------------------------------------


def test_f1_assess_imports_nothing_that_could_call_a_model_or_the_network():
    for name in imported(tree()):
        root = name.split(".")[0]
        assert root not in FORBIDDEN_MODULES, name
        assert not root.startswith("langchain"), name
    assert not re.search(r"\bopenai\b|anthropic|langchain|langgraph|httpx|os\.environ|getenv|dotenv|\.env\b", PATH.read_text(encoding="utf-8"), re.IGNORECASE)


def test_f1_assess_never_calls_a_client():
    called = set()
    for node in ast.walk(tree()):
        if isinstance(node, ast.Call):
            called.add(node.func.id if isinstance(node.func, ast.Name) else getattr(node.func, "attr", ""))
    assert not called & CLIENT_CALLS, called & CLIENT_CALLS


def test_f1_from_the_clients_assess_imports_only_types_and_pure_helpers():
    for node in ast.walk(tree()):
        if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("flysafelz.clients"):
            assert {a.name for a in node.names} <= FROM_CLIENTS, node.module


def test_f5_the_assess_tests_use_the_frozen_clock_and_no_dotenv():
    assert NOW == utc(7, 30)  # the conftest clock, 2026-10-31T07:30Z
    for path in [*TESTS.glob("test_assess_*.py"), *(TESTS.parent / "integration").glob("test_assess_*.py")]:
        if path.name == Path(__file__).name:
            continue  # this file names the forbidden words
        text = path.read_text(encoding="utf-8")
        assert "datetime.now" not in text and "utcnow" not in text, path.name
        assert "dotenv" not in text and "os.environ" not in text, path.name


# --- F7 ------------------------------------------------------------------------------------


def test_f7_assess_defines_no_classes_and_no_decorated_functions():
    nodes = list(ast.walk(tree()))
    assert not [n for n in nodes if isinstance(n, ast.ClassDef)]
    assert not [n for n in nodes if isinstance(n, ast.FunctionDef) and n.decorator_list]  # no lru_cache, no registry decorators
    assert not [n for n in nodes if isinstance(n, (ast.Global, ast.Nonlocal))]


def test_f7_no_module_level_state_is_changed_by_a_run():
    def state() -> dict[str, str]:
        return {k: repr(v) for k, v in vars(module).items() if not k.startswith("__") and not callable(v) and not isinstance(v, type(module))}

    before = state()
    run_assess(window=(utc(9), utc(12)))
    assert state() == before
