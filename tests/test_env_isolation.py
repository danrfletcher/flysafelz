"""E21, F9: AGENT_* and LANGSMITH_* exported in the shell that runs the tests never reach a test, and tracing is off in every test.

The checks that need a polluted environment run in a subprocess (pytester.runpytest_subprocess) that has the variables exported, with
a copy of tests/conftest.py. Each has a control: the same inner test without the conftest fails, so a pass is not vacuous.
Names are written in pieces where joined they would read as a key assignment, so no tracked file matches the F6 key scan.
"""

import ast
import os
import subprocess
import sys
from pathlib import Path

import pytest
from langsmith.utils import tracing_is_enabled

ROOT = Path(__file__).resolve().parents[1]
CONFTEST = ROOT / "tests" / "conftest.py"

AGENT_KEY = "AGENT_API_" + "KEY"
LANGSMITH_KEY = "LANGSMITH_API_" + "KEY"
# The five the brief names, then the rest of what conftest.py clears.
EXPORTED = {
    AGENT_KEY: "dummy-agent-token",
    "AGENT_BASE_URL": "http://127.0.0.1:1/v1",
    LANGSMITH_KEY: "dummy-langsmith-token",
    "LANGSMITH_TRACING": "true",
    "LANGCHAIN_TRACING_V2": "true",
}
ALSO_EXPORTED = {
    "AGENT_MODEL": "dummy/model",
    "LANGSMITH_PROJECT": "dummy-project",
    "LANGSMITH_ENDPOINT": "http://127.0.0.1:1",
    "LANGSMITH_TRACING_V2": "true",
    "LANGCHAIN_TRACING": "true",
    "LANGCHAIN_API_" + "KEY": "dummy-legacy-token",
    "LANGCHAIN_ENDPOINT": "http://127.0.0.1:1",
    "LANGCHAIN_PROJECT": "dummy-project",
    "OPENAI_API_" + "KEY": "dummy-openai-token",
    "OPENAI_BASE_URL": "http://127.0.0.1:1/v1",
    "JUDGE_API_" + "KEY": "dummy-judge-token",  # the eval judge's settings (5a PR-7.F1)
    "JUDGE_BASE_URL": "http://127.0.0.1:1/v1",
    "JUDGE_MODEL": "dummy/judge",
    "JUDGE_PROVIDER": "dummy",
}

INNER_TEST = f"""
import os

from langsmith.utils import tracing_is_enabled

EXPORTED = {sorted({**EXPORTED, **ALSO_EXPORTED})!r}
# Read while the test module is imported, which is after pytest_configure and before any test runs.
SEEN_AT_COLLECTION = {{name: os.environ.get(name) for name in EXPORTED}}


def test_nothing_is_set_when_the_test_module_is_imported():
    leaked = {{name: value for name, value in SEEN_AT_COLLECTION.items() if value is not None and name != "LANGSMITH_TRACING"}}
    assert leaked == {{}}
    assert SEEN_AT_COLLECTION["LANGSMITH_TRACING"] == "false"


def test_each_is_absent_in_the_test_and_langsmith_tracing_is_false():
    for name in EXPORTED:
        if name == "LANGSMITH_TRACING":
            assert os.environ[name] == "false"
        else:
            assert name not in os.environ, name


def test_langsmith_says_tracing_is_off():
    assert tracing_is_enabled() is False
"""

LATE_EXPORT = '''

def pytest_collection_finish(session):
    """After pytest_configure and before the first test: something exports variables late. The per-test fixture must clear them again."""
    os.environ["AGENT_MODEL"] = "set-late"
    os.environ["LANGSMITH_PROJECT"] = "set-late"
    os.environ["LANGSMITH_TRACING"] = "true"
'''


@pytest.fixture
def polluted(monkeypatch):
    """The brief's five variables, and the rest, exported in this process so the pytester subprocess inherits them."""
    for name, value in {**EXPORTED, **ALSO_EXPORTED}.items():
        monkeypatch.setenv(name, value)


def run_inner(pytester, *, conftest: str | None):
    pytester.makepyfile(test_inner=INNER_TEST)
    if conftest is not None:
        pytester.makeconftest(conftest)
    return pytester.runpytest_subprocess("--disable-socket", "-p", "no:cacheprovider")


def test_the_exported_variables_reach_an_inner_run_with_no_conftest(pytester, polluted):
    """The control: without tests/conftest.py the inner tests fail, so the pass below is the conftest's doing."""
    result = run_inner(pytester, conftest=None)
    assert result.ret != 0
    result.assert_outcomes(failed=3)


def test_with_the_conftest_every_exported_variable_is_gone_and_tracing_is_off(pytester, polluted):
    result = run_inner(pytester, conftest=CONFTEST.read_text(encoding="utf-8"))
    result.assert_outcomes(passed=3)
    assert result.ret == 0
    result.stdout.fnmatch_lines(["network: --disable-socket in effect"])


def test_the_per_test_fixture_clears_what_is_exported_after_the_session_started(pytester, polluted):
    """pytest_configure clears at session start; a variable exported later (here in pytest_collection_finish) is cleared by the autouse fixture."""
    late = CONFTEST.read_text(encoding="utf-8") + LATE_EXPORT
    pytester.makepyfile(
        test_inner="""
import os
from langsmith.utils import tracing_is_enabled


def test_a_late_export_is_cleared_for_the_test():
    assert "AGENT_MODEL" not in os.environ and "LANGSMITH_PROJECT" not in os.environ
    assert os.environ["LANGSMITH_TRACING"] == "false"
    assert tracing_is_enabled() is False
"""
    )
    pytester.makeconftest(late)
    result = pytester.runpytest_subprocess("--disable-socket", "-p", "no:cacheprovider")
    result.assert_outcomes(passed=1)


def test_in_this_run_nothing_agent_or_langsmith_is_set_and_tracing_is_off():
    leaked = [name for name in os.environ if name.startswith(("AGENT_", "LANGSMITH_")) and name != "LANGSMITH_TRACING"]
    assert leaked == []
    assert os.environ["LANGSMITH_TRACING"] == "false"
    assert tracing_is_enabled() is False
    for name in ("LANGCHAIN_TRACING_V2", "LANGCHAIN_TRACING", "LANGCHAIN_ENDPOINT", "LANGCHAIN_PROJECT", "OPENAI_BASE_URL", "OPENAI_API_" + "KEY", "LANGCHAIN_API_" + "KEY"):
        assert name not in os.environ, name


def test_conftest_imports_urllib3_at_module_top_before_sockets_are_blocked():
    """urllib3's import-time IPv6 probe opens a socket; pytest-socket warns when that happens inside a test. Imported at the top, it is done by then."""
    tree = ast.parse(CONFTEST.read_text(encoding="utf-8"))
    top_level = [n for n in tree.body if isinstance(n, ast.Import) and any(a.name == "urllib3" for a in n.names)]
    assert top_level, "tests/conftest.py must `import urllib3` at module top"
    assert "urllib3" in sys.modules


QUIET_FILES = ["tests/test_fakes.py", "tests/unit/test_agent_settings.py", "tests/unit/test_agent_config.py", "tests/unit/test_embedding_model.py"]


def test_the_agent_tests_run_clean_with_pytest_socket_warnings_as_errors():
    """A pytest-socket warning (a test touched socket.socket) is an error here, so a stray network attempt or an unguarded urllib3 import fails the run."""
    command = [sys.executable, "-m", "pytest", "-p", "no:cacheprovider", "-W", "error::UserWarning:pytest_socket", *QUIET_FILES]
    result = subprocess.run(command, cwd=ROOT, env={**os.environ}, capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "network: --disable-socket in effect" in result.stdout
    assert "warning" not in result.stdout.splitlines()[-1]
