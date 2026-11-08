"""Shared test setup: a frozen clock, no API key, no dotenv, no tracing, no sockets (addopts)."""

import os
import sys
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path

import pytest

# Imported here, at module top, before pytest-socket blocks sockets: urllib3 (a dependency of langsmith) probes for IPv6 when it is
# first imported, which opens a socket without connecting. Imported inside a test, that probe makes pytest-socket warn "A test tried
# to use socket.socket", which looks like a network call and is not. test_env_isolation.py runs the suite with that warning as an error.
import urllib3  # noqa: F401

pytest_plugins = ["pytester"]

# Written out, not imported from reader/config.py, so a broken config module cannot break every test. test_isolation.py checks they match.
READER_VARIABLES = ("READER_PROVIDER", "READER_BASE_URL", "READER_API_KEY", "READER_MODEL")
DOTENV_PATH = "READER_DOTENV_PATH"

# The agent's settings and LangSmith's are cleared by prefix, so a variable added later is cleared too. The legacy names are the ones
# langsmith and langchain still honour to switch tracing on or to supply a key or an endpoint; the OPENAI_* pair is what an OpenAI client
# would fall back to. Then tracing is switched off explicitly.
CLEARED_PREFIXES = ("AGENT_", "LANGSMITH_")
LEGACY_VARIABLES = (
    "LANGCHAIN_TRACING_V2",
    "LANGCHAIN_TRACING",
    "LANGCHAIN_API_KEY",
    "LANGCHAIN_ENDPOINT",
    "LANGCHAIN_PROJECT",
    "OPENAI_API_KEY",
    "OPENAI_BASE_URL",
)
TRACING_OFF = ("LANGSMITH_TRACING", "false")


def variables_to_clear(environ: Mapping[str, str]) -> list[str]:
    """The names in `environ` that the agent, LangSmith or an OpenAI client would read: every AGENT_* and LANGSMITH_* and the legacy names."""
    return [name for name in environ if name.startswith(CLEARED_PREFIXES) or name in LEGACY_VARIABLES]


FROZEN_NOW = datetime(2026, 10, 31, 7, 30, tzinfo=UTC)


def pytest_configure(config: pytest.Config) -> None:
    """Clear the agent and tracing variables at session start, before any test module imports langsmith, which caches what it reads from the environment.

    The autouse `isolated_environment` fixture clears them again for each test.
    """
    for name in variables_to_clear(os.environ):
        del os.environ[name]
    os.environ[TRACING_OFF[0]] = TRACING_OFF[1]


def pytest_addoption(parser: pytest.Parser) -> None:
    """tests/golden/*.txt are rewritten only when this flag is given, never automatically."""
    parser.addoption("--update-golden", action="store_true", default=False, help="rewrite tests/golden/*.txt from the current CLI output")


@pytest.fixture
def update_golden(request: pytest.FixtureRequest) -> bool:
    return bool(request.config.getoption("--update-golden"))


@pytest.fixture(autouse=True)
def now() -> datetime:
    """The frozen clock every test uses: 2026-10-31T07:30Z. Code under test takes it as an argument."""
    return FROZEN_NOW


@pytest.fixture(autouse=True)
def forget_remembered_areas() -> None:
    """notam_area remembers each NOTAM's area (so a parser miss is logged once); a test starts without any, so it sees its own log."""
    from flysafelz import notam_area

    notam_area.forget_areas()


@pytest.fixture(autouse=True)
def isolated_environment(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """No test sees a real key or reads .env: the key is removed and `import dotenv` fails.

    The reader's settings are cleared too: every READER_* variable is removed, and READER_DOTENV_PATH names a file that does
    not exist, so reader/config.py can never load a real .env (E15). A test that needs reader settings sets its own.

    The agent's settings and LangSmith's are cleared too: every AGENT_* and LANGSMITH_* variable, the legacy LANGCHAIN_* names that
    switch on tracing or supply a key, and OPENAI_API_KEY / OPENAI_BASE_URL are removed, then LANGSMITH_TRACING is set to false. The
    reader's and the agent's model clients would otherwise send every prompt to LangSmith from a developer's shell that has it on. A
    test that needs one sets its own.
    """
    monkeypatch.delenv("OPENAIP_API_KEY", raising=False)
    for name in READER_VARIABLES:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv(DOTENV_PATH, str(tmp_path / "no-such-dotenv" / ".env"))
    for name in variables_to_clear(os.environ):
        monkeypatch.delenv(name)
    monkeypatch.setenv(*TRACING_OFF)
    monkeypatch.setitem(sys.modules, "dotenv", None)
    # The command line keeps readings in a SQLite file at the repository root; a test never touches it.
    from flysafelz.reader import config

    monkeypatch.setattr(config, "reading_store", lambda: tmp_path / "readings.sqlite")


def pytest_report_header(config: pytest.Config) -> str:
    """Show in the pytest header that the socket guard is on (checked by the verification run)."""
    state = "--disable-socket in effect" if config.getoption("--disable-socket") else "SOCKETS ENABLED"
    return f"network: {state}"
