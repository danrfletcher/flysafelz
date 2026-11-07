"""Shared test setup: a frozen clock, no API key, no dotenv, no tracing, no sockets (addopts)."""

import sys
from datetime import UTC, datetime
from pathlib import Path

import pytest

pytest_plugins = ["pytester"]

# Written out, not imported from reader/config.py, so a broken config module cannot break every test. test_isolation.py checks they match.
READER_VARIABLES = ("READER_PROVIDER", "READER_BASE_URL", "READER_API_KEY", "READER_MODEL")
DOTENV_PATH = "READER_DOTENV_PATH"

FROZEN_NOW = datetime(2026, 10, 31, 7, 30, tzinfo=UTC)


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
def isolated_environment(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """No test sees a real key or reads .env: the key is removed and `import dotenv` fails.

    The reader's settings are cleared too: every READER_* variable is removed, and READER_DOTENV_PATH names a file that does
    not exist, so reader/config.py can never load a real .env (E15). A test that needs reader settings sets its own.

    LangSmith tracing is switched off too: the reader's model client would otherwise send every prompt to LangSmith
    from a developer's shell that has it on.
    """
    monkeypatch.delenv("OPENAIP_API_KEY", raising=False)
    for name in READER_VARIABLES:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv(DOTENV_PATH, str(tmp_path / "no-such-dotenv" / ".env"))
    for name in ("LANGSMITH_TRACING", "LANGSMITH_TRACING_V2", "LANGCHAIN_TRACING", "LANGCHAIN_TRACING_V2"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setitem(sys.modules, "dotenv", None)


def pytest_report_header(config: pytest.Config) -> str:
    """Show in the pytest header that the socket guard is on (checked by the verification run)."""
    state = "--disable-socket in effect" if config.getoption("--disable-socket") else "SOCKETS ENABLED"
    return f"network: {state}"
