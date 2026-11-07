"""Shared test setup: a frozen clock, no API key, no dotenv, no sockets (addopts)."""

import sys
from datetime import UTC, datetime

import pytest

pytest_plugins = ["pytester"]

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
def isolated_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """No test sees a real key or reads .env: the key is removed and `import dotenv` fails."""
    monkeypatch.delenv("OPENAIP_API_KEY", raising=False)
    monkeypatch.setitem(sys.modules, "dotenv", None)


def pytest_report_header(config: pytest.Config) -> str:
    """Show in the pytest header that the socket guard is on (checked by the verification run)."""
    state = "--disable-socket in effect" if config.getoption("--disable-socket") else "SOCKETS ENABLED"
    return f"network: {state}"
