"""F5: every test runs with the frozen clock, no API key, no dotenv and no reader settings (E15)."""

import importlib.util
import os
import socket
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import pytest
from pytest_socket import SocketBlockedError

from flysafelz.reader import config
from flysafelz.reader.config import DOTENV_PATH, READER_VARIABLES, ReaderConfigError, reader_config

ROOT = Path(__file__).resolve().parents[2]
THIS_FILE = "tests/unit/test_isolation.py"


def test_now_is_frozen(now):
    assert now == datetime(2026, 10, 31, 7, 30, tzinfo=UTC)


def test_api_key_is_removed():
    assert "OPENAIP_API_KEY" not in os.environ


def test_dotenv_cannot_be_imported():
    with pytest.raises(ImportError):
        import dotenv  # noqa: F401


@pytest.mark.filterwarnings("ignore:A test tried to use socket")
def test_a_socket_attempt_raises_socket_blocked_error():
    with pytest.raises(SocketBlockedError):
        socket.socket(socket.AF_INET, socket.SOCK_STREAM)


# --- E15: no READER_* setting and no real .env reaches a test ------------------------------------------------------------------


def test_every_reader_setting_is_removed_and_the_dotenv_path_is_a_file_that_does_not_exist():
    assert [name for name in READER_VARIABLES if name in os.environ] == []
    path = Path(os.environ[DOTENV_PATH])
    assert not path.exists() and path != ROOT / ".env"


def test_the_reader_config_then_sees_no_settings_and_no_test_can_load_a_real_dotenv(monkeypatch, tmp_path):
    with pytest.raises(ReaderConfigError) as error:
        reader_config()
    assert error.value.missing == READER_VARIABLES

    # Even with a real-looking .env at the place the reader would look by default, the injected path hides it.
    (tmp_path / ".env").write_text("".join(f"{name}=from-a-real-dotenv\n" for name in READER_VARIABLES), encoding="utf-8")
    monkeypatch.setattr(config, "_repository_root", lambda: tmp_path)
    with pytest.raises(ReaderConfigError):
        reader_config()

    # The control: without the injected path the same file is read, so it is the path that protects the test, not an absent file.
    monkeypatch.delenv(DOTENV_PATH)
    assert reader_config().model == "from-a-real-dotenv"


def test_conftest_clears_exactly_the_variables_the_reader_reads():
    spec = importlib.util.spec_from_file_location("shared_conftest", ROOT / "tests" / "conftest.py")
    assert spec is not None and spec.loader is not None
    shared = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(shared)
    assert shared.READER_VARIABLES == READER_VARIABLES and shared.DOTENV_PATH == DOTENV_PATH


def test_reader_settings_exported_in_the_shell_that_runs_pytest_are_cleared_before_a_test_runs(tmp_path):
    """Run two of the checks above in a pytest process that has all four READER_* variables exported, and READER_DOTENV_PATH naming a real file."""
    real_dotenv = tmp_path / "real.env"
    real_dotenv.write_text("".join(f"{name}=exported-{name}\n" for name in READER_VARIABLES), encoding="utf-8")
    exported = {name: f"exported-{name}" for name in READER_VARIABLES} | {DOTENV_PATH: str(real_dotenv)}
    inner = [f"{THIS_FILE}::test_every_reader_setting_is_removed_and_the_dotenv_path_is_a_file_that_does_not_exist", f"{THIS_FILE}::test_the_reader_config_then_sees_no_settings_and_no_test_can_load_a_real_dotenv"]

    result = subprocess.run([sys.executable, "-m", "pytest", "-p", "no:cacheprovider", *inner], cwd=ROOT, env={**os.environ, **exported}, capture_output=True, text=True, check=False)

    assert result.returncode == 0, result.stdout + result.stderr
    assert "2 passed" in result.stdout
    assert "network: --disable-socket in effect" in result.stdout  # the inner run is offline too
