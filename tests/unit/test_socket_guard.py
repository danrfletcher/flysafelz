"""GP2, F5: under the project's addopts, a socket call fails the run with SocketBlockedError."""

import tomllib
from pathlib import Path

PYPROJECT = Path(__file__).resolve().parents[2] / "pyproject.toml"

INNER_TEST = """
import socket

def test_opens_a_connection():
    socket.create_connection(("192.0.2.1", 80), timeout=1)
"""


def test_addopts_disable_sockets():
    config = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
    assert "--disable-socket" in config["tool"]["pytest"]["ini_options"]["addopts"]


def test_a_socket_call_fails_the_run(pytester):
    config = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
    addopts = config["tool"]["pytest"]["ini_options"]["addopts"].split()
    pytester.makepyfile(test_inner=INNER_TEST)
    result = pytester.runpytest_subprocess(*addopts)
    assert result.ret != 0
    result.assert_outcomes(failed=1)
    result.stdout.fnmatch_lines(["*SocketBlockedError*"])
