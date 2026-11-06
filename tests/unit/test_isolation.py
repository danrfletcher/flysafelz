"""F5: every test runs with the frozen clock, no API key and no dotenv."""

import os
import socket
from datetime import UTC, datetime

import pytest
from pytest_socket import SocketBlockedError


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
