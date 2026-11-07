"""agent/config.py: AGENT_* read lazily from the environment, every missing one named, and a ChatOpenAI that never looks elsewhere for a key.

No mocks: the tests set real environment variables (monkeypatch) and build the real client. Sockets are off (addopts), so a
network call from building the client would fail the test.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from flysafelz.agent import config
from flysafelz.agent.config import AGENT_VARIABLES, AgentConfig, AgentConfigError, agent_config, build_chat_model
from flysafelz.agent.settings import SETTINGS

ROOT = Path(__file__).resolve().parents[2]
BASE_URL, MODEL, KEY = AGENT_VARIABLES
VALUES = {BASE_URL: "http://127.0.0.1:6008/api/v1", MODEL: "some/model", KEY: "dummy-" + "token"}


# Building a ChatOpenAI makes langchain_openai create one throwaway socket to see which TCP options the platform accepts. It never connects;
# under --disable-socket pytest-socket blocks it, langchain_openai catches that and goes on, and pytest-socket warns. That warning is not a network call.
PROBE_SOCKET = pytest.mark.filterwarnings("ignore:A test tried to use socket")


@pytest.fixture
def agent_env(monkeypatch):
    """All three AGENT_* variables set to dummy values."""
    for name, value in VALUES.items():
        monkeypatch.setenv(name, value)


def test_the_variable_names_are_the_three_agent_ones_in_order():
    assert AGENT_VARIABLES == ("AGENT_BASE_URL", "AGENT_MODEL", "AGENT_API_KEY")


def test_importing_the_module_with_the_variables_unset_raises_nothing():
    """A fresh interpreter with no AGENT_* in its environment: the import succeeds and reads nothing."""
    environment = {name: value for name, value in os.environ.items() if not name.startswith("AGENT_")}
    result = subprocess.run([sys.executable, "-c", "import flysafelz.agent.config"], cwd=ROOT, env=environment, capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr


def test_the_variables_are_read_when_called_not_when_imported(monkeypatch):
    with pytest.raises(AgentConfigError):
        agent_config()
    for name, value in VALUES.items():
        monkeypatch.setenv(name, value)  # set after the module was imported
    assert agent_config().model == "some/model"
    monkeypatch.setenv(MODEL, "another/model")
    assert agent_config().model == "another/model"


def test_with_nothing_set_one_error_names_all_three_and_no_default_is_used():
    with pytest.raises(AgentConfigError) as error:
        agent_config()
    assert error.value.missing == (BASE_URL, MODEL, KEY) and error.value.invalid == ()
    assert str(error.value) == "agent settings missing: AGENT_BASE_URL, AGENT_MODEL, AGENT_API_KEY (set them in the environment)"
    assert isinstance(error.value, ValueError)


@pytest.mark.parametrize("absent", [(BASE_URL,), (MODEL,), (KEY,), (BASE_URL, KEY), (MODEL, KEY)])
def test_every_missing_variable_is_named_and_the_others_are_not(monkeypatch, agent_env, absent):
    for name in absent:
        monkeypatch.delenv(name)
    with pytest.raises(AgentConfigError) as error:
        agent_config()
    assert error.value.missing == tuple(name for name in AGENT_VARIABLES if name in absent)
    for name in AGENT_VARIABLES:
        assert (name in str(error.value)) == (name in absent)


@pytest.mark.parametrize("blank", ["", "   ", "\t\n"])
@pytest.mark.parametrize("name", AGENT_VARIABLES)
def test_a_blank_variable_counts_as_missing(monkeypatch, agent_env, name, blank):
    monkeypatch.setenv(name, blank)
    with pytest.raises(AgentConfigError) as error:
        agent_config()
    assert error.value.missing == (name,)


@pytest.mark.parametrize("url", ["127.0.0.1:6008/api/v1", "localhost:6008/api/v1", "api.example.com/v1", "ftp://example.com/v1", "http://", "https:///v1", "/api/v1"])
def test_a_base_url_without_an_http_scheme_and_host_is_rejected_by_name(monkeypatch, agent_env, url):
    monkeypatch.setenv(BASE_URL, url)
    with pytest.raises(AgentConfigError) as error:
        agent_config()
    assert error.value.invalid == (BASE_URL,) and error.value.missing == ()
    assert BASE_URL in str(error.value)


def test_a_rejected_base_url_is_not_echoed_because_it_may_hold_a_credential(monkeypatch, agent_env):
    monkeypatch.setenv(BASE_URL, "ftp://user:hunter2@example.com/v1")
    with pytest.raises(AgentConfigError) as error:
        agent_config()
    assert "hunter2" not in str(error.value) and "example.com" not in str(error.value)


def test_a_missing_variable_and_a_bad_url_are_reported_together(monkeypatch, agent_env):
    monkeypatch.setenv(BASE_URL, "no-scheme")
    monkeypatch.delenv(MODEL)
    with pytest.raises(AgentConfigError) as error:
        agent_config()
    assert error.value.missing == (MODEL,) and error.value.invalid == (BASE_URL,)
    assert "AGENT_MODEL" in str(error.value) and "AGENT_BASE_URL" in str(error.value)


@pytest.mark.parametrize("url", ["http://127.0.0.1:6008/api/v1", "https://api.example.com/v1"])
def test_http_and_https_urls_are_accepted(monkeypatch, agent_env, url):
    monkeypatch.setenv(BASE_URL, url)
    assert agent_config().base_url == url


def test_values_are_trimmed_and_the_key_is_hidden_from_the_repr(agent_env, monkeypatch):
    monkeypatch.setenv(MODEL, "  some/model  ")
    loaded = agent_config()
    assert isinstance(loaded, AgentConfig) and loaded.model == "some/model"
    assert loaded.api_key.get_secret_value() == VALUES[KEY]
    assert VALUES[KEY] not in repr(loaded) and VALUES[KEY] not in str(loaded)


def test_a_mapping_can_be_passed_instead_of_the_environment():
    assert agent_config(VALUES).base_url == VALUES[BASE_URL]
    with pytest.raises(AgentConfigError):
        agent_config({})


@PROBE_SOCKET
def test_build_chat_model_makes_a_client_for_the_endpoint_with_the_key_passed_explicitly(agent_env, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "other-" + "token")  # must never be consulted
    monkeypatch.setenv("OPENAI_BASE_URL", "http://127.0.0.1:1/other")
    client = build_chat_model()
    assert isinstance(client, ChatOpenAI)
    assert client.model_name == "some/model"
    assert client.openai_api_base == VALUES[BASE_URL]
    assert isinstance(client.openai_api_key, SecretStr) and client.openai_api_key.get_secret_value() == VALUES[KEY]


@PROBE_SOCKET
def test_the_clients_max_retries_come_from_settings(agent_env):
    client = build_chat_model()
    assert client.max_retries == SETTINGS.model_max_retries == 1


def test_build_chat_model_with_nothing_set_names_every_missing_variable():
    with pytest.raises(AgentConfigError, match="AGENT_BASE_URL, AGENT_MODEL, AGENT_API_KEY"):
        build_chat_model()


@PROBE_SOCKET
def test_build_chat_model_takes_a_config_instead_of_reading_the_environment():
    client = build_chat_model(agent_config(VALUES))
    assert client.model_name == "some/model"


@PROBE_SOCKET
def test_building_the_client_opens_no_connection_and_asks_for_no_model_list(agent_env):
    """Sockets are blocked by addopts, so any connection (a model-list call included) would raise SocketBlockedError here."""
    build_chat_model()
    text = Path(config.__file__).read_text(encoding="utf-8")
    assert "/models" not in text and ".models" not in text
