"""G2: AGENT_* are read in agent/config.py and nowhere else, the model is built lazily, and the client retries once (FR17).

The settings of the config module itself (every missing one named, the key hidden, a bad URL) are in tests/unit/test_agent_config.py.
"""

import ast
import re
from pathlib import Path

import pytest
from langchain_openai import ChatOpenAI

from flysafelz.agent.config import AGENT_VARIABLES, AgentConfigError, LazyChatModel, agent_config, build_chat_model
from flysafelz.agent.settings import SETTINGS

AGENT = Path(__file__).resolve().parents[3] / "src" / "flysafelz" / "agent"
PROBE_SOCKET = pytest.mark.filterwarnings("ignore:A test tried to use socket")
VALUES = dict(zip(AGENT_VARIABLES, ("http://model.invalid/v1", "some/model", "dummy-" + "token"), strict=True))


def agent_files() -> list[Path]:
    return sorted(AGENT.rglob("*.py"))


def test_the_scan_sees_the_agent_package():
    names = {path.name for path in agent_files()}
    assert {"config.py", "graph.py", "middleware.py", "prompts.py", "run.py"} <= names


def test_no_file_of_agent_but_config_py_names_an_agent_variable():
    named = [path.relative_to(AGENT).as_posix() for path in agent_files() if re.search(r"AGENT_[A-Z]", path.read_text(encoding="utf-8"))]
    assert named == ["config.py"]


def test_no_file_of_agent_but_config_py_and_run_py_touches_the_environment():
    reading = [path.relative_to(AGENT).as_posix() for path in agent_files() if re.search(r"environ|getenv|dotenv", path.read_text(encoding="utf-8"))]
    assert reading == ["config.py", "run.py"]


def test_run_py_names_no_agent_variable_and_reads_its_own_six():
    text = (AGENT / "run.py").read_text(encoding="utf-8")
    assert not re.search(r"AGENT_[A-Z]", text)
    assert set(re.findall(r'"([A-Z]{1,4})"', text)) >= {"LAT", "LON", "Q", "TZ", "CASE", "NOW"}


def test_graph_py_builds_its_model_only_through_lazy_chat_model():
    tree = ast.parse((AGENT / "graph.py").read_text(encoding="utf-8"))
    names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)} | {alias.name for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) for alias in node.names}
    assert "LazyChatModel" in names
    assert not names & {"build_chat_model", "ChatOpenAI", "agent_config", "init_chat_model"}


def test_with_nothing_set_building_the_real_model_raises_one_line_naming_every_variable():
    with pytest.raises(AgentConfigError) as raised:
        build_chat_model()
    assert "\n" not in str(raised.value)
    assert str(raised.value).startswith("agent settings missing: AGENT_BASE_URL, AGENT_MODEL, AGENT_API_KEY")


@pytest.mark.parametrize("present", [0, 1, 2])
def test_the_one_line_names_only_the_missing_variables(monkeypatch, present):
    for name in AGENT_VARIABLES[:present]:
        monkeypatch.setenv(name, VALUES[name])
    with pytest.raises(AgentConfigError) as raised:
        agent_config()
    message = str(raised.value)
    assert all(name in message for name in AGENT_VARIABLES[present:])
    assert not any(name in message for name in AGENT_VARIABLES[:present])


@PROBE_SOCKET
def test_the_client_is_built_with_max_retries_1(monkeypatch):
    for name, value in VALUES.items():
        monkeypatch.setenv(name, value)
    client = build_chat_model()
    assert isinstance(client, ChatOpenAI)
    assert client.max_retries == 1 == SETTINGS.model_max_retries
    assert client.openai_api_base == VALUES["AGENT_BASE_URL"]


@PROBE_SOCKET
def test_the_lazy_model_reads_nothing_until_it_is_used(monkeypatch):
    lazy = LazyChatModel()  # no AGENT_* set: constructing it is fine
    assert lazy._real is None
    with pytest.raises(AgentConfigError):
        lazy.bind_tools([])
    assert lazy._real is None  # and a failed build leaves nothing behind
    for name, value in VALUES.items():
        monkeypatch.setenv(name, value)
    lazy.bind_tools([])
    assert isinstance(lazy._real, ChatOpenAI)
