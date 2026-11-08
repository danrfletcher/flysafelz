"""PR-9.F1: the graph is one `create_agent` over the five tools and the one middleware, with the schemas of PR-3, and it builds with no model setting.

Nothing is mocked but the chat model. `create_agent` is wrapped by a recorder (it still runs) to see what the graph was built from.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest
from fakes import FakeToolChatModel
from graph_helpers import GP3_QUESTION, agent_graph, ask, assess_call
from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import AIMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph.state import CompiledStateGraph

from flysafelz.agent import graph as graph_module
from flysafelz.agent.config import AGENT_VARIABLES, AgentConfigError, LazyChatModel
from flysafelz.agent.graph import TOOL_NAMES, build_graph
from flysafelz.agent.middleware import FlySafeMiddleware
from flysafelz.agent.state import DEFAULT_TZ, RequestState, RunContext

ROOT = Path(__file__).resolve().parents[3]
FIVE = ["get_weather", "get_airspace", "get_notams", "assess", "search_guidance"]
PROBE_SOCKET = pytest.mark.filterwarnings("ignore:A test tried to use socket")


@pytest.fixture
def built(monkeypatch):
    """The arguments `create_agent` was called with, for each graph this test builds. The real create_agent still runs."""
    seen = []
    real = graph_module.create_agent

    def recording(*args, **kwargs):
        seen.append((args, kwargs))
        return real(*args, **kwargs)

    monkeypatch.setattr(graph_module, "create_agent", recording)
    return seen


@pytest.fixture(autouse=True)
def fresh_shared_parts(monkeypatch):
    """build_graph() shares one source store and one index per process; each test starts without them."""
    monkeypatch.setattr(graph_module, "_shared", {})


# --- G2: what the graph is made of ---------------------------------------------------------------------------------


def test_the_tools_are_the_five_in_order():
    assert list(TOOL_NAMES) == FIVE
    graph = build_graph(model=FakeToolChatModel())
    assert list(graph.nodes["tools"].bound.tools_by_name) == FIVE


def test_one_create_agent_call_with_the_five_tools_one_middleware_and_the_two_schemas(built):
    model = FakeToolChatModel()
    build_graph(model=model)
    [(args, kwargs)] = built
    assert args == (model,)
    assert [tool.name for tool in kwargs["tools"]] == FIVE
    assert len(kwargs["middleware"]) == 1
    assert isinstance(kwargs["middleware"][0], FlySafeMiddleware) and isinstance(kwargs["middleware"][0], AgentMiddleware)
    assert kwargs["state_schema"] is RequestState
    assert kwargs["context_schema"] is RunContext


def test_the_graph_has_the_models_and_tools_nodes_and_only_the_one_middlewares_hooks():
    graph = build_graph(model=FakeToolChatModel())
    assert isinstance(graph, CompiledStateGraph)
    assert set(graph.nodes) == {"__start__", "model", "tools", "FlySafeMiddleware.before_agent", "FlySafeMiddleware.after_agent"}


def test_an_injected_model_is_the_one_called_and_it_is_given_the_five_tools():
    model = FakeToolChatModel(script=[AIMessage("[path: decline]\nI can only help with whether to fly from this spot.")])
    state = ask(build_graph(model=model), "What is a good pub?")
    assert len(model.calls) == 1
    assert model.bound_tool_names == FIVE
    assert state["answer"].startswith("[path: decline]")


def test_a_server_configuration_in_the_model_place_means_no_model_was_given(built):
    """The LangGraph server calls a one-parameter factory with its run configuration."""
    build_graph({"configurable": {"thread_id": "t"}})
    [(args, _)] = built
    assert isinstance(args[0], LazyChatModel)


def test_the_source_store_and_the_index_are_made_once_and_shared_by_every_graph(monkeypatch):
    seen = []
    real = graph_module.build_agent

    def recording(model, *, source_cache, index, checkpointer=None):
        seen.append((source_cache, index))
        return real(model, source_cache=source_cache, index=index, checkpointer=checkpointer)

    monkeypatch.setattr(graph_module, "build_agent", recording)
    build_graph(model=FakeToolChatModel())
    build_graph(model=FakeToolChatModel())
    (first_sources, first_index), (second_sources, second_index) = seen
    assert first_sources is second_sources and first_index is second_index


# --- G2: the model comes from AGENT_*, lazily ----------------------------------------------------------------------------


def test_importing_the_graph_module_with_no_agent_setting_raises_nothing():
    clean = {name: value for name, value in os.environ.items() if not name.startswith("AGENT_")}
    done = subprocess.run([sys.executable, "-c", "import flysafelz.agent.graph"], env=clean, capture_output=True, text=True, cwd=ROOT, check=False)
    assert done.returncode == 0, done.stderr


def test_building_the_graph_with_no_agent_setting_raises_nothing():
    assert not [name for name in os.environ if name.startswith("AGENT_")]
    assert isinstance(build_graph(), CompiledStateGraph)


def test_a_turn_that_never_reaches_the_model_needs_no_agent_setting():
    state = ask(build_graph(), "Can I fly here?", lz={"lat": 48.85, "lon": 2.35})
    assert state["answer"] == "FlySafeLZ only covers the UK."


def test_the_first_model_call_without_the_settings_raises_one_line_naming_every_missing_variable():
    with pytest.raises(AgentConfigError) as raised:
        ask(build_graph(), GP3_QUESTION)
    message = str(raised.value)
    assert "\n" not in message
    assert all(name in message for name in AGENT_VARIABLES)


def test_a_missing_setting_is_raised_not_turned_into_the_unavailable_answer():
    """E17 covers the endpoint failing; a setting that is not there is the person who set the agent up to fix."""
    graph = build_graph()
    with pytest.raises(AgentConfigError):
        ask(graph, GP3_QUESTION)


def test_with_only_some_settings_the_error_names_the_ones_missing(monkeypatch):
    monkeypatch.setenv(AGENT_VARIABLES[0], "http://model.invalid/v1")
    with pytest.raises(AgentConfigError) as raised:
        ask(build_graph(), GP3_QUESTION)
    assert AGENT_VARIABLES[1] in str(raised.value) and AGENT_VARIABLES[2] in str(raised.value) and AGENT_VARIABLES[0] not in str(raised.value)


@PROBE_SOCKET
def test_with_the_settings_set_the_lazy_model_builds_the_client_once_with_one_retry(monkeypatch):
    monkeypatch.setenv(AGENT_VARIABLES[0], "http://model.invalid/v1")
    monkeypatch.setenv(AGENT_VARIABLES[1], "some/model")
    monkeypatch.setenv(AGENT_VARIABLES[2], "dummy-" + "token")
    lazy = LazyChatModel()
    assert lazy._real is None
    bound = lazy.bind_tools([])
    client = lazy._real
    assert isinstance(client, ChatOpenAI) and client.max_retries == 1
    assert bound.bound is client
    lazy.bind_tools([])
    assert lazy._real is client


@PROBE_SOCKET
def test_the_lazy_model_sends_each_call_to_the_built_client(monkeypatch):
    from langchain_core.outputs import ChatGeneration, ChatResult

    monkeypatch.setenv(AGENT_VARIABLES[0], "http://model.invalid/v1")
    monkeypatch.setenv(AGENT_VARIABLES[1], "some/model")
    monkeypatch.setenv(AGENT_VARIABLES[2], "dummy-" + "token")
    sent = []

    def generate(self, messages, stop=None, run_manager=None, **kwargs):
        sent.append(("sync", messages[0].content))
        return ChatResult(generations=[ChatGeneration(message=AIMessage("from the client"))])

    async def agenerate(self, messages, stop=None, run_manager=None, **kwargs):
        sent.append(("async", messages[0].content))
        return ChatResult(generations=[ChatGeneration(message=AIMessage("from the client"))])

    monkeypatch.setattr(ChatOpenAI, "_generate", generate)
    monkeypatch.setattr(ChatOpenAI, "_agenerate", agenerate)
    lazy = LazyChatModel()
    assert lazy.invoke([HumanMessage("one")]).content == "from the client"
    # An event loop opens a socket pair, which the suite blocks; the fake client awaits nothing, so one send finishes the coroutine.
    waiting = lazy._agenerate([HumanMessage("two")])
    with pytest.raises(StopIteration) as finished:
        waiting.send(None)
    assert finished.value.value.generations[0].message.content == "from the client"
    assert sent == [("sync", "one"), ("async", "two")]


# --- GP2: the schema Studio's form is made from ------------------------------------------------------------------------------


def test_the_input_schema_requires_lat_and_lon_and_defaults_tz_to_europe_london():
    schema = build_graph(model=FakeToolChatModel()).get_input_jsonschema()
    assert {"lat", "lon"} <= set(schema["required"]) and "tz" not in schema["required"]
    assert schema["properties"]["tz"]["default"] == DEFAULT_TZ == "Europe/London"
    assert "answer" not in schema["properties"] and "guard" not in schema["properties"]


def test_the_output_schema_has_the_answer_and_no_private_field():
    properties = build_graph(model=FakeToolChatModel()).get_output_jsonschema()["properties"]
    assert {"answer", "guard", "overlays", "assessment"} <= set(properties)
    assert not {"phone_map", "thread_lz", "lat", "lon", "tz"} & set(properties)


def test_a_follow_up_may_leave_lat_lon_and_tz_out_the_threads_are_used():
    model = FakeToolChatModel(script=[AIMessage("first."), AIMessage("second.")])
    graph = agent_graph(model)
    ask(graph, GP3_QUESTION)
    state = graph.invoke({"messages": [HumanMessage("And later?")]}, {"configurable": {"thread_id": "thread-1"}}, context=RunContext(case="blue_mill_2026-10-31"))
    assert state["answer"] == "second."


# --- the graph runs the tool it is asked for --------------------------------------------------------------------------------


def test_a_scripted_assess_call_reaches_the_real_tool_through_the_graph():
    model = FakeToolChatModel(script=[assess_call(), AIMessage("[path: assess]\nDone.")])
    state = ask(agent_graph(model))
    tool_messages = [m for m in state["messages"] if m.type == "tool"]
    assert [m.name for m in tool_messages] == ["assess"]
    assert tool_messages[0].artifact is not None and "\n" in tool_messages[0].text
