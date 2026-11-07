"""The two test doubles in tests/fakes.py, run with sockets off (the project's addopts)."""

import numpy as np
import pytest
from fakes import EMBEDDING_DIMENSION, FakeEmbedder, FakeToolChatModel, ScriptExhaustedError
from langchain.agents import create_agent
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool


@tool
def count_notams(radius_nm: int) -> str:
    """Count the NOTAMs within a radius."""
    return f"2 NOTAMs within {radius_nm} NM"


TOOL_CALL = {"name": "count_notams", "args": {"radius_nm": 2}, "id": "call_1", "type": "tool_call"}


def test_a_scripted_tool_call_round_trip_by_hand():
    model = FakeToolChatModel(script=[AIMessage(content="", tool_calls=[TOOL_CALL]), AIMessage(content="There are two NOTAMs nearby.")])
    bound = model.bind_tools([count_notams])

    first = bound.invoke([HumanMessage("Anything near the LZ?")])
    assert first.tool_calls == [{**TOOL_CALL, "type": "tool_call"}] and first.content == ""

    result = count_notams.invoke(first.tool_calls[0])
    assert isinstance(result, ToolMessage) and result.content == "2 NOTAMs within 2 NM"

    second = bound.invoke([HumanMessage("Anything near the LZ?"), first, result])
    assert second.content == "There are two NOTAMs nearby." and not second.tool_calls
    assert model.bound_tool_names == ["count_notams"]
    assert [len(call) for call in model.calls] == [1, 3]  # what the code under test sent each time is kept
    assert model.calls[1][2] is result


def test_langchains_create_agent_drives_the_fake_through_a_tool_call():
    """Only to show the fake is a model create_agent accepts; running the real agent on it is a later PR's test."""
    model = FakeToolChatModel(script=[AIMessage(content="", tool_calls=[TOOL_CALL]), AIMessage(content="Two NOTAMs within 2 NM.")])
    agent = create_agent(model, tools=[count_notams])

    state = agent.invoke({"messages": [HumanMessage("Anything near the LZ?")]})

    assert [type(m).__name__ for m in state["messages"]] == ["HumanMessage", "AIMessage", "ToolMessage", "AIMessage"]
    assert state["messages"][2].content == "2 NOTAMs within 2 NM"
    assert state["messages"][-1].content == "Two NOTAMs within 2 NM."
    assert set(model.bound_tool_names) == {"count_notams"}  # create_agent binds the tools on every model call


def test_a_call_after_the_script_is_used_up_fails_loudly():
    model = FakeToolChatModel(script=[AIMessage(content="only reply")])
    assert model.invoke("hi").content == "only reply"
    with pytest.raises(ScriptExhaustedError):
        model.invoke("hi again")


def test_bind_tools_accepts_a_tool_a_dict_and_a_function():
    def plain_function(): ...

    model = FakeToolChatModel(script=[])
    assert model.bind_tools([count_notams, {"name": "from_a_dict"}, plain_function]) is model
    assert model.bound_tool_names == ["count_notams", "from_a_dict", "plain_function"]


def test_the_embedder_is_deterministic_with_the_real_models_dimension():
    embedder = FakeEmbedder()
    first = list(embedder.embed(["danger area", "NOTAM"]))
    second = list(embedder.embed(["danger area", "NOTAM"]))

    assert len(first) == 2
    for a, b in zip(first, second, strict=True):
        assert a.shape == (EMBEDDING_DIMENSION,) and a.dtype == np.float32
        assert np.array_equal(a, b)
    assert not np.array_equal(first[0], first[1])
    assert np.linalg.norm(first[0]) == pytest.approx(1.0, abs=1e-6)
    assert EMBEDDING_DIMENSION == FakeEmbedder.dimension == 384


def test_the_embedder_gives_a_string_one_vector_and_a_query_the_same_vector_as_a_document():
    embedder = FakeEmbedder()
    [single] = list(embedder.embed("danger area"))
    [from_list] = list(embedder.embed(["danger area"]))
    [query] = list(embedder.query_embed(["danger area"]))
    assert np.array_equal(single, from_list) and np.array_equal(single, query)
    assert list(embedder.embed([])) == []
