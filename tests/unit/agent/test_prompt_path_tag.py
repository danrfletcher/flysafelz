"""I5: the prompt asks for a `[path: …]` first line, adds no tool, and what the model writes reaches the end of the middleware untouched.

PR-11's guard strips the line. Until it does, the tag is in the model's message and in `answer` exactly as the model wrote it: nothing here removes,
adds or rewrites it, and the middleware's own pass-through of a good response is checked on the response object itself.
"""

import re

import pytest
from fakes import FakeToolChatModel
from graph_helpers import agent_graph, ask, assess_call, system_prompts, utc
from langchain.agents.middleware import ModelRequest, ModelResponse
from langchain_core.messages import AIMessage, HumanMessage

from flysafelz.agent.middleware import FlySafeMiddleware
from flysafelz.agent.prompts import PATH_TAGS, PATHS, system_prompt


def test_the_prompt_contains_the_tag_instruction_and_all_four_values():
    prompt = system_prompt(utc(2026, 10, 31, 7, 30), "Europe/London")
    assert "Start every final answer with exactly one first line" in prompt
    assert [f"[path: {path}]" for path in PATHS] == list(PATH_TAGS)
    assert all(tag in prompt for tag in PATH_TAGS)


@pytest.mark.parametrize("path", PATHS)
def test_a_final_message_with_the_tag_reaches_the_answer_and_the_messages_unchanged(path):
    written = f"[path: {path}]\nThe answer, with 3 numbers: 5000 m, 90 %, 4.7 mph."
    model = FakeToolChatModel(script=[AIMessage(written)])
    state = ask(agent_graph(model), "Can I fly?")
    assert state["messages"][-1].content == written
    assert state["answer"] == written


def test_the_tag_instruction_adds_no_tool_a_decline_makes_no_tool_call_and_an_assess_still_does():
    declined = FakeToolChatModel(script=[AIMessage("[path: decline]\nI can only help with whether to fly from this spot.")])
    state = ask(agent_graph(declined), "What is a good pub?")
    assert not [m for m in state["messages"] if m.type == "tool"]
    assert declined.bound_tool_names == ["get_weather", "get_airspace", "get_notams", "assess", "search_guidance"]

    assessed = FakeToolChatModel(script=[assess_call(), AIMessage("[path: assess]\nDone.")])
    state = ask(agent_graph(assessed))
    assert [m.name for m in state["messages"] if m.type == "tool"] == ["assess"]
    assert system_prompts(assessed)[0] == system_prompts(declined)[0]


def test_the_middleware_hands_a_good_response_on_as_the_same_object():
    written = AIMessage("[path: explain]\nA NOTAM is a notice.")
    response = ModelResponse(result=[written])
    seen = []

    def handler(request):
        seen.append(request)
        return response

    request = ModelRequest(
        model=FakeToolChatModel(),
        messages=[HumanMessage("What is a NOTAM?")],
        state={"messages": [HumanMessage("What is a NOTAM?")]},
        runtime=type("Runtime", (), {"context": None})(),
    )
    assert FlySafeMiddleware().wrap_model_call(request, handler) is response
    assert response.result[0].content == "[path: explain]\nA NOTAM is a notice."
    assert re.match(r"\[path: (assess|explain|clarify|decline)\]\n", response.result[0].content)
