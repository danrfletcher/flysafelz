"""I5: the prompt asks for a `[path: …]` first line and adds no tool; the guard (PR-11.F1) strips the line, so it is in neither `messages` nor `answer`.

The strip itself is tested in test_guard_path_tag.py; here the prompt side, and that a good tagged answer reaches the end with only the tag gone.
"""

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
def test_a_final_message_with_the_tag_reaches_the_answer_and_the_messages_without_it(path):
    body = "I can only help with whether to fly from this spot."
    model = FakeToolChatModel(script=[AIMessage(f"[path: {path}]\n{body}")])
    state = ask(agent_graph(model), "Can I fly?")
    assert state["messages"][-1].content == body
    assert state["answer"] == body
    assert state["guard"] == "passed"


def test_the_tag_instruction_adds_no_tool_a_decline_makes_no_tool_call_and_an_assess_still_does():
    declined = FakeToolChatModel(script=[AIMessage("[path: decline]\nI can only help with whether to fly from this spot.")])
    state = ask(agent_graph(declined), "What is a good pub?")
    assert not [m for m in state["messages"] if m.type == "tool"]
    assert declined.bound_tool_names == ["get_weather", "get_airspace", "get_notams", "assess", "search_guidance"]

    assessed = FakeToolChatModel(script=[assess_call(), AIMessage("[path: assess]\nDone.")])
    state = ask(agent_graph(assessed))
    assert [m.name for m in state["messages"] if m.type == "tool"] == ["assess"]
    assert system_prompts(assessed)[0] == system_prompts(declined)[0]


def test_the_middleware_hands_a_tool_call_response_on_as_the_same_object():
    written = AIMessage("", tool_calls=assess_call().tool_calls)
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
    assert len(seen) == 1
