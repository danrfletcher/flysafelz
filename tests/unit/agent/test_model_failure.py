"""E17: the model endpoint failing, or answering with tool calls that cannot be read, ends the turn with an answer built by code.

The real graph, the real tools (Blue Mill case) and the real middleware; only the chat model is scripted. The client's own retry (`max_retries=1`)
is the client's: a fake that raises is the endpoint still failing after it.
"""

import logging
import re
from types import SimpleNamespace

import httpx
import openai
import pytest
from fakes import FakeToolChatModel
from graph_helpers import GP3_QUESTION, RaisingModel, ScriptedModel, agent_graph, ask, assess_call, blue_mill_context
from langchain.agents.middleware import ModelRequest, ModelResponse
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from flysafelz.agent.config import AgentConfigError
from flysafelz.agent.middleware import UNAVAILABLE_TEXT, FlySafeMiddleware, is_model_failure, unavailable_answer

LOGGER = "flysafelz.agent.middleware"
REQUEST = httpx.Request("POST", "http://model.invalid/v1/chat/completions")
TIMEOUTS = [
    pytest.param(httpx.ReadTimeout("timed out"), id="httpx.TimeoutException"),
    pytest.param(openai.APITimeoutError(request=REQUEST), id="openai.APITimeoutError"),
    pytest.param(openai.APIConnectionError(request=REQUEST), id="openai.APIConnectionError"),
    pytest.param(httpx.ConnectError("refused"), id="httpx.ConnectError"),
    pytest.param(TimeoutError("timed out"), id="TimeoutError"),
]


def status_error(cls: type[openai.APIStatusError], status: int) -> openai.APIStatusError:
    """The error the openai client raises for a reply with this status."""
    return cls(f"Error code: {status}", response=httpx.Response(status, request=REQUEST), body=None)


SERVER_FAILURES = [
    pytest.param(status_error(openai.InternalServerError, 500), id="500"),
    pytest.param(status_error(openai.InternalServerError, 503), id="503"),
    pytest.param(status_error(openai.RateLimitError, 429), id="429"),
]
WRONG_SETTINGS = [
    pytest.param(status_error(openai.AuthenticationError, 401), id="401-wrong-key"),
    pytest.param(status_error(openai.PermissionDeniedError, 403), id="403"),
    pytest.param(status_error(openai.NotFoundError, 404), id="404-wrong-model"),
    pytest.param(status_error(openai.BadRequestError, 400), id="400"),
    pytest.param(status_error(openai.UnprocessableEntityError, 422), id="422"),
]
INVALID_CALL = {"name": "assess", "args": '{"start": "2026-10-31 10:00", "end": ', "id": "bad-1", "error": "bad JSON", "type": "invalid_tool_call"}


def unparseable() -> AIMessage:
    """What a model returns when its tool call is not valid JSON: only `invalid_tool_calls`."""
    return AIMessage("", invalid_tool_calls=[INVALID_CALL])


def tool_messages(state) -> list[ToolMessage]:
    return [m for m in state["messages"] if isinstance(m, ToolMessage)]


def assert_unavailable_fixed_text(text: str) -> None:
    assert text == UNAVAILABLE_TEXT
    for need in ("could not be checked", "UNVERIFIED", "do not fly on this assessment", "NATS AIS pre-flight briefing"):
        assert need in text


# --- the model fails with nothing run ------------------------------------------------------------------------------------------


@pytest.mark.parametrize("error", TIMEOUTS)
def test_a_failing_model_ends_the_run_with_the_fixed_unavailable_answer_and_no_exception(error, caplog):
    model = RaisingModel(error=error)
    with caplog.at_level(logging.ERROR, logger=LOGGER):
        state = ask(agent_graph(model))
    assert_unavailable_fixed_text(state["answer"])
    assert state["messages"][-1].content == state["answer"]
    assert state["guard"] == "fallback"
    assert tool_messages(state) == []
    assert model.count == 1


def test_the_failure_is_an_error_on_the_log_and_no_guard_feedback_is_written(caplog):
    with caplog.at_level(logging.ERROR, logger=LOGGER):
        state = ask(agent_graph(RaisingModel(error=httpx.ReadTimeout("timed out"))))
    [record] = [r for r in caplog.records if r.name == LOGGER]
    assert record.levelno == logging.ERROR and "ReadTimeout" in record.getMessage()
    assert "guard_fail" not in state
    assert not any("guard_fail" in str(m.content) for m in state["messages"])


def test_the_unavailable_answer_names_no_verdict_and_no_banned_phrase():
    text = UNAVAILABLE_TEXT.lower()
    for phrase in ("safe to fly", "clear to fly", "good to go", "all clear"):
        assert phrase not in text
    assert not re.search(r"\bGO\b", UNAVAILABLE_TEXT)


# --- the model fails after assess ran ------------------------------------------------------------------------------------------


@pytest.mark.parametrize("error", TIMEOUTS)
def test_when_assess_ran_this_turn_the_answer_is_its_text_verbatim(error):
    model = ScriptedModel(script=[assess_call(), error])
    state = ask(agent_graph(model))
    [assessed] = tool_messages(state)
    assert assessed.name == "assess" and assessed.text
    assert state["answer"] == assessed.text
    assert state["messages"][-1].content == assessed.text
    assert state["guard"] == "fallback"
    assert len(model.calls) == 2


def test_the_assess_text_keeps_its_placeholders_for_the_step_that_puts_the_numbers_back():
    state = ask(agent_graph(ScriptedModel(script=[assess_call(), httpx.ReadTimeout("timed out")])))
    [assessed] = tool_messages(state)
    assert state["answer"] == assessed.text
    assert "<PHONE_1>" in state["answer"]  # Blue Mill's contact number is still the placeholder: PR-11's rehydration puts the number back


def test_the_newest_assess_of_the_turn_is_the_one_used():
    model = ScriptedModel(script=[assess_call("call-1"), assess_call("call-2"), TimeoutError("timed out")])
    state = ask(agent_graph(model))
    _, second = tool_messages(state)
    assert state["answer"] == second.text


def test_an_assess_from_an_earlier_turn_does_not_count():
    """Turn 1 ran assess and answered. Turn 2's model fails before any tool runs: the answer is the fixed text, not the old table."""
    model = ScriptedModel(script=[assess_call(), AIMessage("[path: assess]\nDone."), httpx.ReadTimeout("timed out")])
    graph = agent_graph(model)
    ask(graph, GP3_QUESTION)
    state = ask(graph, "And what about the NOTAMs?")
    assert_unavailable_fixed_text(state["answer"])


# --- tool calls that cannot be read ----------------------------------------------------------------------------------------------


def test_unparseable_tool_calls_run_no_tool_and_give_the_fixed_answer(caplog):
    model = ScriptedModel(script=[unparseable()])
    with caplog.at_level(logging.ERROR, logger=LOGGER):
        state = ask(agent_graph(model))
    assert_unavailable_fixed_text(state["answer"])
    assert tool_messages(state) == [] and state["guard"] == "fallback"
    assert any("could not be parsed" in r.getMessage() for r in caplog.records if r.name == LOGGER)
    assert len(model.calls) == 1  # the model is not asked again


def test_unparseable_tool_calls_after_assess_give_the_assess_text_verbatim():
    model = ScriptedModel(script=[assess_call(), unparseable()])
    state = ask(agent_graph(model))
    [assessed] = tool_messages(state)
    assert state["answer"] == assessed.text and state["guard"] == "fallback"


def test_a_message_with_one_good_call_and_one_unreadable_call_runs_neither():
    both = AIMessage("", tool_calls=assess_call().tool_calls, invalid_tool_calls=[INVALID_CALL])
    state = ask(agent_graph(ScriptedModel(script=[both])))
    assert tool_messages(state) == []
    assert_unavailable_fixed_text(state["answer"])


# --- what is not a model failure ----------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("error", TIMEOUTS + SERVER_FAILURES)
def test_is_model_failure_says_yes_to_the_endpoint_failing(error):
    assert is_model_failure(error)


@pytest.mark.parametrize("error", [KeyError("bug"), ValueError("bad"), AgentConfigError(("AGENT_MODEL",)), RuntimeError("x"), AssertionError("x")])
def test_is_model_failure_says_no_to_a_bug_or_a_bad_setting(error):
    assert not is_model_failure(error)


@pytest.mark.parametrize("error", WRONG_SETTINGS)
def test_is_model_failure_says_no_to_a_reply_that_means_a_wrong_key_or_model(error):
    assert not is_model_failure(error)


@pytest.mark.parametrize("error", WRONG_SETTINGS)
def test_a_wrong_key_or_model_reply_is_raised_through_the_graph_not_answered_as_unavailable(error):
    with pytest.raises(type(error)):
        ask(agent_graph(ScriptedModel(script=[error])))


@pytest.mark.parametrize("error", SERVER_FAILURES)
def test_a_5xx_or_rate_limit_reply_gives_the_unavailable_answer_and_the_log_names_the_error_text(error, caplog):
    with caplog.at_level(logging.ERROR, logger=LOGGER):
        state = ask(agent_graph(RaisingModel(error=error)))
    assert_unavailable_fixed_text(state["answer"])
    assert state["guard"] == "fallback"
    assert any(type(error).__name__ in r.getMessage() and f"Error code: {error.status_code}" in r.getMessage() for r in caplog.records if r.name == LOGGER)


def test_a_wrong_key_reply_raised_in_the_async_hook_reaches_the_caller():
    async def refusing(request):
        raise status_error(openai.AuthenticationError, 401)

    with pytest.raises(openai.AuthenticationError):
        finish(FlySafeMiddleware().awrap_model_call(request_for(), refusing))


def test_a_bug_in_the_model_call_is_raised_not_hidden_behind_the_unavailable_answer():
    with pytest.raises(KeyError, match="bug"):
        ask(agent_graph(ScriptedModel(script=[KeyError("bug")])))


def test_a_missing_agent_setting_raised_inside_the_model_call_reaches_the_caller():
    with pytest.raises(AgentConfigError):
        ask(agent_graph(ScriptedModel(script=[AgentConfigError(("AGENT_MODEL",))])))


def test_a_good_answer_after_a_good_run_is_untouched():
    model = FakeToolChatModel(script=[assess_call(), AIMessage("[path: assess]\nNO-GO from 10:00.")])
    state = ask(agent_graph(model))
    assert state["answer"] == "[path: assess]\nNO-GO from 10:00."
    assert state["guard"] is None  # the guard (PR-10, PR-11) is what sets it on a good answer


# --- the unavailable answer itself ----------------------------------------------------------------------------------------------------


def test_unavailable_answer_reads_only_the_messages_after_the_pilots_newest_question():
    old = ToolMessage("OLD TABLE", name="assess", tool_call_id="a")
    new = ToolMessage("NEW TABLE", name="assess", tool_call_id="b")
    other = ToolMessage("WEATHER", name="get_weather", tool_call_id="c")
    assert unavailable_answer([HumanMessage("q1"), old, AIMessage("a1"), HumanMessage("q2")]) == UNAVAILABLE_TEXT
    assert unavailable_answer([HumanMessage("q1"), old, AIMessage("a1"), HumanMessage("q2"), other]) == UNAVAILABLE_TEXT
    assert unavailable_answer([HumanMessage("q1"), old, AIMessage("a1"), HumanMessage("q2"), new, other]) == "NEW TABLE"


# --- the same in the async hook --------------------------------------------------------------------------------------------------------


def request_for(question: str = GP3_QUESTION) -> ModelRequest:
    runtime = SimpleNamespace(context=blue_mill_context())
    return ModelRequest(model=FakeToolChatModel(), messages=[HumanMessage(question)], state={"messages": [HumanMessage(question)]}, runtime=runtime)


def finish(coroutine):
    """Run a coroutine that awaits nothing to its end without an event loop (a loop opens a socket pair, which the suite blocks)."""
    with pytest.raises(StopIteration) as done:
        coroutine.send(None)
    return done.value.value


def test_the_async_hook_gives_the_same_unavailable_answer_for_a_failing_handler():
    async def failing(request):
        raise httpx.ReadTimeout("timed out")

    result = finish(FlySafeMiddleware().awrap_model_call(request_for(), failing))
    assert result.model_response.result[0].content == UNAVAILABLE_TEXT
    assert result.command.update == {"answer": UNAVAILABLE_TEXT, "guard": "fallback"}


def test_the_async_hook_gives_the_prompt_to_a_good_handler_and_returns_its_response_unchanged():
    seen = []
    response = ModelResponse(result=[AIMessage("fine.")])

    async def good(request):
        seen.append(request.system_message.text)
        return response

    assert finish(FlySafeMiddleware().awrap_model_call(request_for(), good)) is response
    assert seen[0].startswith("You are the FlySafeLZ assistant")


def test_the_async_hook_hides_unreadable_tool_calls_and_raises_a_bug():
    async def unreadable(request):
        return ModelResponse(result=[unparseable()])

    async def buggy(request):
        raise KeyError("bug")

    assert finish(FlySafeMiddleware().awrap_model_call(request_for(), unreadable)).model_response.result[0].content == UNAVAILABLE_TEXT
    with pytest.raises(KeyError):
        finish(FlySafeMiddleware().awrap_model_call(request_for(), buggy))
