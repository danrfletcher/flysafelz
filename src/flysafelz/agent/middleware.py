"""The agent's one middleware: the input check, the system prompt, the model-failure answer, and the final `answer`.

    FlySafeMiddleware        the one `AgentMiddleware` subclass in the graph (graph.py passes it as the only item of `middleware`)
    unavailable_answer(messages)   the code-built answer for a turn whose model call failed (E17)
    is_model_failure(error)  whether an exception is the model endpoint failing (transport or server), as against a bug or a bad setting

Its hooks, in the order a turn meets them:

    before_agent      runs `validate_request` (input_check.py): a bad request raises, a request outside the UK is answered by code and jumps to the end
    wrap_model_call   gives the model this turn's system prompt (prompts.py, built from the run context's `now` and the thread's `tz`; it is
                      the system message of this call only and is never stored in `messages`), then makes the call. If the call fails,
                      or the model returns tool calls that cannot be parsed, it returns the code-built answer instead (E17)
    after_agent       sets `answer` to the final message's text when nothing set it already

The guard (PR-10, PR-11) goes into this same class: it checks the answer inside `wrap_model_call`, so a failed draft never reaches the thread's
state. This file holds no guard check, no retry beyond the model client's own (`model_max_retries`, FR17), no fallback after a failed check and
no re-check line.

E17. When the model endpoint times out or fails (after the client's own retry), or returns tool calls that cannot be parsed, nothing escapes to
the caller. The answer is built by code: if `assess` ran this turn, that tool's text exactly as it gave it (placeholders kept, for the rehydration
step to put the numbers back); if not, a fixed text that says the spot could not be checked, to treat it as UNVERIFIED, not to fly on this
assessment, and to use the NATS AIS pre-flight briefing. `guard` is set to `fallback`, because the answer is code's, not the model's. The failure is
logged as an error (its class name and text), and never as guard feedback: no `guard_fail` is written for it.

A setting that is missing or wrong is not a model failure. It is raised, so the person who set up the agent sees it: AgentConfigError, and the endpoint's
401, 403, 404 and other 4xx replies (a wrong API key or model name). Only timeouts, connection errors, 5xx replies and 429 give the unavailable answer.
"""

import logging
from collections.abc import Awaitable, Callable, Mapping, Sequence
from typing import Any

from langchain.agents.middleware import AgentMiddleware, ExtendedModelResponse, ModelRequest, ModelResponse
from langchain.agents.middleware.types import hook_config
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.runtime import Runtime
from langgraph.types import Command

from flysafelz.agent.guard import ASSESS_TOOL
from flysafelz.agent.guard_flow import GuardFlow, GuardOutcome, output_update
from flysafelz.agent.input_check import validate_request
from flysafelz.agent.prompts import system_prompt
from flysafelz.agent.state import DEFAULT_TZ, RequestState, RunContext, run_now

log = logging.getLogger(__name__)

HTTP_PACKAGES = frozenset({"httpx", "httpcore"})
HTTP_TRANSPORT_FAILURES = frozenset({"TimeoutException", "NetworkError"})  # httpx / httpcore base classes: timeouts, and connect/read/write/close errors
ENDPOINT_FAILURES = frozenset({"APIConnectionError", "RateLimitError", "InternalServerError"})  # openai: unreachable or timed out (APITimeoutError is one), 429, and 5xx
MESSAGE_LIMIT = 150  # characters of the exception text that go to the log
UNAVAILABLE_TEXT = (
    "The assistant is unavailable, so this spot could not be checked. Treat it as UNVERIFIED and do not fly on this assessment. "
    "Use the NATS AIS pre-flight briefing to check the NOTAMs, and check the weather and airspace by hand."
)

ModelHandler = Callable[[ModelRequest[RunContext]], ModelResponse[Any]]
AsyncModelHandler = Callable[[ModelRequest[RunContext]], Awaitable[ModelResponse[Any]]]


def class_names(error: BaseException, package: str | frozenset[str]) -> set[str]:
    """The names of the classes in `error`'s MRO that come from `package` (one top-level package name, or a set of them)."""
    packages = {package} if isinstance(package, str) else package
    return {cls.__name__ for cls in type(error).__mro__ if cls.__module__.split(".")[0] in packages}


def is_model_failure(error: BaseException) -> bool:
    """True when `error` is the model endpoint failing: a timeout, a connection error, a 5xx reply or a 429 (rate limit).

    A 400, 401, 403, 404 or 422 reply (a wrong key, a wrong model name, a bad request) is the setting or the code being wrong, not the endpoint
    failing, so it is not a model failure and is raised. Neither is AgentConfigError or a bug.

    openai is not imported here (only langchain_openai may be): the exception's classes are looked up by the package they come from,
    so a subclass counts.
    """
    if isinstance(error, TimeoutError | ConnectionError):
        return True
    if class_names(error, HTTP_PACKAGES) & HTTP_TRANSPORT_FAILURES:
        return True
    return bool(class_names(error, "openai") & ENDPOINT_FAILURES)


def failure_reason(error: BaseException) -> str:
    """The class name and the text of `error`, cut short, for the log line."""
    return f"{type(error).__name__}: {error}"[:MESSAGE_LIMIT]


def this_turn(messages: Sequence[BaseMessage]) -> list[BaseMessage]:
    """The messages after the pilot's newest one: what the model and the tools have said so far this turn."""
    for index in range(len(messages) - 1, -1, -1):
        if isinstance(messages[index], HumanMessage):
            return list(messages[index + 1 :])
    return list(messages)


def unavailable_answer(messages: Sequence[BaseMessage]) -> str:
    """The code-built answer for a failed model call: the newest `assess` text of this turn exactly as the tool gave it, else the fixed text."""
    ran = [message for message in this_turn(messages) if isinstance(message, ToolMessage) and message.name == ASSESS_TOOL]
    return ran[-1].text if ran else UNAVAILABLE_TEXT


def has_invalid_tool_calls(response: ModelResponse[Any]) -> bool:
    """True when the model's message holds a tool call whose arguments could not be parsed. Such a message is never acted on, even beside valid calls."""
    return any(isinstance(message, AIMessage) and message.invalid_tool_calls for message in response.result)


def unavailable_response(messages: Sequence[BaseMessage], reason: str, state: Mapping[str, Any] | None = None) -> ExtendedModelResponse[Any]:
    """The model response that ends the turn with the code-built answer. `reason` goes to the log.

    `answer` has the real phone numbers back (the AI message keeps its placeholders), `guard` is `fallback`, and `assessment` is this turn's `assess` result.
    """
    log.error("The model call failed (%s); the answer is built by code.", reason)
    text = unavailable_answer(messages)
    held = state or {}
    artifacts = [m.artifact for m in this_turn(messages) if isinstance(m, ToolMessage) and m.artifact is not None]
    update = output_update(text, "fallback", artifacts=artifacts, phone_map=held.get("phone_map"), overlays=held.get("overlays") or [])
    return ExtendedModelResponse(model_response=ModelResponse(result=[AIMessage(text)]), command=Command(update=update))


class FlySafeMiddleware(AgentMiddleware[RequestState, RunContext]):
    """The graph's one middleware. See the module doc for its hooks.

    `on_outcome`, when given, is called once for each final message the guard has dealt with, with the turn's `GuardOutcome` (PR-12.F1 sends it as feedback).
    """

    state_schema = RequestState

    def __init__(self, on_outcome: Callable[[GuardOutcome], None] | None = None) -> None:
        super().__init__()
        self.on_outcome = on_outcome

    @hook_config(can_jump_to=["end"])
    def before_agent(self, state: RequestState, runtime: Runtime[RunContext]) -> dict[str, Any] | None:
        return validate_request(state, runtime.context)

    def _with_prompt(self, request: ModelRequest[RunContext]) -> ModelRequest[RunContext]:
        """The request with this turn's system prompt: `now` from the run context, `tz` from the thread's LZ."""
        thread_lz = request.state.get("thread_lz")
        tz = thread_lz["tz"] if thread_lz else DEFAULT_TZ
        return request.override(system_message=SystemMessage(system_prompt(run_now(request.runtime.context), tz)))

    def wrap_model_call(self, request: ModelRequest[RunContext], handler: ModelHandler) -> ModelResponse[Any] | ExtendedModelResponse[Any]:
        request = self._with_prompt(request)
        messages = request.state["messages"]
        try:
            response = handler(request)
            if has_invalid_tool_calls(response):
                return unavailable_response(messages, "tool calls that could not be parsed", request.state)
            flow = GuardFlow(request, this_turn(messages), self.on_outcome)
            while (retry := flow.review(response)) is not None:
                response = handler(retry)
        except Exception as error:  # anything but a model failure is raised again on the next line
            if not is_model_failure(error):
                raise
            return unavailable_response(messages, failure_reason(error), request.state)
        return flow.done  # type: ignore[return-value]  # review() sets it whenever it returns None

    async def awrap_model_call(self, request: ModelRequest[RunContext], handler: AsyncModelHandler) -> ModelResponse[Any] | ExtendedModelResponse[Any]:
        request = self._with_prompt(request)
        messages = request.state["messages"]
        try:
            response = await handler(request)
            if has_invalid_tool_calls(response):
                return unavailable_response(messages, "tool calls that could not be parsed", request.state)
            flow = GuardFlow(request, this_turn(messages), self.on_outcome)
            while (retry := flow.review(response)) is not None:
                response = await handler(retry)
        except Exception as error:  # anything but a model failure is raised again on the next line
            if not is_model_failure(error):
                raise
            return unavailable_response(messages, failure_reason(error), request.state)
        return flow.done  # type: ignore[return-value]  # review() sets it whenever it returns None

    def after_agent(self, state: RequestState, runtime: Runtime[RunContext]) -> dict[str, Any] | None:
        """Set `answer` to the final message's text, unless something set it already (the UK-only reply, the model-failure answer)."""
        if state.get("answer"):
            return None
        messages = state["messages"]
        final = messages[-1] if messages else None
        if isinstance(final, AIMessage) and not final.tool_calls and final.text:
            return {"answer": final.text}
        return None
