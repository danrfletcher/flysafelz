"""Mocks for the v0 baseline's tests (5a PR-9.F1): a fake chat model that answers from its own prompt, and a LangSmith client that records feedback too.

    CannedV0Model(plan=None)     a chat model that supports `with_structured_output`: it reads the window and the NOTAM ids out of the prompt it is given and replies with a
                                 tool call that fills V0Output. `plan(prompt text) -> dict | Exception | None` overrides the reply for a test (None: the default reply)
    segments_for(window, ...)    the default reply's segments
    V0Client                     RunRecordingClient plus create_feedback and read_project, so a whole run's feedback is recorded
    reply_message(output, usage) the AIMessage of one reply, with or without usage
    FAKE_V0_ENV                  every AGENT_* and LANGSMITH_* variable set to an obviously fake value (no READER_*, no JUDGE_*: v0 needs neither)

Everything here is invented; nothing is sent and no key is read.
"""

import re
import uuid
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from eval_runner_helpers import RunRecordingClient
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field

FAKE_V0_ENV = {
    "AGENT_BASE_URL": "http://agent.invalid/v1",
    "AGENT_MODEL": "fake-agent-model",
    "AGENT_API_KEY": "fake-key",
    "LANGSMITH_API_KEY": "fake-key",
}
USAGE = {"input_tokens": 1200, "output_tokens": 80, "total_tokens": 1280}
WINDOW_LINE = re.compile(r"^Window to assess: (\S+) to (\S+)$", re.MULTILINE)
NOTAM_LINE = re.compile(r"^NOTAM (\S+)$", re.MULTILINE)


def segments_for(window: tuple[str, str], *, verdict: str = "PILOT_DISCRETION", approaching_limits: bool = False, conditions: Sequence[str] = ()) -> list[dict[str, Any]]:
    """One segment over the whole window."""
    return [{"start": window[0], "end": window[1], "verdict": verdict, "approaching_limits": approaching_limits, "conditions": list(conditions)}]


def default_reply(prompt: str, *, verdict: str = "PILOT_DISCRETION", action: str = "IGNORE") -> dict[str, Any]:
    """A reply that fits the prompt: the window as one segment, and `action` for every NOTAM the prompt lists."""
    found = WINDOW_LINE.search(prompt)
    assert found, "the prompt has no window line"
    ids = NOTAM_LINE.findall(prompt)
    return {"segments": segments_for((found[1], found[2]), verdict=verdict), "notam_actions": [{"notam_id": i, "action": action} for i in ids]}


def reply_message(output: Mapping[str, Any], usage: Mapping[str, int] | None = USAGE) -> AIMessage:
    """The model's answer as the structured-output tool call the parser reads, with `usage_metadata` when `usage` is given."""
    call = {"name": "V0Output", "args": dict(output), "id": f"call_{uuid.uuid4().hex[:8]}", "type": "tool_call"}
    return AIMessage(content="", tool_calls=[call], usage_metadata=dict(usage) if usage is not None else None)  # type: ignore[arg-type]


class CannedV0Model(BaseChatModel):
    """A chat model for `with_structured_output(V0Output, include_raw=True)`.

    Each call reads the prompt (system and user text), asks `plan(user text)` what to do, and answers with it: a dict is the reply's arguments, an Exception is raised, None means
    the default reply. `prompts` holds every call's messages, `configs_seen` is not kept (the callbacks are the runner's). `usage` False leaves `usage_metadata` off the reply.
    """

    plan: Any = Field(default=None, exclude=True)
    usage: bool = True
    prompts: list[list[BaseMessage]] = Field(default_factory=list)
    bound: list[str] = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "canned-v0-model"

    def _generate(self, messages: list[BaseMessage], stop: list[str] | None = None, run_manager: CallbackManagerForLLMRun | None = None, **kwargs: Any) -> ChatResult:
        self.prompts.append(list(messages))
        user = str(messages[-1].content)
        planned: Callable[[str], Any] | None = self.plan
        chosen = planned(user) if planned is not None else None
        if isinstance(chosen, BaseException):
            raise chosen
        output = default_reply(user) if chosen is None else chosen
        return ChatResult(generations=[ChatGeneration(message=reply_message(output, USAGE if self.usage else None))])

    def bind_tools(self, tools: Sequence[Any], *, tool_choice: str | None = None, **kwargs: Any) -> "CannedV0Model":
        self.bound.extend(getattr(tool, "__name__", str(tool)) for tool in tools)
        return self


class V0Client(RunRecordingClient):
    """The runner's recording client plus the two calls a v0 run adds: feedback on a run, and the project's id for it."""

    def __init__(self, fail_on: Mapping[str, int] | None = None) -> None:
        super().__init__(fail_on)
        self.feedback: list[tuple[Any, str, Any, Any, Mapping[str, Any]]] = []

    def read_project(self, *, project_name: str) -> Any:
        self._record("read_project", project_name=project_name)
        return type("Project", (), {"id": uuid.uuid5(uuid.NAMESPACE_URL, project_name)})()

    def create_feedback(self, run_id: Any, key: str, *, score: Any = None, comment: Any = None, **kwargs: Any) -> None:
        self._record("create_feedback", key=key)
        self.feedback.append((run_id, key, score, comment, kwargs))

    def feedback_keys(self, run_id: Any) -> list[str]:
        return [item[1] for item in self.feedback if item[0] == run_id]
