"""The real agent graph (graph.py's `build_agent`) over a scripted chat model, for the PR-9.F1 tests.

    agent_graph(model, ...)        -> graph    the real graph: the five tools, the one middleware, an in-memory checkpoint
    ScriptedModel(script=[...])    -> model    a chat model that plays its script in order; an exception in it is raised at that call
    ask(graph, question, ...)      -> state    one turn on Blue Mill (or the LZ given), with the case and the frozen clock as the run context
    call(name, **args)             -> AIMessage  a model message that asks for one tool

Only the model is faked (and, where a test says so, what the SourceCache fetches or its reader model). No tool, middleware or prompt is.
"""

from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from fakes import FakeToolChatModel, ScriptExhaustedError
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import Field, PrivateAttr
from tool_helpers import BLUE_MILL_LZ, BLUE_MILL_NOW, BLUE_MILL_WINDOW

from flysafelz.agent.graph import build_agent
from flysafelz.agent.sources import SourceCache
from flysafelz.agent.state import RunContext
from flysafelz.guidance.index import load_index

BLUE_MILL_CASE = "blue_mill_2026-10-31"
GP3_QUESTION = "Can I fly here between 10 and 4 today?"


class ScriptedModel(FakeToolChatModel):
    """FakeToolChatModel whose script may hold exceptions: the call that reaches one records its messages and raises it instead of answering."""

    script: list[Any] = Field(default_factory=list)
    _at: int = PrivateAttr(default=0)

    def _generate(self, messages: list[BaseMessage], stop: list[str] | None = None, run_manager: CallbackManagerForLLMRun | None = None, **kwargs: Any) -> ChatResult:
        self.calls.append(list(messages))
        if self._at >= len(self.script):
            raise ScriptExhaustedError(f"the model was called {self._at + 1} times but the test scripted {len(self.script)}")
        step = self.script[self._at]
        self._at += 1
        if isinstance(step, BaseException):
            raise step
        return ChatResult(generations=[ChatGeneration(message=step)])


class RaisingModel(BaseChatModel):
    """A chat model whose every call raises `error`: the endpoint is down. `count` is how many calls it got."""

    error: Any = None
    count: int = 0

    @property
    def _llm_type(self) -> str:
        return "raising-chat-model"

    def _generate(self, messages: list[BaseMessage], stop: list[str] | None = None, run_manager: CallbackManagerForLLMRun | None = None, **kwargs: Any) -> ChatResult:
        self.count += 1
        raise self.error

    def bind_tools(self, tools: Sequence[Any], **kwargs: Any) -> "RaisingModel":
        return self


def call(name: str, call_id: str = "call-1", **args: Any) -> AIMessage:
    """A model message that asks for the tool `name` with `args`."""
    return AIMessage("", tool_calls=[{"name": name, "args": args, "id": call_id}])


def assess_call(call_id: str = "call-1") -> AIMessage:
    """The GP3 call: assess 10:00 to 16:00 today."""
    return call("assess", call_id, **BLUE_MILL_WINDOW)


def agent_graph(model: BaseChatModel, *, source_cache: SourceCache | None = None, tmp_path: Path | None = None, checkpointer: Any = None):
    """The real graph over `model`. No guidance index is on disk in a test (`tmp_path` is empty), so `search_guidance` answers that it is unavailable."""
    index = load_index(tmp_path or Path("/nonexistent-guidance"))
    return build_agent(model, source_cache=source_cache or SourceCache(), index=index, checkpointer=InMemorySaver() if checkpointer is None else checkpointer)


def thread(thread_id: str = "thread-1") -> dict[str, Any]:
    return {"configurable": {"thread_id": thread_id}}


def blue_mill_context(now: datetime = BLUE_MILL_NOW, **more: Any) -> RunContext:
    return RunContext(now=now, case=BLUE_MILL_CASE, **more)


def ask(graph: Any, question: str = GP3_QUESTION, *, thread_id: str = "thread-1", context: RunContext | None = None, lz: dict[str, float] | None = None, **more: Any) -> dict[str, Any]:
    """One turn on the graph: the question, the LZ (Blue Mill unless `lz`), the run context (the Blue Mill case at the frozen clock unless `context`)."""
    given = {"messages": [HumanMessage(question)], **(BLUE_MILL_LZ if lz is None else lz), **more}
    return graph.invoke(given, thread(thread_id), context=context or blue_mill_context())


def system_prompts(model: FakeToolChatModel | ScriptedModel) -> list[str]:
    """The system message each model call was given, in call order."""
    return [str(next(m for m in messages if isinstance(m, SystemMessage)).content) for messages in model.calls]


def utc(*parts: int) -> datetime:
    return datetime(*parts, tzinfo=UTC)
