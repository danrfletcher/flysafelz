"""The contract graph the agent tests share: the input check alone in a real `create_agent`, a fake chat model, an in-memory checkpoint.

PR-9.F1 builds the real graph; this one has no tools, no guard and no prompt, so a test sees only what the contract does.
"""

from typing import Any

from fakes import FakeToolChatModel
from langchain.agents import create_agent
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver

from flysafelz.agent.input_check import check_input
from flysafelz.agent.state import RequestState, RunContext


def contract_graph(replies: int = 6) -> tuple[Any, FakeToolChatModel]:
    """(graph, model): the model answers "ok." to its first `replies` calls, and `len(model.calls)` counts the calls it got."""
    model = FakeToolChatModel(script=[AIMessage("ok.") for _ in range(replies)])
    graph = create_agent(model=model, tools=[], middleware=[check_input], state_schema=RequestState, context_schema=RunContext, checkpointer=InMemorySaver())
    return graph, model


def thread(thread_id: str) -> dict[str, Any]:
    return {"configurable": {"thread_id": thread_id}}
