"""Run the data tools the way the agent does: in a real `create_agent` graph with a scripted model, so `ToolRuntime`, the reducer of `phone_map` and the progress stream are the real ones.

    run_tools(source_cache, steps, ...) -> ToolRun

`steps` is what the scripted model asks for, one list of tool calls per model step (the calls of one step run together, as they do for a model that asks for
weather, airspace and NOTAMs at once). Nothing is faked but the model and what the caller puts in the SourceCache (its fetcher and its reader model).
"""

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from fakes import FakeToolChatModel
from langchain.agents import create_agent
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver

from flysafelz.agent.input_check import check_input
from flysafelz.agent.sources import SourceCache
from flysafelz.agent.state import RequestState, RunContext
from flysafelz.agent.tools.airspace_tool import build_get_airspace
from flysafelz.agent.tools.notams_tool import build_get_notams
from flysafelz.agent.tools.weather_tool import build_get_weather

BLUE_MILL_NOW = datetime(2026, 10, 31, 7, 30, tzinfo=UTC)
BLUE_MILL_LZ = {"lat": 53.802797, "lon": -0.086644}
BLUE_MILL_WINDOW = {"start": "2026-10-31 10:00", "end": "2026-10-31 16:00"}
ALL_THREE = ("get_weather", "get_airspace", "get_notams")


@dataclass
class ToolRun:
    """What a run did: every tool message (in the order the tools answered), the progress lines, and the thread's state afterwards."""

    messages: list[ToolMessage] = field(default_factory=list)
    progress: list[str] = field(default_factory=list)
    state: dict[str, Any] = field(default_factory=dict)

    def of(self, name: str) -> list[ToolMessage]:
        return [m for m in self.messages if m.name == name]

    def text(self, name: str, index: int = 0) -> str:
        return str(self.of(name)[index].content)

    def artifact(self, name: str, index: int = 0) -> Any:
        return self.of(name)[index].artifact


def calls(*names: str, **args: Any) -> list[tuple[str, dict[str, Any]]]:
    """One model step asking for each of `names` with the same arguments."""
    return [(name, dict(args)) for name in names]


def run_tools(
    source_cache: SourceCache,
    steps: list[list[tuple[str, dict[str, Any]]]],
    *,
    ctx: RunContext | None = None,
    lz: dict[str, float] | None = None,
    tz: str | None = "Europe/London",
    thread_id: str = "thread-1",
) -> ToolRun:
    """Send one turn through a graph whose model asks for `steps` in turn and then says "done."; return what the tools did."""
    ctx = ctx or RunContext(now=BLUE_MILL_NOW, case="blue_mill_2026-10-31")
    tools = {"get_weather": build_get_weather(source_cache), "get_airspace": build_get_airspace(source_cache), "get_notams": build_get_notams(source_cache)}
    script = [AIMessage("", tool_calls=[{"name": name, "args": args, "id": f"call-{n}-{i}"} for i, (name, args) in enumerate(step)]) for n, step in enumerate(steps)]
    model = FakeToolChatModel(script=[*script, AIMessage("done.")])
    graph: Any = create_agent(model=model, tools=list(tools.values()), middleware=[check_input], state_schema=RequestState, context_schema=RunContext, checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": thread_id}}
    given = {"messages": [HumanMessage("Can I fly here between 10 and 4 today?")], **(BLUE_MILL_LZ if lz is None else lz), "tz": tz}
    run, seen = ToolRun(), set()
    for mode, chunk in graph.stream(given, config, context=ctx, stream_mode=["custom", "values"]):
        if mode == "custom":
            run.progress.append(str(chunk))
            continue
        for message in chunk["messages"]:
            if isinstance(message, ToolMessage) and message.id not in seen:
                seen.add(message.id)
                run.messages.append(message)
    run.state = dict(graph.get_state(config).values)
    return run
