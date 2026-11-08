"""The agent graph: one LangChain `create_agent` that Studio, the eval harness and a terminal all talk to.

    build_graph(model=None)      the graph the LangGraph server loads (langgraph.json points here); a factory, so nothing is built at import
    build_agent(model, ...)      the same graph over parts the caller chooses (tests pass a source store, a guidance index and a checkpointer)
    TOOL_NAMES                   the five tools, in the order they are registered

The graph is `create_agent(model, tools=[the five], middleware=[FlySafeMiddleware()], state_schema=RequestState, context_schema=RunContext)`.
Nothing else: no custom node, no sub-agent, no second middleware. The five tools are get_weather, get_airspace, get_notams, assess and
search_guidance. The thread state is state.py's `RequestState` and the run context is its `RunContext`. The one middleware (middleware.py) runs
the input check, gives the model its system prompt (prompts.py) and answers for the model when it fails. The guard (PR-10, PR-11) and tracing
(PR-12) are not here.

The model. `build_graph()` with no model uses a `LazyChatModel` (config.py): the configured ChatOpenAI, built when the first model call needs it.
So importing this module, building the graph, and a turn that never reaches the model all work with no AGENT_* setting; the first model call
without them raises AgentConfigError, one line naming each missing setting. A test passes its own model: `build_graph(model=fake)`.

The LangGraph server calls a one-parameter factory with its run configuration, which lands in `model`. A configuration is a mapping, never a
chat model, so a mapping there is read as "no model given".

The server calls the factory again for each run, but the threads' fetched sources and the guidance index have to outlive one call (the sources
are reused for fifteen minutes within a thread, and the index reads its file once). They are therefore made once per process, the first time
`build_graph` needs them, and every graph that `build_graph` returns shares them.
"""

import threading
from collections.abc import Mapping
from typing import Any

from langchain.agents import create_agent
from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Checkpointer

from flysafelz.agent.config import LazyChatModel
from flysafelz.agent.middleware import FlySafeMiddleware
from flysafelz.agent.sources import SourceCache
from flysafelz.agent.state import RequestState, RunContext
from flysafelz.agent.tools.airspace_tool import build_get_airspace
from flysafelz.agent.tools.assess_tool import build_assess
from flysafelz.agent.tools.guidance_tool import build_search_guidance
from flysafelz.agent.tools.notams_tool import build_get_notams
from flysafelz.agent.tools.weather_tool import build_get_weather
from flysafelz.guidance.index import GuidanceIndex, load_index

Agent = CompiledStateGraph[Any, RunContext, Any, Any]  # what `create_agent` returns
TOOL_NAMES = ("get_weather", "get_airspace", "get_notams", "assess", "search_guidance")

_shared: dict[str, Any] = {}
_shared_lock = threading.Lock()


def shared_source_cache() -> SourceCache:
    """The process's one SourceCache, made on first use."""
    with _shared_lock:
        if "sources" not in _shared:
            _shared["sources"] = SourceCache()
        return _shared["sources"]


def shared_guidance_index() -> GuidanceIndex:
    """The process's one guidance index, loaded on first use. Loading reads the committed file; the embedding model loads at the first search."""
    with _shared_lock:
        if "index" not in _shared:
            _shared["index"] = load_index()
        return _shared["index"]


def build_agent(model: BaseChatModel, *, source_cache: SourceCache, index: GuidanceIndex, checkpointer: Checkpointer | None = None) -> Agent:
    """The agent over these parts: the one `create_agent` call in the code base."""
    tools = [
        build_get_weather(source_cache),
        build_get_airspace(source_cache),
        build_get_notams(source_cache),
        build_assess(source_cache),
        build_search_guidance(index),
    ]
    return create_agent(model, tools=tools, middleware=[FlySafeMiddleware()], state_schema=RequestState, context_schema=RunContext, checkpointer=checkpointer)


def build_graph(model: BaseChatModel | Mapping[str, Any] | None = None) -> Agent:
    """The agent graph. `model` is a chat model to use (tests); left out, the configured one is built at the first model call."""
    if model is None or isinstance(model, Mapping):
        model = LazyChatModel()
    return build_agent(model, source_cache=shared_source_cache(), index=shared_guidance_index())
