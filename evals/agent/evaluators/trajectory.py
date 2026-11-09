"""Trajectory scores (5a PR-6.F1, G10, E9): which tools the run called, and how many steps it took.

    required_tools(example, answer)    every name in `reference.required_tools` is among the run's tool calls
    forbidden_tools(example, answer)   no name in this example's own `reference.forbidden_tools` is among them
    step_cap_ok(example, answer)       the run's steps are within the step cap
    step_cap()                         the cap: the graph's own recursion limit, read from ticket 4's graph (never written here)
    tool_calls_of(state)               the tool calls of a v1 run, from its graph state's messages (for the code that builds a ScoredAnswer)
    guard_of(state)                    the run's guard result, from its graph state

The run's tool calls are `answer.tool_calls`, one entry per call in order. All three scores are v1 scores: a v0 run (one structured call, no tools) gets None (E5),
and none depends on the path, since a compound example's second part is checked here (E9: dev_16 needs `get_notams` and `get_weather`; with either missing
required_tools fails and no other score changes).

required_tools. A superset match through agentevals (`create_trajectory_match_evaluator`, mode "superset", arguments ignored): the run's tool names must contain every
required name. Order and repeats do not matter, so a tool called twice counts once. None when the list is absent or empty. A run with no tool calls fails it.

forbidden_tools is read per example: only the names in that example's own list count. `dev_17_sherburn` (compound clarify + explain) lists four tools and leaves out
`get_airspace`, so calling it passes; `dev_18_blue_mill` lists all five, so the same call fails. None when the list is absent or empty. A run with no tool calls passes it.

Step cap (G10). A step is one tool call, and a tool called twice is two steps (required_tools counts it once). The run passes when its steps are at or under the
cap, so a run at the cap passes and one over fails. Ticket 4 merged with no step-cap setting in settings.yaml (the runner sets no recursion limit either), so the cap
that holds on its graph is the recursion limit `create_agent` puts on it. `step_cap()` reads that from the config of a graph built by `flysafelz.agent.graph.build_agent`
the first time and keeps it. That graph is built over an empty guidance index and a fresh source cache, so reading the cap loads no file (`build_graph` would load the
committed guidance index) and makes no call. If ticket 4 adds a cap of its own, this one function is where it is read.
"""

from collections.abc import Mapping, Sequence
from functools import cache
from typing import Any

from agentevals.trajectory.match import create_trajectory_match_evaluator

from evals.agent.evaluators.answer import V1, ScoredAnswer, ToolCall
from evals.agent.evaluators.labels import Feedback, example_id, feedback
from evals.agent.evaluators.mentions import reference_list
from flysafelz.agent.config import LazyChatModel
from flysafelz.agent.graph import build_agent
from flysafelz.agent.sources import SourceCache
from flysafelz.guidance.index import GuidanceIndex

REQUIRED_KEY = "required_tools"
FORBIDDEN_KEY = "forbidden_tools"
STEP_KEY = "step_cap"
SEARCH_GUIDANCE = "search_guidance"
_SUPERSET = create_trajectory_match_evaluator(trajectory_match_mode="superset", tool_args_match_mode="ignore")


@cache
def step_cap() -> int:
    """The cap on a run's steps: the recursion limit of ticket 4's graph (see the module text). Built once, over an empty index: no file read, no model call, no network."""
    graph = build_agent(LazyChatModel(), source_cache=SourceCache(), index=_no_index())
    return int(graph.config["recursion_limit"])  # type: ignore[index]  # create_agent sets it


def _no_index() -> GuidanceIndex:
    """A guidance index with nothing in it and no file behind it; the tools are built over it but never called."""

    def no_embedder() -> Any:
        raise AssertionError("the step cap's graph is never run")

    return GuidanceIndex([], key=None, stored_key=None, embedder_factory=no_embedder)


def _messages(names: Sequence[str]) -> list[dict[str, Any]]:
    """The tool names as the one assistant message agentevals reads; a run with no tool calls is a plain reply."""
    if not names:
        return [{"role": "assistant", "content": "done"}]
    return [{"role": "assistant", "content": "", "tool_calls": [{"function": {"name": name, "arguments": "{}"}} for name in names]}]


def required_tools(example: Mapping[str, Any], answer: ScoredAnswer) -> Feedback | None:
    required = reference_list(example, REQUIRED_KEY)
    if required is None or answer.kind != V1:
        return None
    called = [call.name for call in answer.tool_calls]
    wanted = list(dict.fromkeys(required))
    matched = bool(_SUPERSET(outputs=_messages(called), reference_outputs=_messages(wanted))["score"])  # type: ignore[index]  # a dict result
    if matched:
        return feedback(REQUIRED_KEY, 1, f"{example_id(example)}: every required tool was called ({', '.join(wanted)})")
    missing = [name for name in wanted if name not in called]
    return feedback(REQUIRED_KEY, 0, f"{example_id(example)}: required tool(s) not called: {', '.join(missing)}; called: {', '.join(dict.fromkeys(called)) or 'none'}")


def forbidden_tools(example: Mapping[str, Any], answer: ScoredAnswer) -> Feedback | None:
    forbidden = reference_list(example, FORBIDDEN_KEY)
    if forbidden is None or answer.kind != V1:
        return None
    used = [name for name in dict.fromkeys(call.name for call in answer.tool_calls) if name in forbidden]
    if used:
        return feedback(FORBIDDEN_KEY, 0, f"{example_id(example)}: forbidden tool(s) called: {', '.join(used)}")
    return feedback(FORBIDDEN_KEY, 1, f"{example_id(example)}: none of the {len(forbidden)} forbidden tools was called")


def step_cap_ok(example: Mapping[str, Any], answer: ScoredAnswer) -> Feedback | None:
    if answer.kind != V1:
        return None
    steps, cap = len(answer.tool_calls), step_cap()
    verdict = "within" if steps <= cap else "over"
    return feedback(STEP_KEY, 1 if steps <= cap else 0, f"{example_id(example)}: {steps} step(s), {verdict} the cap of {cap}")


# --- reading a v1 run's graph state ------------------------------------------------------------------------------------


def _field(item: Any, name: str) -> Any:
    return item.get(name) if isinstance(item, Mapping) else getattr(item, name, None)


def tool_calls_of(state: Mapping[str, Any]) -> tuple[ToolCall, ...]:
    """The tool calls of a run in the order the model made them, from the AI messages in its state. A `search_guidance` call carries the guidance sources its result
    (the matching tool message's artifact) returned; a call whose result is missing carries none."""
    messages = state.get("messages") or []
    artifacts = {_field(m, "tool_call_id"): _field(m, "artifact") for m in messages if _field(m, "type") == "tool"}
    found = []
    for message in messages:
        if _field(message, "type") != "ai":
            continue
        for call in _field(message, "tool_calls") or []:
            name = _field(call, "name")
            guides: tuple[str, ...] = ()
            if name == SEARCH_GUIDANCE:
                chunks = _field(artifacts.get(_field(call, "id")), "chunks") or []
                guides = tuple(dict.fromkeys(str(_field(chunk, "document")) for chunk in chunks if _field(chunk, "document")))
            found.append(ToolCall(str(name), guides))
    return tuple(found)


def guard_of(state: Mapping[str, Any]) -> str | None:
    """The run's guard result (`passed`, `retried` or `fallback`), or None when the state has none."""
    guard = state.get("guard")
    return guard if isinstance(guard, str) else None
