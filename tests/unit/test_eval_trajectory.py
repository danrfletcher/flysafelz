"""Trajectory scores (5a PR-6.F1, G10, E9): required tools (superset), forbidden tools (per example), the step cap, and reading a real graph state."""

from dataclasses import replace
from types import SimpleNamespace

import pytest
from agent_eval.factories import answer, example
from eval_form_helpers import DOCUMENT, dev, passing_answer
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from evals.agent.evaluators import trajectory
from evals.agent.evaluators.aggregate import EVALUATORS, ScoredRun, score_run
from evals.agent.evaluators.answer import ToolCall
from evals.agent.evaluators.trajectory import FORBIDDEN_KEY, REQUIRED_KEY, STEP_KEY, forbidden_tools, guard_of, required_tools, step_cap, step_cap_ok, tool_calls_of
from flysafelz.agent import graph as graph_module
from flysafelz.agent.graph import build_graph
from flysafelz.agent.tools.artifacts import GuidanceArtifact, GuidanceChunk


def real_cap():
    """The recursion limit of the whole graph, built as the app builds it (this loads the committed guidance index)."""
    return build_graph().config["recursion_limit"]  # type: ignore[typeddict-item,index]


def calls(*names, **kwargs):
    return answer(path="explain", tool_calls=list(names), **kwargs)


@pytest.fixture(autouse=True)
def fresh_cap():
    step_cap.cache_clear()
    yield
    step_cap.cache_clear()


# --- required tools: a superset match -------------------------------------------------------------------------------------------


def test_every_required_tool_called_passes():
    ex = example(path="explain", extra={"required_tools": ["get_notams", "get_weather"]})
    got = required_tools(ex, calls("get_weather", "get_notams"))
    assert got and got["key"] == REQUIRED_KEY and got["score"] == 1


def test_extra_tools_and_a_different_order_still_pass():
    ex = example(path="explain", extra={"required_tools": ["get_notams", "get_weather"]})
    assert required_tools(ex, calls("get_airspace", "get_weather", "assess", "get_notams"))["score"] == 1  # type: ignore[index]


def test_a_missing_tool_fails_and_is_named():
    ex = example(path="explain", extra={"required_tools": ["get_notams", "get_weather"]})
    got = required_tools(ex, calls("get_notams"))
    assert got and got["score"] == 0 and "get_weather" in got["comment"].split("not called:")[1].split(";")[0] and "get_notams" not in got["comment"].split(";")[0].split("not called:")[1]


def test_a_run_with_no_tool_calls_fails_required_and_passes_forbidden():
    ex = example(path="explain", extra={"required_tools": ["get_notams"], "forbidden_tools": ["get_airspace", "assess"]})
    none_called = calls()
    assert required_tools(ex, none_called)["score"] == 0  # type: ignore[index]
    assert "called: none" in required_tools(ex, none_called)["comment"]  # type: ignore[index]
    assert forbidden_tools(ex, none_called)["score"] == 1  # type: ignore[index]


def test_duplicate_calls_count_once_for_required_and_once_per_step_for_the_cap():
    ex = example(path="explain", extra={"required_tools": ["get_notams", "get_notams", "get_weather"]})
    twice = calls("get_notams", "get_notams", "get_weather")
    assert required_tools(ex, twice)["score"] == 1  # type: ignore[index]
    assert required_tools(ex, calls("get_notams", "get_notams"))["score"] == 0  # type: ignore[index]  # two calls to one tool are not a call to get_weather
    assert "3 step(s)" in step_cap_ok(ex, twice)["comment"]  # type: ignore[index]


def test_the_tool_arguments_do_not_matter():
    ex = example(path="explain", extra={"required_tools": ["get_notams"]})
    assert required_tools(ex, calls(ToolCall("get_notams")))["score"] == 1  # type: ignore[index]


@pytest.mark.parametrize("value", [None, []])
def test_required_is_none_when_the_list_is_absent(value):
    ex = example(path="explain", extra={} if value is None else {"required_tools": value})
    assert required_tools(ex, calls("get_notams")) is None


# --- E9 compound: dev_16 -------------------------------------------------------------------------------------------------------


def test_e9_dev_16_needs_both_get_notams_and_get_weather():
    ex = dev("dev_16_buttermere")
    assert ex["metadata"]["parts"] == ["explain", "explain"] and set(ex["reference"]["required_tools"]) == {"get_notams", "get_weather"}
    assert required_tools(ex, calls("get_notams", "get_weather"))["score"] == 1  # type: ignore[index]
    assert required_tools(ex, calls("get_notams"))["score"] == 0  # type: ignore[index]
    assert required_tools(ex, calls("get_weather"))["score"] == 0  # type: ignore[index]


def other_scores(feedback):
    """Every registered key but required_tools -> its score (None when it does not apply). Comments may differ (step_cap's step count), scores may not."""
    return {key: None if found is None else found["score"] for key, found in feedback.items() if key != REQUIRED_KEY}


@pytest.mark.parametrize("missing", ["get_notams", "get_weather"])
def test_e9_with_either_tool_missing_only_required_tools_changes(missing):
    """The full passing dev_16 run, then the same run without one tool: every registered score but required_tools is unchanged."""
    ex = dev("dev_16_buttermere")
    full = passing_answer(ex)
    assert {c.name for c in full.tool_calls} >= {"get_notams", "get_weather"}
    without = replace(full, tool_calls=tuple(c for c in full.tool_calls if c.name != missing))
    assert len(without.tool_calls) == len(full.tool_calls) - 1
    before, after = score_run(ScoredRun(ex, full)).feedback, score_run(ScoredRun(ex, without)).feedback
    assert before.keys() == after.keys() == EVALUATORS.keys()
    assert before[REQUIRED_KEY]["score"] == 1 and after[REQUIRED_KEY]["score"] == 0  # type: ignore[index]
    assert missing in after[REQUIRED_KEY]["comment"]  # type: ignore[index]
    assert other_scores(before) == other_scores(after)
    assert before["cited"] is not None and before["must_mention"] is not None and before["guard_pass"] is not None  # the comparison covers real scores, not just Nones


# --- forbidden tools: per example ----------------------------------------------------------------------------------------------


def test_dev_17_lists_four_tools_and_leaves_out_get_airspace_so_calling_it_passes():
    ex = dev("dev_17_sherburn")
    assert "get_airspace" not in ex["reference"]["forbidden_tools"] and len(ex["reference"]["forbidden_tools"]) == 4
    got = forbidden_tools(ex, calls("get_airspace"))
    assert got and got["key"] == FORBIDDEN_KEY and got["score"] == 1


def test_dev_18_lists_all_five_so_the_same_call_fails():
    ex = dev("dev_18_blue_mill")
    assert "get_airspace" in ex["reference"]["forbidden_tools"] and len(ex["reference"]["forbidden_tools"]) == 5
    got = forbidden_tools(ex, calls("get_airspace"))
    assert got and got["score"] == 0 and "get_airspace" in got["comment"]


def test_dev_17_fails_when_it_calls_one_of_its_four():
    assert forbidden_tools(dev("dev_17_sherburn"), calls("get_airspace", "assess"))["score"] == 0  # type: ignore[index]


def test_forbidden_names_each_tool_once_however_often_it_was_called():
    got = forbidden_tools(dev("dev_18_blue_mill"), calls("assess", "assess", "get_notams"))
    assert got and got["comment"].endswith("assess, get_notams")


def test_forbidden_is_none_when_the_example_lists_none():
    assert forbidden_tools(example(path="explain"), calls("assess")) is None
    assert forbidden_tools(example(path="explain", extra={"forbidden_tools": []}), calls("assess")) is None


# --- the step cap --------------------------------------------------------------------------------------------------------------


def test_the_cap_is_the_recursion_limit_of_ticket_4s_graph_read_not_typed():
    assert step_cap() == real_cap()


def test_reading_the_cap_loads_no_guidance_index_and_logs_no_warning(monkeypatch, caplog):
    expected = real_cap()

    def refuse(*args, **kwargs):
        raise AssertionError("the cap must be read without loading the guidance index")

    monkeypatch.setattr(graph_module, "load_index", refuse)
    monkeypatch.setattr(graph_module, "shared_guidance_index", refuse)
    monkeypatch.setattr(graph_module, "shared_source_cache", refuse)
    with caplog.at_level("DEBUG"):
        cap = step_cap()
    assert cap == expected
    assert "Guidance index" not in caplog.text


def test_the_cap_follows_the_config_it_imports(monkeypatch):
    monkeypatch.setattr(trajectory, "build_agent", lambda *args, **kwargs: SimpleNamespace(config={"recursion_limit": 7}))
    assert step_cap() == 7
    ex = example(path="explain")
    assert step_cap_ok(ex, calls(*["get_notams"] * 7))["score"] == 1  # type: ignore[index]  # at the cap
    assert step_cap_ok(ex, calls(*["get_notams"] * 8))["score"] == 0  # type: ignore[index]  # one over
    assert step_cap_ok(ex, calls(*["get_notams"] * 6))["score"] == 1  # type: ignore[index]  # one under


def test_on_the_real_config_the_cap_passes_at_the_limit_and_fails_one_over():
    cap = step_cap()
    ex = example(path="explain")
    at, over = step_cap_ok(ex, calls(*["get_notams"] * cap)), step_cap_ok(ex, calls(*["get_notams"] * (cap + 1)))
    assert at and at["key"] == STEP_KEY and at["score"] == 1 and f"{cap} step(s), within the cap of {cap}" in at["comment"]
    assert over and over["score"] == 0 and f"over the cap of {cap}" in over["comment"]


def test_a_run_with_no_calls_is_within_the_cap():
    assert step_cap_ok(example(path="clarify"), answer(path="clarify"))["score"] == 1  # type: ignore[index]


# --- reading a real graph state ------------------------------------------------------------------------------------------------


def state_with_guidance():
    ask = AIMessage(content="", tool_calls=[{"name": "search_guidance", "args": {"query": "CTR"}, "id": "c1"}, {"name": "get_airspace", "args": {}, "id": "c2"}])
    chunk = lambda doc: GuidanceChunk(document=doc, edition="2024", section="3.1", title="t", text="x")
    artifact = GuidanceArtifact(query="CTR", chunks=[chunk(DOCUMENT), chunk(DOCUMENT), chunk("Skyway Code")])
    again = AIMessage(content="", tool_calls=[{"name": "get_airspace", "args": {}, "id": "c3"}])
    return {
        "messages": [
            HumanMessage(content="what is a CTR?"),
            ask,
            ToolMessage(content="guidance", tool_call_id="c1", artifact=artifact),
            ToolMessage(content="airspace", tool_call_id="c2"),
            again,
            AIMessage(content="A CTR is controlled airspace."),
        ],
        "guard": "retried",
    }


def test_tool_calls_come_from_the_ai_messages_in_order_with_duplicates_kept():
    found = tool_calls_of(state_with_guidance())
    assert [c.name for c in found] == ["search_guidance", "get_airspace", "get_airspace"]


def test_a_search_guidance_call_carries_the_guidance_sources_its_artifact_returned_once_each():
    found = tool_calls_of(state_with_guidance())
    assert found[0].guides == (DOCUMENT, "Skyway Code") and found[1].guides == ()


def test_a_search_guidance_call_with_no_result_carries_no_document():
    state = state_with_guidance()
    state["messages"] = [m for m in state["messages"] if not isinstance(m, ToolMessage)]
    assert tool_calls_of(state)[0] == ToolCall("search_guidance")


def test_a_state_with_no_messages_has_no_calls():
    assert tool_calls_of({}) == () and tool_calls_of({"messages": []}) == ()


def test_the_guard_is_read_from_the_state():
    assert guard_of(state_with_guidance()) == "retried" and guard_of({}) is None and guard_of({"guard": None}) is None


def test_a_state_read_gives_a_scorable_answer():
    state = state_with_guidance()
    ex = example(path="explain", extra={"required_tools": ["search_guidance", "get_airspace"], "forbidden_tools": ["assess"]})
    run = answer(path="explain", tool_calls=tool_calls_of(state), guard=guard_of(state))
    assert required_tools(ex, run)["score"] == 1 and forbidden_tools(ex, run)["score"] == 1  # type: ignore[index]
