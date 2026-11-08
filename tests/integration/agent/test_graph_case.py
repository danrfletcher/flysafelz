"""PR-9.F1 GP3 and I3: the whole graph on the committed Blue Mill case, offline, with a scripted chat model.

The tools, the middleware, the prompt, the input check and the source store are the real ones. Only the chat model is scripted (the agent's, and in the I3
test the NOTAM reader's). A live fetch fails the test.
"""

import pytest
from fakes import FakeToolChatModel
from graph_helpers import BLUE_MILL_CASE, GP3_QUESTION, agent_graph, ask, assess_call, blue_mill_context, thread
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from reader_replies import fixture_readings, model_reply
from reader_set_helpers import ScriptedChat
from unit.reader_helpers import blue_mill_notams

from flysafelz.agent.sources import SourceCache
from flysafelz.agent.state import RunContext

FINAL = "[path: assess]\nPILOT DISCRETION from 10:00 GMT to 16:00 GMT, with conditions."


class NoFetch:
    """The live fetcher: in case mode it must never be called."""

    def __call__(self, *args):
        raise AssertionError("a case-mode run fetched live")


def blue_mill_reader() -> ScriptedChat:
    """A fake reader model: the two NOTAMs readings.yaml covers answer as readings.yaml says; any other gets the generated good reply."""
    by_id, readings = blue_mill_notams(), fixture_readings()
    scripts = {key: [model_reply(by_id[key.replace("_", "/")], readings[key.replace("_", "/")])] for key in ("H7171_26", "J3857_26")}
    return ScriptedChat(list(by_id.values()), **scripts)


def scripted_agent() -> FakeToolChatModel:
    return FakeToolChatModel(script=[assess_call(), AIMessage(FINAL)])


def test_gp3_the_graph_runs_end_to_end_offline_and_sets_answer():
    reader_builds = []

    def reader_model():
        reader_builds.append(1)
        raise AssertionError("the default reader is `fixture`: no reader model is built")

    graph = agent_graph(scripted_agent(), source_cache=SourceCache(fetch=NoFetch(), reader_model=reader_model))
    state = ask(graph, GP3_QUESTION)
    assert state["answer"] == FINAL
    assert state["messages"][-1].content == FINAL
    [assessed] = [m for m in state["messages"] if isinstance(m, ToolMessage)]
    assert assessed.name == "assess" and assessed.status == "success"
    assert "Window: Saturday 31 Oct 10:00 – 16:00 GMT." in assessed.text
    assert "H7171/26" in assessed.text and "J3857/26" in assessed.text
    assert reader_builds == []  # I3: no reader call with the default `reader: fixture`


def test_gp3_the_progress_lines_arrive_on_the_custom_stream_and_the_answer_once():
    graph = agent_graph(scripted_agent(), source_cache=SourceCache(fetch=NoFetch()))
    given = {"messages": [HumanMessage(GP3_QUESTION)], "lat": 53.802797, "lon": -0.086644}
    custom, final = [], None
    for mode, chunk in graph.stream(given, thread(), context=blue_mill_context(), stream_mode=["custom", "values"]):
        if mode == "custom":
            custom.append(chunk)
        else:
            final = chunk
    assert "Assessing…" in custom
    assert final["answer"] == FINAL
    assert FINAL not in custom


def test_i3_the_same_run_with_the_live_reader_reaches_the_reader_model_through_the_graph(tmp_path):
    """No code change between the runs: only the run context differs (reader="live", a reader_cache) and the source store holds a fake reader model."""
    reader = blue_mill_reader()
    source_cache = SourceCache(fetch=NoFetch(), reader_model=lambda: reader)
    live = RunContext(now=blue_mill_context().now, case=BLUE_MILL_CASE, reader="live", reader_cache=tmp_path / "r.sqlite")
    model = FakeToolChatModel(script=[AIMessage("", tool_calls=[{"name": "get_notams", "args": {"start": "2026-10-31 10:00", "end": "2026-10-31 16:00"}, "id": "n1"}]), AIMessage(FINAL)])
    state = ask(agent_graph(model, source_cache=source_cache), GP3_QUESTION, context=live)
    assert len(reader.calls) > 0
    assert set(reader.calls) >= {"H7171/26", "J3857/26"}
    assert state["answer"] == FINAL


def test_i3_the_assess_tool_reads_through_the_reader_too_in_a_live_run(tmp_path):
    reader = blue_mill_reader()
    source_cache = SourceCache(fetch=NoFetch(), reader_model=lambda: reader)
    live = RunContext(now=blue_mill_context().now, case=BLUE_MILL_CASE, reader="live", reader_cache=tmp_path / "r.sqlite")
    state = ask(agent_graph(scripted_agent(), source_cache=source_cache), GP3_QUESTION, context=live)
    assert len(reader.calls) > 0 and state["answer"] == FINAL


def test_a_second_question_on_the_same_thread_uses_the_threads_landing_zone_and_the_new_turns_prompt():
    model = FakeToolChatModel(script=[AIMessage("one."), AIMessage("two.")])
    graph = agent_graph(model, source_cache=SourceCache(fetch=NoFetch()))
    ask(graph, "first question")
    later = blue_mill_context().now.replace(hour=11, minute=45)
    state = graph.invoke({"messages": [HumanMessage("second question")]}, thread(), context=blue_mill_context(later))
    assert state["answer"] == "two."
    assert "The time now is 11:45 GMT." in str(model.calls[-1][0].content)


@pytest.mark.parametrize("lz", [{"lat": 48.85, "lon": 2.35}])
def test_outside_the_uk_the_agent_model_is_never_called(lz):
    model = FakeToolChatModel(script=[])
    state = ask(agent_graph(model), GP3_QUESTION, lz=lz)
    assert state["answer"] == "FlySafeLZ only covers the UK." and model.calls == []
