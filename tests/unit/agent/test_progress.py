"""Progress lines (G5, Q5): fixed code text, written to LangGraph's custom stream; nothing outside a graph run (F3)."""

import re
from typing import TypedDict

import pytest
from langchain_core.runnables import RunnableLambda
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph

from flysafelz.agent import progress
from flysafelz.agent.progress import ASSESSING, CHECKING_AIRSPACE, CHECKING_ANSWER, CHECKING_WEATHER, emit, reading_notams

VERDICT_WORDS = re.compile(r"NO-GO|NO GO|UNVERIFIED|PILOT DISCRETION|\bGO\b|\bsafe\b", re.IGNORECASE)


class State(TypedDict):
    done: bool


def one_node_graph(*lines: str):
    """A graph with one node that emits each line, as a tool would."""

    def node(state: State) -> State:
        for line in lines:
            emit(line)
        return {"done": True}

    builder = StateGraph(State)
    builder.add_node("work", node)
    builder.add_edge(START, "work")
    builder.add_edge("work", END)
    return builder.compile()


def test_each_q5_line_reaches_the_custom_stream_unchanged_and_in_order():
    lines = [CHECKING_WEATHER, reading_notams(1, 2), reading_notams(2, 2), ASSESSING, CHECKING_ANSWER]
    streamed = list(one_node_graph(*lines).stream({"done": False}, stream_mode="custom"))
    assert streamed == ["Checking weather…", "Reading 2 NOTAMs (1/2)…", "Reading 2 NOTAMs (2/2)…", "Assessing…", "Checking the answer…"]


def test_a_custom_stream_is_not_asked_for_so_nothing_is_added_to_the_other_modes():
    values = list(one_node_graph(CHECKING_WEATHER).stream({"done": False}, stream_mode="values"))
    assert values == [{"done": False}, {"done": True}]


def test_emit_outside_a_graph_run_is_a_no_op():
    emit(CHECKING_WEATHER)  # the real get_stream_writer raises RuntimeError outside a run; emit swallows exactly that


def test_emit_from_a_tool_invoked_directly_is_a_no_op(capsys):
    @tool
    def work() -> str:
        """Emit a line, as a tool does."""
        emit(CHECKING_WEATHER)
        return "ok"

    assert work.invoke({}) == "ok"
    captured = capsys.readouterr()
    assert captured.out == "" and captured.err == ""


def test_emit_from_a_runnable_lambda_is_a_no_op(capsys):
    def work(value: int) -> int:
        emit(ASSESSING)
        return value + 1

    assert RunnableLambda(work).invoke(1) == 2
    captured = capsys.readouterr()
    assert captured.out == "" and captured.err == ""


def test_emit_from_a_tool_inside_a_graph_run_still_streams():
    @tool
    def work() -> str:
        """Emit a line, as a tool does."""
        emit(CHECKING_WEATHER)
        return "ok"

    def node(state: State) -> State:
        work.invoke({})
        return {"done": True}

    builder = StateGraph(State)
    builder.add_node("work", node)
    builder.add_edge(START, "work")
    builder.add_edge("work", END)
    assert list(builder.compile().stream({"done": False}, stream_mode="custom")) == ["Checking weather…"]


def test_an_error_from_the_writer_itself_is_not_swallowed():
    def failing(_line: str) -> None:
        raise ValueError("stream is broken")

    class Runtime:
        stream_writer = staticmethod(failing)

    def work(value: int) -> int:
        emit(ASSESSING)
        return value

    with pytest.raises(ValueError, match="stream is broken"):
        RunnableLambda(work).invoke(1, config={"configurable": {"__pregel_runtime": Runtime()}})


def test_the_notam_line_counts_from_one_and_agrees_with_the_spec_text():
    assert reading_notams(1, 2) == "Reading 2 NOTAMs (1/2)…"
    assert reading_notams(2, 2) == "Reading 2 NOTAMs (2/2)…"
    assert reading_notams(1, 1) == "Reading 1 NOTAM (1/1)…"
    assert reading_notams(3, 3) == "Reading 3 NOTAMs (3/3)…"


@pytest.mark.parametrize("i, n", [(0, 2), (3, 2), (-1, 2), (1, 0)])
def test_a_notam_number_outside_the_count_is_refused(i, n):
    with pytest.raises(ValueError, match="out of range"):
        reading_notams(i, n)


@pytest.mark.parametrize("i, n", [(1.0, 2), (1, 2.0), ("1", 2), (True, 2), (1, True)])
def test_only_whole_numbers_go_into_a_line(i, n):
    with pytest.raises(TypeError, match="whole numbers"):
        reading_notams(i, n)


def test_f3_no_progress_line_holds_a_verdict_word_or_a_number_but_the_notam_counts():
    constants = [CHECKING_WEATHER, CHECKING_AIRSPACE, ASSESSING, CHECKING_ANSWER]
    builders = [reading_notams(i, n) for n in (1, 2, 3) for i in range(1, n + 1)]
    for line in constants:
        assert not VERDICT_WORDS.search(line), line
        assert not re.search(r"\d", line), line
    for line in builders:
        assert not VERDICT_WORDS.search(line), line
        # the only digits are the counts i and n, in "(i/n)" and "Reading n"
        counts = re.fullmatch(r"Reading (\d+) NOTAMs? \((\d+)/(\d+)\)…", line)
        assert counts and counts.group(1) == counts.group(3), line


def test_the_progress_constants_are_the_whole_public_list_of_fixed_lines():
    fixed = {name: value for name, value in vars(progress).items() if name.isupper() and isinstance(value, str)}
    assert fixed == {"CHECKING_WEATHER": "Checking weather…", "CHECKING_AIRSPACE": "Checking airspace…", "ASSESSING": "Assessing…", "CHECKING_ANSWER": "Checking the answer…"}
