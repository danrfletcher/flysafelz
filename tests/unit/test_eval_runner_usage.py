"""Usage on the first model response (5a PR-4.F1, GP2, E4): a clear error and a stop before a second example, and the per-run usage kept when present."""

import pytest
from eval_runner_helpers import ReaderSpec, ScriptedGraph, make_records, run
from langchain_core.messages import AIMessage, HumanMessage

from evals.agent import runner
from evals.agent.runner import UsageMissingError


def test_a_first_response_with_no_usage_stops_the_run_before_a_second_example_starts(tmp_path):
    graph = ScriptedGraph(usage="none")
    with pytest.raises(UsageMissingError) as stopped:
        run(make_records(5), tmp_path, graph)
    assert len(graph.calls) == 1  # no second example, and no retry of the first
    assert stopped.value.last_example_id == "inv_dev_01"
    assert stopped.value.partial.records == []


def test_the_error_is_one_line_naming_the_bridge_and_the_missing_field(tmp_path):
    with pytest.raises(UsageMissingError) as stopped:
        run(make_records(2), tmp_path, ScriptedGraph(usage="none"))
    message = str(stopped.value)
    assert "\n" not in message
    assert "AGENT_BASE_URL" in message and "bridge" in message
    assert "usage_metadata" in message and "usage" in message
    assert "zeros" in message
    assert message.endswith("(last example: inv_dev_01)")


def test_the_check_is_on_the_first_response_not_on_a_later_one(tmp_path):
    graph = ScriptedGraph(usage="first_missing", reader={"inv_dev_01": ReaderSpec(failed=("N1/26",))})
    with pytest.raises(UsageMissingError):
        run(make_records(3), tmp_path, graph)
    assert len(graph.calls) == 1


@pytest.mark.parametrize("style", ["metadata", "api"])
def test_usage_in_either_style_is_accepted_and_kept_per_run(tmp_path, style):
    result = run(make_records(2), tmp_path, ScriptedGraph(usage=style), reps=2)
    assert len(result.records) == 4
    for record in result.records:
        assert record.usage == {"input_tokens": 100, "output_tokens": 20, "total_tokens": 120, "model_calls": 1, "calls_without_usage": 0}


def test_usage_is_summed_over_the_model_responses_of_a_run(tmp_path):
    graph = ScriptedGraph(reader={"inv_dev_01": ReaderSpec(failed=("N1/26",))})
    record = run(make_records(1), tmp_path, graph, reps=1).records[0]
    assert record.usage == {"input_tokens": 200, "output_tokens": 40, "total_tokens": 240, "model_calls": 2, "calls_without_usage": 0}


def test_the_experiment_total_is_the_sum_of_the_runs(tmp_path):
    result = run(make_records(3), tmp_path, ScriptedGraph(), reps=2)
    assert result.usage_total() == {"input_tokens": 600, "output_tokens": 120, "total_tokens": 720, "model_calls": 6, "calls_without_usage": 0}


def test_an_errored_run_has_no_usage_and_adds_nothing_to_the_total(tmp_path):
    graph = ScriptedGraph(script={"inv_dev_02": [RuntimeError("x")] * 4})
    result = run(make_records(3), tmp_path, graph, reps=1)
    assert result.records[1].usage is None
    assert result.usage_total()["total_tokens"] == 240


def test_a_later_run_without_usage_keeps_none_never_zeros(tmp_path):
    def no_usage(call):
        reply = AIMessage("NO-GO. Answer.")
        call.model_call(reply)
        return {"messages": [HumanMessage("q"), reply], "answer": "NO-GO. Answer.", "guard": "passed"}

    result = run(make_records(2), tmp_path, ScriptedGraph(script={"inv_dev_02": [no_usage]}), reps=1)
    assert result.records[0].usage is not None
    assert result.records[1].usage is None and not result.records[1].errored
    assert result.usage_total()["total_tokens"] == 120


def test_no_usage_anywhere_gives_no_total_not_a_zero_one():
    assert runner.ExperimentResult("e", "dev", runner.CACHE_DIR / "e.sqlite", 3, 0, []).usage_total() is None


@pytest.mark.parametrize(
    "message",
    [
        AIMessage("x"),
        AIMessage("x", usage_metadata={"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}),
        AIMessage("x", response_metadata={"usage": {"prompt_tokens": 0, "completion_tokens": 0}}),
        AIMessage("x", response_metadata={"usage": {}}),
        AIMessage("x", response_metadata={"usage": {"prompt_tokens": "many", "completion_tokens": 3}}),
        AIMessage("x", response_metadata={"usage": "lots"}),
    ],
)
def test_usage_that_is_absent_empty_zero_or_not_a_number_counts_as_missing(tmp_path, message):
    def answered(call):
        call.model_call(message)
        return {"messages": [HumanMessage("q"), message], "answer": "NO-GO. Answer."}

    with pytest.raises(UsageMissingError):
        run(make_records(2), tmp_path, ScriptedGraph(script={"inv_dev_01": [answered]}))


def test_the_token_usage_name_the_openai_client_uses_is_accepted(tmp_path):
    def answered(call):
        reply = AIMessage("x", response_metadata={"token_usage": {"prompt_tokens": 7, "completion_tokens": 3}})
        call.model_call(reply)
        return {"messages": [HumanMessage("q"), reply], "answer": "NO-GO. Answer."}

    record = run(make_records(1), tmp_path, ScriptedGraph(script={"inv_dev_01": [answered]}), reps=1).records[0]
    assert record.usage["total_tokens"] == 10


def test_a_state_with_no_model_response_does_not_trip_the_check_and_the_next_example_does(tmp_path):
    def coded_answer(call):
        return {"messages": [], "answer": "Outside the UK: this agent covers the UK only.", "guard": "passed"}

    graph = ScriptedGraph(script={"inv_dev_01": [coded_answer]}, usage="none")
    with pytest.raises(UsageMissingError) as stopped:
        run(make_records(3), tmp_path, graph)
    assert stopped.value.last_example_id == "inv_dev_02" and [r.example_id for r in stopped.value.partial.records] == ["inv_dev_01"]


def test_a_draft_that_never_reaches_the_state_is_still_counted_and_a_node_other_than_the_agents_is_not(tmp_path):
    def two_drafts(call):
        call.model_call(AIMessage("draft", usage_metadata={"input_tokens": 10, "output_tokens": 2, "total_tokens": 12}))
        call.model_call(AIMessage("reader", usage_metadata={"input_tokens": 999, "output_tokens": 9, "total_tokens": 1008}), node="tools")
        final = AIMessage("NO-GO. Answer.", usage_metadata={"input_tokens": 20, "output_tokens": 3, "total_tokens": 23})
        call.model_call(final)
        return {"messages": [HumanMessage("q"), final], "answer": "NO-GO. Answer.", "guard": "retried"}

    record = run(make_records(1), tmp_path, ScriptedGraph(script={"inv_dev_01": [two_drafts]}), reps=1).records[0]
    assert record.usage == {"input_tokens": 30, "output_tokens": 5, "total_tokens": 35, "model_calls": 2, "calls_without_usage": 0}


def test_a_message_the_agent_wrote_itself_is_not_the_first_model_response(tmp_path):
    def coded(call):  # an AI message in the state, and no model run behind it
        return {"messages": [HumanMessage("q"), AIMessage("The assistant is unavailable.")], "answer": "The assistant is unavailable.", "guard": "fallback"}

    graph = ScriptedGraph(script={"inv_dev_01": [coded]})
    records = run(make_records(2), tmp_path, graph, reps=1).records
    assert records[0].usage is None and records[1].usage["total_tokens"] == 120
