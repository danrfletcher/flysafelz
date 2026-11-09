"""The strict usage check (5a PR-10.F1, E4): `make eval` stops on the first model response without usage, or on a later one, naming the example, and never sums a gap as 0.

The real graph (ticket 4's `build_agent`) under the real runner, with a scripted chat model returning `AIMessage`s with and without usage. Sockets stay off, no key is read.
"""

import pytest
from eval_runner_helpers import FakeSleeper, RecordingTimeout
from graph_helpers import ScriptedModel, agent_graph
from langchain_core.messages import AIMessage
from tool_helpers import BLUE_MILL_LZ, BLUE_MILL_WINDOW

from evals.agent import runner
from evals.agent.runner import STRICT_USAGE_FIRST, STRICT_USAGE_LATER, UsageMissingError
from flysafelz.agent.sources import SourceCache

BLUE_MILL_CASE = "blue_mill_2026-10-31"
USAGE = {"input_tokens": 50, "output_tokens": 5, "total_tokens": 55}
PLAIN = "[path: explain]\nThis is a plain reply."


class NoFetch:
    def __call__(self, *args):
        raise AssertionError("a case-mode run fetched live")


def reply(text: str = PLAIN, **kwargs) -> AIMessage:
    return AIMessage(text, **kwargs)


def with_usage(text: str = PLAIN) -> AIMessage:
    return reply(text, usage_metadata=dict(USAGE))


def tool_round(*, usage: bool = True) -> AIMessage:
    return AIMessage("", tool_calls=[{"name": "get_weather", "args": BLUE_MILL_WINDOW, "id": "w-1"}], usage_metadata=dict(USAGE) if usage else None)


def example(number: int) -> dict:
    return {"id": f"inv_usage_{number:02d}", "inputs": {**BLUE_MILL_LZ, "question": "Can I fly here today?", "now": "2026-10-31T07:30:00+00:00"}, "fixtures": {"case": BLUE_MILL_CASE, "drop": []}}


def run_real(model, tmp_path, count: int = 3, *, strict: bool = True, **options):
    return runner.run_experiment(
        [example(n) for n in range(1, count + 1)],
        experiment="v1-dev-usage",
        graph_factory=lambda: agent_graph(model, source_cache=SourceCache(fetch=NoFetch())),
        cache_dir=tmp_path,
        reps=1,
        sleep=FakeSleeper(),
        timeout=RecordingTimeout(),
        strict_usage=strict,
        **options,
    )


def test_a_first_response_without_usage_stops_the_run_with_the_first_call_error_before_a_second_example(tmp_path):
    model = ScriptedModel(script=[reply(), with_usage(), with_usage()])
    with pytest.raises(UsageMissingError) as stopped:
        run_real(model, tmp_path)
    assert str(stopped.value) == f"{STRICT_USAGE_FIRST} (last example: inv_usage_01)"
    assert STRICT_USAGE_FIRST == "bridge returned no token usage on the first call: tokens cannot be reported"
    assert len(model.calls) == 1 and stopped.value.partial.records == []


def test_usage_on_the_first_call_but_none_on_a_later_one_stops_the_run_naming_the_example(tmp_path):
    model = ScriptedModel(script=[with_usage(), with_usage(), reply()])
    with pytest.raises(UsageMissingError) as stopped:
        run_real(model, tmp_path)
    assert STRICT_USAGE_LATER in str(stopped.value) and str(stopped.value).endswith("(last example: inv_usage_03)")
    assert [r.example_id for r in stopped.value.partial.records] == ["inv_usage_01", "inv_usage_02"]
    assert all(r.usage["calls_without_usage"] == 0 for r in stopped.value.partial.records)  # nothing was summed over a gap


def test_a_later_response_of_the_same_run_without_usage_stops_the_run_too(tmp_path):
    model = ScriptedModel(script=[tool_round(), reply()])
    with pytest.raises(UsageMissingError, match="later call") as stopped:
        run_real(model, tmp_path)
    assert str(stopped.value).endswith("(last example: inv_usage_01)")


def test_a_tool_round_without_usage_followed_by_one_with_it_is_still_a_gap(tmp_path):
    model = ScriptedModel(script=[tool_round(usage=False), with_usage()])
    with pytest.raises(UsageMissingError, match="first call"):
        run_real(model, tmp_path)


@pytest.mark.parametrize(
    "partial",
    [
        pytest.param({"usage": {"input_tokens": 12}}, id="no-output-tokens"),
        pytest.param({"usage": {"output_tokens": 12}}, id="no-input-tokens"),
        pytest.param({"usage": {"prompt_tokens": 12}}, id="api-style-no-completion"),
        pytest.param({"usage": {"input_tokens": 0, "output_tokens": 0}}, id="zeros"),
        pytest.param({"usage": {"input_tokens": "12", "output_tokens": 3}}, id="not-a-number"),
    ],
)
def test_usage_that_lacks_input_or_output_tokens_counts_as_no_usage(tmp_path, partial):
    with pytest.raises(UsageMissingError, match="first call"):
        run_real(ScriptedModel(script=[reply(response_metadata=partial)]), tmp_path, 1)


def test_the_api_style_usage_with_both_fields_passes(tmp_path):
    model = ScriptedModel(script=[reply(response_metadata={"usage": {"prompt_tokens": 40, "completion_tokens": 6}})] * 2)
    result = run_real(model, tmp_path, 2)
    assert [r.usage["total_tokens"] for r in result.records] == [46, 46]


def test_a_run_with_usage_everywhere_has_no_gap_and_an_exact_total(tmp_path):
    result = run_real(ScriptedModel(script=[with_usage()] * 3), tmp_path)
    assert result.usage_total() == {"input_tokens": 150, "output_tokens": 15, "total_tokens": 165, "model_calls": 3, "calls_without_usage": 0}


def test_the_model_failure_fallback_has_no_response_so_it_is_not_a_gap(tmp_path):
    """A 5xx becomes the agent's own 'unavailable' answer: no model response, so nothing for the strict check to find."""
    import httpx
    import openai

    error = openai.InternalServerError("Error code: 500", response=httpx.Response(500, request=httpx.Request("POST", "http://bridge.invalid/v1")), body=None)
    result = run_real(ScriptedModel(script=[error, with_usage()]), tmp_path, 2)
    assert [r.errored for r in result.records] == [False, False] and result.records[0].usage is None


def test_without_strict_only_the_first_response_is_checked_as_before(tmp_path):
    model = ScriptedModel(script=[with_usage(), reply(), with_usage()])
    result = run_real(model, tmp_path, strict=False)
    assert [r.errored for r in result.records] == [False, False, False]
    assert result.records[1].usage is None  # a later gap, recorded as absent, never as zeros


def test_the_default_non_strict_message_is_unchanged(tmp_path):
    with pytest.raises(UsageMissingError, match=r"AGENT_BASE_URL.*tokens would be reported as zeros"):
        run_real(ScriptedModel(script=[reply()]), tmp_path, 1, strict=False)


# --- R2: the strict check also covers an attempt that failed ------------------------------------------------------------------


class ProbeGraph:
    """A graph whose attempts are scripted: each takes the next step, which says what model responses the attempt got (announced to the config's callbacks, as a real
    chat model run does) and then how it ends. A step is (responses, ending): `responses` are AIMessages, `ending` is an exception to raise or None to return a state."""

    def __init__(self, steps):
        self.steps = list(steps)
        self.invoked: list[str] = []

    def invoke(self, request, config, *, context):
        import uuid

        from langchain_core.outputs import ChatGeneration, LLMResult

        self.invoked.append(config["metadata"]["example_id"])
        responses, ending = self.steps.pop(0)
        for message in responses:
            run_id = uuid.uuid4()
            for handler in config["callbacks"]:
                handler.on_chat_model_start({}, [[]], run_id=run_id, metadata={"langgraph_node": "model"})
                handler.on_llm_end(LLMResult(generations=[[ChatGeneration(message=message)]]), run_id=run_id)
        if ending is not None:
            raise ending
        return {"messages": [*responses], "answer": "GO WITH CONDITIONS. Invented answer.", "guard": "passed"}


def run_probe(graph, tmp_path, count: int = 3, **options):
    options.setdefault("strict_usage", True)
    return runner.run_experiment(
        [example(n) for n in range(1, count + 1)],
        experiment="v1-dev-probe",
        graph_factory=lambda: graph,
        cache_dir=tmp_path,
        reps=1,
        sleep=FakeSleeper(),
        timeout=RecordingTimeout(),
        **options,
    )


def test_a_response_without_usage_in_an_attempt_that_then_raises_stops_the_run_not_an_errored_record(tmp_path):
    graph = ProbeGraph([([with_usage()], None), ([reply()], RuntimeError("the graph blew up")), ([with_usage()], None)])
    with pytest.raises(UsageMissingError) as stopped:
        run_probe(graph, tmp_path)
    assert STRICT_USAGE_LATER in str(stopped.value) and str(stopped.value).endswith("(last example: inv_usage_02)")
    assert graph.invoked == ["inv_usage_01", "inv_usage_02"]  # the third example never ran, and no retry of the second
    assert [r.example_id for r in stopped.value.partial.records] == ["inv_usage_01"]  # the second is not an errored record in the totals


def test_a_first_response_without_usage_in_a_failed_attempt_is_the_first_call_error(tmp_path):
    graph = ProbeGraph([([reply()], RuntimeError("boom")), ([with_usage()], None)])
    with pytest.raises(UsageMissingError) as stopped:
        run_probe(graph, tmp_path, 2)
    assert str(stopped.value) == f"{STRICT_USAGE_FIRST} (last example: inv_usage_01)"
    assert graph.invoked == ["inv_usage_01"] and stopped.value.partial.records == []


def test_a_response_without_usage_in_an_attempt_that_times_out_stops_the_run(tmp_path):
    graph = ProbeGraph([([with_usage()], None), ([reply()], runner.ExampleTimeout("no result within 180 s")), ([with_usage()], None)])
    with pytest.raises(UsageMissingError, match="later call") as stopped:
        run_probe(graph, tmp_path)
    assert str(stopped.value).endswith("(last example: inv_usage_02)") and graph.invoked == ["inv_usage_01", "inv_usage_02"]


def test_a_gap_in_a_failed_first_attempt_stops_the_run_before_the_retry(tmp_path):
    graph = ProbeGraph([([with_usage()], None), ([with_usage(), reply()], RuntimeError("boom")), ([with_usage()], None)])
    with pytest.raises(UsageMissingError):
        run_probe(graph, tmp_path, retries=3)
    assert graph.invoked == ["inv_usage_01", "inv_usage_02"]  # one attempt at the second example, not four


def test_a_failed_attempt_whose_responses_all_have_usage_is_retried_and_counted_as_before(tmp_path):
    graph = ProbeGraph([([with_usage()], RuntimeError("boom")), ([with_usage()], None)])
    [record] = run_probe(graph, tmp_path, 1).records
    assert not record.errored and record.attempts == 2
    assert record.usage == {"input_tokens": 100, "output_tokens": 10, "total_tokens": 110, "model_calls": 2, "calls_without_usage": 0}  # the failed attempt's call was paid for


def test_a_failed_attempt_with_no_model_response_at_all_is_not_a_usage_gap(tmp_path):
    graph = ProbeGraph([([], RuntimeError("boom")), ([with_usage()], None), ([with_usage()], None)])
    result = run_probe(graph, tmp_path, 2, retries=1)
    assert [r.errored for r in result.records] == [False, False] and result.records[0].attempts == 2


def test_without_strict_a_failed_attempt_with_a_later_gap_still_ends_errored_as_before(tmp_path):
    graph = ProbeGraph([([with_usage()], None), ([reply()], RuntimeError("boom"))] + [([reply()], RuntimeError("boom"))] * 3)
    result = run_probe(graph, tmp_path, 2, strict_usage=False)
    assert [r.errored for r in result.records] == [False, True] and result.records[1].usage is None


def test_a_keyboard_interrupt_in_a_failed_attempt_is_not_turned_into_a_usage_error(tmp_path):
    graph = ProbeGraph([([reply()], KeyboardInterrupt())])
    result = run_probe(graph, tmp_path, 2)
    assert result.interrupted and result.records == []
