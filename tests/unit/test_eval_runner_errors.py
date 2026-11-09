"""Errored runs, retries and the reader's part (5a PR-4.F1, G7, E6, E7, E13): the retry table, the backoff, the timeout, the 10% rule, fatal stops, SIGINT."""

import os
import signal
import threading
import time

import httpx
import openai
import pytest
from eval_runner_helpers import NO_ANSWER, FakeClock, FakeSleeper, ReaderSpec, RecordingTimeout, RunRecordingClient, ScriptedGraph, make_records, run
from langchain_core.messages import AIMessage, HumanMessage

from evals.agent import runner
from evals.agent.runner import BACKOFF_SECONDS, EXAMPLE_TIMEOUT_S, ExampleTimeout, FatalRunError

KINDS = {
    "exception": lambda n: RuntimeError(f"graph blew up {n}"),
    "timeout": lambda n: ExampleTimeout(f"no result within {EXAMPLE_TIMEOUT_S:g} s"),
    "no_final_answer": lambda n: NO_ANSWER,
}


# --- E6: the retry table -----------------------------------------------------------------------------------------------


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("failures", [0, 1, 2, 3, 4])
def test_zero_to_three_failures_then_success_is_scored_and_four_failures_are_errored(tmp_path, kind, failures):
    script = {"inv_dev_01": [KINDS[kind](n) for n in range(failures)]}
    graph = ScriptedGraph(script=script)
    sleeper = FakeSleeper()
    result = run(make_records(2), tmp_path, graph, reps=1, sleep=sleeper)
    first = result.records[0]
    errored = failures == 4
    assert first.errored is errored
    assert first.attempts == min(failures + 1, 4)
    assert len([c for c in graph.calls if c.example_id == "inv_dev_01"]) == first.attempts
    assert first.backoff_waits == BACKOFF_SECONDS[: min(failures, 3)]
    assert sleeper.waits == list(first.backoff_waits)  # the example after it needed none
    if errored:
        assert first.output is None and first.answer is None and first.last_error
        assert result.scored == [result.records[1]]
    else:
        assert first.answer and first.output is not None and first.last_error is None
        assert result.scored == result.records
    assert result.records[1].attempts == 1 and not result.records[1].errored  # the next example is unaffected


def test_the_backoff_schedule_and_the_retry_count_are_the_named_constants():
    assert runner.RETRIES == 3 and len(BACKOFF_SECONDS) == runner.RETRIES
    assert list(BACKOFF_SECONDS) == sorted(BACKOFF_SECONDS) and all(w > 0 for w in BACKOFF_SECONDS)


def test_the_waits_go_through_the_injected_sleeper_and_the_clock_not_the_real_one(tmp_path, monkeypatch):
    def no_real_sleep(seconds):
        raise AssertionError("time.sleep was called")

    monkeypatch.setattr(time, "sleep", no_real_sleep)
    clock = FakeClock()
    sleeper = FakeSleeper(clock)
    result = run(make_records(1), tmp_path, ScriptedGraph(script={"inv_dev_01": [RuntimeError("x")] * 4}), reps=1, sleep=sleeper, clock=clock)
    assert sleeper.waits == list(BACKOFF_SECONDS)
    assert result.records[0].latency_s == sum(BACKOFF_SECONDS)  # the fake clock moved only by the waits


def test_each_attempt_is_given_the_timeout_and_the_timeout_is_one_named_constant(tmp_path):
    assert EXAMPLE_TIMEOUT_S == 180
    timeout = RecordingTimeout()
    run(make_records(2), tmp_path, ScriptedGraph(script={"inv_dev_01": [RuntimeError("x")]}), reps=1, timeout=timeout)
    assert timeout.seconds == [180.0, 180.0, 180.0]  # two attempts at the first example, one at the second


def test_every_attempt_is_a_new_thread_but_one_context_value_set(tmp_path):
    graph = ScriptedGraph(script={"inv_dev_01": [RuntimeError("x"), RuntimeError("y")]})
    run(make_records(1), tmp_path, graph, reps=1)
    threads = [c.config["configurable"]["thread_id"] for c in graph.calls]
    assert len(set(threads)) == 3
    assert len({(c.context.case, c.context.reader_cache, c.context.now, c.context.drop) for c in graph.calls}) == 1


def test_the_last_error_is_one_short_line_with_phone_numbers_masked(tmp_path):
    error = RuntimeError("bad\n" + "x" * 500 + " call 01632 960123")
    result = run(make_records(1), tmp_path, ScriptedGraph(script={"inv_dev_01": [error] * 4}), reps=1)
    last = result.records[0].last_error
    assert last.startswith("RuntimeError: bad xxx") and "\n" not in last and len(last) <= runner.ERROR_LIMIT
    short = runner._short(RuntimeError("call 01632 960123 now"))
    assert "960123" not in short


# --- the real timeout --------------------------------------------------------------------------------------------------


def test_call_with_timeout_stops_a_call_that_runs_past_it_and_puts_the_signal_back():
    before = signal.getsignal(signal.SIGALRM)
    started = time.monotonic()
    with pytest.raises(ExampleTimeout, match="0.05 s"):
        runner.call_with_timeout(lambda: time.sleep(5), 0.05)
    assert time.monotonic() - started < 2
    assert signal.getsignal(signal.SIGALRM) is before
    assert signal.getitimer(signal.ITIMER_REAL) == (0.0, 0.0)


def test_call_with_timeout_returns_what_a_quick_call_returns_and_cancels_the_timer():
    assert runner.call_with_timeout(lambda: "done", 5) == "done"
    assert signal.getitimer(signal.ITIMER_REAL) == (0.0, 0.0)


def test_the_agents_own_except_exception_does_not_swallow_the_timeout():
    def swallowing():
        try:
            time.sleep(5)
        except Exception:  # noqa: BLE001 - what the tools and the reader do with a failed call
            return "swallowed"

    with pytest.raises(ExampleTimeout):
        runner.call_with_timeout(swallowing, 0.05)


def test_call_with_timeout_refuses_a_thread_that_is_not_the_main_one():
    seen = []

    def in_thread():
        try:
            runner.call_with_timeout(lambda: None, 1)
        except RuntimeError as error:
            seen.append(str(error))

    worker = threading.Thread(target=in_thread)
    worker.start()
    worker.join()
    assert seen and "main thread" in seen[0]


def test_a_graph_that_hangs_is_errored_after_four_timeouts_with_the_real_timeout(tmp_path):
    graph = ScriptedGraph(script={"inv_dev_01": [lambda call: time.sleep(5)] * 4})
    sleeper = FakeSleeper()
    result = run(make_records(2), tmp_path, graph, reps=1, timeout=runner.call_with_timeout, timeout_s=0.05, sleep=sleeper)
    assert result.records[0].errored and result.records[0].attempts == 4
    assert result.records[0].last_error.startswith("ExampleTimeout: no result within 0.05 s")
    assert not result.records[1].errored
    assert graph.peak == 1


# --- E6: the incomplete threshold --------------------------------------------------------------------------------------


@pytest.mark.parametrize(("errored", "total", "incomplete"), [(0, 87, False), (8, 87, False), (9, 87, True), (1, 10, False), (2, 10, True), (0, 0, False), (3, 29, True), (2, 29, False)])
def test_incomplete_is_more_than_ten_percent_errored(errored, total, incomplete):
    assert runner.is_incomplete(errored, total) is incomplete


@pytest.mark.parametrize(("failing", "incomplete"), [(8, False), (9, True)])
def test_87_runs_with_8_errored_are_complete_and_with_9_are_incomplete(tmp_path, failing, incomplete):
    records = make_records(29)
    # the first `failing` (example, rep) pairs in run order fail on every attempt
    pairs = [(r["id"], rep) for rep in (1, 2, 3) for r in records][:failing]
    graph = ScriptedGraph(script={pair: [RuntimeError("down")] * 4 for pair in pairs})
    result = run(records, tmp_path, graph)
    assert result.planned_runs == 87 and len(result.records) == 87
    assert result.errored_runs == failing
    assert result.incomplete is incomplete
    assert len(result.scored) == 87 - failing
    assert ("incomplete: " + str(incomplete).lower()) in result.summary_lines()


def test_exactly_ten_percent_errored_in_a_real_run_is_complete(tmp_path):
    records = make_records(10)
    result = run(records, tmp_path, ScriptedGraph(script={(records[0]["id"], 1): [RuntimeError("x")] * 4}), reps=1)
    assert result.errored_runs == 1 and result.planned_runs == 10
    assert result.incomplete is False


def test_an_errored_example_still_runs_in_its_later_reps_and_is_listed_once_per_rep(tmp_path):
    records = make_records(3)
    graph = ScriptedGraph(script={("inv_dev_02", 1): [RuntimeError("down")] * 4, ("inv_dev_02", 3): [NO_ANSWER] * 4})
    result = run(records, tmp_path, graph)
    assert [(i, rep) for i, rep in graph.order() if i == "inv_dev_02"] == [("inv_dev_02", 1)] * 4 + [("inv_dev_02", 2)] + [("inv_dev_02", 3)] * 4
    errored = result.errored
    assert [e.example_id for e in errored[1]] == ["inv_dev_02"] and errored[2] == [] and [e.example_id for e in errored[3]] == ["inv_dev_02"]
    assert errored[1][0].error.startswith("RuntimeError: down") and errored[3][0].error.startswith("NoFinalAnswer")
    assert len(result.scored) == 7


def test_an_id_is_listed_once_in_a_rep_even_if_it_appears_twice(tmp_path):
    records = make_records(2)
    records.append(dict(records[0]))
    graph = ScriptedGraph(script={"inv_dev_01": [RuntimeError("down")] * 8})
    result = run(records, tmp_path, graph, reps=1)
    assert result.errored_runs == 1 and [e.example_id for e in result.errored[1]] == ["inv_dev_01"]


# --- G7: what is not errored -------------------------------------------------------------------------------------------


def test_a_guarded_fallback_answer_is_kept_for_scoring(tmp_path):
    def fallback(call):
        return {"messages": [], "answer": "The assistant is unavailable, so this spot could not be checked. Treat it as UNVERIFIED.", "guard": "fallback"}

    graph = ScriptedGraph(script={"inv_dev_01": [fallback]})
    result = run(make_records(1), tmp_path, graph, reps=1)
    assert not result.records[0].errored and result.records[0].attempts == 1 and result.records[0].output["guard"] == "fallback"
    assert len(graph.calls) == 1


def test_a_closing_ai_message_counts_as_the_final_answer_when_the_state_has_no_answer_key(tmp_path):
    def messages_only(call):
        return {"messages": [HumanMessage("q"), AIMessage("NO-GO. Closing message.", usage_metadata={"input_tokens": 1, "output_tokens": 1, "total_tokens": 2})]}

    result = run(make_records(1), tmp_path, ScriptedGraph(script={"inv_dev_01": [messages_only]}), reps=1)
    assert result.records[0].answer == "NO-GO. Closing message." and result.records[0].attempts == 1


@pytest.mark.parametrize("state", [None, "text", [], {"messages": []}, {"answer": "", "messages": []}, {"answer": 3}])
def test_a_state_without_a_final_answer_is_a_failed_attempt(tmp_path, state):
    result = run(make_records(1), tmp_path, ScriptedGraph(script={"inv_dev_01": [lambda call: state] * 4}), reps=1)
    assert result.records[0].errored and result.records[0].attempts == 4


# --- E7: reader failures that became UNVERIFIED ------------------------------------------------------------------------


def test_unverified_segments_from_a_failed_reader_call_are_kept_and_counted_not_errored(tmp_path):
    graph = ScriptedGraph(reader={"inv_dev_01": ReaderSpec(failed=("N1/26", "N2/26")), "inv_dev_02": ReaderSpec(failed=("N3/26",))})
    result = run(make_records(3), tmp_path, graph)
    first = result.records[0]
    assert not first.errored and first.attempts == 1 and first.answer
    assert first.reader_failed_to_unverified == ("N1/26", "N2/26")
    assert result.errored_runs == 0 and len(result.scored) == 9
    assert [len(r.reader_failed_to_unverified) for r in result.records if r.rep == 1] == [2, 1, 0]
    assert result.reader_failed_to_unverified == {"per_rep": {1: 3, 2: 3, 3: 3}, "total": 9}
    assert len(graph.calls) == 9  # nothing was retried


def test_a_failed_reader_call_is_counted_once_per_run_however_many_times_get_notams_named_it(tmp_path):
    graph = ScriptedGraph(reader={"inv_dev_01": ReaderSpec(failed=("N1/26",), notams_calls=3)})
    result = run(make_records(1), tmp_path, graph, reps=1)
    assert result.reader_failed_to_unverified == {"per_rep": {1: 1}, "total": 1}


def test_a_failed_read_that_left_no_unverified_segment_is_not_counted(tmp_path):
    graph = ScriptedGraph(reader={"inv_dev_01": ReaderSpec(failed=("N1/26",), unverified=False)})
    result = run(make_records(1), tmp_path, graph, reps=1)
    assert not result.records[0].errored and result.reader_failed_to_unverified["total"] == 0


# --- E13: the reader limit ---------------------------------------------------------------------------------------------


def test_notams_left_unread_by_the_reader_limit_are_answered_not_errored_not_retried_not_counted(tmp_path):
    graph = ScriptedGraph(reader={"inv_dev_01": ReaderSpec(limited=("L1/26", "L2/26"))})
    result = run(make_records(2), tmp_path, graph, reps=1)
    first = result.records[0]
    assert not first.errored and first.attempts == 1 and first.backoff_waits == ()
    assert first.reader_limited == ("L1/26", "L2/26") and first.reader_failed_to_unverified == ()
    assert result.reader_failed_to_unverified["total"] == 0 and result.errored_runs == 0
    assert [c.example_id for c in graph.calls] == ["inv_dev_01", "inv_dev_02"]


def test_a_failed_read_and_a_limited_one_in_one_run_are_told_apart(tmp_path):
    graph = ScriptedGraph(reader={"inv_dev_01": ReaderSpec(failed=("F1/26",), limited=("L1/26",))})
    first = run(make_records(1), tmp_path, graph, reps=1).records[0]
    assert first.reader_failed_to_unverified == ("F1/26",) and first.reader_limited == ("L1/26",)


# --- fatal stops (GP9) -------------------------------------------------------------------------------------------------


class _RateLimit(Exception):
    status_code = 429


class APIConnectionError(Exception):
    pass


@pytest.mark.parametrize(("error", "words"), [(_RateLimit("slow down"), "rate limited"), (APIConnectionError("no route"), "endpoint unreachable")])
def test_a_rate_limit_or_an_unreachable_bridge_beyond_the_retries_stops_the_run_with_the_records_so_far(tmp_path, error, words):
    graph = ScriptedGraph(script={("inv_dev_03", 1): [error] * 4})
    sleeper = FakeSleeper()
    with pytest.raises(FatalRunError) as stopped:
        run(make_records(5), tmp_path, graph, sleep=sleeper)
    assert words in str(stopped.value) and "(last example: inv_dev_03)" in str(stopped.value)
    assert "\n" not in str(stopped.value)
    assert stopped.value.last_example_id == "inv_dev_03"
    assert [r.example_id for r in stopped.value.partial.records] == ["inv_dev_01", "inv_dev_02"]
    assert stopped.value.partial.errored_runs == 0
    assert sleeper.waits == list(BACKOFF_SECONDS)  # it was retried first
    assert len(graph.calls) == 2 + 4


def test_a_rate_limit_that_clears_within_the_retries_is_not_fatal(tmp_path):
    graph = ScriptedGraph(script={"inv_dev_01": [_RateLimit("slow"), _RateLimit("slow")]})
    result = run(make_records(1), tmp_path, graph, reps=1)
    assert result.records[0].attempts == 3 and not result.records[0].errored


def test_a_rate_limit_inside_another_exception_is_still_recognised(tmp_path):
    try:
        try:
            raise _RateLimit("429")
        except _RateLimit as inner:
            raise RuntimeError("wrapped") from inner
    except RuntimeError as outer:
        error = outer
    with pytest.raises(FatalRunError, match="rate limited"):
        run(make_records(1), tmp_path, ScriptedGraph(script={"inv_dev_01": [error] * 4}), reps=1)


def test_a_graph_that_cannot_be_built_is_fatal_before_any_call(tmp_path):
    def broken():
        raise ValueError("no model")

    with pytest.raises(FatalRunError, match=r"could not be built: ValueError: no model \(last example: none") as stopped:
        runner.run_experiment(make_records(1), experiment="v1-dev-b", graph_factory=broken, cache_dir=tmp_path)
    assert stopped.value.partial.records == []


def test_an_unreachable_langsmith_at_the_start_is_fatal_before_any_graph_call(tmp_path):
    client = RunRecordingClient()  # no dataset was uploaded: reading it fails
    graph = ScriptedGraph()
    with pytest.raises(FatalRunError, match="LangSmith could not open the experiment"):
        run(make_records(2), tmp_path, graph, client=client)
    assert graph.calls == []


def test_a_langsmith_failure_halfway_stops_the_run_and_keeps_the_records_so_far(tmp_path):
    client = RunRecordingClient(fail_on={"create_run": 3})
    client.create_dataset(dataset_name="flysafelz-dev")
    graph = ScriptedGraph()
    with pytest.raises(FatalRunError, match=r"LangSmith could not log the run.*\(last example: inv_dev_03\)") as stopped:
        run(make_records(4), tmp_path, graph, client=client)
    assert [r.example_id for r in stopped.value.partial.records] == ["inv_dev_01", "inv_dev_02", "inv_dev_03"]
    assert len(graph.calls) == 3


# --- SIGINT ------------------------------------------------------------------------------------------------------------


def test_a_keyboard_interrupt_stops_cleanly_is_not_errored_and_leaves_the_cache_file(tmp_path):
    graph = ScriptedGraph(script={("inv_dev_03", 1): [KeyboardInterrupt()]}, touch_cache=True)
    sleeper = FakeSleeper()
    result = run(make_records(5), tmp_path, graph, sleep=sleeper)
    assert result.interrupted is True
    assert [r.example_id for r in result.records] == ["inv_dev_01", "inv_dev_02"]
    assert result.errored_runs == 0 and not result.incomplete
    assert len(graph.calls) == 3 and sleeper.waits == []  # no retry, nothing after it
    assert (tmp_path / "v1-dev-test.sqlite").exists()
    assert any("INTERRUPTED" in line for line in result.summary_lines())


def test_a_real_sigint_during_the_real_timeout_stops_the_run(tmp_path):
    def interrupt(call):
        os.kill(os.getpid(), signal.SIGINT)
        time.sleep(5)

    graph = ScriptedGraph(script={"inv_dev_02": [interrupt]}, touch_cache=True)
    original = signal.getsignal(signal.SIGINT)
    result = run(make_records(3), tmp_path, graph, reps=1, timeout=runner.call_with_timeout)
    assert signal.getsignal(signal.SIGINT) is original
    assert result.interrupted and [r.example_id for r in result.records] == ["inv_dev_01"] and result.errored_runs == 0
    assert (tmp_path / "v1-dev-test.sqlite").exists()
    assert signal.getitimer(signal.ITIMER_REAL) == (0.0, 0.0)


def test_a_closing_message_made_of_content_blocks_is_read_as_its_text(tmp_path):
    def blocks(call):
        reply = AIMessage(content=[{"type": "text", "text": "GO. "}, {"type": "text", "text": "Blocks."}], usage_metadata={"input_tokens": 1, "output_tokens": 1, "total_tokens": 2})
        call.model_call(reply)
        return {"messages": [HumanMessage("q"), reply]}

    result = run(make_records(1), tmp_path, ScriptedGraph(script={"inv_dev_01": [blocks]}), reps=1)
    assert result.records[0].answer == "GO. Blocks."


# --- GP9: a failure the agent hides (its middleware answers for a model call that failed) ------------------------------------


def _bridge_error(cls, status):
    return cls(f"Error code: {status}", response=httpx.Response(status, request=httpx.Request("POST", "http://bridge.invalid/v1")), body=None)


def hidden(error, *, node="model"):
    """An outcome like the real agent's when a model call fails: the call's failure reaches the callbacks, and the state holds the middleware's code-built answer."""

    def outcome(call):
        call.model_error(error, node=node)
        fallback = AIMessage("The assistant is unavailable, so this spot could not be checked.")
        return {"messages": [HumanMessage("q"), fallback], "answer": fallback.content, "guard": "fallback"}

    return outcome


@pytest.mark.parametrize(
    ("error", "words"),
    [
        (_bridge_error(openai.RateLimitError, 429), "rate limited"),
        (openai.APIConnectionError(request=httpx.Request("POST", "http://bridge.invalid")), "endpoint unreachable"),
        (openai.APITimeoutError(request=httpx.Request("POST", "http://bridge.invalid")), "timed out"),  # an APIConnectionError subclass, named as a timeout
    ],
)
def test_a_429_a_refused_connection_or_a_timeout_the_agent_hid_is_retried_and_then_stops_the_run(tmp_path, error, words):
    graph = ScriptedGraph(script={"inv_dev_02": [hidden(error)] * 4})
    sleeper = FakeSleeper()
    with pytest.raises(FatalRunError, match=words) as stopped:
        run(make_records(4), tmp_path, graph, reps=1, sleep=sleeper)
    assert stopped.value.last_example_id == "inv_dev_02" and [r.example_id for r in stopped.value.partial.records] == ["inv_dev_01"]
    assert sleeper.waits == list(BACKOFF_SECONDS) and len(graph.calls) == 1 + 4


def test_a_hidden_429_that_clears_on_a_retry_is_scored_from_the_attempt_that_worked(tmp_path):
    graph = ScriptedGraph(script={"inv_dev_01": [hidden(_bridge_error(openai.RateLimitError, 429))]})
    [record] = run(make_records(1), tmp_path, graph, reps=1).records
    assert record.attempts == 2 and not record.errored and record.model_failures == () and record.output["guard"] == "passed"


def test_a_hidden_5xx_keeps_the_fallback_answer_for_scoring_and_puts_the_failure_on_the_record(tmp_path):
    graph = ScriptedGraph(script={"inv_dev_01": [hidden(_bridge_error(openai.InternalServerError, 503))]})
    result = run(make_records(2), tmp_path, graph, reps=1)
    first = result.records[0]
    assert not first.errored and first.attempts == 1 and first.output["guard"] == "fallback"
    assert len(first.model_failures) == 1 and "InternalServerError" in first.model_failures[0]
    assert result.model_failed_runs == 1 and result.errored_runs == 0 and len(result.scored) == 2


def test_a_failure_in_the_readers_node_is_the_readers_and_does_not_stop_the_run(tmp_path):
    graph = ScriptedGraph(script={"inv_dev_01": [hidden(_bridge_error(openai.RateLimitError, 429), node="tools")] * 4})
    [record] = run(make_records(1), tmp_path, graph, reps=1).records
    assert record.attempts == 1 and not record.errored and record.model_failures == ()


def test_an_error_that_is_not_a_model_failure_is_not_taken_for_one(tmp_path):
    """A 401 (a wrong key) is raised by the agent, not hidden; if one reached the callbacks alone it is still not a bridge outage to stop on."""
    graph = ScriptedGraph(script={"inv_dev_01": [hidden(_bridge_error(openai.AuthenticationError, 401))]})
    [record] = run(make_records(1), tmp_path, graph, reps=1).records
    assert record.attempts == 1 and record.model_failures == ()
