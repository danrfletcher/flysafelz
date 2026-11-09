"""The runner on ticket 4's real graph (5a PR-4.F1, GP9, G7, E4): only the chat models are scripted.

The agent's middleware catches a failed model call (a 429, a refused connection, a timeout, a 5xx) and answers for the model, so nothing escapes `graph.invoke`.
It also drops a draft the guard rejects before it reaches the state. A fake graph that raises, or whose messages hold every model response, cannot show how the
runner copes with either. These tests run `runner.run_experiment` over `build_agent` itself (tests/graph_helpers.py), with chat models that fail the way the bridge does.
Sockets stay off, no key is read, and the cache path is under tmp_path.
"""

from pathlib import Path

import httpx
import openai
import pytest
from eval_runner_helpers import FakeSleeper, RecordingTimeout
from graph_helpers import BLUE_MILL_CASE, RaisingModel, ScriptedModel, agent_graph
from langchain_core.messages import AIMessage
from tool_helpers import BLUE_MILL_LZ, BLUE_MILL_WINDOW

from evals.agent import runner
from flysafelz.agent.sources import SourceCache

REQUEST = httpx.Request("POST", "http://bridge.invalid/v1/chat/completions")
USAGE = {"input_tokens": 50, "output_tokens": 5, "total_tokens": 55}
PLAIN = "[path: explain]\nThis is a plain reply."
BANNED = "[path: explain]\nIt is safe to fly."  # a banned phrase on every path: the guard rejects it and asks again


def status_error(cls: type[openai.APIStatusError], status: int) -> openai.APIStatusError:
    return cls(f"Error code: {status}", response=httpx.Response(status, request=REQUEST), body=None)


def rate_limit() -> openai.RateLimitError:
    return status_error(openai.RateLimitError, 429)  # type: ignore[return-value]


def refused() -> openai.APIConnectionError:
    return openai.APIConnectionError(request=REQUEST)


def server_error() -> openai.InternalServerError:
    return status_error(openai.InternalServerError, 500)  # type: ignore[return-value]


def reply(text: str, *, usage: bool = True) -> AIMessage:
    return AIMessage(text, usage_metadata=dict(USAGE) if usage else None)


def tool_round(name: str) -> AIMessage:
    """A model response that asks for one tool, with usage."""
    return AIMessage("", tool_calls=[{"name": name, "args": BLUE_MILL_WINDOW, "id": f"{name}-1"}], usage_metadata=dict(USAGE))


def example(number: int, **inputs) -> dict:
    """A dev-shaped record on the committed Blue Mill case, at its frozen clock."""
    given = {**BLUE_MILL_LZ, "question": "Can I fly here today?", "now": "2026-10-31T07:30:00+00:00", **inputs}
    return {"id": f"inv_real_{number:02d}", "inputs": given, "fixtures": {"case": BLUE_MILL_CASE, "drop": []}}


class NoFetch:
    def __call__(self, *args):
        raise AssertionError("a case-mode run fetched live")


def run_real(model, tmp_path: Path, records: list[dict], *, reader_model=None, reps: int = 1, sleeper: FakeSleeper | None = None):
    cache = SourceCache(fetch=NoFetch(), reader_model=reader_model) if reader_model else SourceCache(fetch=NoFetch())
    return runner.run_experiment(
        records,
        experiment="v1-dev-real",
        graph_factory=lambda: agent_graph(model, source_cache=cache),
        cache_dir=tmp_path,
        reps=reps,
        sleep=sleeper or FakeSleeper(),
        timeout=RecordingTimeout(),
    )


# --- R1: a bridge that is down or rate limiting stops the run, though the agent hides it -------------------------------------


@pytest.mark.parametrize(("error", "words"), [(rate_limit(), "rate limited"), (refused(), "endpoint unreachable"), (openai.APITimeoutError(request=REQUEST), "timed out")])
def test_a_bridge_that_fails_every_call_stops_the_run_after_the_retries_with_one_message(tmp_path, error, words):
    model, sleeper = RaisingModel(error=error), FakeSleeper()
    with pytest.raises(runner.FatalRunError) as stopped:
        run_real(model, tmp_path, [example(1), example(2)], sleeper=sleeper)
    assert words in str(stopped.value) and "\n" not in str(stopped.value)
    assert str(stopped.value).endswith("(last example: inv_real_01)")
    assert model.count == 1 + runner.RETRIES and sleeper.waits == list(runner.BACKOFF_SECONDS)  # retried first, and the second example never started
    assert stopped.value.partial.records == []


def test_an_outage_halfway_stops_the_run_and_keeps_the_runs_before_it(tmp_path):
    model = ScriptedModel(script=[reply(PLAIN), *[rate_limit()] * (1 + runner.RETRIES)])
    with pytest.raises(runner.FatalRunError, match=r"rate limited.*\(last example: inv_real_02\)") as stopped:
        run_real(model, tmp_path, [example(1), example(2), example(3)])
    assert [r.example_id for r in stopped.value.partial.records] == ["inv_real_01"]
    assert len(model.calls) == 1 + 1 + runner.RETRIES


def test_a_rate_limit_that_clears_within_the_retries_is_scored_with_its_attempts_recorded(tmp_path):
    model, sleeper = ScriptedModel(script=[rate_limit(), rate_limit(), reply(PLAIN)]), FakeSleeper()
    [record] = run_real(model, tmp_path, [example(1)], sleeper=sleeper).records
    assert not record.errored and record.attempts == 3 and record.backoff_waits == runner.BACKOFF_SECONDS[:2] and sleeper.waits == list(runner.BACKOFF_SECONDS[:2])
    assert record.output["guard"] == "passed" and record.answer.startswith("This is a plain reply.")
    assert record.model_failures == ()  # the kept attempt had no failure; the two before it were retried away
    assert record.usage == {"input_tokens": 50, "output_tokens": 5, "total_tokens": 55, "model_calls": 1, "calls_without_usage": 0}


def test_a_5xx_is_a_fail_closed_failure_inside_the_agent_so_its_guarded_answer_is_kept_for_scoring(tmp_path):
    model = ScriptedModel(script=[server_error(), reply(PLAIN)])
    sleeper = FakeSleeper()
    first, second = run_real(model, tmp_path, [example(1), example(2)], sleeper=sleeper).records
    assert not first.errored and first.attempts == 1 and sleeper.waits == []  # not retried, not errored
    assert first.output["guard"] == "fallback" and "unavailable" in first.answer
    assert len(first.model_failures) == 1 and "InternalServerError" in first.model_failures[0]
    assert second.output["guard"] == "passed" and second.model_failures == ()


def test_the_summary_says_how_many_kept_runs_are_the_fallback_after_a_failed_model_call(tmp_path):
    result = run_real(ScriptedModel(script=[server_error(), reply(PLAIN)]), tmp_path, [example(1), example(2)])
    assert result.model_failed_runs == 1
    assert any("fallback answer because its model call failed" in line and line.endswith(": 1") for line in result.summary_lines())


def test_a_reader_that_is_rate_limited_is_the_readers_failure_not_the_bridges(tmp_path):
    """E7: the NOTAM reader's own 429 becomes UNVERIFIED inside the agent. It is not the agent's model failing, so the run is neither retried nor stopped."""
    model = ScriptedModel(script=[tool_round("get_notams"), reply(PLAIN)])
    result = run_real(model, tmp_path, [example(1)], reader_model=lambda: RaisingModel(error=rate_limit()))
    [record] = result.records
    assert not record.errored and record.attempts == 1 and record.model_failures == ()
    assert record.usage["model_calls"] == 2  # the agent's two; the reader's calls are not counted


# --- R2: the first-response usage check looks at a real model response only ------------------------------------------------


@pytest.mark.parametrize("error", [server_error(), pytest.param(None, id="uk-only")])
def test_a_first_example_the_model_never_answered_does_not_trip_the_usage_check(tmp_path, error):
    """A 5xx on the first example gives the agent's own 'unavailable' message, and a request outside the UK is answered by code: neither has usage, neither is the bridge's."""
    if error is None:
        model, first = ScriptedModel(script=[reply(PLAIN)]), example(1, lat=48.85, lon=2.35)
    else:
        model, first = ScriptedModel(script=[error, reply(PLAIN)]), example(1)
    records = run_real(model, tmp_path, [first, example(2)]).records
    assert [r.errored for r in records] == [False, False]
    assert records[0].usage is None  # no model response: no usage, and never zeros
    assert records[1].usage["total_tokens"] == 55


def test_the_check_still_stops_the_run_when_the_first_real_response_has_no_usage(tmp_path):
    model = ScriptedModel(script=[server_error(), reply(PLAIN, usage=False)])
    with pytest.raises(runner.UsageMissingError, match=r"AGENT_BASE_URL.*\(last example: inv_real_02\)") as stopped:
        run_real(model, tmp_path, [example(1), example(2), example(3)])
    assert [r.example_id for r in stopped.value.partial.records] == ["inv_real_01"]
    assert len(model.calls) == 2  # the third example never started


# --- R3: every model call is counted, the guard's discarded drafts too ------------------------------------------------------


def test_the_tokens_of_a_draft_the_guard_threw_away_are_counted(tmp_path):
    model = ScriptedModel(script=[reply(BANNED), reply(PLAIN)])
    [record] = run_real(model, tmp_path, [example(1)]).records
    assert record.output["guard"] == "retried"
    assert [m.type for m in record.output["messages"]].count("ai") == 1  # the rejected draft is not in the state ...
    assert record.usage == {"input_tokens": 100, "output_tokens": 10, "total_tokens": 110, "model_calls": 2, "calls_without_usage": 0}  # ... but it was paid for


def test_a_run_with_a_tool_round_and_a_guard_retry_counts_all_three_calls(tmp_path):
    model = ScriptedModel(script=[tool_round("assess"), reply(BANNED), reply(PLAIN)])
    [record] = run_real(model, tmp_path, [example(1)], reader_model=lambda: RaisingModel(error=rate_limit())).records
    assert record.usage["model_calls"] == 3 and record.usage["total_tokens"] == 165
    assert not record.errored


def test_the_experiment_total_adds_up_every_call_of_every_run(tmp_path):
    model = ScriptedModel(script=[reply(BANNED), reply(PLAIN), reply(PLAIN), reply(PLAIN), reply(PLAIN)])
    result = run_real(model, tmp_path, [example(1), example(2)], reps=2)
    assert [r.usage["model_calls"] for r in result.records] == [2, 1, 1, 1]
    assert result.usage_total()["total_tokens"] == 55 * 5


def test_the_tokens_of_the_attempts_before_a_retry_are_part_of_the_run(tmp_path):
    """The first attempt spent a call and then hit a 429; the retry succeeded. Both were paid for."""
    model = ScriptedModel(script=[tool_round("get_weather"), rate_limit(), reply(PLAIN)])
    [record] = run_real(model, tmp_path, [example(1)]).records
    assert record.attempts == 2 and record.usage["model_calls"] == 2 and record.usage["total_tokens"] == 110
