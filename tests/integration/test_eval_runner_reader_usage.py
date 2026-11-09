"""The NOTAM reader's token usage in a run (5a PR-10.F1, E4, G16): counted per example, strict, and never a zero for a reading the cache returned.

The runner on ticket 4's real graph (tests/graph_helpers.py) with the live reader: the reader reads its NOTAMs on a thread pool inside a tool, where LangChain does not carry
the run's callbacks, so ModelCalls cannot see those calls. `runner.ReaderMeter` wraps the reader's chat model instead. Only the chat models are fakes; sockets stay off.
"""

import threading
from pathlib import Path
from typing import Any

import httpx
import openai
import pytest
from eval_runner_helpers import FakeSleeper, RecordingTimeout
from fakes import FakeToolChatModel
from graph_helpers import BLUE_MILL_CASE, RaisingModel, agent_graph
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field
from reader_replies import fixture_readings, model_reply
from reader_set_helpers import ScriptedChat
from tool_helpers import BLUE_MILL_LZ, BLUE_MILL_WINDOW
from unit.reader_helpers import blue_mill_notams

from evals.agent import runner
from flysafelz.agent.sources import SourceCache

AGENT_USAGE = {"input_tokens": 5, "output_tokens": 1, "total_tokens": 6}
READER_USAGE = {"input_tokens": 7, "output_tokens": 3, "total_tokens": 10}
REQUEST = httpx.Request("POST", "http://reader.invalid/v1/chat/completions")


class MeteredScript(BaseChatModel):
    """The reader's chat model: ScriptedChat's per-NOTAM replies, each with usage from `usages` (a list consumed in call order; an empty entry is a reply with none)."""

    inner: Any = None
    usages: list[dict[str, int] | None] = Field(default_factory=list)
    model_name: str = "fake-reader-model"
    seen: list[int] = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "metered-script"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        reply = self.inner.invoke(messages)
        with LOCK:
            usage = self.usages[len(self.seen)] if len(self.seen) < len(self.usages) else dict(READER_USAGE)
            self.seen.append(1)
        return ChatResult(generations=[ChatGeneration(message=AIMessage(reply.content, usage_metadata=usage) if usage else AIMessage(reply.content))])


LOCK = threading.Lock()


def blue_mill_reader(usages: list[dict[str, int] | None] | None = None) -> MeteredScript:
    by_id, readings = blue_mill_notams(), fixture_readings()
    scripts = {key: [model_reply(by_id[key.replace("_", "/")], readings[key.replace("_", "/")])] for key in ("H7171_26", "J3857_26")}
    return MeteredScript(inner=ScriptedChat(list(by_id.values()), **scripts), usages=usages or [], seen=[])


def agent_steps() -> list[AIMessage]:
    ask = AIMessage("", tool_calls=[{"name": "get_notams", "args": BLUE_MILL_WINDOW, "id": "n-1"}], usage_metadata=dict(AGENT_USAGE))
    return [ask, AIMessage("[path: explain]\nThe NOTAMs are listed above.", usage_metadata=dict(AGENT_USAGE))]


def example(number: int) -> dict:
    return {
        "id": f"inv_reader_{number:02d}",
        "inputs": {**BLUE_MILL_LZ, "question": "Can I fly here today?", "now": "2026-10-31T07:30:00+00:00"},
        "fixtures": {"case": BLUE_MILL_CASE, "drop": []},
    }


class NoFetch:
    def __call__(self, *args):
        raise AssertionError("a case-mode run fetched live")


def run_reader(tmp_path: Path, reader: BaseChatModel, *, count: int = 1, reps: int = 1, agent=None, meter: bool = True, **options):
    """Each attempt gets a fresh agent model (a script is used up by one run); the reader is shared, as the real one's cache is."""
    the_meter = runner.ReaderMeter()

    def build():
        source_cache = SourceCache(fetch=NoFetch(), reader_model=lambda: the_meter.wrap(reader) if meter else reader)
        return agent_graph(agent or ScriptedAgents(), source_cache=source_cache)

    return runner.run_experiment(
        [example(n) for n in range(1, count + 1)],
        experiment="v1-dev-reader",
        graph_factory=build,
        cache_dir=tmp_path,
        reps=reps,
        sleep=FakeSleeper(),
        timeout=RecordingTimeout(),
        reader_meter=the_meter if meter else None,
        **options,
    )


class ScriptedAgents(FakeToolChatModel):
    """An agent model that asks for get_notams and then answers, again for every run (the script cycles)."""

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        done = sum(1 for m in messages if m.type == "ai")
        return ChatResult(generations=[ChatGeneration(message=agent_steps()[done % 2])])


def test_the_readers_replies_are_counted_per_example_apart_from_the_agents(tmp_path):
    reader = blue_mill_reader()
    [record] = run_reader(tmp_path, reader).records
    calls = len(reader.seen)
    assert calls > 0 and not record.errored
    assert record.usage == {"input_tokens": 10, "output_tokens": 2, "total_tokens": 12, "model_calls": 2, "calls_without_usage": 0}  # the agent's two, as before
    assert record.reader_usage == {"input_tokens": 7 * calls, "output_tokens": 3 * calls, "total_tokens": 10 * calls, "model_calls": calls, "calls_without_usage": 0}


def test_a_stored_reading_made_no_call_so_a_later_rep_counts_none_and_not_a_zero(tmp_path):
    reader = blue_mill_reader()
    first, second = run_reader(tmp_path, reader, reps=2).records
    assert first.reader_usage is not None and len(reader.seen) == first.reader_usage["model_calls"]
    assert second.reader_usage is None  # every reading came from the experiment's cache file: no call, nothing counted
    assert second.usage["model_calls"] == 2


def test_strict_mode_does_not_stop_a_run_whose_later_rep_read_only_stored_readings(tmp_path):
    result = run_reader(tmp_path, blue_mill_reader(), reps=2, strict_usage=True)
    assert [r.errored for r in result.records] == [False, False]


def test_a_reader_reply_without_usage_stops_a_strict_run_naming_the_example(tmp_path):
    reader = blue_mill_reader(usages=[dict(READER_USAGE), None])
    with pytest.raises(runner.UsageMissingError) as stopped:
        run_reader(tmp_path, reader, count=2, strict_usage=True)
    assert str(stopped.value) == f"{runner.STRICT_USAGE_LATER}{runner.STRICT_USAGE_READER} (last example: inv_reader_01)"
    assert stopped.value.partial.records == []


def test_a_bare_reader_reply_after_an_agent_reply_with_usage_is_a_later_call(tmp_path):
    """The agent's tool request (with usage) comes before the reader's first reply, so the reader's gap is not the experiment's first call."""
    reader = blue_mill_reader(usages=[None])
    with pytest.raises(runner.UsageMissingError) as stopped:
        run_reader(tmp_path, reader, strict_usage=True)
    assert runner.STRICT_USAGE_LATER in str(stopped.value) and str(stopped.value).endswith("(last example: inv_reader_01)")


def test_without_strict_a_reader_reply_without_usage_is_counted_as_a_call_without_usage_never_as_zero(tmp_path):
    reader = blue_mill_reader(usages=[dict(READER_USAGE), None])
    [record] = run_reader(tmp_path, reader).records
    calls = len(reader.seen)
    assert calls >= 2 and record.reader_usage["model_calls"] == calls and record.reader_usage["calls_without_usage"] == 1
    assert record.reader_usage["input_tokens"] == 7 * (calls - 1)


def test_a_reader_whose_call_fails_has_no_reply_to_count(tmp_path):
    error = openai.RateLimitError("Error code: 429", response=httpx.Response(429, request=REQUEST), body=None)
    [record] = run_reader(tmp_path, RaisingModel(error=error), strict_usage=True).records
    assert not record.errored and record.reader_usage is None and record.usage["model_calls"] == 2


def test_a_reader_with_no_meter_is_not_counted_and_the_run_is_as_before(tmp_path):
    [record] = run_reader(tmp_path, blue_mill_reader(), meter=False).records
    assert record.reader_usage is None and record.usage["model_calls"] == 2


def test_the_meter_is_detached_when_the_example_ends(tmp_path):
    the_meter = runner.ReaderMeter()
    calls = runner.ModelCalls()
    the_meter.attach(calls)
    the_meter.record(AIMessage("x", usage_metadata=dict(READER_USAGE)))
    the_meter.detach()
    the_meter.record(AIMessage("late", usage_metadata=dict(READER_USAGE)))
    assert calls.reader_usage() == {"input_tokens": 7, "output_tokens": 3, "total_tokens": 10, "model_calls": 1, "calls_without_usage": 0}


def test_the_wrapped_model_keeps_the_names_the_cache_key_reads_and_passes_the_arguments_through():
    seen = {}

    class Model:
        model_name = "the-reader"

        def invoke(self, messages, **kwargs):
            seen.update(kwargs, messages=messages)
            return AIMessage("ok", usage_metadata=dict(READER_USAGE))

    the_meter = runner.ReaderMeter()
    calls = runner.ModelCalls()
    the_meter.attach(calls)
    wrapped = the_meter.wrap(Model())
    assert wrapped.model_name == "the-reader" and wrapped.invoke(["m"], response_format={"a": 1}, timeout=9).content == "ok"
    assert seen == {"messages": ["m"], "response_format": {"a": 1}, "timeout": 9}
    with pytest.raises(AttributeError):
        wrapped.no_such_attribute  # noqa: B018 - the access is the test
    assert calls.reader_usage()["total_tokens"] == 10
