"""Mocks and fixtures for the eval runner's tests (5a PR-4.F1). Everything here is invented; nothing is sent and no key is read.

make_records(n, ...)         loader-shaped records (id, inputs, fixtures, reference, metadata) with invented ids, cases and a fixed `now` each
ScriptedGraph(...)           a fake agent graph: per-(example, rep) or per-example outcomes consumed one attempt at a time, usage on or off in either style,
                             reader-status injection through real get_notams and assess artifacts, a record of every call and of the peak in flight, and the
                             callbacks a real chat model run makes (Call.model_call, Call.model_error), which are how the runner sees usage and hidden failures
FakeClock, FakeSleeper       a clock that only the sleeper moves, and a sleeper that records the waits and never sleeps
RecordingTimeout             the timeout function, recording the seconds it was given and calling straight through
RunRecordingClient           tests' RecordingClient plus create_project and create_run (what the runner sends to LangSmith)
FAKE_ENV                     every AGENT_*, READER_*, JUDGE_* and LANGSMITH_* variable set to an obviously fake value, for the command's settings check
"""

import threading
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, LLMResult

from evals.agent.recording_client import RecordingClient
from flysafelz.agent.tools.artifacts import AssessArtifact, AssessNotam, AssessSegment, NotamCounts, NotamItem, NotamsArtifact, SourceInfo

FAKE_ENV = {
    "AGENT_BASE_URL": "http://agent.invalid/v1",
    "AGENT_MODEL": "fake-agent-model",
    "AGENT_API_KEY": "fake-key",
    "READER_PROVIDER": "fake",
    "READER_BASE_URL": "http://reader.invalid/v1",
    "READER_API_KEY": "fake-key",
    "READER_MODEL": "fake-reader-model",
    "LANGSMITH_API_KEY": "fake-key",
    "JUDGE_PROVIDER": "fake",
    "JUDGE_BASE_URL": "http://judge.invalid/v1",
    "JUDGE_API_KEY": "fake-key",
    "JUDGE_MODEL": "fake-judge-model",
}
FROZEN = "2026-01-02T03:04:00Z"
ANSWER = "GO WITH CONDITIONS. Invented answer for a test."
WINDOW = (datetime(2026, 1, 2, 4, tzinfo=UTC), datetime(2026, 1, 2, 6, tzinfo=UTC))


def make_records(n: int = 4, *, cases: int = 3, drops: Mapping[int, list[str]] | None = None) -> list[dict[str, Any]]:
    """`n` dev records inv_dev_01…, example i on case inv_case_{i % cases}, each with its own `now` (a minute apart), question and coordinates."""
    drops = drops or {}
    records = []
    for i in range(1, n + 1):
        records.append(
            {
                "id": f"inv_dev_{i:02d}",
                "inputs": {"lat": 51.0 + i / 100, "lon": -1.0 - i / 100, "question": f"Invented question {i}: can I fly?", "now": f"2026-01-02T03:{i:02d}:00Z"},
                "fixtures": {"case": f"inv_case_{i % cases}", "drop": list(drops.get(i, []))},
                "reference": {"path": "assess"},
                "metadata": {"split": "dev", "path": "assess", "compound": False, "parts": []},
            }
        )
    return records


# --- the clock and the sleeper -----------------------------------------------------------------------------------------


class FakeClock:
    """A monotonic clock that stands still until a sleeper (or a test) moves it."""

    def __init__(self) -> None:
        self.now = 100.0

    def __call__(self) -> float:
        return self.now


class FakeSleeper:
    """Records each wait and moves the clock by it; sleeps for no time at all."""

    def __init__(self, clock: FakeClock | None = None) -> None:
        self.clock = clock
        self.waits: list[float] = []

    def __call__(self, seconds: float) -> None:
        self.waits.append(seconds)
        if self.clock is not None:
            self.clock.now += seconds


class RecordingTimeout:
    """Stands in for runner.call_with_timeout: records the seconds each call was given, then calls straight through."""

    def __init__(self) -> None:
        self.seconds: list[float] = []

    def __call__(self, call: Callable[[], Any], seconds: float) -> Any:
        self.seconds.append(seconds)
        return call()


# --- the LangSmith client ----------------------------------------------------------------------------------------------


class RunRecordingClient(RecordingClient):
    """The loader's RecordingClient, plus the two calls the runner adds: create_project (the experiment) and create_run (one finished run)."""

    def __init__(self, fail_on: Mapping[str, int] | None = None) -> None:
        super().__init__(fail_on)
        self.projects: list[dict[str, Any]] = []
        self.runs: list[dict[str, Any]] = []

    def create_project(self, *, project_name: str, reference_dataset_id: Any = None, description: str | None = None, metadata: Mapping[str, Any] | None = None) -> None:
        self._record("create_project", project_name=project_name)
        self.projects.append({"project_name": project_name, "reference_dataset_id": reference_dataset_id, "description": description, "metadata": dict(metadata or {})})

    def create_run(self, **kwargs: Any) -> None:
        self._record("create_run", name=kwargs.get("name"))
        self.runs.append(kwargs)


# --- artifacts the real tools would have returned ----------------------------------------------------------------------


def _source(name: str) -> SourceInfo:
    return SourceInfo(source=name, state="ok", reason="", fetched_at=None, check_by_hand=None)  # type: ignore[arg-type]


def notams_artifact(unread: Mapping[str, str]) -> NotamsArtifact:
    """get_notams' artifact: one entry per id, each unread for the reason given (the tool's own words: "reader unavailable", "reader failed", "reader limit")."""
    items = [NotamItem(id=i, action="CHECK_SUP", reason="", bearing_deg=None, reading=None, unread_reason=why, e_text="") for i, why in unread.items()]
    start, end = WINDOW
    return NotamsArtifact(
        source=_source("notams"),
        window_start=start,
        window_end=end,
        window_assumed=False,
        overlays=[],
        counts=NotamCounts(within=len(items), awareness=0, ignored=0),
        notams=items,
        assumptions=[],
        issued_at=None,
    )


def assess_artifact(unverified_by: tuple[str, ...]) -> AssessArtifact:
    """assess' artifact with one UNVERIFIED segment that holds `unverified_by` (and, when empty, one PILOT_DISCRETION segment)."""
    notams = [AssessNotam(id=i, action="UNREAD_OVER_LZ", bearing_deg=None, reason="") for i in unverified_by]
    segment = AssessSegment.model_construct(verdict="UNVERIFIED" if unverified_by else "PILOT_DISCRETION", notams=notams)
    return AssessArtifact.model_construct(segments=[segment])


@dataclass(frozen=True)
class ReaderSpec:
    """What the reader did in a scripted run. `failed` ids are unread ("reader unavailable") and, with `unverified`, sit in an UNVERIFIED segment; `limited` are unread by the limit."""

    failed: tuple[str, ...] = ()
    limited: tuple[str, ...] = ()
    unverified: bool = True
    notams_calls: int = 1  # how many times get_notams ran: each names the same NOTAMs


# --- the fake graph ----------------------------------------------------------------------------------------------------

OK = "ok"
NO_ANSWER = "no_answer"


@dataclass
class Call:
    """One graph.invoke: what it was given and when it ran."""

    request: dict[str, Any]
    config: dict[str, Any]
    context: Any
    example_id: str
    rep: int
    order: int  # 0-based position among all calls

    def _handlers(self) -> list[Any]:
        return list(self.config.get("callbacks") or [])

    def model_call(self, message: AIMessage, *, node: str = "model") -> None:
        """One chat model run that ended with `message`, announced to the config's callbacks as LangChain does (`node` is the LangGraph node it runs in)."""
        run_id = uuid.uuid4()
        for handler in self._handlers():
            handler.on_chat_model_start({}, [[]], run_id=run_id, metadata={"langgraph_node": node})
            handler.on_llm_end(LLMResult(generations=[[ChatGeneration(message=message)]]), run_id=run_id)

    def model_error(self, error: BaseException, *, node: str = "model") -> None:
        """One chat model run that failed with `error`, announced to the config's callbacks."""
        run_id = uuid.uuid4()
        for handler in self._handlers():
            handler.on_chat_model_start({}, [[]], run_id=run_id, metadata={"langgraph_node": node})
            handler.on_llm_error(error, run_id=run_id)


@dataclass
class ScriptedGraph:
    """A fake of the compiled agent graph: `invoke(request, config, *, context)` returns a state like the real one's.

    script       {example id | (example id, rep): [outcome, ...]}: the n-th attempt at that example takes the n-th outcome; none left is OK. An outcome is OK,
                 NO_ANSWER, an exception instance (raised: an Exception, ExampleTimeout, KeyboardInterrupt), or a callable(call) -> state.
    usage        "metadata" (usage_metadata on every model response), "api" (response_metadata["usage"] in the API's own names), "none", or
                 "first_missing" (the first model response has none and later ones do)
    reader       {example id: ReaderSpec}: the get_notams and assess artifacts of that example's run
    touch_cache  create the run's reader cache file, as the real reader does on its first write
    """

    script: dict[Any, list[Any]] = field(default_factory=dict)
    usage: str = "metadata"
    reader: dict[str, ReaderSpec] = field(default_factory=dict)
    touch_cache: bool = False
    calls: list[Call] = field(default_factory=list)
    events: list[tuple[str, int]] = field(default_factory=list)
    in_flight: int = 0
    peak: int = 0

    def __post_init__(self) -> None:
        self._taken: dict[Any, int] = {}
        self._lock = threading.Lock()

    def _usage_kwargs(self, index: int) -> dict[str, Any]:
        if self.usage == "none" or (self.usage == "first_missing" and index == 0):
            return {}
        if self.usage == "api":
            return {"response_metadata": {"usage": {"prompt_tokens": 100, "completion_tokens": 20}}}
        return {"usage_metadata": {"input_tokens": 100, "output_tokens": 20, "total_tokens": 120}}

    def _state(self, call: Call) -> dict[str, Any]:
        question = call.request["messages"][0].content
        spec = self.reader.get(call.example_id)
        messages: list[Any] = [HumanMessage(question)]
        if spec is None:
            messages.append(AIMessage(ANSWER, **self._usage_kwargs(0)))
            call.model_call(messages[-1])
        else:
            unread = {**dict.fromkeys(spec.failed, "reader unavailable"), **dict.fromkeys(spec.limited, "reader limit")}
            messages.append(AIMessage("", tool_calls=[{"name": "get_notams", "args": {}, "id": "t1"}], **self._usage_kwargs(0)))
            call.model_call(messages[-1])
            for number in range(spec.notams_calls):
                messages.append(ToolMessage("notams", tool_call_id=f"n{number}", artifact=notams_artifact(unread)))
            messages.append(ToolMessage("assess", tool_call_id="a1", artifact=assess_artifact(spec.failed if spec.unverified else ())))
            messages.append(AIMessage(ANSWER, **self._usage_kwargs(1)))
            call.model_call(messages[-1])
        return {"messages": messages, "answer": ANSWER, "guard": "passed"}

    def _outcome(self, example_id: str, rep: int) -> Any:
        for key in ((example_id, rep), example_id):
            if key in self.script:
                taken = self._taken.get(key, 0)
                self._taken[key] = taken + 1
                return self.script[key][taken] if taken < len(self.script[key]) else OK
        return OK

    def invoke(self, request: dict[str, Any], config: dict[str, Any], *, context: Any) -> dict[str, Any]:
        meta = config["metadata"]
        with self._lock:
            call = Call(request, config, context, meta["example_id"], meta["rep"], len(self.calls))
            self.calls.append(call)
            self.in_flight += 1
            self.peak = max(self.peak, self.in_flight)
            self.events.append(("start", call.order))
        try:
            if self.touch_cache:
                Path(context.reader_cache).parent.mkdir(parents=True, exist_ok=True)
                Path(context.reader_cache).touch()
            outcome: Any = self._outcome(call.example_id, call.rep)
            if isinstance(outcome, BaseException):
                raise outcome
            if callable(outcome):
                return outcome(call)
            state = self._state(call)
            if outcome == NO_ANSWER:
                state["answer"] = "  "
                state["messages"] = [state["messages"][0], AIMessage("", **self._usage_kwargs(0))]
            return state
        finally:
            with self._lock:
                self.in_flight -= 1
                self.events.append(("end", call.order))

    def factory(self) -> "ScriptedGraph":
        """The graph factory the runner is given."""
        return self

    def order(self) -> list[tuple[str, int]]:
        """[(example id, rep)] in the order the graph was called."""
        return [(c.example_id, c.rep) for c in self.calls]


def run(records: list[dict[str, Any]], cache_dir: Path, graph: ScriptedGraph, *, experiment: str = "v1-dev-test", **options: Any) -> Any:
    """runner.run_experiment on the fake graph, with a fake clock, a sleeper that never sleeps and a timeout that only records, unless the test passes its own."""
    from evals.agent import runner

    clock = options.setdefault("clock", FakeClock())
    options.setdefault("sleep", FakeSleeper(clock))
    options.setdefault("timeout", RecordingTimeout())
    options.setdefault("now_fn", lambda: datetime(2026, 1, 2, 3, 4, tzinfo=UTC))
    return runner.run_experiment(records, experiment=experiment, graph_factory=graph.factory, cache_dir=cache_dir, **options)
