"""The v1 experiment runner (5a PR-4.F1, G6, G7, E6, E7, E13): the loader's examples through ticket 4's graph, in order, one at a time.

    run_experiment(records, *, experiment, graph_factory, ...) -> ExperimentResult
    cache_path_for(experiment, cache_dir)     data/eval-cache/<experiment>.sqlite; refuses a name that is not a plain name and a file that already exists
    default_experiment_name(split, moment)    v1-<split>-<UTC stamp>
    call_with_timeout(call, seconds)          the per-example timeout (SIGALRM, main thread only)
    shared_case_groups(records)               the groups of example ids that share a case
    reader_outcomes(state)                    the reader failures that became UNVERIFIED, and the NOTAMs left unread by the reader limit
    ModelCalls                                a LangChain callback handler: the agent's model calls as they happen (usage of every one, failures the agent hides)

The records are the loader's (`evals.agent.loader.load_examples`): id, inputs {lat, lon, question, now}, fixtures {case, drop}. The runner reads nothing
else of an example and opens no case file: it never opens `readings.yaml` (the run's `reader` is "live"), and it imports the agent but does not patch,
wrap or subclass it. Nothing in src/ changes.

One run is one example in one rep. Each example runs through the graph with this run context (ticket 4's `RunContext`, field names as merged):
    now           the example's frozen `inputs.now`, parsed (never the clock)
    case          the example's `fixtures.case`
    reader        "live": the NOTAM reader reads the case's PIB; readings.yaml is not read
    reader_cache  data/eval-cache/<experiment>.sqlite, one file per experiment, shared by all reps and by examples that share a case
    drop          the example's `fixtures.drop` as a tuple (() when it has none; never None)
The step cap is ticket 4's: there is none in its settings (settings.yaml), so the runner sets no recursion limit and LangGraph's own applies.

Order (G6). Three explicit sequential passes: rep 1 over every example, then rep 2, then rep 3, each example at a time (a call finishes, or is stopped
by the timeout, before the next starts). They are not LangSmith `num_repetitions`. Reps 2 and 3 find rep 1's readings in the experiment's cache file, and
examples that share a case find each other's. `shared_case_groups` lists those groups so the report can say so.

Errored (G7, E6). Harness level only: the graph raised an exception, ran past EXAMPLE_TIMEOUT_S, or returned no final answer (no `answer`, and no
closing message without tool calls), on every one of 1 + RETRIES attempts, with BACKOFF_SECONDS waited between attempts. A run that succeeds on
any attempt is scored; its attempt count and waits are on the record. An errored run has no output and is left out of every score; it is listed by example id
with its last error, once per rep. More than INCOMPLETE_ABOVE (10%) of the runs errored sets `incomplete`. Exactly 10% does not.
A fail-closed failure inside the agent is not errored: the reader unavailable, an unread NOTAM leading to UNVERIFIED, a model failure that ends in the
guarded answer all return an answer, and the run is kept. `reader_failed_to_unverified` counts the NOTAMs the reader could not read (reason "reader
unavailable" or "reader failed" in get_notams' artifact) that sit in an UNVERIFIED segment of assess' artifact, once per run. A NOTAM left unread by the
reader limit (reason "reader limit", E13) is answered as designed: not errored, not retried, not counted there (it is listed in `reader_limited`).

Fatal stops (GP9). These end the run with a FatalRunError (one line: the cause and the last example id; `partial` holds the records written so far, in memory):
the first model response has no usage (UsageMissingError), the agent graph cannot be built, LangSmith cannot be reached, and a rate-limit or connection error from
the bridge that still stands after the retries. The agent never lets that last kind out of `graph.invoke`: its middleware catches a failed model call (a 429, a
connection error, a timeout, a 5xx) and returns its own "unavailable" answer with guard `fallback`. So the runner watches the model calls themselves (ModelCalls,
passed in the config's `callbacks`, `on_llm_error`) and treats a 429 or a connection error that the agent hid as a failed attempt: it is retried with the same
backoff, and if it stands on the last attempt the run stops. A timeout is treated like a connection error (the bridge did not answer in time): retried, then fatal.
Only a 5xx that the agent hid is a fail-closed failure (G7): the guarded answer is kept for scoring, with the failure on the record (`model_failures`). (A model
reply with a tool call that cannot be parsed also gets the agent's fallback answer, but no call failed, so nothing is retried and `model_failures` stays empty.)
Nothing is written to src/, to the dataset copy or to the dataset's source folder. A KeyboardInterrupt (SIGINT) stops the run at once and cleanly: the run in progress is not recorded and is not errored, `interrupted` is set, and the cache file is left where it is.

Usage (E4). The first model response the agent gets from the bridge must carry usage: `usage_metadata`, or the API-style `usage` (or `token_usage`) in
`response_metadata`, with tokens above zero. If it carries neither the run stops, naming the bridge and the missing field, before a second example starts.
"A model response" is one the model returned (ModelCalls sees it end); an AI message the agent's code wrote (the model-failure answer, the UK-only reply) is not
one, and a run whose model call failed has no response to check. Each run keeps the sum over every model call it made, in every attempt and including the
drafts the guard threw away (those never reach `messages`, so only the callbacks see them) and the calls of an errored run's failed attempts: `usage`, with
`model_calls`. A run with no usage keeps None,
never zeros. Only the agent's own model node is counted (`langgraph_node` "model"): the NOTAM reader's calls run inside the tool node and are left out.

LangSmith. With a client, the runner makes one experiment (a LangSmith project named after the experiment, linked to the split's dataset) and logs each
finished run to it with `create_run`, tagged `rep:<n>` and carrying the rep in its metadata. Phones in an answer are masked by the client's anonymiser
(tracing.build_client). With client=None nothing is sent (the unit tests).

Everything that touches the world is injected: the graph factory, the LangSmith client, the sleeper, the clocks and the timeout function.
"""

import re
import signal
import threading
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from fractions import Fraction
from pathlib import Path
from typing import Any
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import HumanMessage
from langchain_core.outputs import LLMResult

from evals.agent.loader import DEFAULT_SPLIT, REPO, Record, example_uuid, experiment_target
from flysafelz.agent.middleware import is_model_failure
from flysafelz.agent.state import RunContext
from flysafelz.agent.tools.notams_tool import FAILED_REASON, LIMIT_REASON, UNAVAILABLE_REASON
from flysafelz.redact import redact_phones

EXAMPLE_TIMEOUT_S = 180.0  # per attempt at one example (G6)
RETRIES = 3  # after the first attempt: 4 attempts in all (G7)
BACKOFF_SECONDS = (2.0, 4.0, 8.0)  # the wait before retry 1, 2 and 3
REPS = 3
INCOMPLETE_ABOVE = Fraction(1, 10)  # more than this share of the runs errored: incomplete (E6)
CACHE_DIR = REPO / "data" / "eval-cache"
AGENT_LABEL = "v1"
READER_MODE = "live"
ERROR_LIMIT = 300  # characters of an error kept on a record
UNVERIFIED = "UNVERIFIED"
READER_FAILED_REASONS = frozenset({UNAVAILABLE_REASON, FAILED_REASON})  # get_notams' own words for a NOTAM the reader could not read
EXPERIMENT_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")
AGENT_MODEL_NODE = "model"  # create_agent's model node: the `langgraph_node` of the agent's own model calls (the reader's run in the "tools" node)
RATE_LIMIT_STATUS = 429
FATAL_ERROR_NAMES = frozenset({"RateLimitError", "APIConnectionError"})  # the OpenAI client's names for a 429 and for an endpoint that cannot be reached
BRIDGE_VARIABLE = "AGENT_BASE_URL"

GraphFactory = Callable[[], Any]
Timeout = Callable[[Callable[[], Any], float], Any]


# --- errors ------------------------------------------------------------------------------------------------------------


class RunnerError(Exception):
    """The run cannot start, or cannot go on. The message is one line."""


class CacheExistsError(RunnerError):
    """The experiment's reader cache file is already there: it belongs to an earlier run and is never reused or appended to."""


class FatalRunError(RunnerError):
    """A stop that no retry fixes (GP9). `cause` is one line, `last_example_id` the example in progress (None: before the first), `partial` the records so far."""

    def __init__(self, cause: str, last_example_id: str | None, partial: "ExperimentResult | None" = None):
        self.cause = cause
        self.last_example_id = last_example_id
        self.partial = partial
        super().__init__(f"{cause} (last example: {last_example_id or 'none, nothing had run'})")


class UsageMissingError(FatalRunError):
    """The first model response carries no usage: the bridge must send `usage_metadata` or an API-style `usage` (E4)."""


class ExampleTimeout(BaseException):  # not an Exception: the agent's own `except Exception` handlers (tools, reader) must not swallow it
    """The graph ran past the per-example timeout."""


class ModelCallFailed(Exception):
    """The agent's model call failed with a 429 or a connection error and the agent hid it behind its fallback answer. The attempt is a failed one (it is
    retried); raised `from` the endpoint's own error, so the fatal-stop test finds the 429 or the connection error in the chain."""


class NoFinalAnswer(Exception):
    """The graph returned, but with no final answer: no `answer` and no closing message."""


# --- the timeout -------------------------------------------------------------------------------------------------------


def call_with_timeout(call: Callable[[], Any], seconds: float) -> Any:
    """Run `call()`; if it is still running after `seconds`, stop it where it is and raise ExampleTimeout.

    Uses SIGALRM, so the call is unwound in this thread and nothing is left running: the next example never overlaps it. That needs the main
    thread (the command line is); any other thread gets a RuntimeError at once rather than a timeout that cannot work.
    """
    if threading.current_thread() is not threading.main_thread():
        raise RuntimeError("the per-example timeout needs the main thread (it uses SIGALRM)")

    def on_alarm(_signum: int, _frame: object) -> None:
        raise ExampleTimeout(f"no result within {seconds:g} s")

    previous = signal.signal(signal.SIGALRM, on_alarm)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        return call()
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


# --- names and paths ---------------------------------------------------------------------------------------------------


def default_experiment_name(split: str = DEFAULT_SPLIT, moment: datetime | None = None) -> str:
    """v1-<split>-<UTC stamp>, e.g. v1-dev-20261108-160700."""
    moment = datetime.now(UTC) if moment is None else moment.astimezone(UTC)
    return f"{AGENT_LABEL}-{split}-{moment:%Y%m%d-%H%M%S}"


def cache_path_for(experiment: str, cache_dir: Path = CACHE_DIR) -> Path:
    """The experiment's reader cache file. Raises RunnerError for a name that is not a plain name, CacheExistsError (naming the path) when the file is there."""
    if not EXPERIMENT_NAME.fullmatch(experiment):
        raise RunnerError(f"experiment name {experiment!r} must be letters, digits, '_', '.' or '-' (it names the cache file)")
    path = Path(cache_dir) / f"{experiment}.sqlite"
    if path.exists():
        raise CacheExistsError(f"reader cache {path} already exists: an experiment never reuses or appends to another's cache; choose a new experiment name")
    return path


def shared_case_groups(records: Sequence[Record]) -> list[list[str]]:
    """The example ids that share a case, one list per case with two or more examples, in the order the cases first appear."""
    by_case: dict[str, list[str]] = {}
    for record in records:
        by_case.setdefault(record["fixtures"]["case"], []).append(record["id"])
    return [ids for ids in by_case.values() if len(ids) > 1]


def is_incomplete(errored: int, total: int) -> bool:
    """More than 10% of the runs errored. Exactly 10% is complete; no runs is complete."""
    return total > 0 and Fraction(errored, total) > INCOMPLETE_ABOVE


# --- reading a graph result --------------------------------------------------------------------------------------------


def _text(message: Any) -> str:
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content
    return "".join(block.get("text", "") if isinstance(block, Mapping) else str(block) for block in content or [])


def final_answer(state: Mapping[str, Any]) -> str:
    """The run's final answer: the state's `answer`, else the text of a closing AI message without tool calls. "" when there is none."""
    answer = state.get("answer")
    if isinstance(answer, str) and answer.strip():
        return answer
    messages = state.get("messages") or []
    last = messages[-1] if messages else None
    if last is not None and getattr(last, "type", None) == "ai" and not getattr(last, "tool_calls", None) and _text(last).strip():
        return _text(last)
    return ""


def _usage_of_message(message: Any) -> tuple[int, int] | None:
    """(input tokens, output tokens) of one model response, or None when it has none above zero. `usage_metadata` first, then the API-style usage."""
    candidates: list[tuple[Any, Any]] = []
    meta = getattr(message, "usage_metadata", None)
    if isinstance(meta, Mapping):
        candidates.append((meta.get("input_tokens"), meta.get("output_tokens")))
    response = getattr(message, "response_metadata", None)
    if isinstance(response, Mapping):
        for key in ("usage", "token_usage"):
            usage = response.get(key)
            if isinstance(usage, Mapping):
                candidates.append((usage.get("input_tokens", usage.get("prompt_tokens")), usage.get("output_tokens", usage.get("completion_tokens"))))
    for tokens_in, tokens_out in candidates:
        if all(isinstance(n, int) and not isinstance(n, bool) and n >= 0 for n in (tokens_in, tokens_out)) and tokens_in + tokens_out > 0:
            return tokens_in, tokens_out
    return None


class ModelCalls(BaseCallbackHandler):
    """The agent's model calls of one run, seen as they happen: pass it in the graph config's `callbacks`.

    The graph's final state cannot tell the runner what the model did. A draft the guard rejects never reaches `messages`, so its tokens would be lost, and a model
    call that failed (a 429, a refused connection) is caught by the agent's middleware, which writes its own "unavailable" answer into `messages`: that message has no
    usage and no model behind it. The callbacks see every call, and every failure before the middleware catches it.

    Only the agent's own model node is watched (`langgraph_node` is AGENT_MODEL_NODE): the NOTAM reader's chat model runs inside a tool, in the same config, and its
    calls and failures are the reader's (E7), not the agent's.

    responses  one entry per model response the run got, in order: (input tokens, output tokens), or None when it carried no usage above zero
    failures   the model-endpoint failures (the middleware's own `is_model_failure`) of the attempt in progress, which the agent turned into its fallback answer
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._watched: set[UUID] = set()
        self.responses: list[tuple[int, int] | None] = []
        self.failures: list[BaseException] = []

    def begin_attempt(self) -> None:
        """A new attempt at the same example: its failures start empty. The responses (and so the tokens spent) go on adding up."""
        with self._lock:
            self.failures = []

    def _watch(self, run_id: UUID, metadata: Mapping[str, Any] | None) -> None:
        if (metadata or {}).get("langgraph_node") == AGENT_MODEL_NODE:
            with self._lock:
                self._watched.add(run_id)

    def on_chat_model_start(self, serialized: dict[str, Any], messages: list[list[Any]], *, run_id: UUID, metadata: dict[str, Any] | None = None, **kwargs: Any) -> None:
        self._watch(run_id, metadata)

    def on_llm_end(self, response: LLMResult, *, run_id: UUID, **kwargs: Any) -> None:
        with self._lock:
            if run_id not in self._watched:
                return
            messages = (getattr(generation, "message", None) for generations in response.generations for generation in generations)
            self.responses.append(next(filter(None, map(_usage_of_message, messages)), None))

    def on_llm_error(self, error: BaseException, *, run_id: UUID, **kwargs: Any) -> None:
        with self._lock:
            if run_id in self._watched and is_model_failure(error):
                self.failures.append(error)

    def usage(self) -> dict[str, int] | None:
        """The tokens summed over every model response: input_tokens, output_tokens, total_tokens, model_calls, calls_without_usage. None when none had usage."""
        with self._lock:
            found = list(self.responses)
        counted = [pair for pair in found if pair is not None]
        if not counted:
            return None
        tokens_in, tokens_out = sum(p[0] for p in counted), sum(p[1] for p in counted)
        return {"input_tokens": tokens_in, "output_tokens": tokens_out, "total_tokens": tokens_in + tokens_out, "model_calls": len(found), "calls_without_usage": len(found) - len(counted)}


def _field(item: Any, name: str) -> Any:
    return item.get(name) if isinstance(item, Mapping) else getattr(item, name, None)


def _artifacts(state: Mapping[str, Any], tool: str) -> list[Any]:
    return [m.artifact for m in state.get("messages") or [] if getattr(m, "type", None) == "tool" and _field(getattr(m, "artifact", None), "tool") == tool]


@dataclass(frozen=True)
class ReaderOutcomes:
    failed_to_unverified: tuple[str, ...]  # NOTAM ids, sorted: the reader could not read them and an UNVERIFIED segment holds them
    limited: tuple[str, ...]  # NOTAM ids, sorted: left unread by the reader limit


def reader_outcomes(state: Mapping[str, Any]) -> ReaderOutcomes:
    """What the reader did in this run, from the tool artifacts in the state (get_notams and assess). Each NOTAM id counts once, however many calls named it."""
    failed: set[str] = set()
    limited: set[str] = set()
    for artifact in _artifacts(state, "get_notams"):
        for item in _field(artifact, "notams") or []:
            reason = _field(item, "unread_reason")
            if reason in READER_FAILED_REASONS:
                failed.add(_field(item, "id"))
            elif reason == LIMIT_REASON:
                limited.add(_field(item, "id"))
    unverified: set[str] = set()
    for artifact in _artifacts(state, "assess"):
        for segment in _field(artifact, "segments") or []:
            if _field(segment, "verdict") == UNVERIFIED:
                unverified.update(_field(notam, "id") for notam in _field(segment, "notams") or [])
    return ReaderOutcomes(tuple(sorted(failed & unverified)), tuple(sorted(limited)))


def _short(error: BaseException) -> str:
    """One line for an error: its type and text, phone numbers masked, cut at ERROR_LIMIT characters."""
    text = redact_phones(" ".join(f"{type(error).__name__}: {error}".split()))[0]
    return text if len(text) <= ERROR_LIMIT else text[: ERROR_LIMIT - 1] + "…"


def _is_fatal_kind(error: BaseException) -> str | None:
    """ "rate limited", "timed out" or "endpoint unreachable" when the error (or one it was raised from) says so, else None.

    A timeout is looked for before a connection error because the OpenAI client's APITimeoutError is a subclass of APIConnectionError."""
    seen: set[int] = set()
    current: BaseException | None = error
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if getattr(current, "status_code", None) == RATE_LIMIT_STATUS or type(current).__name__ == "RateLimitError":
            return "rate limited"
        names = {cls.__name__ for cls in type(current).__mro__}
        if "APITimeoutError" in names or isinstance(current, TimeoutError):
            return "timed out"
        if FATAL_ERROR_NAMES & names or isinstance(current, ConnectionError):
            return "endpoint unreachable"
        current = current.__cause__ or current.__context__
    return None


# --- the records -------------------------------------------------------------------------------------------------------


@dataclass
class RunRecord:
    """One example in one rep. `output` is the graph's final state (None when errored); an errored run is left out of every score."""

    example_id: str
    rep: int
    case: str
    status: str  # "answered" or "errored"
    attempts: int
    backoff_waits: tuple[float, ...]
    latency_s: float
    context: RunContext
    output: Mapping[str, Any] | None = None
    answer: str | None = None
    last_error: str | None = None
    usage: dict[str, int] | None = None
    reader_failed_to_unverified: tuple[str, ...] = ()
    reader_limited: tuple[str, ...] = ()
    model_failures: tuple[str, ...] = ()  # model calls that failed with a 5xx and that the agent turned into its fallback answer, in the kept attempt (not 429, timeout, connection)
    langsmith_run_id: uuid.UUID | None = None
    started_at: datetime | None = None

    @property
    def errored(self) -> bool:
        return self.status == "errored"


@dataclass(frozen=True)
class ErroredRun:
    example_id: str
    error: str


@dataclass
class ExperimentResult:
    """Everything a run produced, for the evaluators, the judge and the report. Filled as the run goes, so a stop leaves what was done."""

    experiment: str
    split: str
    cache_path: Path
    reps: int
    planned_runs: int
    shared_case_groups: list[list[str]]
    records: list[RunRecord] = field(default_factory=list)
    interrupted: bool = False

    @property
    def errored(self) -> dict[int, list[ErroredRun]]:
        """The errored runs per rep (every rep is a key), each example once, in run order."""
        listed: dict[int, list[ErroredRun]] = {rep: [] for rep in range(1, self.reps + 1)}
        for record in self.records:
            if record.errored and all(e.example_id != record.example_id for e in listed[record.rep]):
                listed[record.rep].append(ErroredRun(record.example_id, record.last_error or ""))
        return listed

    @property
    def errored_runs(self) -> int:
        return sum(len(v) for v in self.errored.values())

    @property
    def incomplete(self) -> bool:
        return is_incomplete(self.errored_runs, self.planned_runs)

    @property
    def scored(self) -> list[RunRecord]:
        """The runs that count: every one that was not errored."""
        return [r for r in self.records if not r.errored]

    @property
    def reader_failed_to_unverified(self) -> dict[str, Any]:
        """The ops count: {"per_rep": {rep: n}, "total": n}, each failed reader call once per run."""
        per_rep = {rep: sum(len(r.reader_failed_to_unverified) for r in self.records if r.rep == rep) for rep in range(1, self.reps + 1)}
        return {"per_rep": per_rep, "total": sum(per_rep.values())}

    @property
    def model_failed_runs(self) -> int:
        """How many kept runs hold a fallback answer because a model call failed with a 5xx (a 429, a timeout or a connection error is retried and then stops the run instead)."""
        return sum(1 for r in self.scored if r.model_failures)

    def usage_total(self) -> dict[str, int] | None:
        """Tokens summed over every run that has usage, errored ones too (their model calls were paid for); None when no run has any."""
        parts = [r.usage for r in self.records if r.usage]
        if not parts:
            return None
        keys = ("input_tokens", "output_tokens", "total_tokens", "model_calls", "calls_without_usage")
        return {key: sum(p[key] for p in parts) for key in keys}

    def summary_lines(self) -> list[str]:
        """The run in a few lines: the experiment, the cache, the runs per rep in order, the errored ids, the ops count."""
        lines = [f"experiment {self.experiment} (split {self.split}): {len(self.records)} of {self.planned_runs} runs, one example at a time"]
        lines.append(f"reader cache: {self.cache_path}")
        for rep in range(1, self.reps + 1):
            ids = [r.example_id for r in self.records if r.rep == rep]
            span = f"{ids[0]} … {ids[-1]}" if ids else "none"
            lines.append(f"  rep {rep}: {len(ids)} runs ({span})")
        errored = self.errored
        lines.append(
            f"errored: {self.errored_runs} of {self.planned_runs} runs"
            + ("" if not self.errored_runs else " — " + "; ".join(f"rep {rep}: " + ", ".join(e.example_id for e in found) for rep, found in errored.items() if found))
        )
        lines.append(f"incomplete: {str(self.incomplete).lower()}")
        failed = self.reader_failed_to_unverified
        if self.model_failed_runs:
            lines.append(f"runs kept with the agent's fallback answer because its model call failed (5xx): {self.model_failed_runs}")
        lines.append("reader calls failed and became UNVERIFIED: " + str(failed["total"]) + " (" + ", ".join(f"rep {rep}: {n}" for rep, n in failed["per_rep"].items()) + ")")
        if self.shared_case_groups:
            lines.append(f"examples sharing a case (they and the later reps reuse one cache file): {len(self.shared_case_groups)} groups")
        if self.interrupted:
            lines.append("INTERRUPTED: the run was stopped; the cache file was left in place")
        return lines


# --- the run -----------------------------------------------------------------------------------------------------------


def _utc_now() -> datetime:
    return datetime.now(UTC)


class _Run:
    """The state of one experiment while it runs."""

    def __init__(
        self,
        result: ExperimentResult,
        graph: Any,
        client: Any,
        *,
        timeout_s: float,
        retries: int,
        backoff: Sequence[float],
        sleep: Callable[[float], None],
        clock: Callable[[], float],
        now_fn: Callable[[], datetime],
        timeout: Timeout,
    ):
        self.result, self.graph, self.client = result, graph, client
        self.timeout_s, self.retries, self.backoff = timeout_s, retries, tuple(backoff)
        self.sleep, self.clock, self.now_fn, self.timeout = sleep, clock, now_fn, timeout
        self.usage_checked = False

    # the request and the context of one attempt: built fresh each time, from this example only, so nothing carries over between examples

    def context_for(self, record: Record) -> RunContext:
        return RunContext(
            now=datetime.fromisoformat(record["inputs"]["now"]),
            case=record["fixtures"]["case"],
            reader=READER_MODE,
            reader_cache=self.result.cache_path,
            drop=tuple(record["fixtures"].get("drop") or ()),
        )

    def request_for(self, record: Record) -> dict[str, Any]:
        inputs = record["inputs"]
        return {"messages": [HumanMessage(inputs["question"])], "lat": inputs["lat"], "lon": inputs["lon"]}

    def check_usage(self, calls: ModelCalls, example_id: str) -> None:
        """The first model response of the experiment must carry usage (E4). Checked once, on the first run that has a model response at all.

        A run with none (the UK-only reply, or a model call that failed and became the fallback answer) says nothing about the bridge's usage, so it is skipped.
        """
        if self.usage_checked or not calls.responses:
            return
        self.usage_checked = True
        if calls.responses[0] is None:
            raise UsageMissingError(
                f"the model bridge ({BRIDGE_VARIABLE}) sent no usage on the first response: it has neither `usage_metadata` nor an API-style `usage` with tokens above zero; "
                "tokens would be reported as zeros, so the run stops",
                example_id,
                self.result,
            )

    def attempt(self, record: Record, rep: int, number: int, calls: ModelCalls) -> Mapping[str, Any]:
        calls.begin_attempt()
        config = {
            "configurable": {"thread_id": f"{self.result.experiment}:{rep}:{record['id']}:{number}"},
            "metadata": {"experiment": self.result.experiment, "example_id": record["id"], "rep": rep},
            "callbacks": [calls],
        }
        request, context = self.request_for(record), self.context_for(record)
        state = self.timeout(lambda: self.graph.invoke(request, config, context=context), self.timeout_s)
        if not isinstance(state, Mapping):
            raise NoFinalAnswer("the graph returned no state")
        self.check_usage(calls, record["id"])
        hidden = next((error for error in calls.failures if _is_fatal_kind(error) is not None), None)
        if hidden is not None:  # the agent caught it and answered with its fallback: the bridge is limiting or down, and that is not a product answer to score
            raise ModelCallFailed(f"the model call failed and the agent answered with its fallback: {_short(hidden)}") from hidden
        if not final_answer(state):
            raise NoFinalAnswer("the graph returned no final answer")
        return state

    def run_example(self, record: Record, rep: int) -> RunRecord:
        """One example, one rep: up to 1 + retries attempts, waiting between them. Never raises for a failed attempt; FatalRunError for a fatal one."""
        started, started_at = self.clock(), self.now_fn()
        waits: list[float] = []
        calls = ModelCalls()
        last_error: BaseException | None = None
        state: Mapping[str, Any] | None = None
        attempts = 0
        for attempts in range(1, self.retries + 2):
            try:
                state = self.attempt(record, rep, attempts, calls)
                break
            except ExampleTimeout as error:  # a BaseException by design
                last_error = error
            except FatalRunError:
                raise
            except Exception as error:  # noqa: BLE001 - anything the graph raises is a harness failure; it is retried and, at the end, recorded
                last_error = error
            if attempts <= self.retries:
                wait = self.backoff[min(attempts - 1, len(self.backoff) - 1)] if self.backoff else 0.0
                waits.append(wait)
                self.sleep(wait)
        latency = self.clock() - started
        context = self.context_for(record)
        if state is None:
            assert last_error is not None
            kind = _is_fatal_kind(last_error)
            if kind is not None:
                raise FatalRunError(f"the model bridge kept failing ({kind}) after {attempts} attempts: {_short(last_error)}", record["id"], self.result)
            return RunRecord(
                record["id"], rep, record["fixtures"]["case"], "errored", attempts, tuple(waits), latency, context, last_error=_short(last_error), usage=calls.usage(), started_at=started_at
            )
        outcomes = reader_outcomes(state)
        return RunRecord(
            record["id"],
            rep,
            record["fixtures"]["case"],
            "answered",
            attempts,
            tuple(waits),
            latency,
            context,
            output=state,
            answer=final_answer(state),
            usage=calls.usage(),
            reader_failed_to_unverified=outcomes.failed_to_unverified,
            reader_limited=outcomes.limited,
            model_failures=tuple(_short(error) for error in calls.failures),
            started_at=started_at,
        )

    # LangSmith: one experiment, one logged run per finished run

    def open_experiment(self) -> None:
        if self.client is None:
            return
        target = experiment_target(self.result.split)
        try:
            dataset = self.client.read_dataset(dataset_name=target.dataset)
            self.client.create_project(
                project_name=self.result.experiment,
                reference_dataset_id=dataset.id,
                description=f"FlySafeLZ {AGENT_LABEL} over the {self.result.split} split, {self.result.reps} reps, one example at a time",
                metadata={"agent": AGENT_LABEL, "split": self.result.split, "reps": self.result.reps, "split_project": target.project},
            )
        except Exception as error:
            raise FatalRunError(f"LangSmith could not open the experiment {self.result.experiment}: {_short(error)}", None, self.result) from error

    def log(self, run: RunRecord, record: Record) -> None:
        if self.client is None:
            return
        run.langsmith_run_id = uuid.uuid4()
        try:
            self.client.create_run(
                id=run.langsmith_run_id,
                name=f"{AGENT_LABEL}-{record['id']}",
                run_type="chain",
                project_name=self.result.experiment,
                reference_example_id=example_uuid(record["id"]),
                inputs={"question": record["inputs"]["question"], "lat": record["inputs"]["lat"], "lon": record["inputs"]["lon"], "now": record["inputs"]["now"], "fixtures": record["fixtures"]},
                outputs=None if run.errored else {"answer": run.answer, "guard": (run.output or {}).get("guard"), "usage": run.usage},
                error=run.last_error,
                start_time=run.started_at,
                end_time=self.now_fn(),
                tags=[f"rep:{run.rep}", AGENT_LABEL],
                extra={
                    "metadata": {"rep": run.rep, "example_id": record["id"], "experiment": self.result.experiment, "split": self.result.split, "attempts": run.attempts, "latency_s": run.latency_s}
                },
            )
        except Exception as error:
            raise FatalRunError(f"LangSmith could not log the run: {_short(error)}", record["id"], self.result) from error


def run_experiment(
    records: Sequence[Record],
    *,
    experiment: str,
    graph_factory: GraphFactory,
    split: str = DEFAULT_SPLIT,
    client: Any = None,
    cache_dir: Path = CACHE_DIR,
    reps: int = REPS,
    timeout_s: float = EXAMPLE_TIMEOUT_S,
    retries: int = RETRIES,
    backoff: Sequence[float] = BACKOFF_SECONDS,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
    now_fn: Callable[[], datetime] = _utc_now,
    timeout: Timeout = call_with_timeout,
) -> ExperimentResult:
    """Run every record `reps` times, rep by rep, one at a time, and return the result. See the module doc for the rules.

    The experiment name and its cache path are checked first (RunnerError, CacheExistsError), before the graph is built or anything is sent.
    Raises FatalRunError (with `.partial`) for a stop that no retry fixes. A KeyboardInterrupt does not raise: the result comes back with `interrupted` set.
    """
    cache_path = cache_path_for(experiment, cache_dir)
    result = ExperimentResult(experiment, split, cache_path, reps, len(records) * reps, shared_case_groups(records))
    try:
        graph = graph_factory()
    except Exception as error:
        raise FatalRunError(f"the agent graph could not be built: {_short(error)}", None, result) from error
    run = _Run(result, graph, client, timeout_s=timeout_s, retries=retries, backoff=backoff, sleep=sleep, clock=clock, now_fn=now_fn, timeout=timeout)
    try:
        run.open_experiment()
        for rep in range(1, reps + 1):
            for record in records:
                finished = run.run_example(record, rep)
                result.records.append(finished)
                run.log(finished, record)
    except KeyboardInterrupt:
        result.interrupted = True
    return result
