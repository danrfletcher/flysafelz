"""LangSmith tracing (PR-12.F1: G20, G15's feedback clause, FR18, I5, I6, I7): the anonymiser, the client, and what each turn reports.

    ANONYMISER              the function the LangSmith client applies to every traced input, output, metadata and error: phones and emails masked
    build_client()          the LangSmith client with the anonymiser attached, or None when LANGSMITH_API_KEY is not set (then tracing is off)
    shared_client()         the same, made once per process and key, for `build_graph()` (the server builds a graph for every run)
    traced(graph, client)   the graph with a tracer on it that sends its runs through that client
    tag_turn(context)       puts `source` (case / online-seed / pilot, I6) in the turn's root run
    report_outcome(outcome) puts `path` in the root run (I5, I6) and sends the guard's result as `guard_fail` feedback on it (G15, FR18)
    wait_for_feedback()     returns when every feedback sent so far has been sent or has failed (a test, or a process about to exit, calls it)

Nothing here reads LANGSMITH_* at import: config.langsmith_settings() reads them when the first graph is built. With no LANGSMITH_API_KEY there is no client, the graph
has no tracer, and `tag_turn` and `report_outcome` find no run to write to and do nothing. No network call is made.

The anonymiser. Phones are masked with `redact.PHONE_PATTERNS`: ticket 3's own pattern (the same object `redact_phones` uses, so they cannot drift apart) and, after it,
the one for numbers written with hyphens or brackets, which `redact_phones` leaves alone. Both give `<PHONE>`. Emails give `<EMAIL>`. A `<PHONE_n>` placeholder has no phone shape, so it is left exactly as it is. Only strings are touched: a number, a
GeoJSON coordinate, a boolean and None pass through, so `53.802797` keeps its full precision and so do the LZ inputs in state. The tools already hand the model
and the trace `<PHONE_n>` (P1); this catches what is left, above all the real number that `answer`, `assessment` and `overlays` hold after rehydration (I7).

The tracer is LangChain's own `LangChainTracer` and nothing else changes in the graph: it is added to the graph's config (`traced`), so every run of the graph,
its model calls and its tools go through the client that holds the anonymiser. Two things need the run a turn belongs to, which the tracer keeps in its
`run_map`. The middleware's hooks find the turn's root run from the config LangGraph gives the node (`_root_run`): `tag_turn` and `report_outcome` write to it.
`FlySafeTracer` copies `path` from the root run's metadata into its outputs when it ends (I6), so a reader that only sees run outputs gets it too.

`path` is the guard's: the tag the model's answer started with, or `unknown` (I5). `source` is `case` for any run with a case in its run context, whatever
the caller sent (a caller can never make a case-mode run `pilot`); `online-seed` when the caller sent that as `source` in the run's metadata (ticket 7's seed);
otherwise `pilot`. A turn that never reaches the guard (a rejected request, the UK-only answer, a model failure) has a `source` and no `path`, and no feedback.

The feedback. Each guarded turn sends exactly one `guard_fail`: score 0 when the first draft passed, 1 when any check failed (fixed by the retry or ended in the
fallback). The comment holds each failed round's findings, a round filed under `path_tag` when the tag was its only problem. It goes through the same
anonymiser as the run does, because a finding quotes the draft and a draft may hold a number the model made up (E14). Sending is network work, so it runs on a
worker thread: the graph may be running on an event loop (the LangGraph server), and a slow or failing LangSmith must never hold up or change the pilot's answer.
A feedback that fails is logged as a warning and dropped.
"""

import logging
import re
import threading
from collections.abc import Callable, Mapping
from concurrent.futures import Future, ThreadPoolExecutor, wait
from typing import Any, cast
from uuid import UUID
from weakref import WeakKeyDictionary

from langchain_core.runnables import Runnable
from langchain_core.tracers.langchain import LangChainTracer
from langgraph.config import get_config
from langsmith import Client
from langsmith.anonymizer import StringNodeRule, create_anonymizer

from flysafelz.agent.config import langsmith_settings
from flysafelz.agent.guard_flow import GuardOutcome
from flysafelz.agent.state import RunContext
from flysafelz.redact import PHONE_PATTERNS

log = logging.getLogger(__name__)

FEEDBACK_KEY = "guard_fail"
PATH_KEY = "path"
SOURCE_KEY = "source"
SOURCES = ("case", "online-seed", "pilot")  # I6
CASE_SOURCE, SEED_SOURCE, PILOT_SOURCE = SOURCES

PHONE_MASK = "<PHONE>"
EMAIL_MASK = "<EMAIL>"
EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}")  # an address with an @ (NOTAM text writes "(AT)" and ticket 3 keeps that)
RULES: list[StringNodeRule] = [*({"pattern": pattern, "replace": PHONE_MASK} for pattern in PHONE_PATTERNS), {"pattern": EMAIL, "replace": EMAIL_MASK}]
MAX_DEPTH = 64  # how deep into a payload the anonymiser looks; langsmith's own default is 10 and it silently skips what is deeper. No traced payload comes near this
MESSAGE_LIMIT = 150  # characters of an exception's text that go to the log

ANONYMISER: Callable[[Any], Any] = create_anonymizer(RULES, max_depth=MAX_DEPTH)  # edits dicts and lists it is given in place, and returns the result (langsmith's contract)


# --- the client -------------------------------------------------------------------------------------------------------


def project_name(env: Mapping[str, str] | None = None) -> str:
    """The LangSmith project runs go to: LANGSMITH_PROJECT, else settings.yaml's `langsmith_project` (flysafelz)."""
    return langsmith_settings(env).project


def build_client(env: Mapping[str, str] | None = None) -> Client | None:
    """A LangSmith client with the anonymiser attached, or None when LANGSMITH_API_KEY is not set or blank (tracing off). Builds nothing, and reads LANGSMITH_*, only when called."""
    settings = langsmith_settings(env)
    if settings.api_key is None:
        return None
    return Client(api_key=settings.api_key.get_secret_value(), api_url=settings.endpoint, anonymizer=ANONYMISER)


_clients: dict[tuple[str, str], Client] = {}
_clients_lock = threading.Lock()


def shared_client() -> Client | None:
    """`build_client()`, but one client per key and endpoint for the process: the server builds the graph for every run, and a client holds a thread and a queue."""
    settings = langsmith_settings()
    if settings.api_key is None:
        return None
    identity = (settings.api_key.get_secret_value(), settings.endpoint or "")
    with _clients_lock:
        if identity not in _clients:
            built = build_client()
            if built is None:  # the key was removed between the two reads
                return None
            _clients[identity] = built
        return _clients[identity]


# --- the tracer and the graph's runs ------------------------------------------------------------------------------------


class FlySafeTracer(LangChainTracer):
    """LangChain's tracer, which also writes the root run's `path` into the root run's outputs when it ends (I6). It has no state of its own: LangChain copies a tracer."""

    def _on_chain_end(self, run: Any) -> None:
        path = run.metadata.get(PATH_KEY) if run.parent_run_id is None else None
        if path is not None and isinstance(run.outputs, dict):
            run.outputs = {**run.outputs, PATH_KEY: path}  # a new dict: the one LangGraph returned to the caller is not touched
        super()._on_chain_end(run)


def traced[G: Runnable](graph: G, client: Client | None = None) -> G:
    """The graph with a tracer that sends its runs through `client` (`shared_client()` when none is given). Unchanged, and untraced, when there is no client."""
    client = shared_client() if client is None else client
    if client is None:
        return graph
    return cast(G, graph.with_config(callbacks=[FlySafeTracer(client=client, project_name=project_name())]))  # with_config returns the same kind of graph


def _root_run() -> tuple[LangChainTracer, Any] | None:
    """(the tracer, the turn's root run) for the node that is running now, or None when this run is not traced.

    LangGraph hands a node a callback manager whose parent is the node's own run; the tracer holds every run of the turn, and each run knows its trace's root.
    """
    try:
        callbacks = get_config().get("callbacks")
    except RuntimeError:  # not inside a graph run
        return None
    parent = getattr(callbacks, "parent_run_id", None)
    handlers = getattr(callbacks, "handlers", None) or []
    tracer = next((handler for handler in handlers if isinstance(handler, LangChainTracer)), None)
    if tracer is None or parent is None:
        return None
    node = tracer.run_map.get(str(parent))
    root = tracer.run_map.get(str(node.trace_id)) if node is not None else None
    return (tracer, root) if root is not None else None


def source_of(context: RunContext | None, asked: object) -> str:
    """The run's `source` (I6): `case` for a case-mode run whatever was asked; else `online-seed` if the caller asked for exactly that; else `pilot`."""
    if context is not None and context.case:
        return CASE_SOURCE
    return SEED_SOURCE if asked == SEED_SOURCE else PILOT_SOURCE


def tag_turn(context: RunContext | None) -> None:
    """Write the turn's `source` into its root run's metadata. A failure is logged and changes nothing else."""
    try:
        found = _root_run()
        if found is not None:
            metadata = found[1].metadata
            metadata[SOURCE_KEY] = source_of(context, metadata.get(SOURCE_KEY))
    except Exception as error:  # noqa: BLE001 - tracing must never change what the pilot gets
        log.warning("Could not tag the LangSmith run (%s); the answer is unchanged.", _reason(error))


# --- the feedback ---------------------------------------------------------------------------------------------------------


def feedback_comment(outcome: GuardOutcome) -> str | None:
    """The findings of each failed round, one line per finding under a "Round n" line that names the reason, or None for a turn that passed first time."""
    if not outcome.attempts:
        return None
    rounds = [f"Round {number} ({attempt.reason}):\n" + "\n".join(f"- {finding}" for finding in attempt.findings) for number, attempt in enumerate(outcome.attempts, start=1)]
    return "\n".join(rounds)


_executor: ThreadPoolExecutor | None = None
_pending: set[Future[None]] = set()
_pending_lock = threading.Lock()
_project_ids: WeakKeyDictionary[Client, dict[str, UUID]] = WeakKeyDictionary()


def _reason(error: BaseException) -> str:
    return f"{type(error).__name__}: {error}"[:MESSAGE_LIMIT]


def _project_id(client: Client, project: str) -> UUID | None:
    """The project's id, which LangSmith wants with run feedback. Asked once per client and project; None (logged) when it cannot be found, and then the feedback goes without."""
    known = _project_ids.setdefault(client, {})
    if project not in known:
        try:
            known[project] = client.read_project(project_name=project).id
        except Exception as error:  # noqa: BLE001 - the project may not exist yet on the first turn; the feedback is still worth sending
            log.info("LangSmith project %r not found (%s); sending the feedback without its id.", project, _reason(error))
            return None
    return known[project]


def _send(client: Client, run_id: UUID, project: str, failed: int, comment: str | None) -> None:
    try:
        client.create_feedback(
            run_id,
            FEEDBACK_KEY,
            score=failed,  # 0 passed first time, 1 failed a check
            comment=comment,
            trace_id=run_id,  # the root run is the trace: lets the client queue the feedback behind the run it describes
            session_id=_project_id(client, project),
        )
    except Exception as error:  # noqa: BLE001 - LangSmith 5xx, a timeout, whatever it is: the pilot already has the answer
        log.warning("Could not send %s feedback to LangSmith (%s); the answer is unchanged.", FEEDBACK_KEY, _reason(error))


def _submit(client: Client, run_id: UUID, project: str, failed: int, comment: str | None) -> None:
    global _executor
    with _pending_lock:
        if _executor is None:
            _executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="flysafelz-feedback")
        future = _executor.submit(_send, client, run_id, project, failed, comment)
        _pending.add(future)
    future.add_done_callback(_forget)


def _forget(future: Future[None]) -> None:
    with _pending_lock:
        _pending.discard(future)


def wait_for_feedback(timeout: float | None = None) -> None:
    """Return when every feedback sent so far has gone or failed (or after `timeout` seconds)."""
    with _pending_lock:
        futures = list(_pending)
    wait(futures, timeout=timeout)


def report_outcome(outcome: GuardOutcome) -> None:
    """The guard's result for the turn: `path` into the root run, and one `guard_fail` feedback on it. Called once per guarded turn by the middleware's guard.

    Score 0 is a first draft that passed; 1 is any failed check, fixed by the retry or ended in the fallback. Does nothing when the run is not traced, and
    never raises: a tracing failure changes neither `answer` nor `guard`.
    """
    try:
        found = _root_run()
        if found is None:
            return
        tracer, root = found
        root.metadata[PATH_KEY] = outcome.path
        comment = feedback_comment(outcome)
        _submit(tracer.client, root.id, root.session_name or tracer.project_name or project_name(), 1 if outcome.attempts else 0, None if comment is None else ANONYMISER(comment))
    except Exception as error:  # noqa: BLE001 - tracing must never change what the pilot gets
        log.warning("Could not report the guard result to LangSmith (%s); the answer is unchanged.", _reason(error))
