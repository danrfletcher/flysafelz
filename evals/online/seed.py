"""`make online-seed`: ask the 11 questions in `seed_questions.yaml` through the agent on live data, so there are real runs for `make online-eval` to score (7 PR-5.F1, G1, G2, E6).

    load_questions(path=QUESTIONS_FILE) -> list[Question]   the file, checked entry by entry; SeedFileError (one line) names the entry that is wrong
    missing_variables(environ) -> list[str]                  the required variables that are unset or blank, in a fixed order
    default_entry() -> EntryPoint                            ticket 4's graph, built at the first question: `build_graph().invoke(request, config, context=context)`
    main(argv=None, *, environ=None, entry=None, client=None, stdout=None, stderr=None, questions_file=QUESTIONS_FILE) -> int   the command

It makes no scoring decision, defines no metric and posts no feedback: that is `job.py`. It runs only on Dan's Mac, with his keys, and `make test` never calls it.

Order. The file is read and checked first (a malformed file, a missing field, a duplicate `id` or an `intended_path` outside assess, explain, clarify and decline stops it with one line naming
the entry, before any variable is looked at). Then the variables: LANGSMITH_API_KEY, ticket 4's AGENT_BASE_URL, AGENT_MODEL and AGENT_API_KEY, and OPENAIP_API_KEY. Any that is unset, or set to
an empty string (as `.env.example` ships them), is named, one per line and never with a value, and the command exits 2 before the first question, the entry point and the client having had no call (E6).
The project (LANGSMITH_PROJECT, else the agent's own default, `flysafelz`, the same one `make online-eval` pulls from) goes through the pull's project guard, so the seed can never write into the offline
eval's held-out or dev experiment project: that exits 2, too.

The questions. They are asked one at a time, in file order, each in its own thread, never at the same time. Each goes through ticket 4's entry point as the pilot's message with the question's own `lat` and
`lon`, and a run context with no case and no `now`: that is live mode, which reads the live sources and has the NOTAM reader read the live bulletin. (`reader="live"` is a case-mode switch and the agent's
input check refuses it without a case, so it is not set: the live reader is what live mode uses.) The run's metadata carries `source: online-seed`, which ticket 4's tracing keeps for the root run, and the
question's `lz_name`. `intended_path` is never sent: it is not in the metadata, not in the request, not posted anywhere, and this is the only module in `evals/online/` that reads it.

Tracing. The seed turns tracing on itself for its own calls (`tracing_v2_enabled(project_name=..., client=...)` over the agent's client, which masks phones and emails), and collects each root run through the
tracing callback (`collect_runs()`), never by listing the project afterwards. The agent's graph already carries its own tracer; LangChain does not add a second one, so a question is one trace.
The client is flushed after each question and its feedback waited for at the end, so the run is in LangSmith before its URL is asked for.

Output. One line on stdout per question that ran: `<id> <path> <url>`, where `path` is read as `reference.py` reads it for the online checks (the root run's metadata, then its outputs, `unknown` when neither
holds one of the four) and the URL is `client.get_run_url(run=..., project_name=...)`. A run whose URL cannot be made prints `unavailable (<error type>)` there. 11 lines on a clean run.

Exit codes: 0 every question ran and has its line; 1 a question raised, left no single root run, or has no URL (it is named on stderr by id and error type, the others still ran); 2 a bad file, a missing
variable, a protected project or an argument.
"""

import math
import os
import sys
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TextIO

import yaml
from langchain_core.messages import HumanMessage
from langchain_core.tracers.context import collect_runs, tracing_v2_enabled

from evals.online.pull import guard_project
from evals.online.reference import path_of
from flysafelz.agent import tracing
from flysafelz.agent.config import AGENT_VARIABLES, LANGSMITH_KEY
from flysafelz.agent.graph import build_graph
from flysafelz.agent.state import RunContext
from flysafelz.agent.tracing import SEED_SOURCE

QUESTIONS_FILE = Path(__file__).with_name("seed_questions.yaml")
OPENAIP_KEY = "OPENAIP_API_KEY"
REQUIRED_VARIABLES = (LANGSMITH_KEY, *AGENT_VARIABLES, OPENAIP_KEY)
INTENDED_PATHS = ("assess", "explain", "clarify", "decline")
TEXT_FIELDS = ("id", "lz_name", "question", "intended_path")
NUMBER_FIELDS = (("lat", 90.0), ("lon", 180.0))
FIELDS = ("id", "lz_name", "lat", "lon", "question", "intended_path")
FAILED_EXIT = 1
USAGE_EXIT = 2

# What the agent is called with: the graph's input, the run context, and the config (thread and metadata). It returns whatever the graph returns, which is not read.
EntryPoint = Callable[[Mapping[str, Any], RunContext, Mapping[str, Any]], Any]


class SeedFileError(ValueError):
    """`seed_questions.yaml` is wrong. The message is one line and names the entry."""


@dataclass(frozen=True)
class Question:
    id: str
    lz_name: str
    lat: float
    lon: float
    question: str
    intended_path: str  # for reading only: never sent, never scored


# --- the file -------------------------------------------------------------------------------------------------------------------------------


def _entry_name(index: int, raw: object) -> str:
    given = raw.get("id") if isinstance(raw, Mapping) else None
    return f"entry {index}" + (f" ({given})" if isinstance(given, str) and given.strip() else "")


def _text(raw: Mapping[str, Any], field: str, where: str) -> str:
    value = raw[field]
    if not isinstance(value, str) or not value.strip():
        raise SeedFileError(f"{where}: {field} must be a non-empty string")
    return value.strip()


def _number(raw: Mapping[str, Any], field: str, limit: float, where: str) -> float:
    value = raw[field]
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or abs(value) > limit:
        raise SeedFileError(f"{where}: {field} must be a number between -{limit:g} and {limit:g}")
    return float(value)


def _question(index: int, raw: object, seen: set[str]) -> Question:
    where = _entry_name(index, raw)
    if not isinstance(raw, Mapping):
        raise SeedFileError(f"{where}: must be a mapping with {', '.join(FIELDS)}")
    for field in FIELDS:
        if field not in raw:
            raise SeedFileError(f"{where}: missing {field}")
    text = {field: _text(raw, field, where) for field in TEXT_FIELDS}
    numbers = {field: _number(raw, field, limit, where) for field, limit in NUMBER_FIELDS}
    if text["id"] in seen:
        raise SeedFileError(f"{where}: duplicate id {text['id']}")
    if text["intended_path"] not in INTENDED_PATHS:
        raise SeedFileError(f"{where}: intended_path must be one of {', '.join(INTENDED_PATHS)}")
    seen.add(text["id"])
    return Question(lat=numbers["lat"], lon=numbers["lon"], **text)


def load_questions(path: Path = QUESTIONS_FILE) -> list[Question]:
    """The questions, in file order. Raises SeedFileError, with one line naming the entry, for a file that is not YAML, is not a list of mappings, or has an entry with a missing or bad field."""
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as error:
        raise SeedFileError(f"{path.name}: cannot be read as YAML ({type(error).__name__})") from None
    if not isinstance(data, list) or not data:
        raise SeedFileError(f"{path.name}: must be a list of questions")
    seen: set[str] = set()
    return [_question(index, raw, seen) for index, raw in enumerate(data, start=1)]


# --- the keys -------------------------------------------------------------------------------------------------------------------------------


def missing_variables(environ: Mapping[str, str]) -> list[str]:
    """The required variables that are unset, or set to nothing but spaces (an empty `NAME=` line in .env is unset)."""
    return [name for name in REQUIRED_VARIABLES if not environ.get(name, "").strip()]


# --- the questions --------------------------------------------------------------------------------------------------------------------------


def default_entry() -> EntryPoint:
    """Ticket 4's agent, as an entry point. The graph is built at the first question, so a seed that stops at the file or the keys builds nothing."""
    graph: Any = None

    def entry(request: Mapping[str, Any], context: RunContext, config: Mapping[str, Any]) -> Any:
        nonlocal graph
        if graph is None:
            graph = build_graph()
        return graph.invoke(request, config, context=context)

    return entry


def request_for(question: Question) -> dict[str, Any]:
    """The graph's input: the pilot's message and the landing zone. No time zone is sent, so the agent's default (Europe/London) applies."""
    return {"messages": [HumanMessage(question.question)], "lat": question.lat, "lon": question.lon}


def config_for(question: Question) -> dict[str, Any]:
    """Its own thread, and the metadata `make online-eval` reads: `source` and `lz_name`. Nothing else: `intended_path` stays in the file."""
    return {"configurable": {"thread_id": f"seed-{uuid.uuid4()}"}, "metadata": {"source": SEED_SOURCE, "lz_name": question.lz_name}}


def root_run(collected: Sequence[Any]) -> Any:
    """The one root run among the runs the callback collected (the agent's model and tool runs below it are collected too). Raises LookupError for none or several."""
    roots = [run for run in collected if getattr(run, "parent_run_id", None) is None]
    if len(roots) != 1:
        raise LookupError(f"expected one root run, collected {len(roots)}")
    return roots[0]


def ask_question(question: Question, entry: EntryPoint, client: Any) -> Any:
    """Ask one question with the run context of live mode and return its root run. `collect_runs` hands the run back through the tracing callback."""
    with collect_runs() as collector:
        entry(request_for(question), RunContext(), config_for(question))
    client.flush()  # the run is sent in the background: it must be there before its URL is asked for
    return root_run(collector.traced_runs)


def _line(question: Question, run: Any, client: Any, project: str, stdout: TextIO, stderr: TextIO) -> bool:
    """Print the question's line. False when there is no URL for it."""
    path = path_of(run)
    try:
        url = client.get_run_url(run=run, project_name=project)
    except Exception as error:  # noqa: BLE001 - whatever the client raises, the run is there and the line still names it
        print(f"{question.id} {path} unavailable ({type(error).__name__})", file=stdout, flush=True)
        print(f"{question.id}: no run URL: {type(error).__name__}", file=stderr, flush=True)
        return False
    print(f"{question.id} {path} {url}", file=stdout, flush=True)
    return True


def main(
    argv: Sequence[str] | None = None,
    *,
    environ: Mapping[str, str] | None = None,
    entry: EntryPoint | None = None,
    client: Any = None,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
    questions_file: Path = QUESTIONS_FILE,
) -> int:
    """Ask every question in the file once, in order. See the module text for the order of the checks and the exit codes. `entry` and `client` are for tests: ticket 4's agent and the LangSmith client."""
    out = sys.stdout if stdout is None else stdout
    err = sys.stderr if stderr is None else stderr
    env = os.environ if environ is None else environ
    if argv:
        print("make online-seed takes no arguments: edit evals/online/seed_questions.yaml instead.", file=err)
        return USAGE_EXIT
    try:
        questions = load_questions(questions_file)
    except SeedFileError as error:
        print(str(error), file=err)
        return USAGE_EXIT
    missing = missing_variables(env)
    if missing:
        print("make online-seed needs these variables, which are not set (in the environment or in .env):", file=err)
        for name in missing:
            print(name, file=err)
        return USAGE_EXIT
    project = guard_project(environ=env)
    client = tracing.shared_client() if client is None else client
    entry = default_entry() if entry is None else entry
    failed = 0
    with tracing_v2_enabled(project_name=project, client=client):
        for question in questions:
            try:
                run = ask_question(question, entry, client)
            except Exception as error:  # noqa: BLE001 - one question that fails is reported by id and error type, and the rest still run
                print(f"{question.id}: failed: {type(error).__name__}", file=err, flush=True)
                failed += 1
                continue
            failed += not _line(question, run, client, project, out, err)
    tracing.wait_for_feedback()  # the guard's feedback on each run is sent on a worker thread
    client.flush()
    if failed:
        print(f"{failed} of {len(questions)} questions did not finish cleanly.", file=err)
        return FAILED_EXIT
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
