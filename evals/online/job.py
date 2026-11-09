"""Score the production runs the pull let through, and post the scores to LangSmith as feedback (7 PR-2.F1, G7, G9, G10, GP3, GP5, E2, E5, E9).

    ONLINE_CHECKS                                    {online check name: 5a function}, the only place the 5a evaluators are imported
    evaluator_version(root=None) -> str              the first 12 hex characters of a SHA-256 over the files that decide a score (below)
    applicable_keys(run, reference) -> tuple[str]    the feedback keys that apply to a run, without the version marker (PR-3.F1 adds the judge keys here)
    missing_keys(applicable, existing, version)      the keys still to post: those not present at `version`, then the version marker if it is not present either
    score_run(run, reference, tree) -> {key: KeyScore}   every applicable check on one run, pure
    score_and_post(client, project, runs, version=None) -> OnlineResult   score each run, post what it is missing, report each run

Nothing here builds a LangSmith client or reads a key: the client is passed in (PR-4.F1's `make online-eval` builds the real one). It makes no scoring decision of its own: it builds
what 5a's evaluators read (an `example` mapping and a `ScoredAnswer`) from the run and its trace, calls them unchanged, and turns each result into a score of 1, 0 or null.

The checks (G7). ONLINE_CHECKS maps the online name to the 5a function; the feedback key is `online.<name>`, in this order, which is the order they are posted in:
    guard_pass               form.guard_pass              the guard record is the root run's outputs `guard`; none gives null, `no guard record`
    decision_first           form.decision_first
    cited_guidance           form.cited_guidance          applies when the run called `search_guidance`
    recheck_stated           form.recheck_stated
    assumed_window_stated    form.window_stated           5a's name differs
    no_verdict_beyond_day3   safety.no_verdict_beyond_day3
    step_cap                 trajectory.step_cap_ok       5a's name differs
    decline_no_tools         trajectory.forbidden_tools   5a's name differs; the reference's forbidden tools are every agent tool on the decline path, none elsewhere
A run is scored by every check, so a key is always posted; a check that does not apply to the run is a null with a reason.

What 5a is given. The `example` is the synthetic reference (reference.py) in 5a's shape: `inputs.now`, `reference.path`, the tool lists, and, when the trace has an `assess` run, the window it assessed and
its segments (as `reference.window` and `reference.verdict_by_segment`, which is how 5a's `decision_first` finds a verdict to lead with). They are what the run itself assessed, not labels: no
`must_mention`, no sheet label and no `intended_path` is read. The `ScoredAnswer` is the delivered answer (`outputs.answer`), the path, the assessed segments, the guard record, and the tool runs below the root in the
order they started, with the documents a `search_guidance` run returned.

Null reasons (the comment of a null starts with one of these; PR-4.F1 counts them):
    no guard record    `guard_pass` with no `guard` in the root run's outputs
    no assess          a check that needs the assessed window or segments (`decision_first`, `recheck_stated`, `assumed_window_stated`, `no_verdict_beyond_day3`) on a trace with no readable `assess` run
    path unknown       `decline_no_tools` on a run whose path is `unknown` (E2); the path-independent checks still score, and the run is flagged `path_unknown`
    not applicable     5a returned None: the check does not apply to this run (a rules answer has no window, a window within the current hour has no re-check line, ...)
    check error: <C>   the 5a function raised an exception of class C; the other keys still score

A score is exactly 1, 0 or null. True and False, 1.0 and 0.0, and None are normalised; any other value raises ValueError: a 5a function that gives 0.5 is a programming error, not a run's fault.

Posting (G9, G10). One `create_feedback(run_id=, key=online.<name>, score=, comment=, extra=, session_id=)` per key, with `session_id` the project's UUID from `read_project`, read once per job and not at all
when there are no runs. The comment is the reason and ends with ` | evaluator_version=<12 hex>`, and `extra` holds the same version. Before posting, `list_feedback(run_ids=[run.id])` is read; a key counts as present
when feedback with its name carries the current version (in `extra`, else read from the comment). A null score counts as present: a null is never retried. The keys posted are the missing ones in order,
then `online.evaluator_version` last, and only if every key before it posted. A feedback from an older version is never edited or deleted, and a new version posts every key again. A run is `already scored` only when
every applicable key and the version marker exist at this version; it then gets no write. If `create_feedback` raises, the run is reported (run id, key, error class), nothing after that key is posted for it
(the version marker included), and the job goes on to the next run; a rerun posts only what is missing. If `list_feedback` raises, nothing is posted for that run (a blind post could duplicate a key) and the run is reported as errored.
A trace that cannot be read or built into a reference is reported as errored too. `read_project` failing is not caught: with no project UUID nothing can be posted.

The evaluator version covers the files that decide a score: `evals/agent/evaluators/*.py`, `evals/agent/judge.py` (it holds the judge's prompt), `evals/agent/judge_examples.yaml`, and `evals/online/*.py` except `seed.py`.
"""

import hashlib
import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import UUID

from evals.agent.evaluators.answer import UNKNOWN_PATH, V1, ScoredAnswer, ToolCall
from evals.agent.evaluators.form import cited_guidance, decision_first, guard_pass, recheck_stated, window_stated
from evals.agent.evaluators.safety import no_verdict_beyond_day3
from evals.agent.evaluators.trajectory import forbidden_tools, step_cap_ok
from evals.online.reference import OnlineReference, build_reference
from evals.online.runs import final_answer, mapping, outputs_of

ROOT = Path(__file__).resolve().parents[2]
PREFIX = "online."
GUARD_PASS = "guard_pass"
DECISION_FIRST = "decision_first"
CITED_GUIDANCE = "cited_guidance"
RECHECK_STATED = "recheck_stated"
ASSUMED_WINDOW_STATED = "assumed_window_stated"
NO_VERDICT_BEYOND_DAY3 = "no_verdict_beyond_day3"
STEP_CAP = "step_cap"
DECLINE_NO_TOOLS = "decline_no_tools"
VERSION_KEY = "online.evaluator_version"

# The one place the 5a evaluators are imported and named. Each is called unchanged: (example, answer) -> Feedback | None.
ONLINE_CHECKS: dict[str, Callable[..., Any]] = {
    GUARD_PASS: guard_pass,
    DECISION_FIRST: decision_first,
    CITED_GUIDANCE: cited_guidance,
    RECHECK_STATED: recheck_stated,
    ASSUMED_WINDOW_STATED: window_stated,
    NO_VERDICT_BEYOND_DAY3: no_verdict_beyond_day3,
    STEP_CAP: step_cap_ok,
    DECLINE_NO_TOOLS: forbidden_tools,
}
NEEDS_ASSESS = frozenset({DECISION_FIRST, RECHECK_STATED, ASSUMED_WINDOW_STATED, NO_VERDICT_BEYOND_DAY3})

NO_GUARD_RECORD = "no guard record"
PATH_UNKNOWN = "path unknown"
NOT_APPLICABLE = "not applicable"
CHECK_ERROR = "check error"

SCORED, ALREADY_SCORED, ERRORED = "scored", "already scored", "errored"
READ_TRACE, LIST_FEEDBACK, POST = "read trace", "list feedback", "post"  # where a run's error happened

GUARD_RESULTS = ("passed", "retried", "fallback")  # ticket 4's GuardResult
TOOL_RUN = "tool"
SEARCH_GUIDANCE = "search_guidance"
MAX_DEPTH = 6  # how deep into a tool run's outputs the guidance chunks are looked for
VERSION_LENGTH = 12
VERSION_EXTRA = "evaluator_version"
VERSION_PATTERN = re.compile(r"evaluator_version=([0-9a-f]{12})\s*$")
MESSAGE_LIMIT = 200  # characters of an error message kept in the report
COVERED_GLOBS = ("evals/agent/evaluators/*.py", "evals/online/*.py")
COVERED_FILES = ("evals/agent/judge.py", "evals/agent/judge_examples.yaml")
EXCLUDED_FILES = frozenset({"evals/online/seed.py"})


def feedback_key(check: str) -> str:
    """The LangSmith feedback key of an online check: `online.<check>`."""
    return PREFIX + check


DETERMINISTIC_KEYS = tuple(feedback_key(check) for check in ONLINE_CHECKS)


# --- the evaluator version (G10) ------------------------------------------------------------------------------------------------------------


def covered_files(root: Path | None = None) -> list[str]:
    """The relative paths (with `/`) the evaluator version covers, sorted. A file the version needs that is missing raises FileNotFoundError."""
    base = ROOT if root is None else root
    found = {path.relative_to(base).as_posix() for pattern in COVERED_GLOBS for path in base.glob(pattern)}
    for name in COVERED_FILES:
        if not (base / name).is_file():
            raise FileNotFoundError(f"the evaluator version covers {name}, which is not in {base}")
        found.add(name)
    return sorted(found - EXCLUDED_FILES)


def evaluator_version(root: Path | None = None) -> str:
    """The first 12 hex characters of a SHA-256 over the sorted relative paths and bytes of the covered files."""
    base = ROOT if root is None else root
    digest = hashlib.sha256()
    for relative in covered_files(base):
        digest.update(relative.encode("utf-8") + b"\0" + (base / relative).read_bytes() + b"\0")
    return digest.hexdigest()[:VERSION_LENGTH]


def comment_with_version(reason: str, version: str) -> str:
    return f"{reason} | {VERSION_EXTRA}={version}"


def version_of(posted: Any) -> str | None:
    """The evaluator version a posted feedback was written at: its `extra`, else the end of its comment; None for feedback this job did not write."""
    found = mapping(getattr(posted, "extra", None)).get(VERSION_EXTRA)
    if isinstance(found, str) and re.fullmatch(r"[0-9a-f]{12}", found):
        return found
    match = VERSION_PATTERN.search(str(getattr(posted, "comment", None) or ""))
    return match.group(1) if match else None


# --- what 5a is given ------------------------------------------------------------------------------------------------------------------------


def _find_chunks(value: object, depth: int = 0) -> list[Any]:
    """The `chunks` list of a search_guidance result, wherever the run's outputs hold it (a plain tool message or a Command's update)."""
    if depth > MAX_DEPTH:
        return []
    if isinstance(value, Mapping):
        if isinstance(value.get("chunks"), list):
            return value["chunks"]
        children: list[object] = list(value.values())
    elif isinstance(value, list):
        children = list(value)
    else:
        return []
    for item in children:
        found = _find_chunks(item, depth + 1)
        if found:
            return found
    return []


def _guides_of(tool_run: Any) -> tuple[str, ...]:
    documents = (mapping(chunk).get("document") for chunk in _find_chunks(outputs_of(tool_run)))
    return tuple(dict.fromkeys(str(document) for document in documents if document))


def _started(run: Any) -> datetime:
    started = getattr(run, "start_time", None)
    return started if isinstance(started, datetime) and started.tzinfo is not None else datetime.min.replace(tzinfo=UTC)


def tool_calls_of(tree: Sequence[Any]) -> tuple[ToolCall, ...]:
    """One ToolCall per tool run below the root, in the order they started; a `search_guidance` run carries the documents it returned."""
    tools = [child for child in tree if getattr(child, "parent_run_id", None) is not None and getattr(child, "run_type", None) == TOOL_RUN]
    return tuple(ToolCall(str(child.name), _guides_of(child) if child.name == SEARCH_GUIDANCE else ()) for child in sorted(tools, key=_started))


def guard_of(run: Any) -> str | None:
    """The guard record: the root run's outputs `guard` (`passed`, `retried`, `fallback`), or None when it has none or holds something else."""
    found = outputs_of(run).get("guard")
    return found if found in GUARD_RESULTS else None


def answer_of(run: Any, reference: OnlineReference, tree: Sequence[Any]) -> ScoredAnswer:
    """The run as 5a's `ScoredAnswer`: its delivered answer, path, assessed segments, tool calls and guard record."""
    segments = reference.assess.segments if reference.assess is not None else ()
    return ScoredAnswer(path=reference.path, segments=segments, text=final_answer(run), tool_calls=tool_calls_of(tree), guard=guard_of(run), kind=V1)


def example_of(run: Any, reference: OnlineReference) -> dict[str, Any]:
    """The synthetic reference in the shape 5a's evaluators read an example in. Nothing in it is a label: see the module text."""
    held: dict[str, Any] = {"path": reference.path, "required_tools": list(reference.required_tools), "forbidden_tools": list(reference.forbidden_tools)}
    if reference.assess is not None:
        held["window"] = [reference.assess.start.isoformat(), reference.assess.end.isoformat()]
        held["verdict_by_segment"] = [{"start": s.start, "end": s.end, "verdict": s.verdict, "approaching_limits": s.approaching_limits} for s in reference.assess.segments]
        held["tier"] = []
        held["notam_actions"] = {}
    return {"id": str(getattr(run, "id", "run")), "inputs": {"now": reference.now.isoformat()}, "reference": held}


# --- scoring (G7) ----------------------------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class KeyScore:
    """One key's score on one run: 1, 0 or None, and its reason (without the version)."""

    key: str
    score: int | None
    comment: str


def normalise_score(value: object) -> int | None:
    """1, 0 or None from True/False, 1/0, 1.0/0.0 or None. Anything else is a programming error in a check and raises ValueError."""
    if value is None:
        return None
    if isinstance(value, (bool, int, float)) and value in (0, 1):
        return int(value)
    raise ValueError(f"a check's score must be 1, 0 or None, not {value!r}")


def _null_reason(check: str, reference: OnlineReference, answer: ScoredAnswer) -> str:
    if check == GUARD_PASS and answer.guard is None:
        return NO_GUARD_RECORD
    if check == DECLINE_NO_TOOLS and reference.path == UNKNOWN_PATH:
        return PATH_UNKNOWN
    if check in NEEDS_ASSESS and reference.assess is None:
        return reference.assess_reason or NOT_APPLICABLE
    return NOT_APPLICABLE


def _score_one(check: str, function: Callable[..., Any], example: Mapping[str, Any], answer: ScoredAnswer, reference: OnlineReference) -> KeyScore:
    key = feedback_key(check)
    if check in NEEDS_ASSESS and reference.assess is None:
        return KeyScore(key, None, _null_reason(check, reference, answer))  # the example has no window, so 5a is not asked
    if check == GUARD_PASS and answer.guard is None:
        return KeyScore(key, None, NO_GUARD_RECORD)
    try:
        found = function(example, answer)
    except Exception as error:  # noqa: BLE001 - whatever a check raises is that key's null; the run's other keys still score
        return KeyScore(key, None, f"{CHECK_ERROR}: {type(error).__name__}")
    if found is None:
        return KeyScore(key, None, _null_reason(check, reference, answer))
    return KeyScore(key, normalise_score(found["score"]), str(found["comment"]))


def applicable_keys(run: Any, reference: OnlineReference) -> tuple[str, ...]:
    """The feedback keys that apply to a run, in posting order, without `online.evaluator_version`. Every deterministic check applies to every run (a check that does not apply is a null)."""
    return DETERMINISTIC_KEYS


def score_run(run: Any, reference: OnlineReference, tree: Sequence[Any]) -> dict[str, KeyScore]:
    """Every applicable key's score for one run, keyed by feedback key in posting order. Pure: nothing is read from or sent to LangSmith."""
    example, answer = example_of(run, reference), answer_of(run, reference, tree)
    scored = {feedback_key(check): _score_one(check, function, example, answer, reference) for check, function in ONLINE_CHECKS.items()}
    return {key: scored[key] for key in applicable_keys(run, reference)}


# --- what is missing (G9, G10) ---------------------------------------------------------------------------------------------------------------


def missing_keys(applicable: Iterable[str], existing: Iterable[Any], version: str) -> list[str]:
    """The keys to post, in order. A key is present when feedback of that name carries `version`; a null score is present. The missing applicable keys come first, then
    `online.evaluator_version` unless it is present too. An empty list means every applicable key and the version marker exist at this version: the run is already scored."""
    present = {posted.key for posted in existing if version_of(posted) == version}
    missing = [key for key in applicable if key not in present]
    return missing if VERSION_KEY in present else [*missing, VERSION_KEY]


# --- one run, and the job --------------------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class RunError:
    stage: str  # READ_TRACE, LIST_FEEDBACK or POST
    error_class: str
    message: str
    key: str | None = None  # the key that failed to post


@dataclass
class RunResult:
    run_id: UUID
    status: str = SCORED  # SCORED, ALREADY_SCORED or ERRORED
    path: str = UNKNOWN_PATH
    path_unknown: bool = False  # E2: PR-4.F1 counts these
    scores: dict[str, KeyScore] = field(default_factory=dict)  # every applicable key, whether or not this pass posted it
    posted: list[str] = field(default_factory=list)  # the keys this pass posted, in order
    error: RunError | None = None

    def record(self) -> dict[str, Any]:
        """What describes the run's scores and nothing about this pass: the same on a rerun that posts nothing (GP5)."""
        return {"run_id": str(self.run_id), "path": self.path, "path_unknown": self.path_unknown, "scores": {key: (kept.score, kept.comment) for key, kept in self.scores.items()}}


@dataclass
class OnlineResult:
    project: str
    version: str
    runs: list[RunResult] = field(default_factory=list)

    @property
    def posted(self) -> int:
        return sum(len(result.posted) for result in self.runs)

    def with_status(self, status: str) -> list[RunResult]:
        return [result for result in self.runs if result.status == status]

    @property
    def errors(self) -> list[RunResult]:
        return self.with_status(ERRORED)


def _short(error: Exception) -> str:
    return str(error).replace("\n", " ")[:MESSAGE_LIMIT]


def _read_and_score(client: Any, project: str, run: Any, result: RunResult) -> None:
    rows = client.list_runs(project_name=project, trace_id=getattr(run, "trace_id", None) or run.id)
    tree = [row for row in rows if row.id != run.id]
    reference = build_reference(run, tree)
    result.path, result.path_unknown = reference.path, reference.path == UNKNOWN_PATH
    result.scores = score_run(run, reference, tree)


def _post(client: Any, run: Any, keys: Sequence[str], result: RunResult, session_id: UUID, version: str) -> None:
    for key in keys:
        if key == VERSION_KEY:
            score, reason = 1, "all keys posted"
        else:
            kept = result.scores[key]
            score, reason = kept.score, kept.comment
        try:
            client.create_feedback(run_id=run.id, key=key, score=score, comment=comment_with_version(reason, version), extra={VERSION_EXTRA: version}, session_id=session_id)
        except Exception as error:  # noqa: BLE001 - a LangSmith error on one key is reported for this run; the job goes on
            result.status, result.error = ERRORED, RunError(POST, type(error).__name__, _short(error), key)
            return  # the keys after it, and the version marker, wait for a rerun
        result.posted.append(key)


def _process(client: Any, project: str, session_id: UUID, run: Any, version: str) -> RunResult:
    result = RunResult(run.id)
    try:
        _read_and_score(client, project, run, result)
    except Exception as error:  # noqa: BLE001 - a trace that cannot be read or built is this run's error, not the job's
        result.status, result.error = ERRORED, RunError(READ_TRACE, type(error).__name__, _short(error))
        return result
    try:
        existing = list(client.list_feedback(run_ids=[run.id]))
    except Exception as error:  # noqa: BLE001 - without the existing feedback a post could duplicate a key, so nothing is posted
        result.status, result.error = ERRORED, RunError(LIST_FEEDBACK, type(error).__name__, _short(error))
        return result
    keys = missing_keys(result.scores, existing, version)
    if not keys:
        result.status = ALREADY_SCORED
        return result
    _post(client, run, keys, result, session_id, version)
    return result


def score_and_post(client: Any, project: str, runs: Sequence[Any], version: str | None = None) -> OnlineResult:
    """Score each run and post the feedback it is missing at `version` (default: the code's own). With no runs nothing is read or posted, and `read_project` is not called."""
    result = OnlineResult(project, evaluator_version() if version is None else version)
    if not runs:
        return result
    session_id = client.read_project(project_name=project).id
    for run in runs:
        result.runs.append(_process(client, project, session_id, run, result.version))
    return result


def report_lines(result: OnlineResult) -> list[str]:
    """A plain-text report of a pass: one line per run, then the totals."""
    lines = [f"project {result.project}, evaluator version {result.version}: {len(result.runs)} run(s)"]
    for item in result.runs:
        if item.status == ERRORED and item.error is not None:
            where = f"{item.error.stage} {item.error.key}" if item.error.key else item.error.stage
            lines.append(f"{item.run_id} errored at {where}: {item.error.error_class}: {item.error.message} (posted {len(item.posted)} before it)")
        elif item.status == ALREADY_SCORED:
            lines.append(f"{item.run_id} already scored")
        else:
            lines.append(f"{item.run_id} scored, posted {len(item.posted)}: {', '.join(item.posted)}")
    lines.append(f"{len(result.with_status(SCORED))} scored, {len(result.with_status(ALREADY_SCORED))} already scored, {len(result.errors)} errored; {result.posted} feedback posted")
    return lines


__all__ = [
    "ALREADY_SCORED",
    "DETERMINISTIC_KEYS",
    "ERRORED",
    "ONLINE_CHECKS",
    "SCORED",
    "VERSION_KEY",
    "KeyScore",
    "OnlineResult",
    "RunError",
    "RunResult",
    "applicable_keys",
    "evaluator_version",
    "feedback_key",
    "missing_keys",
    "normalise_score",
    "report_lines",
    "score_and_post",
    "score_run",
]
