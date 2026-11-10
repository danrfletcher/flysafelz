"""Score the production runs the pull let through, and post the scores to LangSmith as feedback (7 PR-2.F1, G7, G9, G10, GP3, GP5, E2, E5, E9).

    ONLINE_CHECKS                                    {online check name: 5a function}, the only place the 5a evaluators are imported
    evaluator_version(root=None) -> str              the first 12 hex characters of a SHA-256 over the files that decide a score (below)
    applicable_keys(run, reference) -> tuple[str]    the deterministic feedback keys that apply to a run, without the version marker (the judge keys are `judge_keys_for(path)`: they apply only when a judge is set up)
    missing_keys(applicable, existing, version)      the keys still to post: those not present at `version`, then the version marker if it is not present either
    score_run(run, reference, tree) -> {key: KeyScore}   every applicable check on one run, pure
    score_and_post(client, project, runs, version=None, judge=None, *, judge_missing=(), sample_pct=20) -> OnlineResult   score each run, judge a sample, post what it is missing, report each run
    select_for_judge(run_ids, pct) -> (judged, not_sampled)   which eligible runs the judge is called on: pure, no clock, no randomness (PR-3.F1)
    judge_setup_from_env(environ=None) -> JudgeSetup          JUDGE_SAMPLE_PCT (exit 2 on a bad value) and the judge, or the JUDGE_* names that are not set (PR-3.F1)
    main(argv=None, ...) -> int                       `make online-eval`: pull, score, judge, post, then the summary file `evals/results/online-<date>.md` (evals/online/summary.py); 0, or 1 when a run has an error (PR-4.F1)
    make_client(environ=None) -> Client               the real LangSmith client, built only when LANGSMITH_API_KEY is set (PR-4.F1)

Nothing above `main` builds a LangSmith client or reads a key: the client is passed in. It makes no scoring decision of its own: it builds
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
A trace that cannot be read or built into a reference is reported as errored too (stage `read trace`), and so is a run whose check gives a score outside 1, 0 or null (stage `score`: a programming error in a check, with nothing posted for the run). `read_project` failing is not caught: with no project UUID nothing can be posted.

The sampled judge (PR-3.F1, G8). 5a's judge is imported unchanged (`Judge`, `criteria_for`, `build_judge`) and called as `judge.judge(example, answer)`, once per judged run. It is given what 5a gives it for a labelled
example, built from the run alone: `inputs.question` (the last pilot message of the run's inputs), `inputs.now`, `inputs.lat` and `inputs.lon`, and `reference.path`, which is the run's own path. The answer is the
delivered answer with the path tag removed by ticket 4's `split_path`. 5a's `Judge.judge` takes no LZ name, so the run's `lz_name` is passed as the example's `fixtures.case`, which 5a turns into the place line
(`lz_name_from_case`: the words are capitalised and an underscore is a space, so `Blue Mill` comes back as `Blue Mill`); with no `lz_name` there is no `fixtures`, and 5a writes the coordinates. A run with neither
a usable name (one 5a reduces to nothing, such as `_` or a bare date, is not a name) nor coordinates, or with no pilot question in its inputs, cannot be given to the judge: it is that run's judge error. Nothing reads a label, and no model output is written anywhere but LangSmith feedback.
    key                          posted for
    online.judge.<criterion>     every criterion of the run's path in 5a's table (`criteria_for(path)`); a criterion 5a returns no grade for is a null, `no result from the judge`; 5a's own `judge error: ...` null is posted as it is
    online.judge.sampled         a run the sample left out: value `not_sampled`, score null, a comment holding the reason, `sample_pct=<n>` and the evaluator version
A run whose path is `unknown` (or any path 5a's table has no criteria for) has no judge key: 5a's table gives it none, so the key set is the deterministic one and nothing is posted for the judge.
Eligible runs: not errored, not skipped by the pull, with at least one judge key of their path missing at this version, and with no `not_sampled` record at this version made at the same percentage. A run that is not
eligible is not counted, not judged and not recorded. Of the eligible runs, with 50 or fewer every one is judged; with more, a run is judged when its run-id hash bucket is below `JUDGE_SAMPLE_PCT` (default 20), and the others
get one `not_sampled` record each (`JUDGE_SAMPLE_PCT` is 0 to 100; 100 judges all). A run left out is reconsidered only when `JUDGE_SAMPLE_PCT` differs from the `sample_pct` in its record; when it is left out again nothing is
posted, because the key exists at this version. The order of a run's posts is the deterministic keys, the judge keys (or the `not_sampled` record), then `online.evaluator_version`. A run whose judge call raised keeps its deterministic posts, has no judge key and no version marker, is reported as errored with stage `judge`, and is judged again by the next pass.
5a's `Judge.judge` catches a failed model call and an unreadable reply itself and returns the `judge error: ...` null for every criterion: that is 5a's handling, it is posted, and a null is never retried at this version.
With any of the JUDGE_* variables unset (`judge_setup_from_env`) no judge is built, nothing about the judge is posted, and the names are in `OnlineResult.judge_missing`.

The evaluator version covers the files that decide a score: `evals/agent/evaluators/*.py`, `evals/agent/judge.py` (it holds the judge's prompt), `evals/agent/judge_examples.yaml`, and `evals/online/*.py` except `seed.py`.
"""

import hashlib
import os
import re
import sys
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, NoReturn, Protocol
from uuid import UUID

from evals.agent.evaluators.answer import UNKNOWN_PATH, V1, ScoredAnswer, ToolCall
from evals.agent.evaluators.form import cited_guidance, decision_first, guard_pass, recheck_stated, window_stated
from evals.agent.evaluators.safety import no_verdict_beyond_day3
from evals.agent.evaluators.trajectory import forbidden_tools, step_cap_ok
from evals.agent.judge import Judge, build_judge, criteria_for, lz_name_from_case
from evals.agent.judge import feedback_key as judge_feedback_key
from evals.agent.judge_config import missing_judge_settings
from evals.online.pull import DEFAULT_SINCE, SINCE_VARIABLE, USAGE_EXIT, pull_runs, settings_from_env
from evals.online.reference import OnlineReference, build_reference
from evals.online.runs import final_answer, inputs_of, mapping, outputs_of
from flysafelz.agent.guard_flow import split_path

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
READ_TRACE, SCORE, LIST_FEEDBACK, POST, JUDGE = "read trace", "score", "list feedback", "post", "judge"  # where a run's error happened
JUDGED, NOT_SAMPLED = "judged", "not_sampled"  # what the sample decided for an eligible run; NOT_SAMPLED is also the value of the `online.judge.sampled` feedback
SAMPLED_KEY = PREFIX + "judge.sampled"
SAMPLE_PCT_VARIABLE = "JUDGE_SAMPLE_PCT"
DEFAULT_SAMPLE_PCT = 20
JUDGE_ALL_LIMIT = 50  # with this many eligible runs or fewer, every one is judged
NO_JUDGE_RESULT = "no result from the judge"
SAMPLE_PCT_PATTERN = re.compile(r"sample_pct=([0-9]+)")

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
    """One key's score on one run: 1, 0 or None, and its reason (without the version). `value` and `extra` are for the one key that has them (`online.judge.sampled`)."""

    key: str
    score: int | None
    comment: str
    value: str | None = None
    extra: Mapping[str, Any] | None = None


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
    """The deterministic feedback keys that apply to a run, in posting order, without `online.evaluator_version`. Every deterministic check applies to every run (a check that does not apply is a null).

    The judge keys are not here: they apply only to a run the judge is called on (`judge_keys_for`)."""
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


# --- the sampled judge (G8) ------------------------------------------------------------------------------------------------------------------


def _refuse(message: str) -> NoReturn:
    print(message, file=sys.stderr)
    raise SystemExit(USAGE_EXIT)


def parse_sample_pct(text: str) -> int:
    """`JUDGE_SAMPLE_PCT` as a whole number from 0 to 100. Anything else (`-1`, `101`, `abc`, empty) exits 2 and names the variable."""
    stripped = text.strip()
    if re.fullmatch(r"[0-9]+", stripped) is None or int(stripped) > 100:
        _refuse(f"{SAMPLE_PCT_VARIABLE}={text!r} is not a percentage: use a whole number from 0 to 100, for example 20.")
    return int(stripped)


def _checked_pct(pct: int) -> int:
    if isinstance(pct, bool) or not isinstance(pct, int) or not 0 <= pct <= 100:
        raise ValueError(f"the judge sample percentage must be a whole number from 0 to 100, not {pct!r}")
    return pct


def run_bucket(run_id: object) -> int:
    """The run's place in the sample, 0 to 99: its id's SHA-256 as a number, modulo 100. The same id always has the same bucket."""
    return int(hashlib.sha256(str(run_id).encode()).hexdigest(), 16) % 100


def select_for_judge[T](run_ids: Iterable[T], pct: int) -> tuple[list[T], list[T]]:
    """(judged, not_sampled) of the eligible run ids, each in the order given. With JUDGE_ALL_LIMIT or fewer every run is judged. With more, a run is judged when its bucket is below `pct`.

    Pure: no clock, no randomness, and the split does not depend on the order of the ids."""
    ids = list(run_ids)
    _checked_pct(pct)
    if len(ids) <= JUDGE_ALL_LIMIT:
        return ids, []
    judged = [run_id for run_id in ids if run_bucket(run_id) < pct]
    return judged, [run_id for run_id in ids if run_bucket(run_id) >= pct]


def judge_key(criterion: str) -> str:
    """`online.judge.<criterion>`: 5a's own `judge.<criterion>` key under the online prefix."""
    return PREFIX + judge_feedback_key(criterion)


def judge_keys_for(path: object) -> tuple[str, ...]:
    """The judge keys of a run's own path, from 5a's table in its order. `()` for `unknown` and every path the table has no criteria for."""
    return tuple(judge_key(criterion) for criterion in criteria_for(path))


def _present_at(existing: Iterable[Any], version: str) -> set[str]:
    return {posted.key for posted in existing if version_of(posted) == version}


def sampled_pct_at(existing: Iterable[Any], version: str) -> int | None:
    """The percentage in the run's `not_sampled` record at `version`; None when it has no record at that version, -1 when the record's comment holds no percentage."""
    found: int | None = None
    for posted in existing:
        if posted.key == SAMPLED_KEY and version_of(posted) == version:
            match = SAMPLE_PCT_PATTERN.search(str(getattr(posted, "comment", None) or ""))
            found = int(match.group(1)) if match else -1
    return found


def judge_eligible(path: object, existing: Iterable[Any], version: str, pct: int) -> bool:
    """Whether the judge's sample is to consider this run: its path has judge keys, at least one is missing at `version`, and it has no `not_sampled` record at `version` made at `pct`."""
    keys = judge_keys_for(path)
    existing = list(existing)
    if not keys or all(key in _present_at(existing, version) for key in keys):
        return False
    return sampled_pct_at(existing, version) != pct


def _human_text(message: object) -> str | None:
    """The text of one entry of a run's `messages` input when it is the pilot's: a plain string, or a human / user message as a LangChain or OpenAI mapping."""
    if isinstance(message, str):
        return message.strip() or None
    found = mapping(message)
    found = mapping(found.get("kwargs")) or found  # a serialised LangChain message keeps its fields under `kwargs`
    if found.get("role", found.get("type")) not in ("human", "user"):
        return None
    content = found.get("content")
    if isinstance(content, list):
        content = " ".join(part if isinstance(part, str) else str(mapping(part).get("text") or "") for part in content)
    return content.strip() or None if isinstance(content, str) else None


def question_of(run: Any) -> str:
    """The pilot's question: `inputs.question` when the run has one, else the last pilot message of `inputs.messages`. ValueError when there is none."""
    inputs = inputs_of(run)
    asked = inputs.get("question")
    if isinstance(asked, str) and asked.strip():
        return asked.strip()
    messages = inputs.get("messages")
    for message in reversed(messages if isinstance(messages, list) else []):
        text = _human_text(message)
        if text:
            return text
    raise ValueError("no pilot question in the run's inputs")


def judge_input(run: Any, reference: OnlineReference) -> tuple[dict[str, Any], ScoredAnswer]:
    """What 5a's `Judge.judge` is given for one run: an example with the question, `now`, the place and the run's own path, and the delivered answer with its path tag removed (ticket 4's `split_path`).

    The place is the run's `lz_name` (as the example's `fixtures.case`, see the module text), else the run's coordinates, which 5a writes as `the launch zone at <lat>, <lon>`. ValueError when there is no
    pilot question, or neither a name nor coordinates."""
    name = reference.lz_name if lz_name_from_case(reference.lz_name) is not None else None  # a name 5a reduces to nothing ('_', a bare date) is no name
    if name is None and (reference.lat is None or reference.lon is None):
        raise ValueError("the run has no usable lz_name and no lat and lon in its inputs")
    inputs = {"question": question_of(run), "now": reference.now.isoformat(), "lat": 0.0 if reference.lat is None else reference.lat, "lon": 0.0 if reference.lon is None else reference.lon}
    example: dict[str, Any] = {"id": str(getattr(run, "id", "run")), "inputs": inputs, "reference": {"path": reference.path}}
    if name is not None:
        example["fixtures"] = {"case": name}
    return example, ScoredAnswer(path=reference.path, text=split_path(final_answer(run))[1], kind=V1)


def judge_scores_of(path: object, grades: Mapping[str, Any]) -> dict[str, KeyScore]:
    """5a's grades for one run as online key scores, for the criteria of the run's path only. A criterion with no grade is a null, `no result from the judge`; everything else is 5a's score and comment."""
    scores: dict[str, KeyScore] = {}
    for criterion in criteria_for(path):
        grade = grades.get(judge_feedback_key(criterion))
        if grade is None:
            scores[judge_key(criterion)] = KeyScore(judge_key(criterion), None, NO_JUDGE_RESULT)
        else:
            scores[judge_key(criterion)] = KeyScore(judge_key(criterion), normalise_score(grade["score"]), str(grade.get("comment") or NO_JUDGE_RESULT))
    return scores


def not_sampled_score(eligible: int, pct: int) -> KeyScore:
    """The `online.judge.sampled` record of a run the sample left out: value `not_sampled`, score null, and a comment with the reason and the percentage (the version is added when it is posted)."""
    reason = f"left out of the judge sample: {eligible} eligible runs, more than {JUDGE_ALL_LIMIT}, chosen by run-id hash at sample_pct={pct}"
    return KeyScore(SAMPLED_KEY, None, reason, value=NOT_SAMPLED, extra={"judge_sample_pct": pct})


class JudgeLike(Protocol):
    """What the job needs of a judge: 5a's `Judge.judge`. Tests pass a fake with the same method."""

    def judge(self, example: Mapping[str, Any], answer: ScoredAnswer) -> Mapping[str, Any]: ...


@dataclass(frozen=True)
class JudgeSetup:
    """What the environment says about the judge: the sample percentage, and either the judge or the JUDGE_* names that are not set (then `judge` is None and nothing was built)."""

    judge: Judge | None
    pct: int
    missing: tuple[str, ...] = ()


def judge_setup_from_env(environ: Mapping[str, str] | None = None, build: Callable[[Mapping[str, str]], Judge] = build_judge) -> JudgeSetup:
    """`JUDGE_SAMPLE_PCT` first (a bad value exits 2 naming it, before any client exists), then the judge: `build(environ)` only when every JUDGE_* variable is set, else the missing names and no judge.

    `JUDGE_SAMPLE_PCT` is read from `environ` only, not from the `.env` file; the four JUDGE_* settings are also read from `.env` (judge_config). `make online-eval` starts the job with `uv run --env-file .env`,
    so a `JUDGE_SAMPLE_PCT` in `.env` reaches `environ` there; running `python -m evals.online.job` by hand does not load `.env`."""
    env = os.environ if environ is None else environ
    pct = parse_sample_pct(env[SAMPLE_PCT_VARIABLE]) if SAMPLE_PCT_VARIABLE in env else DEFAULT_SAMPLE_PCT
    missing = tuple(missing_judge_settings(env))
    return JudgeSetup(None if missing else build(env), pct, missing)


# --- one run, and the job --------------------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class RunError:
    stage: str  # READ_TRACE, SCORE, LIST_FEEDBACK, POST or JUDGE
    error_class: str
    message: str
    key: str | None = None  # the key that failed to post


@dataclass
class RunResult:
    run_id: UUID
    status: str = SCORED  # SCORED, ALREADY_SCORED or ERRORED
    path: str = UNKNOWN_PATH
    path_unknown: bool = False  # E2: PR-4.F1 counts these
    scores: dict[str, KeyScore] = field(default_factory=dict)  # every applicable deterministic key, whether or not this pass posted it
    posted: list[str] = field(default_factory=list)  # the keys this pass posted, in order
    error: RunError | None = None
    judge: str | None = None  # JUDGED or NOT_SAMPLED when the sample decided this run this pass; None when it did not (no judge, not eligible, or the run errored first)
    judge_scores: dict[str, KeyScore] = field(default_factory=dict)  # what the judge gave this pass (every key of the run's path, whether or not it was posted); empty when it was not called or raised
    judge_error: RunError | None = None  # the judge call raised: the run's judge keys stay missing for the next pass

    def record(self) -> dict[str, Any]:
        """What describes the run's deterministic scores and nothing about this pass: the same on a rerun that posts nothing (GP5)."""
        return {"run_id": str(self.run_id), "path": self.path, "path_unknown": self.path_unknown, "scores": {key: (kept.score, kept.comment) for key, kept in self.scores.items()}}


@dataclass
class OnlineResult:
    project: str
    version: str
    runs: list[RunResult] = field(default_factory=list)
    judge_enabled: bool = False  # a judge was passed in
    judge_missing: tuple[str, ...] = ()  # the JUDGE_* names that were not set, for PR-4.F1's summary (E4); the judge was not built
    sample_pct: int = DEFAULT_SAMPLE_PCT
    eligible: int = 0  # the runs the sample considered
    judge_calls: int = 0  # times `judge.judge` was called, one per judged run, a call that raised included

    @property
    def posted(self) -> int:
        return sum(len(result.posted) for result in self.runs)

    def with_status(self, status: str) -> list[RunResult]:
        return [result for result in self.runs if result.status == status]

    def with_judge(self, decision: str) -> list[RunResult]:
        """The runs the sample judged (JUDGED) or left out (NOT_SAMPLED) this pass."""
        return [result for result in self.runs if result.judge == decision]

    @property
    def errors(self) -> list[RunResult]:
        return self.with_status(ERRORED)


@dataclass
class _Prepared:
    """One run read and scored, with the feedback it already has: `reference` and `existing` are None when reading or listing failed (the run is then already errored)."""

    run: Any
    result: RunResult
    reference: OnlineReference | None = None
    existing: list[Any] | None = None

    @property
    def ready(self) -> bool:
        return self.reference is not None and self.existing is not None


def _short(error: Exception) -> str:
    return str(error).replace("\n", " ")[:MESSAGE_LIMIT]


def _read(client: Any, project: str, run: Any, result: RunResult) -> tuple[OnlineReference, list[Any]]:
    rows = client.list_runs(project_name=project, trace_id=getattr(run, "trace_id", None) or run.id)
    tree = [row for row in rows if row.id != run.id]
    reference = build_reference(run, tree)
    result.path, result.path_unknown = reference.path, reference.path == UNKNOWN_PATH
    return reference, tree


def _post(client: Any, run: Any, keys: Sequence[str], scores: Mapping[str, KeyScore], result: RunResult, session_id: UUID, version: str) -> None:
    for key in keys:
        if key == VERSION_KEY:
            kept = KeyScore(key, 1, "all keys posted")
        else:
            kept = scores[key]
        extra = {VERSION_EXTRA: version, **(kept.extra or {})}
        value = {} if kept.value is None else {"value": kept.value}
        try:
            client.create_feedback(run_id=run.id, key=key, score=kept.score, comment=comment_with_version(kept.comment, version), extra=extra, session_id=session_id, **value)
        except Exception as error:  # noqa: BLE001 - a LangSmith error on one key is reported for this run; the job goes on
            result.status, result.error = ERRORED, RunError(POST, type(error).__name__, _short(error), key)
            return  # the keys after it, and the version marker, wait for a rerun
        result.posted.append(key)


def _prepare(client: Any, project: str, run: Any) -> _Prepared:
    """Read the run's trace, score it, and read its existing feedback. A failure is the run's error, not the job's, and nothing is posted for it."""
    item = _Prepared(run, RunResult(run.id))
    try:
        reference, tree = _read(client, project, run, item.result)
    except Exception as error:  # noqa: BLE001 - a trace that cannot be read or built is this run's error, not the job's
        item.result.status, item.result.error = ERRORED, RunError(READ_TRACE, type(error).__name__, _short(error))
        return item
    try:
        item.result.scores = score_run(run, reference, tree)
    except Exception as error:  # noqa: BLE001 - a check that gives a score outside 1, 0 or None (normalise_score) is a programming error: named as its own stage, nothing is posted for the run
        item.result.status, item.result.error = ERRORED, RunError(SCORE, type(error).__name__, _short(error))
        return item
    item.reference = reference
    try:
        item.existing = list(client.list_feedback(run_ids=[run.id]))
    except Exception as error:  # noqa: BLE001 - without the existing feedback a post could duplicate a key, so nothing is posted
        item.result.status, item.result.error = ERRORED, RunError(LIST_FEEDBACK, type(error).__name__, _short(error))
    return item


def _decide(prepared: Sequence[_Prepared], judge: JudgeLike | None, pct: int, summary: OnlineResult) -> dict[UUID, str]:
    """For each eligible run, JUDGED or NOT_SAMPLED. Counts the eligible runs in `summary`. With no judge nothing is eligible: the judge is not set up, so no run is considered."""
    if judge is None:
        return {}
    eligible = [item for item in prepared if item.reference is not None and item.existing is not None and judge_eligible(item.reference.path, item.existing, summary.version, pct)]
    summary.eligible = len(eligible)
    judged, left_out = select_for_judge([item.run.id for item in eligible], pct)
    return {**dict.fromkeys(left_out, NOT_SAMPLED), **dict.fromkeys(judged, JUDGED)}


def _complete(client: Any, session_id: UUID, item: _Prepared, summary: OnlineResult, judge: JudgeLike | None, decision: str | None) -> RunResult:
    """Call the judge when the sample picked this run, then post what the run is missing: the deterministic keys, the judge keys or the `not_sampled` record, and the version marker last."""
    result = item.result
    if item.reference is None or item.existing is None:
        return result
    version, scores, applicable = summary.version, dict(result.scores), list(result.scores)
    if decision == JUDGED and judge is not None:
        try:
            example, answer = judge_input(item.run, item.reference)
            summary.judge_calls += 1
            result.judge_scores = judge_scores_of(item.reference.path, judge.judge(example, answer))
        except Exception as error:  # noqa: BLE001 - a judge call that raises is this run's judge error: its judge keys stay missing, and the other runs are still judged
            result.judge_scores = {}
            result.judge_error = RunError(JUDGE, type(error).__name__, _short(error))
        else:
            result.judge = JUDGED
            scores.update(result.judge_scores)
            applicable.extend(result.judge_scores)
    elif decision == NOT_SAMPLED:
        result.judge = NOT_SAMPLED
        if sampled_pct_at(item.existing, version) is None:  # a record at this version at another percentage is kept: nothing new is posted
            scores[SAMPLED_KEY] = not_sampled_score(summary.eligible, summary.sample_pct)
            applicable.append(SAMPLED_KEY)
    keys = missing_keys(applicable, item.existing, version)
    if result.judge_error is not None:
        keys = [key for key in keys if key != VERSION_KEY]  # the marker means every key posted: the judge's did not
        result.status = ERRORED
    elif not keys:
        result.status = ALREADY_SCORED
    _post(client, item.run, keys, scores, result, session_id, version)
    return result


def score_and_post(
    client: Any, project: str, runs: Sequence[Any], version: str | None = None, judge: JudgeLike | None = None, *, judge_missing: Sequence[str] = (), sample_pct: int = DEFAULT_SAMPLE_PCT
) -> OnlineResult:
    """Score each run, judge the sample, and post the feedback each run is missing at `version` (default: the code's own). With no runs nothing is read or posted, and `read_project` is not called.

    `judge` is 5a's judge (or anything with its `.judge(example, answer)`); None leaves every judge key missing for a later pass, and `judge_missing` (the JUDGE_* names `judge_setup_from_env` found unset)
    is carried in the result. `sample_pct` is JUDGE_SAMPLE_PCT. Every run is read before any is posted, because the 50-run rule needs the eligible count of the whole pass.
    """
    _checked_pct(sample_pct)
    result = OnlineResult(project, evaluator_version() if version is None else version, judge_enabled=judge is not None, judge_missing=tuple(judge_missing), sample_pct=sample_pct)
    if not runs:
        return result
    session_id = client.read_project(project_name=project).id
    prepared = [_prepare(client, project, run) for run in runs]
    decisions = _decide(prepared, judge, sample_pct, result)
    for item in prepared:
        result.runs.append(_complete(client, session_id, item, result, judge, decisions.get(item.run.id)))
    return result


def _judge_line(result: OnlineResult) -> str | None:
    if result.judge_missing:
        return f"judge not run: {', '.join(result.judge_missing)} not set; the judge keys stay missing for a later pass"
    if not result.judge_enabled:
        return None
    return (
        f"judge: {result.eligible} eligible, {len(result.with_judge(JUDGED))} judged ({result.judge_calls} call(s)), "
        f"{len(result.with_judge(NOT_SAMPLED))} not sampled, at {SAMPLE_PCT_VARIABLE}={result.sample_pct}"
    )


def report_lines(result: OnlineResult) -> list[str]:
    """A plain-text report of a pass: one line per run (a run whose judge call raised has a second line), the judge's line when there is one, then the totals."""
    lines = [f"project {result.project}, evaluator version {result.version}: {len(result.runs)} run(s)"]
    for item in result.runs:
        if item.status == ERRORED and item.error is not None:
            where = f"{item.error.stage} {item.error.key}" if item.error.key else item.error.stage
            lines.append(f"{item.run_id} errored at {where}: {item.error.error_class}: {item.error.message} (posted {len(item.posted)} before it)")
        elif item.status == ERRORED and item.judge_error is None:
            lines.append(f"{item.run_id} errored")
        elif item.status == ALREADY_SCORED:
            lines.append(f"{item.run_id} already scored")
        elif item.status == SCORED:
            lines.append(f"{item.run_id} scored, posted {len(item.posted)}: {', '.join(item.posted)}")
        if item.judge_error is not None:
            lines.append(f"{item.run_id} errored at {item.judge_error.stage}: {item.judge_error.error_class}: {item.judge_error.message} (judge keys not posted; posted {len(item.posted)} other)")
    judged = _judge_line(result)
    if judged is not None:
        lines.append(judged)
    lines.append(f"{len(result.with_status(SCORED))} scored, {len(result.with_status(ALREADY_SCORED))} already scored, {len(result.errors)} errored; {result.posted} feedback posted")
    return lines


# --- the command: `make online-eval` (PR-4.F1, G11, GP2) -------------------------------------------------------------------------------------


API_KEY_VARIABLE = "LANGSMITH_API_KEY"
PROGRAM = "evals.online.job"
FAILED_EXIT = 1  # the summary was written, and at least one run has an error: a rerun posts only what is missing


def make_client(environ: Mapping[str, str] | None = None, factory: Callable[..., Any] | None = None) -> Any:
    """The real LangSmith client, the one place one is built. Exits 2, naming LANGSMITH_API_KEY and before anything is built or sent, when the key is not set. `factory` is for tests."""
    env = os.environ if environ is None else environ
    key = env.get(API_KEY_VARIABLE, "").strip()
    if not key:
        _refuse(
            f"{API_KEY_VARIABLE} is not set: it is needed to read the production runs and post feedback on LangSmith. Nothing was sent. Set it in the environment or in .env, then run `make online-eval` again."
        )
    if factory is None:
        from langsmith import Client

        factory = Client
    return factory(api_key=key)


def utc_now() -> datetime:
    return datetime.now(UTC)


def main(
    argv: Sequence[str] | None = None,
    *,
    environ: Mapping[str, str] | None = None,
    client: Any = None,
    clock: Callable[[], datetime] = utc_now,
    build: Callable[[Mapping[str, str]], Judge] = build_judge,
    results_dir: Path | None = None,
) -> int:
    """The whole job: pull, allow-list, score, judge the sample, post, then write the summary (`evals/results/online-<date>.md`, dated in Europe/London) and print its lines and path.

    Everything it is told comes through the environment (`LANGSMITH_API_KEY`, `LANGSMITH_PROJECT`, `SINCE`, `LIMIT`, `JUDGE_SAMPLE_PCT`, `JUDGE_*`), never argv. The order is: the project guard, `SINCE` and `LIMIT`
    (each a bad value exits 2 naming it), then the key (exit 2 naming it), then `JUDGE_SAMPLE_PCT` and the judge (a bad percentage exits 2 naming it), then the first LangSmith call, so a refusal writes no file and makes no call. Returns 0, or 1 when the summary was
    written and a run has an error (a post that failed, a judge call that raised, a trace or its feedback that could not be read): the file is never lost, and a rerun posts only what is missing.
    `client`, `clock`, `build` (the judge builder) and `results_dir` are for tests.
    """
    from evals.online import summary  # here, not at the top: the summary reads this module's keys and constants

    env = os.environ if environ is None else environ
    if list(argv or []):
        _refuse(f"{PROGRAM} takes no arguments: set SINCE, LIMIT and JUDGE_SAMPLE_PCT in the environment (make online-eval SINCE=24h)")
    settings = settings_from_env(env)
    live = make_client(env) if client is None else client
    setup = judge_setup_from_env(env, build)
    now = clock()
    pulled = pull_runs(live, settings.project, settings.since, settings.limit, now)
    result = score_and_post(live, settings.project, pulled.runs, judge=setup.judge, judge_missing=setup.missing, sample_pct=setup.pct)
    print("\n".join(report_lines(result)))
    scored = summary.collect(live, result, pulled.runs)
    data = summary.ReportInput(
        day=summary.summary_date(now),
        project=settings.project,
        version=result.version,
        since=env.get(SINCE_VARIABLE, DEFAULT_SINCE).strip(),
        limit=settings.limit,
        limit_hit=pulled.limit_hit,
        read=pulled.read,
        runs=scored,
        skipped_by_source=dict(pulled.by_source),
        skipped_by_reason=dict(pulled.by_reason),
        judge_missing=result.judge_missing,
        link=summary.run_link(live, settings.project, pulled.runs),
    )
    report = summary.build_report(data)
    written = summary.write_summary(summary.render_markdown(report), data.day, summary.RESULTS_DIR if results_dir is None else results_dir)
    print("\n".join(summary.summary_lines(report)))
    print(f"summary written: {written}")
    errored = [item for item in scored if item.error is not None]
    if errored:
        print(f"{len(errored)} run(s) had an error and are listed in the summary: run `make online-eval` again, it posts only what is missing", file=sys.stderr)
        return FAILED_EXIT
    return 0


__all__ = [
    "ALREADY_SCORED",
    "DETERMINISTIC_KEYS",
    "ERRORED",
    "JUDGED",
    "NOT_SAMPLED",
    "ONLINE_CHECKS",
    "SAMPLED_KEY",
    "SCORED",
    "VERSION_KEY",
    "JudgeSetup",
    "KeyScore",
    "OnlineResult",
    "RunError",
    "RunResult",
    "applicable_keys",
    "evaluator_version",
    "feedback_key",
    "judge_eligible",
    "judge_input",
    "judge_key",
    "judge_keys_for",
    "judge_setup_from_env",
    "main",
    "make_client",
    "missing_keys",
    "normalise_score",
    "parse_sample_pct",
    "report_lines",
    "score_and_post",
    "score_run",
    "select_for_judge",
]


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
