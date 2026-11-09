"""The results file of an Experiment (5a PR-10.F1, G16): `evals/results/<experiment>.md`, built from the scored runs of a finished v1 or v0 experiment.

    ReportRun(...)                      one example in one rep, as the report reads it: scores by evaluator key, latency, usage, error, LangSmith run id
    ReportInput(...)                    everything the file shows: the runs, the dataset's examples, the models, the wall-clock time, the price, the extra lines
    from_experiment(result, evaluation, records, ...) -> ReportInput    the runner's ExperimentResult and the evaluators' Evaluation as a ReportInput
    render(report) -> str               the Markdown, sections in this order: Headline, Slices, Failing ids, Errored ids, Ops, Models
    write_report(text, experiment, results_dir) -> Path    the file, written whole in one go (a temp file in the same folder, then os.replace); only under `results_dir`
    ops_of(report) -> Ops               the Ops numbers (latency p50/p95, tokens, reader calls failed, wall-clock), for the tests and the page
    reader_tokens(results) -> list[TokenUsage]    the usage of the reader's model calls; a STORED reading made no call and is skipped, never counted as 0

Headline. One row per metric: missed blockers, hard-restriction recall, verdict match, path accuracy, noise, trajectory (required tools, forbidden tools, step cap), guard and
each judge criterion's pass rate. A value is the mean over the 3 per-rep means with the min and max of those means beside it (spread = min-max). A rep with no score for the
metric is left out of the means; one rep only gives a spread equal to the mean (shown, not hidden). A metric that is null for the whole run (trajectory for v0) is `n/a`, never 0.

Slices. One table per dimension of the example's metadata (path, case_type, failure_source, notam_category, airspace_context, compound; a missing `compound` is false): a row per
value with n (the examples that have runs) and the headline metrics that applied to the slice (pooled over its scored runs). A value with n = 0 has no row.

Failing ids and errored ids. A failing id is an example with a scored run that failed a check (a missed blocker, a recall, verdict match, path, trajectory, guard or judge score
below full); each is listed with the reps and the checks and a LangSmith link for each failing run. An errored id (the harness failed, there is no answer) is listed in its own
section, never scored; the run is `incomplete` when MORE than 10% of the planned runs errored (exactly 10% is not): runner.is_incomplete, the runner's own rule.

Ops. Latency p50 and p95 per example run (`nearest_rank`, imported from evals.reader_eval.report: no second percentile function), the input and output tokens the model
responses reported, the est. cost (evals.agent.pricing: the fixed label, the list price of each model, its source url and date), the reader calls that failed and became UNVERIFIED, and the
wall-clock time. The tokens are the agent's responses plus the NOTAM reader's (a v1 run); a stored reading made no call and adds nothing. The judge's calls are not in them (the file
says so). Missing usage is never a 0: an experiment with none shows `n/a`.

Models. The model per role (agent, reader, judge, v0), then the extra lines the caller adds (PR-7's self-preference line, PR-9's reading-plus-arithmetic line) and the note that examples
sharing a case reused rep-1 readings.

Nothing here scores, calls a model or a service, or opens a path outside the results folder.
"""

import os
import re
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Any
from uuid import UUID

from evals.agent import runner
from evals.agent.evaluators.aggregate import Evaluation
from evals.agent.evaluators.blocker import MISSED_BLOCKER_KEY, RECALL_KEY
from evals.agent.evaluators.form import GUARD_KEY
from evals.agent.evaluators.noise import NOISE_KEY
from evals.agent.evaluators.paths import PATH_KEY
from evals.agent.evaluators.trajectory import FORBIDDEN_KEY, REQUIRED_KEY, STEP_KEY
from evals.agent.evaluators.verdict import VERDICT_KEY
from evals.agent.judge import CRITERIA, feedback_key
from evals.agent.pricing import Price, cost_line, cost_line_by_role
from evals.reader_eval.report import nearest_rank
from flysafelz.reader.contract import ReaderResult, ReaderStatus, TokenUsage

RESULTS_DIR = runner.REPO / "evals" / "results"
NOT_APPLICABLE = "n/a"
SECTIONS = ("Headline", "Slices", "Failing ids", "Errored ids", "Ops", "Models")
ROLES = ("agent", "reader", "judge", "v0")
SLICE_DIMENSIONS = ("path", "case_type", "failure_source", "notam_category", "airspace_context", "compound")
_PIPE = re.compile(r"\|")
SHARED_CASE_NOTE = (
    "Examples that share a case reused rep-1 readings: the reader cache is one file per experiment, so a reading made once is stored and read back by later reps and by other examples of the case."
)
P50, P95 = Fraction(1, 2), Fraction(19, 20)
# (label, evaluator key): the rows of the headline table, in order. A judge criterion's key is `judge.<criterion>`.
HEADLINE: tuple[tuple[str, str], ...] = (
    ("missed blockers (share of runs with one; lower is better)", MISSED_BLOCKER_KEY),
    ("hard-restriction recall", RECALL_KEY),
    ("verdict match", VERDICT_KEY),
    ("path accuracy", PATH_KEY),
    ("noise (share of IGNORE-labelled NOTAMs surfaced; lower is better)", NOISE_KEY),
    ("trajectory: required tools", REQUIRED_KEY),
    ("trajectory: forbidden tools", FORBIDDEN_KEY),
    ("trajectory: step cap", STEP_KEY),
    ("guard pass", GUARD_KEY),
    *((f"judge: {name} pass rate", feedback_key(name)) for name in CRITERIA),
)
SHORT = {key: label.split(" (")[0] for label, key in HEADLINE}
CHECKS = tuple(key for _label, key in HEADLINE if key not in (MISSED_BLOCKER_KEY, NOISE_KEY))  # a score below 1 on any of these fails the run


class ReportError(ValueError):
    """The report cannot be written as asked (a bad experiment name, a bad folder, a reader result with no usage)."""


@dataclass(frozen=True)
class ReportRun:
    """One example in one rep. `scores` maps an evaluator key to its score; a key that is absent or None did not apply (or was a judge error). An errored run has no scores."""

    example_id: str
    rep: int
    errored: bool = False
    scores: Mapping[str, float | None] = field(default_factory=dict)
    latency_s: float = 0.0
    usage: TokenUsage | None = None  # the agent's model responses summed; None: the run made no model call
    calls_without_usage: int = 0
    error: str | None = None
    run_id: UUID | None = None
    reader_failed: int = 0  # NOTAMs the reader could not read that an UNVERIFIED segment holds
    reader_usage: TokenUsage | None = None  # the NOTAM reader's replies summed (runner.ReaderMeter); None: the run made no reader call (a stored reading makes none)


@dataclass(frozen=True)
class ReportInput:
    experiment: str
    split: str
    agent: str  # "v1" or "v0"
    reps: int
    planned_runs: int
    runs: Sequence[ReportRun]
    examples: Mapping[str, Mapping[str, Any]]  # the dataset's records by id (their `metadata` is the slices')
    models: Mapping[str, str | None] = field(default_factory=dict)  # role -> model name; None or absent: the role is not used
    wall_clock_s: float = 0.0
    price: Price | None = None  # the agent's (or v0's) model
    reader_results: Sequence[ReaderResult] = ()
    shared_case_groups: int = 0
    notes: Sequence[str] = ()
    link: Callable[[ReportRun], str | None] = lambda _run: None
    interrupted: bool = False
    reader_price: Price | None = None  # the NOTAM reader's model; needed for the cost line when the reader made calls


# --- numbers --------------------------------------------------------------------------------------------------------


def _mean(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _pct(value: float | None) -> str:
    return NOT_APPLICABLE if value is None else f"{value * 100:.1f}%"


def rep_means(runs: Sequence[ReportRun], key: str, reps: int) -> list[float]:
    """The mean of the key's scores in each rep that has any, in rep order. Errored runs and runs the key did not apply to count for nothing."""
    means = []
    for rep in range(1, reps + 1):
        found = [score for run in runs if run.rep == rep and not run.errored and (score := run.scores.get(key)) is not None]
        if found:
            means.append(sum(found) / len(found))
    return means


def headline_value(runs: Sequence[ReportRun], key: str, reps: int) -> str:
    """`83.3% (80.0%–86.7%)`: the mean of the per-rep means and their min–max; `n/a` when no rep has a score."""
    means = rep_means(runs, key, reps)
    if not means:
        return NOT_APPLICABLE
    return f"{_pct(sum(means) / len(means))} ({_pct(min(means))}–{_pct(max(means))})"


def _scored(runs: Sequence[ReportRun]) -> list[ReportRun]:
    return [run for run in runs if not run.errored]


def _failed_keys(run: ReportRun) -> list[str]:
    failed = [key for key in CHECKS if (score := run.scores.get(key)) is not None and score < 1]
    if (blocker := run.scores.get(MISSED_BLOCKER_KEY)) is not None and blocker > 0:
        failed.insert(0, MISSED_BLOCKER_KEY)
    return failed


# --- the sections ---------------------------------------------------------------------------------------------------


def _table(header: Sequence[str], rows: Sequence[Sequence[str]]) -> list[str]:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join(" --- " for _ in header) + "|"]
    return lines + ["| " + " | ".join(_PIPE.sub(r"\\|", cell) for cell in row) + " |" for row in rows]


def headline_lines(report: ReportInput) -> list[str]:
    rows = [(label, headline_value(report.runs, key, report.reps)) for label, key in HEADLINE]
    return ["Mean over the per-rep means, with the min–max of those means beside it. `n/a`: the metric does not apply to any run.", "", *_table(("metric", "mean (min–max over reps)"), rows)]


def _dimension_value(record: Mapping[str, Any], dimension: str) -> str:
    value = (record.get("metadata") or {}).get(dimension)
    if dimension == "compound":
        return "true" if value else "false"  # a missing compound is false
    return "(none)" if value in (None, "") else str(value)


def slice_lines(report: ReportInput) -> list[str]:
    ids_with_runs = {run.example_id for run in report.runs}
    lines: list[str] = []
    for dimension in SLICE_DIMENSIONS:
        groups: dict[str, list[str]] = {}
        for example_id, record in report.examples.items():
            if example_id in ids_with_runs:
                groups.setdefault(_dimension_value(record, dimension), []).append(example_id)
        rows = []
        for value in sorted(groups):
            ids = set(groups[value])
            if not ids:
                continue  # n = 0: no row, never a 0%
            runs = _scored([run for run in report.runs if run.example_id in ids])
            metrics = [f"{SHORT[key]} {_pct(mean)}" for _label, key in HEADLINE if (mean := _mean([s for run in runs if (s := run.scores.get(key)) is not None])) is not None]
            rows.append((value, str(len(ids)), "; ".join(metrics) or NOT_APPLICABLE))
        if rows:
            lines += [f"### by {dimension}", "", *_table((dimension, "n", "metrics that apply (mean over the slice's scored runs)"), rows), ""]
    return lines[:-1] if lines else ["No example has a run."]


def _run_link(report: ReportInput, run: ReportRun) -> str:
    url = report.link(run)
    if url:
        return f"[rep {run.rep}]({url})"
    return f"rep {run.rep} (run {run.run_id})" if run.run_id else f"rep {run.rep}"


def failing_lines(report: ReportInput) -> list[str]:
    failing: dict[str, list[tuple[ReportRun, list[str]]]] = {}
    for run in _scored(list(report.runs)):
        if failed := _failed_keys(run):
            failing.setdefault(run.example_id, []).append((run, failed))
    if not failing:
        return ["No scored run failed a check." if _scored(list(report.runs)) else "No run was scored."]
    lines = ["An example with a scored run that failed a check (a missed blocker, or a recall, verdict, path, trajectory, guard or judge score below full):", ""]
    for example_id in sorted(failing):
        found = failing[example_id]
        checks = ", ".join(dict.fromkeys(key for _run, keys in found for key in keys))
        lines.append(f"- {example_id}: {checks}. " + "; ".join(_run_link(report, run) for run, _keys in found))
    return lines


def errored_runs(report: ReportInput) -> list[ReportRun]:
    return [run for run in report.runs if run.errored]


def errored_lines(report: ReportInput) -> list[str]:
    errored = errored_runs(report)
    share = f"{len(errored)} of {report.planned_runs} planned runs errored"
    lines = [f"{share}. incomplete: {str(is_incomplete(report)).lower()} (more than 10% of the planned runs errored makes a run incomplete; exactly 10% does not).", ""]
    if not errored:
        return [*lines, "No run errored."]
    by_id: dict[str, list[ReportRun]] = {}
    for run in errored:
        by_id.setdefault(run.example_id, []).append(run)
    for example_id in sorted(by_id):
        reps = ", ".join(str(run.rep) for run in by_id[example_id])
        lines.append(f"- {example_id}: rep {reps}: {by_id[example_id][-1].error or 'no error text'}")
    return lines


def is_incomplete(report: ReportInput) -> bool:
    return runner.is_incomplete(len(errored_runs(report)), report.planned_runs)


# --- Ops --------------------------------------------------------------------------------------------------------------


def reader_tokens(results: Sequence[ReaderResult]) -> list[TokenUsage]:
    """The usage of the reader's model calls. A STORED reading (the cache's, tokens=None) made no call and is skipped; so is an UNAVAILABLE result with no usage (no answer came back).

    A READ, REPAIRED or FAILED result without usage is a call whose usage is missing: that is E4, an error, never a 0.
    """
    found: list[TokenUsage] = []
    for result in results:
        if result.status is ReaderStatus.STORED:
            continue  # no call was made
        if result.tokens is None:
            if result.status is ReaderStatus.UNAVAILABLE:
                continue  # no answer came back at all, so there was no usage to report
            raise ReportError(f"{runner.STRICT_USAGE_FIRST} (reader result for {result.notam_id})")
        found.append(result.tokens)
    return found


def sum_tokens(parts: Sequence[TokenUsage]) -> TokenUsage | None:
    """The tokens added up; None for no parts (an experiment with no usage shows n/a, never zeros)."""
    if not parts:
        return None
    return TokenUsage(input_tokens=sum(p.input_tokens for p in parts), output_tokens=sum(p.output_tokens for p in parts), total_tokens=sum(p.total_tokens for p in parts))


@dataclass(frozen=True)
class Ops:
    latency_p50: float | None
    latency_p95: float | None
    latency_runs: int
    agent_tokens: TokenUsage | None
    reader_tokens: TokenUsage | None
    tokens: TokenUsage | None  # the agent's and the reader's together
    calls_without_usage: int
    reader_failed: int
    wall_clock_s: float


def ops_of(report: ReportInput) -> Ops:
    latencies = [run.latency_s for run in _scored(list(report.runs))]
    agent = sum_tokens([run.usage for run in report.runs if run.usage is not None])  # an errored run's calls were paid for too
    reader = sum_tokens([*(run.reader_usage for run in report.runs if run.reader_usage is not None), *reader_tokens(report.reader_results)])  # a stored reading is in neither
    both = sum_tokens([part for part in (agent, reader) if part is not None])
    return Ops(
        nearest_rank(latencies, P50),
        nearest_rank(latencies, P95),
        len(latencies),
        agent,
        reader,
        both,
        sum(run.calls_without_usage for run in report.runs),
        sum(run.reader_failed for run in _scored(list(report.runs))),
        report.wall_clock_s,
    )


def _seconds(value: float | None) -> str:
    return NOT_APPLICABLE if value is None else f"{value:.2f} s"


def _count(value: int | None) -> str:
    return NOT_APPLICABLE if value is None else f"{value:,}"


def _duration(seconds: float) -> str:
    whole = round(seconds)
    hours, rest = divmod(whole, 3600)
    minutes, secs = divmod(rest, 60)
    return f"{hours} h {minutes} min {secs} s" if hours else f"{minutes} min {secs} s"


def _cost_line(report: ReportInput, ops: Ops) -> str:
    """The est. cost line: the agent's tokens at the agent's price and the reader's at the reader's. Never a 0 or a blank: with tokens but no price it says n/a and why."""
    if ops.tokens is None:
        return f"est. cost: {NOT_APPLICABLE} (no usage)"
    if ops.reader_tokens is None:
        if report.price is None:
            return f"{NOT_APPLICABLE}: no price list entry was given for the cost"
        return cost_line(report.price, ops.tokens.input_tokens, ops.tokens.output_tokens)
    parts: list[tuple[str, Price, int, int]] = []
    if ops.agent_tokens is not None:
        if report.price is None:
            return f"{NOT_APPLICABLE}: no price list entry was given for the cost"
        parts.append(("agent", report.price, ops.agent_tokens.input_tokens, ops.agent_tokens.output_tokens))
    if report.reader_price is None:
        return f"{NOT_APPLICABLE}: no price list entry was given for the reader model's cost"
    parts.append(("reader", report.reader_price, ops.reader_tokens.input_tokens, ops.reader_tokens.output_tokens))
    return cost_line_by_role(parts)


def ops_lines(report: ReportInput) -> list[str]:
    ops = ops_of(report)
    lines = [
        f"- latency per example run (p50 / p95, nearest rank, {ops.latency_runs} scored runs): {_seconds(ops.latency_p50)} / {_seconds(ops.latency_p95)}",
        f"- tokens from response usage: input {_count(ops.tokens.input_tokens if ops.tokens else None)}, output {_count(ops.tokens.output_tokens if ops.tokens else None)}",
    ]
    if report.agent == runner.AGENT_LABEL and ops.tokens is not None:
        agent_part = f"agent model: input {_count(ops.agent_tokens.input_tokens if ops.agent_tokens else None)}, output {_count(ops.agent_tokens.output_tokens if ops.agent_tokens else None)}"
        reader_part = (
            f"reader model: input {_count(ops.reader_tokens.input_tokens)}, output {_count(ops.reader_tokens.output_tokens)}"
            if ops.reader_tokens is not None
            else "reader model: no call came back (readings came from the stored readings, or no NOTAM needed one)"
        )
        lines.append(f"  - {agent_part}; {reader_part}; the judge's calls are not in these totals")
    if ops.tokens is None:
        lines.append("  - no model response reported usage, so no total is shown (never a 0)")
    if ops.calls_without_usage:
        lines.append(f"  - {ops.calls_without_usage} model call(s) reported no usage and are not in the totals")
    lines.append("- " + _cost_line(report, ops))
    if report.agent == runner.AGENT_LABEL:
        lines.append(f"- reader calls that failed and became UNVERIFIED: {ops.reader_failed}")
    lines.append(f"- wall-clock time: {_duration(ops.wall_clock_s)}")
    return lines


def model_lines(report: ReportInput) -> list[str]:
    lines = [f"- {role}: {report.models.get(role) or 'not used in this experiment'}" for role in ROLES]
    if report.shared_case_groups and report.agent == runner.AGENT_LABEL:
        lines += ["", SHARED_CASE_NOTE]
    for note in report.notes:
        lines += ["", note]
    return lines


# --- the file ---------------------------------------------------------------------------------------------------------


def render(report: ReportInput) -> str:
    """The whole Markdown file."""
    errored = len(errored_runs(report))
    status = "incomplete" if is_incomplete(report) else "complete"
    if report.interrupted:
        status += " (interrupted)"
    head = [
        f"# {report.experiment}",
        "",
        f"{report.agent} over the {report.split} split, {report.reps} reps: {len(report.runs)} of {report.planned_runs} planned runs, {errored} errored. Status: {status}.",
        "",
    ]
    bodies = (headline_lines(report), slice_lines(report), failing_lines(report), errored_lines(report), ops_lines(report), model_lines(report))
    lines = head
    for title, body in zip(SECTIONS, bodies, strict=True):
        lines += [f"## {title}", "", *body, ""]
    return "\n".join(lines).rstrip("\n") + "\n"


def write_report(text: str, experiment: str, results_dir: Path = RESULTS_DIR) -> Path:
    """Write `<results_dir>/<experiment>.md` whole in one go: to a temp file in the same folder, then renamed over the name. Nothing outside `results_dir` is touched."""
    if not runner.EXPERIMENT_NAME.fullmatch(experiment) or re.search(r"\.\.", experiment):
        raise ReportError(f"experiment name {experiment!r} must be letters, digits, '_', '.' or '-' (it names the results file)")
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    target = results_dir / f"{experiment}.md"
    handle, temp = tempfile.mkstemp(dir=results_dir, prefix=f".{experiment}.", suffix=".tmp")
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            stream.write(text)
        os.replace(temp, target)
    except BaseException:
        Path(temp).unlink(missing_ok=True)
        raise
    return target


# --- from a finished experiment ---------------------------------------------------------------------------------------


def _usage_of(usage: Mapping[str, int] | None) -> TokenUsage | None:
    if not usage:
        return None
    return TokenUsage(input_tokens=usage["input_tokens"], output_tokens=usage["output_tokens"], total_tokens=usage["total_tokens"])


def from_experiment(
    result: runner.ExperimentResult,
    evaluation: Evaluation,
    records: Sequence[Mapping[str, Any]],
    *,
    models: Mapping[str, str | None],
    wall_clock_s: float,
    price: Price | None,
    link: Callable[[ReportRun], str | None] = lambda _run: None,
    notes: Sequence[str] = (),
    reader_price: Price | None = None,
) -> ReportInput:
    """The runner's result and the evaluators' scores as the report's input. The i-th scored run of `result` has the i-th entry of `evaluation.runs`."""
    scores_by_run = {(s.example_id, s.rep): {key: (item["score"] if item is not None else None) for key, item in s.feedback.items()} for s in evaluation.runs}
    runs = []
    for record in result.records:
        usage = record.usage or {}
        runs.append(
            ReportRun(
                record.example_id,
                record.rep,
                record.errored,
                {} if record.errored else scores_by_run.get((record.example_id, record.rep), {}),
                record.latency_s,
                _usage_of(record.usage),
                usage.get("calls_without_usage", 0),
                record.last_error if record.errored else None,
                record.langsmith_run_id,
                len(record.reader_failed_to_unverified),
                _usage_of(record.reader_usage),
            )
        )
    return ReportInput(
        result.experiment,
        result.split,
        result.agent,
        result.reps,
        result.planned_runs,
        runs,
        {record["id"]: record for record in records},
        models,
        wall_clock_s,
        price,
        (),
        len(result.shared_case_groups),
        notes,
        link,
        result.interrupted,
        reader_price,
    )


def langsmith_link(client: Any, project: str) -> Callable[[ReportRun], str | None]:
    """A run's LangSmith page: the project's own url with the run's id. The project is asked once; a project or a client that cannot say gives no link (the id is printed instead)."""
    known: dict[str, str | None] = {}

    def link(run: ReportRun) -> str | None:
        if run.run_id is None:
            return None
        if "url" not in known:
            try:
                known["url"] = str(client.read_project(project_name=project).url).rstrip("/")
            except Exception:  # noqa: BLE001 - a link is a convenience; the run id is still in the file
                known["url"] = None
        return f"{known['url']}/r/{run.run_id}" if known["url"] else None

    return link


def same_family(left: str | None, right: str | None) -> bool:
    """Whether two model names are from one family: the same first word (`claude-sonnet-5` and `claude-opus-5`, `gpt-5` and `gpt-4o`). A name that is missing is not."""
    if not left or not right:
        return False
    first = re.compile(r"[A-Za-z]+")
    one, other = first.search(left.rsplit("/", 1)[-1]), first.search(right.rsplit("/", 1)[-1])
    return bool(one and other and one.group().casefold() == other.group().casefold())
