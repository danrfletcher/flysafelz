"""The online summary file (7 PR-4.F1, G11, GP4, E1, E2, E4, E5): `evals/results/online-<YYYY-MM-DD>.md`, one page for the production runs of the window. It scores nothing and defines no metric.

    summary_date(moment) -> date                    the day of `moment` in Europe/London (not UTC): 23:30 UTC on 20 Oct 2026 is 21 Oct, 23:30 UTC on 8 Nov is 8 Nov
    collect(client, result, runs) -> [ScoredRun]    each scored run with the feedback LangSmith holds for it now (`list_feedback`), so a run scored on an earlier day counts
    build_report(report_input) -> dict              the one report dict: every number the file and the stdout lines show
    render_markdown(report) -> str                  the Markdown, from the dict alone
    summary_lines(report) -> [str]                  the stdout lines, from the same dict
    write_summary(text, day, results_dir) -> Path   `online-<day>.md`, written whole in one go (5a's `write_report`), overwriting the day's file; `results_dir` is made if it is missing

The shape is 5a's experiment report (`evals.agent.report`: one dict-like input, `render`, a table per section, `n/a` for a rate with nothing under it, one atomic write) without its slices by case type and
its reps column. Only 5a's `write_report`, `RESULTS_DIR` and `NOT_APPLICABLE` are imported from it; its table helper is private, so the few lines of table formatting are copied here, and no metric is.

What the file counts. A run is *scored* when LangSmith holds at least one feedback for it at the current evaluator version. Only feedback at that version is read: a feedback from an older version of the same run is
ignored (it is kept in LangSmith and never counted here). The runs are the ones the pull let through for the window, so the file covers the window and not only what this invocation posted. A rerun that posts
nothing reads the same feedback and writes the same file: there is no time of the job's run in it, no count of what this invocation posted, and every list is ordered by the run's start time and then its id.

    Deterministic checks   per check: pass rate = passes / n, with n the runs with a score of 1 or 0. A null is left out of n and counted on its own, by reason (`no guard record`, `no assess`, `path unknown`, `not applicable`, `check error: <class>`)
    Judge                  per path and criterion (the criteria are 5a's table), then a separate `unknown` row; with the JUDGE_* names unset, those names instead of rates; and the runs marked `not_sampled`
    Failing runs           every scored run with a score of 0 on any key: run link (`client.get_run_url`), path, the launch zone's name, the failing keys. No answer, question or phone number
    Runs skipped           the pull's skips, by source (`case`, `judge`, `v0`, `missing`) and by reason (`experiment`, `source`, `errored`), the runs fully scored at this version, and the runs with an error (E5)
    Evaluator version      the 12 hex characters, SINCE, LIMIT, and a line when LIMIT was hit (G3)
    Not scored online      the fixed line: missed blocker and verdict match need labels, and window IoU, noise and NOTAM action confusion are offline-only too

A rate is `n/a` when n is 0 (E1: no runs, no division). A run whose path is `unknown` is in the `unknown` judge row and in the null counts with the reason `path unknown` (E2); it is never a failure. A cell has `|` and newlines escaped.
"""

import re
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from evals.agent.evaluators.answer import UNKNOWN_PATH
from evals.agent.judge import CRITERIA_BY_PATH, ERROR_PREFIX
from evals.agent.report import NOT_APPLICABLE, RESULTS_DIR, write_report
from evals.online import job
from evals.online.reference import NO_ASSESS
from evals.online.runs import metadata_of

TIMEZONE = ZoneInfo("Europe/London")
NAME_PREFIX = "online-"
READ_BACK = "read back"  # the stage of an error in reading a run's feedback for the file
SECTIONS = ("Deterministic checks", "Judge", "Failing runs", "Runs skipped", "Evaluator version", "Not scored online")
OFFLINE_LABELS_LINE = "Missed blocker and verdict match need labels, which production runs do not have: they are carried by the offline evals."
OFFLINE_ONLY_LINE = "Window IoU, noise and NOTAM action confusion are offline-only too."
JUDGE_ERROR = "judge error"  # 5a's `judge error: <message>` as one reason: the message is not copied into the file
OTHER_REASON = "other"
KNOWN_REASONS = (job.NO_GUARD_RECORD, NO_ASSESS, job.PATH_UNKNOWN, job.NOT_APPLICABLE, job.NO_JUDGE_RESULT)
ERROR_CLASS_LIMIT = 60
_PIPE = re.compile(r"\|")
_BREAKS = re.compile(r"[\r\n]+")
_VERSION_TAIL = " | " + job.VERSION_EXTRA + "="


@dataclass(frozen=True)
class ScoredRun:
    """One pulled run as the file reads it: its path and launch zone, the feedback LangSmith holds for it (every version; the build keeps the current one), and its error, if the job or the read-back had one."""

    run_id: str
    start: datetime | None
    path: str
    feedback: Sequence[Any] = ()
    lz_name: str | None = None
    error: job.RunError | None = None


@dataclass(frozen=True)
class ReportInput:
    """Everything the report dict is built from. `link` gives a failing run's LangSmith url (None for none); it is called for failing runs only."""

    day: date
    project: str
    version: str
    since: str
    limit: int
    limit_hit: bool = False
    read: int = 0
    runs: Sequence[ScoredRun] = ()
    skipped_by_source: Mapping[str, int] = field(default_factory=dict)
    skipped_by_reason: Mapping[str, int] = field(default_factory=dict)
    judge_missing: Sequence[str] = ()
    link: Callable[[str], str | None] = lambda _run_id: None


# --- the date (G11) -------------------------------------------------------------------------------------------------------------------------


def summary_date(moment: datetime) -> date:
    """The calendar day of `moment` in Europe/London. A time with no zone is a mistake (it would be dated by whatever the machine's clock says) and raises ValueError."""
    if moment.tzinfo is None:
        raise ValueError("the summary date needs a time with a zone")
    return moment.astimezone(TIMEZONE).date()


def summary_name(day: date) -> str:
    """`online-2026-11-08`: the file's name without `.md`."""
    return f"{NAME_PREFIX}{day.isoformat()}"


def write_summary(text: str, day: date, results_dir: Path = RESULTS_DIR) -> Path:
    """Write `<results_dir>/online-<day>.md` whole in one go, making the folder if it is missing and replacing that day's file. Nothing is added to git."""
    return write_report(text, summary_name(day), results_dir)


# --- reading the feedback ---------------------------------------------------------------------------------------------------------------


def current_feedback(feedback: Iterable[Any], version: str) -> dict[str, Any]:
    """{key: feedback} of the feedback written at `version`. Feedback of an older version, or none of this job's, is left out; a key posted twice (it never is) keeps the last."""
    return {posted.key: posted for posted in feedback if job.version_of(posted) == version}


def null_reason(comment: object) -> str:
    """The reason of a null score, from its comment: the comment less the version tail, when it is one of the known reasons; `check error: <class>` as written; 5a's judge errors as `judge error`; else `other`."""
    text = str(comment or "").rsplit(_VERSION_TAIL, 1)[0].strip()
    if text in KNOWN_REASONS:
        return text
    if text.startswith(job.CHECK_ERROR):
        return text[:ERROR_CLASS_LIMIT]
    if text.startswith(ERROR_PREFIX.strip()):
        return JUDGE_ERROR
    return OTHER_REASON


def _rate(passed: int, n: int) -> float | None:
    return passed / n if n else None


def _tally(scored: Iterable[Any]) -> dict[str, Any]:
    """n, passed, rate and the null reasons of a set of feedback of one key."""
    n = passed = 0
    nulls: Counter[str] = Counter()
    for posted in scored:
        if posted.score is None:
            nulls[null_reason(posted.comment)] += 1
        else:
            n += 1
            passed += 1 if posted.score >= 1 else 0
    return {"n": n, "passed": passed, "rate": _rate(passed, n), "nulls": dict(sorted(nulls.items())), "null_count": sum(nulls.values())}


def _is_scored(run: ScoredRun, version: str) -> bool:
    return bool(current_feedback(run.feedback, version))


def _is_complete(held: Mapping[str, Any]) -> bool:
    return job.VERSION_KEY in held and all(key in held for key in job.DETERMINISTIC_KEYS)


def _order(runs: Iterable[ScoredRun]) -> list[ScoredRun]:
    return sorted(runs, key=lambda run: (run.start or datetime.min.replace(tzinfo=UTC), run.run_id))


def _key_order(keys: Iterable[str]) -> list[str]:
    """The failing keys in the order the job posts them: the deterministic ones, then the judge's, then anything else by name."""
    known = [*job.DETERMINISTIC_KEYS, *(job.judge_key(name) for names in CRITERIA_BY_PATH.values() for name in names)]
    wanted = set(keys)
    first = list(dict.fromkeys(key for key in known if key in wanted))
    return first + sorted(wanted - set(first))


# --- the report dict --------------------------------------------------------------------------------------------------------------------


def _checks(held: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for key in job.DETERMINISTIC_KEYS:
        rows.append({"key": key, "name": key.removeprefix(job.PREFIX), **_tally(found[key] for found in held if key in found)})
    return rows


def _judge_rows(runs: Sequence[ScoredRun], held: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """One row per criterion and path of 5a's table, then a row for each other path the runs have, `unknown` last and always there. A path with no criteria has one row with no criterion."""
    paths = [*CRITERIA_BY_PATH, *sorted({run.path for run in runs} - set(CRITERIA_BY_PATH) - {UNKNOWN_PATH}), UNKNOWN_PATH]
    rows: list[dict[str, Any]] = []
    for path in paths:
        on_path = [found for run, found in zip(runs, held, strict=True) if run.path == path]
        criteria = CRITERIA_BY_PATH.get(path, ())
        if not criteria:
            rows.append({"path": path, "criterion": None, "key": None, "runs": len(on_path), **_tally(())})
            continue
        for criterion in criteria:
            key = job.judge_key(criterion)
            rows.append({"path": path, "criterion": criterion, "key": key, "runs": len(on_path), **_tally(found[key] for found in on_path if key in found)})
    return rows


def _not_sampled(held: Sequence[Mapping[str, Any]]) -> int:
    """The runs marked `not_sampled` at this version that no judge key has filled in since."""
    judge_keys = {job.judge_key(name) for names in CRITERIA_BY_PATH.values() for name in names}
    return sum(1 for found in held if job.SAMPLED_KEY in found and not judge_keys & found.keys())


def _failing(runs: Sequence[ScoredRun], held: Sequence[Mapping[str, Any]], link: Callable[[str], str | None]) -> list[dict[str, Any]]:
    failing = []
    for run, found in zip(runs, held, strict=True):
        keys = _key_order(key for key, posted in found.items() if posted.score is not None and posted.score < 1)
        if keys:
            failing.append({"run_id": run.run_id, "url": link(run.run_id), "path": run.path, "lz_name": run.lz_name, "keys": keys})
    return failing


def _errors(runs: Sequence[ScoredRun]) -> list[dict[str, Any]]:
    return [
        {"run_id": run.run_id, "stage": run.error.stage, "key": run.error.key, "error_class": run.error.error_class} for run in runs if run.error is not None
    ]


def build_report(data: ReportInput) -> dict[str, Any]:
    """The one report dict. `runs` is every pulled run that was not skipped; the scored ones (some feedback at this version) are in the tables, the others are only in `skipped.errors` if they have an error."""
    ordered = _order(data.runs)
    scored = [run for run in ordered if _is_scored(run, data.version)]
    held = [current_feedback(run.feedback, data.version) for run in scored]
    return {
        "date": data.day.isoformat(),
        "project": data.project,
        "version": data.version,
        "since": data.since,
        "limit": data.limit,
        "limit_hit": data.limit_hit,
        "read": data.read,
        "runs": len(scored),
        "checks": _checks(held),
        "judge": {
            "missing": list(data.judge_missing),
            "rows": _judge_rows(scored, held),
            "not_sampled": _not_sampled(held),
        },
        "failing": _failing(scored, held, data.link),
        "skipped": {
            "total": sum(data.skipped_by_reason.values()),
            "by_source": dict(sorted(data.skipped_by_source.items())),
            "by_reason": dict(sorted(data.skipped_by_reason.items())),
            "fully_scored": sum(1 for found in held if _is_complete(found)),
            "errors": _errors(ordered),
        },
        "offline_only": [OFFLINE_LABELS_LINE, OFFLINE_ONLY_LINE],
    }


# --- the Markdown -----------------------------------------------------------------------------------------------------------------------


def _cell(value: object) -> str:
    """One table cell: a newline becomes a space and `|` is escaped, so a launch zone named `A | B` or with a line break cannot split a row."""
    return _PIPE.sub(r"\\|", _BREAKS.sub(" ", str(value)))


def _table(header: Sequence[str], rows: Sequence[Sequence[object]]) -> list[str]:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join(" --- " for _ in header) + "|"]
    return lines + ["| " + " | ".join(_cell(cell) for cell in row) + " |" for row in rows]


def _pct(rate: float | None) -> str:
    return NOT_APPLICABLE if rate is None else f"{rate * 100:.1f}%"


def _counts(counts: Mapping[str, int]) -> str:
    return ", ".join(f"{name}: {count}" for name, count in counts.items())


def _plural(count: int, noun: str) -> str:
    return f"{count} {noun}" if count == 1 else f"{count} {noun}s"


def _nulls(row: Mapping[str, Any]) -> str:
    return f"{row['null_count']} ({_counts(row['nulls'])})" if row["null_count"] else "0"


def check_lines(report: Mapping[str, Any]) -> list[str]:
    rows = [(row["name"], _pct(row["rate"]), f"{row['passed']}/{row['n']}", row["n"], _nulls(row)) for row in report["checks"]]
    return [
        "Pass rate = passes / n, with n the runs the check scored 1 or 0. A null is left out of n and counted on its own, with its reason. `n/a`: nothing was scored.",
        "",
        *_table(("check", "pass rate", "passes/n", "n", "nulls (reason)"), rows),
    ]


def judge_lines(report: Mapping[str, Any]) -> list[str]:
    judge = report["judge"]
    lines = [f"Runs marked not_sampled (left out of the judge's sample): {judge['not_sampled']}", ""]
    if judge["missing"]:
        return [f"The judge was not run: {', '.join(judge['missing'])} not set. Set them and run again; the judge keys are filled in on a later run.", "", *lines[:1]]
    rows = []
    for row in judge["rows"]:
        if row["criterion"] is None:
            rows.append((row["path"], f"no judge criteria for this path ({_plural(row['runs'], 'run')})", NOT_APPLICABLE, 0, "0"))
        else:
            rows.append((row["path"], row["criterion"], _pct(row["rate"]), row["n"], _nulls(row)))
    return [*lines, *_table(("path", "criterion", "pass rate", "n", "nulls (reason)"), rows)]


def failing_lines(report: Mapping[str, Any]) -> list[str]:
    if not report["failing"]:
        return ["No scored run failed a check." if report["runs"] else "No run was scored."]
    rows = [(item["url"] or f"run {item['run_id']}", item["path"], item["lz_name"] or "-", ", ".join(item["keys"])) for item in report["failing"]]
    return ["A scored run with a score of 0 on any check or judge criterion. A null is never a failure.", "", *_table(("run", "path", "launch zone", "failing keys"), rows)]


def skipped_lines(report: Mapping[str, Any]) -> list[str]:
    skipped = report["skipped"]
    lines = [
        f"- skipped by the pull: {skipped['total']}",
        f"  - by source (runs whose source is not `pilot` or `online-seed`): {_counts(skipped['by_source']) or 'none'}",
        f"  - by reason: {_counts(skipped['by_reason']) or 'none'}",
        f"- fully scored at this evaluator version: {skipped['fully_scored']}",
        f"- runs with an error: {len(skipped['errors'])}",
    ]
    for item in skipped["errors"]:
        where = f"{item['stage']} {item['key']}" if item["key"] else item["stage"]
        lines.append(f"  - {item['run_id']}: {where}: {item['error_class']}. A rerun posts only what is missing.")
    return lines


def version_lines(report: Mapping[str, Any]) -> list[str]:
    hit = f"LIMIT was hit: {report['limit']} runs were read and more were in the window, so the oldest were not read. Raise LIMIT to read them." if report["limit_hit"] else "LIMIT was not hit."
    return [f"- evaluator version: {report['version']}", f"- project: {report['project']}", f"- SINCE: {report['since']}", f"- LIMIT: {report['limit']}", f"- {hit}"]


def render_markdown(report: Mapping[str, Any]) -> str:
    """The whole file, from the report dict alone."""
    head = [f"# Online eval {report['date']}", "", f"{_plural(report['runs'], 'run')} scored at evaluator version {report['version']}, from {_plural(report['read'], 'run')} read in the window.", ""]
    bodies = (check_lines(report), judge_lines(report), failing_lines(report), skipped_lines(report), version_lines(report), list(report["offline_only"]))
    lines = head
    for title, body in zip(SECTIONS, bodies, strict=True):
        lines += [f"## {title}", "", *body, ""]
    return "\n".join(lines).rstrip("\n") + "\n"


def summary_lines(report: Mapping[str, Any]) -> list[str]:
    """The lines printed to stdout, from the same dict."""
    skipped = report["skipped"]
    lines = [f"online eval {report['date']}: {_plural(report['runs'], 'run')} scored at evaluator version {report['version']} (project {report['project']}, SINCE={report['since']}, LIMIT={report['limit']})"]
    for row in report["checks"]:
        lines.append(f"  {row['name']}: {_pct(row['rate'])} ({row['passed']}/{row['n']}), {row['null_count']} null")
    if report["judge"]["missing"]:
        lines.append(f"  judge: not run, {', '.join(report['judge']['missing'])} not set")
    else:
        graded = sum(row["n"] for row in report["judge"]["rows"])
        lines.append(f"  judge: {graded} graded, {report['judge']['not_sampled']} not_sampled")
    lines.append(f"  failing runs: {len(report['failing'])}")
    lines.append(f"  skipped: {skipped['total']}; fully scored: {skipped['fully_scored']}; errors: {len(skipped['errors'])}")
    if report["limit_hit"]:
        lines.append(f"  LIMIT={report['limit']} was hit: more runs were in the window")
    return lines


# --- from LangSmith ---------------------------------------------------------------------------------------------------------------------


def collect(client: Any, result: job.OnlineResult, runs: Iterable[Any]) -> list[ScoredRun]:
    """Every run of the pass with the feedback LangSmith holds for it now: read after the posts, so it is the state a rerun would find, whoever wrote it and whenever.

    `runs` are the pulled runs (for the start time and the launch zone). A run whose feedback cannot be read keeps the error it had, else gets a `read back` error: the file is still written.
    """
    by_id = {run.id: run for run in runs}
    collected = []
    for item in result.runs:
        run = by_id[item.run_id]
        feedback: list[Any] = []
        error = item.error
        try:
            feedback = list(client.list_feedback(run_ids=[item.run_id]))
        except Exception as problem:  # noqa: BLE001 - one run's feedback that cannot be read is listed against it; the other runs and the file still go ahead
            error = error or job.RunError(READ_BACK, type(problem).__name__, "")
        if item.judge_error is not None and error is None:
            error = item.judge_error
        name = metadata_of(run).get("lz_name")
        collected.append(ScoredRun(str(item.run_id), getattr(run, "start_time", None), item.path, feedback, name if isinstance(name, str) and name.strip() else None, error))
    return collected


def run_link(client: Any, project: str, runs: Iterable[Any]) -> Callable[[str], str | None]:
    """A failing run's LangSmith url, from `client.get_run_url`. A client that cannot say gives None, and the file shows the run id."""
    by_id = {str(run.id): run for run in runs}

    def link(run_id: str) -> str | None:
        try:
            return str(client.get_run_url(run=by_id[run_id], project_name=project))
        except Exception:  # noqa: BLE001 - a link is a convenience; the run id is in the file
            return None

    return link
