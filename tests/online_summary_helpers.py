"""Hand-built inputs for the online summary's tests (7 PR-4.F1). No client, no socket, no key.

    posted(key, score, reason, version=)    one feedback as LangSmith returns it: the key, a score of 1, 0 or None, and a comment that ends with the evaluator version
    scored_run(run_id, path=, checks=, judge=, ...)   a ScoredRun whose feedback holds every deterministic key and the version marker (a complete run), with the scores given
    report_for(runs, **kw)                  the report dict of `build_report` over those runs
    check_row(report, name) / judge_row(report, path, criterion)   one row of the report dict

`checks` and `judge` map a name to 1, 0, None (a null, `not applicable`) or `(None, reason)`. A check not named scores 1.
"""

from collections.abc import Mapping, Sequence
from datetime import UTC, date, datetime, timedelta
from types import SimpleNamespace
from typing import Any

from evals.agent.judge import CRITERIA_BY_PATH
from evals.online import job, summary

VERSION, OTHER_VERSION = "0123456789ab", "ba9876543210"
DAY = date(2026, 11, 8)
START = datetime(2026, 11, 2, 9, 0, tzinfo=UTC)
CHECKS = tuple(key.removeprefix(job.PREFIX) for key in job.DETERMINISTIC_KEYS)


def posted(key: str, score: int | None, reason: str = "ok", version: str = VERSION) -> Any:
    return SimpleNamespace(key=key, score=score, comment=f"{reason} | evaluator_version={version}", extra={"evaluator_version": version}, value=None)


def _score(value: Any) -> tuple[int | None, str]:
    if isinstance(value, tuple):
        return value
    return value, job.NOT_APPLICABLE if value is None else "ok"


def scored_run(
    run_id: str,
    *,
    path: str = "assess",
    checks: Mapping[str, Any] | None = None,
    judge: Mapping[str, Any] | None = None,
    version: str = VERSION,
    minutes: int = 0,
    lz_name: str | None = None,
    marker: bool = True,
    only: Sequence[str] | None = None,
    extra: Sequence[Any] = (),
    error: job.RunError | None = None,
) -> summary.ScoredRun:
    """A run with every deterministic key (1 unless `checks` says otherwise), the judge keys of `judge`, the marker, and `extra` feedback. `only` keeps just those deterministic names."""
    feedback = []
    for name in CHECKS if only is None else only:
        score, reason = _score((checks or {}).get(name, 1))
        feedback.append(posted(job.feedback_key(name), score, reason, version))
    for criterion, value in (judge or {}).items():
        score, reason = _score(value)
        feedback.append(posted(job.judge_key(criterion), score, reason, version))
    if marker:
        feedback.append(posted(job.VERSION_KEY, 1, "all keys posted", version))
    return summary.ScoredRun(run_id, START + timedelta(minutes=minutes), path, [*feedback, *extra], lz_name, error)


def report_for(runs: Sequence[summary.ScoredRun], **kw: Any) -> dict[str, Any]:
    fields: dict[str, Any] = {"day": DAY, "project": "flysafelz", "version": VERSION, "since": "7d", "limit": 500, "read": len(runs), "runs": runs, **kw}
    return summary.build_report(summary.ReportInput(**fields))


def check_row(report: Mapping[str, Any], name: str) -> dict[str, Any]:
    return next(row for row in report["checks"] if row["name"] == name)


def judge_row(report: Mapping[str, Any], path: str, criterion: str | None) -> dict[str, Any]:
    return next(row for row in report["judge"]["rows"] if row["path"] == path and row["criterion"] == criterion)


def judge_all(path: str, score: Any = 1) -> dict[str, Any]:
    """A `judge` mapping scoring every criterion of the path's table."""
    return dict.fromkeys(CRITERIA_BY_PATH.get(path, ()), score)
