"""The reader eval's report (G15): one dict, rendered as Markdown and written as JSON, so the two cannot drift.

    build_report(...)        -> the dict (JSON types only: dicts, lists, strings, numbers, booleans, None)
    render_markdown(report)  -> report.md, written from that dict alone
    summary_lines(report)    -> what the command prints on stdout, also from that dict alone
    nearest_rank(values, q)  -> a percentile by the nearest-rank method

Keys: harmful, over_caution, field_accuracy, rates, latency_ms, tokens, skipped, blue_mill, model, prompt_version, runs
(runs per item), and for context items (what was scored), segment_misses and labels_file.

Rates are fractions of item-runs, 0 to 1, shown as percentages by the Markdown. validity is the share of runs that ended
with a reading that passed the checks (read or repaired); repair is the share that needed the one re-ask; failure and
unavailable are the shares that ended unread, because the answer was wrong twice or because there was none (R3.1).
Latency is the seconds the model calls of an item-run took, over all item-runs, in ms. Tokens are summed over the runs that
returned usage; a run with a repair counts only when both of its answers did. With no usage anywhere the sums are null,
shown as "n/a".

Findings keep NOTAM ids, actions, spots and field names. The report holds no contact value and no exemption quote, and no
phone number is put back: a placeholder stays a placeholder (R3.5).
"""

import math
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from typing import Any

from evals.reader_eval.downgrade import READER_FAILED, Finding, RunScore
from evals.reader_eval.fields import NOT_SCORED_FIELDS, aggregate
from evals.reader_eval.labels import LabelSet
from flysafelz.reader.contract import ReaderResult, ReaderStatus

TARGET_HARMFUL = 0  # the aim stated in the heading: no harmful downgrade
PERCENTILES = {"p50": Fraction(1, 2), "p95": Fraction(19, 20)}
MS_PER_S = 1000  # seconds to milliseconds (a unit, not a threshold)
NA = "n/a"
READ_STATUSES = (ReaderStatus.READ, ReaderStatus.REPAIRED)
REPAIR_ASKED = (ReaderStatus.REPAIRED, ReaderStatus.FAILED)  # a read that ended in either of these asked for the one re-ask
MISSING_LABELS = "not found"


@dataclass(frozen=True)
class ItemRuns:
    """One scored item: for each run, in order, what the reader gave, how it scored and its per-field accuracy."""

    notam_id: str
    results: Sequence[ReaderResult]
    scores: Sequence[RunScore]
    accuracy: Sequence[Mapping[str, bool | None]]


# --- numbers ----------------------------------------------------------------------------------------------------------------


def nearest_rank(values: Sequence[float], q: Fraction) -> float | None:
    """The nearest-rank percentile: the value at rank ceil(q * n) of the sorted values. One value gives that value; none gives None."""
    if not values:
        return None
    ordered = sorted(values)
    rank = math.ceil(q * len(ordered))
    return ordered[max(rank, 1) - 1]


def _rate(count: int, total: int) -> float | None:
    return count / total if total else None


def asked_for_repair(result: ReaderResult) -> bool:
    """True when the first answer failed its checks and the one re-ask was made: REPAIRED, FAILED, or UNAVAILABLE with failed checks left over."""
    return result.status in REPAIR_ASKED or (result.status is ReaderStatus.UNAVAILABLE and bool(result.failed_checks))


def _rates(results: Sequence[ReaderResult]) -> dict[str, float | None]:
    total = len(results)
    return {
        "validity": _rate(sum(r.status in READ_STATUSES for r in results), total),
        "repair": _rate(sum(asked_for_repair(r) for r in results), total),
        "failure": _rate(sum(r.status is ReaderStatus.FAILED for r in results), total),
        "unavailable": _rate(sum(r.status is ReaderStatus.UNAVAILABLE for r in results), total),
    }


def _latency(results: Sequence[ReaderResult]) -> dict[str, Any]:
    milliseconds = [r.latency_s * MS_PER_S for r in results]
    return {**{name: nearest_rank(milliseconds, q) for name, q in PERCENTILES.items()}, "item_runs": len(milliseconds)}


def _tokens(results: Sequence[ReaderResult]) -> dict[str, Any]:
    with_usage = [r.tokens for r in results if r.tokens is not None]
    if not with_usage:
        return {"input": None, "output": None, "total": None, "runs_with_usage": 0, "item_runs": len(results)}
    return {
        "input": sum(t.input_tokens for t in with_usage),
        "output": sum(t.output_tokens for t in with_usage),
        "total": sum(t.total_tokens for t in with_usage),
        "runs_with_usage": len(with_usage),
        "item_runs": len(results),
    }


# --- findings ---------------------------------------------------------------------------------------------------------------


def _finding(finding: Finding, run: int) -> dict[str, Any]:
    line = finding.line(run)
    if finding.reader_failed and f"({READER_FAILED})" not in line:
        line += f" ({READER_FAILED})"
    return {"id": finding.notam_id, "run": run, "kind": finding.kind.value, "field": finding.field, "reader_failed": finding.reader_failed, "line": line}


def _findings_section(items: Sequence[ItemRuns], which: str) -> dict[str, Any]:
    """Findings of one sort ("harmful" or "over_caution"): the findings, and the items and item-runs that have any."""
    findings: list[dict[str, Any]] = []
    flagged_items: set[str] = set()
    flagged_runs = 0
    for item in items:
        for run, score in enumerate(item.scores, start=1):
            found = getattr(score, which)
            if found:
                flagged_items.add(item.notam_id)
                flagged_runs += 1
                findings += [_finding(finding, run) for finding in found]
    section: dict[str, Any] = {"items": len(flagged_items), "item_runs": flagged_runs, "findings": findings}
    if which == "harmful":
        section["target"] = TARGET_HARMFUL
    return section


def _field_accuracy(items: Sequence[ItemRuns], runs: int) -> dict[str, dict[str, float]]:
    """{field: {mean, worst_run}} over the runs. A run's number for a field is the share of its items that got it right."""
    per_run = [[item.accuracy[run] for item in items if run < len(item.accuracy)] for run in range(runs)]
    return {name: {"mean": mean, "worst_run": worst} for name, (mean, worst) in aggregate([run for run in per_run if run]).items()}


# --- the sections about the labels ------------------------------------------------------------------------------------------


def _skipped(labels: LabelSet) -> dict[str, Any]:
    return {
        "unlabelled": {"count": len(labels.unlabelled), "ids": list(labels.unlabelled)},
        "partial": {"count": len(labels.partial), "ids": [label.notam_id for label in labels.partial], "missing": {label.notam_id: list(label.missing) for label in labels.partial}},
        "stale": {"count": len(labels.stale), "ids": [label.notam_id for label in labels.stale]},
        "unknown": {"count": len(labels.unknown), "ids": [label.notam_id for label in labels.unknown]},
    }


def build_report(
    *,
    model: str,
    prompt_version: str,
    runs: int,
    items: Sequence[ItemRuns],
    labels: LabelSet,
    labels_file: str,
    labels_found: bool,
    blue_mill: Mapping[str, Any],
) -> dict[str, Any]:
    """The whole report as one dict. `runs` is how many times each item was read; `labels_file` is the path as it is shown."""
    results = [result for item in items for result in item.results]
    segment_misses = [{"id": item.notam_id, "run": run} for item in items for run, score in enumerate(item.scores, start=1) if score.segment_miss]
    return {
        "model": model,
        "prompt_version": prompt_version,
        "runs": runs,
        "items": [{"id": item.notam_id, "reads": len(item.results), "statuses": dict(Counter(r.status.value for r in item.results))} for item in items],
        "labels_file": {"path": labels_file, "found": labels_found},
        "harmful": _findings_section(items, "harmful"),
        "over_caution": _findings_section(items, "over_caution"),
        "field_accuracy": _field_accuracy(items, runs),
        "rates": _rates(results),
        "latency_ms": _latency(results),
        "tokens": _tokens(results),
        "skipped": _skipped(labels),
        "segment_misses": segment_misses,
        "blue_mill": dict(blue_mill),
    }


# --- Markdown ---------------------------------------------------------------------------------------------------------------


def _percent(value: float | None) -> str:
    return NA if value is None else f"{value:.1%}"


def _count(value: int | None) -> str:
    return NA if value is None else f"{value:,}"


def _milliseconds(value: float | None) -> str:
    return NA if value is None else f"{value:,.0f} ms"


def _plural(count: int, noun: str) -> str:
    return f"{count} {noun}" if count == 1 else f"{count} {noun}s"


def statuses_text(statuses: Mapping[str, int]) -> str:
    """For example "read ×2, repaired ×1"."""
    return ", ".join(f"{status} \u00d7{count}" for status, count in statuses.items())


def headline(report: Mapping[str, Any]) -> str:
    """The heading line: harmful downgrades first, as items and as item-runs, and the target."""
    harmful = report["harmful"]
    return f"Harmful downgrades: {_plural(harmful['items'], 'item')}, {_plural(harmful['item_runs'], 'item-run')} (target {harmful['target']})"


def scored_line(report: Mapping[str, Any]) -> str:
    """How many items were scored, or why none were."""
    items = report["items"]
    if items:
        return f"{_plural(len(items), 'labelled item')} scored, {report['runs']} runs each ({_plural(sum(i['reads'] for i in items), 'item-run')})"
    labels_file = report["labels_file"]
    why = f"{labels_file['path']} {MISSING_LABELS}" if not labels_file["found"] else "no item is fully labelled"
    return f"0 labelled items: {why}"


def _findings_md(section: Mapping[str, Any]) -> list[str]:
    counts = f"{_plural(section['items'], 'item')}, {_plural(section['item_runs'], 'item-run')}"
    if not section["findings"]:
        return [counts, "", "none"]
    return [counts, "", *(f"- {finding['line']}" for finding in section["findings"])]


def _skipped_md(skipped: Mapping[str, Any]) -> list[str]:
    def ids(entry: Mapping[str, Any]) -> str:
        return ": " + ", ".join(entry["ids"]) if entry["ids"] else ""

    partial = skipped["partial"]
    partial_ids = ", ".join(f"{notam_id} (unfilled: {', '.join(partial['missing'][notam_id])})" for notam_id in partial["ids"])
    return [
        f"- unlabelled: {skipped['unlabelled']['count']}{ids(skipped['unlabelled'])}",
        f"- partly labelled: {partial['count']}" + (f": {partial_ids}" if partial_ids else ""),
        f"- stale (the NOTAM text changed since it was labelled): {skipped['stale']['count']}{ids(skipped['stale'])}",
        f"- label ids not on the sheet (unknown): {skipped['unknown']['count']}{ids(skipped['unknown'])}",
    ]


def render_markdown(report: Mapping[str, Any]) -> str:
    """report.md, from the dict alone."""
    rates, latency, tokens = report["rates"], report["latency_ms"], report["tokens"]
    lines = [
        f"# {headline(report)}",
        "",
        f"Model {report['model']} · prompt version {report['prompt_version']} · {_plural(report['runs'], 'run')} per item",
        scored_line(report),
        "",
        "## Harmful downgrades",
        "",
        *_findings_md(report["harmful"]),
        "",
        "## Over-caution",
        "",
        *_findings_md(report["over_caution"]),
        "",
        "## Field accuracy",
        "",
        "Mean over the runs, and the worst run. " + " and ".join(NOT_SCORED_FIELDS) + " are not scored.",
        "",
    ]
    if report["field_accuracy"]:
        lines += ["| field | mean | worst run |", "| --- | --- | --- |"]
        lines += [f"| {name} | {_percent(entry['mean'])} | {_percent(entry['worst_run'])} |" for name, entry in report["field_accuracy"].items()]
    else:
        lines.append("no scored items")
    lines += [
        "",
        "## Reader health",
        "",
        f"- validity (a reading that passed the checks): {_percent(rates['validity'])}",
        f"- repair (needed the one re-ask): {_percent(rates['repair'])}",
        f"- failure (wrong twice, unread): {_percent(rates['failure'])}",
        f"- unavailable (no usable answer, unread): {_percent(rates['unavailable'])}",
        f"- latency p50 {_milliseconds(latency['p50'])}, p95 {_milliseconds(latency['p95'])} (over {_plural(latency['item_runs'], 'item-run')})",
        f"- tokens: input {_count(tokens['input'])}, output {_count(tokens['output'])}, total {_count(tokens['total'])}"
        + (f" ({tokens['runs_with_usage']} of {_plural(tokens['item_runs'], 'item-run')} returned usage)" if tokens["runs_with_usage"] != tokens["item_runs"] else ""),
        "",
        "## Skipped items",
        "",
        "Not scored, and not counted as wrong.",
        "",
        *_skipped_md(report["skipped"]),
    ]
    if report["segment_misses"]:
        lines += ["", "## Timing misses", "", "The eval hour missed a NOTAM the label says applies to paramotors; these runs are not counted as agreement.", ""]
        lines += [f"- {miss['id']} run {miss['run']}" for miss in report["segment_misses"]]
    if report["items"]:
        lines += ["", "## Items read", ""]
        lines += [f"- {item['id']}: {_plural(item['reads'], 'read')} ({statuses_text(item['statuses'])})" for item in report["items"]]
    blue_mill = report["blue_mill"]
    lines += ["", "## Blue Mill live check", "", blue_mill["heading"], "", *(f"- {line}" for line in blue_mill["lines"])]
    return "\n".join(lines) + "\n"


def summary_lines(report: Mapping[str, Any]) -> list[str]:
    """What the command prints on stdout (without the path of the report): the headline, what was read and skipped, the Blue Mill lines."""
    skipped = report["skipped"]
    lines = [headline(report), f"Model {report['model']} · prompt version {report['prompt_version']}", scored_line(report)]
    lines += [f"  {item['id']}: read {_plural(item['reads'], 'time')} ({statuses_text(item['statuses'])})" for item in report["items"]]
    lines.append(
        "Skipped: " + f"{skipped['unlabelled']['count']} unlabelled, {skipped['partial']['count']} partly labelled, {skipped['stale']['count']} stale, {skipped['unknown']['count']} unknown label ids"
    )
    for name in ("partial", "stale", "unknown"):
        if skipped[name]["ids"]:
            lines.append(f"  {name}: {', '.join(skipped[name]['ids'])}")
    lines += ["Blue Mill live check:", *(f"  {line}" for line in report["blue_mill"]["lines"])]
    return lines
