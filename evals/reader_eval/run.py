"""`make eval-reader`: read every fully labelled NOTAM with the real model, score the runs, and write the report (G12, G15).

    python -m evals.reader_eval.run

    evaluate(llm, paths, now) -> the report dict (reads the model; writes nothing)
    write_report(report, runs, now) -> the new folder holding report.md and report.json
    main(llm=None, paths=DEFAULT_PATHS, now=None) -> exit code

The only target that calls a model, so it needs the four READER_* settings (flysafelz.reader.config): with any missing it
prints which and exits 2 before it reads a label or makes a folder. The model is built only through `flysafelz.reader`
(this file imports no model client) and is a parameter of `evaluate` and `main`, so the tests hand in a fake.

What it does:

- reads `evals/reader/labels.yaml` (only ever read: it is Dan's file) against the sheet `items.yaml`. Only a complete, current
  label is scored; an unlabelled, partly labelled or stale item is skipped and listed, a label for an id not on the sheet is
  listed as unknown (E13, R3.9). A file that is not usable, or a label whose value is not one the contracts allow, ends the
  run (exit 2) naming the file and the item, so nothing is scored from half a file. No labels.yaml is not an error: it is
  "0 labelled items", and the Blue Mill check still runs;
- reads each scored item `runs` times, `concurrency` at a time, with the cache bypassed (`cache=None`: nothing read from it,
  nothing written, E12), both numbers from the reader's settings.yaml by name. Each run is scored by PR-7's functions against
  the label, a run with no reading as what the pilot would be told (R3.1). The endpoint being down is not an error: every run
  is unavailable and is scored as unread;
- reads the Blue Mill case's NOTAMs the same way and compares them with the case's readings.yaml (blue_mill.py);
- writes `runs/<UTC timestamp>/report.md` and `report.json` from one dict (report.py). Two runs in the same second get
  separate folders; a folder is made, never reused, so no earlier report is overwritten.

A harmful downgrade never changes the exit code: this is a report, not a gate (Q5). Phone numbers are never put back (R3.5).
"""

import json
import logging
import sys
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime
from itertools import count
from pathlib import Path
from typing import Any

from evals.reader_eval.blue_mill import blue_mill_section
from evals.reader_eval.downgrade import score_item_run
from evals.reader_eval.fields import field_accuracy
from evals.reader_eval.items import ITEMS_PATH, LABELS_PATH, PIBS_DIR, READER_DATA, ItemError, load_items, parsed_notam
from evals.reader_eval.labels import LabelError, LabelSet, load_labels
from evals.reader_eval.report import ItemRuns, build_report, render_markdown, summary_lines
from flysafelz.clients.openaip import metres_to_feet
from flysafelz.contracts import ParsedNotam
from flysafelz.notam_action import LandingZone
from flysafelz.reader.cached_read import read_cached
from flysafelz.reader.config import ReaderConfigError, build_llm
from flysafelz.reader.contract import ReaderResult, ReaderStatus
from flysafelz.reader.prompt import PROMPT_VERSION
from flysafelz.reader.read import model_name
from flysafelz.reader.reader_set import read_for_lz
from flysafelz.reader.settings import SETTINGS as READER_SETTINGS
from flysafelz.redact import redact_phones
from flysafelz.sources import CaseSources, SourceError, load_case_sources

log = logging.getLogger(__name__)

# The chat model, built by flysafelz.reader.config.build_llm and handed in. Typed Any so that this file imports no model client (F2).
ChatModel = Any

BLUE_MILL_CASE = "blue_mill_2026-10-31"
RUNS_DIR = READER_DATA / "runs"
REPO_ROOT = READER_DATA.parents[1]
STAMP_FORMAT = "%Y%m%dT%H%M%SZ"
USAGE_ERROR = 2  # the exit code for a setting, a label file or a case that cannot be used


@dataclass(frozen=True)
class EvalPaths:
    """Where the eval reads and writes. The defaults are the repository's; the tests point every one at tmp_path."""

    items: Path = ITEMS_PATH
    labels: Path = LABELS_PATH
    pibs: Path = PIBS_DIR
    runs: Path = RUNS_DIR


DEFAULT_PATHS = EvalPaths()


def shown(path: Path) -> str:
    """The path as the report shows it: relative to the repository when it is inside it."""
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(path)


# --- the labels -------------------------------------------------------------------------------------------------------------


def reject_malformed(labels: LabelSet, path: Path) -> None:
    """Raise LabelError naming the file and the item for a current label with a value the contracts do not allow.

    The label loader treats such a value as unfilled, which would skip the item quietly. A typo in a category is not a
    label Dan has left for later, so the run stops (E13 is for what is blank, not for what is wrong).
    """
    for label in labels.partial:
        if label.problems:
            raise LabelError(f"{path}: {label.notam_id}: {'; '.join(label.problems)}")


# --- reading ----------------------------------------------------------------------------------------------------------------


def _unavailable(parsed: ParsedNotam, llm: ChatModel) -> ReaderResult:
    return ReaderResult(notam_id=parsed.id, reading=None, status=ReaderStatus.UNAVAILABLE, failed_checks=[], model=model_name(llm), prompt_version=PROMPT_VERSION, latency_s=0.0, tokens=None)


def _read_once(parsed: ParsedNotam, llm: ChatModel, now: datetime) -> ReaderResult:
    """One run of one NOTAM, with no cache. Whatever goes wrong inside it costs this run only, which comes back unavailable."""
    try:
        return read_cached(parsed, llm, None, now).result
    except Exception as error:  # noqa: BLE001 - one run's failure, whatever it is, must not stop the others
        log.warning("reader unavailable: %s: the read raised %s: %s", parsed.id, type(error).__name__, redact_phones(str(error))[0])
        return _unavailable(parsed, llm)


def read_items(parsed: list[ParsedNotam], llm: ChatModel, now: datetime) -> list[list[ReaderResult]]:
    """Every NOTAM read `runs` times, at most `concurrency` calls at once (a repair is inside its run's slot). One row per NOTAM, in order."""
    if not parsed:
        return []
    with ThreadPoolExecutor(max_workers=READER_SETTINGS.concurrency) as pool:
        futures = [[pool.submit(_read_once, notam, llm, now) for _ in range(READER_SETTINGS.runs)] for notam in parsed]
        return [[future.result() for future in row] for row in futures]


def run_blue_mill(llm: ChatModel, loaded: CaseSources, case: str = BLUE_MILL_CASE) -> dict[str, Any]:
    """Read the loaded case's reader set with the cache bypassed and compare it with the case's readings.yaml (GP7)."""
    lat, lon, window = loaded.case.lat, loaded.case.lon, (loaded.case.window.start, loaded.case.window.end)
    results = read_for_lz(loaded.pib.notams, (lat, lon), window, llm, None, now=loaded.case.now)
    elevation_m = loaded.forecast.elevation_m
    lz = LandingZone(lat=lat, lon=lon, elevation_ft=metres_to_feet(elevation_m) if elevation_m is not None else 0.0)
    return blue_mill_section(case, lz, window[0], {notam.id: notam for notam in loaded.pib.notams}, loaded.readings, results)


# --- the run ----------------------------------------------------------------------------------------------------------------


def evaluate(llm: ChatModel, paths: EvalPaths = DEFAULT_PATHS, now: datetime | None = None, case: str = BLUE_MILL_CASE) -> dict[str, Any]:
    """Read, score and build the report dict. Raises LabelError, ItemError or SourceError before any model call when an input cannot be used."""
    stamp = datetime.now(UTC) if now is None else now
    sheet = load_items(paths.items)
    labels = load_labels(paths.labels, sheet)
    reject_malformed(labels, paths.labels)
    order = {item["id"]: position for position, item in enumerate(sheet)}
    by_id = {item["id"]: item for item in sheet}
    complete = sorted(labels.complete, key=lambda label: order[label.notam_id])
    parsed = [parsed_notam(by_id[label.notam_id], paths.pibs) for label in complete]
    wanted = [label.reading(notam) for label, notam in zip(complete, parsed, strict=True)]
    loaded = load_case_sources(case)  # a case that will not load stops the run before the model is called

    rows = read_items(parsed, llm, stamp)
    scored = [
        ItemRuns(
            notam.id,
            results,
            [score_item_run(notam, label_reading, result) for result in results],
            [field_accuracy(label_reading, result.reading) for result in results],
        )
        for notam, label_reading, results in zip(parsed, wanted, rows, strict=True)
    ]
    return build_report(
        model=model_name(llm),
        prompt_version=PROMPT_VERSION,
        runs=READER_SETTINGS.runs,
        items=scored,
        labels=labels,
        labels_file=shown(paths.labels),
        labels_found=paths.labels.is_file(),
        blue_mill=run_blue_mill(llm, loaded, case),
    )


def make_run_folder(runs: Path, now: datetime) -> Path:
    """A new folder `runs/<UTC time>`. If that name is taken (a second run in the same second) the next free `<time>-2`, `-3`... Never an existing one."""
    base = now.astimezone(UTC).strftime(STAMP_FORMAT)
    attempts = count(1)
    while True:
        attempt = next(attempts)
        folder = runs / (base if attempt == 1 else f"{base}-{attempt}")
        try:
            folder.mkdir(parents=True, exist_ok=False)
        except FileExistsError:
            continue
        return folder


def write_report(report: Mapping[str, Any], runs: Path, now: datetime) -> Path:
    """Write report.md and report.json, both from the one dict, into a new folder. Returns the folder."""
    folder = make_run_folder(runs, now)
    (folder / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (folder / "report.md").write_text(render_markdown(report), encoding="utf-8")
    return folder


def main(llm: ChatModel | None = None, paths: EvalPaths = DEFAULT_PATHS, now: datetime | None = None) -> int:
    """Run the eval and print where the report is. 0 whatever the scores; 2 for a missing setting, an unusable label file, sheet or case."""
    stamp = datetime.now(UTC) if now is None else now
    try:
        model = build_llm() if llm is None else llm
    except ReaderConfigError as error:
        print(f"flysafelz eval-reader: {error}", file=sys.stderr)
        return USAGE_ERROR
    try:
        report = evaluate(model, paths, stamp)
    except (LabelError, ItemError, SourceError) as error:
        print(f"flysafelz eval-reader: {error}", file=sys.stderr)
        return USAGE_ERROR
    folder = write_report(report, paths.runs, stamp)
    print("\n".join([*summary_lines(report), f"Report: {folder / 'report.md'}", f"        {folder / 'report.json'}"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
