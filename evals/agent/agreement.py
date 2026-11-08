"""Judge calibration, step two (5a PR-8.F1, G13, GP7): read Dan's labels back and say how far the judge agrees with him.

    python -m evals.agent.agreement             what `make agreement EXPERIMENT=<v1 dev experiment>` runs; EXPERIMENT comes from the environment

    pair_stats(pairs) -> Stats                  n, agreement and Cohen's kappa of two columns of 0/1 labels, and why they are weak (if they are)
    parse_label(value, score) -> 0 | 1          Dan's pass/fail (or 1/0); ValueError for anything else
    read_feedback(client, run_ids, copies)      the judge's logged feedback (run 1, on the experiment's runs) and Dan's labels (on the queued copies), from one set of `list_feedback` calls
    second_pass(items, judge, cache_path)       one fresh judge verdict per answer (run 2), cached on disk so a rerun does not call the judge
    render(...) -> list[str]                    the report; main() prints it and writes evals/results/<experiment>-agreement.md
    main(argv, ...) -> int                      0 reported, 2 refused or mis-called, 1 LangSmith or the judge failed

What it reads. The same 39 answers `make annotate` queued (annotate.select over the experiment's runs, so the dev guard and the errored list are the same code). For each, the
criteria that apply come from the judge's one path -> criteria map. Per criterion:

  Judge vs Dan   n, % agreement and Cohen's kappa (sklearn.metrics.cohen_kappa_score) of the judge's run-1 grade (its feedback logged on the experiment's run, `judge.<criterion>`) against
                 Dan's label (feedback under the bare criterion name, on the copy of that run in the calibration project: `make annotate` queues copies so the judge's feedback is never in
                 front of him; a copy names its source run in its `source_run_id` metadata, which is how the two are matched). Only answers where the criterion applies, Dan labelled it and the judge gave a pass or fail count in n. A
                 criterion that does not apply is null on both sides and is not in n (E5). With partial labels only labelled pairs are scored, and the answers still unlabelled
                 are counted per criterion.
  Self-agreement the same 39 answers judged twice: run 1 is the logged feedback, run 2 is one fresh pass of the judge (Judge.judge, the PR-7.F1 prompt and message), by the same
                 n, % and kappa.

Weak (GP7). A kappa is flagged `weak` when n < 20 or kappa < 0.4, so `good_decline` (n = 15 at most) is always flagged. A kappa that cannot be computed (every label in the
criterion identical on both sides: sklearn gives nan) shows `n/a (one class)` and is weak; it is never shown as `nan` and never raises. The report also says how many decline
answers and how many questions the decline numbers rest on ("15 decline answers from 5 questions").

Dan's labels. His feedback is read from LangSmith, from the copies only (feedback under a criterion's name on the experiment's own run is not his); nothing here writes a label. A value
other than pass or fail (or 1 or 0) on one of the four keys is exit 2, naming the run and the key. A feedback key on a copy that is not one of the four criteria (and is not machine
feedback: the judge's `judge.*` keys and anything a model wrote) is ignored and listed. If Dan
changed a label, the latest one counts. No labels at all (`0 of 39 answers labelled`) is exit 2.

The second judge pass (F2, F3). It needs the JUDGE_* settings (PR-7.F1's JudgeConfigError, naming each missing one), checked before the first LangSmith call when no cache exists.
Its raw verdicts are cached at data/eval-cache/<experiment>-judge2.json (git-ignored), written after each answer, so a rerun, or a run after an interruption, judges only what
the file does not hold; a complete cache needs no judge and no JUDGE_* settings. A judge error is not cached (it is retried next time) and the answer drops out of the
self-agreement. Reading the labels comes before the judge is called, so an unlabelled queue costs no model call.

Output. The report goes to stdout and to evals/results/<experiment>-agreement.md (committed). It holds no answer text. The cache and the report are the only files written.
"""

import json
import math
import os
import sys
import tempfile
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sklearn.metrics import cohen_kappa_score

from evals.agent.annotate import CalibrationError, Item, Selection, check_experiment, check_name, experiment_from, fetch_runs, find_copies, queue_name, select, usage_text
from evals.agent.evaluators.answer import ScoredAnswer
from evals.agent.import_dataset import DatasetError
from evals.agent.judge import CRITERIA, KEY_PREFIX, SELF_PREFERENCE_LINE, Judge, JudgeConfigError, build_judge, feedback_key
from evals.agent.loader import DEFAULT_SPLIT, REPO, Record, load_dataset, make_client

PROGRAM = "evals.agent.agreement"
RESULTS_DIR = REPO / "evals" / "results"
CACHE_DIR = REPO / "data" / "eval-cache"
MIN_N = 20  # a kappa over fewer answers is weak
MIN_KAPPA = 0.4  # and so is one below this
KAPPA_EPSILON = 1e-9  # a kappa of exactly 0.4 computed as 0.3999999999999999 is not below 0.4
USAGE_EXIT, FAILED_EXIT = 2, 1
BATCH = 20  # run ids per list_feedback call
PASS_WORDS, FAIL_WORDS = {"pass", "1", "1.0", "true"}, {"fail", "0", "0.0", "false"}
MODEL_SOURCE = "model"
ONE_CLASS = "n/a (one class)"
NO_PAIRS = "n/a (no pairs)"
DECLINE = "decline"

JudgeFactory = Callable[[Mapping[str, str]], Judge]
ClientFactory = Callable[[Mapping[str, str]], Any]


# --- the numbers ---------------------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Stats:
    """n pairs, how many agree, and kappa (None when it cannot be computed). `weak` holds the reasons: empty when the figure is sound."""

    n: int
    agree: int
    kappa: float | None
    weak: tuple[str, ...]

    @property
    def percent(self) -> float | None:
        return 100.0 * self.agree / self.n if self.n else None

    @property
    def kappa_text(self) -> str:
        if self.kappa is not None:
            return f"{self.kappa:.3f}"
        return ONE_CLASS if self.n else NO_PAIRS

    @property
    def percent_text(self) -> str:
        return f"{self.percent:.1f}%" if self.percent is not None else "n/a"

    @property
    def flag(self) -> str:
        return f"weak ({'; '.join(self.weak)})" if self.weak else ""


def weak_reasons(n: int, kappa: float | None) -> tuple[str, ...]:
    """Why a kappa is weak: n under MIN_N, kappa under MIN_KAPPA, or no kappa at all. Empty when none applies."""
    reasons = []
    if n < MIN_N:
        reasons.append(f"n={n} < {MIN_N}")
    if kappa is None:
        reasons.append("kappa undefined")
    elif kappa < MIN_KAPPA - KAPPA_EPSILON:
        reasons.append(f"kappa < {MIN_KAPPA}")
    return tuple(reasons)


def pair_stats(pairs: Sequence[tuple[int, int]]) -> Stats:
    """n, agreement and Cohen's kappa of (first rater, second rater) labels, each 0 or 1.

    One class only (every label in both columns identical, so sklearn would divide zero by zero and return nan) gives kappa None, never nan and never an error.
    """
    first, second = [a for a, _ in pairs], [b for _, b in pairs]
    agree = sum(1 for a, b in pairs if a == b)
    kappa: float | None = None
    if len(set(first) | set(second)) > 1:
        value = float(cohen_kappa_score(first, second))
        kappa = None if math.isnan(value) else value
    return Stats(len(pairs), agree, kappa, weak_reasons(len(pairs), kappa))


def parse_label(value: Any, score: Any = None) -> int:
    """Dan's label as 1 (pass) or 0 (fail): `pass`/`fail` in any case, or 1/0. The value is read first, the score when there is no value. ValueError for anything else."""
    raw = value if value is not None else score
    text = str(raw).strip().lower() if raw is not None and not isinstance(raw, (dict, list)) else ""
    if text in PASS_WORDS:
        return 1
    if text in FAIL_WORDS:
        return 0
    raise ValueError(f"{raw!r} is not pass or fail (or 1 or 0)")


# --- the feedback ----------------------------------------------------------------------------------------------------------------------


@dataclass
class Feedback:
    """What LangSmith holds for the answers: the judge's run-1 grade and Dan's label per (run id, criterion), and what was ignored."""

    judge: dict[tuple[str, str], int | None] = field(default_factory=dict)  # None: the judge's feedback is a judge error (no score)
    dan: dict[tuple[str, str], int] = field(default_factory=dict)
    ignored_keys: Counter[str] = field(default_factory=Counter)  # human feedback under a key that is not one of the four criteria


def _stamp(item: Any) -> datetime:
    moment = getattr(item, "modified_at", None) or getattr(item, "created_at", None) or datetime.min
    return moment if moment.tzinfo else datetime.combine(moment.date(), moment.time(), tzinfo=UTC)  # LangSmith's naive times are UTC


def _source_type(item: Any) -> str:
    source = getattr(item, "feedback_source", None)
    kind = getattr(source, "type", None)
    return str(getattr(kind, "value", kind) or "")


def read_feedback(client: Any, run_ids: Sequence[str], copies: Mapping[str, str]) -> Feedback:
    """The feedback on the answers, split three ways (see the module doc). CalibrationError for a label of Dan's that is not pass or fail, naming the run and the key.

    `run_ids` are the experiment's runs, where the judge's grades are; `copies` maps each to its copy in the calibration project, where Dan's labels are. Both are keyed by the
    experiment's run id in the result.
    """
    originals = set(run_ids)
    original_of = {copy: run for run, copy in copies.items() if run in originals}
    ids = [*run_ids, *original_of]
    found: list[Any] = []
    for start in range(0, len(ids), BATCH):
        found.extend(client.list_feedback(run_ids=ids[start : start + BATCH]))
    latest: dict[tuple[str, str], Any] = {}
    for item in sorted(found, key=_stamp):  # the latest of a run's feedback under one key wins
        latest[str(item.run_id), item.key] = item
    result = Feedback()
    for (run_id, key), item in latest.items():
        if run_id in originals:  # the experiment's own run: only the judge's grades are read from it
            if key.startswith(KEY_PREFIX) and key[len(KEY_PREFIX) :] in CRITERIA:
                score = item.score
                result.judge[run_id, key[len(KEY_PREFIX) :]] = None if score is None else round(float(score))
        elif run_id in original_of:  # the copy: only Dan's labels are read from it
            source = original_of[run_id]
            if key.startswith(KEY_PREFIX) or _source_type(item) == MODEL_SOURCE:
                continue  # machine feedback (the judge's keys, the code evaluators, anything a model wrote): never a label of Dan's
            if key in CRITERIA:
                try:
                    result.dan[source, key] = parse_label(item.value, item.score)
                except ValueError as error:
                    raise CalibrationError(f"run {source} (its queued copy {run_id}): the label under {key!r} is {error}") from None
            else:
                result.ignored_keys[key] += 1
    return result


# --- the second judge pass --------------------------------------------------------------------------------------------------------------


def _verdicts_of(grades: Mapping[str, Any]) -> dict[str, dict[str, Any] | None]:
    return {key: None if grade is None else {"score": grade["score"], "comment": grade["comment"]} for key, grade in grades.items()}


def load_cache(path: Path) -> dict[str, dict[str, Any]]:
    """The cached second-pass verdicts, {run id: {judge.<criterion>: {score, comment} | None}}; {} when there is no file. CalibrationError for a file that is not that."""
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        verdicts = data["verdicts"]
        if not isinstance(verdicts, dict):
            raise TypeError("verdicts is not a mapping")
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise CalibrationError(f"the judge cache {path} cannot be read ({type(error).__name__}); delete it to judge again") from error
    return verdicts


def write_cache(path: Path, experiment: str, verdicts: Mapping[str, Any]) -> None:
    """Write the cache whole and atomically: a stop in the middle leaves the earlier file, not half of one."""
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(dir=path.parent, prefix=path.name, suffix=".tmp")
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            json.dump({"experiment": experiment, "verdicts": verdicts}, stream, indent=1, sort_keys=True, ensure_ascii=False)
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


@dataclass
class SecondPass:
    scores: dict[tuple[str, str], int | None]  # (run id, criterion) -> 1, 0 or None (a judge error)
    judged: int  # answers the judge was called for in this run
    cached: int  # answers taken from the cache
    errors: int  # answers whose judge call failed (not cached)


def second_pass(items: Sequence[Item], get_judge: Callable[[], Judge], cache_path: Path, experiment: str) -> SecondPass:
    """One fresh judge verdict per answer, from the cache when the file holds it. `get_judge` is only called when an answer is not in the cache."""
    verdicts = load_cache(cache_path)
    scores: dict[tuple[str, str], int | None] = {}
    judged = cached = errors = 0
    judge: Judge | None = None
    for item in items:
        grades = verdicts.get(item.run_id)
        if grades is None:
            judge = judge or get_judge()
            fresh = judge.judge(item.record, ScoredAnswer(text=item.answer))
            applicable = [fresh[feedback_key(name)] for name in item.criteria]
            if any(grade is None or grade["score"] is None for grade in applicable):
                errors += 1
                grades = {key: None for key in fresh}
            else:
                grades = _verdicts_of(fresh)
                verdicts[item.run_id] = grades
                write_cache(cache_path, experiment, verdicts)
                judged += 1
        else:
            cached += 1
        for name in item.criteria:
            grade = grades.get(feedback_key(name))
            score = None if grade is None else grade["score"]
            scores[item.run_id, name] = None if score is None else round(float(score))
    return SecondPass(scores, judged, cached, errors)


# --- the report ----------------------------------------------------------------------------------------------------------------------------


@dataclass
class Row:
    criterion: str
    stats: Stats
    unlabelled: int = 0  # applicable answers Dan has not labelled on this criterion
    no_verdict: int = 0  # applicable answers with a label but without a judge grade (or, for self-agreement, without both)


def judge_vs_dan(items: Sequence[Item], feedback: Feedback) -> list[Row]:
    rows = []
    for name in CRITERIA:
        pairs, unlabelled, no_verdict = [], 0, 0
        for item in items:
            if name not in item.criteria:
                continue  # does not apply to this path: null on both sides, not in n (E5)
            dan = feedback.dan.get((item.run_id, name))
            judge = feedback.judge.get((item.run_id, name))
            if dan is None:
                unlabelled += 1
            elif judge is None:
                no_verdict += 1
            else:
                pairs.append((judge, dan))
        rows.append(Row(name, pair_stats(pairs), unlabelled, no_verdict))
    return rows


def self_agreement(items: Sequence[Item], feedback: Feedback, second: SecondPass) -> list[Row]:
    rows = []
    for name in CRITERIA:
        pairs, missing = [], 0
        for item in items:
            if name not in item.criteria:
                continue
            first, again = feedback.judge.get((item.run_id, name)), second.scores.get((item.run_id, name))
            if first is None or again is None:
                missing += 1
            else:
                pairs.append((first, again))
        rows.append(Row(name, pair_stats(pairs), 0, missing))
    return rows


def labelled_answers(items: Sequence[Item], feedback: Feedback) -> int:
    """Answers with at least one label of Dan's on a criterion that applies to them."""
    return sum(1 for item in items if any((item.run_id, name) in feedback.dan for name in item.criteria))


def decline_line(items: Sequence[Item]) -> str:
    """`15 decline answers from 5 questions`: how many decline answers the good_decline numbers rest on, and how many distinct questions they came from."""
    declines = [item for item in items if item.path == DECLINE]
    return f"{len(declines)} decline answers from {len({item.example_id for item in declines})} questions"


def _table(header: Sequence[str], rows: Sequence[Sequence[str]]) -> list[str]:
    widths = [max(len(str(line[column])) for line in [header, *rows]) for column in range(len(header))]
    lines = ["| " + " | ".join(str(cell).ljust(width) for cell, width in zip(line, widths, strict=True)).rstrip() + " |" for line in [header, *rows]]
    lines.insert(1, "|" + "|".join("-" * (width + 2) for width in widths) + "|")
    return lines


def render(
    experiment: str,
    selection: Selection,
    feedback: Feedback,
    vs_dan: Sequence[Row],
    self_rows: Sequence[Row],
    second: SecondPass,
) -> list[str]:
    """The whole report as lines: what was compared, the two tables, the decline line and the footnotes."""
    items = selection.items
    off_path = sum(1 for (run_id, name) in feedback.dan if not any(item.run_id == run_id and name in item.criteria for item in items))
    lines = [
        f"# Judge calibration: {experiment}",
        "",
        f"{len(items)} answers (annotation queue {queue_name(experiment)}), {labelled_answers(items, feedback)} of {len(items)} labelled by Dan.",
        f"{decline_line(items)} (the good_decline figures rest on these).",
    ]
    if selection.errored:
        lines.append("Not queued (errored): " + ", ".join(f"{example} rep {rep}" for example, rep, _why in selection.errored) + ".")
    lines += [
        "",
        "## Judge vs Dan",
        "",
        "Judge = its grade of run 1 (the feedback logged on the experiment). Only labelled pairs count; a criterion that does not apply to an answer's path is null and not counted.",
        "",
        *_table(
            ["criterion", "n", "agreement", "kappa", "unlabelled", "no judge grade", "flag"],
            [[row.criterion, str(row.stats.n), row.stats.percent_text, row.stats.kappa_text, str(row.unlabelled), str(row.no_verdict), row.stats.flag] for row in vs_dan],
        ),
        "",
        "## Judge self-agreement",
        "",
        (
            "The same answers judged twice: run 1 = the logged feedback, run 2 = one fresh judge pass"
            f" ({second.judged} answers judged now, {second.cached} from the cache{f', {second.errors} judge errors' if second.errors else ''})."
        ),
        "",
        *_table(
            ["criterion", "n", "agreement", "kappa", "missing a grade", "flag"],
            [[row.criterion, str(row.stats.n), row.stats.percent_text, row.stats.kappa_text, str(row.no_verdict), row.stats.flag] for row in self_rows],
        ),
        "",
        f"Weak = n < {MIN_N} or kappa < {MIN_KAPPA} or kappa undefined. Kappa is Cohen's (sklearn.metrics.cohen_kappa_score) over pass/fail.",
        SELF_PREFERENCE_LINE,
    ]
    if feedback.ignored_keys:
        lines.append("Ignored feedback keys (not one of the four criteria): " + ", ".join(f"{key} ({count})" for key, count in sorted(feedback.ignored_keys.items())) + ".")
    if off_path:
        lines.append(f"{off_path} label(s) of Dan's are on a criterion that does not apply to the answer's path: ignored.")
    return lines


# --- the command ---------------------------------------------------------------------------------------------------------------------------


def _default_judge(environ: Mapping[str, str]) -> Judge:
    return build_judge(environ)


def main(
    argv: Sequence[str] | None = None,
    *,
    environ: Mapping[str, str] | None = None,
    client: Any = None,
    client_factory: ClientFactory | None = None,
    judge_factory: JudgeFactory | None = None,
    records: Sequence[Record] | None = None,
    results_dir: Path = RESULTS_DIR,
    cache_dir: Path = CACHE_DIR,
    stdout: Any = None,
    stderr: Any = None,
) -> int:
    """The command. The client, the judge, the dev records and the two output folders are arguments, so a test runs it whole on a recording client and a fake judge."""
    out, err = sys.stdout if stdout is None else stdout, sys.stderr if stderr is None else stderr
    environ = os.environ if environ is None else environ
    experiment = experiment_from(environ)
    if experiment is None or list(argv or []):
        print(usage_text("agreement") if experiment is None else f"{PROGRAM} takes no arguments: set EXPERIMENT in the environment (make agreement EXPERIMENT=<name>)", file=err)
        return USAGE_EXIT
    cache_path = Path(cache_dir) / f"{experiment}-judge2.json"
    make_judge = judge_factory or _default_judge
    try:
        check_name(experiment)
        judge: Judge | None = None
        if not cache_path.exists():
            judge = make_judge(environ)  # the settings error comes first, before any LangSmith call or queue read
        dev_records = list(load_dataset(DEFAULT_SPLIT).records) if records is None else list(records)
        if client is None:
            client = (client_factory or make_client)(environ)
        check_experiment(client, experiment)
        selection = select(fetch_runs(client, experiment), dev_records)
        feedback = read_feedback(client, [item.run_id for item in selection.items], find_copies(client, experiment))
        labelled = labelled_answers(selection.items, feedback)
        if labelled == 0:
            raise CalibrationError(f"0 of {len(selection.items)} answers labelled: label the answers in the LangSmith annotation queue {queue_name(experiment)} first, then run it again")

        def get_judge() -> Judge:
            return judge if judge is not None else make_judge(environ)

        second = second_pass(selection.items, get_judge, cache_path, experiment)
        lines = render(experiment, selection, feedback, judge_vs_dan(selection.items, feedback), self_agreement(selection.items, feedback, second), second)
        report = Path(results_dir) / f"{experiment}-agreement.md"
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    except (CalibrationError, DatasetError, JudgeConfigError) as error:
        print(f"{PROGRAM}: {error}", file=err)
        return USAGE_EXIT
    except Exception as error:  # noqa: BLE001 - a LangSmith or judge failure: say what; the cache keeps the verdicts already paid for, so the rerun resumes
        print(f"{PROGRAM}: failed: {type(error).__name__}: {error}; run it again, answers already judged are not judged twice", file=err)
        return FAILED_EXIT
    for line in lines:
        print(line, file=out)
    print(f"\nwritten: {report}", file=out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
