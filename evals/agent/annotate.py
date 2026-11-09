"""Judge calibration, step one (5a PR-8.F1, G13, GP6): push the dev answers Dan labels into a LangSmith annotation queue.

    python -m evals.agent.annotate              what `make annotate EXPERIMENT=<v1 dev experiment>` runs; EXPERIMENT comes from the environment, as CASE does for `make fixtures`

    select(runs, records) -> Selection          the answers to queue, the errored ones left out, the reps found
    queue_name(experiment)                      flysafelz-calibration-<experiment>
    label_tags(criteria)                        label:<criterion> for each criterion that applies to an answer
    check_experiment(client, experiment)        the refusals that need LangSmith (unknown name, v0, held-out, wrong reps), shared with `make agreement`
    calibration_project(experiment)              the project the copies are logged in (flysafelz-calibration-<experiment>)
    find_copies(client, experiment) -> dict      {experiment run id: its copy's run id}, shared with `make agreement`
    main(argv, ...) -> int                      0 queued (or already there), 2 refused or mis-called, 1 LangSmith failed

What is queued (G13). 39 answers of a v1 dev experiment: every dev example's rep-1 answer (29) plus the answers of the dev DECLINE examples from reps 2 and 3 (5 x 2 = 10; 15
decline answers from 5 questions, because good_decline has few cases). The decline examples are the dev records whose reference path is `decline`, read from the loader, not a
list kept here.

What is queued is a COPY of each answer, not the experiment's own run (F5). The experiment's runs carry the judge's feedback from `make eval` (`judge.<criterion>`, a 1/0 score and the
judge's reason), and LangSmith's annotation view shows the feedback a run already has, so queueing them would show Dan the verdict he is meant to check. Each answer is therefore
logged again, once, as a run in a calibration project of its own (`flysafelz-calibration-<experiment>`, the queue's name): the question, the time it was asked and the launch zone (the same place the judge is told, from `judge.place_text`), the answer, and
the `label:<criterion>` tags, nothing else. Its metadata holds `source_run_id`, the experiment run it was copied from, which is how `make agreement` maps Dan's labels (on the copy)
back to the judge's grades (on the original). The experiment's own runs are only read: not updated, not tagged, not moved in time (an `update_run` would stamp a new end time on them).
The copy gets a fixed id derived from the project and the source run, so a rerun finds the copies it made and never makes a second one.

Which criteria Dan ticks (G13, E5). `evals.agent.judge.criteria_for` is the one map from a path to its criteria (assess and explain: answers_the_question, prioritised,
actionable; clarify: answers_the_question; decline: answers_the_question, good_decline); this module keeps no second copy. The path is the example's REFERENCE path, as in the
judge. A queue cannot show a different rubric per item, so each copy is tagged `label:<criterion>` for the criteria that apply to it, and the queue's instructions say to
grade only the criteria an item is tagged with. A criterion that does not apply is not tagged, is never asked of Dan and is null in the agreement report, not scored.

What Dan sees (F5). The copy's question, time and place and its answer, its `label:` tags, and the queue's instructions (the four definitions, word for word from the judge's). Never the judge's pass
or fail, its reason or any other score: no feedback of any kind is on a copy, nothing the judge or any evaluator produced is written to the queue or onto a run, and no model writes
annotation feedback. Dan's labels are feedback he submits himself, on the copy, under the bare criterion name (`prioritised`; the judge's own are `judge.prioritised`).

Refusals (exit 2, before the queue exists or any copy is made, nothing written):
  - EXPERIMENT missing or blank: the usage text, and no LangSmith call (the client is not built);
  - a name that is not a plain name, a v0 experiment, or the held-out project or dataset (F7): refused from the name and the project record;
  - an experiment LangSmith does not know;
  - any run whose example is not one of the committed DEV examples, or whose own metadata says its split is not dev (F7). Every run is checked, not only the name;
  - an experiment with fewer than 3 reps, or a run with no rep.
A dev example's run that is missing or errored (the runner logs an errored run with its error) is left out and named; it is not a crash and gets no placeholder item.

Idempotent. The queue is looked up by its name; a queue that exists is reused. The copies already in the calibration project are found by their `source_run_id` and not made again, and
only the copies the queue does not hold yet are added (keyed by run id). A second experiment has its own queue and its own project. A LangSmith failure part way (a copy made but not yet
queued) is finished by running it again.

Nothing is written to the repository or to the dataset's source folder. LangSmith is written to in three ways only: the queue, the copies (one `create_run` each, in the calibration project)
and the queue's contents.
"""

import os
import re
import sys
import uuid
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from evals.agent.import_dataset import DatasetError
from evals.agent.judge import CRITERIA, CRITERIA_BY_PATH, DEFINITIONS, criteria_for, lz_name_from_case, place_text
from evals.agent.judge import SOURCE as JUDGE_SOURCE
from evals.agent.loader import DEFAULT_SPLIT, Record, example_uuid, experiment_target, load_dataset, make_client
from evals.agent.runner import EXPERIMENT_NAME, REPS

PROGRAM = "evals.agent.annotate"
EXPERIMENT_VARIABLE = "EXPERIMENT"
QUEUE_PREFIX = "flysafelz-calibration-"
LABEL_PREFIX = "label:"
DECLINE = "decline"
USAGE_EXIT, FAILED_EXIT = 2, 1
BATCH = 25  # runs per LangSmith call, so a long list never makes one oversized request
FLUSH_SECONDS = 120.0  # how long to wait for the copies to reach LangSmith before they are queued
SOURCE_KEY = "source_run_id"  # the copy's metadata key that names the experiment run it was copied from
HELD_OUT = "held_out"
V0 = "v0"

Client = Any
ClientFactory = Callable[[Mapping[str, str]], Client]


class CalibrationError(Exception):
    """A refusal: the command stops with exit 2 and this one-line message, having written nothing."""


def usage_text(target: str) -> str:
    """The message when EXPERIMENT is missing: how to call the target, and that nothing was sent."""
    return f"usage: make {target} EXPERIMENT=<v1 dev experiment name>   (for example: make {target} EXPERIMENT=v1-dev-20261108-160700)\nEXPERIMENT is not set; nothing was sent to LangSmith."


def experiment_from(environ: Mapping[str, str]) -> str | None:
    """EXPERIMENT from the environment, stripped; None when it is missing or blank."""
    return environ.get(EXPERIMENT_VARIABLE, "").strip() or None


def queue_name(experiment: str) -> str:
    return f"{QUEUE_PREFIX}{experiment}"


def calibration_project(experiment: str) -> str:
    """The LangSmith project the calibration copies of an experiment's answers are logged in (named like its queue)."""
    return f"{QUEUE_PREFIX}{experiment}"


def label_tags(criteria: Sequence[str]) -> list[str]:
    return [f"{LABEL_PREFIX}{name}" for name in criteria]


# --- the refusals from the name --------------------------------------------------------------------------------------------------


def _held_out_names() -> set[str]:
    target = experiment_target(HELD_OUT)
    return {target.project, target.dataset}


def check_name(experiment: str) -> None:
    """Refuse a name that is not a plain name, a v0 experiment and the held-out project or dataset, from the name alone (no LangSmith call). CalibrationError otherwise."""
    if not EXPERIMENT_NAME.fullmatch(experiment):
        raise CalibrationError(f"experiment name {experiment!r} must be letters, digits, '_', '.' or '-'")
    squashed = re.sub(r"[^a-z0-9]", "", experiment.lower())
    if experiment in _held_out_names() or "heldout" in squashed:
        raise CalibrationError(f"{experiment} is a held-out experiment: held-out answers are never queued or labelled during the build (calibration uses dev only)")
    if re.match(r"v0(?![a-z0-9])", experiment.lower()):
        raise CalibrationError(f"{experiment} is a v0 experiment: the judge is calibrated on v1 answers (use a v1-dev-<stamp> experiment)")


def _extra_metadata(item: Any) -> Mapping[str, Any]:
    extra = getattr(item, "extra", None)
    metadata = extra.get("metadata") if isinstance(extra, Mapping) else None
    return metadata if isinstance(metadata, Mapping) else {}


def check_experiment(client: Client, experiment: str) -> None:
    """Refuse an experiment LangSmith does not know, one its own record says is v0 or held-out, and one logged against the held-out dataset. CalibrationError otherwise."""
    try:
        project = client.read_project(project_name=experiment)
    except Exception as error:  # any failure to find the project means there is no such experiment to calibrate on
        raise CalibrationError(f"experiment {experiment!r} was not found in LangSmith ({type(error).__name__}); check the name") from error
    meta = _extra_metadata(project)
    if meta.get("agent") == V0:
        raise CalibrationError(f"{experiment} is a v0 experiment: the judge is calibrated on v1 answers")
    if meta.get("split") == HELD_OUT:
        raise CalibrationError(f"{experiment} is a held-out experiment: held-out answers are never queued or labelled during the build")
    dataset_id = getattr(project, "reference_dataset_id", None)
    if dataset_id is not None:
        try:
            dataset_name = client.read_dataset(dataset_id=dataset_id).name
        except Exception as error:  # not knowing the dataset is not knowing it is dev: refuse, do not guess
            raise CalibrationError(f"the dataset of experiment {experiment} could not be read ({type(error).__name__}); it cannot be shown to be dev") from error
        if dataset_name in _held_out_names():
            raise CalibrationError(f"{experiment} was run on the held-out dataset {dataset_name}: held-out answers are never queued or labelled during the build")


# --- the selection -----------------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Item:
    """One answer to queue: its run, its example, its rep, the criteria that apply to its path, and the answer text (for `make agreement`'s second judge pass)."""

    run_id: str
    example_id: str
    rep: int
    path: str
    criteria: tuple[str, ...]
    answer: str
    started: datetime | None = field(default=None, compare=False)
    ended: datetime | None = field(default=None, compare=False)
    record: Record = field(default_factory=dict, compare=False, repr=False)


@dataclass
class Selection:
    items: list[Item]
    errored: list[tuple[str, int, str]]  # (example id, rep, why) for each dev answer that should have been queued and cannot be
    reps: tuple[int, ...]

    @property
    def ticks(self) -> int:
        return sum(len(item.criteria) for item in self.items)

    @property
    def rep_one(self) -> list[Item]:
        return [item for item in self.items if item.rep == 1]

    @property
    def extra_declines(self) -> list[Item]:
        return [item for item in self.items if item.rep != 1]


def _rep_of(run: Any) -> int | None:
    for tag in getattr(run, "tags", None) or []:
        found = re.fullmatch(r"rep:(\d+)", str(tag))
        if found:
            return int(found.group(1))
    rep = _extra_metadata(run).get("rep")
    return rep if isinstance(rep, int) and not isinstance(rep, bool) else None


def _answer_of(run: Any) -> str:
    outputs = getattr(run, "outputs", None)
    answer = outputs.get("answer") if isinstance(outputs, Mapping) else None
    return answer if isinstance(answer, str) and answer.strip() else ""


def select(runs: Sequence[Any], records: Sequence[Record]) -> Selection:
    """The answers to queue from an experiment's runs, and the ones that cannot be queued. CalibrationError for a run that is not a dev run and for fewer than 3 reps.

    `records` are the committed DEV records (the loader's): a run whose example is not one of them is refused, and so is a run whose own metadata says its split is not dev, so the
    held-out split can never get in whatever the experiment is called (F7). Rep 1 of every dev example is wanted, and reps 2 and 3 of the decline examples.
    """
    dev: dict[Any, Record] = {example_uuid(record["id"]): record for record in records}
    by_slot: dict[tuple[str, int], Any] = {}
    reps: set[int] = set()
    for run in runs:
        split = _extra_metadata(run).get("split")
        if split is not None and split != DEFAULT_SPLIT:
            raise CalibrationError(f"run {run.id} is on a {split} example, not a dev one: nothing was queued")
        record = dev.get(getattr(run, "reference_example_id", None))
        if record is None:
            raise CalibrationError(f"run {run.id} is not on one of the {DEFAULT_SPLIT} examples: nothing was queued")
        rep = _rep_of(run)
        if rep is None:
            raise CalibrationError(f"run {run.id} has no rep (a `rep:<n>` tag): it is not a run of the v1 runner")
        if (record["id"], rep) in by_slot:
            raise CalibrationError(f"two runs of {record['id']} rep {rep} ({by_slot[record['id'], rep].id} and {run.id}): the experiment is not one clean run")
        by_slot[record["id"], rep] = run
        reps.add(rep)
    if reps != set(range(1, REPS + 1)):
        found = ", ".join(map(str, sorted(reps))) or "none"
        raise CalibrationError(f"the experiment has reps {found}; it needs all {REPS} reps (the decline answers are taken from reps 2 and 3)")
    items: list[Item] = []
    errored: list[tuple[str, int, str]] = []
    for record in records:
        path = record["reference"]["path"]
        criteria = criteria_for(path)
        if not criteria:
            continue
        for rep in (1, 2, 3) if path == DECLINE else (1,):
            run = by_slot.get((record["id"], rep))
            if run is None:
                errored.append((record["id"], rep, "no run was logged"))
            elif getattr(run, "error", None) or not _answer_of(run):
                errored.append((record["id"], rep, str(getattr(run, "error", None) or "no answer")))
            else:
                items.append(Item(str(run.id), record["id"], rep, path, criteria, _answer_of(run), getattr(run, "start_time", None), getattr(run, "end_time", None), record))
    return Selection(items, errored, tuple(sorted(reps)))


def fetch_runs(client: Client, experiment: str) -> list[Any]:
    """The experiment's logged runs (the runner's one root run per example and rep). The judge's own model calls are root runs in the same project
    (metadata `source` "judge") and are not answers, so they are left out."""
    runs = client.list_runs(project_name=experiment, is_root=True)
    return [run for run in runs if ((getattr(run, "extra", None) or {}).get("metadata") or {}).get("source") != JUDGE_SOURCE]


def count_line(selection: Selection) -> str:
    """`39 answers (29 rep 1 + 10 decline from reps 2–3), 96 criterion ticks`, with `; errored, not queued: dev_19_buttermere rep 2` when answers are left out."""
    line = f"{len(selection.items)} answers ({len(selection.rep_one)} rep 1 + {len(selection.extra_declines)} decline from reps 2–3), {selection.ticks} criterion ticks"
    if selection.errored:
        line += "; errored, not queued: " + ", ".join(f"{example} rep {rep}" for example, rep, _why in selection.errored)
    return line


def path_summary(selection: Selection) -> list[str]:
    """One line per path: its answers, the criteria that apply and the ticks (answers x criteria)."""
    counts = Counter(item.path for item in selection.items)
    lines = []
    for path, criteria in CRITERIA_BY_PATH.items():
        if counts[path]:
            lines.append(f"  {path:<8} {counts[path]:>2} answers x {len(criteria)} = {counts[path] * len(criteria):>2} ticks: {', '.join(criteria)}")
    return lines


def instructions() -> str:
    """The queue's instructions: how to grade, with the four definitions as the judge has them. No verdict of the judge's is in them."""
    lines = [
        "Grade each answer pass or fail on the criteria it is tagged with, `label:<criterion>`, and leave every other criterion blank: a criterion that is not tagged does not apply to the answer.",
        "Use the feedback key named after the criterion (for example `prioritised`).",
        "Judge the answer as written, from the question, the time it was asked and the launch zone (`location`). You do not need to verify the weather, airspace or NOTAMs.",
        "",
        *(f"- {name}: {DEFINITIONS[name]}" for name in CRITERIA),
    ]
    return "\n".join(lines)


# --- the queue ---------------------------------------------------------------------------------------------------------------------


def _chunks(items: Sequence[Any]) -> list[Sequence[Any]]:
    return [items[start : start + BATCH] for start in range(0, len(items), BATCH)]


def find_queue(client: Client, name: str) -> Any | None:
    """The annotation queue with exactly this name, or None."""
    return next((queue for queue in client.list_annotation_queues(name=name) if queue.name == name), None)


def copy_run_id(experiment: str, source_run_id: str) -> uuid.UUID:
    """The fixed id of the copy of one experiment run, so the same answer is never copied twice."""
    return uuid.uuid5(uuid.NAMESPACE_URL, f"{calibration_project(experiment)}/{source_run_id}")


def find_copies(client: Client, experiment: str) -> dict[str, str]:
    """{experiment run id: run id of its copy} for the copies already logged in the experiment's calibration project; {} when the project does not exist yet."""
    project = calibration_project(experiment)
    if not client.has_project(project):
        return {}
    copies: dict[str, str] = {}
    for run in client.list_runs(project_name=project, is_root=True):
        source = _extra_metadata(run).get(SOURCE_KEY)
        if isinstance(source, str) and source:
            copies[source] = str(run.id)
    return copies


def location_of(record: Record) -> str:
    """The launch zone as the judge is told it (`Location: ...` in judge_message): the LZ name from the example's case folder, else the coordinates. Dan and the judge grade the same item."""
    inputs = record["inputs"]
    return place_text(lz_name_from_case((record.get("fixtures") or {}).get("case")), float(inputs["lat"]), float(inputs["lon"]))


def copy_answer(client: Client, item: Item, experiment: str) -> str:
    """Log one answer as a run of its own in the calibration project and return the new run's id.

    The run holds the question, when it was asked and where (`location_of`, as the judge sees it), the answer, the `label:<criterion>` tags and the id of the run it was copied from, and nothing the judge or an evaluator
    produced. Its times are the original's, so it does not look newer than the answer is.
    """
    run_id = copy_run_id(experiment, item.run_id)
    inputs = item.record.get("inputs", {})
    started = item.started or item.ended or datetime.now().astimezone()
    client.create_run(
        id=run_id,
        name=f"calibration-{item.example_id}",
        run_type="chain",
        project_name=calibration_project(experiment),
        inputs={"question": inputs.get("question"), "now": inputs.get("now"), "location": location_of(item.record)},
        outputs={"answer": item.answer},
        start_time=started,
        end_time=item.ended or started,
        tags=label_tags(item.criteria),
        extra={"metadata": {SOURCE_KEY: item.run_id, "experiment": experiment, "example_id": item.example_id, "rep": item.rep}},
    )
    return str(run_id)


def queue_answers(client: Client, selection: Selection, experiment: str, out: Any) -> None:
    """Create the queue or reuse it, copy the answers it does not have a copy of yet into the calibration project, and add the copies it does not hold yet.

    The experiment's own runs are only read (F5, R2): the copies carry the label tags, so no run of the experiment is tagged or updated.
    """
    name = queue_name(experiment)
    queue = find_queue(client, name)
    held: set[str] = set()
    if queue is None:
        description = f"Calibration of the FlySafeLZ judge: the dev answers of {experiment} (rep 1 of every dev example, and the decline answers of reps 2 and 3)."
        queue = client.create_annotation_queue(name=name, description=description, rubric_instructions=instructions())
        print(f"queue {name}: created", file=out)
    else:
        held = {str(run.id) for run in client.list_runs_from_annotation_queue(queue.id)}
        print(f"queue {name}: reused ({len(held)} answers already in it)", file=out)
    copies = find_copies(client, experiment)
    made = 0
    for item in selection.items:
        if item.run_id not in copies:  # logged before it is added, so an item is never seen half made
            copies[item.run_id] = copy_answer(client, item, experiment)
            made += 1
    if made:
        client.flush(timeout=FLUSH_SECONDS)  # the copies must have reached LangSmith before they are added to the queue
    print(f"{made} answers copied into the project {calibration_project(experiment)} ({len(selection.items) - made} already there)", file=out)
    new = [copies[item.run_id] for item in selection.items if copies[item.run_id] not in held]
    for chunk in _chunks(new):
        client.add_runs_to_annotation_queue(queue.id, run_ids=list(chunk))
    print(f"added {len(new)} to the queue, {len(selection.items) - len(new)} already there", file=out)


# --- the command ---------------------------------------------------------------------------------------------------------------------


def main(
    argv: Sequence[str] | None = None,
    *,
    environ: Mapping[str, str] | None = None,
    client: Client = None,
    client_factory: ClientFactory | None = None,
    records: Sequence[Record] | None = None,
    stdout: Any = None,
    stderr: Any = None,
) -> int:
    """The command. `client` (or `client_factory`) and `records` are for tests; the real run builds the LangSmith client from LANGSMITH_API_KEY and loads the dev split."""
    out, err = sys.stdout if stdout is None else stdout, sys.stderr if stderr is None else stderr
    environ = os.environ if environ is None else environ
    experiment = experiment_from(environ)
    if experiment is None or list(argv or []):
        print(usage_text("annotate") if experiment is None else f"{PROGRAM} takes no arguments: set EXPERIMENT in the environment (make annotate EXPERIMENT=<name>)", file=err)
        return USAGE_EXIT
    try:
        check_name(experiment)
        dev_records = list(load_dataset(DEFAULT_SPLIT).records) if records is None else list(records)
        if client is None:
            client = (client_factory or make_client)(environ)
        check_experiment(client, experiment)
        selection = select(fetch_runs(client, experiment), dev_records)
        print(f"experiment {experiment}: {len(selection.reps)} reps, {len({item.example_id for item in selection.items})} dev examples with answers to queue", file=out)
        queue_answers(client, selection, experiment, out)
    except (CalibrationError, DatasetError) as error:
        print(f"{PROGRAM}: {error}", file=err)
        return USAGE_EXIT
    except Exception as error:  # noqa: BLE001 - a LangSmith failure part way: say what and where, and the rerun finishes the job (the queue is idempotent)
        print(f"{PROGRAM}: LangSmith failed: {type(error).__name__}: {error}; run it again, nothing is queued twice", file=err)
        return FAILED_EXIT
    print(count_line(selection), file=out)
    for line in path_summary(selection):
        print(line, file=out)
    print(f"label them in LangSmith (annotation queue {queue_name(experiment)}), then: make agreement EXPERIMENT={experiment}", file=out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
