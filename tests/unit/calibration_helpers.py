"""Mocks for the judge calibration tests (5a PR-8.F1). Nothing here calls LangSmith or a model, and no key is read.

CalibrationClient(*experiments)   a recording stand-in for the LangSmith client: it holds the fake experiments (tests/factories.fake_experiment), their feedback and the
                                  annotation queues, answers the methods `make annotate` and `make agreement` call, and records every call in `calls` as (method, kwargs).
                                  It behaves like the service where it matters: `create_run` stores a run in its project (a second one with an id in use is a conflict),
                                  `add_runs_to_annotation_queue` fills the queue, a second `create_annotation_queue` with a name in use is a conflict, and `update_run` does what
                                  the pinned SDK does: it stamps `end_time` with the moment of the call when none is given (so a test can see an experiment run being moved in time).
queue_payloads(client)            every argument of every write call (create_run, queue create, add, update) as one text, for the held-out and no-judge-verdict scans
agree_with_judge(client, exp)     the scenario of the golden path: `make annotate` has run, the judge's run-1 feedback is on every run of the 39 (the experiment's own runs), and Dan's labels
                                  are on all of them (on the queued copies, where he submits them)
FakeJudgeModel                    (re-exported) the chat model the judge tests use; `judge_for(...)` builds a Judge over it with a canned reply per answer
"""

import io
import re
import uuid
from collections.abc import Callable, Iterator, Sequence
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any, cast

from factories import FakeExperiment, dan_feedback, judge_feedback
from judge_helpers import FakeJudgeModel, canned
from langsmith import schemas as ls_schemas

from evals.agent import annotate
from evals.agent.annotate import select
from evals.agent.judge import CRITERIA, Judge


class NotFound(Exception):
    """What the real client raises for an unknown project (langsmith's LangSmithNotFoundError)."""


class CalibrationClient:
    """Records each call; keeps the experiments, their feedback and the queues in memory."""

    def __init__(self, *experiments: FakeExperiment, dataset_name: str = "flysafelz-dev", fail_on: dict[str, int] | None = None) -> None:
        self.experiments = {experiment.name: experiment for experiment in experiments}
        self.runs = {run.id: run for experiment in experiments for run in experiment.runs}
        self.projects: dict[str, list[ls_schemas.Run]] = {}  # projects made by create_run: the calibration copies
        self.feedback: list[ls_schemas.Feedback] = []
        self.queues: dict[str, dict[str, Any]] = {}
        self.dataset_name = dataset_name
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self._fail_on = dict(fail_on or {})
        self._seen: dict[str, int] = {}

    def _record(self, method: str, **kwargs: Any) -> None:
        self.calls.append((method, kwargs))
        self._seen[method] = self._seen.get(method, 0) + 1
        if self._fail_on.get(method) == self._seen[method]:
            raise RuntimeError(f"simulated LangSmith error in {method} (call {self._seen[method]})")

    def forget_calls(self) -> None:
        """Start the record afresh: what a test goes on to check is what the command under test did, not the set-up (`make annotate`, the labelling)."""
        self.calls.clear()
        self._seen.clear()

    def methods_called(self) -> list[str]:
        return [method for method, _kwargs in self.calls]

    def calls_to(self, method: str) -> list[dict[str, Any]]:
        return [kwargs for name, kwargs in self.calls if name == method]

    def writes(self) -> list[str]:
        return [m for m in self.methods_called() if m in WRITES]

    # --- reading the experiment ---------------------------------------------------------------------------------------------

    def read_project(self, *, project_name: str) -> ls_schemas.TracerSessionResult:
        self._record("read_project", project_name=project_name)
        if project_name not in self.experiments:
            raise NotFound(f"project {project_name} not found")
        return self.experiments[project_name].project

    def read_dataset(self, *, dataset_id: Any = None, dataset_name: str | None = None) -> Any:
        self._record("read_dataset", dataset_id=dataset_id, dataset_name=dataset_name)
        return SimpleNamespace(id=dataset_id, name=self.dataset_name)

    def list_runs(self, *, project_name: str, is_root: bool | None = None, **kwargs: Any) -> Iterator[ls_schemas.Run]:
        self._record("list_runs", project_name=project_name, is_root=is_root)
        if project_name in self.projects:
            return iter(list(self.projects[project_name]))
        return iter(list(self.experiments[project_name].runs))

    def has_project(self, project_name: str, **kwargs: Any) -> bool:
        self._record("has_project", project_name=project_name)
        return project_name in self.projects or project_name in self.experiments

    def create_run(self, name: str, inputs: dict[str, Any], run_type: str, *, project_name: str, id: Any, **kwargs: Any) -> None:
        self._record("create_run", name=name, inputs=inputs, run_type=run_type, project_name=project_name, id=id, **kwargs)
        assert project_name not in self.experiments, f"create_run into the experiment {project_name}: an experiment's runs are the runner's"
        run_id = _as_uuid(id)
        assert run_id not in self.runs, f"a run {run_id} already exists (the service refuses a second one)"
        run = ls_schemas.Run(
            id=run_id,
            name=name,
            run_type=run_type,
            trace_id=run_id,
            inputs=inputs,
            outputs=kwargs.get("outputs"),
            start_time=kwargs["start_time"],
            end_time=kwargs.get("end_time"),
            tags=list(kwargs.get("tags") or []),
            extra=kwargs.get("extra"),
        )
        self.runs[run_id] = run
        self.projects.setdefault(project_name, []).append(run)

    def flush(self, timeout: float | None = None) -> None:
        self._record("flush", timeout=timeout)

    def copy_of(self, run: ls_schemas.Run) -> ls_schemas.Run:
        """The calibration copy of an experiment run, as `make annotate` logged it (found by its `source_run_id` metadata, as `make agreement` does)."""
        return next(copy for runs in self.projects.values() for copy in runs if ((copy.extra or {}).get("metadata") or {}).get("source_run_id") == str(run.id))

    def feedback_on(self, run: ls_schemas.Run) -> list[ls_schemas.Feedback]:
        return [item for item in self.feedback if item.run_id == run.id]

    # --- the queue ----------------------------------------------------------------------------------------------------------

    def list_annotation_queues(self, *, name: str | None = None, **kwargs: Any) -> Iterator[ls_schemas.AnnotationQueue]:
        self._record("list_annotation_queues", name=name)
        return iter([queue["queue"] for queue in self.queues.values() if name is None or name in queue["queue"].name])

    def create_annotation_queue(self, *, name: str, description: str | None = None, rubric_instructions: str | None = None, **kwargs: Any) -> ls_schemas.AnnotationQueue:
        self._record("create_annotation_queue", name=name, description=description, rubric_instructions=rubric_instructions, **kwargs)
        assert name not in self.queues, f"a queue named {name} already exists (the service refuses a second one)"
        queue = ls_schemas.AnnotationQueue(id=_uuid(f"queue/{name}"), name=name, description=description, tenant_id=_uuid("tenant"))
        self.queues[name] = {"queue": queue, "run_ids": []}
        return queue

    def _queue_by_id(self, queue_id: Any) -> dict[str, Any]:
        return next(queue for queue in self.queues.values() if queue["queue"].id == queue_id)

    def add_runs_to_annotation_queue(self, queue_id: Any, *, run_ids: Sequence[Any]) -> None:
        self._record("add_runs_to_annotation_queue", queue_id=queue_id, run_ids=list(run_ids))
        held = self._queue_by_id(queue_id)["run_ids"]
        for run_id in map(str, run_ids):
            if run_id not in held:
                held.append(run_id)

    def list_runs_from_annotation_queue(self, queue_id: Any, **kwargs: Any) -> Iterator[ls_schemas.RunWithAnnotationQueueInfo]:
        self._record("list_runs_from_annotation_queue", queue_id=queue_id)
        return iter([_queue_run(self.runs[_as_uuid(run_id)]) for run_id in self._queue_by_id(queue_id)["run_ids"]])

    def update_run(self, run_id: Any, *, tags: list[str] | None = None, end_time: datetime | None = None, **kwargs: Any) -> None:
        """As langsmith 0.14.17's `Client.update_run`: with no `end_time` it sends the moment of the call as the run's end time."""
        self._record("update_run", run_id=run_id, tags=tags, end_time=end_time, **kwargs)
        run = self.runs[_as_uuid(run_id)]
        run.end_time = end_time or datetime.now(UTC)
        if tags is not None:
            run.tags = list(tags)

    def queued(self, name: str) -> list[str]:
        return list(self.queues[name]["run_ids"])

    def queued_runs(self, name: str) -> list[ls_schemas.Run]:
        """The runs in the queue, as the annotation view would open them."""
        return [self.runs[_as_uuid(run_id)] for run_id in self.queues[name]["run_ids"]]

    def queued_sources(self, name: str) -> list[str]:
        """The experiment run each queued item is a copy of (its `source_run_id`)."""
        return [(run.extra or {})["metadata"]["source_run_id"] for run in self.queued_runs(name)]

    # --- feedback -----------------------------------------------------------------------------------------------------------

    def list_feedback(self, *, run_ids: Sequence[Any] | None = None, **kwargs: Any) -> Iterator[ls_schemas.Feedback]:
        self._record("list_feedback", run_ids=[str(r) for r in run_ids or []], **kwargs)
        wanted = {str(r) for r in run_ids or []}
        return iter([item for item in self.feedback if str(item.run_id) in wanted])


def _uuid(text: str) -> uuid.UUID:
    return uuid.uuid5(uuid.NAMESPACE_URL, text)


def _as_uuid(value: Any) -> uuid.UUID:
    return value if isinstance(value, uuid.UUID) else uuid.UUID(str(value))


def _queue_run(run: ls_schemas.Run) -> ls_schemas.RunWithAnnotationQueueInfo:
    return ls_schemas.RunWithAnnotationQueueInfo(id=run.id, name=run.name, run_type="chain", start_time=run.start_time)


WRITES = ("create_run", "create_annotation_queue", "add_runs_to_annotation_queue", "update_run")


def queue_payloads(client: CalibrationClient) -> str:
    """Every argument of every write call (copy create, queue create, add, run update), as text, for the scans over what was sent."""
    return "\n".join(repr(kwargs) for method, kwargs in client.calls if method in WRITES)


# --- scenarios -------------------------------------------------------------------------------------------------------------------


def queued_items(experiment: FakeExperiment) -> list:
    return select(experiment.runs, experiment.records).items


def agree_with_judge(
    client: CalibrationClient,
    experiment: FakeExperiment,
    judge_says: Callable[[Any, str], bool] = lambda item, criterion: True,
    dan_says: Callable[[Any, str], bool | str | None] | None = None,
) -> list:
    """The judge's run-1 feedback and Dan's labels on every queued answer, for the criteria that apply. Dan agrees with the judge unless `dan_says` gives something else (None: not labelled)."""
    items = queued_items(experiment)
    out = io.StringIO()
    assert annotate.main([], environ={"EXPERIMENT": experiment.name}, client=client, stdout=out, stderr=out) == 0, out.getvalue()  # the queue Dan labels in
    for item in items:
        run = next(r for r in experiment.runs if str(r.id) == item.run_id)
        copy = client.copy_of(run)  # the judge graded the experiment's run, Dan grades the copy in the queue
        for criterion in item.criteria:
            said = judge_says(item, criterion)
            client.feedback.append(judge_feedback(run, criterion, said))
            labelled = said if dan_says is None else dan_says(item, criterion)
            if labelled is not None:
                client.feedback.append(dan_feedback(copy, criterion, labelled if isinstance(labelled, str) else ("pass" if labelled else "fail")))
    return items


def judge_for(passed: Callable[[Any, str], bool], count: int, items: Sequence[Any] | None = None, start: int = 0) -> Judge:
    """A Judge over a fake model whose n-th reply grades the answer `items[start + n]`: every criterion `passed(item, criterion)` says (a reply is written for all four)."""
    replies = []
    for index in range(start, start + count if items else 0):  # no items: a judge that must never be called (an experiment the command refuses)
        item = items[index]
        replies.append(canned(**{name: (passed(item, name), "again") for name in CRITERIA}))
    return Judge(FakeJudgeModel(replies=replies), prompt="You grade answers.")


def tags_of(run: ls_schemas.Run) -> list[str]:
    return [tag for tag in run.tags or [] if re.match(r"label:", tag)]


def fake_model(judge: Judge) -> FakeJudgeModel:
    """The scripted chat model inside a judge built by `judge_for`: its `replies` (edit one to plant an error) and `calls`."""
    return cast(FakeJudgeModel, judge.model)
