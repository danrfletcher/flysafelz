"""Fake LangSmith runs and trees, and the recording client with `list_runs`, `read_project`, `list_feedback` and `create_feedback`, for the online eval tests (7 PR-1.F1, PR-2.F1). No socket, no key, no real client.

    fake_run(key, **overrides)       a root run as `client.list_runs` returns it: a real `langsmith.schemas.Run`, id from uuid5 of `key` on a fixed namespace, started one hour before the frozen `now`
    fake_tree(children=[...], ...)   FakeTree(root, children): the root run and the runs below it. A child is a tool name ("assess", "search_guidance", ...) or a ready run
    assess_artifact(...)             the artifact the assess tool's run holds, with the fields the online job reads
    scored_tree(key, ...)            a seed-style trace whose answer passes every deterministic check: a root with a guard record, an `assess` run, and a `search_guidance` run
    OnlineRecordingClient            5a's RecordingClient plus `list_runs(**kw)`, `read_project(project_name)`, `list_feedback(run_ids=)` and `create_feedback(run_id=, key=, score=, ..., session_id=)`,
                                     all recorded in `calls`. It keeps the feedback it was given (`feedback`), so a rerun sees it. It raises on any call the online job must never make

`fake_run` takes the Run fields (`inputs`, `outputs`, `error`, `reference_example_id`, `parent_run_id`, `start_time`, `run_type`, ...) and these shortcuts, which build the root run the way ticket 4 writes it:
    source=          metadata `source` ("pilot" by default; `NO_SOURCE` leaves the key out)
    path=            metadata `path` ("assess" by default; `NO_PATH` leaves the key out)
    output_path=     outputs `path` (left out by default)
    answer=          outputs `answer`; `NO_ANSWER` leaves the key out
    metadata=        more metadata, merged over the above
    now_input=       inputs `now`
    guard=           outputs `guard` (`passed`, `retried`, `fallback`); left out by default, which is a run with no guard record
"""

import uuid
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import Any

from langsmith import schemas as ls_schemas

from evals.agent.recording_client import RecordingClient
from flysafelz.agent.answer import recheck_line
from flysafelz.agent.state import DEFAULT_TZ

NOW = datetime(2026, 10, 31, 7, 30, tzinfo=UTC)  # the conftest `now` fixture (test_online_pull checks they agree); not imported, because "conftest" names a different file in tests/unit/agent
NAMESPACE = uuid.UUID("5c1f7e0a-7c3d-4a52-9d57-0b9a5e3f6a10")
NO_SOURCE = object()
NO_PATH = object()
NO_ANSWER = object()
ANSWER = "[path: assess]\nGO WITH CONDITIONS tomorrow 10:00 to 16:00."


def run_id(key: str) -> uuid.UUID:
    return uuid.uuid5(NAMESPACE, key)


def fake_run(key: str = "run", **overrides: Any) -> ls_schemas.Run:
    """A root run: pilot, path assess, an answer, started an hour before the frozen now. See the module text for the shortcuts."""
    source = overrides.pop("source", "pilot")
    path = overrides.pop("path", "assess")
    output_path = overrides.pop("output_path", None)
    answer = overrides.pop("answer", ANSWER)
    now_input = overrides.pop("now_input", None)
    guard = overrides.pop("guard", None)
    metadata: dict[str, Any] = {}
    if source is not NO_SOURCE:
        metadata["source"] = source
    if path is not NO_PATH:
        metadata["path"] = path
    metadata.update(overrides.pop("metadata", {}))
    outputs: dict[str, Any] = {}
    if answer is not NO_ANSWER:
        outputs["answer"] = answer
    if output_path is not None:
        outputs["path"] = output_path
    if guard is not None:
        outputs["guard"] = guard
    inputs: dict[str, Any] = {"messages": [], "lat": 53.802797, "lon": -0.086644}
    if now_input is not None:
        inputs["now"] = now_input
    started = overrides.pop("start_time", NOW - timedelta(hours=1))
    fields: dict[str, Any] = {
        "id": run_id(key),
        "trace_id": run_id(key),
        "name": "LangGraph",
        "run_type": "chain",
        "start_time": started,
        "end_time": None if started is None else started + timedelta(seconds=20),
        "inputs": inputs,
        "outputs": outputs,
        "extra": {"metadata": metadata},
        **overrides,
    }
    return ls_schemas.Run(**fields)


def assess_artifact(
    *,
    window: tuple[str, str] = ("2026-10-31T10:00:00Z", "2026-10-31T16:00:00Z"),
    segments: Sequence[tuple[str, str, str | None]] = (("2026-10-31T10:00:00+00:00", "2026-10-31T11:00:00+00:00", "PILOT_DISCRETION"),),
) -> dict[str, Any]:
    """What the assess tool's run holds. A segment's verdict None is an OUTLOOK hour."""
    return {
        "tool": "assess",
        "tz": "Europe/London",
        "window_start": window[0],
        "window_end": window[1],
        "segments": [
            {"start": start, "end": end, "verdict": verdict, "approaching_limits": verdict == "PILOT_DISCRETION", "tier": "VERDICT" if verdict else "OUTLOOK"} for start, end, verdict in segments
        ],
    }


def tool_run(name: str, root: ls_schemas.Run, *, outputs: dict[str, Any] | None = None, minutes: int = 1, **overrides: Any) -> ls_schemas.Run:
    """A tool run below `root`. The assess tool's outputs are shaped as a real trace holds them (a Command: output.update.messages[*].artifact)."""
    if outputs is None and name == "assess":
        outputs = {"output": {"graph": None, "update": {"messages": [{"type": "tool", "content": "Window: ...", "artifact": assess_artifact()}]}, "resume": None, "goto": []}}
    started = root.start_time + timedelta(minutes=minutes)
    fields: dict[str, Any] = {
        "id": run_id(f"{root.id}/{name}/{minutes}"),
        "trace_id": root.trace_id,
        "parent_run_id": root.id,
        "name": name,
        "run_type": "tool",
        "start_time": started,
        "end_time": started + timedelta(seconds=2),
        "inputs": {},
        "outputs": outputs if outputs is not None else {"output": {"content": f"{name} result"}},
        **overrides,
    }
    return ls_schemas.Run(**fields)


@dataclass
class FakeTree:
    root: ls_schemas.Run
    children: list[ls_schemas.Run]
    feedback: list[tuple[str, str | None]] = field(default_factory=list)  # (key, evaluator version) of feedback the run already has: `judge_client` stores it


def fake_tree(children: Sequence[str | ls_schemas.Run] = (), root: ls_schemas.Run | None = None, **root_overrides: Any) -> FakeTree:
    """The root run and the runs below it. A child given as a name becomes a tool run of that name, one minute after the one before it."""
    top = root if root is not None else fake_run(**root_overrides)
    return FakeTree(top, [child if isinstance(child, ls_schemas.Run) else tool_run(child, top, minutes=index + 1) for index, child in enumerate(children)])


def guidance_run(root: ls_schemas.Run, documents: Sequence[str] = ("CAP 1535",), *, minutes: int = 1) -> ls_schemas.Run:
    """A `search_guidance` tool run whose result names these guidance documents, shaped as a real trace holds it (output.artifact.chunks)."""
    chunks = [{"document": document, "edition": "1", "section": "1", "title": "Weather", "text": "..."} for document in documents]
    return tool_run("search_guidance", root, outputs={"output": {"content": "guidance", "artifact": {"query": "q", "chunks": chunks}}}, minutes=minutes)


def recheck_text() -> str:
    """The re-check line ticket 4's own function gives for the default assess window, 4 hours after the run's `now`."""
    window_start = datetime(2026, 10, 31, 10, 0, tzinfo=UTC)
    line = recheck_line([SimpleNamespace(window_start=window_start)], NOW - timedelta(hours=1), DEFAULT_TZ, "assess")
    assert line is not None
    return line


def good_answer(documents: Sequence[str] = ("CAP 1535",)) -> str:
    """An answer that passes every deterministic check for the default assess artifact: a verdict word first, the window, the re-check line, and the guidance document."""
    return f"PILOT DISCRETION: fly between 10:00 and 16:00. {recheck_text()} See {' and '.join(documents)}."


def scored_tree(key: str = "run", *, documents: Sequence[str] = ("CAP 1535",), **root_overrides: Any) -> FakeTree:
    """A seed-style trace: a root with a guard record and a good answer, a `search_guidance` run and an `assess` run. Every deterministic check on it scores 1, bar the decline check (null)."""
    fields: dict[str, Any] = {"guard": "passed", "answer": good_answer(documents), **root_overrides}
    root = fake_run(key, **fields)
    return FakeTree(root, [guidance_run(root, documents, minutes=1), tool_run("assess", root, minutes=2)])


class OnlineRecordingClient(RecordingClient):
    """5a's recording client, with the methods the online job calls. `runs` is what `list_runs` returns for a project read, lazily and in order, as the SDK pages through it; with `trace_id=` it
    returns that trace's runs (the root among `runs`, the rest in `children`). Feedback is kept in `feedback`, so a second pass sees the first pass's posts.

    Failure injection: `fail_feedback={(run_id, key)}` makes `create_feedback` raise for that run and key until the pair is removed from the set (the call is recorded, nothing is stored);
    `fail_list_feedback={run_id}` does the same for `list_feedback`. `fail_on={"create_feedback": 2}` is 5a's, by call number.
    The calls the online job must never make (it reads no dataset, example, experiment or other project) raise AssertionError.
    """

    def __init__(
        self,
        runs: Sequence[ls_schemas.Run] = (),
        *,
        children: Sequence[ls_schemas.Run] = (),
        fail_feedback: Iterable[tuple[uuid.UUID, str]] = (),
        fail_list_feedback: Iterable[uuid.UUID] = (),
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.runs = list(runs)
        self.children = list(children)
        self.pages_read = 0  # how many runs the caller took from list_runs
        self.feedback: list[ls_schemas.Feedback] = []
        self.fail_feedback = set(fail_feedback)
        self.fail_list_feedback = set(fail_list_feedback)
        self.fail_run_url = False
        self.duplicate_posts: list[tuple[uuid.UUID, str, str | None]] = []  # a (run id, key, version) that was posted when feedback of that triple already existed

    def list_runs(self, **kw: Any) -> Iterator[ls_schemas.Run]:
        self._record("list_runs", **kw)
        trace = kw.get("trace_id")
        pool = self.runs if trace is None else [run for run in (*self.runs, *self.children) if str(run.trace_id) == str(trace)]

        def lazily() -> Iterator[ls_schemas.Run]:
            for run in pool:
                self.pages_read += 1
                yield run

        return lazily()

    def read_project(self, *, project_name: str) -> ls_schemas.TracerSession:
        self._record("read_project", project_name=project_name)
        return ls_schemas.TracerSession(id=run_id(f"project/{project_name}"), name=project_name, start_time=datetime(2026, 10, 1, tzinfo=UTC), tenant_id=run_id("tenant"), reference_dataset_id=None)

    def list_feedback(self, *, run_ids: Sequence[uuid.UUID] | None = None, **kw: Any) -> Iterator[ls_schemas.Feedback]:
        self._record("list_feedback", run_ids=list(run_ids or []), **kw)
        if any(wanted in self.fail_list_feedback for wanted in run_ids or []):
            raise RuntimeError("simulated LangSmith error in list_feedback")
        wanted_ids = {str(wanted) for wanted in run_ids or []}
        return iter([posted for posted in self.feedback if str(posted.run_id) in wanted_ids])

    def create_feedback(
        self, run_id: uuid.UUID, key: str, *, score: float | None = None, comment: str | None = None, extra: dict[str, Any] | None = None, session_id: uuid.UUID | None = None, **kw: Any
    ) -> ls_schemas.Feedback:
        self._record("create_feedback", run_id=run_id, key=key, score=score, comment=comment, extra=extra, session_id=session_id, **kw)
        if (run_id, key) in self.fail_feedback:
            raise RuntimeError(f"simulated LangSmith error posting {key}")
        triple = (run_id, key, (extra or {}).get("evaluator_version"))
        if triple in self.stored_triples():
            self.duplicate_posts.append(triple)
        stored = ls_schemas.Feedback(
            id=uuid.uuid4(), created_at=NOW, modified_at=NOW, run_id=run_id, trace_id=run_id, key=key, score=score, value=kw.get("value"), comment=comment, extra=extra, session_id=session_id
        )
        self.feedback.append(stored)
        return stored

    def get_run_url(self, *, run: ls_schemas.Run, project_name: str | None = None, project_id: uuid.UUID | None = None, address: Any = None) -> str:
        """LangSmith's `get_run_url(run=, project_name=)`: a fixed url for the run, recorded. `fail_run_url` makes it raise (a client that cannot say)."""
        self._record("get_run_url", run_id=run.id, project_name=project_name)
        if self.fail_run_url:
            raise RuntimeError("simulated LangSmith error in get_run_url")
        return f"https://smith.example/o/org/projects/p/{project_name}/r/{run.id}"

    def seed_feedback(self, run: ls_schemas.Run, key: str, *, version: str | None, score: float | None = 1, comment: str = "earlier", extra_returned: bool = True) -> ls_schemas.Feedback:
        """Feedback already on a run, as an earlier pass left it: written at `version` (in `extra`, unless `extra_returned` is false, and at the end of the comment), or by something else when
        version is None. Not a recorded call."""
        written = {"evaluator_version": version} if version is not None and extra_returned else None
        tail = "" if version is None else f" | evaluator_version={version}"
        stored = ls_schemas.Feedback(id=uuid.uuid4(), created_at=NOW, modified_at=NOW, run_id=run.id, trace_id=run.id, key=key, score=score, comment=comment + tail, extra=written)
        self.feedback.append(stored)
        return stored

    # --- the calls the online job must never make --------------------------------------------------------------------------------

    def list_examples(self, **kw: Any) -> Iterator[Any]:
        raise AssertionError("the online job must not call list_examples")

    def read_dataset(self, **kw: Any) -> Any:
        raise AssertionError("the online job must not call read_dataset")

    def list_projects(self, **kw: Any) -> Any:
        raise AssertionError("the online job must not call list_projects")

    def evaluate(self, *args: Any, **kw: Any) -> Any:
        raise AssertionError("the online job must not run an evaluation")

    evaluate_existing = aevaluate = evaluate

    # --- reading the record ------------------------------------------------------------------------------------------------------

    def list_run_calls(self) -> list[dict[str, Any]]:
        return [kwargs for method, kwargs in self.calls if method == "list_runs"]

    def create_calls(self) -> list[dict[str, Any]]:
        return [kwargs for method, kwargs in self.calls if method == "create_feedback"]

    def stored_triples(self) -> list[tuple[uuid.UUID, str, str | None]]:
        """(run id, key, evaluator version) of every feedback stored, in the order stored: the invariant is that none appears twice."""
        return [(posted.run_id, posted.key, (posted.extra or {}).get("evaluator_version")) for posted in self.feedback]
