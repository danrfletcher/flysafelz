"""Fake LangSmith runs and trees, and the recording client with `list_runs` and `read_project`, for the online eval tests (7 PR-1.F1). No socket, no key, no real client.

    fake_run(key, **overrides)       a root run as `client.list_runs` returns it: a real `langsmith.schemas.Run`, id from uuid5 of `key` on a fixed namespace, started one hour before the frozen `now`
    fake_tree(children=[...], ...)   FakeTree(root, children): the root run and the runs below it. A child is a tool name ("assess", "search_guidance", ...) or a ready run
    assess_artifact(...)             the artifact the assess tool's run holds, with the fields the online job reads
    OnlineRecordingClient            5a's RecordingClient plus `list_runs(**kw)` and `read_project(project_name)`, both recorded in `calls`

`fake_run` takes the Run fields (`inputs`, `outputs`, `error`, `reference_example_id`, `parent_run_id`, `start_time`, `run_type`, ...) and these shortcuts, which build the root run the way ticket 4 writes it:
    source=          metadata `source` ("pilot" by default; `NO_SOURCE` leaves the key out)
    path=            metadata `path` ("assess" by default; `NO_PATH` leaves the key out)
    output_path=     outputs `path` (left out by default)
    answer=          outputs `answer`; `NO_ANSWER` leaves the key out
    metadata=        more metadata, merged over the above
    now_input=       inputs `now`
"""

import uuid
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

from langsmith import schemas as ls_schemas

from evals.agent.recording_client import RecordingClient

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


def fake_tree(children: Sequence[str | ls_schemas.Run] = (), root: ls_schemas.Run | None = None, **root_overrides: Any) -> FakeTree:
    """The root run and the runs below it. A child given as a name becomes a tool run of that name, one minute after the one before it."""
    top = root if root is not None else fake_run(**root_overrides)
    return FakeTree(top, [child if isinstance(child, ls_schemas.Run) else tool_run(child, top, minutes=index + 1) for index, child in enumerate(children)])


class OnlineRecordingClient(RecordingClient):
    """5a's recording client, with the two methods the online job's pull calls. `runs` is what `list_runs` returns, lazily and in order, as the SDK pages through a project."""

    def __init__(self, runs: Sequence[ls_schemas.Run] = (), **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.runs = list(runs)
        self.pages_read = 0  # how many runs the caller took from list_runs

    def list_runs(self, **kw: Any) -> Iterator[ls_schemas.Run]:
        self._record("list_runs", **kw)

        def lazily() -> Iterator[ls_schemas.Run]:
            for run in self.runs:
                self.pages_read += 1
                yield run

        return lazily()

    def read_project(self, *, project_name: str) -> ls_schemas.TracerSession:
        self._record("read_project", project_name=project_name)
        return ls_schemas.TracerSession(id=run_id(f"project/{project_name}"), name=project_name, start_time=datetime(2026, 10, 1, tzinfo=UTC), tenant_id=run_id("tenant"))

    def list_run_calls(self) -> list[dict[str, Any]]:
        return [kwargs for method, kwargs in self.calls if method == "list_runs"]
