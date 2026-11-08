"""A LangSmith client that records instead of sending (PR-12.F1), for the tracing tests. No socket is opened, no key is real.

    recording_client(...)   -> (client, recorder)   a real `langsmith.Client` with the flysafelz anonymiser on it, whose HTTP session is a recording adapter
    Recorder                what the client would have sent: `.sent`, `.runs()`, `.feedback()`, `.text()`

The recipe is the one verified offline with langsmith 0.14.17: `Client(api_url="http://langsmith.invalid", ..., auto_batch_tracing=False, info=LangSmithInfo())`,
and only THEN an adapter mounted on `client.session` for http:// and https://. The Client mounts its own adapters while it is built, so one mounted before
is replaced and the call reaches `getaddrinfo`. `info=` stops the `/info` lookup. With `auto_batch_tracing=False` each run is sent as it starts (`POST /runs`)
and as it ends (`PATCH /runs/<id>`), and feedback as `POST /feedback`.

The adapter answers `GET /sessions` (the project lookup feedback needs) with one project, and everything else with 200 `{}`.
"""

import json
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import parse_qs, urlsplit
from uuid import UUID, uuid4

import requests
import requests.adapters
from fakes import FakeToolChatModel
from graph_helpers import agent_graph, ask
from langchain_core.messages import AIMessage
from langchain_core.tracers.langchain import LangChainTracer
from langsmith import Client
from langsmith.schemas import LangSmithInfo

from flysafelz.agent import tracing
from flysafelz.agent.tracing import ANONYMISER

PROJECT_ID = UUID("00000000-0000-4000-8000-000000000012")
API_URL = "http://langsmith.invalid"


@dataclass(frozen=True)
class Request:
    method: str
    path: str
    body: Any  # the parsed JSON body, or None

    def __str__(self) -> str:
        return f"{self.method} {self.path}"


@dataclass
class Recorder:
    """Every request the client made, in order."""

    sent: list[Request] = field(default_factory=list)
    project: str = "flysafelz"

    def add(self, request: requests.PreparedRequest) -> requests.Response:
        parts = urlsplit(str(request.url))
        body = request.body
        parsed = None
        if body:
            try:
                parsed = json.loads(body)
            except ValueError:
                parsed = {"unparsed": body.decode("utf-8", "replace") if isinstance(body, bytes) else str(body)}
        self.sent.append(Request(str(request.method), parts.path, parsed))
        reply = requests.Response()
        reply.status_code = 200
        reply.url = str(request.url)
        reply.request = request
        reply.headers["Content-Type"] = "application/json"
        if request.method == "GET" and parts.path.endswith("/sessions"):
            name = parse_qs(parts.query).get("name", [self.project])[0]
            reply._content = json.dumps([{"id": str(PROJECT_ID), "name": name, "tenant_id": str(uuid4()), "start_time": "2026-10-31T07:30:00Z", "reference_dataset_id": None}]).encode()
        else:
            reply._content = b"{}"
        return reply

    def of(self, method: str, prefix: str) -> list[Request]:
        return [r for r in self.sent if r.method == method and r.path.startswith(prefix)]

    def feedback(self) -> list[dict[str, Any]]:
        """The body of each `POST /feedback`."""
        return [r.body for r in self.of("POST", "/feedback")]

    def runs(self) -> dict[str, dict[str, Any]]:
        """Each run as LangSmith would hold it at the end: the fields of its `POST /runs`, overwritten by those of its `PATCH /runs/<id>` (outputs, metadata, error)."""
        merged: dict[str, dict[str, Any]] = {}
        for request in self.of("POST", "/runs"):
            merged[request.body["id"]] = dict(request.body)
        for request in self.of("PATCH", "/runs/"):
            run = merged.setdefault(request.path.rsplit("/", 1)[1], {})
            run.update({key: value for key, value in request.body.items() if value is not None})
        return merged

    def root_runs(self) -> list[dict[str, Any]]:
        """The runs with no parent: one per turn."""
        return [run for run in self.runs().values() if not run.get("parent_run_id")]

    def metadata(self, run: dict[str, Any]) -> dict[str, Any]:
        return (run.get("extra") or {}).get("metadata") or {}

    def text(self) -> str:
        """Everything the client sent, as one string to search: every request's path and body."""
        return "\n".join(f"{r} {json.dumps(r.body, ensure_ascii=False)}" for r in self.sent)


class _Adapter(requests.adapters.HTTPAdapter):
    def __init__(self, recorder: Recorder) -> None:
        super().__init__()
        self.recorder = recorder

    def send(self, request: requests.PreparedRequest, *args: Any, **kwargs: Any) -> requests.Response:
        return self.recorder.add(request)


def recording_client[C: Client](client_class: type[C] = Client) -> tuple[C, Recorder]:  # type: ignore[assignment]
    """(a client with the flysafelz anonymiser that records what it sends, the recorder). `client_class` lets a test subclass the client (for example to make a call raise)."""
    client = client_class(api_url=API_URL, web_url=API_URL, api_key="test", anonymizer=ANONYMISER, auto_batch_tracing=False, info=LangSmithInfo())
    recorder = Recorder()
    adapter = _Adapter(recorder)
    for prefix in ("http://", "https://"):
        client.session.mount(prefix, adapter)  # after the client is built: it mounts its own adapters while it is
    return client, recorder


def model_for(*steps: AIMessage | str) -> FakeToolChatModel:
    """A scripted chat model: each step is an AIMessage (a tool call, say) or the text of one."""
    return FakeToolChatModel(script=[step if isinstance(step, AIMessage) else AIMessage(step) for step in steps])


def tracers_of(graph: Any) -> list[LangChainTracer]:
    """The tracers `graph.with_config(callbacks=...)` put on a graph (none for an untraced graph)."""
    callbacks = (getattr(graph, "config", None) or {}).get("callbacks") or []
    return [callback for callback in callbacks if isinstance(callback, LangChainTracer)]


def traced_turn(client: Client, *steps: AIMessage | str, question: str | None = None, thread_id: str = "thread-1", graph: Any = None, **ask_more: Any) -> dict[str, Any]:
    """One turn on the real graph over a model scripted with `steps`, traced through `client`; returns the state. Waits for the feedback before returning."""
    built = graph or tracing.traced(agent_graph(model_for(*steps)), client)
    given = {} if question is None else {"question": question}
    state = ask(built, thread_id=thread_id, **given, **ask_more)
    tracing.wait_for_feedback()
    return state
