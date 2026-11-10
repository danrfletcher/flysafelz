"""`make online-seed` (7 PR-5.F1): the file and key checks before the first question (E6), the order and shape of the calls (G2), the output line, and the failures (one question raising, no URL, an unreadable path).

Real code throughout: `seed.main` runs with a fake entry point that stands in for ticket 4's agent and adds canned run trees to the tracing callback `seed` collects them from, and 5a's recording LangSmith client
(`get_run_url`, `flush` and the rest recorded). The last tests run the real agent graph over a scripted model, with the real client on a recording HTTP session. No key, no socket, no real model.
"""

import io
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from graph_helpers import agent_graph
from langchain_core.tracers.context import run_collector_var, tracing_v2_callback_var
from online_helpers import NO_PATH, OnlineRecordingClient, fake_run, run_id, tool_run
from tracing_helpers import PROJECT_ID, model_for, recording_client

from evals.agent.loader import experiment_target
from evals.online import seed
from evals.online.pull import guard_project
from flysafelz.agent import tracing
from flysafelz.agent.input_check import RequestInvalid, validate_request
from flysafelz.agent.state import RunContext

ROOT = Path(__file__).resolve().parents[2]
SECRET = "do-not-print-this-value"
FULL_ENV = {name: f"{SECRET}-{name}" for name in seed.REQUIRED_VARIABLES}
VARIABLES = ["LANGSMITH_API_KEY", "AGENT_BASE_URL", "AGENT_MODEL", "AGENT_API_KEY", "OPENAIP_API_KEY"]


class FakeEntry:
    """Ticket 4's agent as the seed calls it. Each call records (request, context, config), and adds a root run, with a tool run below it, to the tracing callback the seed is collecting from.

    `paths` gives a question's canned root run its path by the call's number (1-based): a path string, `NO_PATH`, or a ready run. `raises` makes the call with that number raise. `collects=False` collects nothing.
    """

    def __init__(self, *, raises: dict[int, BaseException] | None = None, paths: dict[int, Any] | None = None, collects: bool = True) -> None:
        self.calls: list[tuple[Any, RunContext, dict[str, Any]]] = []
        self.raises = raises or {}
        self.paths = paths or {}
        self.collects = collects
        self.active = 0
        self.most_active = 0
        self.projects: list[str | None] = []

    def __call__(self, request: Any, context: RunContext, config: Any) -> dict[str, str]:
        self.active += 1
        self.most_active = max(self.most_active, self.active)
        try:
            number = len(self.calls) + 1
            self.calls.append((request, context, dict(config)))
            tracer = tracing_v2_callback_var.get()
            self.projects.append(None if tracer is None else tracer.project_name)  # the tracing the seed turned on for this call
            if number in self.raises:
                raise self.raises[number]
            collector = run_collector_var.get()
            assert collector is not None, "the seed must collect runs through collect_runs()"
            if self.collects:
                chosen = self.paths.get(number, "assess")
                root = chosen if hasattr(chosen, "id") else fake_run(f"q{number}", source="online-seed", path=chosen, metadata={"lz_name": config["metadata"]["lz_name"]})
                below = [tool_run("get_weather", root), tool_run("assess", root, minutes=2)] if hasattr(root, "start_time") else []  # the runs below the root are collected too
                collector.traced_runs.extend([below[0], root, *below[1:]] if below else [root])
            return {"answer": "x"}
        finally:
            self.active -= 1


def odd_run(key: str, **fields: Any) -> SimpleNamespace:
    """A collected root run that is not shaped as LangSmith's own (`Run` refuses such values): only its id and no parent are right."""
    return SimpleNamespace(id=run_id(key), parent_run_id=None, **fields)


def run_seed(*, environ=None, entry=None, client=None, questions_file=None):
    """seed.main with captured output: (exit code, stdout lines, stderr lines, entry, client)."""
    entry = FakeEntry() if entry is None else entry
    client = OnlineRecordingClient() if client is None else client
    out, err = io.StringIO(), io.StringIO()
    extra = {} if questions_file is None else {"questions_file": questions_file}
    code = seed.main([], environ=dict(FULL_ENV) if environ is None else environ, entry=entry, client=client, stdout=out, stderr=err, **extra)
    return code, out.getvalue().splitlines(), err.getvalue().splitlines(), entry, client


def written(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "seed_questions.yaml"
    path.write_text(text, encoding="utf-8")
    return path


ENTRY = "- {id: a, lz_name: Blue Mill, lat: 53.802797, lon: -0.086644, question: 'Is it on tomorrow morning?', intended_path: assess}\n"


# --- E6: the keys ---------------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("name", VARIABLES)
def test_e6_one_variable_missing_is_named_and_nothing_is_called(name):
    environ = {key: value for key, value in FULL_ENV.items() if key != name}
    code, out, err, entry, client = run_seed(environ=environ)
    assert code == 2 and out == []
    assert err[1:] == [name]  # a header, then the one name
    assert entry.calls == [] and client.calls == []


def test_e6_all_variables_missing_names_every_one_per_line_in_order():
    code, out, err, entry, client = run_seed(environ={})
    assert code == 2 and out == [] and err[1:] == VARIABLES
    assert entry.calls == [] and client.calls == []


@pytest.mark.parametrize("empty", ["", "   "])
@pytest.mark.parametrize("name", VARIABLES)
def test_e6_a_variable_set_to_an_empty_string_counts_as_missing(name, empty):
    code, _, err, entry, client = run_seed(environ={**FULL_ENV, name: empty})
    assert code == 2 and err[1:] == [name]
    assert entry.calls == [] and client.calls == []


def test_e6_a_value_is_never_printed():
    environ = {**FULL_ENV, "AGENT_MODEL": ""}
    _, out, err, _, _ = run_seed(environ=environ)
    assert SECRET not in "\n".join([*out, *err])


def test_the_variables_are_the_langsmith_key_ticket_4s_agent_settings_and_the_openaip_key():
    assert seed.REQUIRED_VARIABLES == tuple(VARIABLES)
    assert seed.missing_variables({"LANGSMITH_API_KEY": "k"}) == VARIABLES[1:]


def test_a_protected_project_exits_2_before_any_question(capsys):
    for split in ("held_out", "dev"):
        entry, client = FakeEntry(), OnlineRecordingClient()
        with pytest.raises(SystemExit) as stopped:
            seed.main([], environ={**FULL_ENV, "LANGSMITH_PROJECT": experiment_target(split).project}, entry=entry, client=client)
        assert stopped.value.code == 2
        assert entry.calls == [] and client.calls == []
    assert experiment_target("held_out").project in capsys.readouterr().err


def test_an_argument_is_refused():
    out, err = io.StringIO(), io.StringIO()
    entry = FakeEntry()
    assert seed.main(["x"], environ=dict(FULL_ENV), entry=entry, client=OnlineRecordingClient(), stdout=out, stderr=err) == 2
    assert entry.calls == [] and "takes no arguments" in err.getvalue()


# --- the file, before any key check or call -------------------------------------------------------------------------------------------


OMIT = object()


def entry_line(**changes: Any) -> str:
    """One entry of the file as a JSON flow mapping (JSON is YAML), the six fields with `changes` over them; a value of OMIT leaves that field out."""
    fields = {"id": "a", "lz_name": "Blue Mill", "lat": 53.802797, "lon": -0.086644, "question": "Is it on tomorrow morning?", "intended_path": "assess", **changes}
    return "- " + json.dumps({key: value for key, value in fields.items() if value is not OMIT}) + "\n"


def without(field: str) -> str:
    return entry_line(**{field: OMIT})


@pytest.mark.parametrize(
    ("text", "names"),
    [
        ("- id: a\n  question: [unclosed\n", "seed_questions.yaml"),
        ("question: not a list\n", "seed_questions.yaml"),
        ("[]\n", "seed_questions.yaml"),
        ("- just a string\n", "entry 1"),
        *[(without(field), f"entry 1 (a): missing {field}") for field in seed.FIELDS if field != "id"],
        (without("id"), "entry 1: missing id"),
        (entry_line() + entry_line(id="b") + entry_line(id="a"), "entry 3 (a): duplicate id a"),
        (entry_line(intended_path="explian"), "entry 1 (a): intended_path must be one of assess, explain, clarify, decline"),
        (entry_line(intended_path="Assess"), "entry 1 (a): intended_path must be one of"),
        (entry_line(lat="north"), "entry 1 (a): lat must be a number"),
        (entry_line(lon=True), "entry 1 (a): lon must be a number"),
        (entry_line(lat=91), "entry 1 (a): lat must be a number between"),
        ("- {id: a, lz_name: Blue Mill, lat: 53.8, lon: .nan, question: q, intended_path: assess}\n", "entry 1 (a): lon must be a number"),
        (entry_line(question="  "), "entry 1 (a): question must be a non-empty string"),
        (entry_line(id=5), "entry 1: id must be a non-empty string"),
        (entry_line() + entry_line(id="b", lz_name=None), "entry 2 (b): lz_name must be a non-empty string"),
    ],
)
def test_a_bad_file_stops_with_one_line_naming_the_entry_before_any_key_check_or_call(tmp_path, text, names):
    for environ in (dict(FULL_ENV), {}):  # a file that is wrong is reported first, whether or not the keys are there
        code, out, err, entry, client = run_seed(environ=environ, questions_file=written(tmp_path, text))
        assert code == 2 and out == []
        assert len(err) == 1 and names in err[0], err
        assert entry.calls == [] and client.calls == []


def test_a_missing_file_is_one_line_too(tmp_path):
    code, _, err, entry, _ = run_seed(questions_file=tmp_path / "nowhere.yaml")
    assert code == 2 and len(err) == 1 and "nowhere.yaml" in err[0] and entry.calls == []


def test_the_real_file_loads_and_runs_through_the_fake_entry():
    code, out, err, entry, _ = run_seed()
    assert code == 0 and err == [] and len(out) == 11 and len(entry.calls) == 11


# --- G2: the calls --------------------------------------------------------------------------------------------------------------------


def test_g2_the_questions_are_asked_in_file_order_one_at_a_time_each_with_its_own_zone():
    expected = seed.load_questions()
    _, _, _, entry, _ = run_seed()
    assert [call[0]["messages"][0].content for call in entry.calls] == [question.question for question in expected]
    assert [(call[0]["lat"], call[0]["lon"]) for call in entry.calls] == [(question.lat, question.lon) for question in expected]
    assert entry.most_active == 1  # never two at once
    assert len({call[2]["configurable"]["thread_id"] for call in entry.calls}) == 11  # each question has a thread of its own


def test_g2_each_call_is_live_mode_with_source_online_seed_and_the_questions_lz_name_and_nothing_else():
    expected = seed.load_questions()
    _, _, _, entry, _ = run_seed()
    for question, (request, context, config) in zip(expected, entry.calls, strict=True):
        assert context == RunContext() and context.case is None and context.now is None
        assert config["metadata"] == {"source": "online-seed", "lz_name": question.lz_name}
        assert set(request) == {"messages", "lat", "lon"}
        assert "intended_path" not in str(config) and question.intended_path not in str(config)  # f7: the label stays in the file


def test_g2_the_context_passes_ticket_4s_input_check_and_reader_live_does_not_without_a_case():
    question = seed.load_questions()[0]
    update = validate_request(seed.request_for(question), RunContext())
    assert update["answer"] is None and "jump_to" not in update  # a UK point in live mode goes on to the model
    with pytest.raises(RequestInvalid, match="only applies in case mode"):
        validate_request(seed.request_for(question), RunContext(reader="live"))


def test_g2_tracing_is_turned_on_for_the_call_in_the_default_project_and_the_run_is_collected_through_the_callback():
    _, _, _, entry, client = run_seed()
    assert entry.projects == ["flysafelz"] * 11
    assert {call["project_name"] for method, call in client.calls if method == "get_run_url"} == {"flysafelz"}


def test_g2_the_project_follows_langsmith_project():
    _, _, _, entry, client = run_seed(environ={**FULL_ENV, "LANGSMITH_PROJECT": "  my-seed-project "})
    assert entry.projects == ["my-seed-project"] * 11
    assert {call["project_name"] for method, call in client.calls if method == "get_run_url"} == {"my-seed-project"}
    assert guard_project(environ={"LANGSMITH_PROJECT": ""}) == "flysafelz"  # the default is the one the pull uses


def test_g2_tracing_is_off_again_after_the_command():
    run_seed()
    assert tracing_v2_callback_var.get() is None and run_collector_var.get() is None


def test_g2_the_client_is_flushed_after_every_question_and_once_more_at_the_end():
    _, _, _, _, client = run_seed()
    order = client.methods_called()
    assert order.count("flush") == 12 and order[-1] == "flush"
    assert order[:2] == ["flush", "get_run_url"]  # the run is sent before its URL is asked for


# --- the output -----------------------------------------------------------------------------------------------------------------------


def test_the_output_is_one_line_per_question_with_its_id_its_path_and_the_url_from_the_client():
    expected = seed.load_questions()
    code, out, err, entry, client = run_seed()
    assert code == 0 and err == [] and len(out) == 11
    urls = [call["run_id"] for method, call in client.calls if method == "get_run_url"]
    for question, line, asked_run in zip(expected, out, urls, strict=True):
        question_id, path, url = line.split(" ")
        assert question_id == question.id and path == "assess" and url == f"https://smith.example/o/org/projects/p/flysafelz/r/{asked_run}"
    assert len(entry.calls) == 11


def test_the_path_is_read_from_the_metadata_then_the_outputs_and_each_of_the_four_comes_through():
    paths = {1: "assess", 2: "explain", 3: "clarify", 4: "decline", 5: "Decline"}
    entry = FakeEntry(paths=paths)
    _, out, _, _, _ = run_seed(entry=entry)
    assert [line.split(" ")[1] for line in out[:5]] == ["assess", "explain", "clarify", "decline", "decline"]
    from_outputs = fake_run("outputs-only", source="online-seed", path=NO_PATH, output_path="clarify")
    _, out, _, _, _ = run_seed(entry=FakeEntry(paths={1: from_outputs}))
    assert out[0].split(" ")[1] == "clarify"


@pytest.mark.parametrize(
    "chosen",
    [NO_PATH, "banana", "", odd_run("odd", extra="not a mapping", outputs=None), odd_run("odder", extra={"metadata": ["x", 7]}, outputs={"path": ["assess"]}), odd_run("bare")],
    ids=["none", "not a path", "empty", "extra is not a mapping", "path of the wrong type", "no fields at all"],
)
def test_a_run_whose_path_cannot_be_read_prints_unknown_not_a_crash(chosen):
    code, out, err, _, _ = run_seed(entry=FakeEntry(paths={3: chosen}))
    assert code == 0 and err == [] and len(out) == 11
    assert out[2].split(" ")[1] == "unknown"


# --- failures -------------------------------------------------------------------------------------------------------------------------


def test_one_question_raising_is_named_by_id_and_error_type_the_rest_run_and_the_exit_is_non_zero():
    expected = seed.load_questions()
    entry = FakeEntry(raises={3: ValueError(f"the message holds {SECRET}")})
    code, out, err, _, _ = run_seed(entry=entry)
    assert code == 1
    assert len(entry.calls) == 11 and len(out) == 10
    assert err[0] == f"{expected[2].id}: failed: ValueError"
    assert expected[2].id not in "\n".join(out)
    assert [line.split(" ")[0] for line in out] == [question.id for index, question in enumerate(expected) if index != 2]
    assert err[-1] == "1 of 11 questions did not finish cleanly."
    assert SECRET not in "\n".join(err)  # the type, never the message


def test_every_question_raising_prints_no_lines_and_names_each():
    entry = FakeEntry(raises={n: RuntimeError("x") for n in range(1, 12)})
    code, out, err, _, _ = run_seed(entry=entry)
    assert code == 1 and out == [] and len(entry.calls) == 11
    assert [line.split(":")[0] for line in err[:11]] == [question.id for question in seed.load_questions()]


def test_a_question_that_leaves_no_root_run_is_named_and_the_rest_still_run():
    code, out, err, entry, _ = run_seed(entry=FakeEntry(collects=False))
    assert code == 1 and out == [] and len(entry.calls) == 11
    assert err[0].endswith(": failed: LookupError")


def test_a_question_that_leaves_two_root_runs_is_a_failure_too():
    class Two(FakeEntry):
        def __call__(self, request, context, config):
            super().__call__(request, context, config)
            run_collector_var.get().traced_runs.append(fake_run("second-root"))
            return {}

    code, out, err, _, _ = run_seed(entry=Two())
    assert code == 1 and out == [] and err[0].endswith(": failed: LookupError")


def test_a_client_that_cannot_give_a_url_still_prints_the_line_and_the_exit_is_non_zero():
    client = OnlineRecordingClient()
    client.fail_run_url = True
    code, out, err, entry, _ = run_seed(client=client)
    assert code == 1 and len(entry.calls) == 11 and len(out) == 11
    assert all(line.endswith(" unavailable (RuntimeError)") for line in out)
    assert err[0].endswith(": no run URL: RuntimeError")


def test_a_keyboard_interrupt_is_not_swallowed():
    with pytest.raises(KeyboardInterrupt):
        run_seed(entry=FakeEntry(raises={2: KeyboardInterrupt()}))


# --- the default entry point ------------------------------------------------------------------------------------------------------------


def test_the_default_entry_point_builds_ticket_4s_graph_once_at_the_first_question_and_invokes_it_with_the_context(monkeypatch):
    built, invoked = [], []

    class Graph:
        def invoke(self, request, config, *, context):
            invoked.append((request, config, context))
            return {"answer": "ok"}

    monkeypatch.setattr(seed, "build_graph", lambda: built.append(1) or Graph())
    entry = seed.default_entry()
    assert built == []  # nothing is built until a question is asked
    context = RunContext()
    assert entry({"messages": []}, context, {"metadata": {"a": 1}}) == {"answer": "ok"}
    entry({"messages": []}, context, {"metadata": {"a": 2}})
    assert built == [1] and [call[1]["metadata"]["a"] for call in invoked] == [1, 2] and invoked[0][2] is context


def test_a_seed_that_stops_at_the_file_or_the_keys_builds_no_graph(monkeypatch, tmp_path):
    monkeypatch.setattr(seed, "build_graph", lambda: pytest.fail("the graph must not be built"))
    seed.main([], environ={}, client=OnlineRecordingClient(), stdout=io.StringIO(), stderr=io.StringIO())
    seed.main([], environ=dict(FULL_ENV), client=OnlineRecordingClient(), stdout=io.StringIO(), stderr=io.StringIO(), questions_file=written(tmp_path, "[]\n"))


# --- the real graph and the real client -------------------------------------------------------------------------------------------------

OUTSIDE_THE_UK = (
    "- {id: far, lz_name: Paris, lat: 48.8566, lon: 2.3522, question: 'Can I fly tomorrow morning?', intended_path: assess}\n"  # answered by code, so no model and no live source is called
)


def test_the_real_graph_through_the_seed_makes_one_trace_tagged_online_seed_with_the_lz_name_and_a_url_from_the_real_client(tmp_path):
    client, recorder = recording_client()
    graph = tracing.traced(agent_graph(model_for()), client)  # the real graph with the agent's own tracer on it, over a model scripted to say nothing: it is never called

    def entry(request, context, config):
        return graph.invoke(request, config, context=context)

    code, out, err, _, _ = run_seed(entry=entry, client=client, questions_file=written(tmp_path, OUTSIDE_THE_UK))
    assert (code, err) == (0, [])
    (root,) = recorder.root_runs()  # the seed's own tracing did not add a second trace beside the agent's
    assert recorder.metadata(root)["source"] == "online-seed" and recorder.metadata(root)["lz_name"] == "Paris"
    assert "intended_path" not in recorder.text()
    (line,) = out
    question_id, path, url = line.split(" ")
    assert (question_id, path) == ("far", "unknown")  # a request outside the UK never reaches the guard, so it has no path
    assert f"/projects/p/{PROJECT_ID}/r/{root['id']}" in url  # the real client's own URL: the project and the run it just traced


def test_every_question_in_the_real_file_passes_ticket_4s_input_check_as_the_seed_sends_it():
    """The seed's request and context are accepted by the agent: a UK point goes on to the model rather than raising RequestInvalid."""
    for question in seed.load_questions():
        assert validate_request(seed.request_for(question), RunContext())["answer"] is None
