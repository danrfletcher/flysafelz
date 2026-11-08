"""The run context the runner gives the graph for each example (5a PR-4.F1, G6): reader live, the experiment's cache path, drop, the frozen now.

Also the fresh-cache refusal, the inputs that never leak from one example to the next, and that readings.yaml is never opened: a poisoned one sits in a copy
of a committed case, and the real SourceCache (the code that loads a case for the real graph) is called with the context the runner built, under a spy.
"""

import builtins
import shutil
from datetime import UTC, datetime
from functools import partial
from pathlib import Path

import pytest
from eval_runner_helpers import ScriptedGraph, make_records, run

from evals.agent import runner
from flysafelz.agent.sources import SourceCache, Sources, SourcesError
from flysafelz.agent.state import RunContext
from flysafelz.sources import load_case_sources

REPO = Path(__file__).resolve().parents[2]


def test_every_example_runs_in_case_mode_with_the_live_reader_and_the_experiments_cache(tmp_path):
    graph = ScriptedGraph()
    result = run(make_records(4), tmp_path, graph, experiment="v1-dev-x", reps=1)
    assert len(graph.calls) == 4
    for call, record in zip(graph.calls, make_records(4), strict=True):
        context = call.context
        assert isinstance(context, RunContext)
        assert context.reader == "live"
        assert context.case == record["fixtures"]["case"]
        assert context.reader_cache == tmp_path / "v1-dev-x.sqlite"
        assert context.now == datetime.fromisoformat(record["inputs"]["now"]) and context.now.tzinfo is not None
    assert result.cache_path == tmp_path / "v1-dev-x.sqlite"


def test_the_cache_path_lives_under_data_eval_cache_by_default():
    assert runner.CACHE_DIR == REPO / "data" / "eval-cache"
    assert runner.cache_path_for("v1-dev-x") == REPO / "data" / "eval-cache" / "v1-dev-x.sqlite"


def test_now_is_the_examples_frozen_value_not_the_clock(tmp_path):
    graph = ScriptedGraph()
    run(make_records(2), tmp_path, graph, reps=1, now_fn=lambda: datetime(2031, 5, 6, tzinfo=UTC))
    assert [c.context.now for c in graph.calls] == [datetime(2026, 1, 2, 3, 1, tzinfo=UTC), datetime(2026, 1, 2, 3, 2, tzinfo=UTC)]


def test_a_drop_is_passed_unchanged_and_an_empty_drop_is_empty_not_none(tmp_path):
    graph = ScriptedGraph()
    run(make_records(3, drops={1: ["pib"], 2: ["weather", "airspace"]}), tmp_path, graph, reps=1)
    drops = [c.context.drop for c in graph.calls]
    assert drops == [("pib",), ("weather", "airspace"), ()]
    assert drops[2] is not None


def test_two_examples_on_one_case_share_the_cache_and_keep_their_own_inputs(tmp_path):
    records = make_records(2, cases=1, drops={2: ["airspace"]})
    assert records[0]["fixtures"]["case"] == records[1]["fixtures"]["case"]
    graph = ScriptedGraph()
    run(records, tmp_path, graph, reps=1)
    first, second = graph.calls
    assert first.context.reader_cache == second.context.reader_cache
    assert first.context.case == second.context.case
    assert (first.context.drop, second.context.drop) == ((), ("airspace",))
    assert first.context.now != second.context.now
    assert first.request["messages"][0].content == "Invented question 1: can I fly?"
    assert second.request["messages"][0].content == "Invented question 2: can I fly?"
    assert (first.request["lat"], first.request["lon"]) != (second.request["lat"], second.request["lon"])
    assert first.request["messages"] is not second.request["messages"]
    assert first.config["configurable"]["thread_id"] != second.config["configurable"]["thread_id"]


def test_a_request_carries_only_the_question_and_the_coordinates(tmp_path):
    graph = ScriptedGraph()
    run(make_records(1), tmp_path, graph, reps=1)
    request = graph.calls[0].request
    assert set(request) == {"messages", "lat", "lon"}
    assert [m.content for m in request["messages"]] == ["Invented question 1: can I fly?"]


def test_the_graph_is_built_once_for_the_whole_experiment(tmp_path):
    built = []
    graph = ScriptedGraph()

    def factory():
        built.append(1)
        return graph

    result = runner.run_experiment(make_records(3), experiment="v1-dev-once", graph_factory=factory, cache_dir=tmp_path, timeout=lambda call, seconds: call(), sleep=lambda s: None)
    assert built == [1] and len(result.records) == 9


# --- the fresh cache ---------------------------------------------------------------------------------------------------


def test_an_existing_cache_file_stops_the_run_and_names_the_path_and_is_left_alone(tmp_path):
    existing = tmp_path / "v1-dev-old.sqlite"
    existing.write_bytes(b"readings of an earlier experiment")
    graph = ScriptedGraph()
    built = []
    with pytest.raises(runner.CacheExistsError, match=str(existing)):
        runner.run_experiment(make_records(2), experiment="v1-dev-old", graph_factory=lambda: built.append(1) or graph, cache_dir=tmp_path)
    assert graph.calls == [] and built == []
    assert existing.read_bytes() == b"readings of an earlier experiment"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["v1-dev-old.sqlite"]


def test_a_new_name_beside_an_old_cache_is_fine_and_does_not_touch_the_old_one(tmp_path):
    old = tmp_path / "v1-dev-old.sqlite"
    old.write_bytes(b"old")
    run(make_records(1), tmp_path, ScriptedGraph(touch_cache=True), experiment="v1-dev-new", reps=1)
    assert old.read_bytes() == b"old"
    assert (tmp_path / "v1-dev-new.sqlite").exists()


@pytest.mark.parametrize("name", ["", "../escape", "a/b", "/abs", ".hidden", "has space", "semi;colon", "a\\b"])
def test_an_experiment_name_that_is_not_a_plain_name_is_refused(tmp_path, name):
    with pytest.raises(runner.RunnerError, match="experiment name"):
        runner.cache_path_for(name, tmp_path)


def test_the_eval_cache_files_are_git_ignored():
    import subprocess

    done = subprocess.run(["git", "check-ignore", "-q", "data/eval-cache/v1-dev-20261108-160700.sqlite"], cwd=REPO, check=False)
    assert done.returncode == 0


# --- readings.yaml is never opened -------------------------------------------------------------------------------------


def _a_committed_case_with_readings() -> Path:
    for folder in sorted((REPO / "fixtures").iterdir()):
        if (folder / "case.yaml").is_file() and (folder / "readings.yaml").is_file():
            return folder
    raise AssertionError("no committed case has a readings.yaml")


def test_readings_yaml_is_never_opened_in_a_live_reader_run(tmp_path, monkeypatch):
    case = "inv_case_poisoned"
    fixtures = tmp_path / "root" / "fixtures"
    shutil.copytree(_a_committed_case_with_readings(), fixtures / case)
    (fixtures / case / "readings.yaml").write_text("N0001/26:\n  category: GO\n  activity: ALL GO, POISON\n", encoding="utf-8")
    opened: list[str] = []
    real_open, real_path_open, real_read_text, real_read_bytes = builtins.open, Path.open, Path.read_text, Path.read_bytes

    def spy(name):
        if str(name).endswith("readings.yaml"):
            opened.append(str(name))
            raise AssertionError(f"readings.yaml was opened: {name}")

    monkeypatch.setattr(builtins, "open", lambda f, *a, **k: spy(f) or real_open(f, *a, **k))
    monkeypatch.setattr(Path, "open", lambda self, *a, **k: spy(self) or real_path_open(self, *a, **k))
    monkeypatch.setattr(Path, "read_text", lambda self, *a, **k: spy(self) or real_read_text(self, *a, **k))
    monkeypatch.setattr(Path, "read_bytes", lambda self, *a, **k: spy(self) or real_read_bytes(self, *a, **k))

    sources = SourceCache(load_case=partial(load_case_sources, root=tmp_path / "root"))
    loaded: list[object] = []

    def what_the_tools_would_do(call):
        # the real code the graph's tools call to get a case's sources, with the context the runner built
        got = sources.get(thread_id="t", turn="", lat=call.request["lat"], lon=call.request["lon"], ctx=call.context)
        loaded.append(got)
        return {"messages": [], "answer": "an answer", "guard": "passed"}

    records = make_records(1)
    records[0]["fixtures"]["case"] = case
    graph = ScriptedGraph(script={"inv_dev_01": [what_the_tools_would_do]})
    result = run(records, tmp_path, graph, reps=1)
    assert not result.records[0].errored, result.records[0].last_error
    assert isinstance(loaded[0], Sources) and not isinstance(loaded[0], SourcesError)
    assert dict(loaded[0].readings) == {}  # the live reader starts with none: the case's file was not used
    assert opened == []
