"""The command `python -m evals.agent.run_v1` through `main` (5a PR-4.F1, GP2, GP9, F3, F7), on synthetic datasets, a fake graph and a recording client.

No key, no model and no network: the settings are fake values, the graph is scripted and the LangSmith client records. Sockets stay off.
"""

import io

import pytest
from eval_loader_factories import make_examples, write_dataset
from eval_runner_helpers import FAKE_ENV, FakeSleeper, RecordingTimeout, RunRecordingClient, ScriptedGraph

from evals.agent import run_v1, runner


class Boom(AssertionError):
    pass


def never(*args, **kwargs):
    raise Boom("this must not be called")


def invoke(built, tmp_path, *argv, environ=FAKE_ENV, graph=None, client=None, **options):
    """main() on the synthetic dataset; returns (exit code, stdout, stderr, graph, client, cache dir)."""
    graph = ScriptedGraph() if graph is None else graph
    client = RunRecordingClient() if client is None else client
    out, err = io.StringIO(), io.StringIO()
    cache_dir = tmp_path / "cache"
    code = run_v1.main(
        list(argv),
        environ=environ,
        stdout=out,
        stderr=err,
        graph_factory=graph.factory,
        client_factory=lambda env: client,
        data_dir=built.data_dir,
        root=built.root,
        cache_dir=cache_dir,
        sleep=FakeSleeper(),
        timeout=RecordingTimeout(),
        **options,
    )
    return code, out.getvalue(), err.getvalue(), graph, client, cache_dir


def test_a_run_of_the_default_split_runs_the_dev_examples_three_reps_and_exits_0(tmp_path):
    built = write_dataset(tmp_path)
    code, out, err, graph, client, cache_dir = invoke(built, tmp_path)
    assert (code, err) == (0, "")
    dev = [e["id"] for e in built.examples if e["metadata"]["split"] == "dev"]
    assert len(dev) == 29
    assert graph.order() == [(i, rep) for rep in (1, 2, 3) for i in dev]
    assert "87 of 87 runs" in out and "errored: 0 of 87" in out and "incomplete: false" in out
    assert str(cache_dir) in out
    assert list(client.datasets) == ["flysafelz-dev"] and len(client.projects) == 1 and len(client.runs) == 87


def test_no_held_out_example_runs_or_is_uploaded_unless_the_split_is_named(tmp_path):
    built = write_dataset(tmp_path)
    _, _, _, graph, client, _ = invoke(built, tmp_path)
    assert not any("_ho_" in i for i, _ in graph.order())
    assert {e.metadata["split"] for e in client.datasets["flysafelz-dev"].examples.values()} == {"dev"}
    assert "flysafelz-held-out" not in client.datasets


def test_held_out_runs_only_when_named_and_goes_to_its_own_dataset(tmp_path):
    built = write_dataset(tmp_path)
    code, _, _, graph, client, _ = invoke(built, tmp_path, "--split", "held_out")
    assert code == 0
    assert {i for i, _ in graph.order()} == {e["id"] for e in built.examples if e["metadata"]["split"] == "held_out"}
    assert list(client.datasets) == ["flysafelz-held-out"]
    assert client.projects[0]["metadata"]["split"] == "held_out" and client.projects[0]["project_name"].startswith("v1-held_out-")


def test_the_experiment_is_named_from_the_split_and_the_stamp_or_by_the_caller(tmp_path):
    built = write_dataset(tmp_path)
    _, out, _, _, client, _ = invoke(built, tmp_path)
    assert client.projects[0]["project_name"].startswith("v1-dev-") and f"experiment {client.projects[0]['project_name']}" in out
    other = tmp_path / "again"
    other.mkdir()
    _, _, _, _, client, _ = invoke(write_dataset(other), other, "--experiment", "v1-dev-chosen")
    assert client.projects[0]["project_name"] == "v1-dev-chosen"


# --- F3: no settings, no client ----------------------------------------------------------------------------------------


def test_with_no_settings_it_names_every_missing_variable_and_builds_nothing(tmp_path):
    built = write_dataset(tmp_path)
    out, err = io.StringIO(), io.StringIO()
    code = run_v1.main([], environ={}, stdout=out, stderr=err, graph_factory=never, client_factory=never, data_dir=built.data_dir, root=built.root, cache_dir=tmp_path / "cache")
    assert code == 1 and out.getvalue() == ""
    message = err.getvalue()
    assert message.count("\n") == 1
    for name in ("AGENT_BASE_URL", "AGENT_MODEL", "AGENT_API_KEY", "READER_PROVIDER", "READER_BASE_URL", "READER_API_KEY", "READER_MODEL", "LANGSMITH_API_KEY"):
        assert name in message
    assert "nothing was built, sent or written" in message
    assert "uv run --env-file .env python -m evals.agent.run_v1" in message and "or in .env" not in message  # the agent reads AGENT_* from the environment only
    assert not (tmp_path / "cache").exists()


def test_only_the_variables_that_are_missing_are_named(tmp_path):
    built = write_dataset(tmp_path)
    environ = {k: v for k, v in FAKE_ENV.items() if k not in ("AGENT_MODEL", "LANGSMITH_API_KEY")}
    err = io.StringIO()
    code = run_v1.main([], environ=environ, stderr=err, graph_factory=never, client_factory=never, data_dir=built.data_dir, root=built.root, cache_dir=tmp_path / "c")
    assert code == 1
    assert "AGENT_MODEL" in err.getvalue() and "LANGSMITH_API_KEY" in err.getvalue()
    assert "AGENT_API_KEY" not in err.getvalue() and "READER_" not in err.getvalue()


def test_an_agent_url_that_is_not_a_url_is_named(tmp_path):
    built = write_dataset(tmp_path)
    err = io.StringIO()
    run_v1.main([], environ={**FAKE_ENV, "AGENT_BASE_URL": "not a url"}, stderr=err, graph_factory=never, client_factory=never, data_dir=built.data_dir, root=built.root, cache_dir=tmp_path / "c")
    assert "AGENT_BASE_URL (must be an http:// or https:// URL)" in err.getvalue()


def test_blank_settings_count_as_missing():
    assert run_v1.missing_settings({**FAKE_ENV, "AGENT_API_KEY": "  ", "LANGSMITH_API_KEY": ""}) == ["AGENT_API_KEY", "LANGSMITH_API_KEY"]
    assert run_v1.missing_settings(FAKE_ENV) == []


def test_the_default_client_factory_refuses_without_a_key_before_building_a_client():
    from evals.agent.import_dataset import DatasetError

    with pytest.raises(DatasetError, match="LANGSMITH_API_KEY"):
        run_v1.default_client_factory({})


def test_the_default_client_factory_builds_the_anonymising_client_with_the_environment_it_is_given(monkeypatch):
    seen = []
    monkeypatch.setattr("flysafelz.agent.tracing.build_client", lambda env: seen.append(dict(env)) or "client")
    assert run_v1.default_client_factory({"LANGSMITH_API_KEY": "fake-key"}) == "client"
    assert seen == [{"LANGSMITH_API_KEY": "fake-key"}]


# --- GP2: the loader first ---------------------------------------------------------------------------------------------


def test_a_missing_fixture_stops_the_run_before_any_client_graph_call_or_cache_file(tmp_path):
    built = write_dataset(tmp_path, no_folder=frozenset({"inv_case_1", "inv_case_3"}))
    out, err = io.StringIO(), io.StringIO()
    code = run_v1.main([], environ=FAKE_ENV, stdout=out, stderr=err, graph_factory=never, client_factory=never, data_dir=built.data_dir, root=built.root, cache_dir=tmp_path / "cache")
    assert code == 1 and out.getvalue() == ""
    assert "2 fixture folder(s) missing" in err.getvalue() and "inv_case_1" in err.getvalue() and "inv_case_3" in err.getvalue()
    assert not (tmp_path / "cache").exists()


def test_wrong_counts_stop_the_run_before_any_client_or_graph(tmp_path):
    built = write_dataset(tmp_path, make_examples(dev=28))
    err = io.StringIO()
    code = run_v1.main([], environ=FAKE_ENV, stderr=err, graph_factory=never, client_factory=never, data_dir=built.data_dir, root=built.root, cache_dir=tmp_path / "cache")
    assert code == 1 and "dataset counts are wrong" in err.getvalue()


def test_a_stale_sheet_stops_the_run_with_the_loaders_own_message(tmp_path):
    built = write_dataset(tmp_path)
    (built.root / "fixtures" / "inv_case_0" / "weather.json").write_bytes(b'{"changed": true}')
    err = io.StringIO()
    code = run_v1.main([], environ=FAKE_ENV, stderr=err, graph_factory=never, client_factory=never, data_dir=built.data_dir, root=built.root, cache_dir=tmp_path / "cache")
    assert code == 1 and "stale" in err.getvalue() and "make dataset" in err.getvalue()
    assert not (tmp_path / "cache").exists()


def test_the_examples_a_sheet_entry_refuses_are_not_run_and_are_named(tmp_path, capsys):
    examples = make_examples()
    built = write_dataset(tmp_path, examples, not_done=frozenset({examples[0]["id"]}))
    code, _, _, graph, _, _ = invoke(built, tmp_path)
    assert code == 0 and f"refused {examples[0]['id']}" in capsys.readouterr().out  # the loader's upload names it
    assert examples[0]["id"] not in {i for i, _ in graph.order()}


def test_an_existing_cache_file_stops_the_run_naming_it_before_any_client_or_graph(tmp_path):
    built = write_dataset(tmp_path)
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    (cache_dir / "v1-dev-taken.sqlite").write_bytes(b"x")
    err = io.StringIO()
    code = run_v1.main(["--experiment", "v1-dev-taken"], environ=FAKE_ENV, stderr=err, graph_factory=never, client_factory=never, data_dir=built.data_dir, root=built.root, cache_dir=cache_dir)
    assert code == 1 and str(cache_dir / "v1-dev-taken.sqlite") in err.getvalue()
    assert (cache_dir / "v1-dev-taken.sqlite").read_bytes() == b"x"


# --- GP9: a stop is one line -------------------------------------------------------------------------------------------


def test_a_fatal_stop_exits_1_with_one_line_giving_the_cause_and_the_last_example(tmp_path):
    built = write_dataset(tmp_path)
    graph = ScriptedGraph(usage="none")
    code, out, err, graph, _client, _ = invoke(built, tmp_path, graph=graph)
    assert code == 1 and out == ""
    assert err.count("\n") == 1 and err.startswith("evals.agent.run_v1: stopped: ")
    assert "AGENT_BASE_URL" in err and err.rstrip().endswith("(last example: inv_dev_01)")
    assert len(graph.calls) == 1


def test_a_langsmith_failure_at_upload_exits_1_with_one_line(tmp_path):
    built = write_dataset(tmp_path)
    code, _, err, graph, _, _ = invoke(built, tmp_path, client=RunRecordingClient(fail_on={"create_dataset": 1}))
    assert code == 1 and err.count("\n") == 1 and "LangSmith error while creating the dataset" in err
    assert graph.calls == []


def test_a_langsmith_failure_while_logging_exits_1_naming_the_last_example(tmp_path):
    built = write_dataset(tmp_path)
    code, _, err, graph, _, _ = invoke(built, tmp_path, client=RunRecordingClient(fail_on={"create_run": 2}))
    assert code == 1 and err.count("\n") == 1 and "LangSmith could not log the run" in err and err.rstrip().endswith("(last example: inv_dev_02)")
    assert len(graph.calls) == 2


def test_an_interrupt_exits_130_with_the_summary_and_no_error(tmp_path):
    built = write_dataset(tmp_path)
    graph = ScriptedGraph(script={("inv_dev_04", 1): [KeyboardInterrupt()]}, touch_cache=True)
    code, out, err, _, _, cache_dir = invoke(built, tmp_path, graph=graph)
    assert code == 130 and err == ""
    assert "INTERRUPTED" in out and "errored: 0 of" in out
    assert len(list(cache_dir.glob("*.sqlite"))) == 1


def test_run_split_returns_the_result_for_the_scorers(tmp_path):
    built = write_dataset(tmp_path)
    graph = ScriptedGraph()
    result = run_v1.run_split(
        environ=FAKE_ENV,
        graph_factory=graph.factory,
        client_factory=lambda env: RunRecordingClient(),
        data_dir=built.data_dir,
        root=built.root,
        cache_dir=tmp_path / "cache",
        sleep=FakeSleeper(),
        timeout=RecordingTimeout(),
        experiment="v1-dev-scored",
    )
    assert isinstance(result, runner.ExperimentResult) and result.split == "dev" and len(result.records) == 87 and result.cache_path == tmp_path / "cache" / "v1-dev-scored.sqlite"


def test_the_default_graph_factory_builds_the_real_graph_lazily(monkeypatch):
    built = []
    monkeypatch.setattr("flysafelz.agent.graph.build_graph", lambda: built.append(1) or "graph")
    assert run_v1.default_graph_factory() == "graph" and built == [1]
