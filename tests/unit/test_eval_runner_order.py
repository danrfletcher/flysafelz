"""The runner's order (5a PR-4.F1, G6): rep 1 to the end, then rep 2, then rep 3; one graph call at a time; every run tagged with its rep; shared cases."""

import inspect

import pytest
from eval_runner_helpers import RunRecordingClient, ScriptedGraph, make_records, run

from evals.agent import run_v1, runner
from evals.agent.loader import example_uuid


def test_rep_1_runs_for_every_example_before_rep_2_and_rep_2_before_rep_3(tmp_path):
    records = make_records(5)
    graph = ScriptedGraph()
    run(records, tmp_path, graph)
    ids = [r["id"] for r in records]
    assert graph.order() == [(i, rep) for rep in (1, 2, 3) for i in ids]


def test_the_records_come_back_in_run_order_tagged_with_their_rep(tmp_path):
    records = make_records(4)
    result = run(records, tmp_path, ScriptedGraph())
    assert [(r.example_id, r.rep) for r in result.records] == [(rec["id"], rep) for rep in (1, 2, 3) for rec in records]
    assert {r.rep for r in result.records} == {1, 2, 3}
    assert result.planned_runs == 12 and result.reps == 3


def test_at_most_one_graph_call_is_in_flight_and_each_ends_before_the_next_starts(tmp_path):
    graph = ScriptedGraph()
    run(make_records(6), tmp_path, graph)
    assert graph.peak == 1
    assert graph.events == [(kind, n) for n in range(18) for kind in ("start", "end")]


def test_a_failed_attempt_ends_before_its_retry_starts(tmp_path):
    graph = ScriptedGraph(script={"inv_dev_01": [RuntimeError("one"), RuntimeError("two")]})
    run(make_records(2), tmp_path, graph, reps=1)
    assert graph.peak == 1
    assert [e[0] for e in graph.events] == ["start", "end"] * 4


def test_reps_are_not_langsmith_repetitions_each_run_is_logged_once_tagged_with_its_rep(tmp_path):
    records = make_records(3)
    client = RunRecordingClient()
    client.create_dataset(dataset_name="flysafelz-dev")
    result = run(records, tmp_path, ScriptedGraph(), client=client)
    assert len(client.projects) == 1 and client.projects[0]["project_name"] == "v1-dev-test"
    assert client.projects[0]["reference_dataset_id"] == client.datasets["flysafelz-dev"].id
    assert len(client.runs) == 9
    assert {r["project_name"] for r in client.runs} == {"v1-dev-test"}
    for logged, record in zip(client.runs, result.records, strict=True):
        assert logged["tags"] == [f"rep:{record.rep}", "v1"]
        assert logged["extra"]["metadata"]["rep"] == record.rep
        assert logged["reference_example_id"] == example_uuid(record.example_id)
        assert logged["id"] == record.langsmith_run_id
        assert logged["start_time"] <= logged["end_time"]
    assert [r["extra"]["metadata"]["rep"] for r in client.runs] == [1, 1, 1, 2, 2, 2, 3, 3, 3]


def test_no_langsmith_call_is_made_without_a_client(tmp_path):
    result = run(make_records(1), tmp_path, ScriptedGraph(), reps=1)
    assert result.records[0].langsmith_run_id is None


# --- shared cases ------------------------------------------------------------------------------------------------------


def test_shared_case_groups_lists_the_examples_that_share_a_case_in_first_seen_order():
    records = make_records(7, cases=3)  # cases by i % 3: 1,4,7 -> 1; 2,5 -> 2; 3,6 -> 0
    assert runner.shared_case_groups(records) == [["inv_dev_01", "inv_dev_04", "inv_dev_07"], ["inv_dev_02", "inv_dev_05"], ["inv_dev_03", "inv_dev_06"]]


def test_an_example_alone_on_its_case_is_in_no_group():
    records = make_records(4, cases=10)
    assert runner.shared_case_groups(records) == []
    records[3]["fixtures"]["case"] = records[0]["fixtures"]["case"]
    assert runner.shared_case_groups(records) == [["inv_dev_01", "inv_dev_04"]]


def test_the_result_carries_the_groups_and_says_so_in_its_summary(tmp_path):
    result = run(make_records(4, cases=2), tmp_path, ScriptedGraph(), reps=1)
    assert result.shared_case_groups == [["inv_dev_01", "inv_dev_03"], ["inv_dev_02", "inv_dev_04"]]
    assert any("sharing a case" in line for line in result.summary_lines())


def test_all_reps_and_all_examples_use_one_cache_path(tmp_path):
    graph = ScriptedGraph()
    run(make_records(3), tmp_path, graph)
    assert {c.context.reader_cache for c in graph.calls} == {tmp_path / "v1-dev-test.sqlite"}


# --- the split ---------------------------------------------------------------------------------------------------------


def test_the_default_split_is_dev_everywhere_in_the_runner():
    assert inspect.signature(run_v1.run_split).parameters["split"].default == "dev"
    assert inspect.signature(runner.run_experiment).parameters["split"].default == "dev"
    assert inspect.signature(runner.default_experiment_name).parameters["split"].default == "dev"
    assert run_v1._parser().parse_args([]).split == "dev"


def test_held_out_is_reached_only_by_naming_it():
    assert run_v1._parser().parse_args(["--split", "held_out"]).split == "held_out"
    for bad in ("held-out", "heldout", "all", "HELD_OUT"):
        with pytest.raises(SystemExit):
            run_v1._parser().parse_args(["--split", bad])
