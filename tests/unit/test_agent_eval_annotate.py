"""`make annotate` (5a PR-8.F1, G13, GP6, E5, E6): which 39 dev answers are queued, the criteria shown per path, the refusals, and idempotence.

The experiment is a fake 3-rep v1 dev experiment over the committed dev examples (16 assess, 5 explain, 3 clarify, 5 decline); the client is a recording stand-in that holds
the queue. Nothing real is called and no key is read.
"""

import io
import uuid

import pytest
from calibration_helpers import CalibrationClient
from factories import fake_experiment, meta

from evals.agent import annotate, judge
from evals.agent.annotate import CalibrationError, calibration_project, queue_name, select
from evals.agent.import_dataset import DatasetError
from evals.agent.loader import example_uuid, load_dataset

EXPERIMENT = "v1-dev-20261108-160700"
DECLINES = ("dev_08_bamburgh", "dev_09_west_sleekburn", "dev_18_blue_mill", "dev_19_buttermere", "dev_28_crookham")
COUNT_LINE = "39 answers (29 rep 1 + 10 decline from reps 2–3), 96 criterion ticks"


@pytest.fixture(scope="module")
def dev_records():
    return load_dataset("dev").records


def annotate_main(client, experiment=EXPERIMENT, records=None, **options):
    out, err = io.StringIO(), io.StringIO()
    environ = {} if experiment is None else {"EXPERIMENT": experiment}
    code = annotate.main([], environ=environ, client=client, records=records, stdout=out, stderr=err, **options)
    return code, out.getvalue(), err.getvalue()


# --- the selection -------------------------------------------------------------------------------------------------------------


def test_the_selection_is_39_answers_29_from_rep_1_and_10_decline_answers_from_reps_2_and_3(dev_records):
    experiment = fake_experiment()
    assert len(experiment.runs) == 87  # 29 examples x 3 reps: the runner runs every example in every rep
    selection = select(experiment.runs, dev_records)
    assert len(selection.items) == 39 and selection.errored == [] and selection.reps == (1, 2, 3)
    assert [item.example_id for item in selection.rep_one] == [r["id"] for r in dev_records]
    assert sorted((item.example_id, item.rep) for item in selection.extra_declines) == sorted((d, rep) for d in DECLINES for rep in (2, 3))
    assert {item.run_id for item in selection.items} == {str(experiment.run(item.example_id, item.rep).id) for item in selection.items}


def test_the_dev_set_is_16_assess_5_explain_3_clarify_5_decline(dev_records):
    paths = [record["reference"]["path"] for record in dev_records]
    assert (paths.count("assess"), paths.count("explain"), paths.count("clarify"), paths.count("decline")) == (16, 5, 3, 5)


def test_each_answer_has_the_criteria_of_its_path_and_the_ticks_total_96(dev_records):
    selection = select(fake_experiment().runs, dev_records)
    for item in selection.items:
        assert item.criteria == judge.criteria_for(item.path)
    by_path = {path: [item for item in selection.items if item.path == path] for path in ("assess", "explain", "clarify", "decline")}
    assert {path: len(items) for path, items in by_path.items()} == {"assess": 16, "explain": 5, "clarify": 3, "decline": 15}
    assert by_path["assess"][0].criteria == ("answers_the_question", "prioritised", "actionable")
    assert by_path["clarify"][0].criteria == ("answers_the_question",)
    assert by_path["decline"][0].criteria == ("answers_the_question", "good_decline")
    assert selection.ticks == 16 * 3 + 5 * 3 + 3 * 1 + 5 * 2 + 10 * 2 == 96


def test_the_path_to_criteria_map_is_the_judges_one_map_not_a_copy(dev_records, monkeypatch):
    assert annotate.CRITERIA_BY_PATH is judge.CRITERIA_BY_PATH
    monkeypatch.setattr(judge, "CRITERIA_BY_PATH", {"clarify": ("prioritised",)})  # change the one map: the selection follows it
    selection = select(fake_experiment().runs, dev_records)
    assert {item.path for item in selection.items} == {"clarify"} and all(item.criteria == ("prioritised",) for item in selection.items)


def test_the_path_is_the_examples_reference_path_not_anything_on_the_run(dev_records):
    experiment = fake_experiment()
    for run in experiment.runs:
        meta(run)["path"] = "decline"  # what the agent chose is not what decides the criteria
        (run.outputs or {})["path"] = "decline"
    selection = select(experiment.runs, dev_records)
    assert [item.path for item in selection.rep_one] == [r["reference"]["path"] for r in dev_records]


def test_the_count_line_and_the_per_path_summary(dev_records):
    selection = select(fake_experiment().runs, dev_records)
    assert annotate.count_line(selection) == COUNT_LINE
    assert annotate.path_summary(selection) == [
        "  assess   16 answers x 3 = 48 ticks: answers_the_question, prioritised, actionable",
        "  explain   5 answers x 3 = 15 ticks: answers_the_question, prioritised, actionable",
        "  clarify   3 answers x 1 =  3 ticks: answers_the_question",
        "  decline  15 answers x 2 = 30 ticks: answers_the_question, good_decline",
    ]


# --- queuing -------------------------------------------------------------------------------------------------------------------


def test_the_command_creates_the_queue_with_39_answers_and_prints_the_count_line():
    experiment = fake_experiment()
    client = CalibrationClient(experiment)
    code, out, err = annotate_main(client)
    assert (code, err) == (0, "")
    assert f"queue {queue_name(EXPERIMENT)}: created" in out and queue_name(EXPERIMENT) == f"flysafelz-calibration-{EXPERIMENT}"
    assert COUNT_LINE in out.splitlines()
    assert "added 39 to the queue, 0 already there" in out
    assert len(client.queued(queue_name(EXPERIMENT))) == 39
    assert client.methods_called().count("create_annotation_queue") == 1


def test_only_the_dev_rep_1_answers_and_the_decline_answers_of_reps_2_and_3_are_queued():
    experiment = fake_experiment()
    client = CalibrationClient(experiment)
    annotate_main(client)
    queued = client.queued_sources(queue_name(EXPERIMENT))  # the experiment run each queued copy was made from
    wanted = {str(experiment.run(r["id"], 1).id) for r in experiment.records} | {str(experiment.run(d, rep).id) for d in DECLINES for rep in (2, 3)}
    assert len(queued) == 39 and set(queued) == wanted
    assert str(experiment.run("dev_01_blue_mill", 2).id) not in queued  # an assess answer of rep 2 is not queued


def test_each_queued_copy_is_tagged_with_exactly_the_criteria_that_apply_to_its_path():
    experiment = fake_experiment()
    client = CalibrationClient(experiment)
    annotate_main(client)
    copies = {(copy.extra or {})["metadata"]["source_run_id"]: copy for copy in client.queued_runs(queue_name(EXPERIMENT))}
    for run in experiment.runs:
        copy = copies.get(str(run.id))
        if copy is None:
            continue  # a run that is not queued has no copy
        assert copy.tags == [f"label:{name}" for name in judge.criteria_for(experiment.path_of(run))]
    assert copies[str(experiment.run("dev_08_bamburgh", 2).id)].tags == ["label:answers_the_question", "label:good_decline"]
    assert copies[str(experiment.run("dev_07_buttermere", 1).id)].tags == ["label:answers_the_question"]  # E5: prioritised, actionable and good_decline are not shown


def test_each_copy_holds_the_question_the_answer_and_where_it_came_from_and_nothing_else():
    experiment = fake_experiment()
    client = CalibrationClient(experiment)
    annotate_main(client)
    assert len(client.calls_to("create_run")) == 39
    for call in client.calls_to("create_run"):
        source = next(run for run in experiment.runs if str(run.id) == call["extra"]["metadata"]["source_run_id"])
        record = experiment.record(meta(source)["example_id"])
        assert call["project_name"] == calibration_project(EXPERIMENT) == queue_name(EXPERIMENT)
        assert call["inputs"] == {"question": record["inputs"]["question"], "now": record["inputs"]["now"], "location": annotate.location_of(record)}
        assert call["outputs"] == {"answer": source.outputs["answer"]}
        assert set(call["extra"]["metadata"]) == {"source_run_id", "experiment", "example_id", "rep"}
        assert (call["start_time"], call["end_time"]) == (source.start_time, source.end_time)
        assert call["id"] == annotate.copy_run_id(EXPERIMENT, str(source.id))
        assert set(call) == {"name", "inputs", "run_type", "project_name", "id", "outputs", "start_time", "end_time", "tags", "extra"}


def test_each_copy_shows_the_place_the_judge_is_told_so_dan_and_the_judge_grade_the_same_item():
    """R3: the judge's message has `Location: <place>`; the copy carries that same place, not a second rule for it."""
    experiment = fake_experiment()
    client = CalibrationClient(experiment)
    annotate_main(client)
    for call in client.calls_to("create_run"):
        source = next(run for run in experiment.runs if str(run.id) == call["extra"]["metadata"]["source_run_id"])
        record = experiment.record(meta(source)["example_id"])
        message = judge.judge_message(record, call["outputs"]["answer"])
        assert f"\nLocation: {call['inputs']['location']}\n" in message
    assert {call["inputs"]["location"] for call in client.calls_to("create_run")} >= {"Blue Mill", "West Sleekburn", "Buttermere"}


def test_a_copy_without_a_case_folder_names_the_coordinates_as_the_judge_does(dev_records):
    record = {**dev_records[0], "fixtures": {}, "inputs": {**dev_records[0]["inputs"], "lat": 55.123456, "lon": -1.5}}
    assert annotate.location_of(record) == "the launch zone at 55.1235, -1.5000"
    assert f"\nLocation: {annotate.location_of(record)}\n" in judge.judge_message(record, "Answer.")


def test_the_instructions_tell_dan_the_place_is_part_of_what_he_sees():
    text = annotate.instructions()
    assert "launch zone" in text and "`location`" in text


def test_the_experiments_own_runs_are_never_updated_tagged_or_moved_in_time():
    """R2: `update_run` without an end_time stamps the moment of the call on the run (langsmith 0.14.17), so the runs of the experiment must not be updated at all."""
    experiment = fake_experiment()
    before = [run.model_dump() for run in experiment.runs]
    client = CalibrationClient(experiment)
    annotate_main(client)
    annotate_main(client)
    assert "update_run" not in client.methods_called()
    assert [run.model_dump() for run in experiment.runs] == before  # tags, end times, everything as make eval logged them
    assert all(run.end_time is not None and run.end_time.year == 2026 and run.end_time.day == 8 for run in experiment.runs)


def test_the_copies_are_flushed_to_langsmith_before_they_are_added_to_the_queue():
    client = CalibrationClient(fake_experiment())
    annotate_main(client)
    order = client.methods_called()
    assert "flush" in order and order.index("flush") > max(i for i, m in enumerate(order) if m == "create_run") and order.index("flush") < order.index("add_runs_to_annotation_queue")


def test_the_queue_instructions_carry_the_four_definitions_and_the_tagging_rule():
    client = CalibrationClient(fake_experiment())
    annotate_main(client)
    (created,) = client.calls_to("create_annotation_queue")
    text = created["rubric_instructions"]
    for name in judge.CRITERIA:
        assert f"- {name}: {judge.DEFINITIONS[name]}" in text
    assert "label:<criterion>" in text and "leave every other criterion blank" in text
    assert EXPERIMENT in created["description"]


# --- errored answers (E6) --------------------------------------------------------------------------------------------------


def test_an_errored_decline_answer_is_left_out_and_named_with_no_placeholder():
    experiment = fake_experiment(errored=[("dev_19_buttermere", 2)])
    client = CalibrationClient(experiment)
    code, out, err = annotate_main(client)
    assert (code, err) == (0, "")
    assert "38 answers (29 rep 1 + 9 decline from reps 2–3), 94 criterion ticks; errored, not queued: dev_19_buttermere rep 2" in out
    queued = client.queued(queue_name(EXPERIMENT))
    assert len(queued) == 38 and str(experiment.run("dev_19_buttermere", 2).id) not in client.queued_sources(queue_name(EXPERIMENT))
    assert len(client.calls_to("create_run")) == 38  # no placeholder copy


def test_an_errored_rep_1_answer_and_a_run_that_was_never_logged_are_each_named():
    experiment = fake_experiment(errored=[("dev_01_blue_mill", 1)], missing=[("dev_28_crookham", 3)])
    client = CalibrationClient(experiment)
    code, out, _ = annotate_main(client)
    assert code == 0
    assert "37 answers (28 rep 1 + 9 decline from reps 2–3), 91 criterion ticks; errored, not queued: dev_01_blue_mill rep 1, dev_28_crookham rep 3" in out
    assert len(client.queued(queue_name(EXPERIMENT))) == 37


def test_an_errored_answer_of_an_assess_example_in_rep_2_matters_not_at_all():
    experiment = fake_experiment(errored=[("dev_01_blue_mill", 2)])
    client = CalibrationClient(experiment)
    _, out, _ = annotate_main(client)
    assert COUNT_LINE in out.splitlines() and "errored" not in out


# --- the refusals -------------------------------------------------------------------------------------------------------------


def refused(client, experiment=EXPERIMENT, reason=""):
    code, out, err = annotate_main(client, experiment)
    assert code == 2, (out, err)
    assert reason in err
    assert "created" not in out
    assert client.writes() == []  # the queue is not created and no run is tagged
    return err


@pytest.mark.parametrize("name", ["flysafelz-held-out", "v1-held_out-20261108-160700", "v1-held-out-x", "V1-HELD_OUT-20261108", "heldout-run"])
def test_a_held_out_experiment_is_refused_from_its_name_before_any_langsmith_call(name):
    client = CalibrationClient(fake_experiment(name))
    refused(client, name, "held-out experiment")
    assert client.calls == []


def test_a_held_out_project_record_is_refused_even_under_an_ordinary_name():
    client = CalibrationClient(fake_experiment("v1-dev-20261108-160700", project_metadata={"agent": "v1", "split": "held_out"}))
    refused(client, EXPERIMENT, "held-out experiment")


def test_an_experiment_logged_against_the_held_out_dataset_is_refused():
    client = CalibrationClient(fake_experiment(reference_dataset_id=uuid.uuid4()), dataset_name="flysafelz-held-out")
    refused(client, EXPERIMENT, "held-out dataset flysafelz-held-out")


def test_an_experiment_whose_dataset_is_dev_is_accepted():
    client = CalibrationClient(fake_experiment(reference_dataset_id=uuid.uuid4()), dataset_name="flysafelz-dev")
    assert annotate_main(client)[0] == 0
    assert client.methods_called().count("read_dataset") == 1


def test_a_dataset_that_cannot_be_read_is_refused_not_assumed_dev():
    client = CalibrationClient(fake_experiment(reference_dataset_id=uuid.uuid4()), fail_on={"read_dataset": 1})
    refused(client, EXPERIMENT, "cannot be shown to be dev")


def test_one_run_whose_example_metadata_split_is_not_dev_stops_the_whole_command():
    """The filter by split runs on every item, not only on the experiment name (F7)."""
    experiment = fake_experiment(splits={("dev_14_town_moor", 3): "held_out"})
    client = CalibrationClient(experiment)
    err = refused(client, EXPERIMENT, f"run {experiment.run('dev_14_town_moor', 3).id} is on a held_out example")
    assert "nothing was queued" in err and client.queues == {}


def test_a_run_on_an_example_that_is_not_one_of_the_committed_dev_examples_is_refused():
    experiment = fake_experiment()
    experiment.runs[5].reference_example_id = uuid.uuid4()  # an example the dev split does not have, whatever its metadata says
    refused(CalibrationClient(experiment), EXPERIMENT, "is not on one of the dev examples")


def test_every_run_is_checked_a_held_out_example_hidden_among_dev_runs_is_found(dev_records):
    experiment = fake_experiment()
    meta(experiment.runs[-1])["split"] = "held_out"
    with pytest.raises(CalibrationError, match="held_out example"):
        select(experiment.runs, dev_records)


@pytest.mark.parametrize("name", ["v0-dev-20261108-160700", "V0-dev-x", "v0", "v0_baseline"])
def test_a_v0_experiment_is_refused_by_name(name):
    client = CalibrationClient(fake_experiment(name))
    refused(client, name, "v0 experiment")
    assert client.calls == []


def test_a_v0_experiment_is_refused_by_its_project_record_too():
    client = CalibrationClient(fake_experiment(project_metadata={"agent": "v0", "split": "dev"}))
    refused(client, EXPERIMENT, "v0 experiment")


def test_a_name_that_only_starts_like_v0_is_not_a_v0_experiment():
    client = CalibrationClient(fake_experiment("v0x-dev-1"))
    assert annotate_main(client, "v0x-dev-1")[0] == 0


def test_an_unknown_experiment_is_refused_naming_it():
    client = CalibrationClient(fake_experiment())
    refused(client, "v1-dev-20990101-000000", "'v1-dev-20990101-000000' was not found in LangSmith")


@pytest.mark.parametrize("reps", [(1,), (1, 2)])
def test_an_experiment_with_fewer_than_three_reps_is_refused(reps):
    client = CalibrationClient(fake_experiment(reps=reps))
    refused(client, EXPERIMENT, f"reps {', '.join(map(str, reps))}; it needs all 3 reps")


def test_a_run_with_no_rep_is_refused():
    experiment = fake_experiment()
    experiment.runs[0].tags = ["v1"]
    meta(experiment.runs[0]).pop("rep")
    refused(CalibrationClient(experiment), EXPERIMENT, "has no rep")


def test_two_runs_of_one_example_and_rep_are_refused():
    experiment = fake_experiment()
    experiment.runs.append(experiment.runs[0].model_copy(update={"id": uuid.uuid4()}))
    refused(CalibrationClient(experiment), EXPERIMENT, "two runs of dev_01_blue_mill rep 1")


@pytest.mark.parametrize("name", ["../x", "a/b", "a b", "-x", "x;y"])
def test_a_name_that_is_not_a_plain_name_is_refused_before_any_call(name):
    client = CalibrationClient(fake_experiment())
    refused(client, name, "must be letters, digits")
    assert client.calls == []


# --- missing EXPERIMENT -----------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("value", [None, "", "   "])
def test_a_missing_experiment_prints_the_usage_and_exits_2_with_no_client_built(value):
    def no_client(environ):
        raise AssertionError("the LangSmith client was built")

    out, err = io.StringIO(), io.StringIO()
    environ = {} if value is None else {"EXPERIMENT": value}
    code = annotate.main([], environ=environ, client_factory=no_client, stdout=out, stderr=err)
    assert code == 2 and out.getvalue() == ""
    assert err.getvalue().startswith("usage: make annotate EXPERIMENT=<v1 dev experiment name>") and "nothing was sent to LangSmith" in err.getvalue()


def test_command_line_arguments_are_refused_the_experiment_comes_from_the_environment():
    out, err = io.StringIO(), io.StringIO()
    code = annotate.main(["v1-dev-x"], environ={"EXPERIMENT": EXPERIMENT}, client=CalibrationClient(fake_experiment()), stdout=out, stderr=err)
    assert code == 2 and "takes no arguments" in err.getvalue()


def test_no_langsmith_key_is_exit_2_naming_it_and_nothing_is_built():
    out, err = io.StringIO(), io.StringIO()
    code = annotate.main([], environ={"EXPERIMENT": EXPERIMENT}, stdout=out, stderr=err)
    assert code == 2 and "LANGSMITH_API_KEY is not set" in err.getvalue()


def test_a_dataset_problem_is_exit_2_with_its_message(dev_records, monkeypatch):
    def broken(split):
        raise DatasetError("verdict sheet is stale (rules.yaml changed): run make dataset")

    monkeypatch.setattr(annotate, "load_dataset", broken)
    code, _, err = annotate_main(CalibrationClient(fake_experiment()))
    assert code == 2 and "verdict sheet is stale" in err


# --- idempotence -----------------------------------------------------------------------------------------------------------------


def test_running_annotate_twice_reuses_the_queue_and_adds_no_duplicate():
    experiment = fake_experiment()
    client = CalibrationClient(experiment)
    annotate_main(client)
    first = client.queued(queue_name(EXPERIMENT))
    writes_after_first = len(client.writes())
    code, out, _ = annotate_main(client)
    assert code == 0
    assert f"queue {queue_name(EXPERIMENT)}: reused (39 answers already in it)" in out and "added 0 to the queue, 39 already there" in out
    assert client.queued(queue_name(EXPERIMENT)) == first and len(set(first)) == 39
    assert client.methods_called().count("create_annotation_queue") == 1
    assert len(client.writes()) == writes_after_first  # the second run wrote nothing at all: no queue, no copy, no add
    assert len(client.calls_to("create_run")) == 39


def test_a_queue_that_holds_some_answers_gets_only_the_missing_ones():
    experiment = fake_experiment()
    client = CalibrationClient(experiment)
    annotate_main(client)
    queue = client.queues[queue_name(EXPERIMENT)]
    kept = queue["run_ids"][:10]
    queue["run_ids"] = list(kept)  # Dan removed 29 from the queue
    annotate_main(client)
    second_adds = [run_id for call in client.calls_to("add_runs_to_annotation_queue")[2:] for run_id in call["run_ids"]]  # the first run added 39 in two batches of 25 and 14
    assert len(second_adds) == 29 and not set(second_adds) & set(kept)
    assert len(client.queued(queue_name(EXPERIMENT))) == 39


def test_a_different_experiment_gets_its_own_queue():
    other = "v1-dev-20261109-090000"
    client = CalibrationClient(fake_experiment(), fake_experiment(other))
    annotate_main(client)
    annotate_main(client, other)
    assert sorted(client.queues) == [queue_name(EXPERIMENT), queue_name(other)]
    assert not set(client.queued(queue_name(EXPERIMENT))) & set(client.queued(queue_name(other)))
    assert len(client.queued(queue_name(other))) == 39


def test_a_langsmith_failure_part_way_is_exit_1_and_a_rerun_finishes_without_duplicates():
    experiment = fake_experiment()
    client = CalibrationClient(experiment, fail_on={"add_runs_to_annotation_queue": 1})
    code, _, err = annotate_main(client)
    assert code == 1 and "LangSmith failed: RuntimeError" in err and "nothing is queued twice" in err
    code, out, _ = annotate_main(client)
    assert code == 0 and COUNT_LINE in out.splitlines()
    queued = client.queued(queue_name(EXPERIMENT))
    assert len(queued) == 39 == len(set(queued))
    assert len(client.calls_to("create_run")) == 39  # the 39 copies made before the failure are found again, not made twice


def test_the_default_records_are_the_committed_dev_split():
    client = CalibrationClient(fake_experiment())
    assert annotate_main(client, records=None)[0] == 0


def test_the_dev_uuids_match_the_loaders_ids(dev_records):
    experiment = fake_experiment()
    assert {run.reference_example_id for run in experiment.runs} == {example_uuid(r["id"]) for r in dev_records}


# --- the real SDK accepts the copy (the recording client could hide a keyword the SDK refuses) -------------------------------------


def test_the_real_langsmith_client_builds_a_complete_run_from_the_copy_call(monkeypatch):
    """`Client.create_run` of the pinned SDK, with its transport replaced: the body it would POST carries the project, the tags, the answer, the end time and the source id."""
    import json

    from langsmith import Client

    sent = []

    def capture(self, method, pathname, *, request_kwargs=None, **kwargs):
        sent.append((method, pathname, json.loads(request_kwargs["data"]) if request_kwargs and "data" in request_kwargs else None))
        raise RuntimeError("stop at the transport")

    monkeypatch.setattr(Client, "request_with_retries", capture)
    client = Client(api_key="not-a-key", api_url="http://langsmith.invalid", auto_batch_tracing=False)
    experiment = fake_experiment()
    selection = select(experiment.runs, load_dataset("dev").records)
    item = selection.items[0]
    with pytest.raises(RuntimeError, match="stop at the transport"):
        annotate.copy_answer(client, item, EXPERIMENT)
    (method, path, body) = sent[0]
    source = experiment.run(item.example_id, item.rep)
    assert (method, path) == ("POST", "http://langsmith.invalid/runs")
    assert body["session_name"] == calibration_project(EXPERIMENT) and body["id"] == str(annotate.copy_run_id(EXPERIMENT, item.run_id))
    assert body["tags"] == ["label:answers_the_question", "label:prioritised", "label:actionable"] and body["outputs"] == {"answer": source.outputs["answer"]}
    assert body["extra"]["metadata"]["source_run_id"] == item.run_id and body["end_time"] is not None and "reference_example_id" not in body
