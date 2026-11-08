"""The loader's upload (5a PR-3.F1, G5): create or reuse the split's dataset, and make it match the file by example id, writing only what differs.

The only mock is RecordingClient: an in-memory stand-in that implements the seven client methods the loader uses and records every call.
Synthetic datasets (eval_loader_factories), including the held-out upload: no real example is uploaded, nothing is sent, no key is set.
"""

import uuid

import pytest
from eval_loader_factories import TRICKY_QUESTION, make_examples, write_dataset

from evals.agent.import_dataset import DatasetError
from evals.agent.loader import CHUNK, ID_NAMESPACE, UploadSummary, example_uuid, make_client, upload_dataset
from evals.agent.recording_client import RecordingClient

DEV = "flysafelz-dev"
FEWER = {"dev": 28, "held_out": 15, "sheet": 24}


@pytest.fixture(autouse=True)
def no_langsmith_key(monkeypatch):
    monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)


def upload(built, client, split="dev", **kwargs):
    return upload_dataset(split, client, data_dir=built.data_dir, root=built.root, **kwargs)


def test_the_first_upload_creates_every_example_and_prints_one_summary_line(tmp_path, capsys):
    built = write_dataset(tmp_path)
    client = RecordingClient()
    summary = upload(built, client)
    assert summary == UploadSummary(DEV, 29, created=29)
    assert capsys.readouterr().out == "flysafelz-dev: 29 examples (29 created, 0 updated, 0 deleted, 0 unchanged)\n"
    assert len(client.datasets[DEV].examples) == 29
    assert client.methods_called() == ["has_dataset", "create_dataset", "list_examples", "create_examples", "create_examples"]  # 25 + 4


def test_a_second_identical_upload_does_no_writes_at_all(tmp_path, capsys):
    built = write_dataset(tmp_path)
    client = RecordingClient()
    upload(built, client)
    capsys.readouterr()
    writes_before, state_before = len(client.writes()), client.state(DEV)
    summary = upload(built, client)
    assert summary == UploadSummary(DEV, 29, unchanged=29)
    assert capsys.readouterr().out == "flysafelz-dev: 29 examples (0 created, 0 updated, 0 deleted, 29 unchanged)\n"
    assert len(client.writes()) == writes_before  # no create, update or delete call
    assert client.state(DEV) == state_before
    assert client.methods_called()[-3:] == ["has_dataset", "read_dataset", "list_examples"]  # the dataset is reused, not created again


def test_one_changed_question_is_one_update_in_place_and_nothing_else(tmp_path):
    examples = make_examples()
    client = RecordingClient()
    upload(write_dataset(tmp_path, examples), client)
    before = client.state(DEV)
    examples[5]["inputs"]["question"] = "A changed invented question?"
    summary = upload(write_dataset(tmp_path / "second", examples), client)
    assert (summary.created, summary.updated, summary.deleted, summary.unchanged) == (0, 1, 0, 28)
    after = client.state(DEV)
    key = str(example_uuid(examples[5]["id"]))
    assert after[key][0]["question"] == "A changed invented question?"
    assert {k: v for k, v in after.items() if k != key} == {k: v for k, v in before.items() if k != key}
    assert "delete_examples" not in client.methods_called()
    updates = [kwargs["updates"] for method, kwargs in client.calls if method == "update_examples"]
    assert len(updates) == 1 and [u["id"] for u in updates[0]] == [example_uuid(examples[5]["id"])]


def test_a_changed_reference_or_metadata_value_is_an_update_too(tmp_path):
    examples = make_examples()
    client = RecordingClient()
    upload(write_dataset(tmp_path, examples), client)
    examples[6]["reference"]["must_mention"] = ["1500"]
    examples[7]["metadata"]["case_type"] = "tricky"
    summary = upload(write_dataset(tmp_path / "second", examples), client)
    assert (summary.created, summary.updated, summary.deleted) == (0, 2, 0)


def test_an_id_that_left_the_file_is_deleted_and_nothing_else_is_touched(tmp_path):
    examples = make_examples()
    client = RecordingClient()
    upload(write_dataset(tmp_path, examples), client)
    removed = next(e for e in reversed(examples) if e["metadata"]["split"] == "dev" and e["reference"]["path"] != "assess")
    examples.remove(removed)
    summary = upload(write_dataset(tmp_path / "second", examples), client, expected=FEWER)
    assert (summary.created, summary.updated, summary.deleted, summary.unchanged) == (0, 0, 1, 28)
    assert str(example_uuid(removed["id"])) not in client.state(DEV)
    assert len(client.datasets[DEV].examples) == 28
    assert [kwargs["example_ids"] for method, kwargs in client.calls if method == "delete_examples"] == [[example_uuid(removed["id"])]]


def test_a_new_example_is_created_next_to_the_old_ones(tmp_path):
    client = RecordingClient()
    upload(write_dataset(tmp_path, make_examples(dev=28)), client, expected=FEWER)
    summary = upload(write_dataset(tmp_path / "second", make_examples()), client)
    assert (summary.created, summary.updated, summary.deleted, summary.unchanged) == (1, 0, 0, 28)


def test_an_error_halfway_names_the_dataset_and_a_rerun_converges_on_the_same_final_state(tmp_path):
    built = write_dataset(tmp_path)
    clean = RecordingClient()
    upload(built, clean)
    flaky = RecordingClient(fail_on={"create_examples": 2})  # the second batch fails: 25 are written, 4 are not
    with pytest.raises(DatasetError) as caught:
        upload(built, flaky)
    assert str(caught.value).startswith("flysafelz-dev: LangSmith error while creating examples: RuntimeError: simulated LangSmith error in create_examples")
    assert len(flaky.datasets[DEV].examples) == CHUNK  # partly written
    summary = upload(built, flaky)  # the rerun finds the 25 and creates the 4
    assert (summary.created, summary.updated, summary.deleted, summary.unchanged) == (4, 0, 0, 25)
    assert flaky.state(DEV) == clean.state(DEV)
    again = upload(built, flaky)
    assert (again.created, again.updated, again.deleted) == (0, 0, 0)


def test_an_error_halfway_through_updates_converges_too(tmp_path):
    examples = make_examples()
    client = RecordingClient(fail_on={"update_examples": 1})  # the first upload makes no update call, so this fails the first real update
    upload(write_dataset(tmp_path, examples), client)
    for e in examples[:3]:
        e["inputs"]["question"] += " (changed)"
    second = write_dataset(tmp_path / "second", examples)
    with pytest.raises(DatasetError, match=r"flysafelz-dev: LangSmith error while updating examples"):
        upload(second, client)
    summary = upload(second, client)
    assert (summary.created, summary.updated, summary.deleted, summary.unchanged) == (0, 3, 0, 26)


@pytest.mark.parametrize("method", ["has_dataset", "create_dataset", "list_examples", "create_examples"])
def test_a_langsmith_error_in_any_step_surfaces_with_the_dataset_name(tmp_path, method):
    built = write_dataset(tmp_path)
    with pytest.raises(DatasetError, match=r"^flysafelz-dev: LangSmith error while "):
        upload(built, RecordingClient(fail_on={method: 1}))


def test_a_missing_key_raises_before_the_first_call_and_names_the_variable(tmp_path):
    built = write_dataset(tmp_path)
    with pytest.raises(DatasetError) as caught:
        upload_dataset("dev", None, data_dir=built.data_dir, root=built.root, environ={})
    assert "LANGSMITH_API_KEY" in str(caught.value) and "nothing was sent" in str(caught.value)


def test_a_blank_key_counts_as_missing_and_no_client_is_built():
    built_with = []
    with pytest.raises(DatasetError, match="LANGSMITH_API_KEY"):
        make_client({"LANGSMITH_API_KEY": "   "}, factory=lambda **kw: built_with.append(kw))
    assert built_with == []


def test_with_a_key_the_client_is_built_with_it_and_nothing_else_happens():
    seen = []
    client = make_client({"LANGSMITH_API_KEY": "invented-test-key"}, factory=lambda **kw: seen.append(kw) or "client")
    assert client == "client" and seen == [{"api_key": "invented-test-key"}]


def test_a_key_in_the_process_environment_is_read_when_no_environment_is_passed(monkeypatch):
    monkeypatch.setenv("LANGSMITH_API_KEY", "invented-test-key")
    seen = []
    make_client(factory=lambda **kw: seen.append(kw) or "client")
    assert seen == [{"api_key": "invented-test-key"}]


def test_a_data_problem_stops_the_upload_before_a_missing_key_is_even_asked_about(tmp_path):
    built = write_dataset(tmp_path, make_examples(), no_folder=frozenset({"inv_case_1"}))
    with pytest.raises(DatasetError, match="fixture folder"):
        upload_dataset("dev", None, data_dir=built.data_dir, root=built.root, environ={})


def test_the_held_out_split_uploads_to_its_own_dataset_and_never_touches_dev(tmp_path, capsys):
    built = write_dataset(tmp_path)  # synthetic held-out examples
    client = RecordingClient()
    summary = upload(built, client, "held_out")
    assert summary.dataset == "flysafelz-held-out" and summary.created == 15
    assert list(client.datasets) == ["flysafelz-held-out"]
    assert capsys.readouterr().out == "flysafelz-held-out: 15 examples (15 created, 0 updated, 0 deleted, 0 unchanged)\n"
    assert all(kwargs.get("dataset_name") in (None, "flysafelz-held-out") for _method, kwargs in client.calls)


def test_the_default_upload_goes_to_dev_only(tmp_path):
    built = write_dataset(tmp_path)
    client = RecordingClient()
    upload_dataset(client=client, data_dir=built.data_dir, root=built.root)
    assert list(client.datasets) == [DEV]
    assert all(e.metadata["split"] == "dev" for e in client.datasets[DEV].examples.values())


def test_the_example_ids_are_derived_from_the_dataset_ids_and_do_not_move(tmp_path):
    built = write_dataset(tmp_path)
    client = RecordingClient()
    upload(built, client)
    assert set(client.state(DEV)) == {str(uuid.uuid5(ID_NAMESPACE, e["id"])) for e in built.examples if e["metadata"]["split"] == "dev"}
    assert example_uuid("inv_dev_01") == example_uuid("inv_dev_01") != example_uuid("inv_dev_02")


def test_what_is_uploaded_is_inputs_with_fixtures_outputs_as_the_reference_and_the_nine_metadata_fields(tmp_path):
    examples = make_examples()
    examples[0]["inputs"]["question"] = TRICKY_QUESTION
    examples[0]["fixtures"]["drop"] = ["pib"]
    built = write_dataset(tmp_path, examples)
    client = RecordingClient()
    upload(built, client)
    inputs, outputs, metadata = client.state(DEV)[str(example_uuid(examples[0]["id"]))]
    assert inputs == {"lat": 51.9, "lon": -1.1, "now": "2026-01-02T03:04Z", "question": TRICKY_QUESTION, "fixtures": {"case": "inv_case_1", "drop": ["pib"]}}
    assert outputs["path"] == "assess" and set(outputs) >= {"verdict_by_segment", "tier", "notam_actions", "window", "required_tools"}
    assert list(metadata) == ["split", "case_type", "path", "failure_source", "notam_category", "airspace_context", "constructed", "compound", "parts"]


def test_extra_metadata_the_service_adds_does_not_make_an_example_look_changed(tmp_path):
    built = write_dataset(tmp_path)
    client = RecordingClient()
    upload(built, client)
    for stored in client.datasets[DEV].examples.values():
        stored.metadata["dataset_split"] = ["base"]
    assert upload(built, client).unchanged == 29


def test_a_refused_example_is_neither_uploaded_nor_deleted(tmp_path, capsys):
    examples = make_examples()
    built = write_dataset(tmp_path, examples, not_done=frozenset({examples[0]["id"]}))
    client = RecordingClient()
    summary = upload(built, client)
    assert summary.total == 28 and summary.created == 28
    assert str(example_uuid(examples[0]["id"])) not in client.state(DEV)
    assert f"refused {examples[0]['id']}: its verdict sheet entry has actions_done: false" in capsys.readouterr().out
    # it was uploaded earlier and is refused now: it stays where it is
    client.datasets[DEV].examples[example_uuid(examples[0]["id"])] = client.datasets[DEV].examples[example_uuid(examples[1]["id"])]
    assert upload(built, client).deleted == 0
    assert str(example_uuid(examples[0]["id"])) in client.state(DEV)


def test_an_unknown_split_is_refused_before_any_call():
    client = RecordingClient()
    with pytest.raises(DatasetError, match="unknown split"):
        upload_dataset("held-out", client)
    assert client.calls == []
