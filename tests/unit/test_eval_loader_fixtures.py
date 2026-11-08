"""The loader's fixture check (5a PR-3.F1, E1): a missing case folder stops it before any LangSmith call, and the message lists every one.

A folder that exists without a case.yaml counts as missing. The check covers every example's case, whichever split was asked for; a case only the
other split uses is named without example ids. Synthetic datasets (eval_loader_factories), RecordingClient for the call log.
"""

import pytest
from eval_loader_factories import make_examples, write_dataset

from evals.agent.import_dataset import DatasetError
from evals.agent.loader import check_dataset, load_examples, upload_dataset
from evals.agent.recording_client import RecordingClient


def test_two_missing_folders_and_one_without_a_case_yaml_come_in_one_sorted_message_before_any_call(tmp_path):
    examples = make_examples()
    built = write_dataset(tmp_path, examples, no_folder=frozenset({"inv_case_4", "inv_case_1"}), no_case_yaml=frozenset({"inv_case_2"}))
    client = RecordingClient()
    with pytest.raises(DatasetError) as caught:
        upload_dataset("dev", client, data_dir=built.data_dir, root=built.root)
    lines = str(caught.value).splitlines()
    assert lines[0] == "3 fixture folder(s) missing or without case.yaml:"
    assert [line.split(" (")[0].strip() for line in lines[1:]] == ["fixtures/inv_case_1/", "fixtures/inv_case_2/", "fixtures/inv_case_4/"]  # sorted
    assert client.calls == []  # nothing reached LangSmith, not even a lookup


def test_each_missing_folder_lists_every_example_id_that_uses_it_not_just_the_first(tmp_path):
    examples = make_examples()
    built = write_dataset(tmp_path, examples, no_folder=frozenset({"inv_case_1"}))
    dev_users = [e["id"] for e in examples if e["fixtures"]["case"] == "inv_case_1" and e["metadata"]["split"] == "dev"]
    assert len(dev_users) > 1
    with pytest.raises(DatasetError) as caught:
        load_examples("dev", data_dir=built.data_dir, root=built.root)
    assert f"fixtures/inv_case_1/ (used by {', '.join(dev_users)})" in str(caught.value)


def test_a_case_only_the_other_split_uses_is_named_without_example_ids(tmp_path):
    examples = make_examples(dev=29, held_out=15, assess_dev=16, assess_held_out=8)
    for e in examples:  # dev uses only inv_case_0..3; inv_case_4 is held-out's alone
        if e["metadata"]["split"] == "dev" and e["fixtures"]["case"] == "inv_case_4":
            e["fixtures"]["case"] = "inv_case_0"
    held_users = [e["id"] for e in examples if e["fixtures"]["case"] == "inv_case_4"]
    assert held_users and all(i.startswith("inv_ho_") for i in held_users)
    built = write_dataset(tmp_path, examples, no_folder=frozenset({"inv_case_4"}))
    with pytest.raises(DatasetError) as caught:
        load_examples("dev", data_dir=built.data_dir, root=built.root)  # a dev load still sees the held-out gap
    message = str(caught.value)
    assert "fixtures/inv_case_4/" in message
    assert not any(i in message for i in held_users) and "inv_ho_" not in message
    with pytest.raises(DatasetError) as asked:  # asked for by name, the held-out ids are listed
        load_examples("held_out", data_dir=built.data_dir, root=built.root)
    assert all(i in str(asked.value) for i in held_users)


def test_a_folder_with_no_case_yaml_is_missing_even_when_its_other_files_are_there(tmp_path):
    built = write_dataset(tmp_path, make_examples(), no_case_yaml=frozenset({"inv_case_3"}))
    assert (tmp_path / "fixtures" / "inv_case_3" / "weather.json").is_file()
    with pytest.raises(DatasetError, match=r"fixtures/inv_case_3/ \(used by "):
        load_examples("dev", data_dir=built.data_dir, root=built.root)


def test_a_case_name_that_is_a_file_not_a_folder_is_missing(tmp_path):
    built = write_dataset(tmp_path, make_examples(), no_folder=frozenset({"inv_case_0"}))
    (tmp_path / "fixtures" / "inv_case_0").write_text("not a folder", encoding="utf-8")
    with pytest.raises(DatasetError, match=r"fixtures/inv_case_0/"):
        load_examples("dev", data_dir=built.data_dir, root=built.root)


def test_the_gap_is_reported_before_the_sheet_looks_stale(tmp_path):
    """A stamp is sealed over the case's files, so a missing folder would otherwise read as a stale sheet and hide which folders are gone."""
    built = write_dataset(tmp_path, make_examples(), no_folder=frozenset({"inv_case_1"}))
    with pytest.raises(DatasetError) as caught:
        load_examples("dev", data_dir=built.data_dir, root=built.root)
    assert "stale" not in str(caught.value) and "fixtures/inv_case_1/" in str(caught.value)


def test_check_reports_a_missing_folder_the_same_way(tmp_path):
    built = write_dataset(tmp_path, make_examples(), no_folder=frozenset({"inv_case_2"}))
    with pytest.raises(DatasetError, match=r"1 fixture folder\(s\) missing"):
        check_dataset(data_dir=built.data_dir, root=built.root)


def test_with_every_folder_present_the_check_line_counts_the_distinct_cases(tmp_path):
    built = write_dataset(tmp_path, make_examples())
    assert check_dataset(data_dir=built.data_dir, root=built.root) == "dev 29 · held_out 15 · sheet 24 · fixtures 5/5 present · sheet finalised"
