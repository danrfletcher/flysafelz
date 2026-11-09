"""The loader's command line (5a PR-3.F1): --check, --dry-run and --upload through `main`, on synthetic datasets. Nothing is sent: --upload runs without a key.

Also the small error paths of reading: an examples file that cannot be decoded, a bad `now` type, a non-numeric longitude.
"""

import pytest
from eval_loader_factories import make_examples, write_dataset

from evals.agent.import_dataset import DatasetError
from evals.agent.loader import load_examples, main


@pytest.fixture(autouse=True)
def no_langsmith_key(monkeypatch):
    monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)


def run(built, *argv, environ=None):
    return main(list(argv), environ=environ, data_dir=built.data_dir, root=built.root)


def test_check_prints_the_counts_line_and_exits_0(tmp_path, capsys):
    assert run(write_dataset(tmp_path), "--check") == 0
    assert capsys.readouterr().out == "dev 29 · held_out 15 · sheet 24 · fixtures 5/5 present · sheet finalised\n"


def test_check_names_the_refused_examples_count(tmp_path, capsys):
    examples = make_examples()
    assert run(write_dataset(tmp_path, examples, not_done=frozenset({examples[0]["id"]})), "--check") == 0
    assert capsys.readouterr().out.endswith(" · 1 refused (actions_done false)\n")


def test_check_on_a_missing_folder_exits_1_with_the_message_on_stderr_only(tmp_path, capsys):
    built = write_dataset(tmp_path, make_examples(), no_folder=frozenset({"inv_case_1", "inv_case_3"}))
    assert run(built, "--check") == 1
    out, err = capsys.readouterr()
    assert out == "" and err.startswith("evals.agent.loader: 2 fixture folder(s) missing or without case.yaml:\n") and "fixtures/inv_case_1/" in err and "fixtures/inv_case_3/" in err


def test_dry_run_uploads_twice_and_the_second_run_writes_nothing(tmp_path, capsys):
    assert run(write_dataset(tmp_path), "--dry-run") == 0
    lines = capsys.readouterr().out.splitlines()
    assert lines[0].startswith("dry run: dev uploaded twice")
    assert "flysafelz-dev: 29 examples (29 created, 0 updated, 0 deleted, 0 unchanged)" in lines
    assert "flysafelz-dev: 29 examples (0 created, 0 updated, 0 deleted, 29 unchanged)" in lines
    assert [line for line in lines if line.startswith("run ") and "write call(s)" in line] == ["run 1: 3 write call(s)", "run 2: 0 write call(s)"]


def test_dry_run_with_held_out_named_uses_the_held_out_dataset(tmp_path, capsys):
    assert run(write_dataset(tmp_path), "--dry-run", "--split", "held_out") == 0
    assert "flysafelz-held-out: 15 examples (15 created" in capsys.readouterr().out


def test_upload_without_a_key_exits_1_naming_the_variable(tmp_path, capsys):
    assert run(write_dataset(tmp_path), "--upload", environ={}) == 1
    out, err = capsys.readouterr()
    assert out == "" and "LANGSMITH_API_KEY" in err and "nothing was sent" in err


def test_a_command_is_required_and_a_split_must_be_known(tmp_path):
    built = write_dataset(tmp_path)
    with pytest.raises(SystemExit) as no_command:
        run(built)
    with pytest.raises(SystemExit) as bad_split:
        run(built, "--check", "--split", "test")
    assert no_command.value.code == 2 and bad_split.value.code == 2


def test_a_file_that_cannot_be_decoded_is_a_one_line_error_naming_it(tmp_path):
    built = write_dataset(tmp_path)
    (built.data_dir / "examples.yaml").write_bytes(b"\xff\xfe- id: x\n")
    with pytest.raises(DatasetError, match=r"^examples.yaml: cannot be read \(UnicodeDecodeError\)$"):
        load_examples("dev", data_dir=built.data_dir, root=built.root)


def test_an_unreadable_sheet_is_a_dataset_error(tmp_path):
    built = write_dataset(tmp_path)
    (built.data_dir / "verdict_sheet.yaml").write_text("- [unclosed\n", encoding="utf-8")
    with pytest.raises(DatasetError):
        load_examples("dev", data_dir=built.data_dir, root=built.root)


def test_a_non_numeric_longitude_and_a_boolean_latitude_are_refused(tmp_path):
    examples = make_examples()
    examples[2]["inputs"]["lon"] = "west"
    with pytest.raises(DatasetError, match=r"inputs.lon must be a number"):
        load_examples("dev", data_dir=write_dataset(tmp_path / "a", examples).data_dir, root=tmp_path / "a")
    examples[2]["inputs"]["lon"] = -1.1
    examples[2]["inputs"]["lat"] = True
    with pytest.raises(DatasetError, match=r"inputs.lat must be a number"):
        load_examples("dev", data_dir=write_dataset(tmp_path / "b", examples).data_dir, root=tmp_path / "b")
