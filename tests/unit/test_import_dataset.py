"""evals/agent/import_dataset.py: `make dataset` checks Dan's three files against dataset.sha256, then copies them (5a PR-1.S1, G3, E3, F8).

DATASET_SRC is set with monkeypatch.setenv to a tmp folder holding three invented YAML files; dataset.sha256 sits beside it, as Dan's does.
The destination is another tmp folder, so the repo's evals/agent/data/ is never touched.
"""

import hashlib
import os
import stat
import subprocess
import sys
from pathlib import Path

import pytest

from evals.agent import import_dataset
from evals.agent.import_dataset import REQUIRED, DatasetError, main

REPO = Path(__file__).resolve().parents[2]
BODIES = {
    "examples.yaml": b"- id: invented_one\n  question: Is the invented field flyable?\n",
    "verdict_sheet.yaml": b"- id: invented_one\n  rows: []\n",
    "guidance_questions.yaml": b"- question: What is an invented rule?\n",
}


def sha(body: bytes) -> str:
    return hashlib.sha256(body).hexdigest()


def make_source(root: Path, name: str = "dataset", bodies=BODIES, sums_lines=None) -> Path:
    """<root>/<name>/ with the files and <root>/dataset.sha256 beside it."""
    src = root / name
    src.mkdir(parents=True)
    for file, body in bodies.items():
        (src / file).write_bytes(body)
    lines = sums_lines if sums_lines is not None else [f"{sha(body)}  {file}" for file, body in BODIES.items()]
    (root / "dataset.sha256").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return src


@pytest.fixture
def dest(tmp_path):
    return tmp_path / "out" / "data"


def run(monkeypatch, src, dest, capsys):
    monkeypatch.setenv("DATASET_SRC", str(src))
    code = main(dest=dest)
    out, err = capsys.readouterr()
    return code, out, err


def snapshot(folder: Path) -> dict[str, tuple[bytes, int]]:
    return {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in sorted(folder.iterdir())} if folder.exists() else {}


# --- the happy copy -------------------------------------------------------------------------------------------------------------


def test_a_good_source_is_verified_and_copied_byte_for_byte(tmp_path, dest, monkeypatch, capsys):
    src = make_source(tmp_path / "in")
    code, out, err = run(monkeypatch, src, dest, capsys)
    assert code == 0 and err == ""
    assert {p.name for p in dest.iterdir()} == set(REQUIRED)  # no temporary file is left behind
    for name, body in BODIES.items():
        assert (dest / name).read_bytes() == body
        assert sha((dest / name).read_bytes()) == sha(body)
        assert f"copied: {name}" in out
    assert "verified" in out and "dataset.sha256" in out


def test_the_default_destination_is_the_repos_data_folder():
    assert import_dataset.DATA_DIR == REPO / "evals" / "agent" / "data"


def test_a_source_with_spaces_and_a_quote_in_its_path_works_because_it_comes_from_the_environment(tmp_path, dest, monkeypatch, capsys):
    src = make_source(tmp_path / "in put's", name="my data set")
    code, _out, err = run(monkeypatch, src, dest, capsys)
    assert code == 0 and err == ""
    assert (dest / "examples.yaml").read_bytes() == BODIES["examples.yaml"]


def test_a_source_path_with_spaces_and_a_quote_works_end_to_end_through_a_subprocess(tmp_path):
    """The real command line: the path reaches `python -m evals.agent.import_dataset` only through the environment."""
    src = make_source(tmp_path / "in put's", name="my \"data\" set")
    code = (
        "import sys; from pathlib import Path; from evals.agent.import_dataset import main; "
        f"sys.exit(main(dest=Path({str(tmp_path / 'out')!r})))"
    )
    done = subprocess.run([sys.executable, "-c", code], cwd=REPO, env={**os.environ, "DATASET_SRC": str(src)}, capture_output=True, text=True, check=False)
    assert done.returncode == 0, done.stderr
    assert (tmp_path / "out" / "verdict_sheet.yaml").read_bytes() == BODIES["verdict_sheet.yaml"]


def test_the_module_runs_as_python_dash_m_and_exits_1_naming_the_setting_when_it_is_unset():
    env = {k: v for k, v in os.environ.items() if k != "DATASET_SRC"}
    done = subprocess.run([sys.executable, "-m", "evals.agent.import_dataset"], cwd=REPO, env=env, capture_output=True, text=True, check=False)
    assert done.returncode == 1 and "DATASET_SRC is not set" in done.stderr


def test_a_tilde_in_the_source_is_expanded(tmp_path, dest, monkeypatch, capsys):
    src = make_source(tmp_path / "in")
    monkeypatch.setenv("HOME", str(tmp_path / "in"))
    monkeypatch.setenv("DATASET_SRC", "~/dataset")
    assert main(dest=dest) == 0 and src.is_dir()
    assert (dest / "examples.yaml").is_file()


# --- a mismatch is all or nothing -----------------------------------------------------------------------------------------------


@pytest.mark.parametrize("bad", REQUIRED)
def test_one_changed_byte_exits_1_names_the_file_and_leaves_the_destination_as_it_was(tmp_path, dest, monkeypatch, capsys, bad):
    src = make_source(tmp_path / "in")
    dest.mkdir(parents=True)
    (dest / "examples.yaml").write_bytes(b"older copy")
    (dest / "kept.txt").write_bytes(b"not ours")
    before = snapshot(dest)
    body = bytearray((src / bad).read_bytes())
    body[3] ^= 1
    (src / bad).write_bytes(bytes(body))

    code, out, err = run(monkeypatch, src, dest, capsys)

    assert code == 1 and out == ""
    assert bad in err and "does not match" in err
    assert snapshot(dest) == before  # not one file written, replaced or left half-copied


def test_a_mismatch_in_the_last_file_still_copies_none_of_the_first_two(tmp_path, dest, monkeypatch, capsys):
    src = make_source(tmp_path / "in")
    (src / "guidance_questions.yaml").write_bytes(b"tampered")
    code, _out, err = run(monkeypatch, src, dest, capsys)
    assert code == 1 and "guidance_questions.yaml" in err
    assert not dest.exists() or list(dest.iterdir()) == []


def test_every_bad_file_is_named_not_only_the_first(tmp_path, dest, monkeypatch, capsys):
    src = make_source(tmp_path / "in")
    (src / "examples.yaml").write_bytes(b"tampered")
    (src / "verdict_sheet.yaml").unlink()
    code, _out, err = run(monkeypatch, src, dest, capsys)
    assert code == 1 and "examples.yaml" in err and "verdict_sheet.yaml" in err and "missing" in err


def test_a_failure_while_writing_leaves_no_temporary_file_and_no_partial_copy(tmp_path, dest, monkeypatch, capsys):
    src = make_source(tmp_path / "in")
    real = os.replace
    calls = []

    def failing(source, target):
        calls.append(target)
        raise OSError("disk full")

    monkeypatch.setattr(import_dataset.os, "replace", failing)
    code, _out, err = run(monkeypatch, src, dest, capsys)
    monkeypatch.setattr(import_dataset.os, "replace", real)
    assert code == 1 and "cannot write" in err and calls
    assert list(dest.iterdir()) == []


# --- the source and sums themselves ---------------------------------------------------------------------------------------------


def test_an_unset_source_exits_1_naming_the_setting(dest, monkeypatch, capsys):
    monkeypatch.delenv("DATASET_SRC", raising=False)
    assert main(dest=dest) == 1
    assert "DATASET_SRC is not set" in capsys.readouterr().err


def test_an_empty_source_setting_is_unset(dest, monkeypatch, capsys):
    monkeypatch.setenv("DATASET_SRC", "  ")
    assert main(dest=dest) == 1
    assert "DATASET_SRC is not set" in capsys.readouterr().err


def test_a_missing_source_folder_exits_1_naming_it(tmp_path, dest, monkeypatch, capsys):
    code, _out, err = run(monkeypatch, tmp_path / "no such folder", dest, capsys)
    assert code == 1 and "DATASET_SRC is not a folder" in err and "no such folder" in err
    assert not dest.exists()


def test_a_source_that_is_a_file_is_not_a_folder(tmp_path, dest, monkeypatch, capsys):
    (tmp_path / "plain").write_text("x", encoding="utf-8")
    code, _out, err = run(monkeypatch, tmp_path / "plain", dest, capsys)
    assert code == 1 and "not a folder" in err


def test_a_missing_sums_file_exits_1_naming_it(tmp_path, dest, monkeypatch, capsys):
    src = make_source(tmp_path / "in")
    (tmp_path / "in" / "dataset.sha256").unlink()
    code, _out, err = run(monkeypatch, src, dest, capsys)
    assert code == 1 and "dataset.sha256" in err and "missing" in err
    assert not dest.exists()


def test_an_empty_sums_file_exits_1(tmp_path, dest, monkeypatch, capsys):
    src = make_source(tmp_path / "in", sums_lines=[""])
    code, _out, err = run(monkeypatch, src, dest, capsys)
    assert code == 1 and "dataset.sha256" in err and "lists no files" in err
    assert not dest.exists()


def test_a_malformed_sums_line_exits_1_with_its_line_number(tmp_path, dest, monkeypatch, capsys):
    good = [f"{sha(body)}  {file}" for file, body in BODIES.items()]
    src = make_source(tmp_path / "in", sums_lines=[good[0], good[1], "this is not a checksum", good[2]])
    code, _out, err = run(monkeypatch, src, dest, capsys)
    assert code == 1 and "line 3" in err
    assert not dest.exists()


@pytest.mark.parametrize("omitted", REQUIRED)
def test_a_sums_file_that_omits_a_required_file_is_a_failure_not_a_skip(tmp_path, dest, monkeypatch, capsys, omitted):
    src = make_source(tmp_path / "in", sums_lines=[f"{sha(body)}  {file}" for file, body in BODIES.items() if file != omitted])
    code, _out, err = run(monkeypatch, src, dest, capsys)
    assert code == 1 and omitted in err and "does not list" in err
    assert not dest.exists()


def test_a_sums_file_may_use_the_dot_slash_form_and_upper_case_hex(tmp_path, dest, monkeypatch, capsys):
    src = make_source(tmp_path / "in", sums_lines=[f"{sha(body).upper()}  ./{file}" for file, body in BODIES.items()])
    code, _out, _err = run(monkeypatch, src, dest, capsys)
    assert code == 0


def test_a_file_the_sums_list_but_the_source_lacks_fails_even_if_it_is_not_one_of_the_three(tmp_path, dest, monkeypatch, capsys):
    lines = [f"{sha(body)}  {file}" for file, body in BODIES.items()] + [f"{sha(b'x')}  extra_notes.yaml"]
    src = make_source(tmp_path / "in", sums_lines=lines)
    code, _out, err = run(monkeypatch, src, dest, capsys)
    assert code == 1 and "extra_notes.yaml" in err
    assert not dest.exists()


def test_the_sums_are_found_beside_the_source_folder_not_inside_it(tmp_path, dest, monkeypatch, capsys):
    src = make_source(tmp_path / "in")
    (tmp_path / "in" / "dataset.sha256").rename(src / "dataset.sha256")
    code, _out, err = run(monkeypatch, src, dest, capsys)
    assert code == 1 and "dataset.sha256" in err


def test_an_unreadable_source_file_is_named_in_the_error(tmp_path, dest, monkeypatch, capsys):
    src = make_source(tmp_path / "in")
    (src / "examples.yaml").chmod(0)
    try:
        code, _out, err = run(monkeypatch, src, dest, capsys)
    finally:
        (src / "examples.yaml").chmod(0o644)
    assert code == 1 and "examples.yaml" in err
    assert not dest.exists()


# --- a re-run changes nothing ---------------------------------------------------------------------------------------------------


def test_a_second_run_with_an_unchanged_source_rewrites_nothing(tmp_path, dest, monkeypatch, capsys):
    src = make_source(tmp_path / "in")
    assert run(monkeypatch, src, dest, capsys)[0] == 0
    first = snapshot(dest)
    code, out, _err = run(monkeypatch, src, dest, capsys)
    assert code == 0
    assert snapshot(dest) == first  # same bytes and same modification times: no file was written
    assert all(f"unchanged: {name}" in out for name in REQUIRED)


def test_a_changed_source_replaces_only_the_file_that_changed(tmp_path, dest, monkeypatch, capsys):
    src = make_source(tmp_path / "in")
    assert run(monkeypatch, src, dest, capsys)[0] == 0
    first = snapshot(dest)
    new = b"- id: invented_two\n"
    (src / "verdict_sheet.yaml").write_bytes(new)
    sums = [f"{sha(new if file == 'verdict_sheet.yaml' else body)}  {file}" for file, body in BODIES.items()]
    (tmp_path / "in" / "dataset.sha256").write_text("\n".join(sums) + "\n", encoding="utf-8")
    code, out, _err = run(monkeypatch, src, dest, capsys)
    assert code == 0 and (dest / "verdict_sheet.yaml").read_bytes() == new
    assert snapshot(dest)["examples.yaml"] == first["examples.yaml"] and "copied: verdict_sheet.yaml" in out


def test_a_changed_destination_file_is_replaced_by_the_checked_source_bytes(tmp_path, dest, monkeypatch, capsys):
    src = make_source(tmp_path / "in")
    dest.mkdir(parents=True)
    (dest / "examples.yaml").write_bytes(b"hand edited")
    assert run(monkeypatch, src, dest, capsys)[0] == 0
    assert (dest / "examples.yaml").read_bytes() == BODIES["examples.yaml"]


# --- F8: the source is only read ------------------------------------------------------------------------------------------------


def test_a_read_only_source_is_copied_and_its_bytes_and_mtimes_do_not_move(tmp_path, dest, monkeypatch, capsys):
    root = tmp_path / "in"
    src = make_source(root)
    watched = [root, src, root / "dataset.sha256", *(src / name for name in REQUIRED)]

    def state():
        return {str(p): (p.stat().st_mtime_ns, p.read_bytes() if p.is_file() else None) for p in watched}

    for path in watched:
        path.chmod(0o555 if path.is_dir() else 0o444)
    before = state()
    try:
        code, _out, err = run(monkeypatch, src, dest, capsys)
        after = state()
        listing_after = sorted(p.name for p in src.iterdir())
    finally:
        for path in watched:
            path.chmod(0o755 if path.is_dir() else 0o644)
    assert code == 0, err
    assert after == before
    assert listing_after == sorted(REQUIRED)  # nothing was created beside the source files
    assert sorted(p.name for p in root.iterdir()) == ["dataset", "dataset.sha256"]
    assert stat.S_IMODE((dest / "examples.yaml").stat().st_mode) == 0o644  # the copy is not read-only just because its source was


def test_a_mismatch_in_a_read_only_source_also_leaves_it_untouched(tmp_path, dest, monkeypatch, capsys):
    root = tmp_path / "in"
    src = make_source(root)
    (src / "examples.yaml").write_bytes(b"tampered")
    mtimes = {p: p.stat().st_mtime_ns for p in [root, src, *src.iterdir()]}
    for path in [src, root]:
        path.chmod(0o555)
    try:
        code, _out, _err = run(monkeypatch, src, dest, capsys)
    finally:
        for path in [src, root]:
            path.chmod(0o755)
    assert code == 1
    assert {p: p.stat().st_mtime_ns for p in mtimes} == mtimes


def test_dataset_error_is_what_the_helpers_raise(tmp_path):
    with pytest.raises(DatasetError, match="does not list"):
        import_dataset.verified_bytes(make_source(tmp_path / "in", sums_lines=[f"{sha(b'x')}  examples.yaml"]))
