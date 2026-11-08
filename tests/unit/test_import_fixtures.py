"""evals/agent/import_fixtures.py: copy absent labelled folders byte for byte, never overwrite, check everything (5a PR-1.S1, G4, E3, F4, F8).

Two invented case folders in a tmp dir. Nothing here names a real case or holds a real landing zone.
"""

import hashlib
from pathlib import Path

import pytest

from evals.agent import import_fixtures
from evals.agent.import_fixtures import FixtureImportError, main
from evals.agent.import_fixtures import import_fixtures as run_import

FILES = ("case.yaml", "weather.json", "pib.xml")
CASES = {
    "alpha_case": {name: f"alpha {name}\n".encode() for name in FILES},
    "beta_case": {name: f"beta {name}\n".encode() for name in FILES},
}


def sha(body: bytes) -> str:
    return hashlib.sha256(body).hexdigest()


def write_case(root: Path, name: str, files: dict[str, bytes]) -> None:
    for file, body in files.items():
        (root / name / file).parent.mkdir(parents=True, exist_ok=True)
        (root / name / file).write_bytes(body)


@pytest.fixture
def world(tmp_path):
    """src/ holds both cases; dest/ holds the checksum list for both and an empty exception list."""
    src, dest = tmp_path / "src", tmp_path / "dest"
    for name, files in CASES.items():
        write_case(src, name, files)
    dest.mkdir()
    lines = [f"{sha(body)}  ./{name}/{file}" for name, files in CASES.items() for file, body in files.items()]
    (dest / "labelled-fixtures.sha256").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (dest / "labelled_exceptions.yaml").write_text("[]\n", encoding="utf-8")
    return src, dest


def paths(dest: Path):
    return dest / "labelled-fixtures.sha256", dest / "labelled_exceptions.yaml"


def tree(root: Path) -> dict[str, bytes]:
    return {p.relative_to(root).as_posix(): p.read_bytes() for p in sorted(root.rglob("*")) if p.is_file()}


# --- copying only what is absent ------------------------------------------------------------------------------------------------


def test_every_absent_folder_is_copied_byte_for_byte_and_checked(world):
    src, dest = world
    copied, skipped, files = run_import(src, dest, *paths(dest))
    assert copied == ["alpha_case", "beta_case"] and skipped == [] and files == 6
    for name, case_files in CASES.items():
        assert {p.name: p.read_bytes() for p in (dest / name).iterdir()} == case_files
    assert sorted(p.name for p in dest.iterdir()) == ["alpha_case", "beta_case", "labelled-fixtures.sha256", "labelled_exceptions.yaml"]  # no temporary folder is left


def test_only_the_absent_folder_is_copied_when_the_other_exists(world):
    src, dest = world
    write_case(dest, "alpha_case", CASES["alpha_case"])
    copied, skipped, _files = run_import(src, dest, *paths(dest))
    assert copied == ["beta_case"] and skipped == ["alpha_case"]


def test_a_folder_that_already_exists_is_never_overwritten_even_when_its_bytes_differ(world, capsys):
    src, dest = world
    write_case(dest, "alpha_case", {**CASES["alpha_case"], "case.yaml": b"edited after the labelled capture\n"})
    (dest / "alpha_case" / "extra.txt").write_bytes(b"left alone too")
    before = tree(dest / "alpha_case")
    sums, exceptions = paths(dest)
    with pytest.raises(FixtureImportError) as error:  # the check afterwards still sees that folder is wrong
        run_import(src, dest, sums, exceptions)
    assert tree(dest / "alpha_case") == before  # not one byte replaced, no file added or removed
    assert "alpha_case/case.yaml: differs" in str(error.value) and "alpha_case/extra.txt: unlisted" in str(error.value)
    assert (dest / "beta_case" / "case.yaml").read_bytes() == CASES["beta_case"]["case.yaml"]  # the absent one was still copied


def test_the_cli_names_a_folder_it_left_alone_and_exits_0_when_the_check_is_clean(world, capsys):
    src, dest = world
    write_case(dest, "alpha_case", CASES["alpha_case"])
    assert main(["--src", str(src), "--dest", str(dest)]) == 0
    out = capsys.readouterr().out
    assert "already in dest/, not touched: alpha_case" in out and "copied: beta_case" in out
    assert "1 folders copied, 1 already present" in out and "verified 2 folders, 6 files" in out and "0 named exception" in out


def test_a_second_run_copies_nothing_and_changes_nothing(world):
    src, dest = world
    run_import(src, dest, *paths(dest))
    before = tree(dest)
    copied, skipped, _files = run_import(src, dest, *paths(dest))
    assert copied == [] and skipped == ["alpha_case", "beta_case"] and tree(dest) == before


def test_a_file_in_a_subfolder_of_a_case_is_copied_with_its_path(tmp_path):
    src, dest = tmp_path / "src", tmp_path / "dest"
    write_case(src, "alpha_case", {"case.yaml": b"c", "deeper/more.json": b"m"})
    dest.mkdir()
    (dest / "labelled-fixtures.sha256").write_text(f"{sha(b'c')}  ./alpha_case/case.yaml\n{sha(b'm')}  ./alpha_case/deeper/more.json\n", encoding="utf-8")
    (dest / "labelled_exceptions.yaml").write_text("[]\n", encoding="utf-8")
    run_import(src, dest, *paths(dest))
    assert (dest / "alpha_case" / "deeper" / "more.json").read_bytes() == b"m"


def test_the_destination_folder_is_made_when_it_does_not_exist(tmp_path, world):
    src, dest = world
    fresh = tmp_path / "fresh" / "fixtures"
    run_import(src, fresh, *paths(dest))
    assert (fresh / "alpha_case" / "case.yaml").is_file()


def test_the_copy_uses_shutil_copyfile_only(world, monkeypatch):
    src, dest = world
    calls = []
    real = import_fixtures.shutil.copyfile
    monkeypatch.setattr(import_fixtures.shutil, "copyfile", lambda a, b: calls.append((Path(a).name, Path(b).name)) or real(a, b))
    run_import(src, dest, *paths(dest))
    assert len(calls) == 6 and all(a == b for a, b in calls)


# --- a bad source copies nothing ------------------------------------------------------------------------------------------------


def test_a_source_file_that_differs_from_the_list_copies_nothing_at_all(world):
    src, dest = world
    (src / "beta_case" / "pib.xml").write_bytes(b"tampered")
    with pytest.raises(FixtureImportError, match=r"(?s)nothing was copied.*beta_case/pib\.xml: differs"):
        run_import(src, dest, *paths(dest))
    assert sorted(p.name for p in dest.iterdir()) == ["labelled-fixtures.sha256", "labelled_exceptions.yaml"]


def test_a_source_with_a_file_the_list_lacks_copies_nothing(world):
    src, dest = world
    (src / "alpha_case" / "stray.txt").write_bytes(b"not in the list")
    with pytest.raises(FixtureImportError, match=r"alpha_case/stray\.txt: unlisted"):
        run_import(src, dest, *paths(dest))
    assert not (dest / "alpha_case").exists()


def test_a_source_that_lacks_a_listed_file_copies_nothing(world):
    src, dest = world
    (src / "alpha_case" / "weather.json").unlink()
    with pytest.raises(FixtureImportError, match=r"alpha_case/weather\.json: missing"):
        run_import(src, dest, *paths(dest))
    assert not (dest / "alpha_case").exists() and not (dest / "beta_case").exists()


def test_a_missing_source_folder_is_named(world, tmp_path):
    _src, dest = world
    with pytest.raises(FixtureImportError, match="--src is not a folder"):
        run_import(tmp_path / "nowhere", dest, *paths(dest))


def test_a_missing_checksum_list_or_exception_list_is_a_failure(world):
    src, dest = world
    sums, exceptions = paths(dest)
    exceptions.unlink()
    with pytest.raises(FixtureImportError, match=r"labelled_exceptions\.yaml"):
        run_import(src, dest, sums, exceptions)
    sums.unlink()
    with pytest.raises(FixtureImportError, match=r"labelled-fixtures\.sha256"):
        run_import(src, dest, sums, exceptions)
    assert sorted(p.name for p in dest.iterdir()) == []


def test_a_failed_copy_leaves_no_temporary_folder(world, monkeypatch):
    src, dest = world

    def failing(a, b):
        raise OSError("disk full")

    monkeypatch.setattr(import_fixtures.shutil, "copyfile", failing)
    with pytest.raises(FixtureImportError, match="alpha_case: cannot copy"):
        run_import(src, dest, *paths(dest))
    assert sorted(p.name for p in dest.iterdir()) == ["labelled-fixtures.sha256", "labelled_exceptions.yaml"]


# --- exceptions -----------------------------------------------------------------------------------------------------------------


def test_a_named_exception_lets_one_existing_file_differ(world, capsys):
    src, dest = world
    write_case(dest, "alpha_case", {**CASES["alpha_case"], "pib.xml": b"appended to\n"})
    paths(dest)[1].write_text("- path: alpha_case/pib.xml\n  reason: invented reason for the test\n", encoding="utf-8")
    assert main(["--src", str(src), "--dest", str(dest)]) == 0
    out = capsys.readouterr().out
    assert "1 named exception(s)" in out and "alpha_case/pib.xml: invented reason for the test" in out
    assert (dest / "alpha_case" / "pib.xml").read_bytes() == b"appended to\n"


def test_an_exception_for_a_file_that_matches_is_a_stale_exception_and_fails(world, capsys):
    src, dest = world
    paths(dest)[1].write_text("- path: alpha_case/pib.xml\n  reason: nothing differs any more\n", encoding="utf-8")
    assert main(["--src", str(src), "--dest", str(dest)]) == 1
    err = capsys.readouterr().err
    assert "alpha_case/pib.xml: stale_exception" in err and "remove it" in err


def test_the_exception_does_not_cover_a_second_differing_file(world, capsys):
    src, dest = world
    write_case(dest, "alpha_case", {**CASES["alpha_case"], "pib.xml": b"appended to\n", "case.yaml": b"also edited\n"})
    paths(dest)[1].write_text("- path: alpha_case/pib.xml\n  reason: invented\n", encoding="utf-8")
    assert main(["--src", str(src), "--dest", str(dest)]) == 1
    err = capsys.readouterr().err
    assert "alpha_case/case.yaml: differs" in err and "alpha_case/pib.xml" not in err


def test_main_prints_a_clear_failure_for_a_bad_list_line_with_its_number(world, capsys):
    src, dest = world
    paths(dest)[0].write_text("nonsense\n", encoding="utf-8")
    assert main(["--src", str(src), "--dest", str(dest)]) == 1
    assert "line 1" in capsys.readouterr().err


def test_explicit_list_paths_are_used(world, tmp_path, capsys):
    src, dest = world
    other = tmp_path / "elsewhere"
    other.mkdir()
    (other / "s.sha256").write_text(paths(dest)[0].read_text(encoding="utf-8"), encoding="utf-8")
    (other / "e.yaml").write_text("[]\n", encoding="utf-8")
    paths(dest)[0].unlink()
    paths(dest)[1].unlink()
    assert main(["--src", str(src), "--dest", str(dest), "--sums", str(other / "s.sha256"), "--exceptions", str(other / "e.yaml")]) == 0


# --- F8: the source is only read ------------------------------------------------------------------------------------------------


def test_a_read_only_source_is_copied_and_not_changed(world):
    src, dest = world
    watched = [src, *src.rglob("*")]
    names = sorted(p.relative_to(src).as_posix() for p in src.rglob("*"))
    before = {p: (p.stat().st_mtime_ns, p.read_bytes() if p.is_file() else None) for p in watched}
    for path in watched:
        path.chmod(0o555 if path.is_dir() else 0o444)
    try:
        run_import(src, dest, *paths(dest))
        after = {p: (p.stat().st_mtime_ns, p.read_bytes() if p.is_file() else None) for p in watched}
        listing = sorted(p.relative_to(src).as_posix() for p in src.rglob("*"))
    finally:
        for path in watched:
            path.chmod(0o755 if path.is_dir() else 0o644)
    assert after == before
    assert listing == names
    assert (dest / "beta_case" / "pib.xml").read_bytes() == CASES["beta_case"]["pib.xml"]
