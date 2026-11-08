"""evals/agent/checksums.py: reading shasum lists and checking a folder against one (5a PR-1.S1, E3).

Every list and folder here is written into a tmp dir by the test, with invented case names.
"""

import hashlib
from pathlib import Path

import pytest

from evals.agent.checksums import (
    DIFFERS,
    MISSING,
    STALE_EXCEPTION,
    UNLISTED,
    UNREADABLE,
    ChecksumError,
    check_labelled,
    compare_fixtures,
    read_exceptions,
    read_sums,
    sha256_file,
    stale_exceptions,
    verify,
)


def digest(body: bytes) -> str:
    return hashlib.sha256(body).hexdigest()


def write(root: Path, relative: str, body: bytes) -> str:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(body)
    return digest(body)


def sums_file(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "list.sha256"
    path.write_text(text, encoding="utf-8")
    return path


A = "a" * 64
B = "b" * 64


# --- read_sums ------------------------------------------------------------------------------------------------------------------


def test_read_sums_accepts_the_dot_slash_form_and_the_bare_name(tmp_path):
    path = sums_file(tmp_path, f"{A}  ./alpha_case/case.yaml\n{B}  examples.yaml\n")
    assert read_sums(path) == {"alpha_case/case.yaml": A, "examples.yaml": B}


def test_read_sums_skips_blank_lines_and_lowercases_upper_case_hex(tmp_path):
    path = sums_file(tmp_path, f"\n{A.upper()}  ./alpha_case/case.yaml\n   \n\n{B}  examples.yaml\n\n")
    assert read_sums(path) == {"alpha_case/case.yaml": A, "examples.yaml": B}


def test_read_sums_accepts_shasums_binary_marker(tmp_path):
    assert read_sums(sums_file(tmp_path, f"{A} *examples.yaml\n")) == {"examples.yaml": A}


def test_read_sums_keeps_a_path_with_spaces_whole(tmp_path):
    assert read_sums(sums_file(tmp_path, f"{A}  ./alpha case/a file.yaml\n")) == {"alpha case/a file.yaml": A}


@pytest.mark.parametrize(
    "line",
    [
        "not a checksum line",
        f"{A}",  # no path
        f"{A} examples.yaml",  # one space and no marker
        f"{A[:-1]}  examples.yaml",  # 63 hex digits
        f"{'g' * 64}  examples.yaml",  # not hex
        f"{A}  ",  # an empty path
        f"{A}  /etc/passwd",  # leaves the root
        f"{A}  ../outside.yaml",
        f"{A}  ./alpha_case/../../outside.yaml",
    ],
)
def test_read_sums_names_the_line_number_of_a_malformed_line(tmp_path, line):
    path = sums_file(tmp_path, f"{B}  good.yaml\n\n{line}\n")
    with pytest.raises(ChecksumError) as error:
        read_sums(path)
    assert "line 3" in str(error.value) and "list.sha256" in str(error.value)


def test_read_sums_rejects_a_path_listed_twice_naming_the_second_line(tmp_path):
    with pytest.raises(ChecksumError, match=r"line 2.*second time"):
        read_sums(sums_file(tmp_path, f"{A}  x.yaml\n{B}  ./x.yaml\n"))


@pytest.mark.parametrize("text", ["", "\n\n   \n"])
def test_read_sums_rejects_an_empty_list(tmp_path, text):
    with pytest.raises(ChecksumError, match="lists no files"):
        read_sums(sums_file(tmp_path, text))


def test_read_sums_rejects_a_missing_file_naming_it(tmp_path):
    with pytest.raises(ChecksumError, match=r"list\.sha256.*missing"):
        read_sums(tmp_path / "list.sha256")


def test_read_sums_rejects_bytes_that_are_not_text(tmp_path):
    path = tmp_path / "list.sha256"
    path.write_bytes(b"\xff\xfe\x00")
    with pytest.raises(ChecksumError, match="cannot be read"):
        read_sums(path)


# --- verify ---------------------------------------------------------------------------------------------------------------------


def test_verify_returns_nothing_for_a_folder_that_matches(tmp_path):
    sums = {"alpha_case/case.yaml": write(tmp_path, "alpha_case/case.yaml", b"one"), "examples.yaml": write(tmp_path, "examples.yaml", b"two")}
    assert verify(tmp_path, sums) == []


def test_verify_reports_one_row_per_mismatch_kind(tmp_path):
    sums = {
        "alpha_case/case.yaml": write(tmp_path, "alpha_case/case.yaml", b"unchanged"),
        "alpha_case/weather.json": write(tmp_path, "alpha_case/weather.json", b"was this"),
        "alpha_case/pib.xml": digest(b"never on disk"),
    }
    write(tmp_path, "alpha_case/weather.json", b"now that")  # differs
    write(tmp_path, "alpha_case/notes.txt", b"nobody listed me")  # unlisted
    assert verify(tmp_path, sums) == [
        ("alpha_case/notes.txt", UNLISTED),
        ("alpha_case/pib.xml", MISSING),
        ("alpha_case/weather.json", DIFFERS),
    ]


def test_verify_reports_every_mismatch_not_only_the_first(tmp_path):
    sums = {f"alpha_case/file{n}.yaml": write(tmp_path, f"alpha_case/file{n}.yaml", b"before") for n in range(4)}
    for n in range(4):
        write(tmp_path, f"alpha_case/file{n}.yaml", b"after")
    assert verify(tmp_path, sums) == [(f"alpha_case/file{n}.yaml", DIFFERS) for n in range(4)]


def test_verify_finds_an_unlisted_file_in_a_subfolder(tmp_path):
    sums = {"alpha_case/case.yaml": write(tmp_path, "alpha_case/case.yaml", b"x")}
    write(tmp_path, "alpha_case/deep/er/extra.json", b"y")
    assert verify(tmp_path, sums) == [("alpha_case/deep/er/extra.json", UNLISTED)]


def test_verify_does_not_scan_folders_or_top_level_files_the_sums_do_not_mention(tmp_path):
    sums = {"alpha_case/case.yaml": write(tmp_path, "alpha_case/case.yaml", b"x"), "examples.yaml": write(tmp_path, "examples.yaml", b"y")}
    write(tmp_path, "beta_case/case.yaml", b"another case, not in the list")
    write(tmp_path, "readme.txt", b"a loose top-level file")
    assert verify(tmp_path, sums) == []


def test_verify_a_listed_folder_that_is_gone_reports_each_file_missing(tmp_path):
    sums = {"alpha_case/a.yaml": A, "alpha_case/b.yaml": B}
    assert verify(tmp_path, sums) == [("alpha_case/a.yaml", MISSING), ("alpha_case/b.yaml", MISSING)]


def test_verify_reports_a_file_it_cannot_open_as_unreadable_and_goes_on(tmp_path):
    sums = {"alpha_case/a.yaml": write(tmp_path, "alpha_case/a.yaml", b"1"), "alpha_case/b.yaml": write(tmp_path, "alpha_case/b.yaml", b"2")}
    write(tmp_path, "alpha_case/b.yaml", b"changed")
    (tmp_path / "alpha_case" / "a.yaml").chmod(0)
    try:
        got = verify(tmp_path, sums)
        stale = stale_exceptions(tmp_path, sums, {"alpha_case/a.yaml"})
        report = compare_fixtures(tmp_path, sums, set(), {"alpha_case": {"a.yaml": digest(b"1")}})
    finally:
        (tmp_path / "alpha_case" / "a.yaml").chmod(0o644)
    assert got == [("alpha_case/a.yaml", UNREADABLE), ("alpha_case/b.yaml", DIFFERS)]
    assert stale == []
    assert report == ["alpha_case/a.yaml: unreadable against the sha256 list and does not match the sheet fingerprint", "alpha_case/b.yaml: differs against the sha256 list"]


def test_verify_a_directory_where_a_file_should_be_is_missing(tmp_path):
    (tmp_path / "alpha_case" / "case.yaml").mkdir(parents=True)
    assert verify(tmp_path, {"alpha_case/case.yaml": A}) == [("alpha_case/case.yaml", MISSING)]


def test_verify_lets_an_excepted_file_differ_but_not_go_missing(tmp_path):
    sums = {"alpha_case/readings.yaml": write(tmp_path, "alpha_case/readings.yaml", b"labelled"), "alpha_case/case.yaml": write(tmp_path, "alpha_case/case.yaml", b"keep")}
    write(tmp_path, "alpha_case/readings.yaml", b"appended to")
    assert verify(tmp_path, sums, {"alpha_case/readings.yaml"}) == []
    assert verify(tmp_path, sums) == [("alpha_case/readings.yaml", DIFFERS)]
    (tmp_path / "alpha_case" / "readings.yaml").unlink()
    assert verify(tmp_path, sums, {"alpha_case/readings.yaml"}) == [("alpha_case/readings.yaml", MISSING)]


def test_verify_an_exception_does_not_cover_another_file(tmp_path):
    sums = {"alpha_case/a.yaml": write(tmp_path, "alpha_case/a.yaml", b"1"), "alpha_case/b.yaml": write(tmp_path, "alpha_case/b.yaml", b"2")}
    write(tmp_path, "alpha_case/a.yaml", b"changed")
    write(tmp_path, "alpha_case/b.yaml", b"changed")
    assert verify(tmp_path, sums, {"alpha_case/a.yaml"}) == [("alpha_case/b.yaml", DIFFERS)]


def test_changing_one_byte_is_enough(tmp_path):
    body = bytes(range(256)) * 4
    sums = {"alpha_case/blob.bin": write(tmp_path, "alpha_case/blob.bin", body)}
    flipped = bytearray(body)
    flipped[500] ^= 1
    write(tmp_path, "alpha_case/blob.bin", bytes(flipped))
    assert verify(tmp_path, sums) == [("alpha_case/blob.bin", DIFFERS)]


# --- stale exceptions -----------------------------------------------------------------------------------------------------------


def test_an_exception_for_a_file_that_matches_is_stale(tmp_path):
    sums = {"alpha_case/readings.yaml": write(tmp_path, "alpha_case/readings.yaml", b"as labelled")}
    assert stale_exceptions(tmp_path, sums, {"alpha_case/readings.yaml"}) == ["alpha_case/readings.yaml"]
    assert check_labelled(tmp_path, sums, {"alpha_case/readings.yaml"}) == [("alpha_case/readings.yaml", STALE_EXCEPTION)]


def test_an_exception_for_a_file_the_list_does_not_hold_is_stale(tmp_path):
    sums = {"alpha_case/case.yaml": write(tmp_path, "alpha_case/case.yaml", b"x")}
    assert stale_exceptions(tmp_path, sums, {"alpha_case/ghost.yaml"}) == ["alpha_case/ghost.yaml"]


def test_an_exception_for_a_file_that_differs_is_not_stale(tmp_path):
    sums = {"alpha_case/readings.yaml": write(tmp_path, "alpha_case/readings.yaml", b"as labelled")}
    write(tmp_path, "alpha_case/readings.yaml", b"appended to")
    assert stale_exceptions(tmp_path, sums, {"alpha_case/readings.yaml"}) == []
    assert check_labelled(tmp_path, sums, {"alpha_case/readings.yaml"}) == []


def test_an_excepted_file_that_is_gone_is_missing_not_stale(tmp_path):
    sums = {"alpha_case/readings.yaml": A}
    assert stale_exceptions(tmp_path, sums, {"alpha_case/readings.yaml"}) == []
    assert check_labelled(tmp_path, sums, {"alpha_case/readings.yaml"}) == [("alpha_case/readings.yaml", MISSING)]


# --- read_exceptions ------------------------------------------------------------------------------------------------------------


def test_read_exceptions_returns_path_to_reason(tmp_path):
    path = tmp_path / "ex.yaml"
    path.write_text("# comment\n- path: ./alpha_case/readings.yaml\n  reason: appended a reading\n", encoding="utf-8")
    assert read_exceptions(path) == {"alpha_case/readings.yaml": "appended a reading"}


def test_read_exceptions_allows_an_empty_file_and_an_empty_list(tmp_path):
    for text in ("", "[]\n", "# only a comment\n"):
        path = tmp_path / "ex.yaml"
        path.write_text(text, encoding="utf-8")
        assert read_exceptions(path) == {}


@pytest.mark.parametrize(
    "text",
    [
        "path: a/b.yaml\nreason: not a list\n",
        "- just a string\n",
        "- path: a/b.yaml\n",  # no reason
        "- reason: no path\n",
        "- path: a/b.yaml\n  reason: ''\n",  # empty reason
        "- path: a/b.yaml\n  reason: r\n  extra: key\n",
        "- path: ../b.yaml\n  reason: leaves the root\n",
        "- path: a/b.yaml\n  reason: r\n- path: ./a/b.yaml\n  reason: again\n",
        "- path: [unclosed\n",
    ],
)
def test_read_exceptions_rejects_a_bad_list_naming_the_file(tmp_path, text):
    path = tmp_path / "ex.yaml"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ChecksumError, match=r"ex\.yaml"):
        read_exceptions(path)


def test_read_exceptions_rejects_a_missing_file(tmp_path):
    with pytest.raises(ChecksumError, match="missing"):
        read_exceptions(tmp_path / "ex.yaml")


# --- compare_fixtures -----------------------------------------------------------------------------------------------------------


def invented_case(root: Path) -> tuple[dict[str, str], dict[str, str]]:
    """(sums, fingerprint file parts) for an invented draft case whose files all match both."""
    names = ("weather.json", "airspace.json", "pib.xml", "readings.yaml")
    parts = {name: write(root, f"alpha_case/{name}", name.encode()) for name in names}
    sums = {f"alpha_case/{name}": value for name, value in parts.items()}
    sums["alpha_case/case.yaml"] = write(root, "alpha_case/case.yaml", b"case")
    return sums, parts


def test_compare_fixtures_is_empty_when_both_references_hold(tmp_path):
    sums, parts = invented_case(tmp_path)
    assert compare_fixtures(tmp_path, sums, set(), {"alpha_case": parts}) == []


def test_compare_fixtures_says_both_when_a_file_fails_the_list_and_the_fingerprint(tmp_path):
    sums, parts = invented_case(tmp_path)
    write(tmp_path, "alpha_case/pib.xml", b"one byte off")
    assert compare_fixtures(tmp_path, sums, set(), {"alpha_case": parts}) == ["alpha_case/pib.xml: differs against the sha256 list and does not match the sheet fingerprint"]


def test_compare_fixtures_says_the_list_only_for_a_file_the_sheet_does_not_hash(tmp_path):
    sums, parts = invented_case(tmp_path)
    write(tmp_path, "alpha_case/case.yaml", b"changed")
    assert compare_fixtures(tmp_path, sums, set(), {"alpha_case": parts}) == ["alpha_case/case.yaml: differs against the sha256 list"]


def test_compare_fixtures_says_the_fingerprint_only_when_the_list_excepts_the_file(tmp_path):
    sums, parts = invented_case(tmp_path)
    write(tmp_path, "alpha_case/readings.yaml", b"appended to")
    assert compare_fixtures(tmp_path, sums, {"alpha_case/readings.yaml"}, {"alpha_case": parts}) == ["alpha_case/readings.yaml: does not match the sheet fingerprint"]


def test_compare_fixtures_names_every_failing_file(tmp_path):
    sums, parts = invented_case(tmp_path)
    for name in ("weather.json", "airspace.json"):
        write(tmp_path, f"alpha_case/{name}", b"changed")
    got = compare_fixtures(tmp_path, sums, set(), {"alpha_case": parts})
    assert [line.split(":")[0] for line in got] == ["alpha_case/airspace.json", "alpha_case/weather.json"]


def test_compare_fixtures_accepts_an_absent_fingerprint_file_that_is_absent(tmp_path):
    sums, parts = invented_case(tmp_path)
    (tmp_path / "alpha_case" / "readings.yaml").unlink()
    del sums["alpha_case/readings.yaml"]
    parts["readings.yaml"] = "absent"
    assert compare_fixtures(tmp_path, sums, set(), {"alpha_case": parts}) == []


def test_compare_fixtures_reports_a_file_the_fingerprint_hashes_but_the_folder_lacks(tmp_path):
    sums, parts = invented_case(tmp_path)
    (tmp_path / "alpha_case" / "pib.xml").unlink()
    assert compare_fixtures(tmp_path, sums, set(), {"alpha_case": parts}) == ["alpha_case/pib.xml: missing against the sha256 list and does not match the sheet fingerprint"]


def test_sha256_file_is_the_digest_of_the_bytes(tmp_path):
    write(tmp_path, "f.bin", b"abc")
    assert sha256_file(tmp_path / "f.bin") == digest(b"abc")
