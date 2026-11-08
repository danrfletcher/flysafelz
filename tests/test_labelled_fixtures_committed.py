"""G4 / E3 (5a PR-1.S1): Dan's 23 labelled case folders are committed byte for byte, and the sheet's draft fingerprints still match them.

Everything here reads committed files: fixtures/labelled-fixtures.sha256 (a verbatim copy of the source's list), fixtures/labelled_exceptions.yaml
(the one named file allowed to differ) and the repo's evals/agent/data/ copies of the dataset. Nothing is fetched, captured or read from outside the repo.
"""

import hashlib
import shutil
from pathlib import Path

import pytest
import yaml

from evals.agent.checksums import check_labelled, compare_fixtures, read_exceptions, read_sums, stale_exceptions, verify
from flysafelz.pib import parse_pib
from flysafelz.verdict_sheet_fingerprint import FIXTURE_FILES, MODE_DRAFT, file_parts, unseal

REPO = Path(__file__).resolve().parents[1]
FIXTURES = REPO / "fixtures"
DATA = REPO / "evals" / "agent" / "data"
SUMS = read_sums(FIXTURES / "labelled-fixtures.sha256")
EXCEPTIONS = read_exceptions(FIXTURES / "labelled_exceptions.yaml")
LABELLED = sorted({path.split("/")[0] for path in SUMS})
DRAFT_ENTRIES = 14
CASES_IN_EXAMPLES = 25


def load(name: str):
    return yaml.safe_load((DATA / name).read_text(encoding="utf-8"))


def case_of_each_example() -> dict[str, str]:
    return {example["id"]: example["fixtures"]["case"] for example in load("examples.yaml")}


def draft_fingerprints() -> dict[str, dict[str, str] | None]:
    """{entry id: its fixture-file hashes, or None when the stored fingerprint is unreadable}, for every draft entry of the repo's sheet."""
    drafts = {}
    for entry in load("verdict_sheet.yaml"):
        if entry.get("mode") == MODE_DRAFT:
            parts = unseal(entry.get("fingerprint"), MODE_DRAFT)
            drafts[entry["id"]] = None if parts is None else {name: parts[name] for name in FIXTURE_FILES}
    return drafts


def fingerprint_by_case() -> dict[str, dict[str, str]]:
    cases = case_of_each_example()
    by_case: dict[str, dict[str, str]] = {}
    for entry_id, parts in draft_fingerprints().items():
        assert parts is not None, f"{entry_id}: the stored fingerprint is unreadable (a hash or the seal was edited, or the parts are wrong)"
        case = cases[entry_id]
        assert by_case.setdefault(case, parts) == parts, f"{entry_id}: two draft entries on {case} disagree about its fixture files"
    return by_case


# --- the checksum list ------------------------------------------------------------------------------------------------------------


def test_the_list_is_the_sources_115_files_in_23_folders():
    assert len(SUMS) == 115
    assert len(LABELLED) == 23
    assert len((FIXTURES / "labelled-fixtures.sha256").read_text(encoding="utf-8").splitlines()) == 115


def test_every_listed_folder_is_committed_with_exactly_its_listed_files():
    for name in LABELLED:
        assert (FIXTURES / name).is_dir(), name
        assert sorted(p.name for p in (FIXTURES / name).iterdir()) == sorted(path.split("/", 1)[1] for path in SUMS if path.startswith(f"{name}/")), name


def test_the_only_exception_is_the_druridge_readings_the_h2483_append_changed():
    assert EXCEPTIONS == {"druridge_bay_2026-07-22/readings.yaml": "3b PR-8 appended the H2483 reading"}
    raw = yaml.safe_load((FIXTURES / "labelled_exceptions.yaml").read_text(encoding="utf-8"))
    assert [sorted(entry) for entry in raw] == [["path", "reason"]]


def test_all_labelled_folders_match_the_list_except_the_one_named_exception():
    problems = check_labelled(FIXTURES, SUMS, EXCEPTIONS)
    assert problems == [], "\n".join(f"{relative}: {kind}" for relative, kind in problems)
    # exactly one file differs when the exception is not allowed for, and it is the excepted one
    assert verify(FIXTURES, SUMS) == [("druridge_bay_2026-07-22/readings.yaml", "differs")]


def test_no_exception_is_stale():
    assert stale_exceptions(FIXTURES, SUMS, EXCEPTIONS) == []


def test_114_of_the_115_files_match_the_list_exactly():
    not_ok = {relative for relative, _kind in verify(FIXTURES, SUMS)}
    assert len(SUMS) - len(not_ok) == 114


# --- the draft fingerprints -------------------------------------------------------------------------------------------------------


def test_the_sheet_has_14_draft_entries_and_every_fingerprint_is_readable():
    drafts = draft_fingerprints()
    assert len(drafts) == DRAFT_ENTRIES
    unreadable = sorted(entry_id for entry_id, parts in drafts.items() if parts is None)
    assert unreadable == [], f"unreadable fingerprint: {unreadable}"


def test_every_draft_fingerprint_matches_the_committed_case_files():
    by_case = fingerprint_by_case()
    assert by_case
    assert compare_fixtures(FIXTURES, SUMS, EXCEPTIONS, by_case) == []
    for case, parts in by_case.items():
        assert file_parts(FIXTURES / case) == parts, case


def test_the_sheet_and_the_examples_name_the_same_entries_and_every_case_is_committed():
    cases = set(case_of_each_example().values())
    assert len(cases) == CASES_IN_EXAMPLES
    missing = sorted(case for case in cases if not (FIXTURES / case / "case.yaml").is_file())
    assert missing == []
    assert {entry["id"] for entry in load("verdict_sheet.yaml")} <= set(case_of_each_example())


# --- E3: one changed byte is named, with the reference it fails -----------------------------------------------------------------


def copy_of_a_draft_case(tmp_path: Path):
    """A tmp copy of one real draft case folder, the list limited to that folder, and its fingerprint parts."""
    case, parts = next((c, p) for c, p in sorted(fingerprint_by_case().items()) if c in LABELLED)
    shutil.copytree(FIXTURES / case, tmp_path / case)
    return case, parts, {path: digest for path, digest in SUMS.items() if path.startswith(f"{case}/")}


def flip_a_byte(path: Path) -> None:
    body = bytearray(path.read_bytes())
    body[len(body) // 2] ^= 1
    path.write_bytes(bytes(body))


@pytest.mark.parametrize("name", ["weather.json", "airspace.json", "pib.xml"])
def test_one_changed_byte_in_a_fixture_file_fails_naming_the_file_and_both_references(tmp_path, name):
    case, parts, sums = copy_of_a_draft_case(tmp_path)
    exceptions = {path for path in EXCEPTIONS if path.startswith(f"{case}/")}
    assert compare_fixtures(tmp_path, sums, exceptions, {case: parts}) == []  # the copy starts clean
    flip_a_byte(tmp_path / case / name)
    assert compare_fixtures(tmp_path, sums, exceptions, {case: parts}) == [f"{case}/{name}: differs against the sha256 list and does not match the sheet fingerprint"]


def test_one_changed_byte_in_case_yaml_is_a_list_mismatch_only_because_the_sheet_does_not_hash_it(tmp_path):
    case, parts, sums = copy_of_a_draft_case(tmp_path)
    flip_a_byte(tmp_path / case / "case.yaml")
    assert compare_fixtures(tmp_path, sums, set(), {case: parts}) == [f"{case}/case.yaml: differs against the sha256 list"]


def test_a_file_changed_in_step_with_a_re_stamped_list_still_fails_the_fingerprint(tmp_path):
    case, parts, sums = copy_of_a_draft_case(tmp_path)
    flip_a_byte(tmp_path / case / "pib.xml")
    sums = {**sums, f"{case}/pib.xml": hashlib.sha256((tmp_path / case / "pib.xml").read_bytes()).hexdigest()}
    assert compare_fixtures(tmp_path, sums, set(), {case: parts}) == [f"{case}/pib.xml: does not match the sheet fingerprint"]


def test_a_missing_and_an_extra_file_in_a_case_folder_are_reported(tmp_path):
    case, parts, sums = copy_of_a_draft_case(tmp_path)
    (tmp_path / case / "readings.yaml").unlink()
    (tmp_path / case / "extra.json").write_bytes(b"{}")
    assert compare_fixtures(tmp_path, sums, set(), {case: parts}) == [
        f"{case}/extra.json: unlisted against the sha256 list",
        f"{case}/readings.yaml: missing against the sha256 list and does not match the sheet fingerprint",
    ]


# --- Buttermere (Q6) --------------------------------------------------------------------------------------------------------------


def test_buttermere_24_sept_is_committed_on_its_pinned_archive_sha_and_its_pib_holds_h6217():
    case = yaml.safe_load((FIXTURES / "buttermere_2026-09-24" / "case.yaml").read_text(encoding="utf-8"))
    assert str(case["archive_sha"]).startswith("55bc107a")
    _issued, notams = parse_pib((FIXTURES / "buttermere_2026-09-24" / "pib.xml").read_bytes())
    assert "H6217/26" in {n.id for n in notams}


def test_dev_16_is_the_example_that_expects_it():
    assert case_of_each_example()["dev_16_buttermere"] == "buttermere_2026-09-24"
