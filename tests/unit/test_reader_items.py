"""G11 / GP8: the selection rules and the shape of the labelling sheet, evals/reader/items.yaml.

The script runs in-process under the suite's `--disable-socket` against a tmp output file, and its output is compared with the
committed items.yaml, so the sheet cannot drift from the code. F3 (no real number anywhere it should not be) and F4 (no labels)
are checked here too.
"""

import ast
import importlib.util
import logging
import re
import subprocess
import sys
from pathlib import Path

import pytest
import yaml
from factories import notam, utc
from phone_leaks import digit_runs, leak_list

from flysafelz.contracts import Circle
from flysafelz.pib import read_notams
from flysafelz.redact import e_hash

ROOT = Path(__file__).resolve().parents[2]
READER = ROOT / "evals" / "reader"
SCRIPT = ROOT / "scripts" / "build_reader_items.py"

_spec = importlib.util.spec_from_file_location("build_reader_items", SCRIPT)
assert _spec and _spec.loader
script = importlib.util.module_from_spec(_spec)
sys.modules["build_reader_items"] = script
_spec.loader.exec_module(script)

LABEL_KEYS = [
    "category",
    "activity",
    "floor",
    "ceiling",
    "applies_to",
    "exemptions",
    "contact",
    "activates_designators",
    "prohibitive_reason",
]


@pytest.fixture(scope="module")
def committed():
    return yaml.safe_load((READER / "items.yaml").read_text(encoding="utf-8"))["items"]


@pytest.fixture(scope="module")
def selection():
    return yaml.safe_load((READER / "selection.yaml").read_text(encoding="utf-8"))["items"]


def run(*args: str) -> int:
    return script.main(list(args))


def write_selection(path: Path, items: list[dict]) -> Path:
    path.write_text(yaml.safe_dump({"items": items}, sort_keys=False), encoding="utf-8")
    return path


# --- running the script ------------------------------------------------------------------------


def test_the_script_writes_the_committed_items_yaml_byte_for_byte(tmp_path, capsys):
    out = tmp_path / "items.yaml"
    assert run("--out", str(out)) == 0
    assert out.read_bytes() == (READER / "items.yaml").read_bytes()
    printed = capsys.readouterr().out
    assert "40 items written" in printed
    assert "0 phone leaks" in printed


def test_two_runs_give_a_byte_identical_sheet(tmp_path):
    first, second = tmp_path / "one.yaml", tmp_path / "two" / "items.yaml"
    assert run("--out", str(first)) == 0
    assert run("--out", str(second)) == 0
    assert first.read_bytes() == second.read_bytes()


def test_the_script_works_from_any_directory_and_needs_no_network(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert run("--out", str(tmp_path / "items.yaml")) == 0  # the suite runs with --disable-socket: a connection would raise


def test_the_script_runs_as_a_program(tmp_path):
    out = tmp_path / "items.yaml"
    done = subprocess.run([sys.executable, str(SCRIPT), "--out", str(out)], capture_output=True, text=True, cwd=tmp_path, check=False)
    assert done.returncode == 0, done.stderr
    assert "40 items written" in done.stdout
    assert out.read_bytes() == (READER / "items.yaml").read_bytes()


# --- GP8: the sheet ----------------------------------------------------------------------------


def test_forty_items_in_rank_order_matching_the_selection(committed, selection):
    assert [item["rank"] for item in committed] == list(range(1, 41))
    assert [(i["rank"], i["sha"], i["id"], i["type"]) for i in committed] == [(s["rank"], s["sha"], s["id"], s["type"]) for s in selection]


def test_ranks_1_to_24_cover_each_type_once_and_25_to_40_repeat_the_first_16_in_order(committed):
    types = [item["type"] for item in committed]
    assert sorted(types[:24]) == sorted(script.TYPES)
    assert len(set(types[:24])) == 24
    assert types[24:] == types[:16]


def test_the_selection_is_vectors_as_supplied():
    """Pins the supplied selection: rank, id and commit. A change to it is a change to the eval, not to this PR."""
    entries = yaml.safe_load((READER / "selection.yaml").read_text(encoding="utf-8"))["items"]
    spec_start = {1: ("J1458/26", "fc03e3b9"), 5: ("H7171/26", "bdab1465"), 10: ("H5759/26", "27e0709e"), 15: ("B1042/26", "e688158a"), 40: ("Q73/26", "bdab1465")}
    for rank, (notam_id, short) in spec_start.items():
        entry = entries[rank - 1]
        assert (entry["id"], entry["sha"][:8]) == (notam_id, short)
    assert {entry["sha"][:8] for entry in entries} == {"fc03e3b9", "e688158a", "27e0709e", "bdab1465"}


def test_every_item_has_its_redacted_e_coded_fields_area_hash_and_label(committed):
    for item in committed:
        assert list(item) == ["rank", "id", "sha", "type", "e", "phones_redacted", "coded", "area", "e_hash", "label"], item["id"]
        assert item["e"].strip()
        assert list(item["coded"]) == ["q_line", "a", "b", "c", "d", "f", "g"], item["id"]
        assert set(item["coded"]["q_line"]) == {"code", "fir", "traffic", "purpose", "scope", "lower_fl", "upper_fl", "lat", "lon", "radius_nm"}
        assert item["coded"]["a"]
        assert item["coded"]["b"]
        assert item["area"]["kind"] in {"circle", "polygon", "q_circle"}
        assert item["area"]["source"]
        assert re.fullmatch(r"[0-9a-f]{64}", item["e_hash"])


def test_the_e_hash_is_the_hash_of_the_raw_e_text_in_the_committed_pib(committed):
    pibs = {}
    for item in committed:
        if item["sha"] not in pibs:
            pibs[item["sha"]] = {n.id: n for n in read_notams((READER / "pibs" / f"{item['sha']}.xml").read_bytes())[1]}
        raw = pibs[item["sha"]][item["id"]].item_e
        assert item["e_hash"] == e_hash(raw), item["id"]


def test_the_coded_fields_are_the_pibs_own(committed):
    by_id = {}
    for sha in {item["sha"] for item in committed}:
        by_id[sha] = {n.id: n for n in read_notams((READER / "pibs" / f"{sha}.xml").read_bytes())[1]}
    for item in committed:
        parsed = by_id[item["sha"]][item["id"]]
        q = item["coded"]["q_line"]
        assert (q["code"], q["fir"], q["lower_fl"], q["upper_fl"], q["radius_nm"]) == (parsed.q_code, parsed.fir, parsed.lower_fl, parsed.upper_fl, parsed.radius_nm)
        assert item["coded"]["a"] == parsed.item_a
        assert item["coded"]["b"] == parsed.valid_from.strftime("%Y-%m-%dT%H:%MZ")
        assert (item["coded"]["d"] or "").split() == (parsed.item_d or "").split()
        assert (item["coded"]["f"] or "").split() == (parsed.item_f or "").split()
        assert (item["coded"]["g"] or "").split() == (parsed.item_g or "").split()


def test_the_area_shown_is_the_one_build_area_gives(committed):
    h7171 = next(item for item in committed if item["id"] == "H7171/26")
    assert h7171["area"]["kind"] == "circle"
    assert h7171["area"]["lat"] == pytest.approx(53.79222, abs=1e-4)
    assert h7171["area"]["lon"] == pytest.approx(-0.08306, abs=1e-4)
    assert h7171["area"]["radius_nm"] == 0.5
    kinds = {item["area"]["kind"] for item in committed}
    assert kinds == {"circle", "polygon", "q_circle"}  # all three answers are on the sheet
    polygon = next(item for item in committed if item["area"]["kind"] == "polygon")
    assert len(polygon["area"]["points"]) >= 3 and all(len(point) == 2 for point in polygon["area"]["points"])
    q_circle = next(item for item in committed if item["area"]["kind"] == "q_circle")
    assert (q_circle["area"]["lat"], q_circle["area"]["lon"], q_circle["area"]["radius_nm"]) == (
        q_circle["coded"]["q_line"]["lat"],
        q_circle["coded"]["q_line"]["lon"],
        q_circle["coded"]["q_line"]["radius_nm"],
    )


def test_a_metres_radius_item_shows_a_converted_circle(committed):
    item = next(item for item in committed if item["type"] == "radius_in_metres")
    assert item["area"]["kind"] == "circle"
    assert 0 < item["area"]["radius_nm"] < 1


def test_every_e_text_is_redacted_to_phone_placeholders(committed):
    with_phone = [item for item in committed if item["phones_redacted"]]
    assert len(with_phone) >= 5
    for item in with_phone:
        assert re.search(r"<PHONE_\d+>", item["e"] + " ".join(item["coded"][k] or "" for k in "dfg")), item["id"]
    drone = next(item for item in committed if item["type"] == "drone_ops_with_phone")
    assert "<PHONE_1>" in drone["e"]  # a contact is present in E; no number is asserted


def test_placeholders_are_numbered_from_one_per_notam(committed):
    for item in with_phones(committed):
        used = sorted({int(n) for n in re.findall(r"<PHONE_(\d+)>", item["e"])})
        assert used == list(range(1, len(used) + 1)), item["id"]


def with_phones(items):
    return [item for item in items if item["phones_redacted"]]


def test_blank_labels_with_exactly_the_nine_keys_and_no_geometry_key(committed):
    for item in committed:
        assert list(item["label"]) == LABEL_KEYS, item["id"]
        assert all(value is None for value in item["label"].values()), item["id"]
    text = (READER / "items.yaml").read_text(encoding="utf-8")
    assert not re.search(r"^\s*-?\s*geometry\s*:", text, re.MULTILINE)


def test_the_header_states_the_datum_rule():
    header = (READER / "items.yaml").read_text(encoding="utf-8").split("\nitems:")[0]
    assert "when E gives a height twice" in header
    assert "label the AMSL value" in header
    assert all(line.startswith("#") or not line for line in header.splitlines())


def test_the_sheet_is_one_yaml_document_that_reads_back(committed):
    text = (READER / "items.yaml").read_text(encoding="utf-8")
    assert len(list(yaml.safe_load_all(text))) == 1
    assert text.endswith("\n") and not text.endswith("\n\n")
    assert "\r" not in text


# --- F3: no real number in the sheet, the script's output or its log ---------------------------------


def test_no_real_number_is_in_items_yaml_the_scripts_stdout_or_its_log(tmp_path, capsys, caplog):
    leaks = leak_list()
    assert len(leaks) > 100  # the list is real: every 10-11 digit run in the 4 PIBs
    with caplog.at_level(logging.DEBUG):
        assert run("--out", str(tmp_path / "items.yaml")) == 0
    captured = capsys.readouterr()
    haystacks = {
        "items.yaml": (READER / "items.yaml").read_text(encoding="utf-8"),
        "the new sheet": (tmp_path / "items.yaml").read_text(encoding="utf-8"),
        "stdout": captured.out,
        "stderr": captured.err,
        "log": "\n".join(record.getMessage() for record in caplog.records),
    }
    for where, text in haystacks.items():
        squeezed = re.sub(r"\s+", "", text)
        found = sorted(number for number in leaks if number in squeezed)
        assert not found, f"{len(found)} number(s) from the leak list are in {where}"  # the count, never the numbers


def test_no_real_number_is_in_the_source_scripts_tests_or_docs():
    """Nothing this PR adds holds a real number. Real numbers live in the committed PIB and fixture XML; docs and tests use Ofcom's drama-range numbers."""
    leaks = leak_list()
    new_tests = ["test_redact", "test_notam_area", "test_notam_area_corpus", "test_reader_items", "phone_leaks", "fence_scan"]
    files = [ROOT / "README.md", ROOT / "Makefile", *ROOT.glob("src/**/*.py"), *ROOT.glob("src/**/*.yaml"), *ROOT.glob("scripts/*.py"), *ROOT.glob("docs/**/*.md")]
    files += [ROOT / "tests" / "unit" / f"{name}.py" for name in new_tests] + list(READER.glob("*.yaml"))  # ticket 2's test data is not this PR's
    assert len(files) > 20
    for path in files:
        squeezed = re.sub(r"\s+", "", path.read_text(encoding="utf-8"))
        found = [number for number in leaks if number in squeezed]
        assert not found, f"{len(found)} number(s) from the leak list are in {path.relative_to(ROOT)}"


def test_the_leak_list_is_independent_of_the_redactor():
    tree = ast.parse((Path(__file__).parent / "phone_leaks.py").read_text(encoding="utf-8"))
    imported = [node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
    imported += [alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names]
    assert not [name for name in imported if "redact" in name]
    assert digit_runs("CTC 07700 900123 OR 01632\n960001") == ["07700900123", "01632960001"]
    assert digit_runs("FROM 2610311200 TO 2610311800") == []  # date-time groups parse as YYMMDDhhmm
    assert digit_runs("119.130 534732N 0000459W 700FT") == []


def test_the_leak_list_holds_the_numbers_the_sheet_redacted(committed):
    leaks = leak_list()
    redacted_ids = {item["id"] for item in committed if item["phones_redacted"]}
    in_list = {notam_id for ids in leaks.values() for notam_id in ids}
    assert redacted_ids <= in_list  # every NOTAM the redactor changed has a number on the independent list too


def test_the_script_refuses_to_write_a_sheet_with_a_number_left_in_it(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(script, "redact_phones", lambda text: (text, {}))  # a redactor that finds nothing
    out = tmp_path / "items.yaml"
    assert run("--out", str(out)) == 1
    assert "still in the sheet" in capsys.readouterr().err
    assert not out.exists()


# --- F4: no labels ------------------------------------------------------------------------------


def test_the_script_never_names_a_labels_file():
    # No check that labels.yaml is absent from the tree: Dan adds evals/reader/labels.yaml at Review, and
    # make test must keep passing then. The guard is that this script never reads or writes one.
    assert "labels.yaml" not in SCRIPT.read_text(encoding="utf-8")


def test_the_script_writes_only_the_out_file(tmp_path):
    before = sorted(path.relative_to(ROOT) for path in READER.rglob("*"))
    assert run("--out", str(tmp_path / "items.yaml")) == 0
    assert sorted(path.relative_to(ROOT) for path in READER.rglob("*")) == before
    assert [path.name for path in tmp_path.iterdir()] == ["items.yaml"]  # no temp file left


# --- selection errors: non-zero exit, the item named, no partial file ----------------------------------


def broken(tmp_path, selection, change):
    entries = [dict(entry) for entry in selection]
    change(entries)
    return write_selection(tmp_path / "selection.yaml", entries)


def fails(tmp_path, selection_path, capsys, out=None):
    out = out or tmp_path / "out" / "items.yaml"
    assert run("--selection", str(selection_path), "--pibs", str(READER / "pibs"), "--out", str(out)) == 1
    assert not out.exists()  # never a partial items.yaml
    assert not list(out.parent.glob("*")) if out.parent.exists() else True
    return capsys.readouterr().err


def test_a_duplicate_rank_names_the_item(tmp_path, selection, capsys):
    path = broken(tmp_path, selection, lambda e: e[6].update(rank=3))
    err = fails(tmp_path, path, capsys)
    assert "duplicate rank 3" in err
    assert "H1910/26" in err


def test_a_gap_in_the_ranks_names_the_missing_rank(tmp_path, selection, capsys):
    path = broken(tmp_path, selection, lambda e: e.pop(11))
    err = fails(tmp_path, path, capsys)
    assert "rank 12" in err


def test_a_rank_outside_1_to_40_names_the_item(tmp_path, selection, capsys):
    path = broken(tmp_path, selection, lambda e: e[39].update(rank=41))
    err = fails(tmp_path, path, capsys)
    assert "Q73/26" in err and "1 to 40" in err


def test_a_zero_rank_and_a_text_rank_are_refused(tmp_path, selection, capsys):
    zero = broken(tmp_path, selection, lambda e: e[0].update(rank=0))
    assert "J1458/26" in fails(tmp_path, zero, capsys)
    text = broken(tmp_path, selection, lambda e: e[0].update(rank="1"))
    assert "J1458/26" in fails(tmp_path, text, capsys)


def test_an_unknown_type_names_the_item(tmp_path, selection, capsys):
    path = broken(tmp_path, selection, lambda e: e[4].update(type="drone_ops_with_phones"))
    err = fails(tmp_path, path, capsys)
    assert "H7171/26" in err and "unknown type" in err


def test_a_notam_missing_from_its_committed_pib_names_the_item(tmp_path, selection, capsys):
    path = broken(tmp_path, selection, lambda e: e[4].update(id="H9999/26"))
    err = fails(tmp_path, path, capsys)
    assert "H9999/26" in err and "rank 5" in err and "bdab1465" in err


def test_a_notam_in_a_different_commits_pib_is_missing_too(tmp_path, selection, capsys):
    # H7171/26 is in the 31 Oct PIB only; asking for it in the 4 Jul PIB is a (sha, id) pair that is not committed.
    path = broken(tmp_path, selection, lambda e: e[4].update(sha=selection[0]["sha"]))
    err = fails(tmp_path, path, capsys)
    assert "H7171/26" in err and "fc03e3b9" in err


def test_a_commit_with_no_committed_pib_names_the_commit(tmp_path, selection, capsys):
    path = broken(tmp_path, selection, lambda e: e[7].update(sha="0" * 40))
    err = fails(tmp_path, path, capsys)
    assert "00000000" in err and "not committed" in err


def test_a_short_sha_is_refused(tmp_path, selection, capsys):
    path = broken(tmp_path, selection, lambda e: e[7].update(sha="fc03e3b9"))
    assert "H2986/26" in fails(tmp_path, path, capsys)


def test_the_same_notam_twice_is_refused(tmp_path, selection, capsys):
    path = broken(tmp_path, selection, lambda e: e[7].update(id=e[6]["id"]))
    err = fails(tmp_path, path, capsys)
    assert "H1910/26" in err and "already rank 7" in err


def test_fewer_than_forty_items_is_refused(tmp_path, selection, capsys):
    path = broken(tmp_path, selection, lambda e: e.__delitem__(slice(30, None)))
    assert "no item for rank 31" in fails(tmp_path, path, capsys)


def test_an_unreadable_or_malformed_selection_is_refused(tmp_path, capsys):
    missing = tmp_path / "nope.yaml"
    assert "cannot read the selection" in fails(tmp_path, missing, capsys)
    bad = tmp_path / "bad.yaml"
    bad.write_text("items: [unclosed", encoding="utf-8")
    assert "cannot read the selection" in fails(tmp_path, bad, capsys)
    wrong = tmp_path / "wrong.yaml"
    wrong.write_text("rows: []", encoding="utf-8")
    assert "no `items:` list" in fails(tmp_path, wrong, capsys)
    extra = write_selection(tmp_path / "extra.yaml", [{"rank": 1, "id": "J1458/26"}])
    assert "needs exactly rank, sha, id and type" in fails(tmp_path, extra, capsys)


def test_a_failed_run_leaves_an_existing_sheet_as_it_was(tmp_path, selection, capsys):
    out = tmp_path / "items.yaml"
    out.write_text("# the previous sheet\n", encoding="utf-8")
    path = broken(tmp_path, selection, lambda e: e[4].update(id="H9999/26"))
    assert run("--selection", str(path), "--out", str(out)) == 1
    assert out.read_text(encoding="utf-8") == "# the previous sheet\n"
    assert [p.name for p in tmp_path.glob("items.yaml*")] == ["items.yaml"]
    assert "H9999/26" in capsys.readouterr().err


def test_a_pib_that_is_not_xml_is_refused_with_its_commit(tmp_path, selection, capsys):
    pibs = tmp_path / "pibs"
    pibs.mkdir()
    for sha in {entry["sha"] for entry in selection}:
        (pibs / f"{sha}.xml").write_bytes((READER / "pibs" / f"{sha}.xml").read_bytes())
    (pibs / f"{selection[0]['sha']}.xml").write_text("<not xml", encoding="utf-8")
    out = tmp_path / "items.yaml"
    assert run("--pibs", str(pibs), "--out", str(out)) == 1
    assert "fc03e3b9" in capsys.readouterr().err
    assert not out.exists()


# --- the yaml writer reads back what it was given -----------------------------------------------------

AWKWARD_TEXT = [
    "plain",
    "two\nlines",
    "  leading spaces\nsecond",
    "trailing   \nspaces",
    "tab\there\nline",
    "null",
    "true: yes\n# not a comment\n- not a list",
    "'quoted' \"both\" \\ back",
    "caf\u00e9 \u2013 \u4e2d\u6587\nsecond",
    "blank\n\nline between",
    "\nstarts with a newline",
    "ends with a newline\n",
    "carriage\r\nreturn",
    "nul\x00 and bell\x07",
    "line sep\u2028here",
    "{braces}: [brackets], | pipe > fold & anchor * alias ! tag % percent @ at ` tick",
]


@pytest.mark.parametrize("text", AWKWARD_TEXT, ids=range(len(AWKWARD_TEXT)))
def test_any_e_text_is_written_so_that_it_reads_back(text):
    entry = {"rank": 1, "sha": "a" * 40, "id": "H7171/26", "type": "tmz"}
    parsed = notam(item_e=text.replace("\x00", " "), circle=Circle(lat=53.8, lon=-0.08, radius_nm=1.0))
    item = script.build_item(entry, parsed)
    back = yaml.safe_load(script.render([item]))["items"][0]
    assert back == item
    assert back["e"] == "\n".join(line.rstrip() for line in parsed.item_e.split("\n"))  # only trailing spaces are dropped


def test_ids_and_hashes_that_look_like_other_types_stay_strings():
    entry = {"rank": 1, "sha": "1" * 40, "id": "H7171/26", "type": "tmz"}
    item = script.build_item(entry, notam(item_e="TEXT"))
    back = yaml.safe_load(script.render([item]))["items"][0]
    assert isinstance(back["sha"], str) and isinstance(back["e_hash"], str)
    assert back["sha"] == "1" * 40


def test_phones_in_the_d_and_f_and_g_fields_share_the_e_numbering():
    entry = {"rank": 1, "sha": "a" * 40, "id": "H7171/26", "type": "tmz"}
    parsed = notam(item_e="CTC 07700 900123.", item_d="CTC 01632 960001", item_f="SFC", item_g="CTC 07700 900123")
    item = script.build_item(entry, parsed)
    assert item["e"] == "CTC <PHONE_1>."
    assert item["coded"]["d"] == "CTC <PHONE_2>"
    assert item["coded"]["g"] == "CTC <PHONE_1>"
    assert item["coded"]["f"] == "SFC"
    assert item["phones_redacted"] == 2
    assert script.phone_leaks([item]) == []


def test_a_missing_d_f_g_stays_missing_and_an_empty_one_does_not_become_missing():
    entry = {"rank": 1, "sha": "a" * 40, "id": "H7171/26", "type": "tmz"}
    item = script.build_item(entry, notam(item_e="TEXT", item_d=None, item_f="", item_g=None))
    assert (item["coded"]["d"], item["coded"]["f"], item["coded"]["g"]) == (None, "", None)
    assert yaml.safe_load(script.render([item]))["items"][0]["coded"]["f"] == ""


def test_item_c_is_perm_the_end_time_or_an_estimate():
    entry = {"rank": 1, "sha": "a" * 40, "id": "H7171/26", "type": "tmz"}
    assert script.build_item(entry, notam(item_e="T", permanent=True, valid_to=None))["coded"]["c"] == "PERM"
    assert script.build_item(entry, notam(item_e="T", valid_to=utc(18, 0, 10, 11)))["coded"]["c"] == "2026-11-10T18:00Z"
    assert script.build_item(entry, notam(item_e="T", valid_to=utc(18, 0, 10, 11), est=True))["coded"]["c"] == "2026-11-10T18:00Z EST"
    assert script.build_item(entry, notam(item_e="T", valid_to=None))["coded"]["c"] is None


def test_the_script_check_finds_a_leftover_number_and_ignores_date_time_groups():
    item = {"e": "X 07700900123 Y", "coded": {"d": "2610311200", "f": None, "g": "AT 01632 960001"}}
    assert script.phone_leaks([item]) == ["07700900123", "01632960001"]
