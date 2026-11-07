"""G12, GP9, GP10: the runner. A fake chat model, the real reader, the real scoring, the real report; tmp_path for labels, runs and PIBs.

Run `pytest tests/unit/test_reader_eval_run.py -k full_run -s` to see what the command prints.
"""

import json
import os
import re
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import pytest
from cache_helpers import fixture_reply
from phone_leaks import digit_runs, leaked
from phone_leaks import leak_list as committed_leak_list
from reader_eval_run_helpers import FAKE_PHONE, FORBIDDING_LABEL, GOOD_LABEL, IDS, Down, build_sheet, chat, label_entry, synthetic_notams, write_labels
from reader_set_helpers import ScriptedChat, good_reply

from evals.reader_eval import run as runner
from evals.reader_eval.run import LABELS_PATH, USAGE_ERROR, main, shown
from flysafelz.reader.config import READER_VARIABLES
from flysafelz.reader.prompt import PROMPT_VERSION
from flysafelz.reader.settings import SETTINGS as READER_SETTINGS

ROOT = Path(__file__).resolve().parents[2]
H, J = "H7171/26", "J3857/26"
PARTIAL = {"category": "cooperative"}


def runner_now():
    return datetime(2026, 11, 7, 16, 0, tzinfo=UTC)


def labelled(tmp_path, *, third=PARTIAL, third_stale=False, first=GOOD_LABEL, second=GOOD_LABEL):
    """The synthetic sheet with X1/26 (first) and X2/26 (second) labelled and X3/26 given as `third` (None: not in the file)."""
    paths, hashes = build_sheet(tmp_path)
    entries = [label_entry("X1/26", hashes, first), label_entry("X2/26", hashes, second)]
    if third is not None:
        entries.append(label_entry("X3/26", hashes, third, stale=third_stale))
    write_labels(paths, entries)
    return paths, hashes


def read_report(paths):
    (folder,) = sorted(paths.runs.iterdir())
    return folder, json.loads((folder / "report.json").read_text(encoding="utf-8"))


def calls_for(model, notam_id):
    return model.calls.count(notam_id)


# --- what is read, and how often -----------------------------------------------------------------------------------------------------


def test_each_fully_labelled_item_is_read_three_times_and_the_others_are_skipped_and_counted(tmp_path, capsys):
    paths, _ = labelled(tmp_path, third=PARTIAL)
    model = chat()
    assert main(model, paths, runner_now()) == 0
    assert calls_for(model, "X1/26") == 3 and calls_for(model, "X2/26") == 3
    assert calls_for(model, "X3/26") == 0
    _, report = read_report(paths)
    assert report["runs"] == 3 and [i["id"] for i in report["items"]] == ["X1/26", "X2/26"] and all(i["reads"] == 3 for i in report["items"])
    assert report["skipped"]["partial"]["ids"] == ["X3/26"]
    out = capsys.readouterr().out
    assert "2 labelled items scored, 3 runs each (6 item-runs)" in out and "Skipped: 0 unlabelled, 1 partly labelled, 0 stale, 0 unknown label ids" in out


def test_one_complete_one_partial_one_stale_reads_the_complete_one_three_times(tmp_path, capsys):
    paths, hashes = build_sheet(tmp_path)
    write_labels(paths, [label_entry("X1/26", hashes, GOOD_LABEL), label_entry("X2/26", hashes, PARTIAL), label_entry("X3/26", hashes, GOOD_LABEL, stale=True)])
    model = chat()
    assert main(model, paths, runner_now()) == 0
    assert model.calls.count("X1/26") == 3 and "X2/26" not in model.calls and "X3/26" not in model.calls
    out = capsys.readouterr().out
    print(out)
    assert "X1/26: read 3 times" in out and "Skipped: 0 unlabelled, 1 partly labelled, 1 stale, 0 unknown label ids" in out
    assert "partial: X2/26" in out and "stale: X3/26" in out


def test_an_item_with_none_empty_and_null_answers_is_complete_and_scored(tmp_path):
    answers = {**GOOD_LABEL, "floor": None, "exemptions": [], "contact": None, "activates_designators": None, "prohibitive_reason": "none"}
    paths, _ = labelled(tmp_path, first=answers, second=PARTIAL, third=None)
    model = chat()
    assert main(model, paths, runner_now()) == 0
    assert calls_for(model, "X1/26") == 3 and calls_for(model, "X2/26") == 0
    assert read_report(paths)[1]["skipped"]["unlabelled"]["ids"] == ["X3/26"]


def test_runs_and_concurrency_come_from_the_readers_settings_by_name(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "READER_SETTINGS", READER_SETTINGS.model_copy(update={"runs": 2, "concurrency": 1}))
    paths, _ = labelled(tmp_path)
    model = chat()
    assert main(model, paths, runner_now()) == 0
    assert calls_for(model, "X1/26") == 2 and calls_for(model, "X2/26") == 2
    assert model.peak == 1
    assert read_report(paths)[1]["runs"] == 2


def test_at_most_three_calls_run_at_once_and_three_do_run_at_once(tmp_path):
    """The fake holds each of its first three calls until all three are in flight (a barrier), so reaching 3 is shown, and it counts the peak."""
    paths, hashes = build_sheet(tmp_path)
    write_labels(paths, [label_entry(i, hashes, GOOD_LABEL) for i in IDS])
    model = ScriptedChat(synthetic_notams(), rendezvous=3, pause_s=0.02)
    items = [runner.parsed_notam(item, paths.pibs) for item in runner.load_items(paths.items)]
    rows = runner.read_items(items, model, runner_now())
    assert model.peak == 3
    assert len(model.calls) == 9 and all(r.status.value == "read" for row in rows for r in row)


# --- the cache is bypassed -------------------------------------------------------------------------------------------------------------


def test_the_cache_is_never_read_or_written_and_no_sqlite_file_appears(tmp_path, monkeypatch):
    from flysafelz.reader import cached_read

    def boom(*args, **kwargs):
        raise AssertionError("the eval touched the reading cache")

    for name in ("lookup", "store", "record_failure", "was_failed"):
        monkeypatch.setattr(cached_read, name, boom)
    monkeypatch.chdir(tmp_path)
    paths, _ = labelled(tmp_path)
    assert main(chat(), paths, runner_now()) == 0
    _, report = read_report(paths)
    assert report["rates"]["validity"] == 1.0  # every run read, none came back unavailable because the cache raised
    assert not list(tmp_path.rglob("*.sqlite")) and not (tmp_path / "data").exists()


def test_an_earlier_reading_is_never_reused_whatever_the_prompt_version(tmp_path, monkeypatch):
    """Seed a cache with the same model, then run: every run is a fresh model call (none is `stored`) and the cache file is untouched."""
    from flysafelz.reader.cached_read import read_cached

    paths, _ = labelled(tmp_path)
    cache = tmp_path / "data" / "readings.sqlite"
    cache.parent.mkdir()
    seeding = chat()
    notam = synthetic_notams()[0]
    assert read_cached(notam, seeding, cache, runner_now()).result.status.value == "read"
    before = cache.read_bytes()
    monkeypatch.chdir(tmp_path)
    model = chat()
    assert main(model, paths, runner_now()) == 0
    assert calls_for(model, "X1/26") == 3  # not 0: a hit would have cost no call
    assert cache.read_bytes() == before
    _, report = read_report(paths)
    assert all("stored" not in i["statuses"] for i in report["items"])


def test_model_and_prompt_version_are_written_into_both_reports(tmp_path):
    paths, _ = labelled(tmp_path)
    assert main(chat(), paths, runner_now()) == 0
    folder, report = read_report(paths)
    assert (report["model"], report["prompt_version"]) == ("fake-reader-model", PROMPT_VERSION)
    assert f"Model fake-reader-model \u00b7 prompt version {PROMPT_VERSION}" in (folder / "report.md").read_text(encoding="utf-8")


# --- failures ---------------------------------------------------------------------------------------------------------------------------


def test_one_runs_unavailable_does_not_stop_the_others(tmp_path):
    paths, _ = labelled(tmp_path)
    model = chat(X1_26=[ConnectionError("down")])
    assert main(model, paths, runner_now()) == 0
    _, report = read_report(paths)
    statuses = {i["id"]: i["statuses"] for i in report["items"]}
    assert statuses["X1/26"] == {"unavailable": 1, "read": 2} and statuses["X2/26"] == {"read": 3}
    assert report["rates"]["unavailable"] == pytest.approx(1 / 6) and report["rates"]["validity"] == pytest.approx(5 / 6)


def test_a_read_that_raises_is_one_unavailable_run_not_the_end_of_the_eval(tmp_path, monkeypatch, caplog):
    real = runner.read_cached
    seen = []

    def sometimes_raises(parsed, llm, cache, now):
        seen.append(parsed.id)
        if len(seen) == 2:
            raise RuntimeError(f"the client broke while reading {FAKE_PHONE}")
        return real(parsed, llm, cache, now)

    monkeypatch.setattr(runner, "read_cached", sometimes_raises)
    paths, _ = labelled(tmp_path)
    assert main(chat(), paths, runner_now()) == 0
    _, report = read_report(paths)
    assert sum(i["statuses"].get("unavailable", 0) for i in report["items"]) == 1
    assert sum(i["statuses"].get("read", 0) for i in report["items"]) == 5
    assert "RuntimeError" in caplog.text and FAKE_PHONE not in caplog.text  # the warning names the error, and puts no number back


def test_an_endpoint_that_is_down_for_the_whole_run_still_writes_the_report_and_exits_0(tmp_path, capsys):
    paths, _ = labelled(tmp_path, first=FORBIDDING_LABEL, second=FORBIDDING_LABEL)
    assert main(Down(), paths, runner_now()) == 0
    folder, report = read_report(paths)
    assert report["rates"] == {"validity": 0.0, "repair": 0.0, "failure": 0.0, "unavailable": 1.0}
    assert report["harmful"]["item_runs"] == 6 and all(f["reader_failed"] for f in report["harmful"]["findings"])  # unread is scored as what the pilot is told
    assert report["blue_mill"]["ok"] is None and all("unavailable" in line for line in report["blue_mill"]["lines"])
    out = capsys.readouterr().out
    assert "(target 0)" in out and "H7171/26: unavailable" in out
    assert "(reader failed)" in (folder / "report.md").read_text(encoding="utf-8")


def test_harmful_downgrades_never_change_the_exit_code(tmp_path, capsys):
    paths, _ = labelled(tmp_path, first=FORBIDDING_LABEL)
    too_mild = {**good_reply(), "category": "informational"}  # the label forbids entry; this reading only asks for awareness: a harmful downgrade
    assert main(chat(X1_26=[too_mild] * 3), paths, runner_now()) == 0
    _, report = read_report(paths)
    assert report["harmful"]["items"] == 1 and report["harmful"]["item_runs"] == 3
    assert capsys.readouterr().out.startswith("Harmful downgrades: 1 item, 3 item-runs (target 0)")


def test_a_blue_mill_difference_does_not_change_the_exit_code_either(tmp_path):
    paths, _ = labelled(tmp_path)
    assert main(chat(J3857_26=[{**fixture_reply(J), "contact": {"kind": "none", "value": None}}]), paths, runner_now()) == 0
    assert read_report(paths)[1]["blue_mill"]["ok"] is False


# --- the inputs --------------------------------------------------------------------------------------------------------------------------


def test_a_missing_labels_file_gives_zero_labelled_items_and_still_runs_blue_mill(tmp_path, capsys):
    paths, _ = build_sheet(tmp_path)
    model = chat()
    assert main(model, paths, runner_now()) == 0
    folder, report = read_report(paths)
    assert not paths.labels.exists()  # F4: the run does not create it
    assert report["labels_file"] == {"path": str(paths.labels), "found": False}
    assert report["items"] == [] and sorted(model.calls) == [H, J]
    assert f"0 labelled items: {paths.labels} not found" in (folder / "report.md").read_text(encoding="utf-8")
    assert report["skipped"]["unlabelled"]["count"] == 3
    assert "H7171/26: AVOID_SECTOR live vs AVOID_SECTOR fixture" in capsys.readouterr().out


def test_the_default_labels_path_is_shown_as_the_repository_relative_path_of_the_spec():
    assert shown(LABELS_PATH) == "evals/reader/labels.yaml"
    assert shown(Path("/somewhere/else/labels.yaml")) == "/somewhere/else/labels.yaml"


@pytest.mark.parametrize(
    "text, names",
    [("labels: [unclosed", ["labels.yaml", "not valid YAML"]), ("labels: 3\n", ["labels.yaml", "must be a list"]), ("labels:\n  - id: X1/26\n", ["labels.yaml", "entry 1"])],
)
def test_a_labels_file_that_is_not_usable_exits_2_naming_it_and_scores_nothing(tmp_path, capsys, text, names):
    paths, _ = build_sheet(tmp_path)
    paths.labels.parent.mkdir(parents=True)
    paths.labels.write_text(text, encoding="utf-8")
    before = paths.labels.read_bytes()
    model = chat()
    assert main(model, paths, runner_now()) == USAGE_ERROR
    err = capsys.readouterr().err
    assert all(name in err for name in names)
    assert model.calls == [] and not paths.runs.exists()  # nothing read, no folder made
    assert paths.labels.read_bytes() == before


@pytest.mark.parametrize("key, value", [("category", "banana"), ("applies_to", "EVERYONE"), ("contact", "carrier pigeon"), ("prohibitive_reason", "because"), ("ceiling", "somewhere high")])
def test_a_value_the_contracts_do_not_allow_exits_2_naming_the_file_and_the_item(tmp_path, capsys, key, value):
    paths, hashes = build_sheet(tmp_path)
    write_labels(paths, [label_entry("X1/26", hashes, GOOD_LABEL), label_entry("X2/26", hashes, {**GOOD_LABEL, key: value})])
    model = chat()
    assert main(model, paths, runner_now()) == USAGE_ERROR
    err = capsys.readouterr().err
    assert str(paths.labels) in err and "X2/26" in err and key in err
    assert model.calls == [] and not paths.runs.exists()  # the good label was not scored on a half-read file


def test_a_blank_answer_is_unfilled_and_skipped_not_an_error(tmp_path):
    paths, _ = labelled(tmp_path, second={**GOOD_LABEL, "category": None}, third=None)
    assert main(chat(), paths, runner_now()) == 0
    assert read_report(paths)[1]["skipped"]["partial"]["ids"] == ["X2/26"]


def test_a_stale_label_and_an_unknown_one_are_listed_and_not_scored(tmp_path):
    paths, hashes = build_sheet(tmp_path)
    write_labels(paths, [label_entry("X1/26", hashes, GOOD_LABEL), label_entry("X2/26", hashes, GOOD_LABEL, stale=True), {"id": "Q7/26", "e_hash": "x", "label": GOOD_LABEL}])
    model = chat()
    assert main(model, paths, runner_now()) == 0
    report = read_report(paths)[1]
    assert report["skipped"]["stale"]["ids"] == ["X2/26"] and report["skipped"]["unknown"]["ids"] == ["Q7/26"] and report["skipped"]["unlabelled"]["ids"] == ["X3/26"]
    assert "X2/26" not in model.calls and "Q7/26" not in model.calls


def test_a_case_that_will_not_load_ends_the_run_before_the_model_is_called(tmp_path, capsys, monkeypatch):
    def broken(case):
        raise runner.SourceError("the case has no pib.xml")

    monkeypatch.setattr(runner, "load_case_sources", broken)
    paths, _ = labelled(tmp_path)
    model = chat()
    assert main(model, paths, runner_now()) == USAGE_ERROR
    assert "the case has no pib.xml" in capsys.readouterr().err and model.calls == [] and not paths.runs.exists()


# --- the settings ------------------------------------------------------------------------------------------------------------------------


def test_with_no_reader_settings_main_exits_2_naming_all_four_before_anything_else(tmp_path, capsys, monkeypatch):
    monkeypatch.chdir(tmp_path)
    paths, _ = labelled(tmp_path)
    assert main(None, paths, runner_now()) == USAGE_ERROR
    err = capsys.readouterr().err
    assert all(name in err for name in READER_VARIABLES)
    assert not paths.runs.exists()


@pytest.mark.parametrize("missing", READER_VARIABLES)
def test_with_one_setting_missing_only_that_one_is_named(tmp_path, capsys, monkeypatch, missing):
    for name in READER_VARIABLES:
        monkeypatch.setenv(name, "x")
    monkeypatch.delenv(missing)
    paths, _ = labelled(tmp_path)
    assert main(None, paths, runner_now()) == USAGE_ERROR
    err = capsys.readouterr().err
    assert missing in err and sum(name in err for name in READER_VARIABLES) == 1 and not paths.runs.exists()


def test_the_command_as_a_process_with_no_settings_exits_non_zero_names_them_and_makes_no_run_folder(tmp_path):
    runs = ROOT / "evals" / "reader" / "runs"
    before = sorted(p.name for p in runs.iterdir()) if runs.exists() else None
    env = {k: v for k, v in os.environ.items() if k not in READER_VARIABLES}
    env["READER_DOTENV_PATH"] = str(tmp_path / "no-such" / ".env")
    done = subprocess.run([sys.executable, "-m", "evals.reader_eval.run"], cwd=ROOT, env=env, capture_output=True, text=True, check=False, timeout=60)
    assert done.returncode != 0 and all(name in done.stderr for name in READER_VARIABLES)
    assert (sorted(p.name for p in runs.iterdir()) if runs.exists() else None) == before


# --- the Makefile -------------------------------------------------------------------------------------------------------------------------


def test_make_has_an_eval_reader_target_that_test_never_invokes():
    makefile = (ROOT / "Makefile").read_text(encoding="utf-8")
    phony = next(line for line in makefile.splitlines() if line.startswith(".PHONY"))
    assert "eval-reader" in phony.split()
    recipes = dict(re.findall(r"^([a-z-]+):[^\n]*\n((?:\t[^\n]*\n?)+)", makefile, re.MULTILINE))
    assert "evals.reader_eval.run" in recipes["eval-reader"]
    assert "eval-reader" not in recipes["test"] and "reader_eval.run" not in recipes["test"]
    assert "--cov=evals.reader_eval" in recipes["test"] and "--cov-fail-under=90" in recipes["test"]
    comment = makefile[: makefile.index("\neval-reader:")].rsplit("\n\n", 1)[-1]
    assert "only target that calls the model" in comment and "READER_*" in comment


def test_runs_are_ignored_by_git():
    done = subprocess.run(["git", "check-ignore", "-q", "evals/reader/runs/20261031T070000Z/report.md"], cwd=ROOT, check=False)
    assert done.returncode == 0


# --- labels.yaml is only read (F4) -------------------------------------------------------------------------------------------------------


def test_labels_yaml_is_byte_and_mtime_identical_after_a_run(tmp_path):
    paths, _ = labelled(tmp_path)
    before = (paths.labels.read_bytes(), paths.labels.stat().st_mtime_ns)
    assert main(chat(), paths, runner_now()) == 0
    assert (paths.labels.read_bytes(), paths.labels.stat().st_mtime_ns) == before


# --- no real number anywhere (F3) --------------------------------------------------------------------------------------------------------


def test_the_artifact_grep_catches_a_number_wrapped_across_a_line_break_or_split_by_several_spaces():
    leak_list = {"01632960003"}
    assert leaked("reason: CALL 01632\n960003 FIRST", leak_list) == leak_list
    assert leaked("reason: CALL 01632   960003 FIRST", leak_list) == leak_list
    assert leaked("reason: CALL <PHONE_1> FIRST", leak_list) == set()


def test_the_artifact_grep_catches_a_number_that_follows_other_digits():
    """The greedy 10-11 digit rule would join `13001632960` here and miss the number; a substring search does not."""
    leak_list = {"01632960003"}
    assert digit_runs("reason: CTC 124.600 01632960003") != ["01632960003"]  # why digit_runs is not the search
    assert leaked("reason: CTC 124.600 01632960003", leak_list) == leak_list
    assert leaked("CTC 124.600 01632 960003", leak_list) == leak_list
    assert leaked("SFC 2000 01632 960003", leak_list) == leak_list
    assert leaked("CTC 124.600 01632\n960003", leak_list) == leak_list
    assert leaked("CTC 124.600 <PHONE_1>", leak_list) == set()


def test_full_run_leaks_no_phone_number_into_report_md_report_json_or_stdout(tmp_path, capsys, caplog):
    """The leak list is every 10 to 11 digit run in the raw E text of the sheet's NOTAMs and of the two Blue Mill ones."""
    from cache_helpers import blue_mill_set

    paths, _ = labelled(tmp_path, third=GOOD_LABEL, first=FORBIDDING_LABEL)
    leak_list = set(committed_leak_list())
    for notam in [*synthetic_notams(), *blue_mill_set()]:
        leak_list |= set(digit_runs(notam.item_e))
    assert FAKE_PHONE.replace(" ", "") in leak_list and len(leak_list) >= 2  # the drama number and Blue Mill's own: the grep is not vacuous
    with_number = {"kind": "phone", "value": "Someone <PHONE_1>"}
    model = chat(X1_26=[{**good_reply(), "contact": with_number}] * 3, J3857_26=[{**fixture_reply(J), "contact": {"kind": "none", "value": None}}])
    assert main(model, paths, runner_now()) == 0
    folder, report = read_report(paths)
    artifacts = (folder / "report.md").read_text(encoding="utf-8") + (folder / "report.json").read_text(encoding="utf-8") + capsys.readouterr().out + caplog.text
    assert report["blue_mill"]["ok"] is False and report["harmful"]["item_runs"] >= 1  # the report does print reasons and findings
    assert leaked(artifacts, leak_list) == set()
    assert "<PHONE_1>" in artifacts
