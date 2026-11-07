"""G15: the report. Findings come from PR-7's real scoring (score_item_run); the rest is built from results with known statuses.

Run `pytest tests/unit/test_reader_eval_report.py -k heading -s` to see the report these tests render.
"""

import json
from fractions import Fraction

import pytest
from factories import amsl, reading, utc
from reader_eval_helpers import q_notam

from evals.reader_eval.downgrade import score_item_run
from evals.reader_eval.fields import field_accuracy
from evals.reader_eval.labels import LabelSet
from evals.reader_eval.report import ItemRuns, build_report, headline, nearest_rank, render_markdown, scored_line, summary_lines
from evals.reader_eval.run import make_run_folder, write_report
from flysafelz.contracts import Category
from flysafelz.reader.contract import ReaderResult, ReaderStatus, TokenUsage

NOTAM = q_notam(valid_from=utc(0), valid_to=utc(23), upper_fl=100)
LABEL = reading(category=Category.PROHIBITIVE, ceiling=amsl(2000))
TOO_MILD = reading(category=Category.INFORMATIONAL, ceiling=amsl(2000))
BLUE_MILL = {"case": "c", "segment": None, "heading": "c: the live check", "ok": True, "items": [], "lines": ["H7171/26: AVOID_SECTOR live vs AVOID_SECTOR fixture \u2713"]}
HALF = Fraction(1, 2)
P95 = Fraction(19, 20)


def result(status=ReaderStatus.READ, *, latency_s=0.0, tokens=None, failed_checks=(), notam_id="X0001/26", got=LABEL):
    has_reading = status in (ReaderStatus.READ, ReaderStatus.REPAIRED)
    return ReaderResult(
        notam_id=notam_id,
        reading=got if has_reading else None,
        status=status,
        failed_checks=list(failed_checks),
        model="fake-model",
        prompt_version="test",
        latency_s=latency_s,
        tokens=tokens,
    )


def item(notam_id, results, label=LABEL, parsed=NOTAM):
    return ItemRuns(
        notam_id,
        results,
        [score_item_run(parsed, label, r) for r in results],
        [field_accuracy(label, r.reading) for r in results],
    )


NO_LABELS = LabelSet((), ())


def report_for(items, *, runs=3, labels=NO_LABELS, found=True):
    return build_report(model="fake-model", prompt_version="test", runs=runs, items=items, labels=labels, labels_file="evals/reader/labels.yaml", labels_found=found, blue_mill=BLUE_MILL)


def mixed():
    """X0001/26: run 1 too mild (harmful), run 2 unread (harmful, the reader failed), run 3 right. X0002/26: three right."""
    first = item("X0001/26", [result(got=TOO_MILD), result(ReaderStatus.FAILED), result()])
    second = item("X0002/26", [result(notam_id="X0002/26")] * 3)
    return report_for([first, second])


# --- the headline and the order ---------------------------------------------------------------------------------------------------


def test_the_heading_leads_with_harmful_downgrades_as_items_and_item_runs_and_states_the_target():
    report = mixed()
    assert headline(report) == "Harmful downgrades: 1 item, 2 item-runs (target 0)"
    markdown = render_markdown(report)
    assert markdown.splitlines()[0] == "# Harmful downgrades: 1 item, 2 item-runs (target 0)"
    print(markdown)


def test_an_item_counts_once_however_many_of_its_runs_are_harmful():
    report = report_for([item("X0001/26", [result(got=TOO_MILD)] * 3)])
    assert (report["harmful"]["items"], report["harmful"]["item_runs"]) == (1, 3)
    assert headline(report) == "Harmful downgrades: 1 item, 3 item-runs (target 0)"


def test_no_harmful_downgrade_reads_0_items_0_item_runs_and_none():
    report = report_for([item("X0001/26", [result()] * 3)])
    assert headline(report) == "Harmful downgrades: 0 items, 0 item-runs (target 0)"
    assert "## Harmful downgrades\n\n0 items, 0 item-runs\n\nnone" in render_markdown(report)


def test_each_harmful_run_is_named_with_its_item_run_action_spot_and_field():
    lines = [f["line"] for f in mixed()["harmful"]["findings"]]
    assert lines[0].startswith("X0001/26 run 1: ") and "\u2192" in lines[0] and "at the centre" in lines[0] and "category" in lines[0]
    assert "(reader failed)" not in lines[0]


def test_an_unread_run_is_tagged_reader_failed_and_a_read_one_is_not():
    findings = mixed()["harmful"]["findings"]
    failed = [f for f in findings if f["run"] == 2]
    assert len(failed) == 1 and failed[0]["reader_failed"] and failed[0]["line"].count("(reader failed)") == 1
    assert all(not f["reader_failed"] and "(reader failed)" not in f["line"] for f in findings if f["run"] != 2)
    assert "X0001/26 run 2" in render_markdown(mixed()) and "(reader failed)" in render_markdown(mixed())


def test_the_sections_come_in_the_order_of_the_spec():
    markdown = render_markdown(mixed())
    order = ["## Harmful downgrades", "## Over-caution", "## Field accuracy", "## Reader health", "## Skipped items", "## Blue Mill live check"]
    positions = [markdown.index(heading) for heading in order]
    assert positions == sorted(positions)
    assert markdown.index("# Harmful downgrades:") == 0


def test_over_caution_has_its_own_item_and_item_run_counts():
    cautious = reading(category=Category.PROHIBITIVE, ceiling=amsl(2000))
    mild = reading(category=Category.INFORMATIONAL, ceiling=amsl(2000))
    report = report_for([item("X0001/26", [result(got=cautious)] * 3, label=mild)])
    assert report["harmful"]["item_runs"] == 0
    assert (report["over_caution"]["items"], report["over_caution"]["item_runs"]) == (1, 3)
    assert "## Over-caution\n\n1 item, 3 item-runs\n\n- X0001/26 run 1: " in render_markdown(report)


# --- field accuracy ---------------------------------------------------------------------------------------------------------------


def test_field_accuracy_is_the_mean_over_the_runs_and_the_worst_run():
    right, wrong = {"category": True, "ceiling_height": None}, {"category": False, "ceiling_height": None}
    a = ItemRuns("A/26", [result()] * 3, [], [right, wrong, right])
    b = ItemRuns("B/26", [result()] * 3, [], [right, right, right])
    entry = report_for([a, b])["field_accuracy"]["category"]
    assert entry["worst_run"] == pytest.approx(0.5)
    assert entry["mean"] == pytest.approx((1 + 0.5 + 1) / 3)


def test_a_field_that_is_not_applicable_in_any_run_is_not_in_the_table():
    right = {"category": True, "ceiling_height": None}
    report = report_for([ItemRuns("A/26", [result()], [], [right])], runs=1)
    assert "category" in report["field_accuracy"] and "ceiling_height" not in report["field_accuracy"]


def test_the_table_is_in_the_markdown_with_mean_and_worst_run_as_percentages():
    right, wrong = {"category": True}, {"category": False}
    markdown = render_markdown(report_for([ItemRuns("A/26", [result()] * 2, [], [right, wrong])], runs=2))
    assert "| category | 75.0% | 50.0% |" not in markdown  # one item: the runs are 100% and 0%
    assert "| category | 50.0% | 0.0% |" in markdown
    assert "geometry and activity are not scored" in markdown


# --- rates -------------------------------------------------------------------------------------------------------------------------


def test_the_rates_are_shares_of_item_runs():
    results = [
        result(ReaderStatus.READ),
        result(ReaderStatus.REPAIRED, failed_checks=["x"]),
        result(ReaderStatus.FAILED, failed_checks=["x"]),
        result(ReaderStatus.UNAVAILABLE),
    ]
    rates = report_for([item("X0001/26", results)], runs=4)["rates"]
    assert rates == {"validity": 0.5, "repair": 0.5, "failure": 0.25, "unavailable": 0.25}


def test_an_unavailable_run_after_a_failed_first_answer_counts_as_a_repair():
    rates = report_for([item("X0001/26", [result(ReaderStatus.UNAVAILABLE, failed_checks=["x"]), result(ReaderStatus.UNAVAILABLE)])], runs=2)["rates"]
    assert rates["repair"] == 0.5 and rates["unavailable"] == 1.0 and rates["validity"] == 0.0


def test_with_no_item_runs_the_rates_are_none_and_shown_n_a():
    report = report_for([])
    assert report["rates"] == {"validity": None, "repair": None, "failure": None, "unavailable": None}
    assert "- validity (a reading that passed the checks): n/a" in render_markdown(report)


# --- latency and tokens ------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "values, q, expected",
    [
        ([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], HALF, 5),
        ([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], P95, 10),
        (list(range(1, 21)), P95, 19),
        (list(range(1, 21)), HALF, 10),
        ([10, 1, 5], HALF, 5),
        ([7], HALF, 7),
        ([7], P95, 7),
        ([2, 9], HALF, 2),
        ([], HALF, None),
    ],
)
def test_nearest_rank_percentiles(values, q, expected):
    assert nearest_rank(values, q) == expected


def test_latency_is_in_milliseconds_over_every_item_run_and_one_run_still_gives_numbers():
    report = report_for([item("X0001/26", [result(latency_s=0.25)])], runs=1)
    assert report["latency_ms"] == {"p50": 250.0, "p95": 250.0, "item_runs": 1}
    many = report_for([item("X0001/26", [result(latency_s=s / 10) for s in range(1, 11)])], runs=10)["latency_ms"]
    assert many["p50"] == pytest.approx(500) and many["p95"] == pytest.approx(1000)


def test_tokens_are_summed_over_the_runs_that_reported_usage_only():
    usage = TokenUsage(input_tokens=100, output_tokens=20, total_tokens=120)
    report = report_for([item("X0001/26", [result(tokens=usage), result(tokens=usage), result()])])
    assert report["tokens"] == {"input": 200, "output": 40, "total": 240, "runs_with_usage": 2, "item_runs": 3}
    assert "- tokens: input 200, output 40, total 240 (2 of 3 item-runs returned usage)" in render_markdown(report)


def test_tokens_are_n_a_when_no_run_returned_usage():
    report = report_for([item("X0001/26", [result()] * 3)])
    assert report["tokens"]["total"] is None
    assert "- tokens: input n/a, output n/a, total n/a (0 of 3 item-runs returned usage)" in render_markdown(report)


# --- skipped, empty and the stdout summary -------------------------------------------------------------------------------------------


def test_skipped_items_are_separate_counts_with_their_ids(tmp_path):
    from reader_eval_run_helpers import GOOD_LABEL, build_sheet, label_entry, write_labels

    from evals.reader_eval.items import load_items
    from evals.reader_eval.labels import load_labels

    paths, hashes = build_sheet(tmp_path)
    write_labels(
        paths,
        [label_entry("X1/26", hashes, GOOD_LABEL), label_entry("X2/26", hashes, {"category": "cooperative"}), label_entry("X3/26", hashes, GOOD_LABEL, stale=True), {"id": "Z9/26", "e_hash": "x"}],
    )
    labels = load_labels(paths.labels, load_items(paths.items))
    skipped = report_for([], labels=labels)["skipped"]
    assert skipped["unlabelled"] == {"count": 0, "ids": []}
    assert skipped["partial"]["count"] == 1 and skipped["partial"]["ids"] == ["X2/26"] and "activity" in skipped["partial"]["missing"]["X2/26"]
    assert skipped["stale"] == {"count": 1, "ids": ["X3/26"]}
    assert skipped["unknown"] == {"count": 1, "ids": ["Z9/26"]}
    markdown = render_markdown(report_for([], labels=labels))
    assert "- stale (the NOTAM text changed since it was labelled): 1: X3/26" in markdown and "- label ids not on the sheet (unknown): 1: Z9/26" in markdown


def test_no_scored_items_says_why_and_still_renders_everything():
    report = report_for([], found=False)
    assert scored_line(report) == "0 labelled items: evals/reader/labels.yaml not found"
    markdown = render_markdown(report)
    assert "0 labelled items: evals/reader/labels.yaml not found" in markdown and "no scored items" in markdown and "## Blue Mill live check" in markdown
    assert scored_line(report_for([], found=True)) == "0 labelled items: no item is fully labelled"


def test_the_summary_lines_lead_with_the_headline_and_carry_the_blue_mill_lines():
    lines = summary_lines(mixed())
    assert lines[0] == "Harmful downgrades: 1 item, 2 item-runs (target 0)"
    assert lines[3] == "  X0001/26: read 3 times (read \u00d72, failed \u00d71)" and lines[4] == "  X0002/26: read 3 times (read \u00d73)"
    assert lines[2] == "2 labelled items scored, 3 runs each (6 item-runs)"
    assert lines[-2] == "Blue Mill live check:" and lines[-1].endswith("fixture \u2713")


# --- one dict ------------------------------------------------------------------------------------------------------------------------


KEYS = {"harmful", "over_caution", "field_accuracy", "rates", "latency_ms", "tokens", "skipped", "blue_mill", "model", "prompt_version", "runs"}


def test_the_dict_has_every_key_the_spec_names_and_survives_a_json_round_trip():
    report = mixed()
    assert KEYS <= set(report)
    assert json.loads(json.dumps(report)) == report


def test_the_markdown_is_rendered_from_the_dict_alone(tmp_path):
    report = mixed()
    folder = write_report(report, tmp_path / "runs", utc(7))
    from_disk = json.loads((folder / "report.json").read_text(encoding="utf-8"))
    assert from_disk == report
    assert (folder / "report.md").read_text(encoding="utf-8") == render_markdown(from_disk)


def test_changing_a_number_in_the_dict_changes_the_markdown_so_they_cannot_drift():
    report = mixed()
    changed = json.loads(json.dumps(report))
    changed["harmful"]["items"], changed["rates"]["validity"], changed["latency_ms"]["p50"] = 7, 0.123, 4321.0
    markdown = render_markdown(changed)
    assert "Harmful downgrades: 7 items" in markdown and "12.3%" in markdown and "4,321 ms" in markdown
    assert render_markdown(report) != markdown


def test_model_and_prompt_version_are_in_both_reports(tmp_path):
    folder = write_report(mixed(), tmp_path / "runs", utc(7))
    assert json.loads((folder / "report.json").read_text(encoding="utf-8"))["model"] == "fake-model"
    assert "Model fake-model \u00b7 prompt version test \u00b7 3 runs per item" in (folder / "report.md").read_text(encoding="utf-8")


# --- folders ---------------------------------------------------------------------------------------------------------------------------


def test_two_runs_in_the_same_second_get_separate_folders_and_the_first_is_not_touched(tmp_path):
    runs = tmp_path / "runs"
    first = write_report(mixed(), runs, utc(7))
    before = {p.name: p.read_bytes() for p in first.iterdir()}
    second = write_report(report_for([]), runs, utc(7))
    third = make_run_folder(runs, utc(7))
    assert len({first, second, third}) == 3 and all(p.parent == runs for p in (first, second, third))
    assert first.name == "20261031T070000Z" and second.name == "20261031T070000Z-2" and third.name == "20261031T070000Z-3"
    assert {p.name: p.read_bytes() for p in first.iterdir()} == before
    assert sorted(p.name for p in second.iterdir()) == ["report.json", "report.md"]


def test_a_folder_that_already_exists_is_never_reused(tmp_path):
    runs = tmp_path / "runs"
    (runs / "20261031T070000Z").mkdir(parents=True)
    (runs / "20261031T070000Z" / "report.md").write_text("an earlier report", encoding="utf-8")
    folder = write_report(mixed(), runs, utc(7))
    assert folder.name.endswith("-2")
    assert (runs / "20261031T070000Z" / "report.md").read_text(encoding="utf-8") == "an earlier report"
