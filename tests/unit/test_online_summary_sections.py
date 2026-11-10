"""Every GP4 section is in the file, with its fixed lines, and the failing-run links come from the client (7 PR-4.F1, GP4, E2, E4).

The report is built from hand-made feedback; the links come from `summary.run_link` over the online recording client, whose `get_run_url` is recorded.
"""

import pytest
from online_helpers import OnlineRecordingClient, fake_run
from online_summary_helpers import VERSION, judge_all, report_for, scored_run

from evals.agent.judge_config import JUDGE_VARIABLES
from evals.online import job, summary


def sample_report(**kw):
    runs = [
        scored_run("a", path="assess", checks={"step_cap": 0}, judge=judge_all("assess"), lz_name="Blue Mill", minutes=1),
        scored_run("b", path="decline", judge={**judge_all("decline"), "good_decline": 0}, minutes=2),
        scored_run("c", path="unknown", checks={"decline_no_tools": (None, job.PATH_UNKNOWN)}, minutes=3),
    ]
    return report_for(runs, skipped_by_source={"case": 2, "judge": 1, "v0": 1, "missing": 1}, skipped_by_reason={"source": 5, "errored": 1, "experiment": 1}, **kw)


def headings(text):
    return [line.removeprefix("## ") for line in text.splitlines() if line.startswith("## ")]


def test_every_section_heading_is_there_in_the_order_of_the_spec():
    assert headings(summary.render_markdown(sample_report())) == list(summary.SECTIONS)
    assert summary.SECTIONS == ("Deterministic checks", "Judge", "Failing runs", "Runs skipped", "Evaluator version", "Not scored online")


def test_the_file_is_headed_with_its_day_and_the_run_count_and_has_no_slices_and_no_reps_column():
    text = summary.render_markdown(sample_report())
    assert text.startswith("# Online eval 2026-11-08\n\n3 runs scored at evaluator version " + VERSION)
    assert "slice" not in text.lower() and "reps" not in text.lower() and "| rep" not in text.lower()


def test_the_fixed_labels_line_says_missed_blocker_and_verdict_match_need_labels_and_are_carried_offline():
    text = summary.render_markdown(sample_report())
    assert "Missed blocker and verdict match need labels, which production runs do not have: they are carried by the offline evals." in text.splitlines()
    assert "Window IoU, noise and NOTAM action confusion are offline-only too." in text.splitlines()


def test_the_checks_section_gives_each_pass_rate_with_n_and_the_null_reasons():
    section = "\n".join(summary.check_lines(sample_report()))
    assert "| step_cap | 66.7% | 2/3 | 3 | 0 |" in section
    assert "| decline_no_tools | 100.0% | 2/2 | 2 | 1 (path unknown: 1) |" in section  # the unknown run is a null with its reason, never a failure


def test_the_judge_section_has_rows_by_path_the_unknown_row_and_the_not_sampled_count():
    section = "\n".join(summary.judge_lines(sample_report()))
    assert "Runs marked not_sampled (left out of the judge's sample): 0" in section
    assert "| assess | prioritised | 100.0% | 1 | 0 |" in section and "| decline | good_decline | 0.0% | 1 | 0 |" in section
    assert "| unknown | no judge criteria for this path (1 run) | n/a | 0 | 0 |" in section


def test_e4_with_the_judge_unset_the_section_names_each_variable_and_the_deterministic_rates_stay():
    report = sample_report(judge_missing=JUDGE_VARIABLES)
    text = summary.render_markdown(report)
    assert "The judge was not run: JUDGE_PROVIDER, JUDGE_BASE_URL, JUDGE_API_KEY, JUDGE_MODEL not set." in text
    assert "| step_cap | 66.7% | 2/3 | 3 | 0 |" in text and "| good_decline |" not in text
    assert "Runs marked not_sampled" in text


def test_the_skipped_section_counts_by_source_by_reason_fully_scored_and_errors():
    section = "\n".join(summary.skipped_lines(sample_report()))
    assert "- skipped by the pull: 7" in section
    assert "by source (runs whose source is not `pilot` or `online-seed`): case: 2, judge: 1, missing: 1, v0: 1" in section
    assert "by reason: errored: 1, experiment: 1, source: 5" in section
    assert "- fully scored at this evaluator version: 3" in section and "- runs with an error: 0" in section


def test_the_version_section_has_the_12_hex_characters_since_limit_and_the_limit_line():
    section = "\n".join(summary.version_lines(sample_report(limit_hit=True)))
    assert f"- evaluator version: {VERSION}" in section and len(VERSION) == 12
    assert "- SINCE: 7d" in section and "- LIMIT: 500" in section and "LIMIT was hit" in section


# --- the links come from get_run_url ------------------------------------------------------------------------------------------------------


def test_failing_run_links_come_from_get_run_url_on_the_client():
    failing, passing = fake_run("failing"), fake_run("passing")
    client = OnlineRecordingClient([failing, passing])
    link = summary.run_link(client, "flysafelz", [failing, passing])
    report = report_for(
        [scored_run(str(failing.id), checks={"step_cap": 0}, minutes=1), scored_run(str(passing.id), minutes=2)],
        link=link,
    )
    expected = f"https://smith.example/o/org/projects/p/flysafelz/r/{failing.id}"
    assert [item["url"] for item in report["failing"]] == [expected]
    assert [kw for method, kw in client.calls if method == "get_run_url"] == [{"run_id": failing.id, "project_name": "flysafelz"}]  # asked for the failing run only
    assert f"| {expected} | assess | - | online.step_cap |" in summary.render_markdown(report)


def test_a_client_that_cannot_give_a_link_leaves_the_run_id_in_the_file():
    run = fake_run("failing")
    client = OnlineRecordingClient([run])
    client.fail_run_url = True
    report = report_for([scored_run(str(run.id), checks={"step_cap": 0})], link=summary.run_link(client, "flysafelz", [run]))
    assert report["failing"][0]["url"] is None
    assert f"| run {run.id} | assess |" in summary.render_markdown(report)


@pytest.mark.parametrize("text_in", ["answer text", "question text"])
def test_the_failing_section_holds_no_answer_or_question_text(text_in):
    run = fake_run("failing", answer=f"[path: assess]\nGO. {text_in} 01632 960123")
    client = OnlineRecordingClient([run])
    report = report_for([scored_run(str(run.id), checks={"step_cap": 0})], link=summary.run_link(client, "flysafelz", [run]))
    assert text_in not in summary.render_markdown(report) and "01632 960123" not in summary.render_markdown(report)


def test_the_stdout_lines_come_from_the_same_dict():
    report = sample_report()
    lines = summary.summary_lines(report)
    assert lines[0] == f"online eval 2026-11-08: 3 runs scored at evaluator version {VERSION} (project flysafelz, SINCE=7d, LIMIT=500)"
    assert "  step_cap: 66.7% (2/3), 0 null" in lines and "  failing runs: 2" in lines
    assert "  judge: 5 graded, 0 not_sampled" in lines
    assert "  skipped: 7; fully scored: 3; errors: 0" in lines
