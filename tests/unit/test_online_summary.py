"""The report dict and its rendering (7 PR-4.F1, GP4, E1, E2, E4, E5 and the added cases). Hand-built feedback lists: no client.

The rule under test: rate = passes / the runs with a score of 1 or 0 (n); a null is left out of n and counted by its reason; a rate with n = 0 is `n/a`.
"""

import json

import pytest
from online_summary_helpers import DAY, OTHER_VERSION, START, VERSION, check_row, judge_all, judge_row, posted, report_for, scored_run

from evals.online import job, summary

GUARD = "guard_pass"


def guard_runs(scores):
    """One run per entry of `scores` (1, 0, None or (None, reason)), each with its own id and start time."""
    return [scored_run(f"run-{index:02d}", minutes=index, checks={GUARD: score}) for index, score in enumerate(scores)]


# --- rate = passes / non-null count, nulls counted per reason ------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("scores", "n", "passed", "rate", "nulls"),
    [
        pytest.param([1, 1, 1, 1], 4, 4, 1.0, {}, id="all-pass"),
        pytest.param([0, 0, 0], 3, 0, 0.0, {}, id="all-fail"),
        pytest.param([None, None], 0, 0, None, {job.NOT_APPLICABLE: 2}, id="all-null"),
        pytest.param([1, 0, None, 1, (None, "no guard record")], 3, 2, 2 / 3, {job.NOT_APPLICABLE: 1, "no guard record": 1}, id="mixed"),
        pytest.param([(None, "no assess"), (None, "no assess"), (None, "path unknown"), 1], 1, 1, 1.0, {"no assess": 2, "path unknown": 1}, id="nulls-by-reason"),
        pytest.param([], 0, 0, None, {}, id="no-runs"),
    ],
)
def test_the_rate_is_passes_over_the_non_null_count_with_nulls_counted_by_reason(scores, n, passed, rate, nulls):
    row = check_row(report_for(guard_runs(scores)), GUARD)
    assert (row["n"], row["passed"], row["rate"], row["nulls"], row["null_count"]) == (n, passed, rate, nulls, sum(nulls.values()))


@pytest.mark.parametrize(
    ("scores", "shown"),
    [
        pytest.param([1, 1, 1, 1], "| guard_pass | 100.0% | 4/4 | 4 | 0 |", id="all-pass"),
        pytest.param([0, 0, 0], "| guard_pass | 0.0% | 0/3 | 3 | 0 |", id="all-fail"),
        pytest.param([None, None], "| guard_pass | n/a | 0/0 | 0 | 2 (not applicable: 2) |", id="all-null"),
        pytest.param([1, 0, None, 1, (None, "no guard record")], "| guard_pass | 66.7% | 2/3 | 3 | 2 (no guard record: 1, not applicable: 1) |", id="mixed"),
        pytest.param([], "| guard_pass | n/a | 0/0 | 0 | 0 |", id="no-runs"),
    ],
)
def test_the_rendered_row_shows_the_rate_with_n_and_n_a_when_n_is_zero(scores, shown):
    assert shown in summary.render_markdown(report_for(guard_runs(scores))).splitlines()


def test_the_rendered_file_is_built_from_the_dict_alone():
    report = report_for(guard_runs([1, 0, None]))
    assert summary.render_markdown(json.loads(json.dumps(report))) == summary.render_markdown(report)


def test_every_deterministic_check_has_a_row_in_the_order_the_job_posts_them():
    assert [row["key"] for row in report_for([])["checks"]] == list(job.DETERMINISTIC_KEYS)


@pytest.mark.parametrize(
    ("comment", "reason"),
    [
        (f"{job.NO_GUARD_RECORD} | evaluator_version={VERSION}", job.NO_GUARD_RECORD),
        (f"{job.PATH_UNKNOWN} | evaluator_version={VERSION}", job.PATH_UNKNOWN),
        (f"{job.NOT_APPLICABLE} | evaluator_version={VERSION}", job.NOT_APPLICABLE),
        (f"{job.CHECK_ERROR}: KeyError | evaluator_version={VERSION}", "check error: KeyError"),
        (f"{job.NO_JUDGE_RESULT} | evaluator_version={VERSION}", job.NO_JUDGE_RESULT),
        (f"judge error: ConnectionError: the bridge said something long and specific | evaluator_version={VERSION}", "judge error"),
        (f"something nobody wrote | evaluator_version={VERSION}", "other"),
        (None, "other"),
    ],
)
def test_a_null_comment_is_one_of_the_known_reasons_and_never_free_text(comment, reason):
    assert summary.null_reason(comment) == reason


# --- the version -----------------------------------------------------------------------------------------------------------------------------


def test_feedback_from_an_older_evaluator_version_is_not_counted():
    old = scored_run("run-a", version=OTHER_VERSION, checks={GUARD: 0})
    now = scored_run("run-a", checks={GUARD: 1}, extra=old.feedback)  # the same run holds both versions' keys
    report = report_for([now])
    assert report["runs"] == 1 and check_row(report, GUARD)["n"] == 1 and check_row(report, GUARD)["passed"] == 1
    assert report["failing"] == []


def test_a_run_with_feedback_only_from_an_older_version_is_not_a_scored_run():
    report = report_for([scored_run("run-a", version=OTHER_VERSION)])
    assert report["runs"] == 0 and all(row["n"] == 0 for row in report["checks"]) and report["skipped"]["fully_scored"] == 0


def test_feedback_this_job_did_not_write_is_not_counted():
    other = posted(job.feedback_key(GUARD), 0)
    other.extra, other.comment = None, "a person's note"
    run = scored_run("run-a", only=[], marker=False, extra=[other])
    assert report_for([run])["runs"] == 0


# --- judge rows by path ------------------------------------------------------------------------------------------------------------------


def test_the_judge_has_one_row_per_criterion_and_path_and_a_separate_unknown_row():
    runs = [
        scored_run("a", path="assess", judge=judge_all("assess"), minutes=1),
        scored_run("b", path="assess", judge={**judge_all("assess"), "prioritised": 0}, minutes=2),
        scored_run("c", path="decline", judge=judge_all("decline"), minutes=3),
        scored_run("d", path="unknown", checks={"decline_no_tools": (None, job.PATH_UNKNOWN)}, minutes=4),
    ]
    report = report_for(runs)
    assert judge_row(report, "assess", "prioritised")["rate"] == 0.5 and judge_row(report, "assess", "prioritised")["n"] == 2
    assert judge_row(report, "assess", "answers_the_question")["rate"] == 1.0
    assert judge_row(report, "decline", "good_decline")["n"] == 1
    assert judge_row(report, "explain", "actionable")["rate"] is None  # no explain run: n/a, not 0
    unknown = judge_row(report, "unknown", None)
    assert unknown["runs"] == 1 and unknown["n"] == 0 and unknown["rate"] is None
    assert [row["path"] for row in report["judge"]["rows"]][-1] == "unknown"


def test_the_unknown_row_is_there_even_with_no_run():
    assert judge_row(report_for([]), "unknown", None)["runs"] == 0


def test_a_judge_null_is_counted_by_reason_and_left_out_of_n():
    run = scored_run("a", judge={**judge_all("clarify"), "answers_the_question": (None, job.NO_JUDGE_RESULT)}, path="clarify")
    row = judge_row(report_for([run]), "clarify", "answers_the_question")
    assert (row["n"], row["rate"], row["nulls"]) == (0, None, {job.NO_JUDGE_RESULT: 1})


def test_a_path_5a_has_no_criteria_for_gets_its_own_row_with_no_criterion():
    report = report_for([scored_run("a", path="act")])
    assert judge_row(report, "act", None)["runs"] == 1


def test_the_judge_section_names_the_unset_variables_in_place_of_rates():
    report = report_for([scored_run("a", judge=judge_all("assess"))], judge_missing=("JUDGE_BASE_URL", "JUDGE_MODEL"))
    section = summary.judge_lines(report)
    text = "\n".join(section)
    assert "JUDGE_BASE_URL, JUDGE_MODEL not set" in text and "pass rate" not in text
    assert all(row["name"] for row in report["checks"]) and check_row(report, GUARD)["n"] == 1  # the deterministic rates are all there


def test_not_sampled_counts_the_runs_with_the_record_and_no_judge_key():
    sampled = posted(job.SAMPLED_KEY, None, "left out of the judge sample")
    runs = [
        scored_run("a", extra=[sampled], minutes=1),
        scored_run("b", extra=[sampled], judge=judge_all("assess"), minutes=2),  # left out once, judged since: not counted
        scored_run("c", minutes=3),
    ]
    assert report_for(runs)["judge"]["not_sampled"] == 1


# --- failing runs ------------------------------------------------------------------------------------------------------------------------


def test_a_run_with_a_zero_on_any_check_or_judge_criterion_fails_and_a_null_never_does():
    runs = [
        scored_run("a", checks={"step_cap": 0}, minutes=1),
        scored_run("b", judge={**judge_all("assess"), "actionable": 0}, minutes=2),
        scored_run("c", checks={GUARD: None}, minutes=3),
        scored_run("d", path="unknown", checks={"decline_no_tools": (None, job.PATH_UNKNOWN)}, minutes=4),
    ]
    failing = report_for(runs)["failing"]
    assert [(item["run_id"], item["keys"]) for item in failing] == [("a", ["online.step_cap"]), ("b", ["online.judge.actionable"])]


def test_failing_keys_come_in_the_posting_order_and_the_runs_by_start_time_then_id():
    runs = [
        scored_run("b", checks={"step_cap": 0, GUARD: 0}, minutes=5),
        scored_run("a", checks={"step_cap": 0}, minutes=5),
        scored_run("z", checks={"step_cap": 0}, minutes=1),
    ]
    failing = report_for(runs)["failing"]
    assert [item["run_id"] for item in failing] == ["z", "a", "b"]
    assert failing[2]["keys"] == ["online.guard_pass", "online.step_cap"]


def test_the_failing_run_carries_its_link_path_and_launch_zone_and_the_link_is_asked_for_failing_runs_only():
    asked = []

    def link(run_id):
        asked.append(run_id)
        return f"https://smith.example/r/{run_id}"

    report = report_for([scored_run("a", checks={GUARD: 0}, lz_name="Blue Mill", minutes=1), scored_run("b", minutes=2)], link=link)
    assert asked == ["a"]
    assert report["failing"] == [{"run_id": "a", "url": "https://smith.example/r/a", "path": "assess", "lz_name": "Blue Mill", "keys": ["online.guard_pass"]}]


# --- E1: no runs -------------------------------------------------------------------------------------------------------------------------


def test_e1_no_runs_gives_a_file_that_says_0_runs_with_empty_sections_and_n_a_rates():
    text = summary.render_markdown(report_for([], read=0))
    assert "0 runs scored" in text and "No run was scored." in text
    assert all(f"| {name} | n/a | 0/0 | 0 | 0 |" in text for name in ("guard_pass", "step_cap"))
    assert "- skipped by the pull: 0" in text and "- fully scored at this evaluator version: 0" in text


# --- ordering and escaping --------------------------------------------------------------------------------------------------------------------


def test_the_report_does_not_depend_on_the_order_the_runs_arrive_in():
    runs = [scored_run(f"r{index}", checks={GUARD: index % 2}, minutes=index % 3) for index in range(7)]
    assert summary.render_markdown(report_for(runs)) == summary.render_markdown(report_for(list(reversed(runs))))


def test_unicode_and_table_breaking_characters_in_a_launch_zone_name_stay_in_their_cell():
    name = "Crookham – north | pad\nB"
    report = report_for([scored_run("a", checks={GUARD: 0}, lz_name=name)], link=lambda run_id: None)
    rows = [line for line in summary.render_markdown(report).splitlines() if line.startswith("| run a")]
    assert len(rows) == 1
    assert "Crookham – north \\| pad B" in rows[0]
    unescaped_pipes = [index for index, char in enumerate(rows[0]) if char == "|" and rows[0][index - 1] != "\\"]
    assert len(unescaped_pipes) == 5  # the row's own four cells: run, path, launch zone, failing keys


def test_a_run_start_time_in_the_report_is_not_needed_for_the_date_and_the_report_holds_no_timestamp():
    text = summary.render_markdown(report_for([scored_run("a")]))
    assert START.isoformat() not in text and "T09:00" not in text and DAY.isoformat() in text


# --- skipped, complete and errors -----------------------------------------------------------------------------------------------------------


def test_fully_scored_counts_the_runs_with_every_deterministic_key_and_the_marker_at_this_version():
    runs = [
        scored_run("a", judge=judge_all("assess"), minutes=1),
        scored_run("b", judge=judge_all("assess"), marker=False, minutes=2),
        scored_run("c", only=["guard_pass"], judge=judge_all("assess"), minutes=3),
    ]
    assert report_for(runs)["skipped"]["fully_scored"] == 1


def test_a_run_whose_path_has_judge_criteria_is_fully_scored_only_with_its_judge_keys_or_the_not_sampled_record():
    # E4: the judge was never set up, so the job posted the marker with no judge key: a later run fills them in, so the run is not complete yet
    unjudged = scored_run("a", minutes=1)
    partly = scored_run("b", judge={name: 1 for name in list(judge_all("assess"))[:1]}, minutes=2)
    judged = scored_run("c", judge=judge_all("assess"), minutes=3)
    left_out = scored_run("d", extra=[posted(job.SAMPLED_KEY, None, "left out of the judge sample")], minutes=4)
    no_criteria = scored_run("e", path="unknown", minutes=5)  # 5a has no judge criteria for `unknown`: its deterministic keys are all there is
    assert len(judge_all("assess")) > 1
    report = report_for([unjudged, partly, judged, left_out, no_criteria])
    assert report["runs"] == 5 and report["skipped"]["fully_scored"] == 3


def test_a_post_failure_is_listed_under_its_run_with_the_stage_key_and_class_and_no_message():
    error = job.RunError(job.POST, "RuntimeError", "the body of the error, which could hold anything", key="online.step_cap")
    report = report_for([scored_run("a", only=["guard_pass"], marker=False, error=error)])
    assert report["skipped"]["errors"] == [{"run_id": "a", "stage": "post", "key": "online.step_cap", "error_class": "RuntimeError"}]
    text = summary.render_markdown(report)
    assert "a: post online.step_cap: RuntimeError" in text and "which could hold anything" not in text


def test_a_run_with_an_error_and_no_feedback_is_listed_but_is_not_a_scored_run():
    error = job.RunError(job.READ_TRACE, "ValueError", "x")
    report = report_for([scored_run("a", only=[], marker=False, error=error)])
    assert report["runs"] == 0 and len(report["skipped"]["errors"]) == 1


def test_the_skips_are_counted_by_source_and_by_reason():
    report = report_for([], skipped_by_source={"case": 2, "judge": 1, "missing": 1}, skipped_by_reason={"source": 4, "errored": 1, "experiment": 1})
    assert report["skipped"]["total"] == 6
    text = summary.render_markdown(report)
    assert "case: 2, judge: 1, missing: 1" in text and "errored: 1, experiment: 1, source: 4" in text


def test_a_limit_that_was_hit_is_said_and_one_that_was_not_is_said_too():
    hit = summary.render_markdown(report_for([], limit_hit=True, limit=50))
    assert "LIMIT was hit: 50 runs were read" in hit
    assert "LIMIT was not hit." in summary.render_markdown(report_for([]))
