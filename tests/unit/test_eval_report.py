"""The results file (5a PR-10.F1, G16, GP4): headline, slices, failing and errored ids, Ops and models, from hand-built scored runs of 3 examples x 3 reps.

Nothing here calls a model, a service or the network; the runs are objects the test builds.
"""

import uuid
from decimal import Decimal

import pytest

from evals.agent import report
from evals.agent.evaluators.blocker import MISSED_BLOCKER_KEY, RECALL_KEY
from evals.agent.evaluators.noise import NOISE_KEY
from evals.agent.evaluators.trajectory import REQUIRED_KEY
from evals.agent.evaluators.verdict import VERDICT_KEY
from evals.agent.judge import CRITERIA, SELF_PREFERENCE_LINE, feedback_key
from evals.agent.pricing import Price
from evals.agent.report import SECTIONS, ReportInput, ReportRun, headline_value, is_incomplete, render, rep_means
from flysafelz.reader.contract import TokenUsage

IDS = ("ex_a", "ex_b", "ex_c")
EXAMPLES = {
    "ex_a": {"id": "ex_a", "metadata": {"path": "assess", "case_type": "blocker", "failure_source": "none", "notam_category": "danger", "airspace_context": "open", "compound": False}},
    "ex_b": {"id": "ex_b", "metadata": {"path": "assess", "case_type": "quiet", "failure_source": "reader", "notam_category": "other", "airspace_context": "ctr"}},
    "ex_c": {"id": "ex_c", "metadata": {"path": "explain", "case_type": "guidance", "failure_source": "none", "airspace_context": "open", "compound": True}},
}
PRICE = Price("agent-model", Decimal("3.00"), Decimal("15.00"), "https://provider.invalid/pricing", "2026-11-09")
JUDGE_ANSWERS = feedback_key(CRITERIA[0])
JUDGE_LABEL = f"judge: {CRITERIA[0]} pass rate"


def run(example: str, rep: int, scores: dict | None = None, **fields) -> ReportRun:
    return ReportRun(example, rep, scores=scores or {}, latency_s=fields.pop("latency_s", 1.0), run_id=uuid.UUID(int=hash((example, rep)) % 2**64), **fields)


def three_by_three() -> list[ReportRun]:
    """verdict_match: ex_a 1,1,0; ex_b 1,0,0; ex_c 1,1,1 -> rep means 1.0, 0.667, 0.333. ex_c has no recall (None) and a judge score."""
    verdicts = {"ex_a": (1.0, 1.0, 0.0), "ex_b": (1.0, 0.0, 0.0), "ex_c": (1.0, 1.0, 1.0)}
    runs = []
    for example in IDS:
        for rep in (1, 2, 3):
            scores = {
                VERDICT_KEY: verdicts[example][rep - 1],
                RECALL_KEY: None if example == "ex_c" else 1.0,
                MISSED_BLOCKER_KEY: 0.0,
                REQUIRED_KEY: None,
                JUDGE_ANSWERS: 1.0 if example == "ex_c" else None,
            }
            runs.append(run(example, rep, scores, usage=TokenUsage(input_tokens=1000, output_tokens=100, total_tokens=1100)))
    return runs


def make(runs=None, **changes) -> ReportInput:
    runs = three_by_three() if runs is None else runs
    fields = {
        "experiment": "v1-dev-20261109-120000",
        "split": "dev",
        "agent": "v1",
        "reps": 3,
        "planned_runs": 9,
        "runs": runs,
        "examples": EXAMPLES,
        "models": {"agent": "agent-model", "reader": "reader-model", "judge": "judge-model", "v0": None},
        "wall_clock_s": 3725.0,
        "price": PRICE,
    }
    return ReportInput(**{**fields, **changes})


def section(text: str, title: str) -> str:
    """The lines of one `## title` section."""
    lines = text.splitlines()
    start = lines.index(f"## {title}") + 1
    end = next((i for i in range(start, len(lines)) if lines[i].startswith("## ")), len(lines))
    return "\n".join(lines[start:end])


def row(text: str, label: str) -> str:
    return next(line for line in text.splitlines() if line.startswith(f"| {label}"))


# --- headline -----------------------------------------------------------------------------------------------------------------------


def test_rep_means_are_the_mean_of_each_reps_scores_in_rep_order():
    assert rep_means(three_by_three(), VERDICT_KEY, 3) == pytest.approx([1.0, 2 / 3, 1 / 3])


def test_the_headline_value_is_the_mean_of_the_rep_means_with_their_min_and_max():
    assert headline_value(three_by_three(), VERDICT_KEY, 3) == "66.7% (33.3%–100.0%)"


def test_the_headline_table_has_a_row_for_every_metric_the_spec_names():
    headline = section(render(make()), "Headline")
    for label in (
        "missed blockers",
        "hard-restriction recall",
        "verdict match",
        "path accuracy",
        "noise",
        "trajectory: required tools",
        "trajectory: forbidden tools",
        "trajectory: step cap",
        "guard pass",
    ):
        assert f"| {label}" in headline
    for criterion in CRITERIA:
        assert f"| judge: {criterion} pass rate" in headline
    assert row(headline, "verdict match") == "| verdict match | 66.7% (33.3%–100.0%) |"


def test_a_metric_that_is_null_for_the_whole_run_is_n_a_never_zero():
    headline = section(render(make()), "Headline")
    assert row(headline, "trajectory: required tools") == "| trajectory: required tools | n/a |"
    assert row(headline, "path accuracy") == "| path accuracy | n/a |"
    assert "0.0%" not in row(headline, "trajectory: required tools")


def test_a_metric_that_applied_to_some_runs_only_is_averaged_over_those_alone():
    headline = section(render(make()), "Headline")
    assert row(headline, "hard-restriction recall") == "| hard-restriction recall | 100.0% (100.0%–100.0%) |"  # ex_c's None is not a 0
    assert row(headline, JUDGE_LABEL) == f"| {JUDGE_LABEL} | 100.0% (100.0%–100.0%) |"


def test_one_rep_only_shows_the_spread_equal_to_the_mean_and_does_not_hide_it():
    one_rep = [r for r in three_by_three() if r.rep == 1]
    assert headline_value(one_rep, VERDICT_KEY, 1) == "100.0% (100.0%–100.0%)"
    text = render(make(one_rep, reps=1, planned_runs=3))
    assert row(section(text, "Headline"), "verdict match") == "| verdict match | 100.0% (100.0%–100.0%) |"


def test_a_partial_run_averages_only_the_reps_that_have_scores():
    partial = [r for r in three_by_three() if r.rep < 3]
    assert headline_value(partial, VERDICT_KEY, 3) == "83.3% (66.7%–100.0%)"


def test_a_missed_blocker_is_shown_as_the_share_of_runs_that_have_one_and_is_lower_is_better():
    runs = [run("ex_a", rep, {MISSED_BLOCKER_KEY: 1.0 if rep == 1 else 0.0}) for rep in (1, 2, 3)]
    assert row(section(render(make(runs, planned_runs=3)), "Headline"), "missed blockers").endswith("| 33.3% (0.0%–100.0%) |")


# --- slices --------------------------------------------------------------------------------------------------------------------------


def test_slice_tables_group_by_each_dimension_with_n_and_the_metrics_that_apply():
    slices = section(render(make()), "Slices")
    for dimension in ("path", "case_type", "failure_source", "notam_category", "airspace_context", "compound"):
        assert f"### by {dimension}" in slices
    by_path = slices.split("### by case_type")[0]
    assert "| assess | 2 | " in by_path and "| explain | 1 | " in by_path
    assert "verdict match 50.0%" in next(line for line in by_path.splitlines() if line.startswith("| assess |"))
    assert f"{JUDGE_LABEL} 100.0%" not in next(line for line in by_path.splitlines() if line.startswith("| assess |"))  # it did not apply to assess runs here
    assert f"{JUDGE_LABEL} 100.0%" in next(line for line in by_path.splitlines() if line.startswith("| explain |"))


def test_a_missing_compound_counts_as_false():
    compound = section(render(make()), "Slices").split("### by compound")[1]
    assert "| false | 2 |" in compound and "| true | 1 |" in compound


def test_a_slice_with_no_examples_with_runs_is_left_out_not_shown_as_zero():
    runs = [r for r in three_by_three() if r.example_id != "ex_c"]
    slices = section(render(make(runs, planned_runs=6)), "Slices")
    assert "| explain |" not in slices and "| guidance |" not in slices and "| true |" not in slices


def test_an_example_with_no_runs_does_not_make_a_row():
    text = render(make([run("ex_a", 1, {VERDICT_KEY: 1.0})], reps=1, planned_runs=1))
    slices = section(text, "Slices")
    assert "| assess | 1 |" in slices and "| quiet |" not in slices


def test_a_dimension_the_example_lacks_is_listed_as_none_and_never_dropped():
    slices = section(render(make()), "Slices")
    assert "| (none) | 1 |" in slices.split("### by notam_category")[1].split("### by airspace_context")[0]


# --- failing and errored ids -------------------------------------------------------------------------------------------------------------


def test_failing_ids_list_each_example_with_a_link_for_each_failing_run():
    text = render(make(link=lambda r: f"https://smith.invalid/r/{r.run_id}"))
    failing = section(text, "Failing ids")
    assert "- ex_a: verdict_match." in failing and "- ex_b: verdict_match." in failing
    assert "ex_c" not in failing
    assert failing.count("https://smith.invalid/r/") == 3  # ex_a rep 3, ex_b reps 2 and 3
    assert "[rep 3](https://smith.invalid/r/" in failing


def test_without_a_link_the_run_id_is_printed_so_the_run_can_still_be_found():
    failing = section(render(make()), "Failing ids")
    assert f"rep 3 (run {uuid.UUID(int=hash(('ex_a', 3)) % 2**64)})" in failing


def test_a_missed_blocker_is_a_failing_check():
    runs = [run("ex_a", 1, {MISSED_BLOCKER_KEY: 1.0, VERDICT_KEY: 1.0})]
    assert "- ex_a: harmful_missed_blocker." in section(render(make(runs, reps=1, planned_runs=1)), "Failing ids")


def test_noise_is_reported_but_is_not_a_failing_check():
    runs = [run("ex_a", 1, {NOISE_KEY: 1.0, VERDICT_KEY: 1.0})]
    assert "No scored run failed a check." in section(render(make(runs, reps=1, planned_runs=1)), "Failing ids")


def errored(example: str, rep: int, error: str = "the graph raised: boom") -> ReportRun:
    return ReportRun(example, rep, errored=True, error=error)


def test_errored_ids_are_listed_separately_and_never_scored():
    runs = [r for r in three_by_three() if (r.example_id, r.rep) != ("ex_b", 2)] + [errored("ex_b", 2)]
    text = render(make(runs))
    assert "- ex_b: rep 2: the graph raised: boom" in section(text, "Errored ids")
    assert "ex_b" in section(text, "Failing ids")  # its scored reps still count
    assert "rep 2" not in section(text, "Failing ids").split("ex_b")[1]  # the errored rep is not a failing run


def test_an_errored_run_adds_nothing_to_any_mean():
    runs = [run("ex_a", 1, {VERDICT_KEY: 1.0}), errored("ex_a", 2), run("ex_a", 3, {VERDICT_KEY: 1.0})]
    assert headline_value(runs, VERDICT_KEY, 3) == "100.0% (100.0%–100.0%)"


def test_a_run_with_no_errors_says_so():
    errored_section = section(render(make()), "Errored ids")
    assert "0 of 9 planned runs errored. incomplete: false" in errored_section and "No run errored." in errored_section


@pytest.mark.parametrize(("errors", "planned", "incomplete"), [(9, 90, False), (10, 90, True), (1, 10, False), (2, 10, True), (0, 9, False), (1, 9, True)])
def test_incomplete_is_more_than_ten_percent_errored_and_exactly_ten_is_not(errors, planned, incomplete):
    runs = [errored(f"ex_{n}", 1) for n in range(errors)]
    built = make(runs, planned_runs=planned)
    assert is_incomplete(built) is incomplete
    text = render(built)
    assert f"incomplete: {str(incomplete).lower()}" in section(text, "Errored ids")
    assert ("Status: incomplete" in text) is incomplete


def test_all_examples_errored_still_writes_a_report_with_n_a_everywhere_and_every_id_listed():
    runs = [errored(example, rep, f"{example} failed") for example in IDS for rep in (1, 2, 3)]
    text = render(make(runs))
    headline = section(text, "Headline")
    assert [line for line in headline.splitlines() if line.startswith("| ") and "---" not in line][1:] == [line for line in headline.splitlines() if line.endswith("| n/a |")]
    assert len([line for line in headline.splitlines() if line.endswith("| n/a |")]) == len(report.HEADLINE) and "%" not in headline
    errored_section = section(text, "Errored ids")
    assert "9 of 9 planned runs errored. incomplete: true" in errored_section
    assert all(f"- {example}: rep 1, 2, 3: {example} failed" in errored_section for example in IDS)
    assert "Status: incomplete" in text and "No run was scored." in section(text, "Failing ids")
    assert "latency per example run (p50 / p95, nearest rank, 0 scored runs): n/a / n/a" in section(text, "Ops")  # empty percentiles do not crash


def test_a_run_with_no_runs_at_all_still_renders():
    text = render(make([], planned_runs=9))
    assert "No example has a run." in section(text, "Slices")


# --- models, notes, order ----------------------------------------------------------------------------------------------------------------------


def test_the_models_section_gives_the_model_per_role_and_says_when_a_role_is_not_used():
    models = section(render(make()), "Models")
    assert "- agent: agent-model" in models and "- reader: reader-model" in models and "- judge: judge-model" in models
    assert "- v0: not used in this experiment" in models


def test_the_notes_the_caller_gives_are_carried_and_the_shared_case_line_is_said_when_cases_are_shared():
    models = section(render(make(notes=[SELF_PREFERENCE_LINE, "v1's advantage covers reading plus arithmetic"], shared_case_groups=2)), "Models")
    assert SELF_PREFERENCE_LINE in models and "v1's advantage covers reading plus arithmetic" in models
    assert "Examples that share a case reused rep-1 readings" in models


def test_no_shared_case_line_without_shared_cases_and_none_for_v0_which_has_no_reader():
    assert "reused rep-1 readings" not in section(render(make()), "Models")
    assert "reused rep-1 readings" not in section(render(make(agent="v0", shared_case_groups=3)), "Models")


def test_the_sections_are_in_the_order_headline_slices_failing_errored_ops_models():
    text = render(make())
    positions = [text.index(f"## {title}") for title in SECTIONS]
    assert SECTIONS == ("Headline", "Slices", "Failing ids", "Errored ids", "Ops", "Models")
    assert positions == sorted(positions) and text.count("\n## ") == 6


def test_the_title_is_the_experiment_name_and_the_status_is_in_the_first_lines():
    text = render(make())
    assert text.startswith("# v1-dev-20261109-120000\n") and "9 of 9 planned runs, 0 errored. Status: complete." in text.splitlines()[2]
    assert text.endswith("\n") and not text.endswith("\n\n")


def test_an_interrupted_run_is_marked_in_the_status():
    assert "Status: complete (interrupted)." in render(make(interrupted=True))


def test_a_cell_with_a_pipe_does_not_break_the_table():
    examples = {**EXAMPLES, "ex_a": {"id": "ex_a", "metadata": {**EXAMPLES["ex_a"]["metadata"], "case_type": "a|b"}}}
    assert "a\\|b" in section(render(make(examples=examples)), "Slices")
