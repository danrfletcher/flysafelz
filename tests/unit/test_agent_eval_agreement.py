"""`make agreement` (5a PR-8.F1, G13, GP7, E5): the numbers, the weak rule, the labels read back, the second judge pass and its cache.

The client is the recording stand-in holding Dan-style labels; the judge is a fake chat model that answers from a script. Nothing real is called and no key is read.
"""

import io
import json
from pathlib import Path
from typing import Any

import pytest
from calibration_helpers import CalibrationClient, agree_with_judge, fake_model, judge_for, queued_items
from factories import dan_feedback, fake_experiment, judge_feedback

from evals.agent import agreement, annotate
from evals.agent.agreement import MIN_KAPPA, MIN_N, Stats, pair_stats, parse_label, weak_reasons
from evals.agent.judge import CRITERIA, JudgeConfigError

EXPERIMENT = "v1-dev-20261108-160700"
ASSESS_FIRST_FOUR = ("dev_01_blue_mill", "dev_02_west_sleekburn", "dev_03_buttermere", "dev_04_alnwick_pastures")


def table(spec):
    """(judge pass, Dan pass) pairs from counts: spec = (both pass, judge pass / Dan fail, judge fail / Dan pass, both fail)."""
    a, b, c, d = spec
    return [(1, 1)] * a + [(1, 0)] * b + [(0, 1)] * c + [(0, 0)] * d


# --- the numbers: hand-computed 2 x 2 tables -----------------------------------------------------------------------------------------


def test_a_perfect_table_is_100_percent_agreement_and_kappa_1():
    stats = pair_stats(table((10, 0, 0, 10)))
    assert (stats.n, stats.agree, stats.percent, stats.kappa, stats.weak) == (20, 20, 100.0, 1.0, ())
    assert (stats.percent_text, stats.kappa_text, stats.flag) == ("100.0%", "1.000", "")


def test_a_chance_level_table_is_50_percent_and_kappa_0():
    """5 / 5 / 5 / 5: both raters pass half of the answers, so the agreement expected by chance is 0.5 and kappa = (0.5 - 0.5) / (1 - 0.5) = 0."""
    stats = pair_stats(table((5, 5, 5, 5)))
    assert (stats.n, stats.agree, stats.percent) == (20, 10, 50.0)
    assert stats.kappa == pytest.approx(0.0, abs=1e-12) and stats.kappa_text == "0.000"
    assert stats.weak == ("kappa < 0.4",)


def test_a_table_with_a_known_kappa():
    """8 / 2 / 4 / 6: po = 14/20 = 0.7, pe = 0.5 x 0.6 + 0.5 x 0.4 = 0.5, kappa = 0.2 / 0.5 = 0.4."""
    stats = pair_stats(table((8, 2, 4, 6)))
    assert stats.percent == 70.0 and stats.kappa == pytest.approx(0.4)


def test_a_one_class_table_has_no_kappa_and_says_so_without_nan_or_a_raise(recwarn):
    stats = pair_stats([(1, 1)] * 12)
    assert (stats.n, stats.agree, stats.percent) == (12, 12, 100.0)
    assert stats.kappa is None and stats.kappa_text == "n/a (one class)"
    assert stats.weak == ("n=12 < 20", "kappa undefined") and "nan" not in stats.flag + stats.kappa_text
    assert not [w for w in recwarn.list if "single label" in str(w.message) or "undefined" in str(w.message)]  # sklearn is not even called


def test_a_one_class_table_at_n_20_or_more_is_still_weak_because_kappa_is_undefined():
    assert pair_stats([(0, 0)] * 25).weak == ("kappa undefined",)


def test_all_pass_against_all_fail_is_0_percent_and_a_defined_kappa_or_n_a_never_nan():
    stats = pair_stats([(1, 0)] * 6)
    assert stats.percent == 0.0 and "nan" not in stats.kappa_text and stats.weak


def test_no_pairs_is_n_0_with_no_percent_and_is_weak():
    stats = pair_stats([])
    assert (stats.n, stats.percent, stats.kappa, stats.percent_text, stats.kappa_text) == (0, None, None, "n/a", "n/a (no pairs)")
    assert stats.weak == ("n=0 < 20", "kappa undefined")


# --- the weak rule: n = 19 / 20 and kappa = 0.39 / 0.40 --------------------------------------------------------------------------


def test_the_thresholds_are_20_and_0_4():
    assert (MIN_N, MIN_KAPPA) == (20, 0.4)


def test_n_19_is_weak_and_n_20_is_not_with_the_same_perfect_agreement():
    assert pair_stats(table((10, 0, 0, 9))).weak == ("n=19 < 20",)
    assert pair_stats(table((10, 0, 0, 10))).weak == ()


def test_kappa_0_39_is_weak_and_kappa_0_40_is_not_at_n_20():
    """6 / 3 / 3 / 8 gives po = 0.7 and pe = 0.505, kappa = 0.195 / 0.495 = 0.394; 8 / 2 / 4 / 6 gives exactly 0.4."""
    low, edge = pair_stats(table((6, 3, 3, 8))), pair_stats(table((8, 2, 4, 6)))
    assert low.n == edge.n == 20
    assert low.kappa == pytest.approx(0.3939, abs=1e-4) and low.weak == ("kappa < 0.4",)
    assert edge.kappa == pytest.approx(0.4) and edge.weak == ()


@pytest.mark.parametrize(
    ("n", "kappa", "reasons"),
    [
        (19, 0.39, ("n=19 < 20", "kappa < 0.4")),
        (19, 0.4, ("n=19 < 20",)),
        (20, 0.39, ("kappa < 0.4",)),
        (20, 0.4, ()),
        (20, 0.3999999999999999, ()),  # float noise on a kappa that is 0.4
        (39, 0.95, ()),
        (15, 1.0, ("n=15 < 20",)),
        (30, None, ("kappa undefined",)),
    ],
)
def test_weak_reasons(n, kappa, reasons):
    assert weak_reasons(n, kappa) == reasons


def test_the_flag_text_lists_every_reason():
    assert Stats(15, 12, 0.3, ("n=15 < 20", "kappa < 0.4")).flag == "weak (n=15 < 20; kappa < 0.4)"


# --- Dan's label values ----------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("value", ["pass", "PASS", " Pass ", 1, 1.0, "1", True])
def test_these_values_are_a_pass(value):
    assert parse_label(value) == 1


@pytest.mark.parametrize("value", ["fail", "Fail", 0, 0.0, "0", False])
def test_these_values_are_a_fail(value):
    assert parse_label(value) == 0


def test_the_score_is_read_when_there_is_no_value():
    assert (parse_label(None, 1.0), parse_label(None, 0.0)) == (1, 0)


@pytest.mark.parametrize("value", ["maybe", "", "partial", 0.5, 2, "yes", {"a": 1}, ["pass"], None])
def test_anything_else_is_refused(value):
    with pytest.raises(ValueError, match="not pass or fail"):
        parse_label(value)


# --- the scenario: Dan's labels on all 39 answers --------------------------------------------------------------------------------


def judge_run_1(item, criterion):
    """Run 1 (the logged feedback): everything passes, except prioritised on the first four assess answers and good_decline on dev_08_bamburgh's three answers."""
    if criterion == "prioritised":
        return item.example_id not in ASSESS_FIRST_FOUR
    if criterion == "good_decline":
        return item.example_id != "dev_08_bamburgh"
    return True


def dan_labels(item, criterion):
    """Dan agrees with run 1, except: answers_the_question fails on dev_01 and dev_02, and prioritised fails on dev_05_druridge_bay."""
    if criterion == "answers_the_question" and item.example_id in ASSESS_FIRST_FOUR[:2]:
        return False
    if criterion == "prioritised" and item.example_id == "dev_05_druridge_bay":
        return False
    return judge_run_1(item, criterion)


def judge_run_2(item, criterion):
    """Run 2 (the fresh pass) differs from run 1 on prioritised dev_04 (now fails the first three, not four: passes dev_04) and on good_decline dev_08 rep 3 (passes)."""
    if criterion == "prioritised":
        return item.example_id not in ASSESS_FIRST_FOUR[:3]
    if criterion == "good_decline":
        return not (item.example_id == "dev_08_bamburgh" and item.rep != 3)
    return True


class Setup:
    def __init__(self, tmp_path, *, experiment=None, judge_says=judge_run_1, dan_says=dan_labels, label=True, seed=True):
        self.experiment = experiment or fake_experiment()
        self.client = CalibrationClient(self.experiment)
        self.items = []  # seed=False: an experiment `select` refuses, so there is nothing to label
        if seed:
            self.items = agree_with_judge(self.client, self.experiment, judge_says, dan_says if label else lambda item, criterion: None)
            self.client.forget_calls()  # `make annotate` and the labelling are the set-up; the record is of `make agreement` alone
        self.results, self.cache = tmp_path / "results", tmp_path / "cache"
        self.judges_built = 0
        self.judge = None

    def factory(self, passed=judge_run_2, count=39, start=0):
        def build(environ):
            self.judges_built += 1
            self.judge = judge_for(passed, count, self.items, start)
            return self.judge

        return build

    def run(self, name=EXPERIMENT, *, judge_factory: Any = "default", environ=None, **options):
        out, err = io.StringIO(), io.StringIO()
        factory = self.factory() if judge_factory == "default" else judge_factory
        code = agreement.main(
            [], environ={"EXPERIMENT": name, **(environ or {})}, client=self.client, judge_factory=factory, results_dir=self.results, cache_dir=self.cache, stdout=out, stderr=err, **options
        )
        return code, out.getvalue(), err.getvalue()

    @property
    def report(self):
        return (self.results / f"{EXPERIMENT}-agreement.md").read_text(encoding="utf-8")

    @property
    def model_calls(self):
        return len(fake_model(self.judge).calls) if self.judge else 0


def rows_of(text, heading):
    """{criterion: cells} of the table under `## heading`."""
    block = text.split(f"## {heading}")[1].split("##")[0]
    found = {}
    for line in block.splitlines():
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if cells and cells[0] in CRITERIA:
            found[cells[0]] = cells[1:]
    return found


@pytest.fixture
def labelled(tmp_path):
    return Setup(tmp_path)


# --- the golden path (GP7) -------------------------------------------------------------------------------------------------------


def test_the_report_has_the_judge_vs_dan_table_with_each_criterion_n_agreement_and_kappa(labelled):
    code, out, err = labelled.run()
    assert (code, err) == (0, "")
    rows = rows_of(out, "Judge vs Dan")
    assert list(rows) == list(CRITERIA)
    # columns: n, agreement, kappa, unlabelled, no judge grade, flag
    assert rows["answers_the_question"][:3] == ["39", "94.9%", "0.000"]  # judge passed every one, Dan failed two: 37 of 39 agree, no better than chance
    assert rows["prioritised"][:3] == ["21", "95.2%", "0.859"]  # 17 + 4 = 21 answers (16 assess + 5 explain); judge fails 4, Dan fails 5
    assert rows["actionable"][:3] == ["21", "100.0%", "n/a (one class)"]  # every label on both sides is a pass
    assert rows["good_decline"][:3] == ["15", "100.0%", "1.000"]  # 15 decline answers; perfect agreement on three fails and twelve passes
    assert all(row[3] == "0" for row in rows.values())  # nothing left unlabelled


def test_the_n_of_each_criterion_counts_only_the_answers_the_criterion_applies_to(labelled):
    """E5: 39 answers, but prioritised and actionable apply to 21 (assess and explain), good_decline to 15 (decline), answers_the_question to all 39."""
    _, out, _ = labelled.run()
    assert [rows_of(out, "Judge vs Dan")[name][0] for name in CRITERIA] == ["39", "21", "21", "15"]


def test_the_decline_line_names_the_answers_and_the_questions_it_rests_on(labelled):
    _, out, _ = labelled.run()
    assert "15 decline answers from 5 questions (the good_decline figures rest on these)." in out.splitlines()


def test_good_decline_is_flagged_weak_and_so_is_every_kappa_that_cannot_be_trusted(labelled):
    _, out, _ = labelled.run()
    rows = rows_of(out, "Judge vs Dan")
    assert rows["good_decline"][-1] == "weak (n=15 < 20)"
    assert rows["answers_the_question"][-1] == "weak (kappa < 0.4)"
    assert rows["actionable"][-1] == "weak (kappa undefined)"
    assert rows["prioritised"][-1] == ""  # n = 21 and kappa 0.859: sound


def test_the_self_agreement_table_compares_the_logged_feedback_with_one_fresh_judge_pass(labelled):
    code, out, _ = labelled.run()
    assert code == 0
    rows = rows_of(out, "Judge self-agreement")
    # columns: n, agreement, kappa, missing a grade, flag
    assert rows["answers_the_question"][:3] == ["39", "100.0%", "n/a (one class)"]  # both runs pass every one
    assert rows["prioritised"][:3] == ["21", "95.2%", "0.829"]  # run 1 fails four, run 2 fails three: dev_04 differs
    assert rows["actionable"][:3] == ["21", "100.0%", "n/a (one class)"]
    assert rows["good_decline"][:3] == ["15", "93.3%", "0.762"]  # run 1 fails three (dev_08), run 2 fails two: po = 14/15, pe = 0.72
    assert rows["good_decline"][-1].startswith("weak (n=15 < 20")


def test_the_second_judge_pass_is_one_model_call_per_answer_over_the_39_answers(labelled):
    labelled.run()
    assert labelled.judges_built == 1 and labelled.model_calls == 39


def test_the_report_is_printed_and_written_and_the_two_are_the_same(labelled):
    code, out, _ = labelled.run()
    path = labelled.results / f"{EXPERIMENT}-agreement.md"
    assert code == 0 and path.exists()
    assert out == labelled.report + f"\nwritten: {path}\n"
    assert labelled.report.startswith(f"# Judge calibration: {EXPERIMENT}\n")
    assert "39 answers (annotation queue flysafelz-calibration-v1-dev-20261108-160700), 39 of 39 labelled by Dan." in labelled.report


def test_the_report_has_no_answer_text_and_no_nan(labelled):
    labelled.run()
    assert "Invented answer" not in labelled.report and "nan" not in labelled.report.lower().replace("n/a", "")


def test_the_report_names_the_self_preference_risk(labelled):
    labelled.run()
    assert "self-preference risk" in labelled.report


def test_nothing_is_written_to_langsmith(labelled):
    labelled.run()
    assert labelled.client.writes() == []
    assert set(labelled.client.methods_called()) <= {"read_project", "read_dataset", "list_runs", "has_project", "list_feedback"}


def test_the_files_written_are_the_report_and_the_cache_only(labelled, tmp_path):
    labelled.run()
    assert sorted(path.name for path in tmp_path.rglob("*") if path.is_file()) == [f"{EXPERIMENT}-agreement.md", f"{EXPERIMENT}-judge2.json"]


# --- the cache (F2, F3) ----------------------------------------------------------------------------------------------------------


def test_the_cache_holds_the_raw_verdicts_of_the_second_pass_keyed_by_run(labelled):
    labelled.run()
    data = json.loads((labelled.cache / f"{EXPERIMENT}-judge2.json").read_text(encoding="utf-8"))
    assert data["experiment"] == EXPERIMENT and sorted(data["verdicts"]) == sorted(item.run_id for item in labelled.items)
    one = data["verdicts"][labelled.items[0].run_id]
    assert sorted(one) == ["judge.actionable", "judge.answers_the_question", "judge.good_decline", "judge.prioritised"]
    assert one["judge.answers_the_question"] == {"score": 1.0, "comment": "again"} and one["judge.good_decline"] is None  # not graded for an assess answer


def test_a_second_run_calls_the_judge_for_nothing_and_prints_the_same_report(labelled):
    labelled.run()
    first = labelled.report
    built_before, calls_before = labelled.judges_built, labelled.model_calls

    def never(environ):
        raise AssertionError("the judge was built although the cache holds every answer")

    code, out, _ = labelled.run(judge_factory=never)
    assert code == 0
    assert (labelled.judges_built, labelled.model_calls) == (built_before, calls_before)
    assert labelled.report.replace("0 answers judged now, 39 from the cache", "39 answers judged now, 0 from the cache") == first
    assert "0 answers judged now, 39 from the cache" in out


def test_a_complete_cache_needs_no_judge_settings(labelled):
    labelled.run()
    code, _, err = labelled.run(judge_factory=None)  # no factory: the real build_judge would need JUDGE_* in the environment
    assert (code, err) == (0, "")


def test_a_judge_error_is_not_cached_and_the_answer_drops_out_of_the_self_agreement_until_the_rerun(tmp_path):
    setup = Setup(tmp_path)
    ok = judge_for(judge_run_2, 39, setup.items)
    fake_model(ok).replies[10] = RuntimeError("the endpoint timed out")  # the 11th answer's judge call fails

    code, out, _ = setup.run(judge_factory=lambda environ: ok)
    assert code == 0
    assert "38 answers judged now, 0 from the cache, 1 judge errors" in out
    cached = json.loads((setup.cache / f"{EXPERIMENT}-judge2.json").read_text(encoding="utf-8"))["verdicts"]
    assert len(cached) == 38 and setup.items[10].run_id not in cached
    missing = [rows_of(out, "Judge self-agreement")[name][3] for name in CRITERIA]
    assert sum(int(cell) for cell in missing) == len(setup.items[10].criteria)

    rerun = setup.factory(count=1, start=10)  # the rerun judges only the one answer the cache does not hold
    code, out, _ = setup.run(judge_factory=rerun)
    assert code == 0 and setup.model_calls == 1 and "1 answers judged now, 38 from the cache" in out
    assert rows_of(out, "Judge self-agreement")["answers_the_question"][0] == "39"


def test_a_corrupt_cache_is_exit_2_and_is_not_overwritten(labelled):
    labelled.cache.mkdir(parents=True)
    path = labelled.cache / f"{EXPERIMENT}-judge2.json"
    path.write_text("{not json", encoding="utf-8")
    code, _, err = labelled.run()
    assert code == 2 and "cannot be read" in err and "delete it to judge again" in err
    assert path.read_text(encoding="utf-8") == "{not json"
    assert labelled.judges_built == 0 or labelled.model_calls == 0


# --- the judge settings (E: JUDGE_* unset) ---------------------------------------------------------------------------------------


def test_unset_judge_settings_are_exit_2_naming_each_one_before_any_langsmith_call(labelled):
    code, out, err = labelled.run(judge_factory=None)  # the real builder, with the settings of an empty environment
    assert code == 2 and out == ""
    for name in ("JUDGE_PROVIDER", "JUDGE_BASE_URL", "JUDGE_API_KEY", "JUDGE_MODEL"):
        assert name in err
    assert labelled.client.calls == []  # not a single queue read, run read or feedback read came first
    assert not labelled.cache.exists() and not labelled.results.exists()


def test_the_judge_error_is_the_judge_modules_own(labelled):
    def raising(environ):
        raise JudgeConfigError(("JUDGE_MODEL",))

    code, _, err = labelled.run(judge_factory=raising)
    assert code == 2 and err.startswith("evals.agent.agreement: ") and "JUDGE_MODEL" in err


# --- partial and missing labels (E5) -------------------------------------------------------------------------------------------


def test_with_partial_labels_only_the_labelled_pairs_count_and_the_unlabelled_are_named_per_criterion(tmp_path):
    def some(item, criterion):  # Dan labelled the first ten answers only
        return judge_run_1(item, criterion) if labelled_ids.index(item.run_id) < 10 else None

    experiment = fake_experiment()
    labelled_ids = [item.run_id for item in queued_items(experiment)]
    setup = Setup(tmp_path, experiment=experiment, dan_says=some)
    code, out, _ = setup.run()
    assert code == 0
    rows = rows_of(out, "Judge vs Dan")
    first_ten = setup.items[:10]
    for name in CRITERIA:
        applicable = [item for item in setup.items if name in item.criteria]
        counted = [item for item in first_ten if name in item.criteria]
        assert rows[name][0] == str(len(counted)) and rows[name][3] == str(len(applicable) - len(counted)), name
    assert "10 of 39 labelled by Dan" in out


def test_an_answer_labelled_on_one_criterion_only_is_counted_on_that_one(tmp_path):
    experiment = fake_experiment()
    setup = Setup(tmp_path, experiment=experiment, dan_says=lambda item, criterion: judge_run_1(item, criterion) if criterion == "answers_the_question" else None)
    _, out, _ = setup.run()
    rows = rows_of(out, "Judge vs Dan")
    assert rows["answers_the_question"][0] == "39" and rows["prioritised"][0] == "0" and rows["prioritised"][3] == "21"
    assert rows["prioritised"][2] == "n/a (no pairs)" and rows["prioritised"][-1] == "weak (n=0 < 20; kappa undefined)"


def test_no_labels_at_all_is_exit_2_saying_0_of_39_answers_labelled_and_calls_no_judge(tmp_path):
    setup = Setup(tmp_path, label=False)
    code, out, err = setup.run()
    assert code == 2 and out == ""
    assert "0 of 39 answers labelled" in err and "flysafelz-calibration-v1-dev-20261108-160700" in err
    assert setup.model_calls == 0 and not setup.cache.exists() and not setup.results.exists()


def test_the_judges_own_feedback_is_not_a_label(tmp_path):
    """Only run-1 judge feedback and nothing of Dan's: the judge's `judge.*` keys are never read as his labels."""
    setup = Setup(tmp_path, label=False)
    assert any(item.key.startswith("judge.") for item in setup.client.feedback)
    assert setup.run()[0] == 2


def test_a_label_on_the_experiments_own_run_is_not_dans_he_labels_the_queued_copy(tmp_path):
    experiment = fake_experiment()
    setup = Setup(tmp_path, experiment=experiment, label=False)
    for item in setup.items:
        for criterion in item.criteria:
            setup.client.feedback.append(dan_feedback(experiment.run(item.example_id, item.rep), criterion, "pass"))
    code, _, err = setup.run()
    assert code == 2 and "0 of 39 answers labelled" in err


def test_a_label_on_a_copy_is_matched_to_the_judge_grade_of_the_run_it_was_copied_from(tmp_path):
    experiment = fake_experiment()
    setup = Setup(tmp_path, experiment=experiment, judge_says=lambda item, criterion: False, dan_says=lambda item, criterion: True)
    _, out, _ = setup.run()
    rows = rows_of(out, "Judge vs Dan")
    assert rows["answers_the_question"][:2] == ["39", "0.0%"]  # judge: fail on the experiment's run, Dan: pass on the copy


def test_judge_feedback_on_a_copy_is_neither_a_label_nor_an_ignored_key(tmp_path):
    experiment = fake_experiment()
    setup = Setup(tmp_path, experiment=experiment)
    run = setup.client.copy_of(experiment.run("dev_01_blue_mill", 1))
    setup.client.feedback.append(judge_feedback(run, "prioritised", False))
    _, out, _ = setup.run()
    assert "Ignored feedback keys" not in out and rows_of(out, "Judge vs Dan")["prioritised"][:2] == ["21", "95.2%"]


def test_before_make_annotate_there_is_no_calibration_project_and_nothing_is_labelled(tmp_path):
    experiment = fake_experiment()
    client = CalibrationClient(experiment)
    out, err = io.StringIO(), io.StringIO()
    code = agreement.main([], environ={"EXPERIMENT": EXPERIMENT}, client=client, judge_factory=lambda env: None, results_dir=tmp_path / "r", cache_dir=tmp_path / "c", stdout=out, stderr=err)
    assert code == 2 and "0 of 39 answers labelled" in err.getvalue() and client.writes() == []


def test_a_label_of_a_criterion_that_does_not_apply_to_the_path_is_ignored_and_counted(tmp_path):
    experiment = fake_experiment()
    setup = Setup(tmp_path, experiment=experiment)
    clarify = next(item for item in setup.items if item.path == "clarify")
    run = setup.client.copy_of(experiment.run(clarify.example_id, clarify.rep))
    setup.client.feedback.append(dan_feedback(run, "good_decline", "fail"))
    _, out, _ = setup.run()
    rows = rows_of(out, "Judge vs Dan")
    assert rows["good_decline"][0] == "15"  # the stray label is neither a pair nor a disagreement
    assert "1 label(s) of Dan's are on a criterion that does not apply to the answer's path: ignored." in out


def test_a_criterion_that_does_not_apply_is_not_a_disagreement_even_if_the_judge_graded_it(tmp_path):
    experiment = fake_experiment()
    setup = Setup(tmp_path, experiment=experiment)
    clarify = next(item for item in setup.items if item.path == "clarify")
    run = experiment.run(clarify.example_id, clarify.rep)
    setup.client.feedback.append(judge_feedback(run, "prioritised", False))  # a stray judge grade on a path that does not use it
    _, out, _ = setup.run()
    assert rows_of(out, "Judge vs Dan")["prioritised"][:2] == ["21", "95.2%"]


# --- bad and odd labels -------------------------------------------------------------------------------------------------------


def test_a_label_that_is_not_pass_or_fail_is_exit_2_naming_the_run_and_the_key(tmp_path):
    experiment = fake_experiment()
    setup = Setup(tmp_path, experiment=experiment, dan_says=lambda item, criterion: "maybe" if (item.example_id, criterion) == ("dev_03_buttermere", "prioritised") else judge_run_1(item, criterion))
    code, out, err = setup.run()
    run_id = experiment.run("dev_03_buttermere", 1).id
    assert code == 2 and out == ""
    assert f"run {run_id}" in err and "'prioritised'" in err and "'maybe' is not pass or fail" in err
    assert setup.model_calls == 0 and not setup.results.exists()  # a bad label stops the run before the judge is called and before a report is written


def test_a_numeric_label_of_1_or_0_is_read_as_pass_or_fail(tmp_path):
    experiment = fake_experiment()
    setup = Setup(tmp_path, experiment=experiment, label=False)
    for item in setup.items:
        run = setup.client.copy_of(experiment.run(item.example_id, item.rep))
        for criterion in item.criteria:
            setup.client.feedback.append(dan_feedback(run, criterion, None, score=1.0 if judge_run_1(item, criterion) else 0.0))
    code, out, _ = setup.run()
    assert code == 0 and rows_of(out, "Judge vs Dan")["good_decline"][:3] == ["15", "100.0%", "1.000"]


def test_a_key_that_is_not_one_of_the_four_criteria_is_ignored_and_listed(tmp_path):
    experiment = fake_experiment()
    setup = Setup(tmp_path, experiment=experiment)
    run = setup.client.copy_of(experiment.run("dev_01_blue_mill", 1))
    setup.client.feedback += [dan_feedback(run, "tone", "pass"), dan_feedback(setup.client.copy_of(experiment.run("dev_02_west_sleekburn", 1)), "tone", "fail"), dan_feedback(run, "notes", "ok")]
    code, out, _ = setup.run()
    assert code == 0
    assert "Ignored feedback keys (not one of the four criteria): notes (1), tone (2)." in out


def test_feedback_a_model_wrote_is_never_a_label(tmp_path):
    experiment = fake_experiment()
    setup = Setup(tmp_path, experiment=experiment, label=False)
    for item in setup.items:
        run = setup.client.copy_of(experiment.run(item.example_id, item.rep))
        setup.client.feedback.append(dan_feedback(run, "answers_the_question", "pass", source="model"))
    assert setup.run()[0] == 2  # 0 of 39 labelled: a model-written feedback under a criterion's name does not count


def test_the_latest_label_of_a_run_wins(tmp_path):
    experiment = fake_experiment()
    setup = Setup(tmp_path, experiment=experiment)
    run = setup.client.copy_of(experiment.run("dev_01_blue_mill", 1))
    setup.client.feedback.append(dan_feedback(run, "answers_the_question", "pass", minute=30))  # Dan changed his mind: pass (he had failed it)
    _, out, _ = setup.run()
    assert rows_of(out, "Judge vs Dan")["answers_the_question"][:2] == ["39", "97.4%"]  # only dev_02 is still a disagreement


# --- the judge's own run-1 grades ----------------------------------------------------------------------------------------------


def test_an_answer_whose_logged_judge_feedback_was_an_error_is_counted_as_having_no_grade(tmp_path):
    experiment = fake_experiment()
    setup = Setup(tmp_path, experiment=experiment)
    victim = setup.items[0]
    setup.client.feedback = [f for f in setup.client.feedback if not (str(f.run_id) == victim.run_id and f.key == "judge.answers_the_question")]
    setup.client.feedback.append(judge_feedback(experiment.run(victim.example_id, victim.rep), "answers_the_question", None))
    _, out, _ = setup.run()
    row = rows_of(out, "Judge vs Dan")["answers_the_question"]
    assert row[0] == "38" and row[4] == "1"


# --- refusals and failures, shared with annotate --------------------------------------------------------------------------------


@pytest.mark.parametrize(("name", "reason"), [("flysafelz-held-out", "held-out experiment"), ("v0-dev-20261108-160700", "v0 experiment"), ("../x", "must be letters, digits")])
def test_a_held_out_v0_or_bad_name_is_refused_before_any_langsmith_call(tmp_path, name, reason):
    setup = Setup(tmp_path)
    code, out, err = setup.run(name)
    assert code == 2 and out == "" and reason in err
    assert setup.client.calls == []


def test_an_unknown_experiment_and_one_with_too_few_reps_are_refused(tmp_path):
    setup = Setup(tmp_path)
    assert "was not found in LangSmith" in setup.run("v1-dev-20990101-000000")[2]
    short = Setup(tmp_path / "short", experiment=fake_experiment(reps=(1, 2)), seed=False)
    assert short.run()[0] == 2


def test_a_held_out_run_among_the_experiments_runs_is_refused(tmp_path):
    experiment = fake_experiment(splits={("dev_14_town_moor", 3): "held_out"})
    setup = Setup(tmp_path, experiment=experiment, seed=False)
    code, _, err = setup.run()
    assert code == 2 and "held_out example" in err and setup.model_calls == 0


@pytest.mark.parametrize("value", [None, "", "  "])
def test_a_missing_experiment_is_the_usage_and_exit_2_with_no_client_and_no_judge_built(value):
    def boom(environ):
        raise AssertionError("built")

    out, err = io.StringIO(), io.StringIO()
    code = agreement.main([], environ={} if value is None else {"EXPERIMENT": value}, client_factory=boom, judge_factory=boom, stdout=out, stderr=err)
    assert code == 2 and out.getvalue() == ""
    assert err.getvalue().startswith("usage: make agreement EXPERIMENT=<v1 dev experiment name>") and "nothing was sent to LangSmith" in err.getvalue()


def test_arguments_on_the_command_line_are_refused():
    out, err = io.StringIO(), io.StringIO()
    code = agreement.main(["x"], environ={"EXPERIMENT": EXPERIMENT}, stdout=out, stderr=err)
    assert code == 2 and "takes no arguments" in err.getvalue()


def test_a_langsmith_failure_reading_the_labels_is_exit_1_and_says_the_rerun_resumes(tmp_path):
    setup = Setup(tmp_path)
    setup.client._fail_on = {"list_feedback": 1}
    code, _, err = setup.run()
    assert code == 1 and "RuntimeError" in err and "answers already judged are not judged twice" in err
    assert not setup.results.exists()


def test_a_langsmith_failure_after_some_answers_are_judged_keeps_what_was_paid_for(tmp_path):
    setup = Setup(tmp_path)
    judge = judge_for(judge_run_2, 39, setup.items)

    class Crash(KeyboardInterrupt):  # not an Exception: the process is stopped in the middle, as with Ctrl-C
        pass

    fake_model(judge).replies[7] = Crash()
    with pytest.raises(Crash):
        setup.run(judge_factory=lambda environ: judge)
    cached = json.loads((setup.cache / f"{EXPERIMENT}-judge2.json").read_text(encoding="utf-8"))["verdicts"]
    assert len(cached) == 7 and not list(setup.cache.glob("*.tmp"))  # the first seven are kept, whole, and no half-written file is left
    rerun = setup.factory(count=32, start=7)
    assert setup.run(judge_factory=rerun)[0] == 0 and setup.model_calls == 32


def test_the_annotate_and_agreement_commands_choose_the_same_answers(labelled):
    ids = [item.run_id for item in annotate.select(labelled.experiment.runs, labelled.experiment.records).items]
    assert ids == [item.run_id for item in labelled.items]
    assert Path(agreement.__file__).name == "agreement.py"
