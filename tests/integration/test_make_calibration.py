"""`make annotate` and `make agreement` end to end (5a PR-8.F1, G13, GP6, GP7).

The usage and refusal paths run as the real make targets in a subprocess (no key, no socket needed: they stop before LangSmith is reached). The golden path runs the two
modules' `main()` on the recording client and a fake judge, and writes the report into `tmp_path`. The real queue and Dan's labels are the human's step at Review.
"""

import io
import os
import shutil
import subprocess
from pathlib import Path

import pytest
from calibration_helpers import CalibrationClient, agree_with_judge, judge_for
from factories import fake_experiment

from evals.agent import agreement, annotate

ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT = "v1-dev-20261108-160700"
NEEDS_TOOLS = pytest.mark.skipif(shutil.which("make") is None or shutil.which("uv") is None, reason="make and uv are needed to run the targets")


def make(target: str, **environ: str) -> subprocess.CompletedProcess[str]:
    env = {key: value for key, value in os.environ.items() if key != "EXPERIMENT"}
    env.update(UV_OFFLINE="1", **environ)
    return subprocess.run(["make", target], cwd=ROOT, env=env, capture_output=True, text=True, check=False, timeout=170)


# --- the real targets --------------------------------------------------------------------------------------------------------


@NEEDS_TOOLS
@pytest.mark.parametrize("target", ["annotate", "agreement"])
def test_the_target_without_experiment_exits_2_with_the_usage_and_sends_nothing(target):
    done = make(target)
    assert done.returncode == 2 and done.stdout == ""
    assert f"usage: make {target} EXPERIMENT=<v1 dev experiment name>" in done.stderr
    assert "nothing was sent to LangSmith" in done.stderr


@NEEDS_TOOLS
@pytest.mark.parametrize("target", ["annotate", "agreement"])
def test_the_target_with_a_blank_experiment_is_the_same(target):
    done = make(target, EXPERIMENT="  ")
    assert done.returncode == 2 and "EXPERIMENT is not set" in done.stderr


@NEEDS_TOOLS
@pytest.mark.parametrize("target", ["annotate", "agreement"])
def test_the_held_out_experiment_is_refused_by_the_target_before_a_key_is_looked_for(target):
    done = make(target, EXPERIMENT="flysafelz-held-out")
    assert done.returncode == 2 and "held-out experiment" in done.stderr and "LANGSMITH" not in done.stderr


@NEEDS_TOOLS
@pytest.mark.parametrize(("target", "module"), [("annotate", "evals.agent.annotate"), ("agreement", "evals.agent.agreement")])
def test_the_recipe_runs_the_module_and_passes_experiment_through_the_environment_not_the_command_line(target, module):
    done = subprocess.run(["make", "-n", target, f"EXPERIMENT={EXPERIMENT}"], cwd=ROOT, capture_output=True, text=True, check=False)
    assert done.returncode == 0 and f"python -m {module}" in done.stdout
    assert EXPERIMENT not in done.stdout  # make exports it to the program; it is not an argument


# --- the two commands, whole, on the recording client --------------------------------------------------------------------------


def test_annotate_then_label_then_agreement_end_to_end(tmp_path):
    experiment = fake_experiment()
    client = CalibrationClient(experiment)
    environ = {"EXPERIMENT": EXPERIMENT}

    out, err = io.StringIO(), io.StringIO()
    assert annotate.main([], environ=environ, client=client, stdout=out, stderr=err) == 0 and err.getvalue() == ""
    assert "39 answers (29 rep 1 + 10 decline from reps 2–3), 96 criterion ticks" in out.getvalue().splitlines()
    queued = client.queued(f"flysafelz-calibration-{EXPERIMENT}")
    assert len(queued) == 39

    before = len(client.feedback)
    assert agreement.main([], environ=environ, client=client, results_dir=tmp_path / "results", cache_dir=tmp_path / "cache", stdout=io.StringIO(), stderr=io.StringIO()) == 2  # not labelled yet
    assert before == len(client.feedback) == 0

    items = agree_with_judge(client, experiment)  # the judge's run-1 grades and Dan's labels, on the answers that were queued
    assert sorted(item.run_id for item in items) == sorted(client.queued_sources(f"flysafelz-calibration-{EXPERIMENT}"))  # the queue holds copies of the 39 answers
    assert not set(queued) & {str(run.id) for run in experiment.runs}  # not the experiment's own runs, which carry the judge's feedback
    out, err = io.StringIO(), io.StringIO()
    code = agreement.main(
        [],
        environ=environ,
        client=client,
        judge_factory=lambda env: judge_for(lambda item, criterion: True, 39, items),
        results_dir=tmp_path / "results",
        cache_dir=tmp_path / "cache",
        stdout=out,
        stderr=err,
    )
    assert (code, err.getvalue()) == (0, "")
    report = tmp_path / "results" / f"{EXPERIMENT}-agreement.md"
    assert report.read_text(encoding="utf-8") + f"\nwritten: {report}\n" == out.getvalue()
    text = out.getvalue()
    assert "## Judge vs Dan" in text and "## Judge self-agreement" in text
    assert "15 decline answers from 5 questions" in text
    good_decline = next(line for line in text.splitlines() if line.startswith("| good_decline"))
    assert "weak (n=15 < 20" in good_decline
    assert (tmp_path / "cache" / f"{EXPERIMENT}-judge2.json").exists()


def test_the_golden_path_writes_the_report_under_the_experiments_name_only(tmp_path):
    experiment = fake_experiment()
    client = CalibrationClient(experiment)
    items = agree_with_judge(client, experiment)
    code = agreement.main(
        [],
        environ={"EXPERIMENT": EXPERIMENT},
        client=client,
        judge_factory=lambda env: judge_for(lambda item, criterion: True, 39, items),
        results_dir=tmp_path / "results",
        cache_dir=tmp_path / "cache",
        stdout=io.StringIO(),
        stderr=io.StringIO(),
    )
    assert code == 0 and [path.name for path in (tmp_path / "results").iterdir()] == [f"{EXPERIMENT}-agreement.md"]


def test_nothing_is_written_into_the_repos_results_folder_by_these_tests():
    assert not list((ROOT / "evals" / "results").glob(f"{EXPERIMENT}*"))
