"""The fence for judge calibration (5a PR-8.F1): F2 no network and no model, F3 neither command runs in `make test` and no key is read, F5 nothing the judge said reaches the queue, F7 no held-out example, F8 nothing written outside the repo.

Every scan here has a planted violation proving it fires.
"""

import ast
import io
import os
import subprocess
import tomllib
from pathlib import Path

import pytest
import yaml
from calibration_helpers import CalibrationClient, agree_with_judge, judge_for, queue_payloads, queued_items
from factories import fake_experiment, judge_feedback

from evals.agent import agreement, annotate
from evals.agent.annotate import calibration_project, queue_name
from evals.agent.loader import example_uuid

REPO = Path(__file__).resolve().parents[2]
MODULES = [REPO / "evals" / "agent" / "annotate.py", REPO / "evals" / "agent" / "agreement.py"]
NAMES = [path.name for path in MODULES]
EXPERIMENT = "v1-dev-20261108-160700"
NETWORK = {"httpx", "requests", "urllib", "socket", "aiohttp", "http", "openai", "langchain_openai", "langchain_anthropic"}
FEEDBACK_WRITERS = {"create_feedback", "update_feedback", "create_presigned_feedback_token", "create_examples", "update_examples", "update_run", "delete_run", "delete_feedback", "evaluate"}


def source_of(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def imported_roots(source: str) -> set[str]:
    roots: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            roots |= {alias.name.split(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom):
            roots.add((node.module or "").split(".")[0])
    return roots


def called_names(source: str) -> set[str]:
    return {
        node.func.attr if isinstance(node.func, ast.Attribute) else node.func.id
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Call) and isinstance(node.func, (ast.Attribute, ast.Name))
    }


def held_out_ids() -> list[str]:
    """The ids of the held-out examples, read as names only (the loader is never asked for the held-out split)."""
    records = yaml.safe_load((REPO / "evals" / "agent" / "data" / "examples.yaml").read_text(encoding="utf-8"))
    return [record["id"] for record in records if record["metadata"]["split"] != "dev"]


# --- F2, F3: no network, no model, no key ------------------------------------------------------------------------------------------


@pytest.mark.parametrize("path", MODULES, ids=NAMES)
def test_neither_module_imports_a_network_library_or_a_model_client(path):
    assert imported_roots(source_of(path)) & NETWORK == set()


def test_the_scan_for_a_network_import_fires_on_a_planted_one():
    assert imported_roots("import httpx\nfrom openai import OpenAI") & NETWORK == {"httpx", "openai"}


def test_sockets_stay_disabled_for_the_whole_suite():
    addopts = tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))["tool"]["pytest"]["ini_options"]["addopts"]
    assert "--disable-socket" in addopts.split()  # `--allow-unix-socket` (map-data routes) only lets asyncio's local socketpair through


def test_the_tests_run_with_no_langsmith_or_judge_setting_in_the_environment():
    assert [name for name in os.environ if name.startswith(("LANGSMITH_", "JUDGE_"))] == [name for name in os.environ if name == "LANGSMITH_TRACING"]


def test_neither_command_builds_a_langsmith_client_when_the_experiment_is_missing_or_refused():
    def boom(environ):
        raise AssertionError("the client was built")

    for main in (annotate.main, agreement.main):
        assert main([], environ={}, client_factory=boom, stdout=io.StringIO(), stderr=io.StringIO()) == 2
        assert main([], environ={"EXPERIMENT": "flysafelz-held-out"}, client_factory=boom, stdout=io.StringIO(), stderr=io.StringIO()) == 2


def test_the_targets_are_not_part_of_make_test_but_their_modules_are_measured():
    makefile = source_of(REPO / "Makefile")
    recipe = makefile.split("\ntest:\n")[1].split("\n\n")[0]
    assert "--cov=evals.agent.annotate" in recipe and "--cov=evals.agent.agreement" in recipe and "--cov-fail-under=90" in recipe
    assert "annotate" not in recipe.replace("--cov=evals.agent.annotate", "") and "agreement" not in recipe.replace("--cov=evals.agent.agreement", "")
    assert "\nannotate:\n\t@uv run" in makefile and "\nagreement:\n\t@uv run" in makefile


def test_no_test_target_depends_on_the_calibration_targets():
    makefile = source_of(REPO / "Makefile")
    assert "test: annotate" not in makefile and "test: agreement" not in makefile and "eval: annotate" not in makefile


# --- F5: nothing the judge said reaches the queue ----------------------------------------------------------------------------------


def annotate_with_judge_feedback_present():
    """What the real service holds when `make annotate` runs: `make eval` has already logged the judge's grades, with reasons, on every answer's run. Dan has labelled nothing yet."""
    experiment = fake_experiment()
    client = CalibrationClient(experiment)
    for item in queued_items(experiment):
        run = experiment.run(item.example_id, item.rep)
        client.feedback += [judge_feedback(run, criterion, criterion != "prioritised") for criterion in item.criteria]
    assert any(feedback.key == "judge.prioritised" and feedback.score == 0.0 for feedback in client.feedback)
    code = annotate.main([], environ={"EXPERIMENT": EXPERIMENT}, client=client, stdout=io.StringIO(), stderr=io.StringIO())
    assert code == 0
    return experiment, client


def feedback_in_front_of_dan(client, name):
    """Every piece of feedback that is already on a queued run: what the annotation view shows Dan beside the answer."""
    return [item for run in client.queued_runs(name) for item in client.feedback_on(run)]


def test_the_queue_holds_no_run_that_has_judge_feedback_or_any_feedback_at_all():
    """F5, the real-service case: the judge's feedback is on the experiment's runs (make eval logged it), and LangSmith shows a run's feedback in the annotation view."""
    experiment, client = annotate_with_judge_feedback_present()
    name = queue_name(EXPERIMENT)
    assert [item for item in client.feedback if item.key.startswith("judge.")]  # the judge's grades and reasons are on the experiment's runs
    queued = client.queued_runs(name)
    assert len(queued) == 39 and not {run.id for run in queued} & {run.id for run in experiment.runs}  # none of the queued runs is a run of the experiment
    assert feedback_in_front_of_dan(client, name) == []
    for run in queued:
        assert run.extra["metadata"]["source_run_id"] in {str(r.id) for r in experiment.runs}  # kept only so that `make agreement` can map the labels back
        assert all(tag.startswith("label:") for tag in run.tags)
        assert not any(word in repr(run.model_dump()) for word in ("judge", "JUDGE-REASON", "pass", "fail"))


def test_the_scan_for_feedback_in_front_of_dan_fires_when_an_experiment_run_is_queued():
    """Planted: queue the experiment's own run (what the first version did) and the scan finds the judge's grade and reason."""
    experiment, client = annotate_with_judge_feedback_present()
    name = queue_name(EXPERIMENT)
    client.queues[name]["run_ids"].append(str(experiment.runs[0].id))
    shown = feedback_in_front_of_dan(client, name)
    assert shown and all(item.key.startswith("judge.") and item.comment == "JUDGE-REASON" for item in shown)


def test_no_judge_verdict_reaches_a_queue_call_or_a_run():
    experiment, client = annotate_with_judge_feedback_present()
    assert all(feedback.comment == "JUDGE-REASON" for feedback in client.feedback if feedback.key.startswith("judge."))
    payloads = queue_payloads(client)
    assert payloads and "JUDGE-REASON" not in payloads and "judge." not in payloads and "judge error" not in payloads
    for call in client.calls_to("add_runs_to_annotation_queue"):
        assert set(call) == {"queue_id", "run_ids"}  # ids only: no text, no grade
    (created,) = client.calls_to("create_annotation_queue")
    assert created["rubric_instructions"] == annotate.instructions()  # the same static text whatever the judge said
    for run in experiment.runs:
        assert not [tag for tag in run.tags or [] if tag.startswith("judge") or tag in ("pass", "fail")]


def test_the_experiments_own_runs_are_left_exactly_as_make_eval_logged_them():
    """R2: no update_run (the SDK would stamp a new end time on the run), so no tag, time or output of an experiment run changes."""
    experiment = fake_experiment()
    client = CalibrationClient(experiment)
    agree_with_judge(client, experiment)
    before = [run.model_dump() for run in experiment.runs]
    client.forget_calls()
    assert annotate.main([], environ={"EXPERIMENT": EXPERIMENT}, client=client, stdout=io.StringIO(), stderr=io.StringIO()) == 0
    assert [run.model_dump() for run in experiment.runs] == before
    assert "update_run" not in client.methods_called() and all(call["project_name"] == calibration_project(EXPERIMENT) for call in client.calls_to("create_run"))


def test_the_only_things_written_are_one_queue_the_copies_logged_in_the_calibration_project_and_the_copies_added_to_it():
    _, client = annotate_with_judge_feedback_present()
    assert set(client.writes()) == {"create_annotation_queue", "add_runs_to_annotation_queue", "create_run"}
    for call in client.calls_to("create_run"):
        assert call["project_name"] == calibration_project(EXPERIMENT) and all(tag.startswith("label:") for tag in call["tags"])
    (created,) = client.calls_to("create_annotation_queue")
    assert set(created) == {"name", "description", "rubric_instructions"}  # no rubric items, no default dataset, no reviewers pre-filled
    assert "answers_the_question" in created["rubric_instructions"] and "judge's" not in created["rubric_instructions"].lower()


def test_no_label_is_pre_filled_and_no_model_writes_annotation_feedback():
    _, client = annotate_with_judge_feedback_present()
    assert not [method for method in client.methods_called() if "feedback" in method]  # annotate never reads or writes feedback at all
    for path in MODULES:
        assert called_names(source_of(path)) & FEEDBACK_WRITERS == set(), path.name


def test_only_annotate_logs_runs_and_agreement_logs_none():
    assert "create_run" in called_names(source_of(MODULES[0])) and "create_run" not in called_names(source_of(MODULES[1]))
    assert "update_run" not in called_names(source_of(MODULES[0]))


def test_agreement_reads_feedback_and_writes_none(tmp_path):
    experiment = fake_experiment()
    client = CalibrationClient(experiment)
    items = agree_with_judge(client, experiment)
    client.forget_calls()  # what follows is `make agreement` alone
    code = agreement.main(
        [],
        environ={"EXPERIMENT": EXPERIMENT},
        client=client,
        judge_factory=lambda environ: judge_for(lambda item, criterion: True, 39, items),
        results_dir=tmp_path / "r",
        cache_dir=tmp_path / "c",
        stdout=io.StringIO(),
        stderr=io.StringIO(),
    )
    assert code == 0 and "list_feedback" in client.methods_called()
    assert client.writes() == [] and not [m for m in client.methods_called() if m in FEEDBACK_WRITERS]


def test_the_scan_for_a_feedback_writer_fires_on_a_planted_call():
    assert called_names("client.create_feedback(run_id, 'prioritised', score=1)") & FEEDBACK_WRITERS == {"create_feedback"}


# --- F7: held-out is never named, loaded or queued ---------------------------------------------------------------------------------


def test_there_are_held_out_examples_to_look_for():
    assert len(held_out_ids()) == 15 and all(not name.startswith("dev_") for name in held_out_ids())


def test_no_held_out_example_id_or_uuid_is_in_any_recorded_queue_call():
    _, client = annotate_with_judge_feedback_present()
    payloads = queue_payloads(client)
    for name in held_out_ids():
        assert name not in payloads and str(example_uuid(name)) not in payloads


def test_no_held_out_example_id_is_in_anything_either_command_prints_or_writes(tmp_path):
    experiment = fake_experiment()
    client = CalibrationClient(experiment)
    agree_with_judge(client, experiment)
    out = io.StringIO()
    annotate.main([], environ={"EXPERIMENT": EXPERIMENT}, client=client, stdout=out, stderr=io.StringIO())
    everything = out.getvalue() + "\n".join(repr(call) for call in client.calls)
    for name in held_out_ids():
        assert name not in everything


@pytest.mark.parametrize("path", MODULES, ids=NAMES)
def test_neither_module_names_a_held_out_id_nor_loads_the_held_out_split(path):
    text = source_of(path)
    assert not [name for name in held_out_ids() if name in text]
    assert 'load_dataset("held_out"' not in text and 'load_examples("held_out"' not in text


def test_a_run_on_a_held_out_example_stops_annotate_before_a_queue_exists():
    """Every item is filtered by its split, not only the experiment's name (F7): one such run among 87 refuses the lot."""
    experiment = fake_experiment(splits={("dev_20_druridge_bay", 2): "held_out"})
    client = CalibrationClient(experiment)
    err = io.StringIO()
    code = annotate.main([], environ={"EXPERIMENT": EXPERIMENT}, client=client, stdout=io.StringIO(), stderr=err)
    assert code == 2 and client.writes() == [] and client.queues == {}


def test_the_dataset_that_is_read_for_the_labels_is_dev_only():
    assert annotate.DEFAULT_SPLIT == "dev" and agreement.DEFAULT_SPLIT == "dev"


# --- F8: nothing is written outside the repo and data/eval-cache/ ------------------------------------------------------------------------


def test_the_default_output_folders_are_inside_the_repo():
    assert agreement.RESULTS_DIR == REPO / "evals" / "results" and agreement.CACHE_DIR == REPO / "data" / "eval-cache"


def test_annotate_writes_no_file_at_all(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    annotate_with_judge_feedback_present()
    assert list(tmp_path.rglob("*")) == []
    assert "write_text" not in source_of(MODULES[0]) and "open(" not in source_of(MODULES[0])


@pytest.mark.parametrize("path", MODULES, ids=NAMES)
def test_neither_module_touches_the_dataset_source_or_the_vault(path):
    text = source_of(path)
    assert "DATASET_SRC" not in text and "vault" not in text.lower() and "obsidian" not in text.lower()


def test_agreement_writes_only_into_the_two_folders_it_is_given(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    experiment = fake_experiment()
    client = CalibrationClient(experiment)
    items = agree_with_judge(client, experiment)
    code = agreement.main(
        [],
        environ={"EXPERIMENT": EXPERIMENT},
        client=client,
        judge_factory=lambda environ: judge_for(lambda item, criterion: True, 39, items),
        results_dir=tmp_path / "results",
        cache_dir=tmp_path / "cache",
        stdout=io.StringIO(),
        stderr=io.StringIO(),
    )
    assert code == 0
    assert sorted(str(path.relative_to(tmp_path)) for path in tmp_path.rglob("*") if path.is_file()) == [f"cache/{EXPERIMENT}-judge2.json", f"results/{EXPERIMENT}-agreement.md"]


def test_the_cache_folder_is_git_ignored_and_the_report_folder_is_not():
    ignored = subprocess.run(["git", "check-ignore", "-q", f"data/eval-cache/{EXPERIMENT}-judge2.json"], cwd=REPO, check=False).returncode
    tracked = subprocess.run(["git", "check-ignore", "-q", f"evals/results/{EXPERIMENT}-agreement.md"], cwd=REPO, check=False).returncode
    assert (ignored, tracked) == (0, 1)
