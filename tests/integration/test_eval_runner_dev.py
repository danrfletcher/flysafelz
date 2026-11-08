"""The runner on the real committed dev examples (5a PR-4.F1, GP2): the real PR-3.F1 loader, a scripted fake graph and a recording LangSmith client.

`uv run pytest tests/integration/test_eval_runner_dev.py -s` prints the run summary of the scripted run. Nothing real is called: the settings are fake values,
the graph is the fake from tests/unit/eval_runner_helpers.py and the client records. The shipping command has no fake-graph flag.

Held-out is never read here (F7): the dev ids come from the dev loader, and no held-out id is named.
"""

import io
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import pytest
from eval_runner_helpers import FAKE_ENV, FakeSleeper, RecordingTimeout, RunRecordingClient, ScriptedGraph

from evals.agent import run_v1, runner
from evals.agent.loader import REPO, load_examples

STAMP = datetime(2026, 11, 8, 16, 7, tzinfo=UTC)
EXPERIMENT = "v1-dev-20261108-160700"
CACHE = REPO / "data" / "eval-cache" / f"{EXPERIMENT}.sqlite"


@pytest.fixture(scope="module")
def dev_ids():
    return [r["id"] for r in load_examples("dev")]


@pytest.fixture(scope="module")
def scripted_run():
    """The whole `python -m evals.agent.run_v1` (dev by default) on the committed data, once: (exit code, stdout, stderr, graph, client)."""
    assert not CACHE.exists()
    graph, client = ScriptedGraph(), RunRecordingClient()
    out, err = io.StringIO(), io.StringIO()
    code = run_v1.main(
        [],
        environ=FAKE_ENV,
        stdout=out,
        stderr=err,
        graph_factory=graph.factory,
        client_factory=lambda env: client,
        moment=STAMP,
        sleep=FakeSleeper(),
        timeout=RecordingTimeout(),
    )
    return code, out.getvalue(), err.getvalue(), graph, client


def test_the_dev_split_runs_29_examples_three_reps_87_runs_in_rep_order(scripted_run, dev_ids):
    code, out, err, graph, _client = scripted_run
    assert (code, err) == (0, "")
    assert len(dev_ids) == 29
    assert graph.order() == [(i, rep) for rep in (1, 2, 3) for i in dev_ids]
    assert graph.peak == 1
    print()  # the run summary, as the command prints it (shown with -s)
    print(out, end="")
    assert "87 of 87 runs" in out and "errored: 0 of 87 runs" in out and "incomplete: false" in out


def test_every_run_has_live_reader_the_experiments_cache_and_its_own_frozen_now(scripted_run, dev_ids):
    _, _, _, graph, _ = scripted_run
    by_id = {r["id"]: r for r in load_examples("dev")}
    for call in graph.calls:
        record = by_id[call.example_id]
        assert call.context.reader == "live" and call.context.reader_cache == CACHE
        assert call.context.case == record["fixtures"]["case"] and call.context.drop == tuple(record["fixtures"]["drop"])
        assert call.context.now == datetime.fromisoformat(record["inputs"]["now"])


def test_one_experiment_in_langsmith_with_87_runs_tagged_by_rep(scripted_run):
    _, _, _, _, client = scripted_run
    assert [p["project_name"] for p in client.projects] == [EXPERIMENT]
    assert list(client.datasets) == ["flysafelz-dev"] and client.projects[0]["reference_dataset_id"] == client.datasets["flysafelz-dev"].id
    assert len(client.runs) == 87
    assert [r["extra"]["metadata"]["rep"] for r in client.runs] == [1] * 29 + [2] * 29 + [3] * 29
    assert {t for r in client.runs for t in r["tags"]} == {"rep:1", "rep:2", "rep:3", "v1"}
    assert {r["project_name"] for r in client.runs} == {EXPERIMENT}


def test_the_cache_path_is_under_data_eval_cache_and_the_fake_run_wrote_nothing(scripted_run):
    _, out, _, _, _ = scripted_run
    assert str(REPO / "data" / "eval-cache") in out
    assert not CACHE.exists()


def test_the_dev_examples_that_share_a_case_are_grouped(scripted_run, dev_ids):
    from evals.agent.runner import shared_case_groups

    groups = shared_case_groups(load_examples("dev"))
    assert groups and all(len(g) > 1 and set(g) <= set(dev_ids) for g in groups)
    assert sum(len(g) for g in groups) <= 29


def test_a_loader_refusal_means_no_graph_call_no_client_and_no_cache_file(tmp_path):
    """No fixtures under this root: the real loader refuses (E1) before anything else."""
    cache_dir = tmp_path / "cache"
    out, err = io.StringIO(), io.StringIO()

    def never(*args, **kwargs):
        raise AssertionError("this must not be called")

    code = run_v1.main([], environ=FAKE_ENV, stdout=out, stderr=err, graph_factory=never, client_factory=never, root=tmp_path, cache_dir=cache_dir)
    assert code == 1 and out.getvalue() == ""
    assert "fixture folder(s) missing" in err.getvalue() and err.getvalue().count("\n") >= 2
    assert not cache_dir.exists()


def test_the_real_entry_with_no_settings_names_them_and_builds_and_sends_nothing(tmp_path):
    """The shipping entry in a fresh process with an empty environment: the missing-variable message, exit 1, no cache file."""
    env = {"PATH": os.environ["PATH"], "HOME": os.environ["HOME"], "PYTHONPATH": str(REPO), "READER_DOTENV_PATH": str(tmp_path / "no-such" / ".env")}
    before = sorted(p.name for p in (REPO / "data" / "eval-cache").glob("*")) if (REPO / "data" / "eval-cache").exists() else []
    done = subprocess.run([sys.executable, "-m", "evals.agent.run_v1"], cwd=REPO, env=env, capture_output=True, text=True, timeout=300, check=False)
    assert done.returncode == 1 and done.stdout == ""
    for name in ("AGENT_BASE_URL", "AGENT_MODEL", "AGENT_API_KEY", "READER_PROVIDER", "READER_BASE_URL", "READER_API_KEY", "READER_MODEL", "LANGSMITH_API_KEY"):
        assert name in done.stderr
    assert "nothing was built, sent or written" in done.stderr
    after = sorted(p.name for p in (REPO / "data" / "eval-cache").glob("*")) if (REPO / "data" / "eval-cache").exists() else []
    assert before == after


def test_the_runner_module_stays_importable_without_a_graph():
    assert Path(runner.__file__).name == "runner.py" and runner.CACHE_DIR == REPO / "data" / "eval-cache"
