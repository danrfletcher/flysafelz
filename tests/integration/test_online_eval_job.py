"""`make online-eval` end to end (7 PR-4.F1: GP2, GP4, GP5, E1, E4, E5, E7, and the added cases).

`evals.online.job.main` is the real function and the client is 5a's recording client with the online methods; the judge is a fake with fixed grades. Nothing is mocked in the job or the summary.
The 17 runs are 11 seed runs, 2 pilot runs, and the 4 the job must skip (a `case`, a `judge`, an errored one and one with a `reference_example_id`). The tests print what the evidence files hold (`-s` shows it).
"""

import ast
import hashlib
import re
from datetime import UTC, timedelta
from pathlib import Path

import pytest
from online_helpers import OnlineRecordingClient
from online_job_helpers import CLOCK, FICTION_NUMBER, PROJECT, expected_keys, job_env, posted_keys, results, run_job, seed_trees, world
from online_judge_helpers import OTHER_VERSION, FakeJudge, judge_client, judge_tree

from evals.agent.loader import SPLITS, experiment_target
from evals.online import job

FILE = "online-2026-10-31.md"
PROCESS_MODULES = ("subprocess", "git", "dulwich", "pygit2", "sh")
PROCESS_CALLS = ("system", "popen", "exec", "spawn", "posix_spawn")
SCORED = 13


@pytest.fixture(autouse=True)
def no_keys(monkeypatch):
    monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)
    for name in ("JUDGE_PROVIDER", "JUDGE_BASE_URL", "JUDGE_API_KEY", "JUDGE_MODEL", "JUDGE_SAMPLE_PCT"):
        monkeypatch.delenv(name, raising=False)


def path_of(tree) -> str:
    return ((tree.root.extra or {}).get("metadata") or {})["path"]


def digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def show(label: str, text: str) -> None:
    print(f"\n--- {label}")
    print(text)


# --- GP2 and GP4: the whole job ------------------------------------------------------------------------------------------------------------


def test_gp2_the_job_pulls_scores_all_13_judges_all_13_posts_writes_the_file_and_exits_0(tmp_path, capsys):
    client, trees = world()
    code, judge = run_job(client, tmp_path)
    out = capsys.readouterr().out
    assert code == 0

    pulled = client.list_run_calls()[0]
    assert pulled["project_name"] == PROJECT and pulled["is_root"] is True and pulled["start_time"] == CLOCK - timedelta(days=7)  # SINCE defaults to 7d

    by_run = posted_keys(client)
    scored = [tree for tree in trees if tree.root.id in by_run]
    assert len(scored) == SCORED and len(judge.calls) == SCORED  # all 13 judged: 50 or fewer are all judged
    for tree in scored:
        assert by_run[tree.root.id] == expected_keys(path_of(tree)), tree.root.id
    skipped_ids = {tree.root.id for tree in trees} - set(by_run)
    assert len(skipped_ids) == 4  # the case, judge, errored and experiment runs: no feedback at all
    assert client.duplicate_posts == []

    assert results(tmp_path) == [FILE]
    assert f"summary written: {tmp_path / FILE}" in out
    assert "online eval 2026-10-31: 13 runs scored" in out


def test_gp4_the_file_holds_every_section_and_the_numbers_of_the_run(tmp_path):
    client, _trees = world()
    run_job(client, tmp_path, judge=FakeJudge({"prioritised": 0}))
    text = (tmp_path / FILE).read_text(encoding="utf-8")
    show("online-2026-10-31.md", text)
    for heading in ("## Deterministic checks", "## Judge", "## Failing runs", "## Runs skipped", "## Evaluator version", "## Not scored online"):
        assert heading in text
    assert "13 runs scored at evaluator version " + job.evaluator_version() in text
    assert "| guard_pass | 100.0% | 13/13 | 13 | 0 |" in text
    assert "- skipped by the pull: 4" in text and "case: 1, judge: 1" in text and "errored: 1, experiment: 1, source: 2" in text
    assert "- fully scored at this evaluator version: 13" in text and "- runs with an error: 0" in text
    assert "| assess | prioritised | 0.0% | 5 | 0 |" in text  # 4 seed runs and 1 pilot run
    assert "Missed blocker and verdict match need labels" in text
    assert "https://smith.example/o/org/projects/p/flysafelz/r/" in text  # the failing runs link through get_run_url


def test_gp4_the_failing_runs_are_the_ones_with_a_zero_and_the_unsampled_count_is_shown(tmp_path):
    client, trees = world()
    run_job(client, tmp_path, judge=FakeJudge({"prioritised": 0}))
    text = (tmp_path / FILE).read_text(encoding="utf-8")
    failing = text.split("## Failing runs")[1].split("## Runs skipped")[0]
    prioritised = [tree for tree in trees if path_of(tree) in ("assess", "explain") and tree.root.id in posted_keys(client)]
    assert failing.count("online.judge.prioritised") == len(prioritised)
    assert "Runs marked not_sampled (left out of the judge's sample): 0" in text


# --- GP5: the second run -------------------------------------------------------------------------------------------------------------------


def test_gp5_a_second_run_posts_nothing_and_writes_a_byte_identical_file(tmp_path, capsys):
    client, _trees = world()
    code, _ = run_job(client, tmp_path)
    first = (tmp_path / FILE).read_bytes()
    posts_before = len(client.create_calls())
    second_judge = FakeJudge()
    code_again, _ = run_job(client, tmp_path, judge=second_judge)
    second = (tmp_path / FILE).read_bytes()
    out = capsys.readouterr().out

    assert (code, code_again) == (0, 0)
    posted_second = len(client.create_calls()) - posts_before
    assert posted_second == 0 and second_judge.calls == []
    assert second == first and results(tmp_path) == [FILE]
    assert len(re.findall(r"^[0-9a-f-]{36} already scored", out, re.MULTILINE)) == SCORED and "0 scored, 13 already scored, 0 errored; 0 feedback posted" in out
    print(f"\nrerun: {posted_second} posted, file sha256 {digest(first.decode())} then {digest(second.decode())}: {'identical' if first == second else 'DIFFERENT'}")


def test_gp5_the_rerun_file_is_the_same_whatever_order_the_client_returns_the_runs(tmp_path):
    client, _trees = world()
    run_job(client, tmp_path)
    first = (tmp_path / FILE).read_bytes()
    shuffled = OnlineRecordingClient(list(reversed(client.runs)), children=client.children)
    shuffled.feedback = list(client.feedback)
    run_job(shuffled, tmp_path)
    assert (tmp_path / FILE).read_bytes() == first


def test_a_same_day_rerun_overwrites_the_file_in_place_and_writes_no_second_file(tmp_path):
    client, _ = world()
    run_job(client, tmp_path)
    (tmp_path / FILE).write_text("stale", encoding="utf-8")
    run_job(client, tmp_path)
    assert results(tmp_path) == [FILE] and (tmp_path / FILE).read_text(encoding="utf-8").startswith("# Online eval 2026-10-31")


def test_a_run_on_a_different_day_writes_a_new_dated_file_and_leaves_the_old_one_alone(tmp_path):
    client, _ = world()
    run_job(client, tmp_path)
    first = (tmp_path / FILE).read_bytes()
    judge = FakeJudge()
    run_job(client, tmp_path, clock=CLOCK + timedelta(days=1), judge=judge)  # 1 Nov: the runs are still in the 7 days
    assert results(tmp_path) == [FILE, "online-2026-11-01.md"] and (tmp_path / FILE).read_bytes() == first
    later = (tmp_path / "online-2026-11-01.md").read_text(encoding="utf-8")
    assert later.startswith("# Online eval 2026-11-01") and "13 runs scored" in later  # covers runs scored the day before: this invocation posted none
    assert judge.calls == []
    assert later.split("\n", 1)[1] == first.decode().split("\n", 1)[1]


def test_the_file_is_dated_in_europe_london_so_a_late_utc_run_on_a_bst_day_is_the_next_day(tmp_path):
    client, _ = world()
    from datetime import datetime

    late = datetime(2026, 10, 20, 23, 30, tzinfo=UTC)
    young = [judge_tree(f"recent-{index}", source="pilot", start_time=late - timedelta(hours=2)) for index in range(2)]
    client, _ = world(trees=young)
    run_job(client, tmp_path, clock=late)
    assert results(tmp_path) == ["online-2026-10-21.md"]


def test_the_results_folder_is_made_when_it_is_missing_and_nothing_is_added_to_git(tmp_path):
    client, _ = world()
    folder = tmp_path / "evals" / "results"
    run_job(client, folder)
    assert results(folder) == [FILE]
    for module in sorted(Path(job.__file__).parent.glob("*.py")):  # the whole package: the job, the pull, the summary and the seed
        assert process_or_git_use(module.read_text(encoding="utf-8")) == [], module.name


def process_or_git_use(source: str) -> list[str]:
    """What in the module text starts a process or reaches git: an import of `subprocess` or a git library, a call to `os.system`, `os.popen` or `os.exec*`/`spawn*`, or a call with a string that begins with `git`."""
    found = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            found += [alias.name for alias in node.names if alias.name.split(".")[0] in PROCESS_MODULES]
        elif isinstance(node, ast.ImportFrom) and (node.module or "").split(".")[0] in PROCESS_MODULES:
            found.append(node.module or "")
        elif isinstance(node, ast.Call):
            function = node.func
            if isinstance(function, ast.Attribute) and isinstance(function.value, ast.Name) and function.value.id == "os" and function.attr.startswith(PROCESS_CALLS):
                found.append(f"os.{function.attr}")
            found += [f"call with {arg.value!r}" for arg in node.args if isinstance(arg, ast.Constant) and isinstance(arg.value, str) and arg.value.split(" ")[0] == "git"]
    return found


def test_the_scan_for_a_process_or_git_use_finds_each_kind_in_a_module_that_has_one():
    assert process_or_git_use("import subprocess\n") == ["subprocess"]
    assert process_or_git_use("from subprocess import run\n") == ["subprocess"]
    assert process_or_git_use("import git\n") == ["git"]
    assert process_or_git_use("import os\nos.system('ls')\n") == ["os.system"]
    assert process_or_git_use("run('git add -A')\n") == ["call with 'git add -A'"]
    assert process_or_git_use("import os\nos.environ.get('X')\nprint('github')\n") == []


# --- the older version, the window and the experiments -------------------------------------------------------------------------------------


def test_feedback_of_an_older_version_on_the_same_run_is_not_counted(tmp_path):
    client, trees = world()
    target = trees[0].root
    client.seed_feedback(target, job.feedback_key("step_cap"), version=OTHER_VERSION, score=0)  # a failure scored by older code
    run_job(client, tmp_path)
    text = (tmp_path / FILE).read_text(encoding="utf-8")
    assert "| step_cap | 100.0% | 13/13 | 13 | 0 |" in text
    assert str(target.id) not in text
    assert len([item for item in client.feedback if item.run_id == target.id and item.key == job.feedback_key("step_cap")]) == 2  # kept, and the current version's posted beside it


def test_a_run_scored_on_an_earlier_day_still_counts_when_this_invocation_posts_nothing_for_it(tmp_path):
    client, trees = world()
    version = job.evaluator_version()
    early = trees[0]
    for key in expected_keys("assess")[:-1]:
        client.seed_feedback(early.root, key, version=version, score=0 if key == job.feedback_key("step_cap") else 1)
    client.seed_feedback(early.root, job.VERSION_KEY, version=version)
    posts = len(client.create_calls())
    run_job(client, tmp_path)
    assert not [call for call in client.create_calls()[posts:] if call["run_id"] == early.root.id]
    text = (tmp_path / FILE).read_text(encoding="utf-8")
    assert "13 runs scored" in text and "| step_cap | 92.3% | 12/13 | 13 | 0 |" in text
    assert "| step_cap |" in text and "online.step_cap |" in text.split("## Failing runs")[1]  # the earlier day's failure is in the failing runs


def test_f2_the_job_never_lists_the_held_out_or_an_experiment_project(tmp_path):
    client, _ = world()
    run_job(client, tmp_path)
    protected = {experiment_target(split).project for split in SPLITS}
    projects = {kw.get("project_name") for kw in client.list_run_calls()}
    assert projects == {PROJECT} and not projects & protected
    assert set(client.methods_called()) <= {"list_runs", "read_project", "list_feedback", "create_feedback", "get_run_url"}


def test_e2_a_path_unknown_run_is_in_the_unknown_row_and_the_null_counts_and_is_never_a_failure(tmp_path):
    unknown = judge_tree("unknown-0", path="unknown", source="pilot")
    client, _ = world([unknown])
    code, judge = run_job(client, tmp_path)
    text = (tmp_path / FILE).read_text(encoding="utf-8")
    assert code == 0 and len(judge.calls) == SCORED  # 14 scored runs; the unknown one has no judge criteria
    assert "14 runs scored" in text
    assert "| unknown | no judge criteria for this path (1 run) | n/a | 0 | 0 |" in text
    assert "| decline_no_tools | 0.0% | 0/2 | 2 | 12 (not applicable: 11, path unknown: 1) |" in text.splitlines()
    assert str(unknown.root.id) not in text and "unknown |" not in text.split("## Failing runs")[1].split("## Runs skipped")[0]


# --- E1, E4, E5, E7 ------------------------------------------------------------------------------------------------------------------------


def test_e1_no_runs_in_the_window_writes_the_file_with_0_runs_and_exits_0(tmp_path):
    client = OnlineRecordingClient([])
    code, judge = run_job(client, tmp_path)
    text = (tmp_path / FILE).read_text(encoding="utf-8")
    assert code == 0 and judge.calls == []
    assert "0 runs scored" in text and "No run was scored." in text and "| guard_pass | n/a | 0/0 | 0 | 0 |" in text
    assert "read_project" not in client.methods_called() and client.create_calls() == []
    for heading in ("## Deterministic checks", "## Judge", "## Failing runs", "## Runs skipped", "## Evaluator version", "## Not scored online"):
        assert heading in text


def test_e1_runs_that_are_all_skipped_are_0_scored_and_counted_by_source(tmp_path):
    client, _ = world(trees=[judge_tree("c", source="case"), judge_tree("j", source="judge")])
    code, _ = run_job(client, tmp_path)
    text = (tmp_path / FILE).read_text(encoding="utf-8")
    assert code == 0 and "0 runs scored" in text and "case: 1, judge: 1" in text


def test_e4_with_the_judge_variables_unset_every_deterministic_rate_is_there_and_the_section_names_them(tmp_path, capsys):
    client, _trees = world()

    def refuse(_env):
        raise AssertionError("the judge must not be built with a JUDGE_* variable unset")

    env = job_env()
    for name in ("JUDGE_BASE_URL", "JUDGE_MODEL"):
        del env[name]
    code = job.main([], environ=env, client=client, clock=lambda: CLOCK, build=refuse, results_dir=tmp_path)
    text = (tmp_path / FILE).read_text(encoding="utf-8")
    assert code == 0
    assert "The judge was not run: JUDGE_BASE_URL, JUDGE_MODEL not set." in text
    assert "| guard_pass | 100.0% | 13/13 | 13 | 0 |" in text and "online.judge" not in posted_text(client)
    assert all(f"| {key.removeprefix('online.')} |" in text for key in job.DETERMINISTIC_KEYS)
    assert "- fully scored at this evaluator version: 0" in text  # the marker is posted, but no run has its judge keys yet: a later run fills them in
    assert "judge: not run, JUDGE_BASE_URL, JUDGE_MODEL not set" in capsys.readouterr().out
    # a later run with the judge set fills the judge keys in, and the file then shows rates
    run_job(client, tmp_path)
    assert "The judge was not run" not in (tmp_path / FILE).read_text(encoding="utf-8")
    assert "| assess | prioritised | 100.0% |" in (tmp_path / FILE).read_text(encoding="utf-8")


def posted_text(client):
    return " ".join(item.key for item in client.feedback)


def test_e5_a_post_failure_is_listed_under_its_run_and_the_job_exits_1_after_writing_the_file(tmp_path, capsys):
    client, trees = world()
    victim = trees[0].root
    client.fail_feedback = {(victim.id, job.feedback_key("step_cap"))}
    code, _ = run_job(client, tmp_path)
    out = capsys.readouterr()
    text = (tmp_path / FILE).read_text(encoding="utf-8")
    assert code == 1 and results(tmp_path) == [FILE]  # the summary is never lost
    assert "- runs with an error: 1" in text and f"{victim.id}: post online.step_cap: RuntimeError" in text
    assert "- fully scored at this evaluator version: 12" in text
    assert "1 run(s) had an error" in out.err
    show("E5 stderr", out.err)

    client.fail_feedback = set()  # the rerun posts only what the run is missing
    before = len(client.create_calls())
    code_again, _ = run_job(client, tmp_path)
    again = [call["key"] for call in client.create_calls()[before:]]
    assert code_again == 0 and all(call["run_id"] == victim.id for call in client.create_calls()[before:])
    assert again[0] == job.feedback_key("step_cap") and again[-1] == job.VERSION_KEY and len(again) < len(expected_keys("assess"))
    assert "- runs with an error: 0" in (tmp_path / FILE).read_text(encoding="utf-8")


def test_e5_a_judge_call_that_raises_is_an_error_run_in_the_summary_and_exit_1(tmp_path):
    client, trees = world()
    victim = trees[0].root
    code, _ = run_job(client, tmp_path, judge=FakeJudge(raises=[str(victim.id)]))
    text = (tmp_path / FILE).read_text(encoding="utf-8")
    assert code == 1 and f"{victim.id}: judge: ConnectionError" in text and "bridge returned 503" not in text


def test_e7_the_project_guard_exits_2_before_any_call_and_writes_no_file(tmp_path, capsys):
    for split in SPLITS:
        client, _ = world()
        protected = experiment_target(split).project
        with pytest.raises(SystemExit) as raised:
            run_job(client, tmp_path, env=job_env(LANGSMITH_PROJECT=protected))
        assert raised.value.code == 2
        assert protected in capsys.readouterr().err
        assert client.calls == [] and results(tmp_path) == []


@pytest.mark.parametrize(
    ("variable", "value"),
    [("SINCE", "7x"), ("SINCE", "0d"), ("LIMIT", "many"), ("LIMIT", "0"), ("JUDGE_SAMPLE_PCT", "101"), ("JUDGE_SAMPLE_PCT", "-1")],
)
def test_gp2_a_bad_since_limit_or_sample_percentage_exits_2_naming_the_variable_before_any_call(tmp_path, capsys, variable, value):
    client, _ = world()
    with pytest.raises(SystemExit) as raised:
        run_job(client, tmp_path, env=job_env(**{variable: value}))
    assert raised.value.code == 2
    assert variable in capsys.readouterr().err
    assert client.calls == [] and results(tmp_path) == []


def test_gp2_since_limit_and_the_sample_percentage_reach_the_job_through_the_environment(tmp_path):
    client, _ = world()
    judge = FakeJudge()
    run_job(client, tmp_path, env=job_env(SINCE="24h", LIMIT="5", JUDGE_SAMPLE_PCT="100"), judge=judge)
    pulled = client.list_run_calls()[0]
    assert pulled["start_time"] == CLOCK - timedelta(hours=24)
    text = (tmp_path / FILE).read_text(encoding="utf-8")
    assert "- SINCE: 24h" in text and "- LIMIT: 5" in text and "LIMIT was hit: 5 runs were read" in text
    assert "5 runs read" in text or "from 5 runs read in the window" in text


def test_gp2_the_job_takes_no_arguments_and_the_shell_never_parses_the_settings(tmp_path, capsys):
    client, _ = world()
    with pytest.raises(SystemExit) as raised:
        job.main(["SINCE=24h"], environ=job_env(), client=client, clock=lambda: CLOCK, results_dir=tmp_path)
    assert raised.value.code == 2 and "takes no arguments" in capsys.readouterr().err and client.calls == []


def test_gp2_without_langsmith_api_key_it_exits_2_naming_it_and_builds_no_client(tmp_path, capsys):
    built = []
    with pytest.raises(SystemExit) as raised:
        job.make_client({}, factory=lambda **kw: built.append(kw))
    assert raised.value.code == 2 and built == []
    assert "LANGSMITH_API_KEY is not set" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        job.main([], environ=job_env(**{job.API_KEY_VARIABLE: "  "}), build=lambda _env: FakeJudge(), results_dir=tmp_path)  # blank is not set; no client is passed, so the real one would be built
    assert results(tmp_path) == []


def test_gp2_with_the_key_set_the_client_is_built_from_it_and_only_it():
    built = []
    job.make_client({"LANGSMITH_API_KEY": " ls-fake "}, factory=lambda **kw: built.append(kw) or "client")
    assert built == [{"api_key": "ls-fake"}]


# --- the privacy of the file ------------------------------------------------------------------------------------------------------------------


def test_the_file_holds_no_run_ids_of_passing_runs_and_no_timestamps_of_the_job(tmp_path):
    client, trees = world()
    run_job(client, tmp_path)
    text = (tmp_path / FILE).read_text(encoding="utf-8")
    passing = [tree.root for tree in trees if str(tree.root.id) not in text]
    assert len(passing) >= 4
    assert not re.search(r"\d{2}:\d{2}:\d{2}", text) and "2026-10-31T" not in text


def test_the_seed_runs_are_what_the_job_scored(tmp_path):
    client, _ = world()
    run_job(client, tmp_path)
    scored_ids = set(posted_keys(client))
    assert {tree.root.id for tree in seed_trees()} <= scored_ids
    assert judge_client([]).calls == []


def test_privacy_the_file_holds_no_answer_text_question_text_or_phone_number(tmp_path):
    """A fiction-range number is seeded into a failing run's answer, and a marker phrase into its question and answer; the job reads both, and neither reaches the file."""
    from phone_leaks import leak_list, leaked

    marker = "zebra-marker-phrase"
    answer = f"[path: assess]\nNO-GO. {marker}. Ring the field owner on {FICTION_NUMBER}. Source: CAP 1535."
    leaky = judge_tree("leaky", path="assess", source="pilot", answer=answer, question=f"Can I fly at {marker}, my number is {FICTION_NUMBER}?")
    client, _ = world([leaky])
    code, judge = run_job(client, tmp_path, judge=FakeJudge({"prioritised": 0}))
    text = (tmp_path / FILE).read_text(encoding="utf-8")
    assert code == 0 and str(leaky.root.id) in text and "online.judge.prioritised" in text  # the run is there, as a failing run
    assert any(marker in str(call) for call in judge.calls)  # and the judge was shown the text
    assert marker not in text and FICTION_NUMBER not in text and FICTION_NUMBER.replace(" ", "") not in text and "NO-GO" not in text
    assert leaked(text, set(leak_list())) == set()
