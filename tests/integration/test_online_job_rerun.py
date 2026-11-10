"""The whole score-and-post pass over three fake runs on the recording client (7 PR-2.F1: GP3, GP5, E5, and the invariant).

`score_and_post` is the real function and the client is 5a's recording client with the online methods: nothing is mocked in the job. The three runs are a good assess run (guard record, an `assess` run and
a `search_guidance` run), an explain run with no `assess` child, and a decline run that called no tool. The tests print what the evidence files hold (`-s` shows it).
"""

import random

import pytest
from online_helpers import FakeTree, OnlineRecordingClient, fake_run, good_answer, guidance_run, scored_tree

from evals.online import job

PROJECT = "flysafelz"
VERSION, NEXT_VERSION = "0123456789ab", "ba9876543210"
GP3 = [
    "online.guard_pass",
    "online.decision_first",
    "online.cited_guidance",
    "online.recheck_stated",
    "online.assumed_window_stated",
    "online.no_verdict_beyond_day3",
    "online.step_cap",
    "online.decline_no_tools",
    "online.evaluator_version",
]


def three_runs() -> list[FakeTree]:
    assess = scored_tree("run-1")
    explain_root = fake_run("run-2", path="explain", guard="passed", answer="Cloud base is the height of the lowest cloud. See CAP 1535.")
    explain = FakeTree(explain_root, [guidance_run(explain_root)])
    decline = FakeTree(fake_run("run-3", path="decline", guard="passed", answer="I can only advise on flying conditions at a launch zone."), [])
    return [assess, explain, decline]


def client_for(trees: list[FakeTree], **kwargs) -> OnlineRecordingClient:
    return OnlineRecordingClient([tree.root for tree in trees], children=[child for tree in trees for child in tree.children], **kwargs)


def by_run(client: OnlineRecordingClient) -> dict:
    found: dict = {}
    for posted in client.feedback:
        found.setdefault(posted.run_id, []).append(posted)
    return found


def show(label: str, lines: list[str]) -> None:
    print(f"\n--- {label}")
    print("\n".join(lines))


# --- GP3 ------------------------------------------------------------------------------------------------------------------------------------


def test_gp3_a_scored_run_ends_up_with_the_eight_keys_then_the_version_last():
    trees = three_runs()
    client = client_for(trees)
    result = job.score_and_post(client, PROJECT, [tree.root for tree in trees], version=VERSION)
    assert [item.status for item in result.runs] == [job.SCORED] * 3
    posted = by_run(client)
    for tree in trees:
        assert [item.key for item in posted[tree.root.id]] == GP3
        assert all(item.score in (0, 1, None) and item.comment for item in posted[tree.root.id])
        assert posted[tree.root.id][-1].comment == f"all keys posted | evaluator_version={VERSION}"
    first = posted[trees[0].root.id]
    show(
        f"run {trees[0].root.id} (assess), evaluator version {VERSION}",
        [f"{item.key:34} score={item.score!s:5} {item.comment}" for item in first],
    )
    assert [item.score for item in first] == [1, 1, 1, 1, 1, 1, 1, None, 1]
    assert (posted[trees[1].root.id][1].score, posted[trees[1].root.id][1].comment.split(" | ")[0]) == (None, "no assess")  # decision_first on the explain run
    assert [item.score for item in posted[trees[2].root.id]][-2:] == [1, 1]  # decline_no_tools passes, then the marker
    assert client.duplicate_posts == []


# --- GP5 ------------------------------------------------------------------------------------------------------------------------------------


def test_gp5_a_second_pass_on_the_same_client_makes_no_create_feedback_call_and_reports_every_run_already_scored():
    trees = three_runs()
    client = client_for(trees)
    runs = [tree.root for tree in trees]
    first = job.score_and_post(client, PROJECT, runs, version=VERSION)
    writes = len(client.create_calls())
    second = job.score_and_post(client, PROJECT, runs, version=VERSION)
    assert len(client.create_calls()) == writes == 27 and second.posted == 0
    assert [item.status for item in second.runs] == [job.ALREADY_SCORED] * 3
    assert [item.record() for item in second.runs] == [item.record() for item in first.runs]
    lines = job.report_lines(second)
    show("second pass", lines)
    assert [line for line in lines if line.endswith("already scored")] == [f"{tree.root.id} already scored" for tree in trees]
    assert lines[-1] == "0 scored, 3 already scored, 0 errored; 0 feedback posted"


def test_gp5_a_rerun_after_a_version_change_posts_everything_again_once():
    trees = three_runs()
    client = client_for(trees)
    runs = [tree.root for tree in trees]
    job.score_and_post(client, PROJECT, runs, version=VERSION)
    changed = job.score_and_post(client, PROJECT, runs, version=NEXT_VERSION)
    assert changed.posted == 27 and client.duplicate_posts == []
    assert len(client.feedback) == 54


# --- E5 -------------------------------------------------------------------------------------------------------------------------------------


def test_e5_a_post_that_raises_is_reported_the_job_moves_on_and_a_rerun_fills_only_the_gap():
    trees = three_runs()
    client = client_for(trees, fail_feedback={(trees[1].root.id, "online.step_cap")})
    runs = [tree.root for tree in trees]
    first = job.score_and_post(client, PROJECT, runs, version=VERSION)
    failure = first.runs[1]
    assert failure.status == job.ERRORED and failure.error is not None
    assert (failure.error.stage, failure.error.key, failure.error.error_class) == (job.POST, "online.step_cap", "RuntimeError")
    assert failure.posted == GP3[:6]  # run B's keys before the failing one
    assert "online.evaluator_version" not in [item.key for item in by_run(client)[trees[1].root.id]]
    assert [first.runs[0].status, first.runs[2].status] == [job.SCORED, job.SCORED]  # run C was still scored
    assert [item.key for item in by_run(client)[trees[2].root.id]] == GP3
    report = job.report_lines(first)
    show("failure report", report)
    assert any(f"{trees[1].root.id} errored at post online.step_cap: RuntimeError" in line for line in report)

    client.fail_feedback.clear()
    calls_before = len(client.create_calls())
    second = job.score_and_post(client, PROJECT, runs, version=VERSION)
    filled = client.create_calls()[calls_before:]
    show("fill-in rerun posts", [f"{call['run_id']} {call['key']}" for call in filled])
    assert [(call["run_id"], call["key"]) for call in filled] == [(trees[1].root.id, key) for key in GP3[6:]]
    assert [item.status for item in second.runs] == [job.ALREADY_SCORED, job.SCORED, job.ALREADY_SCORED]
    assert client.duplicate_posts == [] and [item.key for item in by_run(client)[trees[1].root.id]] == GP3
    third = job.score_and_post(client, PROJECT, runs, version=VERSION)
    assert third.posted == 0


def test_e5_a_failed_version_marker_is_the_only_thing_a_rerun_posts():
    trees = three_runs()
    client = client_for(trees, fail_feedback={(trees[0].root.id, job.VERSION_KEY)})
    runs = [tree.root for tree in trees]
    first = job.score_and_post(client, PROJECT, runs, version=VERSION)
    assert first.runs[0].status == job.ERRORED and first.runs[0].posted == GP3[:-1]
    client.fail_feedback.clear()
    second = job.score_and_post(client, PROJECT, runs, version=VERSION)
    assert second.runs[0].posted == [job.VERSION_KEY] and client.duplicate_posts == []


def test_e5_when_list_feedback_fails_nothing_is_posted_for_that_run_and_the_job_continues():
    trees = three_runs()
    client = client_for(trees, fail_list_feedback={trees[1].root.id})
    runs = [tree.root for tree in trees]
    first = job.score_and_post(client, PROJECT, runs, version=VERSION)
    failed = first.runs[1]
    assert failed.status == job.ERRORED and failed.error is not None and failed.error.stage == job.LIST_FEEDBACK and failed.error.error_class == "RuntimeError"
    assert failed.posted == [] and trees[1].root.id not in by_run(client)
    assert all(call["run_id"] != trees[1].root.id for call in client.create_calls())
    assert [first.runs[0].status, first.runs[2].status] == [job.SCORED, job.SCORED]
    client.fail_list_feedback.clear()
    second = job.score_and_post(client, PROJECT, runs, version=VERSION)
    assert second.runs[1].posted == GP3 and client.duplicate_posts == []


def test_e5_a_trace_that_cannot_be_read_is_reported_for_that_run_and_the_job_continues():
    trees = three_runs()
    client = client_for(trees)
    original = client.list_runs

    def failing(**kw):
        if str(kw.get("trace_id")) == str(trees[0].root.trace_id):
            raise ConnectionError("simulated")
        return original(**kw)

    client.list_runs = failing  # type: ignore[method-assign]
    result = job.score_and_post(client, PROJECT, [tree.root for tree in trees], version=VERSION)
    assert result.runs[0].status == job.ERRORED and result.runs[0].error is not None and result.runs[0].error.stage == job.READ_TRACE
    assert result.runs[0].error.error_class == "ConnectionError" and result.runs[0].posted == []
    assert [result.runs[1].status, result.runs[2].status] == [job.SCORED, job.SCORED]


def test_a_check_that_scores_outside_1_0_or_none_is_a_score_error_not_a_read_trace_error(monkeypatch):
    trees = three_runs()
    client = client_for(trees)
    monkeypatch.setitem(job.ONLINE_CHECKS, job.STEP_CAP, lambda example, answer: {"score": 0.5, "comment": "half"})
    result = job.score_and_post(client, PROJECT, [tree.root for tree in trees], version=VERSION)
    for item in result.runs:
        assert item.status == job.ERRORED and item.error is not None and item.error.stage == job.SCORE
        assert item.error.error_class == "ValueError" and item.posted == []
    assert client.create_calls() == []  # no run got feedback
    assert "errored at score: ValueError" in "\n".join(job.report_lines(result))


def test_e2_a_run_with_no_path_is_flagged_path_unknown_in_its_record():
    tree = scored_tree("unk", path="unknown", answer=good_answer())
    client = client_for([tree])
    result = job.score_and_post(client, PROJECT, [tree.root], version=VERSION)
    assert result.runs[0].path_unknown is True and result.runs[0].record()["path_unknown"] is True
    assert result.runs[0].scores["online.decline_no_tools"].comment == "path unknown"
    assert [item.status for item in result.runs] == [job.SCORED] and result.runs[0].scores["online.guard_pass"].score == 1


# --- the invariant --------------------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("seed", range(25))
def test_invariant_no_run_key_version_triple_is_posted_twice_across_reruns_and_injected_failures(seed):
    rng = random.Random(seed)
    trees = three_runs()[: rng.randint(1, 3)]
    client = client_for(trees)
    runs = [tree.root for tree in trees]
    for tree in trees:  # some runs start with feedback from an older version
        if rng.random() < 0.5:
            for key in rng.sample(GP3, rng.randint(1, len(GP3))):
                client.seed_feedback(tree.root, key, version=NEXT_VERSION if rng.random() < 0.5 else "cccccccccccc")
    for _ in range(rng.randint(2, 6)):
        version = rng.choice([VERSION, NEXT_VERSION])
        client.fail_feedback = {(tree.root.id, key) for tree in trees for key in GP3 if rng.random() < 0.15}
        client.fail_list_feedback = {tree.root.id for tree in trees if rng.random() < 0.1}
        job.score_and_post(client, PROJECT, runs, version=version)
        triples = client.stored_triples()
        assert len(triples) == len(set(triples)), f"a triple was stored twice (seed {seed})"
        assert client.duplicate_posts == []
    final = rng.choice([VERSION, NEXT_VERSION])
    client.fail_feedback, client.fail_list_feedback = set(), set()
    job.score_and_post(client, PROJECT, runs, version=final)
    assert client.duplicate_posts == [] and len(client.stored_triples()) == len(set(client.stored_triples()))
    for tree in trees:  # after a clean pass every run has every key at the last version, exactly once
        at_final = [item.key for item in client.feedback if item.run_id == tree.root.id and (item.extra or {}).get("evaluator_version") == final]
        assert sorted(at_final) == sorted(GP3)
    assert job.score_and_post(client, PROJECT, runs, version=final).posted == 0
