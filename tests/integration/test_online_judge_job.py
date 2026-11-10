"""The PR-2 job end to end with the sampled judge, on 5a's recording client, fake run trees and a fake judge (7 PR-3.F1: GP2, GP3, GP5, E4, E8, E10 and the added cases).

`pull_runs` and `score_and_post` are the real functions; nothing is mocked in the job. The judge is `FakeJudge` (a fixed grade per criterion, a call count) and never a model or the bridge.
"""

import uuid
from datetime import UTC, datetime

import pytest
from online_helpers import OnlineRecordingClient, fake_run, run_id
from online_judge_helpers import OTHER_VERSION, VERSION, FakeJudge, by_run, generate_trees, judge_client, judge_tree

from evals.agent.judge import CRITERIA_BY_PATH
from evals.online import job, pull

PROJECT = "flysafelz"
NOW = datetime(2026, 10, 31, 7, 30, tzinfo=UTC)
NO_DOTENV = {"READER_DOTENV_PATH": "/no-such-dotenv/.env"}
SEED_PATHS = ["assess"] * 4 + ["explain"] * 3 + ["clarify"] * 2 + ["decline"] * 2  # 11 seed questions: 4 assess, 3 explain, 2 clarify, 2 decline
LZ_NAMES = ["Blue Mill", "West Sleekburn", "Druridge Bay", "Buttermere", "Crookham"]
DETERMINISTIC = list(job.DETERMINISTIC_KEYS)


def seed_trees():
    return [judge_tree(f"seed-{index}", path=path, source="online-seed", lz_name=LZ_NAMES[index % 5]) for index, path in enumerate(SEED_PATHS)]


def pulled(client):
    return pull.pull_runs(client, PROJECT, pull.parse_since("7d"), 500, NOW)


def run_pass(client, judge, *, pct=20, version=VERSION, **kw):
    """The pull and the score-and-post pass of `make online-eval`, as PR-4.F1 will call them."""
    return job.score_and_post(client, PROJECT, pulled(client).runs, version=version, judge=judge, sample_pct=pct, **kw)


def keys_posted(client, run, since=0):
    return [item.key for item in client.feedback[since:] if item.run_id == run.id]


def sampled_ids(trees, pct):
    return {tree.root.id for tree in trees if job.run_bucket(tree.root.id) < pct}


def expected_keys(path, *, deterministic=True, marker=True):
    return (DETERMINISTIC if deterministic else []) + list(job.judge_keys_for(path)) + ([job.VERSION_KEY] if marker else [])


# --- GP2 and GP3 -----------------------------------------------------------------------------------------------------------------------------


def test_gp2_with_14_eligible_runs_every_one_is_judged_and_no_not_sampled_record_is_posted():
    trees = [*seed_trees(), *generate_trees(3, prefix="pilot")]
    client, fake = judge_client(trees), FakeJudge()
    result = run_pass(client, fake)
    assert result.eligible == 14 and result.judge_calls == len(fake.calls) == 14
    assert sorted(fake.judged_ids()) == sorted(str(tree.root.id) for tree in trees)
    assert not [item for item in client.feedback if item.key == job.SAMPLED_KEY]
    assert [item.judge for item in result.runs] == [job.JUDGED] * 14 and result.with_judge(job.NOT_SAMPLED) == []


def test_gp3_each_seed_run_carries_the_judge_keys_of_exactly_its_path_and_the_version_last():
    trees = seed_trees()
    client = judge_client(trees)
    run_pass(client, FakeJudge())
    posted = by_run(client)
    for tree, path in zip(trees, SEED_PATHS, strict=True):
        keys = [item.key for item in posted[tree.root.id]]
        assert keys == expected_keys(path)
        others = {f"online.judge.{name}" for other, names in CRITERIA_BY_PATH.items() if other != path for name in names} - set(job.judge_keys_for(path))
        assert not others & set(keys)
        for item in posted[tree.root.id]:
            assert item.score in (1, 0, None) and item.comment
            if item.key.startswith("online.judge."):
                criterion = item.key.removeprefix("online.judge.")
                assert item.score == 1 and item.comment == f"fake reason for {criterion} | evaluator_version={VERSION}"
    assert client.duplicate_posts == []


def test_every_post_carries_the_project_s_session_id_and_the_version_marker_is_the_last_post_of_each_run():
    trees = seed_trees()
    client = judge_client(trees)
    run_pass(client, FakeJudge())
    assert {call["session_id"] for call in client.create_calls()} == {run_id(f"project/{PROJECT}")}
    for tree in trees:
        assert keys_posted(client, tree.root)[-1] == job.VERSION_KEY
        assert [key for key in keys_posted(client, tree.root) if key == job.VERSION_KEY] == [job.VERSION_KEY]


def test_the_judge_is_given_each_run_s_own_path_and_never_the_intended_path():
    trees = [judge_tree(f"own-{index}", path=path, metadata={"intended_path": "assess"}) for index, path in enumerate(["clarify", "decline", "explain"])]
    client, fake = judge_client(trees), FakeJudge()
    run_pass(client, fake)
    assert [example["reference"]["path"] for example, _answer in fake.calls] == ["clarify", "decline", "explain"]
    assert [answer.path for _example, answer in fake.calls] == ["clarify", "decline", "explain"]
    assert not [example for example, _answer in fake.calls if "intended_path" in str(example)]


def test_a_failed_judge_score_is_posted_as_a_zero_and_a_null_grade_as_a_null_with_the_reason():
    tree = judge_tree("grades", path="assess")
    client = judge_client([tree])
    run_pass(client, FakeJudge({"answers_the_question": 0, "prioritised": None}))
    posted = {item.key: item for item in client.feedback}
    assert posted["online.judge.answers_the_question"].score == 0 and posted["online.judge.prioritised"].score is None
    assert posted["online.judge.prioritised"].comment == "fake reason for prioritised | evaluator_version=" + VERSION


# --- GP5 -------------------------------------------------------------------------------------------------------------------------------------


def test_gp5_a_second_pass_with_the_same_environment_makes_zero_judge_calls_and_zero_posts():
    trees = [*seed_trees(), *generate_trees(3, prefix="pilot")]
    client, fake = judge_client(trees), FakeJudge()
    first = run_pass(client, fake)
    calls, stored, writes = len(fake.calls), len(client.feedback), len(client.create_calls())
    second = run_pass(client, fake)
    assert len(fake.calls) == calls == 14 and len(client.feedback) == stored and len(client.create_calls()) == writes
    assert second.posted == 0 and second.judge_calls == 0 and second.eligible == 0
    assert [item.status for item in second.runs] == [job.ALREADY_SCORED] * 14
    assert [item.record() for item in second.runs] == [item.record() for item in first.runs]
    assert client.duplicate_posts == []


# --- the pull's part -------------------------------------------------------------------------------------------------------------------------


def test_only_the_runs_the_pull_lets_through_are_judged():
    pilot, seed = judge_tree("pilot", source="pilot"), judge_tree("seed", source="online-seed")
    skipped = [judge_tree("case", source="case"), judge_tree("judge-call", source="judge"), judge_tree("errored", error="boom"), judge_tree("experiment", reference_example_id=uuid.uuid4())]
    client, fake = judge_client([pilot, seed, *skipped]), FakeJudge()
    result = run_pass(client, fake)
    assert sorted(fake.judged_ids()) == sorted([str(pilot.root.id), str(seed.root.id)]) and len(result.runs) == 2
    assert not [item for item in client.feedback if item.run_id in {tree.root.id for tree in skipped}]


def test_f2_the_judge_pass_lists_runs_of_the_one_project_and_only_that_project():
    trees = generate_trees(4)
    client = judge_client(trees)
    run_pass(client, FakeJudge())
    assert {call["project_name"] for call in client.list_run_calls()} == {PROJECT}
    assert {name for name in client.methods_called()} == {"list_runs", "read_project", "list_feedback", "create_feedback"}


# --- E4: the judge variables unset -----------------------------------------------------------------------------------------------------------


def test_e4_with_the_judge_variables_unset_the_deterministic_keys_post_nothing_of_the_judge_does_and_the_names_are_returned():
    trees = [*seed_trees(), *generate_trees(2, prefix="pilot")]
    client = judge_client(trees)
    setup = job.judge_setup_from_env({**NO_DOTENV, "JUDGE_MODEL": "m"}, build=lambda environ: pytest.fail("the judge must not be built"))
    assert setup.judge is None and setup.missing == ("JUDGE_PROVIDER", "JUDGE_BASE_URL", "JUDGE_API_" + "KEY")
    result = run_pass(client, setup.judge, judge_missing=setup.missing, pct=setup.pct)
    assert result.judge_missing == setup.missing and result.judge_calls == 0 and result.judge_enabled is False
    for tree in trees:
        assert keys_posted(client, tree.root) == DETERMINISTIC + [job.VERSION_KEY]
    assert not [item for item in client.feedback if item.key.startswith("online.judge.")]
    assert [line for line in job.report_lines(result) if line.startswith("judge not run")] == [
        "judge not run: JUDGE_PROVIDER, JUDGE_BASE_URL, JUDGE_API_" + "KEY not set; the judge keys stay missing for a later pass"
    ]


def test_e4_a_second_pass_with_the_variables_set_posts_only_the_judge_keys_and_no_deterministic_post():
    trees = [*seed_trees(), *generate_trees(2, prefix="pilot")]
    client = judge_client(trees)
    run_pass(client, None, judge_missing=("JUDGE_MODEL",))
    before, writes = len(client.feedback), len(client.create_calls())
    fake = FakeJudge()
    second = run_pass(client, fake)
    assert len(fake.calls) == 13 and second.posted == len(client.feedback) - before
    for tree, path in zip(trees, [*SEED_PATHS, "assess", "assess"], strict=True):
        assert keys_posted(client, tree.root, before) == list(job.judge_keys_for(path))  # no deterministic key, and no version marker: it is already there
    assert {call["key"] for call in client.create_calls()[writes:]} <= {key for names in CRITERIA_BY_PATH.values() for key in map(job.judge_key, names)}
    assert client.duplicate_posts == []
    assert run_pass(client, fake).posted == 0 and len(fake.calls) == 13


def test_e4_a_run_the_judge_never_saw_is_judged_by_a_later_pass_even_when_it_would_have_been_left_out_of_a_sample():
    trees = generate_trees(5)
    client = judge_client(trees)
    run_pass(client, None)
    fake = FakeJudge()
    run_pass(client, fake)
    assert len(fake.calls) == 5 and not [item for item in client.feedback if item.key == job.SAMPLED_KEY]


# --- E8: more than 50 eligible runs ----------------------------------------------------------------------------------------------------------


def test_e8_with_120_eligible_runs_at_20_percent_the_judged_set_is_exactly_the_hash_rules_set_and_the_rest_are_recorded_not_sampled():
    trees = generate_trees(120, prefix="many")
    client, fake = judge_client(trees), FakeJudge()
    result = run_pass(client, fake)
    expected = sampled_ids(trees, 20)
    assert 0 < len(expected) < 120
    assert {uuid.UUID(example["id"]) for example, _answer in fake.calls} == expected and len(fake.calls) == len(expected) == result.judge_calls
    left_out = {tree.root.id for tree in trees} - expected
    sampled = [item for item in client.feedback if item.key == job.SAMPLED_KEY]
    assert {item.run_id for item in sampled} == left_out and len(sampled) == len(left_out)
    for item in sampled:
        assert item.score is None and item.value == job.NOT_SAMPLED
        assert item.comment == f"left out of the judge sample: 120 eligible runs, more than 50, chosen by run-id hash at sample_pct=20 | evaluator_version={VERSION}"
        assert item.extra == {"evaluator_version": VERSION, "judge_sample_pct": 20}
    for tree in trees:
        keys = keys_posted(client, tree.root)
        assert keys == (expected_keys("assess") if tree.root.id in expected else DETERMINISTIC + [job.SAMPLED_KEY, job.VERSION_KEY])
    assert (len(result.with_judge(job.JUDGED)), len(result.with_judge(job.NOT_SAMPLED))) == (len(expected), len(left_out))
    assert client.duplicate_posts == [] and {call["session_id"] for call in client.create_calls()} == {run_id(f"project/{PROJECT}")}


def test_e8_a_rerun_picks_the_same_runs_makes_no_judge_call_and_posts_nothing_new():
    trees = generate_trees(120, prefix="many")
    client, fake = judge_client(trees), FakeJudge()
    first = run_pass(client, fake)
    calls, stored = len(fake.calls), len(client.feedback)
    second = run_pass(client, fake)
    assert len(fake.calls) == calls and len(client.feedback) == stored and second.posted == 0 and second.eligible == 0
    assert [item.status for item in second.runs] == [job.ALREADY_SCORED] * 120
    assert [item.record() for item in second.runs] == [item.record() for item in first.runs]
    third_client = judge_client(trees)  # a fresh store, the same run ids: the same runs are picked
    third_fake = FakeJudge()
    run_pass(third_client, third_fake)
    assert sorted(third_fake.judged_ids()) == sorted(fake.judged_ids())


def test_e8_a_changed_percentage_reconsiders_the_runs_left_out_and_judges_only_those_the_new_percentage_picks():
    trees = generate_trees(120, prefix="many")
    client, fake = judge_client(trees), FakeJudge()
    run_pass(client, fake)
    first_calls, stored = len(fake.calls), len(client.feedback)
    result = run_pass(client, fake, pct=30)
    newly = {tree.root.id for tree in trees if 20 <= job.run_bucket(tree.root.id) < 30}
    assert newly and {uuid.UUID(example["id"]) for example, _answer in fake.calls[first_calls:]} == newly
    assert result.eligible == 120 - len(sampled_ids(trees, 20))
    added = client.feedback[stored:]
    assert {item.run_id for item in added} == newly and not [item for item in added if item.key in (job.SAMPLED_KEY, job.VERSION_KEY) or item.key in DETERMINISTIC]
    assert [item.key for item in added if item.run_id == next(iter(newly))] == list(job.judge_keys_for("assess"))
    # the runs left out again at 30% have a record at 20% and get no new one
    assert len([item for item in client.feedback if item.key == job.SAMPLED_KEY]) == 120 - len(sampled_ids(trees, 20))
    again = run_pass(client, fake, pct=30)
    assert again.posted == 0 and len(fake.calls) == first_calls + len(newly)
    assert client.duplicate_posts == []


def test_a_percentage_of_zero_with_more_than_50_eligible_runs_judges_none_and_records_every_run_not_sampled():
    trees = generate_trees(60, prefix="zero")
    client, fake = judge_client(trees), FakeJudge()
    result = run_pass(client, fake, pct=0)
    assert fake.calls == [] and result.judge_calls == 0
    assert {item.run_id for item in client.feedback if item.key == job.SAMPLED_KEY} == {tree.root.id for tree in trees}
    assert all("sample_pct=0 |" in str(item.comment) for item in client.feedback if item.key == job.SAMPLED_KEY)
    assert not [item for item in client.feedback if item.key.startswith("online.judge.") and item.key != job.SAMPLED_KEY]


def test_a_percentage_of_100_judges_every_run_even_above_50():
    trees = generate_trees(60, prefix="all")
    client, fake = judge_client(trees), FakeJudge()
    run_pass(client, fake, pct=100)
    assert len(fake.calls) == 60 and not [item for item in client.feedback if item.key == job.SAMPLED_KEY]


# --- E10: the boundary -----------------------------------------------------------------------------------------------------------------------


def test_e10_with_exactly_50_eligible_runs_all_50_are_judged_and_none_is_not_sampled():
    trees = generate_trees(50, prefix="edge")
    client, fake = judge_client(trees), FakeJudge()
    result = run_pass(client, fake)
    assert len(fake.calls) == 50 and result.eligible == 50 and not [item for item in client.feedback if item.key == job.SAMPLED_KEY]


def test_e10_with_51_the_sample_applies():
    trees = generate_trees(51, prefix="edge")
    client, fake = judge_client(trees), FakeJudge()
    result = run_pass(client, fake)
    expected = sampled_ids(trees, 20)
    assert result.eligible == 51 and {uuid.UUID(example["id"]) for example, _answer in fake.calls} == expected and len(expected) < 51
    assert {item.run_id for item in client.feedback if item.key == job.SAMPLED_KEY} == {tree.root.id for tree in trees} - expected


# --- what counts towards the 50 --------------------------------------------------------------------------------------------------------------


def test_the_eligible_count_leaves_out_unknown_path_runs():
    trees = [*generate_trees(40, prefix="known"), *generate_trees(20, prefix="unknown", path="unknown")]
    client, fake = judge_client(trees), FakeJudge()
    result = run_pass(client, fake)
    assert result.eligible == 40 and len(fake.calls) == 40 and not [item for item in client.feedback if item.key == job.SAMPLED_KEY]
    for tree in trees[40:]:  # 5a's table has no criteria for unknown: no judge key, no judge call, the deterministic keys and the marker
        assert keys_posted(client, tree.root) == DETERMINISTIC + [job.VERSION_KEY]
        assert str(tree.root.id) not in fake.judged_ids()


def test_the_eligible_count_leaves_out_runs_already_fully_judged():
    done = [judge_tree(f"done-{index}", feedback=[(key, VERSION) for key in job.judge_keys_for("assess")]) for index in range(15)]
    trees = [*done, *generate_trees(45, prefix="todo")]
    client, fake = judge_client(trees), FakeJudge()
    result = run_pass(client, fake)
    assert result.eligible == 45 and len(fake.calls) == 45 and not [item for item in client.feedback if item.key == job.SAMPLED_KEY]
    assert not set(fake.judged_ids()) & {str(tree.root.id) for tree in done}


def test_a_partly_judged_run_counts_towards_the_50():
    partly = [judge_tree(f"partly-{index}", feedback=[(job.judge_keys_for("assess")[0], VERSION)]) for index in range(6)]
    trees = [*partly, *generate_trees(45, prefix="todo")]
    client, fake = judge_client(trees), FakeJudge()
    result = run_pass(client, fake)
    assert result.eligible == 51 and result.with_judge(job.NOT_SAMPLED) and len(fake.calls) < 51
    assert {uuid.UUID(example["id"]) for example, _answer in fake.calls} == sampled_ids(trees, 20)


def test_when_a_sibling_key_exists_at_this_version_only_the_missing_criteria_are_posted_and_the_judge_is_called_once():
    first_key = job.judge_keys_for("assess")[0]
    tree = judge_tree("sibling", feedback=[(first_key, VERSION)])
    client, fake = judge_client([tree]), FakeJudge()
    stored = len(client.feedback)
    result = run_pass(client, fake)
    assert len(fake.calls) == 1
    posted = keys_posted(client, tree.root, stored)
    assert first_key not in posted and posted == DETERMINISTIC + list(job.judge_keys_for("assess")[1:]) + [job.VERSION_KEY]
    assert result.runs[0].judge_scores.keys() == set(job.judge_keys_for("assess")) and client.duplicate_posts == []


def test_when_a_judge_key_exists_only_at_an_older_version_the_run_is_eligible_again_and_the_old_feedback_is_not_touched():
    old = [(key, OTHER_VERSION) for key in job.judge_keys_for("assess")]
    tree = judge_tree("older", feedback=old)
    client, fake = judge_client([tree]), FakeJudge()
    before = list(client.feedback)
    result = run_pass(client, fake)
    assert len(fake.calls) == 1 and result.eligible == 1
    assert client.feedback[: len(before)] == before and all(item is kept for item, kept in zip(client.feedback, before, strict=False))
    posted = keys_posted(client, tree.root, len(before))
    assert posted == expected_keys("assess")
    assert len([item for item in client.feedback if item.key == job.judge_keys_for("assess")[0]]) == 2  # the old one and the new one: a key is unique per version


# --- the judge raises ------------------------------------------------------------------------------------------------------------------------


def test_when_the_judge_raises_on_one_run_its_judge_keys_stay_missing_the_error_is_reported_and_the_others_are_judged():
    trees = generate_trees(5, prefix="raise")
    bad = trees[2]
    client, fake = judge_client(trees), FakeJudge(raises={str(bad.root.id)})
    result = run_pass(client, fake)
    assert len(fake.calls) == 5 and result.judge_calls == 5
    failed = result.runs[2]
    assert failed.status == job.ERRORED and failed.judge_error is not None
    assert (failed.judge_error.stage, failed.judge_error.error_class, failed.judge_error.message) == ("judge", "ConnectionError", "bridge returned 503")
    assert keys_posted(client, bad.root) == DETERMINISTIC  # its deterministic keys posted; no judge key, no marker
    for tree in trees:
        if tree is not bad:
            assert keys_posted(client, tree.root) == expected_keys("assess")
    assert [item.status for item in result.runs].count(job.SCORED) == 4 and len(result.errors) == 1
    report = job.report_lines(result)
    assert [line for line in report if str(bad.root.id) in line] == [f"{bad.root.id} errored at judge: ConnectionError: bridge returned 503 (judge keys not posted; posted 8 other)"]
    assert report[-1] == f"4 scored, 0 already scored, 1 errored; {4 * len(expected_keys('assess')) + len(DETERMINISTIC)} feedback posted"


def test_a_rerun_judges_only_the_run_whose_judge_raised_and_posts_only_its_judge_keys_and_the_marker():
    trees = generate_trees(5, prefix="raise")
    bad = trees[2]
    client = judge_client(trees)
    run_pass(client, FakeJudge(raises={str(bad.root.id)}))
    stored, fake = len(client.feedback), FakeJudge()
    second = run_pass(client, fake)
    assert fake.judged_ids() == [str(bad.root.id)] and second.eligible == 1
    assert [item.key for item in client.feedback[stored:]] == [*job.judge_keys_for("assess"), job.VERSION_KEY]
    assert [item.status for item in second.runs] == [job.ALREADY_SCORED, job.ALREADY_SCORED, job.SCORED, job.ALREADY_SCORED, job.ALREADY_SCORED]
    assert run_pass(client, fake).posted == 0 and client.duplicate_posts == []


def test_a_criterion_the_judge_gives_no_result_for_is_posted_null_with_the_reason_and_the_run_is_scored():
    tree = judge_tree("partial")
    client = judge_client([tree])
    result = run_pass(client, FakeJudge(drop={"actionable"}))
    posted = {item.key: item for item in client.feedback}
    assert posted["online.judge.actionable"].score is None and posted["online.judge.actionable"].comment == f"no result from the judge | evaluator_version={VERSION}"
    assert posted["online.judge.answers_the_question"].score == 1 and result.runs[0].status == job.SCORED and result.runs[0].judge_error is None


def test_a_post_that_fails_after_the_judge_ran_is_reported_and_the_rerun_posts_only_the_gap_without_judging_again():
    tree = judge_tree("post-fails")
    failing = job.judge_keys_for("assess")[1]
    client, fake = judge_client([tree], fail_feedback={(tree.root.id, failing)}), FakeJudge()
    first = run_pass(client, fake)
    assert first.runs[0].status == job.ERRORED and first.runs[0].error is not None and first.runs[0].error.key == failing
    client.fail_feedback.clear()
    stored = len(client.feedback)
    second = run_pass(client, fake)
    assert len(fake.calls) == 2  # the keys the judge made were not stored, so the rerun asks again for the missing ones
    assert [item.key for item in client.feedback[stored:]] == [*job.judge_keys_for("assess")[1:], job.VERSION_KEY] and second.runs[0].status == job.SCORED
    assert client.duplicate_posts == []


# --- the percentage is checked before any call -----------------------------------------------------------------------------------------------


@pytest.mark.parametrize("pct", [-1, 101, "20"])
def test_a_bad_percentage_given_to_the_job_is_refused_before_any_langsmith_call(pct):
    client = judge_client(generate_trees(2))
    with pytest.raises(ValueError, match="percentage"):
        job.score_and_post(client, PROJECT, [], version=VERSION, judge=FakeJudge(), sample_pct=pct)
    assert client.calls == []


@pytest.mark.parametrize("text", ["-1", "101", "abc", ""])
def test_a_bad_percentage_in_the_environment_exits_non_zero_naming_the_variable_before_any_langsmith_call(text, capsys):
    client = judge_client(generate_trees(2))
    with pytest.raises(SystemExit) as stopped:
        job.judge_setup_from_env({**NO_DOTENV, "JUDGE_SAMPLE_PCT": text})
    assert stopped.value.code == 2 and "JUDGE_SAMPLE_PCT" in capsys.readouterr().err
    assert client.calls == []


# --- no runs, no judge -----------------------------------------------------------------------------------------------------------------------


def test_with_no_runs_the_judge_is_not_called_and_the_client_is_not_touched():
    client, fake = OnlineRecordingClient(), FakeJudge()
    result = run_pass(client, fake)
    assert result.runs == [] and fake.calls == [] and client.methods_called() == ["list_runs"]


def test_an_unrelated_run_keeps_its_own_feedback_untouched():
    other = fake_run("someone-else")
    client = judge_client(generate_trees(2))
    client.seed_feedback(other, "user_score", version=None)
    kept = [item for item in client.feedback if item.run_id == other.id]
    run_pass(client, FakeJudge())
    assert [item for item in client.feedback if item.run_id == other.id] == kept
