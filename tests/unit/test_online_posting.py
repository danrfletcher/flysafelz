"""Which keys the job posts, and how (7 PR-2.F1, G9, G10): the missing-key rule as a table, the call each post makes, the version key's place, and the job over zero runs.

The expected post set is derived from the rule in the spec, not from the code: a key is present when feedback of that name carries the current evaluator version (a null score included);
the missing keys are posted in order; `online.evaluator_version` goes last, and only when it is not present at this version either.
"""

import itertools

import pytest
from online_helpers import OnlineRecordingClient, run_id, scored_tree

from evals.online import job

CURRENT, OLDER = "aaaaaaaaaaaa", "bbbbbbbbbbbb"
KEYS = list(job.DETERMINISTIC_KEYS)
VERSION_KEY = "online.evaluator_version"
PRESENCE = {"none": [], "some": KEYS[::2], "all": KEYS}


def expected_posts(present_keys, version_key_present, written_at):
    """The G9/G10 rule, written out: nothing counts as present unless it was written at the current version."""
    counted = set(present_keys) if written_at == CURRENT else set()
    version_counted = version_key_present and written_at == CURRENT
    return [key for key in KEYS if key not in counted] + ([] if version_counted else [VERSION_KEY])


CASES = list(itertools.product(PRESENCE, [CURRENT, OLDER], [True, False]))


def existing_feedback(client, run, present_keys, version_key_present, written_at):
    for key in present_keys:
        client.seed_feedback(run, key, version=written_at)
    if version_key_present:
        client.seed_feedback(run, VERSION_KEY, version=written_at)


@pytest.mark.parametrize(("presence", "written_at", "version_key_present"), CASES, ids=lambda case: str(case))
def test_missing_keys_follow_the_rule_for_every_combination(presence, written_at, version_key_present):
    client, tree = OnlineRecordingClient(), scored_tree("a")
    existing_feedback(client, tree.root, PRESENCE[presence], version_key_present, written_at)
    found = job.missing_keys(KEYS, client.feedback, CURRENT)
    assert found == expected_posts(PRESENCE[presence], version_key_present, written_at)


@pytest.mark.parametrize(("presence", "written_at", "version_key_present"), CASES, ids=lambda case: str(case))
def test_the_job_posts_exactly_the_missing_keys_with_the_version_key_last(presence, written_at, version_key_present):
    tree = scored_tree("a")
    client = OnlineRecordingClient([tree.root], children=tree.children)
    existing_feedback(client, tree.root, PRESENCE[presence], version_key_present, written_at)
    before = len(client.feedback)
    result = job.score_and_post(client, "flysafelz", [tree.root], version=CURRENT)
    expected = expected_posts(PRESENCE[presence], version_key_present, written_at)
    assert [call["key"] for call in client.create_calls()] == expected
    assert result.runs[0].posted == expected
    assert result.runs[0].status == (job.ALREADY_SCORED if not expected else job.SCORED)
    assert len(client.feedback) == before + len(expected)  # nothing older was deleted or edited
    if expected and VERSION_KEY in expected:
        assert expected[-1] == VERSION_KEY and client.create_calls()[-1]["key"] == VERSION_KEY


def test_g10_a_run_with_every_key_and_the_marker_at_this_version_is_already_scored_and_gets_no_write():
    tree = scored_tree("a")
    client = OnlineRecordingClient([tree.root], children=tree.children)
    existing_feedback(client, tree.root, KEYS, True, CURRENT)
    result = job.score_and_post(client, "flysafelz", [tree.root], version=CURRENT)
    assert [item.status for item in result.runs] == [job.ALREADY_SCORED] and result.posted == 0
    assert client.create_calls() == [] and [m for m in client.methods_called() if m == "create_feedback"] == []


def test_g10_all_keys_at_this_version_but_no_marker_posts_only_the_marker():
    tree = scored_tree("a")
    client = OnlineRecordingClient([tree.root], children=tree.children)
    existing_feedback(client, tree.root, KEYS, False, CURRENT)
    result = job.score_and_post(client, "flysafelz", [tree.root], version=CURRENT)
    assert result.runs[0].posted == [VERSION_KEY] and result.runs[0].status == job.SCORED


def test_g9_a_key_posted_at_this_version_with_a_null_score_counts_as_present_and_is_not_retried():
    tree = scored_tree("a")
    client = OnlineRecordingClient([tree.root], children=tree.children)
    job.score_and_post(client, "flysafelz", [tree.root], version=CURRENT)
    nulls = [posted.key for posted in client.feedback if posted.score is None]
    assert nulls == ["online.decline_no_tools"]  # a null really was posted
    again = job.score_and_post(client, "flysafelz", [tree.root], version=CURRENT)
    assert again.runs[0].status == job.ALREADY_SCORED and len(client.create_calls()) == len(KEYS) + 1


def test_g9_feedback_that_the_job_did_not_write_never_counts_as_present():
    client, tree = OnlineRecordingClient(), scored_tree("a")
    client.seed_feedback(tree.root, "online.guard_pass", version=None, comment="a person's own note")
    assert job.missing_keys(KEYS, client.feedback, CURRENT) == [*KEYS, VERSION_KEY]


def test_g9_the_version_is_read_from_the_comment_when_extra_is_not_returned():
    client, tree = OnlineRecordingClient(), scored_tree("a")
    stored = client.seed_feedback(tree.root, "online.guard_pass", version=CURRENT, extra_returned=False)
    assert stored.extra is None and (stored.comment or "").endswith(f"evaluator_version={CURRENT}")
    assert "online.guard_pass" not in job.missing_keys(KEYS, [stored], CURRENT)


def test_g9_a_version_change_posts_every_key_again_and_keeps_the_old_feedback():
    tree = scored_tree("a")
    client = OnlineRecordingClient([tree.root], children=tree.children)
    job.score_and_post(client, "flysafelz", [tree.root], version=OLDER)
    first = list(client.feedback)
    result = job.score_and_post(client, "flysafelz", [tree.root], version=CURRENT)
    assert result.runs[0].posted == [*KEYS, VERSION_KEY]
    assert client.feedback[: len(first)] == first and len(client.feedback) == 2 * len(first)
    assert {(posted.key, (posted.extra or {})["evaluator_version"]) for posted in client.feedback} == {(key, version) for key in [*KEYS, VERSION_KEY] for version in (OLDER, CURRENT)}


# --- the call each post makes ---------------------------------------------------------------------------------------------------------------


def test_g9_every_post_passes_the_project_uuid_read_once_and_no_trace_id():
    first, second = scored_tree("a"), scored_tree("b")
    client = OnlineRecordingClient([first.root, second.root], children=[*first.children, *second.children])
    job.score_and_post(client, "flysafelz", [first.root, second.root], version=CURRENT)
    reads = [kwargs for method, kwargs in client.calls if method == "read_project"]
    assert reads == [{"project_name": "flysafelz"}]
    project = client.read_project(project_name="flysafelz").id
    calls = client.create_calls()
    assert len(calls) == 2 * (len(KEYS) + 1)
    assert all(call["session_id"] == project for call in calls)
    assert all("trace_id" not in call for call in calls)  # a trace_id would make the SDK queue the post, and a failure would not raise here


def test_g9_each_post_carries_a_score_of_one_zero_or_null_a_comment_and_the_version():
    tree = scored_tree("a", guard="retried")
    client = OnlineRecordingClient([tree.root], children=tree.children)
    job.score_and_post(client, "flysafelz", [tree.root], version=CURRENT)
    for call in client.create_calls():
        assert call["score"] in (0, 1, None) and call["score"] is not True
        assert (call["comment"] or "").endswith(f" | evaluator_version={CURRENT}") and call["extra"] == {"evaluator_version": CURRENT}
        assert call["run_id"] == tree.root.id
    guard = next(call for call in client.create_calls() if call["key"] == "online.guard_pass")
    assert guard["score"] == 0 and "retried" in guard["comment"]
    assert next(call for call in client.create_calls() if call["key"] == "online.decline_no_tools")["comment"].startswith("not applicable")


def test_g9_the_job_reads_a_runs_feedback_before_it_posts_anything_for_it():
    tree = scored_tree("a")
    client = OnlineRecordingClient([tree.root], children=tree.children)
    job.score_and_post(client, "flysafelz", [tree.root], version=CURRENT)
    methods = client.methods_called()
    assert methods.index("list_feedback") < methods.index("create_feedback")
    assert [kwargs["run_ids"] for method, kwargs in client.calls if method == "list_feedback"] == [[tree.root.id]]


def test_g10_a_run_is_scored_even_when_nothing_is_posted_so_its_record_is_the_same_on_a_rerun():
    tree = scored_tree("a")
    client = OnlineRecordingClient([tree.root], children=tree.children)
    first = job.score_and_post(client, "flysafelz", [tree.root], version=CURRENT)
    second = job.score_and_post(client, "flysafelz", [tree.root], version=CURRENT)
    assert first.runs[0].record() == second.runs[0].record() and list(second.runs[0].scores) == KEYS


# --- no runs --------------------------------------------------------------------------------------------------------------------------------


def test_with_zero_runs_nothing_is_read_or_posted_and_read_project_is_not_called():
    client = OnlineRecordingClient()
    result = job.score_and_post(client, "flysafelz", [], version=CURRENT)
    assert result.runs == [] and result.posted == 0 and result.errors == []
    assert client.calls == []
    assert job.report_lines(result)[-1].startswith("0 scored, 0 already scored, 0 errored; 0 feedback posted")


def test_the_default_version_is_the_codes_own():
    client = OnlineRecordingClient()
    assert job.score_and_post(client, "flysafelz", []).version == job.evaluator_version()


def test_run_ids_in_the_helpers_are_stable():
    assert run_id("a") == run_id("a") != run_id("b")
