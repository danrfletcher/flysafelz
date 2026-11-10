"""Which runs the judge's sample considers (7 PR-3.F1, G8): which judge keys exist at which version, for every rule of the eligibility line.

A run is eligible when its path has judge keys, at least one is missing at the current version, and it has no `not_sampled` record at the current version made at the same percentage. The expected answer for
each case is written out here from that rule.
"""

import pytest
from online_helpers import OnlineRecordingClient, fake_run

from evals.agent.judge import CRITERIA, CRITERIA_BY_PATH
from evals.online import job

CURRENT, OLDER = "aaaaaaaaaaaa", "bbbbbbbbbbbb"
PCT = 20
SAMPLE_COMMENT = "left out of the judge sample: 120 eligible runs, more than 50, chosen by run-id hash at sample_pct={pct}"
PATHS = sorted(CRITERIA_BY_PATH)  # the paths 5a gives criteria to


def keys_of(path):
    return [f"online.judge.{name}" for name in CRITERIA_BY_PATH[path]]


def existing(*entries):
    """Feedback as an earlier pass left it: (key, version) or (key, version, comment), on one run, stored by the recording client."""
    client, run = OnlineRecordingClient(), fake_run("eligibility")
    for key, version, *comment in entries:
        client.seed_feedback(run, key, version=version, score=None if key == job.SAMPLED_KEY else 1, comment=comment[0] if comment else "earlier")
    return client.feedback


@pytest.mark.parametrize("path", PATHS)
def test_a_run_with_no_judge_feedback_is_eligible(path):
    assert job.judge_eligible(path, existing(), CURRENT, PCT) is True


@pytest.mark.parametrize("path", PATHS)
def test_a_run_with_every_judge_key_of_its_path_at_this_version_is_not_eligible(path):
    assert job.judge_eligible(path, existing(*[(key, CURRENT) for key in keys_of(path)]), CURRENT, PCT) is False


@pytest.mark.parametrize("path", [p for p in PATHS if len(CRITERIA_BY_PATH[p]) > 1])
def test_a_partly_judged_run_is_eligible_and_so_counts(path):
    keys = keys_of(path)
    assert job.judge_eligible(path, existing(*[(key, CURRENT) for key in keys[:-1]]), CURRENT, PCT) is True
    assert job.judge_eligible(path, existing(*[(key, CURRENT) for key in keys[1:]]), CURRENT, PCT) is True


@pytest.mark.parametrize("path", PATHS)
def test_judge_keys_at_an_older_version_only_leave_the_run_eligible(path):
    assert job.judge_eligible(path, existing(*[(key, OLDER) for key in keys_of(path)]), CURRENT, PCT) is True


@pytest.mark.parametrize("path", PATHS)
def test_a_judge_key_with_a_null_score_counts_as_present(path):
    client, run = OnlineRecordingClient(), fake_run("null")
    for key in keys_of(path):
        client.seed_feedback(run, key, version=CURRENT, score=None, comment="judge error: x")
    assert job.judge_eligible(path, client.feedback, CURRENT, PCT) is False


def test_the_version_is_read_from_the_comment_when_extra_is_not_returned():
    client, run = OnlineRecordingClient(), fake_run("comment")
    for key in keys_of("assess"):
        client.seed_feedback(run, key, version=CURRENT, extra_returned=False)
    assert job.judge_eligible("assess", client.feedback, CURRENT, PCT) is False
    assert job.judge_eligible("assess", client.feedback, OLDER, PCT) is True


@pytest.mark.parametrize("path", PATHS)
def test_not_sampled_at_this_version_and_the_same_percentage_is_not_eligible(path):
    assert job.judge_eligible(path, existing((job.SAMPLED_KEY, CURRENT, SAMPLE_COMMENT.format(pct=PCT))), CURRENT, PCT) is False


@pytest.mark.parametrize("other", [0, 19, 21, 100])
def test_not_sampled_at_another_percentage_is_eligible_again(other):
    assert job.judge_eligible("assess", existing((job.SAMPLED_KEY, CURRENT, SAMPLE_COMMENT.format(pct=other))), CURRENT, PCT) is True


def test_not_sampled_at_an_older_version_is_not_a_record_at_this_one():
    assert job.judge_eligible("assess", existing((job.SAMPLED_KEY, OLDER, SAMPLE_COMMENT.format(pct=PCT))), CURRENT, PCT) is True


def test_a_not_sampled_record_whose_comment_has_no_percentage_is_reconsidered():
    assert job.judge_eligible("assess", existing((job.SAMPLED_KEY, CURRENT, "no percentage here")), CURRENT, PCT) is True


def test_the_percentage_in_the_comment_is_whole_number_text_not_a_prefix_of_another():
    assert job.sampled_pct_at(existing((job.SAMPLED_KEY, CURRENT, SAMPLE_COMMENT.format(pct=100))), CURRENT) == 100
    assert job.judge_eligible("assess", existing((job.SAMPLED_KEY, CURRENT, SAMPLE_COMMENT.format(pct=100))), CURRENT, 10) is True


@pytest.mark.parametrize("path", ["unknown", "act", "nonsense", "", None])
def test_a_path_with_no_criteria_is_never_eligible(path):
    assert job.judge_keys_for(path) == ()
    assert job.judge_eligible(path, existing(), CURRENT, PCT) is False


def test_feedback_that_this_job_did_not_write_does_not_count():
    client, run = OnlineRecordingClient(), fake_run("foreign")
    for key in keys_of("assess"):
        client.seed_feedback(run, key, version=None)
    assert job.judge_eligible("assess", client.feedback, CURRENT, PCT) is True


def test_the_judge_keys_of_a_path_are_5as_criteria_for_that_path_under_the_online_prefix():
    for path, criteria in CRITERIA_BY_PATH.items():
        assert job.judge_keys_for(path) == tuple(f"online.judge.{name}" for name in criteria)
    assert {name for names in CRITERIA_BY_PATH.values() for name in names} <= set(CRITERIA)
    assert job.SAMPLED_KEY == "online.judge.sampled" and job.SAMPLED_KEY not in {job.judge_key(name) for name in CRITERIA}
