"""What the job gives 5a's judge for one run (7 PR-3.F1, G8): the final answer as delivered with the path tag stripped by ticket 4's helper, the LZ name or the coordinates, the pilot's question, and the run's own path.

The message the judge builds is 5a's own `judge_message`, called here on what the job hands over; the real `Judge` on a fake chat model shows what a model would be sent.
"""

from datetime import UTC, datetime

import pytest
from judge_helpers import FakeJudgeModel, canned
from online_helpers import NO_PATH, OnlineRecordingClient
from online_judge_helpers import QUESTION, VERSION, FakeJudge, judge_client, judge_tree

from evals.agent.judge import Judge, judge_message
from evals.online import job, pull
from evals.online.reference import build_reference
from flysafelz.agent import guard_flow


def given(tree):
    """(example, answer) as the job gives them to the judge for this tree."""
    return job.judge_input(tree.root, build_reference(tree.root, tree.children))


def message_for(tree):
    example, answer = given(tree)
    return judge_message(example, answer)


# --- the place -------------------------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["Blue Mill", "West Sleekburn", "Crookham", "Ærø Strand", "Môn Aber", "Tŷ Mawr"])
def test_the_lz_name_in_the_run_metadata_is_the_place_the_judge_sees(name):
    assert f"\nLocation: {name}\n" in message_for(judge_tree("named", lz_name=name))


def test_the_lz_name_reaches_5a_as_the_example_s_case_and_nothing_else_of_the_run_s_place_is_added():
    example, _answer = given(judge_tree("named", lz_name="Blue Mill"))
    assert example["fixtures"] == {"case": "Blue Mill"}


def test_with_no_lz_name_the_place_is_the_launch_zone_at_the_run_s_coordinates():
    tree = judge_tree("unnamed")
    assert "fixtures" not in given(tree)[0]
    assert "\nLocation: the launch zone at 53.8028, -0.0866\n" in message_for(tree)


@pytest.mark.parametrize("blank", ["", "   "])
def test_a_blank_lz_name_counts_as_absent(blank):
    assert "\nLocation: the launch zone at 53.8028, -0.0866\n" in message_for(judge_tree("blank", lz_name=blank))


def test_a_name_without_coordinates_is_enough():
    tree = judge_tree("nocoords", lz_name="Buttermere")
    del tree.root.inputs["lat"], tree.root.inputs["lon"]
    assert "\nLocation: Buttermere\n" in message_for(tree)


def test_neither_a_name_nor_coordinates_cannot_be_given_to_the_judge():
    tree = judge_tree("nowhere")
    del tree.root.inputs["lat"], tree.root.inputs["lon"]
    with pytest.raises(ValueError, match="no lz_name and no lat and lon"):
        given(tree)


# --- the answer ------------------------------------------------------------------------------------------------------------------------------


def test_the_path_tag_is_stripped_with_ticket_4s_helper():
    assert job.split_path is guard_flow.split_path
    tree = judge_tree("tagged", answer="[path: assess]\nNO-GO this morning: wind over the limit.")
    example, answer = given(tree)
    assert answer.text == "NO-GO this morning: wind over the limit."
    assert "[path" not in judge_message(example, answer)


def test_the_text_the_job_gives_is_whatever_split_path_returns(monkeypatch):
    seen = []

    def stub(text):
        seen.append(text)
        return None, "STRIPPED BY THE STUB"

    monkeypatch.setattr(job, "split_path", stub)
    tree = judge_tree("stub", answer="[path: assess]\nthe delivered answer")
    assert given(tree)[1].text == "STRIPPED BY THE STUB" and seen == ["[path: assess]\nthe delivered answer"]


def test_an_answer_with_no_tag_is_given_as_it_was_delivered():
    answer = "GO WITH CONDITIONS tomorrow 10:00 to 16:00.\n\nPhone the airfield first on 01632 960123."
    assert given(judge_tree("plain", answer=answer))[1].text == answer


def test_a_unicode_answer_is_kept_as_written():
    answer = "[path: explain]\nCloud base — the height of the lowest cloud (≈ 1 500 ft). Ærø, Môn, Tŷ."
    assert given(judge_tree("unicode", path="explain", answer=answer))[1].text == "Cloud base — the height of the lowest cloud (≈ 1 500 ft). Ærø, Môn, Tŷ."


def test_the_judge_is_given_the_run_s_own_path_and_the_time_and_question_of_the_run():
    tree = judge_tree("own", path="decline", metadata={"intended_path": "assess"})
    example, answer = given(tree)
    assert (example["reference"], answer.path) == ({"path": "decline"}, "decline")
    assert example["inputs"]["question"] == QUESTION and example["inputs"]["now"] == "2026-10-31T06:30:00+00:00"
    assert example["id"] == str(tree.root.id)
    assert "intended_path" not in str(example)


def test_a_path_in_the_outputs_is_the_run_s_path_when_the_metadata_has_none():
    tree = judge_tree("outputs", path=NO_PATH, output_path="clarify")
    assert given(tree)[0]["reference"]["path"] == "clarify"


# --- the question ----------------------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "messages",
    [
        [{"type": "human", "content": "Q1?"}],
        [{"role": "user", "content": "Q1?"}],
        [{"lc": 1, "type": "constructor", "kwargs": {"type": "human", "content": "Q1?"}}],
        [{"type": "human", "content": [{"type": "text", "text": "Q1?"}]}],
        ["Q1?"],
        [{"type": "human", "content": "earlier"}, {"type": "ai", "content": "an answer"}, {"type": "human", "content": "Q1?"}, {"type": "ai", "content": "another answer"}],
    ],
)
def test_the_question_is_the_last_pilot_message_of_the_run_s_inputs(messages):
    tree = judge_tree("shapes")
    tree.root.inputs["messages"] = messages
    assert given(tree)[0]["inputs"]["question"] == "Q1?"


def test_an_inputs_question_is_used_when_the_run_has_one():
    tree = judge_tree("asked")
    tree.root.inputs["question"] = "What airspace is above West Sleekburn?"
    assert given(tree)[0]["inputs"]["question"] == "What airspace is above West Sleekburn?"


@pytest.mark.parametrize("messages", [[], None, [{"type": "ai", "content": "only the agent spoke"}], [{"type": "human", "content": "  "}], "not a list"])
def test_a_run_with_no_pilot_question_cannot_be_given_to_the_judge(messages):
    tree = judge_tree("noquestion")
    tree.root.inputs["messages"] = messages
    with pytest.raises(ValueError, match="no pilot question"):
        given(tree)


# --- an empty final answer never reaches the judge -------------------------------------------------------------------------------------------


@pytest.mark.parametrize("answer", ["", "   \n"])
def test_an_empty_final_answer_is_skipped_as_errored_by_the_pull_and_never_reaches_the_judge(answer):
    tree = judge_tree("empty", answer=answer)
    client = OnlineRecordingClient([tree.root], children=tree.children)
    pulled = pull.pull_runs(client, "flysafelz", pull.parse_since("7d"), 500, datetime(2026, 10, 31, 7, 30, tzinfo=UTC))
    assert pulled.runs == [] and [(item.reason, item.source) for item in pulled.skipped] == [("errored", "pilot")]
    fake = FakeJudge()
    result = job.score_and_post(client, "flysafelz", pulled.runs, version=VERSION, judge=fake)
    assert fake.calls == [] and result.runs == [] and client.create_calls() == []


def test_a_run_whose_answer_is_only_a_tag_reaches_5as_judge_which_makes_no_model_call_and_gives_no_result():
    tree = judge_tree("only-tag", answer="[path: assess]")
    client = judge_client([tree])
    model = FakeJudgeModel(replies=[])
    result = job.score_and_post(client, "flysafelz", [tree.root], version=VERSION, judge=Judge(model))
    assert model.calls == []
    posted = {item.key: item for item in client.feedback}
    for key in job.judge_keys_for("assess"):
        assert posted[key].score is None and posted[key].comment.startswith(f"{job.NO_JUDGE_RESULT} | ")
    assert result.runs[0].status == job.SCORED


# --- through 5a's real judge on a fake chat model --------------------------------------------------------------------------------------------


def test_5as_judge_sends_its_model_the_stripped_answer_the_name_and_the_question():
    tree = judge_tree("real", lz_name="Ærø Strand", answer="[path: assess]\nNO-GO: gusts of 28 mph are over the limit.")
    client = judge_client([tree])
    model = FakeJudgeModel(replies=[canned(prioritised=(False, "the limit comes second"))])
    job.score_and_post(client, "flysafelz", [tree.root], version=VERSION, judge=Judge(model))
    (sent,) = model.user_texts()
    assert "Location: Ærø Strand" in sent and f"Question: {QUESTION}" in sent
    assert sent.endswith("Final answer:\nNO-GO: gusts of 28 mph are over the limit.") and "[path" not in sent
    assert model.calls[0]["metadata"]["source"] == "judge"
    posted = {item.key: item for item in client.feedback}
    assert (posted["online.judge.prioritised"].score, posted["online.judge.prioritised"].comment) == (0, f"the limit comes second | evaluator_version={VERSION}")
    assert posted["online.judge.answers_the_question"].score == 1


def test_an_unreadable_reply_is_5as_own_handling_a_null_with_its_reason_for_every_criterion_and_the_job_adds_no_parser():
    tree = judge_tree("garbled")
    client = judge_client([tree])
    model = FakeJudgeModel(replies=["this is not json"])
    result = job.score_and_post(client, "flysafelz", [tree.root], version=VERSION, judge=Judge(model))
    judged = {item.key: item for item in client.feedback if item.key.startswith("online.judge.")}
    assert sorted(judged) == sorted(job.judge_keys_for("assess")) and len(model.calls) == 1
    assert all(item.score is None and item.comment.startswith("judge error: ") for item in judged.values())
    assert result.runs[0].judge_error is None and result.runs[0].status == job.SCORED
    assert [item.key for item in client.feedback][-1] == "online.evaluator_version"


def test_a_run_with_no_judge_input_is_that_run_s_judge_error_and_the_model_is_not_called():
    tree = judge_tree("noquestion")
    tree.root.inputs["messages"] = []
    client = judge_client([tree])
    model = FakeJudgeModel(replies=[])
    result = job.score_and_post(client, "flysafelz", [tree.root], version=VERSION, judge=Judge(model))
    assert model.calls == [] and result.judge_calls == 0
    assert result.runs[0].judge_error is not None and result.runs[0].judge_error.stage == "judge" and "no pilot question" in result.runs[0].judge_error.message
    assert not [item for item in client.feedback if item.key.startswith("online.judge.")] and "online.evaluator_version" not in [item.key for item in client.feedback]
