"""The judge in the evaluator list, through the runner (5a PR-7.F1, GP3 judge half, E5, F2, ticket 7's `source: judge`).

One committed DEV example per path goes through `runner.run_experiment` on the scripted fake graph and the recording client. Each run's answer is scored with the code
evaluators and the judge's four, on a fake chat model with canned structured replies, and the feedback goes to a real `langsmith.Client` whose HTTP session is the recording adapter
(tests/tracing_helpers.py). No model, LangSmith or bridge is called and no key is read (--disable-socket is on). No held-out example is built or judged (F7).

`uv run pytest tests/integration/test_judge_evaluator.py -s` prints the recorded feedback of the run.
"""

import json
from uuid import UUID

import pytest
from eval_runner_helpers import RunRecordingClient, ScriptedGraph, run
from judge_helpers import FakeJudgeModel, FeedbackClient, canned
from tracing_helpers import PROJECT_ID, recording_client

from evals.agent import judge as judge_module
from evals.agent import judge_config as judge_config_module
from evals.agent.evaluators.aggregate import EVALUATORS, ScoredRun, evaluate_runs
from evals.agent.evaluators.answer import ScoredAnswer
from evals.agent.evaluators.labels import Feedback
from evals.agent.judge import CRITERIA, SELF_PREFERENCE_LINE, Judge, all_evaluators, send_feedback
from evals.agent.loader import load_examples

JUDGE_KEYS = [f"judge.{name}" for name in CRITERIA]
DEV_PER_PATH = {"assess": "dev_01_blue_mill", "explain": "dev_05_druridge_bay", "clarify": "dev_07_buttermere", "decline": "dev_08_bamburgh"}
APPLIES = {
    "assess": {"judge.answers_the_question", "judge.prioritised", "judge.actionable"},
    "explain": {"judge.answers_the_question", "judge.prioritised", "judge.actionable"},
    "clarify": {"judge.answers_the_question"},
    "decline": {"judge.answers_the_question", "judge.good_decline"},
    "act": set(),
}


def dev_records():
    by_id = {r["id"]: r for r in load_examples("dev")}
    chosen = {path: by_id[example_id] for path, example_id in DEV_PER_PATH.items()}
    act = json.loads(json.dumps(by_id["dev_01_blue_mill"]))  # a path the table does not list: no dev example has it, so one is made from an assess one
    act["id"], act["reference"] = "dev_act_probe", {"path": "act"}
    return {**chosen, "act": act}


@pytest.fixture(scope="module")
def experiment(tmp_path_factory):
    """Run the five examples once through the runner; return (the records, the result, the runner's recording client)."""
    records = dev_records()
    client = RunRecordingClient()
    client.create_dataset(dataset_name="flysafelz-dev")
    result = run(list(records.values()), tmp_path_factory.mktemp("cache"), ScriptedGraph(), reps=1, client=client)
    return records, result, client


def scored_runs(records, result):
    by_id = {run.example_id: run for run in result.records}
    return [ScoredRun(record, ScoredAnswer(path=path, text=by_id[record["id"]].answer or "")) for path, record in records.items()]


def judged(records, result, replies):
    model = FakeJudgeModel(replies=list(replies))
    evaluation = evaluate_runs(scored_runs(records, result), all_evaluators(Judge(model)))
    return evaluation, model


def test_the_runner_ran_the_five_dev_examples_without_a_change_to_it(experiment):
    records, result, client = experiment
    assert [r.example_id for r in result.records] == [r["id"] for r in records.values()]
    assert not result.errored_runs and len(client.runs) == 5


def test_the_judge_sits_in_the_evaluator_list_after_the_code_evaluators():
    keys = list(all_evaluators(Judge(FakeJudgeModel())))
    assert keys == [*EVALUATORS, *JUDGE_KEYS]
    assert len(set(keys)) == len(keys)  # no judge key collides with a code evaluator's


def test_judge_feedback_per_applicable_criterion_and_null_for_the_rest(experiment):
    records, result, _client = experiment
    evaluation, model = judged(records, result, [canned(), canned(prioritised=(False, "The decisive NOTAM is in the last line.")), canned(), canned()])
    assert len(model.calls) == 4  # the act example never reaches the model
    for scores in evaluation.runs:
        path = next(p for p, r in records.items() if r["id"] == scores.example_id)
        graded = {key for key in JUDGE_KEYS if scores.feedback[key] is not None}
        assert graded == APPLIES[path], path
        for key in graded:
            assert scores.feedback[key]["key"] == key and scores.feedback[key]["score"] in (0.0, 1.0)
            assert scores.feedback[key]["comment"]


def test_scores_and_reasons_come_from_the_models_reply_for_the_right_example(experiment):
    records, result, _client = experiment
    evaluation, _ = judged(
        records, result, [canned(), canned(prioritised=(False, "The decisive NOTAM is in the last line.")), canned(), canned(good_decline=(False, "It gives launch advice anyway."))]
    )
    by_id = {s.example_id: s.feedback for s in evaluation.runs}
    assert by_id["dev_05_druridge_bay"]["judge.prioritised"] == {"key": "judge.prioritised", "score": 0.0, "comment": "The decisive NOTAM is in the last line."}
    assert by_id["dev_01_blue_mill"]["judge.prioritised"] == {"key": "judge.prioritised", "score": 1.0, "comment": "ok"}
    assert by_id["dev_08_bamburgh"]["judge.good_decline"]["score"] == 0.0
    assert evaluation.mean("judge.prioritised") == 0.5 and evaluation.applied("judge.prioritised") == 2
    assert evaluation.applied("judge.good_decline") == 1 and evaluation.applied("judge.answers_the_question") == 4


def test_judge_and_code_feedback_land_in_the_same_experiment_on_the_same_run(experiment, capsys):
    records, result, runner_client = experiment
    evaluation, _ = judged(records, result, [canned(), canned(), canned(), canned()])
    langsmith, recorder = recording_client()
    by_id = {run.example_id: run for run in result.records}
    for scores in evaluation.runs:
        run_id = by_id[scores.example_id].langsmith_run_id
        assert isinstance(run_id, UUID) and run_id in {r["id"] for r in runner_client.runs}
        send_feedback(langsmith, run_id, scores.feedback, project=result.experiment)  # code evaluators and judge alike: one create_feedback each
    sent = recorder.feedback()
    assert {item["run_id"] for item in sent} == {str(by_id[s.example_id].langsmith_run_id) for s in evaluation.runs}
    assert {item["key"] for item in sent if item["key"].startswith("judge.")} <= set(JUDGE_KEYS)
    assert any(not item["key"].startswith("judge.") for item in sent), "the code evaluators' feedback is in the same experiment"
    per_run = {}
    for item in sent:
        per_run.setdefault(item["run_id"], set()).add(item["key"])
    for scores in evaluation.runs:
        path = next(p for p, r in records.items() if r["id"] == scores.example_id)
        keys = per_run.get(str(by_id[scores.example_id].langsmith_run_id), set())
        assert {k for k in keys if k.startswith("judge.")} == APPLIES[path], path  # nothing sent for a criterion that does not apply
    with capsys.disabled():
        print("\nrecorded feedback (judge keys), per example:")
        for scores in evaluation.runs:
            print(f"  {scores.example_id}: " + ", ".join(f"{k}={'null' if (f := scores.feedback[k]) is None else f['score']}" for k in JUDGE_KEYS))


def test_feedback_items_carry_the_key_a_score_of_one_or_zero_and_the_reason():
    langsmith, recorder = recording_client()
    run_id = UUID(int=0xA1)
    grades = {
        "judge.answers_the_question": {"key": "judge.answers_the_question", "score": 1.0, "comment": "It answers."},
        "judge.prioritised": {"key": "judge.prioritised", "score": 0.0, "comment": "Buried."},
        "judge.actionable": None,
        "judge.good_decline": None,
    }
    assert send_feedback(langsmith, run_id, grades, project="v1-dev-test") == 2
    sent = recorder.feedback()
    assert [(i["key"], i["score"], i["comment"]) for i in sent] == [("judge.answers_the_question", 1.0, "It answers."), ("judge.prioritised", 0.0, "Buried.")]


def test_no_judge_feedback_uses_a_key_outside_the_judge_namespace(experiment):
    records, result, _ = experiment
    evaluation, _ = judged(records, result, [canned()] * 4)
    names = {key for scores in evaluation.runs for key, found in scores.feedback.items() if key.startswith("judge.") and found is not None}
    assert names <= set(JUDGE_KEYS)
    assert all(found is None or found["key"] == key for scores in evaluation.runs for key, found in scores.feedback.items())


# --- null, errors ---------------------------------------------------------------------------------------------------------------


def test_a_v0_run_with_no_prose_gives_null_for_every_criterion_and_never_calls_the_judge(experiment):
    records, _, _ = experiment
    v0 = ScoredAnswer.from_dict({"path": "assess", "segments": [], "notam_actions": {}})
    model = FakeJudgeModel()
    evaluation = evaluate_runs([ScoredRun(records["assess"], v0)], all_evaluators(Judge(model)))
    assert [evaluation.runs[0].feedback[key] for key in JUDGE_KEYS] == [None] * 4
    assert model.calls == []


def test_an_answer_of_blank_text_is_no_answer_text(experiment):
    records, _, _ = experiment
    model = FakeJudgeModel()
    evaluation = evaluate_runs([ScoredRun(records["assess"], ScoredAnswer(path="assess", text=" \n "))], all_evaluators(Judge(model)))
    assert [evaluation.runs[0].feedback[key] for key in JUDGE_KEYS] == [None] * 4 and model.calls == []


def test_a_model_call_that_raises_is_a_judge_error_for_that_example_and_the_run_continues(experiment):
    records, result, _ = experiment
    evaluation, model = judged(records, result, [RuntimeError("endpoint down: 503"), canned(), canned(), canned()])
    first, *rest = evaluation.runs
    assert first.example_id == "dev_01_blue_mill"
    for key in sorted(APPLIES["assess"]):
        assert first.feedback[key] == {"key": key, "score": None, "comment": "judge error: RuntimeError: endpoint down: 503"}
    assert first.feedback["judge.good_decline"] is None
    assert all(s.feedback["judge.answers_the_question"]["score"] == 1.0 for s in rest if s.example_id != "dev_act_probe")  # the others were judged
    assert len(model.calls) == 4
    assert evaluation.applied("judge.prioritised") == 1  # the error is not a fail: it has no score, so it is in no mean
    assert evaluation.mean("judge.prioritised") == 1.0


def test_a_judge_error_is_sent_to_langsmith_as_a_null_with_the_reason_and_not_as_a_fail(experiment):
    records, result, _ = experiment
    evaluation, _ = judged(records, result, [ValueError("bad reply"), canned(), canned(), canned()])
    langsmith, recorder = recording_client()
    first = evaluation.runs[0]
    send_feedback(langsmith, UUID(int=0xA2), first.feedback, project="v1-dev-test")
    errors = [i for i in recorder.feedback() if i["key"].startswith("judge.")]
    assert len(errors) == 3 and all(i.get("score") is None and i["comment"].startswith("judge error: ") for i in errors)


def test_a_reply_that_breaks_the_schema_is_a_judge_error_not_a_fail(experiment):
    records, result, _ = experiment
    evaluation, _ = judged(records, result, ["this is not json", canned(), canned(), canned()])
    first = evaluation.runs[0].feedback
    assert {first[k]["score"] for k in APPLIES["assess"]} == {None}
    assert all(first[k]["comment"].startswith("judge error: ") for k in APPLIES["assess"])


def test_the_key_is_scrubbed_from_a_judge_error():
    secret = "sk-test-abcdefghij"
    example = {"inputs": {"lat": 1.0, "lon": 2.0, "now": "2026-10-31T07:30Z", "question": "q"}, "reference": {"path": "clarify"}, "fixtures": {}}
    grades = Judge(FakeJudgeModel(replies=[RuntimeError(f"401 for key {secret}")]), secrets=(secret,)).judge(example, ScoredAnswer(text="Which day?"))
    assert secret not in grades["judge.answers_the_question"]["comment"] and "***" in grades["judge.answers_the_question"]["comment"]


def test_one_model_call_serves_the_four_evaluators_of_a_run_and_the_next_run_is_judged_afresh(experiment):
    records, result, _ = experiment
    _, model = judged(records, result, [canned()] * 4)
    assert len(model.calls) == 4  # four graded runs, four calls: not 16


# --- tracing: source judge, never pilot -------------------------------------------------------------------------------------------


def test_every_judge_model_call_is_traced_with_source_judge_and_never_pilot(experiment):
    records, result, _ = experiment
    langsmith, recorder = recording_client()
    model = FakeJudgeModel(replies=[canned()] * 4)
    evaluate_runs(scored_runs(records, result), all_evaluators(Judge(model, client=langsmith, project="v1-dev-20261108-160700")))
    langsmith.flush()
    runs = list(recorder.runs().values())
    assert runs, "the judge's model calls were not traced at all"
    assert {recorder.metadata(r).get("source") for r in runs} == {"judge"}
    assert all(recorder.metadata(r).get("source") != "pilot" and "pilot" not in (r.get("tags") or []) for r in runs)  # the prompt's own words use "pilot"; the tag and the source never do
    assert {r.get("session_name") for r in runs} == {"v1-dev-20261108-160700"}
    assert all(m.get("source") == "judge" for m in model_metadata(model))


def model_metadata(model):
    return [call["metadata"] for call in model.calls]


def test_the_tracer_anonymises_a_phone_number_that_slips_into_a_judge_trace():
    langsmith, recorder = recording_client()
    example = {"inputs": {"lat": 1.0, "lon": 2.0, "now": "2026-10-31T07:30Z", "question": "Ring 01632 960123?"}, "reference": {"path": "clarify"}, "fixtures": {}}
    Judge(FakeJudgeModel(replies=[canned()]), client=langsmith, project="p").judge(example, ScoredAnswer(text="Which day?"))
    langsmith.flush()
    assert "960123" not in recorder.text()


def test_judge_without_a_client_sends_nothing_anywhere():
    model = FakeJudgeModel(replies=[canned()])
    example = {"inputs": {"lat": 1.0, "lon": 2.0, "now": "2026-10-31T07:30Z", "question": "q"}, "reference": {"path": "clarify"}, "fixtures": {}}
    Judge(model).judge(example, ScoredAnswer(text="Which day?"))
    assert model.calls[0]["metadata"]["source"] == "judge"


# --- the report line, and nothing written ------------------------------------------------------------------------------------------


def test_the_self_preference_sentence_is_exported_for_the_report():
    assert SELF_PREFERENCE_LINE.startswith("The judge and the agent are the same model family")
    assert "self-preference risk" in SELF_PREFERENCE_LINE and "calibration" in SELF_PREFERENCE_LINE


def test_the_project_id_is_looked_up_once_for_all_the_runs_of_an_experiment():
    """R3: one feedback call per run, so a lookup per call would be 87 extra LangSmith reads in a dev run."""
    langsmith, recorder = recording_client()
    one: dict[str, Feedback | None] = {"judge.actionable": {"key": "judge.actionable", "score": 1.0, "comment": "ok"}}
    for number in range(5):
        assert send_feedback(langsmith, UUID(int=0xB0 + number), one, project="v1-dev-lookup-once") == 1
    assert len(recorder.of("GET", "/sessions")) == 1
    assert {item["session_id"] for item in recorder.feedback()} == {str(PROJECT_ID)}
    send_feedback(langsmith, UUID(int=0xB9), one, project="v1-dev-another")
    assert len(recorder.of("GET", "/sessions")) == 2  # another experiment is another lookup


def test_a_project_that_cannot_be_found_sends_the_feedback_without_its_id():
    client = FeedbackClient()  # it has no read_project at all
    one: dict[str, Feedback | None] = {"judge.actionable": {"key": "judge.actionable", "score": 1.0, "comment": "ok"}}
    assert send_feedback(client, "run-1", one, project="v1-dev-missing") == 1
    assert "session_id" not in client.feedback[0][4]


def test_the_feedback_client_stand_in_records_what_send_feedback_sends():
    client = FeedbackClient()
    send_feedback(client, "run-1", {"judge.actionable": {"key": "judge.actionable", "score": 1.0, "comment": "ok"}, "judge.good_decline": None})
    assert client.keys() == ["judge.actionable"] and client.feedback[0][0] == "run-1"


def test_no_code_path_in_the_judge_writes_to_the_dataset_the_sheet_or_an_annotation_queue():
    """F5: a grep over the judge's source. It reads files (the examples) and sends feedback; it writes no file and calls no dataset, example or queue method."""
    from pathlib import Path

    source = "\n".join(Path(module.__file__).read_text(encoding="utf-8") for module in (judge_module, judge_config_module))
    forbidden = (
        "write_text",
        "write_bytes",
        "open(",
        "evals/agent/data",
        "verdict_sheet",
        "create_dataset",
        "create_examples",
        "update_examples",
        "delete_examples",
        "annotation_queue",
        "create_annotation",
        "add_runs_to",
        "update_feedback",
    )
    assert [word for word in forbidden if word in source] == []
    assert source.count("create_feedback") == 1  # the one call, in send_feedback
