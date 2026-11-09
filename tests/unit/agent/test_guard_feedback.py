"""G15 / FR18, I5, I6 (PR-12.F1): the `guard_fail` feedback and the root run's `path` and `source`, on the real graph over a scripted model.

Only the model is scripted, and the LangSmith client records instead of sending (tests/tracing_helpers.py): its HTTP session is a recording adapter, so no
socket is opened. `FeedbackLog` is that client with `create_feedback` replaced by a recorder of (run_id, key, score, comment), for the rows that read the call.
"""

import json
import logging
import warnings
from typing import Any

import httpx
import pytest
from graph_helpers import GP3_QUESTION, RaisingModel, agent_graph, ask, assess_call, blue_mill_context, call, thread
from langchain_core.messages import HumanMessage
from langsmith import Client
from tool_helpers import BLUE_MILL_LZ, BLUE_MILL_NOW, BLUE_MILL_WINDOW
from tracing_helpers import PROJECT_ID, model_for, recording_client, traced_turn

from flysafelz.agent import tracing
from flysafelz.agent.graph import build_agent
from flysafelz.agent.guard_flow import GuardAttempt, GuardOutcome
from flysafelz.agent.input_check import RequestInvalid
from flysafelz.agent.state import RunContext

LOGGER = "flysafelz.agent.tracing"
OVERALL = "PILOT DISCRETION \u2013 APPROACHING LIMITS \u00b7 WITH CONDITIONS"
GOOD = f"[path: assess]\n{OVERALL}, flyable from 10:00 GMT to 16:00 GMT. Phone Flylogix <PHONE_1> first (J3857/26), and avoid the sector to the SSE (H7171/26)."
BAD = "[path: assess]\nNO-GO from 10:00 GMT to 16:00 GMT."  # a verdict assess did not return (E12)
NOT_A_VERDICT = "'NO-GO' is not a verdict assess returned this turn"
UNTAGGED = GOOD.removeprefix("[path: assess]\n")  # a good answer with no path line: the tag is the only problem
ROUNDED = GOOD.replace("flyable from", "gust spread about 4 mph, at the caution limit, flyable from")  # the Blue Mill spread is 3.8 mph, the caution limit 4 mph (E11)
ACROSS = "'about 4 mph' rounds 3.8 mph across the 4 mph limit"
INVENTED_NUMBER = GOOD.replace("<PHONE_1>", "01632 960001")  # a drama-range number no tool printed (E14)
DECLINE = "[path: decline]\nI can only say whether it is flyable at a spot you click."


class FeedbackLog(Client):
    """The recording client whose `create_feedback` only records what it was asked: `calls` holds (run_id, key, score, comment, extra arguments)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.calls: list[tuple[Any, ...]] = []

    def create_feedback(self, run_id=None, key="", *, score=None, comment=None, **kwargs):  # type: ignore[override]
        self.calls.append((run_id, key, score, comment, kwargs))


class FeedbackFails(Client):
    """A client whose feedback call fails the way LangSmith does when it is down."""

    def create_feedback(self, *args, **kwargs):  # type: ignore[override]
        raise RuntimeError("503 Service Unavailable from api.smith.langchain.com")


def weather_call():
    """The forecast the gust spread comes from: assess alone does not print it."""
    return call("get_weather", "w", **BLUE_MILL_WINDOW)


def turn(client: Client, *steps, **more) -> dict[str, Any]:
    return traced_turn(client, *steps, **more)


def root_id(rec) -> str:
    (root,) = rec.root_runs()
    return str(root["id"])


# --- the score -----------------------------------------------------------------------------------------------------------


def test_a_first_draft_that_passes_sends_one_guard_fail_0_on_the_root_run():
    client, rec = recording_client(FeedbackLog)
    state = turn(client, assess_call("a"), GOOD)
    assert state["guard"] == "passed"
    assert len(client.calls) == 1
    run_id, key, score, comment, _ = client.calls[0]
    assert (str(run_id), key, score, comment) == (root_id(rec), "guard_fail", 0, None)


def test_a_draft_fixed_by_the_retry_sends_one_guard_fail_1_naming_the_finding():
    client, rec = recording_client(FeedbackLog)
    state = turn(client, assess_call("a"), BAD, GOOD)
    assert state["guard"] == "retried"
    ((run_id, key, score, comment, _),) = client.calls
    assert (str(run_id), key, score) == (root_id(rec), "guard_fail", 1)
    assert comment == f"Round 1 (checks):\n- {NOT_A_VERDICT}"  # E12: the verdict named without assess


def test_two_failed_drafts_end_in_the_fallback_and_the_one_feedback_carries_both_rounds():
    client, rec = recording_client(FeedbackLog)
    state = turn(client, assess_call("a"), BAD, BAD)
    assert state["guard"] == "fallback"
    ((run_id, _, score, comment, _),) = client.calls
    assert (str(run_id), score) == (root_id(rec), 1)
    assert comment == f"Round 1 (checks):\n- {NOT_A_VERDICT}\nRound 2 (checks):\n- {NOT_A_VERDICT}"


def test_a_round_whose_only_problem_is_the_path_tag_is_filed_under_path_tag():
    client, rec = recording_client(FeedbackLog)
    state = turn(client, assess_call("a"), UNTAGGED, UNTAGGED)
    ((run_id, _, score, comment, _),) = client.calls
    assert (str(run_id), score) == (root_id(rec), 1)
    assert comment is not None and comment.count("(path_tag)") == 2 and "(checks)" not in comment
    assert state["guard"] == "retried" and state["answer"].startswith(OVERALL)  # the answer passed every other check, so the pilot has it


def test_the_feedback_goes_through_the_real_client_as_one_post_feedback_with_the_project_and_trace_ids():
    client, rec = recording_client()
    turn(client, assess_call("a"), BAD, GOOD)
    (body,) = rec.feedback()
    assert body["run_id"] == body["trace_id"] == root_id(rec)
    assert body["key"] == "guard_fail" and body["score"] == 1
    assert body["session_id"] == str(PROJECT_ID)
    assert [str(r) for r in rec.of("POST", "/feedback")] == ["POST /feedback"]


def test_the_feedback_names_the_project_so_langsmith_does_not_warn_it_is_deprecated():
    client, rec = recording_client()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        turn(client, assess_call("a"), GOOD)
    assert len(rec.feedback()) == 1 and rec.feedback()[0]["session_id"] == str(PROJECT_ID)
    assert not [str(w.message) for w in caught if "feedback" in str(w.message).lower() or "session_id" in str(w.message)]


def test_the_project_asked_for_is_the_one_the_tracer_names(monkeypatch):
    client, rec = recording_client()
    monkeypatch.setenv("LANGSMITH_PROJECT", "my-project")
    turn(client, assess_call("a"), GOOD)
    assert len(rec.of("GET", "/sessions")) == 1  # the project is asked for once for the turn
    assert all(run["session_name"] == "my-project" for run in rec.runs().values())


def test_a_second_turn_that_fails_once_sends_its_own_guard_fail_1_distinct_from_the_first():
    client, rec = recording_client()
    graph = tracing.traced(agent_graph(model_for(assess_call("a"), GOOD, assess_call("b"), BAD, GOOD)), client)
    ask(graph)
    ask(graph, "And between 12 and 3?", context=blue_mill_context())
    tracing.wait_for_feedback()
    first, second = sorted(rec.root_runs(), key=lambda run: run["start_time"])
    feedback = {body["run_id"]: body for body in rec.feedback()}
    assert len(rec.feedback()) == 2 and set(feedback) == {str(first["id"]), str(second["id"])}
    assert feedback[str(first["id"])]["score"] == 0
    assert feedback[str(second["id"])]["score"] == 1 and NOT_A_VERDICT in feedback[str(second["id"])]["comment"]


# --- the comment is anonymised --------------------------------------------------------------------------------------------


def test_e11_a_number_rounded_across_a_limit_is_fixed_by_the_retry_and_the_feedback_names_the_finding():
    """The real Blue Mill gust spread is 3.8 mph against a caution at 4 mph: 'about 4 mph' rounds across that limit (E11)."""
    client, rec = recording_client(FeedbackLog)
    state = turn(client, weather_call(), assess_call("a"), ROUNDED, GOOD)
    assert state["guard"] == "retried" and state["answer"].startswith(OVERALL)
    ((run_id, key, score, comment, _),) = client.calls
    assert (str(run_id), key, score) == (root_id(rec), "guard_fail", 1)
    assert comment is not None and comment.startswith("Round 1 (checks):\n- ") and comment.count("Round") == 1
    assert ACROSS in comment


def test_e11_two_drafts_that_round_across_the_limit_end_in_the_fallback_and_the_one_feedback_names_it_in_both_rounds():
    client, rec = recording_client(FeedbackLog)
    state = turn(client, weather_call(), assess_call("a"), ROUNDED, ROUNDED)
    assert state["guard"] == "fallback"
    ((run_id, _, score, comment, _),) = client.calls
    assert (str(run_id), score) == (root_id(rec), 1)
    assert comment is not None and comment.count(ACROSS) == 2 and comment.index("Round 1") < comment.index("Round 2")


def test_e14_a_number_the_model_invented_fails_the_guard_and_is_masked_in_the_comment_that_quotes_it():
    """The drama-range number 01632 960001 is in the draft: the real guard (c) refuses it, and the finding that quotes it is anonymised on its way to LangSmith."""
    client, rec = recording_client(FeedbackLog)
    state = turn(client, assess_call("a"), INVENTED_NUMBER, GOOD)
    assert state["guard"] == "retried" and "960001" not in state["answer"]
    ((run_id, _, score, comment, _),) = client.calls
    assert (str(run_id), score) == (root_id(rec), 1)
    assert comment == "Round 1 (checks):\n- '<PHONE>' looks like a real phone number, which only a placeholder may stand for"
    assert "960001" not in comment
    sent = [json.dumps({key: value for key, value in (request.body or {}).items() if key != "serialized"}) for request in rec.sent]
    assert not [
        text for text in sent if "960001" in text
    ]  # nor is it in any run, input, output or feedback sent. (`serialized` is the scripted fake model's own repr of its script, which a real model does not have.)


def test_an_email_in_a_finding_is_masked_in_the_comment_too():
    client, _ = recording_client(FeedbackLog)

    def with_an_email(outcome: GuardOutcome) -> None:
        quoted = GuardAttempt(("ask ops@example.org or <PHONE_1>",), "checks")
        tracing.report_outcome(GuardOutcome(outcome.result, outcome.path, (quoted,)))

    graph = tracing.traced(build_agent(model_for(assess_call("a"), GOOD), **agent_parts(), on_outcome=with_an_email), client)
    ask(graph)
    tracing.wait_for_feedback()
    ((_, _, _, comment, _),) = client.calls
    assert comment == "Round 1 (checks):\n- ask <EMAIL> or <PHONE_1>"


def agent_parts() -> dict[str, Any]:
    from pathlib import Path

    from flysafelz.agent.sources import SourceCache
    from flysafelz.guidance.index import load_index

    return {"source_cache": SourceCache(), "index": load_index(Path("/nonexistent-guidance"))}


# --- no feedback when the guard did not run -----------------------------------------------------------------------------


def test_tracing_off_sends_nothing_and_changes_nothing(monkeypatch):
    monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)
    plain = ask(agent_graph(model_for(assess_call("a"), BAD, GOOD)))
    untraced = tracing.traced(agent_graph(model_for(assess_call("a"), BAD, GOOD)))  # no client: the graph itself
    again = ask(untraced)
    tracing.wait_for_feedback()
    assert plain["answer"] == again["answer"] and plain["guard"] == again["guard"] == "retried"


def test_e17_a_model_that_times_out_is_an_error_on_the_run_and_sends_no_guard_fail():
    client, rec = recording_client(FeedbackLog)
    state = turn(client, graph=tracing.traced(agent_graph(RaisingModel(error=httpx.ReadTimeout("timed out"))), client))
    assert state["answer"].startswith("The assistant is unavailable")  # code's own answer (FR17)
    assert client.calls == []
    assert any("ReadTimeout" in (run.get("error") or "") for run in rec.runs().values())


def test_e1_a_request_the_validation_rejects_sends_no_guard_fail_and_the_error_is_validations_own():
    client, rec = recording_client(FeedbackLog)
    graph = tracing.traced(agent_graph(model_for("never called")), client)
    with pytest.raises(RequestInvalid, match="lat must be between -90 and 90"):
        ask(graph, lz={"lat": 99.0, "lon": 2.35})
    tracing.wait_for_feedback()
    assert client.calls == []
    (root,) = rec.root_runs()
    assert "RequestInvalid" in root["error"]
    assert not any(run.get("error") and "RequestInvalid" not in run["error"] for run in rec.runs().values())  # nothing but the rejection is an error
    assert rec.metadata(root)["source"] == "case"  # the turn is tagged before it is checked


def test_e3_codes_uk_only_answer_sends_no_guard_fail_and_traces_without_error():
    client, rec = recording_client(FeedbackLog)
    state = turn(client, "never called", lz={"lat": 48.86, "lon": 2.35})
    assert state["answer"] == "FlySafeLZ only covers the UK."
    assert client.calls == []
    assert not any(run.get("error") for run in rec.runs().values())
    (root,) = rec.root_runs()
    assert rec.metadata(root).get("path") is None and "path" not in root["outputs"]


# --- a tracing failure never changes the pilot's answer --------------------------------------------------------------------


def test_a_feedback_call_that_raises_leaves_answer_and_guard_unchanged_and_logs_a_warning(caplog):
    expected = ask(agent_graph(model_for(assess_call("a"), BAD, GOOD)))
    client, _ = recording_client(FeedbackFails)
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        state = turn(client, assess_call("a"), BAD, GOOD)
    assert state["answer"] == expected["answer"] and state["guard"] == expected["guard"] == "retried"
    warned = [record.getMessage() for record in caplog.records if record.name == LOGGER and record.levelno == logging.WARNING]
    assert len(warned) == 1 and "guard_fail" in warned[0] and "RuntimeError" in warned[0]
    assert len(warned[0]) < 400  # the reason is cut short, not a whole response body


def test_a_client_that_cannot_find_the_project_still_sends_the_feedback_without_its_id(caplog):
    client, _ = recording_client(FeedbackLog)

    def missing(*args, **kwargs):
        raise RuntimeError("404 project not found")

    client.read_project = missing  # type: ignore[method-assign]
    with caplog.at_level(logging.INFO, logger=LOGGER):
        turn(client, assess_call("a"), GOOD)
    ((_, _, score, _, extra),) = client.calls
    assert score == 0 and extra["session_id"] is None


def test_reporting_inside_a_run_that_is_not_traced_does_nothing():
    tracing.report_outcome(GuardOutcome("passed", "assess"))
    tracing.tag_turn(RunContext(now=BLUE_MILL_NOW))
    tracing.wait_for_feedback()


# --- I5: the path -------------------------------------------------------------------------------------------------------


def raw_model_outputs(rec) -> str:
    return "\n".join(str(run.get("outputs")) for run in rec.runs().values() if run.get("run_type") == "llm")


def test_i5_a_tagged_turn_has_path_in_the_root_runs_metadata_and_the_model_output_keeps_the_tag():
    client, rec = recording_client()
    state = turn(client, assess_call("a"), GOOD)
    (root,) = rec.root_runs()
    assert rec.metadata(root)["path"] == "assess"
    assert "[path: assess]" in raw_model_outputs(rec)  # the raw draft, in the model call's own run
    assert "[path:" not in str(root["outputs"]["answer"]) and "[path:" not in state["answer"]
    assert "[path:" not in str(root["outputs"]["messages"])  # the final message in the root output has the tag off


def test_i5_an_untagged_turn_after_the_retry_has_path_unknown_and_a_path_tag_feedback_on_the_same_root_run():
    client, rec = recording_client()
    turn(client, assess_call("a"), UNTAGGED, UNTAGGED)
    (root,) = rec.root_runs()
    assert rec.metadata(root)["path"] == "unknown"
    (body,) = rec.feedback()
    assert body["run_id"] == str(root["id"]) and "path_tag" in body["comment"] and body["score"] == 1


def test_i5_the_path_is_the_tag_of_the_final_answer_not_of_a_failed_draft():
    client, rec = recording_client()
    turn(client, assess_call("a"), BAD.replace("assess", "explain"), GOOD)
    (root,) = rec.root_runs()
    assert rec.metadata(root)["path"] == "assess"


# --- I6: the source and path in outputs -------------------------------------------------------------------------------------


def one_turn(client, *, context: RunContext, metadata: dict[str, Any] | None = None, steps=(DECLINE,)):
    graph = tracing.traced(agent_graph(model_for(*steps)), client)
    config = {**thread(), **({"metadata": metadata} if metadata else {})}
    graph.invoke({"messages": [HumanMessage(GP3_QUESTION)], **BLUE_MILL_LZ}, config, context=context)
    tracing.wait_for_feedback()


def test_i6_a_case_mode_turn_has_source_case():
    client, rec = recording_client()
    one_turn(client, context=blue_mill_context())
    (root,) = rec.root_runs()
    assert rec.metadata(root)["source"] == "case"


def test_i6_a_plain_api_turn_has_source_pilot():
    client, rec = recording_client()
    one_turn(client, context=RunContext(now=BLUE_MILL_NOW))
    (root,) = rec.root_runs()
    assert rec.metadata(root)["source"] == "pilot"


def test_i6_a_turn_that_passes_online_seed_has_source_online_seed():
    client, rec = recording_client()
    one_turn(client, context=RunContext(now=BLUE_MILL_NOW), metadata={"source": "online-seed"})
    (root,) = rec.root_runs()
    assert rec.metadata(root)["source"] == "online-seed"


@pytest.mark.parametrize("asked", ["pilot", "online-seed", "case", "anything"])
def test_i6_a_caller_can_never_make_a_case_mode_turn_anything_but_case(asked):
    client, rec = recording_client()
    one_turn(client, context=blue_mill_context(), metadata={"source": asked})
    (root,) = rec.root_runs()
    assert rec.metadata(root)["source"] == "case"


@pytest.mark.parametrize("asked", ["case", "anything", "", None])
def test_i6_an_api_turn_asking_for_a_source_other_than_online_seed_is_a_pilot_turn(asked):
    client, rec = recording_client()
    one_turn(client, context=RunContext(now=BLUE_MILL_NOW), metadata={"source": asked})
    (root,) = rec.root_runs()
    assert rec.metadata(root)["source"] == "pilot"


def test_i6_path_is_in_the_root_runs_outputs_as_well_as_its_metadata():
    client, rec = recording_client()
    one_turn(client, context=blue_mill_context())
    (root,) = rec.root_runs()
    assert rec.metadata(root)["path"] == root["outputs"]["path"] == "decline"


def test_i6_only_the_root_run_carries_source_and_path():
    client, rec = recording_client()
    turn(client, assess_call("a"), GOOD)
    children = [run for run in rec.runs().values() if run.get("parent_run_id")]
    assert children and not [run for run in children if "source" in rec.metadata(run) or "path" in rec.metadata(run) or "path" in (run.get("outputs") or {})]


def test_the_state_the_caller_gets_has_no_path_key():
    state = turn(recording_client()[0], assess_call("a"), GOOD)
    assert "path" not in state
    assert set(state) >= {"answer", "guard", "overlays", "assessment", "messages"}


def test_a_failure_while_tagging_or_reporting_is_logged_and_never_raised(monkeypatch, caplog):
    def broken():
        raise RuntimeError("the tracer's run map is not what was expected")

    monkeypatch.setattr(tracing, "_root_run", broken)
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        tracing.tag_turn(None)
        tracing.report_outcome(GuardOutcome("passed", "assess"))
    warned = [record.getMessage() for record in caplog.records if record.name == LOGGER]
    assert len(warned) == 2 and "tag" in warned[0] and "guard result" in warned[1]


def test_a_failure_in_the_tracer_leaves_the_pilots_answer_unchanged(monkeypatch):
    expected = ask(agent_graph(model_for(assess_call("a"), GOOD)))

    def broken():
        raise RuntimeError("boom")

    monkeypatch.setattr(tracing, "_root_run", broken)
    client, _ = recording_client()
    state = turn(client, assess_call("a"), GOOD)
    assert (state["answer"], state["guard"]) == (expected["answer"], expected["guard"])
