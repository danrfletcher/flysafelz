"""PR-11.F1: the guard flow around the model call (G15, P4), driven through `FlySafeMiddleware.wrap_model_call` with a stub handler.

The stub plays the model: it returns the scripted responses in order and records the request of each call. Everything else is real: the middleware, the
guard's checks (`guard_answer` over the Blue Mill tool results of a real graph run), the fallback, the re-check and limit lines and the rehydration.
"""

import inspect
from datetime import UTC, datetime
from functools import cache
from types import SimpleNamespace

import pytest
from factories import weather_forecast
from fakes import FakeToolChatModel
from graph_helpers import blue_mill_context
from langchain.agents.middleware import ExtendedModelResponse, ModelRequest, ModelResponse
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from tool_helpers import ALL_THREE, BLUE_MILL_WINDOW, ToolRun, calls, run_tools
from tool_world import World, go

from flysafelz.agent import guard_flow
from flysafelz.agent.guard import CANNOT_HELP, FALLBACK_HEADING, RAW_HEADING
from flysafelz.agent.guard_flow import CHECKER_NAME, GuardOutcome
from flysafelz.agent.middleware import FlySafeMiddleware
from flysafelz.agent.sources import SourceCache
from flysafelz.agent.state import OverlayFeature, OverlayProperties, PointGeometry
from flysafelz.agent.tools.assess_tool import build_assess

OVERALL = "PILOT DISCRETION \u2013 APPROACHING LIMITS \u00b7 WITH CONDITIONS"
GOOD = f"[path: assess]\n{OVERALL}, flyable from 10:00 GMT to 16:00 GMT."
GOOD_BODY = GOOD.removeprefix("[path: assess]\n")
BAD = "[path: assess]\nNO-GO from 10:00 GMT to 16:00 GMT."  # a verdict assess did not return
BAD_TEXT = "NO-GO from 10:00 GMT to 16:00 GMT."
RECHECK = "Based on data as of 07:30 GMT. Re-check the weather and NOTAMs when you get to the LZ."
QUESTION = "Can I fly here between 10 and 4 today?"


@cache
def blue_mill_run():
    return run_tools(SourceCache(), [calls(*ALL_THREE, **BLUE_MILL_WINDOW), calls("assess", **BLUE_MILL_WINDOW)])


def request_for(*, with_tools: bool = True, assess: bool = True, question: str = QUESTION, run: ToolRun | None = None) -> ModelRequest:
    run = run or blue_mill_run()
    kept = [m for m in run.messages if assess or m.name != "assess"]
    messages = [HumanMessage(question), *kept]
    state = {"messages": messages, "phone_map": run.state.get("phone_map") or {}, "thread_lz": run.state["thread_lz"], "overlays": run.state.get("overlays") or []}
    tools = [build_assess(SourceCache())] if with_tools else []
    return ModelRequest(model=FakeToolChatModel(), messages=messages, tools=tools, state=state, runtime=SimpleNamespace(context=blue_mill_context()))


class Stub:
    """The handler: the next scripted response for each call; a call past the script fails the test (never a third draft)."""

    def __init__(self, *drafts: str | AIMessage) -> None:
        self.drafts = list(drafts)
        self.requests: list[ModelRequest] = []

    def __call__(self, request: ModelRequest) -> ModelResponse:
        self.requests.append(request)
        if len(self.requests) > len(self.drafts):
            raise AssertionError(f"the model was called {len(self.requests)} times but the test scripted {len(self.drafts)}")
        draft = self.drafts[len(self.requests) - 1]
        return ModelResponse(result=[draft if isinstance(draft, AIMessage) else AIMessage(draft)])


def flow(*drafts: str | AIMessage, request: ModelRequest | None = None):
    """Run the middleware over a stub scripted with `drafts`: (what it returned, the stub, the GuardOutcomes it reported)."""
    stub, outcomes = Stub(*drafts), []
    result = FlySafeMiddleware(outcomes.append).wrap_model_call(request or request_for(), stub)
    return result, stub, outcomes


def sent(result: ExtendedModelResponse) -> AIMessage:
    return result.model_response.result[0]


def update(result: ExtendedModelResponse) -> dict:
    return result.command.update


# --- a first draft that passes --------------------------------------------------------------------------------------------------------------------


def test_a_first_draft_that_passes_is_one_model_call_and_no_checker_message():
    result, stub, outcomes = flow(GOOD)
    assert len(stub.requests) == 1
    assert not any(getattr(m, "name", None) == CHECKER_NAME for m in stub.requests[0].messages)
    assert update(result)["guard"] == "passed"
    assert sent(result).content == GOOD_BODY  # the tag is off; the message is what the model wrote after it
    assert update(result)["answer"].startswith(GOOD_BODY) and "[path" not in update(result)["answer"]
    assert outcomes == [GuardOutcome("passed", "assess", ())]  # a pass records zero failures


def test_the_answer_is_the_passed_text_then_the_limit_lines_then_the_recheck_line_in_that_order():
    result, _, _ = flow(GOOD)
    answer = update(result)["answer"]
    assert answer.startswith(GOOD_BODY + "\n\nOver caution limit: ")
    assert answer.endswith(RECHECK)
    assert answer.index("Over caution limit") < answer.index(RECHECK)


def test_a_response_that_calls_tools_is_handed_back_untouched_and_nothing_is_checked():
    asking = ModelResponse(result=[AIMessage("", tool_calls=[{"name": "assess", "args": {}, "id": "x"}])])
    outcomes = []
    handed = FlySafeMiddleware(outcomes.append).wrap_model_call(request_for(), lambda request: asking)
    assert handed is asking and outcomes == []


# --- fail then pass ---------------------------------------------------------------------------------------------------------------------------------


def test_a_failed_first_draft_gets_exactly_one_retry_with_the_findings_from_the_checker_and_no_tools():
    result, stub, outcomes = flow(BAD, GOOD)
    assert len(stub.requests) == 2
    first, retry = stub.requests
    assert first.tools and retry.tools == []  # the retry binds nothing
    checker = retry.messages[-1]
    assert isinstance(checker, HumanMessage) and checker.name == CHECKER_NAME
    assert "'NO-GO' is not a verdict assess returned this turn" in checker.content
    assert retry.messages[:-1] == first.messages  # the conversation, then the checker's message
    assert update(result)["guard"] == "retried"
    assert sent(result).content == GOOD_BODY
    [outcome] = outcomes
    assert (outcome.result, outcome.path) == ("retried", "assess")
    assert [a.reason for a in outcome.attempts] == ["checks"] and "'NO-GO' is not a verdict assess returned this turn" in outcome.attempts[0].findings


def test_nothing_of_the_failed_draft_or_the_checker_is_in_what_the_middleware_returns():
    result, _, _ = flow(BAD, GOOD)
    assert BAD_TEXT not in sent(result).content and BAD_TEXT not in update(result)["answer"]
    assert not any(isinstance(m, HumanMessage) for m in result.model_response.result)
    assert set(update(result)) == {"answer", "guard", "assessment"}  # no `messages` key: nothing else is written to the thread


def test_the_retry_is_the_requests_own_copy_and_the_original_request_keeps_its_tools():
    request = request_for()
    flow(BAD, GOOD, request=request)
    assert request.tools and request.messages[-1].type == "tool"  # override() copied; the first request is as it was


# --- fail then fail ---------------------------------------------------------------------------------------------------------------------------------


def test_a_second_failure_is_the_code_built_fallback_and_there_is_never_a_third_call():
    result, stub, outcomes = flow(BAD, BAD)
    assert len(stub.requests) == 2
    assert update(result)["guard"] == "fallback"
    assert sent(result).content.startswith(f"{FALLBACK_HEADING}:")
    assert update(result)["answer"].startswith(f"{FALLBACK_HEADING}:")
    assert BAD_TEXT not in sent(result).content and BAD_TEXT not in update(result)["answer"]
    [outcome] = outcomes
    assert outcome.result == "fallback" and len(outcome.attempts) == 2


def test_the_fallback_answer_is_never_empty_and_gets_the_same_closing_lines():
    result, _, _ = flow(BAD, BAD)
    assert update(result)["answer"].strip() and update(result)["answer"].endswith(RECHECK)


def test_without_assess_a_failed_explain_falls_back_to_the_raw_tool_text_and_a_decline_to_the_fixed_line():
    explain = "[path: explain]\nGO from 10:00 GMT."  # E12: a verdict with no assess this turn
    result, stub, _ = flow(explain, explain, request=request_for(assess=False))
    assert len(stub.requests) == 2
    assert update(result)["guard"] == "fallback" and update(result)["assessment"] is None
    assert sent(result).content.startswith(f"{RAW_HEADING}:")
    declined = "[path: decline]\nGO from 10:00 GMT."
    result, _, _ = flow(declined, declined, request=request_for(assess=False))
    assert sent(result).content == CANNOT_HELP and update(result)["answer"] == CANNOT_HELP  # no data lines on a decline


def test_with_assess_the_fallback_still_sets_the_assessment():
    result, _, _ = flow(BAD, BAD)
    assert update(result)["assessment"] is not None and update(result)["assessment"].overall is not None


# --- empty drafts -----------------------------------------------------------------------------------------------------------------------------------


def test_an_empty_final_message_is_one_finding_then_retry_then_fallback():
    result, stub, outcomes = flow("", "")
    assert len(stub.requests) == 2
    [outcome] = outcomes
    assert [len(a.findings) for a in outcome.attempts] == [1, 1]
    assert update(result)["guard"] == "fallback" and update(result)["answer"].strip()


def test_an_empty_first_draft_then_a_good_retry_is_retried():
    result, _, _ = flow("", GOOD)
    assert update(result)["guard"] == "retried"


def test_a_tag_and_nothing_else_is_empty_too():
    _, _, outcomes = flow("[path: assess]", GOOD)
    assert len(outcomes[0].attempts[0].findings) == 1


def test_a_retry_that_calls_a_tool_anyway_is_a_failure_not_a_third_call():
    calling = AIMessage(GOOD_BODY, tool_calls=[{"name": "assess", "args": {}, "id": "x"}])
    result, stub, _ = flow(BAD, calling)
    assert len(stub.requests) == 2 and update(result)["guard"] == "fallback"


# --- the settings own the count ---------------------------------------------------------------------------------------------------------------------


def test_the_retry_count_is_read_from_settings(monkeypatch):
    from flysafelz.agent.settings import SETTINGS

    assert SETTINGS.guard_retries == 1
    monkeypatch.setattr(guard_flow, "SETTINGS", SimpleNamespace(guard_retries=0, calendar_days=SETTINGS.calendar_days))
    result, stub, _ = flow(BAD)
    assert len(stub.requests) == 1 and update(result)["guard"] == "fallback"
    monkeypatch.setattr(guard_flow, "SETTINGS", SimpleNamespace(guard_retries=2, calendar_days=SETTINGS.calendar_days))
    result, stub, _ = flow(BAD, BAD, GOOD)
    assert len(stub.requests) == 3 and update(result)["guard"] == "retried"


def test_guard_py_names_no_retry_count():
    """There is no literal 1 standing for the retry count in the guard module or the flow: both read `guard_retries`."""
    source = inspect.getsource(guard_flow)
    assert "SETTINGS.guard_retries" in source and "retries_made < 1" not in source and "retries = 1" not in source


# --- no state between calls -------------------------------------------------------------------------------------------------------------------------


def test_the_middleware_keeps_nothing_between_two_model_calls_or_two_turns():
    middleware = FlySafeMiddleware()
    before = dict(vars(middleware))
    stub = Stub(BAD, GOOD, GOOD)
    first = middleware.wrap_model_call(request_for(), stub)
    assert dict(vars(middleware)) == before  # only its configuration
    second = middleware.wrap_model_call(request_for(question="And later?"), stub)
    assert update(first)["guard"] == "retried" and update(second)["guard"] == "passed"  # no retry carried over from turn 1
    assert len(stub.requests) == 3


# --- the guard sees placeholders; answer gets the number ---------------------------------------------------------------------------------------------


def test_the_guard_saw_placeholders_and_answer_has_the_real_number_while_the_message_keeps_the_placeholder():
    number = next(iter(blue_mill_run().state["phone_map"].values()))
    with_placeholder = f"[path: assess]\n{OVERALL}. Phone Flylogix <PHONE_1> first."
    result, stub, _ = flow(with_placeholder)
    assert len(stub.requests) == 1 and update(result)["guard"] == "passed"
    assert "<PHONE_1>" in sent(result).content and number not in sent(result).content
    assert number in update(result)["answer"] and "<PHONE_1>" not in update(result)["answer"]
    # a draft that writes the real number is a finding: the guard only knows the placeholder
    real = f"[path: assess]\n{OVERALL}. Phone Flylogix {number} first."
    _, stub, outcomes = flow(real, with_placeholder)
    assert len(stub.requests) == 2 and outcomes[0].attempts[0].reason == "checks"


def test_a_notam_overlay_text_has_the_number_back_in_the_output_while_the_state_it_came_from_keeps_the_placeholder():
    """I7: the overlays the turn holds go out with the real number in `text`; a feature with no placeholder is not rewritten (and not re-sent)."""
    number = next(iter(blue_mill_run().state["phone_map"].values()))
    request = request_for()
    with_phone = OverlayFeature(
        geometry=PointGeometry(coordinates=[-0.08, 53.8]), properties=OverlayProperties(kind="notam", id="J3857/26", text="CONTACT FLYLOGIX <PHONE_1> BEFORE USE", action="show")
    )
    request.state["overlays"] = [with_phone]
    result, _, _ = flow(GOOD, request=request)
    [out] = update(result)["overlays"]
    assert out.properties.text == f"CONTACT FLYLOGIX {number} BEFORE USE" and with_phone.properties.text == "CONTACT FLYLOGIX <PHONE_1> BEFORE USE"
    request.state["overlays"] = [with_phone.model_copy(update={"properties": with_phone.properties.model_copy(update={"text": "NO NUMBER"})})]
    result, _, _ = flow(GOOD, request=request)
    assert "overlays" not in update(result)


def test_no_phone_number_is_in_any_message_sent_to_the_model_on_the_retry():
    number = next(iter(blue_mill_run().state["phone_map"].values()))
    _, stub, _ = flow(f"[path: assess]\n{OVERALL}. Phone {number}.", GOOD)
    for request in stub.requests:
        assert all(number not in str(m.content) for m in request.messages)


def test_an_invented_phone_number_never_reaches_answer_even_after_a_failed_retry():
    invented = "[path: explain]\nCall Flylogix on 01632 960001."
    result, stub, outcomes = flow(invented, invented)
    assert len(stub.requests) == 2 and update(result)["guard"] == "fallback"
    assert "01632 960001" not in update(result)["answer"] and "01632 960001" not in sent(result).content
    assert any("01632 960001" in finding for finding in outcomes[0].attempts[0].findings)


# --- E11 / E13 --------------------------------------------------------------------------------------------------------------------------------------


def test_e11_rounding_across_a_limit_is_a_retry_and_a_corrected_retry_is_retried_and_a_repeat_is_fallback():
    rounded = f"[path: assess]\n{OVERALL}. The gust spread is about 4 mph, under the 4 mph limit."
    fixed = f"[path: assess]\n{OVERALL}. The gust spread is about 5 mph, over the 4 mph limit."
    _, stub, outcomes = flow(rounded, fixed)
    assert outcomes[0].result == "retried" and len(stub.requests) == 2
    _, stub, outcomes = flow(rounded, rounded)
    assert outcomes[0].result == "fallback" and len(stub.requests) == 2


def test_i10_a_draft_that_puts_a_value_on_the_wrong_side_of_its_limit_is_retried_and_the_limit_line_is_the_codes():
    """I10 prompt rule: gust spread over `> 6` (6.1 mph: the weather layer rounds a spread to 0.1 mph, so 6.04 cannot come from a forecast). 'under the 6 mph limit' is PR-10's position finding; the retry says 'over' and passes."""
    at = datetime(2026, 10, 31, 10, tzinfo=UTC)
    forecast = weather_forecast(datetime(2026, 10, 31, 8, tzinfo=UTC), datetime(2026, 10, 31, 16, tzinfo=UTC), changes={at: {"wind_10m": 10.0, "gust_10m": 16.1}})
    run = go(World(forecast=forecast), [calls("assess", **BLUE_MILL_WINDOW)])
    [artifact] = [m.artifact for m in run.messages if m.name == "assess"]
    assert any(c.rule_id == "gust_spread_no_go" and c.value == pytest.approx(6.1) for s in artifact.segments for c in s.blockers)  # the value the draft is about
    first = "[path: assess]\nThe gust spread is under the 6 mph limit."
    retry = "[path: assess]\nThe gust spread is over the 6 mph limit."
    result, stub, outcomes = flow(first, retry, request=request_for(run=run))
    assert len(stub.requests) == 2
    [outcome] = outcomes
    assert (outcome.result, outcome.attempts[0].reason) == ("retried", "checks")
    [finding] = outcome.attempts[0].findings
    assert "under the 6 mph limit" in finding and "over" in finding  # the position finding, and it alone made the retry
    assert finding in stub.requests[1].messages[-1].content  # the checker told the model so
    assert update(result)["guard"] == "retried"
    assert sent(result).content == "The gust spread is over the 6 mph limit."
    assert "Over limit: gust spread 6.1 mph (NO-GO above 6 mph)." in update(result)["answer"].splitlines()  # a line code built, not the model
    assert "under the 6 mph limit" not in update(result)["answer"] and "Over limit" not in sent(result).content


def test_e13_the_model_saying_no_need_to_recheck_is_a_retry_and_the_code_line_is_not_flagged():
    claim = f"[path: assess]\n{OVERALL}. This data is final, no need to re-check."
    result, stub, outcomes = flow(claim, GOOD)
    assert len(stub.requests) == 2 and outcomes[0].attempts[0].findings
    assert update(result)["guard"] == "retried" and update(result)["answer"].endswith(RECHECK)  # the line the code adds says "Re-check" and passes nothing through the guard


# --- the async hook ---------------------------------------------------------------------------------------------------------------------------------


def finish(coroutine):
    """Run a coroutine that awaits nothing to its end without an event loop (a loop opens a socket pair, which the suite blocks)."""
    with pytest.raises(StopIteration) as done:
        coroutine.send(None)
    return done.value.value


def test_the_async_hook_does_the_same_retry_and_fallback():
    stub, outcomes = Stub(BAD, GOOD), []

    async def handler(request):
        return stub(request)

    result = finish(FlySafeMiddleware(outcomes.append).awrap_model_call(request_for(), handler))
    assert update(result)["guard"] == "retried" and stub.requests[1].tools == [] and len(stub.requests) == 2

    stub = Stub(BAD, BAD)
    result = finish(FlySafeMiddleware().awrap_model_call(request_for(), handler))
    assert update(result)["guard"] == "fallback" and len(stub.requests) == 2


def test_a_tool_message_artifact_is_what_the_flow_reads_not_a_name():
    """The flow takes this turn's results from the messages after the pilot's newest one; a tool message of an earlier turn is not checked against."""
    request = request_for()
    old = ToolMessage("OLD TABLE: Overall: GO", name="assess", tool_call_id="old")
    messages = [HumanMessage("earlier"), old, AIMessage("earlier answer"), *request.state["messages"]]
    request = request.override(state={**request.state, "messages": messages}, messages=messages)
    _, _, outcomes = flow("[path: assess]\nGO from 10:00 GMT.", "[path: assess]\nGO from 10:00 GMT.", request=request)
    assert outcomes[0].result == "fallback"  # "GO" was the earlier turn's verdict, not this turn's
