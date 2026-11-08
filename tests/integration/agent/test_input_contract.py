"""The input contract through a real graph: `create_agent` with the input check as its only middleware, a fake chat model, an in-memory checkpoint.

Everything goes through `graph.invoke`, as a caller would. The fake model counts its calls, so "no model call" is a number, not a claim.
"""

from contextlib import contextmanager
from datetime import datetime

import pytest
from agent_helpers import contract_graph, thread
from langchain_core.messages import AIMessage, HumanMessage

from flysafelz.agent.input_check import UK_ONLY_ANSWER, RequestInvalid
from flysafelz.agent.state import OverlayFeature, RunContext, lz_radius_feature
from flysafelz.rules import RULES

BLUE_MILL = {"lat": 53.802797, "lon": -0.086644}
CASE = RunContext(case="blue_mill_2026-10-31")


@contextmanager
def rejected(message: str):
    """RequestInvalid with exactly this one-line message (LangGraph adds a note to the exception, which `match` would read too)."""
    with pytest.raises(RequestInvalid) as raised:
        yield
    assert str(raised.value) == message


def turn(**fields):
    """One request: a new question (a fresh message each time: LangGraph gives a message an id, and a repeated id replaces the earlier one) and these fields."""
    return {"messages": [HumanMessage("Can I fly here between 10 and 4 today?")], **fields}


@pytest.mark.parametrize(
    ("fields", "message"),
    [
        ({"lon": -0.086644}, "lat is required"),
        ({"lat": 53.802797}, "lon is required"),
        ({"lat": None, "lon": -0.086644}, "lat is required"),
        ({"lat": 53.802797, "lon": None}, "lon is required"),
        ({"lat": None, "lon": None}, "lat is required"),
        ({**BLUE_MILL, "tz": None}, "unknown time zone None"),
        ({}, "lat is required"),
        ({**BLUE_MILL, "tz": "Mars/Olympus"}, "unknown time zone 'Mars/Olympus'"),
        ({**BLUE_MILL, "tz": "europe/london "}, "unknown time zone 'europe/london '"),
        ({"lat": float("nan"), "lon": 2.35}, "lat must be a finite number"),
    ],
    ids=["no lat", "no lon", "lat None", "lon None", "both None", "tz None", "neither", "unknown tz", "tz with a space", "NaN lat"],
)
def test_a_malformed_request_is_rejected_by_invoke_with_no_model_call(fields, message):
    graph, model = contract_graph()
    config = thread("rejected")
    with rejected(message):
        graph.invoke(turn(**fields), config, context=CASE)
    assert len(model.calls) == 0
    saved = graph.get_state(config).values
    assert saved.get("answer") is None
    assert saved.get("guard") is None
    assert "thread_lz" not in saved
    assert "phone_map" not in saved


@pytest.mark.parametrize(("lat", "lon"), [(48.86, 2.35), (0, 0)], ids=["Paris", "null island"])
def test_outside_the_uk_code_answers_and_the_model_is_never_called(lat, lon):
    graph, model = contract_graph()
    config = thread("abroad")
    output = graph.invoke(turn(lat=lat, lon=lon), config, context=CASE)
    assert output["answer"] == "FlySafeLZ only covers the UK."
    assert output["guard"] == "passed"
    assert output["overlays"] == []
    assert output["assessment"] is None
    assert [type(m) for m in output["messages"]] == [HumanMessage, AIMessage]
    assert output["messages"][-1].content == UK_ONLY_ANSWER
    assert len(model.calls) == 0
    assert graph.get_state(config).values["thread_lz"] == {"lat": lat, "lon": lon, "tz": "Europe/London"}


def test_an_outside_uk_first_turn_ties_the_thread_to_its_lz():
    """R2: a follow-up with no coordinates gets the UK-only answer again, and a moved LZ is refused; the model is never called."""
    graph, model = contract_graph()
    config = thread("abroad-thread")
    graph.invoke(turn(lat=48.86, lon=2.35), config, context=CASE)
    again = graph.invoke(turn(), config, context=CASE)
    assert again["answer"] == UK_ONLY_ANSWER
    with rejected("one landing zone per conversation; start a new thread"):
        graph.invoke(turn(**BLUE_MILL), config, context=CASE)
    with rejected("one landing zone per conversation; start a new thread"):
        graph.invoke(turn(lat=0, lon=0), config, context=CASE)
    assert len(model.calls) == 0


def test_a_valid_blue_mill_request_reaches_the_model_once():
    graph, model = contract_graph()
    output = graph.invoke(turn(**BLUE_MILL), thread("blue-mill"), context=CASE)
    assert len(model.calls) == 1
    assert output["messages"][-1].content == "ok."
    assert output["answer"] is None
    assert output["guard"] is None
    assert output["overlays"] == []
    assert "phone_map" not in output
    assert "thread_lz" not in output


def test_a_valid_request_runs_without_a_context_too():
    graph, model = contract_graph()
    graph.invoke(turn(**BLUE_MILL), thread("no-context"))
    assert len(model.calls) == 1


def test_the_output_never_holds_the_phone_map_even_when_the_thread_does():
    graph, _ = contract_graph()
    config = thread("phones")
    graph.invoke(turn(**BLUE_MILL), config, context=CASE)
    graph.update_state(config, {"phone_map": {"<PHONE_1>": "07700 900123"}})
    output = graph.invoke(turn(), config, context=CASE)
    assert "phone_map" not in output
    assert "07700 900123" not in repr(output)
    assert graph.get_state(config).values["phone_map"] == {"<PHONE_1>": "07700 900123"}


def test_a_rejected_request_leaves_the_phone_map_and_stored_lz_alone():
    graph, model = contract_graph()
    config = thread("phones-rejected")
    graph.invoke(turn(**BLUE_MILL), config, context=CASE)
    graph.update_state(config, {"phone_map": {"<PHONE_1>": "07700 900123"}})
    before = graph.get_state(config).values
    with pytest.raises(RequestInvalid):
        graph.invoke(turn(**{**BLUE_MILL, "tz": "Mars/Olympus"}), config, context=CASE)
    after = graph.get_state(config).values
    assert after["phone_map"] == before["phone_map"]
    assert after["thread_lz"] == before["thread_lz"]
    assert len(model.calls) == 1


def test_a_rejected_question_is_never_shown_to_the_model_or_kept():
    graph, model = contract_graph()
    config = thread("rejected-question")
    graph.invoke(turn(**BLUE_MILL), config, context=CASE)
    moved = HumanMessage("moved, call 07700 900123")
    with rejected("one landing zone per conversation; start a new thread"):
        graph.invoke({"messages": [moved], "lat": 54.0, "lon": -0.1}, config, context=CASE)
    assert len(model.calls) == 1
    # LangGraph saved the input before the hook raised; nothing is answered about it.
    assert "moved, call 07700 900123" in [m.content for m in graph.get_state(config).values["messages"]]

    output = graph.invoke({"messages": [HumanMessage("third")]}, config, context=CASE)
    assert len(model.calls) == 2
    seen = [m.content for m in model.calls[-1]]
    assert seen == ["Can I fly here between 10 and 4 today?", "ok.", "third"]
    assert [m.content for m in output["messages"]] == seen + ["ok."]
    assert [m.content for m in graph.get_state(config).values["messages"]] == seen + ["ok."]
    assert "07700 900123" not in repr(graph.get_state(config).values)


def test_a_rejected_question_is_removed_by_an_outside_uk_answer_too():
    graph, model = contract_graph()
    config = thread("rejected-then-abroad")
    with pytest.raises(RequestInvalid):
        graph.invoke(turn(**BLUE_MILL, tz="Mars/Olympus"), config, context=CASE)
    output = graph.invoke({"messages": [HumanMessage("Paris?")], "lat": 48.86, "lon": 2.35}, config, context=CASE)
    assert [m.content for m in output["messages"]] == ["Paris?", UK_ONLY_ANSWER]
    assert len(model.calls) == 0


def test_each_turn_clears_the_last_turns_answer_guard_overlays_and_assessment():
    graph, _ = contract_graph()
    config = thread("reset")
    graph.invoke(turn(**BLUE_MILL), config, context=CASE)
    circle = lz_radius_feature(**BLUE_MILL).model_dump(mode="json")
    leftover = {"answer": "PILOT DISCRETION from last turn", "guard": "retried", "overlays": [circle], "assessment": None}
    graph.update_state(config, leftover)
    assert graph.get_state(config).values["answer"] == "PILOT DISCRETION from last turn"
    output = graph.invoke(turn(), config, context=CASE)
    assert (output["answer"], output["guard"], output["overlays"], output["assessment"]) == (None, None, [], None)


# --- F4: one LZ per thread -----------------------------------------------------------------------------------------------------


def test_one_landing_zone_per_thread():
    graph, model = contract_graph()
    config = thread("one-lz")

    graph.invoke(turn(**BLUE_MILL), config, context=CASE)  # valid: the LZ is stored
    assert len(model.calls) == 1
    stored = graph.get_state(config).values["thread_lz"]
    assert stored == {**BLUE_MILL, "tz": "Europe/London"}

    with rejected("one landing zone per conversation; start a new thread"):  # moved LZ
        graph.invoke(turn(lat=53.9, lon=-0.086644), config, context=CASE)
    assert len(model.calls) == 1
    assert graph.get_state(config).values["thread_lz"] == stored

    graph.invoke(turn(), config, context=CASE)  # no coordinates: the stored pair is used, even after the refusal
    assert len(model.calls) == 2
    assert graph.get_state(config).values["thread_lz"] == stored

    graph.invoke(turn(**BLUE_MILL), config, context=CASE)  # the same pair again
    assert len(model.calls) == 3

    other = thread("another-thread")  # a new thread takes a new LZ
    graph.invoke(turn(lat=51.5, lon=-0.12), other, context=CASE)
    assert graph.get_state(other).values["thread_lz"] == {"lat": 51.5, "lon": -0.12, "tz": "Europe/London"}
    assert len(model.calls) == 4


def test_a_changed_tz_takes_effect_from_that_turn_and_is_kept():
    graph, model = contract_graph()
    config = thread("tz")
    graph.invoke(turn(**BLUE_MILL, tz="Europe/Paris"), config, context=CASE)
    assert graph.get_state(config).values["thread_lz"]["tz"] == "Europe/Paris"
    graph.invoke(turn(), config, context=CASE)  # omitted: the thread's zone stands
    assert graph.get_state(config).values["thread_lz"]["tz"] == "Europe/Paris"
    graph.invoke(turn(tz="America/New_York"), config, context=CASE)
    assert graph.get_state(config).values["thread_lz"]["tz"] == "America/New_York"
    with pytest.raises(RequestInvalid):
        graph.invoke(turn(tz="Mars/Olympus"), config, context=CASE)
    graph.invoke(turn(), config, context=CASE)  # the refused zone left nothing behind
    assert graph.get_state(config).values["thread_lz"]["tz"] == "America/New_York"
    assert len(model.calls) == 4


def test_a_null_is_refused_on_a_follow_up_but_a_key_left_out_uses_the_stored_value():
    """T1: lat, lon or tz sent as null is a bad value, not an omitted one; leaving the key out keeps the thread's LZ and zone."""
    graph, model = contract_graph()
    config = thread("nulls")
    graph.invoke(turn(**BLUE_MILL, tz="Europe/Paris"), config, context=CASE)
    stored = graph.get_state(config).values["thread_lz"]
    for fields, message in [
        ({"lat": None, "lon": None}, "lat is required"),
        ({"lat": None, "lon": BLUE_MILL["lon"]}, "lat is required"),
        ({"lat": BLUE_MILL["lat"], "lon": None}, "lon is required"),
        ({"tz": None}, "unknown time zone None"),
        ({"lat": None, "lon": None, "tz": None}, "lat is required"),
    ]:
        with rejected(message):
            graph.invoke(turn(**fields), config, context=CASE)
        assert len(model.calls) == 1
        assert graph.get_state(config).values["thread_lz"] == stored
    graph.invoke(turn(), config, context=CASE)  # every key left out: the stored LZ and zone
    assert len(model.calls) == 2
    assert graph.get_state(config).values["thread_lz"] == stored


def test_a_rejected_first_turn_does_not_tie_the_thread_to_its_lz():
    graph, model = contract_graph()
    config = thread("first-rejected")
    with pytest.raises(RequestInvalid):
        graph.invoke(turn(lat=51.5, lon=-0.12, tz="Mars/Olympus"), config, context=CASE)
    graph.invoke(turn(**BLUE_MILL), config, context=CASE)  # a different, valid LZ is accepted: nothing was stored
    assert graph.get_state(config).values["thread_lz"]["lat"] == BLUE_MILL["lat"]
    assert len(model.calls) == 1


def test_a_follow_up_with_no_lz_and_no_stored_one_is_refused():
    graph, model = contract_graph()
    with rejected("lat is required"):
        graph.invoke(turn(), thread("never-set"), context=CASE)
    assert len(model.calls) == 0


# --- the run context ---------------------------------------------------------------------------------------------------------------


def test_an_unknown_case_is_rejected_before_the_model():
    graph, model = contract_graph()
    with rejected("unknown case 'nope_2026': no folder fixtures/nope_2026/"):
        graph.invoke(turn(**BLUE_MILL), thread("case"), context=RunContext(case="nope_2026"))
    assert len(model.calls) == 0


def test_a_naive_now_is_rejected_before_the_model():
    graph, model = contract_graph()
    with rejected("now must be a timezone-aware datetime"):
        graph.invoke(turn(**BLUE_MILL), thread("naive"), context=RunContext(now=datetime(2026, 10, 31, 7, 30)))
    assert len(model.calls) == 0


def test_the_context_from_studio_json_is_read_the_same_way():
    """LangGraph builds the RunContext from a dict; a string `now` becomes a datetime and an unknown reader is refused."""
    graph, model = contract_graph()
    graph.invoke(turn(**BLUE_MILL), thread("studio"), context={"case": "blue_mill_2026-10-31", "now": "2026-10-31T07:30:00Z", "reader": "live", "drop": ["pib"]})
    assert len(model.calls) == 1
    with rejected("unknown reader 'mock'; allowed: fixture, live"):
        graph.invoke(turn(**BLUE_MILL), thread("studio-bad"), context={"case": "blue_mill_2026-10-31", "reader": "mock"})
    assert len(model.calls) == 1


@pytest.mark.parametrize(
    "context, message",
    [
        (RunContext(reader="live"), "reader only applies in case mode; set a case"),
        (RunContext(reader_cache="/tmp/readings.sqlite"), "reader_cache only applies in case mode; set a case"),
        (RunContext(drop=("pib",)), "drop only applies in case mode; set a case"),
        (RunContext(case="blue_mill_2026-10-31", reader_cache="/tmp/readings.sqlite"), "reader_cache needs reader 'live'; with reader 'fixture' it would be ignored"),
    ],
    ids=["reader", "reader_cache", "drop", "reader_cache with fixture"],
)
def test_case_mode_switches_that_would_be_ignored_are_rejected(context, message):
    graph, model = contract_graph()
    with rejected(message):
        graph.invoke(turn(**BLUE_MILL), thread("switches"), context=context)
    assert len(model.calls) == 0


def test_the_uk_edges_come_from_the_rules():
    """The rules' own edge, through the graph: inside is answered by the model, one step past by code."""
    edge_lat = float(RULES.uk_bounds_lat_max.value)  # type: ignore[arg-type]
    graph, model = contract_graph()
    graph.invoke(turn(lat=edge_lat, lon=-1.0), thread("edge-in"), context=CASE)
    assert len(model.calls) == 1
    output = graph.invoke(turn(lat=edge_lat + 1e-9, lon=-1.0), thread("edge-out"), context=CASE)
    assert output["answer"] == UK_ONLY_ANSWER
    assert len(model.calls) == 1


def test_the_circle_builder_is_a_valid_overlay_in_state():
    feature = lz_radius_feature(**BLUE_MILL)
    assert isinstance(feature, OverlayFeature)
    graph, _ = contract_graph()
    config = thread("overlay")
    graph.invoke(turn(**BLUE_MILL), config, context=CASE)
    graph.update_state(config, {"overlays": [feature]})
    assert graph.get_state(config).values["overlays"] == [feature]
