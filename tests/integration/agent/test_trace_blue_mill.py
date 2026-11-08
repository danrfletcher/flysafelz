"""PR-12.F1: a traced Blue Mill turn (GP7, F2, I7) on case `blue_mill_2026-10-31` at the frozen clock 2026-10-31T07:30Z.

The flow is GP3 to GP5: get_notams, then assess, then the answer. Only the chat model is scripted. The LangSmith client is a real `langsmith.Client` with the
flysafelz anonymiser on it, whose HTTP session records what it would have sent (tests/tracing_helpers.py): no socket, and `--disable-socket` stays on.
"""

import json
import re
from pathlib import Path
from typing import Any

import pytest
import yaml
from graph_helpers import GP3_QUESTION, agent_graph, ask, assess_call, blue_mill_context
from langchain_core.messages import AIMessage
from tool_helpers import BLUE_MILL_LZ, BLUE_MILL_WINDOW
from tracing_helpers import PROJECT_ID, model_for, recording_client

from flysafelz import redact
from flysafelz.agent import tracing

ROOT = Path(__file__).resolve().parents[3]
READINGS = ROOT / "fixtures" / "blue_mill_2026-10-31" / "readings.yaml"
OVERALL = "PILOT DISCRETION \u2013 APPROACHING LIMITS \u00b7 WITH CONDITIONS"
ANSWER = f"[path: assess]\n{OVERALL}, flyable from 10:00 GMT to 16:00 GMT. Phone Flylogix <PHONE_1> first (J3857/26), and avoid the sector to the SSE (H7171/26)."
BAD = "[path: assess]\nNO-GO from 10:00 GMT to 16:00 GMT."


def notams_call(call_id: str = "n1") -> AIMessage:
    return AIMessage("", tool_calls=[{"name": "get_notams", "args": dict(BLUE_MILL_WINDOW), "id": call_id}])


def phone_values() -> list[str]:
    """Every phone number the case's readings.yaml gives as a contact, read at run time: the test never writes one down."""
    readings = yaml.safe_load(READINGS.read_text(encoding="utf-8"))
    contacts = [r["contact"]["value"] for r in readings if (r.get("contact") or {}).get("kind") == "phone"]
    numbers = [number for contact in contacts for number in redact.redact_phones(contact)[1].values()]
    assert numbers, "readings.yaml gives no phone contact, so this test would prove nothing"
    return numbers


def forms_of(number: str) -> list[str]:
    """The number as written, and with its spaces and punctuation taken out."""
    return [number, re.sub(r"\D", "", number)]


def leaks(text: str, numbers: list[str]) -> list[str]:
    squeezed = re.sub(r"[\s\-().]", "", text)
    return [form for number in numbers for form in forms_of(number) if form in text or form in squeezed]


@pytest.fixture
def blue_mill():
    """(state, recorder) for GP3 to GP5 on a traced graph."""
    client, rec = recording_client()
    graph = tracing.traced(agent_graph(model_for(notams_call(), assess_call("a1"), ANSWER)), client)
    state = ask(graph)
    tracing.wait_for_feedback()
    return state, rec


def runs_of(rec, run_type: str) -> list[dict[str, Any]]:
    return [run for run in rec.runs().values() if run.get("run_type") == run_type]


# --- GP7 -------------------------------------------------------------------------------------------------------------


def test_gp7_the_trace_has_one_root_run_for_the_turn_and_the_rest_are_its_children(blue_mill):
    _, rec = blue_mill
    (root,) = rec.root_runs()
    assert {run["trace_id"] for run in rec.runs().values()} == {root["id"]}
    assert all(run["parent_run_id"] for run in rec.runs().values() if run["id"] != root["id"])
    assert root["name"] == "LangGraph" and len(runs_of(rec, "tool")) == 2 and len(runs_of(rec, "llm")) == 3


def test_gp7_tool_calls_carry_only_start_end_and_query_arguments(blue_mill):
    _, rec = blue_mill
    allowed = {"start", "end", "query"}
    sent = [call["args"] for run in runs_of(rec, "llm") for generation in (run.get("outputs") or {}).get("generations", [[]])[0] for call in generation["message"]["kwargs"]["tool_calls"]]
    assert len(sent) == 2 and all(set(args) <= allowed for args in sent)
    for run in runs_of(rec, "tool"):
        assert set(run["inputs"]) <= allowed, f"{run['name']} was given {sorted(run['inputs'])}"


def test_gp7_tool_outputs_show_the_placeholder_and_never_the_contact_number(blue_mill):
    _, rec = blue_mill
    outputs = "\n".join(json.dumps(run["outputs"], ensure_ascii=False) for run in runs_of(rec, "tool"))
    assert "<PHONE_1>" in outputs and "J3857/26" in outputs
    assert leaks(outputs, phone_values()) == []
    assert '"<PHONE_1>": "<PHONE>"' in outputs  # the thread's phone_map pairs the placeholder with the number, and the number is masked


def test_gp7_the_guard_result_is_feedback_guard_fail_0_on_the_root_run(blue_mill):
    state, rec = blue_mill
    (root,) = rec.root_runs()
    assert state["guard"] == "passed"
    (body,) = rec.feedback()
    assert (body["run_id"], body["trace_id"], body["key"], body["score"], body.get("comment")) == (root["id"], root["id"], "guard_fail", 0, None)
    assert body["session_id"] == str(PROJECT_ID)
    assert rec.of("POST", "/feedback") and [str(r) for r in rec.sent if r.method == "POST" and r.path == "/feedback"] == ["POST /feedback"]


def test_gp7_the_final_output_has_its_phone_masked_while_the_pilot_has_the_number(blue_mill):
    state, rec = blue_mill
    (root,) = rec.root_runs()
    assert "Phone Flylogix <PHONE> first" in root["outputs"]["answer"]
    assert leaks(json.dumps(root["outputs"], ensure_ascii=False), phone_values()) == []
    number = phone_values()[0]
    assert number in state["answer"] and "<PHONE_1>" not in state["answer"]  # the pilot's view is untouched by the tracing


def test_gp7_the_expected_requests_are_made_and_no_others(blue_mill):
    _, rec = blue_mill
    kinds = {(request.method, "/runs/<id>" if request.path.startswith("/runs/") else request.path) for request in rec.sent}
    assert kinds == {("POST", "/runs"), ("PATCH", "/runs/<id>"), ("GET", "/sessions"), ("POST", "/feedback")}


def test_gp7_the_root_runs_metadata_has_the_source_and_path(blue_mill):
    _, rec = blue_mill
    (root,) = rec.root_runs()
    assert (rec.metadata(root)["source"], rec.metadata(root)["path"], root["outputs"]["path"]) == ("case", "assess", "assess")


# --- coordinates and ids are kept ----------------------------------------------------------------------------------------


def test_coordinates_ids_and_times_come_through_the_trace_unchanged(blue_mill):
    _, rec = blue_mill
    text = rec.text()
    assert "53.802797" in text and "-0.086644" in text  # the LZ inputs in state keep every digit
    assert str(BLUE_MILL_LZ["lat"]) in text
    assert "J3857/26" in text and "H7171/26" in text and "2026-10-31T10:00:00" in text
    root_inputs = json.dumps(rec.root_runs()[0]["inputs"])
    assert "53.802797" in root_inputs and "-0.086644" in root_inputs


def tool_artifact_overlays(rec) -> list[dict[str, Any]]:
    """The overlay features in the tool artifacts of the root run's output messages."""
    (root,) = rec.root_runs()
    return [feature for message in root["outputs"]["messages"] if message["type"] == "tool" for feature in message["artifact"]["overlays"]]


def test_geojson_coordinates_in_the_traced_tool_artifacts_are_still_numbers(blue_mill):
    _, rec = blue_mill
    overlays = tool_artifact_overlays(rec)
    assert len(overlays) >= 3
    for feature in overlays:
        geometry = feature["geometry"]
        points = [geometry["coordinates"]] if geometry["type"] == "Point" else [p for ring in geometry["coordinates"] for p in ring]
        assert points and all(isinstance(x, float) for point in points for x in point)


# --- F2 -------------------------------------------------------------------------------------------------------------------------


def test_f2_none_of_the_phone_values_in_readings_yaml_is_anywhere_in_what_the_client_sent(blue_mill):
    _, rec = blue_mill
    assert len(rec.sent) > 10
    assert leaks(rec.text(), phone_values()) == []
    assert "@" not in rec.text().replace("langsmith.invalid", "")


def test_f2_the_check_does_catch_a_leak(blue_mill):
    """The reader of this test can trust the empty result above: the same search finds the number in the pilot's own answer."""
    state, _ = blue_mill
    assert leaks(state["answer"], phone_values()) != []


# --- I7 ---------------------------------------------------------------------------------------------------------------------------


def test_i7_the_pilot_sees_the_real_number_in_the_assessment_and_the_trace_has_only_the_mask(blue_mill):
    state, rec = blue_mill
    number = phone_values()[0]
    conditions = [c for segment in state["assessment"].segments for c in segment.conditions]
    assert any(number in c for c in conditions)
    for run in rec.runs().values():  # root or child, input or output
        assert leaks(json.dumps(run, ensure_ascii=False, default=str), [number]) == [], run["name"]
    assert "<PHONE>" in json.dumps(rec.root_runs()[0]["outputs"]["assessment"], ensure_ascii=False)


def test_i7_the_raw_notam_text_in_the_traced_overlays_holds_the_placeholder_and_no_number(blue_mill):
    _, rec = blue_mill
    texts = [feature["properties"]["text"] for feature in tool_artifact_overlays(rec) if feature["properties"].get("text")]
    assert texts and any("<PHONE_1>" in text for text in texts)
    assert leaks("\n".join(texts), phone_values()) == []


# --- a second turn ------------------------------------------------------------------------------------------------------------------


def test_a_second_turn_that_fails_the_guard_once_has_its_own_root_run_and_guard_fail_1():
    client, rec = recording_client()
    graph = tracing.traced(agent_graph(model_for(notams_call(), assess_call("a1"), ANSWER, assess_call("a2"), BAD, ANSWER)), client)
    first = ask(graph)
    second = ask(graph, "What about 12 to 3?", context=blue_mill_context())
    tracing.wait_for_feedback()
    assert (first["guard"], second["guard"]) == ("passed", "retried")
    roots = sorted(rec.root_runs(), key=lambda run: run["start_time"])
    assert len(roots) == 2 and roots[0]["id"] != roots[1]["id"]
    feedback = {body["run_id"]: body for body in rec.feedback()}
    assert set(feedback) == {roots[0]["id"], roots[1]["id"]} and len(rec.feedback()) == 2
    assert feedback[roots[0]["id"]]["score"] == 0
    assert feedback[roots[1]["id"]]["score"] == 1 and "not a verdict assess returned" in feedback[roots[1]["id"]]["comment"]
    assert leaks(rec.text(), phone_values()) == []
    assert GP3_QUESTION in json.dumps(roots[0]["inputs"])
