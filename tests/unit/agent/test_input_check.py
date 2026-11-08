"""validate_request, one table: every rule of the input check, each row the input and either the exact RequestInvalid message or the exact update.

The rows run the real function on a plain state dict and a real RunContext. The UK edges are built from RULES, so no bound is written here.
"""

import ast
import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import pytest
from langchain.agents.middleware.types import AgentMiddleware
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from flysafelz.agent import input_check
from flysafelz.agent.input_check import DEFAULT_TZ, UK_ONLY_ANSWER, RequestInvalid, check_input, validate_request
from flysafelz.agent.state import RunContext, ThreadLZ
from flysafelz.rules import RULES, Rule

BLUE_MILL = (53.802797, -0.086644)
CASE = "blue_mill_2026-10-31"
QUESTION = [HumanMessage("Can I fly here today?")]
RESET = {"answer": None, "guard": None, "overlays": [], "assessment": None}


class Rejected:
    """Expected: RequestInvalid with exactly this one-line message."""

    def __init__(self, message: str):
        self.message = message


def accepted(lat: float, lon: float, tz: str = DEFAULT_TZ) -> dict:
    return {**RESET, "thread_lz": ThreadLZ(lat=lat, lon=lon, tz=tz)}


def uk_only(lat: float, lon: float, tz: str = DEFAULT_TZ) -> dict:
    """Code's answer for an LZ outside the UK: the thread is tied to that LZ like to any other."""
    return {"answer": UK_ONLY_ANSWER, "guard": "passed", "overlays": [], "assessment": None, "thread_lz": ThreadLZ(lat=lat, lon=lon, tz=tz), "messages": [AIMessage(UK_ONLY_ANSWER)], "jump_to": "end"}


def request(lat: Any = BLUE_MILL[0], lon: Any = BLUE_MILL[1], tz: Any = ..., messages: Any = None, **more: Any) -> dict:
    """A turn's state. `...` leaves a key out (a follow-up that sends no LZ); None sends the key as null, which is refused."""
    state = {"messages": QUESTION if messages is None else messages, "lat": lat, "lon": lon, "tz": tz, **more}
    return {key: value for key, value in state.items() if value is not ...}


def stored(lat=BLUE_MILL[0], lon=BLUE_MILL[1], tz=DEFAULT_TZ) -> dict:
    return {"thread_lz": ThreadLZ(lat=lat, lon=lon, tz=tz)}


def past(edge: float, direction: float) -> float:
    """The next float beyond `edge`: one step, so the bound is tested to the last bit."""
    return math.nextafter(edge, direction)


def bound(rule: Rule) -> float:
    return float(cast(float, rule.value))


LAT_MIN, LAT_MAX = bound(RULES.uk_bounds_lat_min), bound(RULES.uk_bounds_lat_max)
LON_MIN, LON_MAX = bound(RULES.uk_bounds_lon_min), bound(RULES.uk_bounds_lon_max)
MID_LAT, MID_LON = (LAT_MIN + LAT_MAX) / 2, (LON_MIN + LON_MAX) / 2  # inside the box, on no edge


ROWS = [
    # --- G1 / E1: lat and lon are required -------------------------------------------------------------------------
    ("lat missing", request(lat=...), None, Rejected("lat is required")),
    ("lon missing", request(lon=...), None, Rejected("lon is required")),
    ("lat None", request(lat=None), None, Rejected("lat is required")),
    ("lon None", request(lon=None), None, Rejected("lon is required")),
    ("both missing names lat", request(lat=..., lon=...), None, Rejected("lat is required")),
    ("both None names lat", request(lat=None, lon=None), None, Rejected("lat is required")),
    # --- malformed coordinates are invalid, never 'outside the UK' -------------------------------------------------
    ("lat NaN", request(lat=math.nan), None, Rejected("lat must be a finite number")),
    ("lon NaN", request(lon=math.nan), None, Rejected("lon must be a finite number")),
    ("lat +inf", request(lat=math.inf), None, Rejected("lat must be a finite number")),
    ("lon -inf", request(lon=-math.inf), None, Rejected("lon must be a finite number")),
    ("lat bool", request(lat=True), None, Rejected("lat must be a number")),
    ("lon bool", request(lon=False), None, Rejected("lon must be a number")),
    ("lat numeric string", request(lat="53.8"), None, Rejected("lat must be a number")),
    ("lon numeric string", request(lon="-0.08"), None, Rejected("lon must be a number")),
    ("lat list", request(lat=[53.8]), None, Rejected("lat must be a number")),
    ("lat above 90", request(lat=90.0001), None, Rejected("lat must be between -90 and 90")),
    ("lat below -90", request(lat=-90.0001), None, Rejected("lat must be between -90 and 90")),
    ("lon above 180", request(lon=180.0001), None, Rejected("lon must be between -180 and 180")),
    ("lon below -180", request(lon=-180.0001), None, Rejected("lon must be between -180 and 180")),
    ("integers are numbers", request(lat=54, lon=-1), None, accepted(54.0, -1.0)),
    # --- E2: tz ---------------------------------------------------------------------------------------------------------
    ("tz omitted defaults to London", request(tz=...), None, accepted(*BLUE_MILL, "Europe/London")),
    ("tz None is refused, not read as omitted", request(tz=None), None, Rejected("unknown time zone None")),
    ("tz UTC", request(tz="UTC"), None, accepted(*BLUE_MILL, "UTC")),
    ("tz London", request(tz="Europe/London"), None, accepted(*BLUE_MILL, "Europe/London")),
    ("tz New York (only coverage is UK-only)", request(tz="America/New_York"), None, accepted(*BLUE_MILL, "America/New_York")),
    ("tz unknown", request(tz="Mars/Olympus"), None, Rejected("unknown time zone 'Mars/Olympus'")),
    ("tz empty", request(tz=""), None, Rejected("unknown time zone ''")),
    ("tz local", request(tz="local"), None, Rejected("unknown time zone 'local'")),
    ("tz path", request(tz="../etc/passwd"), None, Rejected("unknown time zone '../etc/passwd'")),
    ("tz whitespace", request(tz="europe/london "), None, Rejected("unknown time zone 'europe/london '")),
    ("tz wrong case", request(tz="europe/london"), None, Rejected("unknown time zone 'europe/london'")),
    ("tz number", request(tz=5), None, Rejected("unknown time zone 5")),
    ("tz list", request(tz=["UTC"]), None, Rejected("unknown time zone ['UTC']")),
    # --- E3: the UK ------------------------------------------------------------------------------------------------------
    ("Paris", request(lat=48.86, lon=2.35), None, uk_only(48.86, 2.35)),
    ("null island", request(lat=0, lon=0), None, uk_only(0.0, 0.0)),
    ("lat min edge is inside", request(lat=LAT_MIN, lon=MID_LON), None, accepted(LAT_MIN, MID_LON)),
    ("lat max edge is inside", request(lat=LAT_MAX, lon=MID_LON), None, accepted(LAT_MAX, MID_LON)),
    ("lon min edge is inside", request(lat=MID_LAT, lon=LON_MIN), None, accepted(MID_LAT, LON_MIN)),
    ("lon max edge is inside", request(lat=MID_LAT, lon=LON_MAX), None, accepted(MID_LAT, LON_MAX)),
    ("one step below lat min", request(lat=past(LAT_MIN, -math.inf), lon=MID_LON), None, uk_only(past(LAT_MIN, -math.inf), MID_LON)),
    ("one step above lat max", request(lat=past(LAT_MAX, math.inf), lon=MID_LON), None, uk_only(past(LAT_MAX, math.inf), MID_LON)),
    ("one step below lon min", request(lat=MID_LAT, lon=past(LON_MIN, -math.inf)), None, uk_only(MID_LAT, past(LON_MIN, -math.inf))),
    ("one step above lon max", request(lat=MID_LAT, lon=past(LON_MAX, math.inf)), None, uk_only(MID_LAT, past(LON_MAX, math.inf))),
    ("a follow-up with no coordinates after an outside-UK first turn gets the UK answer again", request(lat=..., lon=..., **stored(48.86, 2.35)), None, uk_only(48.86, 2.35)),
    ("a follow-up that moves the LZ after an outside-UK first turn is refused", request(**stored(48.86, 2.35)), None, Rejected("one landing zone per conversation; start a new thread")),
    ("a follow-up that moves the LZ from the UK to abroad is refused", request(lat=48.86, lon=2.35, **stored()), None, Rejected("one landing zone per conversation; start a new thread")),
    # --- validation order: malformed input is rejected before the UK check ---------------------------------------------
    ("unknown tz beats the UK answer", request(lat=48.86, lon=2.35, tz="Mars/Olympus"), None, Rejected("unknown time zone 'Mars/Olympus'")),
    ("NaN lat is malformed, not outside the UK", request(lat=math.nan, lon=2.35), None, Rejected("lat must be a finite number")),
    ("missing lon beats the UK answer", request(lat=48.86, lon=None), None, Rejected("lon is required")),
    ("bad coordinate beats bad tz", request(lat="x", tz="Mars/Olympus"), None, Rejected("lat must be a number")),
    ("no question beats everything after it", request(lat=None, messages=[]), None, Rejected("no question: the last message must be the pilot's")),
    # --- the question ---------------------------------------------------------------------------------------------------
    ("no messages", request(messages=[]), None, Rejected("no question: the last message must be the pilot's")),
    ("messages key missing", {"lat": BLUE_MILL[0], "lon": BLUE_MILL[1]}, None, Rejected("no question: the last message must be the pilot's")),
    ("last message from the model", request(messages=[HumanMessage("q"), AIMessage("a")]), None, Rejected("no question: the last message must be the pilot's")),
    ("only a system message", request(messages=[SystemMessage("s")]), None, Rejected("no question: the last message must be the pilot's")),
    ("a follow-up question", request(messages=[HumanMessage("q"), AIMessage("a"), HumanMessage("and tomorrow?")]), None, accepted(*BLUE_MILL)),
    # --- F4: one LZ per thread ---------------------------------------------------------------------------------------------
    ("follow-up without coordinates uses the stored LZ and tz", request(lat=..., lon=..., **stored(tz="Europe/Paris")), None, accepted(*BLUE_MILL, "Europe/Paris")),
    ("follow-up with the same pair", request(**stored()), None, accepted(*BLUE_MILL)),
    ("follow-up with a moved lat", request(lat=54.0, **stored()), None, Rejected("one landing zone per conversation; start a new thread")),
    ("follow-up with a moved lon", request(lon=-0.09, **stored()), None, Rejected("one landing zone per conversation; start a new thread")),
    ("follow-up with lat only", request(lon=..., **stored()), None, Rejected("lon is required")),
    ("follow-up with lon only", request(lat=..., **stored()), None, Rejected("lat is required")),
    ("follow-up with lat and lon both null", request(lat=None, lon=None, **stored()), None, Rejected("lat is required")),
    ("follow-up with lat null and lon left out", request(lat=None, lon=..., **stored()), None, Rejected("lat is required")),
    ("follow-up with lat left out and lon null", request(lat=..., lon=None, **stored()), None, Rejected("lat is required")),
    ("follow-up with lat null and the stored lon", request(lat=None, **stored()), None, Rejected("lat is required")),
    ("follow-up with a null tz is refused, not read as the thread's", request(lat=..., lon=..., tz=None, **stored(tz="Europe/Paris")), None, Rejected("unknown time zone None")),
    ("follow-up with the same pair and a null tz", request(tz=None, **stored()), None, Rejected("unknown time zone None")),
    ("follow-up that leaves tz out keeps the thread's", request(**stored(tz="Europe/Paris")), None, accepted(*BLUE_MILL, "Europe/Paris")),
    ("a follow-up after an outside-UK first turn with both null is refused", request(lat=None, lon=None, **stored(48.86, 2.35)), None, Rejected("lat is required")),
    ("follow-up with a malformed lat", request(lat=math.nan, **stored()), None, Rejected("lat must be a finite number")),
    ("a changed tz takes effect from that turn", request(lat=..., lon=..., tz="America/New_York", **stored()), None, accepted(*BLUE_MILL, "America/New_York")),
    ("a bad tz on a follow-up is refused", request(lat=..., lon=..., tz="Mars/Olympus", **stored()), None, Rejected("unknown time zone 'Mars/Olympus'")),
    # --- run context: now ---------------------------------------------------------------------------------------------------
    ("no context", request(), None, accepted(*BLUE_MILL)),
    ("aware now", request(), RunContext(now=datetime(2026, 10, 31, 7, 30, tzinfo=UTC)), accepted(*BLUE_MILL)),
    ("naive now", request(), RunContext(now=datetime(2026, 10, 31, 7, 30)), Rejected("now must be a timezone-aware datetime")),
    ("now that is not a datetime", request(), RunContext(now="yesterday"), Rejected("now must be a timezone-aware datetime")),
    ("naive now as an ISO string", request(), RunContext(now="2026-10-31T07:30:00"), Rejected("now must be a timezone-aware datetime")),
    ("aware now as an ISO string", request(), RunContext(now="2026-10-31T07:30:00Z"), accepted(*BLUE_MILL)),
    # --- run context: case --------------------------------------------------------------------------------------------------
    ("known case", request(), RunContext(case=CASE), accepted(*BLUE_MILL)),
    ("unknown case", request(), RunContext(case="nope_2026"), Rejected("unknown case 'nope_2026': no folder fixtures/nope_2026/")),
    ("empty case", request(), RunContext(case=""), Rejected("unknown case '': no folder fixtures//")),
    ("case path", request(), RunContext(case="../fixtures"), Rejected("unknown case '../fixtures': no folder fixtures/../fixtures/")),
    ("case that is not a string", request(), RunContext(case=5), Rejected("case must be a string")),
    ("unknown case is refused before the coordinates", request(lat=None), RunContext(case="nope_2026"), Rejected("unknown case 'nope_2026': no folder fixtures/nope_2026/")),
    # --- run context: reader, reader_cache, drop (I3) ----------------------------------------------------------------------
    ("defaults in case mode", request(), RunContext(case=CASE, reader="fixture", reader_cache=None, drop=()), accepted(*BLUE_MILL)),
    ("reader live in case mode", request(), RunContext(case=CASE, reader="live"), accepted(*BLUE_MILL)),
    ("reader live with a cache path", request(), RunContext(case=CASE, reader="live", reader_cache=Path("readings.sqlite")), accepted(*BLUE_MILL)),
    ("a cache path as a string (Studio)", request(), RunContext(case=CASE, reader="live", reader_cache="readings.sqlite"), accepted(*BLUE_MILL)),
    ("drop in case mode", request(), RunContext(case=CASE, drop=("weather", "airspace", "pib")), accepted(*BLUE_MILL)),
    ("drop as a list (Studio)", request(), RunContext(case=CASE, drop=["pib"]), accepted(*BLUE_MILL)),
    ("reader live without a case", request(), RunContext(reader="live"), Rejected("reader only applies in case mode; set a case")),
    ("reader_cache without a case", request(), RunContext(reader_cache=Path("x.sqlite")), Rejected("reader_cache only applies in case mode; set a case")),
    ("drop without a case", request(), RunContext(drop=("pib",)), Rejected("drop only applies in case mode; set a case")),
    ("reader_cache with reader fixture", request(), RunContext(case=CASE, reader_cache=Path("x.sqlite")), Rejected("reader_cache needs reader 'live'; with reader 'fixture' it would be ignored")),
    ("unknown reader", request(), RunContext(case=CASE, reader="mock"), Rejected("unknown reader 'mock'; allowed: fixture, live")),
    ("unknown reader without a case names the values", request(), RunContext(reader="mock"), Rejected("unknown reader 'mock'; allowed: fixture, live")),
    ("unknown drop value", request(), RunContext(case=CASE, drop=("weather", "notams")), Rejected("unknown drop value 'notams'; allowed: weather, airspace, pib")),
    ("drop that is not a list", request(), RunContext(case=CASE, drop="pib"), Rejected("drop must be a list of: weather, airspace, pib")),
]


@pytest.mark.parametrize(("state", "ctx", "expected"), [row[1:] for row in ROWS], ids=[row[0] for row in ROWS])
def test_validate_request(state, ctx, expected):
    if isinstance(expected, Rejected):
        with pytest.raises(RequestInvalid) as raised:
            validate_request(state, ctx)
        assert type(raised.value) is RequestInvalid
        assert str(raised.value) == expected.message
        assert "\n" not in str(raised.value)
    else:
        assert validate_request(state, ctx) == expected


def test_a_rejection_is_a_value_error():
    assert issubclass(RequestInvalid, ValueError)


def test_the_uk_answer_ties_the_thread_to_its_lz_and_adds_one_message():
    update = validate_request(request(lat=48.86, lon=2.35, tz="UTC"), None)
    assert update["thread_lz"] == ThreadLZ(lat=48.86, lon=2.35, tz="UTC")
    assert [m.content for m in update["messages"]] == ["FlySafeLZ only covers the UK."]
    assert "phone_map" not in update


def test_the_questions_rejected_requests_left_behind_are_removed():
    """R1: LangGraph saved them before the hook raised; the next accepted turn takes them out, so the model never sees them."""
    messages = [HumanMessage("first", id="q0"), AIMessage("ok.", id="a0"), HumanMessage("moved, call 07700 900123", id="r0"), HumanMessage("again", id="r1"), HumanMessage("third", id="q1")]
    update = validate_request(request(messages=messages, **stored()), None)
    assert [(type(m).__name__, m.id) for m in update["messages"]] == [("RemoveMessage", "r0"), ("RemoveMessage", "r1")]


def test_the_outside_uk_answer_also_removes_the_rejected_questions():
    messages = [HumanMessage("rejected", id="r0"), HumanMessage("Paris?", id="r1")]
    update = validate_request(request(lat=48.86, lon=2.35, messages=messages), None)
    assert [type(m).__name__ for m in update["messages"]] == ["RemoveMessage", "AIMessage"]
    assert update["messages"][0].id == "r0"


def test_answered_questions_and_other_messages_are_kept():
    messages = [HumanMessage("q0", id="q0"), AIMessage("a0", id="a0"), HumanMessage("q1", id="q1"), AIMessage("a1", id="a1"), HumanMessage("q2", id="q2")]
    assert "messages" not in validate_request(request(messages=messages, **stored()), None)


def test_an_accepted_update_never_touches_phone_map_or_messages():
    update = validate_request(request(), RunContext(case=CASE))
    assert "phone_map" not in update
    assert "messages" not in update


def test_the_state_given_is_not_changed():
    state = request(**stored(tz="UTC"))
    before = {key: (list(value) if isinstance(value, list) else value) for key, value in state.items()}
    validate_request(state, None)
    assert state == before


def test_the_known_zone_set_holds_the_accepted_names_and_no_path():
    assert {"UTC", "Europe/London", "America/New_York"} <= input_check.KNOWN_TIME_ZONES
    assert not {"", "local", "../etc/passwd", "europe/london"} & input_check.KNOWN_TIME_ZONES


def test_the_module_defines_the_input_hook_only():
    """F1/F2: one middleware, no second guard, sub-agent or graph node. PR-9.F1 wires it into the single create_agent."""
    middleware = [name for name, value in vars(input_check).items() if isinstance(value, AgentMiddleware)]
    assert middleware == ["check_input"]
    assert check_input.name == "check_input"
    tree = ast.parse(Path(input_check.__file__).read_text(encoding="utf-8"))
    names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)} | {alias.name for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) for alias in node.names}
    assert not names & {"create_agent", "StateGraph", "wrap_model_call", "after_model", "before_model", "after_agent"}


def test_a_name_the_zone_database_lists_but_cannot_load_is_refused(monkeypatch):
    """The set and ZoneInfo are both checked: a listed name that ZoneInfo cannot read is still an unknown zone, never a fallback to London."""
    monkeypatch.setattr(input_check, "KNOWN_TIME_ZONES", frozenset({"Nowhere/Land"}))
    with pytest.raises(RequestInvalid) as raised:
        validate_request(request(tz="Nowhere/Land"), None)
    assert str(raised.value) == "unknown time zone 'Nowhere/Land'"
