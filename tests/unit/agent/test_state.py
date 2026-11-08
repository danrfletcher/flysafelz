"""The state schema: its key set (F4), who sees each field, the guard values, the overlay and assessment shapes, and the LZ circle."""

import json
from typing import Literal, cast, get_args, get_origin, get_type_hints

import pytest
from agent_helpers import contract_graph
from langchain_core.messages import HumanMessage
from pydantic import ValidationError
from shapely.geometry import shape

from flysafelz.agent import state
from flysafelz.agent.state import (
    Assessment,
    GuardResult,
    OverlayFeature,
    OverlayProperties,
    PointGeometry,
    PolygonGeometry,
    RequestState,
    RunContext,
    Segment,
    ThreadLZ,
    VerdictValue,
    lz_radius_feature,
)
from flysafelz.contracts import Tier, Verdict, display_verdict
from flysafelz.geo import distance_nm
from flysafelz.rules import RULES

# What LangChain's AgentState brings, and what the plan lists. A new key fails test_the_state_keys_are_exactly_the_plans until the plan changes (F4: no user, profile, sites or history).
LANGCHAIN_KEYS = {"messages", "jump_to", "structured_response"}
PLAN_KEYS = {"lat", "lon", "tz", "answer", "guard", "overlays", "assessment", "phone_map", "thread_lz"}
LZ = (53.802797, -0.086644)


def test_the_state_keys_are_exactly_the_plans():
    assert set(get_type_hints(RequestState)) == LANGCHAIN_KEYS | PLAN_KEYS


def test_the_context_keys_are_exactly_the_plans():
    assert set(RunContext.__dataclass_fields__) == {"now", "case", "reader", "reader_cache", "drop"}


def test_the_thread_lz_holds_a_location_and_a_zone_only():
    assert set(get_type_hints(ThreadLZ)) == {"lat", "lon", "tz"}


def test_the_input_schema_holds_the_question_and_the_request_only():
    graph, _ = contract_graph()
    assert set(graph.get_input_jsonschema()["properties"]) == {"messages", "lat", "lon", "tz"}


def test_the_output_schema_holds_the_answer_fields_and_no_request_or_private_field():
    graph, _ = contract_graph()
    properties = set(graph.get_output_jsonschema()["properties"])
    assert {"messages", "answer", "guard", "overlays", "assessment"} <= properties
    assert not properties & {"lat", "lon", "tz", "phone_map", "thread_lz", "jump_to"}


def test_answer_guard_overlays_and_assessment_are_output_only():
    graph, _ = contract_graph()
    sendable = set(graph.get_input_jsonschema()["properties"])
    assert not sendable & {"answer", "guard", "overlays", "assessment"}


def test_phone_map_and_the_thread_lz_are_in_the_checkpoint_but_in_neither_schema():
    """Real phone numbers must never reach the contract: a private field persists with the thread and is in no schema."""
    graph, _ = contract_graph()
    config = {"configurable": {"thread_id": "private"}}
    graph.invoke({"messages": [HumanMessage("q")], "lat": LZ[0], "lon": LZ[1], "tz": "UTC"}, config)
    graph.update_state(config, {"phone_map": {"<PHONE_1>": "07700 900123"}})
    saved = graph.get_state(config).values
    assert saved["phone_map"] == {"<PHONE_1>": "07700 900123"}
    assert saved["thread_lz"]["tz"] == "UTC"
    for schema in (graph.get_input_jsonschema(), graph.get_output_jsonschema()):
        assert not set(schema["properties"]) & {"phone_map", "thread_lz"}


def test_the_request_fields_do_not_outlive_their_turn():
    """lat, lon and tz are one turn's input: after the turn the thread holds none of them (the thread_lz holds the accepted pair)."""
    graph, _ = contract_graph()
    config = {"configurable": {"thread_id": "ephemeral"}}
    graph.invoke({"messages": [HumanMessage("q")], "lat": LZ[0], "lon": LZ[1], "tz": "UTC"}, config)
    saved = graph.get_state(config).values
    assert not {"lat", "lon", "tz"} & set(saved)
    assert saved["thread_lz"] == {"lat": LZ[0], "lon": LZ[1], "tz": "UTC"}


def literal_values(annotation) -> set:
    """Every value of every Literal inside an annotation (through Annotated, NotRequired and unions)."""
    if get_origin(annotation) is Literal:
        return set(get_args(annotation))
    return set().union(*(literal_values(part) for part in get_args(annotation)), set())


def test_guard_takes_exactly_three_values():
    assert set(get_args(GuardResult)) == {"passed", "retried", "fallback"}
    assert literal_values(get_type_hints(RequestState, include_extras=True)["guard"]) == {"passed", "retried", "fallback"}


# --- overlays -----------------------------------------------------------------------------------------------------------------

NOTAM_POLYGON = PolygonGeometry(coordinates=[[[-0.10, 53.80], [-0.05, 53.80], [-0.05, 53.84], [-0.10, 53.84], [-0.10, 53.80]]])


def a_notam() -> OverlayFeature:
    return OverlayFeature(
        geometry=NOTAM_POLYGON,
        properties=OverlayProperties(id="J3857/26", kind="notam", action="CONTACT_FIRST", bearing=157.5, text="E) COOPERATIVE UAS BVLOS OPS"),
    )


def an_airspace_point() -> OverlayFeature:
    return OverlayFeature(geometry=PointGeometry(coordinates=[-0.0866, 53.8028]), properties=OverlayProperties(id="EGD496A", kind="airspace", action="AVOID", bearing=10))


def test_a_notam_overlay_round_trips_to_geojson():
    feature = a_notam()
    geojson = feature.model_dump(mode="json")
    assert geojson == {
        "type": "Feature",
        "geometry": {"type": "Polygon", "coordinates": [[[-0.10, 53.80], [-0.05, 53.80], [-0.05, 53.84], [-0.10, 53.84], [-0.10, 53.80]]]},
        "properties": {"id": "J3857/26", "kind": "notam", "action": "CONTACT_FIRST", "bearing": 157.5, "text": "E) COOPERATIVE UAS BVLOS OPS"},
    }
    assert OverlayFeature.model_validate(json.loads(json.dumps(geojson))) == feature
    assert shape(geojson["geometry"]).is_valid


def test_an_airspace_point_overlay_has_no_text_and_is_a_shapely_point():
    feature = an_airspace_point()
    geojson = feature.model_dump(mode="json")
    assert geojson["properties"]["text"] is None
    assert shape(geojson["geometry"]).geom_type == "Point"
    assert OverlayFeature.model_validate(geojson) == feature


def test_a_feature_dict_picks_the_geometry_by_its_type():
    geojson = {"type": "Feature", "geometry": {"type": "Point", "coordinates": [-0.1, 53.8]}, "properties": {"id": "x", "kind": "airspace"}}
    assert isinstance(OverlayFeature.model_validate(geojson).geometry, PointGeometry)


@pytest.mark.parametrize(
    "bad",
    [
        {"geometry": {"type": "LineString", "coordinates": [[0, 0], [1, 1]]}, "properties": {"id": "x", "kind": "notam"}},
        {"geometry": {"type": "Point", "coordinates": [0.0]}, "properties": {"id": "x", "kind": "notam"}},
        {"geometry": {"type": "Point", "coordinates": [181.0, 0.0]}, "properties": {"id": "x", "kind": "notam"}},
        {"geometry": {"type": "Point", "coordinates": [0.0, 91.0]}, "properties": {"id": "x", "kind": "notam"}},
        {"geometry": {"type": "Point", "coordinates": [0.0, float("nan")]}, "properties": {"id": "x", "kind": "notam"}},
        {"geometry": {"type": "Polygon", "coordinates": []}, "properties": {"id": "x", "kind": "notam"}},
        {"geometry": {"type": "Polygon", "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 1]]]}, "properties": {"id": "x", "kind": "notam"}},
        {"geometry": {"type": "Polygon", "coordinates": [[[0, 0], [1, 0], [0, 0]]]}, "properties": {"id": "x", "kind": "notam"}},
        {"geometry": {"type": "Point", "coordinates": [0.0, 0.0]}, "properties": {"id": "x", "kind": "drone"}},
        {"geometry": {"type": "Point", "coordinates": [0.0, 0.0]}, "properties": {"id": "", "kind": "notam"}},
        {"geometry": {"type": "Point", "coordinates": [0.0, 0.0]}, "properties": {"id": "x", "kind": "notam", "bearing": 360}},
        {"geometry": {"type": "Point", "coordinates": [0.0, 0.0]}, "properties": {"id": "x", "kind": "notam", "bearing": -1}},
        {"geometry": {"type": "Point", "coordinates": [0.0, 0.0]}, "properties": {"id": "x", "kind": "airspace", "text": "E) not a notam"}},
        {"geometry": {"type": "Point", "coordinates": [0.0, 0.0]}, "properties": {"id": "x", "kind": "lz_radius", "text": "E) not a notam"}},
        {"geometry": {"type": "Point", "coordinates": [0.0, 0.0]}, "properties": {"id": "x", "kind": "notam", "colour": "red"}},
    ],
    ids=lambda bad: f"{bad['geometry']['type']}-{bad['properties'].get('kind')}-{sorted(bad['properties'])}-{bad['geometry']['coordinates']}",
)
def test_a_malformed_overlay_is_refused(bad):
    with pytest.raises(ValidationError):
        OverlayFeature.model_validate({"type": "Feature", **bad})


def test_only_a_feature_is_an_overlay():
    with pytest.raises(ValidationError):
        OverlayFeature.model_validate({"type": "FeatureCollection", "geometry": {"type": "Point", "coordinates": [0, 0]}, "properties": {"id": "x", "kind": "notam"}})


# --- the LZ circle --------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("lat, lon", [LZ, (49.9, -6.3), (60.8, -0.9), (51.5, 1.9)], ids=["blue mill", "scilly", "shetland", "kent"])
def test_the_lz_circle_is_centred_on_the_lz_with_the_local_area_radius(lat, lon):
    feature = lz_radius_feature(lat, lon)
    geojson = feature.model_dump(mode="json")
    assert geojson["properties"] == {"id": "lz_radius", "kind": "lz_radius", "action": None, "bearing": None, "text": None}
    ring = geojson["geometry"]["coordinates"][0]
    assert ring[0] == ring[-1]
    radius_nm = float(cast(float, RULES.local_area.value))
    assert RULES.local_area.unit == "NM"
    for lon_v, lat_v in ring:
        assert distance_nm(lat, lon, lat_v, lon_v) == pytest.approx(radius_nm, rel=1e-3)
    polygon = shape(geojson["geometry"])
    assert polygon.is_valid
    centre = polygon.centroid
    assert (centre.y, centre.x) == pytest.approx((lat, lon), abs=1e-4)


def test_the_circle_reaches_the_four_compass_points():
    ring = lz_radius_feature(*LZ).geometry.coordinates[0]
    lons = [p[0] for p in ring]
    lats = [p[1] for p in ring]
    assert max(lats) > LZ[0] > min(lats)
    assert max(lons) > LZ[1] > min(lons)
    assert ring[0][0] == pytest.approx(LZ[1])  # the first point is due north


def test_the_circle_radius_follows_the_rule(monkeypatch):
    """The radius is read from RULES.local_area, not written in the builder."""
    wider = RULES.local_area.model_copy(update={"value": 5})
    monkeypatch.setattr(state, "RULES", RULES.model_copy(update={"local_area": wider}))
    ring = lz_radius_feature(*LZ).geometry.coordinates[0]
    assert distance_nm(*LZ, ring[10][1], ring[10][0]) == pytest.approx(5, rel=1e-3)


# --- the assessment (I7) --------------------------------------------------------------------------------------------------------


def a_segment(**changes) -> dict:
    segment = {
        "start": "2026-10-31T10:00:00+00:00",
        "end": "2026-10-31T11:00:00+00:00",
        "tz_abbr": "GMT",
        "tier": "VERDICT",
        "verdict": "PILOT_DISCRETION",
        "approaching_limits": True,
        "with_conditions": True,
        "display": "PILOT DISCRETION \u2013 APPROACHING LIMITS \u00b7 WITH CONDITIONS",
        "conditions": ["Contact the operator before flying."],
    }
    return {**segment, **changes}


def an_outlook_segment() -> dict:
    return a_segment(tier="OUTLOOK", verdict=None, approaching_limits=False, with_conditions=False, display=None, conditions=[])


def an_assessment() -> dict:
    return {
        "segments": [a_segment(), an_outlook_segment()],
        "overall": {"verdict": "PILOT_DISCRETION", "approaching_limits": True, "with_conditions": True, "display": a_segment()["display"]},
    }


def test_the_assessment_round_trips():
    assessment = Assessment.model_validate(an_assessment())
    dumped = assessment.model_dump(mode="json")
    assert dumped == an_assessment()
    assert Assessment.model_validate(json.loads(json.dumps(dumped))) == assessment
    assert assessment.segments[0].verdict is Verdict.PILOT_DISCRETION
    assert assessment.segments[0].tier is Tier.VERDICT
    assert assessment.segments[1].display is None


def test_the_overall_may_be_none_when_no_hour_has_a_verdict():
    assert Assessment.model_validate({"segments": [an_outlook_segment()], "overall": None}).overall is None


def test_a_display_is_exactly_display_verdict_of_the_machine_fields():
    for verdict in Verdict:
        for approaching in (False, True):
            for conditions in (False, True):
                text = display_verdict(verdict, approaching, conditions)
                Segment.model_validate(a_segment(verdict=verdict.value, approaching_limits=approaching, with_conditions=conditions, display=text))
                VerdictValue.model_validate({"verdict": verdict.value, "approaching_limits": approaching, "with_conditions": conditions, "display": text})


@pytest.mark.parametrize("verdict", ["GO", "MARGINAL", "NO-GO", "pilot_discretion", "", 1])
def test_a_verdict_is_one_of_the_three(verdict):
    with pytest.raises(ValidationError):
        Segment.model_validate(a_segment(verdict=verdict))
    with pytest.raises(ValidationError):
        VerdictValue.model_validate({"verdict": verdict, "approaching_limits": False, "with_conditions": False, "display": "PILOT DISCRETION"})


def test_a_segment_with_no_verdict_must_have_no_display():
    with pytest.raises(ValidationError, match="no verdict has no display"):
        Segment.model_validate(a_segment(tier="OUTLOOK", verdict=None, display="PILOT DISCRETION"))


@pytest.mark.parametrize("display", [None, "PILOT DISCRETION", "NO-GO", "GO"])
def test_a_display_that_is_not_the_exact_string_is_refused(display):
    with pytest.raises(ValidationError):
        Segment.model_validate(a_segment(display=display))


@pytest.mark.parametrize("tier, verdict", [("OUTLOOK", "NO_GO"), ("TOO_FAR", "UNVERIFIED"), ("VERDICT", None)])
def test_only_a_verdict_tier_segment_has_a_verdict(tier, verdict):
    with pytest.raises(ValidationError):
        Segment.model_validate(a_segment(tier=tier, verdict=verdict, display=None if verdict is None else "NO-GO"))


def test_a_too_far_segment_has_no_verdict():
    Segment.model_validate(a_segment(tier="TOO_FAR", verdict=None, approaching_limits=False, with_conditions=False, display=None))


@pytest.mark.parametrize("tier", ["", "outlook", "GO", None])
def test_a_tier_is_ticket_2s(tier):
    with pytest.raises(ValidationError):
        Segment.model_validate(a_segment(tier=tier))


@pytest.mark.parametrize("field", ["start", "end"])
@pytest.mark.parametrize("value", ["2026-10-31T10:00:00", "2026-10-31", "ten o'clock", ""])
def test_a_segment_time_is_iso_with_an_offset(field, value):
    with pytest.raises(ValidationError):
        Segment.model_validate(a_segment(**{field: value}))


def test_a_segment_time_may_carry_any_offset():
    Segment.model_validate(a_segment(start="2026-10-31T11:00:00+01:00", end="2026-10-31T12:00:00+01:00", tz_abbr="CET"))


def test_a_segment_has_no_extra_field():
    with pytest.raises(ValidationError):
        Segment.model_validate(a_segment(colour="amber"))


def test_overlay_text_round_trips_unchanged():
    """OverlayFeature.text: a NOTAM's raw E text comes back unchanged, line breaks and all."""
    text = "E) COOPERATIVE UAS BVLOS OPS\nWITHIN 0.15NM"
    feature = OverlayFeature(geometry=NOTAM_POLYGON, properties=OverlayProperties(id="J3857/26", kind="notam", text=text))
    assert OverlayFeature.model_validate(json.loads(feature.model_dump_json())).properties.text == text
