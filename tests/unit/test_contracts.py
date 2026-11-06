"""Contracts round-trip, reject bad input, and display_verdict builds the six strings (F4)."""

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from flysafelz.contracts import (
    Assessment,
    Blocker,
    Caution,
    Circle,
    FlyableWindow,
    HourAssessment,
    IgnoredNotam,
    NotamAction,
    NotamActionResult,
    NotamReading,
    ParsedNotam,
    Segment,
    SourceStatus,
    Tier,
    Verdict,
    WeatherValues,
    display_verdict,
)

T0 = datetime(2026, 10, 31, 8, 0, tzinfo=UTC)
T1 = datetime(2026, 10, 31, 9, 0, tzinfo=UTC)


def h7171_reading() -> dict:
    """H7171/26-shaped: a 0.5 NM circle (no polygon), SFC to 700 ft AGL, phone contact."""
    return {
        "id": "H7171/26",
        "category": "cooperative",
        "activity": "drone operations",
        "geometry": {"kind": "circle", "lat": 53.8, "lon": -0.09, "radius_nm": 0.5},
        "floor": {"value": 0, "unit": "ft", "datum": "SFC"},
        "ceiling": {"value": 700, "unit": "ft", "datum": "AGL"},
        "schedule": "DAILY 0800-1800",
        "exempts_paramotors": False,
        "contact": {"kind": "phone", "value": "07784 546062"},
        "activates_designators": [],
        "prohibitive_reason": "none",
    }


def segment() -> Segment:
    return Segment(
        start=T0,
        end=T1,
        verdict=Verdict.PILOT_DISCRETION,
        approaching_limits=True,
        conditions=["avoid the drone sector to the SSE"],
        cautions=[Caution(rule_id="gust_spread_caution", text="gust spread 5 mph, limit 6", value=5, limit=6)],
        blockers=[],
        notam_actions=[
            NotamActionResult(
                notam_id="H7171/26",
                action=NotamAction.AVOID_SECTOR,
                reason="drone ops 0.15 NM to the SSE",
                bearing_deg=168.6,
            )
        ],
    )


def assessment(**overrides) -> Assessment:
    fields = {
        "window_start": T0,
        "window_end": T1,
        "hours": [HourAssessment(hour=T0, tier=Tier.VERDICT, segments=[segment()], weather=WeatherValues(wind_10m=8))],
        "overall_verdict": Verdict.PILOT_DISCRETION,
        "overall_tier": Tier.VERDICT,
        "flyable_windows": [FlyableWindow(start=T0, end=T1)],
        "source_statuses": [SourceStatus(source="notams", state="ok", issued_at=T0)],
        "assumptions": ["window defaulted"],
        "ignored_notams": [IgnoredNotam(notam_id="J3864/26", reason="checklist")],
    }
    fields.update(overrides)
    return Assessment(**fields)


def test_notam_reading_round_trips():
    reading = NotamReading.model_validate(h7171_reading())
    assert isinstance(reading.geometry, Circle)
    assert reading.geometry.radius_nm == 0.5
    assert reading.contact.value == "07784 546062"
    again = NotamReading.model_validate_json(reading.model_dump_json())
    assert again == reading


def test_assessment_round_trips():
    a = assessment()
    assert Assessment.model_validate_json(a.model_dump_json()) == a


def test_assessment_with_no_verdict_hours_has_no_verdict_and_says_why():
    outlook = HourAssessment(hour=T0, tier=Tier.OUTLOOK, weather=WeatherValues(wind_10m=9, gust_10m=14))
    a = assessment(hours=[outlook], overall_verdict=None, overall_tier=Tier.OUTLOOK, flyable_windows=[])
    assert a.overall_verdict is None
    assert a.overall_tier is Tier.OUTLOOK
    assert Assessment.model_validate_json(a.model_dump_json()) == a
    too_far = assessment(hours=[], overall_verdict=None, overall_tier=Tier.TOO_FAR, flyable_windows=[])
    assert too_far.overall_tier is Tier.TOO_FAR


def test_verdict_and_tier_must_agree():
    with pytest.raises(ValidationError):
        assessment(overall_verdict=None, overall_tier=Tier.VERDICT)
    with pytest.raises(ValidationError):
        assessment(overall_verdict=Verdict.NO_GO, overall_tier=Tier.OUTLOOK)


def test_outlook_is_not_a_verdict():
    with pytest.raises(ValidationError):
        assessment(overall_verdict="OUTLOOK", overall_tier=Tier.OUTLOOK)
    with pytest.raises(ValidationError):
        Segment(start=T0, end=T1, verdict="OUTLOOK")
    assert {v.value for v in Verdict} == {"NO_GO", "UNVERIFIED", "PILOT_DISCRETION"}


@pytest.mark.parametrize(
    "mutate",
    [
        lambda d: d.__setitem__("category", "friendly"),
        lambda d: d.__setitem__("prohibitive_reason", "rude"),
        lambda d: d["contact"].__setitem__("kind", "pigeon"),
        lambda d: d["floor"].__setitem__("unit", "m"),
        lambda d: d["floor"].__setitem__("datum", "MSL"),
        lambda d: d.__setitem__("colour", "red"),
        lambda d: d["geometry"].__setitem__("shape", "blob"),
        lambda d: d.pop("floor"),
    ],
    ids=["category", "prohibitive_reason", "contact kind", "limit unit", "datum", "extra field", "extra geometry field", "missing floor"],
)
def test_notam_reading_rejects_bad_input(mutate):
    data = h7171_reading()
    mutate(data)
    with pytest.raises(ValidationError):
        NotamReading.model_validate(data)


def test_other_contracts_reject_extra_fields_and_naive_times():
    with pytest.raises(ValidationError):
        SourceStatus(source="weather", state="ok", colour="green")
    with pytest.raises(ValidationError):
        SourceStatus(source="weather", state="broken")
    with pytest.raises(ValidationError):
        SourceStatus(source="radar", state="ok")
    with pytest.raises(ValidationError):
        SourceStatus(source="weather", state="ok", fetched_at=datetime(2026, 10, 31, 8, 0))
    with pytest.raises(ValidationError):
        Blocker(source="weather", text="wind", weight=3)


def test_parsed_notam_round_trips():
    parsed = ParsedNotam(
        id="H7171/26",
        series="H",
        number=7171,
        year=26,
        type="N",
        nof="EGGN",
        fir="EGTT",
        q_code="QWULW",
        traffic="IV",
        purpose="NBO",
        scope="W",
        lower_fl=0,
        upper_fl=1,
        centre_lat=53.8,
        centre_lon=-0.09,
        radius_nm=1,
        item_a="EGTT",
        valid_from=T0,
        valid_to=T1,
        item_e="UAS ops",
        section="Area",
    )
    assert ParsedNotam.model_validate_json(parsed.model_dump_json()) == parsed


def test_notam_action_covers_the_six_outcomes_and_the_no_reading_one():
    assert [a.name for a in NotamAction] == [
        "IGNORE",
        "NO_FLY",
        "CONTACT_FIRST",
        "AVOID_SECTOR",
        "AWARENESS",
        "AWARENESS_MAP",
        "UNREAD_OVER_LZ",
    ]
    assert NotamAction.__doc__
    result = NotamActionResult(notam_id="X1/26", action=NotamAction.IGNORE, reason="inactive")
    assert result.bearing_deg is None


DISPLAY_CASES = [
    (Verdict.NO_GO, False, False, "NO-GO"),
    (Verdict.UNVERIFIED, False, False, "UNVERIFIED"),
    (Verdict.PILOT_DISCRETION, True, False, "PILOT DISCRETION \u2013 APPROACHING LIMITS"),
    (Verdict.PILOT_DISCRETION, False, True, "PILOT DISCRETION \u2013 WITH CONDITIONS"),
    (Verdict.PILOT_DISCRETION, True, True, "PILOT DISCRETION \u2013 APPROACHING LIMITS \u00b7 WITH CONDITIONS"),
    (Verdict.PILOT_DISCRETION, False, False, "PILOT DISCRETION"),
]


@pytest.mark.parametrize(("verdict", "limits", "conditions", "expected"), DISPLAY_CASES)
def test_display_verdict(verdict, limits, conditions, expected):
    assert display_verdict(verdict, limits, conditions) == expected


def test_no_qualifier_changes_a_blocking_verdict():
    assert display_verdict(Verdict.NO_GO, True, True) == "NO-GO"
    assert display_verdict(Verdict.UNVERIFIED, True, True) == "UNVERIFIED"


def test_display_verdict_never_returns_a_bare_go():
    import re

    for verdict in Verdict:
        for limits in (False, True):
            for conditions in (False, True):
                text = display_verdict(verdict, limits, conditions)
                assert not re.search(r"(?<!NO-)\bGO\b", text)
