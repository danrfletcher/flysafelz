"""G17 / Q6 / Q11: the verdict of one segment, over every combination of what can be wrong with it.

The expected values are worked out from the rule (blocker, then unavailable or unread, then the rest),
not listed by hand. Each case runs the real assess() on a one-hour window.
"""

import itertools
from datetime import timedelta

import pytest
from factories import (
    NOW,
    airspace_fetch,
    circle_at,
    forecast_failed,
    notam,
    pib_result,
    reading,
    run_assess,
    status,
    utc,
    weather_forecast,
)

from flysafelz.assess import display as display_of
from flysafelz.contracts import Category, ContactKind, SourceName, SourceState, Verdict
from flysafelz.rules import RULES

# What a segment can be hit by. "source" is which source is down (or stale).
SOURCES = ["ok", "weather failed", "airspace failed", "notams failed", "notams stale"]
DAY_WINDOW = (utc(9), utc(10))  # full daylight at Blue Mill: no caution from the sun
TWILIGHT_WINDOW = (utc(6), utc(7))  # civil dawn is 06:21 and sunrise 06:59 at Blue Mill on 31 Oct


def the_segment(assessment, hour):
    """The segment of the hour that holds 30 minutes past it (06:30 sits inside twilight)."""
    middle = hour + timedelta(minutes=30)
    found = [s for h in assessment.hours for s in h.segments if s.start <= middle < s.end]
    assert len(found) == 1
    return found[0]


def expected(blocker: bool, source: str, unread: bool, caution: bool, condition: bool) -> tuple[Verdict, bool, str]:
    """Verdict, approaching_limits and display string worked out from the rule."""
    if blocker:
        return Verdict.NO_GO, caution, "NO-GO"
    if source != "ok" or unread:
        return Verdict.UNVERIFIED, caution, "UNVERIFIED"
    parts = (["APPROACHING LIMITS"] if caution else []) + (["WITH CONDITIONS"] if condition else [])
    return Verdict.PILOT_DISCRETION, caution, "PILOT DISCRETION" + (" \u2013 " + " \u00b7 ".join(parts) if parts else "")


def cases() -> list:
    found = []
    for blocker, source, unread, caution, condition in itertools.product([False, True], SOURCES, [False, True], [False, True], [False, True]):
        if source == "notams failed" and (blocker or unread or condition):
            continue  # a failed NOTAM source returns no NOTAMs, so none of these can be present
        found.append(pytest.param(blocker, source, unread, caution, condition, id=f"blocker={blocker}-{source}-unread={unread}-caution={caution}-condition={condition}"))
    return found


def build(blocker: bool, source: str, unread: bool, condition: bool, window):
    notams, readings = [], {}
    if blocker:  # a prohibitive NOTAM over the LZ from the surface: NO_FLY
        notams.append(notam("J1/26", circle=circle_at(0)))
        readings["J1/26"] = reading(notam_id="J1/26", geometry=circle_at(0), category=Category.PROHIBITIVE)
    if unread:  # no reading for it
        notams.append(notam("J2/26", circle=circle_at(0)))
    if condition:  # a cooperative NOTAM with a phone number: CONTACT_FIRST
        notams.append(notam("J3/26", circle=circle_at(0)))
        readings["J3/26"] = reading(notam_id="J3/26", geometry=circle_at(0))
    stale_hours = RULES.pib_stale_hours.value
    assert not isinstance(stale_hours, list)
    stale_issue = NOW - timedelta(hours=stale_hours + 1)
    forecast = forecast_failed() if source == "weather failed" else weather_forecast(window[0] - timedelta(hours=1), window[1] + timedelta(hours=1))
    airspace = airspace_fetch(state=SourceState.FAILED) if source == "airspace failed" else airspace_fetch()
    if source == "notams failed":
        pib = pib_result(state=SourceState.FAILED)
    elif source == "notams stale":
        pib = pib_result(*notams, issued=stale_issue)
    else:
        pib = pib_result(*notams)
    return run_assess(window=window, forecast=forecast, airspace=airspace, pib=pib, readings=readings, stale_rule=True)  # the live PIB-age rule


@pytest.mark.parametrize(("blocker", "source", "unread", "caution", "condition"), cases())
def test_the_verdict_follows_the_precedence_rule(blocker, source, unread, caution, condition):
    window = TWILIGHT_WINDOW if caution else DAY_WINDOW
    assessment = build(blocker, source, unread, condition, window)
    segment = the_segment(assessment, window[0])
    verdict, approaching, display = expected(blocker, source, unread, caution, condition)
    assert segment.verdict is verdict
    assert segment.approaching_limits is approaching
    assert display_of(segment) == display
    assert (segment.verdict is Verdict.NO_GO) == bool(segment.blockers)


def test_twilight_alone_is_a_caution_not_a_blocker():
    segment = the_segment(build(False, "ok", False, False, TWILIGHT_WINDOW), utc(6))
    assert segment.blockers == []
    assert [c.rule_id for c in segment.cautions] == [RULES.civil_twilight_depression.id]


def test_a_no_go_segment_still_lists_the_failed_source_and_its_reason():
    assessment = build(True, "weather failed", False, False, DAY_WINDOW)
    weather = next(s for s in assessment.source_statuses if s.source is SourceName.WEATHER)
    assert weather.state is SourceState.FAILED
    assert weather.reason


def test_an_unverified_segment_says_why_in_its_conditions():
    segment = the_segment(build(False, "airspace failed", False, False, DAY_WINDOW), utc(9))
    assert segment.verdict is Verdict.UNVERIFIED
    assert any(c.startswith("UNVERIFIED: ") and "airspace" in c for c in segment.conditions)


def test_a_stale_status_from_the_client_is_treated_like_a_failed_one():
    pib = pib_result()
    pib = pib.model_copy(update={"status": status(SourceName.NOTAMS, SourceState.STALE, "PIB older than the client allows", issued_at=utc(7))})
    segment = the_segment(run_assess(pib=pib), utc(9))
    assert segment.verdict is Verdict.UNVERIFIED


# --- ticket 3b PR-3: a CHECK_SUP NOTAM is a condition, so it sits in the precedence rule where CONTACT_FIRST does ----------------


def check_sup_pair():
    """A constructed danger area over the LZ from SFC with no contact: CHECK_SUP."""
    circle = circle_at(0)
    return notam("J4/26", circle=circle, subject="RD", item_f="SFC", item_e="DANGER AREA EGD496A. SEE AIP SUP 108/2026."), reading(
        notam_id="J4/26", geometry=circle, contact=ContactKind.NONE, value=None
    )


@pytest.mark.parametrize("blocker", [False, True])
@pytest.mark.parametrize("unread", [False, True])
@pytest.mark.parametrize("caution", [False, True])
def test_a_check_sup_notam_never_changes_the_precedence_rule_it_only_adds_a_condition(blocker, unread, caution):
    window = TWILIGHT_WINDOW if caution else DAY_WINDOW
    notams, readings = [], {}
    if blocker:
        notams.append(notam("J1/26", circle=circle_at(0)))
        readings["J1/26"] = reading(notam_id="J1/26", geometry=circle_at(0), category=Category.PROHIBITIVE)
    if unread:
        notams.append(notam("J2/26", circle=circle_at(0)))
    parsed, read = check_sup_pair()
    notams.append(parsed)
    readings[parsed.id] = read
    segment = the_segment(run_assess(window=window, pib=pib_result(*notams), readings=readings), window[0])
    verdict, approaching, display = expected(blocker, "ok", unread, caution, True)
    assert (segment.verdict, segment.approaching_limits, display_of(segment)) == (verdict, approaching, display)
    assert any("read the SUP / AIP ENR 5.1 entry for EGD496A before flight; details in AIP SUP 108/2026" in c for c in segment.conditions)
