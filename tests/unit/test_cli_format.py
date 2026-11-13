"""The CLI's formatting functions, fed hand-built Assessment objects. No fixture, no network, no clock."""

import re
from datetime import UTC, datetime, timedelta, timezone

import pytest

from flysafelz import cli
from flysafelz.assess import UNVERIFIED_PREFIX
from flysafelz.contracts import (
    Assessment,
    Blocker,
    Caution,
    FlyableWindow,
    HourAssessment,
    IgnoredNotam,
    NotamAction,
    NotamActionResult,
    Segment,
    SourceName,
    SourceState,
    SourceStatus,
    Tier,
    Verdict,
    WeatherValues,
)

NOW = datetime(2026, 10, 31, 7, 30, tzinfo=UTC)
BARE_GO = re.compile(r"(?<![-\w])GO(?![-\w])")


def at(hour: int, minute: int = 0, day: int = 31, month: int = 10) -> datetime:
    return datetime(2026, month, day, tzinfo=UTC) + timedelta(hours=hour, minutes=minute)  # hour 24 rolls into the next day


def segment(start: datetime, end: datetime, verdict: Verdict = Verdict.PILOT_DISCRETION, **fields) -> Segment:
    return Segment(start=start, end=end, verdict=verdict, **fields)


def hour_of(*segments: Segment) -> HourAssessment:
    first = segments[0].start.replace(minute=0)
    return HourAssessment(hour=first, tier=Tier.VERDICT, segments=list(segments))


def whole_hour(hour: int, verdict: Verdict = Verdict.PILOT_DISCRETION, **fields) -> HourAssessment:
    return hour_of(segment(at(hour), at(hour + 1), verdict, **fields))


def statuses(**changes: SourceStatus) -> list[SourceStatus]:
    ok = {
        "weather": SourceStatus(source=SourceName.WEATHER, state=SourceState.OK, fetched_at=NOW),
        "airspace": SourceStatus(source=SourceName.AIRSPACE, state=SourceState.OK, fetched_at=NOW),
        "notams": SourceStatus(source=SourceName.NOTAMS, state=SourceState.OK, fetched_at=NOW, issued_at=datetime(2026, 10, 31, 7, 0, 37, tzinfo=UTC)),
    }
    ok.update(changes)
    return list(ok.values())


def assessment(*hours: HourAssessment, overall: Verdict | None = None, tier: Tier = Tier.VERDICT, **fields) -> Assessment:
    return Assessment(
        window_start=hours[0].hour,
        window_end=hours[-1].hour + timedelta(hours=1),
        hours=list(hours),
        overall_verdict=overall,
        overall_tier=tier,
        source_statuses=statuses(),
        **fields,
    )


def printed(a: Assessment, **kwargs) -> str:
    return cli.report(a, title="case x", lat=53.8, lon=-0.08, now=NOW, **kwargs)


# --- display strings: one function, six strings ---------------------------------------------


@pytest.mark.parametrize(
    ("verdict", "approaching", "conditions", "expected"),
    [
        (Verdict.NO_GO, False, [], "NO-GO"),
        (Verdict.UNVERIFIED, False, [], "UNVERIFIED"),
        (Verdict.PILOT_DISCRETION, True, [], "PILOT DISCRETION \u2013 APPROACHING LIMITS"),
        (Verdict.PILOT_DISCRETION, False, ["stay below 1,300 ft QNH"], "PILOT DISCRETION \u2013 WITH CONDITIONS"),
        (Verdict.PILOT_DISCRETION, True, ["stay below 1,300 ft QNH"], "PILOT DISCRETION \u2013 APPROACHING LIMITS \u00b7 WITH CONDITIONS"),
        (Verdict.PILOT_DISCRETION, False, [], "PILOT DISCRETION"),
    ],
)
def test_each_of_the_six_display_strings_is_printed_on_its_row(verdict, approaching, conditions, expected):
    row = cli.hour_lines(whole_hour(9, verdict, approaching_limits=approaching, conditions=conditions))[0]
    assert row == f"09:00\u201310:00  {expected}"


@pytest.mark.parametrize("verdict", [Verdict.NO_GO, Verdict.UNVERIFIED])
@pytest.mark.parametrize("approaching", [True, False])
def test_no_go_and_unverified_take_no_qualifiers(verdict, approaching):
    row = cli.hour_lines(whole_hour(9, verdict, approaching_limits=approaching, conditions=["x"]))[0]
    assert row.split("  ", 1)[1] == {Verdict.NO_GO: "NO-GO", Verdict.UNVERIFIED: "UNVERIFIED"}[verdict]


# --- cautions, conditions, blockers and NOTAMs -----------------------------------------------


def test_a_caution_prints_its_numbers_from_the_assessment():
    caution = Caution(rule_id="gust_spread_caution", text="gust spread 5 mph, limit 6", value=5, limit=6)
    lines = cli.hour_lines(whole_hour(9, approaching_limits=True, cautions=[caution]))
    assert lines == ["09:00\u201310:00  PILOT DISCRETION \u2013 APPROACHING LIMITS", "    caution: gust spread 5 mph, limit 6"]


def test_a_caution_without_numbers_prints_its_text():
    caution = Caution(rule_id="civil_twilight_depression", text="civil twilight: between civil dawn and sunrise, or sunset and civil dusk")
    assert "    caution: civil twilight: between civil dawn and sunrise, or sunset and civil dusk" in cli.hour_lines(whole_hour(9, approaching_limits=True, cautions=[caution]))


def test_blockers_come_first_then_conditions_cautions_and_notams_in_the_order_given():
    contact = NotamActionResult(notam_id="J1/26", action=NotamAction.CONTACT_FIRST, reason="ring them first")
    sector = NotamActionResult(notam_id="H2/26", action=NotamAction.AVOID_SECTOR, reason="drone ops 0.15 NM to the SSE", bearing_deg=168.64)
    lines = cli.hour_lines(
        whole_hour(
            9,
            Verdict.NO_GO,
            approaching_limits=True,
            blockers=[Blocker(source="weather", text="visibility 1340 m, limit 1500"), Blocker(source="daylight", text="night")],
            conditions=["phone them", "drone ops 0.15 NM to the SSE"],
            cautions=[Caution(rule_id="r", text="low cloud 100 %, limit 90", value=100, limit=90)],
            notam_actions=[contact, sector],
        )
    )
    assert lines[1:] == [
        "    blocked (weather): visibility 1340 m, limit 1500",
        "    blocked (daylight): night",
        "    condition: phone them",
        "    condition: drone ops 0.15 NM to the SSE",
        "    caution: low cloud 100 %, limit 90",
        "    NOTAM J1/26 CONTACT_FIRST: ring them first",
        "    NOTAM H2/26 AVOID_SECTOR bearing 168.6\u00b0",  # its reason is already a condition, so it is not repeated
    ]


def test_an_unverified_condition_prints_as_unverified_with_its_text():
    text = "TEMPO DANGER AREA (TDA) EGD496A ACTIVATED"
    lines = cli.hour_lines(whole_hour(9, Verdict.UNVERIFIED, conditions=[UNVERIFIED_PREFIX + text]))
    assert lines[1] == f"    unverified: {text}"


# --- sub-rows ---------------------------------------------------------------------------------


def test_an_hour_cut_at_a_minute_edge_prints_a_sub_row_for_each_segment():
    twilight = segment(at(17), at(17, 6), approaching_limits=True, cautions=[Caution(rule_id="t", text="civil twilight")])
    night = segment(at(17, 6), at(18), Verdict.NO_GO, blockers=[Blocker(source="daylight", text="night: outside civil twilight")])
    assert cli.hour_lines(hour_of(twilight, night)) == [
        "17:00\u201318:00",
        "  17:00\u201317:06  PILOT DISCRETION \u2013 APPROACHING LIMITS",
        "      caution: civil twilight",
        "  17:06\u201318:00  NO-GO",
        "      blocked (daylight): night: outside civil twilight",
    ]


def test_an_uncut_hour_is_one_row_with_no_sub_rows():
    lines = cli.hour_lines(whole_hour(9))
    assert lines == ["09:00\u201310:00  PILOT DISCRETION"]


# --- OUTLOOK and TOO_FAR ----------------------------------------------------------------------


def test_an_outlook_hour_prints_outlook_with_wind_gust_and_rain_chance():
    hour = HourAssessment(hour=at(9), tier=Tier.OUTLOOK, weather=WeatherValues(wind_10m=12.34, gust_10m=20.0, precipitation_probability=40.0))
    assert cli.hour_lines(hour) == ["09:00\u201310:00  OUTLOOK  wind 12.3 mph, gust 20.0 mph, rain chance 40.0 %"]


def test_an_outlook_hour_with_a_missing_value_says_so():
    hour = HourAssessment(hour=at(9), tier=Tier.OUTLOOK, weather=WeatherValues(wind_10m=5.0))
    assert cli.hour_lines(hour) == ["09:00\u201310:00  OUTLOOK  wind 5.0 mph, gust n/a, rain chance n/a"]


def test_a_too_far_hour_prints_too_far_out():
    assert cli.hour_lines(HourAssessment(hour=at(9), tier=Tier.TOO_FAR)) == ["09:00\u201310:00  TOO FAR OUT"]


# --- table, date lines and UTC -----------------------------------------------------------------


def test_the_table_is_labelled_utc_and_gets_a_date_line_when_the_day_changes():
    hours = [whole_hour(22), whole_hour(23), HourAssessment(hour=at(0, day=1, month=11), tier=Tier.TOO_FAR)]
    assert cli.table_lines(assessment(*hours, overall=Verdict.PILOT_DISCRETION)) == [
        "2026-10-31 (UTC)",
        "22:00\u201323:00  PILOT DISCRETION",
        "23:00\u201300:00  PILOT DISCRETION",
        "2026-11-01 (UTC)",
        "00:00\u201301:00  TOO FAR OUT",
    ]
    assert "All times UTC" in printed(assessment(whole_hour(9), overall=Verdict.PILOT_DISCRETION))


def test_a_non_utc_time_is_printed_in_utc():
    plus_one = datetime(2026, 10, 31, 10, 0, tzinfo=UTC).astimezone(timezone(timedelta(hours=1)))
    assert plus_one.hour == 11
    assert cli.clock(plus_one) == "10:00"
    assert cli.minute_text(plus_one) == "2026-10-31T10:00Z"


# --- footer ------------------------------------------------------------------------------------


def test_footer_lists_sources_windows_with_exact_edges_and_the_overall_line():
    a = assessment(
        whole_hour(9, approaching_limits=True, conditions=["c"]),
        overall=Verdict.PILOT_DISCRETION,
        flyable_windows=[FlyableWindow(start=at(7), end=at(8)), FlyableWindow(start=at(10), end=at(17, 6))],
        assumptions=["EGD496A: not in the openAIP data"],
    )
    assert cli.footer_lines(a) == [
        "Sources",
        "  weather: ok",
        "  airspace: ok",
        "  NOTAMs: ok (PIB issued 2026-10-31T07:00:37Z)",
        "Flyable windows (UTC)",
        "  2026-10-31T07:00Z to 2026-10-31T08:00Z",
        "  2026-10-31T10:00Z to 2026-10-31T17:06Z",
        "Overall: PILOT DISCRETION \u2013 APPROACHING LIMITS \u00b7 WITH CONDITIONS",
        "Assumptions",
        "  EGD496A: not in the openAIP data",
    ]


def test_footer_with_no_flyable_window_says_none():
    assert "  none" in cli.footer_lines(assessment(whole_hour(9, Verdict.NO_GO), overall=Verdict.NO_GO))


def test_overall_takes_its_qualifiers_from_the_worst_segments_only():
    plain = whole_hour(9)
    careful = whole_hour(10, approaching_limits=True)
    bad = whole_hour(11, Verdict.NO_GO, approaching_limits=True, conditions=["x"])
    assert cli.overall_line(assessment(plain, careful, overall=Verdict.PILOT_DISCRETION)) == "Overall: PILOT DISCRETION \u2013 APPROACHING LIMITS"
    assert cli.overall_line(assessment(plain, careful, bad, overall=Verdict.NO_GO)) == "Overall: NO-GO"


@pytest.mark.parametrize(("tier", "line"), [(Tier.OUTLOOK, "Overall: No verdict: OUTLOOK only"), (Tier.TOO_FAR, "Overall: No verdict: too far out")])
def test_without_a_verdict_the_overall_line_says_why_and_outlook_is_never_a_verdict(tier, line):
    a = assessment(HourAssessment(hour=at(9), tier=tier), overall=None, tier=tier)
    assert cli.overall_line(a) == line
    assert not re.search(r"Overall: OUTLOOK", printed(a))


@pytest.mark.parametrize(
    ("source", "state", "hint"),
    [
        (SourceName.WEATHER, SourceState.FAILED, "check by hand: Open-Meteo (https://open-meteo.com)"),
        (SourceName.AIRSPACE, SourceState.FAILED, "check by hand: openAIP map (https://www.openaip.net)"),
        (SourceName.NOTAMS, SourceState.FAILED, "check by hand: NATS AIS PIB (https://pibs.nats.co.uk)"),
        (SourceName.NOTAMS, SourceState.STALE, "check by hand: NATS AIS PIB (https://pibs.nats.co.uk)"),
    ],
)
def test_a_source_that_is_not_ok_is_named_with_its_reason_and_a_check_by_hand_pointer(source, state, hint):
    status = SourceStatus(source=source, state=state, reason="HTTP 503", fetched_at=NOW)
    a = Assessment(
        window_start=at(9),
        window_end=at(10),
        hours=[whole_hour(9, Verdict.UNVERIFIED)],
        overall_verdict=Verdict.UNVERIFIED,
        overall_tier=Tier.VERDICT,
        source_statuses=[s for s in statuses(**{source.value: status})],
    )
    lines = cli.footer_lines(a)
    label = cli.SOURCE_LABELS[source]
    at_line = lines.index(f"  {label}: {state.value} \u2013 HTTP 503")
    assert lines[at_line + 1] == f"    {hint}"
    assert sum("check by hand" in line for line in lines) == 1  # only the broken one


def test_an_ok_source_has_no_check_by_hand_line():
    assert not [line for line in cli.footer_lines(assessment(whole_hour(9), overall=Verdict.PILOT_DISCRETION)) if "check by hand" in line]


# --- header -------------------------------------------------------------------------------------


def test_the_header_names_the_case_lz_now_pib_commit_and_issue_time():
    notams = SourceStatus(source=SourceName.NOTAMS, state=SourceState.OK, fetched_at=NOW, issued_at=datetime(2026, 10, 31, 7, 0, 37, tzinfo=UTC), sha="a" * 40)
    a = assessment(whole_hour(9), overall=Verdict.PILOT_DISCRETION).model_copy(update={"source_statuses": statuses(notams=notams)})
    assert cli.header_lines(a, "case x", 53.802797, -0.086644, NOW) == [
        "FlySafeLZ assess \u2013 case x",
        "LZ 53.802797, -0.086644 \u00b7 now 2026-10-31T07:30Z",
        f"PIB: archive commit {'a' * 40}, issued 2026-10-31T07:00:37Z",
        "Window 2026-10-31T09:00Z to 2026-10-31T10:00Z (end exclusive). All times UTC.",
    ]


def test_a_live_pib_has_no_commit_and_a_failed_one_says_not_available():
    live = assessment(whole_hour(9), overall=Verdict.PILOT_DISCRETION)
    assert cli.header_lines(live, "live", 53.8, -0.08, NOW)[2] == "PIB: live from NATS, issued 2026-10-31T07:00:37Z"
    failed = SourceStatus(source=SourceName.NOTAMS, state=SourceState.FAILED, reason="HTTP 503", fetched_at=NOW)
    down = live.model_copy(update={"source_statuses": statuses(notams=failed)})
    assert cli.header_lines(down, "live", 53.8, -0.08, NOW)[2] == "PIB: not available"


# --- the debug list ------------------------------------------------------------------------------


def test_the_dropped_notam_list_is_printed_only_with_debug():
    a = assessment(
        whole_hour(9),
        overall=Verdict.PILOT_DISCRETION,
        ignored_notams=[IgnoredNotam(notam_id="J3864/26", reason="starts 2026-11-01T08:00:00Z, at or after the window end")],
    )
    quiet, loud = printed(a), printed(a, debug=True)
    assert "J3864/26" not in quiet and "Dropped" not in quiet
    assert loud.startswith(quiet.rstrip("\n"))
    assert loud.endswith("Dropped or ignored NOTAMs (1)\n  J3864/26: starts 2026-11-01T08:00:00Z, at or after the window end\n")


def test_debug_with_nothing_dropped_says_zero():
    assert printed(assessment(whole_hour(9), overall=Verdict.PILOT_DISCRETION), debug=True).endswith("Dropped or ignored NOTAMs (0)\n")


def test_no_printed_line_has_the_bare_word_go():
    a = assessment(
        whole_hour(8, Verdict.NO_GO, blockers=[Blocker(source="weather", text="x")]),
        whole_hour(9, approaching_limits=True, conditions=["c"]),
        whole_hour(10),
        overall=Verdict.NO_GO,
    )
    assert not BARE_GO.search(printed(a, debug=True))


def test_an_unread_notam_a_danger_area_line_already_names_prints_no_second_copy_of_its_text():
    from flysafelz.contracts import NotamAction, NotamActionResult

    action = NotamActionResult(notam_id="J3307/26", action=NotamAction.UNREAD_OVER_LZ, reason="NOTAM J3307/26 is over the LZ and has not been read: TDA TEXT")
    owned = ["UNVERIFIED: Danger area EGD598B is activated by NOTAM J3307/26, which has not been read: TDA TEXT"]
    assert cli.notam_line(action, owned) == "NOTAM J3307/26 UNREAD_OVER_LZ"
    other = ["UNVERIFIED: Danger area EGD598B is activated by NOTAM J3307/260, which has not been read: TDA TEXT"]  # a longer id is another NOTAM
    assert cli.notam_line(action, other) == f"NOTAM J3307/26 UNREAD_OVER_LZ: {action.reason}"
    assert cli.notam_line(action, []) == f"NOTAM J3307/26 UNREAD_OVER_LZ: {action.reason}"
