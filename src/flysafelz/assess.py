"""Turn fetched sources into an Assessment (G15 to G19). Plain functions, no model, no network.

    assess(lat, lon, *, now, forecast, airspace, pib, readings, reader_status, window, live) -> Assessment
    fetch_horizon(now)       the UTC time (exclusive) up to which weather is worth fetching
    default_window(now)      the window used when none is given
    assessed_span(...)       the hour-aligned span assess looks at (the window the NOTAM reader is asked about)
    display(segment)         the verdict display string of a segment
    notams_in_hour(...)      the NOTAMs an hour is cut on
    cut_points_with_reasons(...)  the minutes an hour is cut at, each with why

assess never calls a client. The caller fetches (up to fetch_horizon), hands the results
in, and gets an Assessment back. Every threshold is a rules.yaml rule.

How an hour becomes segments (Q10). Each UTC hour row is cut at civil dawn and dusk and at
the start and end of every NOTAM that applies in it, to the minute. An edge on HH:00 cuts
nothing. A segment is checked as a whole: weather (PR-3), daylight (PR-2), airspace and
danger areas (PR-5) and notam_action() (PR-6).

The tier of an hour (G16): day 0 is today in Europe/London. Beyond the outlook day it is
TOO_FAR and nothing is looked at. Otherwise the weather decides: VERDICT when every required
variable is there (up to the verdict day), else OUTLOOK. OUTLOOK hours keep their raw weather
and have no segments. When the weather source failed, the hours up to the verdict day are
VERDICT (and UNVERIFIED), the rest OUTLOOK.

The verdict of a segment (G17, Q6), no scores, no weights:
    any blocker from weather, daylight, airspace or a NOTAM action       NO_GO
    else a source that is failed or stale, a hole in the weather, or an
      unread NOTAM over the LZ from SFC                                   UNVERIFIED
    else                                                                  PILOT_DISCRETION
approaching_limits is true when the segment has any caution (Q11). Every reason is kept, so
a NO_GO segment still lists its conditions. An UNVERIFIED segment lists why in its conditions.

Danger areas that a NOTAM activates are decided by airspace.py (G10), which owns their
conditions. The activating NOTAM still appears in the segment with its action, but adds no
second condition (a NO_FLY or unread result still counts, so nothing is lost).

Conditions come in a fixed order: CHECK_SUP (read the SUP), then CONTACT_FIRST, then permission and radio,
then ceilings, then sectors, then warnings, and the order inside each group is the order found. A CHECK_SUP
NOTAM adds a condition and no blocker, so a segment with only that finding is PILOT_DISCRETION WITH CONDITIONS.

A NOTAM the reader was asked about and got no reading for is named in `reader_status` ("failed" or "unavailable"). It is
unread, like any NOTAM without a reading (Q8): over the LZ from SFC it is UNREAD_OVER_LZ and the segment UNVERIFIED, anywhere
else it gets its code-only action. An "unavailable" one also adds an assumption line; a "failed" one does not. Real phone
numbers never enter an Assessment: the E text in an unread reason is the redacted one, and the CLI puts the numbers back
when it prints.

The overall verdict is the worst VERDICT-tier segment. Flyable windows are runs of
PILOT_DISCRETION segments with exact edges. Anything else, or a gap, ends a run.

The debug list holds every NOTAM that was dropped or ignored and never shown in any segment,
with its reason.
"""

from collections.abc import Mapping
from datetime import UTC, datetime, timedelta
from typing import Literal, get_args

from flysafelz.airspace import Activation, Outcome, TimeSpan, assess_airspace, find_item_e_contact
from flysafelz.clients.open_meteo import Forecast
from flysafelz.clients.openaip import AirspaceFetch, metres_to_feet
from flysafelz.clients.pib_source import utc_text
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
    SourceState,
    SourceStatus,
    Tier,
    Verdict,
    WeatherValues,
    display_verdict,
)
from flysafelz.daylight import (
    LONDON,
    ONE_DAY,
    ONE_HOUR,
    ONE_MINUTE,
    Phase,
    daylight_edges,
    daylight_phase,
    london_day_offset,
    london_day_utc,
    utc_hours,
)
from flysafelz.designators import normalise_designator
from flysafelz.geo import check_uk_bounds
from flysafelz.notam_action import ActionSegment, LandingZone, active_intervals, notam_action
from flysafelz.pib import PibResult, SortAction, prefilter, sort_notam
from flysafelz.redact import redact_phones
from flysafelz.rules import RULES, Rule
from flysafelz.weather import HourTier, WeatherRow, build_rows, check_row, weather_tier

# Condition groups, in the order a pilot reads them (G17).
CONDITION_ORDER = ("check_sup", "contact", "permission", "ceiling", "sector", "warning")

# Worst first. The index decides which of two verdicts is the worse.
WORST_FIRST = (Verdict.NO_GO, Verdict.UNVERIFIED, Verdict.PILOT_DISCRETION)

# notam_action() words an informational NOTAM over the LZ from the surface like this (Q13).
# It is a warning condition, not plain awareness. A test runs the real notam_action() against it.
INFORMATIONAL_OVER_LZ = "informational over the LZ from the surface"

UNVERIFIED_PREFIX = "UNVERIFIED: "

AIRSPACE_CONDITION_GROUP = {
    Outcome.CHECK_SUP: "check_sup",
    Outcome.CONTACT_FIRST: "contact",
    Outcome.PERMISSION: "permission",
    Outcome.RADIO: "permission",
    Outcome.CEILING: "ceiling",
    Outcome.AVOID_SECTOR: "sector",
    Outcome.WARNING: "warning",
}

Condition = tuple[str, str]  # (group in CONDITION_ORDER, text)

# Why a NOTAM the reader was asked about has no reading: it came back checked-and-wrong twice, or no usable answer came back.
ReaderStatus = Literal["failed", "unavailable"]
UNAVAILABLE_STATUS: ReaderStatus = "unavailable"


def _number(rule: Rule) -> float:
    if isinstance(rule.value, list):
        raise TypeError(f"rule {rule.id} is a list, not a number")
    return float(rule.value)


def _aware(t: datetime, what: str) -> datetime:
    if t.tzinfo is None:
        raise ValueError(f"{what} must be timezone-aware, got naive {t!r}")
    return t.astimezone(UTC)


# --- horizon and window (G16, G19) ----------------------------------------------------


def fetch_horizon(now: datetime) -> datetime:
    """The end (exclusive, UTC) of the last day that is worth fetching: the outlook day in Europe/London.

    Hours from here on are TOO_FAR. The caller stops its fetch here (a client adds its own extra hour).
    """
    last_day = _aware(now, "now").astimezone(LONDON).date() + timedelta(days=_number(RULES.outlook_horizon_day))
    return london_day_utc(last_day)[1]


def default_window(now: datetime) -> tuple[datetime, datetime]:
    """From now rounded down to the hour to the end of tomorrow in Europe/London (as UTC) (G19)."""
    now = _aware(now, "now")
    tomorrow = now.astimezone(LONDON).date() + ONE_DAY
    return now.replace(minute=0, second=0, microsecond=0), london_day_utc(tomorrow)[1]


def _hour_starts(start: datetime, end: datetime, now: datetime, live: bool) -> list[datetime]:
    """The UTC hours [start, end) touches. A live run leaves out the hours that have ended."""
    hours = utc_hours(start, end)
    return [h for h in hours if h + ONE_HOUR > now] if live else hours


def assessed_span(start: datetime, end: datetime, now: datetime, live: bool) -> tuple[datetime, datetime]:
    """The span assess() looks at: from the first hour row to the end of the last (the window itself when there are no rows).

    This is the window the NOTAM reader is asked about, so it reads every NOTAM assess may need a reading for.
    """
    hours = _hour_starts(_aware(start, "window start"), _aware(end, "window end"), _aware(now, "now"), live)
    return (hours[0], hours[-1] + ONE_HOUR) if hours else (start, end)


# --- sources (E1, E2) -----------------------------------------------------------------


def _checked_notam_status(status: SourceStatus, now: datetime) -> SourceStatus:
    """The NOTAM status, marked stale here too when its PIB was issued too long before now (E2)."""
    limit = _number(RULES.pib_stale_hours)
    if status.state is SourceState.OK and status.issued_at is not None and now - status.issued_at > timedelta(hours=limit):
        reason = f"PIB issued {utc_text(status.issued_at)}, more than {limit:g} h before now"
        return status.model_copy(update={"state": SourceState.STALE, "reason": reason})
    return status


def _unavailable(statuses: list[SourceStatus]) -> list[str]:
    """One line for each source that is not ok. These make every segment UNVERIFIED."""
    return [f"{s.source.value} source {s.state.value}" + (f": {s.reason}" if s.reason else "") for s in statuses if s.state is not SourceState.OK]


# --- the findings for one segment ------------------------------------------------------


def _weather_findings(row: WeatherRow | None) -> tuple[list[Blocker], list[Caution]]:
    """Blockers and cautions from the hour's weather (G2, G3). No row, no findings."""
    if row is None:
        return [], []
    found = check_row(row)
    blockers = [Blocker(source="weather", text=f.text) for f in found if f.kind == "blocker"]
    cautions = [Caution(rule_id=f.rule_id, text=f.text, value=f.value, limit=f.limit) for f in found if f.kind == "caution"]
    return blockers, cautions


def _daylight_findings(lat: float, lon: float, start: datetime) -> tuple[list[Blocker], list[Caution]]:
    """Night is a blocker, twilight a caution (G6). The segment never straddles an edge, so its start decides."""
    phase = daylight_phase(lat, lon, start)
    if phase is Phase.NIGHT:
        return [Blocker(source="daylight", text="night: outside civil twilight")], []
    if phase is Phase.TWILIGHT:
        text = "civil twilight: between civil dawn and sunrise, or sunset and civil dusk"
        return [], [Caution(rule_id=RULES.civil_twilight_depression.id, text=text)]
    return [], []


def _airspace_findings(
    areas: AirspaceFetch,
    lat: float,
    lon: float,
    span: TimeSpan,
    elevation_ft: float,
    pressure: float | None,
    activations: list[Activation],
    notes: list[str],
) -> tuple[list[Blocker], list[Condition], list[str]]:
    """Blockers, conditions and UNVERIFIED reasons from airspace and danger areas (G8 to G10)."""
    result = assess_airspace(areas.areas, (lat, lon), span, site_elevation_ft=elevation_ft, pressure_msl=pressure, activations=activations)
    notes += result.notes
    blockers: list[Blocker] = []
    conditions: list[Condition] = []
    unverified: list[str] = []
    for finding in result.findings:
        if finding.outcome in (Outcome.NO_GO, Outcome.NO_FLY):
            blockers.append(Blocker(source="airspace", text=finding.reason))
        elif finding.outcome is Outcome.UNVERIFIED:
            unverified.append(finding.reason)
        elif finding.outcome in AIRSPACE_CONDITION_GROUP and finding.condition:
            conditions.append((AIRSPACE_CONDITION_GROUP[finding.outcome], finding.condition))
    return blockers, conditions, unverified


def _notam_findings(
    results: list[NotamActionResult], owned_by_airspace: set[str]
) -> tuple[list[Blocker], list[Condition], list[str]]:
    """Blockers, conditions and UNVERIFIED reasons from the NOTAM actions (G14).

    A NOTAM that activates a danger area gets its condition from airspace.py, so it adds none here.
    """
    blockers: list[Blocker] = []
    conditions: list[Condition] = []
    unverified: list[str] = []
    for result in results:
        if result.action is NotamAction.NO_FLY:
            blockers.append(Blocker(source="notam", text=f"NOTAM {result.notam_id}: {result.reason}"))
        elif result.action is NotamAction.UNREAD_OVER_LZ:
            unverified.append(result.reason)
        elif result.notam_id in owned_by_airspace:
            continue
        elif result.action is NotamAction.CHECK_SUP:
            conditions.append(("check_sup", result.reason))
        elif result.action is NotamAction.CONTACT_FIRST:
            conditions.append(("contact", result.reason))
        elif result.action is NotamAction.AVOID_SECTOR:
            conditions.append(("sector" if result.bearing_deg is not None else "ceiling", result.reason))
        elif result.action is NotamAction.AWARENESS and INFORMATIONAL_OVER_LZ in result.reason:
            conditions.append(("warning", result.reason))
    return blockers, conditions, unverified


def _notam_results(
    notams: list[ParsedNotam],
    span: TimeSpan,
    lz: LandingZone,
    readings: Mapping[str, NotamReading],
    pressure: float | None,
    ignored: dict[str, str],
) -> list[NotamActionResult]:
    """The non-IGNORE NOTAM actions for one segment (G15). Every IGNORE goes into `ignored` with its reason."""
    results = []
    for notam in notams:
        sorted_ = sort_notam(notam, (span.start, span.end), (lz.lat, lz.lon))
        if sorted_.action is SortAction.IGNORE:
            ignored.setdefault(notam.id, sorted_.reason)
        elif sorted_.action is SortAction.AWARENESS:
            results.append(NotamActionResult(notam_id=notam.id, action=NotamAction.AWARENESS, reason=sorted_.reason))
        else:
            action = notam_action(notam, readings.get(notam.id), lz, ActionSegment(start=span.start, end=span.end, pressure_msl=pressure))
            if action.action is NotamAction.IGNORE:
                ignored.setdefault(notam.id, action.reason)
            else:
                results.append(NotamActionResult(notam_id=action.notam_id, action=action.action, reason=action.reason, bearing_deg=action.bearing_deg))
    return results


def _ordered(conditions: list[Condition]) -> list[str]:
    """Conditions in CONDITION_ORDER (a stable sort), each text once."""
    ranked = sorted(conditions, key=lambda c: CONDITION_ORDER.index(c[0]))
    return list(dict.fromkeys(text for _group, text in ranked))


def _dedupe(items: list[Blocker]) -> list[Blocker]:
    return list({(b.source, b.text): b for b in items}.values())


# --- one hour ---------------------------------------------------------------------------


def _activations(
    notams: list[ParsedNotam], readings: Mapping[str, NotamReading], lz: LandingZone, start: datetime, end: datetime
) -> list[Activation]:
    """The NOTAMs that activate a danger area, one Activation for each designator and active span (G10).

    The designators come from the reading when there is one, else from item E of a restriction NOTAM. The E text kept
    for an unread reason is the redacted one (no real phone number in an Assessment). Whether the NOTAM gives a phone or
    email is found by code in the original item E and kept as a kind only (find_item_e_contact), with the SUP references.
    """
    activations = []
    for notam in notams:
        reading = readings.get(notam.id)
        item_e_contact = find_item_e_contact(notam.item_e)
        sup_references = notam.sup_references
        if reading is not None:
            found = reading.activates_designators
        else:
            sorted_ = sort_notam(notam, (start, end), (lz.lat, lz.lon))
            found = sorted_.designators if sorted_.action is SortAction.VIA_AIRSPACE else []
        designators = [d for d in map(normalise_designator, found) if d]
        circle = Circle(lat=notam.centre_lat, lon=notam.centre_lon, radius_nm=notam.radius_nm)
        for designator in dict.fromkeys(designators):
            for span_start, span_end in active_intervals(notam, reading, lz, start, end):
                activations.append(
                    Activation(
                        designator=designator,
                        notam_id=notam.id,
                        reading=reading,
                        e_text=" ".join(redact_phones(notam.item_e)[0].split()),
                        start=span_start,
                        end=span_end,
                        circle=circle,
                        item_e_contact=item_e_contact,
                        sup_references=sup_references,
                    )
                )
    return activations


# Why a segment ends at a cut: a daylight phase change (before, after), or a NOTAM edge (id, "starts" | "ends").
DaylightReason = tuple[Phase, Phase]
NotamReason = tuple[str, str]
CutReason = DaylightReason | NotamReason

STARTS = "starts"
ENDS = "ends"
_NOTAM_EDGE_ORDER = (STARTS, ENDS)


def notams_in_hour(
    hour: datetime,
    notams: list[ParsedNotam],
    readings: Mapping[str, NotamReading],
    lz: LandingZone,
    ignored: dict[str, str] | None = None,
) -> list[ParsedNotam]:
    """The NOTAMs that matter somewhere in this hour: the ones sort_notam() does not IGNORE, in input order (Q10).

    This is the set assess cuts the hour on. `notams` are the prefiltered ones. Each IGNORE reason goes
    into `ignored` (when given) with setdefault, so the first reason a NOTAM gets across hours is kept.
    `readings` is not read: sort_notam() works from the NOTAM text alone. It is in the signature so
    callers pass the same arguments they pass to cut_points_with_reasons().
    """
    kept = []
    for notam in notams:
        sorted_ = sort_notam(notam, (hour, hour + ONE_HOUR), (lz.lat, lz.lon))
        if sorted_.action is SortAction.IGNORE:
            if ignored is not None:
                ignored.setdefault(notam.id, sorted_.reason)
        else:
            kept.append(notam)
    return kept


def cut_points_with_reasons(
    hour: datetime, lat: float, lon: float, notams: list[ParsedNotam], readings: Mapping[str, NotamReading], lz: LandingZone
) -> list[tuple[datetime, list[CutReason]]]:
    """Every minute strictly inside the hour where a segment ends, with why, in time order (Q10).

    The cuts are daylight edges and NOTAM starts and ends. `notams` are the ones that apply in the hour
    (see notams_in_hour). A daylight reason is (phase before, phase after); a NOTAM reason is
    (id, "starts" | "ends"). When several fall on one minute they share one cut: daylight first,
    then NOTAMs by id (a start before an end). An edge on HH:00 or a NOTAM active across the hour cuts nothing.
    """
    end = hour + ONE_HOUR
    reasons: dict[datetime, list[CutReason]] = {}
    for t, phase_after in daylight_edges(lat, lon, hour, end):
        reasons.setdefault(t, []).append((daylight_phase(lat, lon, t - ONE_MINUTE), phase_after))
    notam_reasons: dict[datetime, set[NotamReason]] = {}
    for notam in notams:
        for span_start, span_end in active_intervals(notam, readings.get(notam.id), lz, hour, end):
            for t, edge in ((span_start, STARTS), (span_end, ENDS)):
                if hour < t < end:
                    notam_reasons.setdefault(t, set()).add((notam.id, edge))
    for t, found in notam_reasons.items():
        reasons.setdefault(t, []).extend(sorted(found, key=lambda r: (r[0], _NOTAM_EDGE_ORDER.index(r[1]))))
    return sorted(reasons.items())


def _segment(
    span: TimeSpan,
    row: WeatherRow | None,
    hour_problems: list[str],
    lat: float,
    lon: float,
    lz: LandingZone,
    airspace: AirspaceFetch,
    activations: list[Activation],
    notams: list[ParsedNotam],
    readings: Mapping[str, NotamReading],
    ignored: dict[str, str],
    notes: list[str],
) -> Segment:
    """One segment: all the checks, every reason kept, then the verdict by the precedence rule."""
    pressure = row.values.pressure_msl if row else None
    weather_blockers, weather_cautions = _weather_findings(row)
    daylight_blockers, daylight_cautions = _daylight_findings(lat, lon, span.start)
    air_blockers, air_conditions, air_unverified = _airspace_findings(airspace, lat, lon, span, lz.elevation_ft, pressure, activations, notes)
    actions = _notam_results(notams, span, lz, readings, pressure, ignored)
    owned = {a.notam_id for a in activations}
    notam_blockers, notam_conditions, notam_unverified = _notam_findings(actions, owned)

    blockers = _dedupe(weather_blockers + daylight_blockers + air_blockers + notam_blockers)
    cautions = weather_cautions + daylight_cautions
    unverified = list(dict.fromkeys(hour_problems + air_unverified + notam_unverified))
    if blockers:
        verdict = Verdict.NO_GO
    elif unverified:
        verdict = Verdict.UNVERIFIED
    else:
        verdict = Verdict.PILOT_DISCRETION
    conditions = _ordered(air_conditions + notam_conditions) + [UNVERIFIED_PREFIX + text for text in unverified]
    return Segment(
        start=span.start,
        end=span.end,
        verdict=verdict,
        approaching_limits=bool(cautions),
        conditions=conditions,
        cautions=cautions,
        blockers=blockers,
        notam_actions=actions,
    )


def _tier(hour: datetime, now: datetime, rows: list[WeatherRow], weather_ok: bool) -> HourTier:
    """The tier of an hour (G16, G5). The weather source decides VERDICT against OUTLOOK."""
    if london_day_offset(hour, now) > _number(RULES.outlook_horizon_day):
        return HourTier(tier=Tier.TOO_FAR)
    if weather_ok:
        return weather_tier(hour, rows, now)
    if london_day_offset(hour, now) > _number(RULES.verdict_horizon_day):
        return HourTier(tier=Tier.OUTLOOK)
    return HourTier(tier=Tier.VERDICT)


# --- the whole window -----------------------------------------------------------------------


def _worst(verdicts: list[Verdict]) -> Verdict | None:
    return min(verdicts, key=WORST_FIRST.index, default=None)


def _flyable_windows(hours: list[HourAssessment]) -> list[FlyableWindow]:
    """Runs of PILOT_DISCRETION segments, edge to edge. A segment that starts where the last one ended joins it."""
    windows: list[FlyableWindow] = []
    for hour in hours:
        for segment in hour.segments:
            if segment.verdict is not Verdict.PILOT_DISCRETION:
                continue
            if windows and windows[-1].end == segment.start:
                windows[-1] = FlyableWindow(start=windows[-1].start, end=segment.end)
            else:
                windows.append(FlyableWindow(start=segment.start, end=segment.end))
    return windows


def _overall(hours: list[HourAssessment]) -> tuple[Verdict | None, Tier]:
    """The worst VERDICT-tier segment; with none, None and the tier that explains it (G18)."""
    verdict_hours = [h for h in hours if h.tier is Tier.VERDICT]
    if verdict_hours:
        return _worst([s.verdict for h in verdict_hours for s in h.segments]), Tier.VERDICT
    if any(h.tier is Tier.OUTLOOK for h in hours):
        return None, Tier.OUTLOOK
    return None, Tier.TOO_FAR


def display(segment: Segment) -> str:
    """The display string of a segment: always one of the six verdict strings."""
    return display_verdict(segment.verdict, segment.approaching_limits, bool(segment.conditions))


def assess(
    lat: float,
    lon: float,
    *,
    now: datetime,
    forecast: Forecast,
    airspace: AirspaceFetch,
    pib: PibResult,
    readings: Mapping[str, NotamReading] | None = None,
    reader_status: Mapping[str, ReaderStatus] | None = None,
    window: tuple[datetime, datetime] | None = None,
    live: bool = False,
) -> Assessment:
    """Assess every UTC hour of the window [start, end) for the LZ at lat, lon.

    forecast, airspace and pib are what the clients returned, failed or not. readings maps a
    NOTAM id to its NotamReading (none given: every NOTAM is unread, Q8). reader_status names
    the NOTAMs the reader was asked about and got no reading for ("failed" or "unavailable"):
    they are unread too, and each "unavailable" one is listed in an assumption. With no window the
    default one is used and noted. live drops hours that ended before now.
    Raises ValueError outside UK coverage, for a naive time, for an empty or reversed window, and
    when a NOTAM is in both readings and reader_status or has a status other than the two above.
    """
    check_uk_bounds(lat, lon)
    now = _aware(now, "now")
    readings = readings or {}
    reader_status = reader_status or {}
    both = sorted(set(readings) & set(reader_status))
    if both:
        raise ValueError(f"NOTAM {', '.join(both)} has a reading and a reader status: it cannot be both read and unread")
    odd = sorted(f"{i} ({status!r})" for i, status in reader_status.items() if status not in get_args(ReaderStatus))
    if odd:
        raise ValueError(f"reader status must be 'failed' or 'unavailable', got {', '.join(odd)}")
    assumptions: list[str] = []
    if window is None:
        start, end = default_window(now)
        assumptions.append(
            f"No window given: assessing from {utc_text(start)} (now rounded down to the hour) to {utc_text(end)} (the end of tomorrow, Europe/London)."
        )
    else:
        start, end = _aware(window[0], "window start"), _aware(window[1], "window end")
    if start >= end:
        raise ValueError(f"the window must start before it ends, got {utc_text(start)} to {utc_text(end)}")

    hour_starts = _hour_starts(start, end, now, live)

    statuses = [forecast.status, airspace.status, _checked_notam_status(pib.status, now)]
    problems = _unavailable(statuses)
    weather_ok = forecast.status.state is SourceState.OK
    rows = build_rows(forecast, hour_starts[0], hour_starts[-1] + ONE_HOUR) if weather_ok and hour_starts else []
    row_of = {row.hour: row for row in rows}
    assumptions += list(dict.fromkeys(note for row in rows for note in row.notes))

    elevation_ft = metres_to_feet(forecast.elevation_m) if forecast.elevation_m is not None else 0.0
    if forecast.elevation_m is None:
        assumptions.append("Site elevation unknown (the forecast gave none): taken as 0 ft.")
    lz = LandingZone(lat=lat, lon=lon, elevation_ft=elevation_ft)

    unavailable = sorted(i for i, status in reader_status.items() if status == UNAVAILABLE_STATUS)
    if unavailable:
        assumptions.append(f"NOTAM reader unavailable for {', '.join(unavailable)}: treated as unread")

    window_span = assessed_span(start, end, now, live)
    near_notams, dropped = prefilter(pib.notams, (lat, lon), window_span)
    activations = _activations(near_notams, readings, lz, *window_span)

    ignored: dict[str, str] = {}
    notes: list[str] = []
    hours = []
    for hour in hour_starts:
        tier = _tier(hour, now, rows, weather_ok)
        row = row_of.get(hour)
        if tier.tier is not Tier.VERDICT:
            hours.append(HourAssessment(hour=hour, tier=tier.tier, weather=row.values if row and tier.tier is Tier.OUTLOOK else WeatherValues()))
            continue

        # The NOTAMs that matter somewhere in this hour, and the minutes where one starts or ends.
        in_hour = notams_in_hour(hour, near_notams, readings, lz, ignored)
        cuts = cut_points_with_reasons(hour, lat, lon, in_hour, readings, lz)
        edges = [hour, *(t for t, _reasons in cuts), hour + ONE_HOUR]

        hour_problems = problems + ([tier.reason] if tier.unverified else [])
        segments = [
            _segment(TimeSpan(start=a, end=b), row, hour_problems, lat, lon, lz, airspace, activations, in_hour, readings, ignored, notes)
            for a, b in zip(edges, edges[1:])
        ]
        hours.append(HourAssessment(hour=hour, tier=Tier.VERDICT, segments=segments, weather=row.values if row else WeatherValues()))

    shown = {a.notam_id for h in hours for s in h.segments for a in s.notam_actions}
    debug = [*dropped, *(IgnoredNotam(notam_id=i, reason=r) for i, r in ignored.items() if i not in shown)]
    debug = [*pib.superseded, *debug]
    overall_verdict, overall_tier = _overall(hours)
    return Assessment(
        window_start=start,
        window_end=end,
        hours=hours,
        overall_verdict=overall_verdict,
        overall_tier=overall_tier,
        flyable_windows=_flyable_windows(hours),
        source_statuses=statuses,
        assumptions=assumptions + list(dict.fromkeys(notes)),
        ignored_notams=debug,
    )
