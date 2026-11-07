"""What one NOTAM means for the pilot in one segment (G14), from code alone.

    notam_action(parsed, reading | None, lz, segment) -> ActionResult
    active_intervals(parsed, reading, lz, start, end) -> list of (start, end) UTC spans
    merged_limits(parsed, reading, lz, segment) -> (floor, ceiling) in ft QNH, as the table below uses them

No network, no model, no state. Every reading is written by hand or by ticket 3's
reader. Every number is a rules.yaml rule (local_area, band_low, band_high,
ceiling_buffer, the FL conversion); the only constants here are definitions.

The table, first match wins (a NotamAction each; UNREAD_OVER_LZ is the UNVERIFIED of
the unread path):
    1 IGNORE          not active in the segment, beyond the local area, does not apply to
                      a paramotor (drones only, aerodrome users only, or a quoted
                      PARAMOTOR_CLASS exemption)
    2a NO_FLY         over the LZ, floor SFC, Q-code subject RR, RP or RT (restricted, prohibited,
                      temporary restricted area); the reason names the Q-code
    2b NO_FLY         over the LZ, floor SFC, subject RD with a prohibitive reason (live firing,
                      entry prohibited, byelaws); the reason quotes prohibitive_quote
    2c CONTACT_FIRST  over the LZ, floor SFC, subject RD, reason none, with a phone or email
                      contact (the reason names the AIP SUP when the E text has one)
    2d CHECK_SUP      over the LZ, floor SFC, subject RD, reason none, no phone or email contact:
                      "read the SUP / AIP ENR 5.1 entry[ for <designator>] before flight[; details
                      in AIP SUP n/yyyy]", each bracket only when the code found it in the E text
    2  NO_FLY         over the LZ, floor SFC, any other subject: prohibitive (category, or a
                      prohibitive reason), or cooperative with no phone or email contact (row 3
                      comes first)
    3  CONTACT_FIRST  over the LZ, floor SFC, any other subject: cooperative, with a phone or
                      email contact
    4 AVOID_SECTOR    within the local area but not over the LZ; or over the LZ with a floor
                      up to the low band ("stay below <floor QNH - buffer> ft QNH")
    5 AWARENESS       floor up to the high band (shown in the answer)
    6 AWARENESS_MAP   floor above the high band (map and collapsed list only)

A danger area (RD) over the LZ from SFC is decided by prohibitive_reason and the contact alone,
never by the reader's category: the data can rule a flight out or attach a condition, never rule
it in, so an RD is never IGNORE or AWARENESS and is NO_FLY only on a quoted stay-out reason
(F2). Contact is a phone or an email with a value; a radio frequency is not a contact. A NOTAM
whose Q-code is too short or in lower case has no subject and takes the other-subject rows.

An informational reading over the LZ from SFC (any subject but RD) is a warning, not a restriction (Q13): it
is AWARENESS, shown in the answer, and never NO_FLY. With no reading (Q8), a NOTAM over
the LZ from SFC is UNREAD_OVER_LZ with its E text (phone numbers redacted to <PHONE_n>); any other unread NOTAM gets its
code-only action (AVOID_SECTOR, AWARENESS, AWARENESS_MAP or IGNORE). An unread NOTAM is
never NO_FLY, CONTACT_FIRST or CHECK_SUP.

Geometry: the reading's circle or polygon, else the NOTAM's Q-circle. "Over the LZ" is
inside or exactly on the edge. "Within" is the distance to the edge being at most the
local_area rule. A shape that cannot be used falls back to the Q-circle; with no usable
shape the NOTAM is treated as over the LZ (fail closed) and the reason says so.

Limits (E10): the reading's, else F/G, else the Q-line's lower and upper FL. A reading's
floor or ceiling of None means its E text states none, so that side goes on to F/G and then
the Q-line. When the reading and F/G both give a side and differ, the lower floor and the
higher ceiling win and the reason says so. The E text is never read for limits. Floors are converted to
feet QNH (AGL with the site elevation, FL with the segment's pressure_msl) and a floor at
or below the site elevation is SFC. Bands are floor <= band_low, then <= band_high: a
floor exactly on an edge belongs to the lower band, as in airspace.py (the band rules
have the plain setting comparison "=", so there is no ordering operator to apply).

Schedule (E7): the D item (the reading's normalised text when it has one) through
schedule.py, with SR and SS resolved at the LZ. A segment the schedule covers even in
part is active; a schedule that cannot be read is active for all of B-C and says
"schedule not parsed".
"""

import math
import re
from dataclasses import dataclass
from datetime import datetime

from pydantic import BaseModel, Field

from flysafelz.airspace import TimeSpan, limit_qnh_ft
from flysafelz.contracts import (
    STRICT,
    AppliesTo,
    Category,
    Circle,
    ContactKind,
    Datum,
    Exemption,
    ExemptionCovers,
    Limit,
    LimitUnit,
    NotamAction,
    NotamActionResult,
    NotamReading,
    ParsedNotam,
    Polygon,
    ProhibitiveReason,
    Segment,
)
from flysafelz.designators import normalise_designator
from flysafelz.geo import (
    GeometryError,
    circle_edge_bearing_deg,
    circle_edge_distance_nm,
    circle_over,
    compass16,
    shape_edge_bearing_deg,
    shape_edge_distance_nm,
    shape_from_points,
    shape_over,
)
from flysafelz.redact import redact_phones
from flysafelz.rules import RULES, Rule, Rules
from flysafelz.schedule import Schedule, parse_schedule
from flysafelz.sup_refs import sup_detail

UNLIMITED_FL = 999  # a Q-line upper limit of FL999 means no upper limit
SFC_LIMIT = Limit(value=0, unit=LimitUnit.FT, datum=Datum.SFC)
DANGER_AREA = "RD"  # Q-code subject: a danger area, which the data never rules in
RESTRICTED_SUBJECTS = {"RR": "restricted area", "RP": "prohibited area", "RT": "temporary restricted area"}  # row 2a
_LIMIT_TEXT = re.compile(
    r"(?P<sfc>SFC|GND)|(?P<unl>UNL)|FL\s?(?P<fl>\d+)|(?P<ft>\d+)\s?FT\s?(?P<datum>AGL|AMSL)",
    re.IGNORECASE,
)


class LandingZone(BaseModel):
    """The LZ: where it is and how high the ground is (AGL floors are worked out from elevation_ft)."""

    model_config = STRICT

    lat: float
    lon: float
    elevation_ft: float


class ActionSegment(TimeSpan):
    """A span with the hour's forecast pressure (hPa), which flight levels are converted with.

    contracts.Segment can be passed in its place; it carries no pressure, so flight levels
    are then taken at the reference pressure.
    """

    pressure_msl: float | None = None


class ActionResult(NotamActionResult):
    """The action and its one-line reason. debug holds the notes that belong in the debug list only."""

    debug: list[str] = Field(default_factory=list)


# --- limits -------------------------------------------------------------------


@dataclass(frozen=True)
class _Bound:
    """A floor or ceiling: as published, and in feet QNH (infinite for no upper limit)."""

    text: str
    qnh: float


def _show(limit: Limit) -> str:
    if limit.datum is Datum.SFC:
        return "SFC"
    if limit.unit is LimitUnit.FL:
        return f"FL{limit.value:03.0f}"
    return f"{limit.value:,.0f} ft {limit.datum.value}"


def _bound(limit: Limit, lz: LandingZone, pressure: float, rules: Rules) -> _Bound:
    return _Bound(_show(limit), limit_qnh_ft(limit, lz.elevation_ft, pressure, rules))


def _unlimited() -> _Bound:
    return _Bound("no upper limit", math.inf)


def _parse_fg(text: str | None, lz: LandingZone, pressure: float, rules: Rules) -> _Bound | None:
    """Item F or G ("SFC", "700FT AGL", "800FT AMSL", "FL065", "UNL") as a bound; None when absent or unreadable."""
    match = _LIMIT_TEXT.fullmatch((text or "").strip())
    if match is None:
        return None
    if match["unl"]:
        return _unlimited()
    if match["sfc"]:
        return _bound(SFC_LIMIT, lz, pressure, rules)
    if match["fl"]:
        return _bound(Limit(value=int(match["fl"]), unit=LimitUnit.FL, datum=Datum.AMSL), lz, pressure, rules)
    return _bound(Limit(value=int(match["ft"]), unit=LimitUnit.FT, datum=Datum(match["datum"].upper())), lz, pressure, rules)


def _q_floor(parsed: ParsedNotam, lz: LandingZone, pressure: float, rules: Rules) -> _Bound:
    if not parsed.lower_fl:
        return _bound(SFC_LIMIT, lz, pressure, rules)
    return _bound(Limit(value=parsed.lower_fl, unit=LimitUnit.FL, datum=Datum.AMSL), lz, pressure, rules)


def _q_ceiling(parsed: ParsedNotam, lz: LandingZone, pressure: float, rules: Rules) -> _Bound:
    if parsed.upper_fl >= UNLIMITED_FL:
        return _unlimited()
    return _bound(Limit(value=parsed.upper_fl, unit=LimitUnit.FL, datum=Datum.AMSL), lz, pressure, rules)


def _merge(side: str, reading: _Bound | None, fg: _Bound | None, q_line: _Bound, notes: list[str]) -> _Bound:
    """One side of E10: the reading's, else F/G, else the Q-line. On a conflict the more restrictive wins."""
    if reading is not None and fg is not None:
        if math.isclose(reading.qnh, fg.qnh, abs_tol=1.0):
            return reading
        notes.append(f"reading and F/G disagree on the {side} ({reading.text} against {fg.text}); the more restrictive is used")
        return min(reading, fg, key=lambda b: b.qnh) if side == "floor" else max(reading, fg, key=lambda b: b.qnh)
    return reading or fg or q_line


def _limits(parsed: ParsedNotam, reading: NotamReading | None, lz: LandingZone, pressure: float, rules: Rules, notes: list[str]) -> tuple[_Bound, _Bound]:
    floor = _merge(
        "floor",
        _bound(reading.floor, lz, pressure, rules) if reading and reading.floor is not None else None,
        _parse_fg(parsed.item_f, lz, pressure, rules),
        _q_floor(parsed, lz, pressure, rules),
        notes,
    )
    ceiling = _merge(
        "ceiling",
        _bound(reading.ceiling, lz, pressure, rules) if reading and reading.ceiling is not None else None,
        _parse_fg(parsed.item_g, lz, pressure, rules),
        _q_ceiling(parsed, lz, pressure, rules),
        notes,
    )
    return floor, ceiling


def _pressure(segment: Segment | ActionSegment, rules: Rules) -> float:
    """The segment's forecast pressure, else the reference pressure."""
    pressure = segment.pressure_msl if isinstance(segment, ActionSegment) else None
    return _number(rules.fl_reference_hpa) if pressure is None else pressure


def merged_limits(parsed: ParsedNotam, reading: NotamReading | None, lz: LandingZone, segment: Segment | ActionSegment, *, rules: Rules | None = None) -> tuple[float, float]:
    """The floor and ceiling notam_action() works with (the E10 merge), in feet QNH. A ceiling of no upper limit is infinity.

    A wrapper over the same merge, so the reader eval compares heights as numbers (R3.12) without a second copy of the rule.
    """
    rules = RULES if rules is None else rules
    floor, ceiling = _limits(parsed, reading, lz, _pressure(segment, rules), rules, [])
    return floor.qnh, ceiling.qnh


# --- geometry -----------------------------------------------------------------


@dataclass(frozen=True)
class _Where:
    """Where the NOTAM lies against the LZ. usable is False when no shape could be used (then over is True)."""

    over: bool
    distance_nm: float  # to the edge; 0 when over
    bearing_deg: float | None  # to the nearest edge point; None when over
    edge: str  # "circle edge", "polygon edge", "Q-circle edge"
    usable: bool = True


def _circle_where(lz: LandingZone, circle: Circle, edge: str) -> _Where | None:
    numbers = (circle.lat, circle.lon, circle.radius_nm)
    if not all(math.isfinite(x) for x in numbers) or abs(math.radians(circle.lat)) > math.pi * 0.5 or circle.radius_nm <= 0:
        return None
    if circle_over(lz.lat, lz.lon, circle.lat, circle.lon, circle.radius_nm):
        return _Where(True, 0.0, None, edge)
    distance = circle_edge_distance_nm(lz.lat, lz.lon, circle.lat, circle.lon, circle.radius_nm)
    return _Where(False, distance, circle_edge_bearing_deg(lz.lat, lz.lon, circle.lat, circle.lon, circle.radius_nm), edge)


def _polygon_where(lz: LandingZone, polygon: Polygon) -> _Where | None:
    try:
        shape = shape_from_points(polygon.points)
    except GeometryError:
        return None
    if shape_over(lz.lat, lz.lon, shape):
        return _Where(True, 0.0, None, "polygon edge")
    return _Where(False, shape_edge_distance_nm(lz.lat, lz.lon, shape), shape_edge_bearing_deg(lz.lat, lz.lon, shape), "polygon edge")


def _where(parsed: ParsedNotam, reading: NotamReading | None, lz: LandingZone, debug: list[str]) -> _Where:
    """The reading's shape, else the Q-circle, else (fail closed) over the LZ."""
    geometry = reading.geometry if reading else None
    if isinstance(geometry, Circle):
        found = _circle_where(lz, geometry, "circle edge")
    elif isinstance(geometry, Polygon):
        found = _polygon_where(lz, geometry)
    else:
        found = None
    if found is not None:
        return found
    if geometry is not None:
        debug.append("reading geometry not usable; the Q-circle is used")
    q_circle = _circle_where(lz, Circle(lat=parsed.centre_lat, lon=parsed.centre_lon, radius_nm=parsed.radius_nm), "Q-circle edge")
    if q_circle is not None:
        return q_circle
    return _Where(True, 0.0, None, "", usable=False)


# --- schedule -----------------------------------------------------------------


def _schedule(parsed: ParsedNotam, reading: NotamReading | None, lz: LandingZone, start: datetime, end: datetime) -> Schedule:
    d_text = reading.schedule if reading and reading.schedule.strip() else parsed.item_d
    b = max(parsed.valid_from, start)
    c = end if parsed.valid_to is None else min(parsed.valid_to, end)
    return parse_schedule(d_text, b, c, lz.lat, lz.lon)


def active_intervals(parsed: ParsedNotam, reading: NotamReading | None, lz: LandingZone, start: datetime, end: datetime) -> list[tuple[datetime, datetime]]:
    """The UTC spans inside start-end (and inside B-C) in which the NOTAM is active.

    The D item decides, with SR and SS resolved at the LZ; with no D item the NOTAM is
    active all through B-C. A D item that cannot be read gives all of B-C (fail closed).
    PR-8 splits an hour at the edges of these spans.
    """
    return list(_schedule(parsed, reading, lz, start, end).intervals)


# --- the table ----------------------------------------------------------------


def _number(rule: Rule) -> float:
    if isinstance(rule.value, list):
        raise TypeError(f"rule {rule.id} is a list, not a number")
    return float(rule.value)


def _feet(x: float) -> str:
    return f"{math.floor(x):,}"


def _squash(text: str) -> str:
    return " ".join(text.split()).casefold()


def _has_contact(reading: NotamReading) -> bool:
    """A phone or an email. A radio frequency is never a contact (F11)."""
    contact = reading.contact
    return contact.kind in (ContactKind.PHONE, ContactKind.EMAIL) and bool((contact.value or "").strip())


def _prohibition(reading: NotamReading) -> str | None:
    """Why the reading is prohibitive, or None. A prohibitive reason wins over the category."""
    if reading.prohibitive_reason is not ProhibitiveReason.NONE:
        return f"prohibitive ({reading.prohibitive_reason.value})"
    if reading.category is Category.PROHIBITIVE:
        return "prohibitive"
    return None


def _checked(parsed: ParsedNotam, reading: NotamReading, debug: list[str]) -> list[Exemption]:
    """The exemptions whose quote is really in the redacted E text. An invented quote is dropped and logged (Q2).

    The quote is looked for in `redact_phones(parsed.item_e)`: the text the reader and the labelling sheet saw, so a quote that
    holds a `<PHONE_n>` is kept and a real number in a quote no longer matches.
    """
    e_text = _squash(redact_phones(parsed.item_e)[0])
    kept = []
    for exemption in reading.exemptions:
        if exemption.covers is ExemptionCovers.OTHER:
            continue
        quote = _squash(exemption.quote)
        if quote and quote in e_text:
            kept.append(exemption)
        else:
            debug.append(f'exemption quote not found: "{exemption.quote}"')
    return kept


def _not_for_paramotors(parsed: ParsedNotam, reading: NotamReading, exemptions: list[Exemption]) -> str | None:
    """Why the NOTAM does not apply to a paramotor, or None. Wins over geometry, floor and category (E11)."""
    if reading.applies_to is AppliesTo.UAS_ONLY:
        return "applies to drones only"
    if reading.applies_to is AppliesTo.AERODROME_USERS:
        return f"applies only to aircraft using {parsed.item_a}"
    for exemption in exemptions:
        if exemption.covers is ExemptionCovers.PARAMOTOR_CLASS:
            return f'exempts paramotors: "{exemption.quote.strip()}"'
    return None


def _result(parsed: ParsedNotam, action: NotamAction, reason: str, debug: list[str], extras: list[str] | None = None, bearing: float | None = None) -> ActionResult:
    line = "; ".join([" ".join(reason.split()), *(extras or [])])
    return ActionResult(notam_id=parsed.id, action=action, reason=line, bearing_deg=bearing, debug=list(debug))


def _sup_tail(parsed: ParsedNotam) -> str:
    """"; details in AIP SUP 108/2026" for the SUP references in the E text, else nothing."""
    detail = sup_detail(parsed.sup_references)
    return f"; {detail}" if detail else ""


def check_sup_text(parsed: ParsedNotam) -> str:
    """FR5: "read the SUP / AIP ENR 5.1 entry[ for <designator>] before flight[; details in AIP SUP n/yyyy]".

    A bracketed part is there only when the code found it in the E text (every designator and every SUP, in order of appearance).
    """
    where = f" for {', '.join(parsed.designators)}" if parsed.designators else ""
    return f"read the SUP / AIP ENR 5.1 entry{where} before flight{_sup_tail(parsed)}"


def _danger_area(parsed: ParsedNotam, reading: NotamReading, here: str, debug: list[str], extras: list[str]) -> ActionResult:
    """Rows 2b to 2d for a danger area (QRD) over the LZ from the surface. The reader's category is ignored (CR2)."""
    if reading.prohibitive_reason is not ProhibitiveReason.NONE:
        quote = " ".join(redact_phones(reading.prohibitive_quote or "")[0].split())
        said = f': "{quote}"' if quote else ""
        return _result(parsed, NotamAction.NO_FLY, f"{here}: prohibitive ({reading.prohibitive_reason.value}){said}", debug, extras)
    if _has_contact(reading):
        contact = f"{reading.contact.kind.value} {(reading.contact.value or '').strip()}"
        return _result(parsed, NotamAction.CONTACT_FIRST, f"{here}: contact {contact} first and fly only if they agree{_sup_tail(parsed)}", debug, extras)
    # The FR5 text comes first and the fail-closed notes (schedule not parsed, no usable geometry, a ceiling disagreement, a
    # conditional exemption's quote) follow it, as on the other two rows. Only "activates <designator>" is left out of the reason:
    # the FR5 text already names the designator the code found in the E text. It stays in the debug list.
    shown = [note for note in extras if not note.startswith("activates ")]
    hidden = [note for note in extras if note.startswith("activates ")]
    return _result(parsed, NotamAction.CHECK_SUP, check_sup_text(parsed), [*debug, *hidden], shown)


def _over_lz_from_sfc(parsed: ParsedNotam, reading: NotamReading, label: str, conditional: bool, debug: list[str], extras: list[str], schedule_text: str) -> ActionResult:
    """Rows 2a-2d, 2, 3 and the informational warning, for a read NOTAM over the LZ from the surface."""
    here = f"{label} over the LZ from the surface"
    if parsed.subject in RESTRICTED_SUBJECTS:
        return _result(parsed, NotamAction.NO_FLY, f"{here}: Q-code {parsed.q_code} ({RESTRICTED_SUBJECTS[parsed.subject]})", debug, extras)
    if parsed.subject == DANGER_AREA:
        return _danger_area(parsed, reading, here, debug, extras)
    why = _prohibition(reading)
    if why is not None:
        return _result(parsed, NotamAction.NO_FLY, f"{here}: {why}", debug, extras)
    if _has_contact(reading) and (reading.category is Category.COOPERATIVE or conditional):
        contact = f"{reading.contact.kind.value} {(reading.contact.value or '').strip()}"
        return _result(parsed, NotamAction.CONTACT_FIRST, f"{here}: contact {contact} first and fly only if they agree", debug, extras)
    if reading.category is Category.COOPERATIVE:
        radio = " (a radio frequency alone is not a contact)" if reading.contact.kind is ContactKind.RADIO_ONLY else ""
        return _result(parsed, NotamAction.NO_FLY, f"{here}: cooperative but there is no phone or email contact{radio}", debug, extras)
    when = f" ({schedule_text})" if schedule_text else ""
    return _result(parsed, NotamAction.AWARENESS, f"{label}{when}: informational over the LZ from the surface, a warning and not a restriction", debug, extras)


def notam_action(
    parsed: ParsedNotam,
    reading: NotamReading | None,
    lz: LandingZone,
    segment: Segment | ActionSegment,
    *,
    rules: Rules | None = None,
) -> ActionResult:
    """The action for one NOTAM in one segment, with a one-line reason. See the module text for the table."""
    rules = RULES if rules is None else rules
    debug: list[str] = []
    extras: list[str] = []
    pressure = _pressure(segment, rules)

    # Row 1: does it apply to a paramotor, is it active, is it near?
    exemptions = _checked(parsed, reading, debug) if reading else []
    not_for_us = _not_for_paramotors(parsed, reading, exemptions) if reading else None
    if not_for_us is not None:
        return _result(parsed, NotamAction.IGNORE, not_for_us, debug)
    schedule = _schedule(parsed, reading, lz, segment.start, segment.end)
    if not schedule.intervals:
        return _result(parsed, NotamAction.IGNORE, f"not active in this segment (schedule {parsed.item_d or 'none'})", debug)
    if not schedule.parsed:
        extras.append("schedule not parsed")
        debug.append(schedule.note or "schedule not parsed")
    where = _where(parsed, reading, lz, debug)
    local_area = _number(rules.local_area)
    if not where.over and where.distance_nm > local_area:
        return _result(parsed, NotamAction.IGNORE, f"{where.distance_nm:.2f} NM from the LZ, beyond the {local_area:g} NM local area", debug)
    if not where.usable:
        extras.append("no usable geometry, so treated as over the LZ")

    # What is left applies. Collect what the reason should carry.
    floor, ceiling = _limits(parsed, reading, lz, pressure, rules, extras)
    label = reading.activity.strip() if reading and reading.activity.strip() else f"NOTAM {parsed.id}"
    if reading:
        designators = [d for d in map(normalise_designator, reading.activates_designators) if d]
        if designators:
            extras.append(f"activates {', '.join(designators)}")
        conditions = [e for e in exemptions if e.covers is ExemptionCovers.CONDITIONAL]
        for exemption in conditions:
            extras.append(f'condition: "{exemption.quote.strip()}"')
    else:
        conditions = []
    sfc = floor.qnh <= lz.elevation_ft

    if where.over and sfc:
        if reading is None:
            reason = f"NOTAM {parsed.id} is over the LZ from the surface and has not been read: {' '.join(redact_phones(parsed.item_e)[0].split())}"
            return _result(parsed, NotamAction.UNREAD_OVER_LZ, reason, debug, extras)
        return _over_lz_from_sfc(parsed, reading, label, bool(conditions), debug, extras, reading.schedule.strip())

    if not where.over:
        assert where.bearing_deg is not None  # not over, so there is a nearest edge point
        compass = compass16(where.bearing_deg)
        start = "" if sfc else f", from {floor.text}"
        top = ", no upper limit" if math.isinf(ceiling.qnh) else f", up to {ceiling.text}"
        reason = f"{label} {where.distance_nm:.2f} NM to the {compass} ({where.edge}){start}{top}: avoid that sector"
        return _result(parsed, NotamAction.AVOID_SECTOR, reason, debug, extras, where.bearing_deg)

    # Over the LZ with a floor above the ground.
    if floor.qnh <= _number(rules.band_low):
        stay_below = floor.qnh - _number(rules.ceiling_buffer)
        reason = f"{label} over the LZ from {floor.text} ({_feet(floor.qnh)} ft QNH): stay below {_feet(stay_below)} ft QNH"
        return _result(parsed, NotamAction.AVOID_SECTOR, reason, debug, extras)
    if floor.qnh <= _number(rules.band_high):
        return _result(parsed, NotamAction.AWARENESS, f"{label} over the LZ from {floor.text} ({_feet(floor.qnh)} ft QNH): awareness", debug, extras)
    return _result(parsed, NotamAction.AWARENESS_MAP, f"{label} over the LZ from {floor.text} ({_feet(floor.qnh)} ft QNH): map only", debug, extras)
