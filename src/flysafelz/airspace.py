"""What airspace means for one segment, from code alone. No network, no model, no state.

    assess_airspace(areas, lz, segment, *, site_elevation_ft, pressure_msl, activations)
        -> AirspaceAssessment: one finding per airspace that matters, and the lowest ceiling
    danger_area_active(area, segment, activations) -> Activity

A segment is anything with a `start` and an `end` (a contracts.Segment, or a TimeSpan).
Heights are feet; "QNH" means feet above mean sea level.

Over the LZ (the LZ is inside the polygon, or exactly on its edge), per volume:
    CTR, prohibited, restricted, TMZ, from SFC          NO_GO
    ATZ, MATZ, from SFC                                 PERMISSION  "permission from <name> first"
    RMZ, from SFC                                       RADIO       "radio contact with <name> required"
    those types, or a CTA/TMA/airway of class A-E,
      floor up to the low band                          CEILING     "stay below <floor QNH - buffer> ft QNH"
      floor up to the high band                         WARNING
      floor above the high band, class G, any
      other type (gliding, TRA, TSA...)                 AWARENESS   (map and collapsed list only)
A floor at or below the site elevation is SFC, whatever its datum. A CTA, TMA or
airway that reaches the ground has no SFC rule of its own: its ceiling is the site
elevation less the buffer, which is under the minimum, so it is NO_GO.

Stacked volumes over the LZ (E14): the lowest floor above SFC sets the single
CEILING; the others are IGNORE with the reason. Each volume's own SFC outcome still
applies. If the ceiling left above the site is under the ceiling_left_no_go rule (E15) the
finding is NO_GO instead, with the ceiling in the reason.

Within the local area but not over the LZ (distance to the nearest boundary point):
    an SFC outcome (NO_GO, PERMISSION, RADIO, NO_FLY)   AVOID_SECTOR "avoid the sector to the <16-point bearing>"
    a floor that gives CEILING                          CEILING, "stay below ..." (you can fly under it)
    a floor that gives WARNING or AWARENESS             WARNING or AWARENESS, never an avoid
Every one of these carries bearing_deg (from the LZ to that nearest point) and its
compass16 name. Beyond the local area nothing is said.

Danger areas (G10, G13) are decided by activation first, per segment:
    byNotam            active only while an activation for its designator covers the segment
    published hours    active inside them (schedule.py), or while activated
    H24, empty, or hours that cannot be read: active, and noted
Inactive: IGNORE with a debug reason. Active, over the LZ, from SFC, the activating NOTAMs decide the way
notam_action() does for a danger area (the data can rule a flight out or attach a condition, never rule it in):
    an activation whose reading gives a prohibitive reason (quoted)   NO_FLY
    there is no activation at all (hours, H24)                        NO_FLY
    an activation with no reading, on an area that is active alone    NO_FLY (an unread NOTAM never relaxes it)
    an activation with no reading, on a by-NOTAM area                 UNVERIFIED, with the E text
    a read activation with no phone or email in its original item E   CHECK_SUP  "read the SUP / AIP ENR 5.1
                                                                      entry for <designator> before flight[; details in AIP SUP n/yyyy]"
    a read activation with a phone or email in its original item E    CONTACT_FIRST (permission wording, SUP named)
The reader's category never decides it: a prohibitive, cooperative or informational category with reason none
is CONTACT_FIRST or CHECK_SUP like any other. The contact is found by code in the original item E
(Activation.item_e_contact, filled by assess; a <PHONE_n> placeholder counts, a radio frequency, an AR reference
or coordinates do not), never by the model. Where the reading's own contact disagrees, code decides and a note
says so.
Several activations on one area: NO_FLY beats UNVERIFIED beats CHECK_SUP beats CONTACT_FIRST. An active danger
area above SFC follows the height bands like any other volume. One finding per danger area: PR-6 only labels the
activating NOTAM's own row "activates <designator>".

The polygon comes from openAIP. When the area is not in the openAIP data, the Q-circle of
the NOTAM that activates it stands in (drawn a little outside the circle, so it is never smaller).

Every number is a rules.yaml rule: the local area, the bands, the buffer, the ceiling minimum, the
FL-to-feet conversion. The only constants here are definitions (a flight level is hundreds of feet).
"""

import math
import re
from collections.abc import Sequence
from enum import StrEnum

from pydantic import AwareDatetime, BaseModel, field_validator

from flysafelz.clients.openaip import ARC_MINUTES_PER_DEGREE, Airspace
from flysafelz.contracts import (
    STRICT,
    Circle,
    ContactKind,
    Datum,
    Limit,
    LimitUnit,
    NotamReading,
    ProhibitiveReason,
    Segment,
)
from flysafelz.designators import normalise_designator
from flysafelz.geo import (
    Shape,
    compass16,
    shape_edge_bearing_deg,
    shape_edge_distance_nm,
    shape_over,
)
from flysafelz.redact import redact_phones
from flysafelz.rules import RULES, Rule, Rules
from flysafelz.schedule import parse_schedule
from flysafelz.sup_refs import sup_detail
from flysafelz.timetext import check_tz, clock_span

LatLon = tuple[float, float]

DANGER = "danger"
NO_GO_TYPES = frozenset({"ctr", "prohibited", "restricted", "tmz"})
PERMISSION_TYPES = frozenset({"atz", "matz"})
RADIO_TYPES = frozenset({"rmz"})
CONTROLLED_TYPES = frozenset({"cta", "tma", "airway"})  # relevant only when the ICAO class is A-E
CONTROLLED_CLASSES = frozenset({"A", "B", "C", "D", "E"})
CIRCLE_VERTICES = 72
FEET_PER_FLIGHT_LEVEL = 100  # a flight level is hundreds of feet


class Outcome(StrEnum):
    """What one airspace volume means for the segment."""

    NO_GO = "NO_GO"  # over a CTR, prohibited, restricted or TMZ from SFC; or the ceiling left is too low
    PERMISSION = "PERMISSION"
    RADIO = "RADIO"
    CEILING = "CEILING"
    WARNING = "WARNING"
    AWARENESS = "AWARENESS"
    AVOID_SECTOR = "AVOID_SECTOR"
    CONTACT_FIRST = "CONTACT_FIRST"  # an active danger area, activated by a NOTAM with a phone or email in its item E
    CHECK_SUP = "CHECK_SUP"  # an active danger area, activated by a NOTAM with no phone or email: read the SUP
    NO_FLY = "NO_FLY"  # an active danger area with a quoted prohibitive activation, or with no NOTAM (hours, H24)
    UNVERIFIED = "UNVERIFIED"  # an active danger area activated by a NOTAM nobody has read
    IGNORE = "IGNORE"  # debug only


SFC_OUTCOMES = frozenset({Outcome.NO_GO, Outcome.PERMISSION, Outcome.RADIO, Outcome.NO_FLY})


class TimeSpan(BaseModel):
    """A stretch of time. A contracts.Segment can be passed wherever a TimeSpan is asked for."""

    model_config = STRICT

    start: AwareDatetime
    end: AwareDatetime


class Activation(BaseModel):
    """A NOTAM that activates a danger area for a time.

    designator is normalised (D496A and EGD 496A both become EGD496A). reading is None
    when the NOTAM is unread. circle is the NOTAM's Q-circle, used only when openAIP does
    not have the area. e_text is the redacted E text: the raw item E is never kept here.
    item_e_contact is what code found in the NOTAM's original item E (PHONE, EMAIL or NONE;
    see find_item_e_contact) and sup_references are the AIP SUPs it names. Both are filled
    by code, never by the model, and decide CONTACT_FIRST against CHECK_SUP.
    """

    model_config = STRICT

    designator: str
    notam_id: str
    reading: NotamReading | None = None
    e_text: str
    start: AwareDatetime
    end: AwareDatetime
    circle: Circle | None = None
    item_e_contact: ContactKind = ContactKind.NONE
    sup_references: list[str] = []

    @field_validator("designator")
    @classmethod
    def _normalised(cls, value: str) -> str:
        found = normalise_designator(value)
        if found is None:
            raise ValueError(f"{value!r} is not an airspace designator")
        return found

    @field_validator("item_e_contact")
    @classmethod
    def _phone_email_or_none(cls, value: ContactKind) -> ContactKind:
        if value is ContactKind.RADIO_ONLY:
            raise ValueError("a radio frequency is not a contact: item_e_contact is PHONE, EMAIL or NONE")
        return value


_PLACEHOLDER = re.compile(r"<PHONE_\d+>")
# x@y.z, or the NOTAM way of writing it, x(AT)y.z (OPS(AT)FLYLOGIX.COM). The domain needs a dot and a label after it.
_EMAIL = re.compile(r"[A-Z0-9._%+-]+(?:@|\(AT\))[A-Z0-9-]+(?:\.[A-Z0-9-]+)+", re.IGNORECASE)


def find_item_e_contact(item_e: str) -> ContactKind:
    """What contact a NOTAM's item E gives: PHONE, EMAIL or NONE. Code, on the original text, never a model.

    A phone is whatever redact_phones finds, or a <PHONE_n> placeholder it already put there (so redacting the text
    first never turns a contact into no contact). An email is x@y.z or x(AT)y.z. A radio frequency (128.925MHZ),
    an AR reference (AR-2026-7404/05) and coordinates (534732N) are not a contact. A phone wins over an email.
    """
    if redact_phones(item_e)[1] or _PLACEHOLDER.search(item_e):
        return ContactKind.PHONE
    if _EMAIL.search(item_e):
        return ContactKind.EMAIL
    return ContactKind.NONE


class Activity(BaseModel):
    """Whether a danger area is active in a segment, and why. activations are those that cover it."""

    model_config = STRICT

    active: bool
    active_alone: bool = False  # active with no activation at all (H24, no or unreadable hours, inside published hours)
    reason: str
    activations: list[Activation] = []
    notes: list[str] = []


class Finding(BaseModel):
    """One airspace volume's meaning for the segment.

    condition is the text a pilot sees (None for NO_GO, NO_FLY, UNVERIFIED, AWARENESS, IGNORE:
    those carry only the reason). bearing_deg and compass are set for anything within the local
    area but not over the LZ.
    """

    model_config = STRICT

    area: str  # the airspace's name
    outcome: Outcome
    reason: str
    condition: str | None = None
    bearing_deg: float | None = None
    compass: str | None = None
    floor_qnh_ft: float | None = None
    ceiling_qnh_ft: float | None = None


class AirspaceAssessment(BaseModel):
    """Every finding for the segment, the lowest ceiling over the LZ (QNH ft, None when there is none), and notes."""

    model_config = STRICT

    findings: list[Finding] = []
    ceiling_qnh_ft: float | None = None
    notes: list[str] = []


# --- heights ------------------------------------------------------------------


def _value(rule: Rule) -> float:
    if isinstance(rule.value, list):
        raise TypeError(f"rule {rule.id} is a list, not a number")
    return float(rule.value)


def limit_qnh_ft(limit: Limit, site_elevation_ft: float, pressure_msl: float, rules: Rules = RULES) -> float:
    """A limit as feet above mean sea level (QNH).

    SFC is the ground, so the site elevation. AGL adds the site elevation. A flight level is
    FL x 100 less (reference pressure - the hour's pressure_msl) x ft per hPa (E9): the same FL is
    lower on a low-pressure day.
    """
    if limit.datum is Datum.SFC:
        return site_elevation_ft
    if limit.unit is LimitUnit.FL:
        drop = (_value(rules.fl_reference_hpa) - pressure_msl) * _value(rules.fl_ft_per_hpa)
        return limit.value * FEET_PER_FLIGHT_LEVEL - drop
    if limit.datum is Datum.AGL:
        return limit.value + site_elevation_ft
    return limit.value


def _feet(x: float) -> str:
    return f"{math.floor(x):,}"


# --- one volume ---------------------------------------------------------------


def _sfc_outcome(area: Airspace) -> Outcome | None:
    if area.type in NO_GO_TYPES:
        return Outcome.NO_GO
    if area.type in PERMISSION_TYPES:
        return Outcome.PERMISSION
    if area.type in RADIO_TYPES:
        return Outcome.RADIO
    if area.type == DANGER:
        return Outcome.NO_FLY  # a stand-in: G10 decides which of NO_FLY, UNVERIFIED, CHECK_SUP, CONTACT_FIRST
    return None


def _banded(area: Airspace, floor: float, sfc: bool, rules: Rules, notes: list[str]) -> Outcome:
    """The outcome of a volume from its type and floor (all in QNH feet). G8."""
    sfc_outcome = _sfc_outcome(area)
    controlled = area.type in CONTROLLED_TYPES
    if controlled and area.icao_class is None:
        notes.append(f"{area.name}: ICAO class unknown; treated as controlled (class A-E)")
    controlled = controlled and (area.icao_class is None or area.icao_class in CONTROLLED_CLASSES)
    if sfc_outcome is None and not controlled:
        return Outcome.AWARENESS
    if sfc and sfc_outcome is not None:
        return sfc_outcome
    if floor <= _value(rules.band_low):
        return Outcome.CEILING
    if floor <= _value(rules.band_high):
        return Outcome.WARNING
    return Outcome.AWARENESS


def _label(area: Airspace) -> str:
    return f"{area.name} ({area.type.upper()})"


def _sfc_text(outcome: Outcome) -> str:
    return {
        Outcome.NO_GO: "no flying",
        Outcome.PERMISSION: "permission needed",
        Outcome.RADIO: "radio contact needed",
        Outcome.NO_FLY: "active danger area",
    }[outcome]


# --- danger areas -------------------------------------------------------------


def _covers(activation: Activation, segment: Segment | TimeSpan) -> bool:
    return activation.start < segment.end and segment.start < activation.end


def _anchor(shape: Shape) -> LatLon:
    """The middle of the shape's bounding box: where SR and SS are worked out for published hours."""
    points = [p for poly in shape for ring in poly for p in ring]
    lats, lons = [p[0] for p in points], [p[1] for p in points]
    return (min(lats) + max(lats)) * 0.5, (min(lons) + max(lons)) * 0.5


def danger_area_active(area: Airspace, segment: Segment | TimeSpan, activations: Sequence[Activation] = ()) -> Activity:
    """Is this danger area active in the segment (G10)?

    An activation counts when it names the area's designator and overlaps the segment
    (any overlap, so a segment that was not split at the NOTAM's edge fails closed).
    byNotam with no published hours: active only with such an activation. Published hours
    decide when there are any (active inside them, or when activated), so "MON-FRI
    0800-1700, other times by NOTAM" is active in its hours and byNotam is one more way to
    activate it. H24, no hours, or hours that cannot be read are active, and noted.
    active_alone says the area is active without any activation (so an activation, read
    or not, can never make it less than active).
    """
    covering = [a for a in activations if area.designator is not None and a.designator == area.designator and _covers(a, segment)]
    ids = ", ".join(a.notam_id for a in covering)
    has_hours = bool(area.hours and area.hours.strip())
    if area.by_notam and not has_hours and area.hours_readable:
        if area.designator is None:
            return Activity(active=True, reason="by NOTAM but the name has no designator to match", notes=[f"{area.name}: no designator in the name, so no NOTAM can be matched to it; treated as active"])
        if covering:
            return Activity(active=True, reason=f"activated by NOTAM {ids}", activations=covering)
        return Activity(active=False, reason="by NOTAM and no NOTAM activates it in this segment")
    if not area.hours_readable:  # before the no-hours check: openAIP's unreadable hours arrive as hours=None, hours_readable=False
        return Activity(active=True, active_alone=True, reason="published hours could not be read", activations=covering, notes=[f"{area.name}: published hours could not be read; treated as active"])
    if not has_hours:
        return Activity(active=True, active_alone=True, reason="no published hours", activations=covering, notes=[f"{area.name}: no published hours; treated as active"])
    assert area.hours is not None  # has_hours
    lat, lon = _anchor(area.shape)
    schedule = parse_schedule(area.hours, segment.start, segment.end, lat, lon)
    notes = [schedule.note] if schedule.note else []
    if area.hours.strip().upper() == "H24":
        notes.append(f"{area.name}: published H24; active")
    if schedule.intervals:
        return Activity(active=True, active_alone=True, reason=f"inside published hours {area.hours}", activations=covering, notes=notes)
    if covering:
        return Activity(active=True, reason=f"outside published hours but activated by NOTAM {ids}", activations=covering, notes=notes)
    if area.by_notam and area.designator is None:
        notes.append(f"{area.name}: no designator in the name, so no NOTAM can be matched to it; treated as active")
        return Activity(active=True, reason="outside published hours, by NOTAM, but the name has no designator to match", notes=notes)
    return Activity(active=False, reason=f"outside published hours {area.hours}", notes=notes)


def _prohibition(reading: NotamReading) -> str | None:
    """Why this reading rules the flight out, or None. Only a prohibitive reason does (G13, CR2): the category alone never does.

    The quote is the reader's, checked against the redacted E text; it is redacted again so no number can reach a reason.
    """
    if reading.prohibitive_reason is ProhibitiveReason.NONE:
        return None
    quote = " ".join(redact_phones(reading.prohibitive_quote or "")[0].split())
    said = f': "{quote}"' if quote else ""
    return f"prohibited ({reading.prohibitive_reason.value}){said}"


def _sup_tail(activation: Activation) -> str:
    """"; details in AIP SUP 108/2026" for the SUPs the NOTAM names, else nothing (no empty brackets, no trailing "; ")."""
    detail = sup_detail(activation.sup_references)
    return f"; {detail}" if detail else ""


def _contact_condition(label: str, activation: Activation, reading: NotamReading, tz: str | None = None) -> str:
    """The call-first wording. The contact is the reading's when it gives a phone or email, else it points at the NOTAM.

    The activation span is written in `tz` (None: UTC, no zone label). The Activation itself stays UTC.
    """
    what = f" ({reading.activity})" if reading.activity.strip() else ""
    window = clock_span(activation.start, activation.end, tz)
    if _reading_contact(reading) is ContactKind.NONE:
        contact = f"{activation.item_e_contact.value} given in NOTAM {activation.notam_id}"
    else:
        contact = f"{reading.contact.kind.value} {(reading.contact.value or '').strip()}"
    return f"Danger area {label} active {window}{what}: {contact} and fly only if they agree{_sup_tail(activation)}"


def _check_sup_condition(activation: Activation) -> str:
    """The same words as notam_action's CHECK_SUP row, for the area this activation names."""
    return f"read the SUP / AIP ENR 5.1 entry for {activation.designator} before flight{_sup_tail(activation)}"


def _reading_contact(reading: NotamReading) -> ContactKind:
    """The reading's own contact: PHONE or EMAIL with a value, else NONE (a radio frequency is not a contact)."""
    contact = reading.contact
    if contact.kind in (ContactKind.PHONE, ContactKind.EMAIL) and (contact.value or "").strip():
        return contact.kind
    return ContactKind.NONE


def _disagreement(activation: Activation, reading: NotamReading) -> str | None:
    """A note when the reading's contact and the code's disagree on whether the NOTAM has one. Code decides."""
    in_reading = _reading_contact(reading) is not ContactKind.NONE
    in_text = activation.item_e_contact is not ContactKind.NONE
    if in_reading == in_text:
        return None
    if in_reading:
        return f"NOTAM {activation.notam_id}: the reading gives a {reading.contact.kind.value} contact but code finds none in item E; code decides (CHECK_SUP)"
    return f"NOTAM {activation.notam_id}: code finds a {activation.item_e_contact.value} in item E but the reading gives no phone or email; code decides (CONTACT_FIRST)"


def _danger_over_lz(area: Airspace, activity: Activity, notes: list[str], tz: str | None = None) -> Finding:
    """An active danger area over the LZ from SFC: NO_FLY, UNVERIFIED, CHECK_SUP or CONTACT_FIRST, worst first (G13).

    Follows notam_action's rows 2b to 2d for the activating NOTAMs. NO_FLY needs a quoted prohibitive reason (or no
    activation at all, or an unread one on an area that is active alone); otherwise the code-found contact decides
    between CONTACT_FIRST and CHECK_SUP. A disagreement with the reading's contact goes into `notes`. The activation
    span in the CONTACT_FIRST condition is written in `tz`.
    """
    label = area.designator or area.name
    activations = activity.activations
    if not activations:
        return Finding(area=area.name, outcome=Outcome.NO_FLY, reason=f"Danger area {label} is active ({activity.reason}) and no NOTAM gives a way in: do not fly")
    for activation in activations:
        problem = _prohibition(activation.reading) if activation.reading else None
        if problem:
            return Finding(area=area.name, outcome=Outcome.NO_FLY, reason=f"Danger area {label} is active (NOTAM {activation.notam_id}): {problem}; do not fly")
    for activation in activations:
        if activation.reading is None and activity.active_alone:
            reason = f"Danger area {label} is active ({activity.reason}); NOTAM {activation.notam_id} names it but has not been read: {activation.e_text}; do not fly"
            return Finding(area=area.name, outcome=Outcome.NO_FLY, reason=reason)
    for activation in activations:
        if activation.reading is None:
            reason = f"Danger area {label} is activated by NOTAM {activation.notam_id}, which has not been read: {activation.e_text}"
            return Finding(area=area.name, outcome=Outcome.UNVERIFIED, reason=reason)
    for activation in activations:
        reading = activation.reading  # read, as no activation is unread by now
        if reading is None:
            raise ValueError("an activation without a reading was handled above")
        if activation.item_e_contact is ContactKind.NONE:
            note = _disagreement(activation, reading)
            if note:
                notes.append(note)
            reason = f"Danger area {label} is active by NOTAM {activation.notam_id}: no phone or email in the NOTAM text and no prohibitive wording (a radio frequency is not a contact)"
            return Finding(area=area.name, outcome=Outcome.CHECK_SUP, reason=reason, condition=_check_sup_condition(activation))
    activation = activations[0]
    reading = activation.reading
    if reading is None:
        raise ValueError("an activation without a reading was handled above")
    note = _disagreement(activation, reading)
    if note:
        notes.append(note)
    condition = _contact_condition(label, activation, reading, tz)
    return Finding(area=area.name, outcome=Outcome.CONTACT_FIRST, reason=f"Danger area {label} is active by NOTAM {activation.notam_id}", condition=condition)


def _circle_shape(circle: Circle) -> Shape:
    """A polygon just outside the circle, so it is never smaller than the Q-circle."""
    radius_deg = circle.radius_nm / ARC_MINUTES_PER_DEGREE / math.cos(math.pi / CIRCLE_VERTICES)
    lon_scale = math.cos(math.radians(circle.lat))
    ring = [
        (
            circle.lat + radius_deg * math.cos(math.tau * k / CIRCLE_VERTICES),
            circle.lon + radius_deg * math.sin(math.tau * k / CIRCLE_VERTICES) / lon_scale,
        )
        for k in range(CIRCLE_VERTICES)
    ]
    return [[[*ring, ring[0]]]]


def _qcircle_areas(areas: list[Airspace], activations: Sequence[Activation], segment: Segment | TimeSpan) -> tuple[list[Airspace], list[Finding]]:
    """Stand-ins for activated danger areas that openAIP does not have, and findings for those that cannot be placed."""
    known = {a.designator for a in areas if a.designator}
    stand_ins: list[Airspace] = []
    findings: list[Finding] = []
    for activation in activations:
        if activation.designator in known or not _covers(activation, segment):
            continue
        known.add(activation.designator)
        if activation.circle is None:
            reason = f"{activation.designator} is activated by NOTAM {activation.notam_id}, but openAIP does not have the area and the NOTAM has no Q-circle: {activation.e_text}"
            findings.append(Finding(area=activation.designator, outcome=Outcome.UNVERIFIED, reason=reason))
            continue
        # no reading, or a reading whose E text states no floor: the stand-in comes down to the surface (the cautious side)
        floor = (activation.reading.floor if activation.reading else None) or Limit(value=0, unit=LimitUnit.FT, datum=Datum.SFC)
        stand_ins.append(
            Airspace(
                id=f"qcircle:{activation.designator}",
                name=activation.designator,
                type=DANGER,
                lower=floor,
                by_notam=True,
                designator=activation.designator,
                shape=_circle_shape(activation.circle),
                notes=[f"{activation.designator}: not in the openAIP data; the Q-circle of NOTAM {activation.notam_id} is used"],
            )
        )
    return stand_ins, findings


# --- the segment --------------------------------------------------------------


def _sector(area: Airspace, outcome: Outcome, floor: float, distance: float, bearing: float, sfc: bool, buffer: float) -> Finding:
    """A volume within the local area but not over the LZ (G9)."""
    compass = compass16(bearing)
    where = f"{distance:.2f} NM to the {compass}"
    label = _label(area)
    if sfc and outcome in (SFC_OUTCOMES | {Outcome.CEILING}):
        what = _sfc_text(outcome) if outcome in SFC_OUTCOMES else "reaches the ground"
        return Finding(
            area=area.name,
            outcome=Outcome.AVOID_SECTOR,
            reason=f"{label} {where}: {what}",
            condition=f"avoid the sector to the {compass} ({area.name})",
            bearing_deg=bearing,
            compass=compass,
            floor_qnh_ft=floor,
        )
    if outcome is Outcome.CEILING:
        ceiling = floor - buffer
        return Finding(
            area=area.name,
            outcome=outcome,
            reason=f"{label} {where}, floor {_feet(floor)} ft QNH",
            condition=f"stay below {_feet(ceiling)} ft QNH",
            bearing_deg=bearing,
            compass=compass,
            floor_qnh_ft=floor,
            ceiling_qnh_ft=ceiling,
        )
    if outcome is Outcome.WARNING:
        return Finding(area=area.name, outcome=outcome, reason=f"{label} {where}, floor {_feet(floor)} ft QNH", condition=f"{label} {where}: floor {_feet(floor)} ft QNH", bearing_deg=bearing, compass=compass, floor_qnh_ft=floor)
    return Finding(area=area.name, outcome=Outcome.AWARENESS, reason=f"{label} {where}, floor {_feet(floor)} ft QNH", bearing_deg=bearing, compass=compass, floor_qnh_ft=floor)


def _over(area: Airspace, outcome: Outcome, floor: float, activity: Activity | None, notes: list[str], tz: str | None = None) -> Finding | None:
    """A volume over the LZ. CEILING is returned as None: the caller pools those (E14)."""
    label = _label(area)
    if outcome is Outcome.NO_FLY and activity is not None:
        return _danger_over_lz(area, activity, notes, tz)
    if outcome is Outcome.NO_GO:
        return Finding(area=area.name, outcome=outcome, reason=f"over {label} from the surface", floor_qnh_ft=floor)
    if outcome is Outcome.PERMISSION:
        return Finding(area=area.name, outcome=outcome, reason=f"over {label} from the surface", condition=f"permission from {area.name} first", floor_qnh_ft=floor)
    if outcome is Outcome.RADIO:
        return Finding(area=area.name, outcome=outcome, reason=f"over {label} from the surface", condition=f"radio contact with {area.name} required", floor_qnh_ft=floor)
    if outcome is Outcome.WARNING:
        text = f"{label} starts at {_feet(floor)} ft QNH above the LZ"
        return Finding(area=area.name, outcome=outcome, reason=text, condition=text, floor_qnh_ft=floor)
    if outcome is Outcome.AWARENESS:
        return Finding(area=area.name, outcome=outcome, reason=f"{label} floor {_feet(floor)} ft QNH: awareness only", floor_qnh_ft=floor)
    return None


def _ceiling_finding(area: Airspace, floor: float, site_elevation_ft: float, rules: Rules) -> Finding:
    """The single ceiling over the LZ (E14), or NO_GO when too little is left above the site (E15)."""
    ceiling = floor - _value(rules.ceiling_buffer)
    left = ceiling - site_elevation_ft
    label = _label(area)
    if rules.ceiling_left_no_go.breached(left):
        reason = (
            f"{label} floor {_feet(floor)} ft QNH leaves a ceiling of {_feet(ceiling)} ft QNH, "
            f"only {_feet(left)} ft above the site (minimum {_feet(_value(rules.ceiling_left_no_go))} ft)"
        )
        return Finding(area=area.name, outcome=Outcome.NO_GO, reason=reason, floor_qnh_ft=floor, ceiling_qnh_ft=ceiling)
    reason = f"{label} floor {_feet(floor)} ft QNH less the {_feet(_value(rules.ceiling_buffer))} ft buffer"
    return Finding(area=area.name, outcome=Outcome.CEILING, reason=reason, condition=f"stay below {_feet(ceiling)} ft QNH", floor_qnh_ft=floor, ceiling_qnh_ft=ceiling)


def assess_airspace(
    areas: list[Airspace],
    lz: LatLon,
    segment: Segment | TimeSpan,
    *,
    site_elevation_ft: float,
    pressure_msl: float | None = None,
    activations: Sequence[Activation] = (),
    rules: Rules = RULES,
    tz: str | None = None,
) -> AirspaceAssessment:
    """What every airspace volume means for the LZ in this segment.

    pressure_msl is the hour's forecast pressure for flight levels; None means the
    reference pressure, and that is noted. activations are the NOTAMs that activate danger areas.
    tz is an IANA zone name for the times written into the condition text ("active 09:00-19:00 BST"); None writes UTC.
    Nothing structured changes with it, and an unknown name raises ValueError.
    """
    check_tz(tz)
    lat, lon = lz
    notes: list[str] = []
    if pressure_msl is None:
        pressure_msl = _value(rules.fl_reference_hpa)
        notes.append("no forecast pressure: flight levels are taken at the reference pressure")
    activations = list(activations)
    stand_ins, findings = _qcircle_areas(areas, activations, segment)
    # local_area is a plain setting (comparison "=", a radius): rules.yaml holds the radius, and the cut-off is by design
    # `distance > radius`, so a volume exactly at the radius is within, as for the NOTAM prefilter (pib.py) and notam_action.
    local_area = _value(rules.local_area)
    ceilings: list[tuple[float, Airspace]] = []

    for area in [*areas, *stand_ins]:
        over = shape_over(lat, lon, area.shape)
        distance = 0.0 if over else shape_edge_distance_nm(lat, lon, area.shape)
        if not over and distance > local_area:
            continue
        notes += area.notes
        activity = None
        if area.type == DANGER:
            activity = danger_area_active(area, segment, activations)
            notes += activity.notes
            if not activity.active:
                findings.append(Finding(area=area.name, outcome=Outcome.IGNORE, reason=f"{_label(area)} is not active in this segment ({activity.reason})"))
                continue
        floor = limit_qnh_ft(area.lower, site_elevation_ft, pressure_msl, rules)
        sfc = floor <= site_elevation_ft
        outcome = _banded(area, floor, sfc, rules, notes)
        if not over:
            bearing = shape_edge_bearing_deg(lat, lon, area.shape)
            findings.append(_sector(area, outcome, floor, distance, bearing, sfc, _value(rules.ceiling_buffer)))
        elif outcome is Outcome.CEILING:
            ceilings.append((floor, area))
        else:
            finding = _over(area, outcome, floor, activity, notes, tz)
            if finding is not None:
                findings.append(finding)

    lowest: float | None = None
    if ceilings:
        ceilings.sort(key=lambda pair: pair[0])
        floor, area = ceilings[0]
        findings.append(_ceiling_finding(area, floor, site_elevation_ft, rules))
        lowest = floor - _value(rules.ceiling_buffer)
        for other_floor, other in ceilings[1:]:
            reason = f"{_label(other)} floor {_feet(other_floor)} ft QNH is higher than {area.name}, which sets the ceiling"
            findings.append(Finding(area=other.name, outcome=Outcome.IGNORE, reason=reason, floor_qnh_ft=other_floor))
    return AirspaceAssessment(findings=findings, ceiling_qnh_ft=lowest, notes=list(dict.fromkeys(notes)))
