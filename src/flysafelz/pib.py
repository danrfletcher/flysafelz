"""Parse the NATS PIB XML into Parsed NOTAMs, prefilter them, and sort them with code only (G12, G13, G23).

NATS publishes no schema. These element names were read from a real PIB
downloaded once while building this module (never in tests):

    Pib
      AreaPIBHeader
        Issued                      2026-10-31T07:00:37+00:00 (the PIB's issued time)
      FIRSection                    one per FIR
        ICAO, Name
        ADSection                   aerodromes: Code, Name, NotamList, ObsoleteNotamList
        En-route                    NotamList, ObsoleteNotamList
        Warnings                    NotamList, ObsoleteNotamList
          NotamList
            Notam                   attributes PIBSection (AD | FIR | WAR), Reference, OriginalMessage
              NOF                   originating office, e.g. EGGN
              Series, Number, Year  H, 7171, 26  -> id H7171/26
              Type                  N new | R replaces | C cancels
              ReferredSeries, ReferredNumber, ReferredYear     the NOTAM an R or C refers to
              QLine
                FIR, Code23, Code45 subject and condition: RD + CA -> QRDCA
                Traffic, Purpose, Scope
                Lower, Upper        flight levels
              Coordinates           5356N00001E (DDMMN DDDMME), the Q-circle centre
              Radius                NM (a Q-circle radius; 999 = the whole FIR)
              ItemA                 may repeat (EGTT, EGPX)
              StartValidity         item B, YYMMDDHHMM UTC
              EndValidity           item C, YYMMDDHHMM UTC or PERM
              Estimation            EST when item C is an estimate
              ItemD, ItemE, ItemF, ItemG
              Marker, NotamStoreDate
        ObsoleteNotamList           empty in the PIBs seen; its NOTAMs are never read
      MeteoDisclaimer, MergedWarnings      no NOTAMs in them in the PIBs seen

Required: Series/Number/Year (the id), Type, the Q-line (QLine with all eight
children, Coordinates, and a radius from the Radius element or from three digits
after the coordinates) and StartValidity (B). Missing any of them, XML that does
not parse, or a PIB with no Issued time raises PibError, and the source is
`failed` with a one-line reason. ItemD, ItemF and ItemG are optional and give None.
ItemA, ItemE and EndValidity missing are tolerated (A and E are "", a missing end
is open-ended, which fails closed) so one thin NOTAM does not take the whole
source down. If Coordinates carries the radius too ("5356N00001E010") and a Radius
element is also there, the larger wins.

Item E is kept exactly as published: Unicode and stray whitespace included.

The flow:

    read_notams(xml)            -> issued_at, NOTAMs deduped by id (first seen; every section kept)
    drop_superseded(notams)     -> NOTAMs without those an R replaces or a C cancels
    parse_pib(xml)              -> both of the above in one call
    load_pib(source, now)       -> fetch + parse + the source status (ok | failed | stale; case mode never stale)
    prefilter(notams, lz, window)       -> kept, dropped with reasons (G13)
    sort_notam(notam, span, lz)         -> IGNORE | VIA_AIRSPACE | AWARENESS | TO_READER (G23)

The sort uses coded fields only: series, Q code, traffic, purpose, radius and
B/C/D. There is no reading input. Only facts a reader extracts can lower a
NOTAM's action afterwards (notam_action(), PR-6); a reader never drops one the
sort kept. A NOTAMC is ignored here (it never produces an action itself). The
NOTAM an R replaces or a C cancels is dropped by drop_superseded, so it never
reaches the sort.

Every number the rules need (2 NM, FL195, 50 NM, the 3 h stale limit, the subject
lists) is read from rules.yaml. Distances come from geo.py and activity from
schedule.py; nothing is measured or parsed twice.
"""

import re
import xml.etree.ElementTree as ET
from datetime import UTC, datetime, timedelta
from enum import StrEnum

import httpx
from pydantic import BaseModel, ConfigDict, Field

from flysafelz.clients.pib_source import Source, fetch_pib, utc_text
from flysafelz.contracts import IgnoredNotam, NotamType, ParsedNotam, SourceName, SourceState, SourceStatus
from flysafelz.designators import find_designators
from flysafelz.geo import circle_edge_distance_nm
from flysafelz.notam_area import Broken, Shape, area_from_item_e
from flysafelz.rules import RULES, Rule, Rules
from flysafelz.schedule import parse_schedule

LatLon = tuple[float, float]  # the landing zone: (lat, lon) in degrees, east positive
Window = tuple[datetime, datetime]  # [start, end): start is in, end is out

CENTURY = 2000  # the PIB writes years as two digits
MINUTES_PER_DEGREE = 60
SECURITY_SERIES = "V"  # series V carries security advice (subject RO) and volcanic ash (kept)
PERMANENT = "PERM"
ESTIMATE = "EST"
# The ok-status reason in case mode, where the stale rule is off: the archive snapshots the PIB only about twice a day.
CASE_MODE_NOTE = "case mode: captured PIB treated as current at the case's now"

_TIME = re.compile(r"(\d\d)(\d\d)(\d\d)(\d\d)(\d\d)\s*(EST)?", re.IGNORECASE)
_COORDINATES = re.compile(r"(\d\d)(\d\d)([NS])\s?(\d\d\d)(\d\d)([EW])(?:\s?(\d\d\d))?")
_SECTION_FALLBACK = "unknown"


class PibError(ValueError):
    """The PIB cannot be read. The message is the one-line reason the source fails with."""


class SortAction(StrEnum):
    """What the code-only sort decided for one NOTAM in one span."""

    IGNORE = "IGNORE"
    VIA_AIRSPACE = "VIA_AIRSPACE"  # a restriction that names a designator: handled through the airspace polygon
    AWARENESS = "AWARENESS"  # area-wide or jamming: shown, never read
    TO_READER = "TO_READER"


class SortResult(BaseModel):
    """The sort's answer: an action, a one-line reason, and what it found on the way."""

    model_config = ConfigDict(extra="forbid")

    action: SortAction
    reason: str
    designators: list[str] = Field(default_factory=list)  # VIA_AIRSPACE: the designators found in item E
    schedule_note: str | None = None  # set when item D could not be read and B-C was used whole


class PibResult(BaseModel):
    """The NOTAM source after fetch and parse. notams is empty when the source failed."""

    model_config = ConfigDict(extra="forbid")

    status: SourceStatus  # carries issued_at and, from the archive, the commit sha
    notams: list[ParsedNotam] = Field(default_factory=list)
    superseded: list[IgnoredNotam] = Field(default_factory=list)  # replaced or cancelled, for the debug list


# --- parsing ----------------------------------------------------------------------


def _verbatim(el: ET.Element, tag: str) -> str | None:
    """The text of a child exactly as published, or None when the child is absent."""
    child = el.find(tag)
    if child is None:
        return None
    return child.text or ""


def _trimmed(el: ET.Element, tag: str) -> str | None:
    """The stripped text of a child, or None when it is absent or blank."""
    text = _verbatim(el, tag)
    return text.strip() or None if text is not None else None


def _need(el: ET.Element, tag: str, who: str) -> str:
    text = _trimmed(el, tag)
    if text is None:
        raise PibError(f"{who}: missing {tag}")
    return text


def _integer(text: str, tag: str, who: str) -> int:
    try:
        return int(text)
    except ValueError as exc:
        raise PibError(f"{who}: {tag} {text!r} is not a number") from exc


def _notam_id(series: str, number: int, year: int) -> str:
    """H7171/26: the series letter, the number, the two-digit year."""
    return f"{series}{number}/{year:02d}"


def _read_id(el: ET.Element, position: int) -> tuple[str, str, int, int]:
    """(id, series, number, year) of a Notam element. Before the id is known, errors name the NOTAM's position."""
    who = f"NOTAM #{position}"
    series = _need(el, "Series", who).upper()
    number = _integer(_need(el, "Number", who), "Number", who)
    year = _integer(_need(el, "Year", who), "Year", who)
    return _notam_id(series, number, year), series, number, year


def _read_referred(el: ET.Element, who: str) -> str | None:
    """The id an R or C refers to, or None when the PIB does not say."""
    series = _trimmed(el, "ReferredSeries")
    number = _trimmed(el, "ReferredNumber")
    year = _trimmed(el, "ReferredYear")
    if series is None or number is None or year is None:
        return None
    return _notam_id(series.upper(), _integer(number, "ReferredNumber", who), _integer(year, "ReferredYear", who))


def _read_time(text: str, who: str, tag: str) -> tuple[datetime, bool]:
    """(time, estimated) from YYMMDDHHMM with an optional EST suffix."""
    match = _TIME.fullmatch(text.strip())
    if match is None:
        raise PibError(f"{who}: {tag} {text!r} is not YYMMDDHHMM")
    yy, month, day, hour, minute, est = match.groups()
    try:
        when = datetime(CENTURY + int(yy), int(month), int(day), int(hour), int(minute), tzinfo=UTC)
    except ValueError as exc:
        raise PibError(f"{who}: {tag} {text!r} is not a time") from exc
    return when, est is not None


def _read_end(el: ET.Element, who: str) -> tuple[datetime | None, bool, bool]:
    """(C, permanent, estimated). PERM and a missing end are open-ended (None)."""
    estimated = _trimmed(el, "Estimation") == ESTIMATE
    text = _trimmed(el, "EndValidity")
    if text is None:
        return None, False, estimated
    if text.upper().removesuffix(ESTIMATE).strip() == PERMANENT:
        return None, True, estimated or text.upper().endswith(ESTIMATE)
    end, suffix = _read_time(text, who, "EndValidity")
    return end, False, estimated or suffix


def _read_circle(el: ET.Element, who: str) -> tuple[float, float, float]:
    """(lat, lon, radius NM) from Coordinates and Radius, or from Coordinates alone when it carries the radius."""
    text = _need(el, "Coordinates", who)
    match = _COORDINATES.fullmatch(text)
    if match is None:
        raise PibError(f"{who}: Coordinates {text!r} are not DDMMN DDDMME")
    lat_deg, lat_min, ns, lon_deg, lon_min, ew, radius_in_text = match.groups()
    lat = int(lat_deg) + int(lat_min) / MINUTES_PER_DEGREE
    lon = int(lon_deg) + int(lon_min) / MINUTES_PER_DEGREE
    radii = [float(radius_in_text)] if radius_in_text else []
    element_radius = _trimmed(el, "Radius")
    if element_radius is not None:
        try:
            radii.append(float(element_radius))
        except ValueError as exc:
            raise PibError(f"{who}: Radius {element_radius!r} is not a number") from exc
    if not radii:
        raise PibError(f"{who}: missing Radius")
    return (-lat if ns == "S" else lat), (-lon if ew == "W" else lon), max(radii)


def _read_notam(el: ET.Element, position: int) -> ParsedNotam:
    notam_id, series, number, year = _read_id(el, position)
    who = notam_id
    type_text = _need(el, "Type", who).upper()
    try:
        notam_type = NotamType(type_text)
    except ValueError as exc:
        raise PibError(f"{who}: Type {type_text!r} is not N, R or C") from exc

    replaces = _read_referred(el, who) if notam_type is not NotamType.NEW else None

    q_line = el.find("QLine")
    if q_line is None:
        raise PibError(f"{who}: missing QLine")
    lat, lon, radius = _read_circle(el, who)
    item_a = " ".join(a.text.strip() for a in el.findall("ItemA") if a.text and a.text.strip())
    valid_from, from_estimated = _read_time(_need(el, "StartValidity", who), who, "StartValidity")
    valid_to, permanent, estimated = _read_end(el, who)
    section = el.get("PIBSection") or _SECTION_FALLBACK

    return ParsedNotam(
        id=notam_id,
        series=series,
        number=number,
        year=year,
        type=notam_type,
        replaces=replaces,
        nof=_trimmed(el, "NOF") or "",
        fir=_need(q_line, "FIR", who),
        q_code="Q" + _need(q_line, "Code23", who) + _need(q_line, "Code45", who),
        traffic=_need(q_line, "Traffic", who),
        purpose=_need(q_line, "Purpose", who),
        scope=_need(q_line, "Scope", who),
        lower_fl=_integer(_need(q_line, "Lower", who), "Lower", who),
        upper_fl=_integer(_need(q_line, "Upper", who), "Upper", who),
        centre_lat=lat,
        centre_lon=lon,
        radius_nm=radius,
        item_a=item_a,
        valid_from=valid_from,
        valid_to=valid_to,
        permanent=permanent,
        est=estimated or from_estimated,
        item_d=_trimmed(el, "ItemD"),
        item_e=_verbatim(el, "ItemE") or "",
        item_f=_trimmed(el, "ItemF"),
        item_g=_trimmed(el, "ItemG"),
        section=section,
        sections=[section],
    )


def _read_issued(root: ET.Element) -> datetime:
    text = _trimmed(root, "AreaPIBHeader/Issued")
    if text is None:
        raise PibError("PIB has no AreaPIBHeader/Issued time")
    try:
        issued = datetime.fromisoformat(text)
    except ValueError as exc:
        raise PibError(f"PIB Issued time {text!r} is not a time") from exc
    return issued if issued.tzinfo else issued.replace(tzinfo=UTC)


def read_notams(xml: bytes | str) -> tuple[datetime, list[ParsedNotam]]:
    """The issued time and every NOTAM, deduped by id.

    The same id in two sections gives one NOTAM: the first seen, with every
    section it appeared in listed in `sections`. Raises PibError.
    """
    try:
        root = ET.fromstring(xml)
    except ET.ParseError as exc:
        raise PibError(f"PIB is not valid XML: {exc}") from exc
    issued = _read_issued(root)
    found: dict[str, ParsedNotam] = {}
    position = 0
    for notam_list in root.iter("NotamList"):
        for el in notam_list.findall("Notam"):
            position += 1
            notam = _read_notam(el, position)
            first = found.setdefault(notam.id, notam)
            if first is not notam and notam.section not in first.sections:
                first.sections.append(notam.section)
    return issued, list(found.values())


def drop_superseded(notams: list[ParsedNotam]) -> tuple[list[ParsedNotam], list[IgnoredNotam]]:
    """Remove the NOTAMs that an R in the list replaces or a C cancels (E4).

    Returns (kept, dropped with reasons). A replaced or cancelled id that is
    not in the list is not an error. The R and the C themselves stay.
    """
    gone: dict[str, str] = {}
    for notam in notams:
        if notam.type is not NotamType.NEW and notam.replaces and notam.replaces != notam.id:
            verb = "replaced" if notam.type is NotamType.REPLACE else "cancelled"
            gone.setdefault(notam.replaces, f"{verb} by {notam.id}")
    kept = [n for n in notams if n.id not in gone]
    dropped = [IgnoredNotam(notam_id=n.id, reason=gone[n.id]) for n in notams if n.id in gone]
    return kept, dropped


def parse_pib(xml: bytes | str) -> tuple[datetime, list[ParsedNotam]]:
    """(issued_at, NOTAMs): deduped, without the replaced and cancelled ones. Raises PibError."""
    issued, notams = read_notams(xml)
    kept, _dropped = drop_superseded(notams)
    return issued, kept


def _number(rule: Rule) -> float:
    """A numeric rule's value (the list-valued rules are read with .breached instead)."""
    if isinstance(rule.value, list):
        raise TypeError(f"rule {rule.id} is a list, not a number")
    return rule.value


def load_pib(
    source: Source, now: datetime, *, client: httpx.Client | None = None, rules: Rules = RULES, stale_rule: bool = True
) -> PibResult:
    """Fetch the PIB and read it. The status is ok, stale (issued too long before now) or failed.

    A stale PIB still returns its NOTAMs, so the verdict step can let a NO_GO
    from a working source win over UNVERIFIED. A failed one returns none.

    stale_rule: True (live mode) applies the pib_stale_hours rule. False (case mode) does not: the case's
    captured PIB is treated as current at the case's now, because the archive snapshots it only about twice
    a day. The status is then ok whatever the PIB's age, and its reason says so. A PIB that fails to read is
    still failed.
    """
    fetched = fetch_pib(source, now, client=client)
    if fetched.xml is None:
        return PibResult(status=fetched.status)
    fetched_at = fetched.status.fetched_at
    try:
        issued, notams = read_notams(fetched.xml)
    except PibError as exc:
        failed = SourceStatus(
            source=SourceName.NOTAMS, state=SourceState.FAILED, reason=str(exc), fetched_at=fetched_at, sha=fetched.status.sha
        )
        return PibResult(status=failed)
    kept, dropped = drop_superseded(notams)

    limit = _number(rules.pib_stale_hours)
    if not stale_rule:
        reason, state = CASE_MODE_NOTE, SourceState.OK
    elif now - issued > timedelta(hours=limit):
        reason = f"PIB issued {utc_text(issued)}, more than {limit:g} h before now"
        state = SourceState.STALE
    else:
        reason, state = "", SourceState.OK
    status = SourceStatus(
        source=SourceName.NOTAMS, state=state, reason=reason, fetched_at=fetched_at, issued_at=issued, sha=fetched.status.sha
    )
    return PibResult(status=status, notams=kept, superseded=dropped)


# --- prefilter (G13) ---------------------------------------------------------------


def _drop_reason(notam: ParsedNotam, lz: LatLon, window: Window, rules: Rules) -> str | None:
    """Why the prefilter drops this NOTAM, or None to keep it."""
    start, end = window
    edge = circle_edge_distance_nm(lz[0], lz[1], notam.centre_lat, notam.centre_lon, notam.radius_nm)
    if edge > _number(rules.local_area):
        return f"Q-circle edge is {edge:.2f} NM from the LZ, beyond the {_number(rules.local_area):g} NM local area"
    if rules.notam_max_lower_fl.breached(notam.lower_fl):
        return f"Q-line lower limit FL{notam.lower_fl:03d} is above FL{_number(rules.notam_max_lower_fl):03g}"
    if notam.valid_from >= end:
        return f"starts {utc_text(notam.valid_from)}, at or after the window end {utc_text(end)}"
    if notam.valid_to is not None and notam.valid_to <= start:
        return f"ended {utc_text(notam.valid_to)}, at or before the window start {utc_text(start)}"
    return None


def prefilter(
    notams: list[ParsedNotam], lz: LatLon, window: Window, *, rules: Rules = RULES
) -> tuple[list[ParsedNotam], list[IgnoredNotam]]:
    """Keep a NOTAM only when all three hold; return (kept, dropped with reasons).

    1. The LZ is within the local area (rules.yaml) of the Q-circle's edge. Inside the circle counts as 0.
    2. The Q-line lower limit is not above the FL cap (rules.yaml).
    3. B-C overlaps the half-open window [start, end). A PERM C overlaps any window after B.
    """
    kept: list[ParsedNotam] = []
    dropped: list[IgnoredNotam] = []
    for notam in notams:
        reason = _drop_reason(notam, lz, window, rules)
        if reason is None:
            kept.append(notam)
        else:
            dropped.append(IgnoredNotam(notam_id=notam.id, reason=reason))
    return kept, dropped


# --- code-only sort (G23) ----------------------------------------------------------


def _inactive_reason(notam: ParsedNotam, span: Window, lz: LatLon) -> tuple[str | None, str | None]:
    """(why the NOTAM is not active in the span or None, a note when item D could not be read)."""
    start, end = span
    if notam.valid_from >= end:
        return f"not active in the span: starts {utc_text(notam.valid_from)}", None
    if notam.valid_to is not None and notam.valid_to <= start:
        return f"not active in the span: ended {utc_text(notam.valid_to)}", None
    if not notam.item_d:
        return None, None
    # Clip B-C to the span first: any interval left is an overlap, and a PERM C needs an end.
    clipped_from = max(notam.valid_from, start)
    clipped_to = end if notam.valid_to is None else min(notam.valid_to, end)
    schedule = parse_schedule(notam.item_d, clipped_from, clipped_to, lz[0], lz[1])
    if not schedule.parsed:
        return None, schedule.note
    if not schedule.intervals:
        return f"not active in the span: item D schedule has no time in it ({' '.join(notam.item_d.split())})", None
    return None, None


def _ignore_reason(notam: ParsedNotam, rules: Rules) -> str | None:
    """The first coded-field reason to ignore, in the G23 order, or None."""
    subject = notam.subject
    if rules.notam_ignore_traffic.breached(notam.traffic):
        return f"Q traffic {notam.traffic} (IFR only)"
    if rules.notam_ignore_purpose.breached(notam.purpose):
        return f"purpose {notam.purpose} (checklist)"
    if notam.type is NotamType.CANCEL:
        target = f" {notam.replaces}" if notam.replaces else ""
        return f"NOTAMC cancelling{target}; a cancellation never produces an action"
    if notam.series == SECURITY_SERIES and rules.notam_ignore_security_subjects.breached(subject):
        return f"series {notam.series} subject {subject}: security advice to air operators"
    if rules.notam_wide_ignore_subjects.breached(subject) and rules.notam_area_wide_nm.breached(notam.radius_nm):
        return f"subject {subject} with radius {notam.radius_nm:g} NM: FIR-wide AIS admin"
    if rules.notam_ignore_subject_letters.breached(subject[:1]) and not rules.notam_keep_subjects.breached(subject):
        return f"subject {subject} cannot affect a paramotor"
    return None


def sort_notam(notam: ParsedNotam, span: Window, lz: LatLon, *, rules: Rules = RULES) -> SortResult:
    """Decide IGNORE, VIA_AIRSPACE, AWARENESS or TO_READER for one NOTAM in one span [start, end). First match wins.

    1. IGNORE: not active in the span (B/C, then D) · traffic I only · purpose K · cancelled ·
       series V subject RO · subject that cannot affect a paramotor (with its kept exceptions).
    2. Restriction subject (RD, RR, RP, RT): a designator in item E -> VIA_AIRSPACE; none -> TO_READER.
    3. Jamming subject (GW) -> AWARENESS.
    4. Radius at or above the area-wide size -> AWARENESS, unless item E draws an area (G16): a polygon or a circle in
       the text, or coordinate text that did not make one (Broken). Those go to step 5, and notam_action() measures
       the LZ against the drawn shape, or the Q-circle for Broken, with no area-wide cap. A route in item E (coordinates
       with times) or no coordinates keeps this step.
    5. Everything else -> TO_READER.

    Step 1's AF-admin (radius at or above the area-wide size) and series V rules run first for every NOTAM: those are
    FIR-wide admin and security advice, which no drawn area makes relevant.

    Run it after the prefilter. Series Q is not ignored.
    """
    inactive, note = _inactive_reason(notam, span, lz)
    if inactive:
        return SortResult(action=SortAction.IGNORE, reason=inactive)
    ignore = _ignore_reason(notam, rules)
    if ignore:
        return SortResult(action=SortAction.IGNORE, reason=ignore, schedule_note=note)

    subject = notam.subject
    if rules.notam_restriction_subjects.breached(subject):
        found = find_designators(notam.item_e)
        if found:
            reason = f"restriction {subject} naming {', '.join(found)}: handled through the airspace polygon"
            return SortResult(action=SortAction.VIA_AIRSPACE, reason=reason, designators=found, schedule_note=note)
        return SortResult(
            action=SortAction.TO_READER, reason=f"restriction {subject} with no designator in item E", schedule_note=note
        )
    if rules.notam_awareness_subjects.breached(subject):
        return SortResult(action=SortAction.AWARENESS, reason=f"subject {subject} (GNSS or radio jamming)", schedule_note=note)
    if rules.notam_area_wide_nm.breached(notam.radius_nm):
        area = area_from_item_e(notam)
        if isinstance(area, Shape):
            reason = f"radius {notam.radius_nm:g} NM is area-wide, but item E draws a {area.kind}: the LZ is measured against it"
            return SortResult(action=SortAction.TO_READER, reason=reason, schedule_note=note)
        if isinstance(area, Broken):
            reason = f"radius {notam.radius_nm:g} NM is area-wide, but item E coordinates are not usable ({area.reason}): the Q-circle is used"
            return SortResult(action=SortAction.TO_READER, reason=reason, schedule_note=note)
        reason = f"radius {notam.radius_nm:g} NM is area-wide"
        return SortResult(action=SortAction.AWARENESS, reason=reason, schedule_note=note)
    return SortResult(action=SortAction.TO_READER, reason=f"subject {subject} may affect the flight", schedule_note=note)
