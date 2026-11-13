"""Pydantic contracts shared by every module. Plain models and enums only.

The verdict has three values and no bare "go" verdict exists
(display_verdict builds the strings). Models forbid extra fields.
"""

import re
from decimal import Decimal
from enum import StrEnum
from typing import Annotated, Literal, Self

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, model_validator

from flysafelz.designators import find_designators
from flysafelz.sup_refs import find_sup_references

STRICT = ConfigDict(extra="forbid")
# The one NOTAM-id pattern in src/: a series letter, one to four digits, a slash, a two-digit year (H7171/26, Q172/26, V32/26). Real ids have one to four
# digits, so a four-digit-only pattern would let an invented short id through unchecked. The guard and every later reader of an id import this constant.
NOTAM_ID_RE = re.compile(r"\b[A-Z]\d{1,4}/\d{2}\b")
_Q_CODE = re.compile(r"Q?(?P<subject>..)(?P<condition>..)")


# --- verdicts and tiers ---------------------------------------------------


class Verdict(StrEnum):
    """The only verdicts. Hour order, worst first: NO_GO, UNVERIFIED, PILOT_DISCRETION."""

    NO_GO = "NO_GO"
    UNVERIFIED = "UNVERIFIED"
    PILOT_DISCRETION = "PILOT_DISCRETION"


class Tier(StrEnum):
    """How far an hour can be trusted. Only VERDICT hours carry a Verdict."""

    VERDICT = "VERDICT"
    OUTLOOK = "OUTLOOK"
    TOO_FAR = "TOO_FAR"


_PILOT_DISCRETION_TEXT = "PILOT DISCRETION"


def display_verdict(verdict: Verdict, approaching_limits: bool, has_conditions: bool) -> str:
    """Build the display string for a verdict.

    NO-GO and UNVERIFIED stand alone. PILOT DISCRETION takes the qualifiers
    APPROACHING LIMITS and WITH CONDITIONS, one, both or neither.
    """
    if verdict is Verdict.NO_GO:
        return "NO-GO"
    if verdict is Verdict.UNVERIFIED:
        return "UNVERIFIED"
    qualifiers = []
    if approaching_limits:
        qualifiers.append("APPROACHING LIMITS")
    if has_conditions:
        qualifiers.append("WITH CONDITIONS")
    if not qualifiers:
        return _PILOT_DISCRETION_TEXT
    return f"{_PILOT_DISCRETION_TEXT} \u2013 " + " \u00b7 ".join(qualifiers)


type Position = Literal["over", "at", "under"]
_ABOVE_IS_PAST = frozenset({">", ">="})
_BELOW_IS_PAST = frozenset({"<", "<="})


def position_word(value: float | Decimal, limit: float | Decimal, comparison: str) -> Position:
    """Where a value sits against its limit, as one word. The only place this is decided: the guard and the tool phrase and limit line (PR-11) all call it.

    over   past the limit, the way its comparison looks (6.04 against `> 6`; 9.3 against `>= 9`; 4.0 against `< 5`). "above" says the same.
    at     exactly the limit's number (9.0 against `>= 9`; 6.0 against `> 6`, which is not yet over). Exactness is the caller's: both numbers in the one unit.
    under  has not reached the limit (5.9 against `> 6`; 8.7 against `>= 9`). "below" says the same.

    Only the four ordering comparisons are limits; any other (`in`, `=`) has no position and raises ValueError.
    """
    if comparison not in _ABOVE_IS_PAST | _BELOW_IS_PAST:
        raise ValueError(f"comparison {comparison!r} is not a limit, so a value has no position against it")
    if value == limit:
        return "at"
    past = value > limit if comparison in _ABOVE_IS_PAST else value < limit
    return "over" if past else "under"


# --- source status --------------------------------------------------------


class SourceName(StrEnum):
    WEATHER = "weather"
    AIRSPACE = "airspace"
    NOTAMS = "notams"


class SourceState(StrEnum):
    OK = "ok"
    FAILED = "failed"
    STALE = "stale"


class SourceStatus(BaseModel):
    """Health of one data source. Weather and airspace set fetched_at; NOTAMs set issued_at (PIB Issued) and, from the archive, sha."""

    model_config = STRICT

    source: SourceName
    state: SourceState
    reason: str = ""
    fetched_at: AwareDatetime | None = None
    issued_at: AwareDatetime | None = None
    sha: str | None = None  # NOTAMs from the archive: the commit the PIB came from; None for live and other sources


# --- NOTAMs ----------------------------------------------------------------


class NotamType(StrEnum):
    NEW = "N"
    REPLACE = "R"
    CANCEL = "C"


class ParsedNotam(BaseModel):
    """What the code parses from one PIB entry (G12). No model involved."""

    model_config = STRICT

    id: str  # e.g. "H7171/26"
    series: str
    number: int
    year: int
    type: NotamType
    replaces: str | None = None  # id replaced (R) or cancelled (C)
    nof: str  # originating office
    fir: str
    q_code: str  # five letters, e.g. "QRDCA": Q, subject (RD), condition (CA)
    traffic: str
    purpose: str
    scope: str
    lower_fl: int
    upper_fl: int
    centre_lat: float
    centre_lon: float
    radius_nm: float
    item_a: str
    valid_from: AwareDatetime
    valid_to: AwareDatetime | None = None  # None when permanent
    permanent: bool = False
    est: bool = False  # the end time is an estimate (EST)
    item_d: str | None = None  # schedule text
    item_e: str
    item_f: str | None = None
    item_g: str | None = None
    section: str  # the PIB section it was first seen in
    sections: list[str] = Field(default_factory=list)  # every section it appears in, first seen first

    @property
    def subject(self) -> str:
        """The two-letter subject of the Q code (RD in QRDCA); empty when the code is too short."""
        match = _Q_CODE.match(self.q_code)
        return match.group("subject") if match else ""

    @property
    def condition(self) -> str:
        """The two-letter condition of the Q code (CA in QRDCA); empty when the code is too short."""
        match = _Q_CODE.match(self.q_code)
        return match.group("condition") if match else ""

    @property
    def sup_references(self) -> list[str]:
        """The AIP SUP references in item E ("AIP SUP 108/2026"), found by code. Not a field: never serialised, never model-filled."""
        return find_sup_references(self.item_e)

    @property
    def designators(self) -> list[str]:
        """The airspace designators in item E (EGD496A), found by code, for display only. Activation uses NotamReading.activates_designators."""
        return find_designators(self.item_e)


class Category(StrEnum):
    PROHIBITIVE = "prohibitive"
    COOPERATIVE = "cooperative"
    INFORMATIONAL = "informational"


class ContactKind(StrEnum):
    PHONE = "phone"
    EMAIL = "email"
    RADIO_ONLY = "radio-only"
    NONE = "none"


class LimitUnit(StrEnum):
    FT = "ft"
    FL = "FL"


class Datum(StrEnum):
    SFC = "SFC"
    AGL = "AGL"
    AMSL = "AMSL"


class ProhibitiveReason(StrEnum):
    BYELAWS = "byelaws"
    ENTRY_PROHIBITED = "entry prohibited"
    LIVE_FIRING = "live firing"
    NONE = "none"


class Circle(BaseModel):
    model_config = STRICT

    kind: Literal["circle"] = "circle"
    lat: float
    lon: float
    radius_nm: float


class Polygon(BaseModel):
    model_config = STRICT

    kind: Literal["polygon"] = "polygon"
    points: list[tuple[float, float]]  # (lat, lon) pairs


class Limit(BaseModel):
    """A floor or ceiling. For SFC the value is 0."""

    model_config = STRICT

    value: float
    unit: LimitUnit
    datum: Datum


class Contact(BaseModel):
    model_config = STRICT

    kind: ContactKind
    value: str | None = None


class AppliesTo(StrEnum):
    """Which aircraft a NOTAM is addressed to. UAS_ONLY and AERODROME_USERS never concern a paramotor."""

    ALL_AIRCRAFT = "ALL_AIRCRAFT"
    UAS_ONLY = "UAS_ONLY"
    AERODROME_USERS = "AERODROME_USERS"
    NAMED_CLASSES = "NAMED_CLASSES"


class ExemptionCovers(StrEnum):
    """Who an exemption lets off.

    PARAMOTOR_CLASS: paramotors, paragliders, microlights, SSDR, foot-launched, or all light /
    non-transponder aircraft, judged on meaning not wording. CONDITIONAL: only on a condition (in
    contact with ATC, with permission, "exceptions in the AIC", emergency services): never drops
    the NOTAM. OTHER: some other class of aircraft.
    """

    PARAMOTOR_CLASS = "PARAMOTOR_CLASS"
    CONDITIONAL = "CONDITIONAL"
    OTHER = "OTHER"


class Exemption(BaseModel):
    """One exemption in the E text. quote is the sentence from the E text, word for word (it is checked)."""

    model_config = STRICT

    quote: str
    covers: ExemptionCovers


class NotamReading(BaseModel):
    """The semantic fields a reader fills (ticket 3), or a fixture writes by hand.

    floor and ceiling are the heights the E text states. None means the E text states none: notam_action() then
    uses F/G, then the Q-line.
    """

    model_config = STRICT

    id: str
    category: Category
    activity: str
    geometry: Annotated[Circle | Polygon, Field(discriminator="kind")] | None = None  # no longer used to locate a NOTAM (G16): notam_action() reads item E itself
    floor: Limit | None  # as the E text states it; None: E states none
    ceiling: Limit | None  # as the E text states it; None: E states none
    schedule: str  # D-item text, normalised
    applies_to: AppliesTo = AppliesTo.ALL_AIRCRAFT
    exemptions: list[Exemption] = Field(default_factory=list)
    contact: Contact
    activates_designators: list[str] = Field(default_factory=list)  # e.g. ["EGD496A"]
    prohibitive_reason: ProhibitiveReason = ProhibitiveReason.NONE
    prohibitive_quote: str | None = None  # the E-text words the reason rests on, word for word; None when the reason is none


class NotamAction(StrEnum):
    """What a NOTAM means for the pilot in one segment. notam_action() (PR-6) owns the logic.

    First match wins (row 1 first; rows 2a-2d and 3 only over the LZ from the surface). The Q-code
    subject (ParsedNotam.subject) keys rows 2a-2d; a danger area (RD) is never ruled in by the data:
      IGNORE            1   not active in the slot, beyond the local area, exempts paramotors, or irrelevant
      NO_FLY            2a  subject RR, RP or RT (restricted, prohibited or temporary restricted area)
                        2b  subject RD with a prohibitive reason (live firing, entry prohibited, byelaws),
                            quoted from the E text
                        2   any other subject: prohibitive (category or reason), or cooperative with no
                            phone or email contact
      CONTACT_FIRST     2c  subject RD, no prohibitive reason, with a phone or email contact
                        3   any other subject: cooperative (or a conditional exemption), with a phone or
                            email contact
      CHECK_SUP         2d  subject RD, no prohibitive reason, no phone or email contact: read the SUP /
                            AIP ENR 5.1 entry before flight (shows as PILOT DISCRETION - WITH CONDITIONS)
      AVOID_SECTOR      4   within the local area but not over the LZ, or over it with a floor of 0-5,000 ft
      AWARENESS         5   floor 5,000-10,000 ft: shown in the answer
      AWARENESS_MAP     6   floor above 10,000 ft: shown on the map and the collapsed list only
      UNREAD_OVER_LZ        no reading, and the NOTAM lies over the LZ from the surface (Q8):
                            the segment is UNVERIFIED and the E text is printed
    For a danger area the reader's category is ignored: only prohibitive_reason and the contact decide.
    """

    IGNORE = "IGNORE"
    NO_FLY = "NO_FLY"
    CONTACT_FIRST = "CONTACT_FIRST"
    CHECK_SUP = "CHECK_SUP"
    AVOID_SECTOR = "AVOID_SECTOR"
    AWARENESS = "AWARENESS"
    AWARENESS_MAP = "AWARENESS_MAP"
    UNREAD_OVER_LZ = "UNREAD_OVER_LZ"


class NotamActionResult(BaseModel):
    """The action for one NOTAM plus a one-line reason. bearing_deg is set for sector actions."""

    model_config = STRICT

    notam_id: str
    action: NotamAction
    reason: str
    bearing_deg: float | None = None
    # An AWARENESS the pilot must be warned about as a condition (an informational NOTAM over the LZ from the surface, Q13),
    # not just listed. Set by notam_action(), read by assess; left out of dumps so no output changes shape.
    warning: bool = Field(default=False, exclude=True)


# --- assessment -------------------------------------------------------------


class WeatherValues(BaseModel):
    """Raw weather for an hour row. A value is None when the source did not supply it."""

    model_config = STRICT

    wind_10m: float | None = None  # mph
    wind_80m: float | None = None
    wind_120m: float | None = None
    gust_10m: float | None = None
    direction_10m: float | None = None  # degrees
    precipitation: float | None = None  # mm/h
    precipitation_probability: float | None = None  # %
    cape: float | None = None  # J/kg
    cloud_cover: float | None = None  # %
    cloud_cover_low: float | None = None  # %
    visibility: float | None = None  # m
    shortwave_radiation: float | None = None  # W/m2
    pressure_msl: float | None = None  # hPa


class Caution(BaseModel):
    """A limit that is being approached, with its numbers (e.g. gust spread 5 mph, limit 6)."""

    model_config = STRICT

    rule_id: str
    text: str
    value: float | None = None
    limit: float | None = None


class Blocker(BaseModel):
    """A reason the segment is NO_GO. source is weather, daylight, airspace or notam."""

    model_config = STRICT

    source: str
    text: str


class Segment(BaseModel):
    """A span inside one UTC hour row. Split where a daylight or NOTAM edge falls (Q10)."""

    model_config = STRICT

    start: AwareDatetime
    end: AwareDatetime
    verdict: Verdict
    approaching_limits: bool = False
    conditions: list[str] = Field(default_factory=list)
    cautions: list[Caution] = Field(default_factory=list)
    blockers: list[Blocker] = Field(default_factory=list)
    notam_actions: list[NotamActionResult] = Field(default_factory=list)  # non-IGNORE only


class HourAssessment(BaseModel):
    """One UTC hour. OUTLOOK and TOO_FAR hours have no segments; they keep the raw weather."""

    model_config = STRICT

    hour: AwareDatetime
    tier: Tier
    segments: list[Segment] = Field(default_factory=list)
    weather: WeatherValues = Field(default_factory=WeatherValues)


class FlyableWindow(BaseModel):
    model_config = STRICT

    start: AwareDatetime
    end: AwareDatetime


class IgnoredNotam(BaseModel):
    """Debug entry: a NOTAM the code dropped, and why."""

    model_config = STRICT

    notam_id: str
    reason: str


class Assessment(BaseModel):
    """The whole window.

    overall_verdict is the worst VERDICT-tier segment. With no VERDICT-tier
    hours it is None and overall_tier (OUTLOOK or TOO_FAR) says why.
    """

    model_config = STRICT

    window_start: AwareDatetime
    window_end: AwareDatetime
    hours: list[HourAssessment]
    overall_verdict: Verdict | None
    overall_tier: Tier
    flyable_windows: list[FlyableWindow] = Field(default_factory=list)
    source_statuses: list[SourceStatus] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    ignored_notams: list[IgnoredNotam] = Field(default_factory=list)

    @model_validator(mode="after")
    def _verdict_matches_tier(self) -> Self:
        has_verdict = self.overall_verdict is not None
        if has_verdict != (self.overall_tier is Tier.VERDICT):
            raise ValueError("overall_verdict is set exactly when overall_tier is VERDICT")
        return self

