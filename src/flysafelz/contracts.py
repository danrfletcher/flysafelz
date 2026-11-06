"""Pydantic contracts shared by every module. Plain models and enums only.

The verdict has three values and no bare "go" verdict exists
(display_verdict builds the strings). Models forbid extra fields.
"""

import re
from enum import StrEnum
from typing import Annotated, Literal, Self

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, model_validator

STRICT = ConfigDict(extra="forbid")
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


class NotamReading(BaseModel):
    """The semantic fields a reader fills (ticket 3), or a fixture writes by hand."""

    model_config = STRICT

    id: str
    category: Category
    activity: str
    geometry: Annotated[Circle | Polygon, Field(discriminator="kind")] | None = None  # None: use the Q-circle
    floor: Limit
    ceiling: Limit
    schedule: str  # D-item text, normalised
    exempts_paramotors: bool = False
    contact: Contact
    activates_designators: list[str] = Field(default_factory=list)  # e.g. ["EGD496A"]
    prohibitive_reason: ProhibitiveReason = ProhibitiveReason.NONE


class NotamAction(StrEnum):
    """What a NOTAM means for the pilot in one segment. notam_action() (PR-6) owns the logic.

    First match wins:
      IGNORE            not active in the slot, beyond the local area, exempts paramotors, or irrelevant
      NO_FLY            over the LZ from the surface and prohibitive (or cooperative with no phone or email)
      CONTACT_FIRST     over the LZ from the surface, cooperative, with a phone or email contact
      AVOID_SECTOR      within the local area but not over the LZ, or over it with a floor of 0-5,000 ft
      AWARENESS         floor 5,000-10,000 ft: shown in the answer
      AWARENESS_MAP     floor above 10,000 ft: shown on the map and the collapsed list only
      UNREAD_OVER_LZ    no reading, and the NOTAM lies over the LZ from the surface (Q8):
                        the segment is UNVERIFIED and the E text is printed
    """

    IGNORE = "IGNORE"
    NO_FLY = "NO_FLY"
    CONTACT_FIRST = "CONTACT_FIRST"
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

