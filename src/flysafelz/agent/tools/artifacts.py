"""The structured artifact each data tool returns beside its text, for the guard and the UI (FR6, FR16).

The text is what the model reads. The artifact is the same facts as data: every value with its rule id and limit, every finding with its
datums, every NOTAM with its action, and the GeoJSON Features the map draws. Both pass through the redaction boundary
(`agent/redaction.py`), so no real phone number is in either.

    WeatherArtifact   get_weather: hours, each value paired with the rule ids and limits it is checked against
    AirspaceArtifact  get_airspace: findings, the lowest ceiling over the LZ, the notes
    NotamsArtifact    get_notams: the counts and one entry per NOTAM

All three carry `source` (its state, reason, fetch time and the manual check), the window in UTC, and `overlays` (the LZ's local-area circle
and, for the two that have shapes, the features to draw).
"""

from datetime import datetime
from typing import Literal

from pydantic import AwareDatetime, BaseModel, ConfigDict

from flysafelz.agent.state import OverlayFeature

SourceKey = Literal["weather", "airspace", "notams"]  # SourceName's values
SourceStateKey = Literal["ok", "failed", "stale"]  # SourceState's values


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class SourceInfo(_Model):
    """One source as the tool found it. `check_by_hand` is set when the source is not OK: where the pilot can look.

    `source` and `state` are the plain values of `contracts.SourceName` and `SourceState`, not the enums: the artifact is stored on the
    tool message in the thread's checkpoint, and a checkpoint will not load a class it was not told about (LANGGRAPH_STRICT_MSGPACK).
    A test pins the two lists to the enums.
    """

    source: SourceKey
    state: SourceStateKey
    reason: str
    fetched_at: AwareDatetime | None
    check_by_hand: str | None


class DataArtifact(_Model):
    """What every data tool's artifact has."""

    source: SourceInfo
    window_start: AwareDatetime
    window_end: AwareDatetime
    window_assumed: bool
    overlays: list[OverlayFeature]


# --- get_weather ----------------------------------------------------------------------------------------------------


class RuleLimit(_Model):
    """One limit a value is checked against, exactly as rules.yaml has it."""

    rule_id: str
    kind: str  # "caution" or "no-go"
    comparison: str
    limit: float
    unit: str


class WeatherValue(_Model):
    """One value of one hour (None: not forecast) with the limits that apply to it."""

    name: str
    label: str
    value: float | None
    unit: str
    limits: list[RuleLimit]


class WeatherHour(_Model):
    hour: AwareDatetime
    label: str
    values: list[WeatherValue]


class WeatherArtifact(DataArtifact):
    tool: Literal["get_weather"] = "get_weather"
    hours: list[WeatherHour]


# --- get_airspace ---------------------------------------------------------------------------------------------------


class AirspaceItem(_Model):
    """One airspace finding. floor and ceiling are written with their datum as the volume publishes them.

    A danger area openAIP does not have is placed by its activating NOTAM (`from_notam`): its floor and ceiling are then the reading's (None: the
    NOTAM states none or, when `notam_read` is False, the reader has not read it, so nothing is known and the floor is taken as SFC, as
    `assess_airspace` does). `approx` says the shape drawn is the NOTAM's Q-circle. `over_lz` is None only when nothing places the volume.
    """

    name: str
    type: str | None
    outcome: str
    reason: str
    condition: str | None
    floor: str | None
    ceiling: str | None
    over_lz: bool | None
    distance_nm: float | None
    bearing_deg: float | None
    compass: str | None
    floor_qnh_ft: float | None
    ceiling_qnh_ft: float | None
    from_notam: str | None = None
    approx: bool | None = None
    notam_read: bool | None = None


class AirspaceArtifact(DataArtifact):
    tool: Literal["get_airspace"] = "get_airspace"
    findings: list[AirspaceItem]
    ceiling_qnh_ft: float | None
    notes: list[str]


# --- get_notams -----------------------------------------------------------------------------------------------------


class ReadingFacts(_Model):
    """The reader's facts for a NOTAM that was read (the contact value is redacted with the rest of the artifact)."""

    category: str
    activity: str
    floor: str | None
    ceiling: str | None
    schedule: str
    contact_kind: str
    contact: str | None
    activates: list[str]


class NotamItem(_Model):
    """One NOTAM: its action from code, the reading (None: unread, and `unread_reason` says why) and the raw E text as published."""

    id: str
    action: str
    reason: str
    bearing_deg: float | None
    reading: ReadingFacts | None
    unread_reason: str | None
    e_text: str


class NotamCounts(_Model):
    """The counts the text prints. `within` is the NOTAMs to act on or check in the local area; awareness and ignored are the other entries."""

    within: int
    awareness: int
    ignored: int


class NotamsArtifact(DataArtifact):
    tool: Literal["get_notams"] = "get_notams"
    counts: NotamCounts
    notams: list[NotamItem]
    assumptions: list[str]
    issued_at: datetime | None
