"""The structured artifact each data tool returns beside its text, for the guard and the UI (FR6, FR16).

The text is what the model reads. The artifact is the same facts as data: every value with its rule id and limit, every finding with its
datums, every NOTAM with its action, and the GeoJSON Features the map draws. Both pass through the redaction boundary
(`agent/redaction.py`), so no real phone number is in either.

    WeatherArtifact   get_weather: hours, each value paired with the rule ids and limits it is checked against
    AirspaceArtifact  get_airspace: findings, the lowest ceiling over the LZ, the notes
    NotamsArtifact    get_notams: the counts and one entry per NOTAM
    AssessArtifact    assess: every segment with its display string, conditions, cautions and NOTAM actions, the overall verdict, the flyable spans and the counts
    GuidanceArtifact  search_guidance: the query and the chunks that came back, each with its document, edition, section number, title and text

The first three carry `source` (its state, reason, fetch time and the manual check), the window in UTC, and `overlays` (the LZ's local-area circle
and, for the two that have shapes, the features to draw). `assess` carries all three sources in `sources` instead, since it judges them together. The guidance artifact has no source, window or overlays: it is
published text, not a reading of the landing zone.
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


# --- assess ---------------------------------------------------------------------------------------------------------

TierKey = Literal["VERDICT", "OUTLOOK", "TOO_FAR"]  # contracts.Tier's values
VerdictKey = Literal["NO_GO", "UNVERIFIED", "PILOT_DISCRETION"]  # contracts.Verdict's values


class AssessCaution(_Model):
    """A limit being approached: the rule, the value and the limit it is checked against, and the unit rules.yaml gives that rule (None: it has none).

    `label` is what the value is called in `text` (`gust spread`; None: `text` is not a value against a limit, as for twilight). It is worked out once, by
    the tool, from the finding's own text, so the limit line (`answer.limit_lines`) never reads text.
    """

    rule_id: str
    text: str
    value: float | None
    limit: float | None
    unit: str | None
    label: str | None = None


class AssessBlocker(_Model):
    """A reason the segment is NO-GO: the source it came from (weather, daylight, airspace, notam) and ticket 2's words.

    A weather blocker also carries the numbers behind its words, from the same `weather.check_row` finding the assessment used: `rule_id`, `label`,
    `value`, `limit` and `unit` (all None for any other blocker). The text keeps ticket 2's wording.
    """

    source: str
    text: str
    rule_id: str | None = None
    label: str | None = None
    value: float | None = None
    limit: float | None = None
    unit: str | None = None


class AssessNotam(_Model):
    """A NOTAM that bears on a segment, with the action code gave it there."""

    id: str
    action: str
    bearing_deg: float | None
    reason: str


class AssessSegment(_Model):
    """One span of the assessment, as `agent.state.Segment` needs it, plus the evidence behind it.

    `start` and `end` are ISO 8601 times with a UTC offset, on the clock of the thread's `tz`; `tz_abbr` is the abbreviation of that span's own instant.
    `verdict` is the plain value of `contracts.Verdict`, and None for an OUTLOOK or TOO_FAR hour, which has no verdict and so no `display`.
    `display` is always `display_verdict(verdict, approaching_limits, with_conditions)`. `with_conditions` is whether `conditions` is not empty.
    An OUTLOOK hour keeps its raw wind, gust and rain chance (None: not forecast) and the day to ask again; a VERDICT or TOO_FAR hour has none.
    """

    start: str
    end: str
    tz_abbr: str
    tier: TierKey
    verdict: VerdictKey | None
    approaching_limits: bool
    with_conditions: bool
    display: str | None
    conditions: list[str]
    cautions: list[AssessCaution]
    blockers: list[AssessBlocker]
    notams: list[AssessNotam]
    wind_10m: float | None = None
    gust_10m: float | None = None
    rain_chance: float | None = None
    ask_again_on: str | None = None


class AssessOverall(_Model):
    """The overall verdict: the worst VERDICT-tier segment's machine fields and the display string built from them."""

    verdict: VerdictKey
    approaching_limits: bool
    with_conditions: bool
    display: str


class AssessBlock(_Model):
    """A run of consecutive OUTLOOK hours of one London day, or of TOO_FAR hours, as the text prints it.

    The trend of an OUTLOOK block is the lowest and highest forecast wind, gust and rain chance over its hours (None: no hour has a value), with the
    units rules.yaml gives. `ask_again_on` is the date (YYYY-MM-DD) its hours can carry a verdict; None when they are already inside the verdict
    horizon, which means the forecast is incomplete for them and asking again later may help.
    """

    tier: Literal["OUTLOOK", "TOO_FAR"]
    start: str
    end: str
    tz_abbr: str
    hours: int
    wind_min: float | None = None
    wind_max: float | None = None
    wind_unit: str | None = None
    gust_min: float | None = None
    gust_max: float | None = None
    gust_unit: str | None = None
    rain_min: float | None = None
    rain_max: float | None = None
    rain_unit: str | None = None
    ask_again_on: str | None = None


class AssessSpan(_Model):
    """One flyable window, in `tz`, and its length in whole minutes (the text prints it as `X h YY min`)."""

    start: str
    end: str
    tz_abbr: str
    minutes: int


class AssessCounts(_Model):
    """The numbers the text prints. `notams_within` is the distinct NOTAM ids across all segments' actions, within `local_area_nm`."""

    notams_within: int
    local_area_nm: float
    hours_assessed: int
    hours_outlook: int
    hours_too_far: int
    flyable_windows: int


class AssessArtifact(_Model):
    """What `assess` returns beside its text. The guard pairs each caution's value with its limit from here, never from the text.

    `overall` is None when no hour has a verdict, and `overall_tier` then says why. `data_as_of` is the text's fetch time (`07:30 GMT`), from
    `fetched_at`, the oldest fetch among the turn's sources (None: none has one). `sources` holds the three sources as `Assessment.source_statuses`
    judged them, so a PIB that is too old is `stale` here. `tz` is the thread's zone.
    """

    tool: Literal["assess"] = "assess"
    tz: str
    window_start: AwareDatetime
    window_end: AwareDatetime
    window_assumed: bool
    overlays: list[OverlayFeature]
    segments: list[AssessSegment]
    blocks: list[AssessBlock]
    overall: AssessOverall | None
    overall_tier: TierKey
    flyable: list[AssessSpan]
    counts: AssessCounts
    sources: list[SourceInfo]
    pib_issued_at: datetime | None
    assumptions: list[str]
    fetched_at: AwareDatetime | None
    data_as_of: str | None


# --- search_guidance ------------------------------------------------------------------------------------------------


class GuidanceChunk(_Model):
    """One chunk of published guidance as the tool printed it. The guard treats `section` as a section number a tool printed (FR6)."""

    document: str
    edition: str
    section: str
    title: str
    text: str


class GuidanceArtifact(_Model):
    """What `search_guidance` returns beside its text: the query and the chunks that came back, best first. `chunks` is empty when nothing was found or the search could not run."""

    tool: Literal["search_guidance"] = "search_guidance"
    query: str
    chunks: list[GuidanceChunk]
