"""The agent's contract: what a request carries in, what a turn gives back, what the thread keeps privately, and the run context.

    RequestState   the thread state (a TypedDict that extends LangChain's AgentState), with each field's visibility
    RunContext     the per-run context (not state, never shown to the model): `now`, `case`, and the case-mode switches
    run_now(ctx)   the clock a run uses
    OverlayFeature, Assessment, Segment, VerdictValue   the pilot-facing output shapes (what the map and the badges are drawn from)
    lz_radius_feature(lat, lon)   the LZ's local-area circle as an OverlayFeature

Who can see what (LangChain's OmitFromSchema annotations decide the graph's input and output schemas):

    messages                          input and output
    lat, lon, tz                      input only, and they last one turn (see below)
    answer, guard, overlays, assessment   output only: code writes them, a caller cannot send them
    phone_map, thread_lz              private: kept in the thread's checkpoint, in neither schema (real phone numbers must never reach the contract)

The input schema lists `lat` and `lon` as required and gives `tz` the default Europe/London, so Studio's form asks for them. That is
the form only: LangGraph does not check a request against it. The input check does, and a follow-up turn may leave `lat` and `lon` out
(the thread's LZ is used) or `tz` out (the thread's zone is used, else Europe/London).

`lat`, `lon` and `tz` are an EphemeralValue channel: a value sent with a request is there for the input check and is gone
after it. It is not kept in the thread, so a rejected request leaves nothing behind that would reject the next one. The
input check copies the accepted values into `thread_lz` (the thread's one landing zone and its time zone), and every later
step reads the LZ and the time zone from there.
"""

import math
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Literal, NotRequired, Required, Self, TypedDict

from langchain.agents import AgentState
from langchain.agents.middleware.types import OmitFromInput, OmitFromOutput, PrivateStateAttr
from langgraph.channels.ephemeral_value import EphemeralValue
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from flysafelz.contracts import Tier, Verdict, display_verdict
from flysafelz.geo import FULL_CIRCLE, HALF_TURN, MIN_POLYGON_POINTS, QUARTER_TURN
from flysafelz.rules import RULES
from flysafelz.shapes import CIRCLE_VERTICES, circle_ring  # noqa: F401 - CIRCLE_VERTICES is re-exported: it is the vertex count of every circle

# --- run context -----------------------------------------------------------------------------------------------------


DEFAULT_TZ = "Europe/London"  # the time zone of a request that names none
ReaderName = Literal["fixture", "live"]
DropName = Literal["weather", "airspace", "pib"]


@dataclass(frozen=True)
class RunContext:
    """What a caller sets for one run, through the graph's `context` (never through the model, never in the thread state).

    now           the run's clock; None means the wall clock. Must be timezone-aware.
    case          a committed case (fixtures/<case>/): the tools load its frozen sources instead of fetching. None is live mode.
    reader        case mode only: "fixture" reads the case's readings.yaml, "live" has the NOTAM reader read the case's PIB.
    reader_cache  case mode with reader "live" only: the stored-readings file the reader uses (None: the default file).
    drop          case mode only: sources to treat as failed, as the case's own `fixtures.drop` does.

    The defaults leave every existing case-mode run as it was. A caller that builds this from JSON (Studio, an eval harness) may send
    `now` as an ISO 8601 string, `reader_cache` as a path string and `drop` as a list; they are turned into a datetime, a Path and a tuple
    here. The input check refuses a value that is still the wrong kind.
    """

    now: datetime | None = None
    case: str | None = None
    reader: ReaderName = "fixture"
    reader_cache: Path | None = None
    drop: tuple[DropName, ...] = ()

    def __post_init__(self) -> None:
        if isinstance(self.now, str):
            try:
                object.__setattr__(self, "now", datetime.fromisoformat(self.now))
            except ValueError:
                pass  # left as it was: the input check reports it
        if isinstance(self.reader_cache, str):
            object.__setattr__(self, "reader_cache", Path(self.reader_cache))
        if isinstance(self.drop, list):
            object.__setattr__(self, "drop", tuple(self.drop))


def is_aware(moment: object) -> bool:
    """True for a datetime that carries a UTC offset."""
    return isinstance(moment, datetime) and moment.tzinfo is not None and moment.utcoffset() is not None


def _wall_clock() -> datetime:
    return datetime.now(UTC)


def run_now(ctx: RunContext | None) -> datetime:
    """The run's clock: `ctx.now`, or the wall clock (UTC) when there is none. A naive `now` raises ValueError (it would be read as local time)."""
    if ctx is None or ctx.now is None:
        return _wall_clock()
    if not is_aware(ctx.now):
        raise ValueError("now must be a timezone-aware datetime")
    return ctx.now


# --- output shapes: overlays (GeoJSON Features) ----------------------------------------------------------------------


OverlayKind = Literal["notam", "airspace", "lz_radius"]
LAT_LIMIT = QUARTER_TURN  # degrees: the pole. A longitude's limit is geo.HALF_TURN. GeoJSON positions are [lon, lat]
MIN_RING_POSITIONS = MIN_POLYGON_POINTS + 1  # a closed GeoJSON ring: the fewest points of an area, and the first repeated


def _check_position(position: list[float]) -> list[float]:
    try:
        lon, lat = position
    except ValueError:
        raise ValueError("a position is [lon, lat]: two numbers") from None
    if not (math.isfinite(lon) and math.isfinite(lat)):
        raise ValueError("a position is finite")
    if abs(lon) > HALF_TURN or abs(lat) > LAT_LIMIT:
        raise ValueError("a position is [lon, lat] within +-180 and +-90")
    return position


class PointGeometry(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    type: Literal["Point"] = "Point"
    coordinates: list[float]

    @field_validator("coordinates")
    @classmethod
    def _is_a_position(cls, position: list[float]) -> list[float]:
        return _check_position(position)


class PolygonGeometry(BaseModel):
    """A GeoJSON Polygon: the outer ring first, then any holes. Every ring is closed."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    type: Literal["Polygon"] = "Polygon"
    coordinates: list[list[list[float]]]

    @field_validator("coordinates")
    @classmethod
    def _rings_are_closed(cls, rings: list[list[list[float]]]) -> list[list[list[float]]]:
        if not rings:
            raise ValueError("a polygon needs an outer ring")
        for ring in rings:
            if len(ring) < MIN_RING_POSITIONS or ring[0] != ring[-1]:
                raise ValueError("a polygon ring is closed: at least four positions, the last the same as the first")
            for position in ring:
                _check_position(position)
        return rings


class MultiPolygonGeometry(BaseModel):
    """A GeoJSON MultiPolygon (an airspace made of several parts): a list of polygons, each a list of closed rings."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    type: Literal["MultiPolygon"] = "MultiPolygon"
    coordinates: list[list[list[list[float]]]]

    @field_validator("coordinates")
    @classmethod
    def _each_part_is_a_polygon(cls, parts: list[list[list[list[float]]]]) -> list[list[list[list[float]]]]:
        if not parts:
            raise ValueError("a multipolygon needs at least one polygon")
        for rings in parts:
            PolygonGeometry(coordinates=rings)  # the same checks as a Polygon: closed rings, valid positions
        return parts


class OverlayProperties(BaseModel):
    """What the map needs to draw one feature and say what it is.

    id       the NOTAM number, the airspace's name, or "lz_radius"
    kind     notam, airspace, or lz_radius (the LZ's local-area circle)
    action   code's action for a NOTAM or airspace area (None for the circle)
    bearing  degrees clockwise from true north, from the LZ to the feature (None when it is over the LZ, and for the circle)
    text     a NOTAM's raw E text as published (None for airspace and the circle)
    approx   a NOTAM only: True when the shape is its Q-circle because no area could be built from its E text (None for airspace and the circle)
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(min_length=1)
    kind: OverlayKind
    action: str | None = None
    bearing: float | None = Field(default=None, ge=0, lt=FULL_CIRCLE)
    text: str | None = None
    approx: bool | None = None

    @model_validator(mode="after")
    def _only_a_notam_has_text_and_approx(self) -> Self:
        if self.text is not None and self.kind != "notam":
            raise ValueError("only a notam feature carries text")
        if self.approx is not None and self.kind != "notam":
            raise ValueError("only a notam feature carries approx")
        return self


class OverlayFeature(BaseModel):
    """A GeoJSON Feature, so the map is drawn from the contract without parsing any text. `model_dump(mode="json")` is the GeoJSON."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    type: Literal["Feature"] = "Feature"
    geometry: PointGeometry | PolygonGeometry | MultiPolygonGeometry = Field(discriminator="type")
    properties: OverlayProperties


def lz_radius_feature(lat: float, lon: float) -> OverlayFeature:
    """The LZ's local-area circle: a polygon centred on the LZ, its radius the `local_area` rule (read from rules.yaml, in NM).

    The ring is `shapes.circle_ring`, the one circle-to-polygon the NOTAM overlays use too.
    """
    ring = circle_ring(lat, lon, float(RULES.local_area.value))  # type: ignore[arg-type]  # a number rule
    return OverlayFeature(geometry=PolygonGeometry(coordinates=[ring]), properties=OverlayProperties(id="lz_radius", kind="lz_radius"))


# --- output shapes: the assessment ------------------------------------------------------------------------------------


def _check_display(verdict: Verdict | None, approaching_limits: bool, with_conditions: bool, display: str | None) -> None:
    """`display` is exactly ticket 2's display_verdict of the machine fields, or None when there is no verdict."""
    if verdict is None:
        if display is not None:
            raise ValueError("a segment with no verdict has no display")
    elif display != display_verdict(verdict, approaching_limits, with_conditions):
        raise ValueError("display must be exactly display_verdict(verdict, approaching_limits, with_conditions)")


class VerdictValue(BaseModel):
    """The overall verdict. The machine fields are the contract: a colour is picked from them, never by matching `display`."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    verdict: Verdict
    approaching_limits: bool
    with_conditions: bool
    display: str

    @model_validator(mode="after")
    def _display_matches(self) -> Self:
        _check_display(self.verdict, self.approaching_limits, self.with_conditions, self.display)
        return self


class Segment(BaseModel):
    """One span of the assessment, as the pilot sees it (not ticket 2's contracts.Segment, which is in UTC and keeps the evidence).

    start, end   ISO 8601 local times with a UTC offset, in the request's `tz`
    tz_abbr      the zone's abbreviation for that span (GMT, BST, CET, ...)
    tier         VERDICT hours carry a verdict; OUTLOOK and TOO_FAR hours do not
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    start: str
    end: str
    tz_abbr: str = Field(min_length=1)
    tier: Tier
    verdict: Verdict | None
    approaching_limits: bool
    with_conditions: bool
    display: str | None
    conditions: list[str]

    @field_validator("start", "end")
    @classmethod
    def _local_time_with_offset(cls, value: str) -> str:
        if not is_aware(datetime.fromisoformat(value)):
            raise ValueError("a time is ISO 8601 with a UTC offset")
        return value

    @model_validator(mode="after")
    def _verdict_goes_with_tier_and_display(self) -> Self:
        if (self.tier is Tier.VERDICT) != (self.verdict is not None):
            raise ValueError("only a VERDICT-tier segment has a verdict")
        _check_display(self.verdict, self.approaching_limits, self.with_conditions, self.display)
        return self


class Assessment(BaseModel):
    """This turn's `assess` result: its segments and the overall verdict (None when no hour has a verdict, e.g. an OUTLOOK-only window)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    segments: list[Segment]
    overall: VerdictValue | None


# --- thread state -------------------------------------------------------------------------------------------------------


GuardResult = Literal["passed", "retried", "fallback"]


class ThreadLZ(TypedDict):
    """The thread's one landing zone and its current time zone, kept privately. Tools read the LZ and `tz` from here."""

    lat: float
    lon: float
    tz: str


def _number_key(written: str) -> str:
    """A phone number as its digits, with a leading + kept: `01632 960003` and `01632960003` are one number."""
    return ("+" if written.startswith("+") else "") + "".join(char for char in written if char.isdigit())


def merge_phone_maps(current: dict[str, str] | None, update: dict[str, str] | None) -> dict[str, str]:
    """The reducer of `phone_map`: the two maps joined, so tools that run in one step can each add their numbers.

    A placeholder that stands for one number in both is kept once. One that stands for two different numbers (two tools numbered new
    numbers from the same starting map) raises ValueError rather than let `<PHONE_n>` be put back as the wrong number in the answer.
    The message names the placeholder only, never a number.
    """
    merged = dict(current or {})
    for placeholder, number in (update or {}).items():
        kept = merged.setdefault(placeholder, number)
        if _number_key(kept) != _number_key(number):
            raise ValueError(f"{placeholder} was given to two different numbers in one step: call the data tools one at a time")
    return merged


class RequestState(AgentState):
    """The thread state. Add a field here only when the plan changes: tests/unit/agent/test_state.py pins the key set."""

    lat: Required[Annotated[float | None, EphemeralValue, OmitFromOutput]]
    lon: Required[Annotated[float | None, EphemeralValue, OmitFromOutput]]
    tz: NotRequired[Annotated[str | None, EphemeralValue, OmitFromOutput, Field(default=DEFAULT_TZ)]]
    answer: NotRequired[Annotated[str | None, OmitFromInput]]
    guard: NotRequired[Annotated[GuardResult | None, OmitFromInput]]
    overlays: NotRequired[Annotated[list[OverlayFeature], OmitFromInput]]
    assessment: NotRequired[Annotated[Assessment | None, OmitFromInput]]
    # The reducer is the last annotation (LangGraph takes the last one). `| None` keeps LangGraph from starting the channel at {}: a request that
    # never numbered a phone leaves no phone_map in the state, as before.
    phone_map: NotRequired[Annotated[dict[str, str] | None, PrivateStateAttr, merge_phone_maps]]
    thread_lz: NotRequired[Annotated[ThreadLZ | None, PrivateStateAttr]]
