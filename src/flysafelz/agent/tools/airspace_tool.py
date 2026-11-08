"""get_airspace (G10): the permanent airspace over the landing zone and within the local area, as ticket 2 computes it.

The findings are `airspace.assess_airspace(areas, (lat, lon), TimeSpan(window), site_elevation_ft=, pressure_msl=, activations=, tz=)`, printed
as they come: each finding's text, the volume's floor and ceiling with the datum openAIP gives them, the distance and bearing, then
`ceiling_qnh_ft` and the notes unchanged. Danger-area activations are worked out from the same cached PIB the NOTAM tool reads
(`assess._activations`, over the NOTAMs the prefilter keeps), with whatever readings the thread already has. A danger area activated by a
NOTAM nobody has read comes back UNVERIFIED, which is the honest answer until `get_notams` has read it.

The tool gives findings, not a verdict.
"""

from collections.abc import Callable
from dataclasses import dataclass

from langchain.tools import ToolRuntime, tool
from langchain_core.tools import BaseTool
from langgraph.types import Command

from flysafelz import assess
from flysafelz.agent import progress
from flysafelz.agent.sources import SourceCache
from flysafelz.agent.state import OverlayFeature, OverlayProperties, lz_radius_feature
from flysafelz.agent.tools.artifacts import AirspaceArtifact, AirspaceItem
from flysafelz.agent.tools.common import (
    SOURCE_LABEL,
    Prepared,
    Redactor,
    WindowArgs,
    elevation_ft,
    fetch_line,
    g,
    limit_text,
    lowest_pressure,
    prepare,
    reply,
    source_info,
    unusable_text,
    usable,
    window_line,
)
from flysafelz.airspace import DANGER, Activation, Finding, Outcome, TimeSpan, _circle_shape, assess_airspace
from flysafelz.clients.openaip import Airspace
from flysafelz.contracts import ParsedNotam, SourceName
from flysafelz.geo import Shape, shape_edge_distance_nm
from flysafelz.notam_action import LandingZone
from flysafelz.pib import prefilter
from flysafelz.shapes import airspace_geometry, notam_geometry

UNLIMITED = "UNL"
NONE_LISTED = "none"
NOT_STATED = "not stated"
NM_TO_THE = " NM to the "  # how a sector finding's reason states its distance
DISTANCE_FORMAT = ".2f"  # how `assess_airspace` writes a distance (`0.42 NM to the N`)

DESCRIPTION = (
    "Get the permanent airspace over the pilot's landing zone and within the local area for a window: each volume's meaning (no-go, permission, "
    "radio, ceiling, warning, awareness, avoid-sector, or a danger area's activation), its floor and ceiling with datum, and its distance and bearing. "
    "Times are in the pilot's time zone. Give start and end as local times written YYYY-MM-DD HH:MM, or leave both out for the default window. "
    "The place and time zone are the pilot's and cannot be changed here."
)


def localised(finding: Finding, local: Callable[[str], str]) -> Finding:
    """The finding with its reason and condition in the thread's `<PHONE_n>` numbering."""
    return finding.model_copy(update={"reason": local(finding.reason), "condition": None if finding.condition is None else local(finding.condition)})


@dataclass(frozen=True)
class Placement:
    """Where a finding's volume is and what limits it has: from the openAIP volume, or (a danger area openAIP does not have) from its activating NOTAM.

    `shape` is what distance is measured to. It is the volume itself, or for a NOTAM the Q-circle `assess_airspace` placed it with, so the
    distance agrees with the finding's own text. `geometry` is what the map draws (`airspace_geometry`, or `shapes.notam_geometry`, the one
    NOTAM shape everywhere). `approx` says that drawn NOTAM shape is the Q-circle, not an area from the E text (None: not from a NOTAM). `notam_read` says whether the
    NOTAM reader read the activating NOTAM (None: not from a NOTAM). Unread, floor and ceiling are None because nothing was read, not because
    the NOTAM states none.
    """

    type: str
    floor: str | None
    ceiling: str | None
    shape: Shape | None
    geometry: dict
    approx: bool | None
    notam_id: str | None
    notam_read: bool | None = None


def _placement(name: str, by_name: dict[str, Airspace], activating: dict[str, tuple[Activation, ParsedNotam]]) -> Placement | None:
    """Where `name` is, or None when neither openAIP nor an activating NOTAM places it."""
    area = by_name.get(name)
    if area is not None:
        return Placement(area.type, limit_text(area.lower), limit_text(area.upper) or UNLIMITED, area.shape, airspace_geometry(area.shape), None, None, None)
    found = activating.get(name)
    if found is None:
        return None
    activation, parsed = found
    reading = activation.reading
    geometry, approx = notam_geometry(parsed)
    shape = None if activation.circle is None else _circle_shape(activation.circle)
    floor, ceiling = (None, None) if reading is None else (limit_text(reading.floor), limit_text(reading.ceiling))
    return Placement(DANGER, floor, ceiling, shape, geometry, approx, activation.notam_id, reading is not None)


def _item(finding: Finding, placement: Placement | None, lat: float, lon: float) -> AirspaceItem:
    """The finding as data. Over the LZ is the finding's own: ticket 2 gives a bearing only to a volume that is not over it."""
    over = None if placement is None else finding.bearing_deg is None
    distance = None if placement is None or placement.shape is None or over else float(format(shape_edge_distance_nm(lat, lon, placement.shape), DISTANCE_FORMAT))
    return AirspaceItem(
        name=finding.area,
        type=None if placement is None else placement.type,
        outcome=finding.outcome.value,
        reason=finding.reason,
        condition=finding.condition,
        floor=None if placement is None else placement.floor,
        ceiling=None if placement is None else placement.ceiling,
        over_lz=over,
        distance_nm=distance,
        bearing_deg=finding.bearing_deg,
        compass=finding.compass,
        floor_qnh_ft=finding.floor_qnh_ft,
        ceiling_qnh_ft=finding.ceiling_qnh_ft,
        from_notam=None if placement is None else placement.notam_id,
        approx=None if placement is None else placement.approx,
        notam_read=None if placement is None else placement.notam_read,
    )


def _limits_text(item: AirspaceItem) -> str | None:
    if item.from_notam is None:
        return None if item.floor is None else f"Floor {item.floor}, ceiling {item.ceiling}."
    if item.notam_read is False:
        return f"Floor and ceiling: NOTAM {item.from_notam} was not read, see its raw text (get_notams); the floor is taken as SFC."
    floor = item.floor or NOT_STATED
    ceiling = item.ceiling or NOT_STATED
    return f"Floor {floor}, ceiling {ceiling} (as NOTAM {item.from_notam} states them)."


def _item_text(item: AirspaceItem) -> str:
    if item.over_lz is None:
        where = "Location not known"
    elif item.over_lz:
        where = "Over the LZ"
    else:
        # a sector finding already says `0.42 NM to the N` in its reason: the distance is printed there, once, and only the bearing is added
        said = NM_TO_THE in item.reason
        where = ", ".join(
            part
            for part in (
                None if item.distance_nm is None or said else f"{format(item.distance_nm, DISTANCE_FORMAT)} NM",
                None if item.bearing_deg is None else f"bearing {item.bearing_deg:.0f}° ({item.compass})",
            )
            if part
        )
        where = where[:1].upper() + where[1:]
    parts = [f"{item.name}{'' if item.type is None else f' ({item.type})'}: {item.outcome}.", f"{item.reason}."]
    if item.condition:
        parts.append(f"Condition: {item.condition}.")
    limits = _limits_text(item)
    if limits:
        parts.append(limits)
    if where:
        parts.append(f"{where}.")
    if item.from_notam is not None:
        drawn = "its Q-circle (approximate)" if item.approx else "the area in its E text"
        parts.append(f"Not in the openAIP data: located from NOTAM {item.from_notam}, drawn as {drawn}.")
    return " ".join(parts)


def _feature(item: AirspaceItem, placement: Placement) -> OverlayFeature:
    properties = OverlayProperties(id=item.name, kind="airspace", action=item.outcome, bearing=item.bearing_deg)
    return OverlayFeature(geometry=placement.geometry, properties=properties)  # type: ignore[arg-type]  # a dict the discriminator checks


def airspace_answer(prepared: Prepared, redactor: Redactor) -> tuple[str, AirspaceArtifact]:
    """The text and the artifact for the window. Pure over the prepared sources."""
    sources = prepared.sources
    status = sources.airspace.status
    lines = ["Permanent airspace at the landing zone.", window_line(prepared), fetch_line(prepared)]
    items: list[AirspaceItem] = []
    overlays = [lz_radius_feature(prepared.lat, prepared.lon)]
    ceiling: float | None = None
    notes: list[str] = []
    if not usable(status):
        lines.append(unusable_text(SOURCE_LABEL[SourceName.AIRSPACE], status))
    else:
        window = prepared.window
        height, known = elevation_ft(sources.forecast)
        if not known:
            notes.append("Site elevation unknown (the forecast gave none): taken as 0 ft.")
        span = assess.assessed_span(window.start_utc, window.end_utc, prepared.now, prepared.live)
        near, _dropped = prefilter(sources.pib.notams, (prepared.lat, prepared.lon), span, tz=prepared.tz)
        lz = LandingZone(lat=prepared.lat, lon=prepared.lon, elevation_ft=height)
        activations = assess._activations(near, sources.readings, lz, *span)
        result = assess_airspace(
            sources.airspace.areas,
            (prepared.lat, prepared.lon),
            TimeSpan(start=window.start_utc, end=window.end_utc),
            site_elevation_ft=height,
            pressure_msl=lowest_pressure(sources.forecast, window) if usable(sources.forecast.status) else None,
            activations=activations,
            tz=prepared.tz,
        )
        by_name = {area.name: area for area in sources.airspace.areas}
        parsed_by_id = {parsed.id: parsed for parsed in near}
        activating = {a.notam_id: parsed_by_id[a.notam_id] for a in activations if a.notam_id in parsed_by_id}
        # a danger area openAIP does not have is placed by the NOTAM that activates it (the first, as `assess_airspace` does)
        activated: dict[str, tuple[Activation, ParsedNotam]] = {}
        for a in activations:
            if a.notam_id in parsed_by_id:
                activated.setdefault(a.designator, (a, parsed_by_id[a.notam_id]))

        def local(value: str) -> str:
            """Text built from a NOTAM's activation carries that NOTAM's own `<PHONE_n>` numbering: put it into the thread's. The NOTAM is the one the text names."""
            named = [notam_id for notam_id in activating if notam_id in value]
            return redactor.localiser(activating[named[0]].item_e)(value) if named else value

        notes += [local(note) for note in result.notes]
        ceiling = result.ceiling_qnh_ft
        shown = [f for f in result.findings if f.outcome is not Outcome.IGNORE]
        not_in_effect = [f for f in result.findings if f.outcome is Outcome.IGNORE]
        placements = {f.area: _placement(f.area, by_name, activated) for f in shown}
        items = [_item(localised(f, local), placements[f.area], prepared.lat, prepared.lon) for f in shown]
        if not usable(sources.pib.status):
            notes.append(f"NOTAM source {sources.pib.status.state.value}: no NOTAM could activate a danger area here, so an inactive one may be active.")
        lines.append("Findings:" if items else f"Findings: {NONE_LISTED}. No airspace over the LZ or within the local area affects this window.")
        for item in items:
            lines.append(f"- {_item_text(item)}")
            placement = placements[item.name]
            if placement is not None:
                overlays.append(_feature(item, placement))
        for finding in not_in_effect:
            lines.append(f"- Not in effect: {local(finding.reason)}.")
        lines.append(f"Lowest ceiling over the LZ (ceiling_qnh_ft): {g(ceiling)} ft QNH." if ceiling is not None else f"Lowest ceiling over the LZ (ceiling_qnh_ft): {NONE_LISTED}.")
        lines.append("Notes:" if notes else f"Notes: {NONE_LISTED}.")
        lines += [f"- {note}" for note in notes]
    artifact = AirspaceArtifact(
        source=source_info(prepared, status),
        window_start=prepared.window.start_utc,
        window_end=prepared.window.end_utc,
        window_assumed=prepared.window.assumed,
        overlays=overlays,
        findings=items,
        ceiling_qnh_ft=ceiling,
        notes=notes,
    )
    return "\n".join(lines), artifact


def build_get_airspace(source_cache: SourceCache) -> BaseTool:
    """The get_airspace tool over this source store."""

    @tool("get_airspace", args_schema=WindowArgs, description=DESCRIPTION)
    def get_airspace(runtime: ToolRuntime, start: str | None = None, end: str | None = None) -> Command:
        redactor = Redactor(runtime.state.get("phone_map"))
        prepared = prepare(runtime, source_cache, redactor, start, end, progress.CHECKING_AIRSPACE)
        if isinstance(prepared, str):
            return reply(runtime, prepared, None, redactor)
        text, artifact = airspace_answer(prepared, redactor)
        return reply(runtime, text, artifact, redactor)

    return get_airspace
