"""The verdict labelling sheet (4b): blind entries (PR-3.F1) and draft entries (PR-4.F1). Model-free: nothing here reads a NOTAM.

    python scripts/build_verdict_sheet.py --examples examples.yaml --out verdict_sheet.yaml

An example with `metadata.blind: true` gets a blind entry (nothing filled in); any other `assess` example gets a
draft entry (code's verdicts and NOTAM actions, for Dan to correct; see verdict_sheet_draft). The mode is the
example's, never the order of the examples. `--blind` / `--draft` build only the examples of that mode.

    read_examples(path)              Dan's examples file -> list[Example] (validated; raises SheetError)
    build_blind_entry(example, dir)  the blind sheet entry (a dict) for an `assess` example, from fixtures/<case>/
    build_draft_entry(example, dir)  the draft sheet entry for one (a DraftEntry: a dict plus its comments)
    build_entry(example, dir)        whichever the example's `metadata.blind` asks for
    edge_name(reasons)               why a row starts where it does, from assess's own cut reasons
    check_weather_coverage(...)      the case must hold weather for every hour up to the verdict horizon (G7)
    stale_reasons(example, entry, ..) what changed since an entry was written ([] = nothing) (G6)
    append_entries(out, ...)         add new entries to the end of the sheet as text
    load_sheet(path)                 the labelled sheet as Spec section 4 data (verdict_sheet_check)
    main(argv, root=...)             the command; scripts/build_verdict_sheet.py is a thin wrapper

The examples file is a YAML list (or a mapping with an `examples:` list). An example is
    id: blue_mill_2026-10-31
    inputs:    {lat, lon, now, question}
    fixtures:  {case: <folder name under fixtures/>, drop: [weather|airspace|pib]}      (drop is optional)
    reference: {path: assess | explain, window: [start, end] | null}
    metadata:  {blind: true}

The sheet is a YAML list with one entry per `assess` example:
    id, mode (blind | draft), window ("default" or {start, end}), actions_done (false),
    fingerprint, notam_actions, rows
A blind entry's notam_actions is {NOTAM id: null} for every NOTAM within the local area. A draft entry's is code's
worst action per NOTAM, and a comment block of the NOTAMs code ignored for a reason other than distance.

A row is exactly one segment of assess()'s own answer (G1): start, end, edge, verdict: null,
approaching_limits: null (a draft row has code's values). An OUTLOOK or TOO_FAR hour is one row with `tier` and no
label keys (G5), in either mode.
Rows are never re-cut here. The edge names come from assess.cut_points_with_reasons() and
assess.notams_in_hour(), the same code assess() cuts with (G2); this module has no cut-point logic.

An edge on HH:00 is not a cut inside the hour, so it is found with a one-hour probe that starts one
minute before HH:00: an edge at HH:00 is then strictly inside it (G14). A NOTAM that runs across
HH:00 is not an edge there; one that starts or ends exactly on HH:00 is.

A rerun only appends (PR-5.F1, G6). The sheet is created on the first run; after that the entries of examples that
have none yet are added to the end as text, and the bytes already in the file are never changed, so Dan's labels,
comments and spacing survive. Every entry carries a `fingerprint` (verdict_sheet_fingerprint) of what it was built
from; an example whose entry no longer matches is reported `stale` and its entry is left exactly as it is. The script
never deletes an entry. `--check` validates the whole sheet and prints the counts per example (verdict_sheet_check).
The examples file is only read (F2). A case folder that does not exist yet is captured through capture.capture();
one that exists is never recaptured or overwritten.
"""

import argparse
import copy
import os
import re
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import yaml
from pydantic import AwareDatetime, TypeAdapter, ValidationError

from flysafelz import capture
from flysafelz.assess import (
    STARTS,
    CutReason,
    assess,
    cut_points_with_reasons,
    default_window,
    notams_in_hour,
)
from flysafelz.capture import Case, CaptureError, case_day
from flysafelz.cases import load_case_inputs
from flysafelz.cli import OneLineParser
from flysafelz.clients.open_meteo import Forecast
from flysafelz.clients.openaip import KEY_VARIABLE, AirspaceFetch, metres_to_feet
from flysafelz.clients.pib_source import utc_text
from flysafelz.contracts import Assessment, NotamReading, ParsedNotam, SourceName, SourceState, SourceStatus, Tier
from flysafelz.daylight import (
    ONE_HOUR,
    ONE_MINUTE,
    Phase,
    london_date,
    london_day_offset,
    london_days_utc,
    utc_hours,
)
from flysafelz.geo import circle_edge_distance_nm
from flysafelz.notam_action import LandingZone
from flysafelz.pib import PibResult, prefilter
from flysafelz.rules import RULES
from flysafelz.sources import FIXTURES_DIR, SourceError
from flysafelz.verdict_sheet_check import SheetEntry, SheetError, check_entries, parse_sheet, read_sheet, report_line
from flysafelz.verdict_sheet_check import load_sheet as load_sheet  # re-exported: 5a imports it from here
from flysafelz.verdict_sheet_draft import DraftEntry, draft_actions, fill_rows, ignored_by_code, render_entry
from flysafelz.verdict_sheet_fingerprint import (
    MODE_BLIND,
    MODE_DRAFT,
    PART_INPUTS,
    PART_ROWS,
    changed_parts,
    file_parts,
    fingerprint,
    inputs_part,
    rows_part,
    sealed,
    unseal,
)

WINDOW_DEFAULT = "default"
EDGE_WINDOW_START = "window start"
EDGE_HOUR = "hour"
DROP_SOURCES = {"weather": SourceName.WEATHER, "airspace": SourceName.AIRSPACE, "pib": SourceName.NOTAMS}
DROPPED_REASON = "dropped for this example"
JOIN = " + "

# The four daylight edges, by (phase before, phase after) (G2).
DAYLIGHT_NAMES = {
    (Phase.NIGHT, Phase.TWILIGHT): "civil dawn",
    (Phase.TWILIGHT, Phase.DAY): "sunrise",
    (Phase.DAY, Phase.TWILIGHT): "sunset",
    (Phase.TWILIGHT, Phase.NIGHT): "civil dusk",
}

_AWARE = TypeAdapter(AwareDatetime)
_CASE_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")
_DAY_SUFFIX = re.compile(r"_\d+day$")


# --- the examples file ---------------------------------------------------------------------


@dataclass(frozen=True)
class Example:
    """One item of Dan's examples file. Only an `assess` example carries the input fields."""

    id: str
    path: str | None
    blind: bool = False
    lat: float = 0.0
    lon: float = 0.0
    now: datetime | None = None
    window: tuple[datetime, datetime] | None = None
    case: str = ""
    drop: tuple[str, ...] = ()


def _section(raw: dict, key: str, name: str) -> dict:
    value = raw.get(key)
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise SheetError(f"example {name}: {key} must be a mapping")
    return value


def _number(value: object, field: str, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SheetError(f"example {name}: {field} must be a number")
    return float(value)


def _time(value: object, field: str, name: str) -> datetime:
    try:
        return _AWARE.validate_python(value).astimezone(UTC)
    except ValidationError:
        raise SheetError(f"example {name}: {field} must be a date and time with a timezone, e.g. 2026-10-31T07:30Z") from None


def _window(value: object, name: str) -> tuple[datetime, datetime] | None:
    if value is None:
        return None
    if isinstance(value, dict) and set(value) == {"start", "end"}:
        value = [value["start"], value["end"]]
    if not isinstance(value, (list, tuple)):
        raise SheetError(f"example {name}: reference.window must be null or [start, end]")
    try:
        start, end = value
    except ValueError:
        raise SheetError(f"example {name}: reference.window must be null or [start, end]") from None
    return _time(start, "reference.window start", name), _time(end, "reference.window end", name)


def _drop(value: object, name: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, list):
        raise SheetError(f"example {name}: fixtures.drop must be a list of {', '.join(DROP_SOURCES)}")
    for item in value:
        if item not in DROP_SOURCES:
            raise SheetError(f"example {name}: fixtures.drop has {item!r}; it must be one of {', '.join(DROP_SOURCES)}")
    return tuple(dict.fromkeys(value))


def _parse_example(raw: dict, name: str) -> Example:
    reference = _section(raw, "reference", name)
    path = reference.get("path")
    blind = _section(raw, "metadata", name).get("blind") is True
    if path != "assess":
        return Example(id=name, path=path if isinstance(path, str) else None, blind=blind)
    inputs, fixtures = _section(raw, "inputs", name), _section(raw, "fixtures", name)
    case = fixtures.get("case")
    if not isinstance(case, str) or not _CASE_NAME.fullmatch(case):
        raise SheetError(f"example {name}: fixtures.case must be a case folder name under {FIXTURES_DIR}/, not a path")
    return Example(
        id=name,
        path=path,
        blind=blind,
        lat=_number(inputs.get("lat"), "inputs.lat", name),
        lon=_number(inputs.get("lon"), "inputs.lon", name),
        now=_time(inputs.get("now"), "inputs.now", name),
        window=_window(reference.get("window"), name),
        case=case,
        drop=_drop(fixtures.get("drop"), name),
    )


def read_examples(path: Path) -> list[Example]:
    """Every example in the file, in order. The file is only read. Raises SheetError, one line, for anything wrong in it.

    Every example needs a unique `id`. An example that is not `reference.path: assess` is returned with its id
    and path only (it is skipped, not built), so its other fields are not checked.
    """
    try:
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError) as exc:
        raise SheetError(f"{path}: cannot be read ({type(exc).__name__})") from None
    except yaml.YAMLError:
        raise SheetError(f"{path}: not valid YAML") from None
    if isinstance(data, dict):
        data = data.get("examples")
    if not isinstance(data, list):
        raise SheetError(f"{path}: must be a list of examples (or a mapping with an `examples:` list)")
    examples: list[Example] = []
    seen: dict[str, int] = {}
    for position, raw in enumerate(data, start=1):
        if not isinstance(raw, dict):
            raise SheetError(f"example #{position}: must be a mapping")
        name = raw.get("id")
        if not isinstance(name, str) or not name.strip():
            raise SheetError(f"example #{position}: missing id")
        if name in seen:
            raise SheetError(f"example {name}: duplicate id (examples #{seen[name]} and #{position})")
        seen[name] = position
        examples.append(_parse_example(raw, name))
    return examples


# --- edge names (G2, G14) ---------------------------------------------------------------------


def _notam_order(reason: CutReason) -> tuple[str, int]:
    return str(reason[0]), 0 if reason[1] == STARTS else 1


def edge_name(reasons: Sequence[CutReason]) -> str:
    """Why a row starts where it does: the daylight edge names first, then the NOTAM edges by id, joined with ' + '.

    `reasons` are the reasons assess.cut_points_with_reasons() gave for the minute: (phase before, phase after)
    for daylight, (NOTAM id, 'starts' | 'ends') for a NOTAM. None at all: 'hour' (the row starts on HH:00).
    """
    if not reasons:
        return EDGE_HOUR
    daylight = [r for r in reasons if isinstance(r[0], Phase)]
    notams = sorted((r for r in reasons if not isinstance(r[0], Phase)), key=_notam_order)
    names = [DAYLIGHT_NAMES.get((Phase(r[0]), Phase(r[1]))) or f"daylight {r[0]} to {r[1]}" for r in daylight]
    return JOIN.join(names + [f"{r[0]} {r[1]}" for r in notams])


def _at_hour_boundary(hour: datetime, lat: float, lon: float, near: list[ParsedNotam], readings: Mapping[str, NotamReading], lz: LandingZone) -> list[CutReason]:
    """The edge reasons that fall exactly on HH:00, from a one-hour probe starting at HH:00 minus 1 minute (G14).

    A daylight change or a NOTAM start or end at HH:00 is strictly inside the probe; a NOTAM running across HH:00
    is not cut by it. Both come from the same public helpers assess cuts an hour with.
    """
    probe = hour - ONE_MINUTE
    in_probe = notams_in_hour(probe, near, readings, lz)
    return [reason for t, reasons in cut_points_with_reasons(probe, lat, lon, in_probe, readings, lz) if t == hour for reason in reasons]


# --- coverage (G7) ---------------------------------------------------------------------------------


def _verdict_horizon() -> int:
    return int(RULES.verdict_horizon_day.value)  # type: ignore[arg-type]  # a number rule


def _outlook_horizon() -> int:
    return int(RULES.outlook_horizon_day.value)  # type: ignore[arg-type]  # a number rule


def _suggested_case(case: str, now: datetime, window: tuple[datetime, datetime]) -> tuple[str, int]:
    """(a new case name, the `days` it needs) for a window: London days from now's day to the window's last day, up to the outlook day."""
    last = min(london_day_offset(window[1] - ONE_MINUTE, now), _outlook_horizon())
    days = max(last, 0) + 1
    return f"{_DAY_SUFFIX.sub('', case)}_{days}day", days


def check_weather_coverage(example: Example, forecast: Forecast, now: datetime, window: tuple[datetime, datetime]) -> None:
    """Raise SheetError unless the forecast holds a row for every hour of the window up to the verdict horizon.

    Each hour needs its own sample and the one after it (assess builds a row from both). The horizon and the
    days are Europe/London days (a BST day starts at 23:00Z), never UTC dates. The error names the example and
    the first London day with an hour missing, and suggests a new case name with `days`. TOO_FAR and OUTLOOK
    hours past the verdict horizon need no weather.
    """
    needed = [h for h in utc_hours(*window) if london_day_offset(h, now) <= _verdict_horizon()]
    missing = [h for h in needed if h not in forecast.samples or h + ONE_HOUR not in forecast.samples]
    if not missing:
        return
    day = london_date(missing[0])
    suggestion, days = _suggested_case(example.case, now, window)
    raise SheetError(
        f"example {example.id}: case {example.case} has no weather for London day {day} ({len(missing)} hours missing); "
        f"point the example at a new case captured with days, e.g. {suggestion} (case.yaml: days: {days})"
    )


# --- one entry (G1, G5, G8) -------------------------------------------------------------------------


def _dropped(source: SourceName, now: datetime) -> SourceStatus:
    return SourceStatus(source=source, state=SourceState.FAILED, reason=DROPPED_REASON, fetched_at=now)


def apply_drops(drop: Sequence[str], now: datetime, forecast: Forecast, airspace: AirspaceFetch, pib: PibResult) -> tuple[Forecast, AirspaceFetch, PibResult]:
    """`fixtures.drop`: each named source is replaced by a FAILED one, as the clients return it (G8)."""
    if "weather" in drop:
        forecast = Forecast(status=_dropped(SourceName.WEATHER, now))
    if "airspace" in drop:
        airspace = AirspaceFetch(status=_dropped(SourceName.AIRSPACE, now))
    if "pib" in drop:
        pib = PibResult(status=_dropped(SourceName.NOTAMS, now))
    return forecast, airspace, pib


def local_notam_ids(pib: PibResult, lat: float, lon: float) -> list[str]:
    """The id of every NOTAM in the PIB (superseded ones are already gone) whose Q-circle edge is within the local area.

    Distance only. prefilter()'s kept list also drops by time and FL, so it would miss NOTAMs ignored for those reasons.
    """
    local_area = float(RULES.local_area.value)  # type: ignore[arg-type]  # a number rule
    return sorted(n.id for n in pib.notams if circle_edge_distance_nm(lat, lon, n.centre_lat, n.centre_lon, n.radius_nm) <= local_area)


def segment_rows(assessment: Assessment, lat: float, lon: float, near: list[ParsedNotam], readings: Mapping[str, NotamReading], lz: LandingZone) -> list[dict[str, Any]]:
    """One row per segment (or one tier-only row per OUTLOOK / TOO_FAR hour) of the assessment, with its edge name."""
    rows: list[dict[str, Any]] = []
    for hour in assessment.hours:
        first = not rows
        if hour.tier is not Tier.VERDICT:
            edge = EDGE_WINDOW_START if first else EDGE_HOUR
            rows.append({"start": utc_text(hour.hour), "end": utc_text(hour.hour + ONE_HOUR), "edge": edge, "tier": hour.tier.value})
            continue
        in_hour = notams_in_hour(hour.hour, near, readings, lz)
        inside = dict(cut_points_with_reasons(hour.hour, lat, lon, in_hour, readings, lz))
        for segment in hour.segments:
            if first:
                edge = EDGE_WINDOW_START
            elif segment.start == hour.hour:
                edge = edge_name(_at_hour_boundary(hour.hour, lat, lon, near, readings, lz))
            else:
                edge = edge_name(inside[segment.start])  # a cut assess made: it is in the same helper's answer
            first = False
            rows.append({"start": utc_text(segment.start), "end": utc_text(segment.end), "edge": edge, "verdict": None, "approaching_limits": None})
    return rows


def _shown(value: datetime | float) -> str:
    return utc_text(value) if isinstance(value, datetime) else str(value)


@dataclass(frozen=True)
class _Assessed:
    """What one assess() call over an example's case gave, and what the rows and the draft need from around it."""

    example: Example
    case: Case
    window: tuple[datetime, datetime]
    pib: PibResult
    assessment: Assessment
    near: list[ParsedNotam]  # prefilter()'s kept NOTAMs: the set assess cuts on
    readings: Mapping[str, NotamReading]
    lz: LandingZone

    def head(self, mode: str) -> dict[str, Any]:
        """The keys every entry starts with (the sheet's order), up to and including actions_done."""
        window = self.example.window
        return {
            "id": self.example.id,
            "mode": mode,
            "window": WINDOW_DEFAULT if window is None else {"start": utc_text(self.window[0]), "end": utc_text(self.window[1])},
            "actions_done": False,
        }

    def rows(self) -> list[dict[str, Any]]:
        return segment_rows(self.assessment, self.case.lat, self.case.lon, self.near, self.readings, self.lz)


def _assess_example(example: Example, case_dir: Path, *, readings_optional: bool = False) -> _Assessed:
    """The one assess() call for an example, over the committed case in case_dir. Raises SheetError."""
    assert example.now is not None  # an assess example always has its inputs (read_examples)
    try:
        case, forecast, airspace, pib, readings = load_case_inputs(case_dir, readings_optional=readings_optional)
    except (CaptureError, SourceError) as exc:
        raise SheetError(f"example {example.id}: {exc}") from None
    for field, mine, theirs in (("inputs.now", example.now, case.now), ("inputs.lat", example.lat, case.lat), ("inputs.lon", example.lon, case.lon)):
        if mine != theirs:
            raise SheetError(f"example {example.id}: {field} is {_shown(mine)} but case {example.case} has {_shown(theirs)}; the case must be for the example's own moment")
    window = example.window or default_window(case.now)
    if "weather" not in example.drop:
        check_weather_coverage(example, forecast, case.now, window)
    forecast, airspace, pib = apply_drops(example.drop, case.now, forecast, airspace, pib)
    try:
        assessment = assess(case.lat, case.lon, now=case.now, forecast=forecast, airspace=airspace, pib=pib, readings=readings, window=example.window)
    except ValueError as exc:
        raise SheetError(f"example {example.id}: {exc}") from None

    # The NOTAMs assess cuts on: the same prefilter on the same whole-hour span.
    span = (assessment.hours[0].hour, assessment.hours[-1].hour + ONE_HOUR)
    near, _dropped_notams = prefilter(pib.notams, (case.lat, case.lon), span)
    lz = LandingZone(lat=case.lat, lon=case.lon, elevation_ft=metres_to_feet(forecast.elevation_m) if forecast.elevation_m is not None else 0.0)
    return _Assessed(example, case, window, pib, assessment, near, readings, lz)


def build_blind_entry(example: Example, case_dir: Path) -> dict[str, Any]:
    """The blind sheet entry for one `assess` example, from the committed case in case_dir. Raises SheetError.

    The rows are the segments assess() returns for the example (same call, same readings: the case's
    readings.yaml), with verdicts and actions dropped. Nothing is re-segmented. Nothing derived from code's
    verdicts or actions is written: Dan is the reader.
    """
    built = _assess_example(example, case_dir)
    return {
        **built.head(MODE_BLIND),
        "notam_actions": dict.fromkeys(local_notam_ids(built.pib, built.case.lat, built.case.lon)),
        "rows": built.rows(),
    }


def build_draft_entry(example: Example, case_dir: Path) -> DraftEntry:
    """The draft sheet entry for one `assess` example: code's verdicts and NOTAM actions from the same assess() call as the rows.

    A case with no readings.yaml has no readings (every NOTAM over the LZ from the surface is then unread). Raises SheetError.
    """
    built = _assess_example(example, case_dir, readings_optional=True)
    try:
        rows = fill_rows(built.rows(), built.assessment)
    except ValueError as exc:
        raise SheetError(f"example {example.id}: {exc}") from None
    entry = DraftEntry({**built.head(MODE_DRAFT), "notam_actions": draft_actions(built.assessment), "rows": rows})
    entry.ignored = ignored_by_code(built.assessment.ignored_notams, local_notam_ids(built.pib, built.case.lat, built.case.lon), built.pib.superseded)
    return entry


def build_entry(example: Example, case_dir: Path) -> dict[str, Any]:
    """A blind entry for an example marked `metadata.blind: true`, a draft entry for any other. Never one for the other."""
    return build_blind_entry(example, case_dir) if example.blind else build_draft_entry(example, case_dir)


# --- a case folder that does not exist yet (G7) ------------------------------------------------------


def _new_case_text(example: Example, window: tuple[datetime, datetime]) -> str:
    """case.yaml for a case that has to be captured: lat, lon, now, window, and `days` when one UTC day does not hold the window."""
    assert example.now is not None
    start, end = window
    day_start, day_end = case_day(example.now)
    body: dict[str, Any] = {"lat": example.lat, "lon": example.lon, "now": utc_text(example.now)}
    if not day_start <= start < end <= day_end:
        _name, days = _suggested_case(example.case, example.now, window)
        body["days"] = days
        end = min(end, london_days_utc(example.now, days)[1])  # past the outlook day nothing is captured (TOO_FAR)
    body["window"] = {"start": utc_text(start), "end": utc_text(end)}
    return yaml.safe_dump(body, sort_keys=False)


def ensure_case(example: Example, root: Path, *, environ: Mapping[str, str] | None = None, today: datetime | None = None) -> Path:
    """The case folder for the example. An existing one is returned untouched, never recaptured.

    A missing one gets a case.yaml and a capture (network and OPENAIP_API_KEY: only on Dan's Mac). When the
    capture fails the folder made here is removed again and SheetError carries capture's own message.
    """
    case_dir = root / FIXTURES_DIR / example.case
    try:
        case_dir.mkdir(parents=True)
    except FileExistsError:
        return case_dir
    key = None
    try:
        key = capture.find_key(os.environ if environ is None else environ, root)
        if key is None:
            raise CaptureError(f"{KEY_VARIABLE} is not set (add it to .env)")
        (case_dir / "case.yaml").write_text(_new_case_text(example, example.window or default_window(example.now)), encoding="utf-8")  # type: ignore[arg-type]
        with httpx.Client(timeout=capture.TIMEOUT_SECONDS, follow_redirects=True) as client:
            capture.capture(example.case, case_dir, key, client, today=today or datetime.now(UTC))
    except (CaptureError, OSError) as exc:
        _remove_new_folder(case_dir)
        message = str(exc) if isinstance(exc, CaptureError) else f"{case_dir}: cannot write ({type(exc).__name__})"
        raise SheetError(f"example {example.id}: {message.replace(key, '***') if key else message}") from None
    return case_dir


def _remove_new_folder(case_dir: Path) -> None:
    """Take back the folder ensure_case just made (its own files only; anything it cannot remove stays)."""
    try:
        for leftover in case_dir.iterdir():
            leftover.unlink()
        case_dir.rmdir()
    except OSError:
        pass


# --- the rerun: what is new, what is stale (G6) ---------------------------------------------------------


UNREADABLE = "fingerprint unreadable"


def with_fingerprint(entry: dict[str, Any], parts: Mapping[str, str]) -> dict[str, Any]:
    """The entry with its `fingerprint` key added after `actions_done`. A DraftEntry stays one (it keeps its comments)."""
    items: dict[str, Any] = {}
    for key, value in entry.items():
        items[key] = value
        if key == "actions_done":
            items["fingerprint"] = sealed(parts)
    stamped = copy.copy(entry)
    stamped.clear()
    stamped.update(items)
    return stamped


def stale_reasons(example: Example, stored: Mapping[str, Any], root: Path, build: Callable[[Example, Path], dict[str, Any]] = build_entry) -> list[str]:
    """What changed since the sheet entry `stored` was written for this example: [] when nothing did, else the parts.

    The parts, in order: `inputs`, `rows`, then a fixture file by name (draft entries only). When the inputs changed
    the rows are not listed as well: moved rows are what a changed window does. A fingerprint that is
    missing, mangled, edited by hand (its seal no longer matches) or not for the entry's mode is `fingerprint unreadable`; an example whose mode (blind or draft)
    is no longer the entry's is `inputs`. The case is rebuilt to compare the row edges, from the case folder as it is
    (never captured). When the case cannot be built, the parts that can still be compared are returned; if none of
    them changed, SheetError says why the entry cannot be checked. Nothing is written.
    """
    mode = stored.get("mode")
    old = unseal(stored.get("fingerprint"), mode)
    if old is None:
        return [UNREADABLE]
    now_mode = MODE_BLIND if example.blind else MODE_DRAFT
    if mode != now_mode:
        return [f"{PART_INPUTS}: mode was {mode}, now {now_mode}"]
    current = {PART_INPUTS: inputs_part(example)}
    case_dir = root / FIXTURES_DIR / example.case
    problem: SheetError | None = None
    if case_dir.is_dir():
        try:
            current[PART_ROWS] = rows_part(build(example, case_dir)["rows"])
        except SheetError as exc:
            problem = exc
        if now_mode == MODE_DRAFT:
            current.update(file_parts(case_dir))
    else:
        problem = SheetError(f"example {example.id}: no case folder {FIXTURES_DIR}/{example.case}/ to compare its entry with")
    changed = changed_parts(old, current)
    if PART_INPUTS in changed:
        changed = [name for name in changed if name != PART_ROWS]
    if changed or problem is None:
        return changed
    raise problem


def _entries_text(entries: Sequence[dict[str, Any]]) -> str:
    return "".join(render_entry(e) if isinstance(e, DraftEntry) else yaml.safe_dump([e], sort_keys=False, default_flow_style=False, allow_unicode=True) for e in entries)


def append_entries(out: Path, old_text: str | None, old_entries: Sequence[SheetEntry], entries: Sequence[dict[str, Any]]) -> None:
    """Add the entries to the end of the sheet as text. Raises SheetError, with the file untouched, when they cannot be added.

    The text of `entries` is appended after the file's own bytes (a newline first when the file does not end in
    one); nothing already there is read back or re-dumped, so Dan's comments, key order and spacing survive.
    Before anything is written the whole result is parsed again: it must be valid YAML, hold the old entries
    byte for byte and the new ones after them. The file is read once more afterwards. `old_text` is None for a sheet
    that does not exist yet; it is then created.
    """
    addition = _entries_text(entries)
    if old_text and not old_text.endswith("\n"):
        addition = "\n" + addition
    combined = (old_text or "") + addition
    try:
        parsed = parse_sheet(combined, out)
    except SheetError as exc:
        raise SheetError(f"{exc}; nothing written (the new entries do not fit after the text already there)") from None
    kept = len(old_entries)
    same_old = [a.span for a in parsed[:kept]] == [b.span for b in old_entries] or (
        kept > 0 and [a.span for a in parsed[: kept - 1]] == [b.span for b in old_entries[: kept - 1]] and parsed[kept - 1].span.startswith(old_entries[-1].span)
    )
    if len(parsed) != kept + len(entries) or not same_old:
        raise SheetError(f"{out}: the new entries would not leave the existing ones as they are; nothing written")
    try:
        with out.open("ab" if old_text is not None else "xb") as handle:
            handle.write(addition.encode("utf-8"))
        written = out.read_bytes()
    except OSError as exc:
        raise SheetError(f"{out}: cannot be written ({type(exc).__name__})") from None
    if written != combined.encode("utf-8"):
        raise SheetError(f"{out}: changed while it was being written; check it before the next run")


# --- the command ------------------------------------------------------------------------------------


def _parser() -> argparse.ArgumentParser:
    parser = OneLineParser(
        prog="build_verdict_sheet",
        description="Add new examples to a verdict sheet (blind entries for blind examples, draft entries for the rest), or --check the sheet.",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--blind", action="store_true", help="build only the blind examples (metadata.blind: true); the others are skipped")
    mode.add_argument("--draft", action="store_true", help="build only the draft examples (no metadata.blind); the blind ones are skipped")
    parser.add_argument("--check", action="store_true", help="validate the whole sheet and print the counts per example; writes nothing, exit 1 on any error")
    parser.add_argument("--examples", required=True, type=Path, help="Dan's examples.yaml (only read)")
    parser.add_argument("--out", required=True, type=Path, help="the sheet: created on the first run, then only appended to")
    return parser


def _default_windows(examples: Sequence[Example]) -> dict[str, tuple[datetime, datetime]]:
    """The window of each example that names none: assess's default from the example's own `now`."""
    return {e.id: default_window(e.now) for e in examples if e.path == "assess" and e.window is None and e.now is not None}


def _read_sheet_if_there(out: Path) -> tuple[str | None, list[SheetEntry]]:
    return read_sheet(out) if out.exists() else (None, [])


def _rerun(args: argparse.Namespace, root: Path, environ: Mapping[str, str] | None, today: datetime | None, build: Callable[[Example, Path], dict[str, Any]]) -> int:
    try:
        old_text, existing = _read_sheet_if_there(args.out)
        examples = read_examples(args.examples)
    except SheetError as exc:
        print(exc, file=sys.stderr)
        return 1
    targets = [e for e in examples if e.path == "assess" and not (args.blind and not e.blind) and not (args.draft and e.blind)]
    label_errors = [message for report in check_entries(existing, _default_windows(examples)) for message in report.errors]
    for message in label_errors:
        print(message, file=sys.stderr)

    on_sheet = {entry.id: entry for entry in existing if entry.id}
    entries: list[dict[str, Any]] = []
    unchanged, stale, failed = 0, 0, 0
    for example in targets:
        try:
            if example.id in on_sheet:
                reasons = stale_reasons(example, on_sheet[example.id].data, root, build)
                unchanged += not reasons
                stale += bool(reasons)
                if reasons:
                    print(f"example {example.id}: stale ({', '.join(reasons)}); its entry is left as it is")
                continue
            case_dir = ensure_case(example, root, environ=environ, today=today)
            entry = build(example, case_dir)
            entries.append(with_fingerprint(entry, fingerprint(example, entry["rows"], case_dir, entry["mode"])))
        except SheetError as exc:
            failed += 1
            print(exc, file=sys.stderr)
    summary = f"{len(entries)} new, {unchanged} unchanged, {stale} stale, {len(examples) - len(targets)} skipped, {failed} failed"
    if not entries:
        print(f"nothing written: {summary}")
        return 1 if failed or label_errors else 0
    try:
        append_entries(args.out, old_text, existing, entries)
    except SheetError as exc:
        print(exc, file=sys.stderr)
        return 1
    print(f"wrote {args.out}: {summary}")
    return 1 if label_errors else 0


def _check(args: argparse.Namespace, root: Path, build: Callable[[Example, Path], dict[str, Any]]) -> int:
    """`--check`: one line per entry (GP5) and an error line for each problem. Exit 0 when the sheet validates, 1 otherwise.

    A stale entry or an orphan (an id no longer in the examples file) is reported but is not an error.
    """
    try:
        _text, entries = read_sheet(args.out)
        examples = read_examples(args.examples)
    except SheetError as exc:
        print(exc, file=sys.stderr)
        return 1
    by_id = {e.id: e for e in examples}
    reports = check_entries(entries, _default_windows(examples))
    errors = stale = orphans = 0
    for entry, report in zip(entries, reports, strict=True):
        example = by_id.get(entry.id) if entry.id else None
        orphan = entry.id is not None and (example is None or example.path != "assess")
        status = "no"
        if example is not None and example.path == "assess" and isinstance(entry.data, dict):
            try:
                reasons = stale_reasons(example, entry.data, root, build)
            except SheetError as exc:
                report.errors.append(str(exc))
            else:
                status = f"({', '.join(reasons)})" if reasons else "no"
        if entry.id is not None:
            print(report_line(report, status, orphan=orphan))
        for message in report.errors:
            print(f"error: {message}")
        errors += len(report.errors)
        stale += status != "no"
        orphans += orphan
    print(f"{len(entries)} entries checked: {errors} errors, {stale} stale, {orphans} orphans")
    return 1 if errors else 0


def main(
    argv: Sequence[str] | None = None,
    *,
    root: Path = capture.ROOT,
    environ: Mapping[str, str] | None = None,
    today: datetime | None = None,
    build: Callable[[Example, Path], dict[str, Any]] = build_entry,
) -> int:
    """Run the command. Exit 0: the sheet is as it should be (entries added, or nothing to add), 1: an error, 2: bad usage.

    Without --check: the sheet is created on the first run; later runs only append the entries of examples that
    have none yet (G6). Every new entry gets a `fingerprint`. An example that already has an entry is never rebuilt
    into it: if its inputs, row edges or (draft) fixture files have changed since, it is reported `stale` and the
    entry is left byte for byte. Each example is built in its own mode (blind or draft, by `metadata.blind`);
    `--blind` or `--draft` handles only that mode's examples and counts the rest as skipped. A failure inside one
    example (capture, a case that does not fit the example) skips that example with one line and the rest go on; the
    summary counts them. Label errors in entries already on the sheet are printed and make the exit code 1, but the
    new entries are still appended. A sheet that is not valid YAML stops the run with its line number, with nothing
    written and the file untouched. The examples file is never written (F2).

    With --check: nothing is written; see _check.
    """
    parser = _parser()
    args = parser.parse_args(argv)
    if args.check and (args.blind or args.draft):
        parser.error("--check validates the whole sheet; it cannot be combined with --blind or --draft")
    if args.check:
        return _check(args, root, build)
    return _rerun(args, root, environ, today, build)
