"""The verdict labelling sheet, blind mode (4b, PR-3.F1). Model-free: nothing here reads a NOTAM or fills a label.

    python scripts/build_verdict_sheet.py --blind --examples examples.yaml --out verdict_sheet.yaml

    read_examples(path)              Dan's examples file -> list[Example] (validated; raises SheetError)
    build_blind_entry(example, dir)  one sheet entry (a dict) for an `assess` example, from fixtures/<case>/
    edge_name(reasons)               why a row starts where it does, from assess's own cut reasons
    check_weather_coverage(...)      the case must hold weather for every hour up to the verdict horizon (G7)
    main(argv, root=...)             the command; scripts/build_verdict_sheet.py is a thin wrapper

The examples file is a YAML list (or a mapping with an `examples:` list). An example is
    id: blue_mill_2026-10-31
    inputs:    {lat, lon, now, question}
    fixtures:  {case: <folder name under fixtures/>, drop: [weather|airspace|pib]}      (drop is optional)
    reference: {path: assess | explain, window: [start, end] | null}
    metadata:  {blind: true}

The sheet is a YAML list with one entry per `assess` example:
    id, mode (blind), window ("default" or {start, end}), actions_done (false),
    notam_actions ({NOTAM id: null} for every NOTAM within the local area), rows

A row is exactly one segment of assess()'s own answer (G1): start, end, edge, verdict: null,
approaching_limits: null. An OUTLOOK or TOO_FAR hour is one row with `tier` and no label keys (G5).
Rows are never re-cut here. The edge names come from assess.cut_points_with_reasons() and
assess.notams_in_hour(), the same code assess() cuts with (G2); this module has no cut-point logic.

An edge on HH:00 is not a cut inside the hour, so it is found with a one-hour probe that starts one
minute before HH:00: an edge at HH:00 is then strictly inside it (G14). A NOTAM that runs across
HH:00 is not an edge there; one that starts or ends exactly on HH:00 is.

The script writes a new sheet only: an existing `--out` is refused and never read or changed, and the
examples file is only read (F2). A case folder that does not exist yet is captured through
capture.capture(); one that exists is never recaptured or overwritten.
"""

import argparse
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
from flysafelz.capture import CaptureError, case_day
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

MODE_BLIND = "blind"
WINDOW_DEFAULT = "default"
EDGE_WINDOW_START = "window start"
EDGE_HOUR = "hour"
DROP_SOURCES = {"weather": SourceName.WEATHER, "airspace": SourceName.AIRSPACE, "pib": SourceName.NOTAMS}
DROPPED_REASON = "dropped for this example"
DRAFT_LATER = "draft mode lands in PR-4"
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


class SheetError(Exception):
    """Something about an example, the examples file or the output stops the build. The message is the one line printed."""


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


def build_blind_entry(example: Example, case_dir: Path) -> dict[str, Any]:
    """The blind sheet entry for one `assess` example, from the committed case in case_dir. Raises SheetError.

    The rows are the segments assess() returns for the example (same call, same readings: the case's
    readings.yaml), with verdicts and actions dropped. Nothing is re-segmented.
    """
    assert example.now is not None  # an assess example always has its inputs (read_examples)
    try:
        case, forecast, airspace, pib, readings = load_case_inputs(case_dir)
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
    return {
        "id": example.id,
        "mode": MODE_BLIND,
        "window": WINDOW_DEFAULT if example.window is None else {"start": utc_text(window[0]), "end": utc_text(window[1])},
        "actions_done": False,
        "notam_actions": dict.fromkeys(local_notam_ids(pib, case.lat, case.lon)),
        "rows": segment_rows(assessment, case.lat, case.lon, near, readings, lz),
    }


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


# --- the command ------------------------------------------------------------------------------------


def _parser() -> argparse.ArgumentParser:
    parser = OneLineParser(prog="build_verdict_sheet", description="Write a blind verdict sheet from an examples file.")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--blind", action="store_true", help="build every assess example blind (verdicts and actions empty)")
    mode.add_argument("--draft", action="store_true", help="not yet built (PR-4)")
    parser.add_argument("--examples", required=True, type=Path, help="Dan's examples.yaml (only read)")
    parser.add_argument("--out", required=True, type=Path, help="the sheet to write; it must not exist yet")
    return parser


def _usage_error(parser: argparse.ArgumentParser, message: str) -> int:
    """One line on stderr and argparse's own usage-error exit code, as a bad argument gets in the assess command."""
    try:
        parser.error(message)
    except SystemExit as stop:
        return int(stop.code or 0)
    return 0  # parser.error never returns; this keeps the type checker content


def _write_new(out: Path, entries: list[dict[str, Any]]) -> None:
    """Write the sheet to a file that must not exist yet (exclusive create). Raises SheetError."""
    text = yaml.safe_dump(entries, sort_keys=False, default_flow_style=False, allow_unicode=True)
    try:
        with out.open("x", encoding="utf-8") as handle:
            handle.write(text)
    except FileExistsError:
        raise SheetError(f"{out}: already exists; not overwritten (a rerun that appends lands in PR-5)") from None
    except OSError as exc:
        raise SheetError(f"{out}: cannot be written ({type(exc).__name__})") from None


def main(
    argv: Sequence[str] | None = None,
    *,
    root: Path = capture.ROOT,
    environ: Mapping[str, str] | None = None,
    today: datetime | None = None,
    build: Callable[[Example, Path], dict[str, Any]] = build_blind_entry,
) -> int:
    """Run the command. Exit 0: at least one entry written (or nothing to build), 1: an error and no entry, 2: bad usage.

    A failure inside one example (capture, a case that does not fit the example) skips that example with one line
    and the rest are still written; the summary line counts them. A fault in the examples file or the output path
    stops everything with nothing written.
    """
    parser = _parser()
    args = parser.parse_args(argv)
    if args.draft:
        return _usage_error(parser, DRAFT_LATER)
    if args.out.exists():
        print(f"{args.out}: already exists; not overwritten (a rerun that appends lands in PR-5)", file=sys.stderr)
        return 1
    try:
        examples = read_examples(args.examples)
    except SheetError as exc:
        print(exc, file=sys.stderr)
        return 1
    targets = [e for e in examples if e.path == "assess"]
    if not args.blind and not all(e.blind for e in targets):
        return _usage_error(parser, DRAFT_LATER)

    entries: list[dict[str, Any]] = []
    failed = 0
    for example in targets:
        try:
            entries.append(build(example, ensure_case(example, root, environ=environ, today=today)))
        except SheetError as exc:
            failed += 1
            print(exc, file=sys.stderr)
    skipped = len(examples) - len(targets)
    summary = f"{len(entries)} built, {skipped} skipped, {failed} failed"
    if not entries:
        print(f"nothing written: {summary}")
        return 1 if failed else 0
    try:
        _write_new(args.out, entries)
    except SheetError as exc:
        print(exc, file=sys.stderr)
        return 1
    print(f"wrote {args.out}: {summary}")
    return 0
