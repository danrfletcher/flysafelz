"""make fixtures CASE=<case>: capture one frozen, offline-replayable case into fixtures/<case>/ (G20).

    make fixtures CASE=blue_mill_2026-10-31

It reads fixtures/<case>/case.yaml (lat, lon, now, window, and optionally `days` and an
archive SHA) and calls the clients that already exist, directly:

    weather.json   fetch_historical_forecast  never the live forecast endpoint. The case day
                                              00:00-23:59 UTC plus the next day's 00:00, so the
                                              23:00 row can be mapped (G4). With `days: N`, see below.
                                              Written in Open-Meteo's
                                              own shape with exactly those hourly rows, so
                                              open_meteo.parse_response reads it. It is rebuilt from
                                              the parsed rows, not saved verbatim, so a rerun gives
                                              the same bytes (the live body has a changing
                                              generationtime_ms).
    airspace.json  fetch_airspace             the raw openAIP items for the 10 NM box. openAIP is asked for
                                              all of country=GB (its firewall refuses a bbox), page by
                                              page; every page is recorded as it arrives, and only the
                                              items fetch_airspace kept for the box are joined into one
                                              page. The file is the filtered result, frozen.
    pib.xml        fetch_pib("archive", now)  the PIB at the last archive commit at or before `now`,
                                              or the SHA that case.yaml pins (a pinned SHA wins).
                                              Trimmed to NOTAMs whose Q-circle edge is within the
                                              capture_trim_nm rule (25 NM) of the LZ. The XML
                                              structure and AreaPIBHeader/Issued are kept, so
                                              pib.parse_pib reads the trimmed file unchanged.

The resolved SHA and the PIB's Issued time are written back to case.yaml under `capture:`.

Several days (`days: N`, G11). Without `days` a case is one UTC day and nothing below applies.
With `days: N` (a whole number, 1 to 8: London day 7 is the last that can carry a verdict) the
window may run from the start of `now`'s Europe/London day to the end of London day N-1. All day
boundaries come from daylight.london_days_utc / london_day_utc, so a BST day starts at 23:00 UTC
and the clock-change days are 23 or 25 hours. weather.json holds one hourly row for every UTC
hour from the start of London day 0 to the end of London day N-1, plus the row at that end hour,
from one historical-forecast request. pib.xml is still the one archive PIB at `now`. A case whose
last London day has not ended yet is refused before any request: examples must be historical.

A committed case is never recaptured or deleted (openAIP is live, so a recapture would change
it). More days, or a different moment, go on a new case name, e.g. blue_mill_2026-10-31_3day.

readings.yaml is never overwritten: an existing file is left byte-identical and a missing one
is created as an empty list with a header comment.

All three sources are fetched before anything is written. Any failure is one line on stderr,
a non-zero exit and nothing written. Files are written under a temporary name and renamed; if
a rename fails part way the ones already swapped are put back, so a failed write leaves the
previous fixture whole. Nothing is written if any output holds the openAIP key. case.yaml is
edited in place (only its `capture:` block changes), so comments in it survive. Output is
deterministic: JSON is written with sorted keys and two-space indent, and nothing in a fixture
depends on the clock.

The openAIP key comes from the OPENAIP_API_KEY environment variable, else from a KEY=value line
in .env at the repository root. It is checked before any request and never printed.
"""

import json
import math
import os
import re
import sys
import xml.etree.ElementTree as ET
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Literal

import httpx
import yaml
from pydantic import (
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    ValidationInfo,
    field_validator,
)

from flysafelz.clients.open_meteo import (
    UKMO_SEAMLESS,
    VARIABLES,
    Forecast,
    fetch_historical_forecast,
)
from flysafelz.clients.openaip import KEY_VARIABLE, Airspace, fetch_airspace, item_id
from flysafelz.clients.openaip import URL as AIRSPACE_URL
from flysafelz.clients.pib_source import ARCHIVE_RAW_URL, fetch_pib, utc_text
from flysafelz.contracts import Limit, SourceState
from flysafelz.daylight import london_day_date, london_days_utc
from flysafelz.geo import (
    OUTSIDE_UK_MESSAGE,
    circle_edge_distance_nm,
    shape_edge_distance_nm,
)
from flysafelz.pib import PibError, parse_pib
from flysafelz.pib import (
    _read_circle as read_q_circle,  # the one place that reads a Q-circle; reused so the two never differ
)
from flysafelz.rules import RULES

ROOT = Path(__file__).resolve().parent.parent.parent  # the repository root (this file is src/flysafelz/capture.py)

ENV_FILE = ".env"  # at the repository root; gitignored
USAGE = "usage: make fixtures CASE=<case>  (a folder under fixtures/)"
TIMEOUT_SECONDS = 60  # the PIB is a few MB
ONE_HOUR = timedelta(hours=1)
ONE_DAY = timedelta(days=1)
LAST_MINUTE = timedelta(hours=23, minutes=59)  # the case day ends 23:59 UTC
INDENT = "  "  # two spaces

_CASE_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")
_SHA = re.compile(r"[0-9a-f]{40}")
_NO_COMMIT = "no archive commit"  # how pib_source words E19; capture turns it into a line naming the case

READINGS_HEADER = (
    "# Hand-written NotamReading entries for this case (see flysafelz.contracts.NotamReading).\n"
    "# make fixtures never overwrites this file. An empty list means no NOTAM has a reading.\n"
)


class CaptureError(Exception):
    """A capture cannot finish. The message is the one line printed; nothing has been written."""


# --- case.yaml ---------------------------------------------------------------------


class Window(BaseModel):
    model_config = ConfigDict(extra="forbid")

    start: AwareDatetime
    end: AwareDatetime  # exclusive, like the assess window


class Captured(BaseModel):
    """What the last capture resolved. Written by make fixtures, never by hand."""

    model_config = ConfigDict(extra="forbid")

    sha: str
    pib_issued: AwareDatetime


class Case(BaseModel):
    """fixtures/<case>/case.yaml. Validated before any request is made."""

    model_config = ConfigDict(extra="forbid")

    lat: float
    lon: float
    now: AwareDatetime
    # Europe/London days to capture, counting the day `now` is in as day 1. Absent: the single UTC case day (G11).
    # Declared before `window` so the window check can read it.
    days: int | None = Field(default=None, strict=True)
    window: Window
    archive_sha: str | None = None  # pins the archive commit; without it the last commit at or before `now` is used
    capture: Captured | None = None
    # Which PIB-age rule a replay of this case uses. `case` (absent): the captured PIB is treated as current at `now`
    # (the archive snapshots it only about twice a day). `live`: the live 3 h rule applies; only a constructed old-PIB case sets it.
    stale_rule: Literal["case", "live"] = "case"
    # True: the files are made up, not captured from the archive (so make fixtures refuses to capture over them).
    constructed: bool = False

    @field_validator("lat")
    @classmethod
    def _lat_in_the_uk(cls, lat: float) -> float:
        if not math.isfinite(lat) or RULES.uk_bounds_lat_min.breached(lat) or RULES.uk_bounds_lat_max.breached(lat):
            raise ValueError(OUTSIDE_UK_MESSAGE)
        return lat

    @field_validator("lon")
    @classmethod
    def _lon_in_the_uk(cls, lon: float) -> float:
        if not math.isfinite(lon) or RULES.uk_bounds_lon_min.breached(lon) or RULES.uk_bounds_lon_max.breached(lon):
            raise ValueError(OUTSIDE_UK_MESSAGE)
        return lon

    @field_validator("now")
    @classmethod
    def _not_in_the_future(cls, now: datetime, info: ValidationInfo) -> datetime:
        today = (info.context or {}).get("today")
        if today is not None and now > today:
            raise ValueError(f"{utc_text(now)} is in the future; a case replays the past")
        return now

    @field_validator("days")
    @classmethod
    def _days_within_the_outlook(cls, days: int | None, info: ValidationInfo) -> int | None:
        if days is None:
            return days
        last_day = int(RULES.outlook_horizon_day.value)  # type: ignore[arg-type]  # the last day that can carry a verdict
        if days < 1:
            raise ValueError("must be a whole number of 1 or more")
        if days > last_day + 1:
            now = info.data.get("now")
            where = f" ({london_day_date(now, last_day)})" if now is not None else ""
            raise ValueError(f"must be at most {last_day + 1}: London day {last_day}{where} is the last that can carry a verdict")
        return days

    @field_validator("window")
    @classmethod
    def _inside_the_case_day(cls, window: Window, info: ValidationInfo) -> Window:
        """Without `days`: the window stays inside the UTC case day. With `days` the London-day check applies instead."""
        now = info.data.get("now")
        if now is None or info.data.get("days") is not None:  # `now` already failed, or the London days decide
            return window
        day_start, day_end = case_day(now)
        if not day_start <= window.start < window.end <= day_end:
            raise ValueError(f"must start before it ends and stay inside the case day {day_start.date()} (UTC)")
        return window

    @field_validator("window")
    @classmethod
    def _inside_the_london_days(cls, window: Window, info: ValidationInfo) -> Window:
        """With `days`: the window stays inside London day 0 to London day days-1 (Europe/London, not UTC)."""
        now, days = info.data.get("now"), info.data.get("days")
        if now is None or days is None:  # no `days`, or `now` or `days` already failed: their error is the one to show
            return window
        first_start, last_end = london_days_utc(now, days)
        if not first_start <= window.start < window.end <= last_end:
            first, last = london_day_date(now, 0), london_day_date(now, days - 1)
            raise ValueError(f"must start before it ends and stay inside the London days {first} to {last} (Europe/London)")
        return window

    @field_validator("archive_sha")
    @classmethod
    def _full_sha(cls, sha: str | None) -> str | None:
        if sha is not None and not _SHA.fullmatch(sha):
            raise ValueError("must be a full 40-character lowercase commit SHA")
        return sha


def case_day(now: datetime) -> tuple[datetime, datetime]:
    """(00:00 UTC of the day `now` falls on, 00:00 UTC of the next day)."""
    midnight = now.astimezone(UTC).replace(hour=0, minute=0, second=0, microsecond=0)
    return midnight, midnight + ONE_DAY


def _describe(path: Path, exc: ValidationError) -> str:
    """One line naming the first bad field, e.g. 'fixtures/x/case.yaml: window: must start ...'."""
    first = exc.errors()[0]
    field = ".".join(str(part) for part in first["loc"]) or "case"
    return f"{path}: {field}: {first['msg'].removeprefix('Value error, ')}"


def load_case(path: Path, *, today: datetime) -> tuple[Case, dict]:
    """The validated case and the raw mapping it came from. Raises CaptureError with one line."""
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise CaptureError(f"{path}: file not found") from None
    except (OSError, UnicodeDecodeError) as exc:
        raise CaptureError(f"{path}: cannot be read ({type(exc).__name__})") from None
    except yaml.YAMLError:
        raise CaptureError(f"{path}: not valid YAML") from None
    if not isinstance(raw, dict):
        raise CaptureError(f"{path}: must be a mapping with lat, lon, now and window")
    try:
        return Case.model_validate(raw, context={"today": today}), raw
    except ValidationError as exc:
        raise CaptureError(_describe(path, exc)) from None


# --- the key -----------------------------------------------------------------------


def read_env_file(path: Path) -> dict[str, str]:
    """KEY=value lines of a .env file (optional `export `, optional quotes, # comments). Missing file: empty."""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return {}
    values: dict[str, str] = {}
    for line in text.splitlines():
        line = line.strip().removeprefix("export ").strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, _, value = line.partition("=")
        value = value.strip()
        if len(value) > 1 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        values[name.strip()] = value
    return values


def find_key(environ: Mapping[str, str], root: Path) -> str | None:
    """The openAIP key: the environment first, then .env. None when neither has a non-blank value."""
    for value in (environ.get(KEY_VARIABLE), read_env_file(root / ENV_FILE).get(KEY_VARIABLE)):
        if value and value.strip():
            return value.strip()
    return None


# --- the three fetches ---------------------------------------------------------------


@dataclass
class Fetched:
    files: dict[str, bytes]
    sha: str
    pib_issued: datetime
    notams_before: int
    notams_after: int
    areas: list[Airspace]


def dump_json(data: object) -> bytes:
    """Deterministic JSON: sorted keys, two-space indent, a final newline."""
    return (json.dumps(data, sort_keys=True, indent=INDENT) + "\n").encode("utf-8")


def _hours_through(first: datetime, last: datetime) -> list[datetime]:
    """Every UTC hour stamp from first to last, both included."""
    stamps = []
    stamp = first
    while stamp <= last:
        stamps.append(stamp)
        stamp += ONE_HOUR
    return stamps


def _weather_file(forecast: Forecast, case: Case, stamps: list[datetime]) -> bytes:
    """weather.json in Open-Meteo's shape with exactly these hourly rows. A missing row is one line and nothing is written."""
    missing = [s for s in stamps if s not in forecast.samples]
    if missing:
        raise CaptureError(f"Open-Meteo: no hourly row for {utc_text(missing[0])} ({len(missing)} rows missing)")
    hourly: dict[str, list] = {"time": [s.strftime("%Y-%m-%dT%H:%M") for s in stamps]}
    for api_name, field in VARIABLES.items():
        hourly[api_name] = [getattr(forecast.samples[s], field) for s in stamps]
    body = {
        "elevation": forecast.elevation_m,
        "hourly": hourly,
        "latitude": case.lat,
        "longitude": case.lon,
        "model": UKMO_SEAMLESS,
        "timezone": "UTC",
        "wind_speed_unit": "mph",
    }
    return dump_json(body)


def weather_json(forecast: Forecast, case: Case) -> bytes:
    """weather.json: Open-Meteo's shape, with the case day's 00:00-23:00 rows plus the next day's 00:00."""
    day_start, day_end = case_day(case.now)
    return _weather_file(forecast, case, _hours_through(day_start, day_end))


def weather_json_days(forecast: Forecast, case: Case, days: int) -> bytes:
    """weather.json for `days` London days: one row per UTC hour from the start of London day 0 to the end
    of London day days-1, plus the row at that end hour (the multi-day form of "plus the next 00:00 row")."""
    first, end = london_days_utc(case.now, days)
    return _weather_file(forecast, case, _hours_through(first, end))


def _fetch_weather_days(case: Case, days: int, client: httpx.Client) -> bytes:
    """One historical-forecast request for the whole span. The client takes the end date one hour past the
    `end` it is given, so passing the hour before the end row asks for exactly the UTC dates the span covers."""
    first, end = london_days_utc(case.now, days)
    forecast = fetch_historical_forecast(case.lat, case.lon, first, end - ONE_HOUR, client=client, now=case.now)
    if forecast.status.state is not SourceState.OK:
        raise CaptureError(f"Open-Meteo: {forecast.status.reason}")
    return weather_json_days(forecast, case, days)


def _fetch_weather(case: Case, client: httpx.Client) -> bytes:
    if case.days is not None:
        return _fetch_weather_days(case, case.days, client)
    day_start, _ = case_day(case.now)
    forecast = fetch_historical_forecast(
        case.lat, case.lon, day_start, day_start + LAST_MINUTE, client=client, now=case.now
    )
    if forecast.status.state is not SourceState.OK:
        raise CaptureError(f"Open-Meteo: {forecast.status.reason}")
    return weather_json(forecast, case)


def _fetch_airspace(case: Case, key: str, client: httpx.Client) -> tuple[bytes, list[Airspace]]:
    pages: list[dict] = []

    def record(response: httpx.Response) -> None:
        """Keep each successful openAIP page as it arrives, so airspace.json is the raw response."""
        if response.is_success and str(response.url).startswith(AIRSPACE_URL):
            response.read()
            try:
                body = response.json()
            except ValueError:
                return
            if isinstance(body, dict):
                pages.append(body)

    client.event_hooks = {"response": [record], "request": []}
    try:
        result = fetch_airspace(case.lat, case.lon, api_key=key, client=client, now=case.now)
    finally:
        client.event_hooks = {"response": [], "request": []}
    if result.status.state is not SourceState.OK:
        reason = result.status.reason
        rejected = re.search(r"HTTP (401|403)$", reason)
        if rejected:
            raise CaptureError(f"openAIP: the key was rejected (HTTP {rejected.group(1)}); check {KEY_VARIABLE}")
        raise CaptureError(reason if reason.startswith("openAIP") else f"openAIP: {reason}")
    kept = {area.id for area in result.areas}  # the country's items, cut down to the ones the box kept (once each)
    items = []
    for page in pages:
        for item in page.get("items", []):
            if isinstance(item, dict) and item_id(item) in kept:
                kept.discard(item_id(item))
                items.append(item)
    joined = {"items": items, "page": 1, "totalCount": len(items), "totalPages": 1}
    return dump_json(joined), result.areas


def _outside_trim(notam: ET.Element, lz: tuple[float, float]) -> bool:
    """True when the NOTAM's Q-circle edge is further from the LZ than the trim rule allows.

    A NOTAM whose Q-circle cannot be read is not outside: it is kept, so the trim fails safe.
    """
    try:
        lat, lon, radius = read_q_circle(notam, "NOTAM")
    except PibError:
        return False
    return RULES.capture_trim_nm.breached(circle_edge_distance_nm(lz[0], lz[1], lat, lon, radius))


def trim_pib(xml: bytes, lz: tuple[float, float]) -> tuple[bytes, int, int]:
    """(trimmed XML, NOTAMs before, NOTAMs after). Only Notam elements are removed; everything else stays."""
    try:
        root = ET.fromstring(xml)
    except ET.ParseError as exc:
        raise CaptureError(f"GitHub archive: PIB is not valid XML ({exc})") from None
    parents = {child: parent for parent in root.iter() for child in parent}
    notams = list(root.iter("Notam"))
    dropped = [n for n in notams if _outside_trim(n, lz)]
    for notam in dropped:
        parents[notam].remove(notam)
    trimmed = ET.tostring(root, encoding="utf-8", xml_declaration=True) + b"\n"
    return trimmed, len(notams), len(notams) - len(dropped)


def _fetch_pinned(sha: str, client: httpx.Client) -> bytes:
    """The archive file at a SHA that case.yaml pins. fetch_pib looks the commit up itself, so a pin skips it."""
    what = f"GitHub archive: file at {sha}"
    try:
        response = client.get(ARCHIVE_RAW_URL.format(sha=sha))
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        raise CaptureError(f"{what}: HTTP {exc.response.status_code}") from None
    except httpx.TimeoutException:
        raise CaptureError(f"{what}: request timed out") from None
    except httpx.HTTPError as exc:
        raise CaptureError(f"{what}: {type(exc).__name__}") from None
    return response.content


def _fetch_pib(name: str, case: Case, client: httpx.Client) -> tuple[bytes, str]:
    """(raw XML, the SHA it came from)."""
    if case.archive_sha:
        return _fetch_pinned(case.archive_sha, client), case.archive_sha
    fetched = fetch_pib("archive", case.now, client=client)
    if fetched.xml is None or fetched.status.sha is None:
        reason = fetched.status.reason
        if reason.startswith(_NO_COMMIT):
            raise CaptureError(f"{name}: {reason}")
        raise CaptureError(f"GitHub archive: {reason}")
    return fetched.xml, fetched.status.sha


def fetch_case(name: str, case: Case, key: str, client: httpx.Client) -> Fetched:
    """Fetch all three sources. Raises CaptureError on the first failure; nothing has been written."""
    weather = _fetch_weather(case, client)
    airspace, areas = _fetch_airspace(case, key, client)
    raw_pib, sha = _fetch_pib(name, case, client)
    pib, before, after = trim_pib(raw_pib, (case.lat, case.lon))
    try:
        issued, _notams = parse_pib(pib)  # the trimmed file must read through the real parser
    except PibError as exc:
        raise CaptureError(f"GitHub archive: PIB at {sha} cannot be read: {exc}") from None
    files = {"weather.json": weather, "airspace.json": airspace, "pib.xml": pib}
    return Fetched(files, sha, issued.astimezone(UTC), before, after, areas)


# --- writing -------------------------------------------------------------------------


def _roll_back(case_dir: Path, replaced: list[str], backups: dict[str, Path]) -> list[Path]:
    """Put back every file already swapped in: its old bytes, or nothing if it did not exist before.

    Returns the backups that could not be put back. Those hold the only copy of the old bytes
    and must be kept; a new file that could not be removed has no backup, so it is not listed.
    """
    kept: list[Path] = []
    for name in reversed(replaced):
        try:
            if name in backups:
                os.replace(backups[name], case_dir / name)
            else:
                (case_dir / name).unlink(missing_ok=True)
        except OSError:
            if name in backups:
                kept.append(backups[name])
    return kept


def write_files(case_dir: Path, files: dict[str, bytes]) -> dict[str, str]:
    """Write every changed file, all or none. Returns name -> written | unchanged.

    A file whose bytes are already on disk is not touched. Each new file is staged under a
    temporary name and each old one is copied to a backup first; only then are the new files
    renamed into place. If a rename fails part way, the files already swapped are put back from
    their backups (or removed, if they are new), so the fixture is never half old and half new.
    If staging fails, no rename has happened and the previous files stand. If putting a file
    back fails too, its backup is left on disk and named in the error line, never deleted.
    """
    staged: dict[str, Path] = {}
    backups: dict[str, Path] = {}
    replaced: list[str] = []
    kept: list[Path] = []  # backups whose restore failed: the only copy of the old bytes
    try:
        try:
            for name, data in files.items():
                target = case_dir / name
                if target.exists() and target.read_bytes() == data:
                    continue
                temp = case_dir / f".{name}.tmp"
                temp.write_bytes(data)
                staged[name] = temp
                if target.exists():
                    backup = case_dir / f".{name}.bak"
                    backup.write_bytes(target.read_bytes())
                    backups[name] = backup
            for name, temp in staged.items():
                os.replace(temp, case_dir / name)
                replaced.append(name)
        except OSError as exc:
            kept = _roll_back(case_dir, replaced, backups)
            message = f"{case_dir}: cannot write {exc.filename or 'a fixture file'} ({exc.strerror})"
            if kept:
                message += "; could not restore, old copies kept at " + ", ".join(str(path) for path in kept)
            raise CaptureError(message) from None
    finally:
        for leftover in (*staged.values(), *backups.values()):
            if leftover not in kept:
                leftover.unlink(missing_ok=True)
    return {name: "written" if name in staged else "unchanged" for name in files}


_CAPTURE_KEY = re.compile(r"""(["']?)capture\1\s*:""")


def case_yaml_bytes(text: str, raw: dict, fetched: Fetched) -> bytes:
    """case.yaml with the `capture:` block set. Everything else in the file stays as Dan wrote it.

    The existing top-level `capture:` block is replaced where it stands, or a new one is added at
    the end; comments, key order and quoting elsewhere are not touched, so a recapture of an
    unchanged case gives the same bytes. The result is read back and must equal the old mapping
    plus the new block, otherwise the capture stops with one line and writes nothing.
    """
    block = {"sha": fetched.sha, "pib_issued": utc_text(fetched.pib_issued)}
    new_block = yaml.safe_dump({"capture": block}, sort_keys=True, default_flow_style=False, allow_unicode=True)
    lines = text.splitlines(keepends=True)
    start = next((i for i, line in enumerate(lines) if _CAPTURE_KEY.match(line)), None)
    if start is None:
        head = text if not text or text.endswith("\n") else text + "\n"
        updated = head + new_block
    else:
        end = start + 1
        last = start
        while end < len(lines) and (not lines[end].strip() or lines[end][0] in " \t"):
            if lines[end].strip():
                last = end
            end += 1
        updated = "".join(lines[:start]) + new_block + "".join(lines[last + 1 :])
    try:
        ok = yaml.safe_load(updated) == {**raw, "capture": block}
    except yaml.YAMLError:
        ok = False
    if not ok:
        raise CaptureError("case.yaml: cannot set the capture: block; check that it is a plain top-level key")
    return updated.encode("utf-8")


def refuse_key_in_output(files: dict[str, bytes], key: str) -> None:
    """F6 in code: nothing is written if any output holds the key. The line never shows the key."""
    needle = key.encode("utf-8")
    for name, data in files.items():
        if needle in data:
            raise CaptureError(f"{name}: would contain the {KEY_VARIABLE} value; nothing was written")


# --- what the capture found ----------------------------------------------------------


def _limit_text(limit: Limit | None) -> str:
    if limit is None:
        return "unlimited"
    if limit.datum.value == "SFC":
        return "SFC"
    return f"{limit.value:.0f} {limit.unit.value} {limit.datum.value}"


def _activation_text(area: Airspace) -> str:
    if area.by_notam:
        return "by NOTAM"
    if area.hours is not None:
        return f"hours {area.hours}" + ("" if area.hours_readable else " (not readable)")
    return "empty (treated as active)"


def airspace_report(areas: list[Airspace], lat: float, lon: float) -> list[str]:
    """Lines for the capture report: danger-area activation fields, and what lies over or within the local area."""
    danger = [a for a in areas if a.type == "danger"]
    activation_set = [a for a in danger if a.by_notam or a.hours is not None]
    lines = [f"  danger areas: {len(danger)} ({len(activation_set)} with byNotam or hoursOfOperation set, {len(danger) - len(activation_set)} empty)"]
    lines += [f"    {a.designator or a.name}: {_activation_text(a)}" for a in danger]
    local_area = float(RULES.local_area.value)  # type: ignore[arg-type]
    near = sorted(((shape_edge_distance_nm(lat, lon, a.shape), a) for a in areas), key=lambda pair: (pair[0], pair[1].name))
    for distance, area in near:
        if distance == 0:
            where = "over the LZ"
        elif distance <= local_area:
            where = f"{distance:.2f} NM from the LZ"
        else:
            continue
        lines.append(f"  {where}: {area.name} ({area.type}) {_limit_text(area.lower)} to {_limit_text(area.upper)}")
    return lines


def summary(name: str, fetched: Fetched, status: dict[str, str], readings: str, lat: float, lon: float) -> list[str]:
    lines = [f"captured {name}"]
    lines += [f"  {file}: {state}" for file, state in status.items()]
    lines += [
        f"  readings.yaml: {readings}",
        f"  archive SHA: {fetched.sha}",
        f"  PIB issued: {utc_text(fetched.pib_issued)}",
        f"  NOTAMs: {fetched.notams_before} before the trim, {fetched.notams_after} after (within {RULES.capture_trim_nm.value:g} NM)",
        f"  airspace: {len(fetched.areas)} areas",
    ]
    return lines + airspace_report(fetched.areas, lat, lon)


# --- the command ----------------------------------------------------------------------


def refuse_leftover_backups(case_dir: Path) -> None:
    """Stop before anything is fetched or staged if an earlier failed capture left a .bak behind.

    A kept `.{name}.bak` is the only copy of the old bytes (see write_files). Staging a new capture
    would write the same path again and overwrite it, so Dan has to copy it back or delete it first.
    """
    leftovers = sorted(case_dir.glob(".*.bak"))
    if leftovers:
        listed = ", ".join(str(path) for path in leftovers)
        raise CaptureError(
            f"{case_dir}: an earlier capture left old copies that were not restored: {listed}; "
            "copy each one back over its file (or delete it) and run again"
        )


def refuse_a_day_that_has_not_ended(path: Path, case: Case, today: datetime) -> None:
    """E15: with `days`, the end of London day days-1 must not be after today. One line naming that London date."""
    if case.days is None:
        return
    _, end = london_days_utc(case.now, case.days)
    if end > today:
        last = london_day_date(case.now, case.days - 1)
        raise CaptureError(f"{path}: days: London day {last} has not ended yet (it ends {utc_text(end)}); a case replays the past")


def capture(name: str, case_dir: Path, key: str, client: httpx.Client, *, today: datetime) -> list[str]:
    """Capture one case. Returns the summary lines. Raises CaptureError."""
    refuse_leftover_backups(case_dir)
    case, raw = load_case(case_dir / "case.yaml", today=today)
    if case.constructed:
        raise CaptureError(f"{case_dir / 'case.yaml'}: constructed: true; a constructed case is made up and is never captured")
    refuse_a_day_that_has_not_ended(case_dir / "case.yaml", case, today)  # before any request
    fetched = fetch_case(name, case, key, client)
    readings = case_dir / "readings.yaml"
    files = dict(fetched.files)
    files["case.yaml"] = case_yaml_bytes((case_dir / "case.yaml").read_text(encoding="utf-8"), raw, fetched)
    if readings.exists():
        readings_state = "left untouched"
    else:
        files["readings.yaml"] = (READINGS_HEADER + "[]\n").encode("utf-8")
        readings_state = "created (empty list)"
    refuse_key_in_output(files, key)
    status = write_files(case_dir, files)
    status.pop("readings.yaml", None)
    return summary(name, fetched, status, readings_state, case.lat, case.lon)


def main(
    argv: list[str] | None = None,
    *,
    root: Path = ROOT,
    environ: Mapping[str, str] | None = None,
    transport: httpx.BaseTransport | None = None,
    today: datetime | None = None,
) -> int:
    """Run `capture <case>`. Tests pass root, environ, a mock transport and a frozen today."""
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 1 or not _CASE_NAME.fullmatch(argv[0]):
        print(USAGE, file=sys.stderr)
        return os.EX_USAGE  # 64: a bad command line, not a failed capture
    name = argv[0]
    case_dir = root / "fixtures" / name
    if not case_dir.is_dir():
        print(f"unknown case {name!r}: no folder fixtures/{name}. {USAGE}", file=sys.stderr)
        return os.EX_USAGE  # 64: a bad command line, not a failed capture
    key = find_key(os.environ if environ is None else environ, root)
    if key is None:
        print(f"{KEY_VARIABLE} is not set (add it to .env)", file=sys.stderr)
        return 1
    try:
        with httpx.Client(timeout=TIMEOUT_SECONDS, follow_redirects=True, transport=transport) as client:
            lines = capture(name, case_dir, key, client, today=today or datetime.now(UTC))
    except CaptureError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
