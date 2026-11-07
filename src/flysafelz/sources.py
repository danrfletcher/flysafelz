"""Where `assess` gets its inputs for the command line: a committed case, or the live services.

    load_case_sources(name)           fixtures/<name>/ -> a CaseSources (offline, no key, no clock)
    fetch_live_sources(lat, lon, ...) the three clients, for a live run

Case mode replays the files through the real client code: a transport that answers from the
file stands in for the network, so the same openAIP paging and box filter, and the same PIB
parser, run as in a live fetch. Nothing leaves the machine.

Live mode calls open_meteo.fetch_forecast, openaip.fetch_airspace and pib.load_pib, and each
returns a failed status instead of raising. The openAIP key is the only secret read: from the
OPENAIP_API_KEY variable, else from .env at the repository root (find_key in capture.py).
There are no readings in live mode (Q8).
"""

import json
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import httpx
import yaml
from pydantic import ValidationError

from flysafelz import pib
from flysafelz.assess import fetch_horizon
from flysafelz.capture import ROOT, CaptureError, Case, find_key, load_case
from flysafelz.clients import open_meteo, openaip
from flysafelz.clients.open_meteo import Forecast
from flysafelz.clients.openaip import AirspaceFetch
from flysafelz.contracts import NotamReading, SourceName, SourceState, SourceStatus
from flysafelz.pib import PibResult

FIXTURES_DIR = "fixtures"
TIMEOUT_SECONDS = 60  # the PIB is a few MB
REPLAY_KEY = "replay"  # openAIP wants a non-blank key; the replay transport ignores it

# Where a pilot can look when a source fails (G10 / GP10).
CHECK_BY_HAND = {
    SourceName.WEATHER: "Open-Meteo (https://open-meteo.com)",
    SourceName.NOTAMS: "NATS AIS PIB (https://pibs.nats.co.uk)",
    SourceName.AIRSPACE: "openAIP map (https://www.openaip.net)",
}


class SourceError(Exception):
    """A case cannot be loaded. The message is the one line printed."""


@dataclass
class CaseSources:
    case: Case
    forecast: Forecast
    airspace: AirspaceFetch
    pib: PibResult
    readings: dict[str, NotamReading]


def _replay(body: bytes) -> httpx.Client:
    """A client whose every request is answered with this body, so nothing leaves the machine."""
    return httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(httpx.codes.OK, content=body)))


def _shown(path: Path, root: Path) -> str:
    """The path as the pilot types it: fixtures/<case>/<file>."""
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def _read(path: Path, root: Path) -> bytes:
    try:
        return path.read_bytes()
    except FileNotFoundError:
        raise SourceError(f"{_shown(path, root)}: file not found") from None
    except OSError as exc:
        raise SourceError(f"{_shown(path, root)}: cannot be read ({type(exc).__name__})") from None


def _readings(path: Path, root: Path) -> dict[str, NotamReading]:
    try:
        entries = yaml.safe_load(_read(path, root)) or []
        return {r.id: r for r in (NotamReading.model_validate(item) for item in entries)}
    except (yaml.YAMLError, ValidationError, TypeError, AttributeError):
        raise SourceError(f"{_shown(path, root)}: not a list of valid NOTAM readings") from None


def load_case_sources(name: str, *, root: Path | None = None) -> CaseSources:
    """Load fixtures/<name>/. `now` is the case's own; the wall clock is never read.

    Raises SourceError with one line for an unknown case or a file that is missing or unreadable.
    """
    root = ROOT if root is None else root
    folder = root / FIXTURES_DIR / name
    if not name or name != Path(name).name or name.startswith(".") or not folder.is_dir():
        raise SourceError(f"unknown case {name!r}: no folder {FIXTURES_DIR}/{name}/")
    try:
        # A replay never rejects a case for lying in the future; make fixtures does that when it captures.
        case, _raw = load_case(folder / "case.yaml", today=datetime.max.replace(tzinfo=UTC))
    except CaptureError as exc:
        raise SourceError(str(exc)) from None
    try:
        forecast = open_meteo.parse_response(json.loads(_read(folder / "weather.json", root)), case.now)
    except ValueError:
        raise SourceError(f"{_shown(folder / 'weather.json', root)}: not valid JSON") from None
    airspace = openaip.fetch_airspace(
        case.lat, case.lon, api_key=REPLAY_KEY, client=_replay(_read(folder / "airspace.json", root)), now=case.now
    )
    notams = pib.load_pib("live", case.now, client=_replay(_read(folder / "pib.xml", root)))
    if case.capture is not None and notams.status.state is not SourceState.FAILED:
        # The replay has no archive request to learn the commit from; case.yaml recorded it at capture.
        notams = notams.model_copy(update={"status": notams.status.model_copy(update={"sha": case.capture.sha})})
    return CaseSources(case, forecast, airspace, notams, _readings(folder / "readings.yaml", root))


# --- live ---------------------------------------------------------------------------------


def _not_fetched(source: SourceName, now: datetime) -> SourceStatus:
    return SourceStatus(source=source, state=SourceState.OK, reason="not fetched: the window is beyond the outlook horizon", fetched_at=now)


def fetch_live_sources(lat: float, lon: float, window: tuple[datetime, datetime], now: datetime) -> tuple[Forecast, AirspaceFetch, PibResult]:
    """Fetch weather, airspace and the PIB for a live run. A source that fails comes back failed; nothing is raised.

    A window that starts beyond the outlook horizon (G16) needs no data, so nothing is fetched.
    The weather request starts no earlier than the hour we are in and stops at the horizon.
    """
    start, end = window
    horizon = fetch_horizon(now)
    if start >= horizon:
        return (
            Forecast(status=_not_fetched(SourceName.WEATHER, now)),
            AirspaceFetch(status=_not_fetched(SourceName.AIRSPACE, now)),
            PibResult(status=_not_fetched(SourceName.NOTAMS, now)),
        )
    key = find_key(os.environ, ROOT)
    first = max(start, now.replace(minute=0, second=0, microsecond=0))
    with httpx.Client(timeout=TIMEOUT_SECONDS, follow_redirects=True) as client:
        forecast = open_meteo.fetch_forecast(lat, lon, first, min(end, horizon), client=client, now=now)
        airspace = openaip.fetch_airspace(lat, lon, api_key=key, client=client, now=now)
        notams = pib.load_pib("live", now, client=client)
    return forecast, airspace, notams
