"""Shared by the multi-day capture tests (G11): a fake Open-Meteo that answers for the dates it is asked about.

`Server` and the canned openAIP pages and PIB are the ones tests/test_capture.py uses, unchanged. The only new
piece is MultiDayServer: ticket 2's Server always returns one recorded two-day body, which cannot cover a
window of several days, so this one builds its hourly rows from the request's own start_date and end_date.
No recorded network body is returned and nothing here opens a socket (the suite runs with --disable-socket).
"""

from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import httpx
from test_capture import FAKE_KEY, MIDDLE, NEWEST, OLDEST, PIB, Server

from flysafelz import capture
from flysafelz.clients.open_meteo import HISTORICAL_URL, VARIABLES
from flysafelz.clients.openaip import KEY_VARIABLE

__all__ = ["FAKE_KEY", "MIDDLE", "NEWEST", "OLDEST", "TODAY", "MultiDayServer", "case_yaml", "listing", "london_date_of", "make_root", "run", "server_for"]

ONE_HOUR = timedelta(hours=1)
TODAY = datetime(2026, 11, 6, 12, 0, tzinfo=UTC)  # frozen: after every case used here

# A calm, flyable day: every variable constant except the wind, which creeps up with the hour so rows differ.
CONSTANT = {
    "wind_speed_80m": 8.0,
    "wind_speed_120m": 9.0,
    "wind_gusts_10m": 9.0,
    "wind_direction_10m": 200,
    "precipitation": 0.0,
    "precipitation_probability": 10,
    "cape": 0,
    "cloud_cover": 50,
    "cloud_cover_low": 20,
    "visibility": 24000,
    "shortwave_radiation": 100,
    "pressure_msl": 1015.5,
}


def stamp_text(stamp: datetime) -> str:
    return stamp.strftime("%Y-%m-%dT%H:%M")


class MultiDayServer(Server):
    """Server whose Open-Meteo answer holds every hour of every UTC date from start_date to end_date.

    `drop` lists UTC hours to leave out of the answer, to test a body with a missing row.
    """

    def __init__(self, *, drop: set[datetime] | None = None, **kwargs):
        super().__init__(**kwargs)
        self.drop = drop or set()

    def __call__(self, request: httpx.Request) -> httpx.Response:
        if str(request.url).split("?")[0] == HISTORICAL_URL and HISTORICAL_URL not in self.fail:
            self.seen.append(request)
            return httpx.Response(200, json=self.open_meteo(request.url.params["start_date"], request.url.params["end_date"]))
        return super().__call__(request)

    def open_meteo(self, start_date: str, end_date: str) -> dict:
        first = datetime.fromisoformat(start_date).replace(tzinfo=UTC)
        last = datetime.fromisoformat(end_date).replace(tzinfo=UTC) + timedelta(hours=23)
        stamps = []
        stamp = first
        while stamp <= last:
            if stamp not in self.drop:
                stamps.append(stamp)
            stamp += ONE_HOUR
        hourly: dict[str, list] = {"time": [stamp_text(s) for s in stamps]}
        for api_name in VARIABLES:
            if api_name == "wind_speed_10m":
                hourly[api_name] = [5.0 + 0.1 * (int(s.timestamp() // 3600) % 40) for s in stamps]
            else:
                hourly[api_name] = [CONSTANT[api_name]] * len(stamps)
        return {"latitude": 53.8, "longitude": -0.09, "elevation": 14.0, "generationtime_ms": 0.123, "timezone": "UTC", "hourly": hourly}


def server_for(now: datetime, **kwargs) -> MultiDayServer:
    """A MultiDayServer whose archive has a commit just before `now` and whose PIB was issued at 07:00 that day.

    The canned PIB says it was issued on 2026-10-31, which suits a `now` on that day only; for any other
    `now` the archive commits and the PIB's Issued time are moved to its own day, so the PIB is fresh.
    """
    day = now.date()
    commits = [(MIDDLE, f"{day}T01:23:47Z"), (OLDEST, f"{day - timedelta(days=1)}T21:20:57Z")]
    pib = PIB.replace(b"2026-10-31T07:00:37+00:00", f"{day}T07:00:37+00:00".encode())
    return MultiDayServer(commits=commits, pib=pib, **kwargs)


def case_yaml(*, now: str, start: str, end: str, days: int | str | None, comment: str = "") -> str:
    lines = [comment] if comment else []
    lines += ["lat: 53.802797", "lon: -0.086644", f"now: '{now}'"]
    if days is not None:
        lines.append(f"days: {days}")
    lines += ["window:", f"  start: '{start}'", f"  end: '{end}'"]
    return "\n".join(lines) + "\n"


def make_root(tmp_path: Path, name: str, text: str | None, *, files: dict[str, bytes] | None = None) -> Path:
    case_dir = tmp_path / "fixtures" / name
    case_dir.mkdir(parents=True)
    if text is not None:
        (case_dir / "case.yaml").write_text(text, encoding="utf-8")
    for file, data in (files or {}).items():
        (case_dir / file).write_bytes(data)
    return tmp_path


def run(root: Path, name: str, server: Server, *, today: datetime = TODAY) -> int:
    return capture.main([name], root=root, environ={KEY_VARIABLE: FAKE_KEY}, transport=httpx.MockTransport(server), today=today)


def listing(case_dir: Path) -> dict[str, bytes]:
    return {p.name: p.read_bytes() for p in sorted(case_dir.iterdir())}


def london_date_of(now: datetime) -> date:
    """The London date of `now`, worked out here with zoneinfo so the tests do not lean on the code under test."""
    return now.astimezone(ZoneInfo("Europe/London")).date()
