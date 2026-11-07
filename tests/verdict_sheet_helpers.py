"""Synthetic case folders for the verdict-sheet tests: fixtures/<name>/ written into a tmp root, replayed by load_case_inputs.

    write_case(root, name, now=..., window=(start, end), days=None, notams=(), weather=(first, last))

The weather is calm at every UTC hour from `first` to `last` (both included; by default the case's own span, as
capture writes it). `notams` are factories.notam() ParsedNotams, written out as PIB XML at the precision the PIB
has (a minute of arc), so the real parser reads them back. airspace.json holds no areas. Nothing here opens a socket.
"""

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import yaml
from pib_helpers import notam_xml, pib_xml

from flysafelz.capture import case_day
from flysafelz.clients.open_meteo import VARIABLES
from flysafelz.contracts import ParsedNotam
from flysafelz.daylight import london_days_utc

LAT, LON = 53.802797, -0.086644
ONE_HOUR = timedelta(hours=1)
CALM = {
    "wind_speed_10m": 5.0,
    "wind_speed_80m": 8.0,
    "wind_speed_120m": 9.0,
    "wind_gusts_10m": 7.0,
    "wind_direction_10m": 180.0,
    "precipitation": 0.0,
    "precipitation_probability": 0.0,
    "cape": 0.0,
    "cloud_cover": 10.0,
    "cloud_cover_low": 10.0,
    "visibility": 20000.0,
    "shortwave_radiation": 100.0,
    "pressure_msl": 1015.0,
}


def stamp(t: datetime) -> str:
    return t.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _arc(value: float, width: int, plus: str, minus: str) -> str:
    """A coordinate as the PIB writes it: degrees and whole minutes, e.g. 5348N or 00005W."""
    minutes = round(abs(value) * 60)
    return f"{minutes // 60:0{width}d}{minutes % 60:02d}{plus if value >= 0 else minus}"


def notam_element(notam: ParsedNotam) -> str:
    """The PIB XML for a factories.notam() NOTAM (its id, Q line, circle and B-C times)."""
    return notam_xml(
        series=notam.series,
        number=notam.number,
        year=notam.year,
        code23=notam.subject,
        code45=notam.q_code[3:5],
        traffic=notam.traffic,
        purpose=notam.purpose,
        scope=notam.scope,
        lower=str(notam.lower_fl),
        upper=str(notam.upper_fl),
        coordinates=_arc(notam.centre_lat, 2, "N", "S") + _arc(notam.centre_lon, 3, "E", "W"),
        radius=f"{notam.radius_nm:g}",
        start=notam.valid_from.strftime("%y%m%d%H%M"),
        end=None if notam.valid_to is None else notam.valid_to.strftime("%y%m%d%H%M"),
    )


def weather_json(first: datetime, last: datetime) -> str:
    stamps = []
    t = first
    while t <= last:
        stamps.append(t)
        t += ONE_HOUR
    hourly: dict[str, list] = {"time": [s.strftime("%Y-%m-%dT%H:%M") for s in stamps]}
    for api_name in VARIABLES:
        hourly[api_name] = [CALM[api_name]] * len(stamps)
    return json.dumps({"elevation": 14.0, "hourly": hourly, "latitude": LAT, "longitude": LON, "timezone": "UTC"})


def write_case(
    root: Path,
    name: str,
    *,
    now: datetime,
    window: tuple[datetime, datetime],
    days: int | None = None,
    notams: tuple[ParsedNotam, ...] = (),
    weather: tuple[datetime, datetime] | None = None,
    lat: float = LAT,
    lon: float = LON,
) -> Path:
    """fixtures/<name>/ under root, complete (case.yaml, weather.json, airspace.json, pib.xml, empty readings.yaml)."""
    folder = root / "fixtures" / name
    folder.mkdir(parents=True)
    body: dict = {"lat": lat, "lon": lon, "now": stamp(now)}
    if days is not None:
        body["days"] = days
    body["window"] = {"start": stamp(window[0]), "end": stamp(window[1])}
    (folder / "case.yaml").write_text(yaml.safe_dump(body, sort_keys=False), encoding="utf-8")
    if weather is None:
        first, end = london_days_utc(now, days)[0:2] if days is not None else case_day(now)
        weather = (first, end)
    (folder / "weather.json").write_text(weather_json(*weather), encoding="utf-8")
    (folder / "airspace.json").write_text(json.dumps({"items": [], "page": 1, "totalCount": 0, "totalPages": 1}), encoding="utf-8")
    issued = now.replace(hour=7, minute=0, second=0, microsecond=0).isoformat()
    (folder / "pib.xml").write_text(pib_xml(*(notam_element(n) for n in notams), issued=issued), encoding="utf-8")
    (folder / "readings.yaml").write_text("[]\n", encoding="utf-8")
    return folder


def examples_file(path: Path, *examples: dict) -> Path:
    path.write_text(yaml.safe_dump(list(examples), sort_keys=False), encoding="utf-8")
    return path


def example(
    example_id: str,
    case: str,
    *,
    now: datetime,
    window: tuple[datetime, datetime] | None,
    blind: bool = True,
    ref: str = "assess",
    drop: list[str] | None = None,
    lat: float = LAT,
    lon: float = LON,
) -> dict:
    """One example as a mapping, in the shape of Dan's examples file (times as text)."""
    fixtures: dict = {"case": case}
    if drop is not None:
        fixtures["drop"] = drop
    return {
        "id": example_id,
        "inputs": {"lat": lat, "lon": lon, "now": stamp(now), "question": "Can I fly today?"},
        "fixtures": fixtures,
        "reference": {"path": ref, "window": None if window is None else [stamp(window[0]), stamp(window[1])]},
        "metadata": {"blind": blind},
    }
