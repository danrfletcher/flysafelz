"""`assess --lat --lon` (live mode), in process. The network is a mock transport that serves the committed Blue Mill files by host.

No real request is made: sockets are disabled for the whole suite (pyproject addopts).
"""

import re
from datetime import UTC, datetime
from types import SimpleNamespace

import cases
import httpx
import pytest

from flysafelz import cli, sources

LAT, LON = "53.802797", "-0.086644"
LIVE_NOW = datetime(2026, 10, 31, 7, 30, tzinfo=UTC)
SECRET = "test-key-1234567890-not-real"
FOLDER = cases.case_dir(cases.BLUE_MILL)
WINDOW = ["--start", "2026-10-31T07:00Z", "--end", "2026-10-31T18:00Z"]


@pytest.fixture
def served(monkeypatch, tmp_path):
    """Route sources.httpx.Client to a mock that answers by host. Returns .seen (the requests) and .down (hosts that answer 503)."""
    served = SimpleNamespace(seen=[], down=set())
    bodies = {
        "api.open-meteo.com": (FOLDER / "weather.json").read_bytes(),
        "api.core.openaip.net": (FOLDER / "airspace.json").read_bytes(),
        "pibs.nats.co.uk": (FOLDER / "pib.xml").read_bytes(),
    }

    def handler(request: httpx.Request) -> httpx.Response:
        served.seen.append(request)
        if request.url.host in served.down:
            return httpx.Response(httpx.codes.SERVICE_UNAVAILABLE)
        if request.url.host in bodies:
            return httpx.Response(httpx.codes.OK, content=bodies[request.url.host])
        return httpx.Response(httpx.codes.NOT_FOUND)

    real_client = httpx.Client
    monkeypatch.setattr(sources.httpx, "Client", lambda **kwargs: real_client(transport=httpx.MockTransport(handler), **kwargs))
    monkeypatch.setattr(sources, "ROOT", tmp_path)  # no .env here, so the key comes only from the environment
    monkeypatch.setenv("OPENAIP_API_KEY", SECRET)
    return served


def live_at(capsys, now: datetime, *extra: str) -> str:
    assert cli.main(["assess", "--lat", LAT, "--lon", LON, *extra], now=now) == 0
    out, err = capsys.readouterr()
    assert err == ""
    return out


def live(capsys, *extra: str) -> str:
    return live_at(capsys, LIVE_NOW, *extra)


def test_live_run_prints_the_table_with_a_live_header_and_the_three_sources_fetched(capsys, served):
    out = live(capsys, *WINDOW)
    lines = out.splitlines()
    assert lines[0] == "FlySafeLZ assess \u2013 live"
    assert lines[1] == f"LZ {LAT}, {LON} \u00b7 now 2026-10-31T07:30Z"
    assert lines[2].startswith("PIB: live from NATS, issued ")
    assert {r.url.host for r in served.seen} == {"api.open-meteo.com", "api.core.openaip.net", "pibs.nats.co.uk"}
    assert "  weather: ok" in lines and "  airspace: ok" in lines


def test_live_run_has_no_readings_so_a_notam_that_needs_one_is_unverified_not_acted_on(capsys, served):
    out = live(capsys, *WINDOW)
    assert "CONTACT_FIRST" not in out
    # 10:00 is not a weather NO-GO hour: J3857 is over the LZ from the surface and unread, so it reads UNVERIFIED. H7171's E text
    # draws a circle 0.15 NM from the LZ (G16: code reads it, no reading needed), so it is a sector to avoid, not unread.
    row = out.split("10:00\u201311:00  ", 1)[1].splitlines()
    assert row[0] == "UNVERIFIED"
    assert not any("NOTAM H7171/26 is over the LZ" in line for line in row)
    assert "    NOTAM H7171/26 AVOID_SECTOR bearing 168.6\u00b0" in row
    assert any("TEMPO DANGER AREA (TDA) EGD496A ACTIVATED" in line for line in row)  # the NOTAM's E text is printed
    assert "    NOTAM J3857/26 UNREAD_OVER_LZ" in row and not any("H7171/26 UNREAD_OVER_LZ" in line for line in row)
    assert "08:00\u201309:00  NO-GO" in out  # a weather NO-GO stays NO-GO


def test_live_run_sends_the_key_only_as_a_header_and_never_prints_it(capsys, served):
    out = live(capsys, *WINDOW, "--debug")
    airspace = [r for r in served.seen if r.url.host == "api.core.openaip.net"]
    assert airspace and all(r.headers["x-openaip-api-key"] == SECRET for r in airspace)
    assert all(SECRET not in str(r.url) for r in served.seen)
    assert SECRET not in out


def test_live_run_without_a_window_uses_the_default_window_and_records_it(capsys, served):
    out = live(capsys)
    assert "Window 2026-10-31T07:00Z to 2026-11-02T00:00Z (end exclusive). All times UTC." in out
    assert "  No window given: assessing from 2026-10-31T07:00:00Z (now rounded down to the hour) to 2026-11-02T00:00:00Z (the end of tomorrow, Europe/London)." in out


def test_live_run_with_no_key_is_an_airspace_failure_with_a_check_by_hand_line_not_a_crash(capsys, served, monkeypatch):
    monkeypatch.delenv("OPENAIP_API_KEY")
    out = live(capsys, *WINDOW)
    assert "  airspace: failed" in out
    assert "    check by hand: openAIP map (https://www.openaip.net)" in out
    assert "UNVERIFIED" in out
    assert not [r for r in served.seen if r.url.host == "api.core.openaip.net"]


SOURCES = [
    ("weather", "api.open-meteo.com", "Open-Meteo (https://open-meteo.com)"),
    ("airspace", "api.core.openaip.net", "openAIP map (https://www.openaip.net)"),
    ("NOTAMs", "pibs.nats.co.uk", "NATS AIS PIB (https://pibs.nats.co.uk)"),
]


@pytest.mark.parametrize(("label", "host", "pointer"), SOURCES)
def test_one_failed_source_is_named_with_its_reason_and_pointer_and_its_hours_read_unverified(capsys, served, label, host, pointer):
    served.down.add(host)
    out = live(capsys, *WINDOW)
    lines = out.splitlines()
    failed = next(i for i, line in enumerate(lines) if line.startswith(f"  {label}: failed"))
    assert "HTTP 503" in lines[failed]
    assert lines[failed + 1] == f"    check by hand: {pointer}"
    assert out.count("check by hand:") == 1
    assert "UNVERIFIED" in out
    assert "Overall: UNVERIFIED" in out or "Overall: NO-GO" in out
    # a weather NO-GO hour would be unknowable without the weather, so only the other sources keep 08:00 a NO-GO
    if label != "weather":
        assert "08:00\u201309:00  NO-GO" in out


def test_a_stale_pib_is_named_and_its_hours_read_unverified(capsys, served):
    out = live_at(capsys, datetime(2026, 10, 31, 11, 30, tzinfo=UTC), "--start", "2026-10-31T12:00Z", "--end", "2026-10-31T14:00Z")
    lines = out.splitlines()
    stale = next(i for i, line in enumerate(lines) if line.startswith("  NOTAMs: stale"))
    assert lines[stale + 1] == "    check by hand: NATS AIS PIB (https://pibs.nats.co.uk)"
    assert "12:00\u201313:00  UNVERIFIED" in out and "13:00\u201314:00  UNVERIFIED" in out


def test_hours_that_ended_before_now_are_dropped(capsys, served):
    out = live_at(capsys, datetime(2026, 10, 31, 9, 30, tzinfo=UTC), "--start", "2026-10-31T07:00Z", "--end", "2026-10-31T12:00Z")
    rows = [line[:11] for line in out.splitlines() if line[:2].isdigit() and line[2] == ":"]
    assert rows == ["09:00\u201310:00", "10:00\u201311:00", "11:00\u201312:00"]


def test_a_window_beyond_the_outlook_horizon_fetches_nothing_and_every_hour_is_too_far(capsys, served):
    out = live(capsys, "--start", "2026-11-20T07:00Z", "--end", "2026-11-20T10:00Z")
    assert served.seen == []
    assert out.count("TOO FAR OUT") == 3
    assert "Overall: No verdict: too far out" in out


def test_live_output_never_contains_the_bare_word_go(capsys, served):
    out = live(capsys, *WINDOW)
    assert not re.search(r"(?<![-\w])GO(?![-\w])", out)
