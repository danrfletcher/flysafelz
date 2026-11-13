"""G20 (capture side), E19, F1, F5, F6, F7, F10: `make fixtures` through capture.main and its functions.

Every request goes to an httpx.MockTransport serving the small canned files in tests/data/capture/.
No socket opens (the suite runs with sockets disabled), `today` is frozen, and the key is a
fake one that main reads from the environment dict it is given.
"""

import ast
import json
import os
import re
import stat
import xml.etree.ElementTree as ET
from datetime import UTC, datetime
from pathlib import Path

import httpx
import pytest
import yaml

from flysafelz import capture
from flysafelz.capture import CaptureError, load_case, main, trim_pib
from flysafelz.clients.open_meteo import HISTORICAL_URL, parse_response
from flysafelz.clients.openaip import KEY_HEADER, KEY_VARIABLE
from flysafelz.clients.openaip import URL as AIRSPACE_URL
from flysafelz.clients.pib_source import ARCHIVE_COMMITS_URL
from flysafelz.contracts import SourceState
from flysafelz.pib import parse_pib

DATA = Path(__file__).resolve().parent / "data" / "capture"
SRC = Path(capture.__file__)

CASE = "blue_mill_2026-10-31"
FAKE_KEY = "fake-key-9f3c1a7e5d"
TODAY = datetime(2026, 11, 6, 12, 0, tzinfo=UTC)  # frozen: after the case day
LZ = (53.802797, -0.086644)

NEWEST = "bdab14658f0270dc41460eebcd03332abfb230ec"
MIDDLE = "4375391a6241653e9893cd6db63aeb6a1d80cd06"
OLDEST = "7d0f59a7350e85547a4a9f221a024203e41d2824"
COMMITS = [(NEWEST, "2026-10-31T07:56:28Z"), (MIDDLE, "2026-10-31T01:23:47Z"), (OLDEST, "2026-10-30T21:20:57Z")]

CASE_YAML = """\
lat: 53.802797
lon: -0.086644
now: '2026-10-31T07:30:00Z'
window:
  start: '2026-10-31T07:00:00Z'
  end: '2026-10-31T18:00:00Z'
"""

PIB = (DATA / "pib.xml").read_bytes()
OPEN_METEO = json.loads((DATA / "open_meteo.json").read_text(encoding="utf-8"))
PAGES = {"1": json.loads((DATA / "openaip_page1.json").read_text(encoding="utf-8")), "2": json.loads((DATA / "openaip_page2.json").read_text(encoding="utf-8"))}


def raw_url(sha: str) -> str:
    return f"https://raw.githubusercontent.com/Jonty/uk-notam-archive/{sha}/data/PIB.xml"


class Server:
    """A fake Open-Meteo, openAIP and GitHub. Records every request. `fail` maps a URL (without its query) to a response or an exception."""

    def __init__(self, *, fail=None, commits=COMMITS, pib=PIB):
        self.fail = fail or {}
        self.commits = commits
        self.pib = pib
        self.seen: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.seen.append(request)
        url = str(request.url).split("?")[0]
        if url in self.fail:
            outcome = self.fail[url]
            if isinstance(outcome, Exception):
                raise outcome
            return outcome
        if url == HISTORICAL_URL:
            return httpx.Response(200, json=OPEN_METEO)
        if url == AIRSPACE_URL:
            return httpx.Response(200, json=PAGES[request.url.params["page"]])
        if url == ARCHIVE_COMMITS_URL:
            until = request.url.params["until"]
            listing = [{"sha": sha, "commit": {"committer": {"date": date}}} for sha, date in self.commits if date <= until]
            return httpx.Response(200, json=listing)
        for sha, _date in self.commits:
            if url == raw_url(sha):
                return httpx.Response(200, content=self.pib)
        return httpx.Response(404, text="not found")

    @property
    def urls(self) -> list[str]:
        return [str(r.url).split("?")[0] for r in self.seen]


def make_root(tmp_path: Path, case_yaml: str | None = CASE_YAML, *, files: dict[str, bytes] | None = None) -> Path:
    case_dir = tmp_path / "fixtures" / CASE
    case_dir.mkdir(parents=True)
    if case_yaml is not None:
        (case_dir / "case.yaml").write_text(case_yaml, encoding="utf-8")
    for name, data in (files or {}).items():
        (case_dir / name).write_bytes(data)
    return tmp_path


def run(root: Path, server: Server, *, argv=None, environ=None):
    return main([CASE] if argv is None else argv, root=root, environ={KEY_VARIABLE: FAKE_KEY} if environ is None else environ, transport=httpx.MockTransport(server), today=TODAY)


def listing(case_dir: Path) -> dict[str, bytes]:
    return {p.name: p.read_bytes() for p in sorted(case_dir.iterdir())}


# --- case.yaml validation -------------------------------------------------------------------------------


def with_line(old: str, new: str) -> str:
    assert old in CASE_YAML
    return CASE_YAML.replace(old, new)


BAD_CASES = [
    ("lat: 53.802797\n", "", "lat"),  # missing field
    ("lat: 53.802797", "lat: north", "lat"),
    ("lat: 53.802797", "lat: 12.0", "lat"),  # outside UK coverage
    ("lon: -0.086644", "lon: 40.0", "lon"),
    ("now: '2026-10-31T07:30:00Z'", "now: yesterday", "now"),
    ("now: '2026-10-31T07:30:00Z'", "now: '2026-10-31T07:30:00'", "now"),  # no zone
    ("now: '2026-10-31T07:30:00Z'", "now: '2027-01-01T07:30:00Z'", "now"),  # in the future
    ("  start: '2026-10-31T07:00:00Z'", "  start: '2026-10-30T23:00:00Z'", "window"),  # before the case day
    ("  end: '2026-10-31T18:00:00Z'", "  end: '2026-11-01T01:00:00Z'", "window"),  # after the case day
    ("  end: '2026-10-31T18:00:00Z'", "  end: '2026-10-31T07:00:00Z'", "window"),  # not after the start
    ("window:\n", "windows:\n", "window"),
    ("lon: -0.086644\n", "lon: -0.086644\nsha: abc\n", "sha"),  # unknown field
]


@pytest.mark.parametrize(("old", "new", "field"), BAD_CASES, ids=[f"{c[2]}-{i}" for i, c in enumerate(BAD_CASES)])
def test_a_bad_case_yaml_fails_with_one_line_naming_the_field_and_makes_no_request(tmp_path, capsys, old, new, field):
    root = make_root(tmp_path, with_line(old, new))
    server = Server()
    assert run(root, server) == 1
    out, err = capsys.readouterr()
    assert out == ""
    assert err.count("\n") == 1 and re.search(rf"case\.yaml: {field}\b", err), err
    assert server.seen == []
    assert [p.name for p in (root / "fixtures" / CASE).iterdir()] == ["case.yaml"]


@pytest.mark.parametrize(
    ("text", "message"),
    [(None, "file not found"), ("lat: [unclosed", "not valid YAML"), ("- a\n- b\n", "must be a mapping")],
)
def test_a_missing_or_unreadable_case_yaml_is_one_line(tmp_path, text, message):
    root = make_root(tmp_path, text)
    with pytest.raises(CaptureError, match=message) as caught:
        load_case(root / "fixtures" / CASE / "case.yaml", today=TODAY)
    assert "\n" not in str(caught.value)


def test_a_good_case_yaml_loads_and_a_pinned_sha_must_be_full_length(tmp_path):
    root = make_root(tmp_path, CASE_YAML + f"archive_sha: {NEWEST}\n")
    case, raw = load_case(root / "fixtures" / CASE / "case.yaml", today=TODAY)
    assert (case.lat, case.lon, case.archive_sha) == (LZ[0], LZ[1], NEWEST)
    assert case.window.start == datetime(2026, 10, 31, 7, tzinfo=UTC) and case.window.end == datetime(2026, 10, 31, 18, tzinfo=UTC)
    assert raw["lat"] == LZ[0]
    bad = make_root(tmp_path / "other", CASE_YAML + "archive_sha: bdab1465\n")
    with pytest.raises(CaptureError, match="archive_sha"):
        load_case(bad / "fixtures" / CASE / "case.yaml", today=TODAY)


def test_a_now_after_today_is_rejected_at_validation(tmp_path):
    root = make_root(tmp_path)
    earlier = datetime(2026, 10, 31, 7, 29, tzinfo=UTC)
    with pytest.raises(CaptureError, match=r"now: .*in the future"):
        load_case(root / "fixtures" / CASE / "case.yaml", today=earlier)


# --- usage -------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("argv", [[], [""], ["a", "b"], ["../fixtures"], [".hidden"], ["x/y"]])
def test_a_missing_or_odd_case_prints_the_usage_line(tmp_path, capsys, argv):
    server = Server()
    assert run(make_root(tmp_path), server, argv=argv) == os.EX_USAGE
    out, err = capsys.readouterr()
    assert out == "" and err == capture.USAGE + "\n"
    assert server.seen == []


def test_an_unknown_case_prints_a_usage_line_and_creates_nothing(tmp_path, capsys):
    server = Server()
    assert run(tmp_path, server, argv=["no_such_case"]) == os.EX_USAGE
    err = capsys.readouterr().err
    assert err.count("\n") == 1 and "no_such_case" in err and "usage: make fixtures CASE=<case>" in err
    assert server.seen == [] and not (tmp_path / "fixtures").exists()


# --- the key --------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("environ", [{}, {KEY_VARIABLE: ""}, {KEY_VARIABLE: "   "}])
def test_without_the_key_it_prints_one_line_before_any_request_and_writes_nothing(tmp_path, capsys, environ):
    root = make_root(tmp_path)
    server = Server()
    assert run(root, server, environ=environ) == 1
    out, err = capsys.readouterr()
    assert out == "" and err == "OPENAIP_API_KEY is not set (add it to .env)\n"
    assert server.seen == []
    assert [p.name for p in (root / "fixtures" / CASE).iterdir()] == ["case.yaml"]


def test_the_key_is_read_from_the_process_environment_by_default(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv(KEY_VARIABLE, FAKE_KEY)
    root = make_root(tmp_path)
    server = Server()
    assert main([CASE], root=root, transport=httpx.MockTransport(server), today=TODAY) == 0
    assert server.seen[1].headers[KEY_HEADER] == FAKE_KEY


def test_the_key_falls_back_to_the_env_file_at_the_repository_root(tmp_path):
    # The file is made in tmp_path under the name capture itself reads; the real one is never touched.
    (tmp_path / capture.ENV_FILE).write_text(f"# comment\nexport {KEY_VARIABLE}='from-file'\nOTHER=1\n", encoding="utf-8")
    assert capture.find_key({}, tmp_path) == "from-file"
    assert capture.find_key({KEY_VARIABLE: "from-env"}, tmp_path) == "from-env"  # the environment wins
    assert capture.find_key({}, tmp_path / "nowhere") is None


def test_read_env_file_handles_quotes_comments_and_junk(tmp_path):
    path = tmp_path / "vars.txt"
    path.write_text('A=1\n\n# B=2\nC = "three"\nnot a pair\nexport D=four\n', encoding="utf-8")
    assert capture.read_env_file(path) == {"A": "1", "C": "three", "D": "four"}
    assert capture.read_env_file(tmp_path / "missing") == {}


@pytest.mark.parametrize("status", [401, 403])
def test_a_rejected_key_is_one_line_without_the_key_and_writes_nothing(tmp_path, capsys, status):
    root = make_root(tmp_path)
    server = Server(fail={AIRSPACE_URL: httpx.Response(status)})
    assert run(root, server) == 1
    out, err = capsys.readouterr()
    assert out == ""
    assert err.count("\n") == 1 and "rejected" in err and f"HTTP {status}" in err
    assert FAKE_KEY not in err
    assert listing(root / "fixtures" / CASE) == {"case.yaml": CASE_YAML.encode()}


# --- the weather ---------------------------------------------------------------------------------------------


def test_weather_comes_from_the_historical_endpoint_for_the_case_day_and_never_the_live_one(tmp_path):
    server = Server()
    assert run(make_root(tmp_path), server) == 0
    weather_requests = [r for r in server.seen if r.url.host.endswith("open-meteo.com")]
    assert len(weather_requests) == 1
    request = weather_requests[0]
    assert request.url.host == "historical-forecast-api.open-meteo.com"
    assert not any(url == "https://api.open-meteo.com/v1/forecast" for url in server.urls)
    params = dict(request.url.params)
    assert (params["models"], params["wind_speed_unit"], params["timezone"]) == ("ukmo_seamless", "mph", "UTC")
    assert (params["start_date"], params["end_date"]) == ("2026-10-31", "2026-11-01")  # the day, and the next day for its 00:00


def test_weather_json_holds_the_case_day_plus_next_day_midnight_in_the_open_meteo_shape(tmp_path):
    root = make_root(tmp_path)
    assert run(root, Server()) == 0
    text = (root / "fixtures" / CASE / "weather.json").read_text(encoding="utf-8")
    body = json.loads(text)
    times = body["hourly"]["time"]
    assert (len(times), times[0], times[23], times[-1]) == (25, "2026-10-31T00:00", "2026-10-31T23:00", "2026-11-01T00:00")
    assert "generationtime_ms" not in body  # the part of the live body that changes on every call is not kept
    forecast = parse_response(body, datetime(2026, 10, 31, 7, 30, tzinfo=UTC))  # PR-3's own parser reads it
    assert forecast.status.state is SourceState.OK and forecast.elevation_m == 14.0
    assert len(forecast.samples) == 25
    assert forecast.samples[datetime(2026, 11, 1, 0, 0, tzinfo=UTC)].wind_10m == pytest.approx(5 + 24 * 0.1)


def test_json_is_written_with_sorted_keys_and_a_two_space_indent_and_ends_in_a_newline(tmp_path):
    root = make_root(tmp_path)
    assert run(root, Server()) == 0
    for name in ("weather.json", "airspace.json"):
        text = (root / "fixtures" / CASE / name).read_text(encoding="utf-8")
        assert text.endswith("}\n")
        assert text.splitlines()[1].startswith('  "') and not text.splitlines()[1].startswith("   ")
        assert text == json.dumps(json.loads(text), sort_keys=True, indent=2) + "\n"


def test_a_response_missing_the_next_day_midnight_is_a_failure_and_writes_nothing(tmp_path, capsys):
    short = json.loads(json.dumps(OPEN_METEO))
    for column in short["hourly"].values():
        del column[24:]  # the last row is 2026-10-31T23:00
    root = make_root(tmp_path)
    server = Server(fail={HISTORICAL_URL: httpx.Response(200, json=short)})
    assert run(root, server) == 1
    err = capsys.readouterr().err
    assert err.count("\n") == 1 and "Open-Meteo" in err and "2026-11-01T00:00:00Z" in err
    assert [p.name for p in (root / "fixtures" / CASE).iterdir()] == ["case.yaml"]


# --- the airspace -----------------------------------------------------------------------------------------------


def test_airspace_json_is_the_raw_items_inside_the_box_from_every_country_page_and_the_key_is_a_header_only(tmp_path):
    root = make_root(tmp_path)
    server = Server()
    assert run(root, server) == 0
    asked = [r for r in server.seen if str(r.url).startswith(AIRSPACE_URL)]
    assert [r.url.params["page"] for r in asked] == ["1", "2"]
    assert all(r.headers[KEY_HEADER] == FAKE_KEY and FAKE_KEY not in str(r.url) for r in asked)
    saved = json.loads((root / "fixtures" / CASE / "airspace.json").read_text(encoding="utf-8"))
    assert all(dict(r.url.params) == {"country": "GB", "limit": "1000", "page": r.url.params["page"]} for r in asked)  # no bbox: the firewall refuses it
    everything = PAGES["1"]["items"] + PAGES["2"]["items"]
    assert len(everything) == 10 and len(saved["items"]) == 6  # the country has 10 items here (one is repeated); the Edinburgh, just-outside and far no-polygon ones are cut
    near = [item for item in everything if item["name"] not in {"EDINBURGH CTR", "JUST OUTSIDE TMA", "FAR NO POLYGON"}]
    assert saved["items"] == near[:3] + near[4:]  # raw items, page order, the Humberside item that is on both pages once
    assert (saved["page"], saved["totalPages"], saved["totalCount"]) == (1, 1, len(saved["items"]))


def test_the_saved_airspace_file_reads_back_through_the_openaip_client(tmp_path):
    from flysafelz.clients.openaip import fetch_airspace

    root = make_root(tmp_path)
    assert run(root, Server()) == 0
    body = json.loads((root / "fixtures" / CASE / "airspace.json").read_text(encoding="utf-8"))
    client = httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(200, json=body)))
    got = fetch_airspace(*LZ, api_key=FAKE_KEY, client=client, now=TODAY)
    assert got.status.state is SourceState.OK
    assert "EGD496A" in {a.designator for a in got.areas}


# --- the archive PIB ------------------------------------------------------------------------------------------------


def test_the_archive_commit_is_the_latest_at_or_before_now(tmp_path):
    root = make_root(tmp_path)
    server = Server()
    assert run(root, server) == 0
    commits_request = next(r for r in server.seen if str(r.url).startswith(ARCHIVE_COMMITS_URL))
    assert commits_request.url.params["until"] == "2026-10-31T07:30:00Z"
    assert commits_request.url.params["path"] == "data/PIB.xml"
    assert server.urls[-1] == raw_url(MIDDLE)  # the 07:56Z commit is after now
    case = yaml.safe_load((root / "fixtures" / CASE / "case.yaml").read_text(encoding="utf-8"))
    assert case["capture"]["sha"] == MIDDLE


def test_a_commit_exactly_at_now_is_used(tmp_path):
    exact = CASE_YAML.replace("07:30:00Z", "07:56:28Z")
    root = make_root(tmp_path, exact)
    assert run(root, Server()) == 0
    case = yaml.safe_load((root / "fixtures" / CASE / "case.yaml").read_text(encoding="utf-8"))
    assert case["capture"]["sha"] == NEWEST


def test_a_pinned_sha_wins_and_the_commits_api_is_not_asked(tmp_path):
    root = make_root(tmp_path, CASE_YAML + f"archive_sha: {NEWEST}\n")
    server = Server()
    assert run(root, server) == 0
    assert ARCHIVE_COMMITS_URL not in server.urls
    assert server.urls[-1] == raw_url(NEWEST)
    case = yaml.safe_load((root / "fixtures" / CASE / "case.yaml").read_text(encoding="utf-8"))
    assert case["archive_sha"] == NEWEST and case["capture"]["sha"] == NEWEST


def test_the_resolved_sha_and_the_pib_issued_time_go_under_capture_and_the_other_fields_stay(tmp_path):
    root = make_root(tmp_path)
    assert run(root, Server()) == 0
    case = yaml.safe_load((root / "fixtures" / CASE / "case.yaml").read_text(encoding="utf-8"))
    assert case["capture"] == {"pib_issued": "2026-10-31T07:00:37Z", "sha": MIDDLE}
    assert {k: v for k, v in case.items() if k != "capture"} == yaml.safe_load(CASE_YAML)


def test_e19_no_commit_at_or_before_now_names_the_case_and_now_and_writes_nothing(tmp_path, capsys):
    root = make_root(tmp_path, CASE_YAML.replace("07:30:00Z", "01:00:00Z").replace("07:00:00Z", "00:30:00Z"))
    server = Server(commits=[(NEWEST, "2026-10-31T07:56:28Z")])
    assert run(root, server) == 1
    out, err = capsys.readouterr()
    assert out == "" and err == f"{CASE}: no archive commit at or before 2026-10-31T01:00:00Z\n"
    assert not any(url.startswith("https://raw.githubusercontent.com") for url in server.urls)
    assert sorted(p.name for p in (root / "fixtures" / CASE).iterdir()) == ["case.yaml"]


# --- the 25 NM trim ---------------------------------------------------------------------------------------------------


def et_root(xml: bytes) -> ET.Element:
    return ET.fromstring(xml)


def ids(xml: bytes) -> list[str]:
    return [n.id for n in parse_pib(xml)[1]]


def test_the_trim_keeps_notams_whose_circle_edge_is_within_the_limit_including_ones_that_straddle_it():
    trimmed, before, after = trim_pib(PIB, LZ)
    assert ids(PIB) == ["H9001/26", "H9002/26", "H9003/26", "H9004/26", "H9005/26"]
    # 9001 is over the LZ; 9002's centre is about 29.9 NM away but its 6 NM circle reaches in to about 23.9 NM;
    # 9003's 4 NM circle only reaches 25.9 NM; 9004 is FIR-wide; 9005 is 200 NM south.
    assert ids(trimmed) == ["H9001/26", "H9002/26", "H9004/26"]
    assert (before, after) == (5, 3)


def test_the_trim_keeps_the_header_the_structure_and_the_issued_time():
    trimmed, _, _ = trim_pib(PIB, LZ)
    root = et_root(trimmed)
    assert root.findtext("AreaPIBHeader/Issued") == "2026-10-31T07:00:37+00:00"
    assert root.findtext("AreaPIBHeader/ShortSystemTitle") == "EAD"
    assert root.findtext("FIRSection/ICAO") == "EGTT"
    assert root.find("FIRSection/Warnings/NotamList") is not None
    assert root.find("FIRSection/Warnings/ObsoleteNotamList") is not None
    issued, notams = parse_pib(trimmed)  # PR-4's parser reads the trimmed file unchanged
    assert issued == datetime(2026, 10, 31, 7, 0, 37, tzinfo=UTC) and len(notams) == 3


def test_a_notam_with_no_readable_q_circle_is_kept_so_the_trim_fails_safe():
    xml = (DATA / "pib_unreadable_circle.xml").read_bytes()
    trimmed, before, after = trim_pib(xml, LZ)
    kept = [e.findtext("Number") for e in et_root(trimmed).iter("Notam")]
    assert kept == ["9006", "9007"]  # 9007 has no Coordinates; 9008 is far away
    assert (before, after) == (3, 2)


def test_the_trim_limit_is_the_rule_in_rules_yaml():
    rule = capture.RULES.capture_trim_nm
    assert (rule.value, rule.unit, rule.confidence.value) == (25, "NM", "educated guess") and rule.source.strip()


def test_text_that_is_not_xml_is_a_one_line_failure(tmp_path, capsys):
    root = make_root(tmp_path)
    assert run(root, Server(pib=b"this is not xml")) == 1
    err = capsys.readouterr().err
    assert err.count("\n") == 1 and err.startswith("GitHub archive: PIB is not valid XML")
    assert [p.name for p in (root / "fixtures" / CASE).iterdir()] == ["case.yaml"]


def test_a_pib_with_no_issued_time_is_a_one_line_failure(tmp_path, capsys):
    no_issued = PIB.replace(b"<Issued>2026-10-31T07:00:37+00:00</Issued>", b"")
    root = make_root(tmp_path)
    assert run(root, Server(pib=no_issued)) == 1
    err = capsys.readouterr().err
    assert err.count("\n") == 1 and "Issued" in err
    assert [p.name for p in (root / "fixtures" / CASE).iterdir()] == ["case.yaml"]


# --- readings.yaml ------------------------------------------------------------------------------------------------------


def test_an_existing_readings_file_is_left_byte_identical(tmp_path, capsys):
    mine = b"# mine\r\n- id: J3857/26\n\xc3\xa9  \n"
    root = make_root(tmp_path, files={"readings.yaml": mine})
    assert run(root, Server()) == 0
    assert (root / "fixtures" / CASE / "readings.yaml").read_bytes() == mine
    assert "readings.yaml: left untouched" in capsys.readouterr().out


def test_an_existing_empty_readings_file_is_also_left_alone(tmp_path):
    root = make_root(tmp_path, files={"readings.yaml": b""})
    assert run(root, Server()) == 0
    assert (root / "fixtures" / CASE / "readings.yaml").read_bytes() == b""


def test_a_missing_readings_file_is_created_as_an_empty_list_with_a_header_comment(tmp_path, capsys):
    root = make_root(tmp_path)
    assert run(root, Server()) == 0
    text = (root / "fixtures" / CASE / "readings.yaml").read_text(encoding="utf-8")
    assert text.startswith("#") and yaml.safe_load(text) == []
    assert "readings.yaml: created" in capsys.readouterr().out


def test_a_failed_capture_leaves_readings_and_case_yaml_byte_identical_and_writes_no_fixture(tmp_path):
    root = make_root(tmp_path, files={"readings.yaml": b"- keep me\n"})
    before = listing(root / "fixtures" / CASE)
    assert run(root, Server(fail={raw_url(MIDDLE): httpx.Response(500)})) == 1
    assert listing(root / "fixtures" / CASE) == before


# --- source failures --------------------------------------------------------------------------------------------------------


FAILURES = [
    ("Open-Meteo 500", HISTORICAL_URL, httpx.Response(500), "Open-Meteo: HTTP 500"),
    ("Open-Meteo 429", HISTORICAL_URL, httpx.Response(429), "Open-Meteo: HTTP 429"),
    ("Open-Meteo timeout", HISTORICAL_URL, httpx.ReadTimeout("slow"), "Open-Meteo: request timed out"),
    ("openAIP firewall", AIRSPACE_URL, httpx.Response(403, headers={"content-type": "text/html"}, content=b"<html>Access denied</html>"), "openAIP blocked the request (HTML 403, not a key error)"),
    ("openAIP 500", AIRSPACE_URL, httpx.Response(500), "openAIP: HTTP 500"),
    ("openAIP timeout", AIRSPACE_URL, httpx.ConnectTimeout("slow"), "openAIP: ConnectTimeout"),
    ("GitHub commits 403", ARCHIVE_COMMITS_URL, httpx.Response(403), "GitHub archive: archive commits: HTTP 403"),
    ("GitHub commits 500", ARCHIVE_COMMITS_URL, httpx.Response(500), "GitHub archive: archive commits: HTTP 500"),
    ("GitHub commits timeout", ARCHIVE_COMMITS_URL, httpx.ReadTimeout("slow"), "GitHub archive: archive commits: ReadTimeout"),
    ("GitHub file 404", raw_url(MIDDLE), httpx.Response(404), f"GitHub archive: archive file at {MIDDLE}: HTTP 404"),
    ("GitHub file 502", raw_url(MIDDLE), httpx.Response(502), f"GitHub archive: archive file at {MIDDLE}: HTTP 502"),
]


@pytest.mark.parametrize(("name", "url", "outcome", "line"), FAILURES, ids=[f[0] for f in FAILURES])
def test_a_source_error_is_one_line_naming_the_source_and_status_and_writes_nothing(tmp_path, capsys, name, url, outcome, line):
    root = make_root(tmp_path)
    server = Server(fail={url: outcome})
    assert run(root, server) == 1
    out, err = capsys.readouterr()
    assert out == "" and err.count("\n") == 1
    assert line in err, err
    assert FAKE_KEY not in err
    assert listing(root / "fixtures" / CASE) == {"case.yaml": CASE_YAML.encode()}


def test_a_pinned_sha_file_error_names_github_and_the_status(tmp_path, capsys):
    root = make_root(tmp_path, CASE_YAML + f"archive_sha: {NEWEST}\n")
    assert run(root, Server(fail={raw_url(NEWEST): httpx.Response(404)})) == 1
    assert capsys.readouterr().err == f"GitHub archive: archive file at {NEWEST}: HTTP 404\n"
    root = make_root(tmp_path / "t", CASE_YAML + f"archive_sha: {NEWEST}\n")
    assert run(root, Server(fail={raw_url(NEWEST): httpx.ReadTimeout("slow")})) == 1
    assert capsys.readouterr().err == f"GitHub archive: archive file at {NEWEST}: ReadTimeout\n"


def test_a_write_that_fails_leaves_the_folder_as_it_was_and_no_temp_file(tmp_path, capsys):
    old = {"readings.yaml": b"- keep me\n"}
    root = make_root(tmp_path, files=old)
    case_dir = root / "fixtures" / CASE
    case_dir.chmod(stat.S_IRUSR | stat.S_IXUSR)  # the temp file cannot be created here
    try:
        assert run(root, Server()) == 1
    finally:
        case_dir.chmod(stat.S_IRWXU)
    err = capsys.readouterr().err
    assert err.count("\n") == 1 and "cannot write" in err
    assert listing(case_dir) == {**old, "case.yaml": CASE_YAML.encode()}


def fail_a_rename_into(monkeypatch, target_name: str) -> None:
    """Make os.replace fail when a staged temp file is renamed onto `target_name`; rollback renames still work."""
    real = os.replace

    def replace(src, dst, *args, **kwargs):
        if Path(src).name.endswith(".tmp") and Path(dst).name == target_name:
            raise OSError(28, "No space left on device", str(dst))
        return real(src, dst, *args, **kwargs)

    monkeypatch.setattr(capture.os, "replace", replace)


def test_a_rename_that_fails_part_way_puts_every_file_back_so_the_fixture_is_not_half_new(tmp_path, capsys, monkeypatch):
    old = {"readings.yaml": b"- keep me\n"}
    root = make_root(tmp_path, files=old)
    case_dir = root / "fixtures" / CASE
    fail_a_rename_into(monkeypatch, "case.yaml")  # the last rename: weather, airspace and pib.xml are already swapped in
    assert run(root, Server()) == 1
    err = capsys.readouterr().err
    assert err.count("\n") == 1 and "cannot write" in err
    assert listing(case_dir) == {**old, "case.yaml": CASE_YAML.encode()}  # the three new files removed again, no .tmp or .bak left


def test_a_rename_that_fails_on_the_first_file_leaves_the_rest_alone(tmp_path, monkeypatch):
    old = {"readings.yaml": b"- keep me\n"}
    root = make_root(tmp_path, files=old)
    fail_a_rename_into(monkeypatch, "weather.json")
    assert run(root, Server()) == 1
    assert listing(root / "fixtures" / CASE) == {**old, "case.yaml": CASE_YAML.encode()}


NEW_FILES = {"weather.json": b"new weather", "airspace.json": b"new airspace", "pib.xml": b"new pib", "case.yaml": b"new case"}
OLD_FILES = {"weather.json": b"old weather", "airspace.json": b"old airspace", "pib.xml": b"old pib", "case.yaml": b"old case"}


def break_the_last_rename_and_the_first_restore(monkeypatch) -> None:
    """The last rename (into case.yaml) fails, and so does putting weather.json back from its backup."""
    fail_a_rename_into(monkeypatch, "case.yaml")
    real = capture.os.replace  # already wrapped by fail_a_rename_into

    def replace(src, dst, *args, **kwargs):
        if Path(src).name == ".weather.json.bak":
            raise OSError(5, "Input/output error", str(dst))
        return real(src, dst, *args, **kwargs)

    monkeypatch.setattr(capture.os, "replace", replace)


# A committed case is refused before write_files, so a folder that already holds old bytes reaches it only
# when called directly: these two tests give write_files the old files and the new ones explicitly.
def test_a_failed_restore_keeps_the_backup_and_names_it_in_the_error_line(tmp_path, monkeypatch):
    root = make_root(tmp_path, files=OLD_FILES)
    case_dir = root / "fixtures" / CASE
    break_the_last_rename_and_the_first_restore(monkeypatch)
    with pytest.raises(CaptureError) as caught:
        capture.write_files(case_dir, NEW_FILES)
    err = str(caught.value)
    backup = case_dir / ".weather.json.bak"
    assert "\n" not in err and "cannot write" in err and str(backup) in err
    assert backup.read_bytes() == b"old weather"  # the old bytes survive for Dan to copy back
    assert not (case_dir / ".airspace.json.bak").exists() and not (case_dir / ".pib.xml.bak").exists()  # restored, so cleaned up
    assert (case_dir / "airspace.json").read_bytes() == b"old airspace"
    assert (case_dir / "pib.xml").read_bytes() == b"old pib"


def test_a_rerun_refuses_while_a_kept_backup_is_there_and_leaves_it_alone(tmp_path, capsys, monkeypatch):
    root = make_root(tmp_path, CASE_YAML, files={k: v for k, v in OLD_FILES.items() if k != "case.yaml"})
    case_dir = root / "fixtures" / CASE
    break_the_last_rename_and_the_first_restore(monkeypatch)
    with pytest.raises(CaptureError):
        capture.write_files(case_dir, {**NEW_FILES, "case.yaml": CASE_YAML.encode() + b"# new\n"})
    monkeypatch.undo()  # the disk works again, but weather.json is half new and the .bak still holds the old bytes
    before = listing(case_dir)
    assert before[".weather.json.bak"] == b"old weather" and before["weather.json"] != b"old weather"

    server = Server()
    assert run(root, server) == 1
    err = capsys.readouterr().err
    assert err.count("\n") == 1 and str(case_dir / ".weather.json.bak") in err and "copy each one back" in err
    assert server.seen == []  # refused before any request
    assert listing(case_dir) == before  # nothing written, the old copy is untouched


COMMENTED_CASE = """\
# Blue Mill, the first flight of the autumn. Keep this note.
lat: 53.802797   # the LZ
lon: -0.086644
now: '2026-10-31T07:30:00Z'
window:
  # 07:00 to 18:00, the day's flyable hours
  start: '2026-10-31T07:00:00Z'
  end: '2026-10-31T18:00:00Z'
"""


def test_a_capture_keeps_the_comments_and_layout_dan_wrote_in_case_yaml(tmp_path):
    root = make_root(tmp_path, COMMENTED_CASE)
    path = root / "fixtures" / CASE / "case.yaml"
    assert run(root, Server()) == 0
    first = path.read_text(encoding="utf-8")
    assert first.startswith(COMMENTED_CASE)  # every line he wrote is still there, untouched
    assert yaml.safe_load(first)["capture"] == {"pib_issued": "2026-10-31T07:00:37Z", "sha": MIDDLE}


def test_case_yaml_bytes_replaces_a_capture_block_in_the_middle_of_the_file_where_it_stands():
    fetched = capture.Fetched({}, MIDDLE, datetime(2026, 10, 31, 7, 0, 37, tzinfo=UTC), 1, 1, [])
    stale = COMMENTED_CASE.replace("window:", "capture:\n  pib_issued: '2026-10-30T00:00:00Z'\n  sha: '" + "0" * 40 + "'\n# about the window\nwindow:")
    second = capture.case_yaml_bytes(stale, yaml.safe_load(stale), fetched).decode("utf-8")
    assert second == stale.replace("  pib_issued: '2026-10-30T00:00:00Z'\n  sha: '" + "0" * 40 + "'", "  pib_issued: '2026-10-31T07:00:37Z'\n  sha: " + MIDDLE)
    assert capture.case_yaml_bytes(second, yaml.safe_load(second), fetched).decode("utf-8") == second  # and a second edit changes nothing


def test_a_case_yaml_with_no_final_newline_still_gets_its_capture_block(tmp_path):
    root = make_root(tmp_path, CASE_YAML.rstrip("\n"))
    assert run(root, Server()) == 0
    case = yaml.safe_load((root / "fixtures" / CASE / "case.yaml").read_text(encoding="utf-8"))
    assert case["capture"]["sha"] == MIDDLE and case["lat"] == 53.802797


def test_a_one_line_flow_style_capture_block_is_replaced_in_place():
    fetched = capture.Fetched({}, MIDDLE, datetime(2026, 10, 31, 7, 0, 37, tzinfo=UTC), 1, 1, [])
    old = CASE_YAML + "capture: {sha: abc, pib_issued: '2026-01-01T00:00:00Z'}\n"
    text = capture.case_yaml_bytes(old, yaml.safe_load(old), fetched).decode("utf-8")
    assert text.count("capture") == 1 and yaml.safe_load(text)["capture"]["sha"] == MIDDLE


def test_case_yaml_bytes_refuses_a_result_that_does_not_read_back_as_the_old_fields_plus_the_new_block():
    fetched = capture.Fetched({}, MIDDLE, datetime(2026, 10, 31, 7, 0, 37, tzinfo=UTC), 1, 1, [])
    with pytest.raises(CaptureError, match=r"^case\.yaml: cannot set the capture: block"):
        capture.case_yaml_bytes("lat: 1\n", {"lat": 2}, fetched)  # the text and the mapping disagree


def test_a_second_run_on_a_captured_case_is_refused_with_one_line_before_any_request_and_changes_nothing(tmp_path, capsys):
    root = make_root(tmp_path)
    assert run(root, Server()) == 0
    first = listing(root / "fixtures" / CASE)
    capsys.readouterr()
    server = Server()
    assert run(root, server) == 1
    out, err = capsys.readouterr()
    assert out == "" and err.count("\n") == 1 and CASE in err and "already captured" in err and "new case name" in err
    assert server.seen == []
    assert listing(root / "fixtures" / CASE) == first


@pytest.mark.parametrize("held", ["weather.json", "airspace.json", "pib.xml"])
def test_a_case_that_holds_any_fetched_file_is_refused_before_any_request_and_nothing_is_overwritten(tmp_path, capsys, held):
    root = make_root(tmp_path, files={held: b"committed bytes"})
    before = listing(root / "fixtures" / CASE)
    server = Server()
    assert run(root, server) == 1
    out, err = capsys.readouterr()
    assert out == "" and err.count("\n") == 1 and CASE in err and held in err
    assert server.seen == []
    assert listing(root / "fixtures" / CASE) == before


def test_a_case_yaml_with_a_capture_block_is_refused_even_with_no_fixture_files(tmp_path, capsys):
    root = make_root(tmp_path, CASE_YAML + "capture: {sha: abc, pib_issued: '2026-01-01T00:00:00Z'}\n")
    before = listing(root / "fixtures" / CASE)
    server = Server()
    assert run(root, server) == 1
    err = capsys.readouterr().err
    assert err.count("\n") == 1 and CASE in err and "capture: block" in err
    assert server.seen == [] and listing(root / "fixtures" / CASE) == before


def test_a_committed_case_in_the_repository_is_refused_by_the_real_command(capsys):
    """The reported case: make fixtures CASE=blue_mill_2026-10-31 on a folder from the repository. No transport is given, so a request would fail the test."""
    repo = Path(capture.__file__).resolve().parents[2]
    committed = repo / "fixtures" / "blue_mill_2026-10-31"
    before = listing(committed)
    assert main([committed.name], root=repo, environ={KEY_VARIABLE: FAKE_KEY}, today=TODAY) == 1
    err = capsys.readouterr().err
    assert err.count("\n") == 1 and committed.name in err and "already captured" in err
    assert listing(committed) == before


# --- the summary ---------------------------------------------------------------------------------------------------------------


def test_the_summary_lists_files_sha_issued_time_notam_counts_and_airspace_count(tmp_path, capsys):
    root = make_root(tmp_path)
    assert run(root, Server()) == 0
    out, err = capsys.readouterr()
    assert err == ""
    for line in (
        f"captured {CASE}",
        "  weather.json: written",
        "  airspace.json: written",
        "  pib.xml: written",
        "  case.yaml: written",
        "  readings.yaml: created (empty list)",
        f"  archive SHA: {MIDDLE}",
        "  PIB issued: 2026-10-31T07:00:37Z",
        "  NOTAMs: 5 before the trim, 3 after (within 25 NM)",
        "  airspace: 6 areas",
    ):
        assert line in out.splitlines(), line
    assert FAKE_KEY not in out


def test_the_summary_reports_danger_area_activation_coverage_and_what_is_over_or_near_the_lz(tmp_path, capsys):
    assert run(make_root(tmp_path), Server()) == 0
    out = capsys.readouterr().out
    danger = re.search(r"danger areas: (\d+) \((\d+) with byNotam or hoursOfOperation set, (\d+) empty\)", out)
    assert danger and int(danger.group(1)) == int(danger.group(2)) + int(danger.group(3))
    assert "EGD496A" in out
    assert re.search(r"over the LZ: .*\(danger\)", out)


# --- fences --------------------------------------------------------------------------------------------------------------------


def test_f1_f10_capture_imports_no_llm_sdk_and_reads_no_llm_variable():
    text = SRC.read_text(encoding="utf-8")
    assert not re.search(r"OPENAI(?!P_API_KEY)|ANTHROPIC|LLM_|MODEL_|/models", text)  # the same scan as the source-wide F10 test
    assert not re.search(r"langchain|langgraph", text, re.IGNORECASE)
    imported = set()
    for node in ast.walk(ast.parse(text)):
        if isinstance(node, ast.Import):
            imported |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    assert not imported & {"openai", "anthropic", "langchain", "langgraph", "langchain_core", "litellm"}
    environment_reads = re.findall(r"environ\.get\((\w+)", text) + re.findall(r"getenv\((\w+)", text)
    assert set(environment_reads) <= {"KEY_VARIABLE"}


def test_f7_capture_has_no_fetcher_base_class_or_registry():
    tree = ast.parse(SRC.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            bases = {ast.unparse(b) for b in node.bases}
            assert bases <= {"BaseModel", "Exception"}, (node.name, bases)
    assert not re.search(r"[Rr]egistry|[Pp]lugin|\bABC\b|\bProtocol\b", SRC.read_text(encoding="utf-8"))
    # it calls the existing client functions directly
    called = {ast.unparse(n.func) for n in ast.walk(tree) if isinstance(n, ast.Call)}
    assert {"fetch_historical_forecast", "fetch_airspace", "fetch_pib"} <= called


def test_f6_nothing_a_capture_writes_holds_the_key_or_a_request_header(tmp_path):
    root = make_root(tmp_path)
    assert run(root, Server()) == 0
    for path in (root / "fixtures" / CASE).iterdir():
        text = path.read_text(encoding="utf-8")
        assert FAKE_KEY not in text and KEY_HEADER not in text.lower() and KEY_VARIABLE not in text, path.name


class EchoesKey(Server):
    """An openAIP that wrongly repeats the key it was sent inside an item."""

    def __call__(self, request):
        response = super().__call__(request)
        if str(request.url).split("?")[0] == AIRSPACE_URL and request.url.params["page"] == "1":
            body = json.loads(response.content)
            body["items"][0]["name"] += f" {request.headers[KEY_HEADER]}"
            return httpx.Response(200, json=body)
        return response


def test_f6_a_capture_whose_output_would_hold_the_key_is_refused_and_writes_nothing(tmp_path, capsys):
    old = {"readings.yaml": b"- keep me\n"}
    root = make_root(tmp_path, files=old)
    assert run(root, EchoesKey()) == 1
    out, err = capsys.readouterr()
    assert out == "" and err == f"airspace.json: would contain the {KEY_VARIABLE} value; nothing was written\n"
    assert FAKE_KEY not in err
    assert listing(root / "fixtures" / CASE) == {**old, "case.yaml": CASE_YAML.encode()}


def test_f6_a_key_in_a_comment_of_case_yaml_is_refused_too(tmp_path, capsys):
    root = make_root(tmp_path, f"# {FAKE_KEY}\n" + CASE_YAML)
    assert run(root, Server()) == 1
    err = capsys.readouterr().err
    assert err == f"case.yaml: would contain the {KEY_VARIABLE} value; nothing was written\n"
    assert sorted(p.name for p in (root / "fixtures" / CASE).iterdir()) == ["case.yaml"]
