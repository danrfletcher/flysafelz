"""G11 / E15 / E17 / G15 (capture side): `make fixtures` for a case with `days`, through capture.main.

Every request goes to a MultiDayServer (tests/capture_helpers.py): an httpx.MockTransport whose Open-Meteo
answer is built from the request's own start_date and end_date. `today` is frozen, the key is a fake one read
from the environment dict main is given, and no socket opens (the suite runs with --disable-socket).
"""

import ast
import json
import re
from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx
import pytest
import yaml
from capture_helpers import FAKE_KEY, MIDDLE, OLDEST, TODAY, case_yaml, listing, london_date_of, make_root, run, server_for
from test_capture import FAILURES

from flysafelz import capture
from flysafelz.capture import CaptureError, load_case
from flysafelz.clients.open_meteo import FORECAST_URL, HISTORICAL_URL, parse_response
from flysafelz.clients.pib_source import ARCHIVE_COMMITS_URL
from flysafelz.contracts import SourceState
from flysafelz.daylight import london_day_utc
from flysafelz.rules import RULES

NAME = "blue_mill_2026-10-31_5day"
ONE_HOUR = timedelta(hours=1)
SRC = Path(capture.__file__)
REPO = Path(__file__).resolve().parents[1]


def horizon() -> int:
    """The last London day that can carry a verdict, from rules.yaml."""
    return int(RULES.outlook_horizon_day.value)  # type: ignore[arg-type]


def utc(*parts: int) -> datetime:
    return datetime(*parts, tzinfo=UTC)


def text(t: datetime) -> str:
    return t.strftime("%Y-%m-%dT%H:%M:%SZ")


GMT_NOW = utc(2026, 10, 31, 7, 30)
BST_NOW = utc(2026, 9, 1, 7, 30)
CHANGE_NOW = utc(2026, 10, 24, 7, 30)  # BST; the clocks go back on 25 Oct


def expected_span(now: datetime, days: int) -> tuple[datetime, datetime, int]:
    """(first UTC hour, last UTC hour, row count) worked out from the London days, not from capture."""
    day0 = london_date_of(now)
    start = london_day_utc(day0)[0]
    end = london_day_utc(day0 + timedelta(days=days - 1))[1]
    return start, end, int((end - start) / ONE_HOUR) + 1


def whole_span_case(now: datetime, days: int, **kwargs) -> str:
    start, end, _ = expected_span(now, days)
    return case_yaml(now=text(now), start=text(start), end=text(end), days=days, **kwargs)


def captured(tmp_path: Path, now: datetime, days: int, *, name: str = NAME):
    """Capture a whole-span case; returns (root, server)."""
    root = make_root(tmp_path, name, whole_span_case(now, days))
    server = server_for(now)
    assert run(root, name, server) == 0
    return root, server


def weather_times(root: Path, name: str = NAME) -> list[str]:
    return json.loads((root / "fixtures" / name / "weather.json").read_text(encoding="utf-8"))["hourly"]["time"]


# --- `days` validation: one line naming the field, before any request --------------------------------------

DAYS_OK_WINDOW = ("2026-10-31T07:00:00Z", "2026-10-31T18:00:00Z")


@pytest.mark.parametrize(
    "days",
    [0, -1, 2.5, "three", "'3'", "true", "[3]", 9, 100],
    ids=["zero", "negative", "fraction", "word", "quoted-number", "boolean", "list", "nine", "hundred"],
)
def test_a_bad_days_value_is_one_line_naming_days_and_makes_no_request(tmp_path, capsys, days):
    root = make_root(tmp_path, NAME, case_yaml(now="2026-10-31T07:30:00Z", start=DAYS_OK_WINDOW[0], end=DAYS_OK_WINDOW[1], days=days))
    server = server_for(GMT_NOW)
    assert run(root, NAME, server) == 1
    out, err = capsys.readouterr()
    assert out == "" and err.count("\n") == 1 and re.search(r"case\.yaml: days\b", err), err
    assert server.seen == []
    assert [p.name for p in (root / "fixtures" / NAME).iterdir()] == ["case.yaml"]


def test_days_above_the_cap_names_the_london_date_of_the_last_day(tmp_path, capsys):
    cap = horizon() + 1
    root = make_root(tmp_path, NAME, case_yaml(now="2026-10-31T07:30:00Z", start=DAYS_OK_WINDOW[0], end=DAYS_OK_WINDOW[1], days=cap + 1))
    assert run(root, NAME, server_for(GMT_NOW)) == 1
    err = capsys.readouterr().err
    assert f"at most {cap}" in err and "2026-11-07" in err  # London day 7 from 31 Oct


def test_a_days_case_loads_with_days_and_a_case_without_days_has_none(tmp_path):
    root = make_root(tmp_path, NAME, whole_span_case(GMT_NOW, 5))
    case, raw = load_case(root / "fixtures" / NAME / "case.yaml", today=TODAY)
    assert case.days == 5 and raw["days"] == 5
    other = make_root(tmp_path / "x", NAME, case_yaml(now="2026-10-31T07:30:00Z", start=DAYS_OK_WINDOW[0], end=DAYS_OK_WINDOW[1], days=None))
    assert load_case(other / "fixtures" / NAME / "case.yaml", today=TODAY)[0].days is None


# --- the window: London days, not UTC ----------------------------------------------------------------------

BST_DAY0 = "2026-10-23T23:00:00Z"  # London day 0 of 24 Oct starts at 23:00 UTC the evening before


@pytest.mark.parametrize(
    ("days", "start", "end", "accepted", "named"),
    [
        (3, "2026-10-23T23:30:00Z", "2026-10-27T00:00:00Z", True, None),  # E17: starts inside London day 0
        (3, BST_DAY0, "2026-10-27T00:00:00Z", True, None),  # exactly day 0's start, exactly day 2's end
        (3, "2026-10-23T22:30:00Z", "2026-10-27T00:00:00Z", False, "2026-10-24"),  # E17: before London day 0
        (3, "2026-10-24T07:00:00Z", "2026-10-27T00:30:00Z", False, "2026-10-26"),  # ends after London day 2
        (3, "2026-10-24T07:00:00Z", "2026-10-24T07:00:00Z", False, "2026-10-24"),  # not after its start
        (3, "2026-10-25T07:00:00Z", "2026-10-24T07:00:00Z", False, "2026-10-24"),  # reversed
        (1, "2026-10-23T23:30:00Z", "2026-10-24T23:00:00Z", True, None),  # an explicit days: 1 uses London days (UTC would refuse)
        (1, "2026-10-24T07:00:00Z", "2026-10-25T00:00:00Z", False, "2026-10-24"),  # ... and 25 Oct 00:00 BST is not in day 0
    ],
)
def test_the_window_must_sit_inside_london_day_0_to_day_n_minus_1(tmp_path, capsys, days, start, end, accepted, named):
    root = make_root(tmp_path, NAME, case_yaml(now=text(CHANGE_NOW), start=start, end=end, days=days))
    server = server_for(CHANGE_NOW)
    code = run(root, NAME, server)
    out, err = capsys.readouterr()
    if accepted:
        assert code == 0 and err == ""
        return
    assert code == 1 and out == ""
    assert err.count("\n") == 1 and "case.yaml: window:" in err and named in err, err
    assert server.seen == []
    assert [p.name for p in (root / "fixtures" / NAME).iterdir()] == ["case.yaml"]


def test_a_case_without_days_keeps_the_utc_case_day_check(tmp_path, capsys):
    """Ticket 2's rule, untouched: without `days` a window may not leave the UTC day, whatever London says."""
    root = make_root(tmp_path, NAME, case_yaml(now=text(CHANGE_NOW), start="2026-10-23T23:30:00Z", end="2026-10-24T18:00:00Z", days=None))
    server = server_for(CHANGE_NOW)
    assert run(root, NAME, server) == 1
    err = capsys.readouterr().err
    assert "window: must start before it ends and stay inside the case day 2026-10-24 (UTC)" in err
    assert server.seen == []


# --- the cap comes from the rules --------------------------------------------------------------------------


def test_days_up_to_the_outlook_horizon_plus_one_are_accepted_and_one_more_is_refused(tmp_path, capsys):
    cap = horizon() + 1  # London days 0..horizon
    now = BST_NOW
    root, _ = captured(tmp_path, now, cap)
    _, end, rows = expected_span(now, cap)
    assert len(weather_times(root)) == rows and weather_times(root)[-1] == end.strftime("%Y-%m-%dT%H:%M")
    capsys.readouterr()
    refused = make_root(tmp_path / "over", NAME, whole_span_case(now, cap).replace(f"days: {cap}", f"days: {cap + 1}"))
    server = server_for(now)
    assert run(refused, NAME, server) == 1
    err = capsys.readouterr().err
    assert err.count("\n") == 1 and "days" in err and server.seen == []


def test_the_cap_follows_the_rule_not_a_literal(tmp_path, monkeypatch, capsys):
    shorter = RULES.model_copy(update={"outlook_horizon_day": RULES.outlook_horizon_day.model_copy(update={"value": 2})})
    monkeypatch.setattr(capture, "RULES", shorter)
    root, _ = captured(tmp_path, BST_NOW, 3)  # London days 0..2 now fit
    assert len(weather_times(root)) == expected_span(BST_NOW, 3)[2]
    refused = make_root(tmp_path / "over", NAME, whole_span_case(BST_NOW, 3).replace("days: 3", "days: 4"))
    assert run(refused, NAME, server_for(BST_NOW)) == 1
    assert "at most 3" in capsys.readouterr().err


# --- weather.json: one row per UTC hour, London days ------------------------------------------------------

COVERAGE = [
    pytest.param(GMT_NOW, 5, 121, id="gmt-5-days"),  # E17 control: 31 Oct 00:00Z to 5 Nov 00:00Z
    pytest.param(GMT_NOW, 1, 25, id="gmt-1-day"),
    pytest.param(BST_NOW, 1, 25, id="bst-1-day"),
    pytest.param(BST_NOW, 5, 121, id="bst-5-days"),
    pytest.param(BST_NOW, 8, 193, id="bst-8-days"),
    pytest.param(CHANGE_NOW, 3, 74, id="bst-to-gmt-3-days"),  # E17: 25 Oct is 25 hours
    pytest.param(utc(2026, 10, 25, 7, 30), 2, 50, id="clocks-go-back-day-0"),  # day 0 is 25 hours
    pytest.param(utc(2026, 3, 28, 12, 0), 3, 72, id="clocks-go-forward-in-day-1"),  # 29 Mar is 23 hours
    pytest.param(utc(2026, 3, 29, 7, 30), 1, 24, id="clocks-go-forward-day-0"),
    pytest.param(utc(2026, 3, 30, 7, 30), 4, 97, id="bst-after-the-spring-change"),
    pytest.param(utc(2026, 6, 30, 23, 30), 2, 49, id="bst-now-after-23:00-utc-is-the-next-london-day"),
]


@pytest.mark.parametrize(("now", "days", "rows"), COVERAGE)
def test_weather_json_has_a_row_for_every_utc_hour_of_the_london_days_plus_the_end_hour(tmp_path, now, days, rows):
    start, end, expected = expected_span(now, days)
    assert expected == rows  # the arithmetic itself, so a wrong table entry shows up here
    root, server = captured(tmp_path, now, days)
    times = weather_times(root)
    assert len(times) == rows
    assert times[0] == start.strftime("%Y-%m-%dT%H:%M") and times[-1] == end.strftime("%Y-%m-%dT%H:%M")
    stamps = [datetime.fromisoformat(t).replace(tzinfo=UTC) for t in times]
    assert all(b - a == ONE_HOUR for a, b in zip(stamps, stamps[1:], strict=False))  # no gap, no repeat
    # exactly one Open-Meteo request, for the UTC dates that cover the span
    weather = [r for r in server.seen if r.url.host.endswith("open-meteo.com")]
    assert len(weather) == 1 and str(weather[0].url).split("?")[0] == HISTORICAL_URL
    assert (weather[0].url.params["start_date"], weather[0].url.params["end_date"]) == (str(start.date()), str(end.date()))
    assert not any(url == FORECAST_URL for url in server.urls)


def test_the_two_clock_change_boundaries_of_the_e17_case(tmp_path):
    root, _ = captured(tmp_path, CHANGE_NOW, 3)
    times = weather_times(root)
    assert (times[0], times[-1]) == ("2026-10-23T23:00", "2026-10-27T00:00")
    day_1 = [t for t in times if "2026-10-24T23:00" <= t < "2026-10-26T00:00"]
    assert len(day_1) == 25  # 25 Oct in London: 24 Oct 23:00Z up to, not including, 26 Oct 00:00Z
    assert times.count("2026-10-25T00:00") == 1 and times.count("2026-10-25T23:00") == 1


def test_weather_json_is_the_existing_shape_and_open_meteo_parse_response_reads_it(tmp_path):
    root, _ = captured(tmp_path, GMT_NOW, 5)
    body_text = (root / "fixtures" / NAME / "weather.json").read_text(encoding="utf-8")
    body = json.loads(body_text)
    assert body_text == json.dumps(body, sort_keys=True, indent=2) + "\n"  # sorted keys, two-space indent, final newline
    assert "generationtime_ms" not in body
    forecast = parse_response(body, GMT_NOW)
    assert forecast.status.state is SourceState.OK and forecast.elevation_m == 14.0
    assert len(forecast.samples) == 121
    assert utc(2026, 11, 5, 0, 0) in forecast.samples and utc(2026, 10, 31, 0, 0) in forecast.samples
    assert (body["model"], body["timezone"], body["wind_speed_unit"]) == ("ukmo_seamless", "UTC", "mph")


def test_a_body_missing_the_end_hour_is_one_line_and_the_old_fixture_stays_byte_identical(tmp_path, capsys):
    start, end, _ = expected_span(GMT_NOW, 5)
    old = {"weather.json": b"old weather", "airspace.json": b"old airspace", "pib.xml": b"old pib", "readings.yaml": b"- keep me\n"}
    root = make_root(tmp_path, NAME, whole_span_case(GMT_NOW, 5), files=old)
    before = listing(root / "fixtures" / NAME)
    assert run(root, NAME, server_for(GMT_NOW, drop={end})) == 1
    out, err = capsys.readouterr()
    assert out == "" and err.count("\n") == 1 and "Open-Meteo" in err and text(end) in err
    assert listing(root / "fixtures" / NAME) == before


def test_the_first_missing_hour_is_the_one_named(tmp_path, capsys):
    gap = {utc(2026, 11, 2, 5, 0), utc(2026, 11, 1, 9, 0), utc(2026, 11, 3, 5, 0)}
    root = make_root(tmp_path, NAME, whole_span_case(GMT_NOW, 5))
    assert run(root, NAME, server_for(GMT_NOW, drop=gap)) == 1
    err = capsys.readouterr().err
    assert err.count("\n") == 1 and "2026-11-01T09:00:00Z" in err and "3 rows missing" in err
    assert [p.name for p in (root / "fixtures" / NAME).iterdir()] == ["case.yaml"]


# --- the PIB: still one, at `now` -------------------------------------------------------------------------


def test_the_pib_is_one_archive_pib_at_now_and_capture_is_written_back(tmp_path):
    root, server = captured(tmp_path, GMT_NOW, 5)
    commits = [r for r in server.seen if str(r.url).startswith(ARCHIVE_COMMITS_URL)]
    raw = [u for u in server.urls if u.startswith("https://raw.githubusercontent.com")]
    assert len(commits) == 1 and commits[0].url.params["until"] == "2026-10-31T07:30:00Z"
    assert len(raw) == 1 and MIDDLE in raw[0]
    case = yaml.safe_load((root / "fixtures" / NAME / "case.yaml").read_text(encoding="utf-8"))
    assert case["capture"] == {"pib_issued": "2026-10-31T07:00:37Z", "sha": MIDDLE} and case["days"] == 5
    assert (root / "fixtures" / NAME / "pib.xml").read_bytes().startswith(b"<?xml")


def test_a_pinned_archive_sha_still_wins_with_days(tmp_path):
    pinned = OLDEST  # an older archive commit than the one `now` would pick
    root = make_root(tmp_path, NAME, whole_span_case(GMT_NOW, 2) + f"archive_sha: {pinned}\n")
    server = server_for(GMT_NOW)
    assert run(root, NAME, server) == 0
    assert ARCHIVE_COMMITS_URL not in server.urls
    assert yaml.safe_load((root / "fixtures" / NAME / "case.yaml").read_text(encoding="utf-8"))["capture"]["sha"] == pinned


# --- E15: a day that has not ended ------------------------------------------------------------------------

E15_CASE = case_yaml(now="2026-10-31T07:30:00Z", start="2026-10-31T07:00:00Z", end="2026-11-02T00:00:00Z", days=2)  # London day 1 = 1 Nov
DAY_END = utc(2026, 11, 2, 0, 0)  # the end of London day 1 (GMT)


def test_e15_a_day_that_ends_a_minute_after_today_is_refused_with_one_line_and_nothing_written(tmp_path, capsys):
    old = {"weather.json": b"old weather"}
    root = make_root(tmp_path, NAME, E15_CASE, files=old)
    server = server_for(GMT_NOW)
    assert run(root, NAME, server, today=DAY_END - timedelta(minutes=1)) == 1
    out, err = capsys.readouterr()
    assert out == "" and err.count("\n") == 1
    assert "days:" in err and "2026-11-01" in err and "has not ended" in err and FAKE_KEY not in err
    assert server.seen == []
    assert listing(root / "fixtures" / NAME) == {"case.yaml": E15_CASE.encode(), "weather.json": b"old weather"}


@pytest.mark.parametrize("delta", [timedelta(0), timedelta(minutes=1)], ids=["exactly-at-the-end", "a-minute-after"])
def test_e15_a_day_that_has_ended_is_captured(tmp_path, delta):
    root = make_root(tmp_path, NAME, E15_CASE)
    assert run(root, NAME, server_for(GMT_NOW), today=DAY_END + delta) == 0
    assert len(weather_times(root)) == 49  # 31 Oct 00:00Z to 2 Nov 00:00Z, inclusive


def test_e15_uses_the_london_day_so_a_bst_day_ends_an_hour_before_the_utc_one(tmp_path, capsys):
    now = utc(2026, 9, 1, 7, 30)  # London day 0 ends 2026-09-01T23:00Z
    text_ = case_yaml(now=text(now), start="2026-09-01T07:00:00Z", end="2026-09-01T23:00:00Z", days=1)
    root = make_root(tmp_path, NAME, text_)
    server = server_for(now)
    assert run(root, NAME, server, today=utc(2026, 9, 1, 22, 59)) == 1
    assert "2026-09-01" in capsys.readouterr().err and server.seen == []
    assert run(root, NAME, server, today=utc(2026, 9, 1, 23, 1)) == 0  # a UTC-day rule would still refuse until 2 Sep


def test_e15_now_in_the_future_still_fires_first_with_its_own_message(tmp_path, capsys):
    root = make_root(tmp_path, NAME, E15_CASE)
    server = server_for(GMT_NOW)
    assert run(root, NAME, server, today=GMT_NOW - timedelta(minutes=1)) == 1
    err = capsys.readouterr().err
    assert err.count("\n") == 1
    assert err.endswith("case.yaml: now: 2026-10-31T07:30:00Z is in the future; a case replays the past\n")
    assert "has not ended" not in err and server.seen == []


def test_e15_the_replay_path_never_refuses(tmp_path):
    root = make_root(tmp_path, NAME, E15_CASE)
    path = root / "fixtures" / NAME / "case.yaml"
    case, _ = load_case(path, today=datetime.max.replace(tzinfo=UTC))
    assert case.days == 2
    with pytest.raises(CaptureError, match="has not ended"):  # only capture() asks, with its own `today`
        capture.refuse_a_day_that_has_not_ended(path, case, DAY_END - timedelta(minutes=1))
    capture.refuse_a_day_that_has_not_ended(path, case, datetime.max.replace(tzinfo=UTC))


# --- reruns and case.yaml -----------------------------------------------------------------------------------


def test_a_rerun_of_a_days_case_with_an_unchanged_server_is_idempotent(tmp_path, capsys):
    root = make_root(tmp_path, NAME, whole_span_case(GMT_NOW, 5))
    assert run(root, NAME, server_for(GMT_NOW)) == 0
    first = listing(root / "fixtures" / NAME)
    capsys.readouterr()
    assert run(root, NAME, server_for(GMT_NOW)) == 0
    assert listing(root / "fixtures" / NAME) == first
    out = capsys.readouterr().out
    assert out.count(": unchanged") == 4 and ": written" not in out  # weather, airspace, pib, case.yaml


COMMENT = "# Blue Mill, five days. Keep this note.\n"


def test_the_capture_edit_of_case_yaml_keeps_dans_comments_and_the_days_line(tmp_path):
    original = COMMENT + whole_span_case(GMT_NOW, 5).replace("days: 5", "days: 5   # London days, day 0 is today")
    root = make_root(tmp_path, NAME, original)
    path = root / "fixtures" / NAME / "case.yaml"
    assert run(root, NAME, server_for(GMT_NOW)) == 0
    first = path.read_text(encoding="utf-8")
    assert first.startswith(original)  # every line he wrote, the days line included, is still there
    assert "days: 5   # London days, day 0 is today\n" in first
    assert yaml.safe_load(first)["capture"] == {"pib_issued": "2026-10-31T07:00:37Z", "sha": MIDDLE}
    stale = original.replace("window:", "capture:\n  pib_issued: '2026-10-30T00:00:00Z'\n  sha: '" + "0" * 40 + "'\n# about the window\nwindow:")
    path.write_text(stale, encoding="utf-8")
    assert run(root, NAME, server_for(GMT_NOW)) == 0  # a capture block in the middle is replaced where it stands
    second = path.read_text(encoding="utf-8")
    assert second == stale.replace("  pib_issued: '2026-10-30T00:00:00Z'\n  sha: '" + "0" * 40 + "'", "  pib_issued: '2026-10-31T07:00:37Z'\n  sha: " + MIDDLE)
    assert yaml.safe_load(second)["days"] == 5


# --- source errors during a days capture --------------------------------------------------------------------


@pytest.mark.parametrize(("name", "url", "outcome", "line"), FAILURES, ids=[f[0] for f in FAILURES])
def test_a_source_error_during_a_days_capture_is_one_line_and_writes_nothing(tmp_path, capsys, name, url, outcome, line):
    case_text = whole_span_case(GMT_NOW, 5)
    root = make_root(tmp_path, NAME, case_text, files={"weather.json": b"old"})
    server = server_for(GMT_NOW, fail={url: outcome})
    assert run(root, NAME, server) == 1
    out, err = capsys.readouterr()
    assert out == "" and err.count("\n") == 1
    assert line in err, err
    assert FAKE_KEY not in err
    assert listing(root / "fixtures" / NAME) == {"case.yaml": case_text.encode(), "weather.json": b"old"}


# --- the code path uses the London-day helpers and nothing hand-rolled (E17) ---------------------------------

DAYS_PATH = ["_days_within_the_outlook", "_inside_the_london_days", "weather_json_days", "_fetch_weather_days", "refuse_a_day_that_has_not_ended"]


def _functions(tree: ast.AST) -> dict[str, ast.AST]:
    found = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            found[node.name] = node
    return found


def _names(node: ast.AST) -> set[str]:
    return {n.id for n in ast.walk(node) if isinstance(n, ast.Name)} | {n.attr for n in ast.walk(node) if isinstance(n, ast.Attribute)}


def test_every_day_boundary_on_the_days_path_comes_from_the_london_day_helpers():
    functions = _functions(ast.parse(SRC.read_text(encoding="utf-8")))
    assert set(DAYS_PATH) <= set(functions)
    forbidden = {"case_day", "ONE_DAY", "LAST_MINUTE", "timedelta", "UTC", "ZoneInfo", "tzinfo", "astimezone", "utcoffset", "replace"}
    used = set()
    for name in DAYS_PATH:
        names = _names(functions[name])
        assert not names & forbidden, (name, names & forbidden)
        used |= names
    assert {"london_days_utc", "london_day_date"} <= used
    # the single-day path is still its own thing
    assert "case_day" in _names(functions["weather_json"]) and "case_day" in _names(functions["_fetch_weather"])


def test_the_days_cap_has_no_literal_and_reads_the_horizon_rule():
    functions = _functions(ast.parse(SRC.read_text(encoding="utf-8")))
    validator = functions["_days_within_the_outlook"]
    assert "outlook_horizon_day" in _names(validator)
    numbers = {n.value for n in ast.walk(validator) if isinstance(n, ast.Constant) and isinstance(n.value, int) and not isinstance(n.value, bool)}
    assert numbers <= {0, 1}, numbers  # "at least 1" and "one more than the last day"; never 7 or 8


def test_days_adds_no_environment_read():
    source = SRC.read_text(encoding="utf-8")
    reads = re.findall(r"environ\.get\((\w+)", source) + re.findall(r"getenv\((\w+)", source)
    assert set(reads) <= {"KEY_VARIABLE"}


# --- G15: committed cases are untouched ---------------------------------------------------------------------


def test_no_committed_case_yaml_has_a_days_key():
    """The cases committed before `days` existed are untouched. Dan's labelled cases (5a PR-1.S1) are the exception: some were captured with
    `days` and are committed byte for byte, so the folders in labelled-fixtures.sha256 are left out."""
    labelled = {line.split("  ", 1)[1].removeprefix("./").split("/")[0] for line in (REPO / "fixtures" / "labelled-fixtures.sha256").read_text(encoding="utf-8").splitlines() if line.strip()}
    case_files = sorted(p for p in (REPO / "fixtures").glob("*/case.yaml") if p.parent.name not in labelled)
    assert case_files
    for path in case_files:
        assert "days" not in yaml.safe_load(path.read_text(encoding="utf-8")), path


def test_the_makefile_and_docstring_state_the_never_recapture_rule():
    makefile = (REPO / "Makefile").read_text(encoding="utf-8")
    assert "Recapture one case" not in makefile
    assert "days" in makefile and "never recaptured" in makefile and "new case name" in makefile
    doc = capture.__doc__ or ""
    assert "days: N" in doc and "never recaptured" in doc and "new case name" in doc
