"""G12 / G13: flysafelz.cases.load_case_inputs() loads a committed case exactly as the tests always have.

OLD_LOAD is the loader that lived in tests/cases.py before it moved, kept here as the reference.
Mocks: httpx.MockTransport only. No key, no .env, no network.
"""

import json
import shutil
from datetime import UTC, datetime
from pathlib import Path

import cases
import httpx
import pytest
import yaml

from flysafelz.assess import assess
from flysafelz.capture import CaptureError, load_case
from flysafelz.cases import load_case_inputs, load_case_inputs_by_name
from flysafelz.clients.open_meteo import parse_response
from flysafelz.clients.openaip import fetch_airspace
from flysafelz.contracts import NotamReading
from flysafelz.pib import load_pib
from flysafelz.sources import SourceError, load_case_sources

CASES = [cases.BLUE_MILL, cases.WEST_SLEEKBURN]
REPO = Path(__file__).resolve().parents[2]


def served(body):
    content = body if isinstance(body, bytes) else json.dumps(body).encode()
    return httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(200, content=content)))


def old_load(name):
    """The inputs as the tests loaded them by hand, with case mode's PIB rule (the captured PIB is current at the case's now)."""
    folder = cases.case_dir(name)
    case, _raw = load_case(folder / "case.yaml", today=cases.TODAY)
    forecast = parse_response(json.loads((folder / "weather.json").read_text(encoding="utf-8")), case.now)
    airspace = fetch_airspace(
        case.lat, case.lon, api_key="not-a-real-key", client=served(json.loads((folder / "airspace.json").read_text(encoding="utf-8"))), now=case.now
    )
    pib = load_pib("live", case.now, client=served((folder / "pib.xml").read_bytes()), stale_rule=False)
    entries = yaml.safe_load((folder / "readings.yaml").read_text(encoding="utf-8")) or []
    readings = {r.id: r for r in (NotamReading.model_validate(item) for item in entries)}
    return case, forecast, airspace, pib, readings


def dumped(loaded):
    case, forecast, airspace, pib, readings = loaded
    return case.model_dump(), forecast.model_dump(), airspace.model_dump(), pib.model_dump(), {i: r.model_dump() for i, r in readings.items()}


@pytest.fixture
def copy_of_blue_mill(tmp_path):
    folder = tmp_path / "fixtures" / cases.BLUE_MILL
    shutil.copytree(cases.case_dir(cases.BLUE_MILL), folder)
    return tmp_path, folder


@pytest.mark.parametrize("name", CASES)
def test_the_loader_gives_what_the_old_test_loader_gave(name):
    new = load_case_inputs(cases.case_dir(name))
    assert dumped(new) == dumped(old_load(name))
    assert set(new[4]) == set(old_load(name)[4])


@pytest.mark.parametrize("name", CASES)
def test_tests_cases_load_is_the_new_loader(name):
    assert dumped(cases.load(name)) == dumped(load_case_inputs(cases.case_dir(name)))


@pytest.mark.parametrize("name", CASES)
def test_the_source_statuses_assess_returns_are_the_ones_the_old_loader_gave(name):
    case, forecast, airspace, pib, readings = old_load(name)
    old = assess(case.lat, case.lon, now=case.now, forecast=forecast, airspace=airspace, pib=pib, readings=readings, window=(case.window.start, case.window.end))
    assert cases.run(name).source_statuses == old.source_statuses


def test_the_pib_sha_is_not_copied_from_case_yaml_unlike_load_case_sources():
    loaded = load_case_inputs(cases.case_dir(cases.BLUE_MILL))
    sources = load_case_sources(cases.BLUE_MILL)
    assert loaded[3].status.sha is None
    assert sources.pib.status.sha == sources.case.capture.sha
    assert loaded[3].status.model_copy(update={"sha": sources.pib.status.sha}) == sources.pib.status
    assert loaded[3].notams == sources.pib.notams and loaded[1] == sources.forecast and loaded[2] == sources.airspace


def test_today_is_the_cases_own_now_so_a_case_later_than_the_wall_clock_loads(copy_of_blue_mill):
    _root, folder = copy_of_blue_mill
    text = (folder / "case.yaml").read_text(encoding="utf-8")
    text = text.replace("2026-10-31", "2999-12-30")
    (folder / "case.yaml").write_text(text, encoding="utf-8")
    assert datetime.now(UTC) < datetime(2999, 12, 30, tzinfo=UTC)
    case = load_case_inputs(folder)[0]
    assert case.now == datetime(2999, 12, 30, 7, 30, tzinfo=UTC)
    with pytest.raises(CaptureError, match="in the future"):
        load_case(folder / "case.yaml", today=datetime.now(UTC))  # the clock check the loader does not use


@pytest.mark.parametrize(
    ("old", "new", "field"),
    [
        ("lat: 53.802797", "lat: 12.0", "lat"),
        ("now: '2026-10-31T07:30:00Z'", "now: yesterday", "now"),
        ("now: '2026-10-31T07:30:00Z'", "now: '2026-10-31T07:30:00'", "now"),
        ("  end: '2026-10-31T18:00:00Z'", "  end: '2026-11-01T01:00:00Z'", "window"),
        ("lon: -0.086644", "lon: -0.086644\nsha: abc", "sha"),
    ],
)
def test_a_malformed_case_yaml_still_raises_capture_error_naming_the_field(copy_of_blue_mill, old, new, field):
    _root, folder = copy_of_blue_mill
    path = folder / "case.yaml"
    text = path.read_text(encoding="utf-8")
    assert old in text
    path.write_text(text.replace(old, new), encoding="utf-8")
    with pytest.raises(CaptureError, match=rf"case\.yaml: {field}\b"):
        load_case_inputs(folder)


def test_a_case_yaml_that_is_missing_or_not_a_mapping_raises_capture_error(tmp_path):
    with pytest.raises(CaptureError, match="file not found"):
        load_case_inputs(tmp_path)
    (tmp_path / "case.yaml").write_text("- a\n- list\n", encoding="utf-8")
    with pytest.raises(CaptureError, match="must be a mapping"):
        load_case_inputs(tmp_path)


@pytest.mark.parametrize("missing", ["weather.json", "airspace.json", "pib.xml", "readings.yaml"])
def test_a_missing_data_file_raises_an_error_naming_the_file(copy_of_blue_mill, missing):
    _root, folder = copy_of_blue_mill
    (folder / missing).unlink()
    with pytest.raises(SourceError, match=rf"{missing}: file not found"):
        load_case_inputs(folder)


def test_an_empty_readings_file_gives_no_readings(copy_of_blue_mill):
    _root, folder = copy_of_blue_mill
    (folder / "readings.yaml").write_text("", encoding="utf-8")
    assert load_case_inputs(folder)[4] == {}
    (folder / "readings.yaml").write_text("# only a comment\n[]\n", encoding="utf-8")
    assert load_case_inputs(folder)[4] == {}


def test_weather_that_is_not_json_is_reported_by_file(copy_of_blue_mill):
    _root, folder = copy_of_blue_mill
    (folder / "weather.json").write_text("{not json", encoding="utf-8")
    with pytest.raises(SourceError, match=r"weather\.json: not valid JSON"):
        load_case_inputs(folder)


def test_readings_that_are_not_a_list_of_readings_are_reported_by_file(copy_of_blue_mill):
    _root, folder = copy_of_blue_mill
    (folder / "readings.yaml").write_text("- id: 5\n", encoding="utf-8")
    with pytest.raises(SourceError, match=r"readings\.yaml: not a list of valid NOTAM readings"):
        load_case_inputs(folder)


@pytest.mark.parametrize("name", CASES)
def test_by_name_finds_fixtures_from_the_repo_root_whatever_the_working_directory(name, tmp_path, monkeypatch):
    here = dumped(load_case_inputs_by_name(name))
    monkeypatch.chdir(tmp_path)
    assert dumped(load_case_inputs_by_name(name)) == here == dumped(old_load(name))
    assert dumped(load_case_inputs(cases.case_dir(name))) == here


def test_by_name_takes_an_explicit_root(copy_of_blue_mill, monkeypatch, tmp_path_factory):
    root, _folder = copy_of_blue_mill
    monkeypatch.chdir(tmp_path_factory.mktemp("elsewhere"))
    assert dumped(load_case_inputs_by_name(cases.BLUE_MILL, root=root)) == dumped(old_load(cases.BLUE_MILL))


@pytest.mark.parametrize("name", ["", "nope", "../fixtures", ".git", "blue_mill_2026-10-31/../x"])
def test_by_name_refuses_an_unknown_case_in_one_line(name):
    with pytest.raises(SourceError, match="unknown case"):
        load_case_inputs_by_name(name)


def test_the_repository_root_is_worked_out_from_the_package_location():
    from flysafelz import capture

    assert capture.ROOT == REPO and (capture.ROOT / "fixtures" / cases.BLUE_MILL).is_dir()
