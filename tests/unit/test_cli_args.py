"""Argument parsing for `assess`, and every error path: exit code 2, one line on stderr, no traceback."""

import shutil
from datetime import UTC, datetime
from pathlib import Path

import pytest

from flysafelz import cli, sources
from flysafelz import pib as pib_module
from flysafelz.clients import open_meteo, openaip

BLUE_MILL = "blue_mill_2026-10-31"
REPO = Path(__file__).resolve().parents[2]
CASE = ["assess", "--case", BLUE_MILL]


def fails(capsys, argv: list[str]) -> str:
    """Run the CLI expecting exit code 2 and one stderr line. Returns the line."""
    with pytest.raises(SystemExit) as exit_info:
        cli.main(argv)
    out, err = capsys.readouterr()
    assert exit_info.value.code == 2
    assert out == ""
    assert "Traceback" not in err
    assert len(err.strip().splitlines()) == 1, err
    return err.strip()


@pytest.fixture
def no_fetch(monkeypatch):
    """Any live fetch fails the test: the argument errors below must stop before one."""

    def boom(*args, **kwargs):
        raise AssertionError("fetched")

    monkeypatch.setattr(open_meteo, "fetch_forecast", boom)
    monkeypatch.setattr(openaip, "fetch_airspace", boom)
    monkeypatch.setattr(pib_module, "load_pib", boom)


# --- parse_time ------------------------------------------------------------------------------------


@pytest.mark.parametrize("text", ["2026-10-31T07:00Z", "2026-10-31T07:00:00Z", "2026-10-31T07:00+00:00", "2026-10-31T08:00+01:00"])
def test_z_and_an_offset_are_both_accepted_and_become_utc(text):
    assert cli.parse_time(text) == datetime(2026, 10, 31, 7, 0, tzinfo=UTC)
    assert cli.parse_time(text).tzinfo == UTC


# --- the error paths ---------------------------------------------------------------------------------


def test_an_unknown_case_is_one_line_naming_the_missing_folder(capsys):
    assert fails(capsys, ["assess", "--case", "nope"]) == "flysafelz assess: error: unknown case 'nope': no folder fixtures/nope/"


@pytest.mark.parametrize("name", ["", "../fixtures", "..", ".hidden", "a/b", "/etc"])
def test_a_case_name_cannot_point_outside_fixtures(capsys, name):
    assert "unknown case" in fails(capsys, ["assess", "--case", name])


@pytest.mark.parametrize("extra", [["--lat", "53.8"], ["--lon", "-0.08"], ["--lat", "53.8", "--lon", "-0.08"]])
def test_case_together_with_lat_or_lon_is_an_argument_error(capsys, extra, no_fetch):
    assert "--case cannot be used with --lat or --lon" in fails(capsys, [*CASE, *extra])


def test_no_case_and_no_coordinates_is_an_argument_error(capsys):
    assert "give --case, or both --lat and --lon" in fails(capsys, ["assess"])


@pytest.mark.parametrize("extra", [["--lat", "53.8"], ["--lon", "-0.08"]])
def test_only_one_coordinate_is_an_argument_error(capsys, extra, no_fetch):
    assert "give --case, or both --lat and --lon" in fails(capsys, ["assess", *extra])


@pytest.mark.parametrize(
    "window",
    [
        ["--start", "2026-10-31T09:00Z", "--end", "2026-10-31T09:00Z"],  # equal
        ["--start", "2026-10-31T09:00Z", "--end", "2026-10-31T08:00Z"],  # reversed
        ["--start", "2026-10-31T09:00Z", "--end", "2026-10-31T09:00+00:00"],  # equal, spelt differently
        ["--end", "2026-10-31T07:00Z"],  # case start 07:00, so the end is not after it
        ["--start", "2026-10-31T18:00Z"],  # case end 18:00, so the start is not before it
    ],
)
def test_an_end_that_is_not_after_the_start_is_an_argument_error_in_case_mode(capsys, window):
    assert "--end must be after --start" in fails(capsys, [*CASE, *window])


def test_an_end_that_is_not_after_the_start_is_an_argument_error_in_live_mode(capsys, no_fetch):
    argv = ["assess", "--lat", "53.8", "--lon", "-0.08", "--start", "2026-10-31T09:00Z", "--end", "2026-10-31T08:00Z"]
    assert "--end must be after --start" in fails(capsys, argv)


@pytest.mark.parametrize("flag", ["--start", "--end"])
@pytest.mark.parametrize("text", ["2026-10-31T07:00", "2026-10-31", "2026-10-31 07:00"])
def test_a_time_without_a_timezone_is_an_argument_error(capsys, flag, text):
    line = fails(capsys, [*CASE, flag, text])
    assert f"argument {flag}" in line and "no timezone" in line


@pytest.mark.parametrize("text", ["tomorrow", "07:00", "2026-13-45T00:00Z"])
def test_a_time_that_is_not_a_time_is_an_argument_error(capsys, text):
    assert "is not an ISO 8601 time" in fails(capsys, [*CASE, "--start", text])


def test_live_mode_wants_both_ends_of_the_window_or_neither(capsys, no_fetch):
    assert "give both --start and --end, or neither" in fails(capsys, ["assess", "--lat", "53.8", "--lon", "-0.08", "--start", "2026-10-31T07:00Z"])
    assert "give both --start and --end, or neither" in fails(capsys, ["assess", "--lat", "53.8", "--lon", "-0.08", "--end", "2026-10-31T07:00Z"])


@pytest.mark.parametrize("coordinates", [["--lat", "10", "--lon", "0"], ["--lat", "53.8", "--lon", "10"], ["--lat", "nan", "--lon", "0"], ["--lat", "70", "--lon", "-1"]])
def test_an_lz_outside_the_uk_is_the_outside_uk_coverage_message_before_any_fetch(capsys, coordinates, no_fetch):
    assert fails(capsys, ["assess", *coordinates]) == "flysafelz assess: error: outside UK coverage"


def test_a_coordinate_that_is_not_a_number_is_an_argument_error(capsys):
    assert "argument --lat" in fails(capsys, ["assess", "--lat", "north", "--lon", "0"])


def test_no_command_and_an_unknown_command_are_errors_too(capsys):
    assert "required" in fails(capsys, [])
    fails(capsys, ["forecast"])


# --- no way to change a limit (F8) ---------------------------------------------------------------------


@pytest.mark.parametrize("flag", ["--rules", "--rules-path", "--local-area", "--limit", "--wind-limit", "--gust-spread"])
def test_assess_has_no_flag_for_a_limit_or_the_local_area(capsys, flag):
    assert "unrecognized arguments" in fails(capsys, [*CASE, flag, "99"])


# --- a damaged case folder ------------------------------------------------------------------------------


@pytest.fixture
def copy_of_the_case(tmp_path, monkeypatch):
    shutil.copytree(REPO / "fixtures" / BLUE_MILL, tmp_path / "fixtures" / BLUE_MILL)
    monkeypatch.setattr(sources, "ROOT", tmp_path)
    return tmp_path / "fixtures" / BLUE_MILL


@pytest.mark.parametrize("name", ["case.yaml", "weather.json", "airspace.json", "pib.xml", "readings.yaml"])
def test_a_missing_case_file_is_one_line_naming_it(capsys, copy_of_the_case, name):
    (copy_of_the_case / name).unlink()
    assert fails(capsys, CASE).endswith(f"fixtures/{BLUE_MILL}/{name}: file not found")


def test_readings_that_are_not_readings_are_one_line_naming_the_file(capsys, copy_of_the_case):
    (copy_of_the_case / "readings.yaml").write_text("- id: J3857/26\n  category: nonsense\n", encoding="utf-8")
    assert fails(capsys, CASE).endswith(f"fixtures/{BLUE_MILL}/readings.yaml: not a list of valid NOTAM readings")


def test_weather_that_is_not_json_is_one_line_naming_the_file(capsys, copy_of_the_case):
    (copy_of_the_case / "weather.json").write_text("{not json", encoding="utf-8")
    assert fails(capsys, CASE).endswith(f"fixtures/{BLUE_MILL}/weather.json: not valid JSON")


# --- the window fallbacks (case mode) -----------------------------------------------------------------------


def window_line(capsys, argv: list[str]) -> str:
    assert cli.main(argv) == 0
    out = capsys.readouterr().out
    return next(line for line in out.splitlines() if line.startswith("Window "))


def test_the_case_window_is_used_when_start_and_end_are_left_out(capsys):
    assert window_line(capsys, CASE) == "Window 2026-10-31T07:00Z to 2026-10-31T18:00Z (end exclusive). All times UTC."


def test_one_of_start_and_end_falls_back_to_the_case_window_for_the_other(capsys):
    assert window_line(capsys, [*CASE, "--start", "2026-10-31T09:00Z"]) == "Window 2026-10-31T09:00Z to 2026-10-31T18:00Z (end exclusive). All times UTC."
    assert window_line(capsys, [*CASE, "--end", "2026-10-31T09:00Z"]) == "Window 2026-10-31T07:00Z to 2026-10-31T09:00Z (end exclusive). All times UTC."


def test_z_and_plus_00_00_give_the_same_output(capsys):
    cli.main([*CASE, "--start", "2026-10-31T07:00Z", "--end", "2026-10-31T10:00Z"])
    z = capsys.readouterr().out
    cli.main([*CASE, "--start", "2026-10-31T07:00+00:00", "--end", "2026-10-31T11:00+01:00"])
    assert capsys.readouterr().out == z


def test_a_start_with_an_offset_is_read_as_the_same_instant_in_utc(capsys):
    assert window_line(capsys, [*CASE, "--start", "2026-10-31T09:00+01:00", "--end", "2026-10-31T10:00Z"]) == ("Window 2026-10-31T08:00Z to 2026-10-31T10:00Z (end exclusive). All times UTC.")
