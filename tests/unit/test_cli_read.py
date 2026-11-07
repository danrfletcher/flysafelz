"""`cli read` and `cli assess --reader`: argument parsing, the GP3 layout, the count lines, the missing-settings message, no traceback.

No model is called here: the layout tests hand `read_lines` ReaderResults built by hand, and the argument tests stop before a model is
built. The whole chain through a fake model is tests/integration/test_reader_blue_mill.py.
"""

from datetime import UTC, datetime
from pathlib import Path

import pytest
from factories import LZ, circle_at, notam, reading, utc
from reader_helpers import FAKE_MODEL, blue_mill_notams

from flysafelz import cli
from flysafelz import pib as pib_module
from flysafelz.clients.pib_source import PibFetch
from flysafelz.contracts import ContactKind, SourceName, SourceState, SourceStatus
from flysafelz.reader import config as reader_config
from flysafelz.reader.contract import ReaderResult, ReaderStatus

BLUE_MILL = "blue_mill_2026-10-31"
CASE = ["read", "--case", BLUE_MILL]
LIVE = ["read", "--live", "--lat", "53.802797", "--lon", "-0.086644", "--start", "2026-10-31T07:00Z", "--end", "2026-10-31T18:00Z"]
WINDOW = (utc(7), utc(18))
FIXTURES = Path(__file__).resolve().parents[2] / "fixtures"
ONE = "07700 900123"


def fails(capsys, argv: list[str]) -> str:
    """Run the CLI expecting exit code 2 and one stderr line, no traceback and nothing on stdout. Returns the line."""
    with pytest.raises(SystemExit) as exit_info:
        cli.main(argv)
    out, err = capsys.readouterr()
    assert exit_info.value.code == 2
    assert out == ""
    assert "Traceback" not in err
    assert len(err.strip().splitlines()) == 1, err
    return err.strip()


@pytest.fixture
def no_model(monkeypatch):
    """Building the model (after the arguments are checked) fails the test."""

    def boom(*args, **kwargs):
        raise AssertionError("a model was built")

    monkeypatch.setattr(reader_config, "build_llm", boom)


@pytest.fixture
def no_fetch(monkeypatch):
    def boom(*args, **kwargs):
        raise AssertionError("fetched")

    monkeypatch.setattr(pib_module, "fetch_pib", boom)


# --- arguments ---------------------------------------------------------------------------------------------------------------------


def test_an_unknown_case_is_one_line_naming_the_missing_folder(capsys, no_model):
    assert fails(capsys, ["read", "--case", "nope"]) == "flysafelz read: error: unknown case 'nope': no folder fixtures/nope/"


def test_a_case_without_a_pib_is_one_line_naming_the_missing_file(capsys, no_model, monkeypatch, tmp_path):
    from flysafelz import sources

    folder = tmp_path / "fixtures" / "bare"
    folder.mkdir(parents=True)
    (folder / "case.yaml").write_bytes((FIXTURES / BLUE_MILL / "case.yaml").read_bytes())
    for name in ("weather.json", "airspace.json"):
        (folder / name).write_bytes((FIXTURES / BLUE_MILL / name).read_bytes())
    monkeypatch.setattr(sources, "ROOT", tmp_path)
    assert fails(capsys, ["read", "--case", "bare"]).endswith("fixtures/bare/pib.xml: file not found")


@pytest.mark.parametrize("extra", [["--lat", "53.8"], ["--lon", "-0.08"], ["--lat", "53.8", "--lon", "-0.08"]])
def test_case_together_with_lat_or_lon_is_an_argument_error(capsys, no_model, extra):
    assert "--case cannot be used with --lat or --lon" in fails(capsys, [*CASE, *extra])


def test_case_together_with_live_is_an_argument_error(capsys, no_model):
    assert "--case cannot be used with --live" in fails(capsys, [*CASE, "--live"])


def test_neither_case_nor_live_is_an_argument_error(capsys, no_model):
    assert "give --case, or --live" in fails(capsys, ["read"])
    assert "give --case, or --live" in fails(capsys, ["read", "--lat", "53.8", "--lon", "-0.08"])


@pytest.mark.parametrize("missing", ["--lat", "--lon", "--start", "--end"])
def test_live_needs_all_four_of_lat_lon_start_end(capsys, no_model, no_fetch, missing):
    argv = list(LIVE)
    at = argv.index(missing)
    del argv[at : at + 2]
    assert "--live needs --lat, --lon, --start and --end" in fails(capsys, argv)


@pytest.mark.parametrize("text", ["2026-10-31T07:00", "2026-10-31", "2026-10-31 07:00"])
@pytest.mark.parametrize("flag", ["--start", "--end"])
def test_the_time_zone_rule_is_the_one_assess_has(capsys, no_model, flag, text):
    line = fails(capsys, [*CASE, flag, text])
    assert f"argument {flag}" in line and "no timezone" in line


def test_z_and_an_offset_are_one_instant(capsys, no_model):
    assert cli.parse_time("2026-10-31T08:00+01:00") == cli.parse_time("2026-10-31T07:00Z") == datetime(2026, 10, 31, 7, tzinfo=UTC)


def test_an_end_that_is_not_after_the_start_is_an_argument_error(capsys, no_model):
    assert "--end must be after --start" in fails(capsys, [*CASE, "--start", "2026-10-31T09:00Z", "--end", "2026-10-31T08:00Z"])


def test_an_end_that_is_not_after_the_start_is_an_argument_error_when_live(capsys, no_model, no_fetch):
    assert "--end must be after --start" in fails(capsys, [*LIVE[:-2], "--end", "2026-10-31T07:00Z"])


def test_a_place_outside_the_uk_is_refused_before_anything_is_fetched_or_built(capsys, no_model, no_fetch):
    line = fails(capsys, ["read", "--live", "--lat", "48.85", "--lon", "2.35", "--start", "2026-10-31T07:00Z", "--end", "2026-10-31T18:00Z"])
    assert "flysafelz read: error:" in line


# --- the reader's settings ----------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("argv", [CASE, ["assess", "--case", BLUE_MILL, "--reader", "live"]])
def test_missing_settings_are_one_line_naming_all_four_variables_and_exit_2(capsys, argv):
    line = fails(capsys, argv)
    for name in reader_config.READER_VARIABLES:
        assert name in line
    assert list(reader_config.READER_VARIABLES) == sorted(reader_config.READER_VARIABLES, key=line.index)  # in the order config.py gives


def test_only_the_missing_settings_are_named(capsys, monkeypatch):
    monkeypatch.setenv(reader_config.PROVIDER, "x")
    monkeypatch.setenv(reader_config.MODEL, "y")
    line = fails(capsys, CASE)
    assert reader_config.BASE_URL in line and reader_config.KEY in line
    assert reader_config.PROVIDER not in line and reader_config.MODEL not in line


def test_the_message_is_not_a_fallback_nothing_is_written_to_the_store(capsys, tmp_path):
    fails(capsys, CASE)
    assert not (tmp_path / "readings.sqlite").exists()


def test_a_secret_never_shows_in_the_message(capsys, monkeypatch):
    monkeypatch.setenv(reader_config.KEY, "sk-secret-value-123")
    assert "sk-secret-value-123" not in fails(capsys, CASE)


# --- the live path of read ---------------------------------------------------------------------------------------------------------


def test_live_read_gets_the_pib_through_fetch_pib_and_reads_the_same_set(monkeypatch, capsys):
    from reader_replies import fixture_readings, model_reply
    from reader_set_helpers import ScriptedChat

    blue = blue_mill_notams()
    xml = (FIXTURES / BLUE_MILL / "pib.xml").read_bytes()
    asked = []

    def fetch(source, now, *, client=None):
        asked.append((source, now))
        status = SourceStatus(source=SourceName.NOTAMS, state=SourceState.OK, fetched_at=now)
        return PibFetch(xml=xml, status=status)

    readings = fixture_readings()
    llm = ScriptedChat(list(blue.values()), **{k.replace("/", "_"): [model_reply(blue[k], readings[k])] for k in ("H7171/26", "J3857/26")})
    monkeypatch.setattr(pib_module, "fetch_pib", fetch)
    monkeypatch.setattr(reader_config, "build_llm", lambda *a, **k: llm)
    assert cli.main(LIVE, now=utc(7, 30)) == 0
    out = capsys.readouterr().out
    assert asked == [("live", utc(7, 30))]
    assert sorted(llm.calls) == ["H7171/26", "J3857/26"]
    assert out.startswith("FlySafeLZ read \u2013 live") and "Site elevation 0 ft (not known without a forecast: taken as 0 ft)" in out
    assert "Model calls: 2" in out


def test_a_pib_that_could_not_be_fetched_is_one_line_exit_1_and_no_model_call(monkeypatch, capsys):
    def fetch(source, now, *, client=None):
        return PibFetch(status=SourceStatus(source=SourceName.NOTAMS, state=SourceState.FAILED, reason="NATS said no", fetched_at=now))

    monkeypatch.setattr(pib_module, "fetch_pib", fetch)
    monkeypatch.setattr(reader_config, "build_llm", lambda *a, **k: object())
    with pytest.raises(SystemExit) as exit_info:
        cli.main(LIVE, now=utc(7, 30))
    out, err = capsys.readouterr()
    assert exit_info.value.code == 1 and out == ""
    assert err == "flysafelz: NOTAMs unavailable: NATS said no\n"


# --- assess --reader ----------------------------------------------------------------------------------------------------------------


def test_an_unknown_reader_value_is_an_argparse_error(capsys, no_model):
    line = fails(capsys, ["assess", "--case", BLUE_MILL, "--reader", "model"])
    assert "invalid choice: 'model'" in line and "fixture" in line and "live" in line


def test_reader_fixture_without_a_case_names_the_missing_case(capsys, no_model, no_fetch):
    assert "--reader fixture needs --case" in fails(capsys, ["assess", "--lat", "53.8", "--lon", "-0.08", "--reader", "fixture"])


def test_a_live_location_run_without_reader_passes_no_readings(monkeypatch):
    seen = {}
    real = cli.assess

    def spy(*args, **kwargs):
        seen.update(readings=kwargs["readings"], reader_status=kwargs["reader_status"])
        return real(*args, **kwargs)

    from factories import airspace_fetch, pib_result, weather_forecast

    notam_ = notam("J1/26", circle=circle_at(0), item_e="UAS OPS. BOOKING PHONE " + ONE + ".")
    monkeypatch.setattr(cli, "fetch_live_sources", lambda lat, lon, window, now: (weather_forecast(utc(6), utc(20)), airspace_fetch(), pib_result(notam_)))
    monkeypatch.setattr(cli, "assess", spy)
    assert cli.main(["assess", "--lat", str(LZ.lat), "--lon", str(LZ.lon), "--start", "2026-10-31T09:00Z", "--end", "2026-10-31T10:00Z"], now=utc(7, 30)) == 0
    assert seen == {"readings": {}, "reader_status": {}}


# --- the GP3 layout, from hand-built results ---------------------------------------------------------------------------------------


def result(notam_id: str, status: ReaderStatus, *, read=None, checks=()) -> ReaderResult:
    return ReaderResult(notam_id=notam_id, reading=read, status=status, failed_checks=list(checks), model=FAKE_MODEL, prompt_version="test", latency_s=0.5, tokens=None)


def lines_for(results: dict[str, ReaderResult], notams, retried=frozenset()) -> list[str]:
    run = cli.ReaderRun(results, set(retried))
    return cli.read_lines(notams, run, LZ, WINDOW, cli.PhoneBook.of(notams))


def test_the_layout_for_each_status_and_the_count_lines():
    near = notam("J1/26", circle=circle_at(0), item_e="UAS OPS. BOOKING PHONE " + ONE + ".")
    far = notam("J2/26", circle=circle_at(1.0))
    gone = notam("J3/26", circle=circle_at(1.0))
    redacted = reading(notam_id="J1/26", geometry=circle_at(0), value="BOOKING <PHONE_1>")
    other = reading(notam_id="J2/26", geometry=circle_at(1.0), contact=ContactKind.NONE, value=None)
    results = {
        "J1/26": result("J1/26", ReaderStatus.READ, read=redacted),
        "J2/26": result("J2/26", ReaderStatus.STORED, read=other),
        "J3/26": result("J3/26", ReaderStatus.FAILED, checks=["your exemption quote is not in the NOTAM text"]),
    }
    out = lines_for(results, [near, far, gone])
    assert out[0] == "NOTAM J1/26  read"
    assert "NOTAM J2/26  stored  (stored reading, no model call)" in out
    assert "NOTAM J3/26  failed" in out
    assert "  failed check: your exemption quote is not in the NOTAM text" in out
    assert out.count("  reading:") == 2  # the failed NOTAM has no reading to show
    assert [line for line in out if line.startswith("  action 07:00\u201308:00: NOTAM J1/26 CONTACT_FIRST")]  # the first hour of the window: the NOTAM is active all day
    assert out[-2:] == ["Model calls: 3", "Retries: 0"]  # read 1, stored 0, failed 2 (a read and its repair)


def test_the_reading_keeps_its_placeholder_and_the_action_line_has_the_real_number():
    near = notam("J1/26", circle=circle_at(0), item_e="UAS OPS. BOOKING PHONE " + ONE + ".")
    read = reading(notam_id="J1/26", geometry=circle_at(0), value="BOOKING <PHONE_1>")
    out = lines_for({"J1/26": result("J1/26", ReaderStatus.READ, read=read)}, [near])
    yaml_lines = [line for line in out if line.startswith(cli.YAML_PAD)]
    assert any("<PHONE_1>" in line for line in yaml_lines) and not any(ONE in line for line in yaml_lines)
    action = next(line for line in out if line.startswith("  action "))
    assert ONE in action and "<PHONE_" not in action and "first and fly only if they agree" in action


def test_an_unavailable_notam_shows_no_reading_and_its_unread_action_with_the_redacted_text_put_back():
    near = notam("J1/26", circle=circle_at(0), item_e="UAS OPS. BOOKING PHONE " + ONE + ".")
    out = lines_for({"J1/26": result("J1/26", ReaderStatus.UNAVAILABLE)}, [near])
    assert out[0] == "NOTAM J1/26  unavailable" and "  reading:" not in out
    action = next(line for line in out if line.startswith("  action "))
    assert "UNREAD_OVER_LZ" in action and ONE in action  # printed with the real number, from this NOTAM's own map


def test_retries_count_only_the_retried_notams_that_were_not_stored():
    a, b, c = (notam(i, circle=circle_at(0)) for i in ("J1/26", "J2/26", "J3/26"))
    results = {
        "J1/26": result("J1/26", ReaderStatus.READ, read=reading(notam_id="J1/26", geometry=circle_at(0))),
        "J2/26": result("J2/26", ReaderStatus.STORED, read=reading(notam_id="J2/26", geometry=circle_at(0))),
        "J3/26": result("J3/26", ReaderStatus.FAILED, checks=["x"]),
    }
    assert lines_for(results, [a, b, c], retried={"J1/26", "J3/26"})[-1] == "Retries: 2"
    assert lines_for(results, [a, b, c], retried={"J2/26"})[-1] == "Retries: 0"  # stored: not sent to the model again


def test_a_stored_result_says_stored_reading():
    out = lines_for({"J1/26": result("J1/26", ReaderStatus.STORED, read=reading(notam_id="J1/26", geometry=circle_at(0)))}, [notam("J1/26", circle=circle_at(0))])
    assert out[0] == "NOTAM J1/26  stored  (stored reading, no model call)"
    assert out[-2:] == ["Model calls: 0", "Retries: 0"]


def test_unread_and_readings_split_the_results():
    read = reading(notam_id="J1/26")
    run = cli.ReaderRun({"J1/26": result("J1/26", ReaderStatus.REPAIRED, read=read), "J2/26": result("J2/26", ReaderStatus.FAILED), "J3/26": result("J3/26", ReaderStatus.UNAVAILABLE)}, set())
    assert run.readings == {"J1/26": read}
    assert run.unread == {"J2/26": "failed", "J3/26": "unavailable"}
    assert run.calls == 5  # repaired 2 (a read and its repair), failed 2, unavailable with no failed check left 1


def test_the_reader_set_is_listed_in_the_header_of_a_real_case_run(monkeypatch, capsys):
    from reader_set_helpers import ScriptedChat

    blue = blue_mill_notams()
    llm = ScriptedChat([blue["H7171/26"], blue["J3857/26"]], H7171_26=[TimeoutError("down")], J3857_26=[TimeoutError("down")])
    monkeypatch.setattr(reader_config, "build_llm", lambda *a, **k: llm)
    assert cli.main(CASE) == 0
    out = capsys.readouterr().out
    assert "Asked the reader about 2 NOTAMs: " in out
    assert "NOTAM H7171/26  unavailable" in out and "NOTAM J3857/26  unavailable" in out
    assert out.rstrip().splitlines()[-2:] == ["Model calls: 2", "Retries: 0"]
