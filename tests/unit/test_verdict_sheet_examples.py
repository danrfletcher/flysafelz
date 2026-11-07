"""E4 / E5 / E16 / G8 / G9 and the examples file: read_examples() and build_blind_entry() on the committed Blue Mill case.

Every bad file or example gives one line that names the example. Nothing is written, no network, no key.
"""

from datetime import UTC, datetime
from pathlib import Path

import cases
import pytest
import yaml
from verdict_sheet_helpers import examples_file

from flysafelz.assess import assess
from flysafelz.contracts import SourceName, SourceState, Verdict
from flysafelz.verdict_sheet import DROPPED_REASON, Example, SheetError, apply_drops, build_blind_entry, read_examples

GP1 = Path(__file__).resolve().parents[1] / "data" / "verdict_sheet" / "examples_blue_mill.yaml"
NOW = datetime(2026, 10, 31, 7, 30, tzinfo=UTC)
WINDOW = (datetime(2026, 10, 31, 7, tzinfo=UTC), datetime(2026, 10, 31, 18, tzinfo=UTC))


def raw(example_id="e1", **changes):
    """A Blue Mill assess example as the mapping Dan writes (times as text)."""
    base = {
        "id": example_id,
        "inputs": {"lat": 53.802797, "lon": -0.086644, "now": "2026-10-31T07:30:00Z", "question": "Can I fly Blue Mill today?"},
        "fixtures": {"case": cases.BLUE_MILL},
        "reference": {"path": "assess", "window": ["2026-10-31T07:00:00Z", "2026-10-31T18:00:00Z"]},
        "metadata": {"blind": True},
    }
    base.update(changes)
    return base


def read(tmp_path, *examples):
    return read_examples(examples_file(tmp_path / "examples.yaml", *examples))


def example(**changes) -> Example:
    fields = {"id": "e1", "path": "assess", "blind": True, "lat": 53.802797, "lon": -0.086644, "now": NOW, "window": WINDOW, "case": cases.BLUE_MILL}
    fields.update(changes)
    return Example(**fields)


# --- the committed GP1 file and the shapes read_examples accepts ---------------------------------


def test_the_gp1_file_holds_exactly_the_blue_mill_example():
    assert yaml.safe_load(GP1.read_text(encoding="utf-8")) == [
        {
            "id": "blue_mill_2026-10-31",
            "inputs": {"lat": 53.802797, "lon": -0.086644, "now": NOW, "question": "Can I fly Blue Mill today?"},
            "fixtures": {"case": "blue_mill_2026-10-31"},
            "reference": {"path": "assess", "window": [WINDOW[0], WINDOW[1]]},
            "metadata": {"blind": True},
        }
    ]
    assert read_examples(GP1) == [Example(id="blue_mill_2026-10-31", path="assess", blind=True, lat=53.802797, lon=-0.086644, now=NOW, window=WINDOW, case=cases.BLUE_MILL)]


def test_a_mapping_with_an_examples_list_is_read_too(tmp_path):
    path = tmp_path / "examples.yaml"
    path.write_text(yaml.safe_dump({"examples": [raw()]}), encoding="utf-8")
    assert [e.id for e in read_examples(path)] == ["e1"]


def test_g9_a_null_window_and_a_missing_window_both_read_as_no_window(tmp_path):
    no_window = raw("e2")
    del no_window["reference"]["window"]
    first, second = read(tmp_path, raw("e1", reference={"path": "assess", "window": None}), no_window)
    assert first.window is None and second.window is None


def test_a_window_may_be_a_start_end_mapping_and_a_non_utc_time_is_converted(tmp_path):
    changed = raw(reference={"path": "assess", "window": {"start": "2026-10-31T08:00:00+01:00", "end": "2026-10-31T18:00:00Z"}})
    assert read(tmp_path, changed)[0].window == (datetime(2026, 10, 31, 7, tzinfo=UTC), WINDOW[1])


def test_e5_an_explain_example_is_read_with_its_path_and_none_of_its_other_fields_are_checked(tmp_path):
    (explain,) = read(tmp_path, {"id": "why", "reference": {"path": "explain"}, "inputs": "anything"})
    assert explain.id == "why" and explain.path == "explain"


# --- one line naming the example, for everything wrong ---------------------------------------------------


def error_of(tmp_path, *examples) -> str:
    with pytest.raises(SheetError) as excinfo:
        read(tmp_path, *examples)
    return str(excinfo.value)


def test_e4_two_examples_with_the_same_id_name_the_id(tmp_path):
    message = error_of(tmp_path, raw("same"), raw("other"), raw("same"))
    assert "same" in message and "duplicate" in message and "\n" not in message


def test_a_missing_or_blank_id_names_the_position(tmp_path):
    no_id = raw()
    del no_id["id"]
    assert error_of(tmp_path, raw("a"), no_id) == "example #2: missing id"
    assert error_of(tmp_path, raw(" ")) == "example #1: missing id"


def test_an_unknown_fixtures_drop_value_names_the_example_and_the_value(tmp_path):
    message = error_of(tmp_path, raw("dropper", fixtures={"case": cases.BLUE_MILL, "drop": ["weather", "notams"]}))
    assert "dropper" in message and "'notams'" in message and "weather, airspace, pib" in message


def test_a_drop_that_is_not_a_list_is_refused(tmp_path):
    assert "fixtures.drop must be a list" in error_of(tmp_path, raw("d", fixtures={"case": cases.BLUE_MILL, "drop": "weather"}))


@pytest.mark.parametrize("case", ["../fixtures/x", "a/b", "", ".hidden", None, 7])
def test_a_case_is_a_folder_name_never_a_path(tmp_path, case):
    assert "fixtures.case" in error_of(tmp_path, raw("c", fixtures={"case": case}))


@pytest.mark.parametrize(
    ("field", "value", "word"),
    [("lat", "north", "inputs.lat"), ("lon", True, "inputs.lon"), ("now", "2026-10-31T07:30:00", "inputs.now"), ("now", None, "inputs.now")],
)
def test_a_bad_input_names_the_example_and_the_field(tmp_path, field, value, word):
    inputs = {**raw()["inputs"], field: value}
    message = error_of(tmp_path, raw("bad", inputs=inputs))
    assert message.startswith("example bad:") and word in message


@pytest.mark.parametrize("window", ["morning", [1], ["2026-10-31T07:00:00Z", "2026-10-31T18:00:00"], {"start": "x"}])
def test_a_bad_window_names_the_example(tmp_path, window):
    message = error_of(tmp_path, raw("w", reference={"path": "assess", "window": window}))
    assert message.startswith("example w:") and "window" in message


def test_an_unreadable_file_a_bad_yaml_and_a_wrong_shape_are_one_line_each(tmp_path):
    missing = tmp_path / "nope.yaml"
    with pytest.raises(SheetError, match="cannot be read"):
        read_examples(missing)
    broken = tmp_path / "broken.yaml"
    broken.write_text("- id: [unclosed\n", encoding="utf-8")
    with pytest.raises(SheetError, match="not valid YAML"):
        read_examples(broken)
    for text in ("just text\n", "42\n", "- 7\n"):
        wrong = tmp_path / "wrong.yaml"
        wrong.write_text(text, encoding="utf-8")
        with pytest.raises(SheetError, match="must be a"):
            read_examples(wrong)
    undecodable = tmp_path / "bytes.yaml"
    undecodable.write_bytes(b"\xff\xfe\x00bad")
    with pytest.raises(SheetError, match="cannot be read"):
        read_examples(undecodable)


def test_reading_the_file_leaves_its_bytes_and_mtime_alone(tmp_path):
    path = examples_file(tmp_path / "examples.yaml", raw())
    before = (path.read_bytes(), path.stat().st_mtime_ns)
    read_examples(path)
    assert (path.read_bytes(), path.stat().st_mtime_ns) == before


# --- E16: the example must match its case ----------------------------------------------------------------


@pytest.mark.parametrize(
    ("changes", "field"),
    [
        ({"now": datetime(2026, 10, 31, 8, 30, tzinfo=UTC)}, "inputs.now"),
        ({"lat": 53.9}, "inputs.lat"),
        ({"lon": -0.1}, "inputs.lon"),
    ],
)
def test_e16_an_example_that_differs_from_its_case_names_the_example_and_the_field(changes, field):
    with pytest.raises(SheetError) as excinfo:
        build_blind_entry(example(id="moved", **changes), cases.case_dir(cases.BLUE_MILL))
    message = str(excinfo.value)
    assert message.startswith("example moved:") and field in message and cases.BLUE_MILL in message


def test_an_example_whose_case_folder_is_unreadable_is_one_line_naming_the_example(tmp_path):
    with pytest.raises(SheetError, match=r"example lost: .*case.yaml"):
        build_blind_entry(example(id="lost"), tmp_path / "no_such_case")


def test_a_reversed_window_is_one_line_naming_the_example():
    with pytest.raises(SheetError, match="example back: .*must start before it ends"):
        build_blind_entry(example(id="back", window=(WINDOW[1], WINDOW[0])), cases.case_dir(cases.BLUE_MILL))


# --- G8: dropped sources ---------------------------------------------------------------------------------


def test_g8_a_dropped_source_is_a_failed_status_with_the_reason_dropped_for_this_example():
    case, forecast, airspace, pib, _readings = cases.load(cases.BLUE_MILL)
    f, a, p = apply_drops(("weather", "airspace", "pib"), case.now, forecast, airspace, pib)
    for status, source in ((f.status, SourceName.WEATHER), (a.status, SourceName.AIRSPACE), (p.status, SourceName.NOTAMS)):
        assert (status.source, status.state, status.reason) == (source, SourceState.FAILED, DROPPED_REASON)
    assert f.samples == {} and a.areas == [] and p.notams == []
    same = apply_drops((), case.now, forecast, airspace, pib)
    assert same[0] is forecast and same[1] is airspace and same[2] is pib


@pytest.mark.parametrize("source", ["weather", "airspace", "pib"])
def test_g8_assess_marks_the_hours_unverified_and_the_entry_still_carries_no_verdicts(source):
    case, forecast, airspace, pib, readings = cases.load(cases.BLUE_MILL)
    f, a, p = apply_drops((source,), case.now, forecast, airspace, pib)
    assessment = assess(case.lat, case.lon, now=case.now, forecast=f, airspace=a, pib=p, readings=readings, window=WINDOW)
    segments = [s for h in assessment.hours for s in h.segments]
    assert any(s.verdict is Verdict.UNVERIFIED for s in segments)  # the dropped source is why

    entry = build_blind_entry(example(drop=(source,)), cases.case_dir(cases.BLUE_MILL))
    rows = entry["rows"]
    assert [(r["start"], r["end"]) for r in rows] == [(s.start.strftime("%Y-%m-%dT%H:%M:%SZ"), s.end.strftime("%Y-%m-%dT%H:%M:%SZ")) for s in segments]
    assert all(r["verdict"] is None and r["approaching_limits"] is None for r in rows)
    assert entry["actions_done"] is False and entry["mode"] == "blind"
    assert (entry["notam_actions"] == {}) == (source == "pib")


def test_g8_a_dropped_pib_has_no_notam_cuts_so_the_08_00_row_is_just_an_hour():
    entry = build_blind_entry(example(drop=("pib",)), cases.case_dir(cases.BLUE_MILL))
    assert {r["start"][11:16]: r["edge"] for r in entry["rows"]}["08:00"] == "hour"
