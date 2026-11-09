"""The v0 prompt, assembled by one pure function (5a PR-9.F1, G14): every input block, rules from RULES, the action table from NotamAction, abbreviations, now and the window."""

import inspect
from datetime import UTC, datetime

import pytest

from evals.agent import v0_prompt
from evals.agent.loader import load_examples
from evals.agent.v0_prompt import NO_NOTAMS, UNAVAILABLE, CaseFiles, action_table, build_v0_messages, limit_rules
from flysafelz.contracts import NotamAction, SourceState, Verdict
from flysafelz.reader.prompt import ABBREVIATIONS
from flysafelz.rules import RULES
from flysafelz.sources import load_case_sources


@pytest.fixture(scope="module")
def records():
    return {r["id"]: r for r in load_examples("dev") if r["reference"]["path"] == "assess"}


def case_files(record, **changes):
    sources = load_case_sources(record["fixtures"]["case"], with_readings=False)
    parts = {"forecast": sources.forecast, "airspace": sources.airspace, "pib": sources.pib, **changes}
    return CaseFiles(parts["forecast"], parts["airspace"], parts["pib"])


@pytest.fixture(scope="module")
def dev_01(records):
    record = next(r for i, r in records.items() if i.startswith("dev_01"))
    files = case_files(record)
    return record, files, build_v0_messages(record, files)


# --- every input block is there ------------------------------------------------------------------------------------------------------


def test_the_user_message_holds_now_the_window_the_lz_and_each_source_block(dev_01):
    record, _files, prompt = dev_01
    assert "Now: 2026-09-29T06:30:00Z" in prompt.user
    assert "Window to assess: 2026-09-29T09:00:00Z to 2026-09-29T11:00:00Z" in prompt.user
    assert f"latitude {record['inputs']['lat']}, longitude {record['inputs']['lon']}" in prompt.user
    for heading in ("WEATHER: hourly rows", "AIRSPACE: permanent airspace volumes", "NOTAMS: those that overlap the window"):
        assert heading in prompt.user


def test_the_weather_rows_cover_the_window_hour_by_hour_with_every_value(dev_01):
    _record, _files, prompt = dev_01
    rows = [line for line in prompt.user.splitlines() if line.startswith("2026-09-29 ") and "wind_10m" in line]
    assert [line[:24] for line in rows] == ["2026-09-29 09:00Z-10:00Z", "2026-09-29 10:00Z-11:00Z"]
    for name in ("wind_10m", "wind_80m", "wind_120m", "gust_10m", "precipitation", "cape", "visibility", "shortwave_radiation", "gust_spread", "gradient"):
        assert all(f"{name} " in line for line in rows)


def test_the_airspace_block_lists_the_volumes_near_the_lz_as_published_with_nothing_worked_out(dev_01):
    _record, files, prompt = dev_01
    lines = [line for line in prompt.user.splitlines() if line.startswith("- ")]
    assert any(line.startswith("- ALDBROUGH ") and "floor SFC" in line and "ceiling 3200 ft AMSL" in line for line in lines)
    assert len(lines) == len([a for a in files.airspace.areas if a.name in prompt.user])  # one line per volume shown
    assert "bearing" not in prompt.user.lower() and "over the LZ" not in prompt.user


def test_the_notams_are_those_the_prefilter_keeps_with_raw_items_and_the_q_line(dev_01):
    _record, _files, prompt = dev_01
    assert prompt.notam_ids and prompt.notam_ids[0] == "U6768/26"
    assert f"Count: {len(prompt.notam_ids)}" in prompt.user
    block = prompt.user.split("<notam_text>")[1].split("</notam_text>")[0]
    for notam_id in prompt.notam_ids:
        assert f"NOTAM {notam_id}\nQ-line: " in block
    for item in ("B: ", "C: ", "D: ", "E: ", "F: ", "G: "):
        assert block.count(f"\n{item}") >= len(prompt.notam_ids)


def test_the_message_carries_both_texts_as_two_messages_system_then_human(dev_01):
    _record, _files, prompt = dev_01
    system, human = prompt.messages()
    assert (system.type, human.type) == ("system", "human")
    assert system.content == prompt.system and human.content == prompt.user


# --- rules, the action table, abbreviations ------------------------------------------------------------------------------------------


def test_every_limit_value_in_the_prompt_comes_from_rules_not_a_literal(dev_01):
    _record, _files, prompt = dev_01
    shown = limit_rules()
    assert shown and {rule.id for rule in shown} >= {"wind_10m_no_go", "gust_spread_no_go", "cape_no_go", "visibility_no_go", "ceiling_left_no_go", "civil_twilight_depression"}
    for rule in shown:
        value = ", ".join(rule.value) if isinstance(rule.value, list) else f"{rule.value:g}"
        assert f"- {rule.id}: {value} {rule.unit}" in prompt.system
    assert RULES.wind_10m_no_go.note in prompt.system


def test_a_changed_rule_changes_the_prompt(dev_01):
    record, files, base = dev_01
    stricter = RULES.model_copy(update={"wind_10m_no_go": RULES.wind_10m_no_go.model_copy(update={"value": 9.5})})
    changed = build_v0_messages(record, files, rules=stricter)
    assert "- wind_10m_no_go: 9.5 mph" in changed.system and "- wind_10m_no_go: 9.5 mph" not in base.system


def test_the_settings_of_the_build_and_the_reader_are_not_shown_as_limits():
    ids = {rule.id for rule in limit_rules()}
    assert not {i for i in ids if i.startswith(v0_prompt.EXCLUDED_RULE_PREFIXES)}


def test_the_action_table_has_one_line_per_notam_action_taken_from_the_enum_docstring(dev_01):
    _record, _files, prompt = dev_01
    lines = action_table()
    assert [line.split(":")[0] for line in lines] == [action.name for action in NotamAction]
    assert all(line in prompt.system for line in lines)
    assert "AVOID_SECTOR: within the local area but not over the LZ" in prompt.system


def test_an_enum_member_with_no_docstring_row_is_a_loud_error_never_a_gap():
    from enum import StrEnum

    class Undescribed(StrEnum):
        """Things.

        FIRST     1  has a row
        """

        FIRST = "FIRST"
        SECOND = "SECOND"

    with pytest.raises(ValueError, match="Undescribed.SECOND has no description"):
        action_table(Undescribed)


def test_abbreviations_are_limited_to_the_tokens_the_kept_notams_hold(dev_01, records):
    _record, files, prompt = dev_01
    tail = prompt.user.split("Abbreviations used in these NOTAMs:")[1].strip().splitlines()
    shown = {line.split(" = ")[0] for line in tail}
    e_texts = " ".join(n.item_e for n in files.pib.notams if n.id in prompt.notam_ids).upper()
    assert shown and shown <= set(ABBREVIATIONS) and len(shown) < len(ABBREVIATIONS)
    for abbreviation in shown:
        assert abbreviation in e_texts
    for line in tail:
        assert line.split(" = ", 1)[1] == ABBREVIATIONS[line.split(" = ")[0]]


# --- the empty and unavailable cases -----------------------------------------------------------------------------------------------


def test_an_empty_prefiltered_notam_list_says_no_notams_in_the_local_area_and_has_no_tag(dev_01):
    record, files, _base = dev_01
    empty = files.pib.model_copy(update={"notams": []})
    prompt = build_v0_messages(record, case_files(record, pib=empty))
    assert NO_NOTAMS == "no NOTAMs in the local area" and NO_NOTAMS in prompt.user
    assert "<notam_text" not in prompt.user and "</notam_text" not in prompt.user and prompt.notam_ids == ()
    assert "Abbreviations used" not in prompt.user


@pytest.mark.parametrize(("source", "heading"), [("airspace", "AIRSPACE"), ("pib", "NOTAMS"), ("forecast", "WEATHER")])
def test_a_failed_source_is_a_source_unavailable_block_in_place_of_its_data(dev_01, source, heading):
    record, files, base = dev_01
    status = getattr(files, source).status.model_copy(update={"state": SourceState.FAILED, "reason": "dropped for this example"})
    dropped = build_v0_messages(record, case_files(record, **{source: getattr(files, source).model_copy(update={"status": status})}))
    assert f"{heading}\n{UNAVAILABLE}: " in dropped.user and "do not read this as clear" in dropped.user
    assert UNAVAILABLE not in base.user
    other = {"airspace", "pib", "forecast"} - {source}
    assert dropped.user.count(UNAVAILABLE) == 1, other
    if source == "pib":
        assert dropped.notam_ids == () and "<notam_text" not in dropped.user and NO_NOTAMS not in dropped.user  # unavailable is not "none"


def test_the_system_text_says_unavailable_is_unverified_and_never_clear(dev_01):
    _record, _files, prompt = dev_01
    assert UNAVAILABLE in prompt.system and Verdict.UNVERIFIED.value in prompt.system and "Never read a missing source as clear" in prompt.system


# --- purity ------------------------------------------------------------------------------------------------------------------------


def test_one_prompt_per_example_the_same_input_gives_the_same_strings(records):
    for record in list(records.values())[:4]:
        files = case_files(record)
        first, second = build_v0_messages(record, files), build_v0_messages(record, files)
        assert first == second and first.case == record["fixtures"]["case"]


def test_the_prompt_changes_with_the_window_and_with_now(dev_01):
    record, files, base = dev_01
    later = {**record, "reference": {**record["reference"], "window": ["2026-09-29T10:00Z", "2026-09-29T11:00Z"]}}
    assert build_v0_messages(later, files).user != base.user
    other_now = {**record, "inputs": {**record["inputs"], "now": "2026-09-29T07:30:00Z"}}
    assert "Now: 2026-09-29T07:30:00Z" in build_v0_messages(other_now, files).user


def test_the_function_reads_no_file_no_clock_and_no_network():
    source = inspect.getsource(v0_prompt)
    for word in ("open(", "read_text", "datetime.now", "time.time", "requests", "httpx", "os.environ", "socket"):
        assert word not in source


def test_an_example_that_is_not_an_assess_example_with_a_window_is_refused(records, dev_01):
    record, files, _base = dev_01
    with pytest.raises(ValueError, match="assess examples only"):
        build_v0_messages({**record, "reference": {**record["reference"], "path": "explain"}}, files)
    with pytest.raises(ValueError, match="reference.window"):
        build_v0_messages({**record, "reference": {k: v for k, v in record["reference"].items() if k != "window"}}, files)


def test_the_question_is_not_sent(dev_01):
    record, _files, prompt = dev_01
    assert record["inputs"]["question"] not in prompt.user and record["inputs"]["question"] not in prompt.system


def test_the_window_in_the_prompt_is_the_utc_instant_not_a_local_time(dev_01):
    _record, _files, prompt = dev_01
    assert datetime(2026, 9, 29, 9, tzinfo=UTC).strftime("%Y-%m-%dT%H:%M:%SZ") in prompt.user
