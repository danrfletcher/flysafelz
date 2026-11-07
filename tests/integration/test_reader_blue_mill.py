"""PR-6 (ticket 3): the reader connected to the product, on fixtures/blue_mill_2026-10-31/. GP2-GP4, GP6, the failure paths and F3.

Everything runs through `cli.main([...])` in-process with sockets disabled. The only fake is the chat model, put in by
monkeypatching the reader's model factory (flysafelz.reader.config.build_llm); the readings store is a SQLite file in tmp_path
(tests/conftest.py).
"""

import logging
import re
import sqlite3
import textwrap
from pathlib import Path

import cases
import pytest
import yaml
from factories import utc
from phone_leaks import digit_runs
from reader_replies import area_mismatch, fixture_readings, model_reply
from reader_set_helpers import ScriptedChat, bad_reply
from unit.reader_helpers import blue_mill_notams

from flysafelz import cli
from flysafelz.assess import assess
from flysafelz.contracts import Assessment, NotamAction, Verdict
from flysafelz.reader import config as reader_config
from flysafelz.redact import redact_phones

BLUE_MILL = cases.BLUE_MILL
READ = ["read", "--case", BLUE_MILL]
ASSESS = ["assess", "--case", BLUE_MILL, "--start", "2026-10-31T07:00Z", "--end", "2026-10-31T18:00Z"]
GOLDEN = Path(__file__).resolve().parents[1] / "golden" / "blue_mill_2026-10-31.txt"
H7171, J3857, J3864 = "H7171/26", "J3857/26", "J3864/26"
FIRST_HOUR = "08:00" + cli.DASH + "09:00"


def notams():
    return blue_mill_notams()


def fake_model(monkeypatch: pytest.MonkeyPatch, **scripts) -> ScriptedChat:
    """A scripted chat model in place of the reader's model factory. A NOTAM with no script gets its readings.yaml answer."""
    by_id, readings = notams(), fixture_readings()
    answers = {key: model_reply(by_id[key.replace("_", "/")], readings[key.replace("_", "/")]) for key in ("H7171_26", "J3857_26")}
    llm = ScriptedChat(list(by_id.values()), **{key: scripts.get(key, [answers[key]]) for key in answers})
    monkeypatch.setattr(reader_config, "build_llm", lambda *args, **kwargs: llm)
    return llm


def run(argv: list[str], capsys) -> str:
    assert cli.main(argv) == 0
    out, err = capsys.readouterr()
    assert err == ""
    return out


def block(out: str, notam_id: str) -> list[str]:
    """The lines of one NOTAM's block in a `read` printout."""
    lines = out.splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith(f"NOTAM {notam_id}  "))
    end = next((i for i in range(start + 1, len(lines)) if lines[i] == ""), len(lines))
    return lines[start:end]


def reading_in(lines: list[str]) -> dict:
    """The YAML reading in a block, parsed back."""
    start = lines.index("  reading:") + 1
    yaml_lines = []
    for line in lines[start:]:
        if not line.startswith(cli.YAML_PAD):
            break
        yaml_lines.append(line)
    return yaml.safe_load(textwrap.dedent("\n".join(yaml_lines)))


def action_line(lines: list[str]) -> str:
    return next(line for line in lines if line.startswith("  action "))


# --- GP2, GP3, GP4 ----------------------------------------------------------------------------------------------------------------


def test_gp2_exactly_the_reader_set_is_read_and_at_most_three_calls_run_at_once(monkeypatch, capsys):
    llm = fake_model(monkeypatch)
    out = run(READ, capsys)
    assert sorted(llm.calls) == [H7171, J3857]  # J3864/26 (valid 1 Nov) is dropped by the prefilter: never sent to the model
    assert llm.peak <= 3
    assert J3864 not in out
    assert "Asked the reader about 2 NOTAMs: H7171/26, J3857/26" in out or "Asked the reader about 2 NOTAMs: J3857/26, H7171/26" in out


def test_gp3_each_notam_prints_its_id_status_reading_and_action_then_the_counts(monkeypatch, capsys):
    fake_model(monkeypatch)
    out = run(READ, capsys)
    for notam_id in (H7171, J3857):
        lines = block(out, notam_id)
        assert lines[0] == f"NOTAM {notam_id}  read"
        assert reading_in(lines)["category"] == "cooperative"
        assert action_line(lines).startswith(f"  action {FIRST_HOUR}: NOTAM {notam_id} ")
    assert out.rstrip().splitlines()[-2:] == ["Model calls: 2", "Retries: 0"]


def test_gp4_h7171_has_the_code_built_circle_and_is_avoid_sector_at_the_bearing(monkeypatch, capsys):
    fake_model(monkeypatch)
    lines = block(run(READ, capsys), H7171)
    reading = reading_in(lines)
    geometry = reading["geometry"]
    assert geometry["kind"] == "circle" and geometry["radius_nm"] == 0.5
    assert geometry["lat"] == pytest.approx(53 + 47 / 60 + 32 / 3600, abs=1e-6)  # 534732N
    assert geometry["lon"] == pytest.approx(-(4 / 60 + 59 / 3600), abs=1e-6)  # 0000459W
    assert reading["floor"]["datum"] == "SFC" and reading["ceiling"] == {"value": 700.0, "unit": "ft", "datum": "AGL"}
    assert reading["applies_to"] == "ALL_AIRCRAFT" and reading["exemptions"] == []
    match = re.search(r"NOTAM H7171/26 AVOID_SECTOR bearing ([\d.]+)" + cli.DEGREES, action_line(lines))
    assert match, action_line(lines)
    assert float(match.group(1)) == pytest.approx(168.6, abs=0.5)


def test_gp4_j3857_is_contact_first_with_the_placeholder_in_the_yaml_and_the_real_contact_in_the_action(monkeypatch, capsys):
    fake_model(monkeypatch)
    lines = block(run(READ, capsys), J3857)
    e_text = notams()[J3857].item_e
    contact_in_e = [*redact_phones(e_text)[1].values(), *re.findall(r"[\w.+-]+@[\w-]+\.[\w.]+", e_text)]
    assert contact_in_e  # the NOTAM has a contact in its E text; the test does not say which
    assert reading_in(lines)["contact"]["value"].endswith("<PHONE_1>")
    assert not [number for number in contact_in_e if any(number in line for line in lines if not line.startswith("  action "))]
    action = action_line(lines)
    assert f"NOTAM {J3857} {NotamAction.CONTACT_FIRST.value}" in action and "first and fly only if they agree" in action
    assert any(contact in action for contact in contact_in_e) and "<PHONE_" not in action


def test_a_second_run_reads_nothing_again_and_says_stored_reading(monkeypatch, capsys):
    llm = fake_model(monkeypatch)
    run(READ, capsys)
    assert len(llm.calls) == 2
    out = run(READ, capsys)
    assert len(llm.calls) == 2  # no new model call
    for notam_id in (H7171, J3857):
        assert block(out, notam_id)[0] == f"NOTAM {notam_id}  stored  (stored reading, no model call)"
        assert reading_in(block(out, notam_id))["id"] == notam_id
    assert out.rstrip().splitlines()[-2:] == ["Model calls: 0", "Retries: 0"]


def test_a_failed_notam_is_retried_on_the_next_run_and_counted(monkeypatch, capsys):
    llm = fake_model(monkeypatch, J3857_26=[bad_reply(), bad_reply(), model_reply(notams()[J3857], fixture_readings()[J3857])])
    out = run(READ, capsys)
    lines = block(out, J3857)
    assert lines[0] == f"NOTAM {J3857}  failed"
    assert [line for line in lines if line.startswith("  failed check: ")]
    assert "reading:" not in " ".join(lines)
    assert f"NOTAM {J3857} UNREAD_OVER_LZ" in action_line(lines)
    assert out.rstrip().splitlines()[-2:] == ["Model calls: 3", "Retries: 0"]  # H7171 one call; J3857 a read and its repair
    out = run(READ, capsys)
    assert block(out, J3857)[0] == f"NOTAM {J3857}  read"
    assert block(out, H7171)[0].startswith(f"NOTAM {H7171}  stored")
    assert out.rstrip().splitlines()[-2:] == ["Model calls: 1", "Retries: 1"]
    assert llm.calls.count(J3857) == 3


# --- the failure paths through assess ---------------------------------------------------------------------------------------------


def unread_lines(out: str) -> list[str]:
    return [line for line in out.splitlines() if "has not been read" in line]


def chain_assessment() -> Assessment:
    """assess() on the chain's own readings and reader_status, as the CLI calls it, but returned so its text can be checked."""
    case, forecast, airspace, pib, _readings = cases.load(BLUE_MILL)
    run_ = cli.run_reader(reader_config.build_llm(), pib.notams, case.lat, case.lon, (utc(7), utc(18)), case.now)
    return assess(case.lat, case.lon, now=case.now, forecast=forecast, airspace=airspace, pib=pib, readings=run_.readings, reader_status=run_.unread, window=(utc(7), utc(18)))


def segments_from(assessment: Assessment, hour: int = 8):
    return [s for h in assessment.hours for s in h.segments if s.start >= utc(hour)]


def test_the_reader_unavailable_for_j3857_still_finishes_and_the_segments_are_unverified_with_the_assumption(monkeypatch, capsys):
    fake_model(monkeypatch, J3857_26=[TimeoutError("timed out")])
    out = run([*ASSESS, "--reader", "live"], capsys)
    assert "NOTAM reader unavailable for J3857/26: treated as unread" in out
    raw_e = " ".join(notams()[J3857].item_e.split())
    printed = unread_lines(out)
    assert printed and all(raw_e in line for line in printed)  # printed: the E text with its real number put back
    assert not [line for line in out.splitlines() if "CONTACT_FIRST" in line and J3857 in line]
    fake_model(monkeypatch, J3857_26=[TimeoutError("timed out")])
    assessment = chain_assessment()  # the Assessment itself holds the redacted E text
    redacted_e = " ".join(redact_phones(notams()[J3857].item_e)[0].split())
    assert "NOTAM reader unavailable for J3857/26: treated as unread" in assessment.assumptions
    egd = [s for s in segments_from(assessment) if any("EGD496A" in c for c in s.conditions)]
    assert egd
    for segment in egd:
        assert {a.action for a in segment.notam_actions if a.notam_id == J3857} == {NotamAction.UNREAD_OVER_LZ}
        assert segment.verdict in (Verdict.UNVERIFIED, Verdict.NO_GO)  # NO_GO only when the weather or the night blocks it
        if segment.verdict is Verdict.UNVERIFIED:
            assert any(redacted_e in c for c in segment.conditions)


def test_a_notam_that_fails_its_checks_after_one_repair_is_unread_with_the_danger_area_condition_and_no_assumption_line(monkeypatch, capsys):
    fake_model(monkeypatch, J3857_26=[bad_reply(), bad_reply()])
    out = run([*ASSESS, "--reader", "live"], capsys)
    assert "NOTAM reader unavailable" not in out  # failed adds no assumption line
    assert unread_lines(out)
    fake_model(monkeypatch, J3857_26=[bad_reply(), bad_reply()])
    assessment = chain_assessment()
    assert not [a for a in assessment.assumptions if "reader" in a]
    egd = [s for s in segments_from(assessment) if any("EGD496A" in c for c in s.conditions)]
    assert egd and all({a.action for a in s.notam_actions if a.notam_id == J3857} == {NotamAction.UNREAD_OVER_LZ} for s in egd)
    fake_model(monkeypatch, J3857_26=[bad_reply(), bad_reply()])
    shown = run(READ, capsys)
    assert block(shown, J3857)[0] == f"NOTAM {J3857}  failed"
    assert [line for line in block(shown, J3857) if line.startswith("  failed check: ")]


# --- GP6: the whole chain, offline -------------------------------------------------------------------------------------------------


def test_gp6_each_readings_yaml_area_is_what_the_area_builder_makes_from_the_e_text():
    by_id, readings = notams(), fixture_readings()
    problems = [area_mismatch(by_id[notam_id], reading) for notam_id, reading in readings.items()]
    assert [p for p in problems if p] == []


def test_gp6_the_chain_gives_the_same_hour_table_and_notam_lines_as_the_golden_file(monkeypatch, capsys):
    fake_model(monkeypatch)
    out = run([*ASSESS, "--reader", "live"], capsys)
    assert out == GOLDEN.read_text(encoding="utf-8")
    fixture_out = run(ASSESS, capsys)  # the default --reader fixture
    assert fixture_out == out


def test_the_fixture_reader_is_the_default_for_a_case_and_matches_the_golden_file_byte_for_byte(capsys):
    assert run(ASSESS, capsys) == GOLDEN.read_text(encoding="utf-8")
    assert run([*ASSESS, "--reader", "fixture"], capsys) == GOLDEN.read_text(encoding="utf-8")


def comparable(hour) -> str:
    """The hour as JSON with each bearing rounded to a thousandth of a degree.

    readings.yaml writes the Garton circle's centre to eight decimals and the area builder to full precision, so the two bearings differ
    by about 2e-5 degrees. The file is left as it is: ticket 2's JSON golden holds the bearing it gives.
    """
    return re.sub(r'"bearing_deg":(\d+\.\d+)', lambda m: f'"bearing_deg":{float(m.group(1)):.3f}', hour.model_dump_json())


def test_gp6_the_assessment_from_the_chain_equals_the_one_built_straight_from_readings_yaml(monkeypatch):
    fake_model(monkeypatch)
    case, forecast, airspace, pib, readings = cases.load(BLUE_MILL)
    run_ = cli.run_reader(reader_config.build_llm(), pib.notams, case.lat, case.lon, (utc(7), utc(18)), case.now)
    from_chain = assess(case.lat, case.lon, now=case.now, forecast=forecast, airspace=airspace, pib=pib, readings=run_.readings, reader_status=run_.unread, window=(utc(7), utc(18)))
    straight = assess(case.lat, case.lon, now=case.now, forecast=forecast, airspace=airspace, pib=pib, readings=cli.PhoneBook.of(pib.notams).redact_contacts(readings), window=(utc(7), utc(18)))
    assert len(from_chain.hours) == len(straight.hours)
    for chain_hour, straight_hour in zip(from_chain.hours, straight.hours, strict=True):
        assert comparable(chain_hour) == comparable(straight_hour)
    assert from_chain.flyable_windows == straight.flyable_windows and from_chain.overall_verdict == straight.overall_verdict
    segments = [s for h in from_chain.hours for s in h.segments if s.start >= utc(8)]
    assert segments and {a.action for s in segments for a in s.notam_actions} >= {NotamAction.CONTACT_FIRST, NotamAction.AVOID_SECTOR}
    assert Verdict.UNVERIFIED not in {s.verdict for s in segments}


# --- F3: no real number anywhere but captured stdout -------------------------------------------------------------------------------


def stored_rows(path: Path) -> str:
    with sqlite3.connect(path) as connection:
        return "\n".join(connection.iterdump())


def test_f3_no_real_number_in_the_log_the_stored_rows_or_the_assessment_json_after_the_cli_runs(monkeypatch, capsys, caplog, tmp_path):
    caplog.set_level(logging.DEBUG)
    fake_model(monkeypatch, J3857_26=[TimeoutError("timed out waiting for 01632 960001")])
    run([*ASSESS, "--reader", "live"], capsys)
    fake_model(monkeypatch)
    out = run(READ, capsys)
    out += run([*ASSESS, "--reader", "live", "--debug"], capsys)
    runs = {digits for notam in notams().values() for digits in digit_runs(notam.item_e)}
    assert runs  # Blue Mill's E texts carry numbers, so the scan below has something to find
    store = reader_config.reading_store()
    assert store.exists()
    case, forecast, airspace, pib, _readings = cases.load(BLUE_MILL)
    llm = reader_config.build_llm()
    run_ = cli.run_reader(llm, pib.notams, case.lat, case.lon, (utc(7), utc(18)), case.now)
    assessment: Assessment = assess(case.lat, case.lon, now=case.now, forecast=forecast, airspace=airspace, pib=pib, readings=run_.readings, reader_status=run_.unread, window=(utc(7), utc(18)))
    places = {
        "log records": "\n".join(record.getMessage() for record in caplog.records),
        "stored rows": stored_rows(store),
        "assessment json": assessment.model_dump_json(),
    }
    for where, text in places.items():
        squeezed = re.sub(r"\s+", "", text)
        assert not [digits for digits in runs if digits in squeezed], where
    assert any(digits in re.sub(r"\s+", "", out) for digits in runs)  # the real number is in captured stdout, where it belongs
