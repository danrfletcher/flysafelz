"""GP7: the live Blue Mill check. The real reader (read_for_lz) and the real notam_action(), on the frozen case PIB, with a fake chat model.

`run_check` is the helper the container verification uses: it runs the check and returns the section whose `lines` are the report's.
Run `pytest tests/unit/test_reader_eval_blue_mill.py -s` to see the lines.
"""

import pytest
from cache_helpers import fixture_reply
from phone_leaks import digit_runs, leak_list, leaked
from reader_eval_run_helpers import chat

from evals.reader_eval.blue_mill import AGREE, DIFFER, blue_mill_section, segment_on
from evals.reader_eval.run import BLUE_MILL_CASE, run_blue_mill
from flysafelz.reader.contract import ReaderStatus
from flysafelz.reader.reader_set import read_for_lz
from flysafelz.sources import load_case_sources

H, J = "H7171/26", "J3857/26"
NO_CONTACT = {"kind": "none", "value": None}


@pytest.fixture(scope="module")
def loaded():
    return load_case_sources(BLUE_MILL_CASE)


def run_check(loaded, **scripts):
    """The check, with a fake model scripted per NOTAM (keys like H7171_26)."""
    section = run_blue_mill(chat(**scripts), loaded)
    print("\n".join(section["lines"]))
    return section


def entry(section, notam_id):
    return next(e for e in section["items"] if e["id"] == notam_id)


# --- both agree ------------------------------------------------------------------------------------------------------------------------


def test_a_model_that_returns_the_reverse_mapped_fixture_fields_agrees_on_both_notams(loaded):
    section = run_check(loaded)
    assert section["lines"] == [f"{H}: AVOID_SECTOR live vs AVOID_SECTOR fixture {AGREE}", f"{J}: CONTACT_FIRST live vs CONTACT_FIRST fixture {AGREE}"]
    assert section["ok"] is True and section["case"] == BLUE_MILL_CASE
    assert section["heading"].startswith(f"{BLUE_MILL_CASE}: the live reader against the case's readings.yaml") and "08:00-09:00Z on 2026-10-31" in section["heading"]


def test_the_check_reads_the_two_notams_with_the_cache_bypassed_once_each(loaded):
    model = chat()
    run_blue_mill(model, loaded)
    assert sorted(model.calls) == [H, J]


def test_the_check_is_judged_at_the_08_00_segment_of_the_case_day(loaded):
    segment = segment_on(loaded.case.window.start)
    assert (segment.start.hour, segment.start.minute, segment.end.hour) == (8, 0, 9)
    assert segment.start.date() == loaded.case.window.start.date()


def test_the_live_action_is_the_one_the_real_notam_action_gives_for_the_real_reader_result(loaded):
    """The same two functions the assessment uses, called here directly: the section's live action is theirs."""
    from flysafelz.clients.openaip import metres_to_feet
    from flysafelz.notam_action import LandingZone, notam_action

    case = loaded.case
    results = read_for_lz(loaded.pib.notams, (case.lat, case.lon), (case.window.start, case.window.end), chat(), None, now=case.now)
    lz = LandingZone(lat=case.lat, lon=case.lon, elevation_ft=metres_to_feet(loaded.forecast.elevation_m))
    parsed = {n.id: n for n in loaded.pib.notams}
    expected = notam_action(parsed[H], results[H].reading, lz, segment_on(case.window.start)).action.value
    assert entry(run_blue_mill(chat(), loaded), H)["live_action"] == expected == "AVOID_SECTOR"


# --- J3857/26 is a contact present, never a fixed number ---------------------------------------------------------------------------


@pytest.mark.parametrize("contact", [{"kind": "phone", "value": "<PHONE_1>"}, {"kind": "phone", "value": "Someone <PHONE_1>"}, {"kind": "email", "value": "ops@example.invalid"}])
def test_j3857_agrees_with_any_phone_or_email_contact_not_a_particular_number(loaded, contact):
    section = run_check(loaded, J3857_26=[{**fixture_reply(J), "contact": contact}])
    assert entry(section, J)["agree"] is True and entry(section, J)["live_action"] == "CONTACT_FIRST"


def test_j3857_with_no_contact_is_a_difference_naming_both_actions_and_the_reason(loaded):
    section = run_check(loaded, J3857_26=[{**fixture_reply(J), "contact": NO_CONTACT}])
    line = section["lines"][1]
    assert line.startswith(f"{J}: NO_FLY live vs CONTACT_FIRST fixture {DIFFER}")
    assert "no phone or email contact" in line and "contact phone" in line
    assert section["ok"] is False and entry(section, J)["agree"] is False and entry(section, H)["agree"] is True


# --- a fake that drops H7171's area-relevant facts -------------------------------------------------------------------------------------


def test_h7171_read_as_drones_only_is_a_difference_naming_both_actions_and_the_reasons(loaded):
    section = run_check(loaded, H7171_26=[{**fixture_reply(H), "applies_to": "UAS_ONLY"}])
    line = section["lines"][0]
    live = entry(section, H)["live_action"]
    assert live == "IGNORE" and line.startswith(f"{H}: IGNORE live vs AVOID_SECTOR fixture {DIFFER}")
    assert "(live: applies to drones only; fixture: " in line
    assert section["ok"] is False


# --- unavailable ----------------------------------------------------------------------------------------------------------------------


def test_an_endpoint_that_is_down_shows_unavailable_for_both_and_compares_nothing(loaded):
    down = [ConnectionError("the endpoint refused the connection")]
    section = run_check(loaded, H7171_26=down, J3857_26=down)
    assert section["lines"] == [
        f"{H}: unavailable (no live reading to compare; fixture AVOID_SECTOR)",
        f"{J}: unavailable (no live reading to compare; fixture CONTACT_FIRST)",
    ]
    assert section["ok"] is None and all(e["agree"] is None and e["live_action"] is None for e in section["items"])


def test_one_unavailable_notam_does_not_hide_the_other(loaded):
    section = run_check(loaded, H7171_26=[ConnectionError("down")])
    assert section["lines"][0].startswith(f"{H}: unavailable") and section["lines"][1].endswith(AGREE)
    assert section["ok"] is None


def test_a_notam_that_failed_its_checks_twice_says_failed(loaded):
    bad = {**fixture_reply(J), "exemptions": [{"quote": "THIS SENTENCE IS NOT IN THE NOTAM", "covers": "OTHER"}]}
    section = run_check(loaded, J3857_26=[bad, bad])
    assert section["lines"][1] == f"{J}: {ReaderStatus.FAILED.value} (no live reading to compare; fixture CONTACT_FIRST)"


def test_a_difference_outranks_an_unavailable_one_in_ok(loaded):
    down = [ConnectionError("down")]
    section = run_check(loaded, H7171_26=down, J3857_26=[{**fixture_reply(J), "contact": NO_CONTACT}])
    assert section["ok"] is False


# --- inputs the case does not have --------------------------------------------------------------------------------------------------------


def test_a_fixture_reading_whose_notam_is_not_in_the_pib_says_so(loaded):
    parsed = {n.id: n for n in loaded.pib.notams}
    readings = dict(loaded.readings)
    readings["Z9999/26"] = readings[H]
    section = blue_mill_section(BLUE_MILL_CASE, _lz(loaded), loaded.case.window.start, parsed, readings, {})
    assert "Z9999/26: missing (no live reading to compare; no fixture action; not in the case PIB)" in section["lines"]


def test_a_notam_the_reader_set_did_not_include_says_not_read(loaded):
    parsed = {n.id: n for n in loaded.pib.notams}
    section = blue_mill_section(BLUE_MILL_CASE, _lz(loaded), loaded.case.window.start, parsed, loaded.readings, {})
    assert all("not read (no live reading to compare; fixture " in line and "not in the reader set for this case" in line for line in section["lines"])


def _lz(loaded):
    from flysafelz.clients.openaip import metres_to_feet
    from flysafelz.notam_action import LandingZone

    return LandingZone(lat=loaded.case.lat, lon=loaded.case.lon, elevation_ft=metres_to_feet(loaded.forecast.elevation_m))


# --- no real number in any line (F3) ----------------------------------------------------------------------------------------------------


def test_no_line_or_reason_carries_a_real_phone_number_even_when_the_difference_prints_both_reasons(loaded):
    section = run_check(loaded, J3857_26=[{**fixture_reply(J), "contact": NO_CONTACT}], H7171_26=[{**fixture_reply(H), "applies_to": "UAS_ONLY"}])
    text = repr(section).replace("\\n", "\n")  # repr writes a line break as backslash-n: put it back so a wrapped number is seen
    assert "<PHONE_1>" in text  # the fixture's contact is shown as its placeholder
    raw = " ".join(n.item_e for n in loaded.pib.notams if n.id in (H, J))
    assert digit_runs(raw), "the NOTAM text does hold a number, so the grep below is not vacuous"
    numbers = {*leak_list(), *digit_runs(raw)}  # the committed PIBs' numbers and these two NOTAMs' own
    assert leaked(text, numbers) == set()
