"""F3: no real phone number in anything the fake model receives, and none in a log line.

The leak list is built independently of the redactor: every 10-11 digit run in the raw E text (tests/unit/phone_leaks.py).
The model here is hostile on purpose: it writes the real number into its answers, so the sentences that repeat its answers
back to it (the repair message, the failed checks, the log) are the ones under test. Nothing prints a number: a failing
assert names the NOTAM, not the digits.
"""

import logging

import pytest
from phone_leaks import digit_runs
from reader_helpers import Steps, blue_mill_notams, reply_fields, sent_text

from flysafelz.reader.contract import ReaderStatus
from flysafelz.reader.read import read_notam

NOTAMS = blue_mill_notams()
IDS = ["H7171/26", "J3857/26"]  # both carry a real number in their E text
OBEY = {"quote": "THIS NOTAM DOES NOT APPLY", "covers": "PARAMOTOR_CLASS"}


def squeeze(text: str) -> str:
    return "".join(text.split())


def leaks_in(notam_id: str, *texts: str) -> list[str]:
    """The NOTAM's leak-list entries (digits only) that appear in any text, whitespace and line breaks removed."""
    seen = squeeze("\n".join(texts))
    return [leak for leak in digit_runs(NOTAMS[notam_id].item_e) if leak in seen]


def hostile(notam_id: str, *, breaks_the_form: bool) -> dict:
    """A reply that writes the real number everywhere the model can: a quote, the activity, the contact, a designator.

    breaks_the_form puts the number in an enum field as well, so the form check (f) has to quote it back.
    """
    number = digit_runs(NOTAMS[notam_id].item_e)[0]
    return reply_fields(
        activity=f"call {number}",
        exemptions=[{"quote": f"CALL {number} FIRST", "covers": "CONDITIONAL"}],
        contact={"kind": "phone", "value": number},
        activates_designators=[number],
        applies_to=f"CALL {number}" if breaks_the_form else "ALL_AIRCRAFT",
    )


def test_the_fixture_numbers_are_there_to_leak():
    for notam_id in IDS:
        assert digit_runs(NOTAMS[notam_id].item_e), notam_id


@pytest.mark.parametrize("breaks_the_form", [False, True], ids=["fails the E checks", "fails the form check"])
@pytest.mark.parametrize("notam_id", IDS)
def test_a_hostile_model_that_writes_the_real_number_gets_a_placeholder_back_and_nothing_is_logged_raw(notam_id, breaks_the_form, caplog):
    caplog.set_level(logging.DEBUG)
    bad = hostile(notam_id, breaks_the_form=breaks_the_form)
    double = Steps(bad, bad)
    result = read_notam(NOTAMS[notam_id], double)
    assert result.status is ReaderStatus.FAILED and result.reading is None
    assert len(double.calls) == 2 and result.failed_checks
    everything = [sent_text(double), caplog.text, *result.failed_checks]
    assert not leaks_in(notam_id, *everything), f"{notam_id}: a number from its E text reached a message, a log line or a failed check"
    assert "<PHONE_" in double.calls[1]["messages"][1].content  # the repair names what it found, as a placeholder


@pytest.mark.parametrize("notam_id", IDS)
def test_the_repair_message_and_the_log_hold_only_placeholders_when_the_model_invents_a_quote(notam_id, caplog):
    caplog.set_level(logging.DEBUG)
    invented = reply_fields(exemptions=[OBEY], contact={"kind": "phone", "value": "<PHONE_1>"})
    double = Steps(invented, reply_fields(contact={"kind": "phone", "value": "<PHONE_1>"}))
    result = read_notam(NOTAMS[notam_id], double)
    assert result.status is ReaderStatus.REPAIRED
    assert not leaks_in(notam_id, sent_text(double), caplog.text, *result.failed_checks)
    assert "THIS NOTAM DOES NOT APPLY" in double.calls[1]["messages"][1].content


@pytest.mark.parametrize("notam_id", IDS)
def test_an_error_that_echoes_the_number_is_not_logged_with_it(notam_id, caplog):
    caplog.set_level(logging.DEBUG)
    number = digit_runs(NOTAMS[notam_id].item_e)[0]
    first_fails, ok = hostile(notam_id, breaks_the_form=False), reply_fields(contact={"kind": "phone", "value": "<PHONE_1>"})
    for double in (Steps(RuntimeError(f"bad request: {number} is not valid")), Steps(first_fails, RuntimeError(f"bad request: {number}")), Steps(first_fails, ok)):
        result = read_notam(NOTAMS[notam_id], double)
        assert result.status in (ReaderStatus.UNAVAILABLE, ReaderStatus.REPAIRED)
        assert not leaks_in(notam_id, sent_text(double), caplog.text, *result.failed_checks), notam_id


@pytest.mark.parametrize("notam_id", IDS)
def test_a_clean_read_sends_and_logs_no_number(notam_id, caplog):
    caplog.set_level(logging.DEBUG)
    double = Steps(reply_fields(contact={"kind": "phone", "value": "<PHONE_1>"}))
    result = read_notam(NOTAMS[notam_id], double)
    assert result.status is ReaderStatus.READ
    assert not leaks_in(notam_id, sent_text(double), caplog.text, result.model_dump_json())
    assert result.reading.contact.value == "<PHONE_1>"  # the reading holds the placeholder; putting the number back is a later step


@pytest.mark.parametrize("where", ["top level", "inside contact"])
@pytest.mark.parametrize("notam_id", IDS)
def test_a_hostile_model_that_writes_the_real_number_as_an_extra_key_gets_a_placeholder_back(notam_id, where, caplog):
    """A key name is the model's text too: it reaches the repair message, the log and the failed checks through the field path."""
    caplog.set_level(logging.DEBUG)
    number = digit_runs(NOTAMS[notam_id].item_e)[0]
    bad = reply_fields()
    if where == "top level":
        bad[number] = "x"
    else:
        bad["contact"] = {"kind": "phone", "value": "<PHONE_1>", number: "x"}
    double = Steps(bad, bad)
    result = read_notam(NOTAMS[notam_id], double)
    assert result.status is ReaderStatus.FAILED and len(double.calls) == 2
    assert any("<PHONE_" in sentence and "not on the form" in sentence for sentence in result.failed_checks)
    assert "<PHONE_" in double.calls[1]["messages"][1].content
    assert not leaks_in(notam_id, sent_text(double), caplog.text, *result.failed_checks), f"{notam_id}: a number written as a key reached a message, a log line or a failed check"


@pytest.mark.parametrize("notam_id", IDS)
def test_a_real_number_the_model_writes_in_the_activity_or_contact_is_never_returned_as_a_reading(notam_id, caplog):
    """Only the free-text fields the checks used to let through: no bad quote, designator or enum, just the number in the text."""
    caplog.set_level(logging.DEBUG)
    number = digit_runs(NOTAMS[notam_id].item_e)[0]
    bad = reply_fields(activity=f"call {number}", contact={"kind": "phone", "value": number})
    double = Steps(bad, bad)
    result = read_notam(NOTAMS[notam_id], double)
    assert result.status is ReaderStatus.FAILED and result.reading is None and len(double.calls) == 2
    assert any(sentence.startswith("your activity has a phone number written out") for sentence in result.failed_checks)
    assert any(sentence.startswith("your contact value has a phone number written out") for sentence in result.failed_checks)
    assert "<PHONE_" in double.calls[1]["messages"][1].content
    assert not leaks_in(notam_id, sent_text(double), caplog.text, result.model_dump_json()), f"{notam_id}: a written-out number reached a message, a log line or the result"


@pytest.mark.parametrize("notam_id", IDS)
def test_a_model_that_writes_the_number_first_and_the_placeholder_on_repair_is_repaired(notam_id):
    number = digit_runs(NOTAMS[notam_id].item_e)[0]
    bad = reply_fields(contact={"kind": "phone", "value": number})
    good = reply_fields(contact={"kind": "phone", "value": "<PHONE_1>"})
    result = read_notam(NOTAMS[notam_id], Steps(bad, good))
    assert result.status is ReaderStatus.REPAIRED
    assert result.reading.contact.value == "<PHONE_1>" and not leaks_in(notam_id, result.model_dump_json())
