"""The live Blue Mill check (GP7, R3.3): what the real model makes of the case's two NOTAMs, against the case's hand-written readings.

    blue_mill_section(case, lz, day, notams, fixtures, results) -> dict   (the report's `blue_mill` section)

For each NOTAM in the case's readings.yaml, `notam_action()` gives an action at the 08:00 hour for the live reading (the
reader's result, read from the frozen case PIB) and for the fixture reading, both at the case's landing zone. The same
function on both, so the two differ only by the readings. A line says "H7171/26: AVOID_SECTOR live vs AVOID_SECTOR fixture ✓",
or names the difference (both actions and both reasons) after a ✗. A NOTAM whose live read gave no reading says why
("unavailable", "failed") and is neither a ✓ nor a ✗: there was nothing to compare.

A NOTAM that the fixture gives a contact (phone or email) is compared as "a contact present", never as a fixed number: the
live reading must carry a phone or an email too. Real numbers stay out of every line: the fixture's contact is redacted
against the NOTAM's own E text first, and the reasons hold placeholders (R3.5).

The reading itself is done by run.py, which owns every model call; this module only compares.
"""

from collections.abc import Mapping
from datetime import UTC, datetime, time
from typing import Any

from flysafelz.contracts import ContactKind, NotamReading, ParsedNotam
from flysafelz.daylight import ONE_HOUR
from flysafelz.notam_action import ActionSegment, LandingZone, notam_action
from flysafelz.reader.contract import ReaderResult, ReaderStatus
from flysafelz.redact import redact_known, redact_phones

SEGMENT_START = "08:00"  # UTC, on the case's first day: the hour the plan's golden table is read at
AGREE, DIFFER = "\u2713", "\u2717"
CONTACT_KINDS = (ContactKind.PHONE, ContactKind.EMAIL)
READ_STATUSES = (ReaderStatus.READ, ReaderStatus.REPAIRED)


def segment_on(day: datetime) -> ActionSegment:
    """The one-hour segment at SEGMENT_START (UTC) on the day of `day`."""
    start = datetime.combine(day.astimezone(UTC).date(), time.fromisoformat(SEGMENT_START), tzinfo=UTC)
    return ActionSegment(start=start, end=start + ONE_HOUR)


def redacted_fixture(parsed: ParsedNotam, reading: NotamReading) -> NotamReading:
    """The fixture reading with any real number in its contact swapped for its NOTAM's placeholder, so no reason carries one."""
    value = reading.contact.value
    if not value:
        return reading
    text, _map = redact_known(value, redact_phones(parsed.item_e)[1])
    return reading.model_copy(update={"contact": reading.contact.model_copy(update={"value": text})})


def _entry(notam_id: str, **fields: Any) -> dict[str, Any]:
    return {"id": notam_id, "status": None, "live_action": None, "fixture_action": None, "agree": None, "live_reason": None, "fixture_reason": None, "note": None, **fields}


def _compare(parsed: ParsedNotam, fixture: NotamReading, result: ReaderResult | None, lz: LandingZone, segment: ActionSegment) -> dict[str, Any]:
    fixture_result = notam_action(parsed, redacted_fixture(parsed, fixture), lz, segment)
    if result is None:
        return _entry(parsed.id, status="not read", fixture_action=fixture_result.action.value, fixture_reason=fixture_result.reason, note="not in the reader set for this case")
    if result.reading is None or result.status not in READ_STATUSES:
        return _entry(parsed.id, status=result.status.value, fixture_action=fixture_result.action.value, fixture_reason=fixture_result.reason)
    live_result = notam_action(parsed, result.reading, lz, segment)
    contact_lost = fixture.contact.kind in CONTACT_KINDS and result.reading.contact.kind not in CONTACT_KINDS
    agree = live_result.action is fixture_result.action and not contact_lost
    note = f"the live reading has no contact (fixture: {fixture.contact.kind.value})" if contact_lost else None
    return _entry(
        parsed.id,
        status=result.status.value,
        live_action=live_result.action.value,
        fixture_action=fixture_result.action.value,
        agree=agree,
        live_reason=live_result.reason,
        fixture_reason=fixture_result.reason,
        note=note,
    )


def line_for(entry: Mapping[str, Any]) -> str:
    """One report line for a compared NOTAM."""
    notam_id, fixture = entry["id"], entry["fixture_action"]
    if entry["agree"] is None:
        against = f"fixture {fixture}" if fixture else "no fixture action"
        why = f"; {entry['note']}" if entry["note"] else ""
        return f"{notam_id}: {entry['status']} (no live reading to compare; {against}{why})"
    live = entry["live_action"]
    if entry["agree"]:
        return f"{notam_id}: {live} live vs {fixture} fixture {AGREE}"
    note = f"; {entry['note']}" if entry["note"] else ""
    return f"{notam_id}: {live} live vs {fixture} fixture {DIFFER} (live: {entry['live_reason']}; fixture: {entry['fixture_reason']}{note})"


def blue_mill_section(
    case: str,
    lz: LandingZone,
    day: datetime,
    notams: Mapping[str, ParsedNotam],
    fixtures: Mapping[str, NotamReading],
    results: Mapping[str, ReaderResult],
) -> dict[str, Any]:
    """The report's `blue_mill` section: one entry and one line per NOTAM of the case's readings.yaml, in id order.

    `ok` is False when any NOTAM differed, else None when any had no live reading to compare (or there is nothing to
    compare), else True.
    """
    segment = segment_on(day)
    entries = [_compare(notams[notam_id], fixtures[notam_id], results.get(notam_id), lz, segment) for notam_id in sorted(fixtures) if notam_id in notams]
    entries += [_entry(notam_id, status="missing", note="not in the case PIB") for notam_id in sorted(fixtures) if notam_id not in notams]
    verdicts = [entry["agree"] for entry in entries]
    ok = False if False in verdicts else None if (None in verdicts or not verdicts) else True
    return {
        "case": case,
        "segment": f"{segment.start:%Y-%m-%d %H:%M}-{segment.end:%H:%M}Z",
        "heading": f"{case}: the live reader against the case's readings.yaml, action for action, {segment.start:%H:%M}-{segment.end:%H:%M}Z on {segment.start:%Y-%m-%d}",
        "ok": ok,
        "items": entries,
        "lines": [line_for(entry) for entry in entries],
    }
