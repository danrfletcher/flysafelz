"""Ticket 8 PR-6.F1: the Python side of the answer badges, NOTAM links and highlights.

The UI keeps the NOTAM-id pattern in one TypeScript constant. These tests keep it equal to the server's, and keep the recorded chat payloads
the badge and highlight code reads equal to what the merged server emits (the `assessment` and `overlays` output fields).
"""

import json
import re
from pathlib import Path

import pytest

from flysafelz import contracts
from flysafelz.agent.state import Assessment, OverlayFeature

ROOT = Path(__file__).resolve().parents[2]
UI = ROOT / "ui"
CHAT_FIXTURES = UI / "src" / "test" / "fixtures" / "chat"

# The line in ui/src/lib/notamId.ts that holds the pattern: `export const NOTAM_ID_PATTERN = String.raw`...``.
PATTERN_LINE = re.compile(r"^export const NOTAM_ID_PATTERN = String\.raw`(?P<pattern>[^`]*)`$", re.MULTILINE)


def ts_pattern() -> str:
    found = PATTERN_LINE.search((UI / "src" / "lib" / "notamId.ts").read_text(encoding="utf-8"))
    assert found, "ui/src/lib/notamId.ts has no `export const NOTAM_ID_PATTERN = String.raw`...`` line"
    return found["pattern"]


def final_values(name: str) -> dict:
    events = json.loads((CHAT_FIXTURES / name).read_text(encoding="utf-8"))["events"]
    return [e["data"] for e in events if e["event"] == "values" and e["data"].get("answer")][-1]


def test_the_ui_notam_id_pattern_is_the_servers():
    assert ts_pattern() == contracts.NOTAM_ID_RE.pattern


def test_the_pattern_reads_the_ids_the_ui_links_and_no_others():
    pattern = re.compile(ts_pattern())
    for linked in ("H1234/26", "Q172/26", "V32/26"):
        assert pattern.fullmatch(linked)
    for plain in ("XH1234/26", "H12345/26", "h1234/26"):
        assert not pattern.search(plain)


@pytest.mark.parametrize("name", ["blue-mill-assess.json", "blue-mill-highlights.json", "outlook-too-far.json"])
def test_a_recorded_assessment_and_its_overlays_are_what_the_server_emits(name):
    values = final_values(name)
    # extra="forbid" on both models: a field the server does not have, or one it needs and the recording lacks, fails here.
    assessment = Assessment.model_validate(values["assessment"])
    assert assessment.segments
    for feature in values["overlays"]:
        OverlayFeature.model_validate(feature)


def test_the_follow_up_recording_has_no_assessment_and_no_overlays():
    values = final_values("blue-mill-follow-up.json")
    assert values["assessment"] is None
    assert values["overlays"] == []


def test_the_highlights_recording_has_an_approx_notam_an_unknown_action_and_a_notam_id_outside_its_overlays():
    values = final_values("blue-mill-highlights.json")
    features = [OverlayFeature.model_validate(f) for f in values["overlays"]]
    notams = {f.properties.id: f.properties for f in features if f.properties.kind == "notam"}
    assert any(p.approx for p in notams.values())
    assert any(p.action not in {"NO_FLY", "CONTACT_FIRST", "CHECK_SUP", "AVOID_SECTOR", "AWARENESS", "AWARENESS_MAP"} for p in notams.values())
    ids_in_answer = set(contracts.NOTAM_ID_RE.findall(values["answer"]))
    assert ids_in_answer - set(notams), "the answer should name a NOTAM id that is not in the overlays"
    assert set(notams) <= ids_in_answer
