"""R3.5: _checked() looks for an exemption quote in the redacted E text, so a quote holding a real number could never match.

No committed readings.yaml may hold one. (A fixture's contact may: the CLI redacts it on the way in.)
"""

from pathlib import Path

import pytest
import yaml

from flysafelz.redact import redact_phones

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures"
READINGS = sorted(FIXTURES.glob("*/readings.yaml"))


def test_there_are_committed_readings_to_check():
    assert {path.parent.name for path in READINGS} >= {"blue_mill_2026-10-31", "west_sleekburn_2026-10-31"}


@pytest.mark.parametrize("path", READINGS, ids=lambda path: path.parent.name)
def test_no_exemption_quote_in_a_committed_readings_file_contains_a_real_phone_number(path):
    for reading in yaml.safe_load(path.read_text(encoding="utf-8")) or []:
        for exemption in reading.get("exemptions") or []:
            assert redact_phones(exemption["quote"])[1] == {}, (path.parent.name, reading["id"])


def test_the_check_catches_a_quote_with_a_number():
    assert redact_phones("PARAMOTORS EXEMPT, PHONE 07700 900123")[1] != {}
