"""The call-first condition for EGD496A (Blue Mill) in the pilot's zone: assess_airspace(tz=...) -> _contact_condition.

New file so test_danger_area.py stays as it was. The UTC cases here repeat the wording that file pins, to show tz=None and
tz="UTC" change nothing. Fiction-range phone number only.
"""

from datetime import timedelta

import pytest
from airspace_helpers import ELEVATION_FT, LZ, STANDARD_HPA, activation, blue_mill_tda, reading, utc

from flysafelz.airspace import Outcome, TimeSpan, assess_airspace

D = "\u2013"
LONDON = "Europe/London"
TAIL = "(UAS BVLOS ops): phone Flylogix 07700 900123 and fly only if they agree"


def condition(start, end, tz, segment):
    """The CONTACT_FIRST condition for an EGD496A activation [start, end) over a segment it covers."""
    result = assess_airspace(
        [blue_mill_tda()],
        LZ,
        segment,
        site_elevation_ft=ELEVATION_FT,
        pressure_msl=STANDARD_HPA,
        activations=[activation(read=reading(), start=start, end=end)],
        tz=tz,
    )
    [finding] = result.findings
    assert finding.outcome is Outcome.CONTACT_FIRST
    return finding.condition


# (id, activation start, activation end, tz, the range in the condition)
ROWS = [
    ("default None", utc(8), utc(18), None, f"08:00{D}18:00"),
    ("UTC", utc(8), utc(18), "UTC", f"08:00{D}18:00"),
    ("London GMT day (31 Oct)", utc(8), utc(18), LONDON, f"08:00{D}18:00 GMT"),
    ("London BST day (24 Oct)", utc(8, day=24), utc(18, day=24), LONDON, f"09:00{D}19:00 BST"),
    ("Paris", utc(8), utc(18), "Europe/Paris", f"09:00{D}19:00 CET"),
    ("Kolkata :30", utc(8), utc(18), "Asia/Kolkata", f"13:30{D}23:30 IST"),
    ("zone with no letters", utc(8), utc(18), "Pacific/Chatham", f"2026-10-31 21:45 +1345{D}2026-11-01 07:45 +1345"),
    ("across local midnight", utc(14), utc(18), "Asia/Tokyo", f"2026-10-31 23:00 JST{D}2026-11-01 03:00 JST"),
    ("BST to GMT (25 Oct)", utc(23, 30, day=24), utc(2, day=25), LONDON, f"00:30 BST{D}02:00 GMT"),
    ("GMT to BST (29 Mar)", utc(0, 30, day=29, month=3), utc(2, day=29, month=3), LONDON, f"00:30 GMT{D}03:00 BST"),
    ("the repeated hour", utc(0, 30, day=25), utc(1, 30, day=25), LONDON, f"01:30 BST{D}01:30 GMT"),
]


@pytest.mark.parametrize(("start", "end", "tz", "span"), [r[1:] for r in ROWS], ids=[r[0] for r in ROWS])
def test_the_activation_span_is_written_in_the_zone(start, end, tz, span):
    cover = TimeSpan(start=start, end=end)  # the segment is the activation itself, so the activation covers it
    assert condition(start, end, tz, cover) == f"Danger area EGD496A active {span} {TAIL}"


def test_the_utc_wording_is_the_one_test_danger_area_pins():
    assert condition(utc(8), utc(18), None, TimeSpan(start=utc(9), end=utc(10))) == f"Danger area EGD496A active 08:00{D}18:00 {TAIL}"


def test_only_the_time_text_changes_everything_else_in_the_finding_is_equal():
    segment = TimeSpan(start=utc(9, day=24), end=utc(10, day=24))
    runs = {}
    for tz in (None, LONDON, "America/New_York"):
        result = assess_airspace(
            [blue_mill_tda()],
            LZ,
            segment,
            site_elevation_ft=ELEVATION_FT,
            pressure_msl=STANDARD_HPA,
            activations=[activation(read=reading(), start=utc(8, day=24), end=utc(18, day=24))],
            tz=tz,
        )
        runs[tz] = result
    utc_run, london, new_york = runs[None], runs[LONDON], runs["America/New_York"]
    assert "active 08:00" in utc_run.findings[0].condition
    assert "active 09:00" in london.findings[0].condition and "BST" in london.findings[0].condition
    assert "active 04:00" in new_york.findings[0].condition and "EDT" in new_york.findings[0].condition
    for other in (london, new_york):
        a, b = utc_run.model_dump(), other.model_dump()
        a["findings"][0]["condition"] = b["findings"][0]["condition"] = None
        assert a == b


def test_the_activation_itself_is_still_utc():
    made = activation(read=reading(), start=utc(8, day=24), end=utc(18, day=24))
    assess_airspace([blue_mill_tda()], LZ, TimeSpan(start=utc(9, day=24), end=utc(10, day=24)), site_elevation_ft=ELEVATION_FT, activations=[made], tz=LONDON)
    assert (made.start, made.end) == (utc(8, day=24), utc(18, day=24))
    assert made.start.utcoffset() == timedelta(0) and made.end.utcoffset() == timedelta(0)


@pytest.mark.parametrize("tz", ["", "Mars/Olympus"])
def test_an_unknown_zone_raises_even_when_no_time_is_written(tz):
    with pytest.raises(ValueError, match="unknown time zone"):
        assess_airspace([], LZ, TimeSpan(start=utc(9), end=utc(10)), site_elevation_ft=ELEVATION_FT, tz=tz)
