"""assess(tz=...): the time text moves into the pilot's zone and nothing else does (FR1, E18). Frozen NOW, no network.

The BST scenario is 24 Oct 2026 (London on BST, UTC+1): a stale PIB, a danger area activated by a NOTAM with a phone contact, and two
NOTAMs the prefilter drops. Fiction-range phone number only.
"""

import inspect
from datetime import UTC, datetime, timedelta

import pytest
from airspace_helpers import volume
from factories import (
    LZ,
    airspace_fetch,
    circle_at,
    notam,
    pib_result,
    reading,
    run_assess,
    utc,
    weather_forecast,
)

from flysafelz import airspace, assess, pib
from flysafelz.assess import notams_in_hour
from flysafelz.contracts import ContactKind

LONDON = "Europe/London"
NEW_YORK = "America/New_York"
D = "\u2013"
BST_NOW = utc(7, 30, day=24)  # 24 Oct 2026: 08:30 BST
WINDOW = (utc(9, day=24), utc(12, day=24))
MASKED_KEYS = {"assumptions", "conditions", "reason"}  # where time text is written (source statuses, NOTAM actions and ignored_notams use `reason`)


def bst_scenario(**changes):
    """Inputs for a BST-day run with every kind of time text in it. `changes` replace any argument of run_assess."""
    circle = circle_at(0)
    area = volume("danger", name="EGD496A", designator="EGD496A", by_notam=True)
    danger = notam("J4/26", circle=circle, subject="RD", item_f="SFC", valid_from=utc(8, day=20), valid_to=utc(18, day=30), item_e="TEMPORARY DANGER AREA EGD496A ACTIVATED. BOOKING 01632 960123.")
    read = reading(notam_id="J4/26", geometry=circle, contact=ContactKind.PHONE, value="01632 960123", designators=["EGD496A"], activity="UAS BVLOS ops")
    later = notam("J7/26", circle=circle_at(0), valid_from=utc(20, day=24), valid_to=utc(20, day=25))  # starts after the window: dropped by the prefilter
    over = notam("J8/26", circle=circle_at(0), valid_from=utc(8, day=20), valid_to=utc(6, day=24))  # ended before the window: dropped by the prefilter
    arguments = {
        "now": BST_NOW,
        "window": WINDOW,
        "airspace": airspace_fetch(area),
        "pib": pib_result(danger, later, over, issued=BST_NOW - timedelta(hours=3, seconds=1)),
        "readings": {"J4/26": read},
        "stale_rule": True,
    }
    arguments.update(changes)
    return arguments


def mask(value):
    """A dump with every text under MASKED_KEYS replaced by a marker of the same shape (so counts and order are still compared)."""
    if isinstance(value, dict):
        masked = {}
        for key, item in value.items():
            if key in MASKED_KEYS and isinstance(item, str):
                masked[key] = "<time text>"
            elif key in MASKED_KEYS and isinstance(item, list):
                masked[key] = ["<time text>"] * len(item)
            else:
                masked[key] = mask(item)
        return masked
    if isinstance(value, list):
        return [mask(item) for item in value]
    return value


def texts(assessment):
    """Every text that can hold a time, in order: assumptions, each segment's conditions and NOTAM reasons, source reasons, debug reasons."""
    found = list(assessment.assumptions)
    for hour in assessment.hours:
        for segment in hour.segments:
            found += segment.conditions
            found += [a.reason for a in segment.notam_actions]
    found += [s.reason for s in assessment.source_statuses]
    found += [i.reason for i in assessment.ignored_notams]
    return found


def verdicts(assessment):
    return [[s.verdict for s in h.segments] for h in assessment.hours]


def condition_groups(assessment):
    return [[c.split(" ")[0] for c in s.conditions] for h in assessment.hours for s in h.segments]


# --- (1) tz=None is the call with no tz ----------------------------------------------------------------------


def test_tz_none_equals_no_tz_as_a_whole_assessment():
    arguments = bst_scenario()
    forecast = weather_forecast(WINDOW[0] - timedelta(hours=1), WINDOW[1] + timedelta(hours=1))
    without = assess.assess(LZ.lat, LZ.lon, forecast=forecast, **arguments)
    explicit_none = assess.assess(LZ.lat, LZ.lon, forecast=forecast, tz=None, **arguments)
    via_factory = run_assess(forecast=forecast, tz=None, **arguments)
    assert explicit_none.model_dump() == without.model_dump() == via_factory.model_dump()


def test_tz_utc_gives_the_same_assessment_too():
    assert run_assess(tz="UTC", **bst_scenario()).model_dump() == run_assess(**bst_scenario()).model_dump()


def test_the_utc_text_is_the_text_it_always_was():
    out = texts(run_assess(**bst_scenario()))
    assert any("PIB issued 2026-10-24T04:29:59Z, more than 3 h before now" in t for t in out)
    assert any(f"Danger area EGD496A active 09:00{D}12:00 (UAS BVLOS ops)" in t for t in out)
    assert any(t.startswith("starts 2026-10-24T20:00:00Z, at or after the window end") for t in out)


# --- (2) a zone changes only the time text -------------------------------------------------------------------


def test_london_on_a_bst_date_changes_only_the_time_text():
    utc_run = run_assess(**bst_scenario())
    london = run_assess(tz=LONDON, **bst_scenario())
    assert utc_run.model_dump() != london.model_dump()
    assert mask(utc_run.model_dump()) == mask(london.model_dump())  # structure, verdicts, condition order, flyable windows: all equal
    assert [h.hour for h in london.hours] == [h.hour for h in utc_run.hours] == [utc(h, day=24) for h in (9, 10, 11)]  # still UTC
    assert (london.window_start, london.window_end) == WINDOW
    assert len(texts(london)) == len(texts(utc_run))


def test_each_kind_of_time_text_is_in_the_zone():
    london = texts(run_assess(tz=LONDON, **bst_scenario()))
    assert any("PIB issued 2026-10-24 05:29 BST, more than 3 h before now" in t for t in london)  # the status reason
    assert any(t.startswith("UNVERIFIED: ") and "PIB issued 2026-10-24 05:29 BST" in t for t in london)  # and the segment condition quoting it
    assert any(t.startswith(f"Danger area EGD496A active 10:00{D}") and " BST (UAS BVLOS ops)" in t for t in london), london
    assert any(t.startswith("starts 2026-10-24 21:00 BST, at or after the window end") for t in london)
    assert any(t.startswith("ended 2026-10-24 07:00 BST, at or before the window start") for t in london)


def test_verdicts_and_the_condition_order_are_equal_for_new_york_too():
    base = run_assess(**bst_scenario())
    new_york = run_assess(tz=NEW_YORK, **bst_scenario())
    assert mask(new_york.model_dump()) == mask(base.model_dump())
    assert verdicts(new_york) == verdicts(base)
    assert condition_groups(new_york) == condition_groups(base)  # the first word of each condition, in order
    assert any("PIB issued 2026-10-24 00:29 EDT" in t for t in texts(new_york))


def test_the_day_tiers_stay_london_days_whatever_the_zone():
    """now is 23:30 on the 24th in London but 18:30 in New York: the tiers still count London days (FR1)."""
    late = datetime(2026, 10, 24, 22, 30, tzinfo=UTC)
    tiers = {tz: [(h.hour, h.tier) for h in run_assess(window=None, now=late, tz=tz).hours] for tz in (None, LONDON, NEW_YORK, "Asia/Tokyo")}
    assert tiers[None] == tiers[LONDON] == tiers[NEW_YORK] == tiers["Asia/Tokyo"]
    assert "tz" not in inspect.signature(assess.default_window).parameters
    assert "tz" not in inspect.signature(assess.fetch_horizon).parameters


# --- (3) E18: the default window across the 25 Oct change ----------------------------------------------------


def test_the_default_window_across_25_oct_labels_its_start_and_end_separately():
    out = run_assess(window=None, now=BST_NOW, tz=LONDON)
    [line] = [a for a in out.assumptions if a.startswith("No window given")]
    assert line == "No window given: assessing from 2026-10-24 08:00 BST (now rounded down to the hour) to 2026-10-26 00:00 GMT (the end of tomorrow, Europe/London)."
    assert (out.window_start, out.window_end) == (utc(7, day=24), utc(0, day=26))  # still UTC


def test_the_default_window_line_in_utc_is_unchanged():
    [line] = [a for a in run_assess(window=None, now=BST_NOW).assumptions if a.startswith("No window given")]
    assert line == "No window given: assessing from 2026-10-24T07:00:00Z (now rounded down to the hour) to 2026-10-26T00:00:00Z (the end of tomorrow, Europe/London)."


def test_the_default_window_in_another_zone_uses_that_zones_abbreviation():
    [line] = [a for a in run_assess(window=None, now=BST_NOW, tz=NEW_YORK).assumptions if a.startswith("No window given")]
    assert line.startswith("No window given: assessing from 2026-10-24 03:00 EDT (now rounded down")
    assert "to 2026-10-25 20:00 EDT (the end of tomorrow, Europe/London)" in line


def test_two_segments_either_side_of_the_change_get_their_own_abbreviation():
    """Two NOTAMs activate the area, one ending before 01:00 UTC on 25 Oct and one starting after it: each segment says BST or GMT as it was."""
    circle = circle_at(0)
    area = volume("danger", name="EGD496A", designator="EGD496A", by_notam=True)
    before = notam("J1/26", circle=circle, subject="RD", item_f="SFC", valid_from=utc(8, day=20), valid_to=utc(0, 30, day=25), item_e="TEMPORARY DANGER AREA EGD496A ACTIVATED. BOOKING 01632 960123.")
    after = notam("J2/26", circle=circle, subject="RD", item_f="SFC", valid_from=utc(1, 30, day=25), valid_to=utc(18, day=30), item_e="TEMPORARY DANGER AREA EGD496A ACTIVATED. BOOKING 01632 960123.")
    reads = {i: reading(notam_id=i, geometry=circle, contact=ContactKind.PHONE, value="01632 960123", designators=["EGD496A"]) for i in ("J1/26", "J2/26")}
    out = run_assess(
        now=utc(20, day=24),
        window=(utc(23, day=24), utc(3, day=25)),
        airspace=airspace_fetch(area),
        pib=pib_result(before, after, issued=utc(19, day=24)),
        readings=reads,
        tz=LONDON,
    )
    labelled = [(seg.start, c.split(" (")[0].rsplit(" ", 1)[1]) for h in out.hours for seg in h.segments for c in seg.conditions if c.startswith("Danger area")]
    assert {abbreviation for _start, abbreviation in labelled} == {"BST", "GMT"}
    for start, abbreviation in labelled:  # the segment's own label: BST before 01:00 UTC on 25 Oct, GMT from then
        assert abbreviation == ("BST" if start < utc(1, day=25) else "GMT"), (start, abbreviation)


# --- failure cases -------------------------------------------------------------------------------------------


def test_a_reversed_window_error_quotes_the_times_in_the_zone():
    reversed_window = (utc(13, day=24), utc(10, day=24))
    with pytest.raises(ValueError, match="the window must start before it ends, got 2026-10-24T13:00:00Z to 2026-10-24T10:00:00Z"):
        run_assess(window=reversed_window, now=BST_NOW)
    with pytest.raises(ValueError, match="the window must start before it ends, got 2026-10-24 14:00 BST to 2026-10-24 11:00 BST"):
        run_assess(window=reversed_window, now=BST_NOW, tz=LONDON)


@pytest.mark.parametrize("tz", ["", "Mars/Olympus"])
def test_an_unknown_zone_raises_even_when_the_run_writes_no_time(tz):
    with pytest.raises(ValueError, match="unknown time zone"):
        run_assess(tz=tz)  # a calm run: no assumption, no stale reason, nothing to quote


@pytest.mark.parametrize("tz", ["utc", "Etc/UTC"])
def test_utc_spelled_as_a_zone_is_accepted_and_says_utc(tz):
    # Only None and "UTC" promise today's byte-identical text; these give the zone form with the abbreviation UTC.
    [line] = [a for a in run_assess(window=None, tz=tz).assumptions if a.startswith("No window given")]
    assert line.startswith("No window given: assessing from 2026-10-31 07:00 UTC (now rounded down")
    assert "to 2026-11-02 00:00 UTC (the end of tomorrow" in line


@pytest.mark.parametrize("tz", [None, LONDON])
def test_a_naive_time_fails_the_same_way_with_or_without_a_zone(tz):
    with pytest.raises(ValueError, match="now must be timezone-aware"):
        run_assess(now=datetime(2026, 10, 31, 7, 30), tz=tz)
    forecast = weather_forecast(utc(8), utc(13))
    with pytest.raises(ValueError, match="window start must be timezone-aware"):
        run_assess(window=(datetime(2026, 10, 31, 9), utc(12)), forecast=forecast, tz=tz)


def test_the_structured_fields_stay_utc_in_a_zone():
    out = run_assess(window=None, now=BST_NOW, tz="Asia/Tokyo")
    moments = [out.window_start, out.window_end, *(h.hour for h in out.hours), *(f.start for f in out.flyable_windows), *(s.fetched_at for s in out.source_statuses)]
    assert moments and all(m.utcoffset() == timedelta(0) for m in moments)
    assert all(s.start.utcoffset() == timedelta(0) for h in out.hours for s in h.segments)


# --- the signatures: keyword-only, default None --------------------------------------------------------------


@pytest.mark.parametrize("function", [assess.assess, airspace.assess_airspace, pib.load_pib, pib.prefilter, pib.sort_notam, notams_in_hour], ids=lambda f: f.__name__)
def test_tz_is_keyword_only_and_defaults_to_none(function):
    parameter = inspect.signature(function).parameters["tz"]
    assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
    assert parameter.default is None
