"""G18 overall verdict and tier, flyable windows, G19 default window and live runs."""

from datetime import UTC, datetime

import pytest
from factories import (
    circle_at,
    forecast_failed,
    notam,
    pib_result,
    reading,
    run_assess,
    utc,
    weather_forecast,
    wind,
)

from flysafelz import cli
from flysafelz.assess import default_window
from flysafelz.contracts import Category, Tier, Verdict


def window_edges(assessment):
    return [(w.start, w.end) for w in assessment.flyable_windows]


def test_the_overall_verdict_is_the_worst_segment_in_the_verdict_tier():
    quiet = run_assess(window=(utc(9), utc(12)))
    assert (quiet.overall_verdict, quiet.overall_tier) == (Verdict.PILOT_DISCRETION, Tier.VERDICT)
    gale = weather_forecast(utc(8), utc(13), changes={utc(10): wind(25), utc(11): wind(25)})
    assert run_assess(window=(utc(9), utc(12)), forecast=gale).overall_verdict is Verdict.NO_GO
    hole = weather_forecast(utc(8), utc(13), changes={utc(11): {"cape": None}})
    assert run_assess(window=(utc(9), utc(12)), forecast=hole).overall_verdict is Verdict.UNVERIFIED


def test_no_go_beats_unverified_in_the_overall_verdict():
    forecast = weather_forecast(utc(8), utc(13), changes={utc(11): {"cape": None}, utc(10): wind(25), utc(9): wind(25)})
    assessment = run_assess(window=(utc(9), utc(12)), forecast=forecast)
    assert assessment.overall_verdict is Verdict.NO_GO


def test_only_outlook_hours_give_no_verdict_and_the_outlook_tier():
    assessment = run_assess(window=(utc(10, day=5, month=11), utc(13, day=5, month=11)))
    assert (assessment.overall_verdict, assessment.overall_tier) == (None, Tier.OUTLOOK)
    assert assessment.flyable_windows == []


def test_only_too_far_hours_give_no_verdict_and_the_too_far_tier():
    assessment = run_assess(window=(utc(10, day=9, month=11), utc(13, day=9, month=11)))
    assert (assessment.overall_verdict, assessment.overall_tier) == (None, Tier.TOO_FAR)


def test_a_verdict_tier_hour_next_to_outlook_hours_still_gives_a_verdict():
    forecast = weather_forecast(utc(8, day=3, month=11), utc(12, day=3, month=11))  # rows through 11 are complete; the rest is OUTLOOK
    assessment = run_assess(window=(utc(10, day=3, month=11), utc(14, day=3, month=11)), forecast=forecast)
    assert {h.tier for h in assessment.hours} == {Tier.VERDICT, Tier.OUTLOOK}
    assert assessment.overall_tier is Tier.VERDICT and assessment.overall_verdict is Verdict.PILOT_DISCRETION


def test_adjacent_runs_across_an_hour_boundary_merge_into_one_window():
    assessment = run_assess(window=(utc(9), utc(12)))
    assert window_edges(assessment) == [(utc(9), utc(12))]


def test_pilot_discretion_with_cautions_and_conditions_still_counts_as_flyable():
    n = notam("J1/26", circle=circle_at(0))
    assessment = run_assess(window=(utc(9), utc(11)), pib=pib_result(n), readings={n.id: reading(notam_id=n.id, geometry=circle_at(0))})
    assert all(s.conditions for h in assessment.hours for s in h.segments)
    assert window_edges(assessment) == [(utc(9), utc(11))]


def test_a_no_go_hour_breaks_the_window_and_the_edges_are_exact():
    forecast = weather_forecast(utc(8), utc(14), changes={utc(11): wind(25), utc(12): wind(25)})  # rows 10, 11 and 12 read these
    assessment = run_assess(window=(utc(9), utc(14)), forecast=forecast)
    assert window_edges(assessment) == [(utc(9), utc(10)), (utc(13), utc(14))]


def test_an_unverified_hour_breaks_the_window():
    forecast = weather_forecast(utc(8), utc(14), changes={utc(12): {"cape": None}})  # rows 11 and 12 read it
    assessment = run_assess(window=(utc(9), utc(14)), forecast=forecast)
    assert window_edges(assessment) == [(utc(9), utc(11)), (utc(13), utc(14))]


def test_a_notam_edge_inside_an_hour_gives_a_window_edge_to_the_minute():
    n = notam("J1/26", circle=circle_at(0), valid_from=utc(9, 30), valid_to=utc(10, 45))
    prohibited = reading(notam_id=n.id, geometry=circle_at(0), category=Category.PROHIBITIVE)
    assessment = run_assess(window=(utc(9), utc(12)), pib=pib_result(n), readings={n.id: prohibited})
    assert window_edges(assessment) == [(utc(9), utc(9, 30)), (utc(10, 45), utc(12))]


def test_an_unread_notam_breaks_the_window_for_the_minutes_it_is_active():
    n = notam("J1/26", circle=circle_at(0), valid_from=utc(9, 30), valid_to=utc(10, 45))
    assessment = run_assess(window=(utc(9), utc(12)), pib=pib_result(n))
    assert window_edges(assessment) == [(utc(9), utc(9, 30)), (utc(10, 45), utc(12))]


def test_outlook_hours_end_a_window_and_a_failed_weather_source_gives_no_window():
    forecast = weather_forecast(utc(8), utc(12))  # the last complete row is 11
    assessment = run_assess(window=(utc(9), utc(15)), forecast=forecast)
    assert window_edges(assessment) == [(utc(9), utc(12))]
    assert run_assess(forecast=forecast_failed()).flyable_windows == []


def test_the_window_end_is_exclusive_07_to_18_gives_11_rows():
    assessment = run_assess(window=(utc(7), utc(18)))
    assert len(assessment.hours) == 11
    assert assessment.hours[-1].hour == utc(17)


# --- G19: the default window and live runs ------------------------------------------------


def test_default_window_is_now_floored_to_the_hour_to_the_end_of_tomorrow_in_london():
    assert default_window(utc(7, 30)) == (utc(7), utc(0, day=2, month=11))


def test_default_window_end_follows_london_across_the_clock_change():
    now = datetime(2026, 10, 24, 12, 40, tzinfo=UTC)  # BST. Tomorrow, 25 Oct, ends at midnight GMT: 26 Oct 00:00 UTC.
    assert default_window(now) == (datetime(2026, 10, 24, 12, 0, tzinfo=UTC), datetime(2026, 10, 26, 0, 0, tzinfo=UTC))


def test_no_window_given_uses_the_default_and_says_so_in_the_assumptions():
    assessment = run_assess(window=None)
    assert (assessment.window_start, assessment.window_end) == (utc(7), utc(0, day=2, month=11))
    assert len(assessment.hours) == 41  # 07:00 on 31 Oct to 00:00 on 2 Nov
    (line,) = [a for a in assessment.assumptions if a.startswith("No window given")]
    assert "2026-10-31T07:00:00Z" in line and "2026-11-02T00:00:00Z" in line


def test_a_window_given_leaves_no_default_window_line():
    assert not [a for a in run_assess().assumptions if a.startswith("No window given")]


def test_a_live_run_drops_the_hours_that_ended_before_now():
    window = (utc(7), utc(12))
    forecast = weather_forecast(utc(6), utc(13))
    kept = run_assess(window=window, now=utc(9, 30), forecast=forecast, live=True)
    assert [h.hour for h in kept.hours] == [utc(9), utc(10), utc(11)]  # 09:00 ends at 10:00, after now
    assert [h.hour for h in run_assess(window=window, now=utc(9, 30), forecast=forecast).hours] == [utc(h) for h in range(7, 12)]


def test_a_live_run_with_the_window_wholly_in_the_past_has_no_hours_and_no_error():
    assessment = run_assess(window=(utc(7), utc(9)), now=utc(12, 30), live=True, forecast=weather_forecast(utc(6), utc(10)))
    assert assessment.hours == []
    assert assessment.flyable_windows == []
    assert assessment.overall_verdict is None and assessment.overall_tier is Tier.TOO_FAR  # the contract has no tier for "past"
    assert any("has already ended: there is nothing left to assess" in note for note in assessment.assumptions)
    assert cli.overall_line(assessment) == "Overall: No verdict: nothing left to assess in this window"


# --- inputs that are refused (E16 and the window checks) -------------------------------------


@pytest.mark.parametrize(
    ("lat", "lon"),
    [(49.7, -1.0), (61.1, -1.0), (53.8, -8.8), (53.8, 2.1), (float("nan"), -1.0), (53.8, float("inf")), (48.85, 2.35)],  # the last is Paris
)
def test_an_lz_outside_the_uk_bounds_raises_before_any_work(lat, lon):
    with pytest.raises(ValueError, match="outside UK coverage"):
        run_assess(lat=lat, lon=lon)


def test_an_lz_on_the_uk_bounds_is_accepted():
    for lat, lon in [(49.8, -8.7), (61.0, 2.0)]:
        assert run_assess(lat=lat, lon=lon, window=(utc(9), utc(10))).hours


def test_the_uk_check_comes_before_the_window_check():
    with pytest.raises(ValueError, match="outside UK coverage"):
        run_assess(lat=48.85, lon=2.35, window=(utc(12), utc(9)))


@pytest.mark.parametrize("window", [(utc(9), utc(9)), (utc(12), utc(9))])
def test_an_empty_or_reversed_window_raises_a_clear_error(window):
    with pytest.raises(ValueError, match="must start before it ends"):
        run_assess(window=window, forecast=weather_forecast(utc(6), utc(13)))


def test_a_naive_time_is_refused():
    with pytest.raises(ValueError, match="timezone-aware"):
        run_assess(now=datetime(2026, 10, 31, 7, 30))
    with pytest.raises(ValueError, match="timezone-aware"):
        run_assess(window=(datetime(2026, 10, 31, 9), utc(12)), forecast=weather_forecast(utc(6), utc(13)))
