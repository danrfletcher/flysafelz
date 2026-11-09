"""The summary's date (7 PR-4.F1, G11): today in Europe/London, not UTC. The clock is a value the caller passes, never read here."""

import ast
from datetime import UTC, date, datetime, timedelta, timezone
from pathlib import Path

import pytest

from evals.online import summary

SOURCE = Path(summary.__file__)


@pytest.mark.parametrize(
    ("utc", "day", "name"),
    [
        pytest.param(datetime(2026, 10, 20, 23, 30, tzinfo=UTC), date(2026, 10, 21), "online-2026-10-21", id="23:30 UTC on 20 Oct is 00:30 BST on 21 Oct"),
        pytest.param(datetime(2026, 11, 8, 23, 30, tzinfo=UTC), date(2026, 11, 8), "online-2026-11-08", id="23:30 UTC on 8 Nov is GMT: the same day"),
        pytest.param(datetime(2026, 10, 24, 22, 59, tzinfo=UTC), date(2026, 10, 24), "online-2026-10-24", id="22:59 UTC on 24 Oct is 23:59 BST: still the 24th"),
        pytest.param(datetime(2026, 10, 24, 23, 0, tzinfo=UTC), date(2026, 10, 25), "online-2026-10-25", id="23:00 UTC on 24 Oct is midnight BST: the 25th"),
        pytest.param(datetime(2026, 10, 25, 0, 59, tzinfo=UTC), date(2026, 10, 25), "online-2026-10-25", id="00:59 UTC on 25 Oct is 01:59 BST, the hour before the change"),
        pytest.param(datetime(2026, 10, 25, 1, 0, tzinfo=UTC), date(2026, 10, 25), "online-2026-10-25", id="01:00 UTC on 25 Oct: the clocks go back, 01:00 GMT"),
        pytest.param(datetime(2026, 10, 25, 23, 30, tzinfo=UTC), date(2026, 10, 25), "online-2026-10-25", id="23:30 UTC on 25 Oct is GMT now: the 25th, not the 26th"),
        pytest.param(datetime(2026, 7, 1, 23, 30, tzinfo=UTC), date(2026, 7, 2), "online-2026-07-02", id="summer: 23:30 UTC is 00:30 BST the next day"),
        pytest.param(datetime(2026, 12, 31, 23, 59, tzinfo=UTC), date(2026, 12, 31), "online-2026-12-31", id="winter, the last minute of the year"),
    ],
)
def test_the_summary_is_dated_in_europe_london(utc, day, name):
    assert summary.summary_date(utc) == day
    assert summary.summary_name(summary.summary_date(utc)) == name


def test_a_time_in_another_zone_is_dated_by_the_same_instant():
    new_york = timezone(timedelta(hours=-5))
    assert summary.summary_date(datetime(2026, 10, 20, 18, 30, tzinfo=new_york)) == date(2026, 10, 21)  # 23:30 UTC


def test_the_day_is_not_the_utc_day_across_the_change():
    instant = datetime(2026, 10, 20, 23, 30, tzinfo=UTC)
    assert instant.date() == date(2026, 10, 20) and summary.summary_date(instant) == date(2026, 10, 21)


def test_a_time_with_no_zone_is_refused_rather_than_dated_by_the_machine():
    with pytest.raises(ValueError, match="zone"):
        summary.summary_date(datetime(2026, 10, 20, 23, 30))


def test_the_module_never_reads_the_clock_itself():
    calls = {f"{ast.unparse(node.func)}" for node in ast.walk(ast.parse(SOURCE.read_text(encoding="utf-8"))) if isinstance(node, ast.Call)}
    assert not {call for call in calls if call.endswith((".now", ".today", ".utcnow")) or call in ("time.time", "time")}


def test_the_zone_comes_from_zoneinfo_with_the_pinned_tzdata():
    assert str(summary.TIMEZONE) == "Europe/London"
    assert "tzdata" in (SOURCE.parents[2] / "pyproject.toml").read_text(encoding="utf-8")
