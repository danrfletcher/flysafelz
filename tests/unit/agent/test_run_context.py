"""RunContext and run_now: the run's clock, and the defaults that leave an existing case-mode run unchanged."""

from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path

import pytest
from langchain_core.messages import HumanMessage

from flysafelz.agent import state
from flysafelz.agent.input_check import validate_request
from flysafelz.agent.state import RunContext, run_now

WALL_CLOCK = datetime(2030, 1, 2, 3, 4, tzinfo=UTC)


def test_run_now_returns_the_frozen_now(now):
    assert run_now(RunContext(now=now)) == datetime(2026, 10, 31, 7, 30, tzinfo=UTC)


def test_run_now_keeps_the_zone_it_was_given():
    plus_one = datetime(2026, 10, 31, 8, 30, tzinfo=timezone(timedelta(hours=1)))
    assert run_now(RunContext(now=plus_one)) is plus_one


def test_run_now_without_now_reads_the_wall_clock_in_utc(monkeypatch):
    monkeypatch.setattr(state, "_wall_clock", lambda: WALL_CLOCK)
    assert run_now(RunContext()) == WALL_CLOCK
    assert run_now(None) == WALL_CLOCK


def test_the_real_wall_clock_is_aware_utc():
    before = datetime.now(UTC)
    got = run_now(RunContext())
    assert got.utcoffset() == timedelta(0)
    assert before <= got <= datetime.now(UTC)


def test_run_now_refuses_a_naive_datetime():
    with pytest.raises(ValueError, match="timezone-aware"):
        run_now(RunContext(now=datetime(2026, 10, 31, 7, 30)))


def test_the_run_context_sets_exactly_these_fields():
    assert list(RunContext.__dataclass_fields__) == ["now", "case", "reader", "reader_cache", "drop"]


def test_the_defaults_are_the_case_mode_defaults():
    ctx = RunContext()
    assert (ctx.now, ctx.case, ctx.reader, ctx.reader_cache, ctx.drop) == (None, None, "fixture", None, ())


def test_json_from_studio_becomes_the_typed_fields(now):
    ctx = RunContext(now="2026-10-31T07:30:00Z", case="blue_mill_2026-10-31", reader="live", reader_cache="/tmp/r.sqlite", drop=["pib", "weather"])
    assert ctx.now == now
    assert ctx.reader_cache == Path("/tmp/r.sqlite")
    assert ctx.drop == ("pib", "weather")


def test_the_run_context_is_frozen():
    with pytest.raises(AttributeError):
        RunContext().case = "x"  # type: ignore[misc]


def test_the_defaults_leave_an_existing_case_mode_run_unchanged(now):
    """A run that sets only `case` (and `now`), as every case-mode run did before the switches, is accepted with the same update as one that spells out the defaults."""
    request = {"messages": [HumanMessage("Can I fly here between 10 and 4 today?")], "lat": 53.802797, "lon": -0.086644}
    old_style = RunContext(now=now, case="blue_mill_2026-10-31")
    spelled_out = RunContext(now=now, case="blue_mill_2026-10-31", reader="fixture", reader_cache=None, drop=())
    assert old_style == spelled_out
    assert validate_request(request, old_style) == validate_request(request, spelled_out)
    assert validate_request(request, old_style) == validate_request(request, None)
