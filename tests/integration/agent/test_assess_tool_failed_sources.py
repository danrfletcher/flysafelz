"""E9: a failed or stale source through the assess tool in a live run. A failed source never gives PILOT DISCRETION, and the pilot is told where to check by hand."""

from datetime import timedelta

import pytest
from factories import NOW, airspace_fetch, forecast_failed, pib_result
from tool_helpers import calls
from unit.agent.tool_world import WINDOW, World, go

from flysafelz.contracts import SourceName, SourceState
from flysafelz.rules import RULES
from flysafelz.sources import CHECK_BY_HAND

PILOT_DISCRETION = "PILOT DISCRETION"


def failed(source: SourceName) -> dict:
    """The same `failed(source)` pattern as tests/integration/test_assess_failed_sources.py, as the World's own fields."""
    if source is SourceName.WEATHER:
        return {"forecast": forecast_failed()}
    if source is SourceName.AIRSPACE:
        return {"airspace": airspace_fetch(state=SourceState.FAILED)}
    return {"pib": pib_result(state=SourceState.FAILED)}


def stale() -> dict:
    return {"pib": pib_result(issued=NOW - timedelta(hours=float(RULES.pib_stale_hours.value) + 1))}  # type: ignore[arg-type]


CASES = [pytest.param(failed(source), source, "FAILED", "failed", id=f"{source.value}-failed") for source in SourceName]
CASES.append(pytest.param(stale(), SourceName.NOTAMS, "STALE", "stale", id="notams-stale"))


@pytest.mark.parametrize(("world_args", "source", "word", "state"), CASES)
def test_a_failed_or_stale_source_is_named_with_its_reason_the_manual_check_and_the_warning(world_args, source, word, state):
    run = go(World(**world_args), [calls("assess", **WINDOW)])
    text, artifact = run.text("assess"), run.artifact("assess")
    row = next(line for line in text.splitlines() if line.startswith(f"- {source.value} source:"))
    assert word in row and f"Check by hand: {CHECK_BY_HAND[source]}." in row
    assert "Do not fly on this assessment." in text.splitlines()
    info = next(i for i in artifact.sources if i.source == source.value)
    assert (info.state, info.check_by_hand) == (state, CHECK_BY_HAND[source]) and info.reason
    others = [i for i in artifact.sources if i.source != source.value]
    assert all(i.state == "ok" and i.check_by_hand is None for i in others)


@pytest.mark.parametrize(("world_args", "source", "word", "state"), CASES)
def test_each_verdict_tier_segment_reads_unverified_or_no_go_and_never_pilot_discretion(world_args, source, word, state):
    artifact = go(World(**world_args), [calls("assess", **WINDOW)]).artifact("assess")
    segments = [s for s in artifact.segments if s.tier == "VERDICT"]
    assert segments
    for segment in segments:
        assert segment.verdict in {"UNVERIFIED", "NO_GO"} and segment.display in {"UNVERIFIED", "NO-GO"}, segment.start
        if segment.verdict == "NO_GO":
            assert segment.blockers and all(b.source != source.value for b in segment.blockers)  # a working source said no
    assert "UNVERIFIED" in {s.verdict for s in segments}
    assert artifact.flyable == [] and artifact.counts.flyable_windows == 0


@pytest.mark.parametrize(("world_args", "source", "word", "state"), CASES)
def test_neither_the_text_nor_the_artifact_holds_a_pilot_discretion_string_for_those_hours(world_args, source, word, state):
    run = go(World(**world_args), [calls("assess", **WINDOW)])
    shown = run.text("assess") + run.artifact("assess").model_dump_json()
    assert PILOT_DISCRETION not in shown and PILOT_DISCRETION.replace(" ", "_") not in shown


@pytest.mark.parametrize(("world_args", "source", "word", "state"), CASES)
def test_the_unverified_segments_say_why_in_the_text(world_args, source, word, state):
    text = go(World(**world_args), [calls("assess", **WINDOW)]).text("assess")
    assert f"unverified: {source.value} source {state}" in text


def test_all_three_sources_failed_names_all_three_and_the_night_stays_no_go():
    world = World(forecast=forecast_failed(), airspace=airspace_fetch(state=SourceState.FAILED), pib=pib_result(state=SourceState.FAILED))
    run = go(world, [calls("assess", start="2026-10-31 14:00", end="2026-10-31 20:00")])
    text, artifact = run.text("assess"), run.artifact("assess")
    for source in SourceName:
        assert any(line.startswith(f"- {source.value} source: FAILED") for line in text.splitlines())
    assert {s.display for s in artifact.segments} == {"NO-GO", "UNVERIFIED"}
    night = [s for s in artifact.segments if s.start >= "2026-10-31T17:06"]
    assert night and all(s.display == "NO-GO" for s in night)
    assert all(s.display == "UNVERIFIED" for s in artifact.segments if s.end <= "2026-10-31T17:00:00+00:00")
    assert PILOT_DISCRETION not in text + artifact.model_dump_json()


def test_a_stale_pib_is_reported_from_the_assessment_while_the_cache_still_holds_it_as_ok():
    world = World(**stale())
    run = go(world, [calls("get_notams", **WINDOW), calls("assess", **WINDOW)])
    assert run.artifact("get_notams").source.state == "ok"  # what the thread's cache holds
    assert run.artifact("assess").sources[2].source == "notams" and run.artifact("assess").sources[2].state == "stale"
    assert world.calls and len(world.calls) == 1
