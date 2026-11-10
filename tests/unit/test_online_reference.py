"""The synthetic reference (7 PR-1.F1, G6): each field rule over a fake-tree factory, present and absent. The function under test is `build_reference`, called on real `langsmith` Run objects."""

from dataclasses import fields
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest
from online_helpers import NO_PATH, assess_artifact, fake_run, fake_tree, tool_run

from evals.online.reference import NO_ASSESS, OnlineReference, build_reference
from flysafelz.agent.graph import TOOL_NAMES

STARTED = datetime(2026, 10, 31, 6, 30, tzinfo=UTC)  # the fake run's start: an hour before the frozen now


def assessed(ref):
    assert ref.assess is not None
    return ref.assess


def reference(children=(), **root):
    tree = fake_tree(children, **root)
    return build_reference(tree.root, tree.children)


# --- path -----------------------------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("root", "expected"),
    [
        ({"path": "explain"}, "explain"),  # metadata
        ({"path": NO_PATH, "output_path": "clarify"}, "clarify"),  # outputs
        ({"path": "decline", "output_path": "assess"}, "decline"),  # metadata wins over outputs
        ({"path": NO_PATH}, "unknown"),  # neither
        ({"path": "unknown", "output_path": "act"}, "act"),  # an unknown tag is no tag: the outputs are asked
        ({"path": "banana", "output_path": "explain"}, "explain"),
        ({"path": "banana"}, "unknown"),
        ({"path": "unknown"}, "unknown"),
        ({"path": 7}, "unknown"),
        ({"path": " Assess "}, "assess"),  # 5a's normaliser: any case, spaces off
    ],
)
def test_path_is_metadata_then_outputs_then_unknown(root, expected):
    assert reference(**root).path == expected


@pytest.mark.parametrize("answer", ["[path: decline]\nI can only help with flying conditions.", "NO-GO tomorrow. This is an assess answer."])
def test_an_unknown_path_is_never_guessed_from_the_answer(answer):
    assert reference(path=NO_PATH, answer=answer).path == "unknown"


# --- now ------------------------------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("now_input", "expected"),
    [
        ("2026-10-31T07:30:00Z", datetime(2026, 10, 31, 7, 30, tzinfo=UTC)),
        ("2026-10-31T07:30Z", datetime(2026, 10, 31, 7, 30, tzinfo=UTC)),
        ("2026-10-31T08:30:00+01:00", datetime(2026, 10, 31, 7, 30, tzinfo=UTC)),  # converted to UTC
        ("2026-10-31T07:30:00", datetime(2026, 10, 31, 7, 30, tzinfo=UTC)),  # no zone: UTC
        ("2026-11-02T09:00:00+00:00", datetime(2026, 11, 2, 9, 0, tzinfo=UTC)),
        (None, STARTED),  # absent: the run's start time
        ("not a time", STARTED),  # malformed: the start time, and no error
        ("", STARTED),
        ("31/10/2026", STARTED),
        (12345, STARTED),
    ],
)
def test_now_is_inputs_now_else_the_start_time_in_utc(now_input, expected):
    assert reference(now_input=now_input).now == expected


def test_the_start_time_is_converted_to_utc():
    from datetime import timezone

    start = datetime(2026, 10, 31, 9, 0, tzinfo=timezone(timedelta(hours=2)))
    assert reference(start_time=start).now == datetime(2026, 10, 31, 7, 0, tzinfo=UTC)


def test_a_run_with_no_start_time_and_no_now_is_refused_by_name():
    broken = SimpleNamespace(id="run-x", inputs={}, outputs={}, extra={})  # langsmith's Run cannot be built without a start time
    with pytest.raises(ValueError, match="run-x has no start_time"):
        build_reference(broken, [])


# --- tools ----------------------------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("children", "expected"),
    [
        (["search_guidance"], ("search_guidance",)),
        (["get_weather", "search_guidance", "assess"], ("search_guidance",)),
        (["search_guidance", "search_guidance"], ("search_guidance",)),
        ([], ()),
        (["get_weather", "assess"], ()),
    ],
)
def test_required_tools_holds_search_guidance_when_the_tree_called_it(children, expected):
    assert reference(children).required_tools == expected


def test_a_run_that_is_not_a_tool_run_of_that_name_does_not_count():
    root = fake_run("root")
    chain = tool_run("search_guidance", root, run_type="chain")
    other = tool_run("search_guidance_v2", root)
    assert build_reference(root, [chain, other]).required_tools == ()


def test_a_run_with_no_parent_in_the_tree_is_ignored():
    root = fake_run("root")
    stray = tool_run("search_guidance", root, parent_run_id=None)
    assert build_reference(root, [stray]).required_tools == ()


@pytest.mark.parametrize(("path", "forbidden"), [("decline", tuple(TOOL_NAMES)), ("assess", ()), ("explain", ()), ("act", ()), ("clarify", ()), ("unknown", ())])
def test_forbidden_tools_are_all_of_the_agents_tools_for_a_decline_only(path, forbidden):
    ref = reference(path=path)
    assert ref.forbidden_tools == forbidden
    assert (len(ref.forbidden_tools) == 5) == (path == "decline")


def test_forbidden_tools_follow_the_output_path_when_the_metadata_has_none():
    assert reference(path=NO_PATH, output_path="decline").forbidden_tools == tuple(TOOL_NAMES)


# --- labels that are never filled in ----------------------------------------------------------------------------------------------------------


def test_there_is_no_must_mention_no_window_and_no_sheet_label():
    assert {f.name for f in fields(OnlineReference)} == {"path", "now", "required_tools", "forbidden_tools", "lz_name", "lat", "lon", "assess", "assess_reason"}
    ref = reference(["assess", "search_guidance"], path="assess")
    for label in ("must_mention", "must_not_mention", "window", "verdict_by_segment", "tier", "notam_actions"):
        assert not hasattr(ref, label)


# --- assess ---------------------------------------------------------------------------------------------------------------------------------


def test_the_window_and_the_segments_come_from_the_assess_runs_outputs():
    ref = reference(["assess"])
    assert ref.assess_reason is None and ref.assess is not None
    assert (ref.assess.start, ref.assess.end) == (datetime(2026, 10, 31, 10, tzinfo=UTC), datetime(2026, 10, 31, 16, tzinfo=UTC))
    (segment,) = ref.assess.segments
    assert (segment.start, segment.end, segment.verdict.value, segment.approaching_limits) == (datetime(2026, 10, 31, 10, tzinfo=UTC), datetime(2026, 10, 31, 11, tzinfo=UTC), "PILOT_DISCRETION", True)


def test_an_outlook_hour_with_no_verdict_is_not_a_segment():
    root = fake_run("root")
    artifact = assess_artifact(segments=[("2026-10-31T10:00:00+00:00", "2026-10-31T11:00:00+00:00", "NO_GO"), ("2026-10-31T11:00:00+00:00", "2026-10-31T12:00:00+00:00", None)])
    call = tool_run("assess", root, outputs={"output": {"update": {"messages": [{"type": "tool", "artifact": artifact}]}}})
    segments = assessed(build_reference(root, [call])).segments
    assert [s.verdict.value for s in segments] == ["NO_GO"]


def test_a_plain_tool_message_output_is_read_too():
    root = fake_run("root")
    call = tool_run("assess", root, outputs={"output": {"type": "tool", "content": "text", "artifact": assess_artifact()}})
    assert build_reference(root, [call]).assess is not None


def test_the_last_assess_call_is_the_one_read():
    root = fake_run("root")
    first = tool_run("assess", root, minutes=1, outputs={"output": {"artifact": assess_artifact(window=("2026-10-31T10:00:00Z", "2026-10-31T12:00:00Z"))}})
    second = tool_run("assess", root, minutes=3, outputs={"output": {"artifact": assess_artifact(window=("2026-10-31T13:00:00Z", "2026-10-31T15:00:00Z"))}})
    for order in ([first, second], [second, first]):
        assert assessed(build_reference(root, order)).start == datetime(2026, 10, 31, 13, tzinfo=UTC)


def test_the_latest_readable_assess_call_wins_over_a_later_one_that_gave_nothing():
    root = fake_run("root")
    good = tool_run("assess", root, minutes=1)
    failed = tool_run("assess", root, minutes=3, outputs={"output": {"content": "Could not assess"}}, error="boom")
    assert build_reference(root, [good, failed]).assess is not None


@pytest.mark.parametrize(
    "children",
    [
        [],
        ["get_weather", "search_guidance"],
        [("assess", {"output": {"content": "no artifact"}})],
        [("assess", {})],
        [("assess", {"output": {"artifact": {"tool": "assess", "segments": [{"start": "garbage", "end": "x", "verdict": "NO_GO"}], "window_start": "a", "window_end": "b"}}})],
        [("assess", {"output": {"artifact": {"tool": "assess", "segments": [], "window_start": "2026-10-31T10:00:00Z"}}})],
    ],
    ids=["no-children", "other-tools", "no-artifact", "no-outputs", "bad-times", "no-window-end"],
)
def test_no_assess_gives_none_and_the_reason_no_assess(children):
    root = fake_run("root")
    runs = [tool_run(item, root, minutes=i + 1) if isinstance(item, str) else tool_run(item[0], root, outputs=item[1], minutes=i + 1) for i, item in enumerate(children)]
    ref = build_reference(root, runs)
    assert (ref.assess, ref.assess_reason) == (None, NO_ASSESS) and NO_ASSESS == "no assess"


# --- the launch zone and the metadata variants ----------------------------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["Blue Mill", "Crookham", "Wéstern Ünïcode LZ – 風の丘", "Druridge Bay (north)"])
def test_the_lz_name_survives_into_the_reference_unchanged(name):
    assert reference(metadata={"lz_name": name}).lz_name == name


@pytest.mark.parametrize("value", [None, "", "   ", 7, ["Blue Mill"]])
def test_a_missing_or_unusable_lz_name_is_none(value):
    assert reference(metadata={"lz_name": value}).lz_name is None


def test_the_coordinates_come_from_the_inputs():
    ref = reference()
    assert (ref.lat, ref.lon) == (53.802797, -0.086644)
    bare = fake_run("bare", inputs={})
    assert (build_reference(bare, []).lat, build_reference(bare, []).lon) == (None, None)


def test_the_metadata_is_read_from_extra_metadata_and_a_run_without_it_has_an_unknown_path():
    bare = fake_run("bare", extra=None)
    ref = build_reference(bare, [])
    assert (ref.path, ref.lz_name) == ("unknown", None)


def test_the_reference_is_built_from_the_run_alone_and_changes_nothing_in_it():
    tree = fake_tree(["assess", "search_guidance"], path="decline", now_input="2026-10-31T07:30:00Z")
    before = (tree.root.model_dump_json(), [c.model_dump_json() for c in tree.children])
    build_reference(tree.root, tree.children)
    assert (tree.root.model_dump_json(), [c.model_dump_json() for c in tree.children]) == before
