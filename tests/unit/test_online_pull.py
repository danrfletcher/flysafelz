"""The online pull (7 PR-1.F1): SINCE and LIMIT, the list call, limit_hit, the allow-list, skip counting and precedence, child runs, E1 and E3. Real code, a recording client, fake runs, sockets off."""

from collections import Counter
from datetime import timedelta

import pytest
from online_helpers import NO_ANSWER, NO_SOURCE, NOW, OnlineRecordingClient, fake_run

from evals.online import pull
from evals.online.pull import DEFAULT_LIMIT, DEFAULT_SINCE, parse_limit, parse_since, pull_from_env, pull_runs, settings_from_env, skip_reason, source_of

WEEK = timedelta(days=7)


def test_the_helpers_clock_is_the_frozen_now(now):
    assert NOW == now and fake_run().start_time == now - timedelta(hours=1)


def pulled(runs, now, *, limit=500, since=WEEK, project="flysafelz"):
    client = OnlineRecordingClient(runs)
    return pull_runs(client, project, since, limit, now), client


# --- SINCE and LIMIT ------------------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("7d", timedelta(days=7)),
        ("24h", timedelta(hours=24)),
        ("1d", timedelta(days=1)),
        ("1h", timedelta(hours=1)),
        ("30d", timedelta(days=30)),
        (" 7d ", timedelta(days=7)),
        ("007d", timedelta(days=7)),
    ],
)
def test_since_accepts_a_whole_number_of_days_or_hours(text, expected):
    assert parse_since(text) == expected


@pytest.mark.parametrize("text", ["7", "7w", "-1d", "0d", "0h", "", "   ", "d", "7 d", "1.5d", "7D", "7dd", "d7", "+7d", "٧d"])
def test_since_refuses_everything_else_naming_the_variable(text, capsys):
    with pytest.raises(SystemExit) as stop:
        parse_since(text)
    assert stop.value.code == 2
    assert "SINCE" in capsys.readouterr().err


@pytest.mark.parametrize(("text", "expected"), [("500", 500), ("1", 1), (" 25 ", 25), ("10000", 10000)])
def test_limit_accepts_a_whole_number_above_zero(text, expected):
    assert parse_limit(text) == expected


@pytest.mark.parametrize("text", ["0", "-5", "abc", "", "  ", "1.5", "5e2", "+5", "1,000"])
def test_limit_refuses_everything_else_naming_the_variable(text, capsys):
    with pytest.raises(SystemExit) as stop:
        parse_limit(text)
    assert stop.value.code == 2
    assert "LIMIT" in capsys.readouterr().err


def test_the_defaults_are_seven_days_and_five_hundred_runs():
    settings = settings_from_env({})
    assert (DEFAULT_SINCE, DEFAULT_LIMIT) == ("7d", "500")
    assert (settings.project, settings.since, settings.limit) == ("flysafelz", WEEK, 500)


def test_the_settings_come_from_the_environment():
    settings = settings_from_env({"LANGSMITH_PROJECT": "prod-traces", "SINCE": "24h", "LIMIT": "40"})
    assert (settings.project, settings.since, settings.limit) == ("prod-traces", timedelta(hours=24), 40)


@pytest.mark.parametrize(("variable", "value"), [("SINCE", "7"), ("SINCE", ""), ("SINCE", "0d"), ("LIMIT", "0"), ("LIMIT", "abc"), ("LIMIT", "-5")])
def test_a_bad_value_exits_before_the_client_is_built(variable, value, now, capsys):
    built = []
    with pytest.raises(SystemExit) as stop:
        pull_from_env(lambda: built.append(1) or OnlineRecordingClient(), now, {variable: value})
    assert stop.value.code == 2 and variable in capsys.readouterr().err
    assert built == []


# --- the list call (G3) ---------------------------------------------------------------------------------------------------------------------


def test_one_list_call_with_the_project_root_runs_and_the_window_start(now):
    _, client = pulled([fake_run("a")], now)
    assert client.methods_called() == ["list_runs"]
    assert client.list_run_calls() == [{"project_name": "flysafelz", "is_root": True, "start_time": now - WEEK}]


def test_the_window_start_follows_since(now):
    _, client = pulled([], now, since=timedelta(hours=24))
    assert client.list_run_calls()[0]["start_time"] == now - timedelta(hours=24)


def test_a_run_that_starts_exactly_at_the_window_start_is_included(now):
    edge = fake_run("edge", start_time=now - WEEK)
    result, client = pulled([edge], now)
    assert result.runs == [edge]
    assert client.list_run_calls()[0]["start_time"] == edge.start_time


def test_the_pull_through_the_environment_lists_once_and_builds_the_client_once(now):
    client, built = OnlineRecordingClient([fake_run("a"), fake_run("b", source="online-seed")]), []
    result = pull_from_env(lambda: built.append(1) or client, now, {"LANGSMITH_PROJECT": "prod-traces", "SINCE": "24h", "LIMIT": "10"})
    assert built == [1] and client.methods_called() == ["list_runs"]
    assert client.list_run_calls() == [{"project_name": "prod-traces", "is_root": True, "start_time": now - timedelta(hours=24)}]
    assert (result.project, result.since, result.limit, len(result.runs)) == ("prod-traces", timedelta(hours=24), 10, 2)


# --- limit_hit ------------------------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(("available", "hit", "read"), [(501, True, 500), (600, True, 500), (500, False, 500), (499, False, 499), (0, False, 0)])
def test_limit_hit_is_true_only_when_more_than_limit_runs_were_available(available, hit, read, now):
    result, client = pulled([fake_run(f"run-{i}") for i in range(available)], now, limit=500)
    assert (result.limit_hit, result.read, len(result.runs)) == (hit, read, read)
    assert client.pages_read <= 501  # the generator is not drained past one more than the limit


def test_the_limit_counts_runs_read_before_any_is_skipped(now):
    runs = [fake_run(f"case-{i}", source="case") for i in range(3)] + [fake_run("pilot-1"), fake_run("pilot-2")]
    result, _ = pulled(runs, now, limit=3)
    assert (result.limit_hit, result.read, result.runs, Counter(s.source for s in result.skipped)) == (True, 3, [], {"case": 3})


# --- the allow-list (G4) --------------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("source", ["pilot", "online-seed"])
def test_pilot_and_online_seed_runs_are_returned_for_scoring(source, now):
    run = fake_run("ok", source=source)
    result, _ = pulled([run], now)
    assert result.runs == [run] and result.skipped == []


@pytest.mark.parametrize(
    ("source", "counted_as"),
    [
        ("case", "case"),
        ("judge", "judge"),
        ("v0", "v0"),
        ("studio", "studio"),
        ("Pilot", "Pilot"),  # the allow-list matches the value exactly
        (42, "42"),  # a non-string counts as its text
        (True, "True"),
        (NO_SOURCE, "missing"),
        (None, "missing"),
        ("", "missing"),
        ("   ", "missing"),
    ],
)
def test_every_other_source_is_skipped_and_counted_under_its_own_name(source, counted_as, now):
    run = fake_run("other", source=source)
    result, _ = pulled([run], now)
    assert result.runs == [] and [(s.run, s.reason, s.source) for s in result.skipped] == [(run, "source", counted_as)]
    assert result.by_source == {counted_as: 1} and result.by_reason == {"source": 1}


def test_skips_are_counted_by_source_and_by_reason(now):
    runs = [
        fake_run("c1", source="case"),
        fake_run("c2", source="case"),
        fake_run("j", source="judge"),
        fake_run("v", source="v0"),
        fake_run("m", source=NO_SOURCE),
        fake_run("p"),
        fake_run("s", source="online-seed"),
    ]
    result, _ = pulled(runs, now)
    assert [run.id for run in result.runs] == [runs[5].id, runs[6].id]
    assert result.by_source == {"case": 2, "judge": 1, "v0": 1, "missing": 1}
    assert result.by_reason == {"source": 5}


@pytest.mark.parametrize("metadata", [{"source": "pilot"}, {"source": "online-seed"}])
def test_the_source_is_read_from_the_runs_extra_metadata(metadata, now):
    result, _ = pulled([fake_run("a", source=NO_SOURCE, extra={"metadata": metadata})], now)
    assert len(result.runs) == 1


def test_a_source_kept_outside_extra_metadata_is_not_read(now):
    run = fake_run("a", source=NO_SOURCE, extra={"other": {"source": "pilot"}})
    assert source_of(run) == "missing" and pulled([run], now)[0].runs == []


# --- experiment runs, errored runs, precedence ------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("source", ["pilot", "online-seed", "case", "judge", NO_SOURCE])
def test_a_run_with_a_reference_example_id_is_skipped_as_experiment_whatever_its_source(source, now):
    run = fake_run("exp", source=source, reference_example_id="00000000-0000-4000-8000-000000000001")
    result, _ = pulled([run], now)
    assert result.runs == [] and [s.reason for s in result.skipped] == ["experiment"] and result.by_reason == {"experiment": 1} and result.by_source == {}


@pytest.mark.parametrize(
    "overrides",
    [
        {"error": "boom"},
        {"error": "boom", "answer": "An answer that was given"},
        {"answer": NO_ANSWER},
        {"answer": ""},
        {"answer": "   \n"},
        {"answer": None},
        {"outputs": None},
        {"outputs": {}},
    ],
    ids=["error", "error-with-answer", "no-answer-key", "empty-answer", "blank-answer", "null-answer", "no-outputs", "empty-outputs"],
)
def test_a_run_with_an_error_or_no_final_answer_is_skipped_as_errored(overrides, now):
    result, _ = pulled([fake_run("bad", **overrides)], now)
    assert result.runs == [] and [s.reason for s in result.skipped] == ["errored"] and result.by_reason == {"errored": 1}


PRECEDENCE = [
    # (experiment, bad source, errored) -> the one reason the run is counted under
    (True, True, False, "experiment"),
    (True, False, True, "experiment"),
    (False, True, True, "source"),
    (True, True, True, "experiment"),
    (True, False, False, "experiment"),
    (False, True, False, "source"),
    (False, False, True, "errored"),
    (False, False, False, None),
]


@pytest.mark.parametrize(("experiment", "bad_source", "errored", "expected"), PRECEDENCE)
def test_each_run_is_counted_once_under_the_first_reason_experiment_then_source_then_errored(experiment, bad_source, errored, expected, now):
    overrides = {"source": "case" if bad_source else "pilot", "error": "boom" if errored else None}
    if experiment:
        overrides["reference_example_id"] = "00000000-0000-4000-8000-000000000002"
    run = fake_run("p", **overrides)
    assert skip_reason(run) == expected
    result, _ = pulled([run], now)
    assert sum(result.by_reason.values()) + len(result.runs) == 1
    assert [s.reason for s in result.skipped] == ([] if expected is None else [expected])


def test_the_precedence_is_documented_in_the_module_docstring():
    text = " ".join((pull.__doc__ or "").split())
    assert "first that applies, in this fixed order" in text
    assert text.index("1. `experiment`") < text.index("2. `source`") < text.index("3. `errored`")


# --- child runs, E1, E3 ---------------------------------------------------------------------------------------------------------------------


def test_a_child_run_from_a_misbehaving_client_is_dropped_never_scored_or_counted_as_skipped(now):
    root = fake_run("root")
    child = fake_run("child", parent_run_id=root.id)
    result, _ = pulled([child, root], now)
    assert result.runs == [root] and result.skipped == [] and result.dropped_children == 1


def test_no_runs_in_the_window_is_a_result_not_an_error(now):
    result, client = pulled([], now)
    assert (result.runs, result.skipped, result.limit_hit, result.read, result.dropped_children) == ([], [], False, 0, 0)
    assert result.by_reason == Counter() and result.by_source == Counter()
    assert client.methods_called() == ["list_runs"]


def test_an_errored_run_never_reaches_the_scoring_list_and_nothing_is_posted_for_it(now):
    errored = fake_run("bad", error="the graph returned no final answer", answer=NO_ANSWER)
    client = OnlineRecordingClient([errored, fake_run("good")])
    result = pull_from_env(lambda: client, now, {})
    assert [run.id for run in result.runs] == [fake_run("good").id] and errored.id not in {run.id for run in result.runs}
    assert client.methods_called() == ["list_runs"]  # no create_feedback, no list_feedback, no read of the errored run


# --- the project ----------------------------------------------------------------------------------------------------------------------------


def test_only_the_guarded_project_name_is_ever_sent_to_list_runs(now):
    client = OnlineRecordingClient([fake_run("a")])
    pull_runs(client, "flysafelz", WEEK, 10, now)
    pull_runs(client, "  prod-traces  ", WEEK, 10, now)
    pull_from_env(lambda: client, now, {"LANGSMITH_PROJECT": " other "})
    pull_from_env(lambda: client, now, {})
    assert [call["project_name"] for call in client.list_run_calls()] == ["flysafelz", "prod-traces", "other", "flysafelz"]
    assert set(client.methods_called()) == {"list_runs"}


@pytest.mark.parametrize("split", ["held_out", "dev"])
def test_pull_runs_refuses_a_protected_project_before_touching_the_client(split, now, capsys):
    from evals.agent.loader import experiment_target

    name = experiment_target(split).project
    client = OnlineRecordingClient([fake_run("a")])
    with pytest.raises(SystemExit) as stop:
        pull_runs(client, f" {name.upper()} ", WEEK, 10, now)
    assert stop.value.code == 2 and name.upper() in capsys.readouterr().err
    assert client.calls == []


# --- the demo (`pytest -s -k twelve_runs` prints it) ------------------------------------------------------------------------------------------


def test_demo_twelve_runs_split_into_scorable_and_skipped(now):
    runs = {
        "pilot-1": fake_run("pilot-1"),
        "pilot-2": fake_run("pilot-2", path="explain"),
        "seed-1": fake_run("seed-1", source="online-seed", metadata={"lz_name": "Blue Mill"}),
        "seed-2": fake_run("seed-2", source="online-seed", path="decline"),
        "case-1": fake_run("case-1", source="case"),
        "judge-1": fake_run("judge-1", source="judge"),
        "v0-1": fake_run("v0-1", source="v0"),
        "studio-1": fake_run("studio-1", source="studio"),
        "no-source-1": fake_run("no-source-1", source=NO_SOURCE),
        "experiment-1": fake_run("experiment-1", reference_example_id="00000000-0000-4000-8000-000000000003"),
        "errored-1": fake_run("errored-1", error="the graph returned no final answer"),
        "no-answer-1": fake_run("no-answer-1", source="online-seed", answer=NO_ANSWER),
    }
    names = {run.id: name for name, run in runs.items()}
    result, client = pulled(list(runs.values()), now)
    print()
    print(f"read {result.read} root runs from {result.project} (since {result.since.days}d, limit {result.limit}, limit_hit={result.limit_hit})")
    print(f"list_runs called {len(client.list_run_calls())} time: {client.list_run_calls()[0]}")
    print(f"scorable ({len(result.runs)}): " + ", ".join(names[run.id] for run in result.runs))
    print(f"skipped ({len(result.skipped)}): " + ", ".join(f"{names[item.run.id]} [{item.reason}]" for item in result.skipped))
    print("by reason: " + ", ".join(f"{reason}={count}" for reason, count in sorted(result.by_reason.items())))
    print("by source (reason 'source'): " + ", ".join(f"{source}={count}" for source, count in sorted(result.by_source.items())))
    assert [names[run.id] for run in result.runs] == ["pilot-1", "pilot-2", "seed-1", "seed-2"]
    assert result.by_reason == {"source": 5, "experiment": 1, "errored": 2}
    assert result.by_source == {"case": 1, "judge": 1, "v0": 1, "studio": 1, "missing": 1}
    assert result.read == 12 and len(result.runs) + len(result.skipped) == 12
