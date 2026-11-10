"""The judge's sample rule (7 PR-3.F1, G8, E8, E10): `select_for_judge`, its boundary, `JUDGE_SAMPLE_PCT`, and the environment's part of the judge's set-up.

The expected split is worked out in this file with hashlib, from the rule in the spec, not read back from the code.
"""

import ast
import hashlib
import inspect
import random
import textwrap
import uuid

import pytest

from evals.online import job

NAMESPACE = uuid.UUID("0b5e7c1e-4a53-4c4e-9f0e-6f0d9d5a0001")
PINNED_ID = uuid.UUID("11111111-1111-4111-8111-111111111111")
PINNED_BUCKET = 77  # int(sha256("11111111-1111-4111-8111-111111111111").hexdigest(), 16) % 100


def ids(n: int) -> list[uuid.UUID]:
    return [uuid.uuid5(NAMESPACE, f"run-{index}") for index in range(n)]


def bucket(run_id: object) -> int:
    return int(hashlib.sha256(str(run_id).encode()).hexdigest(), 16) % 100


def expected_split(run_ids: list[uuid.UUID], pct: int) -> tuple[list[uuid.UUID], list[uuid.UUID]]:
    """The spec's rule: 50 or fewer are all judged; more are judged when the hash bucket is below the percentage."""
    if len(run_ids) <= 50:
        return list(run_ids), []
    return [i for i in run_ids if bucket(i) < pct], [i for i in run_ids if bucket(i) >= pct]


COUNTS = [0, 1, 50, 51]
PERCENTAGES = [0, 20, 100]


@pytest.mark.parametrize("pct", PERCENTAGES)
@pytest.mark.parametrize("count", COUNTS)
def test_the_split_follows_the_rule_at_every_count_and_percentage(count, pct):
    run_ids = ids(count)
    assert job.select_for_judge(run_ids, pct) == expected_split(run_ids, pct)


@pytest.mark.parametrize("pct", PERCENTAGES)
def test_with_50_or_fewer_every_run_is_judged_whatever_the_percentage(pct):
    for count in (0, 1, 49, 50):
        judged, left_out = job.select_for_judge(ids(count), pct)
        assert (len(judged), left_out) == (count, [])


def test_at_51_the_sample_applies_and_at_0_percent_nothing_is_judged_at_100_everything_is():
    run_ids = ids(51)
    assert job.select_for_judge(run_ids, 0) == ([], run_ids)
    assert job.select_for_judge(run_ids, 100) == (run_ids, [])
    judged, left_out = job.select_for_judge(run_ids, 20)
    assert 0 < len(judged) < 51 and len(judged) + len(left_out) == 51


def test_a_known_run_id_has_a_known_bucket_and_is_judged_exactly_above_it():
    assert job.run_bucket(PINNED_ID) == PINNED_BUCKET == bucket(PINNED_ID)
    others = ids(50)
    assert PINNED_ID not in others
    for pct, picked in ((PINNED_BUCKET, False), (PINNED_BUCKET + 1, True)):
        judged, left_out = job.select_for_judge([PINNED_ID, *others], pct)
        assert (PINNED_ID in judged, PINNED_ID in left_out) == (picked, not picked)


def test_the_bucket_hashes_the_text_of_the_run_id_so_a_string_and_a_uuid_agree():
    assert job.run_bucket(str(PINNED_ID)) == job.run_bucket(PINNED_ID)


def test_the_same_input_twice_gives_the_same_split():
    run_ids = ids(120)
    assert job.select_for_judge(run_ids, 20) == job.select_for_judge(list(run_ids), 20)


@pytest.mark.parametrize("seed", range(5))
def test_the_order_of_the_run_ids_does_not_change_the_split(seed):
    run_ids = ids(120)
    shuffled = list(run_ids)
    random.Random(seed).shuffle(shuffled)
    judged, left_out = job.select_for_judge(run_ids, 20)
    shuffled_judged, shuffled_left_out = job.select_for_judge(shuffled, 20)
    assert (set(judged), set(left_out)) == (set(shuffled_judged), set(shuffled_left_out))
    assert shuffled_judged == [i for i in shuffled if i in set(judged)]  # each side keeps the order it was given in


def test_120_runs_at_the_default_percentage_are_split_as_the_hash_rule_says():
    run_ids = ids(120)
    judged, left_out = job.select_for_judge(run_ids, job.DEFAULT_SAMPLE_PCT)
    assert job.DEFAULT_SAMPLE_PCT == 20
    assert (judged, left_out) == expected_split(run_ids, 20)
    assert len(judged) == sum(1 for i in run_ids if bucket(i) < 20) and 0 < len(judged) < 120


def names_used(function) -> set[str]:
    """Every name and attribute the function's code uses; its docstring is not code."""
    tree = ast.parse(textwrap.dedent(inspect.getsource(function)))
    return {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)} | {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)}


def test_the_selection_takes_no_clock_and_no_randomness():
    assert list(inspect.signature(job.select_for_judge).parameters) == ["run_ids", "pct"]
    used = names_used(job.select_for_judge) | names_used(job.run_bucket)
    assert {"hashlib", "sha256"} <= used
    assert not used & {"random", "time", "datetime", "now", "uuid4", "choice", "sample", "shuffle", "environ", "getenv"}


@pytest.mark.parametrize("pct", [-1, 101, True, "20", 20.0, None])
def test_a_percentage_that_is_not_a_whole_number_from_0_to_100_is_refused(pct):
    with pytest.raises(ValueError, match="percentage"):
        job.select_for_judge(ids(3), pct)


# --- JUDGE_SAMPLE_PCT (the environment's part) ---------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(("text", "pct"), [("0", 0), ("20", 20), ("100", 100), (" 35 ", 35), ("007", 7)])
def test_a_valid_percentage_is_read(text, pct):
    assert job.parse_sample_pct(text) == pct


@pytest.mark.parametrize("text", ["-1", "101", "abc", "", "  ", "2.5", "20%", "٢٠", "1e1", "+5"])
def test_a_bad_percentage_exits_2_and_names_the_variable(text, capsys):
    with pytest.raises(SystemExit) as stopped:
        job.parse_sample_pct(text)
    assert stopped.value.code == 2
    assert "JUDGE_SAMPLE_PCT" in capsys.readouterr().err


NO_DOTENV = {"READER_DOTENV_PATH": "/no-such-dotenv/.env"}  # the judge settings are read from the environment, then from a .env file: never a real one
JUDGE_ENV = {**NO_DOTENV, "JUDGE_PROVIDER": "fake", "JUDGE_BASE_URL": "http://judge.invalid/v1", "JUDGE_API_" + "KEY": "not-a-real-value", "JUDGE_MODEL": "fake-model"}


def never_built(environ):
    raise AssertionError("the judge must not be built")


def test_the_percentage_defaults_to_20_when_unset():
    built = []
    setup = job.judge_setup_from_env(JUDGE_ENV, build=lambda environ: built.append(environ) or "the judge")
    assert (setup.judge, setup.pct, setup.missing) == ("the judge", 20, ()) and built == [JUDGE_ENV]


def test_a_set_percentage_is_used():
    assert job.judge_setup_from_env({**JUDGE_ENV, "JUDGE_SAMPLE_PCT": "100"}, build=lambda environ: "the judge").pct == 100


@pytest.mark.parametrize("text", ["-1", "101", "abc", ""])
def test_a_bad_percentage_in_the_environment_exits_2_before_the_judge_is_built(text, capsys):
    with pytest.raises(SystemExit) as stopped:
        job.judge_setup_from_env({**JUDGE_ENV, "JUDGE_SAMPLE_PCT": text}, build=never_built)
    assert stopped.value.code == 2 and "JUDGE_SAMPLE_PCT" in capsys.readouterr().err


def test_a_bad_percentage_is_refused_even_when_the_judge_variables_are_unset(capsys):
    with pytest.raises(SystemExit) as stopped:
        job.judge_setup_from_env({**NO_DOTENV, "JUDGE_SAMPLE_PCT": "101"}, build=never_built)
    assert stopped.value.code == 2 and "JUDGE_SAMPLE_PCT" in capsys.readouterr().err


@pytest.mark.parametrize("unset", [name for name in sorted(JUDGE_ENV) if name.startswith("JUDGE_")])
def test_with_any_judge_variable_unset_the_names_are_returned_and_nothing_is_built(unset):
    environ = {name: value for name, value in JUDGE_ENV.items() if name != unset}
    setup = job.judge_setup_from_env(environ, build=never_built)
    assert (setup.judge, setup.missing, setup.pct) == (None, (unset,), 20)


def test_with_none_set_all_four_names_are_returned_in_5as_order_and_a_blank_one_counts_as_unset():
    assert job.judge_setup_from_env(NO_DOTENV, build=never_built).missing == ("JUDGE_PROVIDER", "JUDGE_BASE_URL", "JUDGE_API_" + "KEY", "JUDGE_MODEL")
    assert job.judge_setup_from_env({**JUDGE_ENV, "JUDGE_MODEL": "  "}, build=never_built).missing == ("JUDGE_MODEL",)


def test_the_process_environment_is_read_when_none_is_passed(monkeypatch):
    monkeypatch.setenv("JUDGE_SAMPLE_PCT", "55")
    setup = job.judge_setup_from_env(build=never_built)
    assert setup.pct == 55 and setup.judge is None and len(setup.missing) == 4
