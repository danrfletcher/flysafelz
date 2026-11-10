"""The online project guard (7 PR-1.F1, G5, E7): the offline eval's held-out project and dev experiment project are never read. The names come from 5a's config, never from a literal here."""

import os

import pytest
from online_helpers import OnlineRecordingClient, fake_run

from evals.agent.loader import SPLITS, experiment_target
from evals.online.pull import guard_project, protected_projects, pull_from_env
from flysafelz.agent.settings import SETTINGS

HELD_OUT = experiment_target("held_out").project
DEV = experiment_target("dev").project


def test_5as_two_names_are_what_the_guard_protects():
    assert {HELD_OUT, DEV} == {SPLITS["held_out"].project, SPLITS["dev"].project} and HELD_OUT != DEV
    assert set(protected_projects()) == {HELD_OUT.casefold(), DEV.casefold()}


@pytest.mark.parametrize("name", [HELD_OUT, DEV], ids=["held-out", "dev-experiment"])
def test_each_protected_name_in_the_environment_exits_2_and_names_the_project(name, monkeypatch, capsys):
    monkeypatch.setenv("LANGSMITH_PROJECT", name)
    with pytest.raises(SystemExit) as stop:
        guard_project()
    assert stop.value.code == 2
    assert name in capsys.readouterr().err


@pytest.mark.parametrize("name", [HELD_OUT, DEV], ids=["held-out", "dev-experiment"])
def test_a_name_passed_in_is_refused_the_same_way(name, capsys):
    with pytest.raises(SystemExit) as stop:
        guard_project(name)
    assert stop.value.code == 2 and name in capsys.readouterr().err


@pytest.mark.parametrize("name", [HELD_OUT, DEV], ids=["held-out", "dev-experiment"])
def test_a_refusal_happens_before_the_client_is_built_or_called(name, now):
    built, client = [], OnlineRecordingClient([fake_run("a")])
    with pytest.raises(SystemExit) as stop:
        pull_from_env(lambda: built.append(1) or client, now, {"LANGSMITH_PROJECT": name})
    assert stop.value.code == 2
    assert built == [] and client.calls == []


@pytest.mark.parametrize("name", [HELD_OUT, DEV], ids=["held-out", "dev-experiment"])
@pytest.mark.parametrize(
    "shape", [str.upper, str.title, lambda n: f"  {n}", lambda n: f"{n}\t ", lambda n: f" {n.upper()} \n", lambda n: n.swapcase()], ids=["upper", "title", "lead", "trail", "both", "swapcase"]
)
def test_a_name_that_differs_only_in_case_or_surrounding_spaces_is_refused(name, shape):
    with pytest.raises(SystemExit) as stop:
        guard_project(shape(name))
    assert stop.value.code == 2


def test_the_message_names_the_project_as_it_was_given_less_the_spaces(capsys):
    with pytest.raises(SystemExit):
        guard_project(f"  {HELD_OUT.upper()}  ")
    assert f"project {HELD_OUT.upper()}:" in capsys.readouterr().err


def test_an_unset_project_falls_back_to_flysafelz(monkeypatch):
    monkeypatch.delenv("LANGSMITH_PROJECT", raising=False)
    assert guard_project() == "flysafelz" == SETTINGS.langsmith_project


@pytest.mark.parametrize("value", ["", "   "])
def test_a_blank_project_falls_back_to_the_default_too(value, monkeypatch):
    monkeypatch.setenv("LANGSMITH_PROJECT", value)
    assert guard_project() == "flysafelz"


@pytest.mark.parametrize("name", ["flysafelz", "prod-traces", f"{HELD_OUT}-notes", f"x{DEV}", DEV[:-1], "v1-dev-20261108-160700"])
def test_any_other_project_passes_and_comes_back_trimmed(name, monkeypatch):
    monkeypatch.setenv("LANGSMITH_PROJECT", f"  {name} ")
    assert guard_project() == name


def test_an_explicit_name_wins_over_the_environment(monkeypatch):
    monkeypatch.setenv("LANGSMITH_PROJECT", HELD_OUT)
    assert guard_project("prod-traces") == "prod-traces"


def test_the_test_environment_has_no_langsmith_key_or_project():
    """F4: the autouse isolated_environment strips every LANGSMITH_* variable, the key and the project included, so no test can reach a real project or client."""
    assert "LANGSMITH_API_KEY" not in os.environ and "LANGSMITH_PROJECT" not in os.environ
    assert os.environ["LANGSMITH_TRACING"] == "false"
