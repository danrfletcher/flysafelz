"""`make run` (G6, GP9): the entry module's `main()` with an injected fake model and CASE and NOW.

The environment is passed in as a dict, as `make` passes it. The answer goes to stdout once, the progress lines to stderr, and a bad command line is
one line on stderr and exit 2. The fake model is the only fake; the case is Blue Mill, so nothing is fetched.
"""

import io
import os
import subprocess
import sys
import warnings
from pathlib import Path

import pytest
from fakes import FakeToolChatModel
from graph_helpers import assess_call, call
from langchain_core.messages import AIMessage

from flysafelz.agent import graph as graph_module
from flysafelz.agent.run import main

ROOT = Path(__file__).resolve().parents[3]
BODY = "PILOT DISCRETION \u2013 APPROACHING LIMITS \u00b7 WITH CONDITIONS, flyable from 10:00 GMT to 16:00 GMT."
FINAL = f"[path: assess]\n{BODY}"
BLUE_MILL = {"LAT": "53.802797", "LON": "-0.086644", "Q": "Can I fly here between 10 and 4 today?", "CASE": "blue_mill_2026-10-31", "NOW": "2026-10-31T07:30:00+00:00"}


@pytest.fixture(autouse=True)
def fresh_shared_parts(monkeypatch):
    monkeypatch.setattr(graph_module, "_shared", {})


def run(environ, model=None, argv=None):
    out, err = io.StringIO(), io.StringIO()
    code = main(argv, environ=environ, model=model, stdout=out, stderr=err)
    return code, out.getvalue(), err.getvalue()


def scripted():
    return FakeToolChatModel(script=[AIMessage("I will check first.", tool_calls=assess_call().tool_calls), AIMessage(FINAL)])


def test_the_answer_goes_to_stdout_once_whole_and_the_progress_lines_to_stderr():
    code, out, err = run(BLUE_MILL, scripted())
    assert code == 0
    assert out.startswith(BODY) and "[path" not in out and out.endswith("\n")
    assert "Assessing…" in err.splitlines()
    assert BODY not in err and "Assessing…" not in out


def test_nothing_the_model_says_between_tool_calls_is_printed():
    _, out, err = run(BLUE_MILL, scripted())
    assert "I will check first." not in out + err


def test_the_optional_time_zone_reaches_the_prompt():
    model = FakeToolChatModel(script=[AIMessage("[path: decline]\nok.")])
    code, out, _ = run({**BLUE_MILL, "TZ": "America/New_York"}, model)
    assert code == 0 and out == "ok.\n"
    assert "The pilot's time zone is America/New_York." in str(model.calls[0][0].content)


def test_a_question_with_quotes_and_dollar_signs_arrives_as_typed():
    model = FakeToolChatModel(script=[AIMessage("[path: decline]\nok.")])
    question = "Can I fly at \"2\" and 'later' $HOME `date`?"
    run({**BLUE_MILL, "Q": question}, model)
    assert model.calls[0][1].content == question


@pytest.mark.parametrize("tool", ["assess", "get_weather", "get_airspace", "get_notams", "search_guidance"])
def test_a_run_with_no_clock_set_prints_no_warning_when_a_tool_is_called(tool):
    """The tools' `runtime` argument is typed with the RunContext, so pydantic does not warn as it copies the call's arguments (T1)."""
    args = {"query": "what is a danger area"} if tool == "search_guidance" else {}
    environ = {k: v for k, v in BLUE_MILL.items() if k != "NOW"}
    model = FakeToolChatModel(script=[call(tool, **args), AIMessage("[path: explain]\nChecked.")])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        code, out, _ = run(environ, model)
    assert code == 0 and out.startswith("Checked.")
    assert [str(w.message) for w in caught] == []


@pytest.mark.parametrize("missing", ["LAT", "LON", "Q"])
def test_a_missing_argument_is_one_line_on_stderr_and_exit_2(missing):
    environ = {k: v for k, v in BLUE_MILL.items() if k != missing}
    model = FakeToolChatModel(script=[])
    code, out, err = run(environ, model)
    assert code == 2 and out == ""
    assert len(err.splitlines()) == 1 and missing in err and "make run" in err
    assert model.calls == []


def test_all_three_missing_are_named_together():
    code, _, err = run({}, FakeToolChatModel(script=[]))
    assert code == 2 and all(name in err.split(".")[0] for name in ("LAT", "LON", "Q"))


@pytest.mark.parametrize("environ", [{"LAT": "  ", "LON": "1", "Q": "x"}, {"LAT": "1", "LON": "1", "Q": ""}])
def test_a_blank_argument_counts_as_missing(environ):
    assert run(environ, FakeToolChatModel(script=[]))[0] == 2


@pytest.mark.parametrize("bad", ["north", "nan", "inf", "1,5", ""])
def test_a_latitude_that_is_not_a_number_is_exit_2_with_one_line(bad):
    code, out, err = run({**BLUE_MILL, "LAT": bad}, FakeToolChatModel(script=[]))
    assert code == 2 and out == "" and len(err.splitlines()) == 1


@pytest.mark.parametrize(
    "override",
    [{"TZ": "Mars/Olympus"}, {"CASE": "no_such_case"}, {"NOW": "yesterday"}, {"NOW": "2026-10-31T07:30:00"}, {"LAT": "91"}],
)
def test_a_value_the_input_check_refuses_is_exit_2_with_one_line_and_no_model_call(override):
    model = FakeToolChatModel(script=[])
    code, out, err = run({**BLUE_MILL, **override}, model)
    assert code == 2 and out == "" and len(err.splitlines()) == 1
    assert model.calls == []


def test_arguments_on_the_command_line_are_refused_the_values_come_from_the_environment():
    code, _, err = run(BLUE_MILL, FakeToolChatModel(script=[]), argv=["LAT=1"])
    assert code == 2 and "only the environment" in err


def test_outside_the_uk_prints_the_uk_only_answer_and_calls_no_model():
    model = FakeToolChatModel(script=[])
    code, out, err = run({"LAT": "48.85", "LON": "2.35", "Q": "Can I fly here tomorrow?"}, model)
    assert (code, out) == (0, "FlySafeLZ only covers the UK.\n")
    assert model.calls == []
    assert "Assessing…" not in err


def test_outside_the_uk_builds_no_model_when_none_is_injected_and_no_agent_setting_is_set():
    assert not [name for name in os.environ if name.startswith("AGENT_")]
    code, out, _ = run({"LAT": "48.85", "LON": "2.35", "Q": "Can I fly here tomorrow?"})
    assert (code, out) == (0, "FlySafeLZ only covers the UK.\n")


def test_inside_the_uk_with_no_agent_setting_the_missing_variables_are_named_on_one_line_and_exit_1():
    code, out, err = run({**BLUE_MILL})
    assert code == 1 and out == ""
    assert len(err.splitlines()) == 1 and all(name in err for name in ("AGENT_BASE_URL", "AGENT_MODEL", "AGENT_API_KEY"))


def test_a_model_that_fails_still_prints_the_unavailable_answer_and_exits_0():
    import httpx
    from graph_helpers import RaisingModel

    code, out, _ = run(BLUE_MILL, RaisingModel(error=httpx.ReadTimeout("timed out")))
    assert code == 0
    assert "UNVERIFIED" in out and "NATS AIS pre-flight briefing" in out


def test_a_run_that_ends_with_no_answer_is_exit_1(monkeypatch):
    from flysafelz.agent import run as run_module

    monkeypatch.setattr(run_module, "ask", lambda *args: None)
    code, out, err = run(BLUE_MILL, FakeToolChatModel(script=[]))
    assert code == 1 and out == "" and "without an answer" in err


def test_the_module_runs_as_a_program_and_a_missing_argument_is_exit_2():
    """The `__main__` path `make run` takes."""
    clean = {k: v for k, v in os.environ.items() if k not in {"LAT", "LON", "Q"} and not k.startswith("AGENT_")}
    done = subprocess.run([sys.executable, "-m", "flysafelz.agent.run"], env=clean, capture_output=True, text=True, cwd=ROOT, check=False)
    assert done.returncode == 2 and done.stdout == ""
    assert len(done.stderr.splitlines()) == 1 and "LAT, LON, Q" in done.stderr


def test_the_program_outside_the_uk_prints_the_uk_only_answer_with_no_agent_setting():
    clean = {k: v for k, v in os.environ.items() if not k.startswith("AGENT_")}
    env = {**clean, "LAT": "48.85", "LON": "2.35", "Q": "Can I fly here?"}
    done = subprocess.run([sys.executable, "-m", "flysafelz.agent.run"], env=env, capture_output=True, text=True, cwd=ROOT, check=False)
    assert done.returncode == 0 and done.stdout == "FlySafeLZ only covers the UK.\n"
