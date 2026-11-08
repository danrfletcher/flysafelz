"""G20 (PR-12.F1): the LangSmith client and when it exists.

No key, no client: `langsmith.Client` is never constructed, and the graph has no tracer. With a key, the client has the anonymiser attached, and the project is
LANGSMITH_PROJECT or `flysafelz`. Nothing here is read at import. Sockets stay off: constructing a `Client` makes no request (`info=` is not needed to
construct one, and nothing is sent until a run is traced).
"""

import os
import subprocess
import sys
from pathlib import Path

import langsmith
import pytest
from graph_helpers import agent_graph
from langsmith.schemas import LangSmithInfo
from tracing_helpers import model_for, tracers_of

from flysafelz.agent import graph, tracing
from flysafelz.agent.config import langsmith_settings
from flysafelz.agent.settings import SETTINGS

ROOT = Path(__file__).resolve().parents[3]


class Constructed(AssertionError):
    pass


@pytest.fixture
def no_client_may_be_built(monkeypatch):
    """Any `langsmith.Client(...)` in the tracing module fails the test."""

    def refuse(*args, **kwargs):
        raise Constructed("a LangSmith client was built")

    monkeypatch.setattr(tracing, "Client", refuse)


class OfflineClient(langsmith.Client):
    """The real client, told what the server would say about itself (`info`) and not to start its batching thread: otherwise building one looks the server up, and the socket guard objects."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **{**kwargs, "info": LangSmithInfo(), "auto_batch_tracing": False})


@pytest.fixture(autouse=True)
def offline_clients(monkeypatch):
    monkeypatch.setattr(tracing, "_clients", {})
    monkeypatch.setattr(tracing, "Client", OfflineClient)


# --- no key: tracing is off ------------------------------------------------------------------------------------------


@pytest.mark.parametrize("environ", [{}, {"LANGSMITH_API_KEY": ""}, {"LANGSMITH_API_KEY": "   "}, {"LANGSMITH_TRACING": "true"}, {"LANGSMITH_PROJECT": "mine"}])
def test_without_a_key_build_client_returns_none_and_builds_nothing(no_client_may_be_built, environ):
    assert tracing.build_client(environ) is None


def test_without_a_key_the_environment_the_process_has_is_what_is_read(no_client_may_be_built, monkeypatch):
    monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)
    assert tracing.build_client() is None and tracing.shared_client() is None


def test_without_a_key_the_graph_has_no_tracer(no_client_may_be_built):
    compiled = agent_graph(model_for("x"))
    assert tracing.traced(compiled) is compiled  # the same object: nothing wrapped it


def test_without_a_key_build_graph_makes_no_client_and_has_no_tracer(no_client_may_be_built):
    built = graph.build_graph(model_for("x"))
    assert tracers_of(built) == []


def test_without_a_key_the_trace_helpers_find_no_run_and_do_nothing(no_client_may_be_built):
    tracing.tag_turn(None)
    tracing.report_outcome(tracing.GuardOutcome("passed", "assess"))
    tracing.wait_for_feedback()


# --- a key: a client with the anonymiser -----------------------------------------------------------------------------


def test_with_a_key_the_client_has_the_anonymiser_attached():
    client = tracing.build_client({"LANGSMITH_API_KEY": "test"})
    assert isinstance(client, langsmith.Client)
    assert client._anonymizer is tracing.ANONYMISER


def test_the_client_the_real_langsmith_builds_masks_a_phone_in_run_inputs():
    client = tracing.build_client({"LANGSMITH_API_KEY": "test"})
    assert client is not None
    assert client._hide_run_inputs({"text": "CTC 01632 960001 or ops@example.org", "lat": 53.802797}) == {"text": "CTC <PHONE> or <EMAIL>", "lat": 53.802797}


def test_with_a_key_the_endpoint_is_the_one_asked_for():
    client = tracing.build_client({"LANGSMITH_API_KEY": "test", "LANGSMITH_ENDPOINT": "https://eu.api.smith.langchain.com"})
    assert client is not None and client.api_url == "https://eu.api.smith.langchain.com"


def test_with_a_key_set_in_the_process_environment_build_graph_carries_the_tracer(monkeypatch):
    monkeypatch.setenv("LANGSMITH_API_KEY", "test")
    built = graph.build_graph(model_for("x"))
    tracers = tracers_of(built)
    assert len(tracers) == 1 and isinstance(tracers[0], tracing.FlySafeTracer) and tracers[0].client._anonymizer is tracing.ANONYMISER


def test_the_server_building_a_graph_for_every_run_shares_one_client(monkeypatch):
    monkeypatch.setenv("LANGSMITH_API_KEY", "test")
    assert tracing.shared_client() is tracing.shared_client()
    monkeypatch.setenv("LANGSMITH_API_KEY", "another")
    assert tracing.shared_client() is not None and len(tracing._clients) == 2


# --- the project -----------------------------------------------------------------------------------------------------


def test_the_project_is_flysafelz_by_default():
    assert tracing.project_name({}) == "flysafelz" == SETTINGS.langsmith_project


def test_the_project_is_the_override_when_one_is_set():
    assert tracing.project_name({"LANGSMITH_PROJECT": "my-project"}) == "my-project"


def test_a_blank_override_falls_back_to_the_default():
    assert tracing.project_name({"LANGSMITH_PROJECT": "  "}) == "flysafelz"


def test_the_tracer_the_graph_gets_names_that_project(monkeypatch):
    monkeypatch.setenv("LANGSMITH_API_KEY", "test")
    monkeypatch.setenv("LANGSMITH_PROJECT", "my-project")
    built = graph.build_graph(model_for("x"))
    assert [tracer.project_name for tracer in tracers_of(built)] == ["my-project"]
    monkeypatch.delenv("LANGSMITH_PROJECT")
    built = graph.build_graph(model_for("x"))
    assert [tracer.project_name for tracer in tracers_of(built)] == ["flysafelz"]


# --- the settings tracing reads from config.py --------------------------------------------------------------------------


def test_the_key_is_a_secret_that_never_shows_in_a_repr():
    settings = langsmith_settings({"LANGSMITH_API_KEY": "k-12345"})
    assert settings.api_key is not None and settings.api_key.get_secret_value() == "k-12345"
    assert "k-12345" not in repr(settings) and "k-12345" not in str(settings)


def test_a_blank_or_missing_key_is_no_key_and_the_other_settings_still_read():
    assert langsmith_settings({"LANGSMITH_API_KEY": " "}).api_key is None
    settings = langsmith_settings({"LANGSMITH_PROJECT": " p ", "LANGSMITH_ENDPOINT": " https://x.invalid "})
    assert (settings.api_key, settings.project, settings.endpoint) == (None, "p", "https://x.invalid")
    assert langsmith_settings({}).endpoint is None


# --- nothing at import ---------------------------------------------------------------------------------------------------

SPY = """
import os, sys
import langsmith, langchain_core.tracers.langchain  # their own reads of the environment are not ours

reads = []

class Spy(dict):
    def get(self, key, default=None):
        reads.append(key)
        return super().get(key, default)
    def __getitem__(self, key):
        reads.append(key)
        return super().__getitem__(key)
    def __contains__(self, key):
        reads.append(key)
        return super().__contains__(key)

os.environ = Spy(os.environ)
built = []
real = langsmith.Client

class Spied(real):
    def __init__(self, *a, **k):
        built.append(1)
        super().__init__(*a, **k)

langsmith.Client = Spied
import flysafelz.agent.tracing as tracing
assert not [r for r in reads if r.startswith(("LANGSMITH", "LANGCHAIN"))], reads
assert not built and not tracing._clients
print("clean")
"""


def test_importing_tracing_reads_no_langsmith_variable_and_builds_no_client():
    environ = {**os.environ, "LANGSMITH_API_KEY": "test", "LANGSMITH_PROJECT": "x", "LANGSMITH_TRACING": "true"}
    done = subprocess.run([sys.executable, "-c", SPY], cwd=ROOT, env=environ, capture_output=True, text=True, check=False)
    assert done.returncode == 0 and done.stdout.strip() == "clean", done.stderr


def test_f1_the_traced_graph_is_still_the_compiled_graph_with_the_same_nodes(monkeypatch):
    plain = agent_graph(model_for("x"))
    monkeypatch.setenv("LANGSMITH_API_KEY", "test")
    built = graph.build_graph(model_for("x"))
    assert type(built) is type(plain)
    assert sorted(built.nodes) == sorted(plain.nodes) == ["FlySafeMiddleware.after_agent", "FlySafeMiddleware.before_agent", "__start__", "model", "tools"]
