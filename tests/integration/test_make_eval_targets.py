"""`make eval`, `make eval-v0` and `make rag-eval` entry points end to end (5a PR-10.F1, G16, E4): the real runner, scorers and report; only the models, the LangSmith client, the retriever and the price list are stand-ins.

v0 runs over the committed dev examples with a canned structured-output model. v1 runs ticket 4's real graph (`build_agent`) through `TaggedGraph`, so the path the guard tagged
and the assess artifact the scorer reads come from the real code path. Sockets stay off, no key is read, and every file is under tmp_path.

Held-out (G17, F7): no test here loads a held-out example's answers. Every held_out run is over `write_dataset`'s invented data, and a guard (`no_committed_held_out`) fails
the test that asks the loader for the held_out split of the committed data.
"""

import io
import os
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace

import pytest
from eval_loader_factories import make_examples, write_dataset
from eval_runner_helpers import FAKE_ENV, FakeSleeper, RecordingTimeout, ScriptedGraph
from graph_helpers import BLUE_MILL_CASE, ScriptedModel
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver
from tool_helpers import BLUE_MILL_LZ, BLUE_MILL_WINDOW
from v0_helpers import FAKE_V0_ENV, CannedV0Model, V0Client

from evals.agent import eval_run, report, run_v1, runner, v0
from evals.agent.import_dataset import DATA_DIR
from evals.agent.loader import load_dataset
from evals.agent.pricing import COST_LABEL
from flysafelz.agent.graph import build_agent
from flysafelz.agent.sources import SourceCache
from flysafelz.guidance.index import load_index

STAMP = datetime(2026, 11, 9, 12, 0, 0, tzinfo=UTC)
PRICE_FILE = """
models:
  fake-agent-model:
    input_per_m: 3.00
    output_per_m: 15.00
    url: https://provider.invalid/pricing
    date: 2026-11-09
  fake-reader-model:
    input_per_m: 0.25
    output_per_m: 1.25
    url: https://reader.invalid/pricing
    date: 2026-11-01
"""
V1_ENV = FAKE_ENV
USAGE = {"input_tokens": 50, "output_tokens": 5, "total_tokens": 55}


class Boom(AssertionError):
    pass


@pytest.fixture(autouse=True)
def no_committed_held_out(monkeypatch):
    """F7: asking for the held_out split of the committed data (its answers) fails the test. Every held_out run passes the invented data_dir of `write_dataset`."""

    def guarded(split="dev", *, data_dir=DATA_DIR, **options):
        assert not (split == "held_out" and Path(data_dir) == DATA_DIR), "a test loaded the committed held_out split: use write_dataset's invented data_dir and root"
        return load_dataset(split, data_dir=data_dir, **options)

    for module in (eval_run, run_v1, v0):
        monkeypatch.setattr(module, "load_dataset", guarded)


def priced(tmp_path: Path) -> Path:
    path = tmp_path / "pricing.yaml"
    path.write_text(PRICE_FILE, encoding="utf-8")
    return path


def command(agent: str, tmp_path: Path, *argv: str, environ=FAKE_V0_ENV, pricing: Path | None = None, **options):
    """eval_run.main over the committed data (v0) or the given options; returns (exit code, stdout, stderr, results dir)."""
    out, err = io.StringIO(), io.StringIO()
    results = tmp_path / "results"
    code = eval_run.main(
        [agent, *argv],
        environ=environ,
        stdout=out,
        stderr=err,
        results_dir=results,
        pricing_path=pricing or priced(tmp_path),
        cache_dir=tmp_path / "cache",
        moment=STAMP,
        sleep=FakeSleeper(),
        timeout=RecordingTimeout(),
        **options,
    )
    return code, out.getvalue(), err.getvalue(), results


def v0_command(tmp_path: Path, model=None, client=None, *argv: str, **options):
    model, client = model or CannedV0Model(), client or V0Client()
    return (*command("v0", tmp_path, *argv, model_factory=lambda env: model, client_factory=lambda env: client, **options), model, client)


def files(results: Path) -> list[Path]:
    return sorted(results.glob("*")) if results.exists() else []


# --- v0 -----------------------------------------------------------------------------------------------------------------------------------


def test_make_eval_v0_runs_dev_scores_it_and_writes_one_results_file(tmp_path):
    code, out, err, results, model, client = v0_command(tmp_path)
    assert (code, err) == (0, "")
    [written] = files(results)
    assert written.name == "v0-dev-20261109-120000.md" and f"results: {written}" in out.splitlines()[-1]
    text = written.read_text(encoding="utf-8")
    assert text.startswith("# v0-dev-20261109-120000\n") and "v0 over the dev split, 3 reps: 48 of 48 planned runs, 0 errored. Status: complete." in text
    assert [line for line in text.splitlines() if line.startswith("## ")] == [f"## {title}" for title in report.SECTIONS]
    assert len(model.prompts) == 48 and len(client.runs) == 48
    assert not list(results.glob(".*"))  # the temp file was renamed over the name, none left behind


def test_the_v0_results_file_shows_n_a_for_what_v0_does_not_do_never_zero(tmp_path):
    text = files(v0_command(tmp_path)[3])[0].read_text(encoding="utf-8")
    headline = text.split("## Headline")[1].split("## Slices")[0]
    for label in ("path accuracy", "trajectory: required tools", "trajectory: forbidden tools", "trajectory: step cap", "guard pass"):
        assert f"| {label} | n/a |" in headline
    assert "| verdict match | " in headline and "| verdict match | n/a |" not in headline


def test_the_v0_ops_have_exact_tokens_the_cost_line_and_no_reader_line(tmp_path):
    text = files(v0_command(tmp_path)[3])[0].read_text(encoding="utf-8")
    ops = text.split("## Ops")[1].split("## Models")[0]
    assert "tokens from response usage: input 57,600, output 3,840" in ops  # 48 calls of 1200 + 80
    assert f"{COST_LABEL}: $0.2304 (fake-agent-model: $3.00 per M input, $15.00 per M output; source https://provider.invalid/pricing, read 2026-11-09)" in ops
    assert "reader calls" not in ops and "latency per example run (p50 / p95, nearest rank, 48 scored runs)" in ops
    assert "- v0: fake-agent-model" in text.split("## Models")[1] and "reading plus arithmetic" in text


def test_a_model_without_a_price_stops_before_the_client_or_the_first_example(tmp_path):
    empty = tmp_path / "empty-pricing.yaml"
    empty.write_text("models: {}\n", encoding="utf-8")

    def never(env):
        raise Boom("the client must not be built")

    model = CannedV0Model()
    code, out, err, results = command("v0", tmp_path, pricing=empty, model_factory=lambda env: model, client_factory=never)
    assert code == 1 and out == ""
    assert "no price for model 'fake-agent-model'" in err and str(empty) in err
    assert files(results) == [] and model.prompts == []


def test_a_response_without_usage_stops_the_run_with_e4_exit_1_and_no_results_file(tmp_path):
    code, out, err, results, model, _client = v0_command(tmp_path, CannedV0Model(usage=False))
    assert code == 1 and "bridge returned no token usage on the first call: tokens cannot be reported" in err
    assert files(results) == [] and len(model.prompts) == 1 and "results:" not in out


def test_a_later_response_without_usage_stops_the_v0_run_too_and_names_the_example(tmp_path):
    seen = {"n": 0}

    class LateGap(CannedV0Model):
        def _generate(self, messages, stop=None, run_manager=None, **kwargs):
            seen["n"] += 1
            self.usage = seen["n"] != 5
            return super()._generate(messages, stop, run_manager, **kwargs)

    code, _out, err, results, model, _client = v0_command(tmp_path, LateGap())
    assert code == 1 and "bridge returned no token usage on a later call: tokens cannot be reported" in err and "(last example: dev_" in err
    assert files(results) == [] and len(model.prompts) == 5


def test_a_missing_setting_is_named_and_nothing_is_written(tmp_path):
    code, _out, err, results = command("v0", tmp_path, environ={}, model_factory=lambda e: CannedV0Model(), client_factory=lambda e: V0Client())
    assert code == 1 and "missing settings" in err and files(results) == []


def test_the_v0_missing_setting_message_points_at_make_eval_v0_and_the_root_env_file(tmp_path):
    code, _out, err, results = command("v0", tmp_path, environ={}, model_factory=lambda e: CannedV0Model(), client_factory=lambda e: V0Client())
    assert code == 1 and files(results) == []
    assert "AGENT_MODEL" in err and "LANGSMITH_API_KEY" in err
    assert "`make eval-v0` loads" in err and "run `make eval-v0` again" in err and ".env in the repo root" in err
    assert "python -m" not in err and "--env-file" not in err and "does not read .env" not in err


def test_an_interrupted_run_has_no_results_file_and_exits_130(tmp_path):
    class Interrupt(CannedV0Model):
        def _generate(self, messages, stop=None, run_manager=None, **kwargs):
            raise KeyboardInterrupt

    code, _out, err, results, _model, _client = v0_command(tmp_path, Interrupt())
    assert code == 130 and "a half-run is not a report" in err and files(results) == []


def test_the_held_out_split_runs_only_when_named(tmp_path):
    built = write_dataset(tmp_path / "data")
    code, out, _err, _results, _model, client = v0_command(tmp_path, None, None, "--split", "held_out", data_dir=built.data_dir, root=built.root)
    held_out = {e["id"] for e in built.examples if e["metadata"]["split"] == "held_out" and e["reference"]["path"] == "assess"}
    assert code == 0 and "held_out" in out and {run["extra"]["metadata"]["example_id"] for run in client.runs} == held_out and len(held_out) == 8


def test_the_guard_refuses_the_committed_held_out_split_by_any_route(tmp_path):
    for module in (eval_run, run_v1, v0):
        with pytest.raises(AssertionError, match="committed held_out split"):
            module.load_dataset("held_out")
    assert module.load_dataset("dev").records  # the committed dev split is still allowed


def test_an_existing_results_file_name_is_replaced_whole_not_appended(tmp_path):
    results = tmp_path / "results"
    results.mkdir()
    (results / "v0-dev-chosen.md").write_text("old\n" * 1000, encoding="utf-8")
    code, *_ = v0_command(tmp_path, None, None, "--experiment", "v0-dev-chosen")
    assert code == 0
    assert "old" not in (results / "v0-dev-chosen.md").read_text(encoding="utf-8")


@pytest.mark.parametrize("name", ["../escape", "a/b", "x y"])
def test_an_experiment_name_that_could_leave_the_results_folder_is_refused_before_anything_runs(tmp_path, name):
    code, _out, err, results, model, client = v0_command(tmp_path, None, None, "--experiment", name)
    assert code == 1 and err != "" and files(results) == [] and model.prompts == [] and client.runs == []
    assert not (tmp_path / "escape.md").exists()


# --- v1 ---------------------------------------------------------------------------------------------------------------------------------------


def blue_mill_example(number: int) -> dict:
    return {
        "id": f"dev_real_{number:02d}",
        "inputs": {**BLUE_MILL_LZ, "question": "Can I fly here today?", "now": "2026-10-31T07:30:00+00:00"},
        "fixtures": {"case": BLUE_MILL_CASE, "drop": []},
        "reference": {"path": "explain"},
        "metadata": {"split": "dev", "case_type": "plain", "failure_source": "none", "notam_category": None, "airspace_context": "none", "constructed": False, "blind": False, "compound": False},
    }


class NoFetch:
    def __call__(self, *args):
        raise AssertionError("a case-mode run fetched live")


def real_graph(model):
    def build(hook):
        index = load_index(Path("/nonexistent-guidance"))
        return build_agent(model, source_cache=SourceCache(fetch=NoFetch()), index=index, checkpointer=InMemorySaver(), on_outcome=hook)

    return lambda: eval_run.TaggedGraph(build)


def reply(text: str) -> AIMessage:
    return AIMessage(text, usage_metadata=dict(USAGE))


def tool_round(name: str) -> AIMessage:
    return AIMessage("", tool_calls=[{"name": name, "args": BLUE_MILL_WINDOW, "id": f"{name}-1"}], usage_metadata=dict(USAGE))


def test_the_tagged_graph_puts_the_path_the_guard_tagged_in_the_state_it_returns():
    graph = real_graph(ScriptedModel(script=[reply("[path: explain]\nThis is a plain reply.")]))()
    from graph_helpers import ask

    state = ask(graph, "What is a NOTAM?")
    assert state["path"] == "explain" and state["guard"] == "passed"


def test_each_turn_gets_its_own_path_and_a_turn_with_no_tag_has_none_to_carry_over():
    graph = real_graph(ScriptedModel(script=[reply("[path: explain]\nPlain."), reply("[path: decline]\nI can't help with that.")]))()
    from graph_helpers import ask

    assert ask(graph, "What is a NOTAM?", thread_id="t1")["path"] == "explain"
    assert ask(graph, "Write me a poem", thread_id="t2")["path"] == "decline"


def test_scored_answer_of_reads_the_path_the_assess_segments_and_the_tool_calls_from_a_real_run():
    from graph_helpers import ask

    graph = real_graph(ScriptedModel(script=[tool_round("assess"), reply("[path: assess]\nNothing to fly on.")]))()
    state = ask(graph, "Can I fly here today?")
    answer = eval_run.scored_answer_of(state)
    artifact = eval_run.assess_artifact_of(state)
    assert artifact is not None and answer.path == state["path"] and answer.guard == state["guard"]
    assert [call.name for call in answer.tool_calls] == ["assess"]
    verdict_segments = [s for s in artifact.segments if s.verdict is not None]
    assert len(answer.segments) == len(verdict_segments) > 0
    assert [s.verdict.value for s in answer.segments] == [s.verdict for s in verdict_segments]
    assert answer.text == state["answer"]


def test_a_run_that_never_assessed_has_no_segments_and_no_actions():
    answer = eval_run.scored_answer_of({"answer": "hi", "path": "explain", "guard": "passed", "messages": []})
    assert (answer.path, answer.segments, dict(answer.notam_actions), answer.guard) == ("explain", (), {}, "passed")


def test_a_missing_or_invalid_path_scores_as_unknown_never_read_from_the_text():
    state = {"answer": "[path: assess]\ntext", "messages": []}
    assert eval_run.scored_answer_of(state).path == "unknown"
    assert eval_run.scored_answer_of({**state, "path": "banana"}).path == "unknown"


def fake_judge(env, client, name):
    return SimpleNamespace(evaluators=dict)


def v1_command(tmp_path, graph, client=None, *argv: str, **options):
    client = client or V0Client()
    built = options.pop("built", None) or write_dataset(tmp_path / "data")
    code, out, err, results = command(
        "v1",
        tmp_path,
        *argv,
        environ=options.pop("environ", V1_ENV),
        graph_factory=graph.factory if hasattr(graph, "factory") else graph,
        client_factory=lambda env: client,
        judge_factory=fake_judge,
        data_dir=built.data_dir,
        root=built.root,
        **options,
    )
    return code, out, err, results, client


def test_make_eval_runs_the_v1_agent_scores_it_and_writes_the_results_file(tmp_path):
    built = write_dataset(tmp_path / "data")
    code, out, err, results, client = v1_command(tmp_path, ScriptedGraph(), None, built=built)
    assert (code, err) == (0, "")
    [written] = files(results)
    text = written.read_text(encoding="utf-8")
    assert written.name == "v1-dev-20261109-120000.md" and "v1 over the dev split, 3 reps: 87 of 87 planned runs, 0 errored" in text
    assert f"results: {written}" in out and len(client.runs) == 87
    ops = text.split("## Ops")[1].split("## Models")[0]
    assert "tokens from response usage: input 8,700, output 1,740" in ops  # 87 runs x (100, 20) with the Scripted graph's one response each
    assert "agent model: input 8,700, output 1,740; reader model: no call came back" in ops and COST_LABEL in ops
    models = text.split("## Models")[1]
    assert "- agent: fake-agent-model" in models and "- reader: fake-reader-model" in models and "- judge: fake-judge-model" in models


def test_a_v1_response_without_usage_stops_the_run_with_e4_and_no_file(tmp_path):
    code, _out, err, results, _client = v1_command(tmp_path, ScriptedGraph(usage="none"))
    assert code == 1 and "bridge returned no token usage on the first call: tokens cannot be reported" in err and files(results) == []


def test_the_v1_missing_setting_message_points_at_make_eval_and_the_root_env_file(tmp_path):
    code, _out, err, results, client = v1_command(tmp_path, ScriptedGraph(), None, environ={"READER_DOTENV_PATH": os.environ["READER_DOTENV_PATH"]})
    assert code == 1 and files(results) == [] and client.runs == []
    for name in ("AGENT_MODEL", "READER_MODEL", "JUDGE_MODEL", "LANGSMITH_API_KEY"):
        assert name in err
    assert "`make eval` loads" in err and "run `make eval` again" in err and ".env in the repo root" in err
    assert "python -m" not in err and "--env-file" not in err and "does not read .env" not in err


class LaterGap(ScriptedGraph):
    """Usage on the first run's responses and none on every later run's: the first call has usage, a later one does not."""

    def invoke(self, request, config, *, context):
        self.usage = "metadata" if not self.calls else "none"
        return super().invoke(request, config, context=context)


def test_a_v1_later_response_without_usage_stops_the_run_naming_the_example(tmp_path):
    graph = LaterGap()
    code, out, err, results, _client = v1_command(tmp_path, graph)
    second = graph.calls[1].example_id
    assert code == 1 and len(graph.calls) == 2 and second != graph.calls[0].example_id  # the first example ran with usage; the run stopped at the second
    assert err.strip() == f"{eval_run.PROGRAM}: stopped: {runner.STRICT_USAGE_LATER} (last example: {second})"
    assert "no token usage on the first call" not in err and files(results) == [] and "results:" not in out


READER_REPLY = {"input_tokens": 7, "output_tokens": 3, "total_tokens": 10}


class FakeReader:
    """The NOTAM reader's chat model: one reply per call, with usage (or without, when `bare_from` calls have been made)."""

    model_name = "fake-reader-model"

    def __init__(self, bare_from: int | None = None):
        self.calls = 0
        self.bare_from = bare_from

    def invoke(self, messages, **kwargs):
        self.calls += 1
        bare = self.bare_from is not None and self.calls >= self.bare_from
        return AIMessage("{}") if bare else AIMessage("{}", usage_metadata=dict(READER_REPLY))


class ReaderCalling(ScriptedGraph):
    """A ScriptedGraph whose every run also makes one reader call through the meter, as the real graph's tool does."""

    def __init__(self, meter: runner.ReaderMeter, reader: FakeReader):
        super().__init__()
        self.reader_llm = meter.wrap(reader)

    def invoke(self, request, config, *, context):
        state = super().invoke(request, config, context=context)
        self.reader_llm.invoke(["system", "user"])
        return state


def test_a_v1_run_counts_the_readers_tokens_and_prices_them_at_the_readers_own_price(tmp_path):
    meter, reader = runner.ReaderMeter(), FakeReader()
    code, _out, err, results, _client = v1_command(tmp_path, ReaderCalling(meter, reader), reader_meter=meter)
    assert (code, err) == (0, "") and reader.calls == 87
    ops = files(results)[0].read_text(encoding="utf-8").split("## Ops")[1].split("## Models")[0]
    assert "tokens from response usage: input 9,309, output 2,001" in ops  # the agent's 8,700 / 1,740 and the reader's 87 x (7, 3)
    assert "agent model: input 8,700, output 1,740; reader model: input 609, output 261; the judge's calls are not in these totals" in ops
    agent_price, reader_price = (Decimal("3.00"), Decimal("15.00")), (Decimal("0.25"), Decimal("1.25"))
    dollars = Decimal(8_700) / 10**6 * agent_price[0] + Decimal(1_740) / 10**6 * agent_price[1] + Decimal(609) / 10**6 * reader_price[0] + Decimal(261) / 10**6 * reader_price[1]
    assert (
        dollars == Decimal("0.0526785")
        and f"{COST_LABEL}: ${dollars:.4f} (agent fake-agent-model: $3.00 per M input, $15.00 per M output; source https://provider.invalid/pricing, read 2026-11-09; " in ops
    )
    assert "reader fake-reader-model: $0.25 per M input, $1.25 per M output; source https://reader.invalid/pricing, read 2026-11-01)" in ops


def test_a_v1_reader_reply_without_usage_stops_the_run_with_e4_and_no_file(tmp_path):
    meter, reader = runner.ReaderMeter(), FakeReader(bare_from=3)
    graph = ReaderCalling(meter, reader)
    code, out, err, results, _client = v1_command(tmp_path, graph, reader_meter=meter)
    third = graph.calls[2].example_id
    assert code == 1 and len(graph.calls) == 3 and reader.calls == 3
    assert err.strip() == f"{eval_run.PROGRAM}: stopped: {runner.STRICT_USAGE_LATER}{runner.STRICT_USAGE_READER} (last example: {third})" and files(results) == [] and "results:" not in out


def test_the_real_graph_factory_hands_the_agent_a_source_store_whose_reader_model_is_metered(monkeypatch):
    """The wiring the fake graphs skip: default_graph_factory(meter) builds the agent over a SourceCache whose reader model is the configured one, wrapped by the meter."""
    from flysafelz.agent import graph as agent_graph_module
    from flysafelz.agent import sources as sources_module

    built = {}
    reader = FakeReader()
    monkeypatch.setattr(agent_graph_module, "build_agent", lambda model, **kwargs: built.update(kwargs) or SimpleNamespace(invoke=None))
    monkeypatch.setattr(agent_graph_module, "shared_guidance_index", lambda: "index")
    monkeypatch.setattr(sources_module.reader_config, "build_llm", lambda: reader)
    meter = runner.ReaderMeter()
    eval_run.default_graph_factory(meter)
    calls = runner.ModelCalls()
    meter.attach(calls)
    llm = built["source_cache"].reader_llm()
    assert llm.model_name == "fake-reader-model" and llm.invoke(["system", "user"]).content == "{}" and reader.calls == 1
    assert calls.reader_usage()["total_tokens"] == 10
    eval_run.default_graph_factory()  # no meter: the process-wide store, as before
    assert built["source_cache"] is agent_graph_module.shared_source_cache()


def test_v1_without_a_price_for_the_reader_model_stops_before_the_client_is_built(tmp_path):
    only_agent = tmp_path / "agent-only.yaml"
    only_agent.write_text(PRICE_FILE.split("  fake-reader-model:")[0], encoding="utf-8")

    def never(env):
        raise Boom("no client before the prices are known")

    built = write_dataset(tmp_path / "data")
    out, err = io.StringIO(), io.StringIO()
    code = eval_run.main(["v1"], environ=V1_ENV, stdout=out, stderr=err, results_dir=tmp_path / "results", pricing_path=only_agent, client_factory=never, data_dir=built.data_dir, root=built.root)
    assert code == 1 and "no price for model 'fake-reader-model'" in err.getvalue() and str(only_agent) in err.getvalue() and files(tmp_path / "results") == []


def test_v1_without_a_price_stops_before_the_client_is_built(tmp_path):
    empty = tmp_path / "empty.yaml"
    empty.write_text("", encoding="utf-8")

    def never(env):
        raise Boom("no client before the price is known")

    built = write_dataset(tmp_path / "data")
    out, err = io.StringIO(), io.StringIO()
    code = eval_run.main(["v1"], environ=V1_ENV, stdout=out, stderr=err, results_dir=tmp_path / "results", pricing_path=empty, client_factory=never, data_dir=built.data_dir, root=built.root)
    assert code == 1 and "no price for model 'fake-agent-model'" in err.getvalue() and files(tmp_path / "results") == []


def dotenv_only_env(tmp_path: Path) -> tuple[dict[str, str], Path]:
    """A v1 environment where READER_MODEL and JUDGE_MODEL are in no environment variable: only the file READER_DOTENV_PATH names has them."""
    dotenv = tmp_path / "reader-and-judge.env"
    dotenv.write_text("READER_MODEL=reader-from-dotenv\nJUDGE_MODEL=judge-from-dotenv\n", encoding="utf-8")
    env = {name: value for name, value in V1_ENV.items() if name not in ("READER_MODEL", "JUDGE_MODEL")}
    return {**env, "READER_DOTENV_PATH": str(dotenv)}, dotenv


def test_a_reader_model_set_only_in_the_dotenv_file_still_needs_a_price_before_anything_is_sent(tmp_path):
    env, _dotenv = dotenv_only_env(tmp_path)

    def never(e):
        raise Boom("no client before the reader's price is known")

    built = write_dataset(tmp_path / "data")
    out, err = io.StringIO(), io.StringIO()
    pricing = priced(tmp_path)  # fake-agent-model and fake-reader-model, nothing for reader-from-dotenv
    code = eval_run.main(["v1"], environ=env, stdout=out, stderr=err, results_dir=tmp_path / "results", pricing_path=pricing, client_factory=never, data_dir=built.data_dir, root=built.root)
    assert code == 1 and "no price for model 'reader-from-dotenv'" in err.getvalue() and str(pricing) in err.getvalue() and files(tmp_path / "results") == []


def test_the_models_and_the_cost_use_the_reader_and_judge_models_the_run_read_from_the_dotenv_file(tmp_path):
    env, _dotenv = dotenv_only_env(tmp_path)
    pricing = tmp_path / "dotenv-pricing.yaml"
    pricing.write_text(PRICE_FILE.replace("fake-reader-model", "reader-from-dotenv"), encoding="utf-8")
    meter, reader = runner.ReaderMeter(), FakeReader()
    code, _out, err, results, _client = v1_command(tmp_path, ReaderCalling(meter, reader), reader_meter=meter, environ=env, pricing=pricing)
    assert (code, err) == (0, "")
    text = files(results)[0].read_text(encoding="utf-8")
    models = text.split("## Models")[1]
    assert "- reader: reader-from-dotenv" in models and "- judge: judge-from-dotenv" in models and "not used in this experiment" not in models.split("- v0")[0]
    assert "reader reader-from-dotenv: $0.25 per M input, $1.25 per M output" in text


def test_the_self_preference_line_is_in_the_file_when_agent_and_judge_are_one_family(tmp_path):
    env = {**V1_ENV, "AGENT_MODEL": "claude-sonnet-5", "JUDGE_MODEL": "claude-opus-5"}
    priced_file = tmp_path / "p.yaml"
    priced_file.write_text(PRICE_FILE.replace("fake-agent-model", "claude-sonnet-5"), encoding="utf-8")
    built = write_dataset(tmp_path / "data")
    client = V0Client()
    out = io.StringIO()
    code = eval_run.main(
        ["v1"],
        environ=env,
        stdout=out,
        stderr=io.StringIO(),
        results_dir=tmp_path / "results",
        pricing_path=priced_file,
        graph_factory=ScriptedGraph().factory,
        client_factory=lambda e: client,
        judge_factory=fake_judge,
        data_dir=built.data_dir,
        root=built.root,
        cache_dir=tmp_path / "cache",
        sleep=FakeSleeper(),
        timeout=RecordingTimeout(),
    )
    assert code == 0
    assert "self-preference" in files(tmp_path / "results")[0].read_text(encoding="utf-8")


def test_the_synthetic_dataset_has_what_this_file_assumes():
    assert len(make_examples()) == 44 and runner.AGENT_LABEL == "v1"


def test_v1_without_split_runs_dev_only_and_never_touches_held_out(tmp_path):
    built = write_dataset(tmp_path / "data")
    graph = ScriptedGraph()
    code, _out, _err, results, client = v1_command(tmp_path, graph, None, built=built)
    dev = {e["id"] for e in built.examples if e["metadata"]["split"] == "dev"}
    held_out = {e["id"] for e in built.examples if e["metadata"]["split"] == "held_out"}
    assert code == 0 and {i for i, _ in graph.order()} == dev and not ({i for i, _ in graph.order()} & held_out)
    assert list(client.datasets) == ["flysafelz-dev"] and client.projects[0]["project_name"].startswith("v1-dev-")
    text = files(results)[0].read_text(encoding="utf-8")
    assert not any(example_id in text for example_id in held_out)


def test_v1_with_split_held_out_runs_only_held_out_in_a_separate_project(tmp_path):
    built = write_dataset(tmp_path / "data")
    graph = ScriptedGraph()
    code, _out, _err, results, client = v1_command(tmp_path, graph, None, "--split", "held_out", built=built)
    held_out = {e["id"] for e in built.examples if e["metadata"]["split"] == "held_out"}
    assert code == 0 and {i for i, _ in graph.order()} == held_out and list(client.datasets) == ["flysafelz-held-out"]
    [project] = client.projects
    assert project["project_name"] == "v1-held_out-20261109-120000" and project["metadata"]["split"] == "held_out"
    assert [p.name for p in files(results)] == ["v1-held_out-20261109-120000.md"]
    assert "v1 over the held_out split" in files(results)[0].read_text(encoding="utf-8")


def test_v0_with_split_held_out_has_its_own_project_and_file(tmp_path):
    built = write_dataset(tmp_path / "data")
    code, _out, _err, results, _model, client = v0_command(tmp_path, None, None, "--split", "held_out", data_dir=built.data_dir, root=built.root)
    assert code == 0 and list(client.datasets) == ["flysafelz-held-out"]
    assert client.projects[0]["project_name"] == "v0-held_out-20261109-120000" and [p.name for p in files(results)] == ["v0-held_out-20261109-120000.md"]


# --- rag-eval -------------------------------------------------------------------------------------------------------------------------------------


def test_rag_eval_prints_the_score_and_writes_its_file_with_no_agent_judge_or_langsmith_call(tmp_path):
    from evals.agent import rag_eval
    from flysafelz.guidance.hit4 import load_questions

    questions = load_questions(rag_eval.QUESTIONS_PATH)
    chunks = {q.question: [rag_eval.Chunk(q.expected[0].document, "1", q.expected[0].section)] for q in questions[:8]}
    asked = []

    def retriever(question):
        asked.append(question)
        return chunks.get(question, [])

    out, err = io.StringIO(), io.StringIO()
    code = rag_eval.main([], search_factory=lambda data_dir: (retriever, []), stdout=out, stderr=err, moment=STAMP, results_dir=tmp_path / "results")
    assert (code, err.getvalue()) == (0, "")
    lines = out.getvalue().splitlines()
    assert lines[0] == "hit@4: 8/10 (80%)" and sum(line.startswith("miss: ") for line in lines) == 2
    [written] = files(tmp_path / "results")
    assert written.name == "rag-20261109-120000.md" and written.read_text(encoding="utf-8").count("## Q") == 10
    assert asked == [q.question for q in questions]


def test_the_makefile_passes_split_to_the_eval_targets_only_when_it_is_set(tmp_path):
    """`make eval` and `make eval-v0` with a stand-in `uv` that prints its arguments: no model, no key."""
    import os
    import shutil
    import subprocess

    make = shutil.which("make")
    assert make
    uv = tmp_path / "bin" / "uv"
    uv.parent.mkdir()
    uv.write_text('#!/bin/sh\nfor a in "$@"; do echo "[$a]"; done\n', encoding="utf-8")
    uv.chmod(0o755)
    root = Path(__file__).resolve().parents[2]

    def run(target, **variables):
        env = {k: v for k, v in os.environ.items() if k != "SPLIT"}
        done = subprocess.run(
            [make, "-s", target, *(f"{k}={v}" for k, v in variables.items())], cwd=root, env={**env, "PATH": f"{uv.parent}:{env['PATH']}"}, capture_output=True, text=True, check=True
        )
        return done.stdout.splitlines()

    assert run("eval")[-3:] == ["[-m]", "[evals.agent.eval_run]", "[v1]"]
    assert run("eval", SPLIT="held_out")[-5:] == ["[-m]", "[evals.agent.eval_run]", "[v1]", "[--split]", "[held_out]"]
    assert run("eval-v0")[-1] == "[v0]" and run("eval-v0", SPLIT="held_out")[-2:] == ["[--split]", "[held_out]"]
    assert run("rag-eval")[-2:] == ["[-m]", "[evals.agent.rag_eval]"]
    assert run("eval", SPLIT="x; echo pwned")[-2:] == ["[--split]", "[x; echo pwned]"]  # one argument, never parsed by the shell
