"""The v0 baseline end to end on the committed dev examples (5a PR-9.F1, GP5, G14, E5, E8, E9, E14): the real loader, the real fixture folders, a fake chat model and a recording LangSmith client.

`uv run pytest tests/integration/test_v0_run.py -s` prints the run summary. Nothing real is called: the settings are fake values, the model is tests/unit/v0_helpers.py's CannedV0Model
(a real `with_structured_output` parse of a canned tool call) and the client records. The held-out split is never opened here (F7).
"""

import io
from datetime import UTC, datetime
from pathlib import Path

import pytest
from eval_runner_helpers import FakeSleeper, RecordingTimeout
from v0_helpers import FAKE_V0_ENV, USAGE, CannedV0Model, V0Client, default_reply

from evals.agent import runner, v0
from evals.agent.evaluators.aggregate import EVALUATORS
from evals.agent.loader import REPO, load_examples, upload_dataset
from evals.agent.v0_output import V0Output
from flysafelz.agent.state import RunContext

STAMP = datetime(2026, 11, 8, 16, 7, tzinfo=UTC)
EXPERIMENT = "v0-dev-20261108-160700"
NULL_FOR_V0 = ("path_accuracy", "required_tools", "forbidden_tools", "step_cap", "guard_pass", "decision_first", "cited", "recheck_stated")


@pytest.fixture(scope="module")
def dev_assess():
    return [r for r in load_examples("dev") if r["reference"]["path"] == "assess"]


@pytest.fixture(scope="module")
def dev_run(tmp_path_factory):
    """The whole `python -m evals.agent.v0` (dev by default) on the committed data, once: (exit code, stdout, stderr, model, client, cache dir)."""
    model, client = CannedV0Model(), V0Client()
    out, err = io.StringIO(), io.StringIO()
    cache_dir = tmp_path_factory.mktemp("cache") / "eval-cache"
    code = v0.main(
        [],
        environ=FAKE_V0_ENV,
        stdout=out,
        stderr=err,
        model_factory=lambda env: model,
        client_factory=lambda env: client,
        moment=STAMP,
        cache_dir=cache_dir,
        sleep=FakeSleeper(),
        timeout=RecordingTimeout(),
    )
    return code, out.getvalue(), err.getvalue(), model, client, cache_dir


def two_cases(dev_assess):
    """dev_01 (no drop) and dev_10 (drop: airspace): two fixture cases."""
    return [r for r in dev_assess if r["id"].startswith(("dev_01", "dev_10"))]


def run_two(records, tmp_path, model, client=None, *, reps=1, **options):
    """runner.run_experiment on V0Graph over the given records, as v0.run_split calls it."""
    if client is not None:
        upload_dataset("dev", client)  # the runner reads the split's dataset to link the experiment to it
    return runner.run_experiment(
        records,
        experiment="v0-dev-test",
        graph_factory=lambda: v0.V0Graph(model, records),
        client=client,
        cache_dir=tmp_path / "cache",
        reps=reps,
        agent="v0",
        run_metadata=v0.run_metadata,
        require_usage=False,
        **{"sleep": FakeSleeper(), "timeout": RecordingTimeout(), **options},
    )


# --- GP5: the default run -----------------------------------------------------------------------------------------------------------


def test_the_default_run_is_dev_16_assess_examples_three_sequential_reps_one_call_each(dev_run, dev_assess):
    code, out, err, model, client, _cache = dev_run
    assert (code, err) == (0, "")
    assert len(dev_assess) == 16 and {r["reference"]["path"] for r in dev_assess} == {"assess"}
    ids = [r["id"] for r in dev_assess]
    assert [(run["extra"]["metadata"]["rep"], run["extra"]["metadata"]["example_id"]) for run in client.runs] == [(rep, i) for rep in (1, 2, 3) for i in ids]
    assert len(model.prompts) == 48  # one structured call per example per rep, none for explain, clarify, act or decline
    assert "48 of 48 runs" in out and "errored: 0 of 48 runs" in out and "incomplete: false" in out
    print()
    print(out, end="")


def test_the_experiment_is_named_v0_dev_stamp_and_holds_the_fixed_report_line(dev_run):
    _code, out, _err, _model, client, _cache = dev_run
    assert out.splitlines()[0].startswith(f"experiment {EXPERIMENT} (split dev)")
    [project] = client.projects
    assert project["project_name"] == EXPERIMENT
    assert project["metadata"]["agent"] == "v0" and project["metadata"]["split"] == "dev"
    assert project["metadata"]["report_line"] == "v1's advantage covers reading plus arithmetic; not separated (v0b, v0 with reader output, is the named next experiment)"
    assert out.splitlines()[-1] == v0.REPORT_LINE
    assert list(client.datasets) == ["flysafelz-dev"]  # held-out is not named, so it is not touched


def test_every_logged_run_is_traced_as_source_v0_never_pilot(dev_run):
    _code, _out, _err, _model, client, _cache = dev_run
    assert len(client.runs) == 48
    assert {run["extra"]["metadata"]["source"] for run in client.runs} == {"v0"}
    assert {run["name"].split("-", 1)[0] for run in client.runs} == {"v0"} and all("v0" in run["tags"] for run in client.runs)
    assert "pilot" not in str(client.projects)


def test_the_model_call_itself_carries_source_v0_in_its_trace_metadata(dev_assess):
    from langchain_core.callbacks import BaseCallbackHandler

    seen = []

    class Handler(BaseCallbackHandler):
        def on_chat_model_start(self, serialized, messages, *, run_id, metadata=None, **kwargs):
            seen.append(dict(metadata or {}))

    record = dev_assess[0]
    graph = v0.V0Graph(CannedV0Model(), [record])
    context = RunContext(now=datetime.fromisoformat(record["inputs"]["now"]), case=record["fixtures"]["case"], drop=tuple(record["fixtures"].get("drop") or ()))
    graph.invoke({}, {"metadata": {"example_id": record["id"]}, "callbacks": [Handler()]}, context=context)
    assert seen and all(meta.get("source") == "v0" and meta.get("langgraph_node") == runner.AGENT_MODEL_NODE for meta in seen)


def test_nothing_is_written_to_the_reader_cache_and_the_cache_directory_is_never_made(dev_run):
    _code, out, _err, _model, _client, cache_dir = dev_run
    assert not cache_dir.exists()  # the runner only names the path; v0 has no reader to cache
    assert not (REPO / "data" / "eval-cache" / f"{EXPERIMENT}.sqlite").exists()
    assert "reader cache" not in out and "reader calls failed" not in out


def test_usage_of_every_run_is_recorded_per_call(dev_run):
    _code, _out, _err, _model, client, _cache = dev_run
    usages = [run["outputs"]["usage"] for run in client.runs]
    assert all(u == {**USAGE, "model_calls": 1, "calls_without_usage": 0} for u in usages)


def test_the_feedback_of_each_run_comes_from_the_shared_evaluators_and_leaves_the_null_ones_out(dev_run):
    _code, _out, _err, _model, client, _cache = dev_run
    assert client.feedback, "no feedback was logged"
    run_ids = {run["id"] for run in client.runs}
    assert {item[0] for item in client.feedback} <= run_ids
    keys = {item[1] for item in client.feedback}
    assert keys <= set(EVALUATORS) and "verdict_match" in keys and "window_iou" in keys
    assert not keys & set(NULL_FOR_V0)
    assert all(item[4].get("session_id") for item in client.feedback)  # the experiment's project id was found


def test_the_summary_lists_the_null_evaluators_as_n_a_never_zero(dev_run):
    _code, out, _err, _model, _client, _cache = dev_run
    for key in NULL_FOR_V0:
        assert f"  {key}: n/a (0 runs)" in out
    assert "  verdict_match: n/a" not in out and "  window_iou: " in out


def test_blue_mill_today_has_no_compound_key_in_the_file_and_runs_as_compound_false(dev_run, dev_assess):
    _code, _out, _err, _model, client, _cache = dev_run
    [today] = [r for r in dev_assess if r["id"] == "blue_mill_today"]
    assert today["metadata"]["compound"] is False
    assert [run["name"] for run in client.runs].count("v0-blue_mill_today") == 3


def test_a_dropped_source_is_an_unavailable_block_in_the_prompt_of_its_example(dev_run, dev_assess):
    _code, _out, _err, model, _client, _cache = dev_run
    by_id = {r["id"]: str(model.prompts[i][-1].content) for i, r in enumerate(dev_assess)}  # rep 1
    [ten] = [i for i in by_id if i.startswith("dev_10")]
    [twenty_seven] = [i for i in by_id if i.startswith("dev_27")]
    assert "SOURCE UNAVAILABLE: airspace" in by_id[ten] and "SOURCE UNAVAILABLE: NOTAM" not in by_id[ten]
    assert "SOURCE UNAVAILABLE: NOTAM" in by_id[twenty_seven] and "SOURCE UNAVAILABLE: airspace" not in by_id[twenty_seven]
    assert not any("SOURCE UNAVAILABLE" in text for i, text in by_id.items() if i not in (ten, twenty_seven))


def test_the_injected_case_keeps_its_text_inside_one_data_tag(dev_run, dev_assess):
    _code, _out, _err, model, _client, _cache = dev_run
    [index] = [n for n, r in enumerate(dev_assess) if r["id"].startswith("dev_26")]
    user = str(model.prompts[index][-1].content)
    assert user.count("<notam_text") == 1 and user.count("</notam_text") == 1


def test_the_run_never_calls_the_assess_tool_or_the_reader(monkeypatch, dev_assess, tmp_path):
    def boom(*args, **kwargs):
        raise AssertionError("v0 must not reach this")

    monkeypatch.setattr("flysafelz.assess.assess", boom)
    monkeypatch.setattr("flysafelz.reader.read.read_notam", boom)
    result = run_two(two_cases(dev_assess), tmp_path, CannedV0Model())
    assert result.errored_runs == 0 and len(result.scored) == 2


# --- E5, E14, E9: scoring -----------------------------------------------------------------------------------------------------------


def test_a_v0_run_of_a_compound_example_is_scored_by_the_verdict_evaluators_only(dev_run, dev_assess):
    _code, _out, _err, _model, client, _cache = dev_run
    compound = {r["id"] for r in dev_assess if r["metadata"]["compound"]}
    assert len(compound) == 3
    ids_by_run = {run["id"]: run["extra"]["metadata"]["example_id"] for run in client.runs}
    keys = {item[1] for item in client.feedback if ids_by_run[item[0]] in compound}
    assert "must_mention" not in keys and "must_not_mention" not in keys and "verdict_match" in keys


def test_a_segment_outside_the_window_is_clipped_by_the_shared_evaluators(dev_assess, tmp_path):
    record = two_cases(dev_assess)[0]
    start, end = record["reference"]["window"]

    def wide(user):
        reply = default_reply(user)
        reply["segments"][0]["start"], reply["segments"][0]["end"] = "2026-09-29T00:00:00Z", "2026-09-29T23:00:00Z"
        return reply

    plain = run_two([record], tmp_path, CannedV0Model())
    wider = run_two([record], tmp_path / "w", CannedV0Model(plan=wide))
    assert (start, end) == tuple(record["reference"]["window"])
    scores = []
    for result in (plain, wider):
        pairs = v0.scored_runs(result, [record])
        from evals.agent.evaluators.aggregate import evaluate_runs

        evaluation = evaluate_runs(scored for _run, scored in pairs)
        scores.append(evaluation.runs[0].feedback["window_iou"]["score"])  # type: ignore[index]
    assert scores[0] == scores[1]


# --- errored handling ---------------------------------------------------------------------------------------------------------------


def test_an_exception_a_timeout_and_a_schema_invalid_reply_are_failed_attempts_then_errored_and_never_scored(dev_assess, tmp_path):
    records = two_cases(dev_assess)
    bad_verdict = default_reply("Window to assess: 2026-09-29T09:00:00Z to 2026-09-29T11:00:00Z")
    bad_verdict["segments"][0]["verdict"] = "GO"
    extra_field = default_reply("Window to assess: 2026-09-29T09:00:00Z to 2026-09-29T11:00:00Z")
    extra_field["notes"] = "not in the schema"

    for broken in (RuntimeError("endpoint exploded"), bad_verdict, extra_field):
        client = V0Client()
        model = CannedV0Model(plan=lambda user, broken=broken: broken if "latitude 53.802797" in user else None)
        result = run_two(records, tmp_path / str(id(broken)), model, client)
        assert [r.example_id for r in result.records] == [records[0]["id"], records[1]["id"]]
        first, second = result.records
        assert first.errored and first.attempts == 4 and len(first.backoff_waits) == 3 and not second.errored
        assert result.errored_runs == 1 and [e.example_id for e in result.errored[1]] == [records[0]["id"]]
        assert len(model.prompts) == 4 + 1  # three retries on the first, one call on the second
        logged = {run["extra"]["metadata"]["example_id"]: run for run in client.runs}
        assert logged[records[0]["id"]]["outputs"] is None and logged[records[0]["id"]]["error"]
        pairs = v0.scored_runs(result, records)
        assert [scored.errored for _run, scored in pairs] == [True, False] and pairs[0][1].answer is None


def test_a_timeout_is_a_failed_attempt_too(dev_assess, tmp_path):
    def timing_out(call, seconds):
        raise runner.ExampleTimeout(f"no result within {seconds:g} s")

    records = two_cases(dev_assess)[:1]
    result = run_two(records, tmp_path, CannedV0Model(), timeout=timing_out)
    assert result.records[0].errored and result.records[0].attempts == 4 and "no result within" in (result.records[0].last_error or "")


def test_a_reply_with_no_usage_is_kept_with_usage_absent_never_zero(dev_assess, tmp_path):
    client = V0Client()
    result = run_two(two_cases(dev_assess), tmp_path, CannedV0Model(usage=False), client)
    assert result.errored_runs == 0 and all(r.usage is None for r in result.records)
    assert all(run["outputs"]["usage"] is None for run in client.runs)


def test_a_notam_id_not_in_the_prompt_is_not_scored_and_is_recorded_in_the_run_metadata(dev_assess, tmp_path):
    record = two_cases(dev_assess)[0]

    def invents(user):
        reply = default_reply(user)
        reply["notam_actions"].append({"notam_id": "Z9999/26", "action": "NO_FLY"})
        return reply

    client = V0Client()
    result = run_two([record], tmp_path, CannedV0Model(plan=invents), client)
    [run] = result.records
    assert run.output["v0"]["unlisted"] == ("Z9999/26",)  # type: ignore[index]
    assert "Z9999/26" not in run.output["v0"]["scored"].notam_actions  # type: ignore[index]
    assert client.runs[0]["extra"]["metadata"]["unlisted_notam_ids"] == ["Z9999/26"]


def test_the_model_is_asked_for_the_strict_schema_with_the_raw_message_kept(dev_assess):
    model = CannedV0Model()
    v0.V0Graph(model, dev_assess[:1])
    assert model.bound == ["V0Output"] or [b for b in model.bound if "V0Output" in b]
    assert V0Output.model_config.get("extra") == "forbid"


# --- settings -----------------------------------------------------------------------------------------------------------------------


def test_a_missing_agent_model_stops_before_any_client_model_or_example(tmp_path):
    def never(*args, **kwargs):
        raise AssertionError("nothing may be built")

    env = {k: v for k, v in FAKE_V0_ENV.items() if k != "AGENT_MODEL"}
    err = io.StringIO()
    code = v0.main([], environ=env, stdout=io.StringIO(), stderr=err, model_factory=never, client_factory=never, cache_dir=tmp_path / "c")
    assert code == 1 and "AGENT_MODEL" in err.getvalue() and "nothing was built" in err.getvalue()
    assert not (tmp_path / "c").exists()


def test_every_missing_setting_is_named_in_one_message():
    assert v0.missing_settings({}) == ["AGENT_BASE_URL", "AGENT_MODEL", "AGENT_API_KEY", "LANGSMITH_API_KEY"]
    assert v0.missing_settings({**FAKE_V0_ENV, "AGENT_BASE_URL": "not a url"}) == ["AGENT_BASE_URL (must be an http:// or https:// URL)"]
    assert v0.missing_settings(FAKE_V0_ENV) == []


def test_the_split_is_dev_unless_held_out_is_typed():
    assert v0._parser().parse_args([]).split == "dev"
    assert v0.run_split.__kwdefaults__ is None or "split" not in (v0.run_split.__kwdefaults__ or {})
    assert v0.run_split.__defaults__ == ("dev",)


def test_a_fatal_runner_stop_is_one_line_on_stderr_and_exit_1(dev_assess, tmp_path):
    err = io.StringIO()
    code = v0.main(
        [],
        environ=FAKE_V0_ENV,
        stdout=io.StringIO(),
        stderr=err,
        model_factory=lambda env: CannedV0Model(),
        client_factory=lambda env: V0Client(fail_on={"create_project": 1}),
        cache_dir=tmp_path / "c",
        moment=STAMP,
    )
    assert code == 1 and err.getvalue().startswith("evals.agent.v0: stopped: LangSmith could not open the experiment") and err.getvalue().count("\n") == 1


def test_the_command_runs_as_a_module_with_help():
    import subprocess
    import sys

    done = subprocess.run([sys.executable, "-m", "evals.agent.v0", "--help"], capture_output=True, text=True, cwd=Path(__file__).resolve().parents[2], check=False)
    assert done.returncode == 0 and "--split" in done.stdout
