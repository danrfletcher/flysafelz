"""`make eval` and `make eval-v0` (5a PR-10.F1, G16, E4): run an experiment, score it, write `evals/results/<experiment>.md`.

    python -m evals.agent.eval_run v1 [--split dev] [--experiment NAME]     the v1 agent, the code evaluators and the judge
    python -m evals.agent.eval_run v0 [--split dev] [--experiment NAME]     the v0 baseline, the shared code evaluators (the judge is null for v0)

    run_v1(split, ...) -> Finished           the whole v1 run, scored; raises what run_v1.run_split raises, PricingError and ReportError
    run_v0(split, ...) -> Finished           the same for v0
    scored_answer_of(state) -> ScoredAnswer  the answer a v1 run gave, from its final graph state
    main(argv, ...) -> int                   0 a run that finished and has its results file, 1 anything that stopped it, 130 an interrupt

The split is dev unless `--split held_out` is typed; `make eval SPLIT=held_out` is that flag. The runner puts a held-out run in its own LangSmith project (the loader's split project).
This module reads no example's answers: it scores what the run just produced, and writes only under `evals/results/` (report.write_report).

The order. The price of the agent model (v1 also the reader model) is looked up in evals/agent/pricing.yaml first, before the loader, the client or the first example, so a run is never paid for and left with no
cost line. Then the run (run_v1.run_split or v0.run_split) with the strict usage check (runner `strict_usage`): the first response without usage, or a later one, stops the run with
the error text of E4 on stderr, exit 1, and NO results file is written. A run that finished is scored, its feedback is sent to LangSmith, and the file is written whole in one go.
A run stopped by an interrupt has no file either (exit 130): a half-run is not a report.

The v1 path. The predicted path is the one the guard tagged, which the graph reports to `on_outcome` and which the final state does not hold. `TaggedGraph` is the agent graph with that
one hook added: it keeps the outcome of the turn that just ran and puts its path in the state it returns (`path`). It is the graph `build_graph` builds, with one more callback.
v1's reader calls run on a thread pool inside a tool, where the run's callbacks do not reach, and their usage is not in a run's state. So the reader's chat model is wrapped
(runner.ReaderMeter, through a SourceCache subclass whose `reader_llm` returns the wrapped model) and each reply it returns is counted for the example in progress: the v1 token totals are the agent's
responses plus the reader's, each priced at its own model's list price (AGENT_MODEL and READER_MODEL both need an entry in pricing.yaml, looked up before the run). A reading the
cache returns made no call and is not counted. The judge's calls are not in the totals (the report says so).
"""

import argparse
import os
import sys
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from evals.agent import report, runner, v0
from evals.agent import run_v1 as v1
from evals.agent.evaluators.aggregate import Evaluation, ScoredRun, evaluate_runs
from evals.agent.evaluators.answer import ScoredAnswer, ScoredSegment, clean_id, normalise_path
from evals.agent.evaluators.trajectory import guard_of, tool_calls_of
from evals.agent.import_dataset import DATA_DIR, DatasetError
from evals.agent.judge import SELF_PREFERENCE_LINE, Judge, all_evaluators, build_judge, send_feedback
from evals.agent.judge_config import JudgeConfigError, judge_config
from evals.agent.loader import DEFAULT_SPLIT, EXPECTED, REPO, SPLITS, load_dataset
from evals.agent.pricing import PRICING_PATH, Price, PricingError, price_for
from evals.reader_eval.downgrade import SEVERITY
from flysafelz.contracts import NotamAction
from flysafelz.reader.config import ReaderConfigError, reader_config

PROGRAM = "evals.agent.eval_run"
FAILED_EXIT = 1
INTERRUPTED_EXIT = 130
AGENT_MODEL, READER_MODEL, JUDGE_MODEL = "AGENT_MODEL", "READER_MODEL", "JUDGE_MODEL"
ASSESS_TOOL = "assess"
PATH_STATE_KEY = "path"  # the key TaggedGraph adds to a run's state
INTERRUPTED = "interrupted: a half-run is not a report, so no results file was written"

JudgeFactory = Callable[[Mapping[str, str], Any, str], Judge]


class Interrupted(runner.RunnerError):
    """The run was interrupted (SIGINT): the runner returned what it had, and none of it is reported."""


@dataclass
class Finished:
    result: runner.ExperimentResult
    evaluation: Evaluation
    report_path: Path
    feedback_sent: int


class TaggedGraph:
    """The agent graph plus the path the guard tagged, in the state it returns (see the module text). Runs are one at a time, so one slot holds the turn's outcome."""

    def __init__(self, build: Callable[[Callable[[Any], None]], Any]):
        self._lock = threading.Lock()
        self._path: str | None = None
        self.graph = build(self._on_outcome)

    def _on_outcome(self, outcome: Any) -> None:
        with self._lock:
            self._path = outcome.path

    def invoke(self, request: Any, config: Any, *, context: Any) -> dict[str, Any]:
        with self._lock:
            self._path = None
        state = self.graph.invoke(request, config, context=context)
        with self._lock:
            return {**state, PATH_STATE_KEY: self._path}


def default_graph_factory(meter: runner.ReaderMeter | None = None) -> Any:
    """ticket 4's `build_graph`, with the outcome hook (`build_agent(on_outcome=...)`), traced the way `build_graph` traces it. Imported only when a run reaches it.

    With a `meter` the source store is a SourceCache whose `reader_llm` (the NOTAM reader's chat model, built by ticket 4's own code from the reader's settings) comes back wrapped
    by the meter, so its replies are counted; without one it is the process-wide store, as before.
    """
    from flysafelz.agent.config import LazyChatModel
    from flysafelz.agent.graph import SourceCache, build_agent, shared_guidance_index, shared_source_cache
    from flysafelz.agent.tracing import report_outcome, traced

    class MeteredSources(SourceCache):
        def reader_llm(self) -> Any:
            return meter.wrap(super().reader_llm())

    def sources() -> Any:
        return shared_source_cache() if meter is None else MeteredSources()

    def build(hook: Callable[[Any], None]) -> Any:
        def on_outcome(outcome: Any) -> None:
            hook(outcome)
            report_outcome(outcome)

        return traced(build_agent(LazyChatModel(), source_cache=sources(), index=shared_guidance_index(), on_outcome=on_outcome))

    return TaggedGraph(build)


def _field(item: Any, name: str) -> Any:
    return item.get(name) if isinstance(item, Mapping) else getattr(item, name, None)


def assess_artifact_of(state: Mapping[str, Any]) -> Any:
    """The artifact of the run's last `assess` call (a model, or its dict after a round trip), or None when the run never assessed."""
    found = None
    for message in state.get("messages") or []:
        if _field(message, "type") == "tool" and _field(message, "name") == ASSESS_TOOL and _field(message, "artifact") is not None:
            found = _field(message, "artifact")
    return found


def scored_answer_of(state: Mapping[str, Any]) -> ScoredAnswer:
    """The answer a v1 run gave, from its final graph state: the last assess artifact's VERDICT-tier segments and each NOTAM's strictest action, the delivered text, the
    tool calls, the guard result and the path the guard tagged (PATH_STATE_KEY). A run that never assessed has no segments and no actions."""
    path, text = state.get(PATH_STATE_KEY), str(state.get("answer") or "")
    segments: list[ScoredSegment] = []
    actions: dict[str, NotamAction] = {}
    for segment in _field(assess_artifact_of(state), "segments") or []:
        verdict = _field(segment, "verdict")
        if verdict is None:
            continue  # an OUTLOOK or TOO_FAR hour has no verdict to score
        segments.append(ScoredSegment(datetime.fromisoformat(_field(segment, "start")), datetime.fromisoformat(_field(segment, "end")), verdict, bool(_field(segment, "approaching_limits"))))
        for notam in _field(segment, "notams") or []:
            found = NotamAction(_field(notam, "action"))
            held = actions.get(clean_id(_field(notam, "id")))
            if held is None or SEVERITY.index(found) < SEVERITY.index(held):
                actions[clean_id(_field(notam, "id"))] = found
    return ScoredAnswer(normalise_path(path), tuple(segments), actions, text, tool_calls_of(state), guard_of(state))


def v1_pairs(result: runner.ExperimentResult, records: Sequence[Mapping[str, Any]]) -> list[tuple[runner.RunRecord, ScoredRun]]:
    """Every run of a v1 result with the run the scorers take. An errored run is passed as errored: it is listed and never scored."""
    by_id = {record["id"]: record for record in records}
    return [(run, ScoredRun(by_id[run.example_id], None if run.errored else scored_answer_of(run.output or {}), run.rep, run.errored)) for run in result.records]


def _model_of(environ: Mapping[str, str], variable: str) -> str | None:
    """The model a run uses for the role `variable` names, resolved the way the run resolves it: AGENT_MODEL from the environment only (agent_config); READER_MODEL through reader_config and
    JUDGE_MODEL through judge_config, which take the environment first and then the `.env` file (READER_DOTENV_PATH, else the repository root's). None when it is not set anywhere, or the
    role's other settings are incomplete: the run's own settings check names what is missing."""
    if variable == AGENT_MODEL:
        return (environ.get(AGENT_MODEL) or "").strip() or None
    try:
        config = reader_config(environ) if variable == READER_MODEL else judge_config(environ)
    except (ReaderConfigError, JudgeConfigError):
        return None
    return config.model.strip() or None


def _price(environ: Mapping[str, str], pricing_path: Path, variable: str = AGENT_MODEL) -> Price | None:
    """The price of the model named by `variable` (AGENT_MODEL, or READER_MODEL for a v1 run, from the environment or the `.env` file as the run reads it), looked up before anything is sent. Not set: the run's own settings check names it, so no lookup here."""
    model = _model_of(environ, variable)
    return price_for(model, pricing_path) if model else None


def _models(environ: Mapping[str, str], agent: str) -> dict[str, str | None]:
    if agent == v0.AGENT:
        return {"agent": None, "reader": None, "judge": None, "v0": _model_of(environ, AGENT_MODEL)}
    return {"agent": _model_of(environ, AGENT_MODEL), "reader": _model_of(environ, READER_MODEL), "judge": _model_of(environ, JUDGE_MODEL), "v0": None}


def _finish(
    result: runner.ExperimentResult,
    evaluation: Evaluation,
    sent: int,
    records: Sequence[Mapping[str, Any]],
    *,
    environ: Mapping[str, str],
    client: Any,
    price: Price | None,
    wall_clock_s: float,
    notes: Sequence[str],
    results_dir: Path,
    reader_price: Price | None = None,
) -> Finished:
    link = report.langsmith_link(client, result.experiment)
    built = report.from_experiment(result, evaluation, records, models=_models(environ, result.agent), wall_clock_s=wall_clock_s, price=price, link=link, notes=notes, reader_price=reader_price)
    return Finished(result, evaluation, report.write_report(report.render(built), result.experiment, results_dir), sent)


def run_v1(
    split: str = DEFAULT_SPLIT,
    *,
    experiment: str | None = None,
    environ: Mapping[str, str] | None = None,
    graph_factory: Callable[[], Any] | None = None,
    client_factory: Callable[[Mapping[str, str]], Any] = v1.default_client_factory,
    judge_factory: JudgeFactory | None = None,
    data_dir: Path = DATA_DIR,
    root: Path = REPO,
    expected: Mapping[str, int] = EXPECTED,
    results_dir: Path = report.RESULTS_DIR,
    pricing_path: Path = PRICING_PATH,
    clock: Callable[[], float] = time.monotonic,
    reader_meter: runner.ReaderMeter | None = None,
    **options: Any,
) -> Finished:
    """The v1 run, strict about usage, scored by the code evaluators and the judge, and written to `results_dir`.

    `graph_factory` None builds the real agent graph with its NOTAM reader model metered (`reader_meter`, or a new one); a test passes a graph of its own, and a meter to count a fake reader's replies.
    """
    environ = os.environ if environ is None else environ
    price = _price(environ, pricing_path)
    reader_price = _price(environ, pricing_path, READER_MODEL)
    meter = reader_meter or runner.ReaderMeter()
    box: dict[str, Any] = {}

    def client_of(env: Mapping[str, str]) -> Any:
        box["client"] = client_factory(env)
        return box["client"]

    started = clock()
    result = v1.run_split(
        split,
        experiment=experiment,
        environ=environ,
        graph_factory=graph_factory or (lambda: default_graph_factory(meter)),
        client_factory=client_of,
        data_dir=data_dir,
        root=root,
        expected=expected,
        strict_usage=True,
        reader_meter=meter,
        make_target="eval",
        **options,
    )
    wall_clock = clock() - started
    if result.interrupted:
        raise Interrupted(INTERRUPTED)
    records = load_dataset(split, data_dir=data_dir, root=root, expected=expected).records
    client = box["client"]
    make_judge = judge_factory or (lambda env, c, name: build_judge(env, client=c, project=name))
    judge = make_judge(environ, client, result.experiment)
    pairs = v1_pairs(result, records)
    evaluation = evaluate_runs((scored for _run, scored in pairs), all_evaluators(judge))
    sent = 0
    for (run, _scored), scores in zip((pair for pair in pairs if not pair[0].errored), evaluation.runs, strict=True):
        if run.langsmith_run_id is not None:
            sent += send_feedback(client, run.langsmith_run_id, scores.feedback, project=result.experiment)
    models = _models(environ, runner.AGENT_LABEL)
    notes = [SELF_PREFERENCE_LINE] if report.same_family(models["agent"], models["judge"]) else []
    return _finish(result, evaluation, sent, records, environ=environ, client=client, price=price, wall_clock_s=wall_clock, notes=notes, results_dir=results_dir, reader_price=reader_price)


def run_v0(
    split: str = DEFAULT_SPLIT,
    *,
    experiment: str | None = None,
    environ: Mapping[str, str] | None = None,
    model_factory: Callable[[Mapping[str, str]], Any] | None = None,
    client_factory: Callable[[Mapping[str, str]], Any] | None = None,
    data_dir: Path = DATA_DIR,
    root: Path = REPO,
    expected: Mapping[str, int] = EXPECTED,
    results_dir: Path = report.RESULTS_DIR,
    pricing_path: Path = PRICING_PATH,
    clock: Callable[[], float] = time.monotonic,
    **options: Any,
) -> Finished:
    """The v0 baseline, strict about usage, scored by the shared code evaluators (v0.run_split does it), and written to `results_dir`."""
    environ = os.environ if environ is None else environ
    price = _price(environ, pricing_path)
    box: dict[str, Any] = {}
    make_client = client_factory or v0.default_client_factory

    def client_of(env: Mapping[str, str]) -> Any:
        box["client"] = make_client(env)
        return box["client"]

    started = clock()
    run = v0.run_split(
        split,
        experiment=experiment,
        environ=environ,
        model_factory=model_factory,
        client_factory=client_of,
        data_dir=data_dir,
        root=root,
        expected=expected,
        require_usage=True,
        strict_usage=True,
        make_target="eval-v0",
        **options,
    )
    wall_clock = clock() - started
    if run.result.interrupted:
        raise Interrupted(INTERRUPTED)
    records = v0.assess_records(load_dataset(split, data_dir=data_dir, root=root, expected=expected).records)
    return _finish(run.result, run.evaluation, run.feedback_sent, records, environ=environ, client=box["client"], price=price, wall_clock_s=wall_clock, notes=[v0.REPORT_LINE], results_dir=results_dir)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog=f"python -m {PROGRAM}", description="Run v1 or the v0 baseline over a split, score it and write evals/results/<experiment>.md.")
    parser.add_argument("agent", choices=[runner.AGENT_LABEL, v0.AGENT], help="v1: `make eval`; v0: `make eval-v0`")
    parser.add_argument("--split", default=DEFAULT_SPLIT, choices=list(SPLITS), help="the split to run; dev unless held_out is named")
    parser.add_argument("--experiment", default=None, help="the experiment name (default <agent>-<split>-<UTC stamp>); it names the results file")
    return parser


def main(argv: Sequence[str] | None = None, *, environ: Mapping[str, str] | None = None, stdout: Any = None, stderr: Any = None, **options: Any) -> int:
    """The command. `options` are run_v1's or run_v0's keyword arguments (graph_factory, client_factory, results_dir, ...), for tests."""
    out = sys.stdout if stdout is None else stdout
    err = sys.stderr if stderr is None else stderr
    args = _parser().parse_args(argv)
    run = run_v0 if args.agent == v0.AGENT else run_v1
    try:
        finished = run(args.split, experiment=args.experiment, environ=environ, **options)
    except Interrupted as error:
        print(f"{PROGRAM}: {error}", file=err)
        return INTERRUPTED_EXIT
    except runner.FatalRunError as error:
        print(f"{PROGRAM}: stopped: {error}", file=err)
        return FAILED_EXIT
    except (DatasetError, runner.RunnerError, PricingError, report.ReportError) as error:
        print(f"{PROGRAM}: {error}", file=err)
        return FAILED_EXIT
    for line in finished.result.summary_lines():
        print(line, file=out)
    print(f"feedback items sent to LangSmith: {finished.feedback_sent}", file=out)
    print(f"results: {finished.report_path}", file=out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
