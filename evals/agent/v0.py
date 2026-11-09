"""The v0 baseline (5a PR-9.F1, G14, GP5): one structured model call per assess example, scored by the same evaluators as v1.

    python -m evals.agent.v0 [--split dev] [--experiment NAME]

    run_split(split, ...) -> V0Run            the whole run: the runner's result and the evaluators' scores
    main(argv, ...) -> int                    the command: 0 a run that finished, 1 anything that stopped it, 130 an interrupt
    V0Graph(model, records, root)             what the runner calls in place of ticket 4's graph: `invoke(request, config, context)` makes the one model call
    score(result) -> Evaluation               the shared evaluators (evals.agent.evaluators.aggregate) over the run's answered runs
    REPORT_LINE                               the fixed sentence for PR-10.F1 to print, kept in the experiment's metadata

What v0 is. For each example whose `reference.path` is `assess` (16 in dev; explain, clarify, act and decline examples are never sent), one call to the `AGENT_MODEL` endpoint
with a prompt built by code from the case's own files (v0_prompt.build_v0_messages) and a strict structured reply (v0_output.V0Output). No reader, no `assess`, no daylight
code, no guard and no tools: the model reads the raw NOTAMs and does the arithmetic itself. This module imports none of them (tests/unit/test_v0_fence.py scans it).

It runs through ticket 4's runner (evals.agent.runner.run_experiment), so the order, the retries, the errored handling, the timeout and the LangSmith logging are v1's:
3 sequential reps, each example at a time, 3 retries with backoff, then errored, left out of every score and listed by id. A reply that does not fit the schema, an exception
and a timeout are failed attempts, never a miss. The runner is told `agent="v0"` (the experiment is `v0-<split>-<UTC stamp>`) and `require_usage=False`: a bridge that sends no
usage on a response does not stop a v0 run, and the run's usage is then absent (None), never zeros. The usage that is there is summed per run by the runner (`RunRecord.usage`).

The case. The sources come from `flysafelz.sources.load_case_sources(case, with_readings=False)`: `readings.yaml` is never opened. The example's `fixtures.drop` (the run
context's `drop`) is applied with `flysafelz.verdict_sheet.apply_drops`, so a dropped source is a failed one and the prompt carries a SOURCE UNAVAILABLE block in its place (E8).
Nothing is written: v0 never creates the reader cache directory.

The run's data. Each answered run's state holds `answer` (the text built from the segments), `v0` {the ScoredAnswer, the NOTAM ids named that were not in the prompt, the usage
of the model's reply} and no `guard`. Every logged run carries the metadata `source: v0` (ticket 7: never `pilot`) and the ids the model named that were not in the prompt
(`unlisted_notam_ids`); a NOTAM the prompt held and the output did not name counts as not surfaced, as for v1. The experiment's metadata holds REPORT_LINE.

Scoring. `score` runs the evaluators of evals.agent.evaluators.aggregate.EVALUATORS over the answered runs. The judge (PR-7.F1) is not built for v0: v0's text is code-built, so each
of its criteria is null for it and no JUDGE_* setting is needed. An evaluator that cannot apply to v0 gives None for every run; its mean is "n/a", never 0%. The feedback of each
run goes to LangSmith (judge.send_feedback), None scores left out. The results file, `make eval-v0` and the token and cost report are PR-10.F1's.

The order of `run_split`, each step before the next: the loader (a missing fixture or a stale sheet stops the run), the settings (AGENT_* and LANGSMITH_API_KEY; one message names
every one that is missing, before any client, model or example), the LangSmith client and the dataset upload, then the runner. The split is dev unless `--split held_out` is typed.
The model, the client and the clocks are arguments, so a test runs the whole command on a fake chat model and a recording client with no key.
"""

import argparse
import os
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel

from evals.agent import runner
from evals.agent.evaluators.aggregate import EVALUATORS, Evaluation, ScoredRun, evaluate_runs
from evals.agent.evaluators.answer import ScoredAnswer
from evals.agent.import_dataset import DATA_DIR, DatasetError
from evals.agent.judge import send_feedback
from evals.agent.loader import DEFAULT_SPLIT, EXPECTED, REPO, SPLITS, load_dataset, upload_dataset
from evals.agent.v0_output import V0Output, to_scored_answer
from evals.agent.v0_prompt import CaseFiles, build_v0_messages
from flysafelz.agent.config import LANGSMITH_KEY, AgentConfigError, agent_config
from flysafelz.sources import load_case_sources
from flysafelz.verdict_sheet import apply_drops

PROGRAM = "evals.agent.v0"
AGENT = "v0"
SOURCE = "v0"  # ticket 7: the `source` of every v0 trace; never `pilot`
ASSESS = "assess"
FAILED_EXIT = 1
INTERRUPTED_EXIT = 130
REPORT_LINE = "v1's advantage covers reading plus arithmetic; not separated (v0b, v0 with reader output, is the named next experiment)"
NO_STATE = "v0"  # the key of v0's own data in a run's state


class V0ReplyError(Exception):
    """The model's reply did not fit the schema, or came with no parsed output. The runner counts it as a failed attempt."""


def missing_settings(environ: Mapping[str, str]) -> list[str]:
    """What is wrong with the settings, one entry per variable: every missing AGENT_* (and a bad AGENT_BASE_URL) and LANGSMITH_API_KEY. Builds nothing."""
    problems: list[str] = []
    try:
        agent_config(environ)
    except AgentConfigError as error:
        problems += [*error.missing, *(f"{name} (must be an http:// or https:// URL)" for name in error.invalid)]
    if not environ.get(LANGSMITH_KEY, "").strip():
        problems.append(LANGSMITH_KEY)
    return problems


def assess_records(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """The examples v0 answers: `reference.path == "assess"`, in the loader's order."""
    return [dict(record) for record in records if (record.get("reference") or {}).get("path") == ASSESS]


class V0Graph:
    """The runner's `graph`: `invoke(request, config, context)` makes the one structured model call for the example the config names.

    The example is found by `config["metadata"]["example_id"]`, which the runner sets; its window and its question come from the record, its case and drops from the run context
    the runner built. Nothing is cached between calls: the case is loaded afresh each time, from the committed fixture folder.
    """

    def __init__(self, model: BaseChatModel, records: Sequence[Mapping[str, Any]], root: Path = REPO):
        self.records = {record["id"]: record for record in records}
        self.root = root
        self.structured = model.with_structured_output(V0Output, include_raw=True)

    def case_files(self, context: Any) -> CaseFiles:
        sources = load_case_sources(context.case, root=self.root, with_readings=False)
        forecast, airspace, pib = apply_drops(context.drop, context.now, sources.forecast, sources.airspace, sources.pib)
        return CaseFiles(forecast, airspace, pib)

    def invoke(self, request: Mapping[str, Any], config: Mapping[str, Any], *, context: Any) -> dict[str, Any]:
        example_id = (config.get("metadata") or {}).get("example_id")
        if example_id not in self.records:
            raise V0ReplyError(f"no example {example_id!r} to build a prompt for")
        prompt = build_v0_messages(self.records[example_id], self.case_files(context))
        # `langgraph_node` makes the runner's ModelCalls count this call's usage; `source` marks every trace of the call as v0's.
        metadata = {**(config.get("metadata") or {}), "langgraph_node": runner.AGENT_MODEL_NODE, "source": SOURCE}
        reply = self.structured.invoke(prompt.messages(), {**config, "metadata": metadata})  # type: ignore[arg-type]  # the runner's config is a RunnableConfig dict
        return self.state_of(reply, prompt.notam_ids)  # type: ignore[arg-type]  # include_raw gives a dict

    @staticmethod
    def state_of(reply: Mapping[str, Any], notam_ids: Sequence[str]) -> dict[str, Any]:
        parsed, error, raw = reply.get("parsed"), reply.get("parsing_error"), reply.get("raw")
        if error is not None or not isinstance(parsed, V0Output):
            raise V0ReplyError(f"the reply does not fit the schema: {error if error is not None else 'nothing was parsed'}")
        scored = to_scored_answer(parsed, notam_ids)
        usage = getattr(raw, "usage_metadata", None)
        return {
            "answer": scored.answer.text,
            "guard": None,
            NO_STATE: {"scored": scored.answer, "listed": scored.listed, "unlisted": scored.unlisted, "usage": dict(usage) if usage else None},
        }


def run_metadata(run: runner.RunRecord) -> dict[str, Any]:
    """What each logged run adds to its metadata: the source, and the NOTAM ids the model named that were not in the prompt (empty for an errored run)."""
    held = (run.output or {}).get(NO_STATE) or {}
    return {"source": SOURCE, "unlisted_notam_ids": list(held.get("unlisted", ()))}


def scored_runs(result: runner.ExperimentResult, records: Sequence[Mapping[str, Any]]) -> list[tuple[runner.RunRecord, ScoredRun]]:
    """Every run of the result with the run the evaluators take. An errored run is passed as errored, so it is listed and never scored."""
    by_id = {record["id"]: record for record in records}
    pairs = []
    for run in result.records:
        answer: ScoredAnswer | None = None if run.errored else run.output[NO_STATE]["scored"]  # type: ignore[index]  # an answered run has its state
        pairs.append((run, ScoredRun(by_id[run.example_id], answer, run.rep, run.errored)))
    return pairs


@dataclass
class V0Run:
    result: runner.ExperimentResult
    evaluation: Evaluation
    feedback_sent: int

    def summary_lines(self) -> list[str]:
        lines = self.result.summary_lines()
        lines.append("evaluators (mean over the runs each applied to):")
        lines += [f"  {key}: {self.evaluation.mean_text(key)} ({self.evaluation.applied(key)} runs)" for key in EVALUATORS]
        lines.append(f"feedback items sent to LangSmith: {self.feedback_sent}")
        lines.append(REPORT_LINE)
        return lines


def run_split(
    split: str = DEFAULT_SPLIT,
    *,
    experiment: str | None = None,
    environ: Mapping[str, str] | None = None,
    model_factory: Callable[[Mapping[str, str]], BaseChatModel] | None = None,
    client_factory: Callable[[Mapping[str, str]], Any] | None = None,
    data_dir: Path = DATA_DIR,
    root: Path = REPO,
    expected: Mapping[str, int] = EXPECTED,
    cache_dir: Path = runner.CACHE_DIR,
    moment: datetime | None = None,
    make_target: str | None = None,
    **runner_options: Any,
) -> V0Run:
    """The steps in the module text. `runner_options` go to run_experiment (sleep, clock, timeout, reps, ...).

    `make_target` is the make target that started the run (`make eval-v0` passes "eval-v0"): the missing-settings message then names that target and what it loads, instead of this module's own command.
    """
    environ = os.environ if environ is None else environ
    dataset = load_dataset(split, data_dir=data_dir, root=root, expected=expected)
    problems = missing_settings(environ)
    if problems:
        how = (
            f"AGENT_* and LANGSMITH_API_KEY come from the environment only; to use a .env, run `uv run --env-file .env python -m {PROGRAM}`"
            if make_target is None
            else f"AGENT_* and LANGSMITH_API_KEY come from the environment or from a .env in the repo root, which `make {make_target}` loads when the file exists; set them there or export them, then run `make {make_target}` again"
        )
        raise runner.RunnerError(f"missing settings: {', '.join(problems)} ({how}); nothing was built, sent or written")
    records = assess_records(dataset.records)
    name = experiment or runner.default_experiment_name(split, moment, AGENT)
    runner.cache_path_for(name, cache_dir)  # refuses a bad name before any client exists; it creates nothing
    client = (client_factory or default_client_factory)(environ)
    upload_dataset(split, client, data_dir=data_dir, root=root, expected=expected)
    model = (model_factory or default_model_factory)(environ)
    options = {"require_usage": False, **runner_options}  # PR-10.F1: `make eval-v0` passes require_usage=True and strict_usage=True (evals.agent.eval_run)
    result = runner.run_experiment(
        records,
        experiment=name,
        split=split,
        graph_factory=lambda: V0Graph(model, records, root),
        client=client,
        cache_dir=cache_dir,
        agent=AGENT,
        project_metadata={"report_line": REPORT_LINE, "source": SOURCE},
        run_metadata=run_metadata,
        **options,
    )
    pairs = scored_runs(result, records)
    evaluation = evaluate_runs(scored for _run, scored in pairs)
    sent = 0
    for (run, _scored), scores in zip((pair for pair in pairs if not pair[0].errored), evaluation.runs, strict=True):
        if run.langsmith_run_id is not None:
            sent += send_feedback(client, run.langsmith_run_id, scores.feedback, project=name)
    return V0Run(result, evaluation, sent)


def default_model_factory(environ: Mapping[str, str]) -> BaseChatModel:
    """The configured AGENT_* chat model. Only called after the settings are checked. Building it opens no connection."""
    from flysafelz.agent.config import build_chat_model

    return build_chat_model(agent_config(environ))


def default_client_factory(environ: Mapping[str, str]) -> Any:
    """The real LangSmith client with the phone anonymiser. Only called after the settings are checked, so the key is there."""
    from flysafelz.agent.tracing import build_client

    client = build_client(environ)
    if client is None:
        raise DatasetError(f"{LANGSMITH_KEY} is not set")
    return client


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog=f"python -m {PROGRAM}", description="Run the v0 baseline (one structured model call per assess example) over a split: 3 reps, one example at a time.")
    parser.add_argument("--split", default=DEFAULT_SPLIT, choices=list(SPLITS), help="the split to run; dev unless held_out is named")
    parser.add_argument("--experiment", default=None, help="the experiment name (default v0-<split>-<UTC stamp>)")
    return parser


def main(argv: Sequence[str] | None = None, *, environ: Mapping[str, str] | None = None, stdout: Any = None, stderr: Any = None, **options: Any) -> int:
    """The command. `options` are run_split's keyword arguments (model_factory, client_factory, data_dir, root, cache_dir, sleep, ...), for tests."""
    out = sys.stdout if stdout is None else stdout
    err = sys.stderr if stderr is None else stderr
    args = _parser().parse_args(argv)
    try:
        run = run_split(args.split, experiment=args.experiment, environ=environ, **options)
    except runner.FatalRunError as error:
        print(f"{PROGRAM}: stopped: {error}", file=err)
        return FAILED_EXIT
    except (DatasetError, runner.RunnerError) as error:
        print(f"{PROGRAM}: {error}", file=err)
        return FAILED_EXIT
    for line in run.summary_lines():
        print(line, file=out)
    return INTERRUPTED_EXIT if run.result.interrupted else 0


if __name__ == "__main__":
    raise SystemExit(main())
