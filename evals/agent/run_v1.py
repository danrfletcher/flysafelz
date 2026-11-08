"""Run v1 over a split (5a PR-4.F1, GP2, GP9): the loader, then the runner. PR-10 puts `make eval` behind this.

    python -m evals.agent.run_v1 [--split dev] [--experiment NAME]

    run_split(split, ...) -> ExperimentResult    the whole run, for PR-10 to score; raises DatasetError, RunnerError (FatalRunError, CacheExistsError)
    main(argv, ...) -> int                       the command: 0 a run that finished (an `incomplete` one is said so), 1 anything that stopped it, 130 an interrupt

The split is dev unless `--split held_out` is typed; nothing else chooses it (no environment variable, no default that changes).

The order, each step before the next (the first two before any client is built, any graph call or any cache file):
  1. the loader (evals.agent.loader.load_dataset): a missing fixture folder, a stale or unfinalised sheet or a wrong count stops the run, its message as it is;
  2. the settings: AGENT_* (the model), READER_* (the NOTAM reader) and LANGSMITH_API_KEY must be set. One message names every one that is not. Only READER_*
     is also read from .env (reader/config.py); the agent reads AGENT_* from the process environment alone (agent/config.py), so a run that keeps its keys in .env is started as
     `uv run --env-file .env python -m evals.agent.run_v1`, the way `make run` does. PR-10's `make eval` must do the same;
  3. the experiment's cache path: a file already there stops the run, naming it;
  4. the LangSmith client (tracing.build_client, with the phone anonymiser) and the split's dataset upload (loader.upload_dataset);
  5. the runner (evals.agent.runner.run_experiment): rep 1, then rep 2, then rep 3.
A stop after step 1 prints one line on stderr: the cause and the last example id (GP9). Nothing is written to src/, to the dataset copy or to the dataset's source folder; what
the run did write is its cache file and the LangSmith experiment. The records so far are on the error's `.partial` for a caller that wants them.

The model, the client, the sleeper and the clocks are arguments, so a test runs the whole command on a fake graph and a recording client with no key.
"""

import argparse
import os
import sys
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

from evals.agent import runner
from evals.agent.import_dataset import DATA_DIR, DatasetError
from evals.agent.loader import DEFAULT_SPLIT, EXPECTED, REPO, SPLITS, load_dataset, upload_dataset
from flysafelz.agent.config import LANGSMITH_KEY, AgentConfigError, agent_config
from flysafelz.reader.config import ReaderConfigError, reader_config

PROGRAM = "evals.agent.run_v1"
FAILED_EXIT = 1
INTERRUPTED_EXIT = 130  # the shell's number for a run ended by SIGINT


def missing_settings(environ: Mapping[str, str]) -> list[str]:
    """What is wrong with the settings, one entry per variable: every missing AGENT_*, READER_* and LANGSMITH_API_KEY, and an AGENT_BASE_URL that is not a URL. Builds nothing."""
    problems: list[str] = []
    try:
        agent_config(environ)
    except AgentConfigError as error:
        problems += [*error.missing, *(f"{name} (must be an http:// or https:// URL)" for name in error.invalid)]
    try:
        reader_config(environ)
    except ReaderConfigError as error:
        problems += error.missing
    if not environ.get(LANGSMITH_KEY, "").strip():
        problems.append(LANGSMITH_KEY)
    return problems


def default_graph_factory() -> Any:
    """The real agent graph (ticket 4's build_graph), imported only when a run reaches it."""
    from flysafelz.agent.graph import build_graph

    return build_graph()


def default_client_factory(environ: Mapping[str, str]) -> Any:
    """The real LangSmith client with the phone anonymiser. Only called after the settings are checked, so the key is there."""
    from flysafelz.agent.tracing import build_client

    client = build_client(environ)
    if client is None:
        raise DatasetError(f"{LANGSMITH_KEY} is not set")
    return client


def run_split(
    split: str = DEFAULT_SPLIT,
    *,
    experiment: str | None = None,
    environ: Mapping[str, str] | None = None,
    graph_factory: Callable[[], Any] = default_graph_factory,
    client_factory: Callable[[Mapping[str, str]], Any] = default_client_factory,
    data_dir: Path = DATA_DIR,
    root: Path = REPO,
    expected: Mapping[str, int] = EXPECTED,
    cache_dir: Path = runner.CACHE_DIR,
    moment: datetime | None = None,
    **runner_options: Any,
) -> runner.ExperimentResult:
    """The five steps in the module doc. `runner_options` go to run_experiment (sleep, clock, timeout, ...)."""
    environ = os.environ if environ is None else environ
    dataset = load_dataset(split, data_dir=data_dir, root=root, expected=expected)
    problems = missing_settings(environ)
    if problems:
        raise runner.RunnerError(
            f"missing settings: {', '.join(problems)} (AGENT_* and LANGSMITH_API_KEY come from the environment only, this command does not read .env; to use one, run `uv run --env-file .env python -m {PROGRAM}`, as `make run` does); nothing was built, sent or written"
        )
    name = experiment or runner.default_experiment_name(split, moment)
    runner.cache_path_for(name, cache_dir)  # refuses a name or an existing file before any client exists
    client = client_factory(environ)
    upload_dataset(split, client, data_dir=data_dir, root=root, expected=expected)
    return runner.run_experiment(dataset.records, experiment=name, split=split, graph_factory=graph_factory, client=client, cache_dir=cache_dir, **runner_options)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog=f"python -m {PROGRAM}", description="Run the v1 agent over a split of the eval dataset: 3 reps, one example at a time.")
    parser.add_argument("--split", default=DEFAULT_SPLIT, choices=list(SPLITS), help="the split to run; dev unless held_out is named")
    parser.add_argument("--experiment", default=None, help="the experiment name (default v1-<split>-<UTC stamp>); it names the reader cache file, which must not exist")
    return parser


def main(argv: Sequence[str] | None = None, *, environ: Mapping[str, str] | None = None, stdout: Any = None, stderr: Any = None, **options: Any) -> int:
    """The command. `options` are run_split's keyword arguments (graph_factory, client_factory, data_dir, root, cache_dir, sleep, ...), for tests."""
    out = sys.stdout if stdout is None else stdout
    err = sys.stderr if stderr is None else stderr
    args = _parser().parse_args(argv)
    try:
        result = run_split(args.split, experiment=args.experiment, environ=environ, **options)
    except runner.FatalRunError as error:
        print(f"{PROGRAM}: stopped: {error}", file=err)
        return FAILED_EXIT
    except (DatasetError, runner.RunnerError) as error:
        print(f"{PROGRAM}: {error}", file=err)
        return FAILED_EXIT
    for line in result.summary_lines():
        print(line, file=out)
    return INTERRUPTED_EXIT if result.interrupted else 0


if __name__ == "__main__":
    raise SystemExit(main())
