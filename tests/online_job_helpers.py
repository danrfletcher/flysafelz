"""The fixtures `make online-eval`'s tests share (7 PR-4.F1): a project of fake runs on the online recording client, and the environment and clock the job is started with.

    world(extra=...)     (client, trees): 11 seed runs (4 assess, 3 explain, 2 clarify, 2 decline), 2 pilot runs, and the runs the job must skip: 1 `case`, 1 `judge`, 1 errored, 1 with a `reference_example_id`
    job_env(**kw)        the environment of the job: the project and the four JUDGE_* values (fake: no judge is built from them), and no LANGSMITH key (the client is passed in)
    run_job(client, results_dir, ...)   `job.main` on that client with a fixed clock and a fake judge; returns (exit code, the judge)
    results(results_dir) the files in the folder
"""

import uuid
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from online_helpers import FakeTree, OnlineRecordingClient
from online_judge_helpers import FakeJudge, judge_client, judge_tree

from evals.online import job

PROJECT = "flysafelz"
CLOCK = datetime(2026, 10, 31, 8, 0, tzinfo=UTC)  # an hour and a half after the fake runs started (the conftest `now`, 07:30, less the hour the runs are given); GMT, so the file is online-2026-10-31.md
SEED_PATHS = ["assess"] * 4 + ["explain"] * 3 + ["clarify"] * 2 + ["decline"] * 2
LZ_NAMES = ["Blue Mill", "West Sleekburn", "Druridge Bay", "Buttermere", "Crookham"]
FICTION_NUMBER = "01632 960123"  # the fiction range: never a real number
NO_DOTENV = {"READER_DOTENV_PATH": "/no-such-dotenv/.env"}
JUDGE_ENV = {"JUDGE_PROVIDER": "fake", "JUDGE_BASE_URL": "http://judge.invalid/v1", "JUDGE_API_KEY": "not-a-key", "JUDGE_MODEL": "fake-model"}


def job_env(**kw: str) -> dict[str, str]:
    return {"LANGSMITH_PROJECT": PROJECT, **NO_DOTENV, **JUDGE_ENV, **kw}


def seed_trees() -> list[FakeTree]:
    return [judge_tree(f"seed-{index}", path=path, source="online-seed", lz_name=LZ_NAMES[index % 5]) for index, path in enumerate(SEED_PATHS)]


def pilot_trees() -> list[FakeTree]:
    return [judge_tree("pilot-0", source="pilot"), judge_tree("pilot-1", source="pilot", path="explain")]


def skipped_trees() -> list[FakeTree]:
    return [
        judge_tree("case-0", source="case"),
        judge_tree("judge-0", source="judge"),
        judge_tree("errored-0", source="pilot", error="GraphRecursionError: step limit"),
        judge_tree("experiment-0", source="pilot", reference_example_id=uuid.UUID(int=7)),
    ]


def world(extra: list[FakeTree] | None = None, *, trees: list[FakeTree] | None = None, **client_kw: Any) -> tuple[OnlineRecordingClient, list[FakeTree]]:
    """The client over the 17 runs of GP2 (and any `extra` trees); `trees` replaces all of them."""
    everything = trees if trees is not None else [*seed_trees(), *pilot_trees(), *skipped_trees(), *(extra or [])]
    return judge_client(everything, **client_kw), everything


def run_job(client: OnlineRecordingClient, results_dir: Path, *, env: Mapping[str, str] | None = None, clock: datetime = CLOCK, judge: FakeJudge | None = None) -> tuple[int, FakeJudge]:
    fake = judge or FakeJudge()
    code = job.main([], environ=job_env() if env is None else env, client=client, clock=lambda: clock, build=lambda _env: fake, results_dir=results_dir)
    return code, fake


def results(results_dir: Path) -> list[str]:
    return sorted(path.name for path in results_dir.iterdir()) if results_dir.exists() else []


def posted_keys(client: OnlineRecordingClient) -> dict[Any, list[str]]:
    found: dict[Any, list[str]] = {}
    for item in client.feedback:
        found.setdefault(item.run_id, []).append(item.key)
    return found


def expected_keys(path: str) -> list[str]:
    return [*job.DETERMINISTIC_KEYS, *job.judge_keys_for(path), job.VERSION_KEY]

