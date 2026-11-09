"""Which production runs the online job scores (7 PR-1.F1, G3, G4, G5, E1, E3, E7). It decides; it scores and posts nothing.

    guard_project(name=None)               the project to read, or exit 2 when it is 5a's held-out or dev experiment project
    parse_since(text) / parse_limit(text)  `SINCE` (`7d`, `24h`) as a timedelta and `LIMIT` as a count; exit 2, naming the variable, for anything else
    settings_from_env(environ=None)        guard, SINCE and LIMIT from the environment, in that order; PullSettings
    pull_runs(client, project, since, limit, now) -> PullResult   list the root runs once, drop the ones not to score, count why
    pull_from_env(make_client, now, environ=None) -> PullResult   settings first, then the client, then the pull: the client is not built on a refusal

Everything is a pure function over a client that is passed in. Nothing here builds a LangSmith client (the factory is `make_client`'s caller's, in job.py) or opens a socket.

The pull (G3). One `client.list_runs(project_name=project, is_root=True, start_time=now - since)`. The filter is the client's: a run that starts exactly at `now - since` is
returned and kept, and no time is checked again here. `limit` counts the runs read, before any is skipped; one more than `limit` is asked for, so that `limit_hit` is true only
when more were available (exactly `limit` runs in the window is not a hit). A run that has a `parent_run_id` is not a root run: a client that returns one has misbehaved, and the run is dropped
(counted in `dropped_children`, never scored, never skipped-and-counted).

The allow-list (G4). Only a root run whose metadata `source` is `pilot` or `online-seed` is scored. A run is skipped for exactly one reason, the first that applies, in this fixed order:
    1. `experiment`  it has a `reference_example_id` (an eval run), whatever its `source`
    2. `source`      its `source` is not on the allow-list: `case`, `judge`, `v0`, any other value (its `str()`, so a non-string counts as its text), or `missing`
                     when there is no `source` key (a `None` or blank value counts as no key)
    3. `errored`     it has an `error`, or no final answer in its outputs (E3)
Each run is counted once, under its first reason. `PullResult.by_reason` counts every skipped run by reason; `by_source` splits the `source` reason by the source's name.
A run that is skipped is never scored and gets no reference and no feedback.

The project guard (G5, E7). The online job never reads eval experiments or the held-out project (F2). `guard_project` refuses, with exit code 2 and a line on stderr that names the project,
when the project is the held-out project or the dev experiment project of 5a's config (`evals.agent.loader.SPLITS`, never written out here). Both sides are normalised first (surrounding
spaces off, case folded), so the held-out name in capitals, or with spaces round it, is refused too. With `LANGSMITH_PROJECT` unset or blank the project is the agent's own default (`langsmith_project` in settings.yaml).
`pull_runs` runs the same guard on the project it is given, so no caller can reach a protected project through it, and it lists only the guarded name.
"""

import itertools
import os
import re
import sys
from collections import Counter
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, NoReturn

from evals.agent.loader import experiment_target
from evals.online.runs import SOURCE_KEY, final_answer, metadata_of
from flysafelz.agent.settings import SETTINGS
from flysafelz.agent.tracing import PILOT_SOURCE, SEED_SOURCE

PROJECT_VARIABLE = "LANGSMITH_PROJECT"
SINCE_VARIABLE = "SINCE"
LIMIT_VARIABLE = "LIMIT"
DEFAULT_SINCE = "7d"
DEFAULT_LIMIT = "500"
USAGE_EXIT = 2
ALLOWED_SOURCES = (PILOT_SOURCE, SEED_SOURCE)  # ticket 4's `source` values that are real traffic (I6)
MISSING_SOURCE = "missing"
EXPERIMENT, SOURCE, ERRORED = "experiment", "source", "errored"  # the skip reasons, in the order they are checked
REASONS = (EXPERIMENT, SOURCE, ERRORED)
PROTECTED_SPLITS = {"held_out": "held-out", "dev": "dev experiment"}  # 5a's splits whose project is never read, and what each is called in the message
SINCE_PATTERN = re.compile(r"([0-9]+)([dh])")
LIMIT_PATTERN = re.compile(r"[0-9]+")
UNITS = {"d": timedelta(days=1), "h": timedelta(hours=1)}


def _refuse(message: str) -> NoReturn:
    print(message, file=sys.stderr)
    raise SystemExit(USAGE_EXIT)


# --- the project guard (G5) -----------------------------------------------------------------------------------------------------------------


def _normal(name: str) -> str:
    return name.strip().casefold()


def protected_projects() -> dict[str, str]:
    """{normalised project name: what it is called in the refusal} for 5a's held-out project and its dev experiment project, read from 5a's config."""
    return {_normal(experiment_target(split).project): label for split, label in PROTECTED_SPLITS.items()}


def guard_project(name: str | None = None, environ: Mapping[str, str] | None = None) -> str:
    """The project to read: `name`, else LANGSMITH_PROJECT, else the agent's default. Exits 2, naming the project, when it is a protected one. Returns the name without surrounding spaces."""
    if name is None:
        name = (os.environ if environ is None else environ).get(PROJECT_VARIABLE, "")
    project = name.strip() or SETTINGS.langsmith_project
    kind = protected_projects().get(_normal(project))
    if kind is not None:
        _refuse(
            f"Refusing to read the LangSmith project {project}: it is the {kind} project of the offline eval, and the online job never reads eval experiments or the held-out project. Set {PROJECT_VARIABLE} to the production tracing project."
        )
    return project


# --- SINCE and LIMIT (G3) -------------------------------------------------------------------------------------------------------------------


def parse_since(text: str) -> timedelta:
    """`7d` or `24h` as a duration: a whole number above zero and `d` or `h`. Anything else (`7`, `7w`, `-1d`, `0d`, empty) exits 2 and names SINCE."""
    found = SINCE_PATTERN.fullmatch(text.strip())
    if found is None or int(found.group(1)) == 0:
        _refuse(f"{SINCE_VARIABLE}={text!r} is not a duration: use a whole number above zero and d or h, for example 7d or 24h.")
    return int(found.group(1)) * UNITS[found.group(2)]


def parse_limit(text: str) -> int:
    """The most runs to read: a whole number above zero. Anything else (`0`, `-5`, `abc`, empty) exits 2 and names LIMIT."""
    stripped = text.strip()
    if LIMIT_PATTERN.fullmatch(stripped) is None or int(stripped) == 0:
        _refuse(f"{LIMIT_VARIABLE}={text!r} is not a count: use a whole number above zero, for example 500.")
    return int(stripped)


@dataclass(frozen=True)
class PullSettings:
    project: str
    since: timedelta
    limit: int


def settings_from_env(environ: Mapping[str, str] | None = None) -> PullSettings:
    """The project (guarded), SINCE (default 7d) and LIMIT (default 500) from the environment. Exits 2 on a protected project or a bad value, before any client exists."""
    env = os.environ if environ is None else environ
    project = guard_project(environ=env)
    return PullSettings(project, parse_since(env.get(SINCE_VARIABLE, DEFAULT_SINCE)), parse_limit(env.get(LIMIT_VARIABLE, DEFAULT_LIMIT)))


# --- the pull (G3, G4) ----------------------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class SkippedRun:
    run: Any
    reason: str  # one of REASONS
    source: str  # the run's `source` as text, or `missing`


@dataclass
class PullResult:
    project: str
    since: timedelta
    limit: int
    runs: list[Any] = field(default_factory=list)  # the runs to score, in the order the client gave them
    skipped: list[SkippedRun] = field(default_factory=list)
    limit_hit: bool = False
    read: int = 0  # root runs read, before any was skipped (at most `limit`)
    dropped_children: int = 0  # runs the client returned that have a parent: not root runs, never scored

    @property
    def by_reason(self) -> Counter[str]:
        """Skipped runs counted by reason (`experiment`, `source`, `errored`); each run once."""
        return Counter(item.reason for item in self.skipped)

    @property
    def by_source(self) -> Counter[str]:
        """The runs skipped for the `source` reason, counted by source (`case`, `judge`, `v0`, `missing`, any other value under its own name)."""
        return Counter(item.source for item in self.skipped if item.reason == SOURCE)


def source_of(run: Any) -> str:
    """The run's metadata `source` as text; `missing` when there is no such key, or its value is None or blank."""
    value = metadata_of(run).get(SOURCE_KEY)
    text = "" if value is None else str(value).strip()
    return text or MISSING_SOURCE


def skip_reason(run: Any) -> str | None:
    """Why the run is not scored: the first of `experiment`, `source`, `errored` that applies, or None when it is scored."""
    if getattr(run, "reference_example_id", None) is not None:
        return EXPERIMENT
    if source_of(run) not in ALLOWED_SOURCES:
        return SOURCE
    if getattr(run, "error", None) or not final_answer(run):
        return ERRORED
    return None


def _classify(rows: Iterable[Any], result: PullResult) -> None:
    for run in rows:
        if getattr(run, "parent_run_id", None) is not None:
            result.dropped_children += 1
            continue
        reason = skip_reason(run)
        if reason is None:
            result.runs.append(run)
        else:
            result.skipped.append(SkippedRun(run, reason, source_of(run)))


def pull_runs(client: Any, project: str, since: timedelta, limit: int, now: datetime) -> PullResult:
    """Read up to `limit` root runs of `project` that started at or after `now - since` with one `list_runs` call, and split them into the runs to score and the runs skipped.

    `project` goes through the guard first, so a protected project exits 2 before the client is touched, and only the guarded name is sent. Zero runs is a result, not an error (E1).
    """
    guarded = guard_project(project)
    rows = list(itertools.islice(client.list_runs(project_name=guarded, is_root=True, start_time=now - since), limit + 1))
    result = PullResult(guarded, since, limit, limit_hit=len(rows) > limit)
    rows = rows[:limit]
    result.read = len(rows)
    _classify(rows, result)
    return result


def pull_from_env(make_client: Callable[[], Any], now: datetime, environ: Mapping[str, str] | None = None) -> PullResult:
    """The whole pull from the environment: the settings first (a refusal or a bad value exits 2 here), then `make_client()`, then `pull_runs`. `make_client` is called once, and not on an exit."""
    settings = settings_from_env(environ)
    return pull_runs(make_client(), settings.project, settings.since, settings.limit, now)
