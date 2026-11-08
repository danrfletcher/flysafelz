"""The judge's model settings (5a PR-7.F1, G12): the four JUDGE_* values, read the way `src/flysafelz/reader/config.py` reads READER_*.

    judge_config(environ=None) -> JudgeConfig      the four values, or JudgeConfigError naming every missing one, in order
    missing_judge_settings(environ=None) -> list   the same names as a list (empty when all four are set), for the run command's one settings message
    build_judge_llm(config=None) -> ChatOpenAI     a client for that endpoint (the bridge); nothing is sent until it is invoked

The rules are the reader's: each of JUDGE_PROVIDER, JUDGE_BASE_URL, JUDGE_API_KEY and JUDGE_MODEL comes from `os.environ` first, then from the `.env` file, and a
blank value counts as unset, so an empty line in `.env.example` never masks the other source. Nothing falls back to a default. The key is a SecretStr, so it never shows
in a repr, a log line or a traceback. Nothing is read on import.

The `.env` file is the one the reader uses: the repository root's, unless READER_DOTENV_PATH (environment only) names another, which is how tests/conftest.py points it at
a file that does not exist. The parser is the reader's few lines: KEY=VALUE per line, `#` comments, blank lines, an optional `export `, matching quotes around a value
(text after the closing quote is dropped), keys other than JUDGE_* ignored.

The error is raised by whoever starts the experiment (evals.agent.run_v1 checks it with the other settings before any example runs) and by `build_judge`; the judge never
runs on a partial configuration.
"""

import os
from collections.abc import Mapping
from pathlib import Path

from langchain_openai import ChatOpenAI
from pydantic import BaseModel, ConfigDict, SecretStr

from flysafelz.reader.config import _dotenv_path

PROVIDER = "JUDGE_PROVIDER"
BASE_URL = "JUDGE_BASE_URL"
KEY = "JUDGE_API_KEY"
MODEL = "JUDGE_MODEL"
JUDGE_VARIABLES = (PROVIDER, BASE_URL, KEY, MODEL)
JUDGE_TIMEOUT_S = 120.0  # one judge call; a call that takes longer is a judge error for that example, not a hang
_COMMENT = " #"  # an unquoted value ends at a space and a hash
_EXPORT = "export "
_QUOTES = "\"'"


class JudgeConfigError(ValueError):
    """One or more JUDGE_* variables are not set. `missing` has their names, in the order PROVIDER, BASE_URL, API_KEY, MODEL."""

    def __init__(self, missing: tuple[str, ...]):
        self.missing = missing
        super().__init__(f"judge settings missing: {', '.join(missing)} (set them in the environment or in .env)")


class JudgeConfig(BaseModel):
    """The endpoint the judge talks to. The key is a SecretStr, so it never shows in a repr, a log line or a traceback."""

    model_config = ConfigDict(extra="forbid")

    provider: str
    base_url: str
    api_key: SecretStr
    model: str


def parse_env_file(text: str) -> dict[str, str]:
    """The JUDGE_* assignments in the text of a `.env` file. Comments, blank lines and every other key are skipped."""
    values: dict[str, str] = {}
    for raw in text.splitlines():
        line = raw.strip().removeprefix(_EXPORT).strip()
        name, equals, value = line.partition("=")
        name = name.strip()
        if not equals or line.startswith("#") or name not in JUDGE_VARIABLES:
            continue
        value = value.strip()
        closing = value.find(value[0], 1) if value and value[0] in _QUOTES else -1
        if closing > 0:
            value = value[1:closing]  # a quoted value ends at its closing quote; what follows is a comment
        else:
            value = value.partition(_COMMENT)[0].strip()
        values[name] = value
    return values


def read_env_file(path: Path) -> dict[str, str]:
    """The JUDGE_* values in the file. No file there (or a folder) is an empty file."""
    if not path.is_file():
        return {}
    return parse_env_file(path.read_text(encoding="utf-8"))


def _found(environ: Mapping[str, str]) -> dict[str, str]:
    from_file = read_env_file(_dotenv_path(environ))
    found: dict[str, str] = {}
    for name in JUDGE_VARIABLES:
        value = next((v.strip() for v in (environ.get(name), from_file.get(name)) if v and v.strip()), "")
        if value:
            found[name] = value
    return found


def missing_judge_settings(environ: Mapping[str, str] | None = None) -> list[str]:
    """The JUDGE_* names that are not set (environment, then `.env`), in order. Empty when the judge can be built."""
    found = _found(os.environ if environ is None else environ)
    return [name for name in JUDGE_VARIABLES if name not in found]


def judge_config(environ: Mapping[str, str] | None = None) -> JudgeConfig:
    """The judge's four settings: the environment first, then the `.env` file. Raises JudgeConfigError naming every one not set.

    `environ` is os.environ unless a test passes its own.
    """
    found = _found(os.environ if environ is None else environ)
    missing = tuple(name for name in JUDGE_VARIABLES if name not in found)
    if missing:
        raise JudgeConfigError(missing)
    return JudgeConfig(provider=found[PROVIDER], base_url=found[BASE_URL], api_key=SecretStr(found[KEY]), model=found[MODEL])


def build_judge_llm(config: JudgeConfig | None = None) -> ChatOpenAI:
    """A ChatOpenAI client for the configured endpoint. Building it opens no connection.

    No retry (max_retries is 0): a failed call is that example's judge error, and no retry policy is invented here. `timeout` keeps a stuck call from holding the run.
    """
    config = judge_config() if config is None else config
    return ChatOpenAI(model=config.model, base_url=config.base_url, api_key=config.api_key, max_retries=0, timeout=JUDGE_TIMEOUT_S)
