"""The reader's model configuration: the only place in src/ that reads `READER_*`, the environment or a `.env` file (F2, F5).

    reader_config()  -> ReaderConfig       the four READER_* values, or ReaderConfigError naming the missing ones
    build_llm()      -> ChatOpenAI         a client for that endpoint; nothing is sent until it is invoked

Nothing is read on import. The values are read the first time `reader_config()` (or `build_llm()`) is called, so
`import flysafelz.reader` has no side effect and a test can point the reader somewhere else first.

Each of READER_PROVIDER, READER_BASE_URL, READER_API_KEY and READER_MODEL comes from `os.environ`, else from the `.env`
file. A blank value counts as not set, so an empty line in `.env.example` or an empty export never masks the other source.
Nothing falls back to a default: a missing variable is an error that lists exactly the missing names, in that order.

The `.env` file is the repository root's, unless READER_DOTENV_PATH (environment only, never the file) names another.
tests/conftest.py points it at a file that does not exist, so no test can load a real `.env`. A missing file is an empty one.
The parser is the few lines below: KEY=VALUE per line, `#` comments, blank lines, an optional `export `, matching quotes
around a value (text after the closing quote is dropped). Keys other than READER_* are ignored. python-dotenv is not used (the test setup blocks `import dotenv`).
"""

import os
from collections.abc import Mapping
from pathlib import Path

from langchain_openai import ChatOpenAI
from pydantic import BaseModel, SecretStr

from flysafelz.contracts import STRICT

PROVIDER = "READER_PROVIDER"
BASE_URL = "READER_BASE_URL"
KEY = "READER_API_KEY"
MODEL = "READER_MODEL"
READER_VARIABLES = (PROVIDER, BASE_URL, KEY, MODEL)
DOTENV_PATH = "READER_DOTENV_PATH"
_COMMENT = " #"  # an unquoted value ends at a space and a hash
_EXPORT = "export "
_QUOTES = "\"'"


def _repository_root() -> Path:
    """The nearest folder above this file that holds pyproject.toml."""
    return next(folder for folder in Path(__file__).resolve().parents if (folder / "pyproject.toml").is_file())


class ReaderConfigError(ValueError):
    """One or more READER_* variables are not set. `missing` has their names, in the order PROVIDER, BASE_URL, API_KEY, MODEL."""

    def __init__(self, missing: tuple[str, ...]):
        self.missing = missing
        super().__init__(f"reader settings missing: {', '.join(missing)} (set them in the environment or in .env)")


class ReaderConfig(BaseModel):
    """The endpoint the reader talks to. The key is a SecretStr, so it never shows in a repr, a log line or a traceback."""

    model_config = STRICT

    provider: str
    base_url: str
    api_key: SecretStr
    model: str


def parse_env_file(text: str) -> dict[str, str]:
    """The READER_* assignments in the text of a `.env` file. Comments, blank lines and every other key are skipped."""
    values: dict[str, str] = {}
    for raw in text.splitlines():
        line = raw.strip().removeprefix(_EXPORT).strip()
        name, equals, value = line.partition("=")
        name = name.strip()
        if not equals or line.startswith("#") or name not in READER_VARIABLES:
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
    """The READER_* values in the file. No file there (or a folder) is an empty file."""
    if not path.is_file():
        return {}
    return parse_env_file(path.read_text(encoding="utf-8"))


def _dotenv_path(environ: Mapping[str, str]) -> Path:
    named = environ.get(DOTENV_PATH, "").strip()
    return Path(named) if named else _repository_root() / ".env"


def reader_config(environ: Mapping[str, str] | None = None) -> ReaderConfig:
    """The reader's four settings: the environment first, then the `.env` file. Raises ReaderConfigError for any not set.

    `environ` is os.environ unless a test passes its own.
    """
    environ = os.environ if environ is None else environ
    from_file = read_env_file(_dotenv_path(environ))
    found: dict[str, str] = {}
    for name in READER_VARIABLES:
        value = next((v.strip() for v in (environ.get(name), from_file.get(name)) if v and v.strip()), "")
        if value:
            found[name] = value
    missing = tuple(name for name in READER_VARIABLES if name not in found)
    if missing:
        raise ReaderConfigError(missing)
    return ReaderConfig(provider=found[PROVIDER], base_url=found[BASE_URL], api_key=SecretStr(found[KEY]), model=found[MODEL])


def build_llm(config: ReaderConfig | None = None) -> ChatOpenAI:
    """A ChatOpenAI client for the configured endpoint. Building it opens no connection.

    The client does not retry on its own (max_retries is 0): read.py makes the one repair and treats a failed call as
    unavailable, and a hidden retry would double the wait past `timeout_s`.
    """
    config = reader_config() if config is None else config
    return ChatOpenAI(model=config.model, base_url=config.base_url, api_key=config.api_key, max_retries=0)
