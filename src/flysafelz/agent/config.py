"""The agent's model configuration: the only place in src/ that reads `AGENT_*` (the agent's counterpart of reader/config.py).

    agent_config()      -> AgentConfig   the three AGENT_* values, or AgentConfigError naming every missing or bad one
    build_chat_model()  -> ChatOpenAI    a client for that endpoint; nothing is sent until it is invoked

Nothing is read on import: the environment is read each time `agent_config()` is called, so importing this module has no
side effect and a test can set the variables first. Only the process environment is read, never a `.env` file (the
LangGraph server loads that into the environment before the graph is built).

Each of AGENT_BASE_URL, AGENT_MODEL and AGENT_API_KEY must be set and not blank. Nothing falls back to a default model
or URL. The key is always passed to ChatOpenAI explicitly, so the client never looks for another key in the environment.
No call is made to the endpoint's model list.
"""

import os
from collections.abc import Mapping
from urllib.parse import urlsplit

from langchain_openai import ChatOpenAI
from pydantic import BaseModel, ConfigDict, SecretStr

from flysafelz.agent.settings import SETTINGS

BASE_URL = "AGENT_BASE_URL"
MODEL = "AGENT_MODEL"
KEY = "AGENT_API_KEY"
AGENT_VARIABLES = (BASE_URL, MODEL, KEY)
_SCHEMES = ("http", "https")


class AgentConfigError(ValueError):
    """AGENT_* variables are missing or blank (`missing`), or AGENT_BASE_URL is not an http(s) URL (`invalid`). One message names them all."""

    def __init__(self, missing: tuple[str, ...], invalid: tuple[str, ...] = ()):
        self.missing = missing
        self.invalid = invalid
        parts = []
        if missing:
            parts.append(f"agent settings missing: {', '.join(missing)}")
        if invalid:
            parts.append(f"agent settings invalid: {', '.join(invalid)} must be an http:// or https:// URL")
        super().__init__("; ".join(parts) + " (set them in the environment)")


class AgentConfig(BaseModel):
    """The endpoint the agent talks to. The key is a SecretStr, so it never shows in a repr, a log line or a traceback."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    base_url: str
    model: str
    api_key: SecretStr


def _has_scheme_and_host(url: str) -> bool:
    parts = urlsplit(url)
    return parts.scheme in _SCHEMES and bool(parts.netloc)


def agent_config(environ: Mapping[str, str] | None = None) -> AgentConfig:
    """The agent's three settings from the environment. Raises AgentConfigError listing every one missing, blank or bad.

    `environ` is os.environ unless a test passes its own.
    """
    environ = os.environ if environ is None else environ
    found = {name: environ.get(name, "").strip() for name in AGENT_VARIABLES}
    missing = tuple(name for name in AGENT_VARIABLES if not found[name])
    invalid = (BASE_URL,) if found[BASE_URL] and not _has_scheme_and_host(found[BASE_URL]) else ()
    if missing or invalid:
        raise AgentConfigError(missing, invalid)
    return AgentConfig(base_url=found[BASE_URL], model=found[MODEL], api_key=SecretStr(found[KEY]))


def build_chat_model(config: AgentConfig | None = None) -> ChatOpenAI:
    """A ChatOpenAI client for the configured endpoint. Building it opens no connection.

    max_retries comes from settings.yaml (model_max_retries): the client retries a timeout or a bad reply that many times.
    """
    config = agent_config() if config is None else config
    return ChatOpenAI(base_url=config.base_url, model=config.model, api_key=config.api_key, max_retries=SETTINGS.model_max_retries)
