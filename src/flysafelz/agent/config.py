"""The agent's model configuration: the only place in src/ that reads `AGENT_*` (the agent's counterpart of reader/config.py).

    agent_config()      -> AgentConfig   the three AGENT_* values, or AgentConfigError naming every missing or bad one
    build_chat_model()  -> ChatOpenAI    a client for that endpoint; nothing is sent until it is invoked
    LazyChatModel       a chat model that calls build_chat_model() the first time a model call needs it, not when the graph is built

Nothing is read on import: the environment is read each time `agent_config()` is called, so importing this module has no
side effect and a test can set the variables first. Only the process environment is read, never a `.env` file (the
LangGraph server loads that into the environment before the graph is built).

Each of AGENT_BASE_URL, AGENT_MODEL and AGENT_API_KEY must be set and not blank. Nothing falls back to a default model
or URL. The key is always passed to ChatOpenAI explicitly, so the client never looks for another key in the environment.
No call is made to the endpoint's model list.
"""

import os
from collections.abc import Mapping, Sequence
from typing import Any
from urllib.parse import urlsplit

from langchain_core.callbacks import AsyncCallbackManagerForLLMRun, CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.outputs import ChatResult
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, ConfigDict, PrivateAttr, SecretStr

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


class LazyChatModel(BaseChatModel):
    """The graph's model when no model is injected: the configured ChatOpenAI, built the first time a model call needs it.

    A graph builds without any AGENT_* setting, and a turn that never reaches the model (a request outside the UK, a rejected one) never
    needs them. The first `bind_tools` or generate call builds the client with `build_chat_model()`, so a missing or bad setting raises
    AgentConfigError there, in one line naming it, and the client is then reused. `bind_tools` returns the real client's own bound model.
    """

    _real: ChatOpenAI | None = PrivateAttr(default=None)

    @property
    def _llm_type(self) -> str:
        return "flysafelz-lazy-chat-model"

    def _model(self) -> ChatOpenAI:
        if self._real is None:
            self._real = build_chat_model()
        return self._real

    def bind_tools(self, tools: Sequence[Any], *, tool_choice: str | None = None, **kwargs: Any) -> Any:
        return self._model().bind_tools(tools, tool_choice=tool_choice, **kwargs)

    def _generate(self, messages: list[BaseMessage], stop: list[str] | None = None, run_manager: CallbackManagerForLLMRun | None = None, **kwargs: Any) -> ChatResult:
        return self._model()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)

    async def _agenerate(self, messages: list[BaseMessage], stop: list[str] | None = None, run_manager: AsyncCallbackManagerForLLMRun | None = None, **kwargs: Any) -> ChatResult:
        return await self._model()._agenerate(messages, stop=stop, run_manager=run_manager, **kwargs)
