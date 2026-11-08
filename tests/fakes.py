"""Test doubles for the agent: a scripted chat model that can ask for tools, and an embedder that needs no model file.

Neither opens a socket, reads `.env` or downloads anything, so a test using them runs under `--disable-socket`.

    FakeToolChatModel(script=[AIMessage, ...])   returns the scripted messages in order, one per model call; supports bind_tools
    FakeEmbedder()                               fastembed's embed / query_embed, with vectors derived from a hash of the text
"""

import hashlib
import struct
from collections.abc import Iterable, Iterator, Sequence
from typing import Any

import numpy as np
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import BaseTool
from pydantic import Field, PrivateAttr

EMBEDDING_DIMENSION = 384  # BAAI/bge-small-en-v1.5's size, so code written against the fake sees the real shape


class ScriptExhaustedError(AssertionError):
    """The code under test called the model more times than the test scripted."""


class FakeToolChatModel(BaseChatModel):
    """A chat model that returns the scripted AIMessages in order, one per call. A message may carry `tool_calls`.

    `bind_tools` accepts any tools and returns this same model (the script and the call count are shared), after recording
    their names in `bound_tool_names`. `calls` holds the messages each call was given, so a test can check what the agent sent.
    A call after the script is used up raises ScriptExhaustedError rather than inventing a reply.
    """

    script: list[AIMessage] = Field(default_factory=list)
    calls: list[list[BaseMessage]] = Field(default_factory=list)
    bound_tool_names: list[str] = Field(default_factory=list)
    _next: int = PrivateAttr(default=0)

    @property
    def _llm_type(self) -> str:
        return "fake-tool-chat-model"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        self.calls.append(list(messages))
        if self._next >= len(self.script):
            raise ScriptExhaustedError(f"the model was called {self._next + 1} times but the test scripted {len(self.script)}")
        message = self.script[self._next]
        self._next += 1
        return ChatResult(generations=[ChatGeneration(message=message)])

    def bind_tools(self, tools: Sequence[Any], *, tool_choice: str | None = None, **kwargs: Any) -> "FakeToolChatModel":
        """Record the tools' names and return this model: the fake never decides to call a tool, the script does."""
        for tool in tools:
            name = tool.name if isinstance(tool, BaseTool) else tool.get("name") if isinstance(tool, dict) else getattr(tool, "__name__", str(tool))
            self.bound_tool_names.append(str(name))
        return self


class FakeEmbedder:
    """Deterministic stand-in for fastembed's TextEmbedding: the same text always gives the same unit-length vector.

    The vector comes from SHA-256 of the text, so two different texts differ and nothing is downloaded. It has no meaning:
    similar texts are not close. A test of search order must choose texts for which that is enough, or use its own vectors.
    """

    dimension = EMBEDDING_DIMENSION

    def embed(self, documents: str | Iterable[str], **kwargs: Any) -> Iterator[np.ndarray]:
        """One float32 vector of length `dimension` per text, in order (a single string counts as one text)."""
        texts = [documents] if isinstance(documents, str) else list(documents)
        for text in texts:
            yield self._vector(text)

    query_embed = embed

    def _vector(self, text: str) -> np.ndarray:
        raw = b""
        counter = 0
        while len(raw) < self.dimension * 4:
            raw += hashlib.sha256(f"{counter}:{text}".encode()).digest()
            counter += 1
        words = struct.unpack(f">{self.dimension}I", raw[: self.dimension * 4])
        vector = np.array(words, dtype=np.float64)
        vector = vector / np.iinfo(np.uint32).max * 2 - 1  # each component in [-1, 1]
        return (vector / np.linalg.norm(vector)).astype(np.float32)
