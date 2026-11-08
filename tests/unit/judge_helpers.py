"""Fakes and fixtures for the judge's tests (5a PR-7.F1). Nothing here calls a model or LangSmith, and no key is read.

dev_example(id)             a record of the committed DEV split, as the loader builds it (never a held-out one)
FakeJudgeModel              a chat model that returns canned replies and records what it was given; `with_structured_output` parses them with the schema, as a real one does
canned(**changes)           the JSON text of a full, valid reply (all four criteria passed), with `changes` per criterion: canned(prioritised=(False, "buried"))
FeedbackClient              a stand-in for the LangSmith client that records the `create_feedback` calls it gets
answer(text)                a ScoredAnswer of the v1 kind with that prose

The dev examples used: dev_01_blue_mill (assess, no judge note), dev_05_druridge_bay (explain, a note), dev_07_buttermere (clarify, no note), dev_09_west_sleekburn
(decline, a note), dev_08_bamburgh (decline, no note), dev_25_west_sleekburn (clarify, a note). Every answer and reply below is invented.
"""

import json
from collections.abc import Mapping
from functools import cache
from typing import Any

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field

from evals.agent.evaluators.answer import ScoredAnswer
from evals.agent.judge import CRITERIA
from evals.agent.loader import load_dataset


@cache
def _dev_records() -> dict[str, dict[str, Any]]:
    return {record["id"]: record for record in load_dataset("dev").records}


def dev_example(example_id: str) -> dict[str, Any]:
    """A dev record by id. Asking for anything else (a held-out id) is a KeyError: only the dev split is ever loaded."""
    return _dev_records()[example_id]


def answer(text: str, path: str = "assess") -> ScoredAnswer:
    return ScoredAnswer(path=path, text=text)


def canned(**changes: tuple[bool, str]) -> str:
    """A reply with all four criteria passed (reason "ok"), except those named in `changes` as (passed, reason)."""
    grades = {name: {"passed": True, "reason": "ok"} for name in CRITERIA}
    for name, (passed, reason) in changes.items():
        grades[name] = {"passed": passed, "reason": reason}
    return json.dumps(grades)


class FakeJudgeModel(BaseChatModel):
    """Returns `replies` one per call: a str is the model's text, an exception instance is raised. Out of replies is an AssertionError (a call the test did not expect).

    `calls` holds, per call, the messages it was given and the metadata LangChain passed down with the run (the `source` tag among it).
    """

    replies: list[Any] = Field(default_factory=list)
    calls: list[dict[str, Any]] = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "fake-judge"

    def _generate(self, messages: list[BaseMessage], stop: list[str] | None = None, run_manager: CallbackManagerForLLMRun | None = None, **kwargs: Any) -> ChatResult:
        metadata = dict(getattr(run_manager, "metadata", None) or {})
        self.calls.append({"messages": messages, "metadata": metadata})
        assert self.replies, "the judge called the model more often than the test planned"
        reply = self.replies.pop(0)
        if isinstance(reply, BaseException):
            raise reply
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=reply))])

    def with_structured_output(self, schema: Any, **kwargs: Any) -> Any:  # type: ignore[override]
        return self | PydanticOutputParser(pydantic_object=schema)

    def system_texts(self) -> list[str]:
        return [str(call["messages"][0].content) for call in self.calls]

    def user_texts(self) -> list[str]:
        return [str(call["messages"][1].content) for call in self.calls]


class FeedbackClient:
    """Records create_feedback calls as (run_id, key, score, comment, extra keyword arguments)."""

    def __init__(self) -> None:
        self.feedback: list[tuple[Any, str, Any, Any, Mapping[str, Any]]] = []

    def create_feedback(self, run_id: Any, key: str, *, score: Any = None, comment: Any = None, **kwargs: Any) -> None:
        self.feedback.append((run_id, key, score, comment, kwargs))

    def keys(self) -> list[str]:
        return [item[1] for item in self.feedback]
