"""Builders for the reader tests: a recording fake chat model, scripted replies, and the Blue Mill PIB's NOTAMs.

The fake is a langchain_core GenericFakeChatModel that records the messages and keyword arguments of every call and
returns scripted AIMessages. Nothing here opens a socket, reads .env or asks a real model.
"""

import json
from pathlib import Path
from typing import Any

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage
from pydantic import Field

from flysafelz.contracts import ParsedNotam
from flysafelz.pib import read_notams

ROOT = Path(__file__).resolve().parents[2]
BLUE_MILL_PIB = ROOT / "fixtures" / "blue_mill_2026-10-31" / "pib.xml"
FAKE_MODEL = "fake-reader-model"
USAGE = {"input_tokens": 1200, "output_tokens": 150, "total_tokens": 1350}


class RecordingChat(GenericFakeChatModel):
    """Returns the scripted replies in order; records every call's messages and keyword arguments."""

    model_name: str = FAKE_MODEL
    calls: list[dict[str, Any]] = Field(default_factory=list)

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):  # noqa: ANN001, ANN201
        self.calls.append({"messages": list(messages), "kwargs": dict(kwargs)})
        return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)


def reply_fields(**overrides: Any) -> dict[str, Any]:
    """A valid model reply as a dict: H7171/26's reading by default. Override any field."""
    fields: dict[str, Any] = {
        "category": "cooperative",
        "activity": "drone operations",
        "floor": {"value": 0, "unit": "ft", "datum": "SFC"},
        "ceiling": {"value": 700, "unit": "ft", "datum": "AGL"},
        "applies_to": "ALL_AIRCRAFT",
        "exemptions": [],
        "contact": {"kind": "phone", "value": "<PHONE_1>"},
        "activates_designators": [],
        "prohibitive_reason": "none",
    }
    fields.update(overrides)
    return fields


def message(reply: dict[str, Any] | str, usage: dict[str, int] | None = None) -> AIMessage:
    """A scripted AIMessage: a dict is sent as JSON text, a string as it is."""
    content = reply if isinstance(reply, str) else json.dumps(reply)
    return AIMessage(content=content, usage_metadata=usage) if usage else AIMessage(content=content)


def scripted(*replies: dict[str, Any] | str | AIMessage) -> RecordingChat:
    """A fake chat model that answers each call with the next reply."""
    return RecordingChat(messages=iter([r if isinstance(r, AIMessage) else message(r) for r in replies]))


def blue_mill_notams() -> dict[str, ParsedNotam]:
    """Every NOTAM of the committed Blue Mill PIB, by id."""
    _issued, notams = read_notams(BLUE_MILL_PIB.read_bytes())
    return {notam.id: notam for notam in notams}


def sent_text(llm: RecordingChat) -> str:
    """Everything the fake was sent, over all its calls: every message and every keyword argument."""
    return "\n".join(f"{m.type}: {m.content}" for call in llm.calls for m in call["messages"]) + "\n" + json.dumps([call["kwargs"] for call in llm.calls], default=str)
