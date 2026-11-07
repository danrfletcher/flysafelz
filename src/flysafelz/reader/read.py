"""The reader's one model call (G4).

    read_notam(parsed, llm) -> ReaderResult

One `llm.invoke` with the two strings from `render_prompt` and a strict `json_schema` response format built from
`ReaderOutput`. The reply is parsed with `model_validate_json`; anything that is not a valid ReaderOutput raises
`ReaderOutputError` with the reply text attached. This module neither retries nor classifies a failure, and a model
error from `invoke` is not caught either: that belongs to the step that decides what a failed read means.

Code merges the coded fields into the reading, whatever the model said: `id` from the parsed NOTAM, `geometry` from
`build_area` (None is the Q-circle), `schedule` from item D ("" when there is none, so notam_action() keeps using
item D). The model's floor and ceiling reach the reading unchanged, null included. Choosing between them and F/G is
notam_action()'s E10 merge, never this module's.
"""

import time
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from pydantic import ValidationError

from flysafelz.contracts import NotamReading, ParsedNotam
from flysafelz.notam_area import build_area
from flysafelz.reader.contract import ReaderOutput, ReaderOutputError, ReaderResult, ReaderStatus, TokenUsage
from flysafelz.reader.prompt import PROMPT_VERSION, render_prompt

SCHEMA_NAME = "notam_reading"
_UNKNOWN_MODEL = "unknown"


def _strict(schema: Any) -> Any:
    """The JSON schema with what OpenAI's strict mode needs: every property required, no extra properties, no defaults."""
    if isinstance(schema, list):
        return [_strict(item) for item in schema]
    if not isinstance(schema, dict):
        return schema
    out = {key: _strict(value) for key, value in schema.items() if key not in ("default", "properties")}
    if "properties" in schema:
        out["properties"] = {name: _strict(sub) for name, sub in schema["properties"].items()}
        out["required"] = list(schema["properties"])
        out["additionalProperties"] = False
    return out


def response_format() -> dict[str, Any]:
    """The `json_schema` response format for ReaderOutput, in strict mode."""
    return {
        "type": "json_schema",
        "json_schema": {"name": SCHEMA_NAME, "strict": True, "schema": _strict(ReaderOutput.model_json_schema())},
    }


def _text(message: BaseMessage) -> str:
    """The reply as one string, whether the endpoint sent plain content or a list of content parts."""
    content = message.content
    if isinstance(content, str):
        return content
    parts = (part if isinstance(part, str) else part.get("text", "") for part in content if isinstance(part, (str, dict)))
    return "".join(part for part in parts if isinstance(part, str))


def _model_name(llm: BaseChatModel, reply: BaseMessage) -> str:
    """The model the endpoint was asked for, else the one its reply names."""
    for name in (getattr(llm, "model_name", None), getattr(llm, "model", None), reply.response_metadata.get("model_name")):
        if isinstance(name, str) and name:
            return name
    return _UNKNOWN_MODEL


def _tokens(reply: BaseMessage) -> TokenUsage | None:
    """The usage the endpoint reported, or None when it reported none."""
    usage = getattr(reply, "usage_metadata", None)
    if not usage or "input_tokens" not in usage or "output_tokens" not in usage:
        return None
    return TokenUsage(
        input_tokens=usage["input_tokens"],
        output_tokens=usage["output_tokens"],
        total_tokens=usage.get("total_tokens", usage["input_tokens"] + usage["output_tokens"]),
    )


def _parse(raw: str) -> ReaderOutput:
    try:
        return ReaderOutput.model_validate_json(raw)
    except ValidationError as error:
        problems = "; ".join(f"{'.'.join(str(part) for part in e['loc']) or 'reply'}: {e['msg']}" for e in error.errors())
        raise ReaderOutputError(f"the reply is not a valid reading: {problems}", raw) from error


def read_notam(parsed: ParsedNotam, llm: BaseChatModel) -> ReaderResult:
    """Read one NOTAM with one model call. Returns status "read"; raises ReaderOutputError on a reply it cannot use."""
    prompt = render_prompt(parsed)
    started = time.perf_counter()
    reply = llm.invoke([SystemMessage(prompt.system), HumanMessage(prompt.user)], response_format=response_format())
    latency = time.perf_counter() - started
    output = _parse(_text(reply))
    area, _note = build_area(parsed)
    reading = NotamReading(id=parsed.id, geometry=area, schedule=parsed.item_d or "", **output.model_dump())
    return ReaderResult(
        notam_id=parsed.id,
        reading=reading,
        status=ReaderStatus.READ,
        failed_checks=[],
        model=_model_name(llm, reply),
        prompt_version=PROMPT_VERSION,
        latency_s=latency,
        tokens=_tokens(reply),
    )
