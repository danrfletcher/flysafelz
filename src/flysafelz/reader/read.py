"""The reader's model call, its one repair and its error handling (G4, G6, G7).

    read_notam(parsed, llm) -> ReaderResult

One `llm.invoke` with the two strings from `render_prompt` and a strict `json_schema` response format built from
`ReaderOutput`, sent with the `timeout_s` of settings.yaml. Code merges the coded fields into the reading, whatever the
model said: `id` from the parsed NOTAM, `geometry` from `build_area` (None is the Q-circle), `schedule` from item D ("" when
there is none, so notam_action() keeps using item D). The model's floor and ceiling reach the reading unchanged, null
included. Choosing between them and F/G is notam_action()'s E10 merge, never this module's.

Then `check_reading` (checks.py) judges the reading, and `read_notam` never raises for what the model or the endpoint does:

- the checks pass: status READ;
- the model answered but a check failed, or the reply is JSON that breaks the form (a wrong enum, a missing key, a field it
  may not fill): one re-ask with the original prompt plus the failed checks as plain sentences. It passes: REPAIRED. It fails
  again: FAILED, no reading, and the second answer's checks are kept. There is never a third call;
- no usable answer at all, on the first call or the repair (any error from `invoke`: timeout, connection, HTTP status;
  an empty reply; a reply that is not JSON, or is cut off): UNAVAILABLE, no reading. If the repair was the call that got
  no answer, the first answer's failed checks are kept.

Errors are caught at the `invoke` boundary as `Exception`, so this package imports no error class from openai or httpx.
Nothing is written to a log or a message here except placeholders: the checks' sentences hold none else, and an error's
text goes through `redact_phones` first.
"""

import json
import logging
import time
from dataclasses import dataclass
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from pydantic import ValidationError

from flysafelz.contracts import NotamReading, ParsedNotam
from flysafelz.notam_area import Area, build_area
from flysafelz.reader.checks import check_reading, schema_failures
from flysafelz.reader.contract import (
    ReaderOutput,
    ReaderResult,
    ReaderStatus,
    TokenUsage,
)
from flysafelz.reader.prompt import PROMPT_VERSION, render_prompt
from flysafelz.reader.settings import SETTINGS
from flysafelz.redact import redact_phones

log = logging.getLogger(__name__)

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


def _model_name(llm: BaseChatModel, reply: BaseMessage | None) -> str:
    """The model the endpoint was asked for, else the one its reply names (when there was a reply), else "unknown"."""
    replied = reply.response_metadata.get("model_name") if reply is not None else None
    for name in (getattr(llm, "model_name", None), getattr(llm, "model", None), replied):
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


# --- one call ------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class _Answer:
    """What one call came to: the reply (None when `invoke` raised) and the seconds it took."""

    reply: BaseMessage | None
    latency_s: float


def _ask(llm: BaseChatModel, notam_id: str, system: str, user: str) -> _Answer:
    """One `invoke`. Any error from it is no answer (G7): it is logged and returned as a missing reply, never raised."""
    started = time.perf_counter()
    try:
        reply = llm.invoke([SystemMessage(system), HumanMessage(user)], response_format=response_format(), timeout=SETTINGS.timeout_s)
    except Exception as error:  # noqa: BLE001 - the one boundary: openai and httpx raise their own classes, which this package does not import
        log.warning("reader unavailable: %s: the model call raised %s: %s", notam_id, type(error).__name__, redact_phones(str(error))[0])
        return _Answer(None, time.perf_counter() - started)
    return _Answer(reply, time.perf_counter() - started)


@dataclass(frozen=True)
class _Judged:
    """What the checks made of one answer: a reading (no failures), failed-check sentences, or neither (no usable answer)."""

    reading: NotamReading | None = None
    failures: tuple[str, ...] = ()

    @property
    def unavailable(self) -> bool:
        return self.reading is None and not self.failures


def _judge(parsed: ParsedNotam, area: Area, redaction: dict[str, str], answer: _Answer) -> _Judged:
    """Turn an answer into a reading that passed the checks, the sentences it failed, or nothing at all."""
    if answer.reply is None:
        return _Judged()
    raw = _text(answer.reply)
    try:
        data = json.loads(raw)
    except ValueError:  # empty, not JSON, or cut off: no usable answer, which is not the same as a wrong one
        log.warning("reader unavailable: %s: the reply is empty, not JSON, or cut off (%d characters)", parsed.id, len(raw))
        return _Judged()
    try:
        output = ReaderOutput.model_validate(data)
    except ValidationError as error:
        return _Judged(failures=tuple(schema_failures(error)))
    reading = NotamReading(id=parsed.id, geometry=area, schedule=parsed.item_d or "", **output.model_dump())
    failures = check_reading(parsed, reading, redaction)
    return _Judged(reading=None if failures else reading, failures=tuple(failures))


def _repair_text(user: str, failures: tuple[str, ...]) -> str:
    """The original user text with the failed checks after it, outside the NOTAM text block."""
    return "\n".join([user, "", "Your previous answer failed these checks:", *(f"- {failure}" for failure in failures), "Answer again with every field, fixing each one."])


# --- the read ------------------------------------------------------------------------------------------------------


def _total_tokens(answers: list[_Answer]) -> TokenUsage | None:
    """The usage of every reply added up; None when there was no reply or any reply reported none."""
    usages = [_tokens(answer.reply) for answer in answers if answer.reply is not None]
    if not usages or any(usage is None for usage in usages):
        return None
    return TokenUsage(
        input_tokens=sum(usage.input_tokens for usage in usages if usage),
        output_tokens=sum(usage.output_tokens for usage in usages if usage),
        total_tokens=sum(usage.total_tokens for usage in usages if usage),
    )


def read_notam(parsed: ParsedNotam, llm: BaseChatModel) -> ReaderResult:
    """Read one NOTAM: one call, and one repair if a check fails. Never raises for what the model or the endpoint does."""
    prompt = render_prompt(parsed)
    area, _note = build_area(parsed)
    answers: list[_Answer] = []

    def finish(status: ReaderStatus, reading: NotamReading | None, failed_checks: tuple[str, ...]) -> ReaderResult:
        replies = [answer.reply for answer in answers if answer.reply is not None]
        return ReaderResult(
            notam_id=parsed.id,
            reading=reading,
            status=status,
            failed_checks=list(failed_checks),
            model=_model_name(llm, replies[0] if replies else None),
            prompt_version=PROMPT_VERSION,
            latency_s=sum(answer.latency_s for answer in answers),
            tokens=_total_tokens(answers),
        )

    def attempt(user: str) -> _Judged:
        answers.append(_ask(llm, parsed.id, prompt.system, user))
        return _judge(parsed, area, prompt.redaction, answers[-1])

    first = attempt(prompt.user)
    if first.unavailable:
        return finish(ReaderStatus.UNAVAILABLE, None, ())
    if first.reading is not None:
        return finish(ReaderStatus.READ, first.reading, ())
    log.warning("reader check failed: %s: asking once more: %s", parsed.id, "; ".join(first.failures))

    second = attempt(_repair_text(prompt.user, first.failures))
    if second.unavailable:
        return finish(ReaderStatus.UNAVAILABLE, None, first.failures)
    if second.reading is None:
        log.warning("reader failed: %s: still failing after one repair, so the NOTAM is unread: %s", parsed.id, "; ".join(second.failures))
        return finish(ReaderStatus.FAILED, None, second.failures)
    return finish(ReaderStatus.REPAIRED, second.reading, first.failures)
