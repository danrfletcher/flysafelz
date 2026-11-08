"""What happens with the guard's findings (PR-11.F1): one retry with no tools, then a fallback built by code. `guard.py` holds the checks; this holds the flow.

`FlySafeMiddleware.wrap_model_call` (middleware.py) drives a `GuardFlow` around the model call. The checks (guard.py) are unchanged; this part decides what
happens with their findings.

  a response with tool calls         passes through untouched: it is not a final message
  a final message                    "Checking the answer…" is written, the first line (the path tag) is read and stripped, and `guard_answer` checks the
                                     rest against this turn's tool results. No findings: the answer is delivered, `guard` is `passed`
  findings, a retry left             one more model call: the conversation plus a HumanMessage from the checker with the findings as plain sentences,
                                     and NO tools bound. Nothing is carried between the two calls except the flow object of this one wrap call
  findings after the last retry      `fallback_answer`, built by code from the tool results; `guard` is `fallback`. A tag problem alone is not enough
                                     for that: the answer is delivered with the path `unknown`
  after either                       the limit lines and the re-check line are added (answer.py), then the real phone numbers are put back

A failed draft is never written to graph state: the only message that leaves is the one that passed (without its tag) or the
fallback code built. The turn's outcome (how it ended, the path, the findings of each failed attempt) goes to `on_outcome` for PR-12.F1 to send
as feedback. It is not state (P4: drafts and findings are never written to state).
"""

import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import timedelta
from typing import Any, Literal
from zoneinfo import ZoneInfo

from langchain.agents.middleware import ExtendedModelResponse, ModelRequest, ModelResponse
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langgraph.types import Command

from flysafelz.agent import progress
from flysafelz.agent.answer import NO_LINE_PATHS, assessment_of, limit_lines, recheck_line, rehydrate, rehydrate_assessment, rehydrate_overlays
from flysafelz.agent.guard import ASSESS_TOOL, GuardFinding, ToolResult, TurnSources, fallback_answer, guard_answer
from flysafelz.agent.prompts import PATH_TAGS, PATHS
from flysafelz.agent.settings import SETTINGS
from flysafelz.agent.state import DEFAULT_TZ, GuardResult, OverlayFeature, RunContext, run_now

DATA_TOOLS = frozenset({"get_weather", "get_airspace", "get_notams", ASSESS_TOOL})  # the tools whose artifacts say which window and which fetch a turn used
CHECKER_NAME = "checker"
NO_PATH = "unknown"
_TAG_LINE = re.compile(rf"\[path: ({'|'.join(PATHS)})\]")
_ANY_TAG_LINE = re.compile(r"^[ \t]*\[path\b[^\]\n]*\][ \t]*$\n?", re.IGNORECASE | re.MULTILINE)
_LEADING_TAG = re.compile(r"\A\s*\[path\b[^\]\n]*\][ \t]*", re.IGNORECASE)  # a tag with the answer on its own line
PATH_FINDING = GuardFinding("path", f"The first line of the answer must be exactly one of {', '.join(PATH_TAGS)}, and the answer must start on the next line")
EMPTY_FINDING = GuardFinding("empty", "The answer is empty; write the answer")
TOOL_CALL_FINDING = GuardFinding("tools", "No tools are available for this rewrite; answer in plain text from the tool results already in the conversation")

type AttemptReason = Literal["checks", "path_tag"]


@dataclass(frozen=True)
class GuardAttempt:
    """One failed draft: the findings it got, and the reason PR-12.F1 files them under (`path_tag` when the tag was the only problem)."""

    findings: tuple[str, ...]
    reason: AttemptReason


@dataclass(frozen=True)
class GuardOutcome:
    """How a turn's guard ended: `result` is the `guard` state value, `path` the answer's path tag (`unknown` when none was valid), `attempts` the failed drafts.

    A pass has no attempts. A retry that fixed the answer still lists the draft it fixed.
    """

    result: GuardResult
    path: str
    attempts: tuple[GuardAttempt, ...] = ()


def split_path(text: str) -> tuple[str | None, str]:
    """(path, body) of a final message: the path when its first line is exactly `[path: assess|explain|clarify|decline]`, else None; the body is the rest.

    Any other line shaped like a tag is removed, so a tag never reaches the pilot, whether it was valid or not.
    """
    first, _, rest = text.partition("\n")
    tag = _TAG_LINE.fullmatch(first.strip())
    if tag is None:
        return None, _ANY_TAG_LINE.sub("", _LEADING_TAG.sub("", text)).strip()
    return tag[1], _ANY_TAG_LINE.sub("", rest).strip()


def retries_allowed() -> int:
    """The extra model calls the guard may make after a failed first answer (settings.yaml `guard_retries`)."""
    return SETTINGS.guard_retries


def hide_phones(sentence: str, phone_map: Mapping[str, str]) -> str:
    """`sentence` with each real number in the thread's phone map put back to its placeholder. A finding quotes the draft, and a number must never reach the model or a trace."""
    for placeholder, number in phone_map.items():
        sentence = sentence.replace(number, placeholder)
    return sentence


def checker_message(findings: Iterable[str]) -> HumanMessage:
    """The retry's message from the checker: the findings as plain sentences. It is only ever sent to the model, never stored."""
    lines = "\n".join(f"- {finding}" for finding in findings)
    text = (
        "Message from the answer checker: your last answer was not sent to the pilot, because it failed these checks:\n"
        f"{lines}\n"
        "Write the answer again as plain text. Use only the tool results already in this conversation, and call no tools. "
        f"Start with the first line naming your path ({', '.join(PATH_TAGS)}), then the answer."
    )
    return HumanMessage(content=text, name=CHECKER_NAME)


def _results(messages: Iterable[BaseMessage]) -> tuple[ToolResult, ...]:
    return tuple(ToolResult(str(m.name), m.text, m.artifact) for m in messages if isinstance(m, ToolMessage))


def output_update(text: str, result: GuardResult, *, artifacts: Sequence[Any], phone_map: Mapping[str, str] | None, overlays: Sequence[OverlayFeature]) -> dict[str, Any]:
    """The state update that ends a turn: `answer` (the real numbers back in), `guard`, and the two other output fields that can hold a phone number.

    `assessment` is this turn's `assess` result or None, and `overlays` is written only when a NOTAM's text had a placeholder in it. This is the one
    place a phone number is put back (P1, I7): the AI message in `messages` is not touched.
    """
    update: dict[str, Any] = {
        "answer": rehydrate(text, phone_map),
        "guard": result,
        "assessment": rehydrate_assessment(assessment_of(artifacts), phone_map),
    }
    restored = rehydrate_overlays(overlays, phone_map)
    if any(new is not old for new, old in zip(restored, overlays, strict=True)):
        update["overlays"] = restored
    return update


class GuardFlow:
    """One wrap call's guard: `review(response)` is called with each model response until it returns None, then `done` is the response to hand back.

    It holds the state of this one call only (the attempts so far); a new flow is made for every model call the graph makes, so nothing carries over.
    """

    def __init__(
        self,
        request: ModelRequest[RunContext],
        turn_messages: Sequence[BaseMessage],
        on_outcome: Callable[[GuardOutcome], None] | None = None,
    ) -> None:
        self.request = request
        self.on_outcome = on_outcome
        self.results = _results(turn_messages)
        state = request.state
        self.phone_map: Mapping[str, str] = state.get("phone_map") or {}
        self.overlays: list[OverlayFeature] = list(state.get("overlays") or [])
        thread_lz = state.get("thread_lz")
        self.tz = thread_lz["tz"] if thread_lz else DEFAULT_TZ
        self.now = run_now(request.runtime.context)
        today = self.now.astimezone(ZoneInfo(self.tz)).date()
        calendar_days = tuple(today + timedelta(days=offset) for offset in range(SETTINGS.calendar_days + 1))
        lz = (thread_lz["lat"], thread_lz["lon"]) if thread_lz else None
        self.turn = TurnSources(results=self.results, tz=self.tz, phone_keys=frozenset(self.phone_map), calendar=calendar_days, lz=lz)
        self.attempts: list[GuardAttempt] = []
        self.path: str | None = None
        self.retries_made = 0
        self.done: ModelResponse[Any] | ExtendedModelResponse[Any] | None = None

    # -- one response ------------------------------------------------------------------------------------------------

    def review(self, response: ModelResponse[Any]) -> ModelRequest[RunContext] | None:
        """Check a model response. Returns the retry request when one is due, else None with `done` set."""
        message = next((m for m in reversed(response.result) if isinstance(m, AIMessage)), None)
        wants_tools = message is not None and bool(message.tool_calls)
        if message is None or (wants_tools and not self.retries_made):
            self.done = response  # not a final message: the graph goes on to the tools
            return None
        findings, path, body = self._judge(message)
        if path is not None:
            self.path = path
        if not findings:
            self._finish(message, body, "retried" if self.retries_made else "passed")
            return None
        only_the_tag = findings == [PATH_FINDING]
        shown = [hide_phones(finding, self.phone_map) for finding in findings]  # what the model and the trace get: placeholders only (F2)
        self.attempts.append(GuardAttempt(tuple(shown), "path_tag" if only_the_tag else "checks"))
        if self.retries_made < retries_allowed():
            self.retries_made += 1
            return self.request.override(messages=[*self.request.messages, checker_message(shown)], tools=[], tool_choice=None)
        if only_the_tag:
            self._finish(message, body, "retried" if self.retries_made else "passed")  # the answer passed every other check
            return None
        self._fall_back()
        return None

    def _judge(self, message: AIMessage) -> tuple[list[GuardFinding], str | None, str]:
        """(findings, path, body) of a final message. A message that still calls tools is a finding on a retry, where none are bound."""
        progress.emit(progress.CHECKING_ANSWER)
        path, body = split_path(message.text)
        findings: list[GuardFinding] = []
        if message.tool_calls or message.invalid_tool_calls:
            findings.append(TOOL_CALL_FINDING)
        if not body:
            findings.append(EMPTY_FINDING)  # an empty message is one finding, whatever its first line was
            return findings, path, body
        findings.extend(guard_answer(body, self.turn))
        if path is None:
            findings.append(PATH_FINDING)
        return findings, path, body

    # -- how the turn ends -------------------------------------------------------------------------------------------

    def _artifacts(self) -> list[Any]:
        return [r.artifact for r in self.results if r.artifact is not None]

    def _lines(self, path: str | None) -> list[str]:
        """The limit lines and then the re-check line, in that order, as the rules in answer.py give them."""
        artifacts = self._artifacts()
        data = [a for a in artifacts if getattr(a, "tool", None) in DATA_TOOLS]
        lines = [] if path in NO_LINE_PATHS else limit_lines(artifacts)
        recheck = recheck_line(data, self.now, self.tz, path)
        return [*(["\n".join(lines)] if lines else []), *([recheck] if recheck else [])]

    def _finish(self, message: AIMessage, body: str, result: GuardResult) -> None:
        text = "\n\n".join([body, *self._lines(self.path)])
        self._end(message.model_copy(update={"content": body}), text, result)

    def _fall_back(self) -> None:
        text = fallback_answer(self.results, self.path, self.overlays)
        answer = "\n\n".join([text, *self._lines(self.path)])
        self._end(AIMessage(content=text), answer, "fallback")

    def _end(self, message: AIMessage, answer: str, result: GuardResult) -> None:
        update = output_update(answer, result, artifacts=self._artifacts(), phone_map=self.phone_map, overlays=self.overlays)
        self.done = ExtendedModelResponse(model_response=ModelResponse(result=[message]), command=Command(update=update))
        if self.on_outcome is not None:
            self.on_outcome(GuardOutcome(result, self.path or NO_PATH, tuple(self.attempts)))
