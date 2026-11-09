"""The v0 baseline's structured output, and the adapter that turns it into what the shared evaluators read (5a PR-9.F1, G14).

    V0Output                          the strict schema the one model call must fill: segments[{start, end, verdict, approaching_limits, conditions[]}] and notam_actions[{notam_id, action}]
    V0Segment, V0NotamAction          its two parts
    answer_text(output)               the final-answer text v0 gets: built by code from the segments, never written by the model
    V0Scored(answer, listed, unlisted)   the adapter's result
    to_scored_answer(output, prompt_ids) -> V0Scored   the output as the `ScoredAnswer` of kind "v0", and the NOTAM ids it named that are not in the prompt

The schema is strict: a field the model adds, a verdict that is not a `Verdict`, an action that is not a `NotamAction`, a time with no timezone, a segment that ends before it starts
and a missing field are each a validation error. The run treats a reply that breaks it as a failed attempt (the runner retries it, then lists the example as errored); it is never scored.

The adapter (to_scored_answer) maps the output onto the fields the shared evaluators read, so v0 is scored by the same functions as v1 and there is no v0 copy of any of them:
  - `segments`      every segment, as the `ScoredSegment` the evaluators read (start, end, verdict, approaching_limits). `conditions` are kept in the text only;
  - `notam_actions` {NOTAM id: action} for the ids that are in the prompt. An id the model names that was never in the prompt is not scored: it is left out here and listed in
                    `V0Scored.unlisted` for the run's metadata. A NOTAM that was in the prompt and is missing from the output is, as for v1, not surfaced (the evaluators count it as IGNORE);
  - `text`          `answer_text(output)`: one line per segment with its times on the pilot's clock (Europe/London, as the form checks read them), the verdict, "approaching limits" when
                    flagged, and the model's own conditions. The dropped-source check reads the source words in it and the mention checks read it, so the model's conditions are the part of
                    its answer they can see;
  - `path`          unknown, and `kind` "v0": v0 does no routing, and the evaluators that need a path, tool calls or a guard return None for that kind.
Ids are compared with the whitespace removed and the `/26` year kept (`J3219 /26` is `J3219/26`; `J3219/25` is another NOTAM).
"""

from collections.abc import Iterable
from dataclasses import dataclass

from pydantic import AwareDatetime, BaseModel, StrictBool, model_validator

from evals.agent.evaluators.answer import UNKNOWN_PATH, V0, ScoredAnswer, clean_id
from flysafelz.agent.state import DEFAULT_TZ
from flysafelz.contracts import STRICT, NotamAction, Verdict
from flysafelz.timetext import stamp


class V0Segment(BaseModel):
    """One stretch of the window with one verdict. Times are UTC instants with a timezone."""

    model_config = STRICT

    start: AwareDatetime
    end: AwareDatetime
    verdict: Verdict
    approaching_limits: StrictBool  # a real true or false: the string "yes" is a schema error, not a flag
    conditions: list[str]

    @model_validator(mode="after")
    def _ends_after_it_starts(self) -> "V0Segment":
        if self.end < self.start:
            raise ValueError(f"the segment ends at {self.end.isoformat()}, before it starts at {self.start.isoformat()}")
        return self


class V0NotamAction(BaseModel):
    notam_id: str
    action: NotamAction

    model_config = STRICT


class V0Output(BaseModel):
    """What the single model call returns."""

    model_config = STRICT

    segments: list[V0Segment]
    notam_actions: list[V0NotamAction]


@dataclass(frozen=True)
class V0Scored:
    """The adapter's result: the answer the evaluators read, the prompt NOTAM ids the output gave an action to, and the ids it named that were not in the prompt."""

    answer: ScoredAnswer
    listed: tuple[str, ...]
    unlisted: tuple[str, ...]


def answer_text(output: V0Output) -> str:
    """The v0 run's final-answer text: its segments, one line each, on the pilot's clock. Never empty, so a reply with no segments still has an answer to score."""
    if not output.segments:
        return "No segments were given."
    lines = []
    for segment in output.segments:
        line = f"{stamp(segment.start, DEFAULT_TZ)} to {stamp(segment.end, DEFAULT_TZ)}: {segment.verdict.value}"
        if segment.approaching_limits:
            line += " (approaching limits)"
        if segment.conditions:
            line += ". " + "; ".join(condition.strip() for condition in segment.conditions if condition.strip())
        lines.append(line)
    return "\n".join(lines)


def to_scored_answer(output: V0Output, prompt_ids: Iterable[str]) -> V0Scored:
    """The output as a v0 `ScoredAnswer`, with the NOTAM ids that are not in the prompt left out of it (listed in `unlisted`, sorted)."""
    known = {clean_id(i) for i in prompt_ids}
    kept: list[dict[str, str]] = []  # a list of {id, action}, not a dict, so two entries for one id are both read (the stricter action wins in ScoredAnswer)
    unlisted: set[str] = set()
    for item in output.notam_actions:
        notam_id = clean_id(item.notam_id)
        if notam_id in known:
            kept.append({"id": notam_id, "action": item.action.value})
        else:
            unlisted.add(notam_id)
    segments = [{"start": s.start, "end": s.end, "verdict": s.verdict, "approaching_limits": s.approaching_limits} for s in output.segments]
    answer = ScoredAnswer.from_dict({"path": UNKNOWN_PATH, "segments": segments, "notam_actions": kept, "text": answer_text(output), "kind": V0})
    return V0Scored(answer, tuple(sorted({item["id"] for item in kept})), tuple(sorted(unlisted)))
