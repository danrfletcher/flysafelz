"""The v0 baseline's structured output and its adapter (5a PR-9.F1, G14): schema strictness, and the mapping onto what the shared evaluators read."""

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from evals.agent.evaluators.answer import UNKNOWN_PATH, V0
from evals.agent.evaluators.blocker import surfaced
from evals.agent.v0_output import V0Output, answer_text, to_scored_answer
from flysafelz.contracts import NotamAction

SEGMENT = {"start": "2026-09-29T09:00:00Z", "end": "2026-09-29T11:00:00Z", "verdict": "NO_GO", "approaching_limits": True, "conditions": ["gust spread 7 mph, limit 6 mph"]}
ACTION = {"notam_id": "H6217/26", "action": "NO_FLY"}


def output(*, segments=(SEGMENT,), actions=(ACTION,)) -> dict:
    return {"segments": [dict(s) for s in segments], "notam_actions": [dict(a) for a in actions]}


# --- schema strictness --------------------------------------------------------------------------------------------------------------


def test_a_well_formed_reply_parses_into_typed_values():
    parsed = V0Output.model_validate(output())
    [segment] = parsed.segments
    assert segment.start == datetime(2026, 9, 29, 9, tzinfo=UTC) and segment.verdict.value == "NO_GO" and segment.approaching_limits is True
    assert parsed.notam_actions[0].action.value == "NO_FLY"


@pytest.mark.parametrize(
    ("what", "reply"),
    [
        ("an unknown verdict", output(segments=[{**SEGMENT, "verdict": "GO"}])),
        ("a lower-case verdict", output(segments=[{**SEGMENT, "verdict": "no_go"}])),
        ("an unknown action", output(actions=[{**ACTION, "action": "AVOID"}])),
        ("an extra field on the reply", {**output(), "notes": "hello"}),
        ("an extra field on a segment", output(segments=[{**SEGMENT, "confidence": 0.9}])),
        ("an extra field on an action", output(actions=[{**ACTION, "why": "because"}])),
        ("a missing conditions list", output(segments=[{k: v for k, v in SEGMENT.items() if k != "conditions"}])),
        ("a missing approaching_limits", output(segments=[{k: v for k, v in SEGMENT.items() if k != "approaching_limits"}])),
        ("a missing notam_actions", {"segments": [SEGMENT]}),
        ("a time with no timezone", output(segments=[{**SEGMENT, "start": "2026-09-29T09:00:00"}])),
        ("a segment that ends before it starts", output(segments=[{**SEGMENT, "start": "2026-09-29T11:00:00Z", "end": "2026-09-29T09:00:00Z"}])),
        ("a string for approaching_limits", output(segments=[{**SEGMENT, "approaching_limits": "yes"}])),
    ],
)
def test_the_schema_rejects(what, reply):
    with pytest.raises(ValidationError):
        V0Output.model_validate(reply)


# --- the adapter --------------------------------------------------------------------------------------------------------------------


def test_the_adapter_gives_a_v0_answer_with_the_segments_and_actions_the_evaluators_read():
    scored = to_scored_answer(V0Output.model_validate(output()), ["H6217/26", "V30/26"])
    answer = scored.answer
    assert (answer.kind, answer.path) == (V0, UNKNOWN_PATH)
    [segment] = answer.segments
    assert (segment.start, segment.end, segment.verdict, segment.approaching_limits) == (datetime(2026, 9, 29, 9, tzinfo=UTC), datetime(2026, 9, 29, 11, tzinfo=UTC), "NO_GO", True)
    assert dict(answer.notam_actions) == {"H6217/26": "NO_FLY"}
    assert (scored.listed, scored.unlisted) == (("H6217/26",), ())
    assert answer.tool_calls == () and answer.guard is None


def test_an_id_that_is_not_in_the_prompt_is_not_scored_and_is_listed_for_the_run_metadata():
    reply = output(actions=[ACTION, {"notam_id": "Z9999/26", "action": "NO_FLY"}, {"notam_id": "Z9999/26", "action": "IGNORE"}])
    scored = to_scored_answer(V0Output.model_validate(reply), ["H6217/26"])
    assert "Z9999/26" not in scored.answer.notam_actions
    assert (scored.listed, scored.unlisted) == (("H6217/26",), ("Z9999/26",))


def test_ids_match_with_the_whitespace_removed_and_the_year_kept():
    reply = output(actions=[{"notam_id": " H6217 /26 ", "action": "NO_FLY"}, {"notam_id": "H6217/25", "action": "NO_FLY"}])
    scored = to_scored_answer(V0Output.model_validate(reply), ["H6217/26"])
    assert dict(scored.answer.notam_actions) == {"H6217/26": "NO_FLY"}
    assert scored.unlisted == ("H6217/25",)  # another year is another NOTAM


def test_a_prompt_notam_the_output_does_not_name_is_not_surfaced():
    scored = to_scored_answer(V0Output.model_validate(output(actions=[])), ["H6217/26"])
    assert scored.listed == () and dict(scored.answer.notam_actions) == {}
    assert surfaced(scored.answer, "H6217/26") is NotamAction.IGNORE  # as for v1: a NOTAM left out of the answer is IGNORE, which a labelled NO_FLY then misses


def test_the_text_is_built_by_code_from_the_segments_on_the_pilots_clock():
    text = answer_text(V0Output.model_validate(output()))
    assert "10:00" in text and "12:00" in text  # 09:00Z and 11:00Z in Europe/London (BST on 29 September)
    assert "NO_GO" in text and "approaching limits" in text and "gust spread 7 mph, limit 6 mph" in text


def test_a_reply_with_no_segments_still_has_text_to_score():
    reply = output(segments=[], actions=[])
    assert answer_text(V0Output.model_validate(reply)) == "No segments were given."
    assert to_scored_answer(V0Output.model_validate(reply), []).answer.segments == ()
