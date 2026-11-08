"""ScoredAnswer (5a PR-5.F1): built from a recorded Assessment and from a v0-style dict, with identical scores for the same segments. Also the aggregation helper."""

from datetime import UTC, datetime

import pytest

from agent_eval.factories import answer, example, iso, row, seg, t
from evals.agent.evaluators.aggregate import EVALUATORS, ScoredRun, evaluate_runs, score_run
from evals.agent.evaluators.answer import ScoredAnswer, ScoredSegment, clean_id, path_from_metadata
from evals.agent.evaluators.labels import Feedback
from evals.agent.evaluators.minutes import NaiveDatetimeError
from flysafelz.assess import assess
from flysafelz.cases import load_case_inputs_by_name
from flysafelz.contracts import Assessment, NotamAction, Verdict

BLUE_MILL = "blue_mill_2026-10-31"
TEXT = "The 17:06 dusk is the edge."


@pytest.fixture(scope="module")
def recorded() -> Assessment:
    """The committed Blue Mill case through assess() with its stored readings, round-tripped through JSON as a recorded run would be."""
    case, forecast, airspace, pib, readings = load_case_inputs_by_name(BLUE_MILL)
    live = assess(case.lat, case.lon, now=case.now, forecast=forecast, airspace=airspace, pib=pib, readings=readings)
    return Assessment.model_validate_json(live.model_dump_json())


def v0_dict(assessment: Assessment) -> dict:
    """What the v0 baseline's structured output looks like for the same segments (conditions and other keys come too, and are ignored)."""
    segments = [
        {"start": iso(s.start), "end": iso(s.end), "verdict": s.verdict.value, "approaching_limits": s.approaching_limits, "conditions": ["a condition"]}
        for hour in assessment.hours
        for s in hour.segments
    ]
    actions = {}
    for hour in assessment.hours:
        for s in hour.segments:
            for n in s.notam_actions:
                actions.setdefault(n.notam_id, n.action.value)
    return {"path": "assess", "segments": segments, "notam_actions": actions, "text": TEXT, "extra": 1}


def labels_from(assessment: Assessment):
    rows = [row(s.start, s.end, s.verdict.value, s.approaching_limits) for h in assessment.hours for s in h.segments]
    return rows, {"H7171/26": "AVOID_SECTOR", "J3857/26": "CONTACT_FIRST"}


def test_the_recorded_assessment_is_the_blue_mill_case_with_its_dusk_edge(recorded):
    assert recorded.overall_verdict is Verdict.NO_GO
    ends = {s.end for h in recorded.hours for s in h.segments}
    assert datetime(2026, 10, 31, 17, 6, tzinfo=UTC) in ends


def test_both_builders_give_the_same_scored_answer_for_the_same_segments(recorded):
    from_assessment = ScoredAnswer.from_assessment(recorded, path="assess", text=TEXT)
    from_dict = ScoredAnswer.from_dict(v0_dict(recorded))
    assert from_assessment == from_dict
    assert len(from_assessment.segments) == sum(len(h.segments) for h in recorded.hours) > 11 and from_assessment.path == "assess" and from_assessment.text == TEXT
    assert from_assessment.notam_actions == {"H7171/26": NotamAction.AVOID_SECTOR, "J3857/26": NotamAction.CONTACT_FIRST}


def test_both_give_identical_scores_for_every_evaluator_and_the_correct_answer_is_all_ones(recorded):
    rows, actions = labels_from(recorded)
    ex = example(rows, actions=actions, window=(t(7), t(18)), now=t(7, 30))
    a, b = ScoredAnswer.from_assessment(recorded, path="assess", text=TEXT), ScoredAnswer.from_dict(v0_dict(recorded))  # same text: the form scores read it
    scores_a, scores_b = score_run(ScoredRun(ex, a)), score_run(ScoredRun(ex, b))
    assert scores_a == scores_b
    got = {key: f["score"] for key, f in scores_a.feedback.items() if f}
    assert got["harmful_missed_blocker"] == 0 and got["hard_restriction_recall"] == 1
    assert got["verdict_match"] == got["approaching_limits_match"] == got["window_iou"] == got["path_accuracy"] == got["notam_action_match"] == 1
    assert got["no_verdict_beyond_day3"] == 1 and scores_a.feedback["unverified_on_dropped_source"] is None


def test_the_same_edited_segments_give_identical_scores_through_both_builders(recorded):
    rows, actions = labels_from(recorded)
    ex = example(rows, actions=actions, window=(t(7), t(18)), now=t(7, 30))
    data = v0_dict(recorded)
    data["segments"] = [s | ({"verdict": "PILOT_DISCRETION"} if s["start"].startswith("2026-10-31T17:06") else {}) for s in data["segments"]]
    data["notam_actions"] = {"H7171/26": "AWARENESS", "J3857/26": "NO_FLY"}
    edited_dict = ScoredAnswer.from_dict(data)
    edited_assessment = recorded.model_copy(deep=True)
    for hour in edited_assessment.hours:
        for s in hour.segments:
            if s.start == datetime(2026, 10, 31, 17, 6, tzinfo=UTC):
                s.verdict = Verdict.PILOT_DISCRETION
            for n in s.notam_actions:
                n.action = NotamAction.AWARENESS if n.notam_id == "H7171/26" else NotamAction.NO_FLY
    edited = ScoredAnswer.from_assessment(edited_assessment, path="assess", text=TEXT)
    assert edited == edited_dict
    assert score_run(ScoredRun(ex, edited)) == score_run(ScoredRun(ex, edited_dict))
    assert score_run(ScoredRun(ex, edited)).feedback["harmful_missed_blocker"]["score"] == 1  # type: ignore[index]


def test_an_assessment_notam_gets_the_most_restrictive_action_any_segment_gave_it(recorded):
    a = recorded.model_copy(deep=True)
    first = next(s for h in a.hours for s in h.segments if s.notam_actions)
    first.notam_actions[0].action = NotamAction.NO_FLY
    got = ScoredAnswer.from_assessment(a, path="assess")
    assert got.notam_actions[first.notam_actions[0].notam_id] is NotamAction.NO_FLY


def test_outlook_hours_have_no_segments_so_they_add_none(recorded):
    a = recorded.model_copy(deep=True)
    dropped = len(a.hours[0].segments)
    a.hours[0].segments = []
    assert dropped and len(ScoredAnswer.from_assessment(a, path="assess").segments) == sum(len(h.segments) for h in recorded.hours) - dropped


def test_the_dict_accepts_a_list_of_actions_and_cleans_ids_keeping_the_year():
    got = ScoredAnswer.from_dict({"path": "Assess", "notam_actions": [{"id": "H7171 /26", "action": "NO_FLY"}, {"notam_id": "H7171/25", "action": "IGNORE"}]})
    assert got.path == "assess" and got.notam_actions == {"H7171/26": NotamAction.NO_FLY, "H7171/25": NotamAction.IGNORE}
    assert clean_id(" J3857\t/ 26 ") == "J3857/26"


def test_the_same_id_twice_keeps_the_stricter_action():
    got = ScoredAnswer.from_dict({"notam_actions": [{"id": "A/26", "action": "AWARENESS"}, {"id": "A /26", "action": "NO_FLY"}]})
    assert got.notam_actions == {"A/26": NotamAction.NO_FLY}


def test_an_empty_dict_is_an_empty_answer_with_an_unknown_path():
    assert ScoredAnswer.from_dict({}) == ScoredAnswer("unknown", (), {}, "")


def test_a_naive_segment_time_a_bad_verdict_and_a_bad_action_are_refused():
    with pytest.raises(NaiveDatetimeError, match="segment start"):
        ScoredAnswer.from_dict({"segments": [{"start": "2026-10-31T10:00:00", "end": "2026-10-31T11:00:00Z", "verdict": "NO_GO"}]})
    with pytest.raises(ValueError, match="MAYBE"):
        ScoredAnswer.from_dict({"segments": [{"start": "2026-10-31T10:00:00Z", "end": "2026-10-31T11:00:00Z", "verdict": "MAYBE"}]})
    with pytest.raises(ValueError, match="NOTAM A/26.*SHRUG"):
        ScoredAnswer.from_dict({"notam_actions": {"A/26": "SHRUG"}})
    with pytest.raises(ValueError, match="mapping or a list"):
        ScoredAnswer.from_dict({"notam_actions": "NO_FLY"})
    with pytest.raises(ValueError, match="before it starts"):
        ScoredSegment(t(11), t(10), Verdict.NO_GO)


def test_the_path_comes_from_the_root_run_metadata():
    assert path_from_metadata({"path": "clarify", "other": 1}) == "clarify"
    assert path_from_metadata(None) == "unknown" and path_from_metadata({}) == "unknown"


# --- the aggregation helper --------------------------------------------------------------------------------------------------------


def run_of(id, ans, rep=1, errored=False, **kw):
    ex = example([row(t(10), t(11), "NO_GO")], id=id, window=(t(10), t(11)), **kw)
    return ScoredRun(ex, ans, rep, errored)


def test_an_errored_run_is_skipped_not_scored_zero_and_never_reaches_an_evaluator():
    seen = []

    def spy(ex, ans) -> Feedback:
        seen.append(ex["id"])
        return {"key": "spy", "score": 1, "comment": ""}

    good = run_of("good", answer([seg(t(10), t(11), "NO_GO")]))
    bad = run_of("bad", None, errored=True)
    result = evaluate_runs([good, bad], {"spy": spy})
    assert seen == ["good"] and result.skipped == ["bad"] and [r.example_id for r in result.runs] == ["good"]
    assert result.mean("spy") == 1 and result.applied("spy") == 1  # a 0 for the errored run would have made this 0.5


def test_score_run_asserts_on_an_errored_run_and_on_a_missing_answer():
    with pytest.raises(AssertionError, match="errored"):
        score_run(run_of("bad", answer(), errored=True))
    with pytest.raises(AssertionError, match="has an answer"):
        score_run(run_of("none", None))


def test_a_mean_over_no_applicable_run_is_none_never_zero():
    ex = ScoredRun(example(path="explain"), answer(path="explain"))
    result = evaluate_runs([ex])
    assert result.mean("verdict_match") is None and result.applied("verdict_match") == 0
    assert result.mean("path_accuracy") == 1


def test_missed_blockers_are_counted_by_run_and_the_examples_named_once_each():
    miss = answer([seg(t(10), t(11), "PILOT_DISCRETION")])
    ok = answer([seg(t(10), t(11), "NO_GO")])
    result = evaluate_runs([run_of("a", miss, 1), run_of("a", miss, 2), run_of("b", ok), run_of("c", miss)])
    assert result.missed_blockers.count == 3 and result.missed_blockers.examples == ("a", "c")


def test_the_run_carries_both_confusion_matrices():
    ex = example([row(t(10), t(11), "NO_GO")], window=(t(10), t(11)), actions={"A/26": "NO_FLY"})
    result = evaluate_runs([ScoredRun(ex, answer(path="explain", actions={"A/26": NotamAction.CONTACT_FIRST}))])
    assert result.path_confusion["assess"]["explain"] == 1 and result.action_confusion["NO_FLY"]["CONTACT_FIRST"] == 1


def test_every_evaluator_key_is_reported_and_each_returns_the_feedback_shape(recorded):
    rows, actions = labels_from(recorded)
    ex = example(rows, actions=actions, window=(t(7), t(18)), now=t(7, 30), drop=("pib",))
    scores = score_run(ScoredRun(ex, ScoredAnswer.from_assessment(recorded, path="assess", text="notam")))
    assert list(scores.feedback) == list(EVALUATORS)
    for key, found in scores.feedback.items():
        assert found is None or (found["key"] == key and set(found) == {"key", "score", "comment"})
