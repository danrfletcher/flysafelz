"""G9 verdict match, approaching-limits match and window IoU (5a PR-5.F1): OUTLOOK rows out, E12's 14 minutes, E14 clipping, overlap and gaps, IoU per verdict."""

import pytest

from agent_eval.factories import answer, example, row, seg, t, tier_row
from evals.agent.evaluators.verdict import approaching_limits_match, verdict_ious, verdict_match, window_iou
from flysafelz.contracts import Verdict

NO_GO, UNVERIFIED, PD = Verdict.NO_GO, Verdict.UNVERIFIED, Verdict.PILOT_DISCRETION
E12_ROWS = [row(t(16), t(17, 6), "PILOT_DISCRETION", edge="civil dusk"), row(t(17, 6), t(18), "NO_GO")]
E12_WINDOW = (t(16), t(18))  # 120 labelled minutes


def e12():
    return example(E12_ROWS, window=E12_WINDOW)


def test_a_perfect_answer_scores_one_on_all_three():
    ans = answer([seg(t(16), t(17, 6), "PILOT_DISCRETION"), seg(t(17, 6), t(18), "NO_GO")])
    assert [f["score"] for f in (verdict_match(e12(), ans), approaching_limits_match(e12(), ans), window_iou(e12(), ans))] == [1, 1, 1]  # type: ignore[index]


def test_e12_fourteen_disagreeing_minutes_give_a_match_of_n_minus_14_over_n_and_a_no_go_iou_below_one():
    ans = answer([seg(t(16), t(17, 20), "PILOT_DISCRETION"), seg(t(17, 20), t(18), "NO_GO")])
    got = verdict_match(e12(), ans)
    assert got and got["key"] == "verdict_match" and got["score"] == pytest.approx((120 - 14) / 120)
    assert "14 min disagree" in got["comment"] and "2026-10-31 17:06–17:20Z" in got["comment"] and "answer PILOT_DISCRETION, sheet NO_GO" in got["comment"]
    ious = verdict_ious(e12(), ans)
    assert ious == {NO_GO: pytest.approx(40 / 54), PD: pytest.approx(66 / 80)}
    assert ious[NO_GO] < 1  # type: ignore[index]
    assert window_iou(e12(), ans)["score"] == pytest.approx((40 / 54 + 66 / 80) / 2)  # type: ignore[index]


def test_approaching_limits_is_its_own_score_and_does_not_change_the_verdict_score():
    ex = example([row(t(10), t(11), "NO_GO", True), row(t(11), t(12), "NO_GO", False)], window=(t(10), t(12)))
    ans = answer([seg(t(10), t(12), "NO_GO", False)])
    assert verdict_match(ex, ans)["score"] == 1  # type: ignore[index]
    got = approaching_limits_match(ex, ans)
    assert got and got["key"] == "approaching_limits_match" and got["score"] == pytest.approx(0.5)
    assert "60 min disagree" in got["comment"] and "approaching_limits false" in got["comment"] and "approaching_limits true" in got["comment"]
    assert window_iou(ex, ans)["score"] == 1  # type: ignore[index]


def test_outlook_and_too_far_rows_are_left_out_of_every_score():
    ex = example([row(t(10), t(11), "NO_GO")], window=(t(10), t(13)), tiers=[tier_row(t(11), t(12), "OUTLOOK"), tier_row(t(12), t(13), "TOO_FAR")])
    ans = answer([seg(t(10), t(11), "NO_GO"), seg(t(11), t(13), "PILOT_DISCRETION")])  # whatever the answer says over the tier rows is not scored
    assert verdict_match(ex, ans)["score"] == 1  # type: ignore[index]
    assert window_iou(ex, ans)["score"] == 1  # type: ignore[index]
    assert verdict_ious(ex, ans) == {NO_GO: 1.0}


def test_a_window_with_no_verdict_tier_minute_gives_none_not_a_division_by_zero():
    outlook = example([], window=(t(10), t(12)), tiers=[tier_row(t(10), t(12))])
    zero = example([row(t(10), t(11), "NO_GO")], window=(t(10), t(10)))
    for ex in (outlook, zero):
        for evaluate in (verdict_match, approaching_limits_match, window_iou):
            assert evaluate(ex, answer([seg(t(10), t(12), "NO_GO")])) is None
        assert verdict_ious(ex, answer()) is None


@pytest.mark.parametrize("evaluate", [verdict_match, approaching_limits_match, window_iou])
def test_non_assess_examples_and_examples_without_a_sheet_entry_give_none(evaluate):
    assert evaluate(example(path="decline"), answer(path="decline")) is None
    assert evaluate(example(sheet=False), answer()) is None


# --- E14 -------------------------------------------------------------------------------------------------------------------------


def test_e14_a_row_half_outside_the_window_counts_only_its_in_window_part():
    ex = example([row(t(9), t(11), "NO_GO"), row(t(11), t(13), "PILOT_DISCRETION"), row(t(14), t(15), "NO_GO")], window=(t(10), t(12)))
    ans = answer([seg(t(10), t(11), "NO_GO"), seg(t(11), t(12), "PILOT_DISCRETION")])
    assert verdict_match(ex, ans)["score"] == 1  # type: ignore[index]  # 120 labelled minutes, not 300; the wholly-outside row adds none
    assert verdict_ious(ex, ans) == {NO_GO: 1.0, PD: 1.0}


def test_e14_answer_segments_are_clipped_to_the_window_too():
    ex = example([row(t(10), t(11), "NO_GO")], window=(t(10), t(11)))
    ans = answer([seg(t(8), t(11), "NO_GO"), seg(t(11), t(14), "PILOT_DISCRETION")])
    assert verdict_match(ex, ans)["score"] == 1  # type: ignore[index]
    assert verdict_ious(ex, ans) == {NO_GO: 1.0}  # a PILOT_DISCRETION that is wholly outside the window is not in the union


# --- overlap and gaps ------------------------------------------------------------------------------------------------------------


def test_gap_minutes_count_as_disagreement():
    ex = example([row(t(10), t(11), "NO_GO")], window=(t(10), t(11)))
    got = verdict_match(ex, answer([seg(t(10), t(10, 45), "NO_GO")]))
    assert got and got["score"] == pytest.approx(0.75) and "no answer" in got["comment"] and "2026-10-31 10:45–11:00Z" in got["comment"]


def test_overlapping_answer_segments_take_the_worst_verdict():
    ex = example([row(t(10), t(11), "NO_GO")], window=(t(10), t(11)))
    ans = answer([seg(t(10), t(11), "PILOT_DISCRETION"), seg(t(10, 30), t(11), "NO_GO")])
    assert verdict_match(ex, ans)["score"] == pytest.approx(0.5)  # type: ignore[index]
    assert verdict_match(ex, answer([seg(t(10), t(11), "UNVERIFIED"), seg(t(10), t(11), "NO_GO")]))["score"] == 1  # type: ignore[index]


def test_overlapping_sheet_rows_take_the_worst_verdict_too():
    ex = example([row(t(10), t(11), "PILOT_DISCRETION"), row(t(10), t(11), "UNVERIFIED")], window=(t(10), t(11)))
    assert verdict_ious(ex, answer([seg(t(10), t(11), "UNVERIFIED")])) == {UNVERIFIED: 1.0}


# --- IoU -------------------------------------------------------------------------------------------------------------------------


def test_iou_is_per_verdict_and_a_verdict_absent_from_both_sides_is_skipped_not_one():
    ex = example([row(t(10), t(11), "NO_GO")], window=(t(10), t(11)))
    ious = verdict_ious(ex, answer([seg(t(10), t(11), "NO_GO")]))
    assert ious == {NO_GO: 1.0}  # UNVERIFIED and PILOT_DISCRETION are on neither side
    got = window_iou(ex, answer([seg(t(10), t(11), "NO_GO")]))
    assert got and got["score"] == 1 and "over 1 verdict" in got["comment"]


def test_a_verdict_only_the_answer_has_scores_zero_and_enters_the_mean():
    ex = example([row(t(10), t(11), "NO_GO")], window=(t(10), t(11)))
    ans = answer([seg(t(10), t(10, 30), "NO_GO"), seg(t(10, 30), t(11), "UNVERIFIED")])
    assert verdict_ious(ex, ans) == {NO_GO: pytest.approx(0.5), UNVERIFIED: 0.0}
    assert window_iou(ex, ans)["score"] == pytest.approx(0.25)  # type: ignore[index]
    assert "NO_GO 0.500" in window_iou(ex, ans)["comment"] and "UNVERIFIED 0.000" in window_iou(ex, ans)["comment"]  # type: ignore[index]


def test_a_labelled_verdict_the_answer_never_says_scores_zero():
    ex = example([row(t(10), t(11), "NO_GO")], window=(t(10), t(11)))
    assert verdict_ious(ex, answer()) == {NO_GO: 0.0}
