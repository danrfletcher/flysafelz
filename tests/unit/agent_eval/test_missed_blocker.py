"""G8 harmful_missed_blocker and hard-restriction recall (5a PR-5.F1): both triggers, the SEVERITY ranking, a missing NOTAM as IGNORE, E12, recall with no denominator."""

import pytest

from agent_eval.factories import answer, example, row, seg, t
from evals.agent.evaluators.blocker import hard_restriction_recall, harmful_missed_blocker, unlisted_ids
from evals.agent.evaluators.labels import sheet_labels
from evals.reader_eval.downgrade import SEVERITY
from flysafelz.contracts import NotamAction

NO_FLY, CONTACT = NotamAction.NO_FLY, NotamAction.CONTACT_FIRST
HARD = (NO_FLY, CONTACT)
ID = "H4821/26"


def blocker(ex, ans):
    return harmful_missed_blocker(ex, ans)


# --- trigger 1: PILOT_DISCRETION over a NO_GO or UNVERIFIED label ---------------------------------------------------------------


@pytest.mark.parametrize("label", ["NO_GO", "UNVERIFIED"])
def test_pilot_discretion_over_a_no_go_or_unverified_minute_is_a_missed_blocker(label):
    ex = example([row(t(10), t(12), label)])
    got = blocker(ex, answer([seg(t(10), t(11), "NO_GO"), seg(t(11), t(11, 30), "PILOT_DISCRETION"), seg(t(11, 30), t(12), "NO_GO")]))
    assert got and got["key"] == "harmful_missed_blocker" and got["score"] == 1
    assert "syn_1" in got["comment"] and "2026-10-31 11:00–11:30Z" in got["comment"] and "30 min" in got["comment"]


@pytest.mark.parametrize(
    ("label", "said"),
    [("PILOT_DISCRETION", "NO_GO"), ("PILOT_DISCRETION", "UNVERIFIED"), ("NO_GO", "NO_GO"), ("NO_GO", "UNVERIFIED"), ("UNVERIFIED", "UNVERIFIED")],
)
def test_a_stricter_or_equal_answer_is_not_a_missed_blocker(label, said):
    got = blocker(example([row(t(10), t(11), label)]), answer([seg(t(10), t(11), said)]))
    assert got and got["score"] == 0


def test_a_gap_is_not_pilot_discretion_so_it_is_not_this_trigger():
    got = blocker(example([row(t(10), t(11), "NO_GO")]), answer([seg(t(10), t(10, 30), "NO_GO")]))
    assert got and got["score"] == 0


def test_pilot_discretion_over_an_outlook_minute_or_outside_the_window_is_not_a_missed_blocker():
    ex = example([row(t(8), t(9), "NO_GO")], window=(t(8), t(10)), tiers=[{"start": "2026-10-31T09:00:00Z", "end": "2026-10-31T10:00:00Z", "tier": "OUTLOOK"}])
    got = blocker(ex, answer([seg(t(8), t(9), "NO_GO"), seg(t(9), t(14), "PILOT_DISCRETION")]))
    assert got and got["score"] == 0


# --- trigger 2: a hard NOTAM surfaced with a weaker action -----------------------------------------------------------------------


@pytest.mark.parametrize("label", HARD)
@pytest.mark.parametrize("shown", list(NotamAction))
def test_a_hard_notam_surfaced_with_a_weaker_action_is_a_missed_blocker_for_every_pair(label, shown):
    got = blocker(example(actions={ID: label.value}), answer(actions={ID: shown}))
    weaker = SEVERITY.index(shown) > SEVERITY.index(label)
    assert got and got["score"] == (1 if weaker else 0), (label, shown)
    if weaker:
        assert ID in got["comment"] and label.value in got["comment"] and shown.value in got["comment"]


@pytest.mark.parametrize(
    ("label", "shown", "harmful"),
    [
        (NO_FLY, CONTACT, True),
        (CONTACT, NO_FLY, False),
        (CONTACT, NotamAction.CHECK_SUP, False),  # CHECK_SUP is stricter than CONTACT_FIRST
        (CONTACT, NotamAction.AWARENESS, True),
        (NO_FLY, NotamAction.UNREAD_OVER_LZ, True),
    ],
)
def test_the_ranking_reads_as_the_reader_eval_ranks_it(label, shown, harmful):
    got = blocker(example(actions={ID: label.value}), answer(actions={ID: shown}))
    assert got and got["score"] == int(harmful)


def test_a_hard_notam_missing_from_the_answer_counts_as_surfaced_with_ignore():
    got = blocker(example(actions={ID: "NO_FLY", "H1/26": "CONTACT_FIRST"}), answer(actions={ID: NO_FLY}))
    assert got and got["score"] == 1 and "H1/26" in got["comment"] and "IGNORE" in got["comment"] and ID not in got["comment"]


@pytest.mark.parametrize("label", ["AWARENESS", "AWARENESS_MAP", "AVOID_SECTOR", "IGNORE", None])
def test_a_notam_labelled_below_the_hard_actions_is_never_a_missed_blocker(label):
    got = blocker(example(actions={ID: label}), answer())
    assert got and got["score"] == 0


def test_a_notam_id_is_compared_without_whitespace_and_keeps_its_year():
    ex = example(actions={"H4821/26": "NO_FLY"})
    assert blocker(ex, answer(actions={"H4821 /26": NO_FLY}))["score"] == 0  # type: ignore[index]
    assert blocker(ex, answer(actions={"H4821/25": NO_FLY}))["score"] == 1  # type: ignore[index]  # another NOTAM


def test_both_triggers_are_named_in_one_comment():
    ex = example([row(t(10), t(11), "NO_GO")], actions={ID: "NO_FLY"})
    got = blocker(ex, answer([seg(t(10), t(11), "PILOT_DISCRETION")], actions={ID: CONTACT}))
    assert got and got["score"] == 1 and "2026-10-31 10:00–11:00Z" in got["comment"] and f"{ID} labelled NO_FLY, surfaced as CONTACT_FIRST" in got["comment"]


# --- E12 -------------------------------------------------------------------------------------------------------------------------


def test_e12_the_fourteen_minutes_of_pilot_discretion_over_the_no_go_are_a_missed_blocker():
    ex = example([row(t(16), t(17, 6), "PILOT_DISCRETION", edge="civil dusk"), row(t(17, 6), t(18), "NO_GO")], window=(t(16), t(18)))
    ans = answer([seg(t(16), t(17, 20), "PILOT_DISCRETION"), seg(t(17, 20), t(18), "NO_GO")])
    got = blocker(ex, ans)
    assert got and got["score"] == 1 and "2026-10-31 17:06–17:20Z" in got["comment"] and "14 min" in got["comment"]


# --- None where it does not apply ------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("evaluate", [harmful_missed_blocker, hard_restriction_recall])
def test_non_assess_examples_and_examples_without_a_finalised_sheet_entry_give_none(evaluate):
    assert evaluate(example(path="explain"), answer(path="explain")) is None
    assert evaluate(example(sheet=False), answer()) is None


# --- recall ----------------------------------------------------------------------------------------------------------------------


def test_recall_is_none_when_nothing_is_labelled_hard_never_one_or_zero():
    assert hard_restriction_recall(example(actions={}), answer()) is None
    assert hard_restriction_recall(example(actions={ID: "AWARENESS", "H2/26": None}), answer(actions={ID: NO_FLY})) is None


def test_recall_is_the_hard_notams_surfaced_with_that_exact_action_over_those_labelled():
    ex = example(actions={"A1/26": "NO_FLY", "A2/26": "NO_FLY", "A3/26": "CONTACT_FIRST", "A4/26": "CONTACT_FIRST", "A5/26": "AWARENESS"})
    ans = answer(actions={"A1/26": NO_FLY, "A2/26": CONTACT, "A3/26": CONTACT, "A5/26": NotamAction.AWARENESS})
    got = hard_restriction_recall(ex, ans)
    assert got and got["key"] == "hard_restriction_recall" and got["score"] == pytest.approx(0.5)
    assert "A2/26 labelled NO_FLY, surfaced as CONTACT_FIRST" in got["comment"] and "A4/26 labelled CONTACT_FIRST, surfaced as IGNORE" in got["comment"]


def test_a_stricter_action_is_not_a_recall_hit_even_though_it_is_not_a_missed_blocker():
    ex = example(actions={ID: "CONTACT_FIRST"})
    ans = answer(actions={ID: NO_FLY})
    assert hard_restriction_recall(ex, ans)["score"] == 0  # type: ignore[index]
    assert blocker(ex, ans)["score"] == 0  # type: ignore[index]


def test_a_notam_the_sheet_does_not_list_is_ignored_by_recall_and_named_in_the_comment():
    ex = example(actions={ID: "NO_FLY"})
    ans = answer(actions={ID: NO_FLY, "Z9 /26": CONTACT})
    got = hard_restriction_recall(ex, ans)
    assert got and got["score"] == 1 and "Z9/26" in got["comment"] and "not in the sheet" in got["comment"]
    assert unlisted_ids(sheet_labels(ex), ans) == ["Z9/26"]  # type: ignore[arg-type]
    assert "Z9/26" in blocker(ex, ans)["comment"]  # type: ignore[index]
