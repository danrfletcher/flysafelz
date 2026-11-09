"""noise_rate (5a PR-6.F1, G10): IGNORE-labelled NOTAMs the answer surfaced, over the IGNORE-labelled NOTAMs."""

import pytest
from agent_eval.factories import answer, example, row, seg, t
from eval_form_helpers import dev, passing_answer

from evals.agent.evaluators.blocker import surfaced
from evals.agent.evaluators.noise import NOISE_KEY, noise_rate

LABELS = {"A0001/26": "IGNORE", "A0002/26": "IGNORE", "A0003/26": "IGNORE", "A0004/26": "IGNORE"}


def run(actions, labels=None, **kwargs):
    ex = example([row(t(8), t(18), "NO_GO")], actions=labels if labels is not None else LABELS)
    return noise_rate(ex, answer([seg(t(8), t(18), "NO_GO")], actions=actions, **kwargs))


def test_none_of_the_noise_surfaced_scores_zero():
    got = run({})
    assert got and got["key"] == NOISE_KEY and got["score"] == 0 and "0 of 4" in got["comment"]


def test_a_partial_surfacing_scores_the_share_and_names_the_ids():
    got = run({"A0001/26": "AWARENESS", "A0003/26": "NO_FLY"})
    assert got and got["score"] == 0.5 and "2 of 4" in got["comment"] and "A0001/26, A0003/26" in got["comment"]


def test_all_of_the_noise_surfaced_scores_one():
    got = run({i: "AWARENESS" for i in LABELS})
    assert got and got["score"] == 1


def test_an_ignore_action_in_the_answer_is_not_surfacing():
    assert run({"A0001/26": "IGNORE", "A0002/26": "AWARENESS"})["score"] == 0.25  # type: ignore[index]


def test_it_uses_the_extraction_hard_restriction_recall_uses():
    from evals.agent.evaluators.blocker import hard_restriction_recall

    labels = {"A0001/26": "IGNORE", "B0001/26": "NO_FLY"}
    ex = example([row(t(8), t(18), "NO_GO")], actions=labels)
    ans = answer([seg(t(8), t(18), "NO_GO")], actions={"A0001/26": "AWARENESS", "B0001/26": "NO_FLY"})
    assert surfaced(ans, "A0001/26").value == "AWARENESS" and surfaced(ans, "A0002/26").value == "IGNORE"
    assert noise_rate(ex, ans)["score"] == 1 and hard_restriction_recall(ex, ans)["score"] == 1  # type: ignore[index]


def test_awareness_map_labelled_notams_are_not_in_the_denominator():
    labels = {"A0001/26": "IGNORE", "A0002/26": "AWARENESS_MAP", "A0003/26": "AWARENESS_MAP", "A0004/26": "AWARENESS"}
    got = run({"A0002/26": "AWARENESS_MAP", "A0003/26": "AWARENESS_MAP", "A0004/26": "AWARENESS"}, labels)
    assert got and got["score"] == 0 and "0 of 1" in got["comment"]
    assert run({"A0001/26": "AWARENESS"}, labels)["score"] == 1  # type: ignore[index]


def test_a_labelled_notam_with_a_null_label_counts_as_ignore():
    got = run({"A0001/26": "AWARENESS"}, {"A0001/26": None, "A0002/26": "AWARENESS"})
    assert got and got["score"] == 1 and "1 of 1" in got["comment"]


def test_no_ignore_label_is_none_not_zero():
    assert run({"A0001/26": "AWARENESS"}, {"A0001/26": "AWARENESS", "A0002/26": "AWARENESS_MAP"}) is None
    assert run({}, {}) is None


def test_null_notam_actions_means_none_so_the_score_is_none():
    ex = example([row(t(8), t(18), "NO_GO")])
    ex["reference"]["notam_actions"] = None
    assert noise_rate(ex, answer([seg(t(8), t(18), "NO_GO")], actions={"A0001/26": "AWARENESS"})) is None


def test_an_example_with_no_sheet_entry_is_none():
    ex = example(actions=LABELS, sheet=False)
    assert noise_rate(ex, answer(actions={"A0001/26": "AWARENESS"})) is None


@pytest.mark.parametrize("path", ["explain", "clarify", "decline", "act"])
def test_a_path_that_is_not_assess_is_none(path):
    assert noise_rate(example(path=path), answer(path=path, actions={"A0001/26": "AWARENESS"})) is None


def test_an_id_the_sheet_does_not_list_is_left_out():
    got = run({"A0001/26": "AWARENESS", "Z9999/26": "NO_FLY"})
    assert got and got["score"] == 0.25 and "Z9999" not in got["comment"]


def test_a_v0_answer_is_scored_too():
    assert run({"A0001/26": "AWARENESS"}, kind="v0")["score"] == 0.25  # type: ignore[index]


# --- real dev examples ------------------------------------------------------------------------------------------------------------


def test_dev_01_has_thirteen_ignore_labels_and_the_labelled_answer_scores_zero():
    ex = dev("dev_01_blue_mill")
    assert noise_rate(ex, passing_answer(ex))["score"] == 0  # type: ignore[index]
    base = passing_answer(ex)
    noisy = answer(list(base.segments), actions={**base.notam_actions, "B1434/26": "AWARENESS", "V30/26": "AWARENESS"})
    got = noise_rate(ex, noisy)
    assert got and got["score"] == pytest.approx(2 / 13) and "B1434/26, V30/26" in got["comment"]


def test_dev_03_has_no_ignore_label_so_it_is_none():
    ex = dev("dev_03_buttermere")
    assert noise_rate(ex, passing_answer(ex)) is None
