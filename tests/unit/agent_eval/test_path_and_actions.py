"""G9 path accuracy and confusion (with I5's metadata source and E9's compound example) and the NOTAM action confusion (5a PR-5.F1)."""

import pytest

from agent_eval.factories import answer, example
from evals.agent.evaluators.aggregate import ScoredRun, score_run
from evals.agent.evaluators.answer import PATHS, path_from_metadata
from evals.agent.evaluators.paths import (
    PATH_COLUMNS,
    PATH_ROWS,
    action_confusion,
    action_matches,
    action_pairs,
    notam_action_match,
    path_accuracy,
    path_confusion,
    path_pair,
)
from flysafelz.contracts import NotamAction

ID = "H4821/26"


def predicted(metadata):
    """The answer a run with this root-run metadata gives: the path is read from the metadata, never from the text."""
    return answer(path=path_from_metadata(metadata), text="This is an assess answer. Verdict: NO_GO.")


# --- path accuracy (I5) ----------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("path", PATHS)
def test_path_accuracy_applies_to_every_path_and_is_one_when_the_metadata_path_is_the_primary_path(path):
    got = path_accuracy(example(path=path), predicted({"path": path}))
    assert got["key"] == "path_accuracy" and got["score"] == 1 and "hit" in got["comment"]


def test_a_wrong_path_scores_zero_and_the_text_is_never_read():
    got = path_accuracy(example(path="assess"), predicted({"path": "explain"}))  # the text says assess; the tag says explain
    assert got["score"] == 0 and "predicted path explain, the example's path is assess: a miss" in got["comment"]


@pytest.mark.parametrize("metadata", [{}, None, {"path": "unknown"}, {"path": "teleport"}, {"path": None}, {"other": "assess"}])
def test_a_missing_or_unknown_path_tag_is_a_miss_in_the_unknown_column(metadata):
    ans = predicted(metadata)
    assert ans.path == "unknown"
    got = path_accuracy(example(path="assess"), ans)
    assert got["score"] == 0 and "tag was missing" in got["comment"]
    assert path_confusion([path_pair(example(path="assess"), ans)])["assess"]["unknown"] == 1


def test_the_metadata_path_is_matched_whatever_its_case():
    assert path_from_metadata({"path": " Assess "}) == "assess"


def test_e9_a_compound_example_is_scored_on_its_primary_path_only():
    ex = example(path="assess", compound=True)
    ex["reference"]["parts"] = ["assess", "explain"]
    assert path_accuracy(ex, predicted({"path": "assess"}))["score"] == 1  # also explains, primary right
    assert path_accuracy(ex, predicted({"path": "explain"}))["score"] == 0  # the second part is not a way in


def test_e9_a_missing_compound_key_is_false_and_scores_on_the_primary_path_without_a_key_error():
    plain = example(path="assess")
    del plain["metadata"]["compound"]  # the factory always writes the key; take it out so the case really runs
    assert "compound" not in plain["metadata"]
    right, wrong = predicted({"path": "assess"}), predicted({"path": "explain"})
    assert path_accuracy(plain, right)["score"] == 1
    assert path_accuracy(plain, wrong)["score"] == 0
    assert score_run(ScoredRun(plain, right)).feedback["path_accuracy"]["score"] == 1  # type: ignore[index]
    assert score_run(ScoredRun(plain, wrong)).feedback["path_accuracy"]["score"] == 0  # type: ignore[index]


def test_an_example_with_no_primary_path_or_an_unknown_one_is_a_named_error():
    for reference in ({}, {"path": ""}, {"path": "fly"}):
        ex = example()
        ex["reference"] = reference
        with pytest.raises(ValueError, match="example syn_1"):
            path_accuracy(ex, answer())


# --- path confusion --------------------------------------------------------------------------------------------------------------


def test_the_confusion_matrix_has_five_rows_and_the_five_paths_plus_unknown_as_columns_keyed_by_name():
    matrix = path_confusion([])
    assert list(matrix) == list(PATH_ROWS) == ["assess", "explain", "act", "clarify", "decline"]
    assert all(list(r) == list(PATH_COLUMNS) == [*PATH_ROWS, "unknown"] and set(r.values()) == {0} for r in matrix.values())


def test_the_confusion_matrix_counts_every_labelled_and_predicted_pair_over_all_five_paths():
    pairs = [(a, b) for a in PATHS for b in PATHS if a != b] + [(p, p) for p in PATHS] * 2 + [("act", "unknown")]
    matrix = path_confusion(pairs)
    for a in PATHS:
        for b in PATHS:
            assert matrix[a][b] == (2 if a == b else 1)
    assert matrix["act"]["unknown"] == 1 and sum(matrix["assess"]["unknown"] for _ in [0]) == 0
    assert sum(sum(r.values()) for r in matrix.values()) == len(pairs)


@pytest.mark.parametrize("pair", [("fly", "assess"), ("assess", "fly"), ("unknown", "assess")])
def test_a_pair_outside_the_matrix_is_refused(pair):
    with pytest.raises(ValueError, match="outside the matrix"):
        path_confusion([pair])


# --- NOTAM actions ---------------------------------------------------------------------------------------------------------------


def test_awareness_map_predicted_for_an_awareness_label_is_a_match_and_the_reverse_is_not():
    assert action_matches(NotamAction.AWARENESS, NotamAction.AWARENESS_MAP)
    assert not action_matches(NotamAction.AWARENESS_MAP, NotamAction.AWARENESS)
    assert action_matches(NotamAction.AWARENESS, NotamAction.AWARENESS)
    assert action_matches(NotamAction.AWARENESS_MAP, NotamAction.AWARENESS_MAP)


@pytest.mark.parametrize("label", list(NotamAction))
@pytest.mark.parametrize("shown", list(NotamAction))
def test_every_other_pair_is_a_match_only_when_equal(label, shown):
    expected = label is shown or (label is NotamAction.AWARENESS and shown is NotamAction.AWARENESS_MAP)
    got = notam_action_match(example(actions={ID: label.value}), answer(actions={ID: shown}))
    assert got and got["key"] == "notam_action_match" and got["score"] == (1 if expected else 0)


def test_the_confusion_has_every_notam_action_on_both_axes_so_a_new_member_needs_no_code_change():
    matrix = action_confusion([])
    names = [a.value for a in NotamAction]
    assert list(matrix) == names and all(list(r) == names for r in matrix.values())
    assert "CHECK_SUP" in names


def test_the_confusion_counts_label_by_predicted_over_the_sheets_notams():
    ex = example(actions={"A1/26": "NO_FLY", "A2/26": "AWARENESS", "A3/26": "AWARENESS", "A4/26": None})
    ans = answer(actions={"A1/26": NotamAction.CONTACT_FIRST, "A2/26": NotamAction.AWARENESS_MAP, "A9/26": NotamAction.NO_FLY})
    pairs = action_pairs(ex, ans)
    assert pairs is not None and len(pairs) == 4  # the sheet's four; A9/26 is not one of them
    matrix = action_confusion(pairs)
    assert matrix["NO_FLY"]["CONTACT_FIRST"] == 1 and matrix["AWARENESS"]["AWARENESS_MAP"] == 1
    assert matrix["AWARENESS"]["IGNORE"] == 1  # A3/26 is not in the answer
    assert matrix["IGNORE"]["IGNORE"] == 1  # A4/26 has a null label: IGNORE, as in the sheet
    assert sum(sum(r.values()) for r in matrix.values()) == 4


def test_a_notam_the_sheet_does_not_list_is_left_out_and_named_and_ids_compare_without_whitespace():
    ex = example(actions={ID: "NO_FLY"})
    got = notam_action_match(ex, answer(actions={"H4821 /26": NotamAction.NO_FLY, "Q7/26": NotamAction.NO_FLY}))
    assert got and got["score"] == 1 and "Q7/26" in got["comment"] and "not in the sheet" in got["comment"]
    wrong = notam_action_match(ex, answer(actions={"H4821/26": NotamAction.IGNORE}))
    assert wrong and wrong["score"] == 0 and "H4821/26 labelled NO_FLY, answered IGNORE" in wrong["comment"]


def test_no_labelled_notams_non_assess_and_no_sheet_entry_give_none():
    assert notam_action_match(example(actions={}), answer(actions={ID: NotamAction.NO_FLY})) is None
    assert notam_action_match(example(path="act"), answer(path="act")) is None
    assert notam_action_match(example(sheet=False), answer()) is None
    assert action_pairs(example(sheet=False), answer()) is None
