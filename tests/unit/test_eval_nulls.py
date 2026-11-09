"""E5 (5a PR-6.F1): an evaluator that does not apply returns None, never 0 or False. A rule-driven matrix: evaluator x path x run kind (v1, v0).

Each evaluator has one rule (RULES) saying where it applies. The example for each path carries every reference field the evaluators read, so the only thing that
decides None is the rule under test: the path, the run kind, or (for the field-driven ones) the absent field.
"""

import copy

import pytest
from agent_eval.factories import answer, example, row, seg, t

from evals.agent.evaluators.aggregate import EVALUATORS
from evals.agent.evaluators.answer import ToolCall

PATHS = ("assess", "explain", "clarify", "decline", "act")
KINDS = ("v1", "v0")
NOW, WINDOW = t(6, 30), (t(9), t(11))

# evaluator -> (paths it applies to, run kinds it applies to)
EVERY_PATH, ASSESS_ONLY = set(PATHS), {"assess"}
BOTH, V1_ONLY = {"v1", "v0"}, {"v1"}
RULES = {
    "noise_rate": (ASSESS_ONLY, BOTH),
    "required_tools": (EVERY_PATH, V1_ONLY),
    "forbidden_tools": (EVERY_PATH, V1_ONLY),
    "step_cap": (EVERY_PATH, V1_ONLY),
    "guard_pass": (EVERY_PATH, V1_ONLY),
    "decision_first": (ASSESS_ONLY, V1_ONLY),
    "cited": (EVERY_PATH, V1_ONLY),
    "window_stated": (ASSESS_ONLY, BOTH),
    "recheck_stated": (ASSESS_ONLY, V1_ONLY),
    "must_mention": (EVERY_PATH, BOTH),
    "must_not_mention": (EVERY_PATH, BOTH),
    "no_verdict_beyond_day3": (ASSESS_ONLY, BOTH),
    "unverified_on_dropped_source": (ASSESS_ONLY, BOTH),
}
REFERENCE = {
    "required_tools": ["get_notams", "search_guidance"],
    "forbidden_tools": ["assess"],
    "must_mention": ["H6217", "Sherburn"],
    "must_not_mention": ["NO-GO"],
}


def full_example(path):
    ex = example(
        [row(t(9), t(11), "NO_GO")] if path == "assess" else None,
        path=path,
        window=WINDOW if path == "assess" else None,
        now=NOW,
        actions={"A0001/26": "IGNORE"} if path == "assess" else None,
        drop=("airspace",) if path == "assess" else (),
        extra=REFERENCE,
    )
    return ex


def full_answer(path, kind):
    return answer(
        [seg(t(9), t(11), "UNVERIFIED")] if path == "assess" else None,
        path=path,
        actions={"A0001/26": "AWARENESS"} if path == "assess" else None,
        text="UNVERIFIED. 09:00 to 11:00. H6217 Sherburn CAP 1535 airspace",
        tool_calls=[ToolCall("get_notams"), ToolCall("search_guidance", ("CAP 1535",))] if kind == "v1" else [],
        guard="passed" if kind == "v1" else None,
        kind=kind,
    )


def test_every_rule_names_a_registered_evaluator_and_the_key_it_returns():
    """The earlier PRs' evaluators have their own null tests; this matrix is the ones PR-6.F1 adds plus the two safety checks the spec lists for v0."""
    assert set(RULES) <= set(EVALUATORS)


CASES = [(key, path, kind) for key in RULES for path in PATHS for kind in KINDS]


@pytest.mark.parametrize(("key", "path", "kind"), CASES, ids=[f"{k}-{p}-{r}" for k, p, r in CASES])
def test_an_evaluator_returns_none_where_its_rule_says_it_does_not_apply_and_a_score_where_it_does(key, path, kind):
    paths, kinds = RULES[key]
    got = EVALUATORS[key](full_example(path), full_answer(path, kind))
    if path in paths and kind in kinds:
        assert got is not None and got["key"] == key
        assert isinstance(got["score"], int | float) and not isinstance(got["score"], bool) and 0 <= got["score"] <= 1
    else:
        assert got is None


# --- the field-driven ones: an absent field is None on a path and a kind where the evaluator otherwise applies -----------------------

FIELD_RULES = [
    ("required_tools", "required_tools"),
    ("forbidden_tools", "forbidden_tools"),
    ("must_mention", "must_mention"),
    ("must_not_mention", "must_not_mention"),
    ("cited", "must_mention"),
]


@pytest.mark.parametrize(("key", "field"), FIELD_RULES)
@pytest.mark.parametrize("path", PATHS)
def test_an_absent_reference_list_is_none(key, field, path):
    ex = full_example(path)
    del ex["reference"][field]
    if key == "cited":
        ex["reference"]["required_tools"] = ["get_notams"]  # no search_guidance, so no list is left to cite
    assert EVALUATORS[key](ex, full_answer(path, "v1")) is None


def test_noise_is_none_without_an_ignore_label_or_a_sheet_entry():
    no_ignore = full_example("assess")
    no_ignore["reference"]["notam_actions"] = {"A0001/26": "AWARENESS"}
    assert EVALUATORS["noise_rate"](no_ignore, full_answer("assess", "v1")) is None
    bare = copy.deepcopy(full_example("assess"))
    for field in ("verdict_by_segment", "tier", "notam_actions"):
        bare["reference"].pop(field)
    assert EVALUATORS["noise_rate"](bare, full_answer("assess", "v1")) is None


def test_the_dropped_source_check_is_none_with_no_drop():
    ex = full_example("assess")
    ex["fixtures"]["drop"] = []
    assert EVALUATORS["unverified_on_dropped_source"](ex, full_answer("assess", "v0")) is None


def test_the_matrix_has_every_path_and_both_kinds_for_every_rule():
    assert len(CASES) == len(RULES) * len(PATHS) * len(KINDS)
    assert all(paths <= set(PATHS) and kinds <= set(KINDS) for paths, kinds in RULES.values())
