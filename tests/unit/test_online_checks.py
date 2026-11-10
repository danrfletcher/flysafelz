"""The online checks (7 PR-2.F1: G7, E2, E9, the 'check error' and score-normalisation rules): ONLINE_CHECKS wiring, each check on the synthetic reference, the null reasons.

Every case builds a fake trace (tests/online_helpers.py), builds the synthetic reference from it with `build_reference`, and calls `job.score_run`: the real 5a evaluators run on the real reference.
"""

import math

import pytest
from online_helpers import NOW, FakeTree, assess_artifact, fake_run, fake_tree, guidance_run, scored_tree, tool_run

from evals.agent.evaluators import form, safety, trajectory
from evals.online import job
from evals.online.reference import build_reference

GP3_KEYS = [
    "online.guard_pass",
    "online.decision_first",
    "online.cited_guidance",
    "online.recheck_stated",
    "online.assumed_window_stated",
    "online.no_verdict_beyond_day3",
    "online.step_cap",
    "online.decline_no_tools",
]


def scores(tree: FakeTree) -> dict[str, job.KeyScore]:
    return job.score_run(tree.root, build_reference(tree.root, tree.children), tree.children)


def pairs(tree: FakeTree) -> dict[str, int | None]:
    return {key: kept.score for key, kept in scores(tree).items()}


# --- the wiring (G7) ------------------------------------------------------------------------------------------------------------------------


def test_the_online_checks_are_5as_functions_under_the_online_names():
    assert job.ONLINE_CHECKS == {
        "guard_pass": form.guard_pass,
        "decision_first": form.decision_first,
        "cited_guidance": form.cited_guidance,
        "recheck_stated": form.recheck_stated,
        "assumed_window_stated": form.window_stated,
        "no_verdict_beyond_day3": safety.no_verdict_beyond_day3,
        "step_cap": trajectory.step_cap_ok,
        "decline_no_tools": trajectory.forbidden_tools,
    }
    assert all(job.ONLINE_CHECKS[name] is function for name, function in job.ONLINE_CHECKS.items())


def test_the_key_names_are_the_exact_gp3_list_in_posting_order():
    assert list(job.DETERMINISTIC_KEYS) == GP3_KEYS
    assert job.VERSION_KEY == "online.evaluator_version"
    assert [job.feedback_key(name) for name in job.ONLINE_CHECKS] == GP3_KEYS


@pytest.mark.parametrize(
    ("constant", "name"),
    [
        ("GUARD_PASS", "guard_pass"),
        ("DECISION_FIRST", "decision_first"),
        ("CITED_GUIDANCE", "cited_guidance"),
        ("RECHECK_STATED", "recheck_stated"),
        ("ASSUMED_WINDOW_STATED", "assumed_window_stated"),
        ("NO_VERDICT_BEYOND_DAY3", "no_verdict_beyond_day3"),
        ("STEP_CAP", "step_cap"),
        ("DECLINE_NO_TOOLS", "decline_no_tools"),
    ],
)
def test_each_check_name_is_a_module_constant(constant, name):
    assert getattr(job, constant) == name
    assert name in job.ONLINE_CHECKS


def test_the_applicable_keys_are_the_deterministic_keys_for_any_run():
    tree = scored_tree("a")
    assert job.applicable_keys(tree.root, build_reference(tree.root, tree.children)) == tuple(GP3_KEYS)
    bare = fake_tree(path="explain")
    assert job.applicable_keys(bare.root, build_reference(bare.root, bare.children)) == tuple(GP3_KEYS)


# --- each check, called with the synthetic reference --------------------------------------------------------------------------------------


def test_a_good_assess_run_scores_one_on_every_check_but_the_decline_check():
    found = scores(scored_tree("good"))
    assert list(found) == GP3_KEYS
    assert {key: kept.score for key, kept in found.items()} == {**dict.fromkeys(GP3_KEYS, 1), "online.decline_no_tools": None}
    assert found["online.decline_no_tools"].comment == job.NOT_APPLICABLE
    assert all(kept.comment for kept in found.values())


def test_a_score_is_the_integer_one_zero_or_none_never_a_bool_or_float():
    for kept in scores(scored_tree("good", guard="retried")).values():
        assert kept.score is None or (type(kept.score) is int and kept.score in (0, 1))


@pytest.mark.parametrize("guard", ["retried", "fallback"])
def test_a_guard_that_did_not_pass_the_first_draft_scores_zero_with_5as_reason(guard):
    kept = scores(scored_tree("g", guard=guard))["online.guard_pass"]
    assert kept.score == 0 and guard in kept.comment


def test_an_answer_that_does_not_lead_with_a_verdict_fails_decision_first():
    tree = scored_tree("d", answer="Some thoughts about the weather. PILOT DISCRETION later.")
    kept = scores(tree)["online.decision_first"]
    assert kept.score == 0 and "no verdict word" in kept.comment


def test_an_answer_that_leaves_out_the_window_fails_the_window_check():
    tree = scored_tree("w", answer="PILOT DISCRETION. See CAP 1535.")
    assert scores(tree)["online.assumed_window_stated"].score == 0


def test_an_answer_without_the_recheck_line_fails_when_the_window_starts_later():
    tree = scored_tree("r", answer="PILOT DISCRETION: fly between 10:00 and 16:00. See CAP 1535.")
    kept = scores(tree)["online.recheck_stated"]
    assert kept.score == 0 and "lacks the re-check line" in kept.comment


def test_a_guidance_document_the_answer_does_not_name_fails_cited_guidance():
    tree = scored_tree("c", documents=("CAP 1535", "BHPA Pilot Handbook"), answer="PILOT DISCRETION: fly between 10:00 and 16:00. Be careful.")
    assert scores(tree)["online.cited_guidance"].score == 0


def test_a_run_that_never_called_search_guidance_has_cited_guidance_null_not_applicable():
    root = fake_run("nog", guard="passed")
    kept = scores(FakeTree(root, [tool_run("assess", root)]))["online.cited_guidance"]
    assert kept.score is None and kept.comment == job.NOT_APPLICABLE


def test_a_segment_beyond_day_three_fails_that_check():
    late = assess_artifact(segments=(("2026-11-04T10:00:00+00:00", "2026-11-04T11:00:00+00:00", "PILOT_DISCRETION"),))
    root = fake_run("late", guard="passed")
    wrapped = {"output": {"update": {"messages": [{"type": "tool", "content": "...", "artifact": late}]}}}
    tree = FakeTree(root, [tool_run("assess", root, outputs=wrapped)])
    assert scores(tree)["online.no_verdict_beyond_day3"].score == 0


def test_the_step_cap_counts_every_tool_run_and_passes_at_the_cap():
    tree = scored_tree("steps")
    assert scores(tree)["online.step_cap"].score == 1
    cap = trajectory.step_cap()
    root = fake_run("many", guard="passed")
    over = FakeTree(root, [tool_run("get_weather", root, minutes=n + 1) for n in range(cap + 1)])
    assert scores(over)["online.step_cap"].score == 0


def test_a_decline_run_that_called_a_tool_fails_and_one_that_called_none_passes():
    root = fake_run("dec", path="decline", answer="I can only help with flying conditions.", guard="passed")
    called = FakeTree(root, [tool_run("get_weather", root)])
    clean = FakeTree(root, [])
    assert scores(called)["online.decline_no_tools"].score == 0
    assert "get_weather" in scores(called)["online.decline_no_tools"].comment
    assert scores(clean)["online.decline_no_tools"].score == 1


def test_a_non_decline_path_has_no_forbidden_tools_so_the_check_is_null():
    assert scores(fake_tree(path="explain", guard="passed"))["online.decline_no_tools"].score is None


def test_tool_calls_are_read_in_start_order_with_the_documents_search_guidance_returned():
    root = fake_run("order")
    late, early = guidance_run(root, ("CAP 1535", "CAP 1535"), minutes=5), tool_run("get_weather", root, minutes=2)
    calls = job.tool_calls_of([late, early, root])
    assert [(call.name, call.guides) for call in calls] == [("get_weather", ()), ("search_guidance", ("CAP 1535",))]


def test_a_guard_record_that_is_not_a_guard_result_counts_as_none():
    assert job.guard_of(fake_run("x", guard="passed")) == "passed"
    assert job.guard_of(fake_run("x")) is None
    assert job.guard_of(fake_run("x", guard="odd")) is None


# --- null reasons -----------------------------------------------------------------------------------------------------------------------------


def test_g7_a_run_with_no_guard_record_gets_guard_pass_null_with_the_reason():
    kept = scores(scored_tree("noguard", guard=None))["online.guard_pass"]
    assert kept.score is None and kept.comment == "no guard record"
    assert pairs(scored_tree("noguard", guard=None))["online.decision_first"] == 1  # the other checks still score


def test_e9_a_run_with_no_assess_child_nulls_the_checks_that_need_the_window_with_the_reason():
    root = fake_run("noassess", guard="passed", answer="PILOT DISCRETION: fly between 10:00 and 16:00. See CAP 1535.")
    found = scores(FakeTree(root, [guidance_run(root)]))
    needing = ["online.decision_first", "online.recheck_stated", "online.assumed_window_stated", "online.no_verdict_beyond_day3"]
    for key in needing:
        assert (found[key].score, found[key].comment) == (None, "no assess"), key
    assert found["online.guard_pass"].score == 1 and found["online.cited_guidance"].score == 1 and found["online.step_cap"].score == 1


def test_e9_an_assess_run_whose_output_holds_no_artifact_counts_as_no_assess():
    root = fake_run("broken", guard="passed")
    found = scores(FakeTree(root, [tool_run("assess", root, outputs={"output": {"content": "no artifact"}})]))
    assert found["online.decision_first"].comment == "no assess"


def test_e2_path_unknown_nulls_the_decline_check_with_the_reason_and_the_other_checks_still_score():
    tree = scored_tree("unk", path="unknown")
    found = scores(tree)
    assert (found["online.decline_no_tools"].score, found["online.decline_no_tools"].comment) == (None, "path unknown")
    assert found["online.guard_pass"].score == 1 and found["online.cited_guidance"].score == 1 and found["online.step_cap"].score == 1
    assert build_reference(tree.root, tree.children).path == "unknown"


def test_a_5a_function_that_raises_gives_its_key_null_with_the_exception_class_and_the_other_keys_post(monkeypatch):
    def broken(example, answer):
        raise KeyError("boom")

    monkeypatch.setitem(job.ONLINE_CHECKS, "decision_first", broken)
    found = scores(scored_tree("err"))
    assert (found["online.decision_first"].score, found["online.decision_first"].comment) == (None, "check error: KeyError")
    assert "boom" not in found["online.decision_first"].comment
    assert found["online.guard_pass"].score == 1 and found["online.step_cap"].score == 1
    assert list(found) == GP3_KEYS


# --- score normalisation --------------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(("given", "expected"), [(True, 1), (False, 0), (1, 1), (0, 0), (1.0, 1), (0.0, 0), (None, None)])
def test_normalise_score_maps_booleans_integers_floats_and_none_to_one_zero_or_null(given, expected):
    result = job.normalise_score(given)
    assert result == expected and type(result) is (type(None) if expected is None else int)


@pytest.mark.parametrize("given", [0.5, 2, -1, "1", "pass", math.nan, math.inf, [1], {}])
def test_normalise_score_refuses_any_other_value(given):
    with pytest.raises(ValueError, match="1, 0 or None"):
        job.normalise_score(given)


def test_a_check_that_returns_a_fractional_score_fails_loudly_and_is_not_swallowed_as_a_check_error(monkeypatch):
    monkeypatch.setitem(job.ONLINE_CHECKS, "step_cap", lambda example, answer: {"key": "step_cap", "score": 0.5, "comment": "half"})
    with pytest.raises(ValueError, match="0.5"):
        scores(scored_tree("frac"))


def test_a_check_that_returns_a_boolean_score_is_posted_as_one_or_zero(monkeypatch):
    monkeypatch.setitem(job.ONLINE_CHECKS, "step_cap", lambda example, answer: {"key": "step_cap", "score": False, "comment": "no"})
    kept = scores(scored_tree("bool"))["online.step_cap"]
    assert kept.score == 0 and type(kept.score) is int


def test_the_reference_the_checks_get_is_built_from_the_run_alone():
    tree = scored_tree("ref")
    reference = build_reference(tree.root, tree.children)
    example = job.example_of(tree.root, reference)
    assert set(example) == {"id", "inputs", "reference"}
    assert set(example["reference"]) == {"path", "required_tools", "forbidden_tools", "window", "verdict_by_segment", "tier", "notam_actions"}
    assert example["inputs"] == {"now": NOW.replace(hour=6).isoformat()}
    assert example["reference"]["required_tools"] == ["search_guidance"] and example["reference"]["notam_actions"] == {}
