"""The evaluator version (7 PR-2.F1, G10): 12 lowercase hex characters over the files that decide a score. Every case works on a copy of the covered files in `tmp_path`."""

import re
import shutil
from pathlib import Path

import pytest

from evals.online import job

ROOT = Path(__file__).resolve().parents[2]
COVERED = job.covered_files(ROOT)


@pytest.fixture
def tree(tmp_path) -> Path:
    """A copy of the covered files, at the same relative paths."""
    for relative in COVERED:
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, target)
    return tmp_path


def test_the_covered_files_are_5as_evaluators_the_judge_its_examples_and_the_online_modules_except_seed():
    assert "evals/agent/judge.py" in COVERED and "evals/agent/judge_examples.yaml" in COVERED
    assert {f"evals/agent/evaluators/{name}.py" for name in ("form", "safety", "trajectory", "answer", "labels", "mentions")} <= set(COVERED)
    assert {f"evals/online/{name}.py" for name in ("job", "pull", "reference", "runs")} <= set(COVERED)
    assert "evals/online/seed.py" not in COVERED
    assert COVERED == sorted(COVERED)
    assert all(not name.startswith(("src/", "tests/")) and "__pycache__" not in name for name in COVERED)


def test_the_version_is_exactly_twelve_lowercase_hex_characters(tree):
    assert re.fullmatch(r"[0-9a-f]{12}", job.evaluator_version(tree))


def test_the_version_is_stable_across_calls_and_the_same_for_the_same_bytes_at_the_same_paths(tree):
    assert job.evaluator_version(tree) == job.evaluator_version(tree) == job.evaluator_version(ROOT) == job.evaluator_version()


def test_the_version_does_not_depend_on_the_order_the_files_are_listed_in(tree, monkeypatch):
    first = job.evaluator_version(tree)
    original = Path.glob

    def reversed_glob(self, pattern, *args, **kwargs):
        return iter(sorted(original(self, pattern, *args, **kwargs), reverse=True))

    monkeypatch.setattr(Path, "glob", reversed_glob)
    assert job.evaluator_version(tree) == first


@pytest.mark.parametrize("relative", COVERED)
def test_changing_any_covered_file_changes_the_version(tree, relative):
    before = job.evaluator_version(tree)
    with (tree / relative).open("ab") as handle:
        handle.write(b"\n")
    assert job.evaluator_version(tree) != before


def test_changing_seed_py_does_not_change_the_version_and_neither_does_a_file_nobody_scores_with(tree):
    before = job.evaluator_version(tree)
    (tree / "evals/online/seed.py").write_text("QUESTIONS = 11\n", encoding="utf-8")
    assert job.evaluator_version(tree) == before
    (tree / "evals/online/seed.py").write_text("QUESTIONS = 12\n", encoding="utf-8")
    (tree / "evals/online/seed_questions.yaml").write_text("- id: a\n", encoding="utf-8")
    (tree / "evals/agent/runner.py").write_text("x = 1\n", encoding="utf-8")
    (tree / "evals/online/__pycache__").mkdir()
    (tree / "evals/online/__pycache__/job.cpython-312.pyc").write_bytes(b"\x00")
    assert job.evaluator_version(tree) == before


def test_a_new_online_module_changes_the_version(tree):
    before = job.evaluator_version(tree)
    (tree / "evals/online/summary.py").write_text("x = 1\n", encoding="utf-8")
    assert job.evaluator_version(tree) != before


def test_the_path_is_part_of_the_hash_so_a_renamed_file_with_the_same_bytes_changes_it(tree):
    before = job.evaluator_version(tree)
    (tree / "evals/online/pull.py").rename(tree / "evals/online/pulled.py")
    assert job.evaluator_version(tree) != before


def test_a_covered_file_that_is_missing_raises_instead_of_hashing_less(tree):
    (tree / "evals/agent/judge.py").unlink()
    with pytest.raises(FileNotFoundError, match="judge.py"):
        job.evaluator_version(tree)


def test_the_version_marker_comment_round_trips_through_the_reader():
    posted = type("Posted", (), {"extra": None, "comment": job.comment_with_version("no assess", "0123456789ab")})()
    assert job.version_of(posted) == "0123456789ab"
    assert job.version_of(type("Posted", (), {"extra": None, "comment": "written by someone else"})()) is None
    assert job.version_of(type("Posted", (), {"extra": {"evaluator_version": "not-a-version"}, "comment": None})()) is None
