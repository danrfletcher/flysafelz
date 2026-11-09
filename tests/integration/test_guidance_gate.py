"""GP10, the hit@4 part: `make guidance`'s promote gate scores the staging index and the current index through this PR's scorer (eval.hit4_of_index).

Two tiny fixture snapshots and a fake embedder. `ingest._gate` is the function `run_ingest` calls between staging and promote, so this runs the real call path.
"""

import pytest
from factories import EmbedderSpy, guidance_chunk, write_questions, write_snapshot

from flysafelz.agent.settings import SETTINGS
from flysafelz.guidance import eval as guidance_eval
from flysafelz.guidance import ingest
from flysafelz.guidance.hit4 import load_questions

MODEL = SETTINGS.embedding_model
# Both questions are answered by section 1 of their document; the 'worse' snapshot lost the words of the second one.
GOOD = [guidance_chunk("a", "1", "alpha beta gamma"), guidance_chunk("b", "1", "kappa lambda mu")]
EXTRA = [
    guidance_chunk("0", str(n), f"filler{n} padding") for n in range(1, 6)
]  # five chunks that share no word with a question; they sort before the others, so they fill the top four when nothing matches
WORSE = [guidance_chunk("a", "1", "alpha beta gamma"), guidance_chunk("b", "1", "unrelated words here")]
QUESTIONS = (("alpha beta", [("a", "1")]), ("kappa lambda", [("b", "1")]))


@pytest.fixture
def questions(tmp_path):
    return load_questions(write_questions(tmp_path / "questions.yaml", *QUESTIONS))


def gate(tmp_path, questions, staging_chunks, current_chunks, capsys):
    current, staging = tmp_path / "data", tmp_path / "data" / "staging"
    if current_chunks is not None:
        write_snapshot(current, current_chunks, register=False)
    write_snapshot(staging, staging_chunks, register=False)
    capsys.readouterr()
    allowed = ingest._gate(current, staging, questions, EmbedderSpy().embedder, MODEL)  # type: ignore[arg-type]
    captured = capsys.readouterr()
    return allowed, captured.out, captured.err


def test_a_staging_snapshot_that_scores_worse_is_blocked_and_the_questions_that_got_worse_are_named(tmp_path, questions, capsys, monkeypatch):
    asked = []
    real = guidance_eval.hit4_of_index
    monkeypatch.setattr(ingest, "hit4_of_index", lambda index, qs: asked.append(index.path.parent.name) or real(index, qs))  # type: ignore[union-attr]
    allowed, out, err = gate(tmp_path, questions, [*WORSE, *EXTRA], [*GOOD, *EXTRA], capsys)
    assert allowed is False
    assert asked == ["staging", "data"]  # the shared scorer ran on the staging index and on the current index
    assert "hit@4: staging 1/2, current 2/2" in out
    assert "miss: kappa lambda" in out and "expected: b 1" in out  # the miss, with what was expected and what came back
    assert "went from hit to miss: kappa lambda" in out and "alpha beta" not in out.split("went from hit to miss")[1]
    assert "not promoted" in err and "lower" in err


def test_an_equal_staging_snapshot_is_promoted(tmp_path, questions, capsys):
    allowed, out, err = gate(tmp_path, questions, [*GOOD, *EXTRA], [*GOOD, *EXTRA], capsys)
    assert allowed is True and err == ""
    assert "hit@4: staging 2/2, current 2/2" in out and "went from hit to miss" not in out


def test_a_better_staging_snapshot_is_promoted(tmp_path, questions, capsys):
    allowed, out, _err = gate(tmp_path, questions, [*GOOD, *EXTRA], [*WORSE, *EXTRA], capsys)
    assert allowed is True and "hit@4: staging 2/2, current 1/2" in out


def test_equal_scores_with_a_different_question_missed_are_promoted_but_the_regression_is_named(tmp_path, capsys):
    both = load_questions(write_questions(tmp_path / "questions.yaml", *QUESTIONS, ("epsilon zeta", [("d", "1")])))
    current = [*GOOD, guidance_chunk("d", "1", "unrelated about it"), *EXTRA]  # misses the third question
    staging = [guidance_chunk("a", "1", "alpha beta gamma"), guidance_chunk("b", "1", "unrelated"), guidance_chunk("d", "1", "epsilon zeta eta"), *EXTRA]  # hits the third, misses the second
    allowed, out, _err = gate(tmp_path, both, staging, current, capsys)
    assert allowed is True and "hit@4: staging 2/3, current 2/3" in out and "went from hit to miss: kappa lambda" in out


def test_with_no_current_snapshot_the_first_run_is_promoted_and_counts_the_current_as_zero(tmp_path, questions, capsys):
    allowed, out, _err = gate(tmp_path, questions, [*WORSE, *EXTRA], None, capsys)
    assert allowed is True and "current none yet" in out


def test_an_unreadable_current_snapshot_counts_as_zero_and_does_not_block(tmp_path, questions, capsys):
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "embeddings.sqlite").write_bytes(b"not a database" * 30)
    allowed, out, _err = gate(tmp_path, questions, [*GOOD, *EXTRA], None, capsys)
    assert allowed is True and "unreadable" in out


def test_with_no_labels_the_gate_does_not_run_and_promotes(tmp_path, capsys):
    allowed, out, _err = gate(tmp_path, [], GOOD, GOOD, capsys)
    assert allowed is True and out == "hit@4 not run: no labels\n"
