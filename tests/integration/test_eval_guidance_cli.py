"""`make eval-guidance` (G19): `python -m flysafelz.guidance.eval` on fixture snapshots and questions. No model, no network, no key.

The score runs in process with the fake embedder (the real model is not needed); the cases that stop before the index is searched (no labels, a malformed
file, an unknown document) run as the real command in a subprocess, so the exit code and stderr are those of `make eval-guidance`.
"""

import subprocess
import sys
from pathlib import Path

import pytest
from factories import EmbedderSpy, guidance_chunk, write_questions, write_snapshot

from flysafelz.guidance import eval as guidance_eval
from flysafelz.guidance.embedder import EmbeddingModelMissing
from flysafelz.guidance.index import computed_key

ROOT = Path(__file__).resolve().parents[2]
CHUNKS = [
    guidance_chunk("a", "1", "alpha beta gamma"),
    guidance_chunk("a", "2", "delta epsilon zeta"),
    guidance_chunk("b", "1", "eta theta iota"),
    guidance_chunk("b", "2", "kappa lambda mu"),
    guidance_chunk("c", "1", "nu xi omicron"),
    guidance_chunk("z", "1", "omega psi chi"),  # sorts last, so it is never among the top four for a question it does not share words with
]


def command(*arguments: str | Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run([sys.executable, "-m", "flysafelz.guidance.eval", *map(str, arguments)], cwd=ROOT, capture_output=True, text=True, check=False)


@pytest.fixture
def data(tmp_path):
    folder = tmp_path / "data"
    write_snapshot(folder, CHUNKS)
    return folder


def key_of(folder: Path) -> str:
    return computed_key(folder)


def run_main(capsys, *arguments: str | Path, factory=None):
    capsys.readouterr()
    code = guidance_eval.main([str(a) for a in arguments], embedder_factory=factory or EmbedderSpy())
    captured = capsys.readouterr()
    return code, captured.out.splitlines(), captured.err


def test_it_prints_the_score_one_line_per_miss_and_the_corpus_key_and_exits_zero(tmp_path, data, capsys):
    questions = write_questions(tmp_path / "q.yaml", ("alpha beta", [("a", "1")]), ("kappa lambda", [("z", "1"), ("b", "9")]))
    code, out, err = run_main(capsys, "--data-dir", data, "--questions", questions)
    assert code == 0 and err == ""
    assert out == [
        "hit@4: 1/2 (50%)",
        "miss: kappa lambda | expected: z §1, b §9 | got: b §2, a §1, a §2, b §1",
        f"corpus key: {key_of(data)}",
    ]


def test_it_exits_zero_even_when_every_question_misses(tmp_path, data, capsys):
    questions = write_questions(tmp_path / "q.yaml", ("alpha beta", [("z", "1")]), ("kappa lambda", [("z", "1")]))
    code, out, _err = run_main(capsys, "--data-dir", data, "--questions", questions)
    assert code == 0 and out[0] == "hit@4: 0/2 (0%)" and len(out) == 4


def test_a_subsection_counts_and_the_prefix_trap_does_not(tmp_path, capsys):
    folder = tmp_path / "data"
    fillers = [guidance_chunk("0", str(n), f"filler{n} unrelated") for n in range(1, 5)]  # they sort first among chunks equally unlike the question, so they fill the top four
    write_snapshot(folder, [*fillers, guidance_chunk("a", "3.2.1", "alpha beta"), guidance_chunk("a", "3.20", "gamma delta"), guidance_chunk("a", "13.2", "gamma delta")])
    questions = write_questions(tmp_path / "q.yaml", ("alpha beta", [("a", "3.2")]), ("gamma delta", [("a", "3.2")]))
    code, out, _err = run_main(capsys, "--data-dir", folder, "--questions", questions)
    assert code == 0 and out[0] == "hit@4: 1/2 (50%)" and out[1].startswith("miss: gamma delta | expected: a §3.2 | got: a §13.2, a §3.20, 0 §") and "a §3.2.1" not in out[1]


def test_the_gate_and_the_eval_give_the_same_score_for_the_same_snapshot(tmp_path, data, capsys):
    from flysafelz.guidance import ingest
    from flysafelz.guidance.hit4 import load_questions

    path = write_questions(tmp_path / "q.yaml", ("alpha beta", [("a", "1")]), ("kappa lambda", [("a", "2")]))
    assert ingest.hit4_of_index is guidance_eval.hit4_of_index  # one scorer, not two
    _code, out, _err = run_main(capsys, "--data-dir", data, "--questions", path)
    scored = guidance_eval.hit4_of_index(ingest._index_in(data, EmbedderSpy().embedder, ingest.SETTINGS.embedding_model), load_questions(path))  # type: ignore[arg-type]
    assert out[0] == guidance_eval.result_line(scored)


def test_a_missing_questions_file_prints_no_labels_and_the_key_and_exits_zero(tmp_path, data):
    done = command("--data-dir", data, "--questions", tmp_path / "nope.yaml")
    assert done.returncode == 0 and done.stdout.splitlines() == ["hit@4 not run: no labels", f"corpus key: {key_of(data)}"] and done.stderr == ""


@pytest.mark.parametrize("content", ["", "[]\n", "questions: []\n", "# nothing yet\n"])
def test_an_empty_questions_file_prints_no_labels_and_the_key_and_exits_zero(tmp_path, data, content):
    questions = tmp_path / "q.yaml"
    questions.write_text(content, encoding="utf-8")
    done = command("--data-dir", data, "--questions", questions)
    assert done.returncode == 0 and done.stdout.splitlines() == ["hit@4 not run: no labels", f"corpus key: {key_of(data)}"]


def test_no_labels_with_no_snapshot_still_exits_zero_and_says_there_is_no_key(tmp_path):
    done = command("--data-dir", tmp_path / "empty", "--questions", tmp_path / "nope.yaml")
    assert done.returncode == 0 and done.stdout.splitlines()[0] == "hit@4 not run: no labels" and done.stdout.splitlines()[1].startswith("corpus key: none")


@pytest.mark.parametrize(
    ("content", "named"),
    [
        ("- question: [unclosed\n", "not valid YAML"),
        ("- question: fine\n  expected:\n    - {document: a, section: '1'}\n- expected:\n    - {document: a, section: '1'}\n", "question #2"),
        ("- question: one\n  expected: []\n", "question #1"),
        ("- question: one\n", "question #1"),
        ("- question: fine\n  expected:\n    - {document: a, section: '1'}\n- question: bad\n  expected:\n    - {document: a}\n", "question #2"),
        ("- question: fine\n  expected:\n    - {section: '1'}\n", "question #1"),
        ("- question: fine\n  expected:\n    - {document: a, section: ''}\n", "question #1"),
        ("just a string\n", "expected a list of questions"),
    ],
)
def test_a_malformed_questions_file_exits_non_zero_naming_the_entry(tmp_path, data, content, named):
    questions = tmp_path / "q.yaml"
    questions.write_text(content, encoding="utf-8")
    done = command("--data-dir", data, "--questions", questions)
    assert done.returncode != 0 and named in done.stderr and "q.yaml" in done.stderr and done.stdout == ""


def test_an_expected_document_that_sources_yaml_does_not_list_exits_non_zero_naming_it(tmp_path, data):
    questions = write_questions(tmp_path / "q.yaml", ("alpha beta", [("a", "1")]), ("kappa", [("b", "2"), ("ghost-doc", "1")]))
    done = command("--data-dir", data, "--questions", questions)
    assert done.returncode != 0 and "ghost-doc" in done.stderr and "question #2" in done.stderr and done.stdout == ""


def test_a_missing_register_exits_non_zero(tmp_path, data):
    (data / "sources.yaml").unlink()
    questions = write_questions(tmp_path / "q.yaml", ("alpha beta", [("a", "1")]))
    done = command("--data-dir", data, "--questions", questions)
    assert done.returncode != 0 and "sources.yaml" in done.stderr


def test_a_missing_index_exits_non_zero_and_says_so(tmp_path, data, capsys):
    (data / "embeddings.sqlite").unlink()
    questions = write_questions(tmp_path / "q.yaml", ("alpha beta", [("a", "1")]))
    code, out, err = run_main(capsys, "--data-dir", data, "--questions", questions)
    assert code == 1 and out == [] and "index is not available" in err and "embeddings.sqlite" in err


def test_a_model_that_make_setup_has_not_stored_exits_non_zero_naming_make_setup(tmp_path, data, capsys):
    questions = write_questions(tmp_path / "q.yaml", ("alpha beta", [("a", "1")]))
    spy = EmbedderSpy(error=EmbeddingModelMissing("embedding model missing: run make setup"))
    code, out, err = run_main(capsys, "--data-dir", data, "--questions", questions, factory=spy)
    assert code == 1 and out == [] and "make setup" in err


def test_the_make_target_runs_the_module():
    done = subprocess.run(["make", "-n", "eval-guidance"], cwd=ROOT, capture_output=True, text=True, check=False)
    assert done.returncode == 0 and "python -m flysafelz.guidance.eval" in done.stdout
    assert "eval-guidance" in (ROOT / "Makefile").read_text(encoding="utf-8").splitlines()[0]  # .PHONY


# The same cases in process, so that coverage sees the command's own code (a subprocess is not measured).


def test_in_process_no_labels_malformed_unknown_document_and_unreadable_questions(tmp_path, data, capsys):
    code, out, err = run_main(capsys, "--data-dir", data, "--questions", tmp_path / "nope.yaml")
    assert (code, out, err) == (0, ["hit@4 not run: no labels", f"corpus key: {key_of(data)}"], "")
    code, out, err = run_main(capsys, "--data-dir", tmp_path / "empty", "--questions", tmp_path / "nope.yaml")
    assert code == 0 and out[1].startswith("corpus key: none (")
    bad = tmp_path / "bad.yaml"
    bad.write_text("- question: one\n", encoding="utf-8")
    code, out, err = run_main(capsys, "--data-dir", data, "--questions", bad)
    assert code == 1 and out == [] and "question #1" in err
    unknown = write_questions(tmp_path / "unknown.yaml", ("alpha beta", [("ghost-doc", "1")]))
    code, out, err = run_main(capsys, "--data-dir", data, "--questions", unknown)
    assert code == 1 and "ghost-doc" in err and "question #1" in err
    unreadable = tmp_path / "locked.yaml"
    unreadable.mkdir()  # a folder where the file should be: is_file() is False, so it counts as no labels, never a traceback
    code, out, _err = run_main(capsys, "--data-dir", data, "--questions", unreadable)
    assert code == 0 and out[0] == "hit@4 not run: no labels"
    (data / "sources.yaml").write_text("documents: 7\n", encoding="utf-8")
    code, out, err = run_main(capsys, "--data-dir", data, "--questions", write_questions(tmp_path / "q.yaml", ("alpha beta", [("a", "1")])))
    assert code == 1 and "sources.yaml" in err


def test_the_questions_file_is_read_from_evals_guidance_by_default(tmp_path, data, capsys, monkeypatch):
    (tmp_path / "evals" / "guidance").mkdir(parents=True)
    write_questions(tmp_path / "evals" / "guidance" / "questions.yaml", ("alpha beta", [("a", "1")]))
    monkeypatch.setattr(guidance_eval, "repository_root", lambda: tmp_path)
    code, out, _err = run_main(capsys, "--data-dir", data)
    assert code == 0 and out[0] == "hit@4: 1/1 (100%)"
