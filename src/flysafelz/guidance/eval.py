"""`make eval-guidance`: how often the right section is among the top results for Dan's labelled questions (G19).

    python -m flysafelz.guidance.eval [--data-dir D] [--questions Q] [--sources S]    what `make eval-guidance` runs
    hit4_of_index(index, questions)  -> Hit4Result   ask every question of the index and score the top `top_k` chunks (hit4.run_hit4)
    main(argv, embedder_factory)   -> int          the command line; returns the exit code

`hit4_of_index` is the one place a question is turned into a score. `make guidance`'s promote gate (ingest.py) calls it on the staging index and on the
current index, so the number the gate compares and the number printed here come from the same code (GP10). The hit rule and the match of a section
or its subsections are PR-7's `hit4.py`; nothing here copies them.

Output, in this order:
  hit@4: <hits>/<n> (<percent>)
  miss: <question> | expected: <doc> §<section>, ... | got: <doc> §<section>, ...    one line per miss, the top four that came back
  corpus key: <key>
With no questions (the file is missing or empty) it prints `hit@4 not run: no labels` and the corpus key. It exits 0 whatever the score: the decision to
promote belongs to `make guidance`. It exits 1 for a questions file that is malformed, an `expected` document that sources.yaml does not list, an
index that is missing or unreadable, or an embedding model that `make setup` has not stored, each with a message on stderr.
Defaults: data/guidance for the index and the register, evals/guidance/questions.yaml for the questions.
"""

import argparse
import logging
import sys
from collections.abc import Sequence
from pathlib import Path

from flysafelz.agent.settings import SETTINGS, repository_root
from flysafelz.guidance.embedder import EmbeddingModelMissing
from flysafelz.guidance.hit4 import Hit4Result, Question, QuestionResult, QuestionsError, load_questions, run_hit4
from flysafelz.guidance.index import EmbedderFactory, GuidanceIndex, IndexUnavailable, computed_key, default_data_dir, load_index
from flysafelz.guidance.sources import SourcesError, load_sources

QUESTIONS_PATH = Path("evals") / "guidance" / "questions.yaml"
SOURCES_NAME = "sources.yaml"
NO_KEY = "none"


def hit4_of_index(index: GuidanceIndex, questions: Sequence[Question]) -> Hit4Result:
    """hit@`top_k` of the index over the questions: each question is searched and the (document, section) of every chunk that came back is scored."""

    def search(question: str) -> list[tuple[str, str]]:
        return [(chunk.doc, chunk.section) for chunk in index.search(question, SETTINGS.top_k)]

    return run_hit4(questions, search, SETTINGS.top_k)


def result_line(result: Hit4Result) -> str:
    """`hit@4: 3/4 (75%)`."""
    return f"hit@{SETTINGS.top_k}: {result.hits}/{result.total} ({result.hits / result.total:.0%})"


def _places(pairs: Sequence[tuple[str, str]]) -> str:
    return ", ".join(f"{document} §{section}" for document, section in pairs) or "nothing"


def miss_line(miss: QuestionResult) -> str:
    """One line for a miss: the question, every place it expected, and the places that came back."""
    expected = _places([(entry.document, entry.section) for entry in miss.question.expected])
    return f"miss: {miss.question.question} | expected: {expected} | got: {_places(miss.returned)}"


def _fail(message: str) -> int:
    print(message, file=sys.stderr)
    return 1


def _unknown_documents(questions: Sequence[Question], known: set[str]) -> list[str]:
    """One line for each expected document the register does not list, naming the question number and the document."""
    found: list[str] = []
    for position, question in enumerate(questions, start=1):
        for entry in question.expected:
            if entry.document not in known:
                found.append(f"question #{position}: expected document '{entry.document}' is not in {SOURCES_NAME}")
    return found


def _key_text(data_dir: Path) -> str:
    try:
        return computed_key(data_dir)
    except IndexUnavailable as error:
        return f"{NO_KEY} ({error})"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m flysafelz.guidance.eval", description="Score guidance search (hit@k) against the labelled questions.")
    parser.add_argument("--data-dir", type=Path, help="folder with embeddings.sqlite, manifest.yaml and sources.yaml (default: data/guidance)")
    parser.add_argument("--questions", type=Path, help="the labelled questions (default: evals/guidance/questions.yaml)")
    parser.add_argument("--sources", type=Path, help="the register of documents (default: sources.yaml in the data folder)")
    return parser


def main(argv: Sequence[str] | None = None, embedder_factory: EmbedderFactory | None = None) -> int:
    """Run the eval. `embedder_factory` is for tests; the command line uses the stored embedding model."""
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")  # the index's warnings (a key mismatch, a bad file) reach stderr
    args = _parser().parse_args(argv)
    data_dir = default_data_dir() if args.data_dir is None else args.data_dir
    questions_path = repository_root() / QUESTIONS_PATH if args.questions is None else args.questions
    sources_path = data_dir / SOURCES_NAME if args.sources is None else args.sources
    try:
        questions = load_questions(questions_path)
    except QuestionsError as error:
        return _fail(str(error))
    except OSError as error:
        return _fail(f"{questions_path.name}: cannot read the questions: {error.strerror or error}")
    if not questions:
        print(f"hit@{SETTINGS.top_k} not run: no labels")
        print(f"corpus key: {_key_text(data_dir)}")
        return 0
    try:
        known = {source.id for source in load_sources(sources_path)}
    except SourcesError as error:
        return _fail(str(error))
    unknown = _unknown_documents(questions, known)
    if unknown:
        return _fail("\n".join(f"{questions_path.name}: {line}" for line in unknown))
    index = load_index(data_dir, embedder_factory=embedder_factory)
    if not index.available:
        return _fail(f"hit@{SETTINGS.top_k} not run: the guidance index is not available: {index.reason}")
    try:
        result = hit4_of_index(index, questions)
    except EmbeddingModelMissing as error:
        return _fail(f"hit@{SETTINGS.top_k} not run: {error}")
    print(result_line(result))
    for miss in result.misses:
        print(miss_line(miss))
    print(f"corpus key: {index.key}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
