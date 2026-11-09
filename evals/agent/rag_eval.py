"""RAG hit@4 over the guidance questions (5a PR-10.F1, G15, GP8): `make rag-eval`.

    python -m evals.agent.rag_eval [--questions FILE] [--data-dir DIR]

    run(questions, search, catalogue) -> Hit4Result   ask each question and score its top 4 chunks
    is_hit(chunk, expected, catalogue) -> bool        the hit rule for one chunk and one expected place
    render(result, stamp) -> str                      the Markdown of the results file
    main(argv, ...) -> int                            0 on a score (whatever it is), 1 when it cannot run

hit@4 is the share of the 10 questions of evals/agent/data/guidance_questions.yaml (Dan's, with `expected` places as {document, section}) where ANY of the top 4 chunks
`search_guidance` returns is in ANY expected place. The search is the guidance index's own `search` (flysafelz.guidance.index.GuidanceIndex), the one call the `search_guidance`
tool makes: this command makes no agent call, no judge call and no LangSmith call, and it needs only the embedding model `make setup` stored.

The hit rule. A chunk is in an expected place when it is in the same document (compared without regard to case, spaces or punctuation: `CAP 1535` is the index's `cap1535`) and either its heading (`title`) is the expected heading, compared
case-insensitively (or the expected section is a number as printed and the chunk's is that number or below it), or it is a subsection of it: the chunk's section number is the number of a chunk of that document headed with the expected heading, or is below it (`4.2.1` under
`4.2`; the rule is flysafelz.guidance.hit4.section_matches, so `make eval-guidance` and this one cannot disagree). The `catalogue` is every chunk of the index, which is how a heading
is turned into its number. An expected list with the same place twice (Q6's two "Restricted airspace (temporary)" entries) is one hit, not two. When the search returns fewer than 4
chunks the question is scored on what came back; none is a miss.

Output: `hit@4: k/10 (p%)` and one `miss:` line each (the question, every expected place, the top 4 headings that came back) on stdout, and the whole result in
`evals/results/rag-<UTC stamp>.md`: each question's hit or miss and its top-4 headings. Nothing else is written; the exit code is 0 for any score and 1 when it cannot be scored
(a malformed questions file, no index, no embedding model).
"""

import argparse
import logging
import sys
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, NamedTuple

from evals.agent.import_dataset import DATA_DIR
from evals.agent.report import RESULTS_DIR, ReportError, write_report
from flysafelz.guidance.embedder import EmbeddingModelMissing
from flysafelz.guidance.hit4 import Expected, Hit4Result, Question, QuestionResult, QuestionsError, load_questions, section_matches
from flysafelz.guidance.index import GuidanceIndex, default_data_dir, load_index

PROGRAM = "evals.agent.rag_eval"
QUESTIONS_PATH = DATA_DIR / "guidance_questions.yaml"
TOP = 4
FAILED_EXIT = 1
SEPARATOR = " › "


class Chunk(NamedTuple):
    """What the rule reads of a chunk the search returned: its document, its section number as printed and its heading."""

    document: str
    section: str
    title: str


Search = Callable[[str], Sequence[Chunk]]


def _same(left: str, right: str) -> bool:
    return " ".join(left.split()).casefold() == " ".join(right.split()).casefold()


def _same_document(left: str, right: str) -> bool:
    """The questions say `CAP 1535`, the index's document id is `cap1535`: equal when they differ only in case, spaces and punctuation."""
    return _squeeze(left) == _squeeze(right)


def _squeeze(name: str) -> str:
    return "".join(character for character in name.casefold() if character.isalnum())


def is_hit(chunk: Chunk, expected: Expected, catalogue: Sequence[Chunk] = ()) -> bool:
    """Whether the chunk is in the expected place: same document, and the expected heading itself (any case) or a subsection of it."""
    if not _same_document(chunk.document, expected.document):
        return False
    if _same(chunk.title, expected.section) or section_matches(chunk.section, expected.section):
        return True
    numbers = {c.section for c in catalogue if _same_document(c.document, expected.document) and (_same(c.title, expected.section) or _same(c.section, expected.section))}
    return any(section_matches(chunk.section, number) for number in numbers)


def run(questions: Sequence[Question], search: Search, catalogue: Sequence[Chunk] = ()) -> Hit4Result:
    """Ask each question, keep the first TOP chunks, and mark it a hit when any of them is in any expected place."""
    results = []
    for question in questions:
        chunks = list(search(question.question))[:TOP]
        hit = any(is_hit(chunk, expected, catalogue) for chunk in chunks for expected in question.expected)
        results.append(QuestionResult(question, hit, tuple((chunk.document, chunk.title or chunk.section) for chunk in chunks)))
    return Hit4Result(tuple(results))


def score_line(result: Hit4Result) -> str:
    return f"hit@{TOP}: {result.hits}/{result.total}" + (f" ({result.hits / result.total:.0%})" if result.total else "")


def _places(pairs: Sequence[tuple[str, str]]) -> str:
    return "; ".join(f"{document}{SEPARATOR}{heading}" for document, heading in pairs) or "nothing"


def miss_line(miss: QuestionResult) -> str:
    expected = _places([(e.document, e.section) for e in dict.fromkeys(miss.question.expected)])
    return f"miss: {miss.question.question} | expected: {expected} | got: {_places(miss.returned)}"


def render(result: Hit4Result, stamp: datetime, questions_name: str = QUESTIONS_PATH.name) -> str:
    """The results file: the score, then each question with its hit or miss and its top-4 headings."""
    lines = [f"# RAG hit@{TOP} ({stamp:%Y-%m-%d %H:%M} UTC)", "", f"{score_line(result)} over {questions_name}.", ""]
    for number, found in enumerate(result.results, start=1):
        lines += [
            f"## Q{number}: {'hit' if found.hit else 'miss'}",
            "",
            found.question.question,
            "",
            f"- top {TOP} headings: {_places(found.returned)}",
        ]
        if not found.hit:
            lines.append(f"- expected: {_places([(e.document, e.section) for e in dict.fromkeys(found.question.expected)])}")
        lines.append("")
    return "\n".join(lines).rstrip("\n") + "\n"


def index_search(index: GuidanceIndex) -> tuple[Search, list[Chunk]]:
    """The index's own search (what `search_guidance` calls) and its catalogue of chunks."""

    def search(question: str) -> list[Chunk]:
        return [Chunk(c.doc, c.section, c.title) for c in index.search(question, TOP)]

    return search, [Chunk(c.doc, c.section, c.title) for c in index.chunks]


def _fail(message: str, err: Any) -> int:
    print(f"{PROGRAM}: {message}", file=err)
    return FAILED_EXIT


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog=f"python -m {PROGRAM}", description="Score guidance search (hit@4) over the 10 guidance questions. Makes no agent or judge call.")
    parser.add_argument("--questions", type=Path, default=QUESTIONS_PATH, help="the labelled questions (default: evals/agent/data/guidance_questions.yaml)")
    parser.add_argument("--data-dir", type=Path, default=None, help="folder with embeddings.sqlite and manifest.yaml (default: data/guidance)")
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    search_factory: Callable[[Path], tuple[Search, list[Chunk]]] | None = None,
    stdout: Any = None,
    stderr: Any = None,
    moment: datetime | None = None,
    results_dir: Path = RESULTS_DIR,
) -> int:
    """The command. `search_factory(data_dir)` gives (search, catalogue); the default loads the guidance index (tests pass a fake retriever)."""
    out = sys.stdout if stdout is None else stdout
    err = sys.stderr if stderr is None else stderr
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
    args = _parser().parse_args(argv)
    try:
        questions = load_questions(args.questions)
    except QuestionsError as error:
        return _fail(str(error), err)
    except OSError as error:
        return _fail(f"{args.questions.name}: cannot read the questions: {error.strerror or error}", err)
    if not questions:
        return _fail(f"hit@{TOP} not run: no questions in {args.questions.name}", err)
    data_dir = default_data_dir() if args.data_dir is None else args.data_dir
    try:
        if search_factory is None:
            index = load_index(data_dir)
            if not index.available:
                return _fail(f"hit@{TOP} not run: the guidance index is not available: {index.reason}", err)
            search, catalogue = index_search(index)
        else:
            search, catalogue = search_factory(data_dir)
        result = run(questions, search, catalogue)
    except EmbeddingModelMissing as error:
        return _fail(f"hit@{TOP} not run: {error}", err)
    stamp = (moment or datetime.now(UTC)).astimezone(UTC)
    try:
        path = write_report(render(result, stamp), f"rag-{stamp:%Y%m%d-%H%M%S}", results_dir)
    except (ReportError, OSError) as error:
        return _fail(f"the results file could not be written: {error}", err)
    print(score_line(result), file=out)
    for miss in result.misses:
        print(miss_line(miss), file=out)
    print(f"results: {path}", file=out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
