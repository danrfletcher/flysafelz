"""hit@4: how often the right section is among the top results for a labelled question (G19). Pure: no model, no file but the questions.

    load_questions(path)                       -> list[Question]    `question:` and `expected:` (a list of {document, section}); a missing or empty file gives []
    run_hit4(questions, search, top_k)         -> Hit4Result        `search(question)` returns the (document, section) of each result, best first
    rank(query, vectors, top_k)                -> list[int]         positions of the `top_k` vectors most similar to `query` (cosine; ties keep file order)
    section_matches(chunk_section, expected)   -> bool              same section, or a subsection of it: `4.2` matches `4.2.1` and `4.2 (part 2)`, not `4.20`
    regressions(before, after)                 -> list[str]         the questions that were a hit before and are a miss after
    describe_misses(result)                    -> list[str]         one block per miss: the question, what was expected and what came back

A question is a hit when any of the top results matches any of its `expected` entries (same document, and that section or a subsection of it).
`make eval-guidance` (PR-8) and `make guidance`'s gate use these same functions, so the two cannot disagree. `top_k` is passed in; settings.yaml owns it.
"""

import math
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError

PART_SUFFIX = re.compile(r"\s*\(part\s+\d+\)\s*$", re.IGNORECASE)

Search = Callable[[str], Sequence[tuple[str, str]]]


class QuestionsError(ValueError):
    """The questions file is not valid. The message names the file and the question."""


class Expected(BaseModel):
    """A place the answer lives: a document id and a section number as printed."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    document: str = Field(strict=True, min_length=1)
    section: str = Field(strict=True, min_length=1)


class Question(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    question: str = Field(strict=True, min_length=1)
    expected: list[Expected] = Field(min_length=1)


@dataclass(frozen=True)
class QuestionResult:
    question: Question
    hit: bool
    returned: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class Hit4Result:
    """One result per question. With no questions the check did not run."""

    results: tuple[QuestionResult, ...]

    @property
    def ran(self) -> bool:
        return bool(self.results)

    @property
    def hits(self) -> int:
        return sum(1 for result in self.results if result.hit)

    @property
    def total(self) -> int:
        return len(self.results)

    @property
    def misses(self) -> list[QuestionResult]:
        return [result for result in self.results if not result.hit]


def load_questions(path: Path | None) -> list[Question]:
    """The labelled questions. A file that is missing, empty or holds no question gives []: hit@4 is then not run."""
    if path is None or not path.is_file():
        return []
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as error:
        raise QuestionsError(f"{path.name}: not valid YAML: {error}") from error
    if isinstance(data, dict):
        if set(data) != {"questions"}:  # a misspelt key must not quietly switch the gate off
            raise QuestionsError(f"{path.name}: expected a list of questions, or a mapping with only a `questions` key")
        data = data["questions"]
    if not data:
        return []
    if not isinstance(data, list):
        raise QuestionsError(f"{path.name}: expected a list of questions")
    questions = []
    for position, entry in enumerate(data, start=1):
        try:
            questions.append(Question.model_validate(entry))
        except ValidationError as error:
            first = error.errors()[0]
            where = ".".join(str(part) for part in first["loc"])
            raise QuestionsError(f"{path.name}: question #{position}: {where}: {first['msg']}") from error
    return questions


def _canonical(section: str) -> str:
    return " ".join(PART_SUFFIX.sub("", section).split()).casefold().rstrip(".")


def section_matches(chunk_section: str, expected_section: str) -> bool:
    """The same section, or a subsection of it, by the numbers as printed. A split part (`4.2 (part 2)`) counts as its section."""
    chunk, expected = _canonical(chunk_section), _canonical(expected_section)
    return chunk == expected or chunk.startswith(expected + ".")


def run_hit4(questions: Sequence[Question], search: Search, top_k: int) -> Hit4Result:
    """Ask each question and check its top `top_k` results against its expected entries."""
    results = []
    for question in questions:
        returned = tuple(search(question.question)[:top_k])
        hit = any(document == expected.document and section_matches(section, expected.section) for document, section in returned for expected in question.expected)
        results.append(QuestionResult(question, hit, returned))
    return Hit4Result(tuple(results))


def regressions(before: Hit4Result, after: Hit4Result) -> list[str]:
    """The questions that were a hit in `before` and are a miss in `after`, in the order asked."""
    was_hit = {result.question.question for result in before.results if result.hit}
    return [result.question.question for result in after.results if not result.hit and result.question.question in was_hit]


def describe_misses(result: Hit4Result) -> list[str]:
    """For each miss: the question, what was expected, and what came back."""
    lines = []
    for miss in result.misses:
        expected = ", ".join(f"{e.document} {e.section}" for e in miss.question.expected)
        returned = ", ".join(f"{document} {section}" for document, section in miss.returned) or "nothing"
        lines.append(f"miss: {miss.question.question}\n  expected: {expected}\n  got: {returned}")
    return lines


def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
    norms = math.sqrt(sum(x * x for x in left)) * math.sqrt(sum(y * y for y in right))
    return sum(x * y for x, y in zip(left, right, strict=True)) / norms if norms else 0.0


def rank(query: Sequence[float], vectors: Sequence[Sequence[float]], top_k: int) -> list[int]:
    """Positions of the `top_k` vectors most like `query`, best first. Equal similarity keeps the earlier position first."""
    similarity = [_cosine(query, vector) for vector in vectors]
    return sorted(range(len(vectors)), key=lambda position: (-similarity[position], position))[:top_k]
