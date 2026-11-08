"""hit@4 (G19): the questions format, the match rule, misses, regressions and ranking. A fake embedder ranks real chunks for the last tests."""

from pathlib import Path

import pytest
from guidance_helpers import FakeEmbedder

from flysafelz.guidance.hit4 import (
    Expected,
    Question,
    QuestionsError,
    describe_misses,
    load_questions,
    rank,
    regressions,
    run_hit4,
    section_matches,
)

TOP_K = 4


def write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "questions.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def question(text: str, *expected: tuple[str, str]) -> Question:
    return Question(question=text, expected=[Expected(document=d, section=s) for d, s in expected])


@pytest.mark.parametrize(
    ("chunk", "expected", "matches"),
    [
        ("4.2", "4.2", True),
        ("4.2.1", "4.2", True),
        ("4.2.1.3", "4.2", True),
        ("4.2 (part 2)", "4.2", True),
        ("4.2.1 (part 1)", "4.2", True),
        ("4.20", "4.2", False),
        ("4.2", "4.2.1", False),  # the chunk is the parent of what was expected
        ("4", "4.2", False),
        ("42", "4", False),
        ("Appendix A", "appendix a", True),
        ("Appendix A (part 3)", "Appendix A", True),
        ("Appendix B", "Appendix A", False),
        ("4.2", "4.2.", True),
        ("5.1", "4.2", False),
    ],
)
def test_a_section_matches_itself_and_its_subsections_only(chunk, expected, matches):
    assert section_matches(chunk, expected) is matches


def test_questions_load_with_a_list_of_expected_entries(tmp_path):
    path = write(tmp_path, "- question: Can I fly in a danger area?\n  expected:\n    - {document: cap1535, section: '4.2'}\n    - {document: cap2562, section: '3'}\n")
    (loaded,) = load_questions(path)
    assert loaded.question == "Can I fly in a danger area?"
    assert [(e.document, e.section) for e in loaded.expected] == [("cap1535", "4.2"), ("cap2562", "3")]


def test_questions_under_a_questions_key_load_too(tmp_path):
    path = write(tmp_path, "questions:\n  - {question: q, expected: [{document: d, section: '1'}]}\n")
    assert len(load_questions(path)) == 1


@pytest.mark.parametrize("text", ["", "questions: []\n", "[]\n", "# only a comment\n"])
def test_no_labels_means_an_empty_list(tmp_path, text):
    assert load_questions(write(tmp_path, text)) == []


def test_a_missing_file_or_no_path_means_no_labels(tmp_path):
    assert load_questions(tmp_path / "absent.yaml") == [] and load_questions(None) == []


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("- {expected: [{document: d, section: '1'}]}\n", r"question #1: question: Field required"),
        ("- {question: q}\n", r"question #1: expected: Field required"),
        ("- {question: q, expected: []}\n", r"question #1: expected"),
        ("- {question: q, expected: [{document: d}]}\n", r"expected.0.section"),
        ("- {question: q, expected: [{section: '1'}]}\n", r"expected.0.document"),
        ("- {question: q, expected: [{document: d, section: 4.2}]}\n", r"expected.0.section"),  # a YAML number, not text
        ("- {question: q, expected: 'cap1535 4.2'}\n", r"question #1: expected"),
        ("- {question: ok, expected: [{document: d, section: '1'}]}\n- {question: q, expected: []}\n", r"question #2"),
        ("a: [\n", "not valid YAML"),
        ("key: value\n", "expected a list"),
        ("- just a string\n", "question #1"),
    ],
)
def test_a_malformed_questions_file_is_refused_naming_the_question(tmp_path, text, message):
    with pytest.raises(QuestionsError, match=message):
        load_questions(write(tmp_path, text))


def results(**by_question):
    return lambda asked: by_question[asked]


def test_a_question_is_a_hit_when_the_first_expected_entry_matches_a_top_chunk():
    questions = [question("q", ("cap1535", "4.2"), ("cap2562", "3"))]
    outcome = run_hit4(questions, results(q=[("cap1535", "4.2.1")]), TOP_K)
    assert outcome.hits == 1 and outcome.total == 1 and outcome.ran and not outcome.misses


def test_a_question_with_two_expected_entries_is_a_hit_when_only_the_second_matches():
    questions = [question("q", ("cap1535", "4.2"), ("cap2562", "3"))]
    outcome = run_hit4(questions, results(q=[("cap1535", "9"), ("cap2562", "3.1 (part 2)")]), TOP_K)
    assert outcome.hits == 1


def test_a_question_where_neither_expected_entry_matches_is_a_miss_that_reports_what_came_back():
    questions = [question("Where is RA(T)?", ("cap1535", "4.2"), ("cap2562", "3"))]
    outcome = run_hit4(questions, results(**{"Where is RA(T)?": [("cap1535", "4.20"), ("cap2562", "30"), ("bhpa", "4.2")]}), TOP_K)
    assert outcome.hits == 0 and [m.question.question for m in outcome.misses] == ["Where is RA(T)?"]
    (block,) = describe_misses(outcome)
    assert "Where is RA(T)?" in block and "cap1535 4.2, cap2562 3" in block and "cap1535 4.20, cap2562 30, bhpa 4.2" in block


def test_the_document_must_match_as_well_as_the_section():
    outcome = run_hit4([question("q", ("cap1535", "4.2"))], results(q=[("cap2562", "4.2")]), TOP_K)
    assert outcome.hits == 0


def test_only_the_top_k_results_count():
    returned = [("cap1535", "1"), ("cap1535", "2"), ("cap1535", "3"), ("cap1535", "4"), ("cap1535", "9")]
    assert run_hit4([question("q", ("cap1535", "9"))], results(q=returned), 4).hits == 0
    assert run_hit4([question("q", ("cap1535", "9"))], results(q=returned), 5).hits == 1


def test_a_miss_with_nothing_returned_says_so():
    outcome = run_hit4([question("q", ("d", "1"))], results(q=[]), TOP_K)
    assert "got: nothing" in describe_misses(outcome)[0]


def test_no_questions_means_the_check_did_not_run():
    outcome = run_hit4([], results(), TOP_K)
    assert not outcome.ran and outcome.total == 0 and outcome.hits == 0 and describe_misses(outcome) == []


def test_a_regression_is_a_question_that_went_from_hit_to_miss_and_only_that():
    questions = [question("a", ("d", "1")), question("b", ("d", "2")), question("c", ("d", "3")), question("d", ("d", "4"))]
    before = run_hit4(questions, results(a=[("d", "1")], b=[("d", "2")], c=[("x", "0")], d=[("x", "0")]), TOP_K)
    after = run_hit4(questions, results(a=[("x", "0")], b=[("d", "2")], c=[("d", "3")], d=[("x", "0")]), TOP_K)
    assert (before.hits, after.hits) == (2, 2)
    assert regressions(before, after) == ["a"]  # b stayed a hit, c improved, d stayed a miss


def test_rank_orders_by_cosine_similarity_and_keeps_file_order_for_ties():
    vectors = [[0.0, 1.0], [1.0, 0.0], [2.0, 0.0], [1.0, 1.0]]
    assert rank([1.0, 0.0], vectors, 4) == [1, 2, 3, 0]  # 1 and 2 tie at cosine 1, file order breaks it
    assert rank([1.0, 0.0], vectors, 2) == [1, 2]
    assert rank([0.0, 0.0], vectors, 2) == [0, 1]  # a zero query is similar to nothing: file order
    assert rank([1.0, 0.0], [[0.0, 0.0], [1.0, 0.0]], 2) == [1, 0]  # a zero vector scores 0, not an error
    assert rank([1.0], [], 4) == []


def test_ranking_with_the_fake_embedder_finds_the_chunk_that_shares_the_words():
    embedder = FakeEmbedder()
    chunks = [("cap1535", "1", "airspace classes and clearances"), ("cap1535", "2", "danger area activation notices"), ("cap2562", "3", "restricted area temporary RA(T) NOTAM")]
    vectors = embedder.embed([text for _, _, text in chunks])
    (query,) = embedder.query_embed(["when is a danger area activation notice issued"])
    outcome = run_hit4(
        [question("when is a danger area activation notice issued", ("cap1535", "2"))],
        lambda _: [chunks[i][:2] for i in rank(query, vectors, 1)],
        1,
    )
    assert outcome.hits == 1
