"""RAG hit@4 over the guidance questions (5a PR-10.F1, G15, GP8): the hit rule, the run, the printed lines and the results file.

`search_guidance` is a fake retriever keyed by question: no embedding model, no index, no network, no agent or judge call.
"""

import io
from datetime import UTC, datetime

import pytest

from evals.agent import rag_eval
from evals.agent.rag_eval import QUESTIONS_PATH, Chunk, is_hit, main, miss_line, render, run, score_line
from flysafelz.guidance.hit4 import Expected, Question, load_questions

QUESTIONS = load_questions(QUESTIONS_PATH)
STAMP = datetime(2026, 11, 9, 12, 30, 5, tzinfo=UTC)
CATALOGUE = [
    Chunk("cap1535", "4.2", "Control zone (CTR)"),
    Chunk("cap1535", "4.2.1", "Entering a CTR"),
    Chunk("cap1535", "4.20", "Something else"),
    Chunk("cap2562", "2", "500 ft rule"),
]


class FakeSearch:
    """`search_guidance` as a fake: the chunks to return for each question (any other question returns nothing). It records what it was asked."""

    def __init__(self, by_question):
        self.by_question, self.asked = by_question, []

    def __call__(self, question):
        self.asked.append(question)
        return self.by_question.get(question, [])


def expected_chunk(question: Question) -> Chunk:
    place = question.expected[0]
    return Chunk(place.document, "9.9", place.section)  # the heading is the expected one, whatever the number


def wrong_chunks() -> list[Chunk]:
    return [Chunk("cap1535", "99", "Nothing relevant"), Chunk("bhpa-safety", "1", "Insurance")]


# --- the committed questions ---------------------------------------------------------------------------------------------------------------


def test_the_committed_questions_are_ten_and_q6_has_the_duplicate_the_spec_names():
    assert len(QUESTIONS) == 10
    q6 = next(q for q in QUESTIONS if "temporary notice" in q.question)
    assert sum(1 for e in q6.expected if e.section == "Restricted airspace (temporary)") >= 1


# --- the hit rule --------------------------------------------------------------------------------------------------------------------------------


def test_headings_match_case_insensitively_and_ignoring_extra_spaces():
    expected = Expected(document="CAP 1535", section="Control zone (CTR)")
    assert is_hit(Chunk("cap1535", "4.2", "CONTROL ZONE (ctr)"), expected)
    assert is_hit(Chunk("cap1535", "4.2", "control   zone  (CTR)"), expected)
    assert not is_hit(Chunk("cap1535", "4.2", "Control zones"), expected)


def test_the_document_matches_by_its_id_whatever_the_case_or_spacing():
    expected = Expected(document="CAP 1535", section="Control zone (CTR)")
    assert is_hit(Chunk("CAP 1535", "4.2", "Control zone (CTR)"), expected)
    assert is_hit(Chunk("cap-1535", "4.2", "Control zone (CTR)"), expected)
    assert not is_hit(Chunk("cap2562", "4.2", "Control zone (CTR)"), expected)  # the right heading in the wrong document is a miss


def test_a_subsection_of_the_expected_section_counts():
    expected = Expected(document="CAP 1535", section="Control zone (CTR)")
    assert is_hit(Chunk("cap1535", "4.2.1", "Entering a CTR"), expected, CATALOGUE)
    assert is_hit(Chunk("cap1535", "4.2 (part 2)", "Control zone (CTR)"), expected, CATALOGUE)


def test_a_sibling_that_only_shares_a_prefix_is_not_a_subsection():
    expected = Expected(document="CAP 1535", section="Control zone (CTR)")
    assert not is_hit(Chunk("cap1535", "4.20", "Something else"), expected, CATALOGUE)
    assert not is_hit(Chunk("cap1535", "4.3", "Elsewhere"), expected, CATALOGUE)


def test_a_subsection_needs_the_catalogue_to_turn_the_heading_into_its_number():
    expected = Expected(document="CAP 1535", section="Control zone (CTR)")
    assert not is_hit(Chunk("cap1535", "4.2.1", "Entering a CTR"), expected)  # no catalogue: the number of the heading is unknown
    assert is_hit(Chunk("cap1535", "4.2.1", "Entering a CTR"), expected, CATALOGUE)


def test_an_expected_section_may_be_a_number_as_printed():
    assert is_hit(Chunk("cap1535", "4.2.1", "Entering a CTR"), Expected(document="cap1535", section="4.2"))


# --- the run ------------------------------------------------------------------------------------------------------------------------------------


def test_only_the_top_four_chunks_are_looked_at():
    question = QUESTIONS[0]
    chunks = [*wrong_chunks(), *wrong_chunks(), expected_chunk(question)]  # the right one is fifth
    assert run([question], FakeSearch({question.question: chunks})).hits == 0
    assert run([question], FakeSearch({question.question: [*wrong_chunks(), *wrong_chunks()[:1], expected_chunk(question)]})).hits == 1  # fourth


def test_any_chunk_in_any_expected_place_is_a_hit():
    question = QUESTIONS[0]
    for place in question.expected:
        assert run([question], FakeSearch({question.question: [Chunk(place.document, "1", place.section)]})).hits == 1


def test_duplicate_expected_entries_count_as_one_hit_not_two():
    question = Question(question="Restricted?", expected=[Expected(document="CAP 1535", section="Restricted airspace (temporary)")] * 2)
    chunks = [Chunk("cap1535", "7", "Restricted airspace (temporary)"), Chunk("cap1535", "7 (part 2)", "Restricted airspace (temporary)")]
    result = run([question], FakeSearch({"Restricted?": chunks}))
    assert (result.hits, result.total) == (1, 1)
    assert miss_line(run([question], FakeSearch({})).results[0]).count("Restricted airspace (temporary)") == 1  # listed once in a miss line


def test_fewer_than_four_chunks_are_scored_on_what_came_back_and_none_is_a_miss():
    question = QUESTIONS[1]
    assert run([question], FakeSearch({question.question: [expected_chunk(question)]})).hits == 1
    assert run([question], FakeSearch({question.question: [wrong_chunks()[0]]})).hits == 0
    none = run([question], FakeSearch({}))
    assert none.hits == 0 and none.results[0].returned == ()


def test_ten_of_ten_and_zero_of_ten():
    best = FakeSearch({q.question: [expected_chunk(q)] for q in QUESTIONS})
    assert score_line(run(QUESTIONS, best)) == "hit@4: 10/10 (100%)"
    worst = FakeSearch({q.question: wrong_chunks() for q in QUESTIONS})
    assert score_line(run(QUESTIONS, worst)) == "hit@4: 0/10 (0%)"
    assert best.asked == [q.question for q in QUESTIONS]  # each question is asked once, in order


def test_the_returned_headings_are_kept_for_the_report_and_fall_back_to_the_section_number():
    result = run(QUESTIONS[:1], FakeSearch({QUESTIONS[0].question: [Chunk("cap1535", "4.2", "Control zone (CTR)"), Chunk("cap1535", "5", "")]}))
    assert result.results[0].returned == (("cap1535", "Control zone (CTR)"), ("cap1535", "5"))


# --- the output -----------------------------------------------------------------------------------------------------------------------------------


def go(tmp_path, by_question, **options):
    out, err = io.StringIO(), io.StringIO()
    code = main([], search_factory=lambda data_dir: (FakeSearch(by_question), CATALOGUE), stdout=out, stderr=err, moment=STAMP, results_dir=tmp_path / "results", **options)
    return code, out.getvalue(), err.getvalue()


def test_main_prints_the_score_and_a_line_per_miss_and_writes_the_results_file(tmp_path):
    by_question = {q.question: [expected_chunk(q)] for q in QUESTIONS[:7]} | {q.question: wrong_chunks() for q in QUESTIONS[7:]}
    code, out, err = go(tmp_path, by_question)
    assert (code, err) == (0, "")
    lines = out.splitlines()
    assert lines[0] == "hit@4: 7/10 (70%)"
    misses = [line for line in lines if line.startswith("miss: ")]
    assert len(misses) == 3 and all(" | expected: " in line and " | got: cap1535 › Nothing relevant; bhpa-safety › Insurance" in line for line in misses)
    path = tmp_path / "results" / "rag-20261109-123005.md"
    assert lines[-1] == f"results: {path}" and path.is_file()


def test_the_results_file_has_each_questions_hit_or_miss_and_its_top_four_headings(tmp_path):
    by_question = {QUESTIONS[0].question: [Chunk("cap2562", "3", "500 ft rule"), *wrong_chunks()]}
    go(tmp_path, by_question)
    text = (tmp_path / "results" / "rag-20261109-123005.md").read_text(encoding="utf-8")
    assert text.startswith("# RAG hit@4 (2026-11-09 12:30 UTC)\n\nhit@4: 1/10 (10%) over guidance_questions.yaml.\n")
    assert "## Q1: hit" in text and "## Q2: miss" in text
    assert "- top 4 headings: cap2562 › 500 ft rule; cap1535 › Nothing relevant; bhpa-safety › Insurance" in text
    assert text.count("## Q") == 10 and text.count("- expected: ") == 9 and "- top 4 headings: nothing" in text
    assert text.endswith("\n") and not text.endswith("\n\n")


def test_the_exit_code_is_zero_for_any_score_including_zero(tmp_path):
    code, out, _err = go(tmp_path, {})
    assert code == 0 and out.startswith("hit@4: 0/10 (0%)")


def test_a_malformed_questions_file_exits_1_without_a_results_file(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("- question: x\n  expected: []\n", encoding="utf-8")
    out, err = io.StringIO(), io.StringIO()
    code = main(["--questions", str(bad)], search_factory=lambda d: (FakeSearch({}), []), stdout=out, stderr=err, results_dir=tmp_path / "results")
    assert code == 1 and "bad.yaml: question #1" in err.getvalue() and out.getvalue() == "" and not (tmp_path / "results").exists()


def test_a_questions_file_with_no_questions_is_not_a_zero_score(tmp_path):
    empty = tmp_path / "empty.yaml"
    empty.write_text("# nothing yet\n", encoding="utf-8")
    err = io.StringIO()
    assert main(["--questions", str(empty)], search_factory=lambda d: (FakeSearch({}), []), stdout=io.StringIO(), stderr=err, results_dir=tmp_path / "r") == 1
    assert "hit@4 not run: no questions in empty.yaml" in err.getvalue() and not (tmp_path / "r").exists()


def test_a_questions_path_that_is_not_a_file_is_the_no_questions_stop(tmp_path):
    directory = tmp_path / "q.yaml"
    directory.mkdir()
    err = io.StringIO()
    assert main(["--questions", str(directory)], search_factory=lambda d: (FakeSearch({}), []), stdout=io.StringIO(), stderr=err, results_dir=tmp_path / "r") == 1
    assert "hit@4 not run: no questions in q.yaml" in err.getvalue()


def test_a_missing_embedding_model_exits_1_naming_it_and_writes_nothing(tmp_path):
    def no_model(data_dir):
        raise rag_eval.EmbeddingModelMissing("the embedding model is not stored; run `make setup` while online")

    err = io.StringIO()
    code = main([], search_factory=no_model, stdout=io.StringIO(), stderr=err, results_dir=tmp_path / "r")
    assert code == 1 and "hit@4 not run: the embedding model is not stored" in err.getvalue() and not (tmp_path / "r").exists()


def test_with_no_search_factory_an_index_that_is_not_there_exits_1_with_its_reason(tmp_path, monkeypatch):
    class Empty:
        available = False
        reason = "no embeddings.sqlite in the data folder"

    monkeypatch.setattr(rag_eval, "load_index", lambda data_dir: Empty())
    err = io.StringIO()
    code = main(["--data-dir", str(tmp_path)], stdout=io.StringIO(), stderr=err, results_dir=tmp_path / "r")
    assert code == 1 and "the guidance index is not available: no embeddings.sqlite" in err.getvalue() and not (tmp_path / "r").exists()


def test_index_search_is_the_guidance_indexs_own_search_over_the_top_four():
    from flysafelz.guidance.corpus import IndexedChunk

    chunks = [IndexedChunk(f"c{n}", "cap1535", "ed", f"{n}", f"Heading {n}", "text", "hash", (1.0, 0.0)) for n in range(6)]

    class Index:
        def __init__(self):
            self.chunks, self.asked = chunks, []

        def search(self, query, top_k):
            self.asked.append((query, top_k))
            return chunks[:top_k]

    index = Index()
    search, catalogue = rag_eval.index_search(index)  # type: ignore[arg-type]  # a stand-in with the two members used
    assert search("q") == [Chunk("cap1535", str(n), f"Heading {n}") for n in range(4)]
    assert index.asked == [("q", 4)] and len(catalogue) == 6


@pytest.mark.parametrize("name", ["rag-20261109-123005"])
def test_the_file_name_is_rag_and_the_utc_stamp(tmp_path, name):
    go(tmp_path, {})
    assert [p.name for p in (tmp_path / "results").iterdir()] == [f"{name}.md"]


def test_the_rendering_names_the_expected_places_of_a_miss_once():
    result = run(QUESTIONS[4:5], FakeSearch({}))
    assert render(result, STAMP).count("Restricted airspace (temporary)") == 1
