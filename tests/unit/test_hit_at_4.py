"""The hit@4 rule and the aggregate line, table-driven (G19). The rule is PR-7's hit4.py; this file drives it through the scorer `make eval-guidance` and the promote gate share."""

import pytest
from factories import EmbedderSpy, guidance_chunk, write_snapshot

from flysafelz.agent.settings import SETTINGS
from flysafelz.guidance.eval import hit4_of_index, miss_line, result_line
from flysafelz.guidance.hit4 import Expected, Question, run_hit4, section_matches
from flysafelz.guidance.index import load_index

D = "cap1535"
TOP4 = [(D, "9"), (D, "8"), (D, "7"), (D, "6")]


@pytest.mark.parametrize(
    ("returned", "expected", "hit"),
    [
        pytest.param([(D, "3.2")], [(D, "3.2")], True, id="exact"),
        pytest.param([(D, "3.2.1")], [(D, "3.2")], True, id="subsection"),
        pytest.param([(D, "3.2.1.4")], [(D, "3.2")], True, id="deeper-subsection"),
        pytest.param([(D, "3.2 (part 2)")], [(D, "3.2")], True, id="part-of-a-split-section"),
        pytest.param([(D, "3.20")], [(D, "3.2")], False, id="prefix-trap-3.20"),
        pytest.param([(D, "13.2")], [(D, "3.2")], False, id="prefix-trap-13.2"),
        pytest.param([(D, "3")], [(D, "3.2")], False, id="parent-is-not-the-child"),
        pytest.param([("cap2562", "3.2")], [(D, "3.2")], False, id="wrong-document-right-section"),
        pytest.param([*TOP4, (D, "3.2")], [(D, "3.2")], False, id="rank-5"),
        pytest.param([*TOP4[:3], (D, "3.2")], [(D, "3.2")], True, id="rank-4"),
        pytest.param([(D, "1")], [(D, "3.2"), ("cap2562", "1")], False, id="multi-entry-wrong-document-for-the-matching-section"),
        pytest.param([("cap2562", "1.1")], [(D, "3.2"), ("cap2562", "1")], True, id="multi-entry-hit-on-the-second"),
        pytest.param([(D, "3.2")], [(D, "3.2"), ("cap2562", "1")], True, id="multi-entry-hit-on-the-first"),
        pytest.param([(D, "5.5"), ("cap2562", "7")], [(D, "3.2"), ("cap2562", "1")], False, id="multi-entry-miss-on-none"),
        pytest.param([], [(D, "3.2")], False, id="nothing-came-back"),
    ],
)
def test_the_hit_rule(returned, expected, hit):
    question = Question(question="q", expected=[Expected(document=d, section=s) for d, s in expected])
    outcome = run_hit4([question], lambda _q: returned, SETTINGS.top_k)
    assert (outcome.hits, outcome.total) == (int(hit), 1)


@pytest.mark.parametrize(("chunk", "expected", "match"), [("3.2", "3.2", True), ("3.2.1", "3.2", True), ("3.20", "3.2", False), ("13.2", "3.2", False), ("3.2", "3.2.1", False)])
def test_the_section_rule(chunk, expected, match):
    assert section_matches(chunk, expected) is match


def test_the_aggregate_line_counts_hits_and_gives_the_percentage(tmp_path):
    chunks = [guidance_chunk("a", "1", "alpha beta gamma"), guidance_chunk("a", "2", "delta epsilon zeta"), guidance_chunk("b", "1", "eta theta iota"), guidance_chunk("b", "2", "kappa lambda mu")]
    loaded = load_index(_folder(tmp_path, chunks), embedder_factory=EmbedderSpy())
    questions = [
        Question(question="alpha beta gamma", expected=[Expected(document="a", section="1")]),
        Question(question="delta epsilon zeta", expected=[Expected(document="a", section="2")]),
        Question(question="eta theta iota", expected=[Expected(document="b", section="9")]),  # asked for a section that does not exist: a miss
    ]
    result = hit4_of_index(loaded, questions)
    assert (result.hits, result.total) == (2, 3) and result_line(result) == "hit@4: 2/3 (67%)"
    (miss,) = result.misses
    assert miss_line(miss).startswith("miss: eta theta iota | expected: b §9 | got: ") and miss_line(miss).count("§") == 1 + 4


def test_every_question_gets_the_top_k_of_the_settings(tmp_path):
    loaded = load_index(_folder(tmp_path, [guidance_chunk("a", str(n), f"word{n} common") for n in range(1, 8)]), embedder_factory=EmbedderSpy())
    result = hit4_of_index(loaded, [Question(question="common", expected=[Expected(document="a", section="1")])])
    assert len(result.results[0].returned) == SETTINGS.top_k


def _folder(tmp_path, chunks):
    write_snapshot(tmp_path / "g", chunks)
    return tmp_path / "g"
