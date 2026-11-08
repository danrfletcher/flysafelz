"""search_guidance (G13): run in a real `create_agent` graph, over an index of a tiny snapshot built in tmp_path with the fake embedder."""

import json
from pathlib import Path

import pytest
from factories import EmbedderSpy, guidance_chunk, write_snapshot
from guidance_helpers import FakeEmbedder
from tool_helpers import calls, run_tools

from flysafelz.agent.settings import SETTINGS, embedding_folder
from flysafelz.agent.sources import SourceCache
from flysafelz.agent.tools import guidance_tool
from flysafelz.agent.tools.artifacts import GuidanceArtifact
from flysafelz.agent.tools.guidance_tool import NOT_AVAILABLE, build_search_guidance
from flysafelz.guidance import corpus, embedder
from flysafelz.guidance.embedder import EmbeddingModelMissing
from flysafelz.guidance.index import load_index

CHUNKS = [
    guidance_chunk("cap1535", "3.2", "danger areas are airspace where activities dangerous to flight take place", title="Danger areas", edition="2"),
    guidance_chunk("cap1535", "3.3", "restricted areas limit flight for reasons of safety or security", title="Restricted areas", edition="2"),
    guidance_chunk("cap1535", "3.4", "prohibited areas forbid flight altogether", title="Prohibited areas", edition="2"),
    guidance_chunk("cap2562", "1.1", "notams tell pilots of hazards on the route", title="NOTAMs", edition="5"),
    guidance_chunk("cap2562", "1.2", "a notam has an item e that says what the hazard is", title="Item E", edition="5"),
    guidance_chunk("cap2562", "2.1", "the skyway code explains how to read a chart", title="Charts", edition="5"),
]
SPEC_HEADING = "cap1535 2 §3.2 Danger areas"


def search(tool_index, query: str | None, tool=None):
    return run_tools(SourceCache(), [calls("search_guidance", query=query)], extra_tools=[tool or build_search_guidance(tool_index)])


def data(run) -> dict:
    """The artifact of the run's search_guidance message as plain data."""
    return run.artifact("search_guidance").model_dump(mode="json")


def index_of(folder, chunks=CHUNKS, spy=None, **options):
    write_snapshot(folder, chunks, **options)
    return load_index(folder, embedder_factory=spy or EmbedderSpy())


@pytest.fixture
def full(tmp_path):
    return index_of(tmp_path / "g")


def test_the_top_four_come_back_best_first_each_as_a_heading_line_and_its_text(full):
    run = search(full, "danger areas dangerous activities flight")
    text = run.text("search_guidance")
    lines = text.splitlines()
    assert lines[0].startswith("Passages from published guidance")
    headings = [line for line in lines if line.startswith(("cap1535 ", "cap2562 "))]
    assert len(headings) == SETTINGS.top_k == 4 and headings[0] == SPEC_HEADING
    assert lines[lines.index(SPEC_HEADING) + 1] == CHUNKS[0].text  # the text follows its heading line
    artifact = data(run)
    assert artifact["chunks"][0]["section"] == "3.2" and len(artifact["chunks"]) == 4


def test_the_artifact_carries_the_same_fields_as_data(full):
    artifact = data(search(full, "danger areas dangerous activities flight"))
    assert set(artifact) == {"tool", "query", "chunks"} and artifact["tool"] == "search_guidance" and artifact["query"] == "danger areas dangerous activities flight"
    first = artifact["chunks"][0]
    assert first == {"document": "cap1535", "edition": "2", "section": "3.2", "title": "Danger areas", "text": CHUNKS[0].text}
    assert GuidanceArtifact.model_validate(artifact).chunks[0].section == "3.2"
    text = search(full, "danger areas dangerous activities flight").text("search_guidance")
    for chunk in artifact["chunks"]:  # every section number printed in the text is in the artifact, and the other way round
        assert f"{chunk['document']} {chunk['edition']} §{chunk['section']} {chunk['title']}" in text
    assert text.count("§") == len(artifact["chunks"])


def test_query_is_the_only_argument_the_model_sees(full):
    tool = build_search_guidance(full)
    assert tool.name == "search_guidance"
    schema = tool.tool_call_schema.model_json_schema()
    assert set(schema["properties"]) == {"query"} and schema["required"] == ["query"]
    forbidden = {"lat", "lon", "latitude", "longitude", "tz", "timezone", "now", "case", "runtime", "top_k", "start", "end"}
    assert not forbidden & set(tool.args_schema.model_fields) - {"runtime"}  # type: ignore[union-attr]
    from pydantic import ValidationError

    for extra in forbidden - {"runtime"}:
        with pytest.raises(ValidationError):
            tool.args_schema.model_validate({"query": "x", extra: 1})  # type: ignore[union-attr]


def test_top_k_is_read_from_the_settings_not_written_in_the_tool(full):
    assert "SETTINGS.top_k" in Path(guidance_tool.__file__).read_text(encoding="utf-8")
    assert len(data(search(full, "areas"))["chunks"]) == SETTINGS.top_k


def test_fewer_chunks_than_top_k_returns_all_of_them_in_rank_order(tmp_path):
    two = index_of(tmp_path / "g", CHUNKS[:2])
    chunks = data(search(two, "restricted areas safety security"))["chunks"]
    assert [c["section"] for c in chunks] == ["3.3", "3.2"]


def test_equal_similarity_is_broken_by_document_then_section_and_is_stable(tmp_path):
    same = "identical wording about airspace"
    tied = [guidance_chunk("zeta", "1", same), guidance_chunk("alpha", "2", same), guidance_chunk("alpha", "1", same), guidance_chunk("mid", "9", same), guidance_chunk("other", "1", "unrelated")]
    tied_index = index_of(tmp_path / "g", tied)
    runs = [[(c["document"], c["section"]) for c in data(search(tied_index, same))["chunks"]] for _ in range(3)]
    assert runs[0] == runs[1] == runs[2] == [("alpha", "1"), ("alpha", "2"), ("mid", "9"), ("zeta", "1")]


@pytest.mark.parametrize("query", ["", "   ", "\n\t "])
def test_an_empty_or_blank_query_is_a_tool_message_the_model_can_relay(full, query):
    run = search(full, query)
    assert "needs a question or some key words" in run.text("search_guidance")
    assert data(run)["chunks"] == []


def test_a_very_long_query_and_a_unicode_query_return_at_most_four_chunks(full):
    for query in ("danger " * 1500, "zone de danger — 危险区域 🛩️ café naïve", "\u0000\u202e control characters"):
        assert 0 < len(query) <= 10_500
        run = search(full, query)
        assert len(data(run)["chunks"]) <= SETTINGS.top_k
    assert len("danger " * 1500) >= 10_000


def test_nothing_embedding_close_still_returns_the_nearest_chunks(full):
    assert len(data(search(full, "zzzz qqqq"))["chunks"]) == SETTINGS.top_k  # cosine ranks everything; there is no threshold (F7)


def test_a_missing_index_returns_the_fixed_text_and_an_empty_artifact(tmp_path):
    missing = load_index(tmp_path / "none", embedder_factory=EmbedderSpy())
    run = search(missing, "what is a danger area")
    assert run.text("search_guidance") == NOT_AVAILABLE == "Guidance index is not available"
    assert data(run) == {"tool": "search_guidance", "query": "what is a danger area", "chunks": []}


def test_a_corrupt_file_gives_the_same_answer_as_a_missing_one(tmp_path):
    write_snapshot(tmp_path / "g", CHUNKS)
    (tmp_path / "g" / corpus.EMBEDDINGS_NAME).write_bytes(b"not a database" * 50)
    run = search(load_index(tmp_path / "g", embedder_factory=EmbedderSpy()), "danger")
    assert run.text("search_guidance") == NOT_AVAILABLE and data(run)["chunks"] == []


def test_a_model_that_make_setup_has_not_stored_is_text_naming_make_setup_not_a_traceback(tmp_path):
    spy = EmbedderSpy(error=EmbeddingModelMissing("embedding model missing: run make setup"))
    run = search(index_of(tmp_path / "g", spy=spy), "danger areas")
    assert run.text("search_guidance").startswith(NOT_AVAILABLE) and "make setup" in run.text("search_guidance")
    assert data(run)["chunks"] == []


def test_the_embedder_is_built_at_the_first_search_exactly_once_and_reused(tmp_path):
    spy = EmbedderSpy()
    loaded = index_of(tmp_path / "g", spy=spy)
    tool = build_search_guidance(loaded)
    assert spy.constructed == 0  # loading the index and building the tool construct nothing
    first = run_tools(SourceCache(), [calls("search_guidance", query="danger areas")], extra_tools=[tool])
    assert spy.constructed == 1 and data(first)["chunks"]
    run_tools(SourceCache(), [calls("search_guidance", query="notams hazards"), calls("search_guidance", query="charts")], extra_tools=[tool])
    assert spy.constructed == 1


def test_the_real_factory_builds_text_embedding_lazily_with_local_files_only_and_the_settings_cache(tmp_path, monkeypatch):
    """Through the default factory: nothing is constructed at import, at load or when the tool is built; the first search builds `TextEmbedding(model, cache_dir=..., local_files_only=True)` once."""
    constructed: list[tuple[tuple, dict]] = []

    def fake_text_embedding(*args, **kwargs):
        constructed.append((args, kwargs))
        return FakeEmbedder()

    monkeypatch.setattr(embedder, "TextEmbedding", fake_text_embedding)
    write_snapshot(tmp_path / "g", CHUNKS)
    loaded = load_index(tmp_path / "g")
    tool = build_search_guidance(loaded)
    assert constructed == []
    run_tools(SourceCache(), [calls("search_guidance", query="danger areas")], extra_tools=[tool])
    run_tools(SourceCache(), [calls("search_guidance", query="notams")], extra_tools=[tool])
    assert constructed == [((SETTINGS.embedding_model,), {"cache_dir": str(embedding_folder()), "local_files_only": True})]


def test_the_text_passes_through_the_redaction_boundary(tmp_path):
    chunks = [guidance_chunk("cap1535", "9.1", "for advice phone the duty officer on 01234 567890 about danger areas", title="Contacts")]
    run = search(index_of(tmp_path / "g", chunks), "duty officer danger areas")
    assert "567890" not in run.text("search_guidance") and "567890" not in json.dumps(data(run))
    assert "<PHONE_1>" in run.text("search_guidance")


def test_a_link_only_document_is_never_returned(tmp_path):
    chunks = [*CHUNKS, guidance_chunk("bhpa-safety", "1.1", "danger areas danger areas danger areas bhpa advice", title="BHPA advice")]
    run = search(index_of(tmp_path / "g", chunks, link_only=["bhpa-safety"]), "danger areas bhpa advice")
    assert "bhpa" not in run.text("search_guidance").lower()
    assert all(c["document"] != "bhpa-safety" for c in data(run)["chunks"])
