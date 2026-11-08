"""The guidance index (G18, E19): the corpus key decides whether the stored vectors are used; a bad file gives an empty index; the app never writes the file."""

import hashlib
import logging
import os
import sqlite3
from pathlib import Path

import pytest
from factories import EmbedderSpy, guidance_chunk, write_snapshot
from guidance_helpers import FakeEmbedder

from flysafelz.agent.settings import SETTINGS
from flysafelz.guidance import corpus, index
from flysafelz.guidance.embedder import EmbeddingModelMissing
from flysafelz.guidance.index import IndexUnavailable, computed_key, load_index

LOGGER = "flysafelz.guidance.index"
CHUNKS = [
    guidance_chunk("cap1535", "3.2", "danger areas are airspace of defined dimensions where activities dangerous to flight may take place", title="Danger areas"),
    guidance_chunk("cap1535", "3.3", "restricted areas limit flight for reasons of safety or security", title="Restricted areas"),
    guidance_chunk("cap2562", "1.1", "notams tell pilots of hazards on the route", title="NOTAMs"),
]


def snapshot_bytes(folder: Path) -> dict[str, bytes]:
    return {p.name: p.read_bytes() for p in sorted(folder.iterdir())}


def warnings(caplog) -> list[str]:
    return [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING and r.name == LOGGER]


@pytest.fixture
def folder(tmp_path):
    return tmp_path / "guidance"


def test_matching_keys_load_the_stored_vectors_and_embed_no_chunk(folder, caplog):
    embedder = FakeEmbedder()
    key = write_snapshot(folder, CHUNKS, embedder=embedder)
    embedder.calls.clear()
    spy = EmbedderSpy()
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        loaded = load_index(folder, embedder_factory=spy)
    assert loaded.available and len(loaded.chunks) == 3 and loaded.key == loaded.stored_key == key
    assert spy.constructed == 0 and spy.embedder.calls == [] and warnings(caplog) == []  # type: ignore[attr-defined]
    found = loaded.search("danger areas dangerous activities", 4)
    assert found[0].section == "3.2"
    assert [call[0] for call in spy.embedder.calls] == ["query_embed"]  # type: ignore[attr-defined]  # the query only: no chunk was embedded


def test_the_key_is_pr7s_corpus_key_over_the_manifest_the_model_and_the_chunker_version(folder):
    key = write_snapshot(folder, CHUNKS)
    manifest = corpus.read_manifest(folder / corpus.MANIFEST_NAME)
    assert computed_key(folder) == key == corpus.corpus_key(manifest, SETTINGS.embedding_model, SETTINGS.chunker_version)  # type: ignore[arg-type]
    assert computed_key(folder, model="another-model") != key
    assert computed_key(folder, chunker_version=SETTINGS.chunker_version + 1) != key
    source = Path(index.__file__).read_text(encoding="utf-8")
    assert "hashlib" not in source and "sha256" not in source  # no second copy of the key function


def test_a_mismatched_key_embeds_every_chunk_in_memory_warns_once_with_both_keys_and_leaves_the_file_alone(folder, caplog):
    embedder = FakeEmbedder()
    key = write_snapshot(folder, CHUNKS, embedder=embedder, stored_key="0" * 64)
    before, mtime = snapshot_bytes(folder), (folder / corpus.EMBEDDINGS_NAME).stat().st_mtime_ns
    embedder.calls.clear()
    spy = EmbedderSpy(embedder)
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        loaded = load_index(folder, embedder_factory=spy)
        assert spy.constructed == 0 and embedder.calls == []  # nothing is embedded at load, not even on a mismatch
        first = loaded.search("restricted areas safety security", 4)
        second = loaded.search("notams hazards route", 4)
    assert first[0].section == "3.3" and second[0].doc == "cap2562"
    (message,) = warnings(caplog)
    assert "0" * 64 in message and key in message
    assert ("embed", 3) in embedder.calls and spy.constructed == 1  # every chunk, once: the second search reused the vectors
    assert sum(1 for call in embedder.calls if call[0] == "embed") == 1
    assert snapshot_bytes(folder) == before and (folder / corpus.EMBEDDINGS_NAME).stat().st_mtime_ns == mtime


def test_a_file_with_no_stored_key_counts_as_a_mismatch(folder, caplog):
    write_snapshot(folder, CHUNKS)
    with sqlite3.connect(folder / corpus.EMBEDDINGS_NAME) as connection:
        connection.execute("DELETE FROM meta WHERE key = 'corpus_key'")
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        loaded = load_index(folder, embedder_factory=EmbedderSpy())
    assert loaded.stored_key is None and len(warnings(caplog)) == 1
    assert loaded.search("notams", 4)


def test_a_vector_dimension_that_differs_from_the_embedders_is_treated_as_a_key_mismatch(folder, caplog):
    class Narrow:
        """Gives 8-dimensional vectors; the stored ones have the fake embedder's 1024."""

        def __init__(self) -> None:
            self.calls: list[int] = []

        def embed(self, documents, **kwargs):
            texts = list(documents)
            self.calls.append(len(texts))
            return [self._vector(t) for t in texts]

        query_embed = embed

        @staticmethod
        def _vector(text: str) -> list[float]:
            values = [0.0] * 8
            for word in text.split():
                values[hashlib.sha256(word.encode()).digest()[0] % 8] += 1.0
            return values

    key = write_snapshot(folder, CHUNKS)  # the keys match, the dimensions do not
    before = snapshot_bytes(folder)
    narrow = Narrow()
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        loaded = load_index(folder, embedder_factory=lambda: narrow)
        assert warnings(caplog) == []  # the keys agree, so load says nothing; the dimensions show at the first search
        found = loaded.search("danger areas dangerous activities", 4)
        loaded.search("notams", 4)
    assert len(found) == 3
    (message,) = warnings(caplog)
    assert "[1024]" in message and "embedder gives 8," in message and key in message
    assert narrow.calls.count(3) == 1  # all three chunks embedded in memory, once
    assert snapshot_bytes(folder) == before


def test_a_missing_embeddings_file_gives_an_empty_index_and_one_warning_naming_it(folder, caplog):
    write_snapshot(folder, CHUNKS)
    (folder / corpus.EMBEDDINGS_NAME).unlink()
    spy = EmbedderSpy()
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        loaded = load_index(folder, embedder_factory=spy)
    assert not loaded.available and loaded.search("anything", 4) == [] and spy.constructed == 0
    (message,) = warnings(caplog)
    assert str(folder / corpus.EMBEDDINGS_NAME) in message and loaded.reason is not None


def test_a_missing_manifest_gives_an_empty_index_and_a_warning_naming_it(folder, caplog):
    write_snapshot(folder, CHUNKS)
    (folder / corpus.MANIFEST_NAME).unlink()
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        loaded = load_index(folder, embedder_factory=EmbedderSpy())
    assert not loaded.available and loaded.search("anything", 4) == []
    assert str(folder / corpus.MANIFEST_NAME) in " ".join(warnings(caplog))


def test_a_folder_with_neither_file_gives_an_empty_index_naming_both(tmp_path, caplog):
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        loaded = load_index(tmp_path / "nothing-here", embedder_factory=EmbedderSpy())
    (message,) = warnings(caplog)
    assert not loaded.available and corpus.EMBEDDINGS_NAME in message and corpus.MANIFEST_NAME in message


@pytest.mark.parametrize(
    "damage",
    [
        lambda path: path.write_bytes(b"this is not a database at all, just some text\n" * 20),
        lambda path: path.write_bytes(path.read_bytes()[:100]),
        lambda path: path.write_bytes(b""),
        lambda path: _drop_table(path, "chunks"),
        lambda path: _drop_table(path, "meta"),
    ],
    ids=["not-a-database", "truncated", "empty-file", "no-chunks-table", "no-meta-table"],
)
def test_a_corrupt_or_truncated_file_is_treated_like_a_missing_one_and_the_warning_names_the_path(folder, caplog, damage):
    write_snapshot(folder, CHUNKS)
    path = folder / corpus.EMBEDDINGS_NAME
    damage(path)
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        loaded = load_index(folder, embedder_factory=EmbedderSpy())
    assert not loaded.available and loaded.search("anything", 4) == []
    assert str(path) in " ".join(warnings(caplog))


def _drop_table(path: Path, table: str) -> None:
    with sqlite3.connect(path) as connection:
        connection.execute(f"DROP TABLE {table}")


def test_a_manifest_that_is_not_in_the_shape_make_guidance_writes_gives_an_empty_index(folder, caplog):
    write_snapshot(folder, CHUNKS)
    (folder / corpus.MANIFEST_NAME).write_text("documents:\n  - id: cap1535\n", encoding="utf-8")  # no edition, no sha256
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        loaded = load_index(folder, embedder_factory=EmbedderSpy())
    assert not loaded.available and str(folder / corpus.MANIFEST_NAME) in " ".join(warnings(caplog))
    (folder / corpus.MANIFEST_NAME).write_text(": : not yaml [", encoding="utf-8")
    assert not load_index(folder, embedder_factory=EmbedderSpy()).available


def test_computed_key_raises_for_a_missing_manifest(tmp_path):
    with pytest.raises(IndexUnavailable, match="manifest.yaml"):
        computed_key(tmp_path)


def test_loading_never_writes_the_file_or_adds_a_journal_next_to_it(folder):
    write_snapshot(folder, CHUNKS, stored_key="f" * 64)
    names, before = sorted(os.listdir(folder)), snapshot_bytes(folder)
    mtimes = {p.name: p.stat().st_mtime_ns for p in folder.iterdir()}
    loaded = load_index(folder, embedder_factory=EmbedderSpy())
    loaded.search("danger", 4)
    loaded.search("notams", 2)
    assert sorted(os.listdir(folder)) == names and snapshot_bytes(folder) == before
    assert {p.name: p.stat().st_mtime_ns for p in folder.iterdir()} == mtimes


def test_a_document_the_register_marks_link_only_has_no_chunks_in_the_index(folder, caplog):
    chunks = [*CHUNKS, guidance_chunk("bhpa-safety", "2.1", "danger areas are described here by the bhpa", title="BHPA")]
    write_snapshot(folder, chunks, link_only=["bhpa-safety"])  # the register says link-only although the file holds its text
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        loaded = load_index(folder, embedder_factory=EmbedderSpy())
    assert {c.doc for c in loaded.chunks} == {"cap1535", "cap2562"}
    assert all(c.doc != "bhpa-safety" for c in loaded.search("danger areas bhpa", 10))
    assert any("link-only" in m and "bhpa-safety" in m for m in warnings(caplog))


def test_a_folder_whose_only_documents_are_link_only_is_empty(folder):
    write_snapshot(folder, [guidance_chunk("bhpa-safety", "1", "text")], link_only=["bhpa-safety"])
    assert not load_index(folder, embedder_factory=EmbedderSpy()).available


def test_a_register_that_cannot_be_read_marks_nothing_link_only_and_says_so(folder, caplog):
    write_snapshot(folder, CHUNKS)
    (folder / "sources.yaml").write_text("documents: 7\n", encoding="utf-8")
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        loaded = load_index(folder, embedder_factory=EmbedderSpy())
    assert len(loaded.chunks) == 3 and any("link-only" in m for m in warnings(caplog))


def test_chunks_are_held_in_document_then_section_order(folder):
    shuffled = [CHUNKS[2], CHUNKS[1], CHUNKS[0]]
    write_snapshot(folder, shuffled)
    assert [(c.doc, c.section) for c in load_index(folder, embedder_factory=EmbedderSpy()).chunks] == [("cap1535", "3.2"), ("cap1535", "3.3"), ("cap2562", "1.1")]


def test_the_embedder_is_built_at_the_first_search_once_and_a_failing_factory_is_asked_again(folder):
    write_snapshot(folder, CHUNKS)
    failing = EmbedderSpy(error=EmbeddingModelMissing("embedding model missing: run make setup"))
    loaded = load_index(folder, embedder_factory=failing)
    assert failing.constructed == 0
    for attempt in (1, 2):
        with pytest.raises(EmbeddingModelMissing, match="make setup"):
            loaded.search("danger", 4)
        assert failing.constructed == attempt
    working = EmbedderSpy()
    loaded = load_index(folder, embedder_factory=working)
    loaded.search("danger", 4)
    loaded.search("notams", 4)
    assert working.constructed == 1


def test_an_empty_corpus_file_is_an_empty_index(folder):
    write_snapshot(folder, [])
    assert not load_index(folder, embedder_factory=EmbedderSpy()).available
