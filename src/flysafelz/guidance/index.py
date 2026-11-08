"""The guidance search index: the chunks and vectors `make guidance` committed, searched by cosine similarity (G13, G18, E19).

    load_index(data_dir, ...)   -> GuidanceIndex   read `embeddings.sqlite` and `manifest.yaml` from the folder; never raises for a bad or missing file
    GuidanceIndex.search(query, top_k) -> list[IndexedChunk]   the `top_k` chunks most like the query, best first
    GuidanceIndex.available     False when there is nothing to search (a missing or unreadable file); `reason` says why
    computed_key(data_dir)      -> str   the corpus key the manifest, the model and the chunker version give (PR-7's `corpus.corpus_key`)

Load. The stored corpus key (in the file's `meta` table) is compared with the key computed from `manifest.yaml`, the embedding model's name and the
chunker version. When they match the stored vectors are used and nothing is embedded. When they differ (or the stored vectors have another dimension
than the embedder's) every chunk is embedded again in memory, once, at the first search, and a WARNING names both keys. The file is opened read-only
and is never written: only `make guidance` (ingest.py) writes it.

Lazy embedder. Loading the index never builds the embedder: `embedder_factory` is called at the first search, exactly once, and the result is reused
(fastembed takes about a second to load). A factory that raises (the model is not stored: run `make setup`) is called again at the next search.

Missing or unreadable. A missing `embeddings.sqlite` or `manifest.yaml`, a file that is not a database, a table that is missing or a manifest that is
not in the shape `make guidance` writes: the index is empty (`available` is False), one WARNING names every path that is bad, and a search returns [].

Link-only documents. A chunk of a document that sources.yaml marks `indexed: false` is dropped at load, whatever the file holds, so link-only text (G16)
is never served even if it got into the file by mistake.

Order. The chunks are held sorted by document, then section, then id, and equal similarities keep that order, so results are the same on every run.
"""

import logging
import sqlite3
import struct
import threading
from collections.abc import Callable, Sequence
from pathlib import Path

import yaml

from flysafelz.agent.settings import SETTINGS, repository_root
from flysafelz.guidance import corpus
from flysafelz.guidance.corpus import EMBEDDINGS_NAME, MANIFEST_NAME, CorpusError, IndexedChunk
from flysafelz.guidance.embedder import Embedder, load_settings_embedder
from flysafelz.guidance.hit4 import rank
from flysafelz.guidance.sources import SourcesError, load_sources

log = logging.getLogger(__name__)

SOURCES_NAME = "sources.yaml"

EmbedderFactory = Callable[[], Embedder]


class IndexUnavailable(ValueError):
    """A file the index needs is missing or is not in the shape `make guidance` writes. The message names the file."""


def default_data_dir() -> Path:
    return repository_root() / "data" / "guidance"


def computed_key(data_dir: Path, model: str | None = None, chunker_version: int | None = None) -> str:
    """The corpus key of the manifest in `data_dir`, for the model (default: settings.yaml's) and chunker version (default: settings.yaml's).

    Uses PR-7's `corpus.corpus_key`, the function `make guidance` files the embeddings under. Raises IndexUnavailable if there is no usable manifest.
    """
    path = data_dir / MANIFEST_NAME
    try:
        manifest = corpus.read_manifest(path)
    except (CorpusError, yaml.YAMLError, OSError, UnicodeDecodeError) as error:
        raise IndexUnavailable(f"{path} cannot be read: {error}") from error
    if manifest is None:
        raise IndexUnavailable(f"{path} is missing")
    try:
        return corpus.corpus_key(manifest, SETTINGS.embedding_model if model is None else model, SETTINGS.chunker_version if chunker_version is None else chunker_version)
    except (KeyError, TypeError) as error:
        raise IndexUnavailable(f"{path} is not in the shape `make guidance` writes: {type(error).__name__} {error}") from error


def _read_chunks(path: Path) -> tuple[str | None, list[IndexedChunk]]:
    """(the stored corpus key, the chunks) of the embeddings file, opened read-only. Raises IndexUnavailable for a missing or unreadable file."""
    if not path.is_file():
        raise IndexUnavailable(f"{path} is missing")
    try:
        meta, chunks = corpus.read_embeddings(path)
    except (sqlite3.Error, struct.error, OSError) as error:
        raise IndexUnavailable(f"{path} cannot be read ({type(error).__name__}: {error})") from error
    return meta.get("corpus_key"), chunks


def _order(chunk: IndexedChunk) -> tuple[str, str, str]:
    return chunk.doc, chunk.section, chunk.id


def _link_only_documents(data_dir: Path) -> set[str]:
    """The ids sources.yaml marks `indexed: false`. No register, or one that cannot be read, marks none (a warning says so for a bad one)."""
    path = data_dir / SOURCES_NAME
    if not path.is_file():
        return set()
    try:
        return {source.id for source in load_sources(path) if not source.indexed}
    except SourcesError as error:
        log.warning("Guidance index: cannot tell which documents are link-only: %s", error)
        return set()


class GuidanceIndex:
    """Chunks and their vectors, searched by cosine similarity. Build one with `load_index`."""

    def __init__(
        self,
        chunks: Sequence[IndexedChunk],
        *,
        key: str | None,
        stored_key: str | None,
        embedder_factory: EmbedderFactory,
        reason: str | None = None,
        path: Path | None = None,
    ) -> None:
        """`key` is the computed corpus key and `stored_key` the one in the file (None: unknown). `reason` is why the index is empty, when it is."""
        self.chunks: tuple[IndexedChunk, ...] = tuple(sorted(chunks, key=_order))
        self.key = key
        self.stored_key = stored_key
        self.reason = reason
        self.path = path
        self._factory = embedder_factory
        self._lock = threading.RLock()
        self._embedder: Embedder | None = None
        self._warned = False
        keys_match = key is not None and key == stored_key
        self._vectors: list[tuple[float, ...]] | None = [chunk.vector for chunk in self.chunks] if keys_match else None  # None: to be embedded at the first search
        if self.chunks and not keys_match:
            self._warn_key_mismatch()

    @property
    def available(self) -> bool:
        """True when there is something to search."""
        return bool(self.chunks)

    def _warn_key_mismatch(self) -> None:
        self._warned = True
        log.warning(
            "Guidance index: the corpus key stored in %s (%s) does not match the key computed from the manifest, the embedding model and the chunker version (%s). "
            "Embedding all %d chunks in memory at the first search. The file is not changed.",
            self.path,
            self.stored_key,
            self.key,
            len(self.chunks),
        )

    def _get_embedder(self) -> Embedder:
        with self._lock:
            if self._embedder is None:
                self._embedder = self._factory()
            return self._embedder

    def _query_vector(self, query: str) -> tuple[float, ...]:
        (vector,) = list(self._get_embedder().query_embed([query]))
        return tuple(float(x) for x in vector)

    def _vectors_for(self, dimension: int) -> list[tuple[float, ...]]:
        """The vectors to rank against a query of this dimension: the stored ones when they fit, otherwise every chunk embedded here, once."""
        with self._lock:
            vectors = self._vectors
            if vectors is not None and all(len(vector) == dimension for vector in vectors):
                return vectors
            if not self._warned:  # the keys match, so the stored vectors are from another model than the embedder's: the same thing as a key mismatch
                self._warned = True
                log.warning(
                    "Guidance index: the vectors stored in %s have %s dimensions and the embedder gives %d, so they are not from this model "
                    "(stored corpus key %s, computed key %s). Embedding all %d chunks in memory. The file is not changed.",
                    self.path,
                    sorted({len(chunk.vector) for chunk in self.chunks}),
                    dimension,
                    self.stored_key,
                    self.key,
                    len(self.chunks),
                )
            embedded = self._get_embedder().embed([chunk.text for chunk in self.chunks])
            self._vectors = [tuple(float(x) for x in vector) for vector in embedded]
            return self._vectors

    def search(self, query: str, top_k: int) -> list[IndexedChunk]:
        """The `top_k` chunks most like `query`, best first; fewer when the index has fewer. Equal similarity: by document, then section."""
        if not self.chunks:
            return []
        query_vector = self._query_vector(query)
        vectors = self._vectors_for(len(query_vector))
        return [self.chunks[position] for position in rank(query_vector, vectors, top_k)]


def load_index(
    data_dir: Path | None = None,
    *,
    embedder_factory: EmbedderFactory | None = None,
    model: str | None = None,
    chunker_version: int | None = None,
    apply_register: bool = True,
) -> GuidanceIndex:
    """The index over `data_dir`'s `embeddings.sqlite` and `manifest.yaml` (default: data/guidance). Embeds nothing, builds no embedder, writes nothing.

    `embedder_factory` defaults to the embedder for the model settings.yaml names, loaded from the folder `make setup` filled. `model` and `chunker_version`
    default to settings.yaml's; they are part of the corpus key the stored one is compared with. `apply_register` False skips the link-only filter
    (the hit@4 gate scores whole snapshots, one of them in staging, which has no register beside it).
    """
    data_dir = default_data_dir() if data_dir is None else data_dir
    factory = load_settings_embedder if embedder_factory is None else embedder_factory
    path = data_dir / EMBEDDINGS_NAME
    problems: list[str] = []
    stored_key: str | None = None
    chunks: list[IndexedChunk] = []
    key: str | None = None
    try:
        stored_key, chunks = _read_chunks(path)
    except IndexUnavailable as error:
        problems.append(str(error))
    try:
        key = computed_key(data_dir, model, chunker_version)
    except IndexUnavailable as error:
        problems.append(str(error))
    if not problems and not chunks:
        problems.append(f"{path} holds no chunks")
    kept = chunks
    if not problems:
        link_only = _link_only_documents(data_dir) if apply_register else set()
        kept = [chunk for chunk in chunks if chunk.doc not in link_only]
        if len(kept) != len(chunks):
            log.warning("Guidance index: dropped %d chunks of link-only documents (%s) that %s holds", len(chunks) - len(kept), ", ".join(sorted(link_only)), path)
        if not kept:
            problems.append(f"{path} holds no chunks of an indexed document")
    if problems:
        reason = "; ".join(problems)
        log.warning("Guidance index is not available: %s", reason)
        return GuidanceIndex([], key=key, stored_key=stored_key, embedder_factory=factory, reason=reason, path=path)
    return GuidanceIndex(kept, key=key, stored_key=stored_key, embedder_factory=factory, path=path)
