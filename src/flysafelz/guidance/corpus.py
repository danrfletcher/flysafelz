"""What `make guidance` writes and reads: section files, the manifest, the corpus key and `embeddings.sqlite` (G17).

    corpus_key(manifest, model, chunker_version)   -> str    sha256 over the sorted per-document version slugs, the model name and the chunker version
    write_section_files(folder, chunks)            -> None   one `NNNN-<section>.md` per chunk, with frontmatter
    read_section_files(folder)                     -> list[Chunk]
    write_manifest(path, manifest) / read_manifest(path)
    write_embeddings(path, chunks, vectors, meta)  -> None   `meta(key, value)` and `chunks(id, doc, edition, section, title, text, content_hash, vector BLOB)`
    read_embeddings(path)                          -> (meta, list[IndexedChunk])

The manifest is a mapping: `corpus_key`, `model`, `chunker_version` and `documents`, a list of {id, title, edition, url, sha256, chunker_version, sections}; a document's chunker_version is the one its sections were split with.
A document's version slug is its id, its edition and the SHA-256 of the bytes that were downloaded. Vectors are little-endian float32.
Only `make guidance` writes any of this; the app opens `embeddings.sqlite` read-only.
"""

import hashlib
import json
import sqlite3
import struct
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from flysafelz.guidance.chunker import Chunk, section_slug

MANIFEST_NAME = "manifest.yaml"
EMBEDDINGS_NAME = "embeddings.sqlite"
FRONTMATTER_FENCE = "---"
FRONTMATTER_KEYS = ("doc", "edition", "section", "title", "content_hash")


class CorpusError(ValueError):
    """A section file or the manifest is not in the shape `make guidance` writes."""


@dataclass(frozen=True)
class IndexedChunk:
    """A chunk as stored in embeddings.sqlite, with its vector."""

    id: str
    doc: str
    edition: str
    section: str
    title: str
    text: str
    content_hash: str
    vector: tuple[float, ...]


def corpus_key(manifest: Mapping[str, Any], model: str, chunker_version: int) -> str:
    """The key the committed embeddings are filed under. Changes with any document's id, edition or downloaded bytes, the model, or the chunker version.

    Order-free: documents are sorted before they are hashed. Only `manifest["documents"]`' id, edition and sha256 are read.
    """
    slugs = sorted(f"{document['id']}@{document['edition']}@{document['sha256']}" for document in manifest["documents"])
    payload = json.dumps({"documents": slugs, "model": model, "chunker_version": chunker_version}, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def section_file_name(index: int, chunk: Chunk) -> str:
    """`0003-4.2-part-2.md`: the position keeps document order when files are listed by name."""
    return f"{index:04d}-{section_slug(chunk.section)}.md"


def chunk_ids(chunks: Sequence[Chunk]) -> list[str]:
    """`<doc>/<file stem>` for each chunk, numbered within its document the way its section file is: unique across the corpus."""
    position: dict[str, int] = {}
    ids = []
    for chunk in chunks:
        index = position.get(chunk.doc, 0)
        position[chunk.doc] = index + 1
        ids.append(f"{chunk.doc}/{section_file_name(index, chunk).removesuffix('.md')}")
    return ids


def write_section_files(folder: Path, chunks: Sequence[Chunk]) -> None:
    """Write each chunk as `folder/NNNN-<section>.md`: frontmatter (doc, edition, section, title, content_hash), a blank line, the text."""
    folder.mkdir(parents=True, exist_ok=True)
    for index, chunk in enumerate(chunks):
        front = {key: getattr(chunk, key) for key in FRONTMATTER_KEYS}
        header = yaml.safe_dump(front, sort_keys=False, allow_unicode=True, width=float("inf"))
        (folder / section_file_name(index, chunk)).write_text(f"{FRONTMATTER_FENCE}\n{header}{FRONTMATTER_FENCE}\n\n{chunk.text}\n", encoding="utf-8")


def read_section_file(path: Path) -> Chunk:
    """One section file back into a Chunk. Raises CorpusError for a file without the frontmatter keys."""
    text = path.read_text(encoding="utf-8")
    opening = f"{FRONTMATTER_FENCE}\n"
    closing = f"\n{FRONTMATTER_FENCE}\n\n"
    if not text.startswith(opening) or closing not in text:
        raise CorpusError(f"{path.name}: no frontmatter")
    header, body = text[len(opening) :].split(closing, 1)
    front = yaml.safe_load(header)
    if not isinstance(front, dict) or set(front) != set(FRONTMATTER_KEYS) or not all(isinstance(front[key], str) for key in FRONTMATTER_KEYS):
        raise CorpusError(f"{path.name}: frontmatter must hold exactly {', '.join(FRONTMATTER_KEYS)}, as text")
    return Chunk(front["doc"], front["edition"], front["section"], front["title"], body.removesuffix("\n"), front["content_hash"])


def read_section_files(folder: Path) -> list[Chunk]:
    """Every section file in `folder`, in file-name (document) order."""
    return [read_section_file(path) for path in sorted(folder.glob("*.md"))]


def write_manifest(path: Path, manifest: Mapping[str, Any]) -> None:
    path.write_text(yaml.safe_dump(dict(manifest), sort_keys=False, allow_unicode=True, width=float("inf")), encoding="utf-8")


def read_manifest(path: Path) -> dict[str, Any] | None:
    """The manifest, or None when there is none yet (a first run). A manifest that is not a mapping with a document list raises CorpusError."""
    if not path.is_file():
        return None
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not isinstance(data.get("documents"), list):
        raise CorpusError(f"{path.name}: expected a mapping with a `documents` list")
    return data


def pack_vector(values: Sequence[float]) -> bytes:
    return struct.pack(f"<{len(values)}f", *values)


def unpack_vector(blob: bytes) -> tuple[float, ...]:
    return struct.unpack(f"<{len(blob) // struct.calcsize('<f')}f", blob)


def write_embeddings(path: Path, chunks: Sequence[Chunk], vectors: Sequence[Sequence[float]], meta: Mapping[str, str]) -> None:
    """Create `path` afresh with the chunks and their vectors. `meta` should hold corpus_key, model and chunker_version; the dimension is added."""
    if len(chunks) != len(vectors):
        raise CorpusError("every chunk needs exactly one vector")
    path.unlink(missing_ok=True)
    dimension = len(vectors[0]) if vectors else 0
    connection = sqlite3.connect(path)
    try:
        with connection:
            connection.execute("CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
            connection.execute(
                "CREATE TABLE chunks (id TEXT PRIMARY KEY, doc TEXT NOT NULL, edition TEXT NOT NULL, section TEXT NOT NULL, title TEXT NOT NULL, text TEXT NOT NULL, content_hash TEXT NOT NULL, vector BLOB NOT NULL)"
            )
            connection.executemany("INSERT INTO meta VALUES (?, ?)", [*sorted(meta.items()), ("dimension", str(dimension))])
            connection.executemany(
                "INSERT INTO chunks VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                [(chunk_id, c.doc, c.edition, c.section, c.title, c.text, c.content_hash, pack_vector(vector)) for chunk_id, c, vector in zip(chunk_ids(chunks), chunks, vectors, strict=True)],
            )
    finally:
        connection.close()


def read_embeddings(path: Path) -> tuple[dict[str, str], list[IndexedChunk]]:
    """(meta, chunks in stored order). Opened read-only."""
    connection = sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True)
    try:
        meta = dict(connection.execute("SELECT key, value FROM meta"))
        rows = connection.execute("SELECT id, doc, edition, section, title, text, content_hash, vector FROM chunks ORDER BY rowid").fetchall()
    finally:
        connection.close()
    return meta, [IndexedChunk(chunk_id, doc, edition, section, title, text, content_hash, unpack_vector(blob)) for chunk_id, doc, edition, section, title, text, content_hash, blob in rows]
