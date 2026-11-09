"""`make guidance`: fetch the documents in the register, split them by their own headings, embed them, and promote them only if search got no worse (G17).

    python -m flysafelz.guidance.ingest       what `make guidance` runs
    run_ingest(data_dir, transport, embedder) the whole run on a folder; returns the exit code (the tests call this with a mock transport and a fake embedder)
    main()                                    the command line: the register (GUIDANCE_SOURCES or data/guidance/sources.yaml), the stored embedding model, then run_ingest

One run, in order:
  1. Clear `staging/` (a stale one from a failed run is never promoted). Load the register and the labelled questions.
  2. For each indexed document: download it and take its SHA-256. The same hash, edition and chunker version as `manifest.yaml` records for that document prints "<id>: no change"
     (a document kept from an earlier run through a failed download keeps the chunker version it was split with, so it is split again once it downloads).
     Otherwise read the PDF, split it into chunks and keep them in memory. A download or PDF that fails never stops the run (the current snapshot is kept, or the
     document is named as missing on a first run). A PDF with no section headings does stop it.
  3. If nothing changed, stop. Otherwise write the changed documents' section files, the new manifest and the embeddings of every chunk into `staging/`.
  4. Run hit@4 on staging and on the current snapshot with the same questions, through index.py's search and eval.py's `hit4_of_index`: the code the app and
     `make eval-guidance` use, so the gate and the eval cannot disagree. Staging fewer hits than current: exit 1, naming each question that went from hit to miss.
     No questions: promote and say "hit@4 not run: no labels". No current snapshot (a first run): current counts as 0.
  5. Promote by rename, the manifest last, so a crash leaves the old manifest and key consistent with the old embeddings; an error rolls the renames back.
This is the only module in `guidance/` and `agent/` that imports httpx, and it runs only from the command: importing it does nothing.
"""

import hashlib
import os
import shutil
import sqlite3
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from flysafelz.agent.settings import SETTINGS, embedding_folder, repository_root
from flysafelz.guidance import corpus
from flysafelz.guidance.chunker import Chunk, NoHeadingsError, PdfUnreadable, chunk_pages, chunk_typeset, read_pdf_pages, read_pdf_runs
from flysafelz.guidance.corpus import EMBEDDINGS_NAME, MANIFEST_NAME, CorpusError
from flysafelz.guidance.embedder import Embedder, EmbeddingModelMissing, load_embedder
from flysafelz.guidance.eval import hit4_of_index
from flysafelz.guidance.hit4 import Question, QuestionsError, describe_misses, load_questions, regressions
from flysafelz.guidance.index import GuidanceIndex, load_index
from flysafelz.guidance.sources import Source, SourcesError, indexed, load_sources, url_is_allowed

# Bump this when the chunker's rules change (chunker.py's docstring lists them); the value lives in settings.yaml with the other knobs.
# It is part of the corpus key, so a bump rebuilds the committed embeddings.
CHUNKER_VERSION = SETTINGS.chunker_version

SOURCES_VARIABLE = "GUIDANCE_SOURCES"  # a different register, for a test run against a local server; read by main() only
STAGING_NAME = "staging"
PREVIOUS_NAME = "previous"  # inside staging: what a promote moved out of the way, so an error can put it back
QUESTIONS_PATH = Path("evals") / "guidance" / "questions.yaml"
DOWNLOAD_TIMEOUT_SECONDS = 30  # a plain operational limit for one document
DOWNLOAD_LIMIT_BYTES = 200_000_000  # a document larger than this is treated as a failed download
PDF_END_MARKER = b"%%EOF"
PDF_END_WINDOW_BYTES = 1024  # a complete PDF ends with the marker within this many bytes of its end
USER_AGENT = "FlySafeLZ guidance ingest"


class DownloadFailed(Exception):
    """A document could not be fetched whole or is not a PDF. `reason` is what the output shows in brackets."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


class IngestStop(Exception):
    """The run cannot go on and nothing is promoted. The message has already been printed."""


@dataclass(frozen=True)
class PromotePlan:
    replaced: list[str]  # documents whose section folder staging brings
    dropped: list[str]  # documents that are in the old manifest and no longer indexed


def fetch(client: httpx.Client, source: Source) -> bytes:
    """The document's bytes. Raises DownloadFailed for an error status, a timeout, a body shorter than its Content-Length, a PDF with no end, or a redirect off https."""
    try:
        with client.stream("GET", source.url) as response:
            if not response.is_success:
                raise DownloadFailed(str(response.status_code))
            if not url_is_allowed(str(response.url)):
                raise DownloadFailed("redirected to a URL that is not https")
            body = bytearray()
            for piece in response.iter_bytes():
                body += piece
                if len(body) > DOWNLOAD_LIMIT_BYTES:
                    raise DownloadFailed("too large")
            promised = response.headers.get("content-length", "")
            compressed = response.headers.get("content-encoding", "identity").lower() != "identity"  # Content-Length then counts the compressed bytes
            received = response.num_bytes_downloaded if compressed else len(body)
            if promised.isdigit() and int(promised) != received:
                raise DownloadFailed("partial download")
    except httpx.TimeoutException as error:
        raise DownloadFailed("timeout") from error
    except httpx.HTTPError as error:
        raise DownloadFailed(type(error).__name__) from error
    data = bytes(body)
    if PDF_END_MARKER not in data[-PDF_END_WINDOW_BYTES:]:
        raise DownloadFailed("incomplete PDF")
    return data


def _vectors(embedder: Embedder, texts: list[str]) -> list[tuple[float, ...]]:
    """One document vector per text, passed through float32 so it equals what is stored and read back."""
    return [corpus.unpack_vector(corpus.pack_vector([float(x) for x in vector])) for vector in embedder.embed(texts)]


def _snapshot_is_whole(data_dir: Path, entry: dict[str, Any]) -> bool:
    folder = data_dir / entry["id"]
    return folder.is_dir() and len(list(folder.glob("*.md"))) == entry["sections"]


def _index_in(folder: Path, embedder: Embedder, model: str) -> GuidanceIndex:
    """The search index over `folder`'s embeddings.sqlite and manifest.yaml: the same index and ranking the app searches with (index.py)."""
    return load_index(folder, embedder_factory=lambda: embedder, model=model, chunker_version=CHUNKER_VERSION, apply_register=False)


def promote(data_dir: Path, staging: Path, plan: PromotePlan, rename: Callable[[Path, Path], object] = os.replace) -> None:
    """Move staging into place by rename: changed documents' folders, then embeddings.sqlite, then manifest.yaml last.

    What a rename replaces is kept in staging/previous. A folder is moved there first and the new one moved in. A file is hard-linked there (copied where the
    file system has no links) and the new one renamed over it, so `embeddings.sqlite` and `manifest.yaml` are never missing, not even for an instant. If anything
    raises, the renames done so far are undone (with os.replace, not `rename`) and the error is raised again. A crash that skips that (the process killed)
    leaves the old manifest, which is written last, so the old manifest and its key still match the old embeddings unless the crash falls between the last two
    renames; the key then differs between the two files, which the next run (and the key check at load) sees, and the next run rebuilds.
    """
    previous = staging / PREVIOUS_NAME
    previous.mkdir(parents=True, exist_ok=True)
    undo: list[Callable[[], None]] = []

    def move_folder(name: str, replacing: bool) -> None:
        live = data_dir / name
        if live.exists():
            rename(live, previous / name)
            undo.append(lambda: os.replace(previous / name, live))
        if replacing:
            rename(staging / name, live)
            undo.append(lambda: os.replace(live, staging / name))

    def swap_file(name: str) -> None:
        live = data_dir / name
        existed = live.exists()
        if existed:
            try:
                os.link(live, previous / name)
            except OSError:
                shutil.copy2(live, previous / name)
        rename(staging / name, live)
        undo.append(lambda: os.replace(previous / name, live) if existed else live.unlink())

    try:
        for doc in plan.dropped:
            move_folder(doc, replacing=False)
        for doc in plan.replaced:
            move_folder(doc, replacing=True)
        swap_file(EMBEDDINGS_NAME)
        swap_file(MANIFEST_NAME)
    except Exception:
        for step in reversed(undo):
            step()
        raise


def _read_old(data_dir: Path) -> tuple[dict[str, Any] | None, dict[str, dict[str, Any]]]:
    manifest = corpus.read_manifest(data_dir / MANIFEST_NAME)
    return manifest, {entry["id"]: entry for entry in (manifest or {"documents": []})["documents"]}


def _gather(
    sources: list[Source], old: dict[str, dict[str, Any]], data_dir: Path, client: httpx.Client, tokens_min: int, tokens_max: int
) -> tuple[dict[str, dict[str, Any]], dict[str, list[Chunk]], list[str]]:
    """Steps 2: (the manifest entry of each document that will be in the corpus, the new chunks of those that changed, the documents missing). Prints a line for each."""
    entries: dict[str, dict[str, Any]] = {}
    fresh: dict[str, list[Chunk]] = {}
    missing: list[str] = []
    for source in sources:
        kept = old.get(source.id)
        has_snapshot = kept is not None and _snapshot_is_whole(data_dir, kept)
        try:
            data = fetch(client, source)
            sha256 = hashlib.sha256(data).hexdigest()
            if kept and has_snapshot and kept["sha256"] == sha256 and kept["edition"] == source.edition and kept.get("chunker_version") == CHUNKER_VERSION:
                print(f"{source.id}: no change")
                entries[source.id] = {**kept, "title": source.title, "url": source.url}
                continue
            try:
                pages = read_pdf_pages(data)
            except PdfUnreadable as error:
                raise DownloadFailed("unreadable PDF") from error
        except DownloadFailed as failure:
            if kept and has_snapshot:
                print(f"{source.id}: download failed ({failure.reason}), kept current snapshot")
                entries[source.id] = kept
            else:
                print(f"{source.id}: download failed ({failure.reason}), no snapshot yet: missing from the index")
                missing.append(source.id)
            continue
        try:
            try:
                chunks = chunk_pages(pages, source.id, source.edition, tokens_min, tokens_max)
            except NoHeadingsError:
                chunks = chunk_typeset(read_pdf_runs(data), source.id, source.edition, tokens_min, tokens_max)  # rule 7: unnumbered headings
        except (NoHeadingsError, PdfUnreadable) as error:
            message = str(error) if isinstance(error, NoHeadingsError) else "no section headings found"
            print(f"{source.id}: {message}")
            raise IngestStop(message) from error
        print(f"{source.id}: {'updated' if kept else 'added'}, {len(chunks)} sections")
        fresh[source.id] = chunks
        entries[source.id] = {
            "id": source.id,
            "title": source.title,
            "edition": source.edition,
            "url": source.url,
            "sha256": sha256,
            "chunker_version": CHUNKER_VERSION,
            "sections": len(chunks),
        }
    return entries, fresh, missing


def _needs_rebuild(data_dir: Path, manifest: dict[str, Any] | None, entries: dict[str, dict[str, Any]], fresh: dict[str, list[Chunk]], model: str) -> bool:
    """False only when nothing changed: same documents, same model and chunker, and the stored embeddings carry the manifest's key."""
    if manifest is None or fresh or {e["id"] for e in manifest["documents"]} != set(entries):
        return True
    if manifest.get("model") != model or manifest.get("chunker_version") != CHUNKER_VERSION:
        return True
    return _stored_key(data_dir / EMBEDDINGS_NAME) != manifest.get("corpus_key")


def _stored_key(path: Path) -> str | None:
    """The corpus key stored in an embeddings file, or None when there is no file or it cannot be read."""
    if not path.is_file():
        return None
    try:
        return corpus.read_embeddings(path)[0].get("corpus_key")
    except sqlite3.Error:
        return None


def run_ingest(
    data_dir: Path,
    transport: httpx.BaseTransport | None = None,
    embedder: Embedder | None = None,
    *,
    sources_path: Path | None = None,
    questions_path: Path | None = None,
    model: str | None = None,
    tokens_min: int | None = None,
    tokens_max: int | None = None,
    rename: Callable[[Path, Path], object] = os.replace,
) -> int:
    """The whole run on `data_dir` (data/guidance). Returns the exit code: 0 done or nothing to do, 1 stopped with nothing promoted.

    `sources_path` defaults to data_dir/sources.yaml; `questions_path` None means no labelled questions. `model`, `tokens_min` and `tokens_max` default to settings.yaml.
    """
    if embedder is None:
        raise ValueError("run_ingest needs an embedder")
    model = SETTINGS.embedding_model if model is None else model
    tokens_min = SETTINGS.chunk_tokens_min if tokens_min is None else tokens_min
    tokens_max = SETTINGS.chunk_tokens_max if tokens_max is None else tokens_max
    data_dir.mkdir(parents=True, exist_ok=True)
    staging = data_dir / STAGING_NAME
    shutil.rmtree(staging, ignore_errors=True)  # a stale staging from an earlier failed run is never promoted
    try:
        sources = load_sources(sources_path or data_dir / "sources.yaml")
        questions = load_questions(questions_path)
        manifest, old = _read_old(data_dir)
    except (SourcesError, QuestionsError, CorpusError) as error:
        print(str(error), file=sys.stderr)
        return 1
    for source in sources:
        if not source.indexed:
            print(f"{source.id}: link-only, not downloaded")
    with httpx.Client(transport=transport, timeout=DOWNLOAD_TIMEOUT_SECONDS, follow_redirects=True, headers={"User-Agent": USER_AGENT}) as client:
        try:
            entries, fresh, missing = _gather(indexed(sources), old, data_dir, client, tokens_min, tokens_max)
        except IngestStop:
            return 1
    dropped = sorted(set(old) - set(entries))  # in the old manifest, not in the new corpus: no longer indexed, or failed with no whole snapshot to keep
    for doc in dropped:
        if doc not in missing:
            print(f"{doc}: no longer indexed, removed from the index")
    if not entries:
        print("no document could be indexed: nothing to promote", file=sys.stderr)
        return 1
    if not _needs_rebuild(data_dir, manifest, entries, fresh, model):
        print("nothing to promote: every document is unchanged")
        _report_missing(missing)
        return 0

    documents = [entries[doc] for doc in sorted(entries)]
    chunks = [chunk for entry in documents for chunk in (fresh[entry["id"]] if entry["id"] in fresh else corpus.read_section_files(data_dir / entry["id"]))]
    key = corpus.corpus_key({"documents": documents}, model, CHUNKER_VERSION)
    new_manifest = {"corpus_key": key, "model": model, "chunker_version": CHUNKER_VERSION, "documents": documents}
    staging.mkdir()
    for doc, doc_chunks in fresh.items():
        corpus.write_section_files(staging / doc, doc_chunks)
    corpus.write_embeddings(
        staging / EMBEDDINGS_NAME, chunks, _vectors(embedder, [chunk.text for chunk in chunks]), {"corpus_key": key, "model": model, "chunker_version": str(CHUNKER_VERSION)}
    )
    corpus.write_manifest(staging / MANIFEST_NAME, new_manifest)

    if not _gate(data_dir, staging, questions, embedder, model):
        return 1
    promote(data_dir, staging, PromotePlan(replaced=sorted(fresh), dropped=dropped), rename)
    shutil.rmtree(staging, ignore_errors=True)
    print(f"promoted: {len(documents)} documents, {len(chunks)} sections, corpus key {key}")
    _report_missing(missing)
    return 0


def _report_missing(missing: list[str]) -> None:
    if missing:
        print(f"missing from the index: {', '.join(missing)}")


def _gate(data_dir: Path, staging: Path, questions: list[Question], embedder: Embedder, model: str) -> bool:
    """Step 4. True when staging may be promoted. Prints the hit@4 line, the misses, and any question that went from hit to miss."""
    if not questions:
        print("hit@4 not run: no labels")
        return True
    candidate = hit4_of_index(_index_in(staging, embedder, model), questions)
    current_path = data_dir / EMBEDDINGS_NAME
    current = None
    baseline = "current none yet, counted as 0"
    if current_path.is_file():
        current_index = _index_in(data_dir, embedder, model)
        if current_index.available:
            current = hit4_of_index(current_index, questions)
            baseline = f"current {current.hits}/{current.total}"
        else:  # the same file _needs_rebuild treats as "rebuild": it holds no usable snapshot, so it counts as 0
            baseline = "current snapshot unreadable, counted as 0"
    print(f"hit@4: staging {candidate.hits}/{candidate.total}, {baseline}")
    for line in describe_misses(candidate):
        print(line)
    worse = regressions(current, candidate) if current else []
    for question in worse:
        print(f"went from hit to miss: {question}")
    if current and candidate.hits < current.hits:
        print("not promoted: hit@4 on staging is lower than on the current snapshot", file=sys.stderr)
        return False
    return True


def main(data_dir: Path | None = None, folder: Path | None = None) -> int:
    """`make guidance`. The register is GUIDANCE_SOURCES when set, else data/guidance/sources.yaml. The embedding model must already be stored (`make setup`)."""
    root = repository_root()
    data_dir = root / "data" / "guidance" if data_dir is None else data_dir
    try:
        embedder = load_embedder(SETTINGS.embedding_model, embedding_folder() if folder is None else folder)
    except EmbeddingModelMissing as error:
        print(str(error), file=sys.stderr)
        return 1
    chosen = os.environ.get(SOURCES_VARIABLE)
    try:
        return run_ingest(data_dir, embedder=embedder, sources_path=Path(chosen) if chosen else None, questions_path=root / QUESTIONS_PATH)
    except OSError as error:
        print(f"make guidance stopped: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
