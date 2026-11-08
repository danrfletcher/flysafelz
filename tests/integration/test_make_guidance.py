"""`make guidance` end to end (G17, E20, GP10): `run_ingest` on a temporary data folder with `httpx.MockTransport` and a fake embedder. No socket, no model, no key."""

import hashlib
import io
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

import httpx
import pytest
import yaml
from guidance_helpers import FakeEmbedder, build_pdf, document_pages

from flysafelz.guidance import corpus, ingest
from flysafelz.guidance.chunker import count_tokens
from flysafelz.guidance.ingest import PromotePlan, promote, run_ingest

ALPHA_URL = "https://docs.test/alpha.pdf"
BETA_URL = "https://docs.test/beta.pdf"
BHPA_URL = "https://docs.test/bhpa.pdf"
TOKENS_MIN, TOKENS_MAX = 20, 200  # a band sized for the test documents; the real one is read from settings.yaml
SECTIONS = [(str(n), f"Part {name}", 3) for n, name in enumerate(("one", "two", "three", "four", "five", "six"), start=1)]  # a title ending in a number reads as a contents line
PLAIN_QUESTION = "quokka marmot habitat"
OTHER_QUESTION = "wombat dingo habitat"


def pdf(tag: str, marker: str = "", extra: dict[int, list[str]] | None = None) -> bytes:
    return build_pdf(document_pages(SECTIONS, 10, marker, tag, extra))


PLACES = ("hill", "river", "lake", "wood", "moor", "cliff")  # each page names a different place, so the line is not a repeated page header


def mention(words: str, page: int = 1, times: int = 1) -> list[str]:
    """One line naming `words` (repeated `times`, so the chunk matches a question on them better than a chunk that names them once)."""
    return [f"{' '.join([words.capitalize()] * times)} are named near the {PLACES[page - 1]} in this paragraph."]


def strong_mention(words: str, page: int) -> list[str]:
    return mention(words, page, times=3)


# alpha section 1 holds the quokka words; four decoy sections of beta hold them too, so without them in section 1 the question is a miss at top 4
ALPHA_WITH_ANSWER = pdf("alpha", extra={1: strong_mention(PLAIN_QUESTION, 1)})
ALPHA_WITHOUT_ANSWER = pdf("alpha")
BETA_WITH_DECOYS = pdf("beta", extra={n: mention(PLAIN_QUESTION, n) for n in (2, 3, 4, 5)})
BETA_PLAIN = pdf("beta")


@dataclass
class Site:
    """The documents the mock server holds: url -> PDF bytes, an HTTP status, or an exception to raise. `requests` lists every URL asked for, in order."""

    responses: dict[str, object] = field(default_factory=dict)
    requests: list[str] = field(default_factory=list)

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(str(request.url))
        answer = self.responses[str(request.url)]
        if isinstance(answer, Exception):
            raise answer
        if isinstance(answer, int):
            return httpx.Response(answer)
        if isinstance(answer, httpx.Response):
            return answer
        return httpx.Response(200, content=answer, headers={"content-type": "application/pdf"})

    @property
    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self.handler)


@dataclass
class Run:
    code: int
    out: list[str]
    err: str
    embedder: FakeEmbedder


class Project:
    """A data folder, a register and a questions file in a temporary folder."""

    def __init__(self, root: Path, site: Site, capsys: pytest.CaptureFixture[str]) -> None:
        root.mkdir(parents=True, exist_ok=True)
        self.data = root / "data" / "guidance"
        self.register = root / "sources.yaml"
        self.questions = root / "questions.yaml"
        self.site = site
        self.capsys = capsys
        self.documents = [self.entry("alpha", ALPHA_URL), self.entry("beta", BETA_URL)]
        self.write_register()

    @staticmethod
    def entry(name: str, url: str, indexed: bool = True) -> dict:
        return {"id": name, "title": name.title(), "url": url, "edition": "1", "licence": "test", "review_date": "2026-11-07", "indexed": indexed}

    def write_register(self) -> None:
        self.register.write_text(yaml.safe_dump({"documents": self.documents}, sort_keys=False), encoding="utf-8")

    def ask(self, *questions: tuple[str, str, str]) -> None:
        """Write the labelled questions: (question, document, section)."""
        entries = [{"question": q, "expected": [{"document": d, "section": s}]} for q, d, s in questions]
        self.questions.write_text(yaml.safe_dump(entries, sort_keys=False), encoding="utf-8")

    def run(self, embedder: FakeEmbedder | None = None, **options) -> Run:
        embedder = embedder or FakeEmbedder()
        self.capsys.readouterr()
        code = run_ingest(self.data, self.site.transport, embedder, sources_path=self.register, questions_path=self.questions, tokens_min=TOKENS_MIN, tokens_max=TOKENS_MAX, **options)
        captured = self.capsys.readouterr()
        return Run(code, captured.out.splitlines(), captured.err, embedder)

    def manifest(self) -> dict:
        manifest = corpus.read_manifest(self.data / "manifest.yaml")
        assert manifest is not None
        return manifest

    def snapshot(self) -> dict[str, bytes]:
        """Every file under the data folder (staging aside) by relative path."""
        return {p.relative_to(self.data).as_posix(): p.read_bytes() for p in sorted(self.data.rglob("*")) if p.is_file() and "staging" not in p.relative_to(self.data).parts}


@pytest.fixture
def site():
    return Site({ALPHA_URL: ALPHA_WITH_ANSWER, BETA_URL: BETA_WITH_DECOYS, BHPA_URL: 500})


@pytest.fixture
def project(tmp_path, site, capsys):
    return Project(tmp_path, site, capsys)


def sections_in(path: Path) -> int:
    return len(list(path.glob("*.md")))


# --- first run ------------------------------------------------------------------------------------------------------------------------------


def test_the_first_run_promotes_everything_and_the_files_agree(project, site):
    project.documents.append(project.entry("bhpa", BHPA_URL, indexed=False))
    project.write_register()
    project.ask((PLAIN_QUESTION, "alpha", "1"))
    result = project.run()
    assert result.code == 0
    assert "bhpa: link-only, not downloaded" in result.out
    assert "alpha: added, 6 sections" in result.out and "beta: added, 6 sections" in result.out
    assert any(line.startswith("hit@4: staging 1/1, current none yet, counted as 0") for line in result.out)
    manifest = project.manifest()
    assert result.out[-1] == f"promoted: 2 documents, 12 sections, corpus key {manifest['corpus_key']}"
    assert BHPA_URL not in site.requests and sorted(site.requests) == [ALPHA_URL, BETA_URL]  # the link-only document is never asked for

    assert [d["id"] for d in manifest["documents"]] == ["alpha", "beta"]
    assert manifest["documents"][0]["sha256"] == hashlib.sha256(ALPHA_WITH_ANSWER).hexdigest()
    assert manifest["corpus_key"] == corpus.corpus_key(manifest, manifest["model"], manifest["chunker_version"])
    assert (manifest["model"], manifest["chunker_version"]) == (ingest.SETTINGS.embedding_model, ingest.CHUNKER_VERSION)

    meta, rows = corpus.read_embeddings(project.data / "embeddings.sqlite")
    assert meta == {"corpus_key": manifest["corpus_key"], "model": manifest["model"], "chunker_version": str(ingest.CHUNKER_VERSION), "dimension": "1024"}
    assert len(rows) == 12 and {r.doc for r in rows} == {"alpha", "beta"}
    assert all(len(r.vector) == 1024 for r in rows)
    first = rows[0]
    assert first.vector == pytest.approx(FakeEmbedder.vector(first.text), abs=1e-6)  # float32 round trip
    on_disk = corpus.read_section_files(project.data / "alpha")
    assert [(c.doc, c.edition, c.section) for c in on_disk] == [("alpha", "1", str(n)) for n in range(1, 7)]
    assert [c.text for c in on_disk] == [r.text for r in rows if r.doc == "alpha"]
    assert not (project.data / "staging").exists()
    assert sorted(p.name for p in project.data.iterdir()) == ["alpha", "beta", "embeddings.sqlite", "manifest.yaml"]


def test_the_stored_database_has_the_documented_schema(project):
    project.run()
    connection = sqlite3.connect(project.data / "embeddings.sqlite")
    try:
        columns = {table: [row[1] for row in connection.execute(f"PRAGMA table_info({table})")] for table in ("meta", "chunks")}
    finally:
        connection.close()
    assert columns == {"meta": ["key", "value"], "chunks": ["id", "doc", "edition", "section", "title", "text", "content_hash", "vector"]}


def test_chunks_stay_inside_the_band_and_carry_the_hash_of_their_text(project):
    project.run()
    _, rows = corpus.read_embeddings(project.data / "embeddings.sqlite")
    assert all(TOKENS_MIN <= count_tokens(r.text) <= TOKENS_MAX for r in rows)
    assert all(r.content_hash == hashlib.sha256(r.text.encode("utf-8")).hexdigest() for r in rows)
    assert len({r.id for r in rows}) == len(rows)


def test_the_same_bytes_in_two_empty_folders_give_the_same_hashes_and_corpus_key(tmp_path, site, capsys):
    first, second = Project(tmp_path / "one", site, capsys), Project(tmp_path / "two", site, capsys)
    for project in (first, second):
        project.data.parent.mkdir(parents=True)
        assert project.run().code == 0
    assert first.snapshot().keys() == second.snapshot().keys()
    assert first.manifest() == second.manifest()
    assert [(r.id, r.content_hash) for r in corpus.read_embeddings(first.data / "embeddings.sqlite")[1]] == [
        (r.id, r.content_hash) for r in corpus.read_embeddings(second.data / "embeddings.sqlite")[1]
    ]


def test_a_document_with_a_different_running_page_number_gives_the_same_chunks(project, site):
    project.run()
    key = project.manifest()["corpus_key"]
    renumbered = build_pdf([[(x, y, text.replace("of 4", "of 99")) for x, y, text in page] for page in document_pages(SECTIONS, 10, "", "alpha", {1: strong_mention(PLAIN_QUESTION, 1)})])
    site.responses[ALPHA_URL] = renumbered
    result = project.run()
    assert "alpha: updated, 6 sections" in result.out  # the bytes differ, so it is read again ...
    assert [c.content_hash for c in corpus.read_section_files(project.data / "alpha")] == [r.content_hash for r in corpus.read_embeddings(project.data / "embeddings.sqlite")[1] if r.doc == "alpha"]
    assert project.manifest()["corpus_key"] != key  # ... and the document's bytes are part of the key


# --- unchanged re-run -----------------------------------------------------------------------------------------------------------------------


def test_a_second_identical_run_changes_nothing_and_embeds_nothing(project):
    project.ask((PLAIN_QUESTION, "alpha", "1"))
    project.run()
    manifest, embeddings = (project.data / "manifest.yaml"), (project.data / "embeddings.sqlite")
    before = (manifest.read_bytes(), embeddings.read_bytes(), manifest.stat().st_mtime_ns, embeddings.stat().st_mtime_ns, project.snapshot())
    result = project.run()
    assert result.code == 0
    assert result.out[:2] == ["alpha: no change", "beta: no change"]
    assert "nothing to promote: every document is unchanged" in result.out
    assert result.embedder.calls == []
    assert (manifest.read_bytes(), embeddings.read_bytes(), manifest.stat().st_mtime_ns, embeddings.stat().st_mtime_ns, project.snapshot()) == before


def test_a_bumped_chunker_version_reads_every_document_again(project, monkeypatch):
    project.run()
    old_key = project.manifest()["corpus_key"]
    monkeypatch.setattr(ingest, "CHUNKER_VERSION", ingest.CHUNKER_VERSION + 1)
    result = project.run()
    assert "alpha: no change" not in result.out and "alpha: updated, 6 sections" in result.out
    assert project.manifest()["chunker_version"] == ingest.CHUNKER_VERSION and project.manifest()["corpus_key"] != old_key


def test_a_document_kept_through_a_failed_download_is_split_again_once_it_downloads_after_a_chunker_bump(project, site, monkeypatch):
    project.run()
    first_version = ingest.CHUNKER_VERSION
    monkeypatch.setattr(ingest, "CHUNKER_VERSION", first_version + 1)
    site.responses[BETA_URL] = 404
    bumped = project.run()
    assert "beta: download failed (404), kept current snapshot" in bumped.out and "alpha: updated, 6 sections" in bumped.out
    versions = {d["id"]: d["chunker_version"] for d in project.manifest()["documents"]}
    assert versions == {"alpha": first_version + 1, "beta": first_version}  # beta's sections were still split by the old rules, and the manifest says so
    site.responses[BETA_URL] = BETA_WITH_DECOYS
    recovered = project.run()
    assert "beta: no change" not in recovered.out and "beta: updated, 6 sections" in recovered.out and "alpha: no change" in recovered.out
    assert recovered.out[-1].startswith("promoted: 2 documents")
    assert {d["id"]: d["chunker_version"] for d in project.manifest()["documents"]} == {"alpha": first_version + 1, "beta": first_version + 1}
    assert project.run().out[-1] == "nothing to promote: every document is unchanged"


def test_a_manifest_from_before_documents_recorded_their_chunker_version_splits_each_document_again(project):
    project.run()
    manifest = project.manifest()
    for document in manifest["documents"]:
        del document["chunker_version"]
    corpus.write_manifest(project.data / "manifest.yaml", manifest)
    result = project.run()
    assert "alpha: updated, 6 sections" in result.out and "beta: updated, 6 sections" in result.out


def test_an_embeddings_file_with_another_key_is_rebuilt_even_when_no_document_changed(project):
    project.run()
    manifest = project.manifest()
    corpus.write_embeddings(project.data / "embeddings.sqlite", [], [], {"corpus_key": "stale", "model": "x", "chunker_version": "1"})
    result = project.run()
    assert result.out[:2] == ["alpha: no change", "beta: no change"]
    assert result.out[-1].startswith("promoted: 2 documents, 12 sections")
    assert corpus.read_embeddings(project.data / "embeddings.sqlite")[0]["corpus_key"] == manifest["corpus_key"]


def test_an_unreadable_embeddings_file_is_rebuilt(project):
    project.run()
    (project.data / "embeddings.sqlite").write_bytes(b"not a database")
    assert project.run().out[-1].startswith("promoted: 2 documents")
    assert corpus.read_embeddings(project.data / "embeddings.sqlite")[0]["corpus_key"] == project.manifest()["corpus_key"]


def test_an_unreadable_embeddings_file_counts_as_zero_at_the_hit4_gate_and_is_rebuilt(project):
    project.ask((PLAIN_QUESTION, "alpha", "1"))
    project.run()
    (project.data / "embeddings.sqlite").write_bytes(b"not a database")
    result = project.run()
    assert result.code == 0
    assert "hit@4: staging 1/1, current snapshot unreadable, counted as 0" in result.out
    assert result.out[-1].startswith("promoted: 2 documents")
    assert corpus.read_embeddings(project.data / "embeddings.sqlite")[0]["corpus_key"] == project.manifest()["corpus_key"]


def test_a_missing_section_folder_is_read_again(project):
    project.run()
    for path in (project.data / "beta").glob("*.md"):
        path.unlink()
    result = project.run()
    assert "beta: updated, 6 sections" in result.out and "alpha: no change" in result.out
    assert sections_in(project.data / "beta") == 6


# --- hit@4 gate -----------------------------------------------------------------------------------------------------------------------------


def start_with_a_miss(project, site) -> None:
    """A current snapshot in which the quokka question is a miss: section 1 of alpha lacks the words and four decoys in beta hold them."""
    site.responses[ALPHA_URL] = ALPHA_WITHOUT_ANSWER
    project.ask((PLAIN_QUESTION, "alpha", "1"))
    first = project.run()
    assert first.code == 0 and "hit@4: staging 0/1, current none yet, counted as 0" in first.out
    assert any(line.startswith("miss: " + PLAIN_QUESTION) for line in first.out)


def test_a_changed_document_with_better_hit4_is_promoted(project, site):
    start_with_a_miss(project, site)
    before = project.manifest()
    site.responses[ALPHA_URL] = ALPHA_WITH_ANSWER
    result = project.run()
    assert result.code == 0
    assert "alpha: updated, 6 sections" in result.out and "beta: no change" in result.out
    assert "hit@4: staging 1/1, current 0/1" in result.out
    assert project.manifest()["corpus_key"] != before["corpus_key"]
    assert project.manifest()["documents"][0]["sha256"] == hashlib.sha256(ALPHA_WITH_ANSWER).hexdigest()
    assert any("Quokka marmot habitat Quokka marmot habitat Quokka marmot habitat are named near the hill" in c.text for c in corpus.read_section_files(project.data / "alpha"))


def test_a_changed_document_with_equal_hit4_is_promoted(project, site):
    project.ask((PLAIN_QUESTION, "alpha", "1"))
    project.run()
    site.responses[ALPHA_URL] = pdf("alpha", marker="edit", extra={1: strong_mention(PLAIN_QUESTION, 1)})
    result = project.run()
    assert result.code == 0 and "hit@4: staging 1/1, current 1/1" in result.out
    assert result.out[-1].startswith("promoted: ")
    assert project.manifest()["documents"][0]["sha256"] == hashlib.sha256(site.responses[ALPHA_URL]).hexdigest()


def test_a_changed_document_with_worse_hit4_is_not_promoted_and_names_the_question(project, site):
    project.ask((PLAIN_QUESTION, "alpha", "1"), (OTHER_QUESTION, "beta", "6"))
    site.responses[BETA_URL] = pdf("beta", extra={6: strong_mention(OTHER_QUESTION, 6), **{n: mention(PLAIN_QUESTION, n) for n in (2, 3, 4, 5)}})
    first = project.run()
    assert "hit@4: staging 2/2, current none yet, counted as 0" in first.out
    before = project.snapshot()
    site.responses[ALPHA_URL] = ALPHA_WITHOUT_ANSWER
    result = project.run()
    assert result.code == 1
    assert "hit@4: staging 1/2, current 2/2" in result.out
    assert f"went from hit to miss: {PLAIN_QUESTION}" in result.out
    assert OTHER_QUESTION not in "\n".join(line for line in result.out if line.startswith("went from"))
    assert "not promoted" in result.err
    assert not any(line.startswith("promoted") for line in result.out)
    assert project.snapshot() == before  # manifest, embeddings and section files are exactly as they were
    assert (project.data / "staging" / "manifest.yaml").is_file()  # the attempt is kept for inspection, and ignored by git


def test_a_swap_that_keeps_the_count_equal_is_promoted_and_still_names_what_got_worse(project, site):
    project.ask((PLAIN_QUESTION, "alpha", "1"), (OTHER_QUESTION, "alpha", "2"))
    site.responses[BETA_URL] = pdf("beta", extra={n: mention(PLAIN_QUESTION, n) + mention(OTHER_QUESTION, n) for n in (2, 3, 4, 5)})  # four decoys for each question
    site.responses[ALPHA_URL] = pdf("alpha", extra={1: strong_mention(PLAIN_QUESTION, 1)})
    project.run()
    site.responses[ALPHA_URL] = pdf("alpha", extra={2: strong_mention(OTHER_QUESTION, 2)})
    result = project.run()
    assert "hit@4: staging 1/2, current 1/2" in result.out
    assert result.code == 0 and result.out[-1].startswith("promoted: ")
    assert f"went from hit to miss: {PLAIN_QUESTION}" in result.out


@pytest.mark.parametrize("content", [None, "", "questions: []\n", "# nothing yet\n"])
def test_with_no_labelled_questions_it_promotes_and_says_hit4_did_not_run(project, content):
    if content is not None:
        project.questions.write_text(content, encoding="utf-8")
    result = project.run()
    assert result.code == 0
    assert "hit@4 not run: no labels" in result.out
    assert result.out[-1].startswith("promoted: 2 documents")


def test_a_malformed_questions_file_stops_before_anything_is_fetched(project, site):
    project.questions.write_text("- question: no expected\n", encoding="utf-8")
    result = project.run()
    assert result.code == 1 and "question #1" in result.err
    assert site.requests == [] and not (project.data / "manifest.yaml").exists()


# --- failed documents -----------------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("answer", "reason"),
    [(404, "404"), (500, "500"), (503, "503"), (httpx.ReadTimeout("slow"), "timeout"), (httpx.ConnectError("refused"), "ConnectError")],
    ids=["404", "500", "503", "timeout", "connect"],
)
def test_a_failed_download_keeps_the_current_snapshot_and_the_rest_still_go_through(project, site, answer, reason):
    project.ask((PLAIN_QUESTION, "alpha", "1"))
    project.run()
    beta_files = {name: data for name, data in project.snapshot().items() if name.startswith("beta/")}
    beta_entry = project.manifest()["documents"][1]
    site.responses[BETA_URL] = answer
    site.responses[ALPHA_URL] = pdf("alpha", marker="edit", extra={1: strong_mention(PLAIN_QUESTION, 1)})
    result = project.run()
    assert result.code == 0
    assert f"beta: download failed ({reason}), kept current snapshot" in result.out
    assert "alpha: updated, 6 sections" in result.out and any("hit@4:" in line for line in result.out)
    assert {name: data for name, data in project.snapshot().items() if name.startswith("beta/")} == beta_files
    assert project.manifest()["documents"][1] == beta_entry
    _, rows = corpus.read_embeddings(project.data / "embeddings.sqlite")
    assert sum(1 for r in rows if r.doc == "beta") == 6  # its chunks are still searchable
    assert not (project.data / "staging").exists()


def test_a_failed_download_on_a_first_run_is_named_as_missing_and_the_rest_is_promoted(project, site):
    site.responses[BETA_URL] = 404
    result = project.run()
    assert result.code == 0
    assert "beta: download failed (404), no snapshot yet: missing from the index" in result.out
    assert "alpha: added, 6 sections" in result.out
    assert result.out[-2].startswith("promoted: 1 documents, 6 sections") and result.out[-1] == "missing from the index: beta"
    assert [d["id"] for d in project.manifest()["documents"]] == ["alpha"]
    assert not (project.data / "beta").exists()
    assert {r.doc for r in corpus.read_embeddings(project.data / "embeddings.sqlite")[1]} == {"alpha"}


def test_a_document_that_failed_earlier_comes_in_on_a_later_run(project, site):
    site.responses[BETA_URL] = 404
    project.run()
    site.responses[BETA_URL] = BETA_WITH_DECOYS
    result = project.run()
    assert "beta: added, 6 sections" in result.out and "alpha: no change" in result.out
    assert [d["id"] for d in project.manifest()["documents"]] == ["alpha", "beta"]


def test_when_every_document_fails_on_a_first_run_nothing_is_promoted(project, site):
    site.responses.update({ALPHA_URL: 404, BETA_URL: 500})
    result = project.run()
    assert result.code == 1 and "no document could be indexed" in result.err
    assert not (project.data / "manifest.yaml").exists() and not (project.data / "embeddings.sqlite").exists()


@pytest.mark.parametrize(
    ("response", "reason"),
    [
        (httpx.Response(200, content=ALPHA_WITH_ANSWER[:-300], headers={"content-length": str(len(ALPHA_WITH_ANSWER))}), "partial download"),
        (httpx.Response(200, content=ALPHA_WITH_ANSWER[:-300]), "incomplete PDF"),
        (httpx.Response(200, content=b"<html>Not found</html>"), "incomplete PDF"),
        (httpx.Response(200, content=b"not a pdf at all\n%%EOF\n"), "unreadable PDF"),
        (httpx.Response(200, content=b""), "incomplete PDF"),
    ],
    ids=["short-body", "truncated-pdf", "html-page", "garbage-with-eof", "empty"],
)
def test_a_partial_or_unreadable_download_is_a_failed_download_and_never_staged(project, site, response, reason):
    site.responses[ALPHA_URL] = ALPHA_WITHOUT_ANSWER
    project.run()
    before = project.snapshot()
    site.responses[ALPHA_URL] = response
    result = project.run()
    assert result.code == 0
    assert f"alpha: download failed ({reason}), kept current snapshot" in result.out
    assert "beta: no change" in result.out and "nothing to promote: every document is unchanged" in result.out
    assert project.snapshot() == before
    assert not (project.data / "staging").exists()


def test_a_partial_download_on_a_first_run_is_missing_from_the_index(project, site):
    site.responses[ALPHA_URL] = httpx.Response(200, content=ALPHA_WITH_ANSWER[:100], headers={"content-length": "9999"})
    result = project.run()
    assert "alpha: download failed (partial download), no snapshot yet: missing from the index" in result.out
    assert [d["id"] for d in project.manifest()["documents"]] == ["beta"]


def test_a_redirect_to_a_plain_http_address_is_a_failed_download(project, site):
    site.responses[ALPHA_URL] = httpx.Response(302, headers={"location": "http://elsewhere.test/alpha.pdf"})
    site.responses["http://elsewhere.test/alpha.pdf"] = ALPHA_WITH_ANSWER
    result = project.run()
    assert "alpha: download failed (redirected to a URL that is not https), no snapshot yet: missing from the index" in result.out


def test_a_document_larger_than_the_limit_is_a_failed_download(project, site, monkeypatch):
    monkeypatch.setattr(ingest, "DOWNLOAD_LIMIT_BYTES", 1000)
    result = project.run()
    assert "alpha: download failed (too large), no snapshot yet: missing from the index" in result.out and result.code == 1


# --- headings, stale staging, dropped documents, bad register ----------------------------------------------------------------------------------


def test_a_pdf_with_no_headings_stops_the_run_naming_the_document_and_promotes_nothing(project, site):
    prose = build_pdf([[(56, 700, "This page is only prose."), (56, 680, "Nothing here is numbered.")], [(56, 700, "More prose follows on a second page.")]])
    site.responses[BETA_URL] = prose
    result = project.run()
    assert result.code == 1
    assert "beta: no section headings found" in result.out
    assert not (project.data / "manifest.yaml").exists() and not (project.data / "embeddings.sqlite").exists() and not (project.data / "alpha").exists()
    assert result.embedder.calls == []


def test_a_caa_style_pdf_of_contents_links_chart_labels_and_a_footer_stops_the_run_and_promotes_nothing(project, site):
    """The lines the CAA's CAP 1535 and CAP 2562 PDFs gave the heading rule before it learned to refuse them (R6)."""
    contents = [(56, 700 - 20 * n, text) for n, text in enumerate(("Contents", "07 Pilot >", "07 Aircraft >", "13 UK VOLMET Frequencies >"))]
    chart = [(56, 700 - 20 * n, text) for n, text in enumerate(("Weather chart", "0 Dewpoint oC", "10 KT", "2 NM ATZ shown with", "aerodrome symbol", "1 MIN"))]
    footer = [(56, 700, "Paramotors fly below the cloud."), (56, 60, "CAP 2562"), (56, 40, "6 Paramotor Code")]
    footer_too = [(56, 700, "Paramotors fly below the cloud."), (56, 60, "CAP 2562"), (56, 40, "8 Paramotor Code")]
    site.responses[BETA_URL] = build_pdf([contents, chart, footer, footer_too])
    result = project.run()
    assert result.code == 1 and "beta: no section headings found" in result.out
    assert not (project.data / "manifest.yaml").exists() and not (project.data / "alpha").exists() and result.embedder.calls == []


def test_a_stale_staging_folder_is_cleared_and_never_promoted(project):
    stale = project.data / "staging"
    (stale / "alpha").mkdir(parents=True)
    (stale / "alpha" / "0000-9.md").write_text("stale", encoding="utf-8")
    (stale / "manifest.yaml").write_text("corpus_key: stale\ndocuments: []\n", encoding="utf-8")
    result = project.run()
    assert result.code == 0
    assert not stale.exists()
    assert project.manifest()["corpus_key"] != "stale" and "stale" not in "".join(c.text for c in corpus.read_section_files(project.data / "alpha"))


def test_a_stale_staging_folder_is_cleared_even_when_the_run_stops_early(project):
    (project.data / "staging").mkdir(parents=True)
    (project.data / "staging" / "leftover.md").write_text("x", encoding="utf-8")
    project.register.write_text("documents: []\n", encoding="utf-8")
    assert project.run().code == 1
    assert not (project.data / "staging").exists()


def test_a_document_no_longer_indexed_is_removed_from_the_index(project):
    project.run()
    project.documents[1]["indexed"] = False
    project.write_register()
    result = project.run()
    assert "beta: link-only, not downloaded" in result.out and "beta: no longer indexed, removed from the index" in result.out
    assert [d["id"] for d in project.manifest()["documents"]] == ["alpha"]
    assert not (project.data / "beta").exists()
    assert {r.doc for r in corpus.read_embeddings(project.data / "embeddings.sqlite")[1]} == {"alpha"}


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("documents:\n  - {id: a, title: A, url: 'http://x.test/a.pdf', edition: '1', licence: l, review_date: 2026-11-07, indexed: true}\n", "entry 'a': url"),
        ("documents: []\n", "non-empty list"),
    ],
)
def test_a_bad_register_stops_the_run_naming_the_entry(project, site, text, message):
    project.register.write_text(text, encoding="utf-8")
    result = project.run()
    assert result.code == 1 and message in result.err
    assert site.requests == [] and not (project.data / "manifest.yaml").exists()


# --- interrupted promote --------------------------------------------------------------------------------------------------------------------


class Crash(BaseException):
    """A kill: not an Exception, so promote cannot roll back (as when the process is stopped)."""


def failing_after(renames: int, error: BaseException):
    done: list[tuple[Path, Path]] = []

    def rename(source: Path, target: Path) -> None:
        if len(done) == renames:
            raise error
        source.rename(target)
        done.append((source, target))

    return rename


def change_alpha(site) -> None:
    site.responses[ALPHA_URL] = pdf("alpha", marker="edit", extra={1: strong_mention(PLAIN_QUESTION, 1)})


@pytest.mark.parametrize("renames_before_the_error", [1, 2, 3])
def test_an_error_during_promote_is_rolled_back_and_the_old_snapshot_is_whole(project, site, renames_before_the_error):
    project.ask((PLAIN_QUESTION, "alpha", "1"))
    project.run()
    before = project.snapshot()
    change_alpha(site)
    with pytest.raises(OSError, match="disk full"):
        project.run(rename=failing_after(renames_before_the_error, OSError("disk full")))
    assert project.snapshot() == before
    meta, _ = corpus.read_embeddings(project.data / "embeddings.sqlite")
    assert meta["corpus_key"] == project.manifest()["corpus_key"]
    assert project.run().out[-1].startswith("promoted: ")  # and the next run goes through


@pytest.mark.parametrize("renames_before_the_crash", [1, 2, 3])
def test_a_crash_during_promote_leaves_the_old_manifest_in_place_and_the_next_run_recovers(project, site, renames_before_the_crash):
    project.ask((PLAIN_QUESTION, "alpha", "1"))
    project.run()
    old_manifest, old_embeddings = (project.data / "manifest.yaml").read_bytes(), (project.data / "embeddings.sqlite").read_bytes()
    change_alpha(site)
    with pytest.raises(Crash):
        project.run(rename=failing_after(renames_before_the_crash, Crash()))
    manifest_after = (project.data / "manifest.yaml").read_bytes()
    embeddings_after = (project.data / "embeddings.sqlite").read_bytes()
    if renames_before_the_crash < 3:  # the section folder is renamed first and second, the embeddings third, the manifest last of all
        assert (manifest_after, embeddings_after) == (old_manifest, old_embeddings)
    else:  # the new embeddings are in and the manifest is still the old one: the two files name different keys, which is what a key check sees
        assert corpus.read_embeddings(project.data / "embeddings.sqlite")[0]["corpus_key"] != yaml.safe_load(manifest_after)["corpus_key"]
    assert yaml.safe_load(manifest_after)["corpus_key"] == yaml.safe_load(old_manifest)["corpus_key"]
    recovered = project.run()  # the next run sees the difference and finishes the job
    assert recovered.code == 0 and recovered.out[-1].startswith("promoted: ")
    assert project.manifest()["corpus_key"] == corpus.read_embeddings(project.data / "embeddings.sqlite")[0]["corpus_key"]
    assert project.manifest()["documents"][0]["sha256"] == hashlib.sha256(site.responses[ALPHA_URL]).hexdigest()
    assert sections_in(project.data / "alpha") == 6


def test_promote_moves_the_manifest_last(tmp_path):
    data, staging = tmp_path / "data", tmp_path / "data" / "staging"
    (staging / "alpha").mkdir(parents=True)
    for name in ("embeddings.sqlite", "manifest.yaml", "alpha/0000-1.md"):
        (staging / name).write_text("new", encoding="utf-8")
    moved: list[str] = []

    def rename(source: Path, target: Path) -> None:
        if source.parent == staging:
            moved.append(source.name)
        source.rename(target)

    promote(data, staging, PromotePlan(replaced=["alpha"], dropped=[]), rename)
    assert moved == ["alpha", "embeddings.sqlite", "manifest.yaml"]
    assert (data / "manifest.yaml").read_text(encoding="utf-8") == "new"


def test_promote_puts_back_what_it_moved_aside_when_it_fails(tmp_path):
    data, staging = tmp_path / "data", tmp_path / "data" / "staging"
    (data / "alpha").mkdir(parents=True)
    (data / "alpha" / "0000-1.md").write_text("old", encoding="utf-8")
    (data / "manifest.yaml").write_text("old", encoding="utf-8")
    (staging / "alpha").mkdir(parents=True)
    (staging / "alpha" / "0000-1.md").write_text("new", encoding="utf-8")
    (staging / "manifest.yaml").write_text("new", encoding="utf-8")
    (staging / "embeddings.sqlite").write_text("new", encoding="utf-8")
    with pytest.raises(OSError, match="boom"):
        promote(data, staging, PromotePlan(replaced=["alpha"], dropped=[]), failing_after(3, OSError("boom")))
    assert (data / "alpha" / "0000-1.md").read_text(encoding="utf-8") == "old"
    assert (data / "manifest.yaml").read_text(encoding="utf-8") == "old"
    assert not (data / "embeddings.sqlite").exists()


# --- the command line -----------------------------------------------------------------------------------------------------------------------


def test_main_reads_the_register_named_by_guidance_sources_and_nothing_else(tmp_path, monkeypatch):
    seen = {}

    def fake_run(data_dir, transport=None, embedder=None, **options):
        seen.update(data_dir=data_dir, embedder=embedder, **options)
        return 0

    monkeypatch.setattr(ingest, "run_ingest", fake_run)
    monkeypatch.setattr(ingest, "load_embedder", lambda name, folder: "the-embedder")
    monkeypatch.setenv("GUIDANCE_SOURCES", str(tmp_path / "test-register.yaml"))
    assert ingest.main(data_dir=tmp_path / "data") == 0
    assert seen["sources_path"] == tmp_path / "test-register.yaml" and seen["embedder"] == "the-embedder" and seen["data_dir"] == tmp_path / "data"
    assert seen["questions_path"].as_posix().endswith("evals/guidance/questions.yaml")
    monkeypatch.delenv("GUIDANCE_SOURCES")
    ingest.main(data_dir=tmp_path / "data")
    assert seen["sources_path"] is None  # run_ingest then uses data/guidance/sources.yaml


def test_main_reports_an_operating_system_error_instead_of_a_traceback(tmp_path, monkeypatch, capsys):
    def fake_run(*args, **options):
        raise OSError("read-only file system")

    monkeypatch.setattr(ingest, "run_ingest", fake_run)
    monkeypatch.setattr(ingest, "load_embedder", lambda name, folder: object())
    assert ingest.main(data_dir=tmp_path) == 1
    assert "make guidance stopped: read-only file system" in capsys.readouterr().err


def test_run_ingest_refuses_to_run_without_an_embedder(tmp_path):
    with pytest.raises(ValueError, match="needs an embedder"):
        run_ingest(tmp_path)


def test_the_harness_pdfs_are_what_the_chunker_reads(project):
    """The bytes the mock site serves are readable PDFs (a guard on the fixtures, not on the product)."""
    from flysafelz.guidance.chunker import read_pdf_pages

    assert len(read_pdf_pages(io.BytesIO(ALPHA_WITH_ANSWER).getvalue())) == 6
