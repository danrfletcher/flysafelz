"""The guidance register (G16): every rule of the loader, and the shipped data/guidance/sources.yaml."""

from pathlib import Path

import pytest
import yaml

from flysafelz.guidance.sources import SourcesError, indexed, load_sources, url_is_allowed

ROOT = Path(__file__).resolve().parents[2]
SHIPPED = ROOT / "data" / "guidance" / "sources.yaml"

GOOD = {
    "id": "cap1535",
    "title": "CAP 1535",
    "url": "https://example.org/cap1535.pdf",
    "edition": "3",
    "licence": "OGL v3.0",
    "review_date": "2026-11-07",
    "indexed": True,
}
FIELDS = list(GOOD)


def register(tmp_path: Path, *entries: dict) -> Path:
    path = tmp_path / "sources.yaml"
    path.write_text(yaml.safe_dump({"documents": list(entries)}, sort_keys=False), encoding="utf-8")
    return path


def entry(**changes) -> dict:
    return {**GOOD, **changes}


def test_a_complete_entry_loads_with_every_field(tmp_path):
    (source,) = load_sources(register(tmp_path, GOOD))
    assert (source.id, source.title, source.url, source.edition, source.licence, source.indexed) == ("cap1535", "CAP 1535", GOOD["url"], "3", "OGL v3.0", True)
    assert source.review_date.isoformat() == "2026-11-07"


def test_a_yaml_date_loads_too(tmp_path):
    path = tmp_path / "sources.yaml"
    path.write_text("documents:\n  - {id: a, title: A, url: 'https://x.org/a.pdf', edition: '1', licence: l, review_date: 2026-11-07, indexed: true}\n", encoding="utf-8")
    assert load_sources(path)[0].review_date.isoformat() == "2026-11-07"


@pytest.mark.parametrize("field", FIELDS)
def test_each_missing_field_is_named_with_its_entry(tmp_path, field):
    broken = {key: value for key, value in GOOD.items() if key != field}
    with pytest.raises(SourcesError) as error:
        load_sources(register(tmp_path, broken))
    assert field in str(error.value) and "Field required" in str(error.value)
    assert ("entry 'cap1535'" if field != "id" else "entry #1") in str(error.value)


def test_an_unknown_field_is_refused(tmp_path):
    with pytest.raises(SourcesError, match="entry 'cap1535': .*extra"):
        load_sources(register(tmp_path, entry(extra="x")))


def test_a_duplicate_id_names_it_and_both_entries(tmp_path):
    with pytest.raises(SourcesError, match=r"duplicate id 'cap1535' \(entries #1 and #3\)"):
        load_sources(register(tmp_path, GOOD, entry(id="other"), entry(title="again")))


@pytest.mark.parametrize("url", ["http://example.org/a.pdf", "ftp://example.org/a.pdf", "file:///etc/passwd", "example.org/a.pdf", "https://", ""])
def test_a_url_that_is_not_https_is_refused_naming_the_entry(tmp_path, url):
    with pytest.raises(SourcesError, match="entry 'cap1535': url: .*https"):
        load_sources(register(tmp_path, entry(url=url)))


@pytest.mark.parametrize("url", ["http://127.0.0.1:8765/sample.pdf", "http://localhost/sample.pdf"])
def test_plain_http_is_allowed_to_this_machine_only(tmp_path, url):
    assert load_sources(register(tmp_path, entry(url=url)))[0].url == url
    assert url_is_allowed(url)
    assert not url_is_allowed(url.replace("localhost", "localhost.example.org").replace("127.0.0.1", "127.0.0.2"))


@pytest.mark.parametrize("value", ["07/11/2026", "2026-13-01", "2026-02-30", "soon", "2026-11-07T10:00:00", 20261107, "", None])
def test_a_malformed_date_is_refused_naming_the_entry(tmp_path, value):
    with pytest.raises(SourcesError, match="entry 'cap1535': review_date"):
        load_sources(register(tmp_path, entry(review_date=value)))


@pytest.mark.parametrize("value", ["yes", 1, "true", None])
def test_indexed_must_be_a_real_true_or_false(tmp_path, value):
    with pytest.raises(SourcesError, match="entry 'cap1535': indexed"):
        load_sources(register(tmp_path, entry(indexed=value)))


@pytest.mark.parametrize("bad", ["../etc", "a/b", "A", "-a", "a b", ""])
def test_an_id_that_could_be_a_path_is_refused(tmp_path, bad):
    with pytest.raises(SourcesError, match="id"):
        load_sources(register(tmp_path, entry(id=bad)))


@pytest.mark.parametrize(
    ("text", "message"),
    [("", "non-empty list"), ("documents: []\n", "non-empty list"), ("- a\n", "non-empty list"), ("documents: [1]\n", "entry #1: each entry must be a mapping"), ("a: [\n", "not valid YAML")],
)
def test_a_register_of_the_wrong_shape_is_refused(tmp_path, text, message):
    path = tmp_path / "sources.yaml"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(SourcesError, match=message):
        load_sources(path)


def test_a_register_that_cannot_be_read_is_refused(tmp_path):
    with pytest.raises(SourcesError, match="cannot read the register"):
        load_sources(tmp_path / "missing.yaml")


def test_an_entry_that_is_not_indexed_is_skipped(tmp_path):
    sources = load_sources(register(tmp_path, GOOD, entry(id="bhpa", indexed=False)))
    assert [s.id for s in sources] == ["cap1535", "bhpa"]
    assert [s.id for s in indexed(sources)] == ["cap1535"]


def test_the_shipped_register_loads_and_bhpa_is_link_only():
    sources = {s.id: s for s in load_sources(SHIPPED)}
    assert len(sources) == len(load_sources(SHIPPED))
    bhpa = [s for s in sources.values() if "bhpa" in s.id]
    assert len(bhpa) == 1 and bhpa[0].indexed is False
    link_only = {s.id for s in sources.values() if not s.indexed}
    assert link_only == {"bhpa-safety", "caa-rat-notam"}  # the RA(T) guidance is a web page with no PDF, so it is link-only too
    titles = " ".join(s.title for s in sources.values())
    for name in ("1535", "2562", "RA(T)", "BHPA"):
        assert name in titles
