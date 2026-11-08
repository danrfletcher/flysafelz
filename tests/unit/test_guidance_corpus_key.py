"""The corpus key (G17): it changes when any one input changes, and not when the documents are listed in another order."""

import copy

import pytest

from flysafelz.guidance.corpus import corpus_key

MODEL = "BAAI/bge-small-en-v1.5"
MANIFEST = {
    "documents": [
        {"id": "cap1535", "edition": "3", "sha256": "a" * 64, "title": "CAP 1535", "url": "https://example.org/1", "sections": 12},
        {"id": "cap2562", "edition": "1", "sha256": "b" * 64, "title": "CAP 2562", "url": "https://example.org/2", "sections": 7},
    ]
}
BASE = corpus_key(MANIFEST, MODEL, 1)


def changed(index: int, **changes) -> dict:
    manifest = copy.deepcopy(MANIFEST)
    manifest["documents"][index].update(changes)
    return manifest


def test_the_key_is_a_sha256_hex_digest_and_is_stable():
    assert len(BASE) == 64 and set(BASE) <= set("0123456789abcdef")
    assert corpus_key(copy.deepcopy(MANIFEST), MODEL, 1) == BASE


@pytest.mark.parametrize("index", [0, 1])
@pytest.mark.parametrize(("field", "value"), [("id", "other"), ("edition", "99"), ("sha256", "c" * 64)])
def test_the_key_changes_when_any_one_document_input_changes(index, field, value):
    assert corpus_key(changed(index, **{field: value}), MODEL, 1) != BASE


def test_the_key_changes_with_the_model_name_and_the_chunker_version():
    assert corpus_key(MANIFEST, "BAAI/bge-base-en-v1.5", 1) != BASE
    assert corpus_key(MANIFEST, MODEL, 2) != BASE


def test_the_key_does_not_change_with_the_order_the_documents_are_listed_in():
    reversed_manifest = {"documents": list(reversed(MANIFEST["documents"]))}
    assert corpus_key(reversed_manifest, MODEL, 1) == BASE


def test_the_key_ignores_fields_that_are_not_part_of_the_version_slug():
    assert corpus_key(changed(0, title="Renamed", url="https://example.org/moved", sections=99), MODEL, 1) == BASE


def test_a_document_added_or_removed_changes_the_key():
    fewer = {"documents": MANIFEST["documents"][:1]}
    more = {"documents": [*MANIFEST["documents"], {"id": "x", "edition": "1", "sha256": "d" * 64}]}
    assert len({BASE, corpus_key(fewer, MODEL, 1), corpus_key(more, MODEL, 1)}) == 3


def test_fields_cannot_be_shifted_between_each_other_to_give_the_same_key():
    first = {"documents": [{"id": "a", "edition": "bc", "sha256": "d"}]}
    second = {"documents": [{"id": "ab", "edition": "c", "sha256": "d"}]}
    assert corpus_key(first, MODEL, 1) != corpus_key(second, MODEL, 1)
