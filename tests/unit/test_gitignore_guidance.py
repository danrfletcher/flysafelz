"""Git tracking for the guidance outputs: the committed vectors are not ignored, staging and other databases are."""

import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def ignored(path: str) -> int:
    """`git check-ignore -q` exit status: 0 ignored, 1 not ignored."""
    return subprocess.run(["git", "check-ignore", "-q", path], cwd=ROOT, check=False).returncode


def test_the_committed_embeddings_are_not_ignored():
    assert ignored("data/guidance/embeddings.sqlite") == 1


@pytest.mark.parametrize(
    "path", ["data/other.sqlite", "data/guidance/other.sqlite", "data/guidance/staging/x.md", "data/guidance/staging/embeddings.sqlite", "data/guidance/staging/cap1535/0000-1.md"]
)
def test_other_databases_and_staging_are_ignored(path):
    assert ignored(path) == 0


@pytest.mark.parametrize("path", ["data/guidance/manifest.yaml", "data/guidance/sources.yaml", "data/guidance/cap1535/0000-1.md"])
def test_the_manifest_register_and_section_files_are_tracked(path):
    assert ignored(path) == 1


def test_the_exception_follows_the_sqlite_rule_in_gitignore():
    lines = (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
    assert lines[lines.index("*.sqlite") + 1] == "!data/guidance/embeddings.sqlite"
    assert "data/guidance/staging/" in lines


def test_the_embeddings_are_tracked_once_vector_has_committed_them():
    tracked = subprocess.run(["git", "ls-files", "data/guidance/embeddings.sqlite"], cwd=ROOT, check=True, capture_output=True, text=True).stdout
    if not tracked.strip():
        pytest.skip("data/guidance/embeddings.sqlite is committed by Vector's post-merge `make guidance` run")
    assert tracked.strip() == "data/guidance/embeddings.sqlite"
