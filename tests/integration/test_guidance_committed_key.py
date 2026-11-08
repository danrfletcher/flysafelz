"""E19, the committed snapshot: the key in the committed `embeddings.sqlite` is the one the committed `manifest.yaml`, the model and the chunker version give.

Before the first `make guidance` run (GP10) there is no snapshot. The key test is then a strict xfail, so it cannot pass silently; the commit that adds the
snapshot turns it into an ordinary test (a strict xfail that passes is a failure, so the marker cannot be forgotten).
"""

import subprocess
from pathlib import Path

import pytest

from flysafelz.agent.settings import SETTINGS
from flysafelz.guidance import corpus
from flysafelz.guidance.index import computed_key

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "guidance"
SNAPSHOT = DATA / corpus.EMBEDDINGS_NAME
RELATIVE = SNAPSHOT.relative_to(ROOT).as_posix()


@pytest.mark.xfail(not SNAPSHOT.is_file(), strict=True, reason="no snapshot committed (GP10 run pending)")
def test_the_committed_embeddings_key_equals_the_key_computed_from_the_committed_manifest():
    meta, chunks = corpus.read_embeddings(SNAPSHOT)
    assert chunks, "the committed snapshot holds no chunks"
    assert meta["corpus_key"] == computed_key(DATA, SETTINGS.embedding_model, SETTINGS.chunker_version)
    assert meta["model"] == SETTINGS.embedding_model and meta["chunker_version"] == str(SETTINGS.chunker_version)


def test_git_does_not_ignore_the_embeddings_file():
    """`.gitignore` has `*.sqlite`; the negation for this one file is PR-7's. If it is missing, this fails (exit 0 from check-ignore means ignored)."""
    done = subprocess.run(["git", "check-ignore", "--no-index", "-q", RELATIVE], cwd=ROOT, check=False)
    assert done.returncode == 1, f"git ignores {RELATIVE}: .gitignore needs `!{RELATIVE}` after `*.sqlite`"


def test_other_sqlite_files_are_still_ignored():
    done = subprocess.run(["git", "check-ignore", "--no-index", "-q", "data/readings.sqlite"], cwd=ROOT, check=False)
    assert done.returncode == 0
