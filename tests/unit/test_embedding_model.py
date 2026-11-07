"""`make setup`'s embedding-model step (guidance/embedding_model.py), without the network.

The model is stored in the folder settings.yaml names (`make setup` puts it there; the build runner seeds it). With it stored, the
step must download nothing; with the folder empty and sockets blocked it must fail loudly, naming the model and saying to rerun online.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

from flysafelz.agent.settings import SETTINGS, embedding_folder
from flysafelz.guidance.embedding_model import ensure_embedding_model, main

ROOT = Path(__file__).resolve().parents[2]
MODEL = SETTINGS.embedding_model

# With the folder empty fastembed tries the network; pytest-socket blocks the lookup (and warns, which is not a network call). The step reports the failure.
BLOCKED_LOOKUP = pytest.mark.filterwarnings("ignore:A test tried to use socket")


def require_stored_model() -> Path:
    folder = embedding_folder()
    if not folder.is_dir() or not any(folder.iterdir()):
        pytest.fail(f"the embedding model is not stored in {folder}: run `make setup` first (the tests never download it)")
    return folder


def test_with_the_model_stored_nothing_is_downloaded(capsys):
    """Sockets are blocked, so any download attempt would fail the step: exit 0 and 'already stored' show there was none."""
    folder = require_stored_model()
    assert main() == 0
    output = capsys.readouterr()
    assert output.out == f"Embedding model {MODEL} already stored in {folder}.\n"
    assert output.err == ""


def test_the_model_is_loaded_from_the_stored_folder_and_reports_no_download():
    assert ensure_embedding_model(MODEL, require_stored_model()) is False


@BLOCKED_LOOKUP
def test_with_an_empty_folder_and_no_network_the_step_fails_naming_the_model_and_saying_to_rerun_online(tmp_path, capsys):
    empty = tmp_path / "fastembed"
    assert main(folder=empty) == 1
    output = capsys.readouterr()
    assert output.out == ""  # nothing is reported as stored
    assert f"Could not download the embedding model {MODEL} into {empty}" in output.err
    assert "Rerun `make setup` when you are online." in output.err
    assert ".." not in output.err


@BLOCKED_LOOKUP
def test_a_failed_download_raises_from_ensure_so_it_cannot_pass_quietly(tmp_path):
    with pytest.raises(Exception, match="socket"):
        ensure_embedding_model(MODEL, tmp_path / "fastembed")


def test_the_step_run_as_make_setup_runs_it_offline_and_succeeds_with_the_model_stored():
    """`python -m flysafelz.guidance.embedding_model` is the command in the Makefile. HF_HUB_OFFLINE makes fastembed refuse the network, so a missing model fails here, not downloads."""
    folder = require_stored_model()
    result = subprocess.run([sys.executable, "-m", "flysafelz.guidance.embedding_model"], cwd=ROOT, env={**os.environ, "HF_HUB_OFFLINE": "1"}, capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    assert result.stdout == f"Embedding model {MODEL} already stored in {folder}.\n"


def test_the_makefile_runs_uv_sync_and_then_the_step():
    lines = (ROOT / "Makefile").read_text(encoding="utf-8").splitlines()
    start = lines.index("setup:")
    assert lines[start + 1 : start + 3] == ["\tuv sync", "\tuv run python -m flysafelz.guidance.embedding_model"]


def test_a_model_not_stored_is_downloaded_once_into_the_folder_after_the_local_only_try(tmp_path, monkeypatch):
    """The download itself needs the network, so TextEmbedding is replaced by a recorder here; the order of the two tries is what is checked."""
    calls = []

    class Recorder:
        def __init__(self, model_name, **options):
            calls.append((model_name, options))
            if options.get("local_files_only"):
                raise ValueError("not stored")

    monkeypatch.setattr("flysafelz.guidance.embedding_model.TextEmbedding", Recorder)
    assert ensure_embedding_model(MODEL, tmp_path) is True
    assert calls == [(MODEL, {"cache_dir": str(tmp_path), "local_files_only": True}), (MODEL, {"cache_dir": str(tmp_path)})]
