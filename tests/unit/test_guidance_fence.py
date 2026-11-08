"""Fences for the guidance ingest: F5 (no download at runtime), F6 (the per-package import scan), F7 (no literal band or top_k)."""

import ast
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml
from fence_scan import tree_import_violations

from flysafelz.agent.settings import SETTINGS
from flysafelz.guidance import ingest
from flysafelz.guidance.ingest import EmbeddingModelMissing, load_embedder, main

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src" / "flysafelz"
SETTINGS_FILE = SRC / "agent" / "settings.yaml"
DOWNLOADERS = ("httpx", "urllib.request")
FETCH_ONLY_IN = "guidance/ingest.py"
BAND_NAMES = ("top_k", "tokens_min", "tokens_max", "token_band", "band")


def imported_modules(tree: ast.AST) -> list[str]:
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found += [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            found += [module, *(f"{module}.{a.name}" for a in node.names)]
    return found


def downloaders_in(source: str) -> list[str]:
    return [m for m in imported_modules(ast.parse(source)) if any(m == d or m.startswith(d + ".") for d in DOWNLOADERS)]


def files_that_download(src: Path) -> list[str]:
    hits = []
    for folder in ("guidance", "agent"):
        for path in sorted((src / folder).rglob("*.py")):
            if downloaders_in(path.read_text(encoding="utf-8")):
                hits.append(path.relative_to(src).as_posix())
    return hits


def test_only_ingest_imports_httpx_within_guidance_and_agent():
    assert files_that_download(SRC) == [FETCH_ONLY_IN]


@pytest.mark.parametrize("text", ["import httpx\n", "from httpx import Client\n", "import urllib.request\n", "from urllib import request\n", "from urllib.request import urlopen\n"])
def test_the_download_scan_catches_a_planted_import(text):
    # `from urllib import request` names urllib.request as a member, which the scan reads as the module it brings in
    assert downloaders_in(text)


@pytest.mark.parametrize("text", ["import urllib.parse\n", "from urllib.parse import urlsplit\n", "import json\n"])
def test_the_download_scan_lets_the_harmless_imports_through(text):
    assert downloaders_in(text) == []


def test_a_planted_download_in_another_guidance_or_agent_module_is_caught(tmp_path):
    for where in ("guidance/chunker.py", "agent/graph.py"):
        package = tmp_path / where.split("/")[0]
        package.mkdir(parents=True, exist_ok=True)
        (package / Path(where).name).write_text("import httpx\n", encoding="utf-8")
        assert files_that_download(tmp_path) == [where]
        (package / Path(where).name).unlink()


REFUSE_NETWORK = (
    "import socket\ndef refuse(*a, **k):\n    raise AssertionError('network call at import')\nsocket.socket.connect = refuse\nsocket.getaddrinfo = refuse\nsocket.create_connection = refuse\n"
)


def files_written_by(tmp_path: Path, name: str, statement: str) -> list[str]:
    """Run `statement` in a fresh interpreter with every socket call refused and HOME, TMPDIR and the working folder inside a new folder; the files it left behind."""
    root = tmp_path / name
    for part in ("home", "tmp", "cwd"):
        (root / part).mkdir(parents=True)
    environment = {**os.environ, "HOME": str(root / "home"), "TMPDIR": str(root / "tmp"), "PYTHONDONTWRITEBYTECODE": "1"}
    done = subprocess.run([sys.executable, "-c", REFUSE_NETWORK + statement], cwd=root / "cwd", env=environment, check=False, capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
    return sorted(p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file())


def test_importing_ingest_makes_no_network_call_and_writes_no_file(tmp_path):
    """Importing fastembed writes files of its own (onnxruntime's device-id database and session file, on macOS), so those are the baseline: importing ingest adds none."""
    status = ["git", "status", "--porcelain", "--ignored"]
    before = subprocess.run(status, cwd=ROOT, check=True, capture_output=True, text=True).stdout
    baseline = files_written_by(tmp_path, "baseline", "import fastembed\n")
    with_ingest = files_written_by(tmp_path, "ingest", "import flysafelz.guidance.ingest\n")
    assert with_ingest == baseline
    assert subprocess.run(status, cwd=ROOT, check=True, capture_output=True, text=True).stdout == before


def test_a_missing_embedding_model_is_an_error_not_a_download(tmp_path, capsys):
    empty = tmp_path / "model-cache"
    empty.mkdir()
    with pytest.raises(EmbeddingModelMissing, match="embedding model missing: run make setup"):
        load_embedder(SETTINGS.embedding_model, empty)
    assert list(empty.iterdir()) == []  # nothing was fetched into it
    assert main(data_dir=tmp_path / "data", folder=empty) == 1
    assert "embedding model missing: run make setup" in capsys.readouterr().err
    assert not (tmp_path / "data").exists()  # it stopped before touching the data folder


def test_the_model_loader_never_asks_fastembed_to_download():
    source = (SRC / "guidance" / "ingest.py").read_text(encoding="utf-8")
    calls = [n for n in ast.walk(ast.parse(source)) if isinstance(n, ast.Call) and getattr(n.func, "id", "") == "TextEmbedding"]
    assert len(calls) == 1
    assert {k.arg: ast.literal_eval(k.value) for k in calls[0].keywords if k.arg == "local_files_only"} == {"local_files_only": True}


def test_f6_the_import_scan_still_passes_on_the_real_tree():
    assert tree_import_violations(SRC) == {}


def test_f6_a_planted_fastembed_import_in_assess_is_caught(tmp_path):
    copy = tmp_path / "flysafelz"
    for path in SRC.rglob("*.py"):
        target = copy / path.relative_to(SRC)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
    assert tree_import_violations(copy) == {}
    with (copy / "assess.py").open("a", encoding="utf-8") as handle:
        handle.write("\nimport fastembed\n")
    assert tree_import_violations(copy) == {"flysafelz/assess.py": ["fastembed"]}


# --- F7 -----------------------------------------------------------------------------------------------------------------------------------


def settings_values() -> dict[str, int]:
    data = yaml.safe_load(SETTINGS_FILE.read_text(encoding="utf-8"))
    return {key: data[key] for key in ("chunk_tokens_min", "chunk_tokens_max", "top_k")}


def band_literals(source: str, values: dict[str, int]) -> list[str]:
    """Where the code writes a settings value as a number: the token band anywhere, `top_k` wherever a band-like name is given a number."""
    hits = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Constant) and type(node.value) is int and node.value in (values["chunk_tokens_min"], values["chunk_tokens_max"]):
            hits.append(f"{node.value} at line {node.lineno}")
    for node in ast.walk(ast.parse(source)):
        targets: list[tuple[str, ast.AST | None]] = []
        if isinstance(node, ast.Assign):
            targets = [(t.id, node.value) for t in node.targets if isinstance(t, ast.Name)]
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            targets = [(node.target.id, node.value)]
        elif isinstance(node, ast.keyword) and node.arg:
            targets = [(node.arg, node.value)]
        elif isinstance(node, ast.arguments):
            names = [a.arg for a in [*node.posonlyargs, *node.args, *node.kwonlyargs]]
            defaults = [*node.defaults, *node.kw_defaults]
            targets = [(name, default) for name, default in zip(names[len(names) - len(defaults) :], defaults, strict=False)]
        for name, value in targets:
            if any(word in name.lower() for word in BAND_NAMES) and isinstance(value, ast.Constant) and type(value.value) is int:
                hits.append(f"{name}={value.value} at line {value.lineno}")
    return hits


def test_the_settings_file_holds_the_band_and_top_k():
    assert settings_values() == {"chunk_tokens_min": SETTINGS.chunk_tokens_min, "chunk_tokens_max": SETTINGS.chunk_tokens_max, "top_k": 4}
    assert SETTINGS.chunk_tokens_min < SETTINGS.chunk_tokens_max


@pytest.mark.parametrize("path", sorted((SRC / "guidance").glob("*.py")), ids=lambda p: p.name)
def test_f7_no_literal_token_band_or_top_k_in_guidance(path):
    assert band_literals(path.read_text(encoding="utf-8"), settings_values()) == []


@pytest.mark.parametrize(
    "planted",
    ["TOKENS = 300\n", "limit = 800\n", "def f(x, top_k=4):\n    pass\n", "search(q, top_k=4)\n", "tokens_max: int = 500\n", "def f(tokens_min=100):\n    pass\n"],
)
def test_f7_the_scan_catches_a_planted_literal(planted):
    assert band_literals(planted, settings_values())


def test_f7_the_ingest_reads_the_band_and_top_k_from_the_settings():
    source = (SRC / "guidance" / "ingest.py").read_text(encoding="utf-8")
    for name in ("SETTINGS.chunk_tokens_min", "SETTINGS.chunk_tokens_max", "SETTINGS.top_k"):
        assert name in source
    assert ingest.SETTINGS is SETTINGS and ingest.CHUNKER_VERSION == SETTINGS.chunker_version
