"""F1 and F8 for the results file, the pricing and the eval commands (5a PR-10.F1).

F8  nothing is written outside `evals/results/` (and the run's own reader cache): the write calls of the new modules sit in `report.write_report` and nowhere else, and a
    whole run (v0 dev, then rag-eval) is watched at the file-system calls: every path opened for writing, created, renamed or deleted is under the two folders the test gave
    it, and none is under DATASET_SRC, the vault or the repo's own folders. An experiment name that could leave the folder is refused.
F1  the new modules import nothing from `src/flysafelz` that mutates state: the imports are a short list of read-only modules (contracts, the reader's contract, the guidance
    index and its hit rule, the agent's graph, state and tracing reads). Every scan has a planted violation proving it fires.
"""

import ast
import builtins
import io
import os
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

import pytest
from eval_runner_helpers import FAKE_ENV, FakeSleeper, RecordingTimeout
from test_eval_import_fence import VAULT, write_calls
from v0_helpers import FAKE_V0_ENV, CannedV0Model, V0Client

from evals.agent import eval_run, rag_eval, report
from evals.agent.pricing import load_prices
from flysafelz.guidance.hit4 import load_questions

REPO = Path(__file__).resolve().parents[2]
AGENT = REPO / "evals" / "agent"
NEW = ("report.py", "rag_eval.py", "eval_run.py", "pricing.py")
MAY_WRITE = {"report.py": {"write_report"}}
READ_ONLY_IMPORTS = {
    "flysafelz.contracts",
    "flysafelz.reader.contract",
    "flysafelz.reader.config",  # READER_* settings: read from the environment and the .env file, nothing written (eval_run resolves READER_MODEL as the run does)
    "flysafelz.guidance.hit4",
    "flysafelz.guidance.index",
    "flysafelz.guidance.embedder",
    "flysafelz.agent.config",
    "flysafelz.agent.graph",
    "flysafelz.agent.tracing",
}
STAMP = datetime(2026, 11, 9, 12, 0, 0, tzinfo=UTC)
PRICE_FILE = "models:\n  fake-agent-model:\n    input_per_m: 3.00\n    output_per_m: 15.00\n    url: https://provider.invalid/pricing\n    date: 2026-11-09\n"


# --- the scans ---------------------------------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("name", NEW)
def test_write_calls_of_the_new_modules_sit_only_in_write_report(name):
    stray = [(function, call) for function, call in write_calls((AGENT / name).read_text(encoding="utf-8")) if function not in MAY_WRITE.get(name, set())]
    assert stray == [], f"{name} writes outside {sorted(MAY_WRITE.get(name, set()))}"


def test_write_report_is_where_the_results_file_is_written():
    assert {function for function, _call in write_calls((AGENT / "report.py").read_text(encoding="utf-8"))} == {"write_report"}


def test_the_write_scan_fires_on_a_planted_write_in_another_function():
    planted = (AGENT / "report.py").read_text(encoding="utf-8") + "\n\ndef planted(p):\n    p.write_text('x')\n"
    assert ("planted", "write_text") in write_calls(planted)


@pytest.mark.parametrize("name", NEW)
def test_no_new_module_names_the_vault_or_a_notes_folder(name):
    tree = ast.parse((AGENT / name).read_text(encoding="utf-8"))
    assert [n.value for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, str) and VAULT.search(n.value)] == []


def flysafelz_imports(source: str) -> set[str]:
    found = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.ImportFrom) and node.module and node.module.split(".")[0] == "flysafelz":
            found.add(node.module)
        elif isinstance(node, ast.Import):
            found |= {a.name for a in node.names if a.name.split(".")[0] == "flysafelz"}
    return found


@pytest.mark.parametrize("name", NEW)
def test_the_new_modules_import_only_read_only_parts_of_the_product(name):
    assert flysafelz_imports((AGENT / name).read_text(encoding="utf-8")) - READ_ONLY_IMPORTS == set()


@pytest.mark.parametrize("planted", ["from flysafelz.capture import capture\n", "import flysafelz.clients.open_meteo\n", "from flysafelz.agent.run import main\n"])
def test_the_import_scan_fires_on_a_planted_import_of_a_mutating_module(planted):
    assert flysafelz_imports(planted) - READ_ONLY_IMPORTS


def test_nothing_in_the_new_modules_calls_the_network_or_a_source_client():
    for name in NEW:
        assert not any("clients" in module or "capture" in module for module in flysafelz_imports((AGENT / name).read_text(encoding="utf-8")))


# --- the folder ----------------------------------------------------------------------------------------------------------------------------------------


def test_the_results_folder_is_evals_results_in_the_repo():
    assert report.RESULTS_DIR == REPO / "evals" / "results"


@pytest.mark.parametrize("name", ["../escape", "..", "a/b", "/abs", "x y", "", "a..b/../c"])
def test_an_experiment_name_that_could_leave_the_folder_is_refused_and_nothing_is_written(tmp_path, name):
    folder = tmp_path / "results"
    with pytest.raises(report.ReportError):
        report.write_report("text\n", name, folder)
    assert not list(tmp_path.rglob("*.md")) and not (tmp_path / "escape.md").exists()


def test_the_file_is_the_experiment_name_in_the_folder_and_replaces_whole(tmp_path):
    folder = tmp_path / "results"
    path = report.write_report("first, long long long\n", "v1-dev-x", folder)
    again = report.write_report("second\n", "v1-dev-x", folder)
    assert path == again == folder / "v1-dev-x.md" and path.read_text(encoding="utf-8") == "second\n"
    assert [p.name for p in folder.iterdir()] == ["v1-dev-x.md"]


def test_a_failed_write_leaves_no_partial_file_and_no_temp_file(tmp_path, monkeypatch):
    folder = tmp_path / "results"
    report.write_report("keep me\n", "v1-dev-x", folder)

    def broken(source, target):
        raise OSError("disk full")

    monkeypatch.setattr(os, "replace", broken)
    with pytest.raises(OSError, match="disk full"):
        report.write_report("new\n", "v1-dev-x", folder)
    assert [p.name for p in folder.iterdir()] == ["v1-dev-x.md"] and (folder / "v1-dev-x.md").read_text(encoding="utf-8") == "keep me\n"


# --- a whole run, watched at the file system -------------------------------------------------------------------------------------------------------


class Writes:
    """Every path a call asks to change, recorded by patching the calls the standard library makes for a write."""

    def __init__(self, monkeypatch):
        self.paths: list[Path] = []
        real_open, real_os_open, real_connect = builtins.open, os.open, sqlite3.connect
        real_replace, real_rename, real_mkdir, real_unlink, real_remove = os.replace, os.rename, os.mkdir, os.unlink, os.remove

        def note(*paths):
            self.paths.extend(Path(os.fsdecode(p)).resolve() for p in paths if isinstance(p, str | bytes | os.PathLike))

        def spy_open(file, mode="r", *args, **kwargs):
            if set(mode) & set("wax+"):
                note(file)
            return real_open(file, mode, *args, **kwargs)

        def spy_os_open(path, flags, *args, **kwargs):
            if flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_APPEND | os.O_TRUNC):
                note(path)
            return real_os_open(path, flags, *args, **kwargs)

        def spy_connect(database, *args, **kwargs):
            note(database)
            return real_connect(database, *args, **kwargs)

        def two(real):
            def spy(source, target, *args, **kwargs):
                note(source, target)
                return real(source, target, *args, **kwargs)

            return spy

        def one(real):
            def spy(path, *args, **kwargs):
                note(path)
                return real(path, *args, **kwargs)

            return spy

        monkeypatch.setattr(builtins, "open", spy_open)
        monkeypatch.setattr("io.open", spy_open)
        monkeypatch.setattr(os, "open", spy_os_open)
        monkeypatch.setattr(sqlite3, "connect", spy_connect)
        monkeypatch.setattr(os, "replace", two(real_replace))
        monkeypatch.setattr(os, "rename", two(real_rename))
        for name, real in (("mkdir", real_mkdir), ("unlink", real_unlink), ("remove", real_remove)):
            monkeypatch.setattr(os, name, one(real))


def inside(path: Path, folder: Path) -> bool:
    return path == folder.resolve() or folder.resolve() in path.parents


def test_a_whole_v0_run_and_a_rag_eval_write_only_under_the_two_folders_they_were_given(tmp_path, monkeypatch):
    dataset_src, vault = tmp_path / "dataset-src", tmp_path / "vault" / "Notes"
    dataset_src.mkdir()
    vault.mkdir(parents=True)
    (dataset_src / "examples.yaml").write_text("untouched\n", encoding="utf-8")
    monkeypatch.setenv("DATASET_SRC", str(dataset_src))
    pricing = tmp_path / "pricing.yaml"
    pricing.write_text(PRICE_FILE, encoding="utf-8")
    results, cache = tmp_path / "results", tmp_path / "cache"
    before = {p: p.read_bytes() for p in dataset_src.rglob("*") if p.is_file()}
    writes = Writes(monkeypatch)

    code = eval_run.main(
        ["v0"],
        environ=FAKE_V0_ENV,
        stdout=io.StringIO(),
        stderr=io.StringIO(),
        results_dir=results,
        cache_dir=cache,
        pricing_path=pricing,
        moment=STAMP,
        model_factory=lambda env: CannedV0Model(),
        client_factory=lambda env: V0Client(),
        sleep=FakeSleeper(),
        timeout=RecordingTimeout(),
    )
    questions = load_questions(rag_eval.QUESTIONS_PATH)
    chunks = {q.question: [rag_eval.Chunk(q.expected[0].document, "1", q.expected[0].section)] for q in questions}
    rag = rag_eval.main([], search_factory=lambda data_dir: (lambda question: chunks[question], []), stdout=io.StringIO(), stderr=io.StringIO(), moment=STAMP, results_dir=results)

    assert (code, rag) == (0, 0)
    assert writes.paths, "the spy saw no write at all, so it proves nothing"
    assert [p for p in writes.paths if not (inside(p, results) or inside(p, cache))] == []
    assert [p for p in writes.paths if inside(p, dataset_src) or inside(p, vault.parent) or inside(p, REPO)] == []
    assert sorted(p.name for p in results.iterdir()) == ["rag-20261109-120000.md", "v0-dev-20261109-120000.md"]
    assert {p: p.read_bytes() for p in dataset_src.rglob("*") if p.is_file()} == before


def test_the_spy_fires_on_a_planted_write_outside_the_folders(tmp_path, monkeypatch):
    writes = Writes(monkeypatch)
    (tmp_path / "elsewhere.txt").write_text("x", encoding="utf-8")
    assert [p for p in writes.paths if not inside(p, tmp_path / "results")] == [(tmp_path / "elsewhere.txt").resolve()]


def test_a_run_that_stops_for_a_missing_price_writes_nothing_at_all(tmp_path, monkeypatch):
    empty = tmp_path / "pricing.yaml"
    empty.write_text("models: {}\n", encoding="utf-8")
    writes = Writes(monkeypatch)
    code = eval_run.main(
        ["v0"],
        environ=FAKE_V0_ENV,
        stdout=io.StringIO(),
        stderr=io.StringIO(),
        results_dir=tmp_path / "results",
        cache_dir=tmp_path / "cache",
        pricing_path=empty,
        model_factory=lambda env: CannedV0Model(),
        client_factory=lambda env: V0Client(),
    )
    assert code == 1 and writes.paths == [] and not (tmp_path / "results").exists() and not (tmp_path / "cache").exists()


def test_the_committed_price_list_is_a_valid_empty_list_and_the_fake_env_names_a_priced_model_only_in_tests():
    assert load_prices() == {} and FAKE_ENV["AGENT_MODEL"] == "fake-agent-model"
