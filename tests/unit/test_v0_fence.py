"""The fence for the v0 baseline (5a PR-9.F1): F1/G14 no reader, no `assess`, no daylight code, no guard and no agent graph; eval never reads readings.yaml; F7 dev unless told.

Every scan has a planted violation proving it fires. The run tests use tests/unit/v0_helpers.py's fake chat model and recording client: nothing real is called.
"""

import ast
import io
import os
import shutil
from datetime import datetime
from pathlib import Path

import pytest
from eval_runner_helpers import FakeSleeper, RecordingTimeout
from v0_helpers import FAKE_V0_ENV, CannedV0Model, V0Client

from evals.agent import v0
from evals.agent.loader import load_examples
from flysafelz.agent.state import RunContext

REPO = Path(__file__).resolve().parents[2]
AGENT = REPO / "evals" / "agent"
V0_MODULES = sorted(AGENT.glob("v0*.py"))
FORBIDDEN_IMPORTS = (
    "flysafelz.assess",
    "flysafelz.daylight",
    "flysafelz.reader.read",
    "flysafelz.reader.cached_read",
    "flysafelz.agent.graph",
    "flysafelz.agent.guard",
    "flysafelz.agent.guard_flow",
    "flysafelz.agent.tools",
)
FORBIDDEN_NAMES = ("read_cached", "read_notam", "assess", "build_graph", "check_answer")


def forbidden(source: str) -> list[str]:
    """Findings: an import of the reader's read module, assess, daylight, the guard, the tools or the agent graph, and of a name that is one of them."""
    found = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            modules = [alias.name for alias in node.names]
            names: list[str] = []
        elif isinstance(node, ast.ImportFrom):
            modules, names = [node.module or ""], [alias.name for alias in node.names]
        else:
            continue
        for module in modules:
            found += [f"import {module}" for bad in FORBIDDEN_IMPORTS if module == bad or module.startswith(bad + ".")]
        if modules and modules[0] in ("flysafelz", "flysafelz.reader", "flysafelz.agent"):
            found += [f"from {modules[0]} import {name}" for name in names if f"{modules[0]}.{name}" in FORBIDDEN_IMPORTS]
        found += [f"from {modules[0]} import {name}" for name in names if name in FORBIDDEN_NAMES]
    return found


# --- F1/G14: the import scan --------------------------------------------------------------------------------------------------------


def test_the_scan_covers_all_three_v0_modules():
    assert [p.name for p in V0_MODULES] == ["v0.py", "v0_output.py", "v0_prompt.py"]


@pytest.mark.parametrize("path", V0_MODULES, ids=lambda p: p.name)
def test_no_v0_module_imports_the_reader_assess_daylight_the_guard_the_tools_or_the_agent_graph(path):
    assert forbidden(path.read_text(encoding="utf-8")) == []


@pytest.mark.parametrize(
    "planted",
    [
        "from flysafelz.assess import assess",
        "import flysafelz.daylight",
        "from flysafelz.daylight import civil_window",
        "from flysafelz.reader.read import read_notam",
        "from flysafelz.reader.cached_read import read_cached",
        "from flysafelz.reader import read",
        "from flysafelz.agent.graph import build_graph",
        "from flysafelz.agent import graph",
        "from flysafelz.agent.guard import check_answer",
        "from flysafelz.agent.tools.assess_tool import assess_tool",
    ],
)
def test_the_scan_fires_on_each_planted_import(planted):
    assert forbidden(planted + "\n"), planted


def test_the_scan_leaves_the_allowed_imports_alone():
    allowed = "from flysafelz.weather import build_rows\nfrom flysafelz.reader.prompt import ABBREVIATIONS, abbreviation_rows, data_tag\nfrom flysafelz.agent.config import agent_config\nfrom flysafelz.verdict_sheet import apply_drops\n"
    assert forbidden(allowed) == []


def test_v0_does_not_import_the_v1_runner_internals_or_the_agent_graph_through_the_runner():
    source = (AGENT / "v0.py").read_text(encoding="utf-8")
    assert "from evals.agent import runner" in source and "run_experiment" in source and "graph_factory" in source


# --- F1/G14: the run never reaches them ------------------------------------------------------------------------------------------------


def boom(*args, **kwargs):
    raise AssertionError("v0 must not reach this")


def one_rep_run(tmp_path, **options):
    model, client = CannedV0Model(), V0Client()
    code = v0.main(
        ["--experiment", "v0-dev-fence"],
        environ=FAKE_V0_ENV,
        stdout=io.StringIO(),
        stderr=(err := io.StringIO()),
        model_factory=lambda env: model,
        client_factory=lambda env: client,
        cache_dir=tmp_path / "cache",
        sleep=FakeSleeper(),
        timeout=RecordingTimeout(),
        reps=1,
        **options,
    )
    return code, err.getvalue(), model, client


def test_a_run_with_assess_and_the_reader_set_to_raise_still_passes(monkeypatch, tmp_path):
    monkeypatch.setattr("flysafelz.assess.assess", boom)
    monkeypatch.setattr("flysafelz.reader.read.read_notam", boom)
    monkeypatch.setattr("flysafelz.reader.cached_read.read_cached", boom)
    code, err, model, client = one_rep_run(tmp_path)
    assert (code, err) == (0, "")
    assert len(model.prompts) == 16 and len(client.runs) == 16


@pytest.mark.parametrize(
    "target",
    ["flysafelz.daylight.sunrise_sunset", "flysafelz.daylight.daylight_phase", "flysafelz.daylight.daylight_edges", "flysafelz.agent.graph.build_graph", "flysafelz.agent.graph.build_agent"],
)
def test_no_daylight_or_agent_graph_code_runs_during_a_run(monkeypatch, tmp_path, target):
    monkeypatch.setattr(target, boom)  # the name must exist: no raising=False
    code, err, _model, _client = one_rep_run(tmp_path)
    assert (code, err) == (0, "")


# --- eval never reads readings.yaml; v0 never creates the cache --------------------------------------------------------------------------


@pytest.mark.skipif(os.geteuid() == 0, reason="root can read a file whatever its mode")
def test_readings_yaml_unreadable_for_a_v0_run_and_the_run_still_passes(tmp_path):
    record = next(r for r in load_examples("dev") if r["id"].startswith("dev_01"))
    case = record["fixtures"]["case"]
    shutil.copytree(REPO / "fixtures" / case, tmp_path / "fixtures" / case)
    readings = tmp_path / "fixtures" / case / "readings.yaml"
    assert readings.exists()
    readings.chmod(0)
    try:
        with pytest.raises(PermissionError):
            readings.read_bytes()  # the guard is real: this file cannot be read
        model = CannedV0Model()
        graph = v0.V0Graph(model, [record], root=tmp_path)
        context = RunContext(now=datetime.fromisoformat(record["inputs"]["now"]), case=case, drop=())
        state = graph.invoke({}, {"metadata": {"example_id": record["id"]}}, context=context)
        assert state["answer"] and len(model.prompts) == 1
    finally:
        readings.chmod(0o644)


def test_v0s_own_graph_calls_open_no_readings_yaml(monkeypatch, tmp_path):
    """Only v0's calls are watched. The shared loader's staleness check (an earlier slice) hashes every file of a case, readings.yaml among them, before the run starts: that is not v0 reading it."""
    opened, inside = [], []
    real_invoke = v0.V0Graph.invoke
    real_open, real_read_bytes, real_read_text = io.open, Path.read_bytes, Path.read_text

    def note(path):
        if inside and Path(str(path)).name == "readings.yaml":
            opened.append(str(path))

    def invoke(self, *args, **kwargs):
        inside.append(True)
        try:
            return real_invoke(self, *args, **kwargs)
        finally:
            inside.pop()

    monkeypatch.setattr(v0.V0Graph, "invoke", invoke)
    monkeypatch.setattr("builtins.open", lambda file, *a, **k: (note(file), real_open(file, *a, **k))[1])
    monkeypatch.setattr(Path, "read_bytes", lambda self: (note(self), real_read_bytes(self))[1])
    monkeypatch.setattr(Path, "read_text", lambda self, *a, **k: (note(self), real_read_text(self, *a, **k))[1])
    code, err, model, _client = one_rep_run(tmp_path)
    assert (code, err) == (0, "")
    assert len(model.prompts) == 16 and opened == []


def test_the_watcher_would_see_a_read_of_readings_yaml(monkeypatch, tmp_path):
    seen, real_read_bytes = [], Path.read_bytes
    monkeypatch.setattr(Path, "read_bytes", lambda self: (seen.append(self.name), real_read_bytes(self))[1])
    from flysafelz.sources import load_case_sources

    load_case_sources("blue_mill_2026-09-29")  # with_readings defaults to True: the proof the guard above can fail
    assert "readings.yaml" in seen
    seen.clear()
    load_case_sources("blue_mill_2026-09-29", with_readings=False)
    assert "readings.yaml" not in seen


def test_a_v0_run_never_creates_the_eval_cache_directory(tmp_path):
    cache = tmp_path / "cache"
    code, _err, _model, _client = one_rep_run(tmp_path)
    assert code == 0 and not cache.exists()
    assert not (REPO / "data" / "eval-cache" / "v0-dev-fence.sqlite").exists()


# --- F7/G17: dev unless told ---------------------------------------------------------------------------------------------------------


def test_the_default_run_opens_the_dev_dataset_only(tmp_path):
    _code, _err, _model, client = one_rep_run(tmp_path)
    assert list(client.datasets) == ["flysafelz-dev"]
    assert not any("held" in str(call) for call in client.calls)


def test_held_out_needs_the_explicit_flag():
    assert v0._parser().parse_args([]).split == "dev"
    assert v0._parser().parse_args(["--split", "held_out"]).split == "held_out"


def test_no_key_is_read_from_the_environment_by_the_tests_or_the_module():
    source = (AGENT / "v0.py").read_text(encoding="utf-8")
    assert "dotenv" not in source and "load_dotenv" not in source
    assert "LANGSMITH_API_KEY" not in os.environ and "AGENT_API_KEY" not in os.environ  # conftest cleared them: a test that needs them passes fakes in
