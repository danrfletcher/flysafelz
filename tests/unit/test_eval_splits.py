"""The split rule (5a PR-3.F1, G17): dev by default everywhere, held_out only by name, and fixed dataset and project names.

No environment variable chooses a dataset or a project: LANGSMITH_PROJECT (and the legacy name) is set to the other split's project in the tests below,
and every answer stays the same. Synthetic data for the loads; nothing is sent.
"""

import ast
import inspect
import re
from pathlib import Path

import pytest
from eval_loader_factories import write_dataset

from evals.agent import loader
from evals.agent.import_dataset import DatasetError
from evals.agent.loader import SPLITS, SplitTarget, experiment_target, load_dataset, load_examples, upload_dataset
from evals.agent.recording_client import RecordingClient


def test_the_split_table():
    assert SPLITS == {
        "dev": SplitTarget(dataset="flysafelz-dev", project="flysafelz-dev"),
        "held_out": SplitTarget(dataset="flysafelz-held-out", project="flysafelz-held-out"),
    }


@pytest.mark.parametrize("function", [load_examples, load_dataset, upload_dataset, experiment_target], ids=lambda f: f.__name__)
def test_the_default_split_is_dev_in_every_entry_point(function):
    parameters = list(inspect.signature(function).parameters.values())
    assert parameters[0].name == "split" and parameters[0].default == "dev"


def test_experiment_target_without_an_argument_is_dev():
    assert experiment_target() == SplitTarget("flysafelz-dev", "flysafelz-dev")
    assert experiment_target("held_out") == SplitTarget("flysafelz-held-out", "flysafelz-held-out")


@pytest.mark.parametrize("name", ["held-out", "heldout", "HELD_OUT", "held_out ", "", "test", "all", "dev,held_out", None])
def test_held_out_is_reached_only_by_the_literal_name(name):
    with pytest.raises(DatasetError, match="unknown split"):
        experiment_target(name)
    with pytest.raises(DatasetError, match="unknown split"):
        load_examples(name)
    client = RecordingClient()
    with pytest.raises(DatasetError, match="unknown split"):
        upload_dataset(name, client)
    assert client.calls == []


def test_a_default_load_and_a_default_upload_never_return_or_write_a_held_out_example(tmp_path):
    built = write_dataset(tmp_path)
    records = load_examples(data_dir=built.data_dir, root=built.root)
    assert records and {r["metadata"]["split"] for r in records} == {"dev"}
    client = RecordingClient()
    upload_dataset(client=client, data_dir=built.data_dir, root=built.root)
    assert list(client.datasets) == ["flysafelz-dev"]
    assert {e.metadata["split"] for e in client.datasets["flysafelz-dev"].examples.values()} == {"dev"}


@pytest.mark.parametrize("variable", ["LANGSMITH_PROJECT", "LANGCHAIN_PROJECT", "LANGSMITH_DATASET", "LANGSMITH_SPLIT", "SPLIT", "PROJECT"])
@pytest.mark.parametrize("value", ["flysafelz-dev", "flysafelz-held-out", "somewhere-else", ""])
def test_no_environment_variable_changes_a_project_or_a_dataset(monkeypatch, variable, value):
    monkeypatch.setenv(variable, value)
    assert experiment_target("dev") == SplitTarget("flysafelz-dev", "flysafelz-dev")
    assert experiment_target("held_out") == SplitTarget("flysafelz-held-out", "flysafelz-held-out")
    assert experiment_target() == experiment_target("dev")


def test_an_upload_goes_to_the_fixed_dataset_when_the_environment_names_another_project(tmp_path, monkeypatch):
    monkeypatch.setenv("LANGSMITH_PROJECT", "flysafelz-dev")
    built = write_dataset(tmp_path)
    client = RecordingClient()
    upload_dataset("held_out", client, data_dir=built.data_dir, root=built.root)
    assert list(client.datasets) == ["flysafelz-held-out"]  # a held-out upload is never routed into dev, nor a dev one into held-out
    monkeypatch.setenv("LANGSMITH_PROJECT", "flysafelz-held-out")
    other = RecordingClient()
    upload_dataset("dev", other, data_dir=built.data_dir, root=built.root)
    assert list(other.datasets) == ["flysafelz-dev"]


def test_the_two_splits_never_share_a_dataset_or_a_project():
    assert SPLITS["dev"].dataset != SPLITS["held_out"].dataset and SPLITS["dev"].project != SPLITS["held_out"].project


def code_strings(source: str) -> list[str]:
    """Every string constant in the source that is not a docstring."""
    tree = ast.parse(source)
    docstrings = {id(node.body[0].value) for node in ast.walk(tree) if isinstance(node, (ast.Module, ast.FunctionDef, ast.ClassDef)) and node.body and isinstance(node.body[0], ast.Expr)}
    return [n.value for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, str) and id(n) not in docstrings]


def test_the_loader_names_one_environment_variable_in_its_code_and_it_is_the_key():
    strings = code_strings(Path(loader.__file__).read_text(encoding="utf-8"))
    assert [s for s in strings if re.fullmatch(r"(LANGSMITH|LANGCHAIN)_\w+", s)] == ["LANGSMITH_API_KEY"]


def test_the_scan_sees_a_planted_project_variable():
    assert "LANGSMITH_PROJECT" in code_strings('import os\nproject = os.environ.get("LANGSMITH_PROJECT")\n')
    assert code_strings('def f():\n    """LANGSMITH_PROJECT in a docstring"""\n') == []
