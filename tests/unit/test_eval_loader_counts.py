"""The loader's count checks (5a PR-3.F1, G5): exactly 29 dev, 15 held_out and 24 sheet entries, and an error that states expected against found.

Rule driven, over synthetic datasets built to the real shape and then moved one example either way. A wrong count stops the loader before any upload:
the RecordingClient's call log stays empty.
"""

import pytest
import yaml
from eval_loader_factories import make_examples, sheet_entry, write_case, write_dataset

from evals.agent.import_dataset import DatasetError
from evals.agent.loader import EXPECTED, load_examples, upload_dataset
from evals.agent.recording_client import RecordingClient


def examples_with(dev=29, held_out=15):
    return make_examples(dev, held_out, 16, 8)


def extra_sheet_entry(tmp_path, examples):
    """A 25th sheet entry: for a non-assess example, which the stale check leaves alone and the count does not."""
    other = next(e for e in examples if e["reference"]["path"] != "assess")
    case = write_case(tmp_path, other["fixtures"]["case"])
    return sheet_entry(other, stamped_at=case)


# (rule, dataset builder arguments, the text the error must carry)
COUNT_RULES = [
    ("dev 28", {"dev": 28}, "dev expected 29, found 28"),
    ("dev 30", {"dev": 30}, "dev expected 29, found 30"),
    ("held-out 14", {"held_out": 14}, "held_out expected 15, found 14"),
    ("held-out 16", {"held_out": 16}, "held_out expected 15, found 16"),
    ("sheet 23", {"drop_entry": True}, "sheet expected 24, found 23"),
    ("sheet 25", {"extra_entry": True}, "sheet expected 24, found 25"),
]


def build(tmp_path, dev=29, held_out=15, drop_entry=False, extra_entry=False):
    examples = examples_with(dev, held_out)
    if drop_entry:
        return write_dataset(tmp_path, examples, no_entry=frozenset({examples[0]["id"]}))
    built = write_dataset(tmp_path, examples)
    if extra_entry:
        sheet = [*built.sheet, extra_sheet_entry(tmp_path, examples)]
        (built.data_dir / "verdict_sheet.yaml").write_text(yaml.safe_dump(sheet, sort_keys=False), encoding="utf-8")
    return built


def test_the_expected_counts_are_29_15_and_24():
    assert EXPECTED == {"dev": 29, "held_out": 15, "sheet": 24}


def test_the_right_counts_load(tmp_path):
    built = build(tmp_path)
    assert len(load_examples("dev", data_dir=built.data_dir, root=built.root)) == 29
    assert len(load_examples("held_out", data_dir=built.data_dir, root=built.root)) == 15


@pytest.mark.parametrize(("rule", "args", "text"), COUNT_RULES, ids=[r[0] for r in COUNT_RULES])
@pytest.mark.parametrize("split", ["dev", "held_out"])
def test_a_wrong_count_is_an_error_stating_expected_against_found_whichever_split_was_asked_for(tmp_path, rule, args, text, split):
    built = build(tmp_path, **args)
    with pytest.raises(DatasetError) as caught:
        load_examples(split, data_dir=built.data_dir, root=built.root)
    assert text in str(caught.value)
    assert str(caught.value).startswith("dataset counts are wrong: ")


@pytest.mark.parametrize(("rule", "args", "text"), COUNT_RULES, ids=[r[0] for r in COUNT_RULES])
def test_a_wrong_count_stops_an_upload_before_the_first_call(tmp_path, rule, args, text):
    built = build(tmp_path, **args)
    client = RecordingClient()
    with pytest.raises(DatasetError, match="expected"):
        upload_dataset("dev", client, data_dir=built.data_dir, root=built.root)
    assert client.calls == []


def test_every_wrong_count_is_named_in_one_message(tmp_path):
    examples = examples_with(28, 16)
    built = write_dataset(tmp_path, examples, no_entry=frozenset({examples[0]["id"]}))
    with pytest.raises(DatasetError) as caught:
        load_examples("dev", data_dir=built.data_dir, root=built.root)
    counts = "dataset counts are wrong: dev expected 29, found 28; held_out expected 15, found 16; sheet expected 24, found 23"
    assert str(caught.value) == f"{counts}; assess example(s) with no verdict sheet entry: {examples[0]['id']}"  # the id is named too


def test_a_split_other_than_dev_or_held_out_is_an_error_naming_the_example(tmp_path):
    examples = examples_with()
    examples[3]["metadata"]["split"] = "test"
    built = write_dataset(tmp_path, examples)
    with pytest.raises(DatasetError) as caught:
        load_examples("dev", data_dir=built.data_dir, root=built.root)
    assert str(caught.value) == f"example {examples[3]['id']}: metadata.split is 'test'; it must be dev or held_out"


def test_a_split_asked_for_that_is_not_dev_or_held_out_is_an_error(tmp_path):
    built = build(tmp_path)
    with pytest.raises(DatasetError, match=r"unknown split 'test': use dev or held_out"):
        load_examples("test", data_dir=built.data_dir, root=built.root)


def test_the_count_error_is_the_same_whatever_the_example_moved_to(tmp_path):
    """A dev example moved to held_out leaves both counts wrong; the message names both."""
    examples = examples_with()
    examples[0]["metadata"]["split"] = "held_out"
    built = write_dataset(tmp_path, examples)
    with pytest.raises(DatasetError, match=r"dev expected 29, found 28; held_out expected 15, found 16"):
        load_examples("dev", data_dir=built.data_dir, root=built.root)
