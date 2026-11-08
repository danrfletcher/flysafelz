"""The loader's records (5a PR-3.F1, G5): metadata mapping and defaults, the reference pass-through, frozen `now`, the sheet join, malformed input.

Synthetic datasets only (eval_loader_factories): small examples.yaml and sheet files, a stamp made by the real stamp code, case folders in tmp_path.
Each test calls the real load_examples / load_dataset. No LangSmith key is set and none is needed.
"""

import copy

import pytest
import yaml
from eval_loader_factories import NOW, TRICKY_QUESTION, make_examples, sheet_entry, write_case, write_dataset

from evals.agent.import_dataset import DatasetError
from evals.agent.loader import SHEET_LABELS, load_dataset, load_examples

NINE = ["split", "case_type", "path", "failure_source", "notam_category", "airspace_context", "constructed", "compound", "parts"]


@pytest.fixture(autouse=True)
def no_langsmith_key(monkeypatch):
    monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)


def load(built, split="dev", **kwargs):
    return load_examples(split, data_dir=built.data_dir, root=built.root, **kwargs)


def small(dev=3, held_out=2, assess_dev=2, assess_held_out=1):
    return make_examples(dev, held_out, assess_dev, assess_held_out)


SMALL_COUNTS = {"dev": 3, "held_out": 2, "sheet": 3}


def test_a_record_has_the_four_sections_and_the_nine_metadata_fields(tmp_path):
    built = write_dataset(tmp_path)
    records = load(built)
    assert len(records) == 29
    first = records[0]
    assert set(first) == {"id", "inputs", "fixtures", "reference", "metadata"}
    assert set(first["inputs"]) == {"lat", "lon", "question", "now"}
    assert list(first["metadata"]) == NINE
    assert first["fixtures"] == {"case": "inv_case_1", "drop": []}


def test_the_default_split_is_dev_and_holds_only_dev_examples(tmp_path):
    built = write_dataset(tmp_path)
    records = load_examples(data_dir=built.data_dir, root=built.root)
    assert {r["metadata"]["split"] for r in records} == {"dev"} and len(records) == 29
    held = load(built, "held_out")
    assert {r["metadata"]["split"] for r in held} == {"held_out"} and len(held) == 15


def test_path_comes_from_reference_path_and_a_missing_compound_and_parts_load_as_false_and_empty(tmp_path):
    examples = make_examples()
    del examples[0]["metadata"]["compound"]  # the dataset's blue_mill_today has none
    examples[1]["metadata"].update(compound=True, parts=["assess", "explain"])
    built = write_dataset(tmp_path, examples)
    by_id = {r["id"]: r for r in load(built)}
    plain, compound = by_id[examples[0]["id"]], by_id[examples[1]["id"]]
    assert plain["metadata"]["compound"] is False and plain["metadata"]["parts"] == []
    assert compound["metadata"]["compound"] is True and compound["metadata"]["parts"] == ["assess", "explain"]
    assert plain["metadata"]["path"] == "assess" == plain["reference"]["path"]
    explain = by_id[next(e["id"] for e in examples if e["reference"]["path"] == "explain" and e["metadata"]["split"] == "dev")]
    assert explain["metadata"]["path"] == "explain"


def test_the_metadata_values_are_the_files_and_blind_is_not_one_of_the_nine(tmp_path):
    examples = make_examples()
    examples[0]["metadata"].update(case_type="tricky", failure_source="reader", notam_category="danger_area", airspace_context="tda", constructed=True)
    built = write_dataset(tmp_path, examples)
    meta = load(built)[0]["metadata"]
    assert meta == {
        "split": "dev",
        "case_type": "tricky",
        "path": "assess",
        "failure_source": "reader",
        "notam_category": "danger_area",
        "airspace_context": "tda",
        "constructed": True,
        "compound": False,
        "parts": [],
    }


def test_drop_defaults_to_an_empty_list_and_a_listed_drop_is_kept_in_order_without_repeats(tmp_path):
    examples = make_examples()
    examples[0]["fixtures"]["drop"] = ["pib", "weather", "pib"]
    examples[1]["fixtures"]["drop"] = None
    built = write_dataset(tmp_path, examples)
    records = load(built)
    assert records[0]["fixtures"]["drop"] == ["pib", "weather"]
    assert records[1]["fixtures"]["drop"] == [] and records[2]["fixtures"]["drop"] == []


def test_every_reference_key_passes_through_and_judge_notes_comes_with_it(tmp_path):
    examples = make_examples()
    examples[0]["reference"].update(must_mention=["1500", "INV1/26"], must_not_mention=["go ahead"], forbidden_tools=["get_airspace"], judge_notes="Invented check: names the second hazard.")
    other = next(e for e in examples if e["reference"]["path"] != "assess" and e["metadata"]["split"] == "dev")
    other["reference"].update(judge_notes="Invented note on a non-assess answer.")
    built = write_dataset(tmp_path, examples)
    by_id = {r["id"]: r for r in load(built)}
    ref = by_id[examples[0]["id"]]["reference"]
    for key, value in examples[0]["reference"].items():
        assert ref[key] == value, key
    assert by_id[other["id"]]["reference"]["judge_notes"] == "Invented note on a non-assess answer."
    assert "judge_notes" not in by_id[examples[1]["id"]]["reference"]


def test_now_is_the_files_frozen_text_not_the_clock(tmp_path, now):
    built = write_dataset(tmp_path)
    records = load(built)
    assert {r["inputs"]["now"] for r in records} == {NOW}
    assert NOW != now.strftime("%Y-%m-%dT%H:%MZ")  # the autouse clock is a different moment: a loader reading it would show


def test_a_question_with_apostrophes_curly_quotes_and_non_ascii_is_kept_byte_for_byte(tmp_path):
    examples = make_examples()
    examples[0]["inputs"]["question"] = TRICKY_QUESTION
    built = write_dataset(tmp_path, examples)
    question = load(built)[0]["inputs"]["question"]
    assert question == TRICKY_QUESTION
    assert question.encode("utf-8") == TRICKY_QUESTION.encode("utf-8")
    assert "\u2019" in question and "\u201c" in question and "\u00e9" in question and "'" in question


def test_an_assess_example_carries_the_sheets_labels_and_other_paths_do_not(tmp_path):
    built = write_dataset(tmp_path)
    records = load(built)
    assess = [r for r in records if r["metadata"]["path"] == "assess"]
    other = [r for r in records if r["metadata"]["path"] != "assess"]
    assert len(assess) == 16 and len(other) == 13
    for record in assess:
        assert all(key in record["reference"] for key in SHEET_LABELS)
        assert [row["verdict"] for row in record["reference"]["verdict_by_segment"]] == ["NO_GO", "PILOT_DISCRETION"]
        assert record["reference"]["verdict_by_segment"][0] == {"start": "2026-01-02T04:00:00Z", "end": "2026-01-02T05:00:00Z", "verdict": "NO_GO", "approaching_limits": False}
        assert record["reference"]["tier"] == []
        assert record["reference"]["notam_actions"] == {"INV1/26": "NO_FLY", "INV2/26": "AWARENESS"}
        assert type(record["reference"]["notam_actions"]) is dict  # plain, so it serialises
    for record in other:
        assert not any(key in record["reference"] for key in SHEET_LABELS)


def test_a_null_notam_action_loads_as_ignore(tmp_path):
    examples = make_examples()
    sheet = [sheet_entry(e, stamped_at=write_case(tmp_path, e["fixtures"]["case"]), notam_actions={"INV1/26": "NO_FLY", "INV2/26": None}) for e in examples if e["reference"]["path"] == "assess"]
    built = write_dataset(tmp_path, examples, sheet=sheet)
    assert load(built)[0]["reference"]["notam_actions"] == {"INV1/26": "NO_FLY", "INV2/26": "IGNORE"}


def test_an_assess_example_with_no_sheet_entry_is_an_error_naming_its_id(tmp_path):
    examples = small()
    built = write_dataset(tmp_path, examples, no_entry=frozenset({examples[1]["id"]}))
    with pytest.raises(DatasetError) as caught:
        load(built, expected={"dev": 3, "held_out": 2, "sheet": 2})
    assert examples[1]["id"] in str(caught.value) and "no verdict sheet entry" in str(caught.value)


def test_a_missing_sheet_entry_under_the_real_expected_counts_states_the_counts_and_names_the_id(tmp_path):
    examples = make_examples()  # 29 dev, 15 held_out, 24 assess: the default EXPECTED
    lost = [e["id"] for e in examples if e["reference"]["path"] == "assess"][:2]
    built = write_dataset(tmp_path, examples, no_entry=frozenset(lost))
    with pytest.raises(DatasetError) as caught:
        load_examples("dev", data_dir=built.data_dir, root=built.root)  # no `expected=`: the real counts
    message = str(caught.value)
    assert "sheet expected 24, found 22" in message
    assert f"no verdict sheet entry: {', '.join(lost)}" in message


def test_an_entry_with_actions_done_false_refuses_only_that_example_by_id(tmp_path):
    examples = small()
    built = write_dataset(tmp_path, examples, not_done=frozenset({examples[0]["id"]}))
    result = load_dataset("dev", data_dir=built.data_dir, root=built.root, expected=SMALL_COUNTS)
    assert list(result.refused) == [examples[0]["id"]] and "actions_done" in result.refused[examples[0]["id"]]
    assert [r["id"] for r in result.records] == [examples[1]["id"], examples[2]["id"]]  # the rest still load
    assert [r["id"] for r in load(built, expected=SMALL_COUNTS)] == [examples[1]["id"], examples[2]["id"]]


def test_a_sheet_label_already_in_the_reference_is_an_error_not_an_overwrite(tmp_path):
    examples = small()
    examples[0]["reference"]["tier"] = ["mine"]
    built = write_dataset(tmp_path, examples)
    with pytest.raises(DatasetError, match=rf"example {examples[0]['id']}: reference already has tier"):
        load(built, expected=SMALL_COUNTS)


# --- the stale sheet: PR-2.F1's message goes through unchanged --------------------------------------------------------------------


def test_a_changed_fixture_file_makes_the_loader_refuse_with_the_stale_message_unchanged(tmp_path):
    examples = small()
    built = write_dataset(tmp_path, examples)
    (tmp_path / "fixtures" / "inv_case_1" / "weather.json").write_bytes(b'{"changed": 1}')
    with pytest.raises(DatasetError) as caught:
        load(built, expected=SMALL_COUNTS)
    assert str(caught.value) == "verdict sheet is stale (weather.json changed): run make dataset"


def test_an_unfinalised_entry_is_refused_with_the_rerun_message_and_no_record_is_returned(tmp_path):
    examples = small()
    built = write_dataset(tmp_path, examples)
    sheet = yaml.safe_load((built.data_dir / "verdict_sheet.yaml").read_text(encoding="utf-8"))
    del sheet[0]["finalised"]
    (built.data_dir / "verdict_sheet.yaml").write_text(yaml.safe_dump(sheet, sort_keys=False), encoding="utf-8")
    with pytest.raises(DatasetError) as caught:
        load(built, expected=SMALL_COUNTS)
    assert str(caught.value) == "verdict sheet is stale (finalised stamp missing): run make dataset"


def test_a_hand_edited_label_is_stale_too(tmp_path):
    examples = small()
    built = write_dataset(tmp_path, examples)
    sheet = yaml.safe_load((built.data_dir / "verdict_sheet.yaml").read_text(encoding="utf-8"))
    sheet[0]["notam_actions"]["INV2/26"] = "NO_FLY"
    (built.data_dir / "verdict_sheet.yaml").write_text(yaml.safe_dump(sheet, sort_keys=False), encoding="utf-8")
    with pytest.raises(DatasetError, match=r"verdict sheet is stale \(labels changed\): run make dataset"):
        load(built, expected=SMALL_COUNTS)


# --- malformed examples.yaml: one line naming the example -------------------------------------------------------------------------


def mutated(number_or_id, change):
    examples = small()
    target = next(e for e in examples if e["id"] == number_or_id) if isinstance(number_or_id, str) else examples[number_or_id]
    change(target)
    return examples, target["id"]


def set_in(section, key, value):
    def apply(example):
        example[section][key] = value

    return apply


def delete_in(section, key):
    def apply(example):
        del example[section][key]

    return apply


NON_ASSESS = "inv_dev_03"  # small(): dev_01 and dev_02 are assess, dev_03 is not
RULES = [
    ("a bad drop value, assess", "inv_dev_01", set_in("fixtures", "drop", ["radar"]), "example inv_dev_01: fixtures.drop has 'radar'"),
    ("a bad drop value, other path", NON_ASSESS, set_in("fixtures", "drop", ["radar"]), f"example {NON_ASSESS}: fixtures.drop has 'radar'"),
    ("drop not a list, assess", "inv_dev_01", set_in("fixtures", "drop", "pib"), "example inv_dev_01: fixtures.drop must be a list"),
    ("drop not a list, other path", NON_ASSESS, set_in("fixtures", "drop", "pib"), f"example {NON_ASSESS}: fixtures.drop must be a list"),
    ("missing fixtures.case, assess", "inv_dev_01", delete_in("fixtures", "case"), "example inv_dev_01: fixtures.case must be a case folder name"),
    ("missing fixtures.case, other path", NON_ASSESS, delete_in("fixtures", "case"), f"example {NON_ASSESS}: fixtures.case must be a case folder name"),
    ("a case that is a path", NON_ASSESS, set_in("fixtures", "case", "../outside"), f"example {NON_ASSESS}: fixtures.case must be a case folder name"),
    ("a split that is not dev or held_out", NON_ASSESS, set_in("metadata", "split", "test"), f"example {NON_ASSESS}: metadata.split is 'test'"),
    ("held-out spelt with a hyphen", NON_ASSESS, set_in("metadata", "split", "held-out"), f"example {NON_ASSESS}: metadata.split is 'held-out'"),
    ("no split", NON_ASSESS, delete_in("metadata", "split"), f"example {NON_ASSESS}: metadata.split is None"),
    ("no question", NON_ASSESS, delete_in("inputs", "question"), f"example {NON_ASSESS}: inputs.question must be text"),
    ("a blank question", NON_ASSESS, set_in("inputs", "question", "   "), f"example {NON_ASSESS}: inputs.question must be text"),
    ("now without a timezone", NON_ASSESS, set_in("inputs", "now", "2026-01-02T03:04"), f"example {NON_ASSESS}: inputs.now must be a date and time with a timezone"),
    ("now missing", NON_ASSESS, delete_in("inputs", "now"), f"example {NON_ASSESS}: inputs.now must be a date and time with a timezone"),
    ("lat not a number", NON_ASSESS, set_in("inputs", "lat", "north"), f"example {NON_ASSESS}: inputs.lat must be a number"),
    ("no reference.path", NON_ASSESS, delete_in("reference", "path"), f"example {NON_ASSESS}: reference.path must be text"),
    ("compound not a bool", NON_ASSESS, set_in("metadata", "compound", "yes"), f"example {NON_ASSESS}: metadata.compound must be true or false"),
    ("parts not a list", NON_ASSESS, set_in("metadata", "parts", "assess"), f"example {NON_ASSESS}: metadata.parts must be a list"),
    ("no case_type", NON_ASSESS, delete_in("metadata", "case_type"), f"example {NON_ASSESS}: metadata.case_type is missing"),
    ("no notam_category key", NON_ASSESS, delete_in("metadata", "notam_category"), f"example {NON_ASSESS}: metadata.notam_category is missing"),
    ("constructed missing", NON_ASSESS, delete_in("metadata", "constructed"), f"example {NON_ASSESS}: metadata.constructed must be true or false"),
    ("inputs not a mapping", NON_ASSESS, lambda e: e.update(inputs="x"), f"example {NON_ASSESS}: inputs must be a mapping"),
]


@pytest.mark.parametrize(("name", "target", "change", "message"), RULES, ids=[r[0] for r in RULES])
def test_a_malformed_example_is_a_one_line_error_naming_it(tmp_path, name, target, change, message):
    examples, _ = mutated(target, change)
    built = write_dataset(tmp_path, examples)
    with pytest.raises(DatasetError) as caught:
        load(built, expected=SMALL_COUNTS)
    assert message in str(caught.value) and "\n" not in str(caught.value)


def test_a_duplicate_id_is_an_error_naming_it(tmp_path):
    examples = small()
    examples[1]["id"] = examples[0]["id"]
    built = write_dataset(tmp_path, examples, sheet=[])
    with pytest.raises(DatasetError, match=r"example inv_dev_01: duplicate id"):
        load(built, expected=SMALL_COUNTS)


def test_an_examples_file_that_is_not_a_list_is_an_error(tmp_path):
    built = write_dataset(tmp_path, small(), examples_text="examples: []\n")
    with pytest.raises(DatasetError, match=r"examples.yaml: must be a list of examples"):
        load(built, expected=SMALL_COUNTS)


def test_invalid_yaml_and_a_missing_file_are_errors_naming_the_file(tmp_path):
    built = write_dataset(tmp_path, small(), examples_text="- id: [unclosed\n")
    with pytest.raises(DatasetError, match=r"examples.yaml: not valid YAML"):
        load(built, expected=SMALL_COUNTS)
    (built.data_dir / "examples.yaml").unlink()
    with pytest.raises(DatasetError, match=r"examples.yaml: no such file"):
        load(built, expected=SMALL_COUNTS)


def test_an_item_that_is_not_a_mapping_and_an_item_with_no_id_are_errors(tmp_path):
    with pytest.raises(DatasetError, match=r"example #2: must be a mapping"):
        load(write_dataset(tmp_path / "a", small(), examples_text="- {id: a}\n- just text\n"), expected=SMALL_COUNTS)
    with pytest.raises(DatasetError, match=r"example #1: missing id"):
        load(write_dataset(tmp_path / "b", small(), examples_text="- {inputs: {}}\n"), expected=SMALL_COUNTS)


def test_an_unquoted_now_that_yaml_reads_as_a_datetime_is_refused_not_turned_into_text(tmp_path):
    examples = small()
    text = yaml.safe_dump(examples, sort_keys=False).replace("now: 2026-01-02T03:04Z", "now: 2026-01-02T03:04:00Z", 1)
    assert "now: 2026-01-02T03:04:00Z" in text
    built = write_dataset(tmp_path, examples, examples_text=text)
    with pytest.raises(DatasetError, match=r"example inv_dev_01: inputs.now must be a date and time with a timezone, as quoted text"):
        load(built, expected=SMALL_COUNTS)


def test_the_example_data_is_not_changed_by_loading_twice(tmp_path):
    built = write_dataset(tmp_path)
    before = copy.deepcopy(load(built))
    assert load(built) == before
