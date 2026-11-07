"""E13 / R3.9: labels.yaml is read, never written, and only a complete, current label is scored. Fixtures are inline YAML."""

import pytest
from factories import notam
from reader_eval_helpers import FAKE_PHONE, PLACEHOLDER

from evals.reader_eval.labels import LABEL_KEYS, LabelError, LabelStatus, load_labels
from flysafelz.contracts import AppliesTo, Category, ContactKind, Datum, ExemptionCovers, LimitUnit, ProhibitiveReason
from flysafelz.notam_area import build_area

ITEMS = [{"id": "X0001/26", "e_hash": "hash-1"}, {"id": "X0002/26", "e_hash": "hash-2"}, {"id": "X0003/26", "e_hash": "hash-3"}]

COMPLETE = f"""
labels:
  - id: "X0001/26"
    e_hash: "hash-1"
    label:
      category: prohibitive
      activity: restricted area
      floor: SFC
      ceiling: 700FT AGL
      applies_to: ALL_AIRCRAFT
      exemptions:
        - quote: "DOES NOT APPLY TO ACFT WITH PERMISSION, CONTACT {PLACEHOLDER}"
          covers: CONDITIONAL
      contact: {{kind: phone, value: "{PLACEHOLDER}"}}
      activates_designators: [D496A]
      prohibitive_reason: none
"""


def write(tmp_path, text):
    path = tmp_path / "labels.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def only(label_set):
    (label,) = label_set.labels
    return label


def block(**overrides):
    """A complete label block as YAML text, with any key replaced by its YAML (a key set to None is left out)."""
    keys = {
        "category": "cooperative",
        "activity": "drone ops",
        "floor": "SFC",
        "ceiling": "800FT AMSL",
        "applies_to": "UAS_ONLY",
        "exemptions": "[]",
        "contact": "none",
        "activates_designators": "[]",
        "prohibitive_reason": "none",
    }
    keys.update(overrides)
    lines = "".join(f"      {name}: {value}\n" for name, value in keys.items() if value is not None)
    return f'labels:\n  - id: "X0001/26"\n    e_hash: "hash-1"\n    label:\n{lines}'


def test_a_complete_label_becomes_a_reading_the_way_a_models_does(tmp_path):
    labels = load_labels(write(tmp_path, COMPLETE), ITEMS)
    label = only(labels)
    assert label.status is LabelStatus.COMPLETE and label.missing == () and label.problems == ()
    parsed = notam("X0001/26", item_d="1000-1200", item_e="NO SHAPE HERE.")
    reading = label.reading(parsed)
    assert reading.id == "X0001/26" and reading.schedule == "1000-1200"  # code's id and item D, merged in
    assert reading.geometry == build_area(parsed)[0]  # code's area
    assert reading.category is Category.PROHIBITIVE and reading.applies_to is AppliesTo.ALL_AIRCRAFT
    assert (reading.floor.datum, reading.ceiling.datum, reading.ceiling.value, reading.ceiling.unit) == (Datum.SFC, Datum.AGL, 700, LimitUnit.FT)
    assert reading.activates_designators == ["D496A"]
    assert reading.prohibitive_reason is ProhibitiveReason.NONE
    assert (labels.scored, labels.skipped) == (1, 2)  # X0002 and X0003 have no label


def test_label_quotes_and_contacts_stay_as_placeholders(tmp_path):
    label = only(load_labels(write(tmp_path, COMPLETE), ITEMS))
    (exemption,) = label.output.exemptions
    assert exemption.quote.endswith(PLACEHOLDER) and exemption.covers is ExemptionCovers.CONDITIONAL
    assert label.output.contact.value == PLACEHOLDER


@pytest.mark.parametrize(
    "text, floor, ceiling",
    [
        ("SFC", (0, LimitUnit.FT, Datum.SFC), None),
        ("GND", (0, LimitUnit.FT, Datum.SFC), None),
        ("700FT AGL", (700, LimitUnit.FT, Datum.AGL), None),
        ("700 ft amsl", (700, LimitUnit.FT, Datum.AMSL), None),
        ("FL065", (65, LimitUnit.FL, Datum.AMSL), None),
        ("{value: 1200, unit: ft, datum: AMSL}", (1200, LimitUnit.FT, Datum.AMSL), None),
    ],
)
def test_heights_may_be_written_as_text_or_as_a_mapping(tmp_path, text, floor, ceiling):
    label = only(load_labels(write(tmp_path, block(floor=text)), ITEMS))
    assert label.status is LabelStatus.COMPLETE
    assert (label.output.floor.value, label.output.floor.unit, label.output.floor.datum) == floor


@pytest.mark.parametrize("spelling", ["none", "None", "NONE", "null", "~"])
def test_none_and_null_are_explicit_answers_for_the_heights(tmp_path, spelling):
    label = only(load_labels(write(tmp_path, block(floor=spelling, ceiling=spelling)), ITEMS))
    assert label.status is LabelStatus.COMPLETE
    assert label.output.floor is None and label.output.ceiling is None


@pytest.mark.parametrize("spelling", ["[]", "none", "null"])
def test_an_empty_list_none_and_null_are_explicit_answers_for_exemptions_and_designators(tmp_path, spelling):
    label = only(load_labels(write(tmp_path, block(exemptions=spelling, activates_designators=spelling)), ITEMS))
    assert label.status is LabelStatus.COMPLETE
    assert label.output.exemptions == [] and label.output.activates_designators == []


@pytest.mark.parametrize("spelling", ["none", "null", "{kind: none}"])
def test_no_contact_and_no_prohibitive_reason_are_explicit_answers(tmp_path, spelling):
    label = only(load_labels(write(tmp_path, block(contact=spelling, prohibitive_reason=spelling if spelling != "{kind: none}" else "none")), ITEMS))
    assert label.status is LabelStatus.COMPLETE
    assert label.output.contact.kind is ContactKind.NONE and label.output.contact.value is None
    assert label.output.prohibitive_reason is ProhibitiveReason.NONE


@pytest.mark.parametrize("key", LABEL_KEYS)
def test_a_missing_key_makes_the_label_partial_and_is_listed(tmp_path, key):
    label = only(load_labels(write(tmp_path, block(**{key: None})), ITEMS))
    assert label.status is LabelStatus.PARTIAL
    assert label.missing == (key,)
    assert label.output is None
    with pytest.raises(ValueError, match="partial"):
        label.reading(notam("X0001/26"))


@pytest.mark.parametrize("key", ["category", "activity", "applies_to"])
def test_a_blank_category_activity_or_applies_to_is_unfilled_not_an_answer(tmp_path, key):
    label = only(load_labels(write(tmp_path, block(**{key: "null"})), ITEMS))
    assert label.status is LabelStatus.PARTIAL and label.missing == (key,)


def test_a_partial_label_lists_all_its_missing_keys(tmp_path):
    label = only(load_labels(write(tmp_path, block(category=None, ceiling=None, contact=None)), ITEMS))
    assert label.missing == ("category", "ceiling", "contact")


def test_an_item_with_nothing_filled_in_is_partial_and_never_complete(tmp_path):
    text = 'labels:\n  - id: "X0001/26"\n    e_hash: "hash-1"\n    label: {}\n'
    label = only(load_labels(write(tmp_path, text), ITEMS))
    assert label.status is LabelStatus.PARTIAL and label.missing == LABEL_KEYS
    text = 'labels:\n  - id: "X0001/26"\n    e_hash: "hash-1"\n'  # no label block at all
    assert only(load_labels(write(tmp_path, text), ITEMS)).missing == LABEL_KEYS


def test_a_value_that_is_not_allowed_is_unfilled_and_says_why(tmp_path):
    label = only(load_labels(write(tmp_path, block(category="friendly", floor="seven hundred feet")), ITEMS))
    assert label.status is LabelStatus.PARTIAL
    assert label.missing == ("category", "floor")
    assert len(label.problems) == 2 and label.problems[0].startswith("category:") and "seven hundred feet" in label.problems[1]


def test_a_partial_label_is_skipped_and_counted_never_scored(tmp_path):
    labels = load_labels(write(tmp_path, block(ceiling=None)), ITEMS)
    assert (labels.scored, labels.skipped) == (0, 3)
    assert [label.notam_id for label in labels.partial] == ["X0001/26"]
    assert labels.unlabelled == ("X0002/26", "X0003/26")


def test_a_label_whose_hash_no_longer_matches_is_stale_and_not_scored(tmp_path):
    labels = load_labels(write(tmp_path, COMPLETE.replace("hash-1", "an-old-hash")), ITEMS)
    assert [label.status for label in labels.labels] == [LabelStatus.STALE]
    assert only(labels).output is None
    assert (labels.scored, labels.skipped) == (0, 3)  # X0001 is stale; X0002 and X0003 have no label
    with pytest.raises(ValueError, match="stale"):
        only(labels).reading(notam("X0001/26"))


def test_a_label_for_an_id_not_on_the_sheet_is_unknown_and_not_an_item(tmp_path):
    labels = load_labels(write(tmp_path, COMPLETE.replace("X0001/26", "X9999/26")), ITEMS)
    assert [label.status for label in labels.labels] == [LabelStatus.UNKNOWN]
    assert labels.unknown and labels.scored == 0
    assert labels.skipped == 3  # all three sheet items are unlabelled; the unknown label is not one of them


@pytest.mark.parametrize("text", ["", "# only a comment\n", "labels: []\n", "labels:\n", "{}\n"])
def test_an_empty_file_gives_zero_scored_and_everything_skipped(tmp_path, text):
    labels = load_labels(write(tmp_path, text), ITEMS)
    assert (labels.scored, labels.skipped, labels.labels) == (0, 3, ())


def test_a_missing_file_gives_zero_scored_and_no_crash(tmp_path):
    labels = load_labels(tmp_path / "absent.yaml", ITEMS)
    assert (labels.scored, labels.skipped) == (0, 3)


def test_a_file_with_only_unknown_ids_scores_nothing(tmp_path):
    text = COMPLETE.replace("X0001/26", "X8000/26") + COMPLETE.replace("X0001/26", "X8001/26").replace("labels:\n", "")
    labels = load_labels(write(tmp_path, text), ITEMS)
    assert len(labels.unknown) == 2 and (labels.scored, labels.skipped) == (0, 3)


def test_all_three_outcomes_in_one_file(tmp_path):
    complete = COMPLETE.split("labels:\n")[1]
    partial = complete.replace("X0001/26", "X0002/26").replace("hash-1", "hash-2").replace("      floor: SFC\n", "")
    stale = complete.replace("X0001/26", "X0003/26")  # hash-1 is not X0003's hash
    labels = load_labels(write(tmp_path, "labels:\n" + complete + partial + stale), ITEMS)
    assert [label.status for label in labels.labels] == [LabelStatus.COMPLETE, LabelStatus.PARTIAL, LabelStatus.STALE]
    assert (labels.scored, labels.skipped) == (1, 2)
    assert labels.get("X0002/26").missing == ("floor",)
    assert labels.get("X0404/26") is None


def test_the_sheets_own_items_with_label_blocks_are_read_the_same_way(tmp_path):
    text = "items:\n" + COMPLETE.split("labels:\n")[1]
    assert only(load_labels(write(tmp_path, text), ITEMS)).status is LabelStatus.COMPLETE


@pytest.mark.parametrize(
    "text, message",
    [
        ("labels: [", "not valid YAML"),
        ("labels: 3\n", "must be a list"),
        ("- just a list\n", "must be a list"),
        ('labels:\n  - id: "X0001/26"\n', "needs an id and an e_hash"),
        ('labels:\n  - "X0001/26"\n', "needs an id and an e_hash"),
        (COMPLETE + COMPLETE.split("labels:\n")[1], "more than once"),
    ],
)
def test_a_file_that_is_not_usable_is_an_error_naming_it(tmp_path, text, message):
    path = write(tmp_path, text)
    with pytest.raises(LabelError, match=message) as caught:
        load_labels(path, ITEMS)
    assert str(path) in str(caught.value)


def test_a_file_that_cannot_be_read_is_a_label_error(tmp_path):
    with pytest.raises(LabelError, match="cannot read"):
        load_labels(tmp_path, ITEMS)  # a directory


def test_loading_does_not_change_the_file(tmp_path):
    path = write(tmp_path, COMPLETE)
    before = (path.read_bytes(), path.stat().st_mtime_ns)
    load_labels(path, ITEMS)
    assert (path.read_bytes(), path.stat().st_mtime_ns) == before


def test_the_fake_phone_is_in_the_drama_range():
    assert FAKE_PHONE.startswith("07700 900")
