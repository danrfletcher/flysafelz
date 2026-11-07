"""G14: per-field accuracy of a reading against a label, and the mean and worst run over several runs."""

import pytest
from factories import agl, amsl, exemption, fl, reading

from evals.reader_eval.fields import FIELDS, NOT_SCORED_FIELDS, SCORED_FIELDS, aggregate, field_accuracy
from evals.reader_eval.settings import SETTINGS
from flysafelz.contracts import AppliesTo, Category, ContactKind, ExemptionCovers, ProhibitiveReason

LABEL = reading(category=Category.PROHIBITIVE, floor=amsl(1000), ceiling=amsl(2000), designators=["D496A"], exemptions=[exemption("A", ExemptionCovers.CONDITIONAL)])


def same(**overrides):
    """LABEL with some fields changed."""
    return LABEL.model_copy(update=overrides)


def test_a_reading_equal_to_the_label_is_right_in_every_scored_field():
    got = field_accuracy(LABEL, same())
    assert set(got) == set(FIELDS)
    assert all(got[name] is True for name in SCORED_FIELDS)


def test_geometry_and_activity_are_not_scored_never_true():
    wrong = same(activity="an entirely different activity", geometry=None)
    for read in (same(), wrong):
        got = field_accuracy(LABEL, read)
        assert got["geometry"] is None and got["activity"] is None
    assert set(NOT_SCORED_FIELDS) == {"geometry", "activity"}


@pytest.mark.parametrize(
    "field, change",
    [
        ("category", {"category": Category.COOPERATIVE}),
        ("applies_to", {"applies_to": AppliesTo.UAS_ONLY}),
        ("prohibitive_reason", {"prohibitive_reason": ProhibitiveReason.LIVE_FIRING}),
        ("activates_designators", {"activates_designators": []}),
        ("activates_designators", {"activates_designators": ["D496A", "D412"]}),
        ("exemption_covers", {"exemptions": []}),
        ("exemption_covers", {"exemptions": [exemption("A", ExemptionCovers.PARAMOTOR_CLASS)]}),
    ],
)
def test_an_exact_match_field_is_wrong_when_it_differs(field, change):
    got = field_accuracy(LABEL, same(**change))
    assert got[field] is False
    assert [name for name in SCORED_FIELDS if got[name] is False] == [field]  # and nothing else moves


def test_contact_kind_is_compared_not_the_contact_value():
    assert field_accuracy(LABEL, same(contact=LABEL.contact.model_copy(update={"value": "<PHONE_9>"})))["contact_kind"] is True
    assert field_accuracy(LABEL, same(contact=LABEL.contact.model_copy(update={"kind": ContactKind.EMAIL})))["contact_kind"] is False


def test_designators_are_a_normalised_set():
    label = same(activates_designators=["D496A", "D412"])
    assert field_accuracy(label, same(activates_designators=["EGD412", "EGD496A"]))["activates_designators"] is True  # order and the EG prefix do not matter
    assert field_accuracy(label, same(activates_designators=["EGD496A", "EGD496A", "D412"]))["activates_designators"] is True


def test_exemption_covers_is_a_set_and_the_quotes_are_not_compared():
    label = same(exemptions=[exemption("ONE", ExemptionCovers.CONDITIONAL), exemption("TWO", ExemptionCovers.PARAMOTOR_CLASS)])
    read = same(exemptions=[exemption("OTHER TWO", ExemptionCovers.PARAMOTOR_CLASS), exemption("OTHER ONE", ExemptionCovers.CONDITIONAL), exemption("THREE", ExemptionCovers.CONDITIONAL)])
    assert field_accuracy(label, read)["exemption_covers"] is True


def test_datum_and_unit_are_exact_for_the_floor_and_the_ceiling_each():
    got = field_accuracy(same(floor=agl(500), ceiling=fl(30)), same(floor=amsl(500), ceiling=amsl(3000)))
    assert (got["floor_datum"], got["floor_unit"], got["ceiling_datum"], got["ceiling_unit"]) == (False, True, True, False)


def test_e6_800_ft_amsl_against_500_ft_agl_is_a_datum_mismatch_and_a_height_mismatch_at_500_ft_ground():
    label, read = same(ceiling=agl(500)), same(ceiling=amsl(800))
    got = field_accuracy(label, read)
    assert got["ceiling_datum"] is False
    assert got["ceiling_height"] is False  # 1,000 ft against 800 ft at the 500 ft site
    at_300 = SETTINGS.model_copy(update={"site_elevation_ft": 300})
    assert field_accuracy(label, read, at_300)["ceiling_height"] is True  # equal at 300 ft ground, the datum still wrong
    assert field_accuracy(label, read, at_300)["ceiling_datum"] is False


def test_heights_are_equal_after_converting_flight_levels_to_feet():
    assert field_accuracy(same(ceiling=fl(30)), same(ceiling=amsl(3000)))["ceiling_height"] is True
    assert field_accuracy(same(ceiling=fl(30)), same(ceiling=amsl(3000)))["ceiling_unit"] is False


@pytest.mark.parametrize("gap, right", [(0, True), (100, True), (101, False), (-100, True), (-101, False)])
def test_height_tolerance_is_the_settings_value(gap, right):
    assert SETTINGS.field_height_tolerance_ft == 100
    got = field_accuracy(same(ceiling=amsl(2000)), same(ceiling=amsl(2000 + gap)))
    assert got["ceiling_height"] is right
    assert got["floor_height"] is True


def test_a_tighter_tolerance_from_the_settings_changes_the_answer():
    tight = SETTINGS.model_copy(update={"field_height_tolerance_ft": 10})
    assert field_accuracy(same(ceiling=amsl(2000)), same(ceiling=amsl(2050)), tight)["ceiling_height"] is False


def test_null_against_null_is_a_match_and_null_against_a_value_is_not():
    both = field_accuracy(same(ceiling=None, floor=None), same(ceiling=None, floor=None))
    assert all(both[name] is True for name in ("ceiling_datum", "ceiling_unit", "ceiling_height", "floor_datum", "floor_unit", "floor_height"))
    label_null = field_accuracy(same(ceiling=None), same(ceiling=amsl(2000)))
    read_null = field_accuracy(same(ceiling=amsl(2000)), same(ceiling=None))
    for got in (label_null, read_null):
        assert (got["ceiling_datum"], got["ceiling_unit"], got["ceiling_height"]) == (False, False, False)
        assert got["floor_height"] is True


def test_an_unread_run_scores_every_scored_field_wrong_and_the_others_none():
    got = field_accuracy(LABEL, None)
    assert all(got[name] is False for name in SCORED_FIELDS)
    assert all(got[name] is None for name in NOT_SCORED_FIELDS)


# --- aggregate -----------------------------------------------------------------------------------------------------------------


def run_of(*category_right):
    """One run over several items: field_accuracy for each, where the item's category is right or wrong."""
    return [field_accuracy(LABEL, same(category=Category.PROHIBITIVE if right else Category.INFORMATIONAL)) for right in category_right]


def test_aggregate_gives_the_mean_over_runs_and_the_worst_run_per_field():
    runs = [run_of(True, True), run_of(True, False), run_of(False, False)]  # category accuracy 1.0, 0.5, 0.0
    got = aggregate(runs)
    assert got["category"] == pytest.approx((0.5, 0.0))
    assert got["applies_to"] == (1.0, 1.0)  # right in every run
    assert set(got) == set(SCORED_FIELDS)  # the not-scored fields are left out


def test_aggregate_over_three_runs_of_one_item_each():
    runs = [field_accuracy(LABEL, same()), field_accuracy(LABEL, same(category=Category.COOPERATIVE)), field_accuracy(LABEL, None)]
    got = aggregate(runs)  # a bare dict stands for a run of one item
    assert got["category"] == pytest.approx((1 / 3, 0.0))
    assert got["floor_height"] == pytest.approx((2 / 3, 0.0))  # right in run 1 and 2, wrong in the unread run 3
    assert got["applies_to"] == pytest.approx((2 / 3, 0.0))


def test_aggregate_leaves_out_none_items_and_a_field_never_scored():
    assert aggregate([]) == {}
    assert aggregate([[{"category": None}]]) == {}
    mixed = aggregate([[{"category": True}, {"category": None}, {"category": False}]])
    assert mixed == {"category": (0.5, 0.5)}  # the None item is not counted
