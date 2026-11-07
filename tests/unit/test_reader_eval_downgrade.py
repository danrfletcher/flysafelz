"""G13: harmful downgrades and over-caution. notam_action() is the real one, on both sides, at about 25 real spots; nothing is mocked.

Run `pytest tests/unit/test_reader_eval_downgrade.py -k harmful -s` to read the finding lines these tests print.
"""

from itertools import product

import pytest
from factories import agl, amsl, exemption, fl, reading, utc
from reader_eval_helpers import FAKE_PHONE, q_notam, read_result

from evals.reader_eval import downgrade
from evals.reader_eval.downgrade import SEVERITY, FindingKind, score_item_run, severity_rank
from evals.reader_eval.settings import SETTINGS
from evals.reader_eval.spots import SpotKind, eval_segment, plan_spots
from flysafelz.contracts import AppliesTo, Category, Circle, ContactKind, ExemptionCovers, NotamAction, ProhibitiveReason
from flysafelz.designators import normalise_designator
from flysafelz.notam_action import ActionSegment, LandingZone, notam_action

NOTAM = q_notam(valid_from=utc(0), valid_to=utc(23), upper_fl=100)  # the Q-line reaches 10,000 ft, so it never clips a test height
PROHIBITIVE = {"category": Category.PROHIBITIVE, "ceiling": amsl(2000)}  # NO_FLY inside, AVOID_SECTOR outside
COOPERATIVE = {"category": Category.COOPERATIVE, "ceiling": amsl(2000)}  # CONTACT_FIRST inside with a phone, NO_FLY without one
INFORMATIONAL = {"category": Category.INFORMATIONAL, "ceiling": amsl(2000)}  # AWARENESS inside


def score(label, read, parsed=NOTAM, settings=SETTINGS):
    """One run: the reader's reading (None: the reader failed) against a label."""
    return score_item_run(parsed, label, read_result(read), settings)


def show(run, finding):
    """Print the finding as it will read in the report, for `-k harmful -s`."""
    print(finding.line(run))


# --- the severity order ---------------------------------------------------------------------------------------------------------


def test_the_severity_order_is_the_spec_s():
    assert SEVERITY == (
        NotamAction.NO_FLY,
        NotamAction.UNREAD_OVER_LZ,
        NotamAction.CONTACT_FIRST,
        NotamAction.AVOID_SECTOR,
        NotamAction.AWARENESS,
        NotamAction.AWARENESS_MAP,
        NotamAction.IGNORE,
    )


def test_the_severity_order_covers_every_action_once_so_a_new_member_fails_here():
    assert len(SEVERITY) == len(set(SEVERITY))
    assert set(SEVERITY) == set(NotamAction), "a NotamAction member was added or removed: place it in downgrade.SEVERITY"


@pytest.mark.parametrize("stricter, milder", list(product(NotamAction, NotamAction)), ids=lambda action: action.value)
def test_every_ordered_pair_of_actions_compares_totally(stricter, milder):
    expected = list(SEVERITY).index(stricter) - list(SEVERITY).index(milder)
    got = severity_rank(stricter) - severity_rank(milder)
    assert (got > 0) == (expected > 0) and (got < 0) == (expected < 0) and (got == 0) == (stricter is milder)


# --- actions: harmful, over-caution, agreement ------------------------------------------------------------------------------------


def test_a_reading_equal_to_the_label_is_agreement():
    run = score(reading(**PROHIBITIVE), reading(**PROHIBITIVE))
    assert run.agreement and run.harmful == [] and run.over_caution == [] and not run.reader_failed and not run.segment_miss
    assert run.spots == 25 and run.dropped_spots == 0


def test_harmful_a_uas_only_reading_of_an_all_aircraft_label_is_named_with_applies_to():
    label = reading(**PROHIBITIVE)
    run = score(label, reading(applies_to=AppliesTo.UAS_ONLY, **PROHIBITIVE))
    (finding,) = run.harmful
    show(2, finding)
    assert run.over_caution == [] and not run.agreement
    assert (finding.kind, finding.field) == (FindingKind.ACTION, "applies_to")
    assert (finding.label_action, finding.read_action) == (NotamAction.NO_FLY, NotamAction.IGNORE)
    assert finding.spot.kind is SpotKind.CENTRE
    assert finding.line(2) == "X0001/26 run 2: NO_FLY → IGNORE at the centre (applies_to UAS_ONLY vs ALL_AIRCRAFT labelled)"


def test_a_uas_only_label_gives_ignore_on_both_sides_and_agrees():
    label = reading(applies_to=AppliesTo.UAS_ONLY, **PROHIBITIVE)
    run = score(label, reading(applies_to=AppliesTo.UAS_ONLY, **PROHIBITIVE))
    assert run.agreement
    plan = plan_spots(None, NOTAM)
    segment = eval_segment(NOTAM, label)
    for spot in plan.spots:
        lz = LandingZone(lat=spot.lat, lon=spot.lon, elevation_ft=SETTINGS.site_elevation_ft)
        assert notam_action(NOTAM, label, lz, segment).action is NotamAction.IGNORE
    assert not run.segment_miss  # a UAS-only label is expected to give IGNORE at the centre


def test_over_caution_a_stricter_reading_that_is_milder_nowhere():
    run = score(reading(applies_to=AppliesTo.UAS_ONLY, **PROHIBITIVE), reading(**PROHIBITIVE))
    (finding,) = run.over_caution
    print(finding.line(1))
    assert run.harmful == [] and not run.agreement
    assert (finding.label_action, finding.read_action) == (NotamAction.IGNORE, NotamAction.NO_FLY)
    assert finding.field == "applies_to"


def test_harmful_wins_when_a_reading_is_milder_at_some_spots_and_stricter_at_none():
    run = score(reading(**PROHIBITIVE), reading(**INFORMATIONAL))
    (finding,) = run.harmful
    show(1, finding)
    assert (finding.label_action, finding.read_action, finding.field) == (NotamAction.NO_FLY, NotamAction.AWARENESS, "category")
    assert run.over_caution == []


def test_harmful_the_nearest_milder_spot_to_the_centre_is_named():
    label = reading(**PROHIBITIVE)
    run = score(label, reading(**INFORMATIONAL))
    segment = eval_segment(NOTAM, label)
    plan = plan_spots(None, NOTAM)

    def act(r, spot):
        return notam_action(NOTAM, r, LandingZone(lat=spot.lat, lon=spot.lon, elevation_ft=SETTINGS.site_elevation_ft), segment).action

    milder = [spot for spot in plan.spots if severity_rank(act(reading(**INFORMATIONAL), spot)) > severity_rank(act(label, spot))]
    assert {spot.kind for spot in milder} == {SpotKind.CENTRE, SpotKind.INSIDE}  # outside the edge the two agree
    assert run.harmful[0].spot is not None and run.harmful[0].spot == plan.spots[0]  # the centre is nearest to itself


def test_the_finding_names_the_milder_spot_nearest_the_centre_of_those_it_is_given():
    label = reading(**PROHIBITIVE)
    read = reading(**INFORMATIONAL)
    plan = plan_spots(None, NOTAM)
    segment = eval_segment(NOTAM, label)
    judging = downgrade._Judging(NOTAM, label, read, segment, SETTINGS)
    outside = [spot for spot in plan.spots if spot.kind is SpotKind.OUTSIDE]
    far = [spot for spot in plan.spots if spot.kind is SpotKind.FAR]
    rows = [(spot, NotamAction.NO_FLY, NotamAction.AWARENESS) for spot in (*far, *outside)]
    finding = downgrade._action_finding(judging, plan.spots[0], rows)
    assert finding.spot in outside  # nearer the centre than any far spot, whatever the order of the rows


# --- unread --------------------------------------------------------------------------------------------------------------------


def test_harmful_an_unread_notam_over_a_no_fly_label_is_tagged_reader_failed():
    run = score(reading(**PROHIBITIVE), None)
    (finding,) = run.harmful
    show(3, finding)
    assert run.reader_failed and finding.reader_failed and finding.field == "reader failed"
    assert (finding.label_action, finding.read_action) == (NotamAction.NO_FLY, NotamAction.UNREAD_OVER_LZ)
    assert finding.line(3) == "X0001/26 run 3: NO_FLY → UNREAD_OVER_LZ at the centre (reader failed)"


def test_an_unread_notam_over_a_contact_first_label_is_over_caution_not_harm():
    run = score(reading(**COOPERATIVE), None)
    (finding,) = run.over_caution
    print(finding.line(1))
    assert run.harmful == [] and run.reader_failed
    assert (finding.label_action, finding.read_action, finding.field) == (NotamAction.CONTACT_FIRST, NotamAction.UNREAD_OVER_LZ, "reader failed")


def test_an_unread_result_is_scored_on_notam_action_with_no_reading(monkeypatch):
    seen = []
    real = downgrade.notam_action

    def spy(parsed, read, lz, segment, **kwargs):
        seen.append(read)
        return real(parsed, read, lz, segment, **kwargs)

    monkeypatch.setattr(downgrade, "notam_action", spy)
    score(reading(**PROHIBITIVE), None)
    assert None in seen and any(r is not None for r in seen)  # the reader's side had no reading, the label's side did


# --- danger areas ---------------------------------------------------------------------------------------------------------------


def test_harmful_a_designator_the_label_activates_and_the_reading_lacks():
    run = score(reading(designators=["D496A"], **PROHIBITIVE), reading(**PROHIBITIVE))
    (finding,) = run.harmful
    show(1, finding)
    assert (finding.kind, finding.field, finding.message) == (FindingKind.DESIGNATORS, "designators", "EGD496A not activated")
    assert finding.line(1) == "X0001/26 run 1: EGD496A not activated"
    assert run.over_caution == []


def test_an_extra_designator_is_over_caution():
    run = score(reading(**PROHIBITIVE), reading(designators=["D412"], **PROHIBITIVE))
    (finding,) = run.over_caution
    assert finding.message == f"{normalise_designator('D412')} activated but not labelled" and run.harmful == []


def test_designators_are_compared_after_normalising_them():
    run = score(reading(designators=["D496A"], **PROHIBITIVE), reading(designators=["EGD496A"], **PROHIBITIVE))
    assert run.agreement


def test_one_missing_and_one_extra_designator_are_one_harm_and_one_over_caution():
    run = score(reading(designators=["D496A"], **PROHIBITIVE), reading(designators=["D412"], **PROHIBITIVE))
    assert [f.message for f in run.harmful] == ["EGD496A not activated"]
    assert [f.message for f in run.over_caution] == ["EGD412 activated but not labelled"]


# --- heights (R3.12) -------------------------------------------------------------------------------------------------------------


def height_kinds(run):
    return [(f.field, f.message) for f in run.harmful + run.over_caution if f.kind is FindingKind.HEIGHT]


@pytest.mark.parametrize(
    "read_top, harmful, over_caution",
    [
        (2000, False, False),
        (1900, False, False),  # exactly the tolerance below: neither
        (1899, True, False),  # 101 ft below: harmful
        (1800, True, False),
        (2100, False, False),  # exactly the tolerance above: neither
        (2101, False, True),  # 101 ft above: over-caution
    ],
)
def test_harmful_a_top_more_than_the_tolerance_below_the_label_s(read_top, harmful, over_caution):
    run = score(reading(**PROHIBITIVE), reading(category=Category.PROHIBITIVE, ceiling=amsl(read_top)))
    found = [f for f in run.harmful if f.kind is FindingKind.HEIGHT]
    cautious = [f for f in run.over_caution if f.kind is FindingKind.HEIGHT]
    assert (len(found) == 1, len(cautious) == 1) == (harmful, over_caution)
    if found:
        show(2, found[0])
        assert found[0].field == "ceiling" and f"top {read_top:,} ft vs 2,000 ft labelled" in found[0].message


@pytest.mark.parametrize(
    "read_floor, harmful, over_caution",
    [
        (1000, False, False),
        (1100, False, False),  # exactly the tolerance above: neither
        (1101, True, False),  # 101 ft above: a bottom too high leaves part of the NOTAM unread
        (900, False, False),  # exactly the tolerance below: neither
        (899, False, True),  # 101 ft below: over-caution
    ],
)
def test_harmful_a_bottom_more_than_the_tolerance_above_the_label_s(read_floor, harmful, over_caution):
    label = reading(category=Category.PROHIBITIVE, floor=amsl(1000), ceiling=amsl(3000))
    run = score(label, reading(category=Category.PROHIBITIVE, floor=amsl(read_floor), ceiling=amsl(3000)))
    found = [f for f in run.harmful if f.kind is FindingKind.HEIGHT]
    cautious = [f for f in run.over_caution if f.kind is FindingKind.HEIGHT]
    assert (len(found) == 1, len(cautious) == 1) == (harmful, over_caution)
    if found:
        assert found[0].field == "floor" and f"bottom {read_floor:,} ft vs 1,000 ft labelled" in found[0].message


def test_heights_are_compared_after_the_conversion_to_feet():
    run = score(reading(category=Category.PROHIBITIVE, ceiling=fl(30)), reading(category=Category.PROHIBITIVE, ceiling=amsl(3000)))
    assert height_kinds(run) == []  # FL030 is 3,000 ft at the reference pressure


def test_a_reading_with_no_ceiling_against_a_labelled_ceiling_is_a_height_over_caution():
    label = reading(category=Category.PROHIBITIVE, ceiling=amsl(1500))
    run = score(label, reading(category=Category.PROHIBITIVE, ceiling=amsl(1500)).model_copy(update={"ceiling": None}))
    assert [f.field for f in run.over_caution if f.kind is FindingKind.HEIGHT] == ["ceiling"]
    assert any("top 10,000 ft vs 1,500 ft labelled" in f.message for f in run.over_caution)  # E10: with no ceiling the Q-line's FL100 stands


def test_harmful_e6_800_ft_amsl_against_500_ft_agl_is_compared_as_numbers_at_500_ft_ground():
    label = reading(category=Category.PROHIBITIVE, ceiling=agl(500))  # 1,000 ft AMSL at 500 ft ground
    read = reading(category=Category.PROHIBITIVE, ceiling=amsl(800))
    run = score(label, read)
    (finding,) = [f for f in run.harmful if f.kind is FindingKind.HEIGHT]
    show(1, finding)
    assert finding.field == "ceiling" and "top 800 ft vs 1,000 ft labelled" in finding.message and "ground 500 ft" in finding.message
    at_300 = SETTINGS.model_copy(update={"site_elevation_ft": 300})  # 500 AGL is 800 AMSL there: equal
    assert height_kinds(score(label, read, settings=at_300)) == []


# --- field attribution -----------------------------------------------------------------------------------------------------------


def test_harmful_the_field_that_flips_the_spot_is_named_among_several_that_differ():
    label = reading(applies_to=AppliesTo.ALL_AIRCRAFT, **COOPERATIVE)  # CONTACT_FIRST
    read = reading(applies_to=AppliesTo.UAS_ONLY, contact=ContactKind.NONE, value=None, **COOPERATIVE)  # IGNORE
    (finding,) = score(label, read).harmful
    show(2, finding)
    assert finding.field == "applies_to"  # putting the label's contact into the reading changes nothing at this spot
    assert "contact" not in finding.field


def test_over_caution_a_missing_contact_is_the_field_that_made_it_stricter():
    label = reading(**COOPERATIVE)
    read = reading(contact=ContactKind.NONE, value=None, **COOPERATIVE)
    (finding,) = score(label, read).over_caution
    print(finding.line(1))
    assert (finding.label_action, finding.read_action, finding.field) == (NotamAction.CONTACT_FIRST, NotamAction.NO_FLY, "contact")
    assert "contact none vs phone labelled" in finding.message


def test_harmful_a_category_downgrade_is_named_category():
    (finding,) = score(reading(**COOPERATIVE), reading(**INFORMATIONAL)).harmful
    assert (finding.label_action, finding.read_action, finding.field) == (NotamAction.CONTACT_FIRST, NotamAction.AWARENESS, "category")


QUOTED_E = "TEST AREA ACTIVE. DOES NOT APPLY TO PARAMOTORS. EXCEPT WITH PERMISSION OF THE CONTROLLER."  # an exemption's quote must be in E (notam_action checks it)
QUOTED = q_notam(valid_from=utc(0), valid_to=utc(23), upper_fl=100, item_e=QUOTED_E)
PARAMOTORS_EXEMPT = exemption("DOES NOT APPLY TO PARAMOTORS", ExemptionCovers.PARAMOTOR_CLASS)
PERMISSION = exemption("EXCEPT WITH PERMISSION OF THE CONTROLLER", ExemptionCovers.CONDITIONAL)


def test_harmful_e3_a_paramotor_class_exemption_the_label_only_has_as_a_condition():
    label = reading(exemptions=[PERMISSION], **PROHIBITIVE)
    read = reading(exemptions=[PARAMOTORS_EXEMPT], **PROHIBITIVE)
    (finding,) = score(label, read, QUOTED).harmful
    show(1, finding)
    assert (finding.kind, finding.field, finding.read_action) == (FindingKind.ACTION, "exemptions", NotamAction.IGNORE)
    assert "DOES NOT APPLY" not in finding.message  # the finding shows the kinds of exemption, not their quotes
    assert "exemptions [PARAMOTOR_CLASS] vs [CONDITIONAL] labelled" in finding.message


def test_e3_a_conditional_exemption_the_label_does_not_have_is_not_harmful():
    run = score(reading(**PROHIBITIVE), reading(exemptions=[PERMISSION], **PROHIBITIVE), QUOTED)
    assert run.harmful == []


def test_a_difference_that_changes_nothing_at_any_spot_is_not_a_finding():
    run = score(reading(**PROHIBITIVE), reading(activity="something else entirely", **PROHIBITIVE))
    assert run.agreement  # activity is free text and not part of the action
    run = score(reading(**PROHIBITIVE), reading(reason=ProhibitiveReason.LIVE_FIRING, **PROHIBITIVE))
    assert run.agreement


# --- the paramotor timing check --------------------------------------------------------------------------------------------------


def test_a_label_that_applies_to_paramotors_and_gives_ignore_at_the_centre_is_a_segment_miss(monkeypatch):
    """A segment outside B-C (as a wrong segment would be): the real notam_action says the NOTAM is not active, so the label gives IGNORE at the centre."""
    label = reading(**PROHIBITIVE)
    day_after = ActionSegment(start=utc(0, day=20, month=11), end=utc(1, day=20, month=11), pressure_msl=1013.25)
    monkeypatch.setattr(downgrade, "eval_segment", lambda parsed, read, settings=SETTINGS: day_after)
    run = score(label, reading(**PROHIBITIVE))
    assert run.segment_miss and not run.agreement
    assert run.harmful == [] and run.over_caution == []  # both sides said IGNORE: the miss is the only signal


@pytest.mark.parametrize(
    "extra",
    [
        {"applies_to": AppliesTo.UAS_ONLY},
        {"applies_to": AppliesTo.AERODROME_USERS},
        {"exemptions": [exemption("PARAMOTORS EXEMPT", ExemptionCovers.PARAMOTOR_CLASS)]},
    ],
)
def test_labels_that_do_not_apply_to_paramotors_are_expected_to_give_ignore(extra):
    label = reading(**{**PROHIBITIVE, **extra})
    run = score(label, label)
    assert run.agreement and not run.segment_miss


@pytest.mark.parametrize("applies_to", [AppliesTo.ALL_AIRCRAFT, AppliesTo.NAMED_CLASSES])
def test_a_label_that_applies_to_paramotors_is_not_a_miss_when_the_centre_is_not_ignored(applies_to):
    label = reading(applies_to=applies_to, **PROHIBITIVE)
    assert not score(label, label).segment_miss


# --- fences ---------------------------------------------------------------------------------------------------------------------


def test_f1_the_models_area_and_schedule_are_replaced_by_codes_before_notam_action(monkeypatch):
    parsed = q_notam(valid_from=utc(0), valid_to=utc(23), upper_fl=100, item_d="1000-1200")
    elsewhere = Circle(lat=40.0, lon=10.0, radius_nm=2.0)
    seen = []
    real = downgrade.notam_action

    def spy(parsed_notam, read, lz, segment, **kwargs):
        seen.append(read)
        return real(parsed_notam, read, lz, segment, **kwargs)

    monkeypatch.setattr(downgrade, "notam_action", spy)
    liar = reading(geometry=elsewhere, schedule="0000-0001", **PROHIBITIVE)  # a model's area and schedule that put the NOTAM nowhere
    honest = reading(**PROHIBITIVE)
    run = score_item_run(parsed, honest, read_result(liar))
    assert seen and all(r.geometry != elsewhere and r.schedule == "1000-1200" for r in seen if r is not None)
    assert run.agreement  # so the model's area made no difference to its score
    assert score_item_run(parsed, honest, read_result(honest)).agreement


def test_f3_the_findings_keep_phone_numbers_out_of_every_string(monkeypatch):
    label = reading(contact=ContactKind.PHONE, value=FAKE_PHONE, exemptions=[exemption(f"CALL {FAKE_PHONE} FIRST", ExemptionCovers.CONDITIONAL)], designators=["D496A"], **COOPERATIVE)
    read = reading(contact=ContactKind.NONE, value=None, category=Category.INFORMATIONAL, ceiling=amsl(1000), exemptions=[exemption(f"RING {FAKE_PHONE}", ExemptionCovers.PARAMOTOR_CLASS)])
    run = score(label, read)
    assert run.harmful or run.over_caution
    text = repr(run) + " ".join(f.line(1) + f.message + f.field for f in run.harmful + run.over_caution)
    assert FAKE_PHONE not in text and "900123" not in text and "CALL" not in text and "RING" not in text
    unread = score(label, None)
    assert FAKE_PHONE not in repr(unread)


def test_a_circle_that_is_the_q_circle_scores_all_25_spots():
    run = score(reading(**PROHIBITIVE), reading(**PROHIBITIVE))
    assert (run.spots, run.dropped_spots) == (25, 0)


def test_action_segment_is_the_real_type():
    assert isinstance(eval_segment(NOTAM, None), ActionSegment)
