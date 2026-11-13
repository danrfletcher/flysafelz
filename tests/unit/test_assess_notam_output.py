"""G15 and G17: the NOTAMs a segment lists against the debug list, and the order of the conditions."""

from airspace_helpers import volume
from factories import (
    LZ,
    airspace_fetch,
    amsl,
    circle_at,
    exemption,
    notam,
    pib_result,
    reading,
    run_assess,
    segment,
    utc,
)

from flysafelz.assess import CONDITION_ORDER, _notam_findings, display
from flysafelz.contracts import Category, ContactKind, NotamAction, NotamActionResult, ProhibitiveReason, Verdict
from flysafelz.notam_action import notam_action


def only_segment(assessment):
    segments = [s for h in assessment.hours for s in h.segments]
    assert len(segments) == 1
    return segments[0]


def read_all(*specs):
    """specs: (notam_id, circle, reading options). Returns the PIB and the readings."""
    notams, readings = [], {}
    for notam_id, circle, options in specs:
        notams.append(notam(notam_id, circle=circle))
        readings[notam_id] = reading(notam_id=notam_id, geometry=circle, **options)
    return pib_result(*notams), readings


def test_a_segment_lists_each_non_ignore_notam_with_its_action_and_a_one_line_reason():
    pib, readings = read_all(
        ("J1/26", circle_at(0), {"activity": "alpha contact"}),
        ("J2/26", circle_at(1.0), {"activity": "charlie sector"}),
    )
    segment = only_segment(run_assess(window=(utc(9), utc(10)), pib=pib, readings=readings))
    actions = {a.notam_id: a for a in segment.notam_actions}
    assert set(actions) == {"J1/26", "J2/26"}
    assert actions["J1/26"].action is NotamAction.CONTACT_FIRST
    assert actions["J2/26"].action is NotamAction.AVOID_SECTOR and actions["J2/26"].bearing_deg is not None
    assert all(a.reason and "\n" not in a.reason for a in segment.notam_actions)


def test_ignored_notams_are_only_in_the_debug_list_each_with_its_reason():
    far = notam("J1/26", circle=circle_at(5.0))  # well beyond the local area: dropped before any reading
    exempted = notam("J2/26", circle=circle_at(0), item_e="UAS OPR WI 0.5NM. PARAMOTORS EXEMPT.")
    exempt_reading = reading(notam_id="J2/26", geometry=circle_at(0), exemptions=[exemption("PARAMOTORS EXEMPT")])
    shown = notam("J3/26", circle=circle_at(0))
    shown_reading = reading(notam_id="J3/26", geometry=circle_at(0))
    assessment = run_assess(
        window=(utc(9), utc(10)),
        pib=pib_result(far, exempted, shown),
        readings={"J2/26": exempt_reading, "J3/26": shown_reading},
    )
    segment = only_segment(assessment)
    assert [a.notam_id for a in segment.notam_actions] == ["J3/26"]
    debug = {i.notam_id: i.reason for i in assessment.ignored_notams}
    assert set(debug) == {"J1/26", "J2/26"}
    assert all(debug.values())
    assert "paramotors" in debug["J2/26"].lower()
    assert not any(i.notam_id in {a.notam_id for a in segment.notam_actions} for i in assessment.ignored_notams)


def test_every_notam_in_the_pib_is_either_in_a_segment_or_in_the_debug_list():
    specs = [("J1/26", circle_at(0), {}), ("J2/26", circle_at(0.5), {}), ("J3/26", circle_at(6.0), {})]
    pib, readings = read_all(*specs)
    assessment = run_assess(window=(utc(9), utc(11)), pib=pib, readings=readings)
    listed = {a.notam_id for h in assessment.hours for s in h.segments for a in s.notam_actions}
    ignored = {i.notam_id for i in assessment.ignored_notams}
    assert listed | ignored == {"J1/26", "J2/26", "J3/26"}
    assert listed & ignored == set()


def test_conditions_are_ordered_contact_permission_ceiling_sector_warning():
    pib, readings = read_all(
        # Written in the reverse of the expected order, so the sort has work to do.
        ("J5/26", circle_at(0), {"activity": "echo warning", "category": Category.INFORMATIONAL}),
        ("J4/26", circle_at(1.0), {"activity": "delta sector"}),
        ("J3/26", circle_at(0), {"activity": "charlie ceiling", "floor": amsl(1000)}),
        ("J1/26", circle_at(0), {"activity": "alpha contact"}),
    )
    atz = volume("atz")  # permission from the ATZ first
    segment = only_segment(run_assess(window=(utc(9), utc(10)), pib=pib, readings=readings, airspace=airspace_fetch(atz)))
    order = ["alpha contact", "permission from", "charlie ceiling", "delta sector", "echo warning"]
    positions = [next(i for i, c in enumerate(segment.conditions) if key in c) for key in order]
    assert positions == sorted(positions)
    assert len(segment.conditions) == 5


def test_the_order_is_stable_inside_a_group():
    pib, readings = read_all(
        ("J2/26", circle_at(0), {"activity": "second contact"}),
        ("J1/26", circle_at(0), {"activity": "first contact"}),
    )
    segment = only_segment(run_assess(window=(utc(9), utc(10)), pib=pib, readings=readings))
    first, second = (next(i for i, c in enumerate(segment.conditions) if key in c) for key in ("second contact", "first contact"))
    assert first < second  # the order the NOTAMs are found in is kept


def test_a_radio_condition_sits_with_permission_before_ceilings():
    pib, readings = read_all(("J1/26", circle_at(0), {"activity": "charlie ceiling", "floor": amsl(1000)}))
    segment = only_segment(run_assess(window=(utc(9), utc(10)), pib=pib, readings=readings, airspace=airspace_fetch(volume("rmz"))))
    assert "radio contact" in segment.conditions[0]
    assert "charlie ceiling" in segment.conditions[1]


def test_a_condition_text_is_never_repeated():
    pib, readings = read_all(("J1/26", circle_at(0), {}))
    segment = only_segment(run_assess(window=(utc(9), utc(10)), pib=pib, readings=readings, airspace=airspace_fetch(volume("atz"), volume("atz", name="TEST ATZ"))))
    assert len(segment.conditions) == len(set(segment.conditions))


# --- ticket 3b PR-3: CHECK_SUP in the segment ---------------------------------------------------------------------------------

DANGER_E = "TEMPORARY DANGER AREA EGD496A ACTIVATED. UAS BVLOS OPS. SEE AIP SUP 108/2026."
GP4_CONDITION = "read the SUP / AIP ENR 5.1 entry for EGD496A before flight; details in AIP SUP 108/2026"


def danger(notam_id="J4/26", item_e=DANGER_E, **options):
    """A constructed QRD NOTAM over the LZ from SFC, active, with its reading: no phone, no email, no placeholder, no real number."""
    circle = circle_at(0)
    parsed = notam(notam_id, circle=circle, subject="RD", item_f="SFC", item_e=item_e)
    read = reading(notam_id=notam_id, geometry=circle, contact=ContactKind.NONE, value=None, **options)
    return parsed, read


def segment_of(*pairs, **kw):
    notams = [p for p, _r in pairs]
    readings = {r.id: r for _p, r in pairs}
    return only_segment(run_assess(window=(utc(9), utc(10)), pib=pib_result(*notams), readings=readings, **kw))


def test_cr3_the_check_sup_slot_sits_directly_before_contact():
    assert CONDITION_ORDER.index("check_sup") + 1 == CONDITION_ORDER.index("contact")
    assert CONDITION_ORDER == ("check_sup", "contact", "permission", "ceiling", "sector", "warning")


def test_gp4_a_constructed_danger_area_with_no_contact_is_pilot_discretion_with_conditions_naming_the_designator_and_the_sup():
    parsed, read = danger()
    assert notam_action(parsed, read, LZ, segment()).action is NotamAction.CHECK_SUP
    seg = segment_of((parsed, read))
    assert [a.action for a in seg.notam_actions] == [NotamAction.CHECK_SUP]
    assert seg.verdict is Verdict.PILOT_DISCRETION and seg.blockers == []
    assert seg.conditions == [GP4_CONDITION]
    assert display(seg) == "PILOT DISCRETION \u2013 WITH CONDITIONS"
    print(f"GP4 verdict: {display(seg)}\nGP4 condition: {seg.conditions[0]}")


def test_gp4_without_the_sup_reference_the_condition_has_no_details_tail_and_no_empty_brackets():
    parsed, read = danger(item_e="TEMPORARY DANGER AREA EGD496A ACTIVATED. UAS BVLOS OPS.")
    seg = segment_of((parsed, read))
    assert seg.conditions == ["read the SUP / AIP ENR 5.1 entry for EGD496A before flight"]
    assert "details in" not in seg.conditions[0] and "[" not in seg.conditions[0] and "()" not in seg.conditions[0]
    assert display(seg) == "PILOT DISCRETION \u2013 WITH CONDITIONS"


def test_a_segment_whose_only_finding_is_check_sup_is_never_plain_pilot_discretion_and_is_flyable():
    assessment = run_assess(window=(utc(9), utc(10)), pib=pib_result(danger()[0]), readings={"J4/26": danger()[1]})
    seg = only_segment(assessment)
    assert display(seg) != "PILOT DISCRETION" and seg.conditions
    assert [(w.start, w.end) for w in assessment.flyable_windows] == [(utc(9), utc(10))]
    assert assessment.overall_verdict is Verdict.PILOT_DISCRETION


def test_the_check_sup_condition_is_listed_before_the_contact_condition_whatever_the_notam_order():
    contact_circle = circle_at(0)
    contact = (notam("J1/26", circle=contact_circle), reading(notam_id="J1/26", geometry=contact_circle, activity="alpha contact", value="07700 900123"))
    for pairs in ((contact, danger()), (danger(), contact)):
        seg = segment_of(*pairs)
        assert len(seg.conditions) == 2
        assert seg.conditions[0] == GP4_CONDITION and "alpha contact" in seg.conditions[1]


def test_a_check_sup_notam_beside_a_no_fly_notam_is_no_go_and_the_check_sup_condition_is_still_listed():
    circle = circle_at(0)
    stay_out = (notam("J1/26", circle=circle, subject="RR"), reading(notam_id="J1/26", geometry=circle))
    seg = segment_of(stay_out, danger())
    assert seg.verdict is Verdict.NO_GO and display(seg) == "NO-GO"
    assert [b.source for b in seg.blockers] == ["notam"] and "J1/26" in seg.blockers[0].text
    assert seg.conditions == [GP4_CONDITION]


def test_a_check_sup_notam_that_the_airspace_path_owns_adds_no_condition_here():
    result = NotamActionResult(notam_id="J4/26", action=NotamAction.CHECK_SUP, reason=GP4_CONDITION)
    assert _notam_findings([result], {"J4/26"}, []) == ([], [], [])
    assert _notam_findings([result], set(), []) == ([], [("check_sup", GP4_CONDITION)], [])
    contact = NotamActionResult(notam_id="J5/26", action=NotamAction.CONTACT_FIRST, reason="call them")
    assert _notam_findings([contact], {"J5/26"}, []) == ([], [], [])  # as CONTACT_FIRST does today


def test_an_unread_danger_area_is_unverified_and_never_check_sup():
    parsed, _read = danger()
    seg = only_segment(run_assess(window=(utc(9), utc(10)), pib=pib_result(parsed)))
    assert seg.verdict is Verdict.UNVERIFIED
    assert [a.action for a in seg.notam_actions] == [NotamAction.UNREAD_OVER_LZ]
    assert not any("read the SUP" in c for c in seg.conditions)


def test_check_sup_maps_to_pilot_discretion_and_nothing_else():
    seg = segment_of(danger())
    assert seg.verdict is Verdict.PILOT_DISCRETION
    assert [v.name for v in Verdict] == ["NO_GO", "UNVERIFIED", "PILOT_DISCRETION"]


# --- ticket 3b PR-4: a danger area the airspace path owns (G13) -------------------------------------------------------------


def owned_danger(designator, item_e, *, contact=ContactKind.NONE, value=None, notam_id="J4/26", **options):
    """An openAIP danger area over the LZ and the constructed QRD NOTAM that activates it, with its reading."""
    circle = circle_at(0)
    area = volume("danger", name=designator, designator=designator, by_notam=True)
    parsed = notam(notam_id, circle=circle, subject="RD", item_f="SFC", item_e=item_e)
    read = reading(notam_id=notam_id, geometry=circle, contact=contact, value=value, designators=[designator], activity="UAS BVLOS ops", **options)
    return area, parsed, read


def owned_segment(*triples):
    areas = [a for a, _p, _r in triples]
    return only_segment(
        run_assess(
            window=(utc(9), utc(10)),
            airspace=airspace_fetch(*areas),
            pib=pib_result(*[p for _a, p, _r in triples]),
            readings={r.id: r for _a, _p, r in triples},
        )
    )


OWNED_CHECK_SUP = owned_danger("EGD496A", DANGER_E)
OWNED_CONTACT = owned_danger("EGD496B", "TEMPORARY DANGER AREA EGD496B ACTIVATED. BOOKING 01632 960123.", contact=ContactKind.PHONE, value="01632 960123", notam_id="J5/26")


def test_g13_a_danger_area_with_no_contact_is_check_sup_from_the_airspace_path_with_the_designator_and_the_sup():
    seg = owned_segment(OWNED_CHECK_SUP)
    assert [a.action for a in seg.notam_actions] == [NotamAction.CHECK_SUP]
    assert seg.verdict is Verdict.PILOT_DISCRETION and seg.blockers == []
    assert seg.conditions == [GP4_CONDITION]  # once: the NOTAM path adds no second condition for an owned NOTAM
    assert display(seg) == "PILOT DISCRETION \u2013 WITH CONDITIONS"


def test_g13_the_airspace_check_sup_sits_in_its_slot_ahead_of_contact_whatever_the_order():
    for triples in ((OWNED_CONTACT, OWNED_CHECK_SUP), (OWNED_CHECK_SUP, OWNED_CONTACT)):
        seg = owned_segment(*triples)
        assert len(seg.conditions) == 2, seg.conditions
        assert seg.conditions[0] == GP4_CONDITION
        assert seg.conditions[1].startswith("Danger area EGD496B active") and "fly only if they agree" in seg.conditions[1]
        assert seg.verdict is Verdict.PILOT_DISCRETION and display(seg) == "PILOT DISCRETION \u2013 WITH CONDITIONS"


def test_g13_an_owned_contact_notam_adds_no_second_condition_from_the_notam_path():
    seg = owned_segment(OWNED_CONTACT)
    assert [a.action for a in seg.notam_actions] == [NotamAction.CONTACT_FIRST]
    assert len(seg.conditions) == 1 and "EGD496B" in seg.conditions[0]


def test_g13_the_airspace_condition_names_the_sup_of_the_item_e_it_was_found_in():
    area, parsed, read = owned_danger("EGD496A", "TEMPORARY DANGER AREA EGD496A ACTIVATED. UAS BVLOS OPS.")
    assert owned_segment((area, parsed, read)).conditions == ["read the SUP / AIP ENR 5.1 entry for EGD496A before flight"]


def test_g13_a_check_sup_beside_a_no_fly_activation_of_another_area_is_no_go_and_still_lists_the_condition():
    byelaws = owned_danger(
        "EGD496B",
        "TEMPORARY DANGER AREA EGD496B ACTIVATED. ENTRY PROHIBITED UNDER BYELAWS.",
        contact=ContactKind.PHONE,
        value="01632 960123",
        notam_id="J5/26",
        reason=ProhibitiveReason.BYELAWS,
        quote="ENTRY PROHIBITED UNDER BYELAWS",
    )
    seg = owned_segment(OWNED_CHECK_SUP, byelaws)
    assert seg.verdict is Verdict.NO_GO and display(seg) == "NO-GO"
    assert any("EGD496B" in b.text and "J5/26" in b.text and "byelaws" in b.text for b in seg.blockers)
    assert seg.conditions == [GP4_CONDITION]


def test_an_unread_notam_that_activates_a_danger_area_adds_one_unverified_line_not_two():
    area, parsed, _read = owned_danger("EGD496A", DANGER_E)
    seg = only_segment(run_assess(window=(utc(9), utc(10)), airspace=airspace_fetch(area), pib=pib_result(parsed)))
    assert seg.verdict is Verdict.UNVERIFIED
    assert [a.action for a in seg.notam_actions] == [NotamAction.UNREAD_OVER_LZ]
    [line] = seg.conditions  # the danger area owns it: the NOTAM's own "has not been read" line is not added again
    assert line.startswith("UNVERIFIED: Danger area EGD496A is activated by NOTAM J4/26, which has not been read")


def test_an_unread_notam_the_airspace_path_said_nothing_about_still_adds_its_own_unverified_line():
    result = NotamActionResult(notam_id="J4/26", action=NotamAction.UNREAD_OVER_LZ, reason="NOTAM J4/26 is over the LZ and has not been read")
    assert _notam_findings([result], {"J4/26"}, []) == ([], [], [result.reason])  # owned, but no airspace reason names it
    assert _notam_findings([result], {"J4/26"}, ["Danger area EGD496A is activated by NOTAM J4/26, which has not been read"]) == ([], [], [])
    assert _notam_findings([result], {"J4/26"}, ["a reason about NOTAM J4/260"]) == ([], [], [result.reason])  # a longer id is another NOTAM
    assert _notam_findings([result], set(), ["NOTAM J4/26 named here"]) == ([], [], [result.reason])  # not owned


def test_an_informational_warning_is_a_condition_whatever_its_reason_says_and_plain_awareness_is_not():
    warn = NotamActionResult(notam_id="J5/26", action=NotamAction.AWARENESS, reason="reworded entirely", warning=True)
    plain = NotamActionResult(notam_id="J6/26", action=NotamAction.AWARENESS, reason="informational over the LZ from the surface")
    assert _notam_findings([warn, plain], set(), []) == ([], [("warning", "reworded entirely")], [])
    assert "warning" not in warn.model_dump()  # the flag is internal: no output changes shape
