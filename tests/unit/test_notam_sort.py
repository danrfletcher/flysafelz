"""G23: the code-only NOTAM sort. One row per IGNORE reason, the order of the steps, and rows shaped like real NOTAMs."""

from datetime import timedelta

import pytest
from pib_helpers import LZ, SPAN, TRIMMED, parsed, utc, with_rules

from flysafelz.contracts import NotamType
from flysafelz.pib import SortAction, SortResult, parse_pib, prefilter, sort_notam

IGNORE, VIA, AWARE, READER = SortAction.IGNORE, SortAction.VIA_AIRSPACE, SortAction.AWARENESS, SortAction.TO_READER
START, END = SPAN


def run(notam, span=SPAN, rules=None) -> SortResult:
    return sort_notam(notam, span, LZ) if rules is None else sort_notam(notam, span, LZ, rules=rules)


# (row name, notam, expected action, text the reason must contain)
ROWS = [
    # --- step 1: every IGNORE reason, one row each -----------------------------------------------------------
    ("not active: B after the span end", parsed(valid_from=END), IGNORE, "not active in the span"),
    ("not active: C before the span start", parsed(valid_from=START - timedelta(days=2), valid_to=START), IGNORE, "not active in the span"),
    ("not active: D has no time in the span", parsed(item_d="0000-0100"), IGNORE, "item D"),
    ("traffic I only", parsed(traffic="I"), IGNORE, "IFR only"),
    ("purpose K", parsed(purpose="K"), IGNORE, "checklist"),
    ("cancelled: a NOTAMC", parsed(type=NotamType.CANCEL, replaces="H100/26"), IGNORE, "cancelling H100/26"),
    ("series V, subject RO", parsed("V34/26", subject="RO", radius_nm=999.0), IGNORE, "security advice"),
    ("subject L (lighting)", parsed("L1/26", subject="LX"), IGNORE, "cannot affect a paramotor"),
    ("subject F (aerodrome facilities)", parsed("L1/26", subject="FU", item_e="AVGAS NOT AVBL"), IGNORE, "cannot affect a paramotor"),
    ("subject M (movement area)", parsed("L1/26", subject="MR"), IGNORE, "cannot affect a paramotor"),
    ("subject N (nav aids)", parsed("L1/26", subject="NB"), IGNORE, "cannot affect a paramotor"),
    ("subject I (instrument approach)", parsed("L1/26", subject="IL"), IGNORE, "cannot affect a paramotor"),
    ("subject C (comms)", parsed("L1/26", subject="CA"), IGNORE, "cannot affect a paramotor"),
    ("subject S (ATS services)", parsed("L1/26", subject="SE"), IGNORE, "cannot affect a paramotor"),
    ("subject O (other, OA)", parsed("U6768/26", subject="OA", radius_nm=999.0), IGNORE, "cannot affect a paramotor"),
    ("AF at 50 NM is FIR-wide admin", parsed("L1/26", subject="AF", radius_nm=50.0), IGNORE, "FIR-wide AIS admin"),
    (
        "Q172/26: series Q, AF, 999 NM, MIL AIP checklist",
        parsed("Q172/26", subject="AF", radius_nm=999.0, purpose="NBO", item_e="UK MIL AIP 1226 AMENDMENT RELEASED"),
        IGNORE,
        "FIR-wide AIS admin",
    ),
    # --- kept exceptions to the subject-letter list -------------------------------------------------------------
    ("FA (drones, fireworks, lasers) is kept", parsed("L1/26", subject="FA"), READER, ""),
    ("OB (obstacle) is kept", parsed("L1/26", subject="OB"), READER, ""),
    ("OL (obstacle lights) is kept", parsed("L1/26", subject="OL"), READER, ""),
    ("AF below 50 NM is kept", parsed("L1/26", subject="AF", radius_nm=49.0), READER, ""),
    ("A subjects are kept", parsed("L1/26", subject="AX"), READER, ""),
    ("RO outside series V is not security advice", parsed("L1/26", subject="RO"), READER, ""),
    ("W subjects are kept", parsed("L1/26", subject="WE"), READER, ""),
    ("series V with another subject (volcanic ash) is kept", parsed("V1/26", subject="WV", radius_nm=10.0), READER, ""),
    ("series Q is not ignored", parsed("Q99/26", subject="WU"), READER, ""),
    ("traffic IV is kept", parsed(traffic="IV"), READER, ""),
    ("traffic V is kept", parsed(traffic="V"), READER, ""),
    ("a NOTAMR itself is a live NOTAM", parsed(type=NotamType.REPLACE, replaces="H100/26"), READER, ""),
    # --- step 2: restrictions --------------------------------------------------------------------------------------
    (
        "RD with EGD496A in E -> via airspace",
        parsed("J3857/26", subject="RD", radius_nm=10.0, item_e="TEMPO DANGER AREA (TDA) EGD496A ACTIVATED."),
        VIA,
        "EGD496A",
    ),
    ("RD with the designator spelt D 496A", parsed("J3864/26", subject="RD", radius_nm=10.0, item_e="TDA D 496A ACTIVATED"), VIA, "EGD496A"),
    ("RR with a designator -> via airspace", parsed("L1/26", subject="RR", item_e="RESTRICTED AREA EGR106 ACTIVE"), VIA, "EGR106"),
    ("RP with a designator -> via airspace", parsed("L1/26", subject="RP", item_e="PROHIBITED AREA EGP108 ACTIVE"), VIA, "EGP108"),
    ("RT with a designator -> via airspace", parsed("L1/26", subject="RT", item_e="TRIGGER EGD323A"), VIA, "EGD323A"),
    (
        "J1458/26: RT, Duxford RA(T), no designator -> reader",
        parsed("J1458/26", subject="RT", item_e="RESTRICTED AREA (TEMPORARY) ACT WI 2.2NM RADIUS OF PSN 520528N 0001055E (DUXFORD)"),
        READER,
        "no designator",
    ),
    ("RR without a designator -> reader", parsed("L1/26", subject="RR", item_e="RESTRICTED AREA WI 5NM OF PSN"), READER, "no designator"),
    ("TDA trigger named only by place -> reader", parsed("L1/26", subject="RT", item_e="TDA TRIGGER NOTAM FOR LEUCHARS"), READER, "no designator"),
    # --- step 3: jamming -----------------------------------------------------------------------------------------
    ("B1104/26: GW jamming near Silverstone -> awareness", parsed("B1104/26", subject="GW", radius_nm=25.0, item_e="GNSS INTERFERENCE"), AWARE, "jamming"),
    # --- step 4: area-wide -----------------------------------------------------------------------------------------
    ("H7297/26: 999 NM pipeline inspection -> awareness", parsed("H7297/26", subject="WY", radius_nm=999.0, traffic="V", purpose="M"), AWARE, "area-wide"),
    ("radius exactly 50 NM -> awareness", parsed("L1/26", subject="WX", radius_nm=50.0), AWARE, "area-wide"),
    ("radius just under 50 NM -> reader", parsed("L1/26", subject="WX", radius_nm=49.9), READER, ""),
    # --- step 5: the reader -----------------------------------------------------------------------------------------
    ("H7171/26: UAS, W subject -> reader", parsed("H7171/26", subject="WU", item_d="0800-1800"), READER, ""),
    (
        "I5452/26: FA, UAS at the Newcastle RVI -> reader",
        parsed("I5452/26", subject="FA", item_e="UAS OPS AT THE ROYAL VICTORIA INFIRMARY, NEWCASTLE"),
        READER,
        "",
    ),
    ("FA 'NO OUT OF HOURS MOVEMENTS' -> reader", parsed("L1/26", subject="FA", item_e="NO OUT OF HOURS MOVEMENTS"), READER, ""),
    ("series Q, FA -> reader, not ignored for series", parsed("Q200/26", subject="FA", item_e="LASER ACTIVITY"), READER, ""),
    ("L2714/26: OB crane -> reader", parsed("L2714/26", subject="OB", item_e="CRANE 150FT AGL"), READER, ""),
    (
        "U7871/26: AF, temporary TMZ at Fairford -> reader",
        parsed("U7871/26", subject="AF", radius_nm=10.0, item_e="TEMPORARY TRANSPONDER MANDATORY ZONE"),
        READER,
        "",
    ),
    # --- order: first match wins; each row matches two or more steps ------------------------------------------------------
    ("order: inactive beats traffic I", parsed(valid_from=END, traffic="I"), IGNORE, "not active in the span"),
    ("order: traffic I beats purpose K", parsed(traffic="I", purpose="K"), IGNORE, "IFR only"),
    ("order: purpose K beats a NOTAMC", parsed(purpose="K", type=NotamType.CANCEL), IGNORE, "checklist"),
    ("order: a NOTAMC beats series V RO", parsed("V34/26", subject="RO", type=NotamType.CANCEL, replaces="V30/26"), IGNORE, "NOTAMC"),
    ("order: series V RO beats the subject-letter list", parsed("V34/26", subject="RO", radius_nm=999.0), IGNORE, "security advice"),
    ("order: IGNORE beats restriction (traffic I with a designator)", parsed(subject="RD", traffic="I", item_e="EGD496A"), IGNORE, "IFR only"),
    ("order: restriction with designator beats area-wide", parsed(subject="RD", radius_nm=999.0, item_e="EGD496A"), VIA, "EGD496A"),
    ("order: restriction without designator beats area-wide", parsed(subject="RR", radius_nm=999.0, item_e="NO NAME"), READER, "no designator"),
    ("order: jamming beats area-wide (the reason names jamming)", parsed(subject="GW", radius_nm=999.0), AWARE, "jamming"),
    ("order: AF 999 NM is ignored, not sent to awareness", parsed(subject="AF", radius_nm=999.0), IGNORE, "FIR-wide AIS admin"),
]
IGNORE_ROWS = [row for row in ROWS if row[2] is IGNORE]


@pytest.mark.parametrize(("name", "notam", "action", "reason_has"), ROWS, ids=[r[0] for r in ROWS])
def test_the_sort_row(name, notam, action, reason_has):
    result = run(notam)
    assert result.action is action, (name, result)
    assert result.reason.strip() and "\n" not in result.reason
    assert reason_has in result.reason


@pytest.mark.parametrize(("name", "notam", "action", "reason_has"), IGNORE_ROWS, ids=[r[0] for r in IGNORE_ROWS])
def test_every_ignore_carries_a_one_line_reason(name, notam, action, reason_has):
    result = run(notam)
    assert result.reason and "\n" not in result.reason and len(result.reason) > len("IGNORE")


def test_via_airspace_carries_the_designators_it_found():
    result = run(parsed("J3857/26", subject="RD", radius_nm=10.0, item_e="TDA EGD496A ACTIVATED. ALSO D323A AND EGD 323B."))
    assert result.action is VIA
    assert result.designators == ["EGD496A", "EGD323A", "EGD323B"]


def test_only_via_airspace_carries_designators():
    assert run(parsed(subject="WU", item_e="EGD496A MENTIONED")).designators == []


def test_the_security_series_comes_from_rules_yaml():
    q_security = parsed("Q34/26", subject="RO", radius_nm=999.0)
    assert run(q_security).action is not IGNORE  # RO outside series V is not security advice
    assert run(q_security, rules=with_rules(notam_security_series=["Q"])).action is IGNORE
    v_security = parsed("V34/26", subject="RO", radius_nm=999.0)
    assert run(v_security, rules=with_rules(notam_security_series=["Q"])).action is not IGNORE


def test_no_series_letter_is_ignored_on_its_own():
    # Series Q is not ignored (UK military NOTAMs are series Q from 1 Nov 2026, U6768/26). V is judged by subject.
    for series in "ABCDEGHIJLMNQRU":
        assert run(parsed(f"{series}1/26", subject="WU")).action is READER, series


# --- activity in the span ------------------------------------------------------------------------------------


def test_d_active_in_the_span_goes_on_to_the_other_steps():
    assert run(parsed(item_d="0800-1800")).action is READER
    assert run(parsed(item_d="2300-0600")).action is IGNORE  # the span is 07:00-18:00
    assert run(parsed(item_d="H24")).action is READER


def test_d_is_judged_inside_b_to_c_and_the_span():
    notam = parsed(item_d="0800-1800", valid_from=utc(31, 17, 30), valid_to=utc(1, 12, month=11))
    assert run(notam).action is READER  # 17:30-18:00 is inside the span
    assert run(notam, span=(utc(31, 7), utc(31, 17, 30))).action is IGNORE  # B is the span end


def test_a_perm_notam_with_a_schedule_is_judged_in_the_span():
    notam = parsed(item_d="0800-1800", valid_from=utc(1, 0), valid_to=None, permanent=True)
    assert run(notam).action is READER
    assert run(notam, span=(utc(31, 20), utc(31, 22))).action is IGNORE


def test_an_unparseable_d_item_is_active_for_all_of_b_to_c_and_the_note_is_carried():
    notam = parsed(item_d="WHEN THE MOOD TAKES US", valid_from=utc(1, 0), valid_to=utc(1, 0, month=12))
    result = run(notam)
    assert result.action is READER
    assert result.schedule_note is not None and "WHEN THE MOOD TAKES US" in result.schedule_note
    assert "treated as active" in result.schedule_note


def test_the_schedule_note_is_carried_on_later_decisions_too():
    result = run(parsed(subject="GW", item_d="WHEN THE MOOD TAKES US"))
    assert (result.action, result.schedule_note is not None) == (AWARE, True)
    result = run(parsed(subject="LX", item_d="WHEN THE MOOD TAKES US"))
    assert (result.action, result.schedule_note is not None) == (IGNORE, True)
    result = run(parsed(subject="RD", item_e="EGD496A", item_d="WHEN THE MOOD TAKES US"))
    assert (result.action, result.schedule_note is not None) == (VIA, True)
    result = run(parsed(subject="RR", item_e="NO NAME", item_d="WHEN THE MOOD TAKES US"))
    assert (result.action, result.schedule_note is not None) == (READER, True)
    result = run(parsed(subject="WY", radius_nm=999.0, item_d="WHEN THE MOOD TAKES US"))
    assert (result.action, result.schedule_note is not None) == (AWARE, True)


def test_a_readable_d_item_has_no_note():
    assert run(parsed(item_d="0800-1800")).schedule_note is None


def test_a_span_is_half_open_for_b_and_c():
    assert run(parsed(valid_from=END)).action is IGNORE  # B at the span end
    assert run(parsed(valid_from=END - timedelta(minutes=1))).action is READER
    assert run(parsed(valid_from=START - timedelta(days=1), valid_to=START)).action is IGNORE  # C at the span start
    assert run(parsed(valid_from=START - timedelta(days=1), valid_to=START + timedelta(minutes=1))).action is READER


# --- rules.yaml drives the lists --------------------------------------------------------------------------------


def test_changing_the_area_wide_radius_changes_the_result():
    notam = parsed(subject="WX", radius_nm=60.0)
    assert run(notam).action is AWARE
    assert run(notam, rules=with_rules(notam_area_wide_nm=70)).action is READER
    assert run(parsed(subject="AF", radius_nm=60.0)).action is IGNORE
    assert run(parsed(subject="AF", radius_nm=60.0), rules=with_rules(notam_area_wide_nm=70)).action is READER


def test_changing_a_subject_list_changes_the_result():
    assert run(parsed(subject="LX")).action is IGNORE
    assert run(parsed(subject="LX"), rules=with_rules(notam_ignore_subject_letters=["F"])).action is READER
    assert run(parsed(subject="OB")).action is READER
    assert run(parsed(subject="OB"), rules=with_rules(notam_keep_subjects=[])).action is IGNORE
    assert run(parsed(subject="GW")).action is AWARE
    assert run(parsed(subject="GW"), rules=with_rules(notam_awareness_subjects=["GX"])).action is READER


# --- the whole chain on the real (trimmed) PIB ------------------------------------------------------------------------


def test_the_trimmed_pib_sorts_as_the_31_oct_notams_do():
    _issued, notams = parse_pib(TRIMMED.read_bytes())
    span = (utc(31, 7), utc(7, 7, month=11))
    near, _far = prefilter(notams, LZ, span)
    sorted_ = {n.id: sort_notam(n, span, LZ) for n in near}
    assert {k: v.action for k, v in sorted_.items()} == {
        "H7171/26": READER,
        "H7297/26": AWARE,
        "J3857/26": VIA,
        "J3864/26": VIA,
        "Q172/26": IGNORE,
        "V34/26": IGNORE,
    }
    assert sorted_["J3857/26"].designators == ["EGD496A"]
    assert sorted_["J3864/26"].designators == ["EGD496A"]
    assert "AF" in sorted_["Q172/26"].reason
    assert "series V" in sorted_["V34/26"].reason


# --- G16: an area drawn in item E replaces the area-wide cap (step 4 only) -----------------------------------------

SQUARE = "551000N 0013000W - 551000N 0012000W - 550500N 0012000W - 550500N 0013000W"  # fiction, ~80 NM north of the default Q centre
DRAWN = {
    "polygon": f"EXERCISE WI AREA BOUNDED BY {SQUARE}. SFC-UNL.",
    "circle": "EXERCISE WI 5NM RADIUS OF PSN 551000N 0013000W. SFC-UNL.",
    "broken": "EXERCISE WI 5NM RADIUS OF PSN 551000N. SFC-UNL.",  # a latitude with no longitude
}
TRANSIT = "TRANSIT\n551000N 0013000W 0900\n551000N 0012000W 0910\n550500N 0012000W 0920"


@pytest.mark.parametrize(("kind", "reason"), [("polygon", "draws a polygon"), ("circle", "draws a circle"), ("broken", "Q-circle is used")])
def test_g16_an_area_wide_notam_that_draws_an_area_or_tried_to_goes_to_the_reader_not_awareness(kind, reason):
    result = run(parsed(subject="WE", radius_nm=152.0, item_e=DRAWN[kind]))
    assert result.action is READER
    assert reason in result.reason and "area-wide" in result.reason


@pytest.mark.parametrize("item_e", [TRANSIT, "EXERCISE SFC-UNL. FOR INFO CTC 07700 900123.", ""], ids=["route", "no coordinates", "empty"])
def test_g16_a_route_or_no_area_keeps_the_area_wide_rule(item_e):
    result = run(parsed(subject="WE", radius_nm=152.0, item_e=item_e))
    assert result.action is AWARE
    assert result.reason == "radius 152 NM is area-wide"


def test_g16_a_drawn_area_changes_nothing_below_the_area_wide_radius():
    for item_e in DRAWN.values():
        assert run(parsed(subject="WE", radius_nm=49.0, item_e=item_e)).action is READER


@pytest.mark.parametrize("kind", list(DRAWN))
def test_g16_af_admin_and_series_v_stay_ignored_with_a_parseable_area(kind):
    """Step 1 runs first: those are FIR-wide admin and security advice, which no drawn area makes relevant."""
    assert run(parsed("L1/26", subject="AF", radius_nm=60.0, item_e=DRAWN[kind])).action is IGNORE
    assert run(parsed("V34/26", subject="RO", radius_nm=999.0, item_e=DRAWN[kind])).action is IGNORE


def test_g16_a_gnss_jamming_notam_is_still_awareness_whatever_it_draws():
    assert run(parsed(subject="GW", radius_nm=152.0, item_e=DRAWN["polygon"])).action is AWARE
