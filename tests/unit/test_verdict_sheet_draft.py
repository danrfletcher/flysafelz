"""G3 / G4 / CQ2 / F4: the draft pieces of the verdict sheet, over hand-built Segments and Assessments.

why-joining and the one-line comment, the `unread` tag, the worst-action table (every pair of the seven actions),
the choice of ignored NOTAMs by id, and the rendered text (it parses back to the same entry, and the comments add
no key). Where a test needs assess()'s own output (a reading's polygon outside 2 NM) it runs the real assess().
"""

import itertools
from datetime import UTC, datetime

import pytest
import yaml
from factories import circle_at, notam, pib_result, reading, run_assess, square_at, utc

from flysafelz.contracts import (
    Assessment,
    Blocker,
    Caution,
    HourAssessment,
    IgnoredNotam,
    NotamAction,
    NotamActionResult,
    Segment,
    Tier,
    Verdict,
)
from flysafelz.verdict_sheet import local_notam_ids
from flysafelz.verdict_sheet_draft import (
    IGNORED_HEADER,
    NO_WHY,
    SEVERITY,
    UNREAD_NOTE,
    DraftEntry,
    draft_actions,
    fill_rows,
    ignored_by_code,
    one_line,
    render_entry,
    unread_ids,
    why_of,
    why_text,
    worst_action,
)

A = NotamAction
HOUR = datetime(2026, 10, 31, 9, tzinfo=UTC)


def result(notam_id: str, action: NotamAction) -> NotamActionResult:
    return NotamActionResult(notam_id=notam_id, action=action, reason="because")


def seg(
    verdict: Verdict = Verdict.PILOT_DISCRETION,
    *,
    start: datetime = HOUR,
    end: datetime = datetime(2026, 10, 31, 10, tzinfo=UTC),
    blockers: tuple[str, ...] = (),
    cautions: tuple[str, ...] = (),
    conditions: tuple[str, ...] = (),
    actions: tuple[NotamActionResult, ...] = (),
    approaching: bool = False,
) -> Segment:
    return Segment(
        start=start,
        end=end,
        verdict=verdict,
        approaching_limits=approaching,
        conditions=list(conditions),
        cautions=[Caution(rule_id="r", text=t) for t in cautions],
        blockers=[Blocker(source="weather", text=t) for t in blockers],
        notam_actions=list(actions),
    )


def assessment(*segments: Segment, ignored: tuple[IgnoredNotam, ...] = (), tier: Tier = Tier.VERDICT) -> Assessment:
    """One VERDICT hour per segment, so any segments can be given (their own start and end are what the rows use)."""
    hours = [HourAssessment(hour=s.start.replace(minute=0), tier=tier, segments=[s]) for s in segments]
    worst = None if tier is not Tier.VERDICT else Verdict.PILOT_DISCRETION
    return Assessment(window_start=HOUR, window_end=HOUR, hours=hours, overall_verdict=worst, overall_tier=tier, ignored_notams=list(ignored))


# --- G3: why --------------------------------------------------------------------------------------


def test_g3_why_joins_blockers_then_cautions_then_conditions_with_semicolons_in_that_order():
    s = seg(blockers=("b1", "b2"), cautions=("c1",), conditions=("UNVERIFIED: weather is missing", "fly high"))
    assert why_text(s) == "b1; b2; c1; UNVERIFIED: weather is missing; fly high"


def test_g3_a_segment_with_nothing_to_say_reads_the_fixed_text():
    assert why_text(seg()) == NO_WHY == "no blockers, cautions or conditions"


def test_g3_why_comes_from_the_text_fields_not_the_rule_id_or_source():
    s = seg(blockers=("wind too high",))
    assert "weather" not in why_text(s) and "r" != why_text(s)


@pytest.mark.parametrize(
    ("raw", "shown"),
    [
        ("NOTAM H7171/26: drone ops # not a key", "NOTAM H7171/26: drone ops # not a key"),
        ('say "yes" and \'no\'', 'say "yes" and \'no\''),
        ("line one\nline two\r\nline three", "line one line two line three"),
        ("tab\tsplit  and   runs", "tab split and runs"),
        ("ceiling 800 ft \u00b0 and \u2013 dash \u00e9", "ceiling 800 ft \u00b0 and \u2013 dash \u00e9"),
        ("nel\u0085ls\u2028ps\u2029bom\ufeffnul\x00end", "nel ls ps bom nul end"),
    ],
)
def test_one_line_collapses_breaks_and_keeps_text_and_non_ascii(raw, shown):
    assert one_line(raw) == shown and "\n" not in one_line(raw)


def awkward_entry(why: str) -> DraftEntry:
    s = seg(blockers=(why,))
    a = assessment(s)
    rows = fill_rows([{"start": "2026-10-31T09:00:00Z", "end": "2026-10-31T10:00:00Z", "edge": "window start", "verdict": None, "approaching_limits": None}], a)
    entry = DraftEntry({"id": "x", "mode": "draft", "window": "default", "actions_done": False, "notam_actions": {}, "rows": rows})
    entry.ignored = [("A1/26", "reason with # and : and \"quotes\"\nand a newline and \u00b0")]
    return entry


@pytest.mark.parametrize("why", ["NOTAM H7171/26: avoid #1 \"sector\"", "up\nand\ndown", "wind 9 mph \u00b0 \u2013 caf\u00e9", "- starts like a list: yes", "key: [flow, {mapping}]", "a \x00 b"])
def test_f4_an_awkward_why_stays_one_comment_line_the_file_parses_and_no_row_gains_a_key(why):
    entry = awkward_entry(why)
    text = render_entry(entry)
    (parsed,) = yaml.safe_load(text)
    assert parsed == dict(entry) and set(parsed["rows"][0]) == {"start", "end", "edge", "verdict", "approaching_limits"}
    comments = [line for line in text.splitlines() if "# why:" in line]
    assert len(comments) == 1 and comments[0].startswith("    # why: ") and len(text.splitlines()) == text.count("\n")
    assert not [k for k in parsed if any(w in k for w in ("why", "reason", "ignored"))]


# --- G3: unread -----------------------------------------------------------------------------------------


def test_g3_unverified_with_unread_notams_gets_them_sorted():
    s = seg(Verdict.UNVERIFIED, actions=(result("J2/26", A.UNREAD_OVER_LZ), result("H9/26", A.AVOID_SECTOR), result("B1/26", A.UNREAD_OVER_LZ)))
    assert unread_ids(s) == ["B1/26", "J2/26"]


@pytest.mark.parametrize("verdict", [Verdict.NO_GO, Verdict.PILOT_DISCRETION])
def test_g3_other_verdicts_never_get_unread_even_with_an_unread_action(verdict):
    assert unread_ids(seg(verdict, actions=(result("J2/26", A.UNREAD_OVER_LZ),))) == []


def test_g3_unverified_without_an_unread_action_gets_nothing():
    assert unread_ids(seg(Verdict.UNVERIFIED, conditions=("UNVERIFIED: weather is missing",), actions=(result("H9/26", A.CONTACT_FIRST),))) == []
    assert unread_ids(seg(Verdict.UNVERIFIED)) == []


def test_g3_fill_rows_takes_the_segments_own_verdict_flag_and_tag_from_one_assessment():
    unread = seg(Verdict.UNVERIFIED, start=HOUR, end=datetime(2026, 10, 31, 9, 30, tzinfo=UTC), actions=(result("J1/26", A.UNREAD_OVER_LZ),), conditions=("UNVERIFIED: unread",))
    plain = seg(Verdict.PILOT_DISCRETION, start=datetime(2026, 10, 31, 9, 30, tzinfo=UTC), end=datetime(2026, 10, 31, 10, tzinfo=UTC), approaching=True)
    a = Assessment(
        window_start=HOUR,
        window_end=HOUR,
        hours=[HourAssessment(hour=HOUR, tier=Tier.VERDICT, segments=[unread, plain])],
        overall_verdict=Verdict.UNVERIFIED,
        overall_tier=Tier.VERDICT,
    )
    blind = [
        {"start": "2026-10-31T09:00:00Z", "end": "2026-10-31T09:30:00Z", "edge": "window start", "verdict": None, "approaching_limits": None},
        {"start": "2026-10-31T09:30:00Z", "end": "2026-10-31T10:00:00Z", "edge": "x", "verdict": None, "approaching_limits": None},
    ]
    first, second = fill_rows(blind, a)
    assert (first["verdict"], first["approaching_limits"], first["unread"], why_of(first)) == ("UNVERIFIED", False, ["J1/26"], "UNVERIFIED: unread")
    assert (second["verdict"], second["approaching_limits"], "unread" in second, why_of(second)) == ("PILOT_DISCRETION", True, False, NO_WHY)
    assert blind[0]["verdict"] is None  # the blind rows given are not changed


def test_g5_tier_rows_stay_tier_only_and_take_no_segment():
    tier_row = {"start": "2026-11-03T09:00:00Z", "end": "2026-11-03T10:00:00Z", "edge": "hour", "tier": "OUTLOOK"}
    a = assessment(seg(blockers=("b",)))
    (kept, row) = fill_rows([tier_row, {"start": "2026-10-31T09:00:00Z", "end": "2026-10-31T10:00:00Z", "edge": "x", "verdict": None, "approaching_limits": None}], a)
    assert kept is tier_row and kept == {"start": "2026-11-03T09:00:00Z", "end": "2026-11-03T10:00:00Z", "edge": "hour", "tier": "OUTLOOK"}
    assert why_of(kept) is None and why_of(row) == "b"
    entry = DraftEntry({"id": "x", "rows": [kept], "notam_actions": {}})
    assert "# why" not in render_entry(entry)


def test_fill_rows_refuses_rows_that_are_not_this_assessments_segments():
    a = assessment(seg())
    wrong = {"start": "2026-10-31T09:00:00Z", "end": "2026-10-31T09:45:00Z", "edge": "x", "verdict": None, "approaching_limits": None}
    with pytest.raises(ValueError, match="not a segment"):
        fill_rows([wrong], a)
    with pytest.raises(ValueError, match="leave out a segment"):
        fill_rows([], a)
    with pytest.raises(ValueError, match="not a segment"):
        fill_rows([{**wrong, "end": "2026-10-31T10:00:00Z"}] * 2, a)


# --- G4: the worst action ------------------------------------------------------------------------------------


def test_g4_severity_order_is_the_plans():
    assert SEVERITY == (A.NO_FLY, A.UNREAD_OVER_LZ, A.CHECK_SUP, A.CONTACT_FIRST, A.AVOID_SECTOR, A.AWARENESS, A.AWARENESS_MAP, A.IGNORE)
    assert set(SEVERITY) == set(NotamAction)


@pytest.mark.parametrize(("first", "second"), list(itertools.product(NotamAction, repeat=2)))
def test_g4_worst_action_of_every_pair_of_the_eight_actions(first, second):
    expected = first if SEVERITY.index(first) <= SEVERITY.index(second) else second
    assert worst_action(first, second) is expected
    assert worst_action(second, first) is expected


def test_g4_check_sup_is_below_unread_over_lz_and_above_contact_first():
    assert worst_action(A.CHECK_SUP, A.UNREAD_OVER_LZ) is A.UNREAD_OVER_LZ
    assert worst_action(A.CHECK_SUP, A.NO_FLY) is A.NO_FLY
    assert worst_action(A.CHECK_SUP, A.CONTACT_FIRST) is A.CHECK_SUP
    assert worst_action(A.CONTACT_FIRST, A.CHECK_SUP) is A.CHECK_SUP
    assert SEVERITY.index(A.CHECK_SUP) == SEVERITY.index(A.UNREAD_OVER_LZ) + 1 == SEVERITY.index(A.CONTACT_FIRST) - 1


def test_g4_a_segment_s_worst_action_picks_check_sup_over_contact_first_and_lists_it_in_severity_order():
    a = assessment(
        seg(start=HOUR, actions=(result("B2/26", A.CONTACT_FIRST), result("C3/26", A.CHECK_SUP), result("D4/26", A.AVOID_SECTOR))),
        seg(start=datetime(2026, 10, 31, 10, tzinfo=UTC), end=datetime(2026, 10, 31, 11, tzinfo=UTC), actions=(result("B2/26", A.CHECK_SUP), result("A1/26", A.UNREAD_OVER_LZ))),
    )
    listed = draft_actions(a)
    assert listed == {"A1/26": "UNREAD_OVER_LZ", "B2/26": "CHECK_SUP", "C3/26": "CHECK_SUP", "D4/26": "AVOID_SECTOR"}
    assert list(listed) == ["A1/26", "B2/26", "C3/26", "D4/26"]


def test_g4_each_notam_is_listed_once_with_its_worst_action_in_severity_order_then_id():
    a = assessment(
        seg(start=HOUR, actions=(result("Z1/26", A.AWARENESS), result("B2/26", A.CONTACT_FIRST), result("A3/26", A.AVOID_SECTOR))),
        seg(start=datetime(2026, 10, 31, 10, tzinfo=UTC), end=datetime(2026, 10, 31, 11, tzinfo=UTC), actions=(result("B2/26", A.NO_FLY), result("A3/26", A.AWARENESS_MAP), result("C4/26", A.UNREAD_OVER_LZ), result("A0/26", A.AVOID_SECTOR))),
    )
    listed = draft_actions(a)
    assert listed == {"B2/26": "NO_FLY", "C4/26": "UNREAD_OVER_LZ", "A0/26": "AVOID_SECTOR", "A3/26": "AVOID_SECTOR", "Z1/26": "AWARENESS"}
    assert list(listed) == ["B2/26", "C4/26", "A0/26", "A3/26", "Z1/26"]


def test_g4_a_notam_contact_first_in_one_segment_and_ignore_in_another_is_listed_once_as_contact_first():
    a = assessment(
        seg(actions=(result("J1/26", A.IGNORE),)),
        seg(start=datetime(2026, 10, 31, 10, tzinfo=UTC), end=datetime(2026, 10, 31, 11, tzinfo=UTC), actions=(result("J1/26", A.CONTACT_FIRST),)),
    )
    assert draft_actions(a) == {"J1/26": "CONTACT_FIRST"}
    assert list(draft_actions(assessment(seg(actions=(result("J1/26", A.CONTACT_FIRST),)), seg(actions=(result("J1/26", A.IGNORE),)))).items()) == [("J1/26", "CONTACT_FIRST")]


def test_g4_only_ignore_means_nothing_is_listed():
    assert draft_actions(assessment(seg(actions=(result("J1/26", A.IGNORE),)))) == {}
    assert draft_actions(assessment(seg())) == {}


def test_g4_tier_only_hours_add_no_actions():
    assert draft_actions(assessment(seg(actions=(result("J1/26", A.NO_FLY),)), tier=Tier.OUTLOOK)) == {}


# --- G4: the ignored block, chosen by id -------------------------------------------------------------------


def ig(notam_id: str, reason: str = "r") -> IgnoredNotam:
    return IgnoredNotam(notam_id=notam_id, reason=reason)


def test_g4_the_block_has_the_local_ids_only_sorted_with_their_reason_and_never_a_superseded_one():
    ignored = [ig("V32/26", "series V subject RO"), ig("Z9/26", "far"), ig("A1/26", "subject OA cannot affect a paramotor"), ig("S5/26", "replaced by S6/26"), ig("J3864/26", "starts later")]
    block = ignored_by_code(ignored, {"V32/26", "A1/26", "J3864/26", "S5/26"}, [ig("S5/26")])
    assert block == [("A1/26", "subject OA cannot affect a paramotor"), ("J3864/26", "starts later"), ("V32/26", "series V subject RO")]


def test_g4_selection_is_by_id_not_by_reason_text():
    # A reason that reads like a distance drop is kept when its id is local; one that reads like a time drop is not when its id is not.
    ignored = [ig("A1/26", "Q-circle edge is 9.00 NM from the LZ, beyond the 2 NM local area"), ig("B2/26", "starts 2026-11-01T08:00:00Z, at or after the window end")]
    assert ignored_by_code(ignored, {"A1/26"}, []) == [("A1/26", "Q-circle edge is 9.00 NM from the LZ, beyond the 2 NM local area")]


def test_g4_a_repeated_id_is_listed_once_with_its_first_reason():
    assert ignored_by_code([ig("A1/26", "first"), ig("A1/26", "second")], {"A1/26"}, []) == [("A1/26", "first")]


def test_g4_nothing_ignored_gives_an_empty_block():
    assert ignored_by_code([], {"A1/26"}, []) == [] and ignored_by_code([ig("A1/26")], set(), []) == []


def test_g4_a_notam_ignored_from_a_readings_own_polygon_beyond_2_nm_is_in_the_block_through_the_real_assess():
    near = notam("R1/26", circle=circle_at(1.0))  # its Q-circle is 1 NM away: it passes the prefilter
    far = notam("F1/26", circle=circle_at(5.0))  # its Q-circle is 5 NM away: the prefilter drops it
    pib = pib_result(near, far)
    result_ = run_assess(window=(utc(9), utc(12)), pib=pib, readings={"R1/26": reading(notam_id="R1/26", geometry=square_at(5.0))})
    reasons = {i.notam_id: i.reason for i in result_.ignored_notams}
    assert "R1/26" in reasons and "F1/26" in reasons and "local area" in reasons["R1/26"]
    block = ignored_by_code(result_.ignored_notams, local_notam_ids(pib, 53.802797, -0.086644), pib.superseded)
    assert [i for i, _ in block] == ["R1/26"]
    assert draft_actions(result_) == {}


def test_g4_a_notam_shown_in_a_segment_is_not_in_assess_ignored_list_so_not_in_the_block():
    shown = notam("S1/26", circle=circle_at(0.0))
    pib = pib_result(shown)
    result_ = run_assess(window=(utc(9), utc(12)), pib=pib, readings={"S1/26": reading(notam_id="S1/26")})
    assert "S1/26" in draft_actions(result_)
    assert "S1/26" not in [i.notam_id for i in result_.ignored_notams]
    assert ignored_by_code(result_.ignored_notams, local_notam_ids(pib, 53.802797, -0.086644), pib.superseded) == []


# --- the text ---------------------------------------------------------------------------------------------------


def entry_with(actions: dict[str, str], ignored: list[tuple[str, str]]) -> DraftEntry:
    entry = DraftEntry({"id": "x", "mode": "draft", "window": "default", "actions_done": False, "notam_actions": actions, "rows": []})
    entry.ignored = ignored
    return entry


def test_no_ignored_notam_writes_no_block_header_and_no_actions_writes_an_empty_mapping():
    text = render_entry(entry_with({"J1/26": "CONTACT_FIRST"}, []))
    assert IGNORED_HEADER not in text and "#" not in text
    assert "  notam_actions:\n    J1/26: CONTACT_FIRST\n  rows: []\n" in text
    assert "  notam_actions: {}\n  rows: []\n" in render_entry(entry_with({}, []))


def test_an_empty_action_list_is_followed_by_the_block_when_there_is_one():
    text = render_entry(entry_with({}, [("A1/26", "subject OA"), ("B2/26", "starts later")]))
    assert "  notam_actions: {}\n    " + IGNORED_HEADER + "\n    # A1/26: subject OA\n    # B2/26: starts later\n  rows: []\n" in text
    (parsed,) = yaml.safe_load(text)
    assert parsed["notam_actions"] == {} and set(parsed) == {"id", "mode", "window", "actions_done", "notam_actions", "rows"}


def test_q3_an_unread_value_is_written_and_its_line_ends_with_the_unread_comment():
    text = render_entry(entry_with({"J1/26": "UNREAD_OVER_LZ", "H2/26": "AVOID_SECTOR"}, []))
    assert f"    J1/26: UNREAD_OVER_LZ  {UNREAD_NOTE}\n" in text and UNREAD_NOTE == "# unread: read the NOTAM and set its action"
    assert "H2/26: AVOID_SECTOR\n" in text and text.count("# unread") == 1
    (parsed,) = yaml.safe_load(text)
    assert parsed["notam_actions"] == {"J1/26": "UNREAD_OVER_LZ", "H2/26": "AVOID_SECTOR"}


def test_the_block_comes_after_every_action_line_under_notam_actions_and_before_rows():
    text = render_entry(entry_with({"J1/26": "CONTACT_FIRST"}, [("A1/26", "x")]))
    assert text.index("J1/26") < text.index(IGNORED_HEADER) < text.index("# A1/26") < text.index("rows:")


def test_a_reason_is_one_comment_line_whatever_it_holds():
    text = render_entry(entry_with({}, [("A1/26", "reason: a # b\nsecond line \u00b0 \"q\"")]))
    assert "    # A1/26: reason: a # b second line \u00b0 \"q\"\n" in text
    (parsed,) = yaml.safe_load(text)
    assert parsed["notam_actions"] == {}


def test_a_row_with_unread_writes_the_ids_in_one_line_and_the_why_after_the_keys():
    s = seg(Verdict.UNVERIFIED, actions=(result("J1/26", A.UNREAD_OVER_LZ), result("B2/26", A.UNREAD_OVER_LZ)), conditions=("UNVERIFIED: x",))
    rows = fill_rows([{"start": "2026-10-31T09:00:00Z", "end": "2026-10-31T10:00:00Z", "edge": "window start", "verdict": None, "approaching_limits": None}], assessment(s))
    text = render_entry(DraftEntry({"id": "x", "notam_actions": {}, "rows": rows}))
    assert "    unread: [B2/26, J1/26]\n    # why: UNVERIFIED: x\n" in text
    assert text.endswith("# why: UNVERIFIED: x\n")
    (parsed,) = yaml.safe_load(text)
    assert parsed["rows"][0]["unread"] == ["B2/26", "J1/26"]
