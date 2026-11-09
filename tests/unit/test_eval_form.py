"""Form scores (5a PR-6.F1, G11): guard, decision_first, cited (two parts and composed), the assumed window, the re-check line, and the two safety checks
that G11 lists (day 3 with the BST to GMT change, and a dropped source), plus every dev example through all the evaluators against a fake passing run."""

import copy
from datetime import datetime
from types import SimpleNamespace

import pytest
from agent_eval.factories import answer, example, iso, row, seg, t, tier_row
from eval_form_helpers import DOCUMENT, dev, dev_records, passing_answer

from evals.agent.evaluators.aggregate import EVALUATORS, ScoredRun, score_run
from evals.agent.evaluators.answer import ToolCall
from evals.agent.evaluators.form import (
    CITED_KEY,
    DECISION_KEY,
    GUARD_KEY,
    RECHECK_KEY,
    WINDOW_KEY,
    cited,
    cited_guidance,
    cited_notams,
    decision_first,
    first_sentence,
    guard_pass,
    recheck_stated,
    verdict_words,
    window_stated,
)
from evals.agent.evaluators.safety import DROPPED_KEY, HORIZON_KEY, no_verdict_beyond_day3, unverified_on_dropped_source
from flysafelz.agent.answer import RECHECK_SENTENCE, recheck_line
from flysafelz.contracts import Verdict, display_verdict

# --- guard_pass ------------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(("result", "score"), [("passed", 1), ("retried", 0), ("fallback", 0)])
def test_the_guard_result_is_read_from_the_run(result, score):
    got = guard_pass(example(path="explain"), answer(path="explain", guard=result))
    assert got and got["key"] == GUARD_KEY and got["score"] == score and result in got["comment"]


def test_a_run_with_no_guard_result_gives_none_not_zero():
    assert guard_pass(example(path="explain"), answer(path="explain")) is None


# --- decision_first --------------------------------------------------------------------------------------------------------------

NOW, WINDOW = t(6, 30), (t(9), t(11))
ASSESS = example([row(t(9), t(11), "NO_GO")], window=WINDOW, now=NOW)


def first(text):
    return decision_first(ASSESS, answer([seg(t(9), t(11), "NO_GO")], text=text))


def test_the_words_come_from_display_verdict():
    assert verdict_words() == tuple(display_verdict(v, False, False) for v in Verdict) == ("NO-GO", "UNVERIFIED", "PILOT DISCRETION")


@pytest.mark.parametrize(
    "text",
    [
        "NO-GO until 11:00. Wind is above the limit.",
        "no-go: the CTR is active.",
        "UNVERIFIED: the weather source was unavailable. Check it yourself.",
        "PILOT DISCRETION – APPROACHING LIMITS from 09:00 to 11:00.",
        "PILOT   DISCRETION with gusts near the limit.",
        "NO-GO (wind above 6 mph at 10:00)\nMore detail follows.",
        "Verdict: NO-GO",
    ],
)
def test_a_verdict_word_in_the_first_sentence_passes(text):
    got = first(text)
    assert got and got["key"] == DECISION_KEY and got["score"] == 1


@pytest.mark.parametrize(
    "text",
    [
        "Wind is 12 mph. NO-GO.",
        "Conditions look fine at first. PILOT DISCRETION later.",
        "Good to go from 09:00.",
        "It is a no go today.",
        "NOGO",
    ],
)
def test_a_verdict_word_later_or_missing_fails(text):
    assert first(text)["score"] == 0  # type: ignore[index]


def test_a_full_stop_inside_a_number_does_not_end_the_sentence():
    assert first("Gusts of 4.5 mph make it PILOT DISCRETION.")["score"] == 1  # type: ignore[index]
    assert first_sentence("Gusts of 4.5 mph. Then more.") == "Gusts of 4.5 mph"


@pytest.mark.parametrize("text", ["", "   ", "\n\t "])
def test_an_empty_or_blank_answer_fails_on_the_assess_path(text):
    got = first(text)
    assert got and got["score"] == 0 and "(empty answer)" in got["comment"]


def test_an_outlook_only_window_gives_none():
    """The OUTLOOK rule: a window whose minutes are all OUTLOOK has no verdict to lead with."""
    outlook = example(tiers=[tier_row(t(6), t(20))], window=(t(6), t(20)), now=NOW)
    assert decision_first(outlook, answer(text="Nothing to say yet.")) is None


def test_dev_11_the_outlook_only_dev_case_gives_none_and_dev_13_scores():
    assert decision_first(dev("dev_11_west_sleekburn"), answer(text="It will probably be fine.")) is None
    assert decision_first(dev("dev_13_bamburgh"), answer(text="It will probably be fine."))["score"] == 0  # type: ignore[index]


@pytest.mark.parametrize("path", ["explain", "clarify", "decline", "act"])
def test_decision_first_is_none_off_the_assess_path(path):
    assert decision_first(example(path=path), answer(path=path, text="NO-GO")) is None


def test_an_assess_example_with_no_sheet_entry_gives_none():
    assert decision_first(example(sheet=False, window=WINDOW), answer(text="NO-GO")) is None


# --- cited: two parts and composed -----------------------------------------------------------------------------------------------


def with_refs(**reference):
    return example(path="explain", extra=reference)


def guided(*docs, text):
    return answer(path="explain", text=text, tool_calls=[ToolCall("search_guidance", tuple(docs))])


def test_cited_notams_needs_every_notam_id_entry():
    ex = with_refs(must_mention=["H6217", "H6449/26", "Sherburn"])
    ok = cited_notams(ex, answer(path="explain", text="See h6217 and H6449 / 26."))
    assert ok and ok["score"] == 1
    miss = cited_notams(ex, answer(path="explain", text="See H6217 only."))
    assert miss and miss["score"] == 0 and "H6449/26" in miss["comment"] and "Sherburn" not in miss["comment"]


def test_cited_notams_ignores_entries_that_are_not_notam_ids():
    ex = with_refs(must_mention=["Sherburn", "ATZ", "1500", "Newcastle CTA 2", "CAP 1535"])
    assert cited_notams(ex, answer(path="explain", text="")) is None


def test_cited_notams_is_none_without_a_must_mention_list():
    assert cited_notams(with_refs(), answer(path="explain")) is None


def test_cited_guidance_needs_a_document_the_search_returned():
    ex = with_refs(required_tools=["search_guidance"])
    assert cited_guidance(ex, guided(DOCUMENT, text=f"{DOCUMENT} says so."))["score"] == 1  # type: ignore[index]
    assert cited_guidance(ex, guided(DOCUMENT, text="cap1535 says so."))["score"] == 1  # type: ignore[index]  # the mention normaliser


def test_cited_guidance_fails_for_a_document_the_search_did_not_return():
    got = cited_guidance(with_refs(required_tools=["search_guidance"]), guided(DOCUMENT, text="The Skyway Code says so."))
    assert got and got["score"] == 0 and DOCUMENT in got["comment"]


def test_cited_guidance_fails_when_the_search_returned_nothing_or_was_never_called():
    ex = with_refs(required_tools=["search_guidance"])
    assert cited_guidance(ex, guided(text=f"{DOCUMENT}"))["score"] == 0  # type: ignore[index]
    assert cited_guidance(ex, answer(path="explain", text=f"{DOCUMENT}"))["score"] == 0  # type: ignore[index]
    assert "no document" in cited_guidance(ex, guided(text="x"))["comment"]  # type: ignore[index]


def test_cited_guidance_is_none_without_search_guidance_in_required_tools():
    assert cited_guidance(with_refs(required_tools=["get_notams"]), guided(DOCUMENT, text=DOCUMENT)) is None
    assert cited_guidance(with_refs(), guided(DOCUMENT, text=DOCUMENT)) is None


def test_cited_is_one_when_every_part_that_applies_passes():
    ex = with_refs(must_mention=["H6217"], required_tools=["search_guidance"])
    got = cited(ex, guided(DOCUMENT, text=f"H6217 and {DOCUMENT}"))
    assert got and got["key"] == CITED_KEY and got["score"] == 1 and "H6217" in got["comment"] and DOCUMENT in got["comment"]


@pytest.mark.parametrize("text", ["H6217 only", f"{DOCUMENT} only", "neither"])
def test_cited_is_zero_when_either_part_fails(text):
    ex = with_refs(must_mention=["H6217"], required_tools=["search_guidance"])
    got = cited(ex, guided(DOCUMENT, text=text))
    assert got and got["score"] == 0


def test_cited_with_one_part_is_that_part():
    notams, guidance = with_refs(must_mention=["H6217"]), with_refs(required_tools=["search_guidance"])
    assert cited(notams, answer(path="explain", text="H6217"))["score"] == 1  # type: ignore[index]
    assert cited(notams, answer(path="explain", text="no"))["score"] == 0  # type: ignore[index]
    assert cited(guidance, guided(DOCUMENT, text=DOCUMENT))["score"] == 1  # type: ignore[index]
    assert cited(guidance, guided(DOCUMENT, text="no"))["score"] == 0  # type: ignore[index]


def test_cited_is_none_when_neither_part_applies():
    assert cited(with_refs(required_tools=["get_weather"], must_mention=["Sherburn"]), answer(path="explain", text="x")) is None
    assert cited(example(path="decline"), answer(path="decline")) is None


def test_the_cited_composition_agrees_with_its_parts_on_the_dev_cases():
    for ex in (dev("dev_16_buttermere"), dev("dev_24_sherburn"), dev("dev_04_alnwick_pastures"), dev("dev_23_blue_mill")):
        ans = passing_answer(ex)
        parts = [f for f in (cited_notams(ex, ans), cited_guidance(ex, ans)) if f]
        whole = cited(ex, ans)
        assert (whole is None) == (not parts) and (whole is None or whole["score"] == int(all(p["score"] for p in parts)))


def test_dev_24_and_dev_23_need_guidance_and_dev_16_needs_the_notam():
    assert cited(dev("dev_16_buttermere"), answer(path="explain", text="no id here"))["score"] == 0  # type: ignore[index]
    assert cited_guidance(dev("dev_24_sherburn"), answer(path="explain", text="x"))["score"] == 0  # type: ignore[index]
    assert cited(dev("dev_01_blue_mill"), answer(text="x")) is None


# --- assumed window stated -------------------------------------------------------------------------------------------------------


def stated(text, ex=ASSESS, **kw):
    return window_stated(ex, answer([seg(t(9), t(11), "NO_GO")], text=text, **kw))


def test_the_window_start_and_end_as_london_hh_mm_are_stated():
    got = stated("Window: 09:00 to 11:00.")
    assert got and got["key"] == WINDOW_KEY and got["score"] == 1 and "09:00-11:00" in got["comment"]


def test_the_match_goes_through_the_mention_normaliser():
    assert stated("From 09:00 until 11:00 GMT")["score"] == 1  # type: ignore[index]
    assert stated("between 9:00am and 11:00am")["score"] == 0  # type: ignore[index]


@pytest.mark.parametrize("text", ["until 11:00", "from 09:00", "NO-GO", ""])
def test_a_missing_start_or_end_fails_and_is_named(text):
    got = stated(text)
    assert got and got["score"] == 0 and "missing" in got["comment"]


def test_the_window_is_stated_in_london_time_not_utc():
    ex = dev("dev_01_blue_mill")  # 09:00Z to 11:00Z on 29 Sep: BST, so 10:00 to 12:00 in London
    assert window_stated(ex, answer(text="Window 10:00 to 12:00"))["score"] == 1  # type: ignore[index]
    assert window_stated(ex, answer(text="Window 09:00 to 11:00"))["score"] == 0  # type: ignore[index]


def test_the_window_is_stated_in_gmt_after_the_clocks_change():
    ex = dev("blue_mill_today")  # 31 Oct: GMT, London time is UTC
    assert window_stated(ex, answer(text="Window 07:00 to 18:00"))["score"] == 1  # type: ignore[index]


def test_window_stated_applies_to_both_run_kinds_and_only_to_assess_with_a_window():
    assert stated("09:00-11:00", kind="v0")["score"] == 1  # type: ignore[index]
    assert window_stated(example(path="explain"), answer(path="explain", text="09:00 11:00")) is None
    no_window = copy.deepcopy(ASSESS)
    del no_window["reference"]["window"]
    assert window_stated(no_window, answer(text="")) is None


def test_a_bad_window_raises_naming_the_example():
    bad = copy.deepcopy(ASSESS)
    bad["reference"]["window"] = [iso(t(11)), iso(t(9))]
    with pytest.raises(ValueError, match="syn_1"):
        window_stated(bad, answer(text=""))


# --- recheck_stated ---------------------------------------------------------------------------------------------------------------


def window_ex(now, start, end):
    return example([row(start, end, "NO_GO")], window=(start, end), now=now)


def recheck(ex, text, **kw):
    return recheck_stated(ex, answer([seg(t(9), t(11), "NO_GO")], text=text, **kw))


def test_a_window_that_starts_after_the_current_hour_needs_the_line():
    ex = window_ex(t(6, 30), t(9), t(11))
    got = recheck(ex, f"NO-GO. {RECHECK_SENTENCE}")
    assert got and got["key"] == RECHECK_KEY and got["score"] == 1
    missing = recheck(ex, "NO-GO until 11:00.")
    assert missing and missing["score"] == 0 and RECHECK_SENTENCE in missing["comment"]


def test_the_line_ticket_4_builds_is_the_one_checked_whether_or_not_it_has_a_fetch_time():
    ex = window_ex(t(6, 30), t(9), t(11))
    artifact = SimpleNamespace(window_start=t(9), source=SimpleNamespace(fetched_at=t(6, 20)))
    real = recheck_line([artifact], t(6, 30), "Europe/London", "assess")
    assert real and real.startswith("Based on data as of") and real.endswith(RECHECK_SENTENCE)
    assert recheck(ex, f"NO-GO. {real}")["score"] == 1  # type: ignore[index]


def test_the_boundary_is_the_end_of_the_current_hour():
    """At 06:59 a window from 07:00 starts after the current hour; at 07:00 it starts in it."""
    assert recheck(window_ex(t(6, 59), t(7), t(9)), "no line")["score"] == 0  # type: ignore[index]
    assert recheck(window_ex(t(7), t(7), t(9)), "no line") is None
    assert recheck(window_ex(t(6, 30), t(6, 45), t(9)), "no line") is None
    assert recheck(window_ex(t(6, 30), t(6, 30), t(9)), "no line") is None


def test_the_current_hour_is_the_london_hour():
    """23:30Z on 6 Aug is 00:30 on 7 Aug in London: a window from 00:00Z (01:00 London) starts after the current hour, one from 23:45Z does not."""
    now = t(23, 30, day=6, month=8)
    assert recheck(window_ex(now, t(0, day=7, month=8), t(3, day=7, month=8)), "no line")["score"] == 0  # type: ignore[index]
    assert recheck(window_ex(now, t(23, 45, day=6, month=8), t(3, day=7, month=8)), "no line") is None


def test_recheck_is_none_for_v0_non_assess_and_no_window():
    ex = window_ex(t(6, 30), t(9), t(11))
    assert recheck(ex, "no line", kind="v0") is None
    assert recheck_stated(example(path="explain"), answer(path="explain")) is None
    no_window = copy.deepcopy(ex)
    del no_window["reference"]["window"]
    assert recheck_stated(no_window, answer()) is None


# --- G11 no verdict beyond day 3 (E11) and the BST to GMT change -----------------------------------------------------------------


def horizon(now, start, end):
    return no_verdict_beyond_day3(example(now=now, window=(start, end)), answer([seg(start, end, "NO_GO")]))


def test_e11_across_the_clock_change_a_day_3_2359_london_minute_passes_and_day_4_fails():
    now = t(9, day=23, month=10)  # BST. Day 0 is 23 Oct; day 3 is 26 Oct, a GMT day, and it ends 2026-10-27 00:00Z
    at_2359 = horizon(now, t(23, 58, day=26, month=10), t(23, 59, day=26, month=10))
    assert at_2359 and at_2359["key"] == HORIZON_KEY and at_2359["score"] == 1
    assert horizon(now, t(23, 59, day=26, month=10), t(0, 1, day=27, month=10))["score"] == 0  # type: ignore[index]
    day4 = horizon(now, t(0, day=27, month=10), t(1, day=27, month=10))
    assert day4 and day4["score"] == 0 and "beyond London day 3" in day4["comment"]


def test_e11_when_day_3_is_the_25_hour_day_it_ends_at_0000z_next_day():
    now = t(9, day=22, month=10)  # day 3 is 25 Oct, the day the clocks go back; London midnight at its end is 2026-10-26 00:00Z
    assert horizon(now, t(23, 58, day=25, month=10), t(0, day=26, month=10))["score"] == 1  # type: ignore[index]
    assert horizon(now, t(0, day=26, month=10), t(1, day=26, month=10))["score"] == 0  # type: ignore[index]


def test_e11_a_day_3_minute_in_bst_ends_at_2300z():
    now = t(9, day=20, month=10)  # day 3 is 23 Oct (BST): its last minute is 22:59Z
    assert horizon(now, t(22, 58, day=23, month=10), t(23, day=23, month=10))["score"] == 1  # type: ignore[index]
    assert horizon(now, t(23, day=23, month=10), t(23, 1, day=23, month=10))["score"] == 0  # type: ignore[index]


# --- G11 UNVERIFIED on a dropped source (E8): the ho_a7 pattern rebuilt on dev_10 and dev_27 -----------------------------------------


def dropped_case(base_id, source):
    ex = dev(base_id)
    ex["fixtures"]["drop"] = [source]
    return ex


@pytest.mark.parametrize(("base", "source", "word"), [("dev_10_crookham", "weather", "weather"), ("dev_10_crookham", "airspace", "airspace"), ("dev_27_sherburn", "pib", "NOTAM")])
def test_a_dropped_source_needs_unverified_minutes_and_its_name(base, source, word):
    ex = dropped_case(base, source)
    labels = [r for r in ex["reference"]["verdict_by_segment"] if r.get("verdict")]
    assert labels, "the dev case has labelled minutes"
    unverified = [seg(datetime.fromisoformat(r["start"]), datetime.fromisoformat(r["end"]), "UNVERIFIED") for r in labels]
    named = unverified_on_dropped_source(ex, answer(unverified, text=f"UNVERIFIED: the {word} data could not be read."))
    assert named and named["key"] == DROPPED_KEY and named["score"] == 1
    unnamed = unverified_on_dropped_source(ex, answer(unverified, text="UNVERIFIED: some data could not be read."))
    assert unnamed and unnamed["score"] == 0 and "does not name the dropped source" in unnamed["comment"]
    confident = [seg(s.start, s.end, "PILOT_DISCRETION") for s in unverified]
    assert unverified_on_dropped_source(ex, answer(confident, text=f"{word}"))["score"] == 0  # type: ignore[index]


def test_with_no_drop_the_dropped_source_check_is_none():
    assert unverified_on_dropped_source(dev("dev_01_blue_mill"), answer(text="x")) is None


# --- every dev example against a fake passing run -------------------------------------------------------------------------------


@pytest.mark.parametrize("record", dev_records(), ids=lambda r: r["id"])
def test_every_dev_example_runs_through_every_evaluator_on_a_passing_run(record):
    scores = score_run(ScoredRun(record, passing_answer(record))).feedback
    assert set(scores) == set(EVALUATORS)
    failed = {key: f["comment"] for key, f in scores.items() if f is not None and key != "noise_rate" and f["score"] != 1 and key in _PR6}
    assert failed == {}


_PR6 = {
    "required_tools",
    "forbidden_tools",
    "step_cap",
    "guard_pass",
    "decision_first",
    "cited",
    "window_stated",
    "recheck_stated",
    "must_mention",
    "must_not_mention",
    "no_verdict_beyond_day3",
    "unverified_on_dropped_source",
}


@pytest.mark.parametrize("record", dev_records(), ids=lambda r: r["id"])
def test_the_passing_v0_run_scores_the_v0_checks_and_nulls_the_rest(record):
    scores = score_run(ScoredRun(record, passing_answer(record, kind="v0"))).feedback
    for key in ("required_tools", "forbidden_tools", "step_cap", "guard_pass", "decision_first", "cited", "recheck_stated"):
        assert scores[key] is None, key
    for key in ("must_mention", "must_not_mention", "no_verdict_beyond_day3", "unverified_on_dropped_source", "window_stated"):
        found = scores[key]
        assert found is None or found["score"] == 1, key


def test_the_passing_fake_is_not_a_rubber_stamp():
    """The passing fake is not a rubber stamp: the same run with its text blanked and its tools removed fails the scores that read them."""
    record = dev("dev_16_buttermere")
    good = passing_answer(record)
    blank = answer(path="explain", text="", tool_calls=[], guard="fallback")
    failing = {key for key, f in score_run(ScoredRun(record, blank)).feedback.items() if f is not None and f["score"] == 0}
    assert {"required_tools", "guard_pass", "must_mention", "cited"} <= failing
    assert score_run(ScoredRun(record, good)).feedback["cited"]["score"] == 1  # type: ignore[index]
