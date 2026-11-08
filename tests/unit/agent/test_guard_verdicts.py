"""G14 (a): verdict words against what `assess` returned this turn: the six display strings, how they are matched, the first sentence, the banned phrases, an overall of None, quotations."""

import pytest
from assess_tool_helpers import render
from factories import forecast_failed, run_assess, utc
from guard_world import blue_mill_turn, findings, mixed_turn, outlook_turn, text_turn, turn_of

from flysafelz.agent.guard import DISPLAYS, ToolResult, guard_answer, verdicts_in
from flysafelz.agent.tools.artifacts import GuidanceArtifact, GuidanceChunk
from flysafelz.contracts import Verdict, display_verdict

OVERALL = "PILOT DISCRETION \u2013 APPROACHING LIMITS \u00b7 WITH CONDITIONS"
NOT_RETURNED = "'{}' is not a verdict assess returned this turn"
NO_ASSESS = "a verdict was given but assess did not run this turn"


def test_the_six_strings_are_the_ones_display_verdict_builds():
    built = {display_verdict(v, limits, conditions) for v in Verdict for limits in (False, True) for conditions in (False, True)}
    assert set(DISPLAYS) == built and len(DISPLAYS) == 6


def test_the_blue_mill_turn_has_the_overall_assess_printed():
    turn = blue_mill_turn()
    assert turn.assess_ran and turn.overall == OVERALL and turn.displays == {OVERALL}
    assert turn.phone_keys == {"<PHONE_1>"} and {tier for _, tier in turn.tiers} == {"VERDICT"}


def test_the_overall_verdict_in_the_first_sentence_passes():
    assert findings(f"{OVERALL} for 10:00\u201316:00 GMT.", blue_mill_turn()) == []


@pytest.mark.parametrize(
    "written",
    [
        OVERALL.lower(),
        "pilot discretion - approaching limits \u00b7 with conditions",  # hyphen for the en dash
        "PILOT DISCRETION \u2014 APPROACHING LIMITS \u00b7 WITH CONDITIONS",  # em dash
        "PILOT DISCRETION \u2212 APPROACHING LIMITS \u2022 WITH CONDITIONS",  # minus sign and bullet
        "PILOT\u00a0DISCRETION\u00a0\u2013\u00a0APPROACHING LIMITS\u00a0\u00b7\u00a0WITH CONDITIONS",  # non-breaking spaces
        "PILOT DISCRETION\u2013APPROACHING LIMITS\u00b7WITH CONDITIONS",  # no spaces
        "Pilot Discretion  \u2013  Approaching Limits  \u00b7  With Conditions",  # double spaces
    ],
)
def test_the_verdict_is_matched_ignoring_case_dash_dot_and_spacing(written):
    assert verdicts_in(written + " for the window.") == [OVERALL]
    assert findings(written + " for the window.", blue_mill_turn()) == []


@pytest.mark.parametrize("written", ["NO-GO", "no-go", "NO\u2013GO", "NO\u2014GO", "No Go", "NO - GO"])
def test_no_go_is_matched_however_it_is_dashed(written):
    assert verdicts_in(f"{written} today.") == ["NO-GO"]
    assert findings(f"{written} today.", blue_mill_turn()) == [NOT_RETURNED.format("NO-GO")]


def test_the_longest_string_wins_so_one_verdict_is_not_counted_twice():
    assert verdicts_in(f"{OVERALL}.") == [OVERALL]
    assert verdicts_in("PILOT DISCRETION, then PILOT DISCRETION \u2013 WITH CONDITIONS.") == ["PILOT DISCRETION", "PILOT DISCRETION \u2013 WITH CONDITIONS"]


@pytest.mark.parametrize("wrong", ["PILOT DISCRETION", "PILOT DISCRETION \u2013 WITH CONDITIONS", "PILOT DISCRETION \u2013 APPROACHING LIMITS", "NO-GO", "UNVERIFIED"])
def test_a_display_string_assess_did_not_return_is_a_finding(wrong):
    """Each is one of the six, and none is the one assess gave; the first sentence is not also reported, since the string is already refused."""
    assert findings(f"{wrong} for 10:00\u201316:00 GMT.", blue_mill_turn()) == [NOT_RETURNED.format(wrong)]


def test_a_verdict_further_down_the_answer_is_checked_too():
    answer = f"{OVERALL}. Later it is NO-GO at 12:00."
    assert findings(answer, blue_mill_turn()) == [NOT_RETURNED.format("NO-GO")]


def test_the_first_sentence_must_name_the_overall_verdict_when_it_names_a_returned_one():
    turn = mixed_turn()
    assert turn.overall == "NO-GO" and "PILOT DISCRETION" in turn.displays
    assert findings("PILOT DISCRETION from 08:00 to 16:00 GMT. Overall it is NO-GO.", turn) == ["the first sentence says 'PILOT DISCRETION' but the overall verdict assess gave is 'NO-GO'"]
    assert findings("NO-GO overall. PILOT DISCRETION from 08:00 to 16:00 GMT.", turn) == []


def test_an_answer_that_gives_a_verdict_later_must_still_lead_with_the_overall_one():
    turn = mixed_turn()
    lead = "the first sentence names no verdict but the overall verdict assess gave is 'NO-GO'"
    assert findings("Conditions look workable this morning. PILOT DISCRETION from 08:00.", turn) == [lead]
    assert findings("Conditions look workable this morning. NO-GO overall.", turn) == [lead]
    assert findings("NO-GO overall. Conditions look workable this morning. PILOT DISCRETION from 08:00.", turn) == []
    assert findings(f"Conditions look workable. {OVERALL}.", blue_mill_turn()) == [f"the first sentence names no verdict but the overall verdict assess gave is '{OVERALL}'"]
    assert findings("Conditions look workable this morning.", turn) == []  # no verdict word at all: the first-sentence rule has nothing to check


def test_the_first_sentence_ends_at_a_full_stop_a_question_mark_an_exclamation_mark_or_a_line_break():
    turn = mixed_turn()
    wrong = "the first sentence says 'PILOT DISCRETION' but the overall verdict assess gave is 'NO-GO'"
    for end in (". ", "! ", "? ", "\n"):
        assert findings(f"PILOT DISCRETION at 08:00{end}Overall NO-GO.", turn) == [wrong], repr(end)
    assert findings("NO-GO overall and PILOT DISCRETION at 08:00, so decide.", turn) == []  # both are in the first sentence, and one is the overall


def test_a_decimal_point_does_not_end_the_first_sentence():
    assert findings("Gust spread is 5.2 mph, so NO-GO.", mixed_turn()) == ["'5.2 mph' is not a number any tool printed this turn, or a rounding of one"]


def test_a_verdict_with_no_assess_result_is_the_e12_finding():
    for word in ("PILOT DISCRETION", "NO-GO"):
        assert findings(f"{word} for tomorrow.", turn_of()) == [NO_ASSESS]
    assert findings("PILOT DISCRETION.", text_turn("a weather result, but no assess")) == [NO_ASSESS]


def test_a_failed_assess_with_no_artifact_is_no_assessment():
    assert findings("NO-GO.", turn_of(ToolResult("assess", "Could not assess: the window is too long."))) == [NO_ASSESS]


def test_overall_none_allows_no_verdict_word_anywhere():
    turn = outlook_turn()
    assert turn.assess_ran and turn.overall is None and turn.displays == frozenset()
    expected = "is a verdict word but assess gave no overall verdict this turn (every hour is OUTLOOK or too far out)"
    for word in ("PILOT DISCRETION", "NO-GO", "UNVERIFIED", OVERALL):
        assert findings(f"It is {word} on 5 Nov.", turn) == [f"'{word}' {expected}"], word
    assert findings("Thursday 5 Nov is only an outlook; ask again on Wednesday 4 Nov.", turn) == []


def test_a_verdict_word_on_an_outlook_hour_fails_while_the_overall_is_not_none():
    turn = mixed_turn()
    line = "'NO-GO' is attached to an OUTLOOK hour, which has no verdict"
    assert findings("NO-GO overall, and the outlook for Wednesday is NO-GO as well.", turn) == [line]
    assert findings("NO-GO overall. On Wednesday 4 Nov at 08:00 it is NO-GO too.", turn) == [line]  # a date makes the hour exact
    assert findings("NO-GO overall. At 03:00 it is NO-GO too.", turn) == [line]  # 03:00 is an hour only the outlook has
    assert findings("NO-GO overall. At 08:00 it is PILOT DISCRETION.", turn) == []  # 08:00 has a verdict on Tuesday, so the guard does not guess
    assert findings("NO-GO overall. On Tuesday 3 Nov at 08:00 it is PILOT DISCRETION.", turn) == []


def test_pilot_discretion_when_assess_overall_is_unverified_fails():
    window = (utc(10), utc(13))
    text, artifact = render(run_assess(window=window, forecast=forecast_failed()), window)
    turn = turn_of(ToolResult("assess", text, artifact))
    assert turn.overall == "UNVERIFIED"
    assert findings("PILOT DISCRETION for 10:00\u201313:00.", turn) == [NOT_RETURNED.format("PILOT DISCRETION")]
    assert findings("UNVERIFIED for 10:00\u201313:00.", turn) == []


# --- banned phrases, on every path -------------------------------------------------------------------------------------


@pytest.mark.parametrize("phrase", ["safe to fly", "clear to fly", "good to go", "all clear", "Safe To Fly", "ALL CLEAR", "all  clear"])
def test_a_banned_phrase_is_a_finding_with_no_tools_and_with_them(phrase):
    answer = f"It is {phrase} today."
    assert len(findings(answer, turn_of())) == 1 and "banned phrase" in findings(answer, turn_of())[0]
    assert any("banned phrase" in f for f in findings(answer, blue_mill_turn()))


@pytest.mark.parametrize("answer", ["It is GO today.", "GO.", "Go or not: GO!", "(GO)", "say GO now"])
def test_the_capital_go_on_its_own_is_a_finding(answer):
    assert len(findings(answer, turn_of())) == 1 and "banned word" in findings(answer, turn_of())[0]


@pytest.mark.parametrize("answer", ["NO-GO is the verdict.", "GOV.UK has the forms.", "Let's go and look at the map.", "It is a good go-around.", "AGO", "GOLF", "no-go area"])
def test_go_inside_a_word_or_after_a_hyphen_or_in_lower_case_is_not_the_bare_word(answer):
    assert not [f for f in findings(answer, turn_of()) if "banned" in f]


def test_safe_alone_and_overall_clearance_are_allowed():
    assert findings("Safe flying is your call. Overall clearance of the airspace is for you to confirm.", turn_of()) == []


def test_a_decline_with_no_tools_and_no_verdict_words_passes_and_one_saying_good_to_go_fails():
    assert findings("I can only help with flying conditions at a place you click on the map.", turn_of()) == []
    assert findings("Sure, it's good to go.", turn_of()) == ["'good to go' is a banned phrase on every path"]


def test_a_banned_phrase_with_a_verdict_on_the_assess_path_is_one_finding_each():
    assert findings(f"{OVERALL}. It is all clear.", blue_mill_turn()) == ["'all clear' is a banned phrase on every path"]


# --- quotations --------------------------------------------------------------------------------------------------------

RAW = "UAS OPR WI 0.5NM RADIUS OF 534732N 0000459W. GO AHEAD AT 7777 FT FOR INFO 1234 FT"
EXCERPTS = {
    "double quotes": f'The NOTAM says "{RAW}" in full.',
    "curly quotes": f"The NOTAM says \u201c{RAW}\u201d in full.",
    "a code span": f"The NOTAM says `{RAW}` in full.",
    "a code block": f"The NOTAM says:\n```\n{RAW}\n```\nThat is all.",
    "a block quote": f"The NOTAM says:\n> {RAW}\n\nThat is all.",
    "bar lines": f"The NOTAM says:\n| {RAW}\nThat is all.",
}


@pytest.mark.parametrize("answer", EXCERPTS.values(), ids=EXCERPTS)
def test_words_and_numbers_inside_a_quotation_of_a_tool_result_do_not_count(answer):
    assert findings(answer, text_turn(f"NOTAM text:\n  | {RAW}\n")) == []


@pytest.mark.parametrize("answer", EXCERPTS.values(), ids=EXCERPTS)
def test_the_same_quotation_of_text_no_tool_printed_is_the_answers_own_words(answer):
    assert any("banned word" in f for f in findings(answer, text_turn("a different NOTAM")))


def test_an_unquoted_copy_of_the_same_text_is_checked():
    turn = text_turn(f"NOTAM text:\n  | {RAW}\n")
    unquoted = f"The NOTAM says {RAW} in full."
    assert findings(unquoted, turn) != [] and any("banned word" in f for f in findings(unquoted, turn))


def test_a_short_or_mid_token_quotation_is_not_an_excerpt_even_when_its_letters_are_in_a_tool_result():
    """The quoted text must be a stretch of whole tokens and long enough to be an excerpt; 'go' is inside 'category' and 'no-go', '1 mph' inside '3.1 mph'."""
    turn = blue_mill_turn()
    assert findings(f'{OVERALL}. It is "GO" for you.', turn) == ["'GO' on its own is a banned word on every path"]
    assert findings(f'{OVERALL}. Gust spread is "1 mph".', turn) == ["'1 mph' is not a number any tool printed this turn, or a rounding of one"]
    assert findings('It is "all clear".', text_turn("the overall clearance is pending")) == ["'all clear' is a banned phrase on every path"]
    # three words or more, but starting or ending inside a word of the tool's text
    printed = text_turn("category no-go area, overall clearance is pending")
    assert findings('It says "gory NO-GO area".', printed) != []
    assert findings('It says "category no-go ar".', printed) != []
    clear = text_turn("the overall clear of the area is pending")
    assert findings('It says "all clear of the area".', clear) == ["'all clear' is a banned phrase on every path"]
    assert findings('It says "the overall clear of the area".', clear) == []


def test_a_quotation_of_whole_tokens_of_a_tool_result_is_still_an_excerpt():
    printed = text_turn("NOTAM text: GO AHEAD AT 7777 FT, the overall clearance is pending")
    assert findings('It says "GO AHEAD AT 7777 FT".', printed) == []
    assert findings('It says "the overall clearance is pending".', printed) == []


def test_a_quoted_excerpt_does_not_hide_the_words_around_it():
    turn = text_turn(f"  | {RAW}\n")
    assert any("banned phrase" in f for f in findings(f'It says "{RAW}" and it is good to go.', turn))


def test_a_quotation_across_a_wrapped_raw_text_still_matches():
    turn = text_turn("  | UAS OPR WI 0.5NM RADIUS\n  | OF 534732N 0000459W. GO AHEAD\n")
    assert findings('It reads "UAS OPR WI 0.5NM RADIUS OF 534732N 0000459W. GO AHEAD".', turn) == []


@pytest.mark.parametrize("dash", ["\u2013", "\u2014", "\u2212", "-"])
def test_an_excerpt_with_any_kind_of_dash_is_still_an_excerpt(dash):
    """R9: the answer is made one-dash by `_canon`, so the tool's text must be too, or a verbatim excerpt with an en dash would be 'GO' on its own."""
    raw = f"E) PJE WI 2NM RADIUS 534732N 0000459W SFC{dash}FL100 GO AHEAD"
    turn = text_turn(raw)
    assert findings(f'It reads "{raw}".', turn) == []
    assert findings(f'It reads "{raw.replace(dash, "-")}".', turn) == []  # whichever dash the answer uses
    assert any("banned word" in f for f in findings(f"It reads {raw}.", turn))  # unquoted, it is the answer's own words


def test_a_quoted_line_assess_printed_still_counts_as_a_verdict():
    """R8: only a NOTAM's raw text or a guidance chunk is an excerpt. The quoted hour line is assess's own text, so its verdict is the answer's, and the answer never leads with the overall."""
    turn = mixed_turn()
    assert turn.overall == "NO-GO"
    line = next(row for row in turn.results[0].text.splitlines() if row.strip().endswith("PILOT DISCRETION")).strip()
    assert line in turn.results[0].text and len(line) > 15
    first = "the first sentence says 'PILOT DISCRETION' but the overall verdict assess gave is 'NO-GO'"
    assert findings(f'"{line}"', turn) == [first]
    assert findings(f'Morning hours read "{line}".', turn) == [first]
    assert findings(f'NO-GO overall. Morning hours read "{line}".', turn) == []  # the same words, behind the overall verdict, are checked and fine
    assert findings('Morning hours read "GMT PILOT DISCRETION 08:00".', turn) == [first]  # the reviewer's probe


def test_a_text_only_result_of_another_tool_is_not_an_excerpt_source():
    printed = text_turn("NOTAM text: GO AHEAD AT 7777 FT")
    assert findings('It says "GO AHEAD AT 7777 FT".', printed) == []
    other = turn_of(ToolResult("get_weather", "NOTAM text: GO AHEAD AT 7777 FT"))
    assert any("banned word" in f for f in findings('It says "GO AHEAD AT 7777 FT".', other))


def test_a_guidance_chunks_text_is_an_excerpt_and_the_text_the_tool_printed_around_it_is_not():
    chunk = GuidanceChunk(document="caa-cap-1234", edition="2026", section="4.2", title="Rules", text="Pilots must not treat a briefing as a GO decision")
    result = ToolResult("search_guidance", "Result 1: CAP 1234, section 4.2, Rules. Pilots must not treat a briefing as a GO decision", GuidanceArtifact(query="rules", chunks=[chunk]))
    turn = turn_of(result)
    assert findings('The guidance says "Pilots must not treat a briefing as a GO decision".', turn) == []
    assert any("banned word" in f for f in findings('The tool printed "Result 1: CAP 1234, section 4.2, Rules. Pilots must not treat a briefing as a GO decision".', turn))


def test_guard_answer_is_the_function_these_tests_call():
    assert guard_answer("", turn_of()) == []
