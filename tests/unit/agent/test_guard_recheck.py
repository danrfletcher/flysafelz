"""G14 (d), a closed list (I10): a sentence is a finding only when it holds one of the exact phrases in `FINAL_PHRASES`. Any other wording is a known gap, not a finding. The re-check line itself is added by code after the guard, so its absence is not a finding."""

import pytest
from guard_world import findings, turn

from flysafelz.agent.guard import FINAL_PHRASES

# The phrases the spec lists, one row each (the E13 sentence is a row of its own below).
SPEC_PHRASES = [
    "the data is final",
    "the forecast is final",
    "the NOTAMs are final",
    "the forecast is now final",
    "this forecast is definitely final",
    "this is the final forecast",
    "the data won't change",
    "the forecast won't change",
    "the NOTAMs won't change",
    "no need to check again",
    "no need to re-check",
    "no need to recheck",
    "no need to check the forecast again",
    "no need to check the NOTAMs again",
    "you don't need to check the NOTAMs again",
    "no need to double check",
    "no need to double-check",
    "you don't need to double check the NOTAMs",
    "no further checks are needed",
    "no further re-check is needed",
    "you can skip the re-check",
    "you needn't bother checking again",
]


def _finding(sentence: str) -> str:
    return f"'{sentence.rstrip('.')}' says the data is final, current enough or needs no re-check"


def test_the_phrases_live_in_one_tuple_and_are_the_spec_list():
    assert list(FINAL_PHRASES) == SPEC_PHRASES


@pytest.mark.parametrize("phrase", SPEC_PHRASES)
def test_each_closed_list_phrase_is_a_finding(phrase):
    sentence = f"{phrase[0].upper()}{phrase[1:]}."
    assert findings(f"{sentence} Fly safe.", turn()) == [_finding(sentence)]


@pytest.mark.parametrize("phrase", SPEC_PHRASES)
def test_a_phrase_is_found_ignoring_case_inside_a_longer_sentence_and_with_curly_apostrophes(phrase):
    sentence = f"Rest assured, {phrase.upper()}.".replace("'", "\u2019")
    assert findings(sentence, turn()) == [_finding(sentence)]


def test_the_e13_sentence_is_one_finding():
    sentence = "The forecast is final, no need to check again"
    assert findings(f"{sentence}.", turn()) == [_finding(sentence)]


@pytest.mark.parametrize(
    "sentence",
    [
        "The data won't change again.",  # before "again"
        "The NOTAMs won't change.",
        "The forecast won't change!",
        "The forecast won't change",
        "The forecast won\u2019t change.",
        "The forecast won't  change.",
        "No need to double\u2011check.",  # a non-breaking hyphen is a dash
        "No need to\u00a0check again.",  # a non-breaking space is a space
    ],
)
def test_the_wont_change_phrases_count_at_a_sentence_end_or_before_again(sentence):
    found = findings(sentence, turn())
    assert len(found) == 1 and found[0].endswith("says the data is final, current enough or needs no re-check")


@pytest.mark.parametrize(
    "sentence",
    [
        "The weather looks settled.",
        "The weather looks settled through the morning.",
        "Conditions stay settled until midday.",
        "The weather is settled.",
        "Conditions are certain.",
        "The NOTAM area is fixed around the mast.",
        "The airspace has a fixed lower base.",
        "The final hour of the forecast window is the last one.",
        "The forecast won't change your verdict much.",
        "It is a final approach area.",
        "Settled weather is forecast for the morning.",
        # wordings outside the list are a known gap, not a finding (I10)
        "The forecast is definitive.",
        "The picture is locked in.",
        "You don't need to recheck before you fly.",
        "There is no need for a re-check.",
        "This is current enough to rely on.",
        "Nothing more to check.",
        "The forecast won't change before you fly.",
        # advice to re-check, and negated forms of the phrases
        "Re-check the forecast before you fly.",
        "The forecast may change, so check again on the day.",
        "There is a need to check the forecast again.",
        "Do not skip the re-check.",
        "You can't skip the double check.",
        "Do a final check of the forecast before you go.",
        "The forecast is not final.",
        "The forecast isn't final, so check again.",
        "No further wind is forecast.",
        "Further checks are needed on the day.",
        "The finalised forecast is not out yet.",
        "The forecast is finalising.",
        "Stay clear of the fixed-wing circuit.",
    ],
)
def test_other_wording_is_not_a_finding(sentence):
    assert findings(sentence, turn()) == []


def test_a_phrase_in_a_quoted_notam_excerpt_is_not_checked_as_the_answers_words():
    notam = "TEMPORARY AREA. THE DATA IS FINAL. NO NEED TO CHECK AGAIN."
    from guard_world import text_turn

    assert findings('The NOTAM says "the data is final. no need to check again".', text_turn(notam)) == []
    assert findings("The NOTAM says the data is final. no need to check again.", text_turn(notam)) != []


def test_the_re_check_line_is_added_by_code_so_its_absence_is_not_a_finding():
    assert findings("A plain note with no verdict and no advice to re-check.", turn()) == []
