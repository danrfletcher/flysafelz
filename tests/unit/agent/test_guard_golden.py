"""The GP5-style Blue Mill answer passes against the real tool results, and five single-mutation variants each give exactly one finding; the entry point's contract."""

import pytest
from guard_world import blue_mill_turn, findings, turn, turn_of

from flysafelz.agent.guard import GuardFinding, guard_answer

GOLDEN = (
    "PILOT DISCRETION \u2013 APPROACHING LIMITS \u00b7 WITH CONDITIONS for Saturday 31 Oct, 10:00\u201316:00 GMT. "
    "Danger area EGD496A is active by NOTAM J3857/26: phone Flylogix <PHONE_1> first. "
    "H7171/26 is a drone operation to the SSE, so avoid that sector. "
    "This is decision support only; re-check the forecast before you fly."
)

GOLD_SENTENCES = (
    "PILOT DISCRETION \u2013 APPROACHING LIMITS \u00b7 WITH CONDITIONS for Saturday 31 Oct, 10:00\u201316:00 GMT.",
    "Danger area EGD496A is active by NOTAM J3857/26: phone Flylogix <PHONE_1> first.",
    "H7171/26 is a drone operation to the SSE, so avoid that sector.",
    "This is decision support only; re-check the forecast before you fly.",
)


def test_the_golden_blue_mill_answer_passes_against_the_real_tool_results():
    run = blue_mill_turn()
    assert {"get_weather", "get_airspace", "get_notams", "assess"} == {r.name for r in run.results}
    assert " ".join(GOLD_SENTENCES) == GOLDEN
    assert findings(GOLDEN, run) == []


# (the sentence to change, its replacement, the one finding that results)
MUTATIONS = {
    "a verdict assess did not return": (
        0,
        "PILOT DISCRETION \u2013 APPROACHING LIMITS \u00b7 WITH CONDITIONS",
        "NO-GO",
        ["'NO-GO' is not a verdict assess returned this turn"],
    ),
    "a time no tool printed": (0, "10:00\u201316:00", "10:00\u201317:30", ["'17:30' is not a time any tool printed this turn"]),
    "an invented NOTAM id": (1, "J3857/26", "J3858/26", ["NOTAM id 'J3858/26' is not in any tool result this turn"]),
    "a placeholder that is not the thread's": (1, "<PHONE_1>", "<PHONE_2>", ["'<PHONE_2>' is not one of this thread's phone placeholders"]),
    "a data-is-final sentence": (
        3,
        "re-check the forecast before you fly",
        "no need to re-check the forecast before you fly",
        ["'This is decision support only; no need to re-check the forecast before you fly' says the data is final, current enough or needs no re-check"],
    ),
}


@pytest.mark.parametrize(("index", "old", "new", "expected"), MUTATIONS.values(), ids=MUTATIONS)
def test_one_mutation_of_the_golden_answer_gives_exactly_one_finding(index, old, new, expected):
    sentences = list(GOLD_SENTENCES)
    assert old in sentences[index]
    sentences[index] = sentences[index].replace(old, new)
    assert findings(" ".join(sentences), blue_mill_turn()) == expected


def test_findings_are_fixed_wording_in_a_stable_order_with_no_repeats():
    answer = "NOTAM V99/26 at 17:30 is safe to fly, 55.5 mph."
    first, second = guard_answer(answer, turn()), guard_answer(answer, turn())
    assert [str(f) for f in first] == [str(f) for f in second]
    assert first and all(isinstance(f, GuardFinding) and f.check in "abcd" for f in first)
    assert len(first) == len(set(first))


@pytest.mark.parametrize(
    "answer",
    [
        "",
        " ",
        "\n\n",
        "\u200b",
        "x" * 5000,
        "\u00ff" * 50,
        "1/2/3/4",
        "-" * 40,
        "1e999",
        "99999999999999999999999.9 mph",
        "0." + "9" * 80 + " mph",
        '"' * 7,
        "`" * 9,
        ">" * 5,
        "<PHONE_>",
        "H/26",
        "PILOT\u00a0DISCRETION\u00a0\u2013\u00a0APPROACHING LIMITS \u00b7 WITH CONDITIONS \u2014 \u2212 \u2012",
        "NO\u2011GO \u2010 \u2015",
    ],
)
def test_any_odd_text_returns_a_list_without_raising(answer):
    assert isinstance(guard_answer(answer, turn()), list)
    assert isinstance(guard_answer(answer, turn_of()), list)


@pytest.mark.parametrize("answer", ["", " ", "\n\t \n", "\u00a0\u00a0"])
def test_an_empty_or_whitespace_answer_has_no_findings(answer):
    assert guard_answer(answer, turn()) == []


def test_the_check_letters_are_a_to_d_and_name_the_check_that_failed():
    answer = "PILOT DISCRETION. V99/26 at 5.2 mph; safe to fly. No need to check again."
    by_check = {f.check for f in guard_answer(answer, turn_of())}
    assert by_check == {"a", "b", "c", "d"}
