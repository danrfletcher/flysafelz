"""G14 (b), the Q2 number rule (FR6): a written number passes only when a number a tool printed rounds to it, in the same unit or one the code table converts, without crossing the limit the artifact pairs with the value.

Every row is (what the answer writes, the value the tool printed, its unit, the limit and its comparison from the artifact, the finding or None). The tool value and its limit are in the artifact
only: the text of the result is empty, so nothing but the artifact can make a row pass.
"""

import random
from dataclasses import replace

import pytest
from guard_world import blue_mill_turn, findings, text_turn, turn_of, weather_turn

from flysafelz.agent.guard import ToolResult

NOT_PRINTED = "'{}' is not a number any tool printed this turn, or a rounding of one"
ACROSS = "'{}' rounds {} across the {} limit"
IS_THE_LIMIT = "'{}' is the limit that gust spread {} {}. Give each value exactly as the tool printed it. Don't restate limit values: code adds a line stating each limit reached."
NONE = None

# (written, value, unit, limit, comparison, finding) -- the finding is the whole sentence the guard gives, or None for a pass
PRECISION = [
    ("10 mph", 9.9, "mph", None, ">=", NONE),  # a tool value rounds to the written integer
    ("9.9 mph", 9.86, "mph", None, ">=", NONE),  # and to one decimal
    ("9.86 mph", 9.86, "mph", None, ">=", NONE),  # and to two
    ("9.9 mph", 9.9, "mph", None, ">=", NONE),
    ("9.8 mph", 9.9, "mph", None, ">=", NOT_PRINTED.format("9.8 mph")),  # the written number is not what 9.9 rounds to
    ("9.87 mph", 9.86, "mph", None, ">=", NOT_PRINTED.format("9.87 mph")),
    ("9 mph", 9.9, "mph", None, ">=", NOT_PRINTED.format("9 mph")),  # rounding goes up at .9, not down
    ("10.0 mph", 9.96, "mph", None, ">=", NONE),  # a trailing .0 is one decimal
    ("10.0 mph", 9.9, "mph", None, ">=", NOT_PRINTED.format("10.0 mph")),
    ("10.00 mph", 9.996, "mph", None, ">=", NONE),
    ("9.90 mph", 9.9, "mph", None, ">=", NONE),
    ("9.91 mph", 9.9, "mph", None, ">=", NOT_PRINTED.format("9.91 mph")),
    ("10 mph", 9.5, "mph", None, ">=", NONE),  # half rounds away from zero
    ("9 mph", 9.5, "mph", None, ">=", NOT_PRINTED.format("9 mph")),
    ("-3 \u00b0C", -3.2, "\u00b0C", None, ">=", NONE),
    ("3 \u00b0C", -3.2, "\u00b0C", None, ">=", NOT_PRINTED.format("3 \u00b0C")),  # the sign matters
    ("5 mph", 4.5, "mph", None, ">=", NONE),
    ("1,520 m", 1520.0, "m", None, ">=", NONE),  # a thousands separator
]
UNITS = [
    ("1.5 km", 1520.0, "m", None, ">=", NONE),  # m to km
    ("1.6 km", 1520.0, "m", None, ">=", NOT_PRINTED.format("1.6 km")),
    ("1520 m", 1.52, "km", None, ">=", NONE),  # km to m
    ("4921 ft", 1500.0, "m", None, ">=", NONE),  # m to ft
    ("1500 m", 4921.0, "ft", None, ">=", NONE),  # ft to m
    ("8 kt", 9.2, "mph", None, ">=", NONE),  # mph to kt
    ("4 m/s", 8.9, "mph", None, ">=", NONE),  # mph to m/s
    ("9 mph", 7.8, "kt", None, ">=", NONE),  # kt to mph
    ("9 mph", 4.0, "m/s", None, ">=", NONE),  # m/s to mph
    ("3.7 km", 2.0, "NM", None, ">=", NOT_PRINTED.format("3.7 km")),  # NM is not in the table
    ("2 NM", 3.7, "km", None, ">=", NOT_PRINTED.format("2 NM")),
    ("2 NM", 2.0, "NM", None, ">=", NONE),  # the printed unit itself
    ("2 miles", 2.0, "NM", None, ">=", NOT_PRINTED.format("2 miles")),
    ("30 %", 10.0, "mph", None, ">=", NOT_PRINTED.format("30 %")),  # a percentage is not a speed
    ("10 %", 10.0, "mph", None, ">=", NOT_PRINTED.format("10 %")),
    ("30 %", 30.0, "%", None, ">=", NONE),
    ("1013 hPa", 1013.0, "mph", None, ">=", NOT_PRINTED.format("1013 hPa")),  # hPa is not in the table
    ("1013 hPa", 1013.0, "hPa", None, ">=", NONE),
    ("5", 5.0, "mph", None, ">=", NOT_PRINTED.format("5")),  # no unit written: only a unitless printed value can match
    ("30", 30.0, "%", None, ">=", NONE),  # a bare number may be a percentage
    ("10 mph", 10.0, "m", None, ">=", NOT_PRINTED.format("10 mph")),  # a unit from another family
    ("2 mm/h", 2.0, "mm/h", None, ">=", NONE),
    ("2 m", 2.0, "mm/h", None, ">=", NOT_PRINTED.format("2 m")),
]
LIMITS = [  # I10: number-only. A hedged number, or one that is not the limit, "rounds across"; the limit itself beside a value that crosses is "the limit that"
    ("about 9 mph", 9.3, "mph", 9.0, ">=", ACROSS.format("about 9 mph", "9.3 mph", "9 mph")),  # the case in the spec
    ("about 6 mph", 6.04, "mph", 6.0, ">", ACROSS.format("about 6 mph", "6.04 mph", "6 mph")),  # a hedged number is a rounded value, not the limit
    ("9 mph", 9.3, "mph", 9.0, ">", IS_THE_LIMIT.format("9 mph", "9.3 mph", "exceeds")),  # lands on the limit while the value is above it
    ("9.0 mph", 9.04, "mph", 9.0, ">", IS_THE_LIMIT.format("9.0 mph", "9.04 mph", "exceeds")),
    ("6 mph", 6.04, "mph", 6.0, ">", IS_THE_LIMIT.format("6 mph", "6.04 mph", "exceeds")),  # just over a limit, rounded down onto it
    ("6.0 mph", 6.04, "mph", 6.0, ">", IS_THE_LIMIT.format("6.0 mph", "6.04 mph", "exceeds")),
    ("6.04 mph", 6.04, "mph", 6.0, ">", NONE),  # at its own precision it is the value
    ("6 mph", 6.0, "mph", 6.0, ">", NONE),  # exactly at the limit
    ("6 mph", 6.0, "mph", 6.0, ">=", NONE),
    ("6.0 mph", 6.0, "mph", 6.0, ">=", NONE),
    ("9 mph", 8.7, "mph", 9.0, ">=", IS_THE_LIMIT.format("9 mph", "8.7 mph", "falls below")),  # just under a >= limit, rounded up onto it
    ("9 mph", 9.2, "mph", 9.0, ">=", IS_THE_LIMIT.format("9 mph", "9.2 mph", "exceeds")),  # lands on it while the value is not exactly there
    ("10 mph", 9.9, "mph", 9.0, ">=", NONE),  # both sides are over
    ("11 mph", 10.6, "mph", 10.0, ">", NONE),
    ("10 mph", 10.4, "mph", 10.0, ">", IS_THE_LIMIT.format("10 mph", "10.4 mph", "exceeds")),
    ("10 mph", 9.6, "mph", 9.8, ">", ACROSS.format("10 mph", "9.6 mph", "9.8 mph")),  # the rounding takes it over a limit it is under
    ("10 mph", 9.6, "mph", 9.8, ">=", ACROSS.format("10 mph", "9.6 mph", "9.8 mph")),
    ("5 mph", 5.3, "mph", 5.0, "<", IS_THE_LIMIT.format("5 mph", "5.3 mph", "exceeds")),  # the < comparisons
    ("4 mph", 4.4, "mph", 5.0, "<", NONE),
    ("5 mph", 4.8, "mph", 5.0, "<=", IS_THE_LIMIT.format("5 mph", "4.8 mph", "falls below")),
    ("5 mph", 5.0, "mph", 5.0, "<=", NONE),
    ("5 mph", 5.0, "mph", 5.0, "<", NONE),
    ("2.9 km", 2920.0, "m", 5000.0, "<", NONE),  # a limit in another unit
    ("3 km", 2920.0, "m", 3000.0, "<", IS_THE_LIMIT.format("3 km", "2920 m", "falls below")),
    ("3 km", 2920.0, "m", 5000.0, "<", NONE),
    ("98 %", 98.0, "%", 90.0, ">", NONE),
    ("90 %", 91.0, "%", 90.0, ">", NONE),  # 91 does not round to 90 at any precision, so nothing crosses; the limit printed beside the value grounds the 90
    ("100 mph", 95.0, "mph", 90.0, ">", NOT_PRINTED.format("100 mph")),  # neither the value nor the limit
]
PERCENT = [
    ("about 10 mph", 9.4, "mph", None, ">=", NOT_PRINTED.format("about 10 mph")),  # within 10 %, but not what 9.4 rounds to
    ("about 10 mph", 9.6, "mph", None, ">=", NONE),
    ("about 9 mph", 9.6, "mph", None, ">=", NOT_PRINTED.format("about 9 mph")),
    ("roughly 100 mph", 95.0, "mph", None, ">=", NOT_PRINTED.format("roughly 100 mph")),
]
ROWS = {"precision": PRECISION, "units": UNITS, "limit crossing": LIMITS, "no percentage tolerance": PERCENT}
CASES = [pytest.param(*row, id=f"{kind}: {row[0]} vs {row[1]} {row[2]}" + (f" ({row[4]} {row[3]:g})" if row[3] is not None else "")) for kind, rows in ROWS.items() for row in rows]


@pytest.mark.parametrize(("written", "value", "unit", "limit", "comparison", "expected"), CASES)
def test_the_number_rule_over_precision_unit_limit_and_tolerance(written, value, unit, limit, comparison, expected):
    turn = weather_turn(value, unit, limit, comparison)
    assert findings(f"Gust spread is {written} now.", turn) == ([] if expected is None else [expected])


# --- number words: only directly before a counted noun ------------------------------------------------------------------

WORDS = [
    ("There are two NOTAMs here.", "2 NOTAMs within 2 NM. 0 awareness-only.", NONE),
    ("There are two NOTAMs here.", "3 NOTAMs within 2 NM.", NOT_PRINTED.format("two NOTAMs")),
    ("There are no NOTAMs here.", "0 NOTAMs within 2 NM.", NONE),
    ("There are no NOTAMs here.", "2 NOTAMs within 2 NM. 0 awareness-only.", NONE),  # a printed 0 is a count a tool printed
    ("There are no NOTAMs here.", "2 NOTAMs within 2 NM.", NOT_PRINTED.format("no NOTAMs")),
    ("There are zero NOTAMs here.", "2 NOTAMs within 2 NM.", NOT_PRINTED.format("zero NOTAMs")),
    ("It is flyable for three hours.", "flyable for 3 h 06 min", NONE),
    ("It is flyable for four hours.", "flyable for 3 h 06 min", NOT_PRINTED.format("four hours")),
    ("It is flyable for a couple of hours.", "flyable for 2 h 00 min", NONE),
    ("It is flyable for a couple of hours.", "flyable for 3 h 06 min", NOT_PRINTED.format("a couple of hours")),
    ("It is flyable for a few hours.", "flyable for 3 h 00 min", NONE),
    ("It is flyable for a few hours.", "flyable for 5 h 00 min", NOT_PRINTED.format("a few hours")),
    ("There are ten minutes of light left.", "light left 10 min", NONE),
    ("There are five minutes of light left.", "light left 10 min", NOT_PRINTED.format("five minutes")),
    ("It has one segment.", "1 segments", NONE),
    ("It has two segments.", "1 segments", NOT_PRINTED.format("two segments")),
    ("Three conditions apply.", "3 conditions", NONE),
    ("Four conditions apply.", "3 conditions", NOT_PRINTED.format("Four conditions")),
    ("One area is close.", "1 area", NONE),
    ("Two airspaces are close.", "1 airspace", NOT_PRINTED.format("Two airspaces")),
    ("one thing to note: the wind", "", NONE),  # not before a counted noun
    ("There are a couple of options.", "", NONE),
    ("There is no need to worry.", "", NONE),
    ("Give it a few minutes of thought.", "", NOT_PRINTED.format("a few minutes")),  # but this is before a counted noun
    ("There are two things to check, and no hours to waste.", "no hours listed", NOT_PRINTED.format("no hours")),
    ("Two options: NOTAMs and weather.", "", NONE),  # the word is not directly before the noun
]


@pytest.mark.parametrize(("answer", "printed", "expected"), WORDS)
def test_a_number_word_is_checked_only_directly_before_a_counted_noun(answer, printed, expected):
    assert findings(answer, text_turn(printed)) == ([] if expected is None else [expected])


# --- arithmetic nobody printed -----------------------------------------------------------------------------------------


def test_a_difference_a_tool_did_not_print_is_a_finding_and_the_printed_spread_is_not():
    printed = "wind at 10 m 6 mph; gust 11.2 mph"
    assert findings("Gust minus the mean wind is 5.2 mph.", text_turn(printed)) == [NOT_PRINTED.format("5.2 mph")]
    assert findings("Gust spread is 5.2 mph.", text_turn(printed + "; gust spread 5.2 mph")) == []


@pytest.mark.parametrize("answer", ["Together that is 17.2 mph.", "The two winds average 8.6 mph.", "Wind is 6 mph and gusts 11.2 mph, 5.2 mph apart."])
def test_a_sum_or_an_average_no_tool_printed_is_a_finding(answer):
    found = findings(answer, text_turn("wind 6 mph; gust 11.2 mph"))
    assert found and all(f.endswith("is not a number any tool printed this turn, or a rounding of one") for f in found)


def test_a_converted_span_passes_only_when_the_rounding_rule_holds():
    turn = text_turn("flyable for 3 h 06 min")
    assert findings("Flyable for just over 3 hours.", turn) == []  # 186 min is 3.1 h, which rounds to 3
    assert findings("Flyable for 3.1 hours.", turn) == []
    assert findings("Flyable for 186 minutes.", turn) == []
    assert findings("Flyable for just over 4 hours.", turn) == [NOT_PRINTED.format("just over 4 hours")]
    assert findings("Flyable for 3.2 hours.", turn) == [NOT_PRINTED.format("3.2 hours")]
    assert findings("Flyable for 3 h 06 min.", turn) == []
    assert findings("Flyable for 3 h 07 min.", turn) == [NOT_PRINTED.format("3 h 07 min")]


def test_a_range_shares_its_unit_and_each_end_is_checked():
    turn = text_turn("wind at 10 m 4.9 mph; later 8.1 mph")
    assert findings("Wind runs 5-8 mph.", turn) == []
    assert findings("Wind runs 5 to 9 mph.", turn) == [NOT_PRINTED.format("9 mph")]


def test_a_number_equal_to_a_printed_limit_is_grounded_unless_a_value_that_crosses_rounds_onto_it():
    near = weather_turn(3.8, "mph", 4.0, ">=")  # 3.8 rounds to 4, the limit's number
    assert findings("The caution limit is 4 mph.", near) == [IS_THE_LIMIT.format("4 mph", "3.8 mph", "falls below")]  # the crossing wins over the limit-equal source
    assert findings("Gust spread is 4 mph.", near) == [IS_THE_LIMIT.format("4 mph", "3.8 mph", "falls below")]
    assert findings("The limit is 5 mph.", near) == [NOT_PRINTED.format("5 mph")]  # a number that is neither a value nor a limit
    far = weather_turn(3.0, "mph", 4.0, ">=")  # nothing rounds onto 4: the limit grounds it
    assert findings("The caution limit is 4 mph.", far) == []
    assert findings("Gust spread is 3 mph, under the 4 mph limit.", far) == []


def test_the_fixed_sentence_for_a_limit_a_crossing_value_rounds_onto():
    """The spec's own example, word for word: the retry reads it and knows the fix."""
    found = findings("Gust spread is 6 mph, which is over the limit.", weather_turn(6.04, "mph", 6.0, ">"))
    assert found == [
        "'6 mph' is the limit that gust spread 6.04 mph exceeds. Give each value exactly as the tool printed it. Don't restate limit values: code adds a line stating each limit reached."
    ]


# I10: the guard reads numbers, not prose. The same number gives the same answer in every frame of words, whatever the frame says about the limit.
FRAMES = [
    "Gust spread is {n}.",
    "Gust spread is {n}, over the limit.",
    "Gust spread is {n}, not over the limit.",
    "Gust spread is well clear of the {n} limit.",
    "The limit is {n}.",
    "The {n} limit applies.",
    "Gust spread never goes above the {n} threshold.",
    "Gust spread is basically fine next to the {n} limit.",
    "Gust spread is under, not over, the caution limit of {n}.",
]
NUMBER_ONLY = [  # (written, value, limit, comparison): what the number alone gives
    ("6 mph", 6.04, 6.0, ">"),  # lands on the limit, value over it
    ("about 6 mph", 6.04, 6.0, ">"),
    ("4 mph", 3.8, 4.0, ">="),
    ("9 mph", 9.3, 9.0, ">="),
    ("6 mph", 6.0, 6.0, ">"),  # exactly at the limit
    ("6 mph", 5.0, 6.0, ">"),  # the limit, clear of the value
    ("5 mph", 5.0, 6.0, ">"),  # the value, clear of the limit
    ("7 mph", 5.0, 6.0, ">"),  # neither
]


@pytest.mark.parametrize(("written", "value", "limit", "comparison"), NUMBER_ONLY, ids=[f"{w} vs {v} ({c} {lim:g})" for w, v, lim, c in NUMBER_ONLY])
def test_the_words_around_a_number_never_change_what_the_number_gives(written, value, limit, comparison):
    sources = weather_turn(value, "mph", limit, comparison)
    alone = findings(f"Gust spread is {written}.", sources)
    for frame in FRAMES:
        assert findings(frame.format(n=written), sources) == alone, frame


def test_a_sentence_about_a_values_position_is_out_of_scope():
    """Known limit (Dan 2026-11-08): the guard checks numbers, ids and verdict words; code states every limit position in its own lines. A prose sentence that
    contradicts those lines is not caught, so none of these is a finding for a value that is clear of its limit, or exactly at it."""
    over = weather_turn(7.0, "mph", 6.0, ">")
    assert findings("Gust spread is 7 mph, which is not over the limit.", over) == []
    assert findings("Gust spread is 7 mph and nowhere near the caution limit.", over) == []
    at = weather_turn(6.0, "mph", 6.0, ">")
    assert findings("Gust spread is 6 mph, so under the limit.", at) == []


def test_the_removed_prose_readers_are_gone():
    """I10: no clause, position or negation reading in guard.py."""
    from flysafelz.agent import guard

    for name in ("_read_position", "_written_as_limit", "_clause_start", "_position_finding", "_tie_finding", "_POSITION_WORD", "_NEGATING", "_FINALITY", "_TOPIC", "_NOT_FINALITY"):
        assert not hasattr(guard, name), name


def test_the_blue_mill_gust_spread_at_the_caution_limit_is_a_finding():
    """The real turn: a gust spread of 3.8 mph against a caution at >= 4 mph, written as 'about 4 mph, at the caution limit'."""
    found = findings("Gust spread about 4 mph, at the caution limit.", blue_mill_turn())
    assert any(f.startswith("'about 4 mph' rounds 3.8 mph across the 4 mph limit") for f in found), found


def test_a_unitless_year_a_list_marker_and_a_date_are_not_numbers():
    turn = text_turn("Saturday 31 Oct 10:00 GMT")
    assert findings("1. The window starts Saturday 31 Oct 10:00 GMT.\n2. It is the 2026 season.", turn) == []


def test_a_number_from_an_earlier_turn_is_a_finding():
    """Prior turns' tool results are not in TurnSources, so a number only they printed is not printed."""
    earlier = text_turn("wind at 10 m 9.9 mph")
    this_turn = text_turn("wind at 10 m 4.9 mph")
    assert findings("Earlier it was 9.9 mph.", earlier) == []
    assert findings("Earlier it was 9.9 mph.", this_turn) == [NOT_PRINTED.format("9.9 mph")]


def test_a_number_in_a_tool_text_with_no_artifact_is_printed():
    assert findings("Visibility is 2920 m.", turn_of(ToolResult("get_weather", "visibility 2920 m"))) == []


# --- F2: the guard never accepts a number, coordinate or phone no tool printed ----------------------------------------------


def test_random_numbers_absent_from_the_sources_always_produce_a_finding():
    """A property test with a seeded generator: numbers of six digits, written as the tools would, in each unit family, against sources that print only small values."""
    rng = random.Random(20261031)
    turn = text_turn("wind at 10 m 4.9 mph; visibility 2920 m; 2 NOTAMs within 2 NM; 98 %; 6 h 00 min")
    for _ in range(300):
        number = f"{rng.randint(100000, 999999)}.{rng.randint(0, 99):02d}"
        unit = rng.choice(["mph", "m", "km", "ft", "kt", "NM", "%", "h", "min", ""])
        found = findings(f"It is {number} {unit}.".replace(" .", "."), turn)
        assert found and all("not a number any tool printed" in f for f in found), (number, unit, found)


def test_a_limit_in_a_unit_the_value_cannot_be_compared_in_is_not_a_crossing():
    """A limit in per cent cannot say which side of it a speed is on, so only the rounding rule applies."""
    sources = weather_turn(9.3, "mph", 9.0, ">=", limit_unit="%")
    assert findings("Gust spread is about 9 mph.", sources) == []
    assert findings("Gust spread is about 10 mph.", sources) == [NOT_PRINTED.format("about 10 mph")]


def test_the_finding_is_stable():
    sources = weather_turn(6.04, "mph", 6.0, ">")
    assert findings("Gust spread is 6 mph, not over the limit.", sources) == findings("Gust spread is 6 mph, not over the limit.", sources) == [IS_THE_LIMIT.format("6 mph", "6.04 mph", "exceeds")]


# --- I10 rev.: the one fixed limit phrase "over|above|under|below|at the <number> <unit> limit" ------------------------------
# Its number must be a printed limit, its word must be what contracts.position_word gives, and not / n't / never right before it is a finding.
# Every other wording of a position stays unread (see the out-of-scope tests above).

NO_LIMIT = "'{}' names a limit that no tool printed this turn"
NEGATED = "'{}' follows a negation (not, n't, never); say where the value is with over, at or under and no negation"
WRONG_WORD = "'{}' says {}, but the value the tool printed, {}, is {} the {} limit"
SPREAD = (6.04, 6.0, ">")  # gust spread 6.04 mph against > 6 mph: over
WIND_OVER = (9.3, 9.0, ">=")  # 9.3 against >= 9 mph: over
WIND_AT = (9.0, 9.0, ">=")  # exactly at the limit
WIND_UNDER = (8.7, 9.0, ">=")
PHRASES = [
    # --- 6.04 mph gust spread against > 6 mph (NO-GO)
    ("gust spread 6.04 mph", SPREAD, NONE),
    ("gust spread about 6 mph", SPREAD, ACROSS.format("about 6 mph", "6.04 mph", "6 mph")),  # a bare number still gets the rounding check
    ("over the 6 mph limit", SPREAD, NONE),
    ("above the 6 mph limit", SPREAD, NONE),
    ("Over The 6 MPH Limit", SPREAD, NONE),
    ("over the 6.0 mph limit", SPREAD, NONE),
    ("under the 6 mph limit", SPREAD, WRONG_WORD.format("under the 6 mph limit", "under", "6.04 mph", "over", "6 mph")),
    ("below the 6 mph limit", SPREAD, WRONG_WORD.format("below the 6 mph limit", "below", "6.04 mph", "over", "6 mph")),
    ("at the 6 mph limit", SPREAD, WRONG_WORD.format("at the 6 mph limit", "at", "6.04 mph", "over", "6 mph")),
    ("not over the 6 mph limit", SPREAD, NEGATED.format("over the 6 mph limit")),
    ("isn't over the 6 mph limit", SPREAD, NEGATED.format("over the 6 mph limit")),
    ("isn\u2019t over the 6 mph limit", SPREAD, NEGATED.format("over the 6 mph limit")),
    ("**not** over the 6 mph limit", SPREAD, NEGATED.format("over the 6 mph limit")),  # emphasis markup around the negation is not a gap
    ("_not_ over the 6 mph limit", SPREAD, NEGATED.format("over the 6 mph limit")),
    ("~~not~~ over the 6 mph limit", SPREAD, NEGATED.format("over the 6 mph limit")),
    ("**isn't** over the 6 mph limit", SPREAD, NEGATED.format("over the 6 mph limit")),
    ("**isn\u2019t** over the 6 mph limit", SPREAD, NEGATED.format("over the 6 mph limit")),
    ("*never* above the 6 mph limit", SPREAD, NEGATED.format("above the 6 mph limit")),
    ("isn't **over the 6 mph limit**", SPREAD, NEGATED.format("over the 6 mph limit")),
    ("not _over the 6 mph limit_", SPREAD, NEGATED.format("over the 6 mph limit")),
    ("**not over the 6 mph limit**", SPREAD, NEGATED.format("over the 6 mph limit")),
    ("**not** at the 6 mph limit", SPREAD, NEGATED.format("at the 6 mph limit")),
    ("**over the 6 mph limit**", SPREAD, NONE),  # markup alone is no negation
    ("_under the 6 mph limit_", SPREAD, WRONG_WORD.format("under the 6 mph limit", "under", "6.04 mph", "over", "6 mph")),  # _italic_ edges do not hide the phrase
    ("**note** over the 6 mph limit", SPREAD, NONE),  # a word that merely ends in 'not'/starts like it is not a negation
    ("cannot go over the 6 mph limit", SPREAD, NONE),  # "go" sits between: the documented known gap
    ("doesn't go over the 6 mph limit", SPREAD, NONE),  # the word before the phrase is "go", not a negation: no other negation is read
    ("never over the 6 mph limit", SPREAD, NEGATED.format("over the 6 mph limit")),
    ("Never over the 6 mph limit", SPREAD, NEGATED.format("over the 6 mph limit")),
    ("not at the 6 mph limit", SPREAD, NEGATED.format("at the 6 mph limit")),
    ("over the 7 mph limit", SPREAD, NO_LIMIT.format("over the 7 mph limit")),  # 7 mph is no printed limit
    ("over the 6 kt limit", SPREAD, NO_LIMIT.format("over the 6 kt limit")),  # nor is 6 kt: a limit is grounded in its own unit
    ("6.04 mph, over the 6 mph limit", SPREAD, NONE),
    ("6.04 mph, under the 6 mph limit", SPREAD, WRONG_WORD.format("under the 6 mph limit", "under", "6.04 mph", "over", "6 mph")),
    # --- wind 9.3 mph against >= 9 mph (caution)
    ("wind 9.3 mph, over the 9 mph limit", WIND_OVER, NONE),
    ("wind over the 9 mph limit", WIND_OVER, NONE),
    ("wind at the 9 mph limit", WIND_OVER, WRONG_WORD.format("at the 9 mph limit", "at", "9.3 mph", "over", "9 mph")),
    ("wind under the 9 mph limit", WIND_OVER, WRONG_WORD.format("under the 9 mph limit", "under", "9.3 mph", "over", "9 mph")),
    # --- wind 9.0 mph against >= 9 mph: exactly at it
    ("wind at the 9 mph limit", WIND_AT, NONE),
    ("wind 9 mph, at the 9 mph limit", WIND_AT, NONE),
    ("wind over the 9 mph limit", WIND_AT, WRONG_WORD.format("over the 9 mph limit", "over", "9 mph", "at", "9 mph")),
    ("wind under the 9 mph limit", WIND_AT, WRONG_WORD.format("under the 9 mph limit", "under", "9 mph", "at", "9 mph")),
    # --- wind 8.7 mph against >= 9 mph: not reached
    ("wind under the 9 mph limit", WIND_UNDER, NONE),
    ("wind below the 9 mph limit", WIND_UNDER, NONE),
    ("wind at the 9 mph limit", WIND_UNDER, WRONG_WORD.format("at the 9 mph limit", "at", "8.7 mph", "under", "9 mph")),
    ("wind over the 9 mph limit", WIND_UNDER, WRONG_WORD.format("over the 9 mph limit", "over", "8.7 mph", "under", "9 mph")),
    # --- a < limit: past it is below it
    ("visibility over the 5 mph limit", (4.0, 5.0, "<"), NONE),
    ("visibility under the 5 mph limit", (6.0, 5.0, "<"), NONE),
    ("visibility at the 5 mph limit", (5.0, 5.0, "<="), NONE),
    ("visibility under the 5 mph limit", (4.0, 5.0, "<"), WRONG_WORD.format("under the 5 mph limit", "under", "4 mph", "over", "5 mph")),
    # --- the same limit written in another unit of the table (9 mph is 7.8 kt)
    ("wind over the 7.8 kt limit", WIND_OVER, NONE),
    ("wind under the 7.8 kt limit", WIND_OVER, WRONG_WORD.format("under the 7.8 kt limit", "under", "9.3 mph", "over", "9 mph")),
]


@pytest.mark.parametrize(("answer", "world", "expected"), PHRASES, ids=[f"{w[1:]} {a}" for a, w, _ in PHRASES])
def test_the_limit_phrase_is_checked_for_its_number_its_word_and_a_negation(answer, world, expected):
    assert findings(f"Reading: {answer}.", weather_turn(world[0], "mph", world[1], world[2])) == ([] if expected is None else [expected])


def test_a_limit_phrase_that_is_not_the_fixed_phrase_is_not_read():
    """Only 'over|above|under|below|at the <number> <unit> limit' is a phrase; the number in any other wording is a bare number, whatever the words say."""
    clear = weather_turn(7.0, "mph", 6.0, ">")  # the 6 is a printed limit and nothing rounds onto it
    for answer in ("Gust spread is 7 mph, past the 6 mph limit.", "Gust spread is 7 mph, under 6 mph.", "Gust spread is 7 mph and the 6 mph threshold is under.", "Gust spread is 7 mph, under the 6 mph limits."):
        assert findings(answer, clear) == [], answer
    near = weather_turn(6.04, "mph", 6.0, ">")
    assert findings("Gust spread is 6.04 mph, past the 6.0 mph limit.", near) == [IS_THE_LIMIT.format("6.0 mph", "6.04 mph", "exceeds")]  # number-only, as before
    assert findings("Gust spread is over the limit of 6 mph.", near) == [IS_THE_LIMIT.format("6 mph", "6.04 mph", "exceeds")]


def test_a_phrase_number_is_not_also_checked_as_a_value():
    """The 6 in 'over the 6 mph limit' is the limit: it gives one finding at most, never the rounding finding of a bare 6 mph beside 6.04."""
    turn = weather_turn(6.04, "mph", 6.0, ">")
    assert findings("Gust spread is 6.04 mph, over the 6 mph limit.", turn) == []
    assert findings("Gust spread is 6.04 mph, under the 6 mph limit.", turn) == [WRONG_WORD.format("under the 6 mph limit", "under", "6.04 mph", "over", "6 mph")]


def test_a_phrase_is_read_where_the_answer_wrote_it_not_in_a_quoted_excerpt():
    turn = weather_turn(6.04, "mph", 6.0, ">")
    assert findings('The NOTAM says "wind under the 6 mph limit" for the site.', turn) == [WRONG_WORD.format("under the 6 mph limit", "under", "6.04 mph", "over", "6 mph")]  # not a NOTAM excerpt


def _two_hours(first: tuple[float, float, str], second: tuple[float, float, str]):
    """One turn that has two weather results, one value each."""
    a, b = weather_turn(first[0], "mph", first[1], first[2]), weather_turn(second[0], "mph", second[1], second[2])
    return replace(a, results=a.results + b.results)


def test_with_two_values_the_sentence_names_the_one_it_means_otherwise_all_must_agree():
    turn = _two_hours((7.0, 6.0, ">"), (5.0, 6.0, ">"))  # one over the limit, one under it
    assert findings("Gust spread is 7 mph, over the 6 mph limit.", turn) == []
    assert findings("Gust spread is 5 mph, under the 6 mph limit.", turn) == []
    assert findings("Gust spread is 5 mph, over the 6 mph limit.", turn) == [WRONG_WORD.format("over the 6 mph limit", "over", "5 mph", "under", "6 mph")]
    # the sentence names no value, so it must be true of both; it is not, and the guard does not pick one
    assert len(findings("Gust spread is over the 6 mph limit.", turn)) == 1
    assert len(findings("Gust spread is under the 6 mph limit.", turn)) == 1
    agree = _two_hours((7.0, 6.0, ">"), (8.0, 6.0, ">"))
    assert findings("Gust spread is over the 6 mph limit.", agree) == []


def test_a_limit_only_a_tools_text_gives_is_no_printed_limit():
    """Limits come from an artifact's rule limits; a number in a tool's text is a number, not a limit, so a phrase naming it is a finding."""
    turn = text_turn("wind at 10 m 4.9 mph; caution at 6 mph")
    assert findings("Wind is over the 6 mph limit.", turn) == [NO_LIMIT.format("over the 6 mph limit")]  # only an artifact prints a limit


def test_a_limit_with_no_comparison_has_no_position_to_check():
    turn = weather_turn(7.0, "mph", 6.0, "=")
    assert findings("Gust spread is 7 mph, under the 6 mph limit.", turn) == []
