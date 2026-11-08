"""The limit phrase (PR-11.F1): the tools write where a value sits against its limit as `over|at|under the N unit limit`, the word from `position_word`.

Each case runs the real tool (or `limit_phrase` itself) and reads the guard's verdict on the result, so the tools and the guard are held to one phrase.
"""

import re
from datetime import UTC, datetime

import pytest
from factories import weather_forecast
from guard_world import blue_mill_turn, findings
from tool_helpers import calls
from tool_world import WINDOW, World, forecast_fixture, go

from flysafelz.agent.tools.common import limit_phrase
from flysafelz.contracts import position_word

PHRASE = re.compile(r"(?P<word>over|at|under) the (?P<number>[\d.]+) (?P<unit>\S+) limit")
ASSESS = [calls("assess", **WINDOW)]
WEATHER = [calls("get_weather", **WINDOW)]


@pytest.mark.parametrize(
    ("value", "limit", "comparison", "unit", "phrase"),
    [
        (6.04, 6.0, ">", "mph", "over the 6 mph limit"),
        (6.0, 6.0, ">", "mph", "at the 6 mph limit"),
        (5.9, 6.0, ">", "mph", "under the 6 mph limit"),
        (9.3, 9.0, ">=", "mph", "over the 9 mph limit"),
        (9.0, 9.0, ">=", "mph", "at the 9 mph limit"),
        (8.7, 9.0, ">=", "mph", "under the 9 mph limit"),
        (2920.0, 5000.0, "<=", "m", "over the 5000 m limit"),  # past a lower-is-worse limit is still `over`
        (24140.0, 5000.0, "<=", "m", "under the 5000 m limit"),
        (0.14, 0.1, ">", "mm/h", "over the 0.1 mm/h limit"),
    ],
)
def test_limit_phrase_is_position_word_of_the_true_value(value, limit, comparison, unit, phrase):
    assert limit_phrase(value, limit, comparison, unit) == phrase
    assert phrase.split()[0] == position_word(value, limit, comparison)


def test_the_phrase_after_every_weather_limit_is_the_position_of_the_artifact_value():
    run = go(World(forecast=forecast_fixture("full_48h")), WEATHER)
    text, artifact = run.text("get_weather"), run.artifact("get_weather")
    row = next(line for line in text.splitlines() if line.startswith("Sat 31 Oct 10:00 GMT"))
    hour = artifact.hours[0]
    expected = [limit_phrase(value.value, limit.limit, limit.comparison, limit.unit) for value in hour.values if value.value is not None for limit in value.limits]
    assert expected
    assert [match.group(0) for match in PHRASE.finditer(row)] == expected


def test_a_value_the_forecast_lacks_gets_no_position():
    hours = (9, 10)  # the 10:00 row takes the worse of 09:00 and 10:00, so both must lack the value
    forecast = weather_forecast(
        datetime(2026, 10, 31, 8, tzinfo=UTC), datetime(2026, 10, 31, 16, tzinfo=UTC), changes={datetime(2026, 10, 31, h, tzinfo=UTC): {"cloud_cover_low": None} for h in hours}
    )
    row = next(line for line in go(World(forecast=forecast), WEATHER).text("get_weather").splitlines() if line.startswith("Sat 31 Oct 10:00 GMT"))
    assert "low cloud not forecast (caution >= 90 %)" in row


def test_assess_caution_lines_say_where_the_value_sits_and_the_guard_accepts_that():
    turn = blue_mill_turn()
    text = next(result.text for result in turn.results if result.name == "assess")
    assert "    caution: gust spread 4.7 mph: over the 4 mph limit (caution)" in text
    assert "    caution: low cloud 98 %: over the 90 % limit (caution)" in text
    assert findings("Gust spread reaches 4.7 mph, over the 4 mph limit.", turn) == []


def test_the_guard_rejects_the_wrong_word_for_a_value_a_tool_printed():
    turn = blue_mill_turn()
    wrong = findings("Gust spread reaches 4.7 mph, under the 4 mph limit.", turn)
    assert len(wrong) == 1 and "under the 4 mph limit" in wrong[0]


def test_a_no_go_weather_blocker_carries_its_numbers_and_the_line_places_the_value():
    forecast = weather_forecast(
        datetime(2026, 10, 31, 8, tzinfo=UTC), datetime(2026, 10, 31, 16, tzinfo=UTC), changes={datetime(2026, 10, 31, hour, tzinfo=UTC): {"precipitation": 0.6} for hour in (10, 11)}
    )
    run = go(World(forecast=forecast), ASSESS)
    blockers = [b for segment in run.artifact("assess").segments for b in segment.blockers if b.source == "weather"]
    assert blockers
    blocker = blockers[0]
    assert (blocker.rule_id, blocker.value, blocker.limit, blocker.unit, blocker.label) == ("precipitation_no_go", 0.6, 0.1, "mm/h", "precipitation")
    assert "    blocked (weather): precipitation 0.6 mm/h: over the 0.1 mm/h limit (NO-GO)" in run.text("assess")


def test_a_caution_whose_text_does_not_match_its_numbers_keeps_ticket_2s_words():
    from flysafelz.agent.tools.assess_tool import finding_parts

    assert finding_parts("gust spread 4.7 mph, limit 4", 4.7, 4.0, "mph") is not None
    assert finding_parts("gust spread 4.7 mph, limit 4", 5.7, 4.0, "mph") is None  # the text's value is not the finding's
    assert finding_parts("gust spread 4.7 mph, limit 4", 4.7, 5.0, "mph") is None  # nor its limit
    assert finding_parts("gust spread 4.7 mph, limit 4", 4.7, 4.0, "m") is None  # nor its unit
    assert finding_parts("twilight ends in 20 min", 4.7, 4.0, "mph") is None
