"""G14 (c): NOTAM ids, <PHONE_n> placeholders, real-looking phones and coordinates, and the two helpers the guard added to contracts.py and notam_area.py."""

import random

import pytest
from guard_world import findings, point_feature, text_turn, turn, turn_of, weather_turn

from flysafelz.agent.guard import ToolResult
from flysafelz.agent.tools.artifacts import GuidanceArtifact, GuidanceChunk
from flysafelz.contracts import NOTAM_ID_RE


@pytest.mark.parametrize("notam_id", ["H7171/26", "J3857/26"])
def test_a_notam_id_in_a_tool_result_passes(notam_id):
    assert findings(f"See {notam_id}.", turn()) == []


@pytest.mark.parametrize("notam_id", ["H7172/26", "V99/26", "Q172/26", "H717/26", "A0001/26", "H7171/27"])
def test_an_invented_notam_id_is_a_finding_whatever_its_digit_count(notam_id):
    assert findings(f"See {notam_id}.", turn()) == [f"NOTAM id '{notam_id}' is not in any tool result this turn"]


def test_a_short_id_a_tool_printed_passes():
    assert findings("See V32/26.", text_turn("V32/26 is active")) == []


def test_a_notam_id_is_not_read_as_a_number_or_a_date():
    assert findings("See H7171/26.", turn()) == []


def test_a_placeholder_must_be_the_threads_and_in_a_tool_result_this_turn():
    assert findings("Phone <PHONE_1>.", turn()) == []
    assert findings("Phone <PHONE_2>.", turn()) == ["'<PHONE_2>' is not one of this thread's phone placeholders"]
    other_turn = text_turn("no phone here", phones=("<PHONE_1>",))
    assert findings("Phone <PHONE_1>.", other_turn) == ["'<PHONE_1>' is not in any tool result this turn"]


@pytest.mark.parametrize("phone", ["01632 960001", "+44 1632 960001", "07700 900123", "07700900123"])
def test_a_real_looking_phone_number_is_a_finding(phone):
    found = findings(f"Ring {phone} to ask.", turn())
    assert len(found) == 1 and found[0].endswith("looks like a real phone number, which only a placeholder may stand for")


def test_a_phone_number_inside_a_quoted_tool_excerpt_is_still_a_finding():
    """The tools redact phones before the model reads them, so a real number in the answer has another source and is refused, even in quotes."""
    found = findings('It says "FOR INFO 07700 900123".', text_turn("FOR INFO 07700 900123"))
    assert len(found) == 1 and "real phone number" in found[0]


# --- coordinates -------------------------------------------------------------------------------------------------------


COORD_TURN = text_turn("NOTAM area centre 534732N 0000459W, radius 0.5 NM", lz=(53.8028, -0.0866))


@pytest.mark.parametrize("written", ["53.8028, -0.0866", "53.8028N 0.0866W", "53.8028 N, 0.0866 W", "53.80, -0.09", "53.79, -0.08"])
def test_the_landing_zone_or_a_notams_converted_point_passes(written):
    assert findings(f"Position {written}.", COORD_TURN) == []


def test_a_notam_dms_point_converted_to_decimal_passes():
    assert findings("The area is centred on 53.7922, -0.0831.", COORD_TURN) == []


@pytest.mark.parametrize("written", ["53.81, -0.0866", "53.8028, -0.0900", "54.1234, -0.0866", "53.8028N 0.0866E"])
def test_a_coordinate_pair_no_tool_printed_is_a_finding(written):
    found = findings(f"Position {written}.", COORD_TURN)
    assert len(found) == 1 and found[0].endswith("is not a place any tool printed this turn")


@pytest.mark.parametrize("written", ["534732N 0000459W", "534732N0000459W", "534732n 0000459w", "534810N 0000512W"])
def test_a_dms_pair_the_answer_writes_passes_when_a_tool_printed_that_place(written):
    """The first is the NOTAM's own text; the last is the landing zone 53.8028, -0.0866 as degrees, minutes, seconds."""
    assert findings(f"The area is centred on {written}.", COORD_TURN) == []


@pytest.mark.parametrize("written", ["534800N 0001000W", "534733N 0000459W", "534732N 0000500W", "534732S 0000459W", "534732N 0000459E"])
def test_a_dms_pair_no_tool_printed_is_a_finding(written):
    assert findings(f"The area is centred on {written}.", COORD_TURN) == [f"'{written}' is not a place any tool printed this turn"]


IMPOSSIBLE = "'{}' is not a possible place, so no tool printed it"
NOT_PRINTED = "'{}' is not a place any tool printed this turn"


@pytest.mark.parametrize(
    ("answer", "expected"),
    [
        ("Centre 534800N 0001000W and 995959N 0001000W.", [NOT_PRINTED.format("534800N 0001000W"), IMPOSSIBLE.format("995959N 0001000W")]),  # latitude past 90
        ("Centre 534800N 0001000W, also 539999N 0001000W.", [NOT_PRINTED.format("534800N 0001000W"), IMPOSSIBLE.format("539999N 0001000W")]),  # minutes past 59
        ("Centre 995959N 0001000W, then 534800N 0001000W.", [IMPOSSIBLE.format("995959N 0001000W"), NOT_PRINTED.format("534800N 0001000W")]),  # the bad pair first
        ("Centre 534732N 0000459W, not 539999N 0001000W.", [IMPOSSIBLE.format("539999N 0001000W")]),  # a printed pair next to a bad one passes
        ("Centre 534732N 0000459W, not 534732N 0006099W.", [IMPOSSIBLE.format("534732N 0006099W")]),  # seconds past 59 in the longitude
        ("Centre 534800N 0001000W, 534801N 0001000W and 995959N 0001000W.", [NOT_PRINTED.format("534800N 0001000W"), NOT_PRINTED.format("534801N 0001000W"), IMPOSSIBLE.format("995959N 0001000W")]),
    ],
)
def test_one_impossible_dms_pair_does_not_switch_off_the_check_of_the_others(answer, expected):
    """R7: an invented place next to an impossible one is still a finding, and the impossible one is a finding of its own."""
    assert findings(answer, text_turn("centre 534732N 0000459W")) == expected


def test_an_impossible_pair_in_a_tool_text_does_not_hide_the_possible_one_beside_it():
    printed = text_turn("area 995959N 0001000W and 534732N 0000459W")
    assert findings("The area is centred on 534732N 0000459W.", printed) == []
    assert findings("The area is centred on 534800N 0001000W.", printed) == [NOT_PRINTED.format("534800N 0001000W")]


def test_a_dms_pair_in_a_quoted_excerpt_is_not_checked_and_the_same_text_unquoted_is():
    turn = text_turn("area 534800N 0001000W is active, contact the unit")
    assert findings('It says "area 534800N 0001000W is active".', turn) == []
    assert findings("The area 534800N 0001000W is active.", turn) == []  # printed, so it passes
    assert findings("The area 534801N 0001000W is active.", turn) == ["'534801N 0001000W' is not a place any tool printed this turn"]


def test_a_coordinate_pair_follows_its_written_decimals():
    assert findings("Position 53.80, -0.09.", COORD_TURN) == []
    assert findings("Position 53.8, -0.1.", COORD_TURN) == []
    assert findings("Position 53.8029, -0.0866.", COORD_TURN) != []


def test_two_measurements_with_units_are_not_a_coordinate_pair():
    found = findings("Gust 5.2 mph, 3.1 mph.", text_turn("gust 5.2 mph then 3.1 mph"))
    assert found == []


def test_a_pair_outside_the_uk_is_not_read_as_a_place():
    assert findings("It is 48.8566, 2.3522.", turn()) != [] and all("not a place" not in f for f in findings("It is 48.8566, 2.3522.", turn()))


@pytest.mark.parametrize("not_an_id", ["XH1234/26", "H12345/26", "AB1/26", "H1234/2026", "H1234/6"])
def test_a_longer_or_malformed_token_is_not_read_as_a_notam_id(not_an_id):
    assert NOTAM_ID_RE.search(not_an_id) is None
    assert findings(f"See {not_an_id}.", turn()) == []


@pytest.mark.parametrize("notam_id", ["Q172/26", "G110/26", "V32/26"])
def test_a_real_short_id_from_this_turns_tools_passes_and_gives_no_number_finding(notam_id):
    assert findings(f"Also {notam_id} applies.", text_turn(f"{notam_id} active 0800-1800")) == []


def test_an_invented_short_id_fails_even_when_a_real_short_id_is_in_the_turn():
    assert findings("See V99/26.", text_turn("V32/26 and Q172/26")) == ["NOTAM id 'V99/26' is not in any tool result this turn"]


# --- the two helpers the guard added elsewhere ---------------------------------------------------------------------------


def test_the_one_notam_id_pattern_takes_one_to_four_digits_and_two_year_digits():
    assert NOTAM_ID_RE.findall("H7171/26 Q172/26 V32/26 J1/26 and not H71712/26, h7171/26, H7171/2026 or 7171/26") == ["H7171/26", "Q172/26", "V32/26", "J1/26"]


def test_coordinates_in_reads_every_dms_pair_in_order_and_nothing_else():
    from flysafelz.notam_area import coordinates_in

    points = coordinates_in("AREA 534732N 0000459W TO 534800N 0001000E")
    assert [(round(lat, 4), round(lon, 4)) for lat, lon in points] == [(53.7922, -0.0831), (53.8, 0.1667)]
    assert coordinates_in("NO COORDINATES HERE, 5 NM") == []
    assert [(round(lat, 4), round(lon, 4)) for lat, lon in coordinates_in("995959N 0001000W THEN 534800N 0001000E")] == [(53.8, 0.1667)]  # a pair that is no place is left out
    assert coordinates_in("") == []


def test_the_guard_imports_no_model_network_or_file_module():
    import ast
    from pathlib import Path

    from flysafelz.agent import guard

    banned = {"httpx", "requests", "socket", "urllib", "openai", "langchain", "langchain_core", "langgraph", "pathlib", "subprocess", "os", "random", "time"}
    imported = {a.name.split(".")[0] for n in ast.walk(ast.parse(Path(guard.__file__).read_text())) if isinstance(n, ast.Import) for a in n.names}
    imported |= {n.module.split(".")[0] for n in ast.walk(ast.parse(Path(guard.__file__).read_text())) if isinstance(n, ast.ImportFrom) and n.module}
    assert not imported & banned


def test_random_ids_coordinates_and_phones_absent_from_the_sources_always_produce_a_finding():
    """F2, with a seeded generator: whatever is written that no tool printed this turn is a finding."""
    rng = random.Random(20261031)
    sources = text_turn("H7171/26 at 534732N 0000459W; call <PHONE_1>", phones=("<PHONE_1>",), lz=(53.8028, -0.0866))
    for _ in range(200):
        notam_id = f"{rng.choice('ABCDEFGHJKLMNPQRSTUVWXYZ')}{rng.randint(1, 9999)}/{rng.randint(20, 29)}"
        if notam_id == "H7171/26":
            continue
        assert findings(f"See {notam_id}.", sources) == [f"NOTAM id '{notam_id}' is not in any tool result this turn"]
        lat, lon = f"{rng.uniform(50.0, 58.0):.4f}", f"-{rng.uniform(0.1, 5.0):.4f}"
        if abs(float(lat) - 53.8028) < 0.001 or abs(float(lat) - 53.7922) < 0.001:
            continue
        assert findings(f"Position {lat}, {lon}.", sources) == [f"'{lat}, {lon}' is not a place any tool printed this turn"]
        dms = f"{rng.randint(50, 58)}{rng.randint(0, 59):02d}{rng.randint(0, 59):02d}N {rng.randint(1, 5):03d}{rng.randint(0, 59):02d}{rng.randint(0, 59):02d}W"
        if dms != "534732N 0000459W":
            assert findings(f"Centre {dms}.", sources) == [f"'{dms}' is not a place any tool printed this turn"], dms
        phone = f"07{rng.randint(100, 999)} {rng.randint(100000, 999999)}"
        found = findings(f"Ring {phone}.", sources)
        assert len(found) == 1 and found[0].startswith(f"'{phone}' looks like a real phone number"), (phone, found)


def test_an_overlay_point_and_the_dms_in_an_overlays_text_are_places_a_tool_printed():
    turn_with_map = weather_turn(5.0, "mph", overlays=[point_feature(53.9, -0.5), point_feature(53.2, -1.1, text="AREA 534732N 0000459W")])
    assert findings("Marked at 53.9, -0.5 and 53.2, -1.1 and 53.7922, -0.0831.", turn_with_map) == []
    assert findings("Marked at 53.9, -0.6.", turn_with_map) == ["'53.9, -0.6' is not a place any tool printed this turn"]


# --- section numbers a guidance tool printed ---------------------------------------------------------------------------


def guidance_turn(*sections: str):
    chunks = [GuidanceChunk(document="caa-cap-1234", edition="2026", section=s, title="Rules", text=f"See section {s}") for s in sections]
    artifact = GuidanceArtifact(query="rules", chunks=chunks)
    return turn_of(ToolResult("search_guidance", " ".join(f"Section {s}" for s in sections), artifact))


def test_a_multi_part_section_number_a_tool_printed_passes_and_another_fails():
    sources = guidance_turn("4.2.1")
    assert findings("See section 4.2.1 of the guidance.", sources) == []
    assert findings("See section 4.2.2 of the guidance.", sources) == ["'4.2.2' is not a section number any tool printed this turn"]


def test_a_two_part_section_is_a_number_a_guidance_tool_printed():
    assert findings("See section 4.2.", guidance_turn("4.2")) == []
    assert findings("See section 4.3.", guidance_turn("4.2")) == ["'4.3' is not a number any tool printed this turn, or a rounding of one"]
