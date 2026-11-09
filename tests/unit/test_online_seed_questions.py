"""G1 (7 PR-5.F1): the rules `evals/online/seed_questions.yaml` keeps, checked on the real file, and each rule's own negative fixture showing that it fires.

Counts and split, the five landing zones and their points, the airspace questions, unique ids and required fields; relative times only (no ISO date, `dd/mm`, month name or four-digit year);
no NOTAM id (`contracts.NOTAM_ID_RE`); no question that is one of the dataset's 44. Phone numbers: none is written here.
"""

import re
from collections import Counter
from pathlib import Path

import pytest
import yaml

from evals.online import seed
from flysafelz.contracts import NOTAM_ID_RE

ROOT = Path(__file__).resolve().parents[2]
SEED_FILE = ROOT / "evals" / "online" / "seed_questions.yaml"
EXAMPLES = ROOT / "evals" / "agent" / "data" / "examples.yaml"
WEST_SLEEKBURN_CASE = ROOT / "fixtures" / "west_sleekburn_2026-10-31" / "case.yaml"
SPLIT = {"assess": 4, "explain": 3, "clarify": 2, "decline": 2}
LZ_CASES = {"Blue Mill": "blue_mill", "Druridge Bay": "druridge_bay", "Buttermere": "buttermere", "Crookham": "crookham"}  # the dataset's case names, for the four points that come from it
WEST_SLEEKBURN = "West Sleekburn"

MONTHS = r"jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|june?|july?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?"  # `may` is below: it is also a word
DATE_RULES = {
    "an ISO date": re.compile(r"\b\d{4}-\d{2}-\d{2}\b"),
    "a dd/mm date": re.compile(r"\b\d{1,2}/\d{1,2}(?:/\d{2,4})?\b"),
    "a month name": re.compile(rf"\b(?:{MONTHS})\b|\b\d{{1,2}}(?:st|nd|rd|th)?(?:\s+of)?\s+may\b|\bmay\s+\d{{1,2}}\b", re.IGNORECASE),
    "a four-digit year": re.compile(r"\b(?:19|20)\d{2}\b"),
}


def questions() -> list[seed.Question]:
    return seed.load_questions(SEED_FILE)


def raw_entries() -> list[dict]:
    return yaml.safe_load(SEED_FILE.read_text(encoding="utf-8"))


def date_rules_hit(text: str) -> list[str]:
    return [name for name, pattern in DATE_RULES.items() if pattern.search(text)]


def normalised(text: str) -> str:
    return " ".join(text.casefold().split())


def dataset_questions() -> list[str]:
    return [entry["inputs"]["question"] for entry in yaml.safe_load(EXAMPLES.read_text(encoding="utf-8"))]


def dataset_points(case_prefix: str) -> set[tuple[float, float]]:
    found = set()
    for entry in yaml.safe_load(EXAMPLES.read_text(encoding="utf-8")):
        if entry["fixtures"]["case"].startswith(case_prefix + "_"):
            found.add((entry["inputs"]["lat"], entry["inputs"]["lon"]))
    return found


# --- the file ---------------------------------------------------------------------------------------------------------------------------


def test_the_file_holds_11_questions_split_4_assess_3_explain_2_clarify_2_decline():
    found = questions()
    assert len(found) == 11
    assert Counter(question.intended_path for question in found) == SPLIT


def test_every_entry_has_exactly_the_six_fields_and_ids_are_unique():
    entries = raw_entries()
    assert [set(entry) for entry in entries] == [set(seed.FIELDS)] * 11
    ids = [entry["id"] for entry in entries]
    assert len(set(ids)) == 11 and all(isinstance(name, str) and name.strip() for name in ids)


def test_two_explain_questions_are_about_airspace_and_each_names_its_own_lz_and_the_third_is_about_the_rules():
    explain = [question for question in questions() if question.intended_path == "explain"]
    airspace = [question for question in explain if "airspace" in question.question.casefold()]
    rules = [question for question in explain if question not in airspace]
    assert len(airspace) == 2 and len(rules) == 1
    for question in airspace:
        assert question.lz_name.casefold() in question.question.casefold(), question.id
    assert len({question.lz_name for question in airspace}) == 2  # two different landing zones: the tool is exercised at two places


def test_the_example_from_the_spec_is_asked_for_west_sleekburn():
    assert "What airspace is above West Sleekburn, and does any of it affect me?" in [question.question for question in questions()]


def test_the_landing_zones_are_exactly_the_five():
    assert {question.lz_name for question in questions()} == {WEST_SLEEKBURN, *LZ_CASES}


def test_west_sleekburn_is_the_point_of_its_case_file():
    case = yaml.safe_load(WEST_SLEEKBURN_CASE.read_text(encoding="utf-8"))
    assert (case["lat"], case["lon"]) == (55.1576, -1.5699)
    assert {(question.lat, question.lon) for question in questions() if question.lz_name == WEST_SLEEKBURN} == {(case["lat"], case["lon"])}


@pytest.mark.parametrize("lz_name, prefix", sorted(LZ_CASES.items()))
def test_the_other_four_points_are_the_ones_examples_yaml_holds(lz_name, prefix):
    (point,) = dataset_points(prefix)  # the dataset has one point for the zone
    assert {(question.lat, question.lon) for question in questions() if question.lz_name == lz_name} == {point}


def test_every_zone_is_asked_about_at_least_once_and_always_at_the_same_point():
    by_zone: dict[str, set[tuple[float, float]]] = {}
    for question in questions():
        by_zone.setdefault(question.lz_name, set()).add((question.lat, question.lon))
    assert len(by_zone) == 5 and all(len(points) == 1 for points in by_zone.values())


# --- relative times only ----------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("question", questions(), ids=lambda question: question.id)
def test_a_question_has_no_date_in_it(question):
    assert date_rules_hit(question.question) == []


@pytest.mark.parametrize(
    "text, rule",
    [
        ("Is 2026-11-08 any good for a fly?", "an ISO date"),
        ("Can I fly on 8/11?", "a dd/mm date"),
        ("Can I fly on 08/11/26?", "a dd/mm date"),
        ("Can I fly on 8 November?", "a month name"),
        ("Is Nov 8 any good?", "a month name"),
        ("Is the 5th of May any good?", "a month name"),
        ("Is May 5 any good?", "a month name"),
        ("What was it like in 2026?", "a four-digit year"),
    ],
)
def test_each_date_rule_fires_on_its_negative_fixture(text, rule):
    assert rule in date_rules_hit(text)


@pytest.mark.parametrize("text", ["Can I fly tomorrow morning?", "May I fly this afternoon?", "Is it worth going out this evening, around 6?", "Is the wind under 12 mph at 10 am?"])
def test_the_date_rules_leave_a_relative_time_and_a_plain_number_alone(text):
    assert date_rules_hit(text) == []


# --- no NOTAM id ------------------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("question", questions(), ids=lambda question: question.id)
def test_a_question_names_no_notam_id(question):
    assert NOTAM_ID_RE.search(question.question) is None


@pytest.mark.parametrize("text", ["What does H1234/26 mean?", "Is V32/26 still on?", "Tell me about A0001/26 please", "See J3857/26."])
def test_the_notam_rule_fires_on_an_id_short_ones_included(text):
    assert NOTAM_ID_RE.search(text) is not None


# --- novelty ----------------------------------------------------------------------------------------------------------------------------


def test_the_dataset_has_its_44_questions():
    assert len(dataset_questions()) == 44


@pytest.mark.parametrize("question", questions(), ids=lambda question: question.id)
def test_a_question_is_not_one_of_the_datasets(question):
    assert normalised(question.question) not in {normalised(text) for text in dataset_questions()}


def test_the_novelty_rule_fires_on_a_dataset_question_whatever_its_case_or_spacing():
    dataset = {normalised(text) for text in dataset_questions()}
    original = "Can I fly Blue Mill today?"
    assert original in dataset_questions()
    assert normalised("  can i   FLY blue mill\ttoday? ") in dataset
    assert normalised("Can I fly Blue Mill tomorrow?") not in dataset


# --- the loader reads the real file -----------------------------------------------------------------------------------------------------


def test_the_loader_gives_the_questions_in_file_order_with_numbers_as_numbers():
    found = questions()
    assert [question.id for question in found] == [entry["id"] for entry in raw_entries()]
    assert all(isinstance(question.lat, float) and isinstance(question.lon, float) for question in found)
