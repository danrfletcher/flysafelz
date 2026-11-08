"""The judge's 8 prompt examples (5a PR-7.F1, F5/F6): their shape, that they are invented, that their phone numbers are fiction, and that all of them are in the system prompt."""

import re
from datetime import datetime
from pathlib import Path

import pytest
import yaml
from judge_helpers import dev_example
from phone_leaks import digit_runs

from evals.agent.judge import CRITERIA, DEFINITIONS, EXAMPLES_PATH, example_reason, judge_message, load_examples, lz_name_from_case, render_message, system_prompt
from flysafelz.redact import redact_phones

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "evals" / "agent" / "data"
FIXTURES = REPO / "fixtures"
FICTION_PREFIXES = ("0163296", "07700900")
NOTAM_ID = re.compile(r"\b[A-Z]\d{3,4}/\d{2}\b")


def raw_examples() -> dict:
    return yaml.safe_load(EXAMPLES_PATH.read_text(encoding="utf-8"))


# --- shape ---------------------------------------------------------------------------------------------------------------------


def test_the_file_has_the_four_criteria_each_with_exactly_one_pass_and_one_fail():
    data = raw_examples()
    assert list(data) == list(CRITERIA)
    for criterion in CRITERIA:
        assert sorted(data[criterion]) == ["fail", "pass"]


def test_every_example_has_its_message_parts_and_its_reason():
    for criterion, grades in load_examples().items():
        for grade, item in grades.items():
            for name in ("lz", "now", "question", "answer", "why_passes" if grade == "pass" else "why_fails"):
                assert item[name].strip(), f"{criterion} {grade}: {name}"
            assert datetime.fromisoformat(item["now"]).tzinfo is not None, f"{criterion} {grade}: now has no zone"


def test_each_fail_example_names_its_single_reason_and_a_pass_does_not_carry_one():
    for criterion, grades in raw_examples().items():
        assert "why_fails" in grades["fail"] and "why_passes" not in grades["fail"], criterion
        assert "why_passes" in grades["pass"] and "why_fails" not in grades["pass"], criterion
        reason = grades["fail"]["why_fails"]
        assert "\n" not in reason.strip() and reason.count(".") <= 1, f"{criterion}: a fail has one reason, in one sentence"


@pytest.mark.parametrize(
    "broken",
    [
        {"answers_the_question": {"pass": {}, "fail": {}}},  # missing criteria
        {name: {"pass": {}} for name in CRITERIA},  # no fail
    ],
)
def test_a_file_of_the_wrong_shape_is_refused(broken, tmp_path):
    path = tmp_path / "judge_examples.yaml"
    path.write_text(yaml.safe_dump(broken), encoding="utf-8")
    with pytest.raises(ValueError, match="judge_examples.yaml"):
        load_examples(path)


def test_an_example_missing_a_field_is_refused_naming_it(tmp_path):
    data = raw_examples()
    del data["prioritised"]["fail"]["why_fails"]
    path = tmp_path / "judge_examples.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    with pytest.raises(ValueError, match="prioritised fail example needs a non-empty `why_fails`"):
        load_examples(path)


# --- F5: invented --------------------------------------------------------------------------------------------------------------


def dataset_strings() -> dict[str, set[str]]:
    """What must not appear in the judge examples: the dataset's LZ names, case folders, questions and NOTAM ids (the dataset is read as text; nothing is judged)."""
    records = yaml.safe_load((DATA / "examples.yaml").read_text(encoding="utf-8"))
    cases = {r["fixtures"]["case"] for r in records}
    names = {name for case in cases if (name := lz_name_from_case(case))}
    questions = {r["inputs"]["question"] for r in records}
    sheet_text = (DATA / "verdict_sheet.yaml").read_text(encoding="utf-8")
    examples_text = (DATA / "examples.yaml").read_text(encoding="utf-8")
    notam_ids = set(NOTAM_ID.findall(sheet_text)) | set(NOTAM_ID.findall(examples_text))
    return {"names": names, "cases": cases | {p.name for p in FIXTURES.iterdir() if p.is_dir()}, "questions": questions, "notam_ids": notam_ids}


def test_the_dataset_strings_are_found_so_the_checks_below_can_fail():
    found = dataset_strings()
    assert len(found["names"]) >= 5 and len(found["cases"]) >= 10 and len(found["questions"]) >= 40
    assert len(found["notam_ids"]) >= 10


def test_no_lz_name_from_the_dataset_appears_in_the_examples():
    text = EXAMPLES_PATH.read_text(encoding="utf-8").lower()
    assert [name for name in dataset_strings()["names"] if name.lower() in text] == []


def test_no_case_folder_name_from_the_dataset_appears_in_the_examples():
    text = EXAMPLES_PATH.read_text(encoding="utf-8")
    assert [case for case in dataset_strings()["cases"] if case in text] == []


def test_no_question_from_the_dataset_appears_in_the_examples():
    text = " ".join(EXAMPLES_PATH.read_text(encoding="utf-8").split())
    assert [q for q in dataset_strings()["questions"] if " ".join(q.split()) in text] == []


def test_no_notam_id_from_the_dataset_or_the_sheet_appears_in_the_examples():
    ids = set(NOTAM_ID.findall(EXAMPLES_PATH.read_text(encoding="utf-8")))
    assert ids, "the examples name no NOTAM at all, so this check would pass on anything"
    assert ids & dataset_strings()["notam_ids"] == set()


def test_the_example_lz_names_are_not_a_dataset_location_by_any_word():
    """A shared word is not proof of copying, but a whole dataset name inside an example's `lz` is."""
    dataset = {name.lower() for name in dataset_strings()["names"]}
    assert {item["lz"].lower() for grades in load_examples().values() for item in grades.values()} & dataset == set()


# --- F6: fiction phones, no coordinates ----------------------------------------------------------------------------------------


def test_every_ten_or_eleven_digit_run_in_the_examples_is_in_a_fiction_range():
    runs = digit_runs(EXAMPLES_PATH.read_text(encoding="utf-8"))
    assert runs, "the examples hold no phone number, so this check would pass on anything"
    assert [run for run in runs if not run.startswith(FICTION_PREFIXES)] == []


def test_every_ten_or_eleven_digit_run_in_the_judge_test_fixtures_is_in_a_fiction_range():
    here = Path(__file__).parent
    for name in ("judge_helpers.py", "test_judge_criteria.py", "test_judge_message.py", "test_judge_examples.py", "test_judge_config.py"):
        assert [run for run in digit_runs((here / name).read_text(encoding="utf-8")) if not run.startswith(FICTION_PREFIXES)] == [], name
    integration = (REPO / "tests" / "integration" / "test_judge_evaluator.py").read_text(encoding="utf-8")
    assert [run for run in digit_runs(integration) if not run.startswith(FICTION_PREFIXES)] == []


def test_no_coordinates_appear_in_the_examples():
    assert re.findall(r"-?\d{1,3}\.\d{3,}", EXAMPLES_PATH.read_text(encoding="utf-8")) == []
    assert not re.search(r"\blat(itude)?\b|\blon(gitude)?\b", EXAMPLES_PATH.read_text(encoding="utf-8"), re.IGNORECASE)


# --- the system prompt -----------------------------------------------------------------------------------------------------------


SPEC_6_DEFINITIONS = {
    "answers_the_question": "It answers what was asked, not a neighbouring question.",
    "prioritised": "Hard limits and NO_FLY / CONTACT_FIRST come before cautions and awareness.",
    "actionable": "The pilot knows what to do (phone first, stay below X ft, avoid a sector, wait until a time).",
    "good_decline": "Decline path only: polite, says what it can do, points to the right human where safety-relevant.",
}


def test_the_definitions_are_spec_section_6s_text_not_a_paraphrase():
    """Pinned here as written in the Spec's 'Pass when' table: comparing DEFINITIONS with themselves could not catch a changed condition."""
    assert dict(DEFINITIONS) == SPEC_6_DEFINITIONS


def test_every_definition_is_in_the_system_prompt_word_for_word():
    prompt = system_prompt()
    for criterion in CRITERIA:
        assert SPEC_6_DEFINITIONS[criterion] in prompt, criterion
        assert f"- {criterion}: {SPEC_6_DEFINITIONS[criterion]}" in prompt


def test_each_definition_is_one_line():
    assert all("\n" not in text and text.endswith(".") for text in DEFINITIONS.values())


def masked(text: str) -> str:
    return redact_phones(text)[0]


def test_every_example_is_in_the_system_prompt_as_its_real_message_with_its_grade():
    prompt = system_prompt()
    for criterion, grades in load_examples().items():
        for grade, item in grades.items():
            message = render_message(datetime.fromisoformat(item["now"]), item["lz"], masked(item["question"]), masked(item["answer"].strip()))
            assert message in prompt, f"{criterion} {grade}"
            assert example_reason(criterion, grade, load_examples()) in prompt
            assert f"Grade for {criterion}: " in prompt


def test_the_prompt_shows_a_phone_number_as_a_placeholder_never_as_a_number():
    """The rules say a number appears as <PHONE_n>; an example that showed one raw would contradict them."""
    prompt = system_prompt()
    raw = digit_runs(EXAMPLES_PATH.read_text(encoding="utf-8"))
    assert raw, "the examples hold no phone number, so this check would pass on anything"
    assert [run for run in raw if run in re.sub(r"\D", "", prompt)] == [], "a raw number's digits are in the prompt"
    assert "960123" not in prompt and "01632" not in prompt
    assert "the site contact is on <PHONE_1>." in prompt


def test_the_prompt_holds_eight_examples_and_no_other_example_text():
    assert len(re.findall(r"^Example \d+: ", system_prompt(), re.MULTILINE)) == 8


def test_an_example_in_the_prompt_is_the_message_a_real_run_of_it_would_get():
    """A real message for the example's text (phone and all) is the one the prompt shows: the two are built with the same masking, so they cannot drift apart."""
    prompt = system_prompt()
    for criterion, grades in load_examples().items():
        for grade, item in grades.items():
            real = judge_message(
                {"inputs": {"lat": 0.0, "lon": 0.0, "now": item["now"], "question": item["question"]}, "reference": {}, "fixtures": {}},
                item["answer"].strip(),
                lz_name=item["lz"],
            )
            assert real in prompt, f"{criterion} {grade}"


def test_the_prompt_tells_the_judge_the_example_specific_line_is_not_a_fifth_criterion():
    assert "not a fifth criterion" in system_prompt()


def test_the_prompt_is_built_without_reading_a_dev_example():
    """The prompt is the same for every example: nothing of an example under test is in it."""
    assert dev_example("dev_01_blue_mill")["inputs"]["question"] not in system_prompt()
