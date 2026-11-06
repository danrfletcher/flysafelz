"""G23: the NOTAM sort lists and radius load from rules.yaml with source and confidence; a missing key stops the loader."""

from collections.abc import Callable
from pathlib import Path

import pytest
import yaml

from flysafelz.rules import DEFAULT_PATH, RULES, Rule, RulesError, load_rules

SOURCE = "ICAO Doc 8126 / UK AIP GEN 3.1 §3.6.3.4"
SORT_RULES = {
    "notam_ignore_subject_letters": ["L", "F", "M", "N", "I", "C", "S", "O"],
    "notam_keep_subjects": ["FA", "OB", "OL"],
    "notam_wide_ignore_subjects": ["AF"],
    "notam_ignore_traffic": ["I"],
    "notam_ignore_purpose": ["K"],
    "notam_ignore_security_subjects": ["RO"],
    "notam_restriction_subjects": ["RD", "RR", "RP", "RT"],
    "notam_awareness_subjects": ["GW"],
    "notam_area_wide_nm": 50,
}
PREFILTER_RULES = {"notam_max_lower_fl": 195}


def rule(rule_id: str) -> Rule:
    return getattr(RULES, rule_id)


@pytest.mark.parametrize(("rule_id", "value"), list(SORT_RULES.items()))
def test_the_sort_rules_load_with_their_values(rule_id, value):
    assert rule(rule_id).value == value
    assert rule(rule_id).id == rule_id


@pytest.mark.parametrize("rule_id", list(SORT_RULES))
def test_every_sort_rule_has_the_g23_source_and_a_note(rule_id):
    assert rule(rule_id).source == SOURCE
    assert rule(rule_id).note.strip()


def test_the_fifty_nm_radius_is_an_educated_guess():
    assert rule("notam_area_wide_nm").confidence == "educated guess"
    assert rule("notam_area_wide_nm").comparison == ">="
    assert rule("notam_area_wide_nm").unit == "NM"


@pytest.mark.parametrize(("rule_id", "value"), list(PREFILTER_RULES.items()))
def test_the_prefilter_rules_load_with_source_and_confidence(rule_id, value):
    assert rule(rule_id).value == value
    assert rule(rule_id).source.strip()
    assert rule(rule_id).confidence == "educated guess"


def test_a_r_and_w_are_never_in_the_ignore_letters():
    assert not {"A", "R", "W"} & set(rule("notam_ignore_subject_letters").value)


def test_the_rules_match_as_the_sort_uses_them():
    assert rule("notam_ignore_subject_letters").breached("L") is True
    assert rule("notam_ignore_subject_letters").breached("W") is False
    assert rule("notam_keep_subjects").breached("FA") is True
    assert rule("notam_keep_subjects").breached("FU") is False
    assert rule("notam_area_wide_nm").breached(50) is True
    assert rule("notam_area_wide_nm").breached(49.9) is False
    assert rule("notam_max_lower_fl").breached(195) is False
    assert rule("notam_max_lower_fl").breached(196) is True


def _write(tmp_path: Path, mutate: Callable[[dict], None]) -> Path:
    data = yaml.safe_load(DEFAULT_PATH.read_text(encoding="utf-8"))
    mutate(data)
    path = tmp_path / "rules.yaml"
    path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return path


@pytest.mark.parametrize("rule_id", [*SORT_RULES, *PREFILTER_RULES])
def test_removing_one_key_makes_the_loader_raise_naming_it(tmp_path, rule_id):
    with pytest.raises(RulesError, match=rule_id):
        load_rules(_write(tmp_path, lambda data: data.pop(rule_id)))


@pytest.mark.parametrize("field", ["source", "confidence"])
@pytest.mark.parametrize("rule_id", [*SORT_RULES, *PREFILTER_RULES])
def test_a_rule_without_a_source_or_confidence_raises(tmp_path, rule_id, field):
    with pytest.raises(RulesError, match=rule_id):
        load_rules(_write(tmp_path, lambda data: data[rule_id].pop(field)))
