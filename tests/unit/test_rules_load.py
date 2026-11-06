"""G21: rules.yaml loads into a frozen model, and every bad rule fails naming its id."""

from collections.abc import Callable
from pathlib import Path

import pytest
import yaml

from flysafelz import rules
from flysafelz.rules import DEFAULT_PATH, RULES, Comparison, Confidence, Rule, Rules, RulesError, load_rules

ID = "wind_10m_no_go"


def _write(tmp_path: Path, mutate: Callable[[dict], None]) -> Path:
    data = yaml.safe_load(DEFAULT_PATH.read_text(encoding="utf-8"))
    mutate(data)
    path = tmp_path / "rules.yaml"
    path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return path


def _duplicate_key(tmp_path: Path) -> Path:
    text = DEFAULT_PATH.read_text(encoding="utf-8")
    path = tmp_path / "rules.yaml"
    path.write_text(text + f"\n{ID}:\n  value: 1\n", encoding="utf-8")
    return path


def _duplicate_field(tmp_path: Path) -> Path:
    text = DEFAULT_PATH.read_text(encoding="utf-8")
    path = tmp_path / "rules.yaml"
    path.write_text(text.replace(f"{ID}:\n  value: 16\n", f"{ID}:\n  value: 16\n  value: 17\n"), encoding="utf-8")
    return path


def _delete(key: str) -> Callable[[dict], None]:
    return lambda d: d.pop(key)


def _set(key: str, field: str, value: object) -> Callable[[dict], None]:
    return lambda d: d[key].__setitem__(field, value)


def _drop_field(key: str, field: str) -> Callable[[dict], None]:
    return lambda d: d[key].pop(field)


BAD_RULES = [
    ("missing rule", _delete(ID), ID),
    ("missing source", _drop_field(ID, "source"), ID),
    ("empty source", _set(ID, "source", ""), ID),
    ("missing confidence", _drop_field(ID, "confidence"), ID),
    ("bad confidence", _set(ID, "confidence", "pretty sure"), ID),
    ("unknown comparison", _set(ID, "comparison", "~="), ID),
    ("non-numeric value", _set(ID, "value", "sixteen"), ID),
    ("boolean value", _set(ID, "value", True), ID),
    ("list value without 'in'", _set(ID, "value", ["a"]), ID),
    ("'in' without a list value", _set(ID, "comparison", "in"), ID),
    ("extra field on a rule", _set(ID, "colour", "red"), ID),
    ("unknown rule id", lambda d: d.__setitem__("wind_10m_extreme", dict(d[ID])), "wind_10m_extreme"),
    ("rule is not a mapping", lambda d: d.__setitem__(ID, 16), ID),
]


@pytest.mark.parametrize(("name", "mutate", "rule_id"), BAD_RULES, ids=[b[0] for b in BAD_RULES])
def test_bad_rule_fails_naming_the_rule(tmp_path, name, mutate, rule_id):
    path = _write(tmp_path, mutate)
    with pytest.raises(RulesError, match=rule_id):
        load_rules(path)


def test_duplicate_rule_key_fails_naming_the_rule(tmp_path):
    with pytest.raises(RulesError, match=ID):
        load_rules(_duplicate_key(tmp_path))


def test_duplicate_field_inside_a_rule_fails_naming_the_rule(tmp_path):
    with pytest.raises(RulesError, match=ID):
        load_rules(_duplicate_field(tmp_path))


def test_shipped_file_loads_into_rules_at_import():
    assert isinstance(RULES, Rules)
    assert rules.RULES is RULES
    assert RULES == load_rules(DEFAULT_PATH)


def test_one_field_per_rule_id_and_ids_match_the_file():
    ids = list(yaml.safe_load(DEFAULT_PATH.read_text(encoding="utf-8")))
    assert list(Rules.model_fields) == ids
    assert [r.id for r in RULES.all()] == ids
    assert Rules.model_config["extra"] == "forbid"


def test_every_rule_has_source_note_and_known_confidence():
    for rule in RULES.all():
        assert rule.source.strip()
        assert rule.note.strip()
        assert rule.confidence in set(Confidence)
        assert rule.comparison in set(Comparison)


def test_list_value_with_in_is_accepted():
    rule = Rule(
        id="subjects_ignored",
        value=["L", "F"],
        unit="list",
        comparison="in",
        source="ICAO Doc 8126",
        confidence="sourced",
        note="example list rule for PR-4",
    )
    assert rule.breached("L") is True
    assert rule.breached("A") is False


def test_non_yaml_file_fails(tmp_path):
    path = tmp_path / "rules.yaml"
    path.write_text("key: [unclosed", encoding="utf-8")
    with pytest.raises(RulesError, match="not valid YAML"):
        load_rules(path)
