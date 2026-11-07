"""Load rules.yaml into a frozen model. Every threshold the code uses lives there.

Importing this module loads the shipped file (RULES). A missing, unknown,
duplicated or malformed rule raises RulesError, and the message names the rule.
Nothing lets a user point the program at another file.
"""

import operator
from enum import StrEnum
from importlib.resources import files
from importlib.resources.abc import Traversable
from pathlib import Path
from typing import Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, StrictFloat, StrictInt, StrictStr, ValidationError, model_validator


class RulesError(ValueError):
    """rules.yaml is missing a rule, has an unknown or duplicate one, or a rule is malformed."""


class Comparison(StrEnum):
    """How a value is tested against a rule.

    The four ordering operators are limits. IN tests membership of a list
    value. EQ marks a plain setting (a buffer, a radius, a horizon) that is
    read, not breached.
    """

    GE = ">="
    GT = ">"
    LT = "<"
    LE = "<="
    IN = "in"
    EQ = "="


class Confidence(StrEnum):
    SOURCED = "sourced"
    BEST_PRACTICE = "best practice"
    EDUCATED_GUESS = "educated guess"


_OPERATORS = {
    Comparison.GE: operator.ge,
    Comparison.GT: operator.gt,
    Comparison.LT: operator.lt,
    Comparison.LE: operator.le,
    Comparison.EQ: operator.eq,
}


class Rule(BaseModel):
    """One rules.yaml entry."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(min_length=1)
    value: StrictInt | StrictFloat | list[StrictStr]
    unit: str = Field(min_length=1)
    comparison: Comparison
    source: str = Field(min_length=1)
    confidence: Confidence
    note: str = Field(min_length=1)

    @model_validator(mode="after")
    def _value_matches_comparison(self) -> Self:
        is_list = isinstance(self.value, list)
        if self.comparison is Comparison.IN and not is_list:
            raise ValueError("comparison 'in' needs a list value")
        if self.comparison is not Comparison.IN and is_list:
            raise ValueError("a list value needs comparison 'in'")
        return self

    def breached(self, x: float | str) -> bool:
        """Apply this rule's own comparison to x: `x <comparison> value`.

        A value exactly at the limit follows the operator, so 16.0 breaches
        ">= 16" and 6.0 does not breach "> 6". For "in" it is list membership.
        """
        if isinstance(self.value, list):
            return x in self.value
        return _OPERATORS[self.comparison](x, self.value)


class Rules(BaseModel):
    """One named field per rule id. A missing or unknown id fails."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    wind_10m_caution: Rule
    wind_10m_no_go: Rule
    gust_spread_caution: Rule
    gust_spread_no_go: Rule
    wind_aloft_caution: Rule
    gradient_caution: Rule
    gradient_no_go: Rule
    precipitation_no_go: Rule
    rain_chance_caution: Rule
    visibility_caution: Rule
    visibility_no_go: Rule
    low_cloud_caution: Rule
    thermic_shortwave_caution: Rule
    thermic_cape_caution: Rule
    cape_no_go: Rule
    ceiling_left_no_go: Rule
    ceiling_buffer: Rule
    fl_ft_per_hpa: Rule
    fl_reference_hpa: Rule
    local_area: Rule
    airspace_box_nm: Rule
    band_low: Rule
    band_high: Rule
    verdict_horizon_day: Rule
    outlook_horizon_day: Rule
    pib_stale_hours: Rule
    uk_bounds_lat_min: Rule
    uk_bounds_lat_max: Rule
    uk_bounds_lon_min: Rule
    uk_bounds_lon_max: Rule
    civil_twilight_depression: Rule
    notam_max_lower_fl: Rule
    notam_ignore_subject_letters: Rule
    notam_keep_subjects: Rule
    notam_wide_ignore_subjects: Rule
    notam_ignore_traffic: Rule
    notam_ignore_purpose: Rule
    notam_ignore_security_subjects: Rule
    notam_restriction_subjects: Rule
    notam_awareness_subjects: Rule
    notam_area_wide_nm: Rule
    capture_trim_nm: Rule

    def all(self) -> list[Rule]:
        """Every rule, in file order."""
        return [getattr(self, name) for name in type(self).model_fields]


DEFAULT_PATH = files("flysafelz") / "rules.yaml"


def _refuse_duplicate_keys(node: yaml.Node | None, path: str) -> None:
    """Walk the YAML node tree and raise on a repeated key (PyYAML keeps the last silently)."""
    if isinstance(node, yaml.MappingNode):
        seen: set[object] = set()
        for key_node, value_node in node.value:
            key = key_node.value
            where = f"{path}.{key}" if path else str(key)
            if key in seen:
                line = key_node.start_mark.line + 1
                raise RulesError(f"rule {path or key}: duplicate key {key!r} (line {line})")
            seen.add(key)
            _refuse_duplicate_keys(value_node, where)
    elif isinstance(node, yaml.SequenceNode):
        for item in node.value:
            _refuse_duplicate_keys(item, path)


def _describe(exc: ValidationError) -> str:
    return "; ".join(f"{'.'.join(str(p) for p in e['loc']) or 'rule'}: {e['msg']}" for e in exc.errors())


def load_rules(path: Path | Traversable) -> Rules:
    """Parse a rules file into Rules, or raise RulesError naming the bad rule."""
    text = path.read_text(encoding="utf-8")
    try:
        node = yaml.compose(text, Loader=yaml.SafeLoader)
        data = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise RulesError(f"rules file is not valid YAML: {exc}") from exc
    _refuse_duplicate_keys(node, "")
    if not isinstance(data, dict):
        raise RulesError("rules file must be a mapping of rule id to rule")

    known = list(Rules.model_fields)
    unknown = [str(k) for k in data if k not in known]
    if unknown:
        raise RulesError(f"unknown rule id: {', '.join(unknown)}")
    missing = [k for k in known if k not in data]
    if missing:
        raise RulesError(f"missing rule: {', '.join(missing)}")

    built: dict[str, Rule] = {}
    for rule_id in known:
        entry = data[rule_id]
        if not isinstance(entry, dict) or "id" in entry:
            raise RulesError(f"rule {rule_id}: must be a mapping of fields, without an 'id' key")
        try:
            built[rule_id] = Rule.model_validate({**entry, "id": rule_id})
        except ValidationError as exc:
            raise RulesError(f"rule {rule_id}: {_describe(exc)}") from exc
    return Rules(**built)


RULES = load_rules(DEFAULT_PATH)
