"""E17: a value exactly at a limit follows the rule's own comparison."""

import pytest

from flysafelz.rules import RULES

# (rule, just below, exactly at, just above, breached at those three points)
CASES = [
    # ">=": breached at and above the limit.
    (RULES.wind_10m_caution, 11.9, 12.0, 12.1, (False, True, True)),
    # ">": breached only above the limit.
    (RULES.wind_10m_no_go, 15.9, 16.0, 16.1, (False, False, True)),
    (RULES.gust_spread_no_go, 5.9, 6.0, 6.1, (False, False, True)),
    # "<": breached only below the limit.
    (RULES.visibility_no_go, 1499, 1500, 1501, (True, False, False)),
    # "<=": breached at and below the limit.
    (RULES.visibility_caution, 4999, 5000, 5001, (True, True, False)),
]


@pytest.mark.parametrize(("rule", "below", "at", "above", "expected"), CASES, ids=[c[0].id for c in CASES])
def test_breached_follows_the_rules_own_comparison(rule, below, at, above, expected):
    assert (rule.breached(below), rule.breached(at), rule.breached(above)) == expected


def test_less_or_equal_is_breached_at_and_below(tmp_path):
    # visibility_no_go ships as "<"; build a "<=" variant of it through the real loader.
    from flysafelz.rules import DEFAULT_PATH, load_rules

    text = DEFAULT_PATH.read_text(encoding="utf-8")
    old = 'comparison: "<"\n  source: "Plan 2 G22; compare'
    assert old in text
    path = tmp_path / "rules.yaml"
    path.write_text(text.replace(old, 'comparison: "<="\n  source: "Plan 2 G22; compare'), encoding="utf-8")
    rule = load_rules(path).visibility_no_go
    assert (rule.breached(1499), rule.breached(1500), rule.breached(1501)) == (True, True, False)


def test_wind_exactly_16_and_gust_spread_exactly_6_are_not_no_go():
    assert RULES.wind_10m_no_go.breached(16.0) is False
    assert RULES.wind_10m_no_go.breached(16.1) is True
    assert RULES.gust_spread_no_go.breached(6.0) is False
