"""GP8: `cli rules` prints one Markdown table (the README links to rules.yaml rather than copying it)."""

import re

import pytest

from flysafelz import cli
from flysafelz.rules import RULES

HEADER = ["id", "value", "unit", "comparison", "source", "confidence", "note"]


def _cells(line: str) -> list[str]:
    assert line.startswith("| ") and line.endswith(" |")
    return re.split(r"(?<!\\) \| ", line[2:-2])


def test_rules_prints_header_and_one_row_per_rule(capsys):
    assert cli.main(["rules"]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert _cells(lines[0]) == HEADER
    assert set(_cells(lines[1])) == {"---"}
    rows = [_cells(line) for line in lines[2:]]
    assert len(rows) == len(RULES.all())
    for cells, rule in zip(rows, RULES.all(), strict=True):
        assert len(cells) == len(HEADER)
        assert cells[0] == rule.id
        if isinstance(rule.value, list):
            assert cells[1] == ", ".join(rule.value)
        else:
            assert float(cells[1]) == rule.value
        assert cells[2] == rule.unit
        assert cells[3] == rule.comparison.value
        assert cells[4] == rule.source
        assert cells[5] == rule.confidence.value
        assert cells[6] == rule.note
        assert all(cell.strip() for cell in cells)


def test_the_word_go_never_appears_alone_in_the_output(capsys):
    cli.main(["rules"])
    printed = capsys.readouterr().out
    # The one place "GO" stands alone is the precipitation note's "0.0 mm/h is GO" (decided by Dan, G14): a
    # forecast reading, not a verdict. Every other use of the word would be a verdict name.
    assert printed.count("0.0 mm/h is GO;") == 1
    assert not re.search(r"(?<!NO-)(?<!NO_)\bGO\b", printed.replace("0.0 mm/h is GO;", ""))


def test_unknown_command_and_missing_command_are_usage_errors():
    with pytest.raises(SystemExit) as unknown:
        cli.main(["assess"])
    assert unknown.value.code == 2
    with pytest.raises(SystemExit) as missing:
        cli.main([])
    assert missing.value.code == 2
