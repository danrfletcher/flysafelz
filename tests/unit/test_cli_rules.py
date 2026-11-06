"""GP8: `cli rules` prints one Markdown table, and the README carries the same table."""

import re
from pathlib import Path

import pytest

from flysafelz import cli
from flysafelz.rules import RULES

README = Path(__file__).resolve().parents[2] / "README.md"
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


def test_readme_rules_table_is_byte_identical_to_the_output(capsys):
    cli.main(["rules"])
    printed = capsys.readouterr().out
    lines = README.read_text(encoding="utf-8").splitlines()
    start = lines.index("## Rules")
    table = []
    for line in lines[start + 1 :]:
        if line.startswith("|"):
            table.append(line)
        elif table:
            break
    assert "\n".join(table) + "\n" == printed


def test_the_word_go_never_appears_alone_in_the_output(capsys):
    cli.main(["rules"])
    assert not re.search(r"(?<!NO-)(?<!NO_)\bGO\b", capsys.readouterr().out)


def test_unknown_command_and_missing_command_are_usage_errors():
    with pytest.raises(SystemExit) as unknown:
        cli.main(["assess"])
    assert unknown.value.code == 2
    with pytest.raises(SystemExit) as missing:
        cli.main([])
    assert missing.value.code == 2
