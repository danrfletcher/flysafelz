"""Command line entry point. Only the `rules` subcommand exists so far."""

import argparse
import sys

from flysafelz.rules import RULES, Rule

COLUMNS = ("id", "value", "unit", "comparison", "source", "confidence", "note")


def format_value(rule: Rule) -> str:
    if isinstance(rule.value, list):
        return ", ".join(rule.value)
    if isinstance(rule.value, int):
        return str(rule.value)
    return format(rule.value, "g")


def _cell(text: str) -> str:
    return text.replace("|", "\\|")


def _row(cells: list[str]) -> str:
    return "| " + " | ".join(_cell(c) for c in cells) + " |"


def rules_table() -> str:
    """The Markdown table of every rule, as it appears in the README."""
    lines = [_row(list(COLUMNS)), _row(["---"] * len(COLUMNS))]
    for rule in RULES.all():
        lines.append(
            _row(
                [
                    rule.id,
                    format_value(rule),
                    rule.unit,
                    rule.comparison.value,
                    rule.source,
                    rule.confidence.value,
                    rule.note,
                ]
            )
        )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="flysafelz")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("rules", help="print rules.yaml as a Markdown table")
    args = parser.parse_args(argv)
    if args.command == "rules":
        sys.stdout.write(rules_table())
    return 0


if __name__ == "__main__":
    sys.exit(main())
