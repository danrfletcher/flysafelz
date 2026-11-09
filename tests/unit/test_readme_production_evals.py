"""G12 / GP6 (7 PR-5.F1): the README's "Production evals" section is there, sits after "Assessing a landing zone from the terminal", says what GP6 lists, and the Quickstart block lists the two commands.

The section is read from the real README. Each check also has a negative fixture: the same text with the phrase taken out fails it.
"""

import re
from pathlib import Path

import pytest

README = Path(__file__).resolve().parents[2] / "README.md"
HEADING = "## Production evals"
AFTER = "### Assessing a landing zone from the terminal (deterministic core only)"  # a `###` under Quickstart in the current README

# GP6: what the section says. Each pattern is one claim, written as the words the README uses for it.
CLAIMS = {
    "production runs have no labels": r"no labels",
    "the label-needing metrics never run online": r"missed blocker.*never run online",
    "the offline evals carry the safety claim": r"offline evals carry the safety claim",
    "the first runs were seeded, not organic": r"seeded, not organic",
    "it would run from cron or a scheduled job": r"cron or a scheduled job",
    "an alert on online.guard_pass = 0": r"alert[^.]*`online\.guard_pass` = 0",
    "pilot thumbs up/down is a next step": r"thumbs up/down",
}


def sections(text: str) -> dict[str, str]:
    """{heading line: its text} for each `## ` section; fenced blocks are kept in their section, and a `## ` inside one is not a heading."""
    found: dict[str, list[str]] = {}
    current = ""
    fenced = False
    for line in text.splitlines():
        if line.startswith("```"):
            fenced = not fenced
        if line.startswith("## ") and not fenced:
            current = line.strip()
            found[current] = []
        elif current:
            found[current].append(line)
    return {heading: "\n".join(lines) for heading, lines in found.items()}


def missing_claims(section: str) -> list[str]:
    flat = " ".join(section.split())
    return [claim for claim, pattern in CLAIMS.items() if not re.search(pattern, flat, re.IGNORECASE)]


def readme() -> str:
    return README.read_text(encoding="utf-8")


def test_the_section_exists_once_and_comes_right_after_assessing_a_landing_zone():
    headings = [line for line in readme().splitlines() if re.match(r"#{2,3} ", line)]  # `##` and `###`, so the `###` Assessing heading is found
    assert headings.count(HEADING) == 1
    assert headings[headings.index(AFTER) + 1] == HEADING


def test_the_section_makes_every_claim_gp6_lists():
    assert missing_claims(sections(readme())[HEADING]) == []


@pytest.mark.parametrize("claim", sorted(CLAIMS))
def test_each_claim_check_fires_when_its_phrase_is_taken_out(claim):
    section = sections(readme())[HEADING]
    removed = re.sub(CLAIMS[claim], "", " ".join(section.split()), flags=re.IGNORECASE)
    assert claim in missing_claims(removed)


def test_the_section_names_the_two_commands_the_seed_and_the_job_and_the_summary_file():
    section = sections(readme())[HEADING]
    for text in ("make online-seed", "make online-eval", "evals/online/seed_questions.yaml", "evals/results/online-<date>.md", "online.evaluator_version", "`source: online-seed`"):
        assert text in section, text


def test_the_quickstart_block_lists_make_online_seed_and_make_online_eval():
    setup = sections(readme())["## Quickstart"]
    block = setup.split("```")[1]
    assert re.search(r"^make online-seed\s+#", block, re.MULTILINE) and re.search(r"^make online-eval\s+#", block, re.MULTILINE)
    assert block.index("make test") < block.index("make online-seed") < block.index("make online-eval")


def test_the_section_says_nothing_is_scheduled_and_installs_nothing():
    section = " ".join(sections(readme())[HEADING].split())
    assert "Nothing is scheduled here" in section
    assert not re.search(r"crontab|launchctl|\.plist|dagu", section, re.IGNORECASE)


def test_the_section_scan_is_not_fooled_by_a_heading_inside_a_code_block():
    text = "## One\n```\n## Production evals\n```\n## Two\nbody\n"
    assert list(sections(text)) == ["## One", "## Two"]
