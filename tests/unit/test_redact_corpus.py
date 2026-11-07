"""G1/E5 over the 4 committed archive PIBs: redaction leaves every schedule, frequency and reference alone."""

from pathlib import Path

import pytest

from flysafelz.pib import read_notams
from flysafelz.redact import redact_phones

ROOT = Path(__file__).resolve().parents[2]
PIBS = sorted((ROOT / "evals" / "reader" / "pibs").glob("*.xml"))


@pytest.fixture(scope="module")
def corpus():
    return {path.stem: read_notams(path.read_bytes())[1] for path in PIBS}


def test_the_four_pibs_are_committed():
    assert len(PIBS) == 4


@pytest.mark.parametrize("field", ["item_d", "item_f", "item_g"])
def test_redaction_changes_no_d_f_or_g_item_in_the_four_pibs(corpus, field):
    """None of them holds a phone number, so a change would be a date, a time or a schedule turned into a placeholder."""
    checked = 0
    for sha, notams in corpus.items():
        for parsed in notams:
            text = getattr(parsed, field) or ""
            assert redact_phones(text) == (text, {}), f"{sha[:8]} {parsed.id} {field}"
            checked += bool(text)
    assert checked > 0


def test_the_corpus_holds_the_day_lists_the_scan_is_for(corpus):
    """Guards the test above: item D with a list of two-digit days, which a loose UK pattern read as a number."""
    day_lists = [p.id for notams in corpus.values() for p in notams if p.item_d and "03 04 05" in p.item_d]
    assert day_lists, "no day list found: the corpus test above would prove nothing"
