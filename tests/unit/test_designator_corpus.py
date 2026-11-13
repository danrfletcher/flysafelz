"""No-drift check over every <ItemE> in the committed PIBs: the designator finder and the SUP finder change nothing else.

The old (one-letter suffix, optional spaces and prefix letters) and the current designator patterns are run over all item E text, and
the differences are pinned: designators lost are approach-chart 'D 527' style false hits, designators gained are the siblings in a
slash list such as 'EGD510A/B/C'. Reads only committed files.
"""

import re
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from flysafelz.designators import find_designators
from flysafelz.sup_refs import find_sup_references

ROOT = Path(__file__).resolve().parents[2]
PIB_FILES = sorted(
    [
        *(ROOT / "fixtures").glob("*/pib.xml"),
        *(ROOT / "tests" / "data" / "capture").glob("*.xml"),
        ROOT / "tests" / "fixtures" / "pib" / "pib_trimmed.xml",
        *(ROOT / "evals" / "reader" / "pibs").glob("*.xml"),
    ]
)

# designators.py before the slash-list and spacing fix: an optional single-letter suffix, optional spaces and prefix letters.
_OLD_DESIGNATOR = r"(?:E?G?\s?)?([DRP])\s?(\d{3})([A-Z]?)"
_OLD_FIND = re.compile(rf"\b{_OLD_DESIGNATOR}\b", re.IGNORECASE)
_CANONICAL_SUP = re.compile(r"AIP SUP [0-9]{1,3}/[0-9]{2,4}")
_SUP_DIGITS = re.compile(r"SUP\s*\d", re.IGNORECASE)
_MISSPELT_SUP = re.compile(r"SUPPELEMENT", re.IGNORECASE)  # the pinned miss: 'AIP SUPPELEMENT 022/2026' (test_sup_refs.py)


def _old_find_designators(text: str) -> list[str]:
    found = (f"EG{m.group(1)}{m.group(2)}{m.group(3)}".upper() for m in _OLD_FIND.finditer(text))
    return list(dict.fromkeys(found))


def _items_e(path: Path) -> list[str]:
    return [el.text or "" for el in ET.parse(path).iter() if el.tag.rsplit("}", 1)[-1] == "ItemE"]


@pytest.fixture(scope="module")
def corpus() -> dict[Path, list[str]]:
    return {path: _items_e(path) for path in PIB_FILES}


def test_the_corpus_is_the_committed_pibs_and_is_not_empty(corpus):
    names = {p.parent.name if p.name == "pib.xml" and p.parent.parent.name == "fixtures" else p.name for p in corpus}
    assert {"blue_mill_2026-10-31", "west_sleekburn_2026-10-31", "pib_trimmed.xml", "pib_unreadable_circle.xml"} <= names
    assert len([p for p in corpus if p.parent == ROOT / "evals" / "reader" / "pibs"]) == 4
    assert sum(len(items) for items in corpus.values()) > 1000
    assert all(items for items in corpus.values())


def test_the_new_pattern_differs_from_the_old_only_by_the_tightening(corpus):
    """Lost: bare 'D 527' (an approach-chart category letter and a height) and 'D 125'. Gained: the siblings in 'EGD510A/B/C'."""
    gained, lost = set(), set()
    for items in corpus.values():
        for text in items:
            old, new = set(_old_find_designators(text)), set(find_designators(text))
            gained |= new - old
            lost |= old - new
    assert gained == {"EGD510B", "EGD510C", "EGD395B"}
    assert lost == {"EGD527", "EGD575", "EGD650", "EGD670", "EGD125", "EGD410", "EGD380"}


def test_every_designator_the_new_pattern_drops_was_a_category_letter_and_a_height(corpus):
    """Each lost one sits in an instrument-approach item (a 'CAT D 125(94)' or 'A/B/C/D 670(632)'), never in a danger-area item."""
    for items in corpus.values():
        for text in items:
            lost = set(_old_find_designators(text)) - set(find_designators(text))
            if lost:
                assert re.search(r"OCA|OCH|APPROACH", text), text[:80]
                assert not re.search(r"DANGER AREA|TDA|AIP SUP", text), text[:80]


def test_the_slash_lists_in_the_corpus_are_read_in_full(corpus):
    found = {d for items in corpus.values() for text in items for d in find_designators(text)}
    assert {"EGD510A", "EGD510B", "EGD510C", "EGD395A", "EGD395B"} <= found


def test_the_old_and_new_patterns_both_find_designators_in_the_corpus(corpus):
    """Guards the check above against an empty comparison: the corpus really holds designators."""
    found = {d for items in corpus.values() for text in items for d in find_designators(text)}
    assert {"EGD496A", "EGD496B", "EGD323A"} <= found


def test_every_item_e_with_a_sup_number_gives_at_least_one_reference_in_the_canonical_shape(corpus):
    checked = 0
    for path, items in corpus.items():
        for text in items:
            refs = find_sup_references(text)
            assert all(_CANONICAL_SUP.fullmatch(ref) for ref in refs), (path.name, refs)
            if _SUP_DIGITS.search(text) and not _MISSPELT_SUP.search(text):
                assert refs, (path.name, text[:60])
                checked += 1
    assert checked > 20  # the corpus really holds SUP references


def test_the_misspelt_supplement_in_the_corpus_is_the_known_miss(corpus):
    misspelt = [text for items in corpus.values() for text in items if _MISSPELT_SUP.search(text)]
    assert misspelt, "the pinned miss should still be in the committed PIBs"
    assert all(find_sup_references(text) == [] for text in misspelt)


def test_the_multi_sup_items_keep_their_order_of_appearance(corpus):
    """Real items that name two or more supplements: the first one in the text is the first one returned."""
    multi = 0
    for items in corpus.values():
        for text in items:
            refs = find_sup_references(text)
            if len(refs) > 1:
                multi += 1
                positions = [re.search(rf"SUP(?:PLEMENT)?\s+{re.escape(ref.removeprefix('AIP SUP '))}", text, re.IGNORECASE) for ref in refs]
                assert all(positions), (refs, text[:60])
                starts = [m.start() for m in positions if m]
                assert starts == sorted(starts), (refs, text[:60])
    assert multi >= 10  # the plan counts 13 across the eval PIBs and fixtures
