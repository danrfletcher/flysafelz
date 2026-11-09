"""Rule 7: documents whose headings are set in larger type, not numbered (the CAA's CAP 1535 and CAP 2562)."""

import pytest

from flysafelz.guidance.chunker import NoHeadingsError, chunk_typeset, read_pdf_runs
from tests.guidance_helpers import build_pdf

BODY = "Pilots must keep clear of the area while it is active and check the notice before every flight in the region."


def page(title: str, sections: list[tuple[str, int]]) -> list[tuple[int, int, str, int]]:
    """A page title at 22 points, then each subhead at 14 points followed by `lines` lines of 11 point body text."""
    items = [(56, 760, title, 22)]
    y = 730
    for subhead, lines in sections:
        items.append((56, y, subhead, 14))
        y -= 18
        for _ in range(lines):
            items.append((56, y, BODY, 11))
            y -= 14
    return items


def chunks(pages):
    return chunk_typeset(read_pdf_runs(build_pdf(pages)), "doc", "v1", 300, 800)


DOCUMENT = [
    page("Airspace hazards", [("Danger areas", 8), ("T emporary restrictions", 8)]),
    page("Airspace hazards", [("Glider sites", 8)]),  # the page title runs on: not a new section
    page("Flying sites", [("Be considerate", 8), ("Landowner permission", 8)]),
    page("Flying sites", [("Weather", 8)]),
    page("Flying sites", [("Equipment", 8)]),
]


def test_page_titles_and_subheads_become_numbered_sections_in_reading_order():
    assert [(c.section, c.title) for c in chunks(DOCUMENT)] == [
        ("1", "Airspace hazards"),
        ("1.1", "Danger areas"),
        ("1.2", "Temporary restrictions"),
        ("1.3", "Glider sites"),
        ("2", "Flying sites"),
        ("2.1", "Be considerate"),
        ("2.2", "Landowner permission"),
        ("2.3", "Weather"),
        ("2.4", "Equipment"),
    ]


def test_a_section_keeps_its_heading_with_its_body():
    danger = next(c for c in chunks(DOCUMENT) if c.title == "Danger areas")
    assert danger.text.startswith("Danger areas\nPilots must keep clear") and danger.doc == "doc" and danger.edition == "v1"


def test_a_kerned_capital_is_joined_but_a_and_i_stay_words():
    pages = [page("Airspace hazards", [("A note on Y ou and I", 8)])] * 5
    assert [c.title for c in chunks(pages)][1] == "A note on You and I"


def test_one_size_of_type_throughout_has_no_headings():
    prose = [[(56, 700, BODY), (56, 680, BODY)]] * 5
    with pytest.raises(NoHeadingsError, match="no section headings found"):
        chunks(prose)


def test_a_size_used_on_fewer_than_five_pages_is_body_text():
    pages = [page("Airspace hazards", [("Danger areas", 8)]), page("Flying sites", [("Be considerate", 8)])]
    with pytest.raises(NoHeadingsError):
        chunks(pages)
