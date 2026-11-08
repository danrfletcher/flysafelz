"""The chunker (G17): the heading rule, the token band (merge and split), text normalisation and deterministic hashes, on the committed sample PDF."""

import hashlib
import re
from pathlib import Path

import pytest
from guidance_helpers import SOFT_HYPHEN, build_pdf, sample_pages

from flysafelz.guidance.chunker import (
    MAX_FURNITURE_WORDS,
    Chunk,
    NoHeadingsError,
    PdfUnreadable,
    chunk_pages,
    clean_line,
    count_tokens,
    drop_page_furniture,
    paragraphs,
    parse_heading,
    read_pdf_pages,
    section_slug,
)

ROOT = Path(__file__).resolve().parents[2]
SAMPLE = ROOT / "tests" / "data" / "guidance" / "sample_cap.pdf"
BAND = (20, 40)  # a band small enough for the sample's few lines


@pytest.mark.parametrize(
    ("line", "expected"),
    [
        ("1 Introduction", ("1", "Introduction")),
        ("4.2 Danger areas", ("4.2", "Danger areas")),
        ("4.2.1 Activation times", ("4.2.1", "Activation times")),
        ("4. Overview", ("4", "Overview")),
        ("10.12 Notices to air missions", ("10.12", "Notices to air missions")),
        ("Chapter 3", ("Chapter 3", "")),
        ("Chapter 3 Airspace classes", ("Chapter 3", "Airspace classes")),
        ("CHAPTER 3: AIRSPACE", ("Chapter 3", "AIRSPACE")),
        ("Chapter 3 - Airspace", ("Chapter 3", "Airspace")),
        ("Appendix A", ("Appendix A", "")),
        ("Appendix A Glossary", ("Appendix A", "Glossary")),
        ("APPENDIX b: Contacts", ("Appendix B", "Contacts")),
        # not headings
        ("Contents", None),
        ("4.2 danger areas in lower case", None),
        ("4.2 Danger areas .......... 17", None),
        ("4.2 Danger areas 17", None),
        ("Chapter 3 Airspace .... 9", None),
        ("1. The pilot checks the notice before every flight.", None),
        ("4.2 Pilots call the unit first;", None),
        ("29 October 2018", None),
        ("2024 Edition of this guidance", None),
        ("10.5 mph", None),
        ("Chapter 3 describes the classes of airspace.", None),
        ("Appendix a list of terms", None),
        ("Chapters 3 Airspace", None),
        ("Page 3 of 4", None),
        ("4.2.1.1.1.1.1 Too deep to be a heading", None),
        ("1 " + " ".join(["word"] * 16), None),
        ("1 " + " ".join(["Word"] * 15), ("1", " ".join(["Word"] * 15))),
        ("", None),
        # lines found in the CAA's CAP 1535 and CAP 2562 PDFs that are not headings
        ("07 Pilot >", None),  # a contents-page number and a navigation marker
        ("10 Flight Information Regions >", None),
        ("0 Dewpoint oC", None),  # a table cell
        ("2 NM ATZ shown with aerodrome symbol", None),  # a chart label: a number and a unit
        ("500 ft AMSL", None),
        ("20 KT", None),
        ("1 MIN", None),
        ("3 NM for every 1,000 ft of height you", None),
        ("1 US gal = 3.785 litres 1 litre = 0.264 US gal", None),
        ("27 CAT III A227 CAT IIIA2", None),
        ("1 EU 376/2014", None),  # a footnote: a bare reference
        ("1 Regulation (EU) No 376/2014", None),  # a footnote citing a regulation
        ("2 A227", None),
        ("2 ECCAIRS stands for European Co-ordination Centre for", None),  # a sentence cut at the end of the printed line
        ("4 Straight-in approaches are not", None),
        ("Chapter 3 and", None),
        # near misses that stay headings
        ("1.0 Introduction", ("1.0", "Introduction")),
        ("5 Minimum equipment", ("5", "Minimum equipment")),
        ("2 Notices to air missions (NOTAMs)", ("2", "Notices to air missions (NOTAMs)")),
        ("3 Airspace for pilots", ("3", "Airspace for pilots")),
        ("2 Hours of operation", ("2", "Hours of operation")),
        ("4 Feet and inches", ("4", "Feet and inches")),
        ("5 Knots and wind speed", ("5", "Knots and wind speed")),
        ("6 Regulation (EU) 2019/947", ("6", "Regulation (EU) 2019/947")),
        ("7 Form CA48 and ATC", ("7", "Form CA48 and ATC")),
    ],
)
def test_the_heading_rule(line, expected):
    assert parse_heading(line) == expected


def numbered(number: str, tokens: int, title: str = "Title", filler: str = "w") -> list[str]:
    """A heading line, then body lines that bring the section to exactly `tokens` tokens in all."""
    heading = f"{number} {title}"
    body = tokens - count_tokens(heading)
    assert body >= 0
    return [heading, *([" ".join([filler] * body)] if body else [])]


def doc(*sections: list[str]) -> list[list[str]]:
    """One page holding the sections one after the other; a blank line between sections so their bodies stay separate paragraphs."""
    return [[line for section in sections for line in section]]


def sections_of(chunks: list[Chunk]) -> list[str]:
    return [c.section for c in chunks]


def test_the_token_count_is_words_and_punctuation():
    assert count_tokens("") == 0
    assert count_tokens("Class D airspace, 4.2: yes") == 9  # Class D airspace , 4 . 2 : yes


@pytest.mark.parametrize(("size", "kept"), [(19, False), (20, True), (21, True)])
def test_the_lower_edge_of_the_band_just_under_on_and_just_over(size, kept):
    """A child under the minimum merges into its parent; on the minimum and over it, it stays a chunk of its own."""
    parent = numbered("1", 20, "Parent")
    child = numbered("1.1", size, "Child")
    chunks = chunk_pages(doc(parent, child), "d", "1", *BAND)
    assert sections_of(chunks) == (["1", "1.1"] if kept else ["1"])
    if not kept:
        assert chunks[0].text.endswith(" ".join(["w"] * (size - count_tokens("1.1 Child"))))
        assert "1.1 Child" in chunks[0].text


@pytest.mark.parametrize(("size", "parts"), [(39, 1), (40, 1), (41, 2)])
def test_the_upper_edge_of_the_band_just_under_on_and_just_over(size, parts):
    """Two paragraphs: at 41 tokens the second no longer fits the first part and the section splits; at 40 it fits."""
    first_paragraph = ["1 Title", "A " + " ".join(["w"] * 18) + "."]  # 1 + 1 heading tokens... counted below
    first_tokens = count_tokens("1 Title\n\n" + first_paragraph[1])
    second_tokens = size - first_tokens
    second = "B " + " ".join(["w"] * (second_tokens - 1 - 1)) + "."  # B, the w's and the full stop
    chunks = chunk_pages([[*first_paragraph, second]], "d", "1", *BAND)
    assert count_tokens(chunks[0].text if parts == 1 else "1 Title\n\n" + first_paragraph[1] + "\n\n" + second) == size
    assert len(chunks) == parts
    assert sections_of(chunks) == (["1"] if parts == 1 else ["1 (part 1)", "1 (part 2)"])
    assert all(c.title == "Title" for c in chunks)
    if parts == 2:
        assert chunks[1].text == second


def test_an_over_band_section_splits_only_on_paragraph_boundaries_and_keeps_every_word():
    lines = ["4.2 Long section"] + [f"Paragraph {i} " + " ".join(["w"] * 14) + "." for i in range(7)]
    chunks = chunk_pages([lines], "d", "1", *BAND)
    assert sections_of(chunks) == [f"4.2 (part {n})" for n in range(1, len(chunks) + 1)] and len(chunks) > 1
    assert all(count_tokens(c.text) <= BAND[1] for c in chunks)
    assert "\n\n".join(c.text for c in chunks).replace("\n\n", "\n").split("\n") == lines


def test_a_single_paragraph_larger_than_the_band_stays_whole():
    chunks = chunk_pages([["1 Title", "Start " + " ".join(["w"] * 80) + "."]], "d", "1", *BAND)
    assert sections_of(chunks) == ["1"] and count_tokens(chunks[0].text) > BAND[1]


def test_deeper_small_sections_merge_first_into_the_nearest_present_parent():
    """4.1.1 and 4.1.2 are small and merge into 4.1; 4.1 stays small, so it merges into 4; 4 is then the only chunk but 5 is untouched."""
    chunks = chunk_pages(doc(numbered("4", 10), numbered("4.1", 4, "T"), numbered("4.1.1", 6), numbered("4.1.2", 6), numbered("5", 25)), "d", "1", *BAND)
    assert sections_of(chunks) == ["4", "5"]
    assert [line.split()[0] for line in chunks[0].text.splitlines() if line and line[0].isdigit()] == ["4", "4.1", "4.1.1", "4.1.2"]


def test_a_section_whose_parent_is_absent_merges_into_the_nearest_ancestor_that_is_present():
    chunks = chunk_pages(doc(numbered("4", 30), numbered("4.2.1", 8)), "d", "1", *BAND)
    assert sections_of(chunks) == ["4"] and "4.2.1 Title" in chunks[0].text


def test_a_small_section_merges_into_the_nearest_earlier_section_when_its_parent_number_repeats():
    """Numbering restarts, or a body line such as `1 Pilots must check NOTAMs` reads as a heading: 1.1 belongs to the `1` before it, not the last `1`."""
    pages = doc(numbered("1", 30, "Introduction"), numbered("1.1", 8, "Scope"), numbered("2", 30, "Airspace"), numbered("1", 30, "Pilots must check NOTAMs before flight", "W"))
    chunks = chunk_pages(pages, "d", "1", *BAND)
    assert sections_of(chunks) == ["1", "2", "1"]
    assert "1.1 Scope" in chunks[0].text and "1.1 Scope" not in chunks[2].text
    assert chunks[2].title == "Pilots must check NOTAMs before flight"


def test_a_small_section_with_a_repeated_parent_number_merges_into_the_one_just_before_it():
    pages = doc(numbered("1", 30, "First"), numbered("2", 30, "Second"), numbered("1", 30, "Again"), numbered("1.1", 8, "Scope"))
    chunks = chunk_pages(pages, "d", "1", *BAND)
    assert sections_of(chunks) == ["1", "2", "1"]
    assert "1.1 Scope" in chunks[2].text and "1.1 Scope" not in chunks[0].text


def test_a_small_section_with_no_parent_stays_a_chunk_of_its_own():
    chunks = chunk_pages(doc(numbered("1", 5), numbered("2", 5), numbered("Appendix A", 5)), "d", "1", *BAND)
    assert sections_of(chunks) == ["1", "2", "Appendix A"]


def test_a_merge_that_pushes_the_parent_over_the_band_splits_the_parent_afterwards():
    parent = ["1 Parent", "P " + " ".join(["w"] * 30) + "."]
    child = ["1.1 Child", "C " + " ".join(["w"] * 8) + "."]
    chunks = chunk_pages([parent + child], "d", "1", *BAND)
    assert sections_of(chunks) == ["1 (part 1)", "1 (part 2)"]
    assert chunks[1].text.startswith("1.1 Child")


def test_chunks_carry_their_document_edition_title_and_the_hash_of_their_text():
    chunks = chunk_pages(doc(numbered("2", 25, "Airspace")), "cap1535", "Edition 3", *BAND)
    (chunk,) = chunks
    assert (chunk.doc, chunk.edition, chunk.section, chunk.title) == ("cap1535", "Edition 3", "2", "Airspace")
    assert chunk.content_hash == hashlib.sha256(chunk.text.encode("utf-8")).hexdigest()


def test_text_before_the_first_heading_is_dropped():
    chunks = chunk_pages([["Cover page words", "Contents", *numbered("1", 25)]], "d", "1", *BAND)
    assert sections_of(chunks) == ["1"] and "Cover" not in chunks[0].text


@pytest.mark.parametrize("pages", [[[]], [["Just a sentence.", "And another one."]], [["Contents", "1 Intro .......... 3"]], []])
def test_a_document_with_no_headings_is_an_error(pages):
    with pytest.raises(NoHeadingsError, match="no section headings found"):
        chunk_pages(pages, "d", "1", *BAND)


def test_a_numbered_sentence_that_wraps_onto_a_lower_case_line_is_not_a_heading():
    pages = [["1 Introduction", "Text.", "2 The standard rule under ICAO is for a flight to not fly", "above the cloud base", "3 Airspace", "More text."]]
    assert sections_of(chunk_pages(pages, "d", "1", 1, 1000)) == ["1", "3"]
    assert "2 The standard rule" in chunk_pages(pages, "d", "1", 1, 1000)[0].text  # it stays in the text of the section it is in


def test_a_short_heading_followed_by_a_lower_case_line_is_still_a_heading():
    assert sections_of(chunk_pages([["1 Scope", "this guide covers airspace."]], "d", "1", 1, 1000)) == ["1"]


def test_a_numbered_line_that_ends_several_pages_is_a_footer_not_a_heading():
    """CAP 2562 prints `6 Paramotor Code` as the last line of every left-hand page, with the page number in it."""
    pages = [["1 Intro", "Some text here.", "CAP 2562", "6 Paramotor Code"], ["More text of the same section.", "A last line of text."], ["Page three text.", "CAP 2562", "8 Paramotor Code"]]
    (chunk,) = chunk_pages(pages, "d", "1", 1, 1000)
    assert chunk.section == "1" and "More text" in chunk.text and "Page three" in chunk.text


def test_a_numbered_line_under_dropped_footer_lines_is_a_footer_even_when_it_appears_once():
    """CAP 2562 page 6 ends `CAP 2562` / `CAA / September 2024` / `6 Paramotor Code`; no other page has the last line in that form."""
    furniture = ["CAA / September 2024", "CAP 2562"]
    pages = [
        ["1 Intro", "Text of the first page.", *furniture],
        ["Text of the second page.", "Nothing more here.", *furniture],
        ["Text of the third page.", "Still more words.", *furniture],
        ["Text of the fourth page.", "And the last words.", *furniture, "6 Paramotor Code"],
    ]
    (chunk,) = chunk_pages(pages, "d", "1", 1, 1000)
    assert chunk.section == "1" and "fourth page" in chunk.text


def test_a_heading_left_alone_at_the_foot_of_a_page_is_still_a_heading():
    """pypdf often leaves a heading as the last line of a page, with its body on the next page."""
    pages = [["1 Introduction", "Text here.", "2 Danger areas"], ["Danger areas are shown on charts.", "More."]]
    chunks = chunk_pages(pages, "d", "1", 1, 1000)
    assert sections_of(chunks) == ["1", "2"] and "Danger areas are shown" in chunks[1].text


def test_a_heading_with_a_lettered_list_straight_under_it_is_still_a_heading():
    pages = [["1 Introduction", "Text here.", "2 Rules for flight in controlled airspace", "a) obtain a clearance;", "b) keep a listening watch.", "3 Next", "End."]]
    chunks = chunk_pages(pages, "d", "1", 1, 1000)
    assert sections_of(chunks) == ["1", "2", "3"] and "a) obtain a clearance;" in chunks[1].text and "a) obtain" not in chunks[0].text


def test_a_heading_at_the_top_of_a_page_is_a_heading():
    pages = [["1 Intro", "Some text here."], ["2 Airspace", "More text of the section."]]
    assert sections_of(chunk_pages(pages, "d", "1", 1, 1000)) == ["1", "2"]


def test_a_contents_list_and_chart_labels_alone_are_not_a_document_with_headings():
    """The lines of a real CAP 1535 contents page and chart pages: the run must stop rather than promote them as sections."""
    pages = [
        ["Contents", "07 Pilot >", "07 Aircraft >", "08 Pre-flight planning >", "13 UK VOLMET Frequencies >", "89 T ransponder use >"],
        ["Dewpoint", "0 Dewpoint oC", "10 KT", "30 KT", "2 NM ATZ shown with", "aerodrome symbol", "1 MIN", "1 MIN", "20 NM"],
    ]
    with pytest.raises(NoHeadingsError, match="no section headings found"):
        chunk_pages(pages, "cap1535", "1", *BAND)


# --- normalisation -------------------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "clean"),
    [
        ("of\ufb01cial \ufb02ight", "official flight"),  # fi and fl ligatures
        (f"oper{SOFT_HYPHEN}ator", "operator"),
        ("a\u00a0b\u2009c\td\u3000e", "a b c d e"),  # no-break, thin, tab, ideographic space
        ("  a   b  ", "a b"),
        ("zero\u200bwidth\ufeff", "zerowidth"),
        ("bell\x07ring", "bell ring"),
        ("\uff14\uff0e\uff12 wide", "4.2 wide"),  # full-width digits and stop fold to plain ones
        ("", ""),
        ("Air Traf/f_ic Zones", "Air Traffic Zones"),  # pypdf writes an unmapped ligature glyph as its name
        ("o/f_f_ice e/f_f_icient a/f_f_lict di/f_fer /f_lag", "office efficient afflict differ flag"),
        ("a/b_c /file 1/2 f_i", "a/b_c /file 1/2 f_i"),  # not a glyph name
    ],
)
def test_text_is_normalised(raw, clean):
    assert clean_line(raw) == clean


def test_a_heading_with_a_ligature_or_odd_space_is_still_a_heading():
    assert parse_heading(clean_line("4.2\u00a0Of\ufb01cial   notices")) == ("4.2", "Official notices")


def test_running_headers_footers_and_page_numbers_are_dropped_only_where_they_repeat():
    pages = [
        ["CAP 1535 Edition 3", "1 Intro", "text one", "Page 1 of 3"],
        ["CAP 1535 Edition 3", "text two", "Page 2 of 3"],
        ["CAP 1535 Edition 3", "text three", "Page 3 of 3"],
    ]
    assert drop_page_furniture(pages) == [["1 Intro", "text one"], ["text two"], ["text three"]]


def test_a_footer_printed_differently_on_left_and_right_pages_is_dropped():
    """`6 Guide` on the even pages and `Guide 7` on the odd ones: each is on half the pages, and on every page of its own kind."""
    pages = [["text a", "Guide A"], ["text b", "B Guide"], ["text c", "Guide C"], ["text d", "D Guide"]]
    pages = [[line, re.sub(r"^[A-D] | [A-D]$", lambda m: " 7" if m.group().startswith(" ") else "7 ", footer)] for line, footer in pages]
    assert drop_page_furniture(pages) == [["text a"], ["text b"], ["text c"], ["text d"]]


def test_a_heading_repeated_with_only_its_number_changed_on_the_right_hand_pages_is_not_furniture():
    """Three chapters each start on an odd page with `N.1 Introduction`: 3 of the 4 odd pages, 3 of 7 pages. They are headings."""
    pages = [
        ["1.1 Introduction", "Text of chapter one."],
        ["More text of chapter one."],
        ["2.1 Introduction", "Text of chapter two."],
        ["More text of chapter two."],
        ["3.1 Introduction", "Text of chapter three."],
        ["More text of chapter three."],
        ["Closing text of the last page."],
    ]
    assert drop_page_furniture(pages) == pages
    assert sections_of(chunk_pages(pages, "d", "1", 1, 1000)) == ["1.1", "2.1", "3.1"]


def test_a_footer_missing_from_the_cover_is_still_furniture_of_its_side():
    """Eleven pages with a footer on the right-hand (odd) pages except the cover: 5 of the 6 right-hand pages, under half of all pages."""
    words = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta", "iota", "kappa", "lambda", "mu"]
    bodies = [[f"{word} line from page {chr(96 + n)}{chr(96 + n)}" for word in words] for n in range(1, 12)]  # no digits: page 3 and page 5 share no key
    pages = [[*body, f"Guide {n}"] if n % 2 == 1 and n > 1 else list(body) for n, body in enumerate(bodies, start=1)]
    assert drop_page_furniture(pages) == bodies


def test_a_line_on_half_the_pages_in_no_pattern_is_kept():
    pages = [["text a", "Note"], ["text b", "other"], ["text c", "different"], ["text d", "Note"]]
    assert drop_page_furniture(pages) == pages


def test_a_line_in_the_middle_of_a_page_is_not_furniture_even_when_it_repeats():
    pages = [[*(f"{n}{page}" for n in "abcde"), "Note: keep clear", *(f"{n}{page}" for n in "fghij")] for page in "xy"]
    assert drop_page_furniture(pages) == pages


def test_a_long_line_is_body_text_however_often_it_repeats_at_the_edge_of_a_page():
    """A repeated line of exactly the limit in words is furniture; one word more is body text."""
    at_limit = " ".join(["word"] * MAX_FURNITURE_WORDS)
    over = f"{at_limit} more"
    assert drop_page_furniture([[at_limit, "a"], [at_limit, "b"]]) == [["a"], ["b"]]
    pages = [[over, "Short header", body] for body in ("alpha", "beta", "gamma")]
    assert drop_page_furniture(pages) == [[over, "alpha"], [over, "beta"], [over, "gamma"]]


def test_a_line_on_only_half_the_pages_is_kept_and_one_page_keeps_everything():
    pages = [["Header", "a"], ["b"]]
    assert drop_page_furniture(pages) == pages
    assert drop_page_furniture([["Header", "a"]]) == [["Header", "a"]]


@pytest.mark.parametrize(
    ("lines", "expected"),
    [
        (["one two", "three four."], ["one two three four."]),
        (["First sentence.", "Second starts here"], ["First sentence.", "Second starts here"]),
        (["First sentence.", "and goes on in lower case"], ["First sentence. and goes on in lower case"]),
        (["Intro:", "• first", "• second"], ["Intro:", "• first", "• second"]),
        (["Intro", "(a) first", "(b) second"], ["Intro", "(a) first", "(b) second"]),
        (["a b", "", "c d"], ["a b", "c d"]),
        ([], []),
    ],
)
def test_lines_are_reflowed_into_paragraphs(lines, expected):
    assert paragraphs(lines) == expected


def test_the_slug_of_a_section_is_a_safe_file_name_part():
    assert [section_slug(s) for s in ("4.2 (part 2)", "Appendix A", "Chapter 3", "4.2.1", "../x", "(!)")] == ["4.2-part-2", "appendix-a", "chapter-3", "4.2.1", "x", "section"]


# --- the sample PDF ------------------------------------------------------------------------------------------------------------------------


def test_the_committed_sample_pdf_is_what_the_helper_builds():
    assert SAMPLE.read_bytes() == build_pdf(sample_pages())


def test_the_sample_pdf_splits_by_its_own_numbered_headings():
    pages = read_pdf_pages(SAMPLE.read_bytes())
    assert len(pages) == 4
    chunks = chunk_pages(pages, "cap0", "1", 10, 40)
    assert sections_of(chunks) == ["1", "2", "2.1", "3", "3.1", "3.2", "4", "4.1", "Appendix A"]
    by_section = {c.section: c for c in chunks}
    assert by_section["3.2"].text.startswith("3.2 Contact")  # the right-hand column of the two-column page
    assert "operator's rules" in by_section["2.1"].text  # the soft hyphen is gone
    assert not any("CAP 0000" in c.text or "Page " in c.text for c in chunks)  # header and footer
    assert not any("......" in c.text for c in chunks)  # the contents list is not a set of sections


def test_the_sample_pdf_merges_its_small_sections_into_their_parents_in_the_real_band():
    chunks = chunk_pages(read_pdf_pages(SAMPLE.read_bytes()), "cap0", "1", 30, 60)
    assert sections_of(chunks) == ["1", "2", "2.1", "3", "4", "Appendix A"]  # 2.1 has 33 tokens, over the minimum, so it stays
    assert "3.1 Activation" in chunks[3].text and "3.2 Contact" in chunks[3].text and "4.1 Reading a NOTAM" in chunks[4].text


def test_the_same_bytes_give_the_same_chunks_hashes_and_text():
    first = chunk_pages(read_pdf_pages(SAMPLE.read_bytes()), "cap0", "1", 10, 40)
    second = chunk_pages(read_pdf_pages(SAMPLE.read_bytes()), "cap0", "1", 10, 40)
    assert first == second and [c.content_hash for c in first] == [c.content_hash for c in second]
    assert len({c.content_hash for c in first}) == len(first)


def test_a_header_with_a_different_page_number_each_page_does_not_change_the_chunks():
    plain = chunk_pages(read_pdf_pages(build_pdf(sample_pages())), "cap0", "1", 10, 40)
    renumbered = [[(x, y, text.replace("of 4", "of 9")) for x, y, text in page] for page in sample_pages()]
    assert [c.content_hash for c in chunk_pages(read_pdf_pages(build_pdf(renumbered)), "cap0", "1", 10, 40)] == [c.content_hash for c in plain]


@pytest.mark.parametrize("data", [b"", b"not a pdf at all", SAMPLE.read_bytes()[:300], b"%PDF-1.4\n1 0 obj\n<< >>\nendobj\n"])
def test_a_file_pypdf_cannot_open_is_unreadable(data):
    with pytest.raises(PdfUnreadable):
        read_pdf_pages(data)


def test_an_encrypted_pdf_is_unreadable():
    from pypdf import PdfReader, PdfWriter

    writer = PdfWriter(clone_from=PdfReader(SAMPLE))
    writer.encrypt("secret")
    import io

    buffer = io.BytesIO()
    writer.write(buffer)
    with pytest.raises(PdfUnreadable, match="encrypted"):
        read_pdf_pages(buffer.getvalue())
