"""Build small text PDFs for the guidance tests, with no PDF library but the standard one: the bytes are written out by hand.

    build_pdf(pages)    -> bytes    `pages` is a list of pages, each a list of (x, y, text) in points from the bottom left; one Helvetica line per item
    sample_pages()      -> list     the pages of tests/data/guidance/sample_cap.pdf: numbered headings, running header and footer, one two-column page
    document_pages(...) -> list     one page per (number, title, paragraph count) with a header and footer, for the integration tests that need many sizes

tests/data/guidance/sample_cap.pdf is `build_pdf(sample_pages())`; a test checks the committed file still equals it.
Text is WinAnsi (cp1252), so a soft hyphen (0xAD) can be written; a ligature cannot, and is tested on strings.
"""

PAGE_WIDTH = 595
PAGE_HEIGHT = 842
TOP = 790
LEFT = 56
LINE = 16
RIGHT_COLUMN = 310
HEADER = "CAP 0000 Sample Guidance"
SOFT_HYPHEN = "\u00ad"

Item = tuple[int, int, str]


def _literal(text: str) -> bytes:
    """A PDF string literal: every byte outside plain ASCII, and the three that need it, written as an octal escape."""
    out = bytearray(b"(")
    for byte in text.encode("cp1252"):
        if byte in b"()\\" or byte < 32 or byte > 126:
            out += b"\\%03o" % byte
        else:
            out.append(byte)
    return bytes(out + b")")


def _content(items: list[Item]) -> bytes:
    return b"\n".join(b"BT /F1 11 Tf %d %d Td %s Tj ET" % (x, y, _literal(text)) for x, y, text in items)


def build_pdf(pages: list[list[Item]]) -> bytes:
    """A PDF with one page per list of items. Deterministic: the same pages always give the same bytes."""
    objects: list[bytes] = [b"<< /Type /Catalog /Pages 2 0 R >>", b"", b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>"]
    kids = []
    for items in pages:
        page_number = len(objects) + 1
        kids.append(b"%d 0 R" % page_number)
        objects.append(b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 %d %d] /Resources << /Font << /F1 3 0 R >> >> /Contents %d 0 R >>" % (PAGE_WIDTH, PAGE_HEIGHT, page_number + 1))
        stream = _content(items)
        objects.append(b"<< /Length %d >>\nstream\n%s\nendstream" % (len(stream), stream))
    objects[1] = b"<< /Type /Pages /Kids [%s] /Count %d >>" % (b" ".join(kids), len(kids))
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for number, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += b"%d 0 obj\n%s\nendobj\n" % (number, body)
    xref = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objects) + 1)
    for offset in offsets:
        out += b"%010d 00000 n \n" % offset
    out += b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (len(objects) + 1, xref)
    return bytes(out)


def _page(number: int, lines: list[str], right_lines: list[str] | None = None) -> list[Item]:
    """Header at the top, footer at the bottom, the lines between them from the top down; `right_lines` make a second column, written after the first."""
    items: list[Item] = [(LEFT, TOP + LINE, HEADER)]
    items += [(LEFT, TOP - LINE * index, line) for index, line in enumerate(lines)]
    items += [(RIGHT_COLUMN, TOP - LINE * index, line) for index, line in enumerate(right_lines or [])]
    items.append((LEFT, LINE * 3, f"Page {number} of 4"))
    return items


def sample_pages() -> list[list[Item]]:
    """Four pages: contents and sections 1 to 2.1, section 3, a two-column page (3.1 to 4), and 4.1 with Appendix A. A running header and footer on every page."""
    contents = [
        "Contents",
        "1 Introduction .......... 2",
        "2 Airspace .......... 2",
        "3 Danger areas .......... 3",
        "4 NOTAMs .......... 3",
        "Appendix A Glossary .......... 4",
    ]
    first = [
        "1 Introduction",
        "This guidance explains how pilots find out what airspace is active.",
        "It is a sample used for testing and is not real guidance.",
        "2 Airspace",
        "Controlled airspace needs a clearance before a pilot enters it.",
        "A pilot stays clear of it unless a clearance is given.",
        "2.1 Class D",
        "Class D airspace is controlled, so the pilot calls the unit first.",
        f"Radio failure while inside it is covered by the oper{SOFT_HYPHEN}ator's rules.",
    ]
    second = [
        "3 Danger areas",
        "A danger area warns of activity that is dangerous to aircraft.",
    ]
    left = [
        "3.1 Activation",
        "A danger area is active when its notice says so.",
        "The pilot checks the notice before every flight.",
    ]
    right = [
        "3.2 Contact",
        "Where a notice gives a contact, the pilot calls it first.",
        "4 NOTAMs",
        "A NOTAM is a notice to air missions.",
    ]
    last = [
        "4.1 Reading a NOTAM",
        "Item E holds the text and item Q holds the area.",
        "Appendix A Glossary",
        "NOTAM: notice to air missions.",
    ]
    return [_page(1, contents + first), _page(2, second), _page(3, left, right), _page(4, last)]


def document_pages(sections: list[tuple[str, str, int]], words_per_paragraph: int, paragraph_marker: str = "", tag: str = "word", extra: dict[int, list[str]] | None = None) -> list[list[Item]]:
    """One page per (number, title, paragraph count): a heading and that many paragraphs of `words_per_paragraph` words, then the repeated header and footer.

    Each paragraph is a single line of plain words that ends with a full stop and starts with a capital, so the chunker reads it as one paragraph. The words
    start with `tag`, so two documents (or two versions of one) can be told apart by the words they hold; `paragraph_marker` is added to each paragraph (a
    changed document is a changed marker). `extra` maps a 1-based page number to more paragraph lines for that page, each starting with a capital and ending with a full stop.
    """
    pages = []
    for number, (heading_number, title, count) in enumerate(sections, start=1):
        lines = [f"{heading_number} {title}"]
        for index in range(count):
            words = " ".join(f"{tag}{number}x{index}x{n}" for n in range(words_per_paragraph))
            lines.append(f"Paragraph {words} {paragraph_marker}.")
        lines += (extra or {}).get(number, [])
        pages.append(_page(number, lines))
    return pages


DIMENSION = 1024  # wide enough that two unrelated texts seldom share a slot


class FakeEmbedder:
    """A deterministic stand-in for fastembed's TextEmbedding: each word is hashed to one of DIMENSION slots, so texts that share words get similar vectors.

    `calls` records ("embed" | "query_embed", number of texts) for each call, so a test can assert that no embedding call was made.
    """

    def __init__(self) -> None:
        self.calls: list[tuple[str, int]] = []

    @staticmethod
    def vector(text: str) -> list[float]:
        import hashlib
        import re

        values = [0.0] * DIMENSION
        for word in re.findall(r"[a-z0-9]+", text.casefold()):
            values[int.from_bytes(hashlib.sha256(word.encode()).digest()[:4], "big") % DIMENSION] += 1.0
        return values

    def embed(self, documents, **kwargs):
        texts = list(documents)
        self.calls.append(("embed", len(texts)))
        return [self.vector(text) for text in texts]

    def query_embed(self, query, **kwargs):
        texts = list(query)
        self.calls.append(("query_embed", len(texts)))
        return [self.vector(text) for text in texts]
