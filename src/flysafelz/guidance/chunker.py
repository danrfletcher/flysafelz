"""Turn a document's pages into guidance chunks: one chunk per numbered section, inside the token band (G17).

    read_pdf_pages(data)                               -> list[list[str]]  the text lines of each page (PdfUnreadable when pypdf cannot open the file)
    chunk_pages(pages, doc, edition, tokens_min, tokens_max) -> list[Chunk]   normalise, find the document's own headings, merge, split
    count_tokens(text)                                 -> int              the size the band is measured in
    section_slug(section)                              -> str              a section number as a file-name part

The rules, in order (bump CHUNKER_VERSION in ingest.py whenever any of them changes):
  1. Every line is normalised: compatibility forms (ligatures such as the fi one) are folded, a ligature that pypdf wrote as a glyph name
     (`/f_i`, `/f_f_l`) becomes its letters, soft hyphens and zero-width marks are dropped, any other whitespace becomes one space.
  2. A short line (at most MAX_FURNITURE_WORDS words) that sits among the first or last few lines of more than half the pages, or of at least
     four fifths of the left-hand pages or of the right-hand pages (running headers, footers, page numbers; digits ignored) is dropped.
  3. A heading is a line that starts with a section number as printed: `4`, `4.2`, `4.2.1`, `Chapter 3` or `Appendix A`, followed by a title.
     Text before the first heading is dropped. No heading anywhere is an error (NoHeadingsError). A line is not a heading when its number has a
     leading zero (`07`, a contents-page number), its title ends with a navigation marker (`Pilot >`), or its title starts with a unit
     abbreviation (`2 NM ATZ shown`, `10 KT`: a chart label or a table cell; `2 Hours of operation` is a heading), or it looks like a table cell or a footnote (`=` in it, a code glued to digits
     such as `A227` or a pair of numbers around a slash such as `376/2014` in a title of two words or fewer, or two such references, or a legal
     citation `No 376/2014`), or it is the first line of a numbered sentence that wrapped (a title of
     five words or more and the next line starts in lower case but is not a list item such as `a) ...`, or the title ends in a word such as `for`, `not` or `the`: `1 The standard rule is not to` / `fly ...`), or it is
     the last printed line of its page and either the same line (digits ignored) ends at least one other page too or the line above it was dropped
     as a header or footer (a footer such as `6 Paramotor Code` under `CAP 2562`; a heading left alone at the foot of a page, with its body on
     the next, is kept).
  4. A section's text is its heading line, then its body re-flowed into paragraphs; the heading line and the first paragraph are never separated.
  5. A section under the band merges into its parent: the nearest section before it in the document whose number is one of its ancestors
     (`4.2.1` into an earlier `4.2`, else an earlier `4`; a number that repeats later in the document is not its parent); with no parent it stays as it is.
     Deeper sections merge first, so a parent that grows past the minimum is left alone.
  6. A section over the band splits on paragraph boundaries into parts numbered `4.2 (part 1)`, `4.2 (part 2)`. A paragraph larger than
     the band on its own stays whole.
Token counts are an approximation (a word or a punctuation mark is one token): the band is a size guide, not a model limit.
"""

import hashlib
import io
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, field

from pypdf import PdfReader

PAGE_SIDES = 1 + 1  # left-hand and right-hand pages: every other page is on the same side
SIDE_SHARE_PERCENT = 80  # a line is furniture of one side only when it is at the edge of this share of that side's pages (the cover may lack it)
PAGE_EDGE_LINES = 5  # how many lines at the top and bottom of a page can be a running header or footer
MAX_FURNITURE_WORDS = 11  # a longer line is body text, however often it repeats
SENTENCE_WORDS = 5  # a title this long, with its sentence going on in lower case on the next line, is body text that starts with a number
MAX_HEADING_WORDS = 15  # a longer line is a sentence that happens to start with a number
TOKEN = re.compile(r"\w+|[^\w\s]")
DASHES = "–—-"  # the hyphen last, so inside a character class it is a hyphen and not a range
NUMBERED = re.compile(r"(?P<number>\d{1,2}(?:\.\d{1,2}){0,5})\.?\s+(?P<title>[A-Z].*)")
CHAPTER = re.compile(rf"(?P<word>chapter)\s+(?P<label>\d{{1,2}})\b[\s:.{DASHES}]*(?P<title>.*)", re.IGNORECASE)
APPENDIX = re.compile(rf"(?P<word>appendix)\s+(?P<label>[A-Za-z])\b[\s:.{DASHES}]*(?P<title>.*)", re.IGNORECASE)
MONTH_START = re.compile(r"(January|February|March|April|May|June|July|August|September|October|November|December)\b")
CONTENTS_LINE = re.compile(r"\.{2,}|\s\d+$")  # dot leaders, or a page number at the end: a table-of-contents entry, not a heading
LIST_START = re.compile(r"(?:[•●▪\-–*]\s|\(?[a-z0-9]{1,3}[.)]\s)")
SENTENCE_END = (".", "!", "?", ":")
NAVIGATION_END = (">", "›", "»")  # a contents-page link such as "Pilot >"
UNITS = frozenset(
    [
        "nm",
        "km",
        "m",
        "ft",
        "kt",
        "kts",
        "mph",
        "mb",
        "hpa",
        "min",
        "mins",
        "sec",
        "secs",
        "hr",
        "hrs",
        "kg",
        "lb",
        "lbs",
        "gal",
        "imp",
        "deg",
        "oc",
    ]
)  # a number followed by one of these abbreviations is a measurement ("2 NM", "500 ft"), not a section number; spelled-out words ("2 Hours of operation") can start a heading
UNFINISHED_END = frozenset(["a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "if", "in", "is", "of", "on", "or", "that", "the", "to", "which", "with", "not"])  # a title cannot end here
REFERENCE = re.compile(r"\b[A-Z]{1,4}\d+\b|\d/\d")  # a code glued to digits (A227, CA48) or a pair of numbers around a slash (376/2014)
CITATION = re.compile(r"\bNo\.?\s+\d+/\d+")  # "No 376/2014": a regulation cited in a footnote
BARE_REFERENCE_WORDS = 1 + 1  # a title this short around one reference is a footnote (`1 EU 376/2014`); a longer one is a heading (`7 Form CA48 and ATC`)
GLYPH_LIGATURES = re.compile(r"/(f_f_i|f_f_l|f_f|f_i|f_l)")  # pypdf writes an unmapped ligature glyph as its name, in the middle of the word: Traf/f_ic
INVISIBLE = dict.fromkeys(map(ord, "\u00ad\u200b\u200c\u200d\u2060\ufeff"))
PART_SEPARATOR = "\n\n"


class PdfUnreadable(ValueError):
    """pypdf could not open the file or read a page of it."""


class NoHeadingsError(ValueError):
    """The text holds no recognisable section heading, so it cannot be split."""


@dataclass(frozen=True)
class Chunk:
    """One section (or one part of a long section) with where it came from. `content_hash` is the SHA-256 of `text`."""

    doc: str
    edition: str
    section: str
    title: str
    text: str
    content_hash: str


@dataclass
class _Node:
    number: str
    title: str
    text: str
    tokens: int = field(init=False)

    def __post_init__(self) -> None:
        self.tokens = count_tokens(self.text)


def count_tokens(text: str) -> int:
    """Words and punctuation marks. The band in settings.yaml is applied to this count."""
    return len(TOKEN.findall(text))


def clean_line(line: str) -> str:
    """Rule 1 for one line: folded forms, no invisible marks, control characters and every run of whitespace as one space."""
    text = GLYPH_LIGATURES.sub(lambda found: found[1].replace("_", ""), unicodedata.normalize("NFKC", line)).translate(INVISIBLE)
    text = "".join(" " if unicodedata.category(char) == "Cc" else char for char in text)
    return " ".join(text.split())


def read_pdf_pages(data: bytes) -> list[list[str]]:
    """The cleaned text lines of each page. Raises PdfUnreadable when pypdf cannot open the file, it is encrypted, or a page will not read."""
    try:
        reader = PdfReader(io.BytesIO(data))
        if reader.is_encrypted:
            raise PdfUnreadable("the PDF is encrypted")
        return [[clean_line(line) for line in (page.extract_text() or "").splitlines()] for page in reader.pages]
    except PdfUnreadable:
        raise
    except Exception as error:  # pypdf raises its own classes and plain ValueError, KeyError, EOFError for a damaged file
        raise PdfUnreadable(f"pypdf could not read the PDF ({type(error).__name__})") from error


def _furniture_key(line: str) -> str:
    return re.sub(r"\d+", "#", line).casefold()


def _could_be_furniture(line: str) -> bool:
    return bool(line) and len(line.split()) <= MAX_FURNITURE_WORDS


def _edge_positions(count: int) -> set[int]:
    return set(range(min(PAGE_EDGE_LINES, count))) | set(range(max(count - PAGE_EDGE_LINES, 0), count))


def _furniture_flags(pages: list[list[str]]) -> list[list[bool]]:
    """For every line of every page: True when rule 2 drops it. A document of one page keeps every line.

    A line is furniture when it is at the edge of more than half of all the pages, or of at least four fifths of the odd pages, or of at least four
    fifths of the even pages (a left-hand page and a right-hand page often print their footers differently: `6 Paramotor Code` and `Paramotor Code 7`).
    A side needs that much because a heading that repeats with only its number changed (`1.1 Introduction`, `2.1 Introduction`) starts chapters on
    pages of one side, and it must not be taken for a footer.
    """
    if len(pages) <= 1:
        return [[False] * len(lines) for lines in pages]
    edge_keys = [{_furniture_key(lines[i]) for i in _edge_positions(len(lines)) if _could_be_furniture(lines[i])} for lines in pages]
    furniture: set[str] = set()
    counts = Counter(key for keys in edge_keys for key in keys)
    furniture |= {key for key, pages_with in counts.items() if pages_with + pages_with > len(edge_keys)}
    for side in (edge_keys[::PAGE_SIDES], edge_keys[1::PAGE_SIDES]):
        counts = Counter(key for keys in side for key in keys)
        furniture |= {key for key, pages_with in counts.items() if len(side) > 1 and pages_with * 100 >= len(side) * SIDE_SHARE_PERCENT}
    flags = []
    for lines in pages:
        edge = _edge_positions(len(lines))
        flags.append([i in edge and _could_be_furniture(line) and _furniture_key(line) in furniture for i, line in enumerate(lines)])
    return flags


def drop_page_furniture(pages: list[list[str]]) -> list[list[str]]:
    """Rule 2: the pages without their running headers, footers and page numbers."""
    return [[line for line, dropped in zip(lines, flags, strict=True) if not dropped] for lines, flags in zip(pages, _furniture_flags(pages), strict=True)]


def _table_cell_or_footnote(title: str) -> bool:
    """An `=`, or a legal citation (`Regulation (EU) No 376/2014`: a footnote), or several references (`CAT III A227 CAT IIIA2`), or a bare reference
    with no other title words (`EU 376/2014`)."""
    references = sum(1 for word in title.split() if REFERENCE.search(word))
    return "=" in title or CITATION.search(title) is not None or references > 1 or (references == 1 and len(title.split()) <= BARE_REFERENCE_WORDS)


def parse_heading(line: str) -> tuple[str, str] | None:
    """(section number as printed, title) when the line is a heading, else None. Rule 3."""
    chapter = CHAPTER.fullmatch(line) or APPENDIX.fullmatch(line)
    if chapter:
        number = f"{chapter['word'].capitalize()} {chapter['label'].upper()}"
        title = chapter["title"].strip()
        if title and not title[0].isupper() and not title[0].isdigit():
            return None  # "Chapter 3 describes ..." is a sentence
    else:
        numbered = NUMBERED.fullmatch(line)
        if not numbered:
            return None
        number, title = numbered["number"], numbered["title"].strip()
        if MONTH_START.match(title):
            return None  # a date at the start of a line: "29 October 2018"
        if number.startswith("0") or title.split()[0].casefold().strip(".,;:()") in UNITS or _table_cell_or_footnote(title):
            return None  # "07 Pilot", "0 Dewpoint oC": a page number or a table cell; "2 NM ATZ shown": a measurement; "1 litre = 0.22": a table
    if title and title.split()[-1].casefold() in UNFINISHED_END:
        return None  # "2 ECCAIRS stands for European Co-ordination Centre for": a sentence cut at the end of the printed line
    if CONTENTS_LINE.search(title) or title.endswith((".", ";", ",", *NAVIGATION_END)) or len(title.split()) > MAX_HEADING_WORDS:
        return None
    return number, title


def _wrapped_sentence(title: str, next_line: str) -> bool:
    """True when a title of SENTENCE_WORDS words or more is followed by a line in lower case: it is the first line of a sentence, not a heading.

    A list item (`a) obtain a clearance;`) under a heading also starts in lower case, but it is not the rest of the title's sentence.
    """
    return len(title.split()) >= SENTENCE_WORDS and next_line[:1].islower() and not LIST_START.match(next_line)


def _starts_new_paragraph(previous: str, line: str) -> bool:
    return bool(LIST_START.match(line)) or (previous.endswith(SENTENCE_END) and not line[0].islower())


def paragraphs(lines: list[str]) -> list[str]:
    """Re-flow printed lines into paragraphs: a blank line, a list marker, or a sentence end followed by a capital starts a new one."""
    result: list[str] = []
    current: list[str] = []
    for line in lines:
        if not line:
            if current:
                result.append(" ".join(current))
                current = []
            continue
        if current and _starts_new_paragraph(current[-1], line):
            result.append(" ".join(current))
            current = []
        current.append(line)
    if current:
        result.append(" ".join(current))
    return result


def _sections(lines: list[str], page_ends: frozenset[int] = frozenset()) -> list[_Node]:
    """Rules 3 and 4: the sections in document order, each with its heading line and re-flowed body.

    `page_ends` holds the index of the last printed line of each page that looks like a footer (see rule 3): a heading-like line there is not a heading.
    """
    found: list[tuple[str, str]] = []  # (number, title) of each heading
    bodies: list[list[str]] = []  # the lines under each heading
    for index, line in enumerate(lines):
        heading = parse_heading(line)
        if heading and index not in page_ends and not _wrapped_sentence(heading[1], lines[index + 1] if index + 1 < len(lines) else ""):
            found.append(heading)
            bodies.append([])
        elif bodies:
            bodies[-1].append(line)
    if not found:
        raise NoHeadingsError("no section headings found")
    nodes = []
    for (number, title), body in zip(found, bodies, strict=True):
        heading_line = f"{number} {title}".strip()
        first, *rest = paragraphs(body) or [""]
        nodes.append(_Node(number, title, PART_SEPARATOR.join([f"{heading_line}\n{first}".rstrip(), *rest])))  # the heading stays with its first paragraph
    return nodes


def _ancestors(number: str) -> list[str]:
    """The numbers above a section, nearest first: `4.2.1` -> [`4.2`, `4`]. A chapter or appendix has none."""
    parts = number.split(".") if number[0].isdigit() else [number]
    return [".".join(parts[:i]) for i in range(len(parts) - 1, 0, -1)]


def _depth(number: str) -> int:
    return len(_ancestors(number))


def _merge_small_sections(nodes: list[_Node], tokens_min: int) -> list[_Node]:
    """Rule 5. Repeats until no section under the band has a parent present.

    A section's parent is the nearest section before it in the document whose number is one of its ancestors. A number can repeat (numbering restarts per
    chapter or appendix), so the parent is found by position and never by number alone.
    """
    while True:
        pairs = []  # (small section, its parent), in document order
        for position, node in enumerate(nodes):
            if node.tokens >= tokens_min:
                continue
            ancestors = set(_ancestors(node.number))
            target = next((earlier for earlier in reversed(nodes[:position]) if earlier.number in ancestors), None)
            if target is not None:
                pairs.append((node, target))
        if not pairs:
            return nodes
        deepest = max(_depth(node.number) for node, _ in pairs)
        node, target = next(pair for pair in pairs if _depth(pair[0].number) == deepest)
        target.text = target.text + PART_SEPARATOR + node.text
        target.tokens = count_tokens(target.text)
        nodes = [other for other in nodes if other is not node]


def _split_over_band(text: str, tokens_max: int) -> list[str]:
    """Rule 6: pack whole paragraphs into parts of at most `tokens_max` tokens."""
    parts: list[list[str]] = [[]]
    size = 0
    for paragraph in text.split(PART_SEPARATOR):
        tokens = count_tokens(paragraph)
        if parts[-1] and size + tokens > tokens_max:
            parts.append([])
            size = 0
        parts[-1].append(paragraph)
        size += tokens
    return [PART_SEPARATOR.join(part) for part in parts]


def chunk_pages(pages: list[list[str]], doc: str, edition: str, tokens_min: int, tokens_max: int) -> list[Chunk]:
    """The chunks of one document, in document order. Raises NoHeadingsError when the text has no section heading."""
    lines: list[str] = []
    ends: list[tuple[int, str, bool]] = []  # (index in `lines`, furniture key, sits under a dropped line) of the last printed line of each page
    for page, flags in zip(pages, _furniture_flags(pages), strict=True):
        last: tuple[int, str, bool] | None = None
        under_dropped = False  # the nearest printed line above was dropped as furniture
        for line, dropped in zip(page, flags, strict=True):
            if dropped:
                under_dropped = True
            else:
                lines.append(line)
                if line:
                    last = (len(lines) - 1, _furniture_key(line), under_dropped)
                    under_dropped = False
        if last:
            ends.append(last)
    repeated = {key for key, count in Counter(key for _, key, _ in ends).items() if count > 1}
    page_ends = frozenset(index for index, key, under_footer in ends if key in repeated or under_footer)
    nodes = _merge_small_sections(_sections(lines, page_ends), tokens_min)
    chunks = []
    for node in nodes:
        texts = _split_over_band(node.text, tokens_max) if node.tokens > tokens_max else [node.text]
        for index, text in enumerate(texts, start=1):
            section = node.number if len(texts) == 1 else f"{node.number} (part {index})"
            chunks.append(Chunk(doc, edition, section, node.title, text, hashlib.sha256(text.encode("utf-8")).hexdigest()))
    return chunks


def section_slug(section: str) -> str:
    """A section number as a file-name part: `4.2 (part 2)` -> `4.2-part-2`, `Appendix A` -> `appendix-a`."""
    return re.sub(r"[^a-z0-9.]+", "-", section.casefold()).strip("-.") or "section"
