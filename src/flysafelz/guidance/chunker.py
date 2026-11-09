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
  7. A document with no numbered heading (rule 3) is read again by type size (chunk_typeset, read_pdf_runs): the CAA's CAP 1535 and CAP 2562
     set their headings in larger type with no numbers. The two larger sizes used on the most pages are page titles and subheads; headings are
     numbered by place, and only a near-empty section merges into its parent. NoHeadingsError when neither rule finds a heading.
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


# --- Typeset headings: documents whose headings are not numbered (rule 7) -------------------------------------------------------------------

KERNED_CAPITAL = re.compile(r"\b([B-HJ-Z]) (?=[a-z])")  # pypdf splits a kerned capital from its word (`T emporary`, `Y ou`); A and I are words
TYPESET_MIN_TOKENS = 40  # printed subheads are short: only a near-empty one merges into its page title, so its title stays searchable
HEADING_LEVELS = ("page titles", "subheads")
HEADING_MIN_PAGES = 5  # a heading size is used on at least this many pages
HEADING_WORD = re.compile(r"[^\W\d_]{3}")  # a heading has a word of three letters or more; else a chart label or an icon (`i`, `!`, `L`)
HEADING_SIZE_STEP = 1  # a run more than this many points larger than the body text is a heading
TypesetRun = tuple[str, float]  # (text, size in points)


def _run_size(font_size: float, cm: list[float], tm: list[float]) -> float:
    tm_x, _, _, tm_y, *_ = tm  # the vertical scale, else the horizontal one, of the text and the page matrices
    cm_x, _, _, cm_y, *_ = cm
    scale = abs(tm_y or tm_x or 1) * abs(cm_y or cm_x or 1)
    return round(font_size * scale, 1)


def read_pdf_runs(data: bytes) -> list[list[TypesetRun]]:
    """The text runs of each page with their printed size, in reading order. Raises PdfUnreadable like read_pdf_pages."""
    try:
        reader = PdfReader(io.BytesIO(data))
        if reader.is_encrypted:
            raise PdfUnreadable("the PDF is encrypted")
        pages: list[list[TypesetRun]] = []
        for page in reader.pages:
            runs: list[TypesetRun] = []

            def visit(text: str, cm: list[float], tm: list[float], _font: object, font_size: float, runs: list[TypesetRun] = runs) -> None:
                for piece in text.splitlines():
                    cleaned = KERNED_CAPITAL.sub(r"\1", clean_line(piece))
                    if cleaned:
                        runs.append((cleaned, _run_size(font_size, cm, tm)))

            page.extract_text(visitor_text=visit)
            pages.append(runs)
        return pages
    except PdfUnreadable:
        raise
    except Exception as error:
        raise PdfUnreadable(f"pypdf could not read the PDF ({type(error).__name__})") from error


def _body_size(pages: list[list[TypesetRun]]) -> float:
    sizes: Counter[float] = Counter()
    for page in pages:
        for text, size in page:
            sizes[size] += len(text)
    return sizes.most_common(1)[0][0] if sizes else 0.0


def _typeset_heading(text: str) -> bool:
    words = text.split()
    return 0 < len(words) <= MAX_HEADING_WORDS and HEADING_WORD.search(text) is not None and not text.endswith((".", ",", ";", *NAVIGATION_END))


def _heading_sizes(pages: list[list[TypesetRun]], body: float) -> list[float]:
    """The two sizes above the body text used on the most pages (each on at least HEADING_MIN_PAGES), largest first: page titles, then subheads."""
    pages_of: dict[float, set[int]] = {}
    for number, page in enumerate(pages):
        for text, size in page:
            if size > body + HEADING_SIZE_STEP and _typeset_heading(text):
                pages_of.setdefault(size, set()).add(number)
    common = sorted((size for size, seen in pages_of.items() if len(seen) >= HEADING_MIN_PAGES), key=lambda size: (-len(pages_of[size]), -size))
    return sorted(common[: len(HEADING_LEVELS)], reverse=True)


def chunk_typeset(pages: list[list[TypesetRun]], doc: str, edition: str, tokens_min: int, tokens_max: int) -> list[Chunk]:
    """Rule 7. The chunks of a document whose headings are set in larger type rather than numbered (CAP 1535, CAP 2562).

    The body size is the size most of the text is set in. Of the sizes more than HEADING_SIZE_STEP points larger, the two used for heading-like
    runs (rule 3's word limit and endings, a word of three letters) on the most pages are the heading levels: the larger is level 1 (page titles),
    the other level 2 (subheads); other sizes (chart labels, chapter covers) are body text. Consecutive level-2 runs are one wrapped subhead. A
    level-1 heading that repeats the one in force (a running page title) is dropped. Headings are numbered by place (1, 1.1, 1.2, 2 ...), so
    rules 4 and 6 apply unchanged (a capital kerned apart from its word is joined back); rule 5 merges only a section under TYPESET_MIN_TOKENS, so a short subhead keeps its own title. Raises NoHeadingsError when no size qualifies.
    """
    body = _body_size(pages)
    levels = _heading_sizes(pages, body)
    level_of = {size: index for index, size in enumerate(levels)}
    furniture = _furniture_flags([[text for text, _ in page] for page in pages])
    kept = [run for page, flags in zip(pages, furniture, strict=True) for run, dropped in zip(page, flags, strict=True) if not dropped or run[1] in level_of]
    headings: list[tuple[int, str]] = []  # (level, title)
    bodies: list[list[str]] = []
    pending: tuple[float, list[str]] | None = None
    current_top = ""

    def close_pending() -> None:
        nonlocal pending, current_top
        if pending is None:
            return
        size, parts = pending
        pending = None
        title = " ".join(parts)
        level = level_of[size]
        if not _typeset_heading(title):
            if bodies:
                bodies[-1].append(title)
            return
        if level == 0:
            if _squeeze(title) == _squeeze(current_top):
                return  # a running page title
            current_top = title
        headings.append((level, title))
        bodies.append([])

    for text, size in kept:
        if size in level_of:
            if pending is not None and pending[0] == size and level_of[size] == 1:
                pending[1].append(text)
            else:
                close_pending()
                pending = (size, [text])
            continue
        close_pending()
        if bodies:
            bodies[-1].append(text)
    close_pending()
    if not headings:
        raise NoHeadingsError("no section headings found")
    top = sub = 0
    nodes: list[_Node] = []
    for (level, title), body_lines in zip(headings, bodies, strict=True):
        if level == 0 or top == 0:
            top, sub = top + 1, 0
            number = str(top)
        else:
            sub += 1
            number = f"{top}.{sub}"
        first, *rest = paragraphs(body_lines) or [""]
        nodes.append(_Node(number, title, PART_SEPARATOR.join([f"{title}\n{first}".rstrip(), *rest])))
    nodes = _merge_small_sections(nodes, min(tokens_min, TYPESET_MIN_TOKENS))
    chunks = []
    for node in nodes:
        texts = _split_over_band(node.text, tokens_max) if node.tokens > tokens_max else [node.text]
        for index, text in enumerate(texts, start=1):
            section = node.number if len(texts) == 1 else f"{node.number} (part {index})"
            chunks.append(Chunk(doc, edition, section, node.title, text, hashlib.sha256(text.encode("utf-8")).hexdigest()))
    return chunks


def _squeeze(text: str) -> str:
    return "".join(ch for ch in text.casefold() if ch.isalnum())
