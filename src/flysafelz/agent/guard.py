"""The guard's checks (G14): is a final answer grounded in this turn's tool results? Pure code: no model, no network, no I/O, the same findings for the same input.

    guard_answer(answer, turn) -> list[GuardFinding]     [] means the answer passes

`turn` is a `TurnSources`: what this turn's tools returned (text and artifact), the thread's phone placeholders, the calendar the system prompt listed and the
landing zone. Nothing from an earlier turn is in it, so a number from an earlier turn is a number no tool printed. A finding is one plain sentence that names the
text and says why; the sentences are fixed wording, so the retry prompt and the trace feedback are reproducible. What the retry, the fallback, the re-check line and
`rehydrate` do with them is the middleware's job (PR-11.F1), not this module's.

(a) Verdict words
    The six display strings (`display_verdict` builds them), matched ignoring case, the kind of dash, the middle dot and the spacing. Every one the answer writes
    must be one `assess` returned this turn. The verdict the first sentence names must include the overall one, and an answer that writes any verdict must
    name one in its first sentence. With no `assess` result, no verdict word is allowed; with an overall of None (every hour OUTLOOK or too far out) none is
    allowed either, and none may sit in a sentence about an OUTLOOK hour. On every path, the bare capital verdict word and the phrases the prompt bans
    (BANNED_PHRASES) are findings.
(b) Numbers (Q2). A written number passes when some number the tools printed gives it:
    precision   the printed number, rounded half up to the decimals the answer writes, is the written number ("10 mph" for 9.9; "9.9 mph" for 9.86; not "9.8 mph")
    unit        the unit is the printed one, or a change in UNIT_TABLE (m, km, ft; mph, kt, m/s; h, min). A bare number matches a count or a percentage only
    limit       rounding never crosses a limit, and the check reads numbers only (I10), never the words around them: a printed value that has a limit in
                an artifact (a weather value, a caution) may not be written so that the limit's comparison gives another answer, nor so that it lands on the
                limit when the value is not exactly there ("about 9 mph" for 9.3 against 9). A number equal to a printed limit is grounded by that limit
                for the precision check, but this check wins: when a value that crosses rounds onto the limit's number ("6 mph" beside 6.04 against a limit
                of 6) the sentence is a finding that says to give the value as printed and not to restate the limit (`_crossing_finding`). What a sentence
                says about where a value sits against its limit is not read, with one exception: the fixed phrase "over|above|under|below|at the <number> <unit>
                limit" (`_LIMIT_PHRASE`). Its number must be a limit a tool printed, its word must be what `contracts.position_word` gives for the printed value
                (over = past the limit, 9.3 against >= 9 included; at = exactly the limit; under = not reached), and a not / n't / never directly before it is a
                finding. Code states every other limit position in its own lines. There is no percentage tolerance
    words       "no", "one" .. "ten", "a couple of", "a few" count only directly before NOTAM(s), hour(s), minute(s), segment(s), condition(s), area(s),
                airspace(s); the number must be a count (or a span, for hours and minutes) a tool printed. "one thing to note" is not a count
    arithmetic  a sum, a difference, an average no tool printed is a number no tool printed
    times       HH:MM (also 4pm) must be a time a tool printed
    dates       a date ("Sunday 1 Nov", 2026-11-01) must be one the calendar or a tool lists; years are not checked
    exempt      NOTAM ids, <PHONE_n> placeholders, coordinates (they have their own check), years, list markers. A section number of three parts or more
                (3.2.1) must be one a tool printed
(c) References. Every NOTAM id (`contracts.NOTAM_ID_RE`) and <PHONE_n> must be in a tool result this turn, and a placeholder must be one of the thread's. A
    real-looking phone number (the shapes `redact_phones` finds) is a finding. A coordinate pair, at the decimals it is written, must be one of an artifact's
    (the landing zone, an overlay's point) or a pair converted from a NOTAM's raw degrees-minutes-seconds text. A pair the answer writes in degrees, minutes,
    seconds (534732N 0000459W) is converted by code and must be one of those places to the arc second.
(d) Re-check. A closed list of phrases (`FINAL_PHRASES`: "the forecast is final", "no need to re-check", ...), matched ignoring case, at word boundaries,
    with dashes and apostrophes made one. A sentence with one of them is a finding; every other wording is a known gap (the guard reads no meaning). The
    re-check line itself is added by code after the guard, so its absence is not a finding.

Words inside a quoted excerpt (double quotes, a code span or block, a block quote or `|` lines) do not count, but only when the quoted text really is a stretch of
a NOTAM's raw text or a guidance chunk this turn, compared with any kind of dash made one (`_excerpt_corpus`): anything else in quotes, a line `assess` printed
included, is the answer's own words.
"""

import calendar
import operator
import re
import unicodedata
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from decimal import Decimal
from itertools import product
from typing import Any, NamedTuple, Self
from zoneinfo import ZoneInfo

from flysafelz.agent.prompts import BANNED_PHRASES, BARE_VERDICT_WORD
from flysafelz.agent.tools.artifacts import (
    AirspaceArtifact,
    AssessArtifact,
    GuidanceArtifact,
    NotamsArtifact,
    WeatherArtifact,
)
from flysafelz.contracts import NOTAM_ID_RE, Verdict, display_verdict, position_word
from flysafelz.geo import METRES_PER_KM, METRES_PER_NM, OutsideUKCoverage, check_uk_bounds
from flysafelz.notam_area import coordinates_each, coordinates_in
from flysafelz.redact import redact_phones
from flysafelz.rules import RULES
from flysafelz.timetext import clock

# --- the inputs and the output ----------------------------------------------------------------------------------------


class GuardFinding(str):
    """One failed check as one plain sentence. It is a `str`, so it is the sentence wherever a sentence is wanted; `check` says which of (a) to (d)."""

    check: str

    def __new__(cls, check: str, sentence: str) -> Self:
        finding = super().__new__(cls, sentence)
        finding.check = check
        return finding


@dataclass(frozen=True)
class ToolResult:
    """One tool's answer this turn: its name, the text the model read and the artifact beside it (None: the tool gave none, for instance an error)."""

    name: str
    text: str
    artifact: Any = None


@dataclass(frozen=True)
class TurnSources:
    """Everything the guard may check an answer against.

    results     this turn's tool results, in the order the tools answered
    tz          the thread's time zone (an IANA name), for the times artifacts hold
    phone_keys  the thread's phone_map keys (`<PHONE_1>`)
    calendar    the days the system prompt's calendar listed: today and the days after it, in `tz`
    lz          the landing zone (lat, lon), None when unknown
    """

    results: tuple[ToolResult, ...]
    tz: str
    phone_keys: frozenset[str]
    calendar: tuple[date, ...]
    lz: tuple[float, float] | None = None

    @property
    def assessments(self) -> tuple[AssessArtifact, ...]:
        return tuple(r.artifact for r in self.results if isinstance(r.artifact, AssessArtifact))

    @property
    def assess_ran(self) -> bool:
        """Whether `assess` gave an assessment this turn (a tool error has no artifact, so it did not)."""
        return bool(self.assessments)

    @property
    def overall(self) -> str | None:
        """The overall verdict's display string, or None when `assess` did not run or every hour was OUTLOOK or too far out."""
        return next((a.overall.display for a in self.assessments if a.overall is not None), None)

    @property
    def displays(self) -> frozenset[str]:
        """Every verdict display string `assess` returned this turn, the overall and the hours'."""
        shown = {a.overall.display for a in self.assessments if a.overall is not None}
        shown |= {s.display for a in self.assessments for s in a.segments if s.display is not None}
        return frozenset(shown)

    @property
    def tiers(self) -> tuple[tuple[str, str], ...]:
        """(start, tier) of every hour or block `assess` judged, in order; start is the ISO time the artifact holds."""
        return tuple((item.start, item.tier) for a in self.assessments for item in (*a.segments, *a.blocks))


# --- text: one canonical form, and what is a quotation ------------------------------------------------------------------

_DASHES = re.compile("[\u2010-\u2015\u2212]")
_DOTS = re.compile("[\u2022\u2219\u22c5\u30fb]")
_INVISIBLE = re.compile("[\u200b-\u200f\u2060\ufeff]")
_PATH_TAG = re.compile(r"\A\s*\[path:\s*[a-z]+\][ \t]*\n?", re.IGNORECASE)
_QUOTED = re.compile(r'```.*?```|`[^`\n]+`|"[^"]+"|\u201c[^\u201d]+\u201d|(?:^[ \t]*[|>][^\n]*(?:\n|\Z))+', re.DOTALL | re.MULTILINE)
_LINE_MARK = re.compile(r"^[ \t]*[|>]", re.MULTILINE)
_QUOTE_MARK = re.compile('[`"\u201c\u201d]')
_NOT_NEWLINE = re.compile(r"[^\n]")
_SENTENCE_END = re.compile(r"[.!?](?=\s|$)|\n")


def _canon(text: str) -> str:
    """NFKC (a non-breaking space is a space), one dash, one middle dot, no zero-width characters, no leading path tag."""
    text = _INVISIBLE.sub("", unicodedata.normalize("NFKC", text))
    return _PATH_TAG.sub("", _DOTS.sub("\u00b7", _DASHES.sub("-", text)))


def _flat(text: str) -> str:
    """Lower case, no quote or block marks, one space between words: the form in which a quotation is compared with a tool result."""
    return " ".join(_QUOTE_MARK.sub(" ", _LINE_MARK.sub(" ", text)).lower().split())


def _blank(text: str) -> str:
    """The same length, all spaces, line breaks kept, so positions and sentences stay where they were."""
    return _NOT_NEWLINE.sub(" ", text)


def _sentence_bounds(text: str) -> list[tuple[int, int]]:
    """(start, end) of every sentence of `text`: split at a `.` `!` or `?` before a space or the end, and at a line break."""
    bounds, start = [], 0
    for end in _SENTENCE_END.finditer(text):
        bounds.append((start, end.start()))
        start = end.end()
    bounds.append((start, len(text)))
    return [(first, last) for first, last in bounds if text[first:last].strip()]


MIN_EXCERPT_CHARS = 15  # a shorter quotation ("GO", "1 mph") is a phrase of the answer's own, whatever a tool's text happens to contain


def _is_excerpt(quoted: str, corpus: str) -> bool:
    """Whether `quoted` (flattened) is a stretch of `corpus` that starts and ends on whole tokens: not inside a word, a hyphenated word (no-go), a number
    (3.1 mph) or an id, and at least MIN_EXCERPT_CHARS long."""
    if len(quoted) < MIN_EXCERPT_CHARS:
        return False
    return re.search(rf"(?<![\w./-]){re.escape(quoted)}(?![\w/-]|\.\d)", corpus) is not None


def _without_excerpts(text: str, corpus: str) -> str:
    """The answer with every quoted excerpt blanked. Quoted text counts as an excerpt only when it is a whole-token stretch of `corpus` (this turn's NOTAM raw
    text and guidance text, see `_excerpt_corpus`), and long enough to be one (a short quotation is the answer's own phrase)."""

    def swap(match: re.Match[str]) -> str:
        return _blank(match.group()) if _is_excerpt(_flat(match.group()), corpus) else match.group()

    return _QUOTED.sub(swap, text)


_EXCERPT_TOOLS = frozenset({"get_notams", "search_guidance"})  # the tools whose text is a source's own words


def _excerpt_corpus(turn: TurnSources) -> str:
    """The text a quotation may be a stretch of, in the answer's form (`_canon`, then `_flat`): the raw E text of this turn's NOTAMs and the text of its guidance
    chunks, and nothing else. What `assess` or any other tool printed is code's wording, not a source's, so quoting it hides nothing. A NOTAM or guidance
    result with no artifact (an error, or a text-only result) offers its text, since there is no raw text to take instead."""
    parts: list[str] = []
    for result in turn.results:
        if isinstance(result.artifact, NotamsArtifact):
            parts += [item.e_text for item in result.artifact.notams]
        elif isinstance(result.artifact, GuidanceArtifact):
            parts += [chunk.text for chunk in result.artifact.chunks]
        elif result.artifact is None and result.name in _EXCERPT_TOOLS:
            parts.append(result.text)
    return _flat(" ".join(_canon(part) for part in parts))


# --- units and numbers -----------------------------------------------------------------------------------------------

# How many of its family's base unit (metre, metre per second, minute) one of each unit is. These are conversion factors, and they are the only
# conversions the guard allows: a unit in another family (NM, %, hPa, mm/h) never matches a different unit. Decimal strings, so the figures are exact.
UNIT_TABLE: dict[str, tuple[str, Decimal]] = {
    "m": ("length", Decimal(1)),
    "km": ("length", Decimal(METRES_PER_KM)),
    "ft": ("length", Decimal("0.3048")),
    "m/s": ("speed", Decimal(1)),
    "mph": ("speed", Decimal("1609.344") / Decimal(3600)),
    "kt": ("speed", Decimal(METRES_PER_NM) / Decimal(3600)),
    "min": ("time", Decimal(1)),
    "h": ("time", Decimal(60)),
}
_ALIASES = {
    "mph": "mph", "mi/h": "mph", "kt": "kt", "kts": "kt", "kn": "kt", "knot": "kt", "knots": "kt", "m/s": "m/s", "mps": "m/s",
    "m": "m", "metre": "m", "metres": "m", "meter": "m", "meters": "m", "km": "km", "kilometre": "km", "kilometres": "km", "kilometer": "km", "kilometers": "km",
    "ft": "ft", "foot": "ft", "feet": "ft", "nm": "NM", "nautical mile": "NM", "nautical miles": "NM",
    "mi": "mi", "mile": "mi", "miles": "mi", "%": "%", "percent": "%", "per cent": "%", "hpa": "hPa", "mm/h": "mm/h", "mm": "mm", "j/kg": "J/kg", "w/m\u00b2": "W/m\u00b2", "w/m2": "W/m\u00b2",
    "\u00b0c": "\u00b0C", "\u00b0": "deg", "degree": "deg", "degrees": "deg",
    "h": "h", "hr": "h", "hrs": "h", "hour": "h", "hours": "h", "min": "min", "mins": "min", "minute": "min", "minutes": "min",
}  # fmt: skip
_UNIT = "|".join(re.escape(alias).replace(r"\ ", r"\s") for alias in sorted(_ALIASES, key=len, reverse=True))
_HEDGE = r"(?:about|around|roughly|approximately|approx\.?|nearly|almost|just over|just under|over|under|more than|less than|up to|~)"
_NUMBER = re.compile(
    rf"(?:(?<!\w)(?P<hedge>{_HEDGE})\s*)?(?<![\w./])(?P<sign>-)?(?P<int>\d{{1,3}}(?:,\d{{3}})+|\d+)(?P<frac>\.\d+)?(?:\s?(?P<unit>{_UNIT}))?(?![\w])",
    re.IGNORECASE,
)
_UNIT_AFTER = re.compile(rf"\s?(?:{_UNIT})(?![\w])", re.IGNORECASE)
_RANGE_JOIN = re.compile(r"\s*(?:-|to)\s*", re.IGNORECASE)
_YEAR = re.compile(r"(?:19|20)\d{2}")
_SPAN = re.compile(r"(?<![\w.])(?P<h>\d+)\s*(?:h|hr|hrs|hours?)\s*(?:and\s+)?(?P<m>\d+)\s*(?:min|mins|minutes?)(?![\w])", re.IGNORECASE)
_SECTION = re.compile(r"(?<![\w.])\d+(?:\.\d+){2,}(?![\w])")
_LIST_MARK = re.compile(r"^[ \t]*\d+[.)](?=\s)", re.MULTILINE)
_NUMBER_WORDS = {word: Decimal(index) for index, word in enumerate(["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"])}
_NUMBER_WORDS |= {"no": _NUMBER_WORDS["zero"], "a couple of": _NUMBER_WORDS["two"], "a couple": _NUMBER_WORDS["two"], "couple of": _NUMBER_WORDS["two"], "a few": _NUMBER_WORDS["three"]}
_COUNTED = {"notam": None, "hour": "h", "minute": "min", "segment": None, "condition": None, "area": None, "airspace": None}
_COUNTED_WORD = re.compile(
    rf"\b(?P<word>{'|'.join(sorted(map(re.escape, _NUMBER_WORDS), key=len, reverse=True))})\s+(?P<noun>{'|'.join(_COUNTED)})s?\b",
    re.IGNORECASE,
)
ONE_HOUR = timedelta(hours=1)
EPSILON = Decimal("1E-9")  # equal, apart from arithmetic noise
DMS_TOLERANCE = Decimal(1) / Decimal(7200) + EPSILON  # half an arc second, in degrees: a pair written in seconds is its printed place, however the tool printed it
_HALF = Decimal("0.5")
_TRIGGERED = {">=": operator.ge, ">": operator.gt, "<": operator.lt, "<=": operator.le}


class Quantity(NamedTuple):
    """A number as written: its position, value, unit (None: no unit), the decimals it is written with and its words."""

    pos: int
    value: Decimal
    unit: str | None
    places: int
    written: str


@dataclass(frozen=True)
class Limit:
    """A limit a value is checked against: how (`>=`, `>`, `<`, `<=`; None: not known), the limit and its unit."""

    comparison: str | None
    value: Decimal
    unit: str | None


@dataclass(frozen=True)
class Source:
    """A number a tool printed, with the limits the artifact pairs it with and what it is called ("gust spread"; empty when the tool gave no name).

    `is_limit` marks a limit itself: it grounds a written number equal to it, but is never itself checked against a limit.
    """

    value: Decimal
    unit: str | None
    limits: tuple[Limit, ...] = ()
    is_limit: bool = False
    name: str = ""


def _unit(raw: str | None) -> str | None:
    """The plain name of a unit as written or as an artifact gives it: `mph`, `NM`, `%`, `h`; None for no unit. An unknown unit stays as it is."""
    if raw is None:
        return None
    key = " ".join(unicodedata.normalize("NFKC", raw).lower().split())
    return _ALIASES.get(key, raw)


def _convert(value: Decimal, unit: str | None, to: str | None) -> Decimal | None:
    """`value` in `unit` as a number of `to`, or None when the table does not allow it. A bare number takes a count or a percentage only."""
    if unit == to:
        return value
    if to is None:
        return value if unit == "%" else None
    if unit is None or unit not in UNIT_TABLE or to not in UNIT_TABLE or UNIT_TABLE[unit][0] != UNIT_TABLE[to][0]:
        return None
    return value * UNIT_TABLE[unit][1] / UNIT_TABLE[to][1]


def _half(places: int) -> Decimal:
    """Half of the last written decimal place: how far a printed number may be from a written one and still round to it."""
    return _HALF.scaleb(-places)


def _rounds_to(printed: Decimal, written: Decimal, places: int) -> bool:
    """Whether `printed` rounds to `written` at `places` decimals, half away from zero: 9.85 is 9.9, 9.84 is not."""
    if printed != 0 and written != 0 and (printed < 0) != (written < 0):
        return False
    low = abs(written) - _half(places)
    return low <= abs(printed) < abs(written) + _half(places)


def _plain(value: Decimal, unit: str | None) -> str:
    return " ".join(part for part in (format(value.normalize(), "f"), unit) if part)


def _crossing(written: Quantity, source: Source) -> Limit | None:
    """The first limit the written number crosses: it changes what the limit's comparison says about the printed value, or it lands on the limit
    when the printed value is not exactly there."""
    for limit in source.limits:
        shown = _convert(written.value, written.unit, limit.unit)
        printed = _convert(source.value, source.unit, limit.unit)
        if shown is None or printed is None:
            continue
        lands = abs(shown - limit.value) <= EPSILON < abs(printed - limit.value)
        test = _TRIGGERED.get(limit.comparison or "")
        if lands or (test is not None and test(printed, limit.value) != test(shown, limit.value)):
            return limit
    return None


def _is_the_limit(written: Quantity, limit: Limit) -> bool:
    shown = _convert(written.value, written.unit, limit.unit)
    return shown is not None and abs(shown - limit.value) <= EPSILON


_HEDGED = re.compile(rf"\A{_HEDGE}", re.IGNORECASE)
NOT_PRINTED = "'{}' is not a number any tool printed this turn, or a rounding of one"


def _crossing_finding(written: Quantity, source: Source, limit: Limit) -> str:
    """The finding for a written number that a printed value rounds onto, across or onto its limit. Number-only: the words around the number are not read.

    A number that is the limit itself (unhedged, equal to the limit) gets the fixed sentence that says how to fix it. A hedged number, or one that is not the limit
    ("about 9 mph" for 9.3, "10 mph" for 9.6 against 9.8), is a rounding that crosses.
    """
    if _HEDGED.match(written.written) is None and _is_the_limit(written, limit):
        printed = _convert(source.value, source.unit, limit.unit)
        verb = "exceeds" if printed is not None and printed > limit.value else "falls below"
        return (
            f"'{written.written}' is the limit that {source.name or 'the value'} {_plain(source.value, source.unit)} {verb}. "
            "Give each value exactly as the tool printed it. Don't restate limit values: code adds a line stating each limit reached."
        )
    return f"'{written.written}' rounds {_plain(source.value, source.unit)} across the {_plain(limit.value, limit.unit)} limit"


def _number_finding(written: Quantity, sources: Iterable[Source]) -> str | None:
    """None when the written number is a rounding of a printed one that it does not misplace against that value's limit, else the finding sentence.

    The number is checked as a number (I10): the words around it are never read, so no sentence about a limit can excuse it. A printed value that rounds to
    the written number and has the written number on the other side of its limit, or on a limit it is not exactly at, fails (`_crossing`). That wins over
    every other source that happens to equal the written number, the limit itself included: the limit is printed beside the value, so "about 9 mph" for a
    gradient of 9.3 would otherwise pass on the strength of the "9 mph" in "caution >= 9 mph". A number equal to a printed limit, with no value rounding onto
    it, is grounded by that limit for the precision check.
    """
    supported = False
    for source in sources:
        converted = _convert(source.value, source.unit, written.unit)
        if converted is None or not _rounds_to(converted, written.value, written.places):
            continue
        supported = True
        if not source.is_limit and (limit := _crossing(written, source)) is not None:
            return _crossing_finding(written, source, limit)
    return None if supported else NOT_PRINTED.format(written.written)



# --- scanning text for what the guard checks ---------------------------------------------------------------------------

_MONTHS = {name.lower(): abbr.lower() for name, abbr in zip(calendar.month_name, calendar.month_abbr, strict=True) if name} | {abbr.lower(): abbr.lower() for abbr in calendar.month_abbr if abbr}
_MONTHS["sept"] = _MONTHS["sep"]
_WEEKDAYS = {name.lower(): abbr.lower() for name, abbr in zip(calendar.day_name, calendar.day_abbr, strict=True)} | {abbr.lower(): abbr.lower() for abbr in calendar.day_abbr}
_WEEKDAYS |= {"tues": _WEEKDAYS["tue"], "thur": _WEEKDAYS["thu"], "thurs": _WEEKDAYS["thu"]}


def _alternatives(words: dict[str, str]) -> str:
    return "|".join(sorted(words, key=len, reverse=True))


_DATE = re.compile(
    rf"(?<![\w:.])(?:(?P<wd>{_alternatives(_WEEKDAYS)})\.?,?\s+)?(?P<day>\d{{1,2}})(?:st|nd|rd|th)?\s+(?:of\s+)?(?P<mon>{_alternatives(_MONTHS)})\b\.?(?:\s+(?:19|20)\d{{2}}\b)?",
    re.IGNORECASE,
)
_ISO_DATE = re.compile(r"(?<![\w-])(?:19|20)\d{2}-(?P<mon>\d{2})-(?P<day>\d{2})(?![\w])")
_CLOCK = re.compile(r"(?<![\w:.])(?P<h>[01]?\d|2[0-3]):(?P<m>[0-5]\d)(?::[0-5]\d)?(?![\w:])")
_CLOCK_12 = re.compile(r"(?<![\w:.])(?P<h>\d{1,2})(?::(?P<m>[0-5]\d))?\s?(?P<ap>[ap])\.?m\b\.?", re.IGNORECASE)
_CLOCK_RANGE = re.compile(  # 0800-1800, as a reading's schedule is printed; not 1000-1500 ft
    rf"(?<![\w.])(?P<ah>[01]\d|2[0-3])(?P<am>[0-5]\d)\s*-\s*(?P<bh>[01]\d|2[0-3])(?P<bm>[0-5]\d)(?![\w])(?!\s?(?:{_UNIT})(?![\w]))", re.IGNORECASE
)
_COORDINATE_PAIR = re.compile(
    r"(?<![\w.\-])(?P<lat>-?\d+\.\d+)\s*\u00b0?\s*(?P<ns>[NS])?(?P<sep>\s*[,;]\s*|\s+)(?P<lon>-?\d+\.\d+)\s*\u00b0?\s*(?P<ew>[EW])?(?!\w|\.\d)",
    re.IGNORECASE,
)


class Dated(NamedTuple):
    weekday: str | None
    day: int
    month: str
    written: str


class Timed(NamedTuple):
    clock: str
    written: str


class Placed(NamedTuple):
    """A decimal coordinate pair as written: both values and the decimals each is written with."""

    lat: Decimal
    lon: Decimal
    lat_places: int
    lon_places: int
    written: str


@dataclass
class Scan:
    """What a text mentions that the guard checks: numbers, times, dates, coordinate pairs and multi-part section numbers."""

    numbers: list[Quantity] = field(default_factory=list)
    times: list[Timed] = field(default_factory=list)
    dates: list[Dated] = field(default_factory=list)
    places: list[Placed] = field(default_factory=list)
    sections: list[str] = field(default_factory=list)


def _padded(hour: str, minute: str) -> str:
    return f"{int(hour):02d}:{minute}"


def _scan_dates_and_times(text: str, found: Scan) -> str:
    """Record and blank the dates and the times of day; return what is left."""

    def iso(match: re.Match[str]) -> str:
        number = int(match["mon"])
        if 0 < number < len(calendar.month_abbr):
            found.dates.append(Dated(None, int(match["day"]), calendar.month_abbr[number].lower(), match.group()))
        return _blank(match.group())

    def dated(match: re.Match[str]) -> str:
        weekday = _WEEKDAYS[match["wd"].lower()] if match["wd"] else None
        found.dates.append(Dated(weekday, int(match["day"]), _MONTHS[match["mon"].lower()], match.group().strip()))
        return _blank(match.group())

    def twelve(match: re.Match[str]) -> str:
        try:
            clock_text = datetime.strptime(f"{match['h']}:{match['m'] or '00'} {match['ap'].upper()}M", "%I:%M %p").strftime("%H:%M")
        except ValueError:
            return match.group()
        written = match.group().strip()
        found.times.append(Timed(clock_text, written[:-1] if written.endswith(".") and "." not in written[:-1] else written))
        return _blank(match.group())

    def twenty_four(match: re.Match[str]) -> str:
        found.times.append(Timed(_padded(match["h"], match["m"]), match.group()))
        return _blank(match.group())

    def hhmm_range(match: re.Match[str]) -> str:
        found.times.append(Timed(_padded(match["ah"], match["am"]), match.group()))
        found.times.append(Timed(_padded(match["bh"], match["bm"]), match.group()))
        return _blank(match.group())

    text = _ISO_DATE.sub(iso, text)
    text = _DATE.sub(dated, text)
    text = _CLOCK_12.sub(twelve, text)
    text = _CLOCK.sub(twenty_four, text)
    return _CLOCK_RANGE.sub(hhmm_range, text)


def _decimals(digits: str) -> int:
    """How many decimals a number is written with: 4 for 53.8028."""
    return -int(Decimal(digits).as_tuple().exponent)


def _scan_places(text: str, found: Scan) -> str:
    """Record and blank decimal coordinate pairs inside the UK bounds: `53.8028, -0.0866`, `53.8028N 0.0866W`. A pair followed by a unit is two measurements."""

    def pair(match: re.Match[str]) -> str:
        if (match["ns"] is None and not match["sep"].strip()) or _UNIT_AFTER.match(text, match.end()):
            return match.group()
        lat, lon = Decimal(match["lat"]), Decimal(match["lon"])
        if match["ns"] is not None:
            lat = -abs(lat) if match["ns"].upper() == "S" else abs(lat)
        if match["ew"] is not None:
            lon = -abs(lon) if match["ew"].upper() == "W" else abs(lon)
        try:
            check_uk_bounds(float(lat), float(lon))
        except OutsideUKCoverage:
            return match.group()
        found.places.append(Placed(lat, lon, _decimals(match["lat"]), _decimals(match["lon"]), match.group().strip()))
        return _blank(match.group())

    return _COORDINATE_PAIR.sub(pair, text)


def _scan_numbers(text: str, found: Scan, *, words: bool) -> None:
    """Record every number, and (`words`) every number word directly before a counted noun. A number is not read when letters touch it (EGD496A, 534732N)."""
    text = _LIST_MARK.sub(lambda m: _blank(m.group()), text)

    def section(match: re.Match[str]) -> str:
        found.sections.append(match.group())
        return _blank(match.group())

    def span(match: re.Match[str]) -> str:
        minutes = Decimal(match["h"]) * UNIT_TABLE["h"][1] + Decimal(match["m"])
        found.numbers.append(Quantity(match.start(), minutes, "min", 0, match.group()))
        return _blank(match.group())

    text = _SECTION.sub(section, text)
    text = _SPAN.sub(span, text)
    for match in _COUNTED_WORD.finditer(text) if words else ():
        unit = _COUNTED[match["noun"].lower()]
        found.numbers.append(Quantity(match.start(), _NUMBER_WORDS[" ".join(match["word"].lower().split())], unit, 0, match.group()))
    pending: list[re.Match[str]] = list(_NUMBER.finditer(text))
    for index, match in enumerate(pending):
        unit = _unit(match["unit"])
        following = pending[index + 1] if index + 1 < len(pending) else None
        if unit is None and following is not None and following["unit"] and _RANGE_JOIN.fullmatch(text[match.end() : following.start()]):
            unit = _unit(following["unit"])  # "5-10 mph", "4 to 6 mph": the unit is for both
        if unit is None and not match["frac"] and _YEAR.fullmatch(match["int"]):
            continue
        value = Decimal(match["int"].replace(",", "") + (match["frac"] or ""))
        found.numbers.append(Quantity(match.start(), -value if match["sign"] else value, unit, len(match["frac"] or ".") - 1, match.group().strip()))


def scan(text: str, *, words: bool = True) -> Scan:
    """Everything in `text` the checks look at. `text` must already be canonical. `words` is False for a tool's text, which counts in digits."""
    found = Scan()
    text = _scan_dates_and_times(text, found)
    text = _scan_places(text, found)
    _scan_numbers(NOTAM_ID_RE.sub(lambda m: _blank(m.group()), text), found, words=words)
    return found


# --- what the tools printed -----------------------------------------------------------------------------------------


def _dec(value: float) -> Decimal:
    return Decimal(repr(value))


def _measure(value: float | None, unit: str | None, limits: Iterable[Limit] = (), name: str = "") -> list[Source]:
    return [] if value is None else [Source(_dec(value), _unit(unit), tuple(limits), name=name)]


def _limit_of(rule_id: str, value: float | None, unit: str | None) -> list[Limit]:
    """The limit of a caution as the artifact has it, and how rules.yaml compares against it (None when the rule id is not one)."""
    rule = getattr(RULES, rule_id, None)
    if value is None:
        return []
    return [Limit(rule.comparison.value if rule is not None else None, _dec(value), _unit(unit))]


def _caution_name(rule_id: str) -> str:
    """What a caution's value is called: its rule id without the `_caution` or `_no_go` ending, as words (`gust_spread_caution` is "gust spread")."""
    return re.sub(r"_(?:caution|no_go)\Z", "", rule_id).replace("_", " ")


def _instant(value: str | datetime | None) -> datetime | None:
    if isinstance(value, datetime):
        return value
    try:
        return datetime.fromisoformat(value) if value else None
    except ValueError:
        return None


@dataclass
class Printed:
    """What this turn's tools printed, as the checks need it."""

    numbers: list[Source] = field(default_factory=list)
    times: set[str] = field(default_factory=set)
    dates: set[tuple[str | None, int, str]] = field(default_factory=set)
    sections: set[str] = field(default_factory=set)
    ids: set[str] = field(default_factory=set)
    placeholders: set[str] = field(default_factory=set)
    places: list[tuple[Decimal, Decimal]] = field(default_factory=list)
    text: str = ""


def _date_key(day: date) -> tuple[str | None, int, str]:
    return calendar.day_abbr[day.weekday()].lower(), day.day, calendar.month_abbr[day.month].lower()


def _add_instant(printed: Printed, moment: str | datetime | None, tz: str) -> None:
    found = _instant(moment)
    if found is None or found.tzinfo is None:
        return
    printed.times.add(clock(found, tz))
    printed.dates.add(_date_key(found.astimezone(ZoneInfo(tz)).date()))


def _from_artifact(printed: Printed, artifact: Any, tz: str) -> None:
    if isinstance(artifact, WeatherArtifact):
        for hour in artifact.hours:
            _add_instant(printed, hour.hour, tz)
            for value in hour.values:
                limits = [Limit(item.comparison, _dec(item.limit), _unit(item.unit)) for item in value.limits]
                printed.numbers += _measure(value.value, value.unit, limits, value.label)
                printed.numbers += [Source(limit.value, limit.unit, is_limit=True) for limit in limits]
    elif isinstance(artifact, AirspaceArtifact):
        printed.numbers += _measure(artifact.ceiling_qnh_ft, "ft")
        for item in artifact.findings:
            printed.numbers += _measure(item.distance_nm, "NM") + _measure(item.bearing_deg, "deg")
            printed.numbers += _measure(item.floor_qnh_ft, "ft") + _measure(item.ceiling_qnh_ft, "ft")
            printed.ids |= {item.from_notam} if item.from_notam else set()
    elif isinstance(artifact, NotamsArtifact):
        printed.numbers += [Source(_dec(n), None) for n in (artifact.counts.within, artifact.counts.awareness, artifact.counts.ignored)]
        for item in artifact.notams:
            printed.numbers += _measure(item.bearing_deg, "deg")
            printed.ids.add(item.id)
            printed.places += _decimal_pairs(coordinates_in(item.e_text))
        _add_instant(printed, artifact.issued_at, tz)
    elif isinstance(artifact, AssessArtifact):
        _from_assess(printed, artifact, tz)
    elif isinstance(artifact, GuidanceArtifact):
        for chunk in artifact.chunks:
            printed.sections.add(chunk.section)
            printed.numbers += _section_number(chunk.section)
    if isinstance(artifact, (WeatherArtifact, AirspaceArtifact, NotamsArtifact)):
        _add_instant(printed, artifact.window_start, tz)
        _add_instant(printed, artifact.window_end, tz)
        _add_instant(printed, artifact.source.fetched_at, tz)
    for feature in getattr(artifact, "overlays", ()):
        if feature.geometry.type == "Point":
            lon, lat, *_ = feature.geometry.coordinates
            printed.places.append((_dec(lat), _dec(lon)))
        if feature.properties.text:
            printed.places += _decimal_pairs(coordinates_in(feature.properties.text))


def _from_assess(printed: Printed, artifact: AssessArtifact, tz: str) -> None:
    _add_instant(printed, artifact.window_start, tz)
    _add_instant(printed, artifact.window_end, tz)
    _add_instant(printed, artifact.fetched_at, tz)
    _add_instant(printed, artifact.pib_issued_at, tz)
    counts = artifact.counts
    printed.numbers += [Source(_dec(n), None) for n in (counts.notams_within, counts.hours_assessed, counts.hours_outlook, counts.hours_too_far, counts.flyable_windows)]
    printed.numbers += _measure(counts.local_area_nm, "NM")
    for segment in artifact.segments:
        _add_instant(printed, segment.start, tz)
        _add_instant(printed, segment.end, tz)
        printed.numbers += _measure(segment.rain_chance, "%")
        for caution in segment.cautions:
            limits = _limit_of(caution.rule_id, caution.limit, caution.unit)
            printed.numbers += _measure(caution.value, caution.unit, limits, _caution_name(caution.rule_id))
            printed.numbers += [Source(limit.value, limit.unit, is_limit=True) for limit in limits]
        for notam in segment.notams:
            printed.ids.add(notam.id)
            printed.numbers += _measure(notam.bearing_deg, "deg")
    for block in artifact.blocks:
        _add_instant(printed, block.start, tz)
        _add_instant(printed, block.end, tz)
        printed.numbers += _measure(float(block.hours), None) + _measure(float(block.hours), "h")
        for low, high, unit in ((block.wind_min, block.wind_max, block.wind_unit), (block.gust_min, block.gust_max, block.gust_unit), (block.rain_min, block.rain_max, block.rain_unit)):
            printed.numbers += _measure(low, unit) + _measure(high, unit)
    for span in artifact.flyable:
        _add_instant(printed, span.start, tz)
        _add_instant(printed, span.end, tz)
        printed.numbers.append(Source(Decimal(span.minutes), "min"))


def _section_number(section: str) -> list[Source]:
    """A guidance section such as `4.2` is also a number a tool printed."""
    try:
        return [Source(Decimal(section), None)]
    except ArithmeticError:
        return []


def _decimal_pairs(points: Iterable[tuple[float, float]]) -> list[tuple[Decimal, Decimal]]:
    return [(_dec(lat), _dec(lon)) for lat, lon in points]


def _printed(turn: TurnSources) -> Printed:
    """Gather everything the turn's tools printed: from each text (what the model read) and from each artifact (the values with their limits)."""
    printed = Printed(text=_flat(" ".join(result.text for result in turn.results)))
    for result in turn.results:
        text = _canon(result.text)
        found = scan(text, words=False)
        printed.numbers += [Source(item.value, item.unit) for item in found.numbers]
        printed.times |= {item.clock for item in found.times}
        printed.dates |= {(item.weekday, item.day, item.month) for item in found.dates}
        printed.sections |= set(found.sections)
        printed.places += [(item.lat, item.lon) for item in found.places]
        printed.places += _decimal_pairs(coordinates_in(text))
        printed.ids |= set(NOTAM_ID_RE.findall(text))
        printed.placeholders |= set(_PLACEHOLDER.findall(text))
        _from_artifact(printed, result.artifact, turn.tz)
    printed.dates |= {_date_key(day) for day in turn.calendar}
    if turn.lz is not None:
        printed.places.append((_dec(turn.lz[0]), _dec(turn.lz[1])))
    return printed


# --- (a) verdict words --------------------------------------------------------------------------------------------------

_PLACEHOLDER = re.compile(r"<PHONE_\d+>")


def _display_pattern(display: str) -> re.Pattern[str]:
    """A display string as a pattern: any case, any dash, the spacing around the dash and the middle dot free, NO-GO also written with a space."""
    parts = []
    for piece in re.split(r"(\s*[-\u00b7]\s*)", _canon(display)):
        if piece.strip() in ("-", "\u00b7"):
            parts.append(r"\s*" + re.escape(piece.strip()) + r"\s*")
        else:
            parts.append(r"\s+".join(re.escape(word) for word in piece.split()))
    pattern = rf"NO[\s-]+{BARE_VERDICT_WORD}" if display == "NO-GO" else "".join(parts)
    return re.compile(rf"(?<![\w-]){pattern}(?![\w-])", re.IGNORECASE)


DISPLAYS = tuple(sorted({display_verdict(v, a, c) for v, a, c in product(Verdict, (False, True), (False, True))}, key=len, reverse=True))
_DISPLAY_PATTERNS = {display: _display_pattern(display) for display in DISPLAYS}
_BARE_GO = re.compile(rf"(?<![\w-]){re.escape(BARE_VERDICT_WORD)}(?![\w-])")
_BANNED = {phrase: re.compile(rf"(?<!\w){r'\s+'.join(map(re.escape, phrase.split()))}(?!\w)", re.IGNORECASE) for phrase in BANNED_PHRASES}


def verdicts_in(text: str) -> list[str]:
    """The display strings `text` writes, in order, the longest first at each place (so PILOT DISCRETION inside a longer one is not counted again)."""
    found: list[tuple[int, str]] = []
    left = _canon(text)
    for display in DISPLAYS:
        for match in _DISPLAY_PATTERNS[display].finditer(left):
            found.append((match.start(), display))
        left = _DISPLAY_PATTERNS[display].sub(lambda m: _blank(m.group()), left)
    return [display for _, display in sorted(found)]


class HourKey(NamedTuple):
    """A whole hour of the assessment as a pilot writes it: the date (day, month) and the clock, in the thread's zone."""

    day: int
    month: str
    clock: str


@dataclass(frozen=True)
class Hours:
    """The hours `assess` judged: those with a verdict, and those with none (OUTLOOK or too far out)."""

    verdict: frozenset[HourKey]
    none: frozenset[HourKey]


def _hour_key(moment: datetime, tz: str) -> HourKey:
    local = moment.astimezone(ZoneInfo(tz))
    return HourKey(local.day, calendar.month_abbr[local.month].lower(), clock(moment, tz))


def _hours(turn: TurnSources) -> Hours:
    """Every hour of every assessment, by tier. A block (a run of OUTLOOK or too-far hours) is listed hour by hour from its start."""
    verdict, none = set(), set()
    for artifact in turn.assessments:
        for segment in artifact.segments:
            moment = _instant(segment.start)
            if moment is not None and moment.tzinfo is not None:
                (verdict if segment.tier == "VERDICT" else none).add(_hour_key(moment, turn.tz))
        for block in artifact.blocks:
            moment = _instant(block.start)
            if moment is not None and moment.tzinfo is not None:
                none.update(_hour_key(moment + n * ONE_HOUR, turn.tz) for n in range(block.hours))
    return Hours(frozenset(verdict), frozenset(none))


def _verdict_findings(body: str, turn: TurnSources, hours: Hours) -> list[GuardFinding]:
    findings = []
    written = verdicts_in(body)
    if written:
        shown = ", ".join(f"'{display}'" for display in dict.fromkeys(written))
        if not turn.assess_ran:
            findings.append(GuardFinding("a", "a verdict was given but assess did not run this turn"))
        elif turn.overall is None:
            findings.append(GuardFinding("a", f"{shown} is a verdict word but assess gave no overall verdict this turn (every hour is OUTLOOK or too far out)"))
        else:
            findings += [GuardFinding("a", f"'{d}' is not a verdict assess returned this turn") for d in dict.fromkeys(written) if d not in turn.displays]
            first = next(iter(_sentences(body)), "")
            written_first = verdicts_in(first)
            named = [d for d in written_first if d in turn.displays]  # one that assess did not return is already a finding above
            if named and turn.overall not in named:
                findings.append(GuardFinding("a", f"the first sentence says '{named[0]}' but the overall verdict assess gave is '{turn.overall}'"))
            elif not written_first:  # the answer gives a verdict somewhere, so it must lead with the overall one
                findings.append(GuardFinding("a", f"the first sentence names no verdict but the overall verdict assess gave is '{turn.overall}'"))
        if turn.overall is not None:  # with no overall at all, the finding above already covers every verdict word
            for sentence in _sentences(body):
                if verdicts_in(sentence) and _about_outlook(sentence, hours):
                    findings.append(GuardFinding("a", f"'{verdicts_in(sentence)[0]}' is attached to an OUTLOOK hour, which has no verdict"))
    return findings


def _sentences(text: str) -> list[str]:
    return [text[first:last].strip() for first, last in _sentence_bounds(text)]


def _about_outlook(sentence: str, hours: Hours) -> bool:
    """Whether the sentence says OUTLOOK, or names an hour with no verdict.

    A time with a date is the hour on that date. A time with no date is an hour with no verdict only when no hour with a verdict is at that clock time
    (otherwise the pilot may mean either, and the guard does not guess against the answer).
    """
    if re.search(r"\boutlook\b", sentence, re.IGNORECASE):
        return True
    found = scan(sentence)
    days = {(item.day, item.month) for item in found.dates}
    for item in found.times:
        if days:
            if any(HourKey(day, month, item.clock) in hours.none for day, month in days):
                return True
        elif item.clock in {key.clock for key in hours.none} - {key.clock for key in hours.verdict}:
            return True
    return False


def _banned_findings(body: str) -> list[GuardFinding]:
    findings = []
    for display in DISPLAYS:  # a verdict string is not the bare word, whatever its spacing
        body = _DISPLAY_PATTERNS[display].sub(lambda m: _blank(m.group()), body)
    if _BARE_GO.search(body):
        findings.append(GuardFinding("a", f"'{BARE_VERDICT_WORD}' on its own is a banned word on every path"))
    findings += [GuardFinding("a", f"'{phrase}' is a banned phrase on every path") for phrase, pattern in _BANNED.items() if pattern.search(body)]
    return findings


# --- (b) the limit phrase (I10) -------------------------------------------------------------------------------------

# (Its two edges are "not next to a letter or digit" rather than \b, so _italic_ markup around the phrase does not hide it.)
# The one place the guard reads a value's position against its limit: this fixed phrase, and nothing else. Any other wording of a position is a known gap (code
# states every limit position in its own lines). `position_word` in contracts decides what the right word is; the phrase's tool-side twin in PR-11 uses it too.
_LIMIT_PHRASE = re.compile(
    rf"(?<![^\W_])(?P<word>over|above|under|below|at)\s+the\s+(?P<number>(?:\d{{1,3}}(?:,\d{{3}})+|\d+)(?P<frac>\.\d+)?)\s?(?P<unit>{_UNIT})(?![\w])\s+limit(?![^\W_])",
    re.IGNORECASE,
)
# Directly before the phrase: not, never, or a word ending in n't (isn't, doesn't). Emphasis markers (* _ ~) may sit on either side of the gap or around the word
# (**not** over, _never_ above, isn't **over ...**): they are markup, the words are still directly before the phrase. `_` is a word character, so the start of the
# negation is "not after a letter or digit" rather than \b.
_NEGATED = re.compile(r"(?<![^\W_])(?:not|never|[^\W_]+n't)[*_~]*\s+[*_~]*\Z", re.IGNORECASE)
_SAME_WORD = {"above": "over", "below": "under"}


def _names_limit(number: Decimal, unit: str | None, places: int, limit: Limit) -> bool:
    """Whether the number the phrase gives is this limit: the same number in the limit's own unit, or (another unit of the table) what the limit rounds to."""
    if unit == limit.unit:
        return abs(number - limit.value) <= EPSILON
    own = _convert(limit.value, limit.unit, unit)
    return own is not None and _rounds_to(own, number, places)


def _held_by(source: Source, numbers: Iterable[Quantity]) -> bool:
    """Whether some number written in the sentence is a rounding of this printed value (so the sentence is about this value)."""
    for written in numbers:
        shown = _convert(source.value, source.unit, written.unit)
        if shown is not None and _rounds_to(shown, written.value, written.places):
            return True
    return False


def _positions(number: Decimal, unit: str | None, places: int, sources: Iterable[Source], sentence: list[Quantity]) -> list[tuple[Source, Limit, str]]:
    """(value, limit, position word) for every printed value that has the phrase's limit. Values the sentence itself gives are the ones meant; with none given, all."""
    found = []
    for source in sources:
        for limit in source.limits:
            printed = _convert(source.value, source.unit, limit.unit)
            if source.is_limit or printed is None or limit.comparison not in _TRIGGERED or not _names_limit(number, unit, places, limit):
                continue
            printed = limit.value if abs(printed - limit.value) <= EPSILON else printed  # equal apart from the noise of a unit change
            found.append((source, limit, position_word(printed, limit.value, limit.comparison)))
    named = [item for item in found if _held_by(item[0], sentence)]
    return named or found


def _limit_phrase_findings(body: str, printed: Printed) -> tuple[str, list[GuardFinding]]:
    """The findings for every limit phrase in `body`, and `body` with each phrase's number and unit blanked (they are the limit, not a value: the rounding check
    must not read them).

    The number must be a limit a tool printed. The phrase may not follow not / n't / never. The position word must be the one `position_word` gives for each
    printed value that has that limit; when the sentence writes a value itself, those are the values it means, otherwise every value with the limit must agree.
    """
    matches = list(_LIMIT_PHRASE.finditer(body))
    bare = body
    for match in matches:
        bare = bare[: match.start("number")] + _blank(bare[match.start("number") : match.end("unit")]) + bare[match.end("unit") :]
    findings: list[GuardFinding] = []
    spelled = _APOSTROPHES.sub("'", body)  # the same length, so the same places
    sources = [item for item in printed.numbers if item.limits or item.is_limit]
    for match in matches:
        number, unit, places = Decimal(match["number"].replace(",", "")), _unit(match["unit"]), len(match["frac"] or ".") - 1
        phrase = " ".join(match.group().split())
        if not any(_names_limit(number, unit, places, limit) for item in sources for limit in (*item.limits, *([Limit(None, item.value, item.unit)] if item.is_limit else []))):
            findings.append(GuardFinding("b", f"'{phrase}' names a limit that no tool printed this turn"))
        elif _NEGATED.search(spelled[: match.start()]):
            findings.append(GuardFinding("b", f"'{phrase}' follows a negation (not, n't, never); say where the value is with over, at or under and no negation"))
        else:
            first, last = next(((a, b) for a, b in _sentence_bounds(bare) if a <= match.start() < b), (0, len(bare)))
            word = match["word"].lower()
            for source, limit, truth in _positions(number, unit, places, sources, scan(bare[first:last], words=False).numbers):
                if truth != _SAME_WORD.get(word, word):
                    shown = _plain(_convert(source.value, source.unit, limit.unit) or source.value, limit.unit)
                    findings.append(GuardFinding("b", f"'{phrase}' says {word}, but the value the tool printed, {shown}, is {truth} the {_plain(limit.value, limit.unit)} limit"))
                    break
    return bare, findings


# --- (b) numbers, times, dates --------------------------------------------------------------------------------------


def _number_findings(body: str, found: Scan, printed: Printed) -> list[GuardFinding]:
    findings = []
    for written in found.numbers:
        finding = _number_finding(written, printed.numbers)
        if finding is not None:
            findings.append(GuardFinding("b", finding))
    findings += [GuardFinding("b", f"'{item.written}' is not a time any tool printed this turn") for item in found.times if item.clock not in printed.times]
    for item in found.dates:
        if not any(day == item.day and month == item.month and (item.weekday is None or weekday is None or item.weekday == weekday) for weekday, day, month in printed.dates):
            findings.append(GuardFinding("b", f"'{item.written}' is not a date the calendar or any tool lists this turn"))
    findings += [GuardFinding("b", f"'{item}' is not a section number any tool printed this turn") for item in found.sections if item not in printed.sections]
    return findings


# --- (c) ids, placeholders, phones, coordinates ----------------------------------------------------------------------


def _reference_findings(text: str, body: str, found: Scan, printed: Printed, turn: TurnSources) -> list[GuardFinding]:
    findings = []
    for notam_id in dict.fromkeys(NOTAM_ID_RE.findall(body)):
        if notam_id not in printed.ids:
            findings.append(GuardFinding("c", f"NOTAM id '{notam_id}' is not in any tool result this turn"))
    for placeholder in dict.fromkeys(_PLACEHOLDER.findall(body)):
        if placeholder not in turn.phone_keys:
            findings.append(GuardFinding("c", f"'{placeholder}' is not one of this thread's phone placeholders"))
        elif placeholder not in printed.placeholders:
            findings.append(GuardFinding("c", f"'{placeholder}' is not in any tool result this turn"))
    _, phones = redact_phones(text)
    findings += [GuardFinding("c", f"'{number}' looks like a real phone number, which only a placeholder may stand for") for number in dict.fromkeys(phones.values())]
    for place in found.places:
        if not any(_rounds_to(lat, place.lat, place.lat_places) and _rounds_to(lon, place.lon, place.lon_places) for lat, lon in printed.places):
            findings.append(GuardFinding("c", f"'{place.written}' is not a place any tool printed this turn"))
    for written, point in coordinates_each(body.upper()):  # degrees, minutes, seconds, as a NOTAM writes them (the scan above reads decimals only)
        if point is None:
            findings.append(GuardFinding("c", f"'{written}' is not a possible place, so no tool printed it"))
        elif not any(abs(_dec(point[0]) - p_lat) <= DMS_TOLERANCE and abs(_dec(point[1]) - p_lon) <= DMS_TOLERANCE for p_lat, p_lon in printed.places):
            findings.append(GuardFinding("c", f"'{written}' is not a place any tool printed this turn"))
    return findings


# --- (d) the data is final ---------------------------------------------------------------------------------------------

# A closed list (I10): the guard finds a sentence that says the data is final or needs no re-check only by these exact phrases. It does not read meaning, so
# any other wording is a known gap that the re-check line code appends is there to cover. Matched ignoring case, at word boundaries, with every kind of dash
# and apostrophe made one. The "won't change" phrases count only at the end of a sentence or before "again", so "The forecast won't change your verdict much"
# passes.
FINAL_PHRASES = (
    "the data is final",
    "the forecast is final",
    "the NOTAMs are final",
    "the forecast is now final",
    "this forecast is definitely final",
    "this is the final forecast",
    "the data won't change",
    "the forecast won't change",
    "the NOTAMs won't change",
    "no need to check again",
    "no need to re-check",
    "no need to recheck",
    "no need to check the forecast again",
    "no need to check the NOTAMs again",
    "you don't need to check the NOTAMs again",
    "no need to double check",
    "no need to double-check",
    "you don't need to double check the NOTAMs",
    "no further checks are needed",
    "no further re-check is needed",
    "you can skip the re-check",
    "you needn't bother checking again",
)
_APOSTROPHES = re.compile("[\u2018\u2019\u02bc\u2032]")
_SENTENCE_END_OR_AGAIN = r"(?=[\s\"')\]]*\Z|\s+again\b)"


def _phrase_pattern(phrase: str) -> re.Pattern[str]:
    words = r"\s+".join(re.escape(word) for word in phrase.split())
    tail = _SENTENCE_END_OR_AGAIN if phrase.endswith("won't change") else ""
    return re.compile(rf"(?<![\w'-]){words}(?![\w'-]){tail}", re.IGNORECASE)


_FINAL_PATTERNS = tuple(_phrase_pattern(phrase) for phrase in FINAL_PHRASES)


def _says_final(sentence: str) -> bool:
    """Whether the sentence holds one of FINAL_PHRASES."""
    sentence = _APOSTROPHES.sub("'", sentence)
    return any(pattern.search(sentence) for pattern in _FINAL_PATTERNS)


def _final_findings(body: str) -> list[GuardFinding]:
    return [GuardFinding("d", f"'{sentence}' says the data is final, current enough or needs no re-check") for sentence in _sentences(body) if _says_final(sentence)]


# --- the entry point ---------------------------------------------------------------------------------------------------


def guard_answer(answer: str, turn: TurnSources) -> list[GuardFinding]:
    """The findings for one final AI message, checked against this turn's tool results only; [] when it passes. Never raises on text, however odd."""
    text = _canon(answer)
    body = _without_excerpts(text, _excerpt_corpus(turn))
    printed = _printed(turn)
    body_without_phones, _ = redact_phones(body)  # the digits of a real-looking phone are the phone finding, not four more numbers
    bare, phrase_findings = _limit_phrase_findings(body_without_phones, printed)  # a limit phrase's number is the limit, checked there and not as a value
    found = scan(bare)
    findings = [
        *_verdict_findings(body, turn, _hours(turn)),
        *_banned_findings(body),
        *phrase_findings,
        *_number_findings(bare, found, printed),
        *_reference_findings(text, body, found, printed, turn),
        *_final_findings(body),
    ]
    return list(dict.fromkeys(findings))
