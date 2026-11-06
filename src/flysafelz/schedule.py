"""Turn a NOTAM item D (the daily schedule text) into UTC intervals inside B-C.

    parse_schedule(d_text, b, c, lat, lon, horizon_end=None) -> Schedule

All times in a NOTAM are UTC, and so are the days here. The intervals are
clipped to B-C, sorted and merged. Zero intervals is a valid answer: the
NOTAM is not active in that window.

Fail closed:
- text that cannot be read gives [(B, C)] with parsed=False and a note that quotes it;
- empty text means "no schedule": active for all of B-C, parsed=True;
- SR and SS come from astral at the LZ for each day, and the active span rounds
  outward (a start rounds down to the minute, an end rounds up), so the span is
  never shorter than published.

Formats read (each comma-separated clause can carry different days):
    0800-1700                      a time pair; several pairs in one clause are allowed
    2300-0600                      an end at or before the start runs into the next day
    H24, DAILY                     the whole day
    SR-SS, SR-30 TO SS+30          sun times with minute offsets
    MON-FRI, TUE WED THU SAT       weekday ranges and sets (an overnight span belongs to its start day)
    ... EXC SAT SUN                weekdays left out
    OCT 26 27 NOV 02 0900-2030     date lists; a month carries over to the bare days after it, even across commas
    SEP 29-OCT 05, OCT 25-JAN 01   ranges across months and into the next year
    27 0900-1600, 28 ...           bare days: any date in B-C with that day of the month
    1500-1600, 1900-2000           a clause with only times adds to the clause before it
Spaces and newlines are interchangeable, spacing around "-" is free, and empty clauses are ignored.
"""

import re
from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta

from flysafelz.daylight import round_down_minute, round_up_minute, sunrise_sunset

MINUTES_PER_HOUR = 60
MINUTES_PER_DAY = 24 * MINUTES_PER_HOUR
END_OF_DAY = (24, 0)  # "2400" as an end time
LEAP_YEAR = 2024  # only to check that a month and day exist (29 February is allowed)
ONE_DAY = timedelta(days=1)

MONTHS = ("JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC")
WEEKDAYS = ("MON", "TUE", "WED", "THU", "FRI", "SAT", "SUN")  # date.weekday() order

_SUN_EDGE = r"(?:SR|SS)(?:[+-]\d{1,3}(?!\d))?"
_EDGE = rf"(?:\d{{4}}|{_SUN_EDGE})"
_TOKEN = re.compile(
    rf"""
      (?P<span>(?P<a>{_EDGE})\s*(?:-|TO)\s*(?P<b>{_EDGE})(?!\w))
    | (?P<h24>H24\b)
    | (?P<daily>DAILY\b)
    | (?P<exc>EXC\b)
    | (?P<month>(?:{"|".join(MONTHS)})\b)
    | (?P<weekday>(?:{"|".join(WEEKDAYS)})\b)
    | (?P<num>\d{{1,2}}(?!\d))
    | (?P<dash>-)
    | (?P<comma>,)
    | (?P<space>\s+)
    """,
    re.VERBOSE,
)
_SUN_PARTS = re.compile(r"(?P<anchor>SR|SS)(?P<offset>[+-]\d+)?")

Token = tuple[str, str]  # (kind, text); a time span is ("span", "<start>|<end>")


class ScheduleError(ValueError):
    """The text is not a schedule this parser reads."""


@dataclass(frozen=True)
class Schedule:
    intervals: list[tuple[datetime, datetime]]  # UTC, inside B-C, sorted, merged
    parsed: bool  # False: the text was not understood and intervals is all of B-C
    note: str | None


@dataclass(frozen=True)
class _Edge:
    sun: str | None  # "SR", "SS", or None for a clock time
    minutes: int  # minutes after midnight (clock time) or after the sun event (sun time)


Span = tuple[_Edge, _Edge]
ALL_DAY: Span = (_Edge(None, 0), _Edge(None, MINUTES_PER_DAY))
_MonthDay = tuple[int | None, int]  # (month, or None for a bare day; day of month)
_DayRange = tuple[_MonthDay, _MonthDay]


@dataclass
class _Group:
    """Time spans and the days they apply to."""

    ranges: list[_DayRange]  # empty: any date
    weekdays: set[int] | None  # None: any weekday
    excluded: set[int]
    spans: list[Span]


@dataclass
class _Clause:
    """One comma-separated part of the text."""

    ranges: list[_DayRange]
    weekdays: set[int] | None
    excluded: set[int]
    spans: list[Span]
    has_days: bool  # it names dates, weekdays or DAILY


# --- reading the text -------------------------------------------------------


def _tokens(text: str) -> list[Token]:
    found: list[Token] = []
    pos = 0
    while pos < len(text):
        match = _TOKEN.match(text, pos)
        if match is None:
            raise ScheduleError(f"cannot read {text[pos:].split(maxsplit=1)[0]!r}")
        pos = match.end()
        kind = match.lastgroup
        if kind == "span":
            found.append((kind, f"{match.group('a')}|{match.group('b')}"))
        elif kind != "space":
            found.append((kind or "", match.group()))
    return found


def _parse_edge(text: str) -> _Edge:
    sun = _SUN_PARTS.fullmatch(text)
    if sun:
        return _Edge(sun.group("anchor"), int(sun.group("offset") or 0))
    hour, minute = divmod(int(text), 100)
    if (hour, minute) != END_OF_DAY:
        try:
            time(hour, minute)
        except ValueError as exc:
            raise ScheduleError(f"{text} is not a time of day") from exc
    return _Edge(None, hour * MINUTES_PER_HOUR + minute)


def _month_day(month: int | None, day: int) -> _MonthDay:
    try:
        date(LEAP_YEAR, month or 1, day)
    except ValueError as exc:
        raise ScheduleError(f"no such date: month {month}, day {day}") from exc
    return month, day


def _weekday_run(tokens: list[Token], i: int) -> tuple[set[int], int]:
    """One weekday, or a range such as MON-FRI, starting at tokens[i]. Returns the weekday numbers and the next index."""
    first = WEEKDAYS.index(tokens[i][1])
    i += 1
    if i < len(tokens) and tokens[i][0] == "dash":
        i += 1
        if i >= len(tokens) or tokens[i][0] != "weekday":
            raise ScheduleError("a weekday range needs a weekday after the dash")
        last = WEEKDAYS.index(tokens[i][1])
        i += 1
        length = (last - first) % len(WEEKDAYS) + 1
        return {(first + step) % len(WEEKDAYS) for step in range(length)}, i
    return {first}, i


def _date_run(tokens: list[Token], i: int, month: int | None) -> tuple[_DayRange, int | None, int]:
    """One day or day range starting at tokens[i] (a number). Returns the range, the month in force and the next index."""
    first = _month_day(month, int(tokens[i][1]))
    last = first
    i += 1
    if i < len(tokens) and tokens[i][0] == "dash":
        i += 1
        if i < len(tokens) and tokens[i][0] == "month":
            if first[0] is None:
                raise ScheduleError("a date range that ends in a named month needs a month before its first day")
            month = MONTHS.index(tokens[i][1]) + 1
            i += 1
        if i >= len(tokens) or tokens[i][0] != "num":
            raise ScheduleError("a date range needs a day after the dash")
        last = _month_day(month, int(tokens[i][1]))
        i += 1
        if last[0] == first[0] and last[1] < first[1]:
            raise ScheduleError("a range inside one month must not run backwards")
    return (first, last), month, i


def _parse_clause(tokens: list[Token], month: int | None) -> tuple[_Clause, int | None]:
    """Read one clause. `month` is the month in force from earlier text; the new one is returned."""
    clause = _Clause(ranges=[], weekdays=None, excluded=set(), spans=[], has_days=False)
    i = 0
    while i < len(tokens):
        kind, value = tokens[i]
        if kind == "span":
            start, stop = value.split("|")
            clause.spans.append((_parse_edge(start), _parse_edge(stop)))
            i += 1
        elif kind == "h24":
            clause.spans.append(ALL_DAY)
            i += 1
        elif kind == "daily":
            clause.has_days = True
            i += 1
        elif kind == "weekday":
            days, i = _weekday_run(tokens, i)
            clause.weekdays = (clause.weekdays or set()) | days
            clause.has_days = True
        elif kind == "exc":
            i += 1
            if i >= len(tokens) or tokens[i][0] != "weekday":
                raise ScheduleError("EXC needs weekdays after it")
            while i < len(tokens) and tokens[i][0] == "weekday":
                days, i = _weekday_run(tokens, i)
                clause.excluded |= days
        elif kind == "month":
            month = MONTHS.index(value) + 1
            i += 1
            if i >= len(tokens) or tokens[i][0] != "num":
                raise ScheduleError(f"month {value} needs a day after it")
        elif kind == "num":
            day_range, month, i = _date_run(tokens, i, month)
            clause.ranges.append(day_range)
            clause.has_days = True
        else:
            raise ScheduleError(f"unexpected {value!r}")
    return clause, month


def _groups(text: str) -> list[_Group]:
    """The day groups in the text. A clause with only times joins the group before it."""
    clauses: list[list[Token]] = [[]]
    for token in _tokens(text.upper()):
        if token[0] == "comma":
            clauses.append([])
        else:
            clauses[-1].append(token)
    groups: list[_Group] = []
    month: int | None = None
    for tokens in clauses:
        if not tokens:
            continue
        clause, month = _parse_clause(tokens, month)
        spans = clause.spans or [ALL_DAY]
        if not clause.has_days and not clause.excluded and groups:
            groups[-1].spans.extend(spans)
        else:
            groups.append(_Group(clause.ranges, clause.weekdays, clause.excluded, spans))
    return groups


# --- placing it on the calendar ---------------------------------------------


def _in_ranges(ranges: list[_DayRange], day: date) -> bool:
    for (m1, d1), (m2, d2) in ranges:
        if m1 is None:  # bare days: that day of the month, in any month
            if d1 <= day.day <= d2:
                return True
            continue
        here, start, end = (day.month, day.day), (m1, d1), (m2, d2)
        if start <= end:
            if start <= here <= end:
                return True
        elif here >= start or here <= end:  # runs over New Year
            return True
    return False


def _group_applies(group: _Group, day: date) -> bool:
    if group.ranges and not _in_ranges(group.ranges, day):
        return False
    if group.weekdays is not None and day.weekday() not in group.weekdays:
        return False
    return day.weekday() not in group.excluded


def _edge_time(edge: _Edge, day: date, lat: float, lon: float) -> datetime:
    if edge.sun is None:
        return datetime.combine(day, time(), tzinfo=UTC) + timedelta(minutes=edge.minutes)
    rise, set_ = sunrise_sunset(lat, lon, day)
    return (rise if edge.sun == "SR" else set_) + timedelta(minutes=edge.minutes)


def _span_on(span: Span, day: date, lat: float, lon: float) -> tuple[datetime, datetime]:
    """The span starting on `day`. The start rounds down and the end rounds up; an end at or before the start is next day's."""
    start = round_down_minute(_edge_time(span[0], day, lat, lon))
    end = round_up_minute(_edge_time(span[1], day, lat, lon))
    if end <= start:
        end = round_up_minute(_edge_time(span[1], day + ONE_DAY, lat, lon))
    return start, end


def _merge(intervals: list[tuple[datetime, datetime]]) -> list[tuple[datetime, datetime]]:
    merged: list[tuple[datetime, datetime]] = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def _intervals(d_text: str, b: datetime, end: datetime, lat: float, lon: float) -> list[tuple[datetime, datetime]]:
    groups = _groups(d_text)
    found = []
    day = b.date() - ONE_DAY  # the day before B, because an overnight span can run into B
    while day <= end.date():
        for group in groups:
            if _group_applies(group, day):
                for span in group.spans:
                    start, stop = _span_on(span, day, lat, lon)
                    start, stop = max(start, b), min(stop, end)
                    if start < stop:
                        found.append((start, stop))
        day += ONE_DAY
    return _merge(found)


def _utc(t: datetime) -> datetime:
    if t.tzinfo is None:
        raise ValueError(f"datetime must be timezone-aware, got naive {t!r}")
    return t.astimezone(UTC)


def parse_schedule(
    d_text: str | None,
    b: datetime,
    c: datetime | None,
    lat: float,
    lon: float,
    horizon_end: datetime | None = None,
) -> Schedule:
    """The UTC intervals inside B-C in which the NOTAM's item D says it is active.

    c is None for a permanent NOTAM, and then horizon_end must say where to stop
    (ValueError if it is missing). lat and lon place SR and SS.
    """
    end = c if c is not None else horizon_end
    if end is None:
        raise ValueError("c is None (PERM): horizon_end is required")
    b, end = _utc(b), _utc(end)
    whole = [(b, end)] if b < end else []
    if d_text is None or not d_text.strip():
        return Schedule(whole, True, None)
    try:
        return Schedule(_intervals(d_text, b, end, lat, lon), True, None)
    except ValueError as exc:  # ScheduleError, or astral finding no sunrise or sunset that day
        quoted = " ".join(d_text.split())
        return Schedule(whole, False, f'could not read schedule "{quoted}" ({exc}); treated as active for all of B-C')
