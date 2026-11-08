"""Minute expansion (5a PR-5.F1, G9, E12, E14): rows and segments as UTC minutes, the one unit every time-based evaluator counts in.

    aware(value, what)                       a datetime (or ISO text) as an aware datetime; a naive one raises NaiveDatetimeError
    minute_span(start, end, what)            the minutes of the half-open span [start, end), as a `range` of minute numbers
    window_of(window, what)                  the minutes of a `[start, end]` example window
    expand(items, span)                      {minute: (verdict, approaching_limits)} for rows or segments, clipped to `span`
    excluded_minutes(rows, span)             the minutes of the rows that carry a `tier` (OUTLOOK / TOO_FAR), clipped to `span`
    runs(minutes)                            the contiguous runs of a set of minutes, in time order
    range_text(run)                          "2026-10-31 17:06–17:20Z" for a run
    WORST_FIRST                              NO_GO, UNVERIFIED, PILOT_DISCRETION

A minute is a whole number of minutes since 1970-01-01 00:00 UTC. A span [start, end) holds the minute starts that are at or after `start` and before `end`, so
a row ending at 17:06 and a row starting at 17:06 never share a minute, and a span that starts or ends between two minutes holds the minutes whose start falls in it.

An item is a mapping with `start` and `end` (aware datetimes, or ISO text with `Z` or an offset) and either `tier` (a row of the verdict sheet that has no
verdict) or `verdict` and, optionally, `approaching_limits` (a sheet row or an answer segment; `edge`, `unread` and any other key are ignored). The same
rule serves the sheet and the answer:
  * clipping: only the minutes inside the span are kept, so a row that straddles the window edge gives its in-window part and a row wholly outside gives none (E14);
  * overlap: two items on one minute give the worst verdict (NO_GO, then UNVERIFIED, then PILOT_DISCRETION); `approaching_limits` is then true when any item
    carrying that worst verdict says so;
  * gaps: a minute no item covers is simply absent, so the evaluator that looks it up sees "no answer" and counts a disagreement.

A datetime with no timezone is refused with NaiveDatetimeError, whose message names what it was for: the evaluators never guess a zone. `Z` and `+01:00` inputs that
name the same instant give identical minutes.
"""

from collections.abc import Iterable, Mapping
from datetime import UTC, datetime, timedelta
from typing import Any

from flysafelz.contracts import Verdict

MINUTE = timedelta(minutes=1)
EPOCH = datetime(1970, 1, 1, tzinfo=UTC)
# Worst first. Verdict's own docstring gives the same order; test_minutes.py checks the two agree.
WORST_FIRST: tuple[Verdict, ...] = (Verdict.NO_GO, Verdict.UNVERIFIED, Verdict.PILOT_DISCRETION)

type Minutes = dict[int, tuple[Verdict, bool]]


class NaiveDatetimeError(ValueError):
    """A time with no timezone was given. The message names what it was for."""


def aware(value: object, what: str) -> datetime:
    """`value` as an aware datetime. Text is read as ISO 8601 (`Z` or an offset). Raises NaiveDatetimeError for a time with no zone, ValueError for anything else."""
    moment = value
    if isinstance(value, str):
        try:
            moment = datetime.fromisoformat(value.strip())
        except ValueError:
            raise ValueError(f"{what}: {value!r} is not an ISO 8601 time") from None
    if not isinstance(moment, datetime):
        raise ValueError(f"{what}: {value!r} is not a time")  # noqa: TRY004  # every bad input is a ValueError, so one except clause serves
    if moment.tzinfo is None or moment.utcoffset() is None:
        raise NaiveDatetimeError(f"{what}: {value!r} has no timezone; write it with Z or an offset such as +01:00")
    return moment


def _ceil_minute(moment: datetime) -> int:
    """The number of the first minute start at or after `moment`."""
    return -((EPOCH - moment) // MINUTE)


def minute_span(start: object, end: object, what: str = "span") -> range:
    """The minutes of [start, end): every minute start at or after `start` and before `end`. Empty when end is not after start."""
    first, last = _ceil_minute(aware(start, f"{what} start")), _ceil_minute(aware(end, f"{what} end"))
    return range(first, max(first, last))


def window_of(window: object, what: str = "window") -> range:
    """The minutes of an example's `[start, end]` window. Raises ValueError for anything but two times, or an end before the start."""
    try:
        first, last = window  # type: ignore[misc]  # anything that is not a pair fails here
    except (TypeError, ValueError):
        raise ValueError(f"{what} must be a list of a start and an end time, not {window!r}") from None
    start, end = aware(first, f"{what} start"), aware(last, f"{what} end")
    if end < start:
        raise ValueError(f"{what} ends at {last!r}, before it starts at {first!r}")
    return minute_span(start, end, what)


def minute_time(minute: int) -> datetime:
    """The UTC time a minute number starts at."""
    return EPOCH + minute * MINUTE


def _field(item: Any, name: str) -> Any:
    return item.get(name) if isinstance(item, Mapping) else getattr(item, name, None)


def _clip(span: range, inside: range) -> range:
    first, last = max(span.start, inside.start), min(span.stop, inside.stop)
    return range(first, max(first, last))


def _verdict(value: object, what: str) -> Verdict:
    try:
        return Verdict(value)
    except ValueError:
        raise ValueError(f"{what}: {value!r} is not one of {', '.join(v.value for v in Verdict)}") from None


def expand(items: Iterable[Any], span: range, what: str = "item") -> Minutes:
    """{minute: (verdict, approaching_limits)} for every item that has a verdict, clipped to `span`. Items with a `tier` or no verdict add nothing. See the module text."""
    found: Minutes = {}
    for number, item in enumerate(items, start=1):
        if _field(item, "tier") is not None or _field(item, "verdict") is None:
            continue
        label = f"{what} {number}"
        verdict = _verdict(_field(item, "verdict"), label)
        limits = bool(_field(item, "approaching_limits"))
        for minute in _clip(minute_span(_field(item, "start"), _field(item, "end"), label), span):
            held = found.get(minute)
            if held is None or WORST_FIRST.index(verdict) < WORST_FIRST.index(held[0]):
                found[minute] = (verdict, limits)
            elif verdict is held[0]:
                found[minute] = (verdict, held[1] or limits)
    return found


def excluded_minutes(rows: Iterable[Any], span: range, what: str = "tier row") -> set[int]:
    """The minutes inside `span` that a row with a `tier` covers (OUTLOOK or TOO_FAR rows carry no verdict and are left out of the verdict scores)."""
    left: set[int] = set()
    for number, row in enumerate(rows, start=1):
        if _field(row, "tier") is not None:
            left.update(_clip(minute_span(_field(row, "start"), _field(row, "end"), f"{what} {number}"), span))
    return left


def runs(minutes: Iterable[int]) -> list[range]:
    """The contiguous runs of a set of minutes, in time order: {5, 6, 7, 9} gives [range(5, 8), range(9, 10)]."""
    found: list[range] = []
    for minute in sorted(set(minutes)):
        if found and found[-1].stop == minute:
            found[-1] = range(found[-1].start, minute + 1)
        else:
            found.append(range(minute, minute + 1))
    return found


def range_text(run: range) -> str:
    """A run as UTC text: "2026-10-31 17:06–17:20Z" (the end is the first minute not in the run), or with both dates when it crosses midnight."""
    start, end = minute_time(run.start), minute_time(run.stop)
    if start.date() == end.date():
        return f"{start:%Y-%m-%d %H:%M}–{end:%H:%M}Z"
    return f"{start:%Y-%m-%d %H:%M}Z–{end:%Y-%m-%d %H:%M}Z"
