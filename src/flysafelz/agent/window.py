"""The window a tool assesses: the model's local `start` and `end`, read in the thread's time zone and turned into UTC.

    parse_window(start, end, *, now, tz)   -> Window | WindowError
    local_label(moment, tz)                -> "10:00 GMT" or "10:00 BST"

The model gives a window as two local times written `YYYY-MM-DD HH:MM`, or leaves both out. It never gives a time zone or a
place: `tz` and `now` are keyword arguments the caller reads from the thread (`thread_lz["tz"]`) and the run context
(`run_now(ctx)`), so no value of either can come from the model.

Rules, in the order they are applied:

    both missing        the default window: from now rounded down to the hour (UTC, as ticket 2's `default_window` does) to the end of
                        tomorrow in `tz`. `words` starts "assumed window: " so the answer can state what was assumed.
    only one given      WindowError
    malformed value     WindowError (a wrong shape, or a shape that is no real date and time)
    a time that does not exist   WindowError. The hour the clocks skip (01:30 on 29 Mar 2026 in London) names no instant.
    a time that exists twice     the first occurrence (`fold=0`): 01:30 on 25 Oct 2026 is 01:30 BST, not 01:30 GMT.
    end not after start WindowError
    end not after now   WindowError "That time has passed."
    start at or after `fetch_horizon(now)`   WindowError "too far out to assess"; nothing is fetched for it.
    a window that starts inside the horizon and ends beyond it is returned as it is: `assess` marks the hours past the horizon TOO_FAR.

Only the window and the words that describe it are decided here. The day tiers (which hours get a verdict, which an outlook) stay
ticket 2's Europe/London days whatever `tz` is, so `fetch_horizon` is the same for every zone.

Every time in `words` is written on the clock of `tz` with the abbreviation of its own instant (GMT before the clocks change, BST
after), and an end at local midnight is written as 24:00 of the day it closes.
"""

import re
from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta
from zoneinfo import ZoneInfo

from flysafelz.assess import fetch_horizon
from flysafelz.timetext import EN_DASH

ASSUMED_PREFIX = "assumed window: "
TIME_FORMAT = "%Y-%m-%d %H:%M"
_SHAPE = re.compile(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}")
END_OF_DAY = "24:00"
ONE_DAY = timedelta(days=1)

ONLY_ONE_GIVEN = "Give both start and end, or neither to use the default window."
NOT_AFTER_START = "The end must be after the start."
HAS_PASSED = "That time has passed: the window ended before now."
TOO_FAR_OUT = "That window is too far out to assess: it starts beyond the forecast horizon."


@dataclass(frozen=True)
class Window:
    """A window to assess: UTC instants [start_utc, end_utc), the words that describe it in the pilot's zone, and whether code assumed it."""

    start_utc: datetime
    end_utc: datetime
    words: str
    assumed: bool


@dataclass(frozen=True)
class WindowError:
    """Why the window cannot be assessed: one sentence the model relays to the pilot."""

    text: str


def _aware(moment: datetime, name: str) -> datetime:
    if moment.tzinfo is None or moment.utcoffset() is None:
        raise ValueError(f"{name} must be a timezone-aware datetime")
    return moment


def local_label(moment: datetime, tz: str) -> str:
    """The time on the clock of `tz` with the abbreviation of that instant: "10:00 GMT" before the clocks change, "10:00 BST" after."""
    local = _aware(moment, "moment").astimezone(ZoneInfo(tz))
    return f"{local:%H:%M} {local.tzname()}"


def _missing(value: str | None) -> bool:
    """A time left out: not given, or an empty string (a model sends one for "none")."""
    return value is None or not value.strip()


def _parse_local(name: str, text: str, zone: ZoneInfo, tz: str) -> datetime | WindowError:
    """One local time as a UTC instant, or the WindowError for it. `name` is "start" or "end"."""
    written = text.strip()
    if not _SHAPE.fullmatch(written):
        return WindowError(f"The {name} must be a local time written YYYY-MM-DD HH:MM, for example 2026-10-31 10:00.")
    try:
        naive = datetime.strptime(written, TIME_FORMAT)
    except ValueError:
        return WindowError(f"The {name} is not a real date and time; write it as YYYY-MM-DD HH:MM.")
    instant = naive.replace(tzinfo=zone).astimezone(UTC)  # fold=0: a time that happens twice is its first occurrence
    if instant.astimezone(zone).replace(tzinfo=None) != naive:
        # The wall time came back as another one, so no instant has it: it falls in the hour the clocks skip.
        return WindowError(f"The {name} {written} does not exist in {tz}: the clocks go forward then, so that hour is skipped.")
    return instant


def _day_text(day: date) -> str:
    return f"{day:%A} {day.day} {day:%b}"


def _words(start: datetime, end: datetime, tz: str) -> str:
    """The window in words, like Saturday 31 Oct 10:00 – 16:00 GMT. The date shows on the end only when it is another day; the abbreviation shows once when both ends share it."""
    zone = ZoneInfo(tz)
    first, last = start.astimezone(zone), end.astimezone(zone)
    last_day, last_clock = last.date(), f"{last:%H:%M}"
    if last.time() == time(0, 0):
        last_day, last_clock = last_day - ONE_DAY, END_OF_DAY  # an end at midnight closes the day before
    left = f"{_day_text(first.date())} {first:%H:%M}"
    right = last_clock if last_day == first.date() else f"{_day_text(last_day)} {last_clock}"
    if first.tzname() == last.tzname():
        return f"{left} {EN_DASH} {right} {last.tzname()}"
    return f"{left} {first.tzname()} {EN_DASH} {right} {last.tzname()}"


def _default_window(now: datetime, tz: str) -> Window:
    """From now rounded down to the hour to the end of tomorrow in `tz`."""
    zone = ZoneInfo(tz)
    start = now.astimezone(UTC).replace(minute=0, second=0, microsecond=0)
    tomorrow = now.astimezone(zone).date() + ONE_DAY
    end = datetime.combine(tomorrow + ONE_DAY, time(0, 0), tzinfo=zone).astimezone(UTC)  # midnight ending tomorrow
    return Window(start, end, ASSUMED_PREFIX + _words(start, end, tz), assumed=True)


def parse_window(start: str | None, end: str | None, *, now: datetime, tz: str) -> Window | WindowError:
    """The window the model asked for, or the default one, or the reason it cannot be assessed.

    `start` and `end` are the model's local times (`YYYY-MM-DD HH:MM` in `tz`) or None. `now` is the run's clock and `tz` the thread's
    IANA zone name, both from the caller's state and context. A `tz` that is not a zone raises (the input check has already refused it).
    """
    now = _aware(now, "now")
    zone = ZoneInfo(tz)
    if _missing(start) and _missing(end):
        return _default_window(now, tz)
    if _missing(start) or _missing(end):
        return WindowError(ONLY_ONE_GIVEN)
    first = _parse_local("start", str(start), zone, tz)
    if isinstance(first, WindowError):
        return first
    last = _parse_local("end", str(end), zone, tz)
    if isinstance(last, WindowError):
        return last
    if last <= first:
        return WindowError(NOT_AFTER_START)
    if last <= now:
        return WindowError(HAS_PASSED)
    if first >= fetch_horizon(now):
        return WindowError(TOO_FAR_OUT)
    return Window(first, last, _words(first, last, tz), assumed=False)
