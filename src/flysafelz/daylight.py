"""Daylight at the landing zone (LZ) and Europe/London day counting.

Civil dawn, sunrise, sunset and civil dusk come from astral. The civil
twilight angle is the civil_twilight_depression rule. Edges are whole UTC
minutes and are rounded so the caution zone grows (fail closed):

    civil dawn  round UP    (night lasts to the next whole minute)
    sunrise     round UP    (twilight lasts to the next whole minute)
    sunset      round DOWN  (twilight starts at the whole minute before)
    civil dusk  round DOWN  (night starts at the whole minute before)

So Blue Mill on 2026-10-31 (astral dusk 17:06:19) has dusk at 17:06.
Phases are NIGHT, then TWILIGHT from dawn, DAY from sunrise, TWILIGHT from
sunset, NIGHT from dusk. A phase begins at its edge minute exactly.

Where the sun never gets far enough below the horizon for civil twilight to
end (northern Scotland in midsummer) astral raises ValueError for dawn and
dusk. There is no night then, only twilight, so the missing edges are left
out and the dark hours read TWILIGHT. Nothing crashes.
"""

from datetime import UTC, date, datetime, time, timedelta
from enum import StrEnum
from zoneinfo import ZoneInfo

from astral import Observer
from astral.sun import dawn, dusk, sunrise, sunset

from flysafelz.rules import RULES

LONDON = ZoneInfo("Europe/London")
ONE_MINUTE = timedelta(minutes=1)
ONE_HOUR = timedelta(hours=1)
ONE_DAY = timedelta(days=1)


class Phase(StrEnum):
    DAY = "DAY"
    TWILIGHT = "TWILIGHT"
    NIGHT = "NIGHT"


def _utc(t: datetime) -> datetime:
    if t.tzinfo is None:
        raise ValueError(f"datetime must be timezone-aware, got naive {t!r}")
    return t.astimezone(UTC)


def round_up_minute(t: datetime) -> datetime:
    floor = t.replace(second=0, microsecond=0)
    return floor if floor == t else floor + ONE_MINUTE


def round_down_minute(t: datetime) -> datetime:
    return t.replace(second=0, microsecond=0)


def sunrise_sunset(lat: float, lon: float, day: date) -> tuple[datetime, datetime]:
    """Exact (to the second) sunrise and sunset in UTC at the LZ for a UTC date.

    Raises ValueError where the sun does not rise or set that day.
    """
    observer = Observer(lat, lon)
    return sunrise(observer, day, tzinfo=UTC), sunset(observer, day, tzinfo=UTC)


def _day_events(lat: float, lon: float, day: date) -> list[tuple[datetime, Phase]]:
    """The edges astral gives for one date, rounded as in the module doc. Missing edges are left out."""
    observer = Observer(lat, lon)
    depression = RULES.civil_twilight_depression.value
    rounded = (
        (lambda: dawn(observer, day, depression=depression, tzinfo=UTC), round_up_minute, Phase.TWILIGHT),
        (lambda: sunrise(observer, day, tzinfo=UTC), round_up_minute, Phase.DAY),
        (lambda: sunset(observer, day, tzinfo=UTC), round_down_minute, Phase.TWILIGHT),
        (lambda: dusk(observer, day, depression=depression, tzinfo=UTC), round_down_minute, Phase.NIGHT),
    )
    events = []
    for compute, rounding, phase_after in rounded:
        try:
            events.append((rounding(compute()), phase_after))
        except ValueError:
            continue  # the sun never gets that far from the horizon on this date
    return events


def _phase_changes(lat: float, lon: float, first: date, last: date) -> list[tuple[datetime, Phase]]:
    """Every phase change for the UTC dates first..last plus one day either side, in time order.

    The extra days mean a window that crosses UTC midnight sees the events of
    both dates, and an early dawn just after midnight is not lost. Duplicates
    are merged and an event that would not change the phase is dropped.
    """
    events: set[tuple[datetime, Phase]] = set()
    day = first - ONE_DAY
    while day <= last + ONE_DAY:
        events.update(_day_events(lat, lon, day))
        day += ONE_DAY
    changes = []
    current = None
    for t, phase_after in sorted(events):
        if phase_after != current:
            changes.append((t, phase_after))
            current = phase_after
    return changes


def daylight_phase(lat: float, lon: float, t: datetime) -> Phase:
    """DAY, TWILIGHT or NIGHT at the LZ at time t. An edge minute already belongs to the phase after it.

    If no edge can be found at all (not possible inside the UK bounds) the answer is TWILIGHT, never DAY or NIGHT.
    """
    t = _utc(t)
    phase = Phase.TWILIGHT
    for edge, phase_after in _phase_changes(lat, lon, t.date(), t.date()):
        if edge > t:
            break
        phase = phase_after
    return phase


def daylight_edges(lat: float, lon: float, start: datetime, end: datetime) -> list[tuple[datetime, Phase]]:
    """Every phase change strictly inside (start, end), as (UTC minute, phase after), in time order.

    A change exactly at start is not listed: the phase at start already is the phase after it.
    The end is exclusive, like the hour rows.
    """
    start, end = _utc(start), _utc(end)
    return [(t, phase) for t, phase in _phase_changes(lat, lon, start.date(), end.date()) if start < t < end]


# --- Europe/London days (hours stay UTC) -------------------------------------


def london_day_offset(t_utc: datetime, now_utc: datetime) -> int:
    """Whole Europe/London calendar days from now to t. Day 0 is today in London."""
    return (_utc(t_utc).astimezone(LONDON).date() - _utc(now_utc).astimezone(LONDON).date()).days


def london_day_utc(day: date) -> tuple[datetime, datetime]:
    """The UTC start and end (exclusive) of a Europe/London calendar day.

    That is 24 hours, except 25 on the day the clocks go back and 23 when they go forward.
    """
    start = datetime.combine(day, time(), tzinfo=LONDON)
    end = datetime.combine(day + ONE_DAY, time(), tzinfo=LONDON)
    return start.astimezone(UTC), end.astimezone(UTC)


def london_date(t_utc: datetime) -> date:
    """The Europe/London calendar date that t falls on."""
    return _utc(t_utc).astimezone(LONDON).date()


def london_day_date(now_utc: datetime, day: int) -> date:
    """The Europe/London date of day number `day` counted from now (day 0 is the day `now` is in)."""
    return london_date(now_utc) + day * ONE_DAY


def london_days_utc(now_utc: datetime, days: int) -> tuple[datetime, datetime]:
    """The UTC start of London day 0 (the day `now` is in) and the UTC end (exclusive) of London day days-1.

    So days=1 is today in London, and days=3 on 24 Oct 2026 (BST) is 23 Oct 23:00Z to 27 Oct 00:00Z.
    """
    if days < 1:
        raise ValueError(f"days must be at least 1, got {days}")
    return london_day_utc(london_day_date(now_utc, 0))[0], london_day_utc(london_day_date(now_utc, days - 1))[1]


def utc_hours(start: datetime, end: datetime) -> list[datetime]:
    """UTC hour starts covering start up to end, end excluded.

    The first hour is the one containing start, so a start of 07:30 gives 07:00 first.
    """
    start, end = _utc(start), _utc(end)
    hour = start.replace(minute=0, second=0, microsecond=0)
    hours = []
    while hour < end:
        hours.append(hour)
        hour += ONE_HOUR
    return hours
