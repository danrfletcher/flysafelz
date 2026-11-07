"""Times written into sentences, in UTC (the default) or in the pilot's zone. Pure functions, stdlib only.

    check_tz(tz)               raise ValueError for a zone name that is not a real IANA zone
    clock(t, tz)               08:00
    labelled_clock(t, tz)      08:00 (UTC default), 08:00 GMT / 09:00 BST (a zone)
    stamp(t, tz)               2026-10-31T07:30:00Z (UTC default), 2026-10-31 07:30 GMT (a zone)
    clock_span(start, end, tz) 08:00-18:00 (UTC default), 08:00-18:00 GMT, 00:30 BST-02:00 GMT (a zone)

`tz` is an IANA name (Europe/London) or None. None and "UTC" give today's UTC text, byte for byte. Any other
name is looked up in zoneinfo, which also gives the offset and the abbreviation for that instant (GMT or BST,
never a table of ours). "utc" and "Etc/UTC" are real zones, so they are written as a zone: "... UTC", not "...Z".

An unknown or empty name raises ValueError naming it. It never falls back to London or UTC. Every time must be
timezone-aware; a naive one raises ValueError. A time is converted from its UTC instant, so the repeated hour when
the clocks go back (01:30 BST, then 01:30 GMT) is never guessed: the two instants give two different texts.

This only writes strings. Structured fields (segment starts, windows, fetch times) stay UTC datetimes.
"""

from datetime import UTC, datetime, tzinfo
from zoneinfo import ZoneInfo

UTC_NAME = "UTC"
EN_DASH = "\u2013"

_CLOCK = "%H:%M"
_LOCAL_STAMP = "%Y-%m-%d %H:%M"
_UTC_STAMP = "%Y-%m-%dT%H:%M:%SZ"


def _zone(tz: str | None) -> tzinfo | None:
    """None for the UTC default (None or "UTC"), else the zone. ValueError naming `tz` when it is not a zone."""
    if tz is None or tz == UTC_NAME:
        return None
    # Any spelling of UTC is the one zone; a case-sensitive disk would not find "utc" otherwise.
    name = UTC_NAME if tz.casefold() == UTC_NAME.casefold() else tz
    try:
        return ZoneInfo(name)
    except (KeyError, ValueError, OSError) as exc:
        # KeyError: no such zone. ValueError: "", an absolute or ../ path. OSError: a folder such as "Europe".
        raise ValueError(f"unknown time zone {tz!r}: use an IANA name such as Europe/London") from exc


def check_tz(tz: str | None) -> None:
    """Raise ValueError when `tz` is not None and not a real zone. Entry points call it so a bad name fails even if nothing is written."""
    _zone(tz)


def _aware(t: datetime) -> datetime:
    if t.tzinfo is None:
        raise ValueError(f"time must be timezone-aware, got naive {t!r}")
    return t


def _local(t: datetime, zone: tzinfo | None) -> datetime:
    """The instant `t` on the clock of `zone` (UTC when None). Converting from an aware time sets the fold."""
    return _aware(t).astimezone(UTC if zone is None else zone)


def _abbreviation(local: datetime) -> str:
    """GMT, BST, JST ... as zoneinfo gives it for this instant; a zone with no letters gives its offset (+0530)."""
    return local.tzname() or ""


def clock(t: datetime, tz: str | None = None) -> str:
    """HH:MM on the clock of `tz`, with no zone label."""
    return _local(t, _zone(tz)).strftime(_CLOCK)


def labelled_clock(t: datetime, tz: str | None = None) -> str:
    """HH:MM, followed by the zone abbreviation for that instant when a zone is set (01:30 BST, 01:30 GMT)."""
    zone = _zone(tz)
    local = _local(t, zone)
    return local.strftime(_CLOCK) if zone is None else f"{local.strftime(_CLOCK)} {_abbreviation(local)}"


def stamp(t: datetime, tz: str | None = None) -> str:
    """A full time: 2026-10-31T07:30:00Z for the UTC default, else date, time and abbreviation (2026-10-31 07:30 GMT)."""
    zone = _zone(tz)
    local = _local(t, zone)
    if zone is None:
        return local.strftime(_UTC_STAMP)
    return f"{local.strftime(_LOCAL_STAMP)} {_abbreviation(local)}"


def clock_span(start: datetime, end: datetime, tz: str | None = None) -> str:
    """A range of times of day, ends in the order given.

    UTC default: 08:00-18:00. A zone, both ends in one local day: the abbreviation once when the ends agree
    (08:00-18:00 GMT), on each end when they differ (00:30 BST-02:00 GMT). Different local days: both ends as full
    stamps, so the dates show (2026-10-31 23:00 JST-2026-11-01 03:00 JST).
    """
    zone = _zone(tz)
    first, last = _local(start, zone), _local(end, zone)
    if zone is None:
        return f"{first.strftime(_CLOCK)}{EN_DASH}{last.strftime(_CLOCK)}"
    if first.date() != last.date():
        return f"{stamp(start, tz)}{EN_DASH}{stamp(end, tz)}"
    if _abbreviation(first) == _abbreviation(last):
        return f"{first.strftime(_CLOCK)}{EN_DASH}{last.strftime(_CLOCK)} {_abbreviation(last)}"
    return f"{labelled_clock(start, tz)}{EN_DASH}{labelled_clock(end, tz)}"
