"""Command line entry point: `rules` prints rules.yaml as a table, `assess` prints the hourly table, `read` runs the NOTAM reader.

    python -m flysafelz.cli rules
    python -m flysafelz.cli assess --case blue_mill_2026-10-31 [--start T --end T] [--reader fixture|live] [--debug]
    python -m flysafelz.cli assess --lat 53.802797 --lon -0.086644 [--start T --end T] [--reader live] [--debug]
    python -m flysafelz.cli read --case blue_mill_2026-10-31 [--start T --end T]
    python -m flysafelz.cli read --live --lat 53.802797 --lon -0.086644 --start T --end T

This file is argparse plus a few formatting functions. It makes no decision: every verdict,
condition, caution and number it prints comes from the Assessment that assess() returned, and
every limit from rules.yaml. There is no flag that changes a limit or the local area.

--case replays fixtures/<case>/ with the case's own frozen `now`. --lat/--lon runs live. The readings come
from --reader: `fixture` (the default with --case) is the case's readings.yaml; `live` asks the NOTAM reader
(a model, set up in reader/config.py: with a setting missing the command stops with the one line config.py
gives, naming each); a run by --lat/--lon with no --reader has no readings at all (Q8). `read` shows what the
reader makes of each NOTAM it is asked about, with the action notam_action() gives for the first hour the
NOTAM is active. Every bad argument ends in one line on stderr and exit code 2. Times are UTC.

Phone numbers (R3.5). The Assessment, its reasons, the debug list, the logs and the stored readings hold
`<PHONE_n>` only; so do the readings handed to assess (a fixture's contact is redacted on the way in). The real
number is put back here, in the lines this file writes to stdout and nowhere earlier. Placeholders are numbered
per NOTAM, so each line is put back with the map of the NOTAM it speaks about (PhoneBook), and a line whose
NOTAM cannot be told keeps its placeholder rather than risk another NOTAM's number.
"""

import argparse
import re
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime

import yaml

from flysafelz import pib as pib_module
from flysafelz.assess import UNVERIFIED_PREFIX, assess, assessed_span, default_window, display
from flysafelz.clients.openaip import metres_to_feet
from flysafelz.clients.pib_source import utc_text
from flysafelz.contracts import (
    Assessment,
    HourAssessment,
    NotamActionResult,
    NotamReading,
    ParsedNotam,
    Segment,
    SourceName,
    SourceState,
    SourceStatus,
    Tier,
    display_verdict,
)
from flysafelz.daylight import ONE_HOUR
from flysafelz.designators import find_designators
from flysafelz.geo import OutsideUKCoverage, check_uk_bounds
from flysafelz.notam_action import ActionSegment, LandingZone, active_intervals, notam_action
from flysafelz.reader import config as reader_config
from flysafelz.reader.cached_read import model_calls
from flysafelz.reader.contract import ReaderResult, ReaderStatus
from flysafelz.reader.reader_set import failed_before, read_for_lz
from flysafelz.redact import redact_known, redact_phones, rehydrate
from flysafelz.rules import RULES, Rule
from flysafelz.sources import CHECK_BY_HAND, CaseSources, SourceError, fetch_live_sources, load_case_sources

COLUMNS = ("id", "value", "unit", "comparison", "source", "confidence", "note")

# Indents are strings, not counts: a count such as 4 or 6 would equal a rules.yaml value (the F3 scan).
DETAIL_PAD = "    "  # under an uncut hour's row
SUB_ROW_PAD = "  "  # a segment's own row inside a cut hour
SUB_DETAIL_PAD = SUB_ROW_PAD + DETAIL_PAD  # under that row
DASH = "\u2013"
DEGREES = "\u00b0"
MISSING = "n/a"
SOURCE_LABELS = {SourceName.WEATHER: "weather", SourceName.AIRSPACE: "airspace", SourceName.NOTAMS: "NOTAMs"}


def format_value(rule: Rule) -> str:
    if isinstance(rule.value, list):
        return ", ".join(rule.value)
    if isinstance(rule.value, int):
        return str(rule.value)
    return format(rule.value, "g")


def _cell(text: str) -> str:
    return text.replace("|", "\\|")


def _row(cells: list[str]) -> str:
    return "| " + " | ".join(_cell(c) for c in cells) + " |"


def rules_table() -> str:
    """The Markdown table of every rule, as it appears in the README."""
    lines = [_row(list(COLUMNS)), _row(["---"] * len(COLUMNS))]
    for rule in RULES.all():
        lines.append(
            _row(
                [
                    rule.id,
                    format_value(rule),
                    rule.unit,
                    rule.comparison.value,
                    rule.source,
                    rule.confidence.value,
                    rule.note,
                ]
            )
        )
    return "\n".join(lines) + "\n"


# --- formatting an Assessment (all UTC) -----------------------------------------------------------


def clock(t: datetime) -> str:
    return t.astimezone(UTC).strftime("%H:%M")


def minute_text(t: datetime) -> str:
    """2026-10-31T07:30Z"""
    return t.astimezone(UTC).strftime("%Y-%m-%dT%H:%MZ")


def span_text(start: datetime, end: datetime) -> str:
    return f"{clock(start)}{DASH}{clock(end)}"


def overall_display(assessment: Assessment) -> str:
    """The display string of the overall verdict, composed from the segments that carry it.

    Those are the worst VERDICT-tier segments, so an overall PILOT_DISCRETION is not shown as
    plain when they have cautions or conditions. NO-GO and UNVERIFIED take no qualifiers.
    """
    verdict = assessment.overall_verdict
    if verdict is None:
        raise ValueError("there is no overall verdict to display")
    worst = [s for h in assessment.hours if h.tier is Tier.VERDICT for s in h.segments if s.verdict is verdict]
    return display_verdict(verdict, any(s.approaching_limits for s in worst), any(s.conditions for s in worst))


def number_text(value: float | None, unit: str) -> str:
    return MISSING if value is None else f"{value:.1f} {unit}"


# --- phone numbers (R3.5) ---------------------------------------------------------------------------

PLACEHOLDER = re.compile(r"<PHONE_\d+>")
YAML_PAD = "      "


@dataclass
class PhoneBook:
    """Each NOTAM's map from `<PHONE_n>` back to the number in its E text, and the words that tie a printed line to the NOTAM.

    Placeholders are numbered per NOTAM, so `<PHONE_1>` of one NOTAM is not `<PHONE_1>` of another: a line is put back with
    the map of the NOTAM it speaks about only. A NOTAM with no number has no map.
    """

    maps: dict[str, dict[str, str]] = field(default_factory=dict)
    markers: dict[str, tuple[str, ...]] = field(default_factory=dict)

    @classmethod
    def of(cls, notams: Sequence[ParsedNotam]) -> "PhoneBook":
        book = cls()
        for notam in notams:
            _redacted, mapping = redact_phones(notam.item_e)
            book.markers[notam.id] = (notam.id, *find_designators(notam.item_e))  # every NOTAM, so a line about one with no number is not put back with another's
            if mapping:
                book.maps[notam.id] = mapping
        return book

    def redact_contacts(self, readings: Mapping[str, NotamReading]) -> dict[str, NotamReading]:
        """The readings with any real number in a contact swapped for its NOTAM's placeholder (a fixture's hand-written contact).

        A number that is in the NOTAM's E text gets the placeholder the E text has; one that is not gets the next number up and
        the map learns it. Readings made by the reader already hold placeholders and come back unchanged.
        """
        out: dict[str, NotamReading] = {}
        for notam_id, reading in readings.items():
            known = self.markers.get(notam_id, (notam_id,))
            self.markers[notam_id] = (*known, *(d for d in reading.activates_designators if d not in known))  # a designator the reading names ties a line to this NOTAM too
            value = reading.contact.value
            if value:
                text, mapping = redact_known(value, self.maps.get(notam_id, {}))
                if text != value:
                    self.maps[notam_id] = mapping
                    reading = reading.model_copy(update={"contact": reading.contact.model_copy(update={"value": text})})
            out[notam_id] = reading
        return out

    def restore_for(self, notam_id: str, line: str) -> str:
        """The line with the numbers of this one NOTAM put back. An unknown placeholder stays as it is."""
        return rehydrate(line, self.maps.get(notam_id, {}))

    def restore(self, line: str, actions: Sequence[NotamActionResult]) -> str:
        """A line that is not an Action line, put back with the map of the NOTAM it speaks about.

        The line speaks about a NOTAM when it holds that NOTAM's id or one of its designators, or the reason of that NOTAM's
        action in this segment. Ids and designators are matched against every NOTAM in the book, not only the segment's
        actions: an airspace condition can come from a NOTAM that was ignored (a drone-only reading still activates its danger
        area) and so has no action. When the line names no NOTAM, every NOTAM in the book is a candidate. A placeholder is put
        back only when every candidate that has it agrees on the number, and a named NOTAM with no number has nothing to put
        back; otherwise it stays, so a number is never taken from a NOTAM the line is not about.
        """
        wanted = set(PLACEHOLDER.findall(line))
        if not wanted:
            return line
        reasons = {a.notam_id for a in actions if a.reason and a.reason in line}
        named = reasons | {notam_id for notam_id, markers in self.markers.items() if any(m in line for m in markers)}
        pool = {i for i in (named or self.maps) if i in self.maps}
        agreed: dict[str, str] = {}
        for placeholder in wanted:
            numbers = {self.maps[i][placeholder] for i in pool if placeholder in self.maps[i]}
            if len(numbers) == 1:
                agreed[placeholder] = numbers.pop()
        return rehydrate(line, agreed)


def notam_line(action: NotamActionResult, conditions: list[str]) -> str:
    """NOTAM <id> <ACTION> [bearing <deg>]: <reason>. The reason is left out when a condition already says it."""
    line = f"NOTAM {action.notam_id} {action.action.value}"
    if action.bearing_deg is not None:
        line += f" bearing {action.bearing_deg:.1f}{DEGREES}"
    if action.reason and not any(action.reason in condition for condition in conditions):
        line += f": {action.reason}"
    return line


def segment_details(segment: Segment, pad: str, phones: PhoneBook | None = None) -> list[str]:
    """Why the segment reads as it does: blockers (in the order assess gave them, weather first), conditions, cautions, NOTAMs.

    A caution prints its own text, which carries its numbers from the Assessment (e.g. "gust spread 5 mph, limit 6").
    With `phones` the real numbers go back into each line as it is made (R3.5): an Action line from its own NOTAM's map,
    any other line by PhoneBook.restore. Without it the lines keep their `<PHONE_n>`.
    """
    lines = [f"{pad}blocked ({b.source}): {b.text}" for b in segment.blockers]
    for condition in segment.conditions:
        if condition.startswith(UNVERIFIED_PREFIX):
            lines.append(f"{pad}unverified: {condition.removeprefix(UNVERIFIED_PREFIX)}")
        else:
            lines.append(f"{pad}condition: {condition}")
    lines += [f"{pad}caution: {c.text}" for c in segment.cautions]
    if phones is not None:
        lines = [phones.restore(line, segment.notam_actions) for line in lines]
        return lines + [phones.restore_for(a.notam_id, f"{pad}{notam_line(a, segment.conditions)}") for a in segment.notam_actions]
    lines += [f"{pad}{notam_line(a, segment.conditions)}" for a in segment.notam_actions]
    return lines


def hour_lines(hour: HourAssessment, phones: PhoneBook | None = None) -> list[str]:
    """One row for the UTC hour. An hour cut by a daylight or NOTAM edge prints its segments as sub-rows."""
    label = span_text(hour.hour, hour.hour + ONE_HOUR)
    if hour.tier is Tier.TOO_FAR:
        return [f"{label}  TOO FAR OUT"]
    if hour.tier is Tier.OUTLOOK:
        w = hour.weather
        values = f"wind {number_text(w.wind_10m, 'mph')}, gust {number_text(w.gust_10m, 'mph')}, rain chance {number_text(w.precipitation_probability, '%')}"
        return [f"{label}  OUTLOOK  {values}"]
    if len(hour.segments) == 1:
        segment = hour.segments[0]
        return [f"{label}  {display(segment)}", *segment_details(segment, DETAIL_PAD, phones)]
    lines = [label]
    for segment in hour.segments:
        lines.append(f"{SUB_ROW_PAD}{span_text(segment.start, segment.end)}  {display(segment)}")
        lines += segment_details(segment, SUB_DETAIL_PAD, phones)
    return lines


def table_lines(assessment: Assessment, phones: PhoneBook | None = None) -> list[str]:
    """Every hour, with a date line (UTC) whenever the day changes."""
    lines: list[str] = []
    day = None
    for hour in assessment.hours:
        if hour.hour.date() != day:
            day = hour.hour.date()
            lines.append(f"{day.isoformat()} (UTC)")
        lines += hour_lines(hour, phones)
    return lines


def status_lines(status: SourceStatus) -> list[str]:
    """A source's state and reason and, when it is not ok, where to check it by hand."""
    line = f"  {SOURCE_LABELS[status.source]}: {status.state.value}"
    if status.state is SourceState.OK and status.reason:
        line += f" ({status.reason})"
    elif status.reason:
        line += f" {DASH} {status.reason}"
    if status.source is SourceName.NOTAMS and status.issued_at is not None:
        line += f" (PIB issued {utc_text(status.issued_at)})"
    lines = [line]
    if status.state is not SourceState.OK:
        lines.append(f"    check by hand: {CHECK_BY_HAND[status.source]}")
    return lines


def overall_line(assessment: Assessment) -> str:
    if assessment.overall_verdict is not None:
        return f"Overall: {overall_display(assessment)}"
    if assessment.overall_tier is Tier.OUTLOOK:
        return "Overall: No verdict: OUTLOOK only"
    return "Overall: No verdict: too far out"


def footer_lines(assessment: Assessment) -> list[str]:
    lines = ["Sources"]
    for status in assessment.source_statuses:
        lines += status_lines(status)
    lines.append("Flyable windows (UTC)")
    if assessment.flyable_windows:
        lines += [f"  {minute_text(w.start)} to {minute_text(w.end)}" for w in assessment.flyable_windows]
    else:
        lines.append("  none")
    lines.append(overall_line(assessment))
    if assessment.assumptions:
        lines.append("Assumptions")
        lines += [f"  {a}" for a in assessment.assumptions]
    return lines


def debug_lines(assessment: Assessment) -> list[str]:
    """Every NOTAM the code dropped or ignored, with its reason. Printed only with --debug (G23)."""
    lines = [f"Dropped or ignored NOTAMs ({len(assessment.ignored_notams)})"]
    lines += [f"  {i.notam_id}: {i.reason}" for i in assessment.ignored_notams]
    return lines


def header_lines(assessment: Assessment, title: str, lat: float, lon: float, now: datetime) -> list[str]:
    notams = next((s for s in assessment.source_statuses if s.source is SourceName.NOTAMS), None)
    if notams is None or notams.issued_at is None:
        pib = "PIB: not available"
    else:
        where = f"archive commit {notams.sha}" if notams.sha else "live from NATS"
        pib = f"PIB: {where}, issued {utc_text(notams.issued_at)}"
    return [
        f"FlySafeLZ assess {DASH} {title}",
        f"LZ {lat}, {lon} \u00b7 now {minute_text(now)}",
        pib,
        f"Window {minute_text(assessment.window_start)} to {minute_text(assessment.window_end)} (end exclusive). All times UTC.",
    ]


def report(assessment: Assessment, *, title: str, lat: float, lon: float, now: datetime, debug: bool = False, phones: PhoneBook | None = None) -> str:
    """The whole printout: header, one row per UTC hour, footer, and with debug the dropped-NOTAM list."""
    blocks = [header_lines(assessment, title, lat, lon, now), table_lines(assessment, phones), footer_lines(assessment)]
    if debug:
        blocks.append(debug_lines(assessment))
    return "\n\n".join("\n".join(block) for block in blocks) + "\n"


# --- arguments --------------------------------------------------------------------------------------


class OneLineParser(argparse.ArgumentParser):
    """argparse, except an error is one line on stderr (no usage block) and exit code 2."""

    def print_usage(self, file=None) -> None:
        return None


def parse_time(text: str) -> datetime:
    """An ISO 8601 time with a timezone (Z or +00:00 both work), as UTC."""
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{text!r} is not an ISO 8601 time like 2026-10-31T07:00Z") from None
    if parsed.tzinfo is None:
        raise argparse.ArgumentTypeError(f"{text!r} has no timezone; add Z or +00:00")
    return parsed.astimezone(UTC)


READING_SOURCES = ("fixture", "live")
FIXTURE, LIVE = READING_SOURCES


def build_parser() -> tuple[argparse.ArgumentParser, argparse.ArgumentParser, argparse.ArgumentParser]:
    """The main parser, the assess parser and the read parser (the checks made after parsing report through the latter two)."""
    parser = OneLineParser(prog="flysafelz")
    commands = parser.add_subparsers(dest="command", required=True, parser_class=OneLineParser)
    commands.add_parser("rules", help="print rules.yaml as a Markdown table")
    assess_parser = commands.add_parser("assess", help="the hourly table for a landing zone")
    assess_parser.add_argument("--case", help="replay fixtures/<case>/ with its own frozen now")
    assess_parser.add_argument("--lat", type=float, help="latitude, for a live run")
    assess_parser.add_argument("--lon", type=float, help="longitude, for a live run")
    assess_parser.add_argument("--start", type=parse_time, help="window start, ISO 8601 with timezone")
    assess_parser.add_argument("--end", type=parse_time, help="window end (exclusive), ISO 8601 with timezone")
    assess_parser.add_argument("--debug", action="store_true", help="also list the dropped or ignored NOTAMs with their reasons")
    assess_parser.add_argument("--reader", choices=READING_SOURCES, help="where the NOTAM readings come from: the case's readings.yaml (fixture, the default with --case) or the NOTAM reader (live)")
    read_parser = commands.add_parser("read", help="run the NOTAM reader on the NOTAMs that matter for a landing zone and window")
    read_parser.add_argument("--case", help="replay fixtures/<case>/ with its own frozen now")
    read_parser.add_argument("--live", action="store_true", help="get the PIB live; needs --lat, --lon, --start and --end")
    read_parser.add_argument("--lat", type=float, help="latitude, with --live")
    read_parser.add_argument("--lon", type=float, help="longitude, with --live")
    read_parser.add_argument("--start", type=parse_time, help="window start, ISO 8601 with timezone")
    read_parser.add_argument("--end", type=parse_time, help="window end (exclusive), ISO 8601 with timezone")
    return parser, assess_parser, read_parser


# --- the reader --------------------------------------------------------------------------------------


def reader_model(parser: argparse.ArgumentParser):
    """The reader's model client. Missing settings end in the one line reader/config.py words, naming each (exit code 2)."""
    try:
        return reader_config.build_llm()
    except reader_config.ReaderConfigError as exc:
        parser.error(str(exc))


@dataclass
class ReaderRun:
    """What one pass of the reader gave: a result per NOTAM it was asked about, and the ids that were retries."""

    results: dict[str, ReaderResult]
    retried: set[str]

    @property
    def readings(self) -> dict[str, NotamReading]:
        return {i: r.reading for i, r in self.results.items() if r.reading is not None}

    @property
    def unread(self) -> dict[str, str]:
        """{id: "failed" | "unavailable"} for the NOTAMs the reader was asked about and got no reading for."""
        return {i: r.status.value for i, r in self.results.items() if r.reading is None}

    @property
    def calls(self) -> int:
        return sum(model_calls(r) for r in self.results.values())

    @property
    def retries(self) -> int:
        return sum(1 for i, r in self.results.items() if i in self.retried and r.status is not ReaderStatus.STORED)


def run_reader(llm, notams: list[ParsedNotam], lat: float, lon: float, window: tuple[datetime, datetime], now: datetime) -> ReaderRun:
    """Read the NOTAMs of the reader set for this landing zone and window. Retries are counted from the stored failures read_for_lz keeps."""
    store = reader_config.reading_store()
    retried = failed_before(notams, (lat, lon), window, store)
    return ReaderRun(read_for_lz(notams, (lat, lon), window, llm, store, now=now), retried)


# --- assess ------------------------------------------------------------------------------------------


def run_assess(args: argparse.Namespace, parser: argparse.ArgumentParser, live_now: datetime | None) -> str:
    """The printout for an assess command. Any problem with the arguments ends in parser.error (one line, exit code 2)."""
    loaded: CaseSources | None = None
    if args.case is not None:
        if args.lat is not None or args.lon is not None:
            parser.error("--case cannot be used with --lat or --lon")
        try:
            loaded = load_case_sources(args.case, readings_optional=args.reader == LIVE)
        except SourceError as exc:
            parser.error(str(exc))
        lat, lon, now = loaded.case.lat, loaded.case.lon, loaded.case.now
        start, end = args.start or loaded.case.window.start, args.end or loaded.case.window.end
        window, live, title = (start, end), False, f"case {args.case}"
    else:
        if args.reader == FIXTURE:
            parser.error("--reader fixture needs --case (a fixture is a case's readings.yaml)")
        if args.lat is None or args.lon is None:
            parser.error("give --case, or both --lat and --lon")
        if (args.start is None) != (args.end is None):
            parser.error("give both --start and --end, or neither")
        lat, lon, now = args.lat, args.lon, live_now or datetime.now(UTC)
        window = (args.start, args.end) if args.start else None  # None: assess uses and records the default window
        start, end = window or default_window(now)
        live, title = True, "live"
    if start >= end:
        parser.error(f"--end must be after --start (got {minute_text(start)} to {minute_text(end)})")
    mode = args.reader or (FIXTURE if loaded is not None else None)
    llm = reader_model(parser) if mode == LIVE else None
    try:
        check_uk_bounds(lat, lon)  # before any fetch
        if loaded is None:
            forecast, airspace, pib = fetch_live_sources(lat, lon, (start, end), now)
            readings = {}
        else:
            forecast, airspace, pib, readings = loaded.forecast, loaded.airspace, loaded.pib, loaded.readings
        phones = PhoneBook.of(pib.notams)
        unread: dict[str, str] = {}
        if mode == LIVE:
            run = run_reader(llm, pib.notams, lat, lon, assessed_span(start, end, now, live), now)
            readings, unread = run.readings, run.unread
        readings = phones.redact_contacts(readings)
        assessment = assess(lat, lon, now=now, forecast=forecast, airspace=airspace, pib=pib, readings=readings, reader_status=unread, window=window, live=live)
    except OutsideUKCoverage as exc:
        parser.error(str(exc))
    return report(assessment, title=title, lat=lat, lon=lon, now=now, debug=args.debug, phones=phones)


# --- read --------------------------------------------------------------------------------------------


def reading_yaml(reading: NotamReading, pad: str) -> list[str]:
    """The reading as YAML, indented. The contact holds `<PHONE_n>`: this is the stored reading, never put back."""
    text = yaml.safe_dump(reading.model_dump(mode="json"), sort_keys=False, allow_unicode=True)
    return [f"{pad}{line}" for line in text.splitlines()]


def action_span(notam: ParsedNotam, reading: NotamReading | None, lz: LandingZone, window: tuple[datetime, datetime]) -> tuple[datetime, datetime]:
    """The segment the action is worked out for: the first hour the NOTAM is active in the window (the window's first hour when never)."""
    intervals = active_intervals(notam, reading, lz, *window)
    start, stop = intervals[0] if intervals else window
    return start, min(start + ONE_HOUR, stop)


def status_text(result: ReaderResult) -> str:
    if result.status is ReaderStatus.STORED:
        return f"{result.status.value}  (stored reading, no model call)"
    return result.status.value


def read_lines(notams: list[ParsedNotam], run: ReaderRun, lz: LandingZone, window: tuple[datetime, datetime], phones: PhoneBook) -> list[str]:
    """One block per NOTAM read: id and status, the reading, the checks that failed and the action. Then the counts."""
    by_id = {n.id: n for n in notams}
    lines: list[str] = []
    for notam_id, result in run.results.items():
        notam = by_id[notam_id]
        start, stop = action_span(notam, result.reading, lz, window)
        action = notam_action(notam, result.reading, lz, ActionSegment(start=start, end=stop))
        lines.append(f"NOTAM {notam_id}  {status_text(result)}")
        if result.reading is not None:
            lines += ["  reading:", *reading_yaml(result.reading, YAML_PAD)]
        lines += [f"  failed check: {check}" for check in result.failed_checks]
        # The Action line only is put back with the NOTAM's own map; the reading above keeps its placeholders.
        action_line = f"  action {span_text(start, stop)}: {notam_line(action, [])}"
        lines += [phones.restore_for(notam_id, action_line), ""]
    lines += [f"Model calls: {run.calls}", f"Retries: {run.retries}"]
    return lines


def run_read(args: argparse.Namespace, parser: argparse.ArgumentParser, live_now: datetime | None) -> str:
    """The printout for a read command. Any problem with the arguments ends in parser.error (one line, exit code 2)."""
    forecast_elevation_m: float | None = None
    if args.case is not None:
        if args.live:
            parser.error("--case cannot be used with --live")
        if args.lat is not None or args.lon is not None:
            parser.error("--case cannot be used with --lat or --lon")
        try:
            loaded = load_case_sources(args.case, readings_optional=True)
        except SourceError as exc:
            parser.error(str(exc))
        lat, lon, now, pib = loaded.case.lat, loaded.case.lon, loaded.case.now, loaded.pib
        start, end = args.start or loaded.case.window.start, args.end or loaded.case.window.end
        forecast_elevation_m, title = loaded.forecast.elevation_m, f"case {args.case}"
    else:
        if not args.live:
            parser.error("give --case, or --live with --lat, --lon, --start and --end")
        if args.lat is None or args.lon is None or args.start is None or args.end is None:
            parser.error("--live needs --lat, --lon, --start and --end")
        lat, lon, now, start, end, title = args.lat, args.lon, live_now or datetime.now(UTC), args.start, args.end, "live"
    if start >= end:
        parser.error(f"--end must be after --start (got {minute_text(start)} to {minute_text(end)})")
    try:
        check_uk_bounds(lat, lon)  # before any fetch
    except OutsideUKCoverage as exc:
        parser.error(str(exc))
    llm = reader_model(parser)
    if args.case is None:
        pib = pib_module.load_pib("live", now)
    if not pib.notams and pib.status.state is SourceState.FAILED:
        parser.exit(1, f"flysafelz: NOTAMs unavailable: {pib.status.reason}\n")
    elevation_ft = metres_to_feet(forecast_elevation_m) if forecast_elevation_m is not None else 0.0
    lz = LandingZone(lat=lat, lon=lon, elevation_ft=elevation_ft)
    run = run_reader(llm, pib.notams, lat, lon, (start, end), now)
    header = [
        f"FlySafeLZ read {DASH} {title}",
        f"LZ {lat}, {lon} \u00b7 now {minute_text(now)}",
        f"Window {minute_text(start)} to {minute_text(end)} (end exclusive). All times UTC.",
        f"Site elevation {elevation_ft:.0f} ft" + ("" if forecast_elevation_m is not None else " (not known without a forecast: taken as 0 ft)"),
        f"Asked the reader about {len(run.results)} NOTAM{'' if len(run.results) == 1 else 's'}" + (f": {', '.join(run.results)}" if run.results else ""),
    ]
    return "\n".join(header) + "\n\n" + "\n".join(read_lines(pib.notams, run, lz, (start, end), PhoneBook.of(pib.notams))) + "\n"


def main(argv: list[str] | None = None, *, now: datetime | None = None) -> int:
    """Run a command. `now` stands in for the clock in a live run (tests freeze it); --case always uses the case's own."""
    parser, assess_parser, read_parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "rules":
        sys.stdout.write(rules_table())
    elif args.command == "read":
        sys.stdout.write(run_read(args, read_parser, now))
    else:
        sys.stdout.write(run_assess(args, assess_parser, now))
    return 0


if __name__ == "__main__":
    sys.exit(main())
