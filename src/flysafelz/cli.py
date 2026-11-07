"""Command line entry point: `rules` prints rules.yaml as a table, `assess` prints the hourly table.

    python -m flysafelz.cli rules
    python -m flysafelz.cli assess --case blue_mill_2026-10-31 [--start T --end T] [--debug]
    python -m flysafelz.cli assess --lat 53.802797 --lon -0.086644 [--start T --end T] [--debug]

This file is argparse plus a few formatting functions. It makes no decision: every verdict,
condition, caution and number it prints comes from the Assessment that assess() returned, and
every limit from rules.yaml. There is no flag that changes a limit or the local area.

--case replays fixtures/<case>/ with the case's own frozen `now`. --lat/--lon runs live with no
readings (Q8). Every bad argument ends in one line on stderr and exit code 2. Times are UTC.
"""

import argparse
import sys
from datetime import UTC, datetime

from flysafelz.assess import UNVERIFIED_PREFIX, assess, default_window, display
from flysafelz.clients.pib_source import utc_text
from flysafelz.contracts import (
    Assessment,
    HourAssessment,
    NotamActionResult,
    Segment,
    SourceName,
    SourceState,
    SourceStatus,
    Tier,
    display_verdict,
)
from flysafelz.daylight import ONE_HOUR
from flysafelz.geo import OutsideUKCoverage, check_uk_bounds
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


def notam_line(action: NotamActionResult, conditions: list[str]) -> str:
    """NOTAM <id> <ACTION> [bearing <deg>]: <reason>. The reason is left out when a condition already says it."""
    line = f"NOTAM {action.notam_id} {action.action.value}"
    if action.bearing_deg is not None:
        line += f" bearing {action.bearing_deg:.1f}{DEGREES}"
    if action.reason and not any(action.reason in condition for condition in conditions):
        line += f": {action.reason}"
    return line


def segment_details(segment: Segment, pad: str) -> list[str]:
    """Why the segment reads as it does: blockers (in the order assess gave them, weather first), conditions, cautions, NOTAMs.

    A caution prints its own text, which carries its numbers from the Assessment (e.g. "gust spread 5 mph, limit 6").
    """
    lines = [f"{pad}blocked ({b.source}): {b.text}" for b in segment.blockers]
    for condition in segment.conditions:
        if condition.startswith(UNVERIFIED_PREFIX):
            lines.append(f"{pad}unverified: {condition.removeprefix(UNVERIFIED_PREFIX)}")
        else:
            lines.append(f"{pad}condition: {condition}")
    lines += [f"{pad}caution: {c.text}" for c in segment.cautions]
    lines += [f"{pad}{notam_line(a, segment.conditions)}" for a in segment.notam_actions]
    return lines


def hour_lines(hour: HourAssessment) -> list[str]:
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
        return [f"{label}  {display(segment)}", *segment_details(segment, DETAIL_PAD)]
    lines = [label]
    for segment in hour.segments:
        lines.append(f"{SUB_ROW_PAD}{span_text(segment.start, segment.end)}  {display(segment)}")
        lines += segment_details(segment, SUB_DETAIL_PAD)
    return lines


def table_lines(assessment: Assessment) -> list[str]:
    """Every hour, with a date line (UTC) whenever the day changes."""
    lines: list[str] = []
    day = None
    for hour in assessment.hours:
        if hour.hour.date() != day:
            day = hour.hour.date()
            lines.append(f"{day.isoformat()} (UTC)")
        lines += hour_lines(hour)
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


def report(assessment: Assessment, *, title: str, lat: float, lon: float, now: datetime, debug: bool = False) -> str:
    """The whole printout: header, one row per UTC hour, footer, and with debug the dropped-NOTAM list."""
    blocks = [header_lines(assessment, title, lat, lon, now), table_lines(assessment), footer_lines(assessment)]
    if debug:
        blocks.append(debug_lines(assessment))
    return "\n\n".join("\n".join(block) for block in blocks) + "\n"


# --- arguments --------------------------------------------------------------------------------------


class OneLineParser(argparse.ArgumentParser):
    """argparse, except an error is one line on stderr (no usage block) and exit code 2."""

    def print_usage(self, file=None) -> None:  # noqa: ANN001
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


def build_parser() -> tuple[argparse.ArgumentParser, argparse.ArgumentParser]:
    """The main parser and the assess parser (the checks made after parsing report through the latter)."""
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
    return parser, assess_parser


def run_assess(args: argparse.Namespace, parser: argparse.ArgumentParser, live_now: datetime | None) -> str:
    """The printout for an assess command. Any problem with the arguments ends in parser.error (one line, exit code 2)."""
    loaded: CaseSources | None = None
    if args.case is not None:
        if args.lat is not None or args.lon is not None:
            parser.error("--case cannot be used with --lat or --lon")
        try:
            loaded = load_case_sources(args.case)
        except SourceError as exc:
            parser.error(str(exc))
        lat, lon, now = loaded.case.lat, loaded.case.lon, loaded.case.now
        start, end = args.start or loaded.case.window.start, args.end or loaded.case.window.end
        window, live, title = (start, end), False, f"case {args.case}"
    else:
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
    try:
        check_uk_bounds(lat, lon)  # before any fetch
        if loaded is None:
            forecast, airspace, pib = fetch_live_sources(lat, lon, (start, end), now)
            readings = {}
        else:
            forecast, airspace, pib, readings = loaded.forecast, loaded.airspace, loaded.pib, loaded.readings
        assessment = assess(lat, lon, now=now, forecast=forecast, airspace=airspace, pib=pib, readings=readings, window=window, live=live)
    except OutsideUKCoverage as exc:
        parser.error(str(exc))
    return report(assessment, title=title, lat=lat, lon=lon, now=now, debug=args.debug)


def main(argv: list[str] | None = None, *, now: datetime | None = None) -> int:
    """Run a command. `now` stands in for the clock in a live run (tests freeze it); --case always uses the case's own."""
    parser, assess_parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "rules":
        sys.stdout.write(rules_table())
    else:
        sys.stdout.write(run_assess(args, assess_parser, now))
    return 0


if __name__ == "__main__":
    sys.exit(main())
