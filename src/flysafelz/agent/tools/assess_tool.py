"""assess (G12): the verdict for the landing zone and a window, exactly as ticket 2's `assess` gives it, with nothing added or decided here.

The model gives `start` and `end` (local `YYYY-MM-DD HH:MM` in the pilot's zone) or neither. The LZ, zone, clock and case come from the thread and
the run context. The tool takes the three sources from the thread's `SourceCache`, so it shares `get_weather`, `get_airspace` and `get_notams`'s
fetch and their fetch time, and reads the NOTAMs the same way `get_notams` does (`read_chosen`: the reading store first, at most
`notam_reader_calls` model reads, the case's `readings.yaml` in a fixture case). A verdict therefore never depends on `get_notams` having run first.
It then calls `assess.assess(lat, lon, now=, forecast=, airspace=, pib=, readings=, reader_status=, window=, live=, tz=)` with an explicit UTC window,
`live=True` with no case and `live=False` with one (FR2), and writes the answer in two forms:

    text      plain text for the model: the window, the fetch time, the overall verdict, the counts, the flyable windows with their lengths, the state of
              each source (a source that is not ok with its reason, where to check by hand and "Do not fly on this assessment."), every VERDICT-tier
              segment with its times in `tz`, the OUTLOOK hours with their trend and the day to ask again, the hours too far out, and the assumptions
    artifact  `AssessArtifact`: the same facts as data, so the guard pairs each value with its limit and the UI draws its badges without parsing the text

Every verdict string is `display_verdict` (through `assess.display` for a segment, `cli.overall_display` for the overall). The statuses come from
`Assessment.source_statuses`, not from the thread's stored statuses, so a PIB that is too old is reported stale. An OUTLOOK or TOO_FAR hour has no verdict and no verdict
string anywhere; a VERDICT-tier hour made UNVERIFIED by a failed source shows "UNVERIFIED" (or "NO-GO" when another source says so), never a
PILOT DISCRETION string.

Phone numbers. Ticket 2 and 3's strings number a NOTAM's phones from 1 for that NOTAM alone. They are put back with the NOTAM's own map (the CLI's
`PhoneBook`, which keeps a placeholder it cannot place), and a placeholder that stays is replaced by a plain phrase, since `<PHONE_1>` of one NOTAM
could otherwise pass for the thread's `<PHONE_1>`. Then the redaction boundary (`reply`) numbers every number for the thread, in the order the text
reaches it.
"""

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from typing import NamedTuple
from zoneinfo import ZoneInfo

from langchain.tools import ToolRuntime, tool
from langchain_core.tools import BaseTool
from langgraph.types import Command

from flysafelz import assess
from flysafelz.agent import progress
from flysafelz.agent.sources import SourceCache, fetch_time
from flysafelz.agent.state import OverlayFeature, OverlayProperties, lz_radius_feature
from flysafelz.agent.tools.artifacts import (
    AssessArtifact,
    AssessBlock,
    AssessBlocker,
    AssessCaution,
    AssessCounts,
    AssessNotam,
    AssessOverall,
    AssessSegment,
    AssessSpan,
)
from flysafelz.agent.tools.common import Prepared, Redactor, WindowArgs, g, limit_phrase, measured, prepare, reply, source_info_at, usable
from flysafelz.agent.tools.notams_tool import read_chosen
from flysafelz.agent.window import Window
from flysafelz.cli import PhoneBook, overall_display
from flysafelz.contracts import Assessment, Blocker, Caution, HourAssessment, NotamActionResult, ParsedNotam, Segment, SourceName, SourceState, SourceStatus, Tier
from flysafelz.daylight import ONE_HOUR, ONE_MINUTE, london_date
from flysafelz.reader.contract import ReaderStatus
from flysafelz.reader.reader_set import reader_set
from flysafelz.rules import RULES
from flysafelz.shapes import notam_geometry
from flysafelz.sources import CHECK_BY_HAND
from flysafelz.timetext import EN_DASH, clock_span, labelled_clock, stamp
from flysafelz.weather import Finding, build_rows, check_row

OUTLOOK_WORD = "OUTLOOK (no verdict)"
TOO_FAR_WORD = "too far out to assess"
NO_VERDICT_OUTLOOK = "No verdict: OUTLOOK only"
NO_VERDICT_TOO_FAR = f"No verdict: {TOO_FAR_WORD}"
NO_FLYABLE = "no flyable window in this window"
DO_NOT_FLY = "Do not fly on this assessment."
NOT_FORECAST = "not forecast"
NUMBER_WITHHELD = "[phone number: see the NOTAM]"  # what stands where a placeholder cannot be told from another NOTAM's
DEGREES = "\u00b0"
PAD = "    "  # a string, not a count, so it is no number to the literal scan
MINUTES_PER_HOUR = int(ONE_HOUR / ONE_MINUTE)

LEFTOVER = re.compile(r"<PHONE_\d+>")
UNITS = {rule.id: rule.unit for rule in RULES.all()}  # a caution's unit is its rule's
WIND_UNIT = RULES.wind_10m_caution.unit
GUST_UNIT = RULES.gust_spread_caution.unit
RAIN_UNIT = RULES.rain_chance_caution.unit
ASK_AGAIN_AFTER = timedelta(days=float(RULES.verdict_horizon_day.value))  # type: ignore[arg-type]  # a number rule: an hour can carry a verdict this many days before it

DESCRIPTION = (
    "Assess the pilot's landing zone for a window and give the verdict. This is the only source of a verdict: never state one that this tool did not give. "
    "It lists each segment with its times in the pilot's time zone, its verdict, its conditions, the limits it is approaching and the NOTAM actions behind it, "
    "then the overall verdict, the flyable windows with their lengths, the hours that only have an outlook (no verdict) with the day to ask again, the hours "
    "too far out, any source that failed or is stale with where to check it by hand, and when the data was fetched. "
    "Give start and end as local times written YYYY-MM-DD HH:MM, or leave both out for the default window. "
    "The place and time zone are the pilot's and cannot be changed here."
)


# --- phone numbers ---------------------------------------------------------------------------------------------------


class Restore:
    """Puts the real number back into a string that holds a placeholder numbered for one NOTAM, and withholds one it cannot place.

    Without a book (no NOTAMs, or a hand-built assessment in a test) a string is returned as it is.
    """

    def __init__(self, book: PhoneBook | None = None):
        self.book = book

    def _clean(self, text: str) -> str:
        return LEFTOVER.sub(NUMBER_WITHHELD, text)

    def line(self, text: str, actions: Sequence[NotamActionResult] = ()) -> str:
        """A condition, blocker or assumption: put back with the map of the NOTAM it names."""
        return text if self.book is None else self._clean(self.book.restore(text, actions))

    def action(self, notam_id: str, text: str) -> str:
        """A NOTAM action's reason: put back with that NOTAM's own map."""
        return text if self.book is None else self._clean(self.book.restore_for(notam_id, text))


# --- small helpers ---------------------------------------------------------------------------------------------------


def day_text(day: date) -> str:
    """Saturday 31 Oct."""
    return f"{day:%A} {day.day} {day:%b}"


def _local(moment: datetime, tz: str) -> datetime:
    return moment.astimezone(ZoneInfo(tz))


def _iso(moment: datetime, tz: str) -> str:
    return _local(moment, tz).isoformat()


def _abbr(moment: datetime, tz: str) -> str:
    return _local(moment, tz).tzname() or ""


def span_label(start: datetime, end: datetime, tz: str) -> str:
    """A span on the clock of `tz`: `Saturday 31 Oct 10:00–16:00 GMT`. A span over two local days is written as two full stamps (the dates show)."""
    first, last = _local(start, tz), _local(end, tz)
    clock = clock_span(start, end, tz)
    return f"{day_text(first.date())} {clock}" if first.date() == last.date() else clock


def _minutes(start: datetime, end: datetime) -> int:
    return int((end - start) / ONE_MINUTE)


def _length(minutes: int) -> str:
    hours, rest = divmod(minutes, MINUTES_PER_HOUR)
    return f"{hours} h {rest:02d} min"


def _range(values: Sequence[float | None]) -> tuple[float | None, float | None]:
    present = [value for value in values if value is not None]
    return (min(present), max(present)) if present else (None, None)


def _trend_part(label: str, low: float | None, high: float | None, unit: str) -> str:
    if low is None or high is None:
        return f"{label} {NOT_FORECAST}"
    low_text, high_text = measured(low), measured(high)
    return f"{label} {low_text} {unit}" if low_text == high_text else f"{label} {low_text}{EN_DASH}{high_text} {unit}"


def _ask_again(hour: datetime, now: datetime) -> date | None:
    """The London day this hour first carries a verdict, or None when that day is not after today (the forecast is incomplete for it, not too far out)."""
    day = london_date(hour) - ASK_AGAIN_AFTER
    return day if day > london_date(now) else None


# --- a finding's words and numbers -----------------------------------------------------------------------------------

# How ticket 2 words a weather finding (weather._finding, _on_the_limit): `gust spread 4.7 mph, limit 4`, or `CAPE 1000 J/kg, on the limit 1000; <hint>`.
# Used for one thing only: the label in front of the value (`gust spread`) and the hint after it. The numbers it finds must agree with the finding's own
# value, limit and unit, which come from `weather.check_row` and rules.yaml; where they do not, the line keeps ticket 2's words as they are.
_FINDING = re.compile(
    r"\A(?P<label>.+?) (?P<shown>-?\d+(?:\.\d+)?) (?P<unit>\S+), (?:limit (?P<limit>-?\d+(?:\.\d+)?)|on the limit (?P<on>-?\d+(?:\.\d+)?)(?:; (?P<hint>.*))?)\Z",
    re.DOTALL,
)


class FindingParts(NamedTuple):
    label: str
    hint: str | None


def finding_parts(text: str, value: float | None, limit: float | None, unit: str | None) -> FindingParts | None:
    """The label and hint of a weather finding's text, or None when the text is not that shape or its numbers are not `value`, `limit` and `unit`.

    The value must be what the text prints once `value` is written to the same number of decimals (weather.py prints one, two for a blocker on its limit).
    """
    found = _FINDING.match(text)
    if found is None or value is None or limit is None or unit is None or found["unit"] != unit:
        return None
    shown, printed_limit = found["shown"], found["limit"] if found["limit"] is not None else found["on"]
    places = len(shown.rpartition(".")[-1]) if "." in shown else 0
    if float(printed_limit) != limit or float(shown) != float(f"{value:.{places}f}"):
        return None
    return FindingParts(found["label"], found["hint"])


def _value_line(label: str, rule_id: str | None, value: float | None, limit: float | None, unit: str | None, kind: str) -> str | None:
    """`gust spread 6.04 mph: over the 6 mph limit (NO-GO)`: the value, then where it sits against its rule's limit. None when any number or the rule is not known.

    The comparison is the rule's (rules.yaml); the position word is `contracts.position_word` of the true value, through `limit_phrase`.
    """
    rule = None if rule_id is None else getattr(RULES, rule_id, None)
    if rule is None or value is None or limit is None or unit is None:
        return None
    comparison = rule.comparison.value
    return f"{label} {measured(value, [(comparison, limit)])} {unit}: {limit_phrase(value, limit, comparison, unit)} ({kind})"


# --- the artifact's parts --------------------------------------------------------------------------------------------


def _blocker(blocker: Blocker, hour: datetime, restore: Restore, segment: Segment, facts: Mapping[tuple[datetime, str], Finding]) -> AssessBlocker:
    """A blocker for the artifact. A weather one also gets its rule, label and numbers from the finding `weather.check_row` made for this hour, matched by its text."""
    text = restore.line(blocker.text, segment.notam_actions)
    finding = facts.get((hour, blocker.text)) if blocker.source == "weather" else None
    if finding is None:
        return AssessBlocker(source=blocker.source, text=text)
    unit = UNITS.get(finding.rule_id)
    parts = finding_parts(blocker.text, finding.value, finding.limit, unit)
    return AssessBlocker(source=blocker.source, text=text, rule_id=finding.rule_id, label=parts.label if parts else None, value=finding.value, limit=finding.limit, unit=unit)


def _caution(caution: Caution) -> AssessCaution:
    unit = UNITS.get(caution.rule_id)
    parts = finding_parts(caution.text, caution.value, caution.limit, unit)
    return AssessCaution(rule_id=caution.rule_id, text=caution.text, value=caution.value, limit=caution.limit, unit=unit, label=parts.label if parts else None)


def _segment(segment: Segment, tz: str, restore: Restore, facts: Mapping[tuple[datetime, str], Finding]) -> AssessSegment:
    conditions = [restore.line(condition, segment.notam_actions) for condition in segment.conditions]
    hour = segment.start.astimezone(UTC).replace(minute=0, second=0, microsecond=0)  # a segment never straddles an hour row
    return AssessSegment(
        start=_iso(segment.start, tz),
        end=_iso(segment.end, tz),
        tz_abbr=_abbr(segment.start, tz),
        tier=Tier.VERDICT.value,
        verdict=segment.verdict.value,
        approaching_limits=segment.approaching_limits,
        with_conditions=bool(conditions),
        display=assess.display(segment),
        conditions=conditions,
        cautions=[_caution(c) for c in segment.cautions],
        blockers=[_blocker(b, hour, restore, segment, facts) for b in segment.blockers],
        notams=[AssessNotam(id=a.notam_id, action=a.action.value, bearing_deg=a.bearing_deg, reason=restore.action(a.notam_id, a.reason)) for a in segment.notam_actions],
    )


def _hour_segment(hour: HourAssessment, tz: str, now: datetime) -> AssessSegment:
    """An OUTLOOK or TOO_FAR hour as a segment: no verdict, no display, and for an OUTLOOK hour its raw weather and the day to ask again."""
    outlook = hour.tier is Tier.OUTLOOK
    ask = _ask_again(hour.hour, now) if outlook else None
    return AssessSegment(
        start=_iso(hour.hour, tz),
        end=_iso(hour.hour + ONE_HOUR, tz),
        tz_abbr=_abbr(hour.hour, tz),
        tier=hour.tier.value,
        verdict=None,
        approaching_limits=False,
        with_conditions=False,
        display=None,
        conditions=[],
        cautions=[],
        blockers=[],
        notams=[],
        wind_10m=hour.weather.wind_10m if outlook else None,
        gust_10m=hour.weather.gust_10m if outlook else None,
        rain_chance=hour.weather.precipitation_probability if outlook else None,
        ask_again_on=None if ask is None else ask.isoformat(),
    )


def _runs(hours: list[HourAssessment], tier: Tier) -> list[list[HourAssessment]]:
    """Consecutive hours of one tier. OUTLOOK hours are also split where the London day changes: the day to ask again belongs to a London day."""
    runs: list[list[HourAssessment]] = []
    for hour in hours:
        if hour.tier is not tier:
            continue
        last = runs[-1][-1] if runs else None
        joins = last is not None and last.hour + ONE_HOUR == hour.hour and (tier is not Tier.OUTLOOK or london_date(last.hour) == london_date(hour.hour))
        if joins:
            runs[-1].append(hour)
        else:
            runs.append([hour])
    return runs


def _block(run: list[HourAssessment], tier: Tier, tz: str, now: datetime) -> AssessBlock:
    start, end = run[0].hour, run[-1].hour + ONE_HOUR
    if tier is Tier.TOO_FAR:
        return AssessBlock(tier="TOO_FAR", start=_iso(start, tz), end=_iso(end, tz), tz_abbr=_abbr(start, tz), hours=len(run))
    wind = _range([hour.weather.wind_10m for hour in run])
    gust = _range([hour.weather.gust_10m for hour in run])
    rain = _range([hour.weather.precipitation_probability for hour in run])
    ask = _ask_again(start, now)
    return AssessBlock(
        tier="OUTLOOK",
        start=_iso(start, tz),
        end=_iso(end, tz),
        tz_abbr=_abbr(start, tz),
        hours=len(run),
        wind_min=wind[0],
        wind_max=wind[1],
        wind_unit=WIND_UNIT,
        gust_min=gust[0],
        gust_max=gust[1],
        gust_unit=GUST_UNIT,
        rain_min=rain[0],
        rain_max=rain[1],
        rain_unit=RAIN_UNIT,
        ask_again_on=None if ask is None else ask.isoformat(),
    )


def _overall(assessment: Assessment) -> AssessOverall | None:
    """The overall verdict's machine fields (those of the worst VERDICT-tier segments, as `cli.overall_display` reads them) and its display string."""
    verdict = assessment.overall_verdict
    if verdict is None:
        return None
    worst = [s for h in assessment.hours if h.tier is Tier.VERDICT for s in h.segments if s.verdict is verdict]
    return AssessOverall(
        verdict=verdict.value,
        approaching_limits=any(s.approaching_limits for s in worst),
        with_conditions=any(s.conditions for s in worst),
        display=overall_display(assessment),
    )


def _spans(assessment: Assessment, tz: str) -> list[AssessSpan]:
    return [AssessSpan(start=_iso(w.start, tz), end=_iso(w.end, tz), tz_abbr=_abbr(w.start, tz), minutes=_minutes(w.start, w.end)) for w in assessment.flyable_windows]


# --- the text --------------------------------------------------------------------------------------------------------


def _source_line(status: SourceStatus, tz: str) -> str:
    name = status.source.value
    if status.state is SourceState.OK:
        issued = f" (PIB issued {stamp(status.issued_at, tz)})" if status.source is SourceName.NOTAMS and status.issued_at is not None else ""
        return f"- {name} source: ok{issued}"
    why = f" ({status.reason})" if status.reason else ""
    return f"- {name} source: {status.state.value.upper()}{why}. Check by hand: {CHECK_BY_HAND[status.source]}."


def _blocker_line(blocker: AssessBlocker) -> str:
    """`blocked (weather): gust spread 6.04 mph: over the 6 mph limit (NO-GO)` for a weather blocker whose numbers are known, else ticket 2's words."""
    line = None if blocker.label is None else _value_line(blocker.label, blocker.rule_id, blocker.value, blocker.limit, blocker.unit, "NO-GO")
    return f"{PAD}blocked ({blocker.source}): {blocker.text if line is None else line}"


def _caution_line(caution: AssessCaution) -> str:
    """`caution: low cloud 98 %: over the 90 % limit (caution)` for a finding that is a value against a limit, else ticket 2's words."""
    parts = finding_parts(caution.text, caution.value, caution.limit, caution.unit)
    line = None if parts is None else _value_line(parts.label, caution.rule_id, caution.value, caution.limit, caution.unit, "caution")
    if line is None or parts is None:
        return f"{PAD}caution: {caution.text}"
    return f"{PAD}caution: {line}" + (f"; {parts.hint}" if parts.hint else "")


def _segment_lines(segment: AssessSegment, tz: str) -> list[str]:
    label = clock_span(datetime.fromisoformat(segment.start), datetime.fromisoformat(segment.end), tz)
    lines = [f"{label}  {segment.display}"]
    lines += [_blocker_line(b) for b in segment.blockers]
    for condition in segment.conditions:
        if condition.startswith(assess.UNVERIFIED_PREFIX):
            lines.append(f"{PAD}unverified: {condition.removeprefix(assess.UNVERIFIED_PREFIX)}")
        else:
            lines.append(f"{PAD}condition: {condition}")
    lines += [_caution_line(c) for c in segment.cautions]
    for notam in segment.notams:
        line = f"{PAD}NOTAM {notam.id} {notam.action}"
        if notam.bearing_deg is not None:
            # whole degrees, as `get_notams` and `get_airspace` print it, so the model sees one figure; the artifact keeps the exact bearing
            line += f" bearing {notam.bearing_deg:.0f}{DEGREES}"
        if notam.reason and not any(notam.reason in condition for condition in segment.conditions):
            line += f": {notam.reason}"
        lines.append(line)
    return lines


def _block_line(block: AssessBlock, tz: str) -> str:
    label = span_label(datetime.fromisoformat(block.start), datetime.fromisoformat(block.end), tz)
    if block.tier == "TOO_FAR":
        return f"- {label}: {TOO_FAR_WORD}."
    trend = ", ".join(
        [
            _trend_part("wind", block.wind_min, block.wind_max, block.wind_unit or ""),
            _trend_part("gust", block.gust_min, block.gust_max, block.gust_unit or ""),
            _trend_part("rain chance", block.rain_min, block.rain_max, block.rain_unit or ""),
        ]
    )
    if block.ask_again_on is None:
        return f"- {label}: {OUTLOOK_WORD}. Trend: {trend}. The forecast is not complete enough for a verdict yet; ask again later."
    return f"- {label}: {OUTLOOK_WORD}. Trend: {trend}. Ask again on {day_text(date.fromisoformat(block.ask_again_on))}."


def _segments_text(segments: list[AssessSegment], tz: str) -> list[str]:
    lines: list[str] = []
    shown: date | None = None
    for segment in segments:
        day = datetime.fromisoformat(segment.start).date()  # the segment's own local date: its ISO time is on the clock of tz
        if day != shown:
            shown = day
            lines.append(f"{day_text(day)}:")
        lines += _segment_lines(segment, tz)
    return lines


@dataclass(frozen=True)
class Inputs:
    """What `build` needs beyond the assessment: the zone, the window and clock of the call, the overlays, the restorer and the tool's own assumptions."""

    tz: str
    window: Window
    now: datetime
    overlays: list[OverlayFeature]
    restore: Restore = field(default_factory=Restore)
    extra_assumptions: Sequence[str] = ()
    facts: Mapping[tuple[datetime, str], Finding] = field(default_factory=dict)  # (UTC hour, text) -> the weather finding a weather blocker came from


def build(assessment: Assessment, given: Inputs) -> tuple[str, AssessArtifact]:
    """The text and the artifact for an assessment. Pure: every number and string is the assessment's, written in the zone."""
    tz, restore = given.tz, given.restore
    try:
        fetched: datetime | None = fetch_time(assessment.source_statuses)
    except ValueError:
        fetched = None
    data_as_of = None if fetched is None else labelled_clock(fetched, tz)

    every_segment = [
        item for hour in assessment.hours for item in ([_segment(s, tz, restore, given.facts) for s in hour.segments] if hour.tier is Tier.VERDICT else [_hour_segment(hour, tz, given.now)])
    ]
    verdict_segments = [item for item in every_segment if item.tier == Tier.VERDICT.value]
    outlook = [_block(run, Tier.OUTLOOK, tz, given.now) for run in _runs(assessment.hours, Tier.OUTLOOK)]
    too_far = [_block(run, Tier.TOO_FAR, tz, given.now) for run in _runs(assessment.hours, Tier.TOO_FAR)]
    overall = _overall(assessment)
    spans = _spans(assessment, tz)
    notam_ids = {a.notam_id for h in assessment.hours for s in h.segments for a in s.notam_actions}
    local_area = float(RULES.local_area.value)  # type: ignore[arg-type]  # a number rule
    counts = AssessCounts(
        notams_within=len(notam_ids),
        local_area_nm=local_area,
        hours_assessed=sum(1 for h in assessment.hours if h.tier is Tier.VERDICT),
        hours_outlook=sum(1 for h in assessment.hours if h.tier is Tier.OUTLOOK),
        hours_too_far=sum(1 for h in assessment.hours if h.tier is Tier.TOO_FAR),
        flyable_windows=len(spans),
    )
    assumptions = [restore.line(text) for text in assessment.assumptions] + list(given.extra_assumptions)
    notams_status = next((s for s in assessment.source_statuses if s.source is SourceName.NOTAMS), None)

    window = given.window
    lines = [window.words if window.assumed else f"Window: {window.words}."]
    lines.append(f"Data as of {data_as_of}." if data_as_of else "Data as of: no source returned data.")
    if overall is not None:
        lines.append(f"Overall: {overall.display}")
    else:
        lines.append(f"Overall: {NO_VERDICT_OUTLOOK if assessment.overall_tier is Tier.OUTLOOK else NO_VERDICT_TOO_FAR}")
    noun = "NOTAM" if counts.notams_within == 1 else "NOTAMs"
    lines.append(f"{counts.notams_within} {noun} within {g(local_area)} NM. {counts.hours_assessed} h assessed, {counts.hours_outlook} OUTLOOK, {counts.hours_too_far} too far.")
    if spans:
        lines.append("Flyable windows:")
        lines += [f"- {span_label(w.start, w.end, tz)}, flyable for {_length(span.minutes)}" for w, span in zip(assessment.flyable_windows, spans, strict=True)]
    else:
        lines.append(f"Flyable windows: {NO_FLYABLE}.")
    lines.append("Sources:")
    lines += [_source_line(status, tz) for status in assessment.source_statuses]
    if any(status.state is not SourceState.OK for status in assessment.source_statuses):
        lines.append(DO_NOT_FLY)
    if verdict_segments:
        lines.append("Segments:")
        lines += _segments_text(verdict_segments, tz)
    if outlook:
        lines.append("OUTLOOK hours (no verdict):")
        lines += [_block_line(block, tz) for block in outlook]
    if too_far:
        lines.append(f"Hours too far out ({TOO_FAR_WORD}, no verdict):")
        lines += [_block_line(block, tz) for block in too_far]
    if assumptions:
        lines.append("Assumptions:")
        lines += [f"- {text}" for text in assumptions]

    artifact = AssessArtifact(
        tz=tz,
        window_start=window.start_utc,
        window_end=window.end_utc,
        window_assumed=window.assumed,
        overlays=given.overlays,
        segments=every_segment,
        blocks=[*outlook, *too_far],
        overall=overall,
        overall_tier=assessment.overall_tier.value,
        flyable=spans,
        counts=counts,
        sources=[source_info_at(status, fetched) for status in assessment.source_statuses],
        pib_issued_at=None if notams_status is None else notams_status.issued_at,
        assumptions=assumptions,
        fetched_at=fetched,
        data_as_of=data_as_of,
    )
    return "\n".join(lines), artifact


# --- the tool --------------------------------------------------------------------------------------------------------


def _reader_status(prepared: Prepared) -> dict[str, str]:
    """The NOTAMs the reader was asked about and gave no reading for, as the two words `assess` takes. Anything else the thread's store holds counts as unavailable."""
    sources = prepared.sources
    known = {ReaderStatus.FAILED.value, ReaderStatus.UNAVAILABLE.value}
    return {notam_id: (status.value if status.value in known else ReaderStatus.UNAVAILABLE.value) for notam_id, status in sources.unread.items() if notam_id not in sources.readings}


def _overlays(assessment: Assessment, notams: Mapping[str, ParsedNotam], prepared: Prepared) -> list[OverlayFeature]:
    """The LZ circle and one feature for each NOTAM that bears on a segment, shaped by `shapes.notam_geometry` as `get_notams` shapes them.

    A NOTAM's action is the first it has, in time order. Its text is the raw E text as published; the redaction boundary numbers its phones.
    """
    overlays = [lz_radius_feature(prepared.lat, prepared.lon)]
    first: dict[str, NotamActionResult] = {}
    for hour in assessment.hours:
        for segment in hour.segments:
            for action in segment.notam_actions:
                first.setdefault(action.notam_id, action)
    for notam_id, action in first.items():
        parsed = notams.get(notam_id)
        if parsed is None:
            continue
        geometry, approx = notam_geometry(parsed)
        properties = OverlayProperties(id=notam_id, kind="notam", action=action.action.value, bearing=action.bearing_deg, text=parsed.item_e, approx=approx)
        overlays.append(OverlayFeature(geometry=geometry, properties=properties))  # type: ignore[arg-type]  # a dict the discriminator checks
    return overlays


def _weather_facts(assessment: Assessment, prepared: Prepared) -> dict[tuple[datetime, str], Finding]:
    """The weather blockers of each VERDICT hour as `weather.check_row` found them, keyed by (UTC hour, text).

    `assess.assess` keeps only a blocker's words. Running `check_row` again over the same forecast rows gives the same findings with their rule, value and limit,
    so the artifact can carry the numbers behind the words without reading them out of text.
    """
    hours = [hour.hour for hour in assessment.hours if hour.tier is Tier.VERDICT]
    forecast = prepared.sources.forecast
    if not hours or not usable(forecast.status):
        return {}
    rows = build_rows(forecast, hours[0], hours[-1] + ONE_HOUR)
    return {(row.hour, found.text): found for row in rows if row.hour in set(hours) for found in check_row(row) if found.kind == "blocker"}


def assess_answer(prepared: Prepared, source_cache: SourceCache) -> tuple[str, AssessArtifact]:
    """Read the NOTAMs the window needs, assess, and write the answer. The window is always the explicit UTC one (never `None`)."""
    sources, window = prepared.sources, prepared.window
    lz = (prepared.lat, prepared.lon)
    span = assess.assessed_span(window.start_utc, window.end_utc, prepared.now, prepared.live)
    chosen: list[ParsedNotam] = []
    if usable(sources.pib.status):
        chosen = reader_set(sources.pib.notams, lz, span)
        read_chosen(source_cache, prepared, span, chosen)
    book = PhoneBook.of(sources.pib.notams)
    readings = book.redact_contacts(dict(sources.readings))
    assessment = assess.assess(
        prepared.lat,
        prepared.lon,
        now=prepared.now,
        forecast=sources.forecast,
        airspace=sources.airspace,
        pib=sources.pib,
        readings=readings,
        reader_status=_reader_status(prepared),  # type: ignore[arg-type]  # the two words, as plain strings
        window=(window.start_utc, window.end_utc),
        live=prepared.live,
        tz=prepared.tz,
    )
    limited = [parsed.id for parsed in chosen if parsed.id in sources.limited]
    extra = [f"Reader limit reached: {', '.join(limited)} not read, treated as unread."] if limited else []
    overlays = _overlays(assessment, {parsed.id: parsed for parsed in sources.pib.notams}, prepared)
    given = Inputs(tz=prepared.tz, window=window, now=prepared.now, overlays=overlays, restore=Restore(book), extra_assumptions=extra, facts=_weather_facts(assessment, prepared))
    return build(assessment, given)


def build_assess(source_cache: SourceCache) -> BaseTool:
    """The assess tool over this source store."""

    @tool("assess", args_schema=WindowArgs, description=DESCRIPTION)
    def assess_tool(runtime: ToolRuntime, start: str | None = None, end: str | None = None) -> Command:
        redactor = Redactor(runtime.state.get("phone_map"))
        prepared = prepare(runtime, source_cache, redactor, start, end, progress.ASSESSING)
        if isinstance(prepared, str):
            return reply(runtime, prepared, None, redactor)
        text, artifact = assess_answer(prepared, source_cache)
        return reply(runtime, text, artifact, redactor)

    return assess_tool
