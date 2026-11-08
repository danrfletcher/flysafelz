"""What code adds to a checked answer, and what it puts back in it (PR-11.F1). Pure functions: no model, no network, no state.

    recheck_line(artifacts, now, tz, path)   "Based on data as of 07:30 GMT. Re-check the weather and NOTAMs when you get to the LZ." or None (FR8, FR9)
    limit_lines(artifacts)                   one line for each value this turn's `assess` put at or over a limit, or [] (I10)
    assessment_of(artifacts)                 the `assessment` output field: this turn's `assess` segments merged, and the worst overall (I7)
    rehydrate(text, phone_map)               each known `<PHONE_n>` in `text` replaced by its number (P1)
    rehydrate_assessment, rehydrate_overlays the same for the two other output fields that hold a phone number (I7)

The order of a turn's last step is: guard, then the limit lines and the re-check line, then rehydration. So the guard only ever sees placeholders,
the lines are code text the guard never reads, and the real numbers exist only in the three output fields `answer`, `assessment` and `overlays`
(P1, I7). `messages`, every tool message and every tool artifact keep their placeholders.

The re-check line (FR8). Added when a data tool (weather, airspace, notams or assess) ran this turn for a window that starts after the current hour
in `tz`. Not for a window that starts in the current hour or before it (the default window starts in the current hour), not when no data tool ran,
and never on a clarify or decline answer. FR9: the time in it is the oldest fetch time among the sources this turn's data tools used, in `tz` with
its abbreviation. A turn whose sources carry no fetch time at all gets the second sentence alone, because there is no time to name.

The limit lines (I10). Built from the `assess` artifacts only: every caution and weather blocker holds the rule, the value, the limit and the unit
the check used. The comparison is the rule's, from rules.yaml, and the position word is `contracts.position_word` of the true numbers, so a line can
not say "over" for a value that has not reached its limit. A value below its limit gets no line. One line for each rule: the most extreme value any
hour had. They are not added to a clarify or decline answer either.
"""

import logging
import re
from collections.abc import Iterable, Mapping, Sequence
from datetime import UTC, datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo

from flysafelz.agent.state import Assessment, OverlayFeature, Segment, VerdictValue
from flysafelz.agent.tools.artifacts import AssessArtifact, AssessSegment
from flysafelz.agent.tools.common import g, measured
from flysafelz.contracts import Tier, Verdict, position_word
from flysafelz.rules import RULES
from flysafelz.timetext import labelled_clock

log = logging.getLogger(__name__)

RECHECK_SENTENCE = "Re-check the weather and NOTAMs when you get to the LZ."
NO_LINE_PATHS = frozenset({"clarify", "decline"})
PLACEHOLDER = re.compile(r"<PHONE_\d+>")
ONE_HOUR = timedelta(hours=1)
# How a rule's comparison reads in a limit line: `>` is "above 6 mph", `>=` is "at or above 4 mph".
COMPARISON_WORDS = {">": "above", ">=": "at or above", "<": "below", "<=": "at or below"}
POSITIONS_SHOWN = {"over": "Over", "at": "At"}  # "under" gets no line
WORST_FIRST = (Verdict.NO_GO, Verdict.UNVERIFIED, Verdict.PILOT_DISCRETION)


# --- the re-check line (FR8, FR9) ------------------------------------------------------------------------------------


def fetch_time(artifact: Any) -> datetime | None:
    """When the source(s) behind one data tool's artifact were fetched: an `assess` artifact holds the oldest of its three, the others hold their one."""
    if isinstance(artifact, AssessArtifact):
        return artifact.fetched_at
    source = getattr(artifact, "source", None)
    return getattr(source, "fetched_at", None)


def recheck_line(artifacts: Iterable[Any], now: datetime, tz: str, path: str | None) -> str | None:
    """The re-check line for a turn whose data tools gave `artifacts`, or None when the rules above say there is none.

    `now` is the run's clock (timezone-aware) and `tz` the thread's zone, from state. `path` is the path the answer took (None: unknown).
    """
    if path in NO_LINE_PATHS:
        return None
    ran = list(artifacts)
    current_hour = now.astimezone(ZoneInfo(tz)).replace(minute=0, second=0, microsecond=0).astimezone(UTC)
    if not any(artifact.window_start.astimezone(UTC) >= current_hour + ONE_HOUR for artifact in ran):
        return None
    fetched = [moment for moment in map(fetch_time, ran) if moment is not None]
    if not fetched:
        return RECHECK_SENTENCE
    return f"Based on data as of {labelled_clock(min(fetched), tz)}. {RECHECK_SENTENCE}"


# --- the limit lines (I10) -------------------------------------------------------------------------------------------


def _reached(artifacts: Iterable[AssessArtifact]) -> dict[str, tuple[float, str, str, float, str, str, str]]:
    """For each rule an `assess` hour reached: (value, label, kind, limit, unit, comparison, position word) of its most extreme value, in first-seen order."""
    found: dict[str, tuple[float, str, str, float, str, str, str]] = {}
    for artifact in artifacts:
        for segment in artifact.segments:
            if segment.tier != Tier.VERDICT.value:
                continue
            checks = [(c.rule_id, c.label, c.value, c.limit, c.unit, "caution") for c in segment.cautions]
            checks += [(b.rule_id, b.label, b.value, b.limit, b.unit, "NO-GO") for b in segment.blockers if b.source == "weather"]
            for rule_id, label, value, limit, unit, kind in checks:
                rule = None if rule_id is None else getattr(RULES, rule_id, None)
                if rule_id is None or rule is None or label is None or value is None or limit is None or unit is None:
                    continue
                comparison = rule.comparison.value
                word = position_word(value, limit, comparison)
                if word not in POSITIONS_SHOWN:
                    continue
                kept = found.get(rule_id)
                past_by = value if comparison in {">", ">="} else -value
                if kept is None or past_by > (kept[0] if kept[5] in {">", ">="} else -kept[0]):
                    found[rule_id] = (value, label, kind, limit, unit, comparison, word)
    return found


def limit_lines(artifacts: Iterable[Any]) -> list[str]:
    """`Over limit: gust spread 6.04 mph (NO-GO above 6 mph).` for each value this turn's `assess` put at or over a limit; `At caution limit: ...` for a caution."""
    lines = []
    for value, label, kind, limit, unit, comparison, word in _reached(a for a in artifacts if isinstance(a, AssessArtifact)).values():
        head = f"{POSITIONS_SHOWN[word]} {'caution ' if kind == 'caution' else ''}limit"
        shown = measured(value, [(comparison, limit)])
        lines.append(f"{head}: {label} {shown} {unit} ({kind} {COMPARISON_WORDS[comparison]} {g(limit)} {unit}).")
    return lines


# --- the assessment output field (I7) --------------------------------------------------------------------------------


def _worst_overall(artifacts: Sequence[AssessArtifact]) -> VerdictValue | None:
    overalls = [a.overall for a in artifacts if a.overall is not None]
    if not overalls:
        return None
    worst = min(overalls, key=lambda o: (WORST_FIRST.index(Verdict(o.verdict)), not o.approaching_limits, not o.with_conditions))
    return VerdictValue(verdict=Verdict(worst.verdict), approaching_limits=worst.approaching_limits, with_conditions=worst.with_conditions, display=worst.display)


def _segment(segment: AssessSegment) -> Segment:
    return Segment(
        start=segment.start,
        end=segment.end,
        tz_abbr=segment.tz_abbr,
        tier=Tier(segment.tier),
        verdict=None if segment.verdict is None else Verdict(segment.verdict),
        approaching_limits=segment.approaching_limits,
        with_conditions=segment.with_conditions,
        display=segment.display,
        conditions=list(segment.conditions),
    )


def assessment_of(artifacts: Iterable[Any]) -> Assessment | None:
    """The `assessment` field from this turn's `assess` artifacts, or None when `assess` did not run.

    Segments of every call in start-time order; a later call's segment for the same span replaces the earlier one. The overall is the worst
    VERDICT-tier overall among the calls (NO-GO, then UNVERIFIED, then PILOT DISCRETION; with limits or conditions as the tie-break).
    """
    ran = [a for a in artifacts if isinstance(a, AssessArtifact)]
    if not ran:
        return None
    by_span = {(s.start, s.end): _segment(s) for artifact in ran for s in artifact.segments}
    segments = sorted(by_span.values(), key=lambda s: datetime.fromisoformat(s.start))
    return Assessment(segments=segments, overall=_worst_overall(ran))


# --- rehydration (P1, I7) --------------------------------------------------------------------------------------------


def rehydrate(text: str, phone_map: Mapping[str, str] | None) -> str:
    """`text` with each `<PHONE_n>` that `phone_map` knows replaced by its number. An unknown placeholder stays as it is and is logged (its name only)."""
    numbers = phone_map or {}

    def put_back(found: re.Match[str]) -> str:
        placeholder = found.group(0)
        if placeholder not in numbers:
            log.warning("%s is not in the thread's phone map; it is left in the text.", placeholder)
            return placeholder
        return numbers[placeholder]

    return PLACEHOLDER.sub(put_back, text)


def rehydrate_assessment(assessment: Assessment | None, phone_map: Mapping[str, str] | None) -> Assessment | None:
    """The assessment with the real numbers in its condition strings."""
    if assessment is None:
        return None
    segments = [segment.model_copy(update={"conditions": [rehydrate(c, phone_map) for c in segment.conditions]}) for segment in assessment.segments]
    return assessment.model_copy(update={"segments": segments})


def rehydrate_overlays(overlays: Sequence[OverlayFeature], phone_map: Mapping[str, str] | None) -> list[OverlayFeature]:
    """The overlays with the real numbers in each NOTAM's raw text; a feature with no placeholder in it is the same object."""
    out = []
    for feature in overlays:
        text = feature.properties.text
        if text is None or not PLACEHOLDER.search(text):
            out.append(feature)
            continue
        out.append(feature.model_copy(update={"properties": feature.properties.model_copy(update={"text": rehydrate(text, phone_map)})}))
    return out
