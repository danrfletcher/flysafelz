"""Harmful downgrades and over-caution: what a reader's mistakes would do to a pilot (G13, R3.1, R3.4, R3.7, R3.11, R3.12).

    score_item_run(parsed, label_reading, result, settings) -> RunScore

For one NOTAM and one run, `notam_action()` is called twice at each of about 25 spots (spots.py) in one hour (spots.eval_segment):
on the reader's result and on Dan's label. A result with a reading is judged on that reading. A result with none (failed or
unavailable) is judged as the pilot would be told: `notam_action(parsed, None, ...)`, which is UNREAD_OVER_LZ over the LZ and
a code-only action elsewhere (R3.1). Both sides use code's area and code's item D, whatever the reading carries (F1): the
model never decides where a NOTAM is or when it applies. Ground is `site_elevation_ft` and pressure the rules' reference
pressure on both sides, so the two actions differ only by the readings.

    harmful     the reading is milder than the label at some spot (named with the spot nearest the area's centre), or it
                lacks a danger area the label activates, or its top is more than `height_tolerance_ft` below the label's
                or its bottom more than that above it (after the E10 merge, as ft AMSL at the site's ground)
    over-caution  stricter than the label at some spot and milder at none, or an extra danger area, or a top or bottom that
                far on the safe side
    agreement   neither

Severity, strictest first: NO_FLY, UNREAD_OVER_LZ, CHECK_SUP, CONTACT_FIRST, AVOID_SECTOR, AWARENESS, AWARENESS_MAP, IGNORE. An unread
NOTAM is thus stricter than a label of CONTACT_FIRST (over-caution, not harm), and milder than NO_FLY (harm, "reader failed").
An unread result is judged on its actions only: it has no designators or heights to compare, and the airspace path adds the
danger-area condition for it anyway.

Each finding names the field that caused it. For an action it is found by putting each field of the label that differs from the
reading into the reading in turn and running the spot that flipped again: the fields that bring back the label's action are
named, else those that change the action at all, else every field that differs.

The timing check: a label that applies to paramotors (ALL_AIRCRAFT or NAMED_CLASSES, no PARAMOTOR_CLASS exemption) must give
something other than IGNORE at the centre in the eval segment. If it does not, the segment missed the NOTAM and the run is
flagged `segment_miss`, never counted as agreement.

Findings name the NOTAM, the actions, the spot and the field. They show contact kinds, never contact values or exemption quotes.
"""

import math
from dataclasses import dataclass, field
from enum import StrEnum

from evals.reader_eval.settings import SETTINGS, EvalSettings
from evals.reader_eval.spots import Spot, eval_segment, plan_spots
from flysafelz.contracts import AppliesTo, Datum, ExemptionCovers, Limit, LimitUnit, NotamAction, NotamReading, ParsedNotam
from flysafelz.designators import normalise_designator
from flysafelz.geo import distance_nm
from flysafelz.notam_action import ActionSegment, LandingZone, merged_limits, notam_action
from flysafelz.notam_area import Area, build_area
from flysafelz.reader.contract import ReaderResult

# Strictest first. A new NotamAction member must be placed here: severity_rank raises until it is (and the test fails).
SEVERITY: tuple[NotamAction, ...] = (
    NotamAction.NO_FLY,
    NotamAction.UNREAD_OVER_LZ,
    NotamAction.CHECK_SUP,
    NotamAction.CONTACT_FIRST,
    NotamAction.AVOID_SECTOR,
    NotamAction.AWARENESS,
    NotamAction.AWARENESS_MAP,
    NotamAction.IGNORE,
)
READER_FAILED = "reader failed"
# The fields of a reading that can change what notam_action() says, in the order they are tried and named.
ACTION_FIELDS = ("category", "applies_to", "exemptions", "contact", "prohibitive_reason", "floor", "ceiling")


def severity_rank(action: NotamAction) -> int:
    """0 for the strictest action; a larger number is milder."""
    return SEVERITY.index(action)


class FindingKind(StrEnum):
    ACTION = "action"
    DESIGNATORS = "designators"
    HEIGHT = "height"


@dataclass(frozen=True)
class Finding:
    """One thing a run got wrong, or got over-cautiously. `message` is the line without the NOTAM and run; `field` names the cause."""

    notam_id: str
    kind: FindingKind
    field: str
    message: str
    reader_failed: bool = False
    label_action: NotamAction | None = None
    read_action: NotamAction | None = None
    spot: Spot | None = None

    def line(self, run: int | None = None) -> str:
        """For example "J1458/26 run 2: NO_FLY → AVOID_SECTOR at the centre (applies_to UAS_ONLY vs ALL_AIRCRAFT)"."""
        who = self.notam_id if run is None else f"{self.notam_id} run {run}"
        return f"{who}: {self.message}"


@dataclass(frozen=True)
class RunScore:
    harmful: list[Finding] = field(default_factory=list)
    over_caution: list[Finding] = field(default_factory=list)
    reader_failed: bool = False
    segment_miss: bool = False  # the eval segment missed a NOTAM the label says applies to paramotors
    dropped_spots: int = 0  # candidate spots on the wrong side of the edge, left out
    spots: int = 0  # spots judged

    @property
    def agreement(self) -> bool:
        return not (self.harmful or self.over_caution or self.segment_miss)


# --- words ----------------------------------------------------------------------------------------------------------------------


def _limit_text(limit: Limit | None) -> str:
    if limit is None:
        return "none stated"
    if limit.datum is Datum.SFC:
        return "SFC"
    if limit.unit is LimitUnit.FL:
        return f"FL{limit.value:03.0f}"
    return f"{limit.value:,.0f} ft {limit.datum.value}"


def _value_text(name: str, reading: NotamReading) -> str:
    value = getattr(reading, name)
    if name in ("floor", "ceiling"):
        return _limit_text(value)
    if name == "contact":
        return value.kind.value
    if name == "exemptions":
        return "[" + ", ".join(sorted({exemption.covers.value for exemption in value})) + "]"
    return value.value


def _feet(qnh: float) -> str:
    return "no upper limit" if math.isinf(qnh) else f"{qnh:,.0f} ft"


# --- one run --------------------------------------------------------------------------------------------------------------------


def _coded(reading: NotamReading, parsed: ParsedNotam, area: Area) -> NotamReading:
    """The reading with code's area and code's item D in place of whatever it carried (F1, R3.7)."""
    return reading.model_copy(update={"geometry": area, "schedule": parsed.item_d or ""})


def _applies_to_paramotors(label: NotamReading) -> bool:
    if label.applies_to in (AppliesTo.UAS_ONLY, AppliesTo.AERODROME_USERS):
        return False
    return not any(exemption.covers is ExemptionCovers.PARAMOTOR_CLASS for exemption in label.exemptions)


def _differing(label: NotamReading, reading: NotamReading) -> list[str]:
    return [name for name in ACTION_FIELDS if getattr(label, name) != getattr(reading, name)]


@dataclass(frozen=True)
class _Judging:
    parsed: ParsedNotam
    label: NotamReading
    reading: NotamReading | None
    segment: ActionSegment
    settings: EvalSettings

    def lz(self, spot: Spot) -> LandingZone:
        return LandingZone(lat=spot.lat, lon=spot.lon, elevation_ft=self.settings.site_elevation_ft)

    def action(self, reading: NotamReading | None, spot: Spot) -> NotamAction:
        return notam_action(self.parsed, reading, self.lz(spot), self.segment).action


def _cause(judging: _Judging, spot: Spot, label_action: NotamAction, read_action: NotamAction) -> tuple[str, str]:
    """(the field names, the detail) of the fields that account for a flip at this spot. See the module text."""
    reading = judging.reading
    assert reading is not None  # an unread result has no fields to blame
    differing = _differing(judging.label, reading)
    trials = {name: judging.action(reading.model_copy(update={name: getattr(judging.label, name)}), spot) for name in differing}
    names = [name for name in differing if trials[name] is label_action] or [name for name in differing if trials[name] is not read_action] or differing
    if not names:
        return "other", "no field of the reading differs from the label's"
    return " + ".join(names), "; ".join(f"{name} {_value_text(name, reading)} vs {_value_text(name, judging.label)} labelled" for name in names)


def _action_finding(judging: _Judging, centre: Spot, rows: list[tuple[Spot, NotamAction, NotamAction]]) -> Finding:
    """The finding for the spot of `rows` nearest the centre."""
    spot, label_action, read_action = min(rows, key=lambda row: distance_nm(centre.lat, centre.lon, row[0].lat, row[0].lon))
    failed = judging.reading is None
    if failed:
        name, detail = READER_FAILED, READER_FAILED
    else:
        name, detail = _cause(judging, spot, label_action, read_action)
    message = f"{label_action.value} → {read_action.value} at {spot.describe()} ({detail})"
    return Finding(judging.parsed.id, FindingKind.ACTION, name, message, failed, label_action, read_action, spot)


def _designator_findings(parsed: ParsedNotam, label: NotamReading, reading: NotamReading) -> tuple[list[Finding], list[Finding]]:
    """(harmful, over-cautious): a danger area the label activates and the reading lacks, and one the reading adds."""

    def normalised(source: NotamReading) -> set[str]:
        return {found for found in map(normalise_designator, source.activates_designators) if found}

    wanted, given = normalised(label), normalised(reading)
    harmful = [Finding(parsed.id, FindingKind.DESIGNATORS, "designators", f"{name} not activated") for name in sorted(wanted - given)]
    cautious = [Finding(parsed.id, FindingKind.DESIGNATORS, "designators", f"{name} activated but not labelled") for name in sorted(given - wanted)]
    return harmful, cautious


def _difference(read: float, labelled: float) -> float:
    """read - labelled, with two equal heights (two infinities too) 0."""
    return 0.0 if read == labelled else read - labelled


def _height_findings(judging: _Judging, centre: Spot) -> tuple[list[Finding], list[Finding]]:
    """(harmful, over-cautious) heights, as numbers: the E10-merged floor and ceiling of both, ft AMSL at the site's ground (R3.12)."""
    parsed, tolerance = judging.parsed, judging.settings.height_tolerance_ft
    lz = judging.lz(centre)
    label_floor, label_top = merged_limits(parsed, judging.label, lz, judging.segment)
    read_floor, read_top = merged_limits(parsed, judging.reading, lz, judging.segment)
    ground = f"(ft AMSL, ground {judging.settings.site_elevation_ft:g} ft)"
    harmful: list[Finding] = []
    cautious: list[Finding] = []
    top, bottom = _difference(read_top, label_top), _difference(read_floor, label_floor)
    for side, name, gap, read, labelled in (("top", "ceiling", top, read_top, label_top), ("bottom", "floor", -bottom, read_floor, label_floor)):
        # gap is how far the reading's side is on the safe side of the label's: a higher top, or a lower bottom
        message = f"{side} {_feet(read)} vs {_feet(labelled)} labelled {ground}"
        if gap < -tolerance:
            harmful.append(Finding(parsed.id, FindingKind.HEIGHT, name, message))
        elif gap > tolerance:
            cautious.append(Finding(parsed.id, FindingKind.HEIGHT, name, message))
    return harmful, cautious


def score_item_run(parsed: ParsedNotam, label_reading: NotamReading, result: ReaderResult, settings: EvalSettings = SETTINGS) -> RunScore:
    """Score one run of the reader on one NOTAM against Dan's label. See the module text."""
    area = build_area(parsed)[0]
    label = _coded(label_reading, parsed, area)
    reading = None if result.reading is None else _coded(result.reading, parsed, area)
    plan = plan_spots(area, parsed, settings)
    segment = eval_segment(parsed, label, settings)
    judging = _Judging(parsed, label, reading, segment, settings)
    centre = plan.spots[0]

    rows = [(spot, judging.action(label, spot), judging.action(reading, spot)) for spot in plan.spots]
    milder = [(spot, labelled, read) for spot, labelled, read in rows if severity_rank(read) > severity_rank(labelled)]
    stricter = [(spot, labelled, read) for spot, labelled, read in rows if severity_rank(read) < severity_rank(labelled)]

    harmful: list[Finding] = []
    over_caution: list[Finding] = []
    if milder:
        harmful.append(_action_finding(judging, centre, milder))
    elif stricter:
        over_caution.append(_action_finding(judging, centre, stricter))
    if reading is not None:
        designators_harmful, designators_cautious = _designator_findings(parsed, label, reading)
        heights_harmful, heights_cautious = _height_findings(judging, centre)
        harmful += designators_harmful + heights_harmful
        over_caution += designators_cautious + heights_cautious

    miss = _applies_to_paramotors(label) and rows[0][1] is NotamAction.IGNORE
    return RunScore(harmful, over_caution, reading is None, miss, plan.dropped, len(plan.spots))
