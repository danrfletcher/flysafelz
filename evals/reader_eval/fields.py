"""Per-field accuracy of a reading against Dan's label (G14), and the mean and worst run over several runs.

    field_accuracy(label_reading, reading) -> {field: True | False | None}
    aggregate(runs) -> {field: (mean, worst_run)}

Exact match: category, applies_to, contact kind, datum and unit (of the floor and of the ceiling, each), prohibitive_reason,
activates_designators (as a set, each normalised: D496A is EGD496A) and exemption `covers` (as a set; the quotes are not compared).
Heights: equal after unit conversion (FL x 100 -> ft) within `field_height_tolerance_ft`, converted at `site_elevation_ft`
and the reference pressure, the way notam_action() does. A null against a null is a match and a null against a value is not.
Geometry is code's (R3.7) and activity is free text: both come back as None, never True, and are left out of the mean.
An unread run (reading None) scores every scored field False.
"""

from collections.abc import Mapping, Sequence

from evals.reader_eval.settings import SETTINGS, EvalSettings, reference_pressure_hpa
from flysafelz.airspace import limit_qnh_ft
from flysafelz.contracts import Limit, NotamReading
from flysafelz.designators import normalise_designator

SCORED_FIELDS = (
    "category",
    "applies_to",
    "contact_kind",
    "floor_datum",
    "floor_unit",
    "floor_height",
    "ceiling_datum",
    "ceiling_unit",
    "ceiling_height",
    "prohibitive_reason",
    "activates_designators",
    "exemption_covers",
)
NOT_SCORED_FIELDS = ("geometry", "activity")
FIELDS = SCORED_FIELDS + NOT_SCORED_FIELDS


def _same_limit_part(label: Limit | None, read: Limit | None, part: str) -> bool:
    """Datum or unit: both null is a match, one null is not, otherwise the values must be equal."""
    if label is None or read is None:
        return label is None and read is None
    return getattr(label, part) is getattr(read, part)


def _same_height(label: Limit | None, read: Limit | None, settings: EvalSettings) -> bool:
    """Equal after conversion to ft QNH, to within the tolerance. Null against null matches; null against a value does not."""
    if label is None or read is None:
        return label is None and read is None
    pressure = reference_pressure_hpa()
    label_ft = limit_qnh_ft(label, settings.site_elevation_ft, pressure)
    read_ft = limit_qnh_ft(read, settings.site_elevation_ft, pressure)
    return abs(label_ft - read_ft) <= settings.field_height_tolerance_ft


def _designators(reading: NotamReading) -> set[str]:
    return {found for found in map(normalise_designator, reading.activates_designators) if found}


def field_accuracy(label_reading: NotamReading, reading: NotamReading | None, settings: EvalSettings = SETTINGS) -> dict[str, bool | None]:
    """For each field of FIELDS: did the reading get it right? None for the fields that are not scored. An unread run is all False."""
    if reading is None:
        return {name: None if name in NOT_SCORED_FIELDS else False for name in FIELDS}
    label = label_reading
    return {
        "category": label.category is reading.category,
        "applies_to": label.applies_to is reading.applies_to,
        "contact_kind": label.contact.kind is reading.contact.kind,
        "floor_datum": _same_limit_part(label.floor, reading.floor, "datum"),
        "floor_unit": _same_limit_part(label.floor, reading.floor, "unit"),
        "floor_height": _same_height(label.floor, reading.floor, settings),
        "ceiling_datum": _same_limit_part(label.ceiling, reading.ceiling, "datum"),
        "ceiling_unit": _same_limit_part(label.ceiling, reading.ceiling, "unit"),
        "ceiling_height": _same_height(label.ceiling, reading.ceiling, settings),
        "prohibitive_reason": label.prohibitive_reason is reading.prohibitive_reason,
        "activates_designators": _designators(label) == _designators(reading),
        "exemption_covers": {e.covers for e in label.exemptions} == {e.covers for e in reading.exemptions},
        "geometry": None,
        "activity": None,
    }


def aggregate(runs: Sequence[Mapping[str, bool | None] | Sequence[Mapping[str, bool | None]]]) -> dict[str, tuple[float, float]]:
    """{field: (mean, worst_run)} over the runs, for every field that was scored in at least one run.

    A run is the results of one pass: a list of field_accuracy dicts, one per item (a bare dict stands for a run of one item).
    A run's accuracy for a field is the fraction of its items that got it right (items where the field is None are left out).
    The mean is the mean of the runs' accuracies and the worst run is the lowest of them.
    """
    per_run: dict[str, list[float]] = {}
    for run in runs:
        items = [run] if isinstance(run, Mapping) else list(run)
        for name in FIELDS:
            scored = [bool(item[name]) for item in items if item.get(name) is not None]
            if scored:
                per_run.setdefault(name, []).append(sum(scored) / len(scored))
    return {name: (sum(values) / len(values), min(values)) for name, values in per_run.items()}
