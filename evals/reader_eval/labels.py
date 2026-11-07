"""Dan's labels for the reader eval, and which of them can be scored (E13, R3.9).

    load_labels(path, items) -> LabelSet

`labels.yaml` is written by Dan alone and only ever read here (F4). Its shape:

    labels:
      - id: "X0001/26"
        e_hash: "<the sheet's e_hash for that NOTAM>"
        label:
          category: prohibitive
          activity: restricted area
          floor: SFC                  # or {value: 0, unit: ft, datum: SFC}; "700FT AGL" and "FL065" are read too
          ceiling: {value: 700, unit: ft, datum: AGL}
          applies_to: ALL_AIRCRAFT
          exemptions: []              # or [{quote: "...", covers: CONDITIONAL}], quoted as on the redacted sheet
          contact: {kind: phone, value: "<PHONE_1>"}   # or none
          activates_designators: []
          prohibitive_reason: none

The sheet's own `items:` entries with their `label:` blocks filled in are read the same way. An entry is:

    unknown   its id is not on the sheet
    stale     its e_hash is not the sheet's e_hash for that id: the NOTAM text changed since it was labelled (not scored)
    partial   some of the nine semantic keys are unfilled (not scored; `missing` lists them)
    complete  every key is filled, and it becomes a NotamReading like a model's: code's area and item D merged in

`none`, `[]` and `null` are explicit answers where they mean something (no floor, no exemptions, no contact, no designators,
no prohibitive reason). A missing key is unfilled. So is a blank category, activity or applies_to, because none of those has
a "none" answer, and so is a value that is not one of the allowed ones (`problems` says which). A sheet item with nothing
filled in is therefore partial, never complete. A label's exemption quote is kept as written on the redacted sheet and is
checked against the redacted E text later, with no rehydration (G5e).
"""

import re
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

import yaml

from flysafelz.contracts import AppliesTo, Category, Contact, ContactKind, Datum, Exemption, Limit, LimitUnit, NotamReading, ParsedNotam, ProhibitiveReason
from flysafelz.notam_area import build_area
from flysafelz.reader.contract import ReaderOutput, merge_reading

LABEL_KEYS = ("category", "activity", "floor", "ceiling", "applies_to", "exemptions", "contact", "activates_designators", "prohibitive_reason")
# A blank here is unfilled, not an answer: there is no "none" category, activity or applies_to.
BLANK_IS_UNFILLED = ("category", "activity", "applies_to")
_LIMIT_TEXT = re.compile(r"(?:(?P<sfc>SFC|GND)|FL\s?(?P<fl>\d+(?:\.\d+)?)|(?P<ft>\d+(?:\.\d+)?)\s?FT\s?(?P<datum>AGL|AMSL))", re.IGNORECASE)


class LabelError(Exception):
    """labels.yaml is not usable (not YAML, the wrong shape, an entry without an id and e_hash, a repeated id)."""


class LabelStatus(StrEnum):
    COMPLETE = "complete"
    PARTIAL = "partial"
    STALE = "stale"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class Label:
    """One entry of labels.yaml. output is set only for a complete label."""

    notam_id: str
    e_hash: str
    status: LabelStatus
    missing: tuple[str, ...] = ()
    problems: tuple[str, ...] = ()
    output: ReaderOutput | None = None

    def reading(self, parsed: ParsedNotam) -> NotamReading:
        """The label as a reading, built the way a model's is: its fields plus code's id, area and item D (merge_reading)."""
        if self.output is None:
            raise ValueError(f"{self.notam_id}: a {self.status} label has no reading")
        return merge_reading(parsed, build_area(parsed)[0], self.output)


@dataclass(frozen=True)
class LabelSet:
    """Every label in the file, and the sheet's items that have none."""

    labels: tuple[Label, ...]
    unlabelled: tuple[str, ...]

    def _with(self, status: LabelStatus) -> tuple[Label, ...]:
        return tuple(label for label in self.labels if label.status is status)

    @property
    def complete(self) -> tuple[Label, ...]:
        return self._with(LabelStatus.COMPLETE)

    @property
    def partial(self) -> tuple[Label, ...]:
        return self._with(LabelStatus.PARTIAL)

    @property
    def stale(self) -> tuple[Label, ...]:
        return self._with(LabelStatus.STALE)

    @property
    def unknown(self) -> tuple[Label, ...]:
        return self._with(LabelStatus.UNKNOWN)

    @property
    def scored(self) -> int:
        """How many items will be scored."""
        return len(self.complete)

    @property
    def skipped(self) -> int:
        """How many of the sheet's items are not scored: unlabelled, partly labelled or stale. An unknown label is no item of the sheet."""
        return len(self.unlabelled) + len(self.partial) + len(self.stale)

    def get(self, notam_id: str) -> Label | None:
        return next((label for label in self.labels if label.notam_id == notam_id), None)


# --- one key at a time --------------------------------------------------------------------------------------------------------


def _none(value: object) -> bool:
    return value is None or (isinstance(value, str) and value.strip().casefold() == "none")


def _text(value: object) -> str:
    if not isinstance(value, str):
        raise TypeError("must be text")
    return value


def _limit(value: object) -> Limit | None:
    if _none(value):
        return None
    if isinstance(value, Mapping):
        return Limit.model_validate(dict(value))
    match = _LIMIT_TEXT.fullmatch(value.strip()) if isinstance(value, str) else None
    if match is None:
        raise ValueError(f"cannot read the height {value!r}: write SFC, 700FT AGL, 800FT AMSL, FL065 or value/unit/datum")
    if match["sfc"]:
        return Limit(value=0, unit=LimitUnit.FT, datum=Datum.SFC)
    if match["fl"]:
        return Limit(value=float(match["fl"]), unit=LimitUnit.FL, datum=Datum.AMSL)
    return Limit(value=float(match["ft"]), unit=LimitUnit.FT, datum=Datum(match["datum"].upper()))


def _exemptions(value: object) -> list[Exemption]:
    if _none(value):
        return []
    if not isinstance(value, list):
        raise TypeError("must be a list of {quote, covers}")
    return [Exemption.model_validate(entry) for entry in value]


def _contact(value: object) -> Contact:
    if _none(value):
        return Contact(kind=ContactKind.NONE)
    if isinstance(value, Mapping):
        return Contact.model_validate(dict(value))
    return Contact(kind=ContactKind(value))


def _designators(value: object) -> list[str]:
    if _none(value):
        return []
    if not isinstance(value, list) or not all(isinstance(entry, str) for entry in value):
        raise TypeError("must be a list of designators")
    return list(value)


def _reason(value: object) -> ProhibitiveReason:
    return ProhibitiveReason.NONE if _none(value) else ProhibitiveReason(value)


_CONVERTERS: dict[str, Callable[[Any], object]] = {
    "category": Category,
    "activity": _text,
    "floor": _limit,
    "ceiling": _limit,
    "applies_to": AppliesTo,
    "exemptions": _exemptions,
    "contact": _contact,
    "activates_designators": _designators,
    "prohibitive_reason": _reason,
}


def _read_block(block: object) -> tuple[ReaderOutput | None, tuple[str, ...], tuple[str, ...]]:
    """(the fields as a ReaderOutput when every key is filled, the unfilled keys, a sentence for each impossible value)."""
    given = block if isinstance(block, Mapping) else {}
    values: dict[str, object] = {}
    missing: list[str] = []
    problems: list[str] = []
    for key in LABEL_KEYS:
        if key not in given or (key in BLANK_IS_UNFILLED and given[key] is None):
            missing.append(key)
            continue
        try:
            values[key] = _CONVERTERS[key](given[key])
        except (ValueError, TypeError) as exc:  # a Pydantic ValidationError is a ValueError
            missing.append(key)
            problems.append(f"{key}: {str(exc).splitlines()[0]}")
    if missing:
        return None, tuple(missing), tuple(problems)
    # Dan's labels have no quote (nine keys, not scored): the reading carries None, which is what merge_reading gives the reason `none` anyway.
    return ReaderOutput.model_validate({**values, "prohibitive_quote": None}), (), ()


# --- the file -----------------------------------------------------------------------------------------------------------------


def _entries(path: Path) -> list[Mapping[str, Any]]:
    """The file's label entries. A missing or empty file has none."""
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return []
    except OSError as exc:
        raise LabelError(f"cannot read {path}: {exc.strerror or exc}") from exc
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise LabelError(f"{path} is not valid YAML: {exc}") from exc
    if data is None:
        return []
    entries = data.get("labels", data.get("items")) if isinstance(data, dict) else None
    if entries is None and isinstance(data, dict):
        return []  # a mapping with neither key holds no labels
    if not isinstance(entries, list):
        raise LabelError(f"{path}: `labels:` must be a list")
    for position, entry in enumerate(entries, start=1):
        if not isinstance(entry, Mapping) or not isinstance(entry.get("id"), str) or not isinstance(entry.get("e_hash"), str):
            raise LabelError(f"{path}: entry {position} needs an id and an e_hash")
    return entries


def load_labels(path: Path, items: Iterable[Mapping[str, Any]]) -> LabelSet:
    """Read the labels and sort them against the sheet's items (each has an `id` and an `e_hash`).

    A missing, empty or label-free file gives no labels and every item unlabelled. Raises LabelError for a file that is not
    usable or names one id twice.
    """
    sheet = {item["id"]: item["e_hash"] for item in items}
    labels: list[Label] = []
    for entry in _entries(path):
        notam_id, hashed = entry["id"], entry["e_hash"]
        if any(label.notam_id == notam_id for label in labels):
            raise LabelError(f"{path}: {notam_id} is labelled more than once")
        if notam_id not in sheet:
            labels.append(Label(notam_id, hashed, LabelStatus.UNKNOWN))
        elif hashed != sheet[notam_id]:
            labels.append(Label(notam_id, hashed, LabelStatus.STALE))
        else:
            output, missing, problems = _read_block(entry.get("label"))
            labels.append(Label(notam_id, hashed, LabelStatus.PARTIAL if missing else LabelStatus.COMPLETE, missing, problems, output))
    labelled = {label.notam_id for label in labels}
    return LabelSet(tuple(labels), tuple(notam_id for notam_id in sheet if notam_id not in labelled))
