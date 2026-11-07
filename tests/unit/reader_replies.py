"""The fake model's answers for the Blue Mill chain test (GP6): readings.yaml's semantic fields, the numbers put back as placeholders.

readings.yaml is hand-written and holds real numbers. The model is never sent one, so what it would answer holds the placeholder of
the NOTAM's own redaction map instead. `reverse_map` is that step, done with the map and not with a second redaction: every
real number the map knows is swapped for its placeholder in every string of the answer.
"""

from typing import Any

import yaml
from reader_helpers import BLUE_MILL_PIB

from flysafelz.contracts import NotamReading, ParsedNotam
from flysafelz.notam_area import build_area
from flysafelz.reader.contract import ReaderOutput
from flysafelz.redact import redact_phones

READINGS_YAML = BLUE_MILL_PIB.with_name("readings.yaml")
CODE_BUILT = {"id", "geometry", "schedule"}  # id, area and item D are made by code, never by the model (F1)


def fixture_readings() -> dict[str, NotamReading]:
    return {entry["id"]: NotamReading(**entry) for entry in yaml.safe_load(READINGS_YAML.read_text(encoding="utf-8"))}


def reverse_map(value: Any, redaction: dict[str, str]) -> Any:
    """The value with each number in `redaction` ({placeholder: number as written}) swapped back for its placeholder, in every string."""
    if isinstance(value, str):
        for placeholder, number in redaction.items():
            value = value.replace(number, placeholder)
        return value
    if isinstance(value, list):
        return [reverse_map(item, redaction) for item in value]
    if isinstance(value, dict):
        return {key: reverse_map(item, redaction) for key, item in value.items()}
    return value


def model_reply(parsed: ParsedNotam, reading: NotamReading) -> dict[str, Any]:
    """What a good model answers for this NOTAM: the semantic fields of its readings.yaml entry, no geometry, numbers as placeholders."""
    fields = reading.model_dump(mode="json", include=set(ReaderOutput.model_fields))
    assert not CODE_BUILT & set(fields)
    reply = reverse_map(fields, redact_phones(parsed.item_e)[1])
    assert not redact_phones(yaml.safe_dump(reply))[1], "a real number is still in the model's answer"
    return reply


def area_mismatch(parsed: ParsedNotam, reading: NotamReading) -> str | None:
    """None when readings.yaml's area is what the area builder makes from the E text; else both shapes, for the failure message."""
    built, _note = build_area(parsed)
    if built is None and reading.geometry is None:
        return None
    if built is not None and reading.geometry is not None and built.kind == reading.geometry.kind:
        near = all(abs(a - b) <= 1e-6 for a, b in zip(_numbers(built), _numbers(reading.geometry), strict=True))
        if near:
            return None
    return f"{parsed.id}: readings.yaml has {reading.geometry!r} but the area builder makes {built!r} from the E text"


def _numbers(area: Any) -> list[float]:
    if area.kind == "circle":
        return [area.lat, area.lon, area.radius_nm]
    return [value for point in area.points for value in point]
