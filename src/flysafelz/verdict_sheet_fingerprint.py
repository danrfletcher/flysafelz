"""The per-entry fingerprint of the verdict sheet (4b, PR-5.F1, G6): what an entry was built from, so a rerun can tell it is out of date.

    fingerprint(example, rows, case_dir, mode)   {part name: sha256 hex}: the parts that make up the entry's fingerprint
    changed_parts(stored, current)               which parts differ ('inputs', 'rows', a fixture file name)
    sealed(parts) / unseal(stored, mode)         the stored form: the parts plus a seal over them; the parts back, or None if unreadable

A fingerprint is a few sha256 hashes, kept apart so that `--check` can name what changed:
    inputs          lat, lon, now, window as written (null stays null), reference.path, fixtures.drop, case name
    rows            the row boundaries as built: start, end and tier of every row
    <file name>     draft entries only: the bytes of the case's weather.json, airspace.json, pib.xml, readings.yaml
So a blind entry goes stale only when its inputs or its row edges change: a readings.yaml edit that moves no row edge
leaves it alone. A draft entry also goes stale when any of the four fixture files changes by one byte.
Code and src/flysafelz/rules.yaml are not inputs: a change to either would otherwise stale every entry.

The stored fingerprint also carries a `seal`, a sha256 over its parts. A hash edited by hand, even to another valid hex
string, no longer matches the seal, so the entry reads as unreadable instead of as a change to the example.

Hashes are over a canonical serialisation (JSON, sorted keys, ISO-UTC times), so they do not move with YAML formatting.
The mode is not hashed: it is stored in the entry, and a mode that no longer matches the example's is reported as 'inputs'.
"""

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

from flysafelz.clients.pib_source import utc_text

if TYPE_CHECKING:
    from flysafelz.verdict_sheet import Example

PART_INPUTS = "inputs"
PART_ROWS = "rows"
FIXTURE_FILES = ("weather.json", "airspace.json", "pib.xml", "readings.yaml")
ABSENT = "absent"  # a fixture file the case does not have (readings.yaml is optional in draft mode)
MODE_BLIND = "blind"
MODE_DRAFT = "draft"
SEAL = "seal"
_HEX = re.compile(r"[0-9a-f]{64}")


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def inputs_part(example: "Example") -> str:
    """The hash of what the example says: lat, lon, now, window as written, reference.path, fixtures.drop, case name."""
    window = None if example.window is None else [utc_text(example.window[0]), utc_text(example.window[1])]
    return _sha256(
        _canonical(
            {
                "lat": example.lat,
                "lon": example.lon,
                "now": None if example.now is None else utc_text(example.now),
                "window": window,
                "path": example.path,
                "drop": sorted(example.drop),
                "case": example.case,
            }
        )
    )


def rows_part(rows: Sequence[Mapping[str, Any]]) -> str:
    """The hash of the row boundaries as built: [start, end, tier or null] for each row, in order."""
    return _sha256(_canonical([[row["start"], row["end"], row.get("tier")] for row in rows]))


def file_parts(case_dir: Path) -> dict[str, str]:
    """The hash of each fixture file's raw bytes, by file name ('absent' for a file the case does not have)."""
    parts: dict[str, str] = {}
    for name in FIXTURE_FILES:
        try:
            parts[name] = _sha256((case_dir / name).read_bytes())
        except FileNotFoundError:
            parts[name] = ABSENT
    return parts


def fingerprint(example: "Example", rows: Sequence[Mapping[str, Any]], case_dir: Path, mode: str) -> dict[str, str]:
    """The fingerprint of an entry built for `example` in `mode` ('blind' or 'draft') from the case in case_dir."""
    parts = {PART_INPUTS: inputs_part(example), PART_ROWS: rows_part(rows)}
    if mode == MODE_DRAFT:
        parts.update(file_parts(case_dir))
    return parts


def expected_names(mode: object) -> tuple[str, ...] | None:
    """The part names a stored fingerprint of this mode must have; None for a mode the sheet does not know."""
    if mode == MODE_BLIND:
        return (PART_INPUTS, PART_ROWS)
    return (PART_INPUTS, PART_ROWS, *FIXTURE_FILES) if mode == MODE_DRAFT else None


def is_readable(stored: object, mode: object) -> bool:
    """True when `stored` is a mapping with exactly the parts of this mode and a sha256 (or 'absent') for each."""
    names = expected_names(mode)
    if names is None or not isinstance(stored, dict) or set(stored) != set(names):
        return False
    return all(isinstance(v, str) and (_HEX.fullmatch(v) is not None or (k in FIXTURE_FILES and v == ABSENT)) for k, v in stored.items())


def sealed(parts: Mapping[str, str]) -> dict[str, str]:
    """The form kept in the sheet: the parts, then `seal`, a sha256 over them (so a hand edit of any part shows)."""
    return {**parts, SEAL: _sha256(_canonical(dict(parts)))}


def unseal(stored: object, mode: object) -> dict[str, str] | None:
    """The parts of a stored fingerprint, or None when it is unreadable: not a mapping, wrong parts for the mode, no seal or a seal that does not match."""
    if not isinstance(stored, dict) or SEAL not in stored:
        return None
    parts = {k: v for k, v in stored.items() if k != SEAL}
    if not is_readable(parts, mode) or stored[SEAL] != sealed(parts)[SEAL]:
        return None
    return parts


def changed_parts(stored: Mapping[str, str], current: Mapping[str, str]) -> list[str]:
    """The names of the parts whose hash differs, in fingerprint order (inputs, rows, then the fixture files).

    Compares only the parts `current` has, so a caller that could compute only some of them (the rows need a
    successful build) still gets the changes it can see.
    """
    return [name for name, value in current.items() if stored.get(name) != value]
