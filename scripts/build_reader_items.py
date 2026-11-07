"""Build the NOTAM reader labelling sheet, evals/reader/items.yaml. Offline: no model, no network.

    uv run python scripts/build_reader_items.py [--selection FILE] [--pibs DIR] [--out FILE]

Reads the committed selection (evals/reader/selection.yaml: rank, archive commit sha, NOTAM id, type for 40 NOTAMs) and the
committed PIBs (evals/reader/pibs/<sha>.xml), and writes one item per NOTAM in rank order: the E text with every phone number
swapped for <PHONE_n>, the fields code parsed (Q-line, A, B/C, D, F/G), the area code built from the E text (G3), the e_hash,
and a blank `label:` block for Dan to fill in. Nothing here writes a label: the only file this script writes is the sheet.

A bad selection (a duplicate rank, a rank outside 1-40, an unknown type, a PIB or a NOTAM that is not in the committed data)
exits 1 and names the item. The sheet is built in memory, checked, and only then written, in one step, so a failed run leaves
the file as it was. Two runs on the same inputs give the same bytes.
"""

import argparse
import json
import logging
import re
import sys
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import NoReturn

import yaml

from flysafelz.contracts import Circle, ParsedNotam
from flysafelz.notam_area import build_area
from flysafelz.pib import PibError, read_notams
from flysafelz.redact import e_hash, redact_phones

ROOT = Path(__file__).resolve().parents[1]
READER = ROOT / "evals" / "reader"
log = logging.getLogger("build_reader_items")

ITEM_COUNT = 40
TYPES = (
    "rat_all_aircraft_paramotor_trap",
    "rat_uas_only",
    "tda_trigger_by_place",
    "tda_with_designator",
    "drone_ops_with_phone",
    "model_flying",
    "parachuting",
    "tethered_balloon",
    "glider_winch_launching",
    "air_display_fly_in",
    "aerodrome_drone",
    "aerodrome_fireworks_laser",
    "aerodrome_users_only",
    "crane_obstacle",
    "tmz",
    "military_exercise_series_q",
    "multi_height_agl_amsl",
    "polygon_area",
    "radius_in_metres",
    "no_coordinates_in_e",
    "radio_only_contact",
    "email_contact",
    "emergency_services_exemption",
    "survey_inspection",
)
LABEL_FIELDS = (
    "category",
    "activity",
    "floor",
    "ceiling",
    "applies_to",
    "exemptions",
    "contact",
    "activates_designators",
    "prohibitive_reason",
)
HEADER = """\
# NOTAM reader labelling sheet: 40 real NOTAMs from the 4 archived PIBs, in priority order.
# Built by scripts/build_reader_items.py from evals/reader/selection.yaml and evals/reader/pibs/. This repo copy is never
# edited by hand and never holds labels: you fill the `label:` blocks in the vault copy, and Vector adds your labels to the
# repo afterwards, keyed by id and e_hash (a label whose e_hash no longer matches is stale).
# Ranks 1-24 cover each of the 24 types once; ranks 25-40 give a second example of the types of ranks 1-16, in the same
# order. Stopping at rank 24 still covers every type.
#
# E is the NOTAM text as published, with each phone number replaced by <PHONE_n> (numbered per NOTAM) and the spaces at
# the ends of lines removed.
# The coded fields are what code parsed. `area` is what code built from E: a circle, a polygon, or the Q-circle when E
# gives no usable shape. You label what the NOTAM says, never a geometry: there is no geometry key.
#
# Datum rule: when E gives a height twice (for example 400FT AGL and 534FT AMSL), label the AMSL value.
#
# Fill each `label:` block: category, activity, floor, ceiling, applies_to, exemptions, contact, activates_designators,
# prohibitive_reason.
"""
_SHA = re.compile(r"[0-9a-f]{40}")
_DIGIT_RUN = re.compile(r"\d{10,11}")  # a phone number's digits once whitespace is gone (the leak check, below)
_PLAIN_BLOCK = re.compile(r"[^\x00-\x08\x0b-\x1f\x7f-\x9f\u2028\u2029\ufeff\r\t]*")


class SheetError(Exception):
    """The selection or a committed PIB is wrong. The message names the item."""


def _fail(message: str) -> NoReturn:
    raise SheetError(message)


# --- the selection ---------------------------------------------------------------------------


def read_selection(path: Path) -> list[dict]:
    """The selection items in rank order, every one checked. Raises SheetError."""
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        _fail(f"cannot read the selection {path}: {exc}")
    entries = data.get("items") if isinstance(data, dict) else None
    if not isinstance(entries, list):
        _fail(f"{path} has no `items:` list")
    by_rank: dict[int, dict] = {}
    seen_pairs: dict[tuple[str, str], int] = {}
    for position, entry in enumerate(entries, start=1):
        label = f"selection entry {position}"
        if not isinstance(entry, dict) or set(entry) != {"rank", "sha", "id", "type"}:
            _fail(f"{label}: needs exactly rank, sha, id and type")
        rank, sha, notam_id, kind = entry["rank"], entry["sha"], entry["id"], entry["type"]
        label = f"rank {rank} ({notam_id})"
        if isinstance(rank, bool) or not isinstance(rank, int) or not 1 <= rank <= ITEM_COUNT:
            _fail(f"{label}: rank must be a whole number from 1 to {ITEM_COUNT}")
        if rank in by_rank:
            _fail(f"{label}: duplicate rank {rank} (also {by_rank[rank]['id']})")
        if not isinstance(notam_id, str) or not isinstance(sha, str) or not _SHA.fullmatch(sha):
            _fail(f"{label}: needs a NOTAM id and a full 40-character commit sha")
        if kind not in TYPES:
            _fail(f"{label}: unknown type {kind!r}")
        if (sha, notam_id) in seen_pairs:
            _fail(f"{label}: {notam_id} at {sha[:8]} is already rank {seen_pairs[(sha, notam_id)]}")
        by_rank[rank] = entry
        seen_pairs[(sha, notam_id)] = rank
    missing = [rank for rank in range(1, ITEM_COUNT + 1) if rank not in by_rank]
    if missing:
        _fail(f"the selection has no item for rank {', '.join(map(str, missing))}")
    return [by_rank[rank] for rank in sorted(by_rank)]


def read_pib(pibs: Path, sha: str, cache: dict[str, dict[str, ParsedNotam]]) -> dict[str, ParsedNotam]:
    """The NOTAMs of the committed PIB for a commit, by id. Raises SheetError."""
    if sha not in cache:
        path = pibs / f"{sha}.xml"
        try:
            cache[sha] = {notam.id: notam for notam in read_notams(path.read_bytes())[1]}
        except OSError as exc:
            _fail(f"the PIB for commit {sha[:8]} is not committed ({path}): {exc.strerror or exc}")
        except PibError as exc:
            _fail(f"the PIB for commit {sha[:8]} cannot be read ({path}): {exc}")
    return cache[sha]


# --- one item --------------------------------------------------------------------------------


def _utc(moment: datetime | None) -> str | None:
    return moment.astimezone(UTC).strftime("%Y-%m-%dT%H:%MZ") if moment else None


def _end(notam: ParsedNotam) -> str | None:
    """Item C as a string: PERM, the end time (with EST when it is an estimate), or None when there is none."""
    if notam.permanent:
        return "PERM"
    end = _utc(notam.valid_to)
    return None if end is None else end + (" EST" if notam.est else "")


def build_item(entry: dict, notam: ParsedNotam) -> dict:
    """The sheet entry for one NOTAM. Redaction covers every free-text field with one numbering, so <PHONE_1> means one number."""
    texts = {"e": notam.item_e, "d": notam.item_d, "f": notam.item_f, "g": notam.item_g}
    joined, phones = redact_phones("\x00".join(text or "" for text in texts.values()))
    # Trailing spaces are dropped from each line so the text reads as a block (e_hash ignores them: it is taken from the raw text).
    redacted = {name: "\n".join(line.rstrip() for line in text.split("\n")) for name, text in zip(texts, joined.split("\x00"), strict=True)}
    area, note = build_area(notam)
    if area is None:
        shape: dict = {"kind": "q_circle", "lat": notam.centre_lat, "lon": notam.centre_lon, "radius_nm": notam.radius_nm}
    elif isinstance(area, Circle):
        shape = {"kind": "circle", "lat": area.lat, "lon": area.lon, "radius_nm": area.radius_nm}
    else:
        shape = {"kind": "polygon", "points": [[lat, lon] for lat, lon in area.points]}
    return {
        "rank": entry["rank"],
        "id": notam.id,
        "sha": entry["sha"],
        "type": entry["type"],
        "e": redacted["e"],
        "phones_redacted": len(phones),
        "coded": {
            "q_line": {
                "code": notam.q_code,
                "fir": notam.fir,
                "traffic": notam.traffic,
                "purpose": notam.purpose,
                "scope": notam.scope,
                "lower_fl": notam.lower_fl,
                "upper_fl": notam.upper_fl,
                "lat": notam.centre_lat,
                "lon": notam.centre_lon,
                "radius_nm": notam.radius_nm,
            },
            "a": notam.item_a,
            "b": _utc(notam.valid_from),
            "c": _end(notam),
            "d": redacted["d"] if texts["d"] is not None else None,
            "f": redacted["f"] if texts["f"] is not None else None,
            "g": redacted["g"] if texts["g"] is not None else None,
        },
        "area": {"source": note, **shape},
        "e_hash": e_hash(notam.item_e),
        "label": dict.fromkeys(LABEL_FIELDS),
    }


# --- writing the yaml ------------------------------------------------------------------------


def _scalar(value: object) -> str:
    """One YAML scalar. A string is always double-quoted (JSON is valid YAML), so ids like 'J1458/26' and 'null' stay strings."""
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return repr(value)
    if isinstance(value, int):
        return str(value)
    return json.dumps(value, ensure_ascii=False)


def _text(value: str, indent: str) -> str:
    """A text field: a readable block scalar when it has several lines and can hold the text exactly, else a quoted string."""
    lines = value.split("\n")
    plain = (
        len(lines) > 1
        and _PLAIN_BLOCK.fullmatch(value.replace("\n", "")) is not None
        and not value.startswith((" ", "\n"))
        and not value.endswith("\n")
        and all(line == line.rstrip() for line in lines)
    )
    if not plain:
        return " " + _scalar(value)
    return " |-\n" + "\n".join(f"{indent}  {line}" if line else "" for line in lines)


def _emit(key: str, value: object, indent: str) -> list[str]:
    if isinstance(value, dict):
        lines = [f"{indent}{key}:"]
        for child_key, child in value.items():
            lines += _emit(child_key, child, indent + "  ")
        return lines
    if isinstance(value, list):
        if all(isinstance(point, list) for point in value):
            return [f"{indent}{key}:"] + [f"{indent}  - [{', '.join(_scalar(n) for n in point)}]" for point in value]
        return [f"{indent}{key}: [{', '.join(_scalar(n) for n in value)}]"]
    if isinstance(value, str) and key in {"e", "d", "f", "g"}:
        return [f"{indent}{key}:{_text(value, indent)}"]
    return [f"{indent}{key}: {_scalar(value)}".rstrip()]


def render(items: list[dict]) -> str:
    lines = [HEADER.rstrip("\n"), "", "items:"]
    for item in items:
        if item is not items[0]:
            lines.append("")
        first = True
        for key, value in item.items():
            chunk = _emit(key, value, "    ")
            if first:
                chunk[0] = "  - " + chunk[0].lstrip()
                first = False
            lines += chunk
    return "\n".join(lines) + "\n"


# --- the checks and the run ------------------------------------------------------------------


def _is_date_time_group(digits: str) -> bool:
    """YYMMDDhhmm: ten digits that read as a date and a time are not a phone number."""
    if len(digits) != len("YYMMDDhhmm"):
        return False
    try:
        datetime.strptime(digits, "%y%m%d%H%M")
    except ValueError:
        return False
    return True


def phone_leaks(items: list[dict]) -> list[str]:
    """Every run of 10-11 digits still in a NOTAM text field of the sheet with whitespace removed, date-time groups excepted.

    Only the text fields are read: the sheet's own numbers (coordinates written to 15 places) are not NOTAM text.
    """
    leaks = []
    for item in items:
        for text in (item["e"], item["coded"]["d"], item["coded"]["f"], item["coded"]["g"]):
            squeezed = re.sub(r"\s+", "", text or "")
            leaks += [run for run in _DIGIT_RUN.findall(squeezed) if not _is_date_time_group(run)]
    return leaks


def build_sheet(selection: Path, pibs: Path) -> list[dict]:
    """All 40 items, or SheetError naming the first bad one. Nothing is written."""
    items: list[dict] = []
    cache: dict[str, dict[str, ParsedNotam]] = {}
    for entry in read_selection(selection):
        notams = read_pib(pibs, entry["sha"], cache)
        notam = notams.get(entry["id"])
        if notam is None:
            _fail(f"rank {entry['rank']} ({entry['id']}): not in the committed PIB for commit {entry['sha'][:8]}")
        items.append(build_item(entry, notam))
    return items


def write_atomically(path: Path, text: str) -> None:
    """Write beside the target, then rename over it: the file is the old one or the new one, never half of either."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    try:
        temp.write_text(text, encoding="utf-8", newline="\n")
        temp.replace(path)
    finally:
        temp.unlink(missing_ok=True)


def main(argv: Sequence[str] | None = None, root: Path = ROOT) -> int:
    reader = root / "evals" / "reader"
    parser = argparse.ArgumentParser(description="Build evals/reader/items.yaml from the committed selection and PIBs.")
    parser.add_argument("--selection", type=Path, default=reader / "selection.yaml")
    parser.add_argument("--pibs", type=Path, default=reader / "pibs")
    parser.add_argument("--out", type=Path, default=reader / "items.yaml")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    try:
        items = build_sheet(args.selection, args.pibs)
        text = render(items)
        # The text round-trips: what is written is the data built, nothing lost to quoting.
        if yaml.safe_load(text)["items"] != items:
            _fail("the sheet does not read back as the data it was built from")
        leaks = phone_leaks(items)
        if leaks:
            _fail(f"{len(leaks)} possible phone number(s) are still in the sheet: nothing written")
    except SheetError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    write_atomically(args.out, text)
    print(f"{len(items)} items written to {args.out}")
    print(f"{sum(item['phones_redacted'] for item in items)} phone numbers redacted, 0 phone leaks")
    return 0


if __name__ == "__main__":
    sys.exit(main())
