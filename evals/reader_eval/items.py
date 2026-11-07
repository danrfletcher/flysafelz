"""The labelling sheet (evals/reader/items.yaml) and the NOTAM each of its items names. Offline: the PIBs are committed.

    load_items(path) -> list of the sheet's item entries, in rank order
    parsed_notam(item, pibs) -> the ParsedNotam rebuilt from the committed PIB for the item's commit

The sheet never holds labels (labels.py reads those from a separate file). An item whose NOTAM text in the committed PIB no
longer hashes to the item's `e_hash` is an error here, so a label is never judged against text it was not written for.
"""

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

from flysafelz.contracts import ParsedNotam
from flysafelz.pib import PibError, read_notams
from flysafelz.redact import e_hash

READER_DATA = Path(__file__).resolve().parents[1] / "reader"
ITEMS_PATH = READER_DATA / "items.yaml"
LABELS_PATH = READER_DATA / "labels.yaml"
PIBS_DIR = READER_DATA / "pibs"


class ItemError(Exception):
    """The sheet or a committed PIB is not usable. The message names the item or the file."""


def load_items(path: Path = ITEMS_PATH) -> list[dict[str, Any]]:
    """The sheet's items. Raises ItemError when the file is missing, is not YAML, or has no `items:` list of entries with an id and an e_hash."""
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ItemError(f"cannot read the sheet {path}: {exc}") from exc
    entries = data.get("items") if isinstance(data, dict) else None
    if not isinstance(entries, list):
        raise ItemError(f"{path} has no `items:` list")
    for position, entry in enumerate(entries, start=1):
        if not isinstance(entry, dict) or not isinstance(entry.get("id"), str) or not isinstance(entry.get("e_hash"), str):
            raise ItemError(f"{path}: item {position} has no id and e_hash")
    return entries


def parsed_notam(item: Mapping[str, Any], pibs: Path = PIBS_DIR) -> ParsedNotam:
    """The NOTAM an item names, from the committed PIB `<pibs>/<sha>.xml`. Raises ItemError if it is missing or its text has changed."""
    path = pibs / f"{item['sha']}.xml"
    try:
        found = [notam for notam in read_notams(path.read_bytes())[1] if notam.id == item["id"]]
    except OSError as exc:
        raise ItemError(f"{item['id']}: the PIB {path} cannot be read: {exc.strerror or exc}") from exc
    except PibError as exc:
        raise ItemError(f"{item['id']}: the PIB {path} is not a PIB: {exc}") from exc
    if not found:
        raise ItemError(f"{item['id']}: not in the PIB {path}")
    if e_hash(found[0].item_e) != item["e_hash"]:
        raise ItemError(f"{item['id']}: its E text in {path} no longer matches the sheet's e_hash")
    return found[0]
