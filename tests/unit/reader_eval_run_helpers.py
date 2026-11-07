"""Builders for the reader eval runner's tests: a synthetic sheet with its PIB and labels under tmp_path, and fake chat models.

The three NOTAMs are obviously synthetic (X1/26 to X3/26, "TEST AREA ACTIVE"); X1/26 carries a phone number in the
Ofcom drama range (07700 900123) so a leak into an artifact is easy to see. Their area is the Q-circle round Blue Mill. The
sheet is items.yaml, the NOTAMs are one PIB under pibs/, and labels.yaml holds whatever the test writes. Nothing here touches
the repository's own evals/reader/ folder.

`ScriptedChat` (reader_set_helpers) is the fake chat model: it answers each NOTAM from its own script and, once a script is
used up, with `good_reply()`, a cooperative all-aircraft drone area to 700 ft AGL with no contact: exactly GOOD_LABEL.
"""

from pathlib import Path
from typing import Any

import yaml
from cache_helpers import blue_mill_set, fixture_reply
from pib_helpers import notam_xml, pib_xml
from reader_set_helpers import ScriptedChat

from evals.reader_eval.run import EvalPaths
from flysafelz.contracts import ParsedNotam
from flysafelz.pib import read_notams
from flysafelz.redact import e_hash

SHA = "synthetic-pib"
IDS = ("X1/26", "X2/26", "X3/26")
FAKE_PHONE = "07700 900123"
# The texts differ, so the fake model can tell the NOTAMs apart by what it is sent.
E_TEXTS = {
    "X1/26": f"TEST AREA ONE ACTIVE. NO SHAPE IS GIVEN IN THIS TEXT. TELEPHONE {FAKE_PHONE}.",
    "X2/26": "TEST AREA TWO ACTIVE. NO SHAPE IS GIVEN IN THIS TEXT.",
    "X3/26": "TEST AREA THREE ACTIVE. NO SHAPE IS GIVEN IN THIS TEXT.",
}

# What good_reply() says, written the way Dan writes a label: so a run that answers with good_reply() agrees with it.
GOOD_LABEL: dict[str, Any] = {
    "category": "cooperative",
    "activity": "drone ops",
    "floor": "SFC",
    "ceiling": "700FT AGL",
    "applies_to": "ALL_AIRCRAFT",
    "exemptions": [],
    "contact": "none",
    "activates_designators": [],
    "prohibitive_reason": "none",
}
# A label that applies to every aircraft and forbids entry (NO_FLY over the centre): a reading that only asks for awareness, or no reading at all, is a harmful downgrade of it.
FORBIDDING_LABEL = {**GOOD_LABEL, "category": "prohibitive", "prohibitive_reason": "entry prohibited"}


def synthetic_pib() -> bytes:
    return pib_xml(*(notam_xml(series="X", number=int(notam_id[1:].split("/")[0]), year=26, radius="3", e=E_TEXTS[notam_id]) for notam_id in IDS)).encode("utf-8")


def synthetic_notams() -> list[ParsedNotam]:
    return read_notams(synthetic_pib())[1]


def build_sheet(tmp_path: Path) -> tuple[EvalPaths, dict[str, str]]:
    """items.yaml and the PIB for the three synthetic NOTAMs under tmp_path. Returns the paths (labels.yaml not written) and each id's e_hash."""
    notams = synthetic_notams()
    assert [notam.id for notam in notams] == list(IDS)
    hashes = {notam.id: e_hash(notam.item_e) for notam in notams}
    pibs = tmp_path / "pibs"
    pibs.mkdir()
    (pibs / f"{SHA}.xml").write_bytes(synthetic_pib())
    sheet = {"items": [{"rank": rank, "id": notam_id, "sha": SHA, "e_hash": hashes[notam_id]} for rank, notam_id in enumerate(IDS, start=1)]}
    (tmp_path / "items.yaml").write_text(yaml.safe_dump(sheet, sort_keys=False), encoding="utf-8")
    return EvalPaths(items=tmp_path / "items.yaml", labels=tmp_path / "reader" / "labels.yaml", pibs=pibs, runs=tmp_path / "runs"), hashes


def write_labels(paths: EvalPaths, entries: list[dict[str, Any]]) -> Path:
    """labels.yaml from entries like {"id": ..., "e_hash": ..., "label": {...}}."""
    paths.labels.parent.mkdir(parents=True, exist_ok=True)
    paths.labels.write_text(yaml.safe_dump({"labels": entries}, sort_keys=False), encoding="utf-8")
    return paths.labels


def label_entry(notam_id: str, hashes: dict[str, str], label: dict[str, Any] | None = None, *, stale: bool = False) -> dict[str, Any]:
    entry: dict[str, Any] = {"id": notam_id, "e_hash": "stale-hash" if stale else hashes[notam_id]}
    if label is not None:
        entry["label"] = label
    return entry


def blue_mill_scripts() -> dict[str, list[Any]]:
    """One good answer each for Blue Mill's two NOTAMs: readings.yaml's fields, the number as its placeholder."""
    return {"H7171_26": [fixture_reply("H7171/26")], "J3857_26": [fixture_reply("J3857/26")]}


def chat(**scripts: Any) -> ScriptedChat:
    """A fake model that knows the synthetic NOTAMs and Blue Mill's two. Blue Mill answers like the fixture unless a script says otherwise."""
    return ScriptedChat(synthetic_notams() + blue_mill_set(), **{**blue_mill_scripts(), **scripts})


class Down:
    """A model double for an endpoint that is down: every call raises."""

    model_name = "down-model"

    def __init__(self) -> None:
        self.calls = 0

    def invoke(self, messages: list[Any], **kwargs: Any) -> Any:
        self.calls += 1
        raise ConnectionError("the endpoint refused the connection")
