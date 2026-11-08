"""Ticket 8 PR-3.F1: the Python side of the map layers' fence.

The UI's own tests run with `make test-ui`. These keep three things the browser code copies from the server from drifting apart:
the airspace type names, the field names of the two routes, and the recorded fixtures, which must be what the real routes emit.
"""

import json
import re
from pathlib import Path

from flysafelz.clients.openaip import CODES

ROOT = Path(__file__).resolve().parents[2]
UI = ROOT / "ui"
MAP = UI / "src" / "map"
FIXTURES = UI / "src" / "test" / "fixtures"


def quoted_names(text: str) -> list[str]:
    return re.findall(r"'([a-z_]+)'", text)


def test_the_ui_lists_every_openaip_airspace_type_in_code_order():
    source = (MAP / "airspaceStyle.ts").read_text(encoding="utf-8")
    block = source.split("export const AIRSPACE_TYPES = [", 1)[1].split("] as const", 1)[0]
    server = [name for name in CODES["type"].values() if name is not None]
    assert quoted_names(block) == server


def test_the_ui_names_no_type_the_server_does_not_have():
    # The colour rule's keys (restricted, danger ...) must be real type names, or the rule would never match.
    source = (MAP / "airspaceStyle.ts").read_text(encoding="utf-8")
    rule = source.split("const BY_TYPE", 1)[1].split("}", 1)[0]
    keys = re.findall(r"^\s+([a-z_]+):", rule, re.MULTILINE)
    assert keys and set(keys) <= {name for name in CODES["type"].values() if name is not None}


def test_the_ui_types_name_the_fields_the_routes_serve():
    airspace = json.loads((FIXTURES / "map-airspace.json").read_text(encoding="utf-8"))
    notams = json.loads((FIXTURES / "map-notams.json").read_text(encoding="utf-8"))
    types = (MAP / "types.ts").read_text(encoding="utf-8")
    for payload in (airspace, notams):
        assert set(payload) == {"type", "features", "updated_at", "stale"}
        assert payload["type"] == "FeatureCollection"
    for collection in (airspace, notams):
        for feature in collection["features"]:
            for field in feature["properties"]:
                assert re.search(rf"\b{field}\??:", types), f"{field} is served but not in ui/src/map/types.ts"
