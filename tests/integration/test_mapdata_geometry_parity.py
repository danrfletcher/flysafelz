"""I8 parity guard: for each NOTAM of the Blue Mill case that the agent's get_notams returns in `overlays`, /map/notams serves the same geometry and the same
`approx` flag, coordinate for coordinate. Both come from shapes.notam_geometry; this fails when either side starts building its own shape.

The agent side runs the real get_notams tool in a real graph (tool_helpers.run_tools). The map side runs the real app on the case's committed PIB.
"""

import json
from datetime import UTC, datetime
from pathlib import Path

from mapdata_helpers import SETTINGS, Upstream
from starlette.testclient import TestClient
from tool_helpers import BLUE_MILL_NOW, BLUE_MILL_WINDOW, calls, run_tools

from flysafelz.agent.sources import SourceCache
from flysafelz.agent.state import RunContext
from flysafelz.mapdata.app import create_app

ROOT = Path(__file__).resolve().parents[2]
CASE = "blue_mill_2026-10-31"
CTX = RunContext(now=BLUE_MILL_NOW, case=CASE, reader="fixture")


def agent_overlays() -> dict[str, dict]:
    run = run_tools(SourceCache(), [calls("get_notams", **BLUE_MILL_WINDOW)], ctx=CTX)
    found = {}
    for overlay in run.artifact("get_notams").overlays:
        if overlay.properties.kind == "notam":
            found[overlay.properties.id] = json.loads(overlay.geometry.model_dump_json()) | {"approx": overlay.properties.approx}
    return found


def map_features(tmp_path) -> dict[str, dict]:
    pib = (ROOT / "fixtures" / CASE / "pib.xml").read_bytes()
    app = create_app(root=tmp_path, clock=lambda: datetime(2026, 10, 31, 7, 30, tzinfo=UTC), settings=SETTINGS, client_factory=Upstream(pib=pib).factory)
    body = TestClient(app).get("/map/notams").json()
    return {f["properties"]["id"]: f for f in body["features"]}


def test_the_agents_notam_overlays_and_the_maps_features_have_the_same_shapes(tmp_path):
    overlays, features = agent_overlays(), map_features(tmp_path)
    assert overlays, "the Blue Mill case returns NOTAM overlays"
    for notam_id, overlay in overlays.items():
        assert notam_id in features, f"{notam_id} is in the agent's overlays but not on the map"
        assert features[notam_id]["geometry"] == {"type": overlay["type"], "coordinates": overlay["coordinates"]}  # coordinate for coordinate
        assert features[notam_id]["properties"]["approx"] is overlay["approx"]


def test_the_comparison_covers_a_built_shape_and_a_q_circle():
    flags = {notam_id: overlay["approx"] for notam_id, overlay in agent_overlays().items()}
    assert True in flags.values() and False in flags.values()
