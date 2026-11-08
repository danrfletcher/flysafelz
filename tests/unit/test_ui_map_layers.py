"""Ticket 8 PR-3.F1: the Python side of the map layers' fence.

The UI's own tests run with `make test-ui`. These keep three things the browser code copies from the server from drifting apart:
the airspace type names, the field names of the two routes, and the made-up map fixtures, which must be shaped as the real routes send them (they are not recorded from the routes).
"""

import json
import re
from pathlib import Path

from flysafelz.airspace import Outcome
from flysafelz.clients.openaip import CODES, parse_airspace
from flysafelz.contracts import Datum, Limit, LimitUnit
from flysafelz.mapdata.airspace import airspace_feature, limit_feet, limit_text

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


# --- the made-up fixtures are shaped as the routes send them ----------------------------------------------------------------

# What the UI fixtures hold that the routes never send, on purpose, to show the UI survives it (each is exercised by a UI test that says so):
# an airspace type openAIP's list does not have (the route turns an unknown type into "other"), and the lower-limit forms "GND" and "" (limit_text writes
# SFC, FLn, "n ft AGL/AMSL" or UNL). Nothing else may differ from the route's output.
MADE_UP_TYPES = {"balloon_zone"}
MADE_UP_LOWER_FORMS = {"GND", ""}
LIMIT_FORM = re.compile(r"^(SFC|UNL|FL\d+|\d+ ft (AGL|AMSL))$")


def fixture_features(prefix: str) -> list[dict]:
    found = []
    for path in sorted(FIXTURES.glob(f"{prefix}*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert set(payload) == {"type", "features", "updated_at", "stale"}, path.name
        found += payload["features"]
    assert found
    return found


def route_airspace_properties() -> set[str]:
    """The properties the route builds: airspace_feature over a real fixture item."""
    item = json.loads((ROOT / "tests" / "fixtures" / "openaip" / "page1.json").read_text(encoding="utf-8"))["items"][0]
    return set(airspace_feature(parse_airspace(item))["properties"])


def test_every_airspace_fixture_feature_has_the_properties_the_route_builds_and_no_action():
    expected = route_airspace_properties()
    for feature in fixture_features("map-airspace"):
        assert set(feature["properties"]) == expected, feature["properties"]["name"]
        assert feature["properties"]["action"] is None, feature["properties"]["name"]  # the base layer maps no action (airspace_feature)
        assert feature["geometry"]["type"] in {"Polygon", "MultiPolygon"}


def test_the_airspace_fixtures_differ_from_the_routes_output_only_in_the_made_up_cases_named_here():
    server_types = {name for name in CODES["type"].values() if name is not None}
    for feature in fixture_features("map-airspace"):
        properties = feature["properties"]
        assert properties["type"] in server_types | MADE_UP_TYPES, properties["name"]
        for field in ("lower", "upper"):
            assert LIMIT_FORM.match(properties[field]) or (field == "lower" and properties[field] in MADE_UP_LOWER_FORMS), (properties["name"], properties[field])


def test_the_limit_text_and_feet_in_the_airspace_fixtures_are_what_limit_text_and_limit_feet_write():
    for feature in fixture_features("map-airspace"):
        properties = feature["properties"]
        for field in ("lower", "upper"):
            text, feet = properties[field], properties[f"{field}_ft"]
            if text in MADE_UP_LOWER_FORMS:
                continue
            if text == "UNL":
                limit = None
            elif text == "SFC":
                limit = Limit(value=0, unit=LimitUnit.FT, datum=Datum.SFC)
            elif text.startswith("FL"):
                limit = Limit(value=int(text[2:]), unit=LimitUnit.FL, datum=Datum.AMSL)
            else:
                value, _, datum = text.split(" ")
                limit = Limit(value=int(value), unit=LimitUnit.FT, datum=Datum(datum))
            assert (limit_text(limit), limit_feet(limit)) == (text, feet), (properties["name"], field)


def test_every_notam_fixture_feature_has_the_properties_the_route_builds():
    from datetime import UTC, datetime

    from mapdata_helpers import notam, pib_of

    from flysafelz.mapdata.notams import notam_feature
    from flysafelz.pib import parse_pib

    start = datetime(2026, 11, 8, 10, tzinfo=UTC)
    _issued, parsed = parse_pib(pib_of(notam(1, start, start.replace(hour=18))))
    expected = set(notam_feature(parsed[0])["properties"])
    for feature in fixture_features("map-notams"):
        assert set(feature["properties"]) == expected, feature["properties"]["id"]
        assert feature["geometry"]["type"] in {"Polygon", "MultiPolygon"}


def test_the_ui_colours_every_airspace_outcome_the_agent_can_put_on_an_overlay():
    """airspace_tool passes every outcome except IGNORE into an answer's overlays; one the UI does not know would be drawn slate with a console warning."""
    source = (MAP / "actionStyle.ts").read_text(encoding="utf-8")
    table = source.split("const BY_ACTION", 1)[1].split("\n}\n", 1)[0]
    keys = set(re.findall(r"^\s+([A-Z_]+)\b", table, re.MULTILINE))
    assert {outcome.value for outcome in Outcome if outcome is not Outcome.IGNORE} <= keys
