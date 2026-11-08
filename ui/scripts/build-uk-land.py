"""Rebuilds ui/src/data/uk-land.geojson, the land polygon the map's click test (ui/src/pick/land.ts) reads.

    uv run python ui/scripts/build-uk-land.py ne_10m_admin_0_countries.geojson

The input is Natural Earth's 1:10m admin-0 countries file (public domain), from
https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/ne_10m_admin_0_countries.geojson

It keeps the United Kingdom (Great Britain, Northern Ireland and the Scottish islands) and the Isle of Man, clips them to the
UK box the app opens on (rules.yaml uk_bounds_*), simplifies to about 0.005 degrees, rounds coordinates to 4 decimals and
drops islets smaller than MIN_AREA. The output is one MultiPolygon, bundled with the app and never fetched at run time.
This is a dev tool: nothing in the app or the tests runs it.
"""

import json
import sys
from pathlib import Path

from shapely.geometry import MultiPolygon, box, mapping, shape
from shapely.ops import unary_union

KEEP = {"GBR", "IMN"}
UK_BOX = box(-8.7, 49.8, 2.0, 61.0)  # west, south, east, north: the uk_bounds_* numbers
TOLERANCE = 0.005
DECIMALS = 4
MIN_AREA = 0.00005  # square degrees, about 0.6 km2: smaller islets are dropped
OUT = Path(__file__).resolve().parent.parent / "src" / "data" / "uk-land.geojson"


def round_ring(ring: list[tuple[float, float]]) -> list[list[float]]:
    return [[round(x, DECIMALS), round(y, DECIMALS)] for x, y in ring]


def main(source: str) -> None:
    features = json.loads(Path(source).read_text())["features"]
    land = unary_union([shape(f["geometry"]) for f in features if f["properties"]["ADM0_A3"] in KEEP])
    land = land.intersection(UK_BOX).simplify(TOLERANCE, preserve_topology=True)
    parts = list(land.geoms) if isinstance(land, MultiPolygon) else [land]
    polygons = []
    for part in parts:
        if part.area < MIN_AREA:
            continue
        coords = mapping(part)["coordinates"]
        polygons.append([round_ring(ring) for ring in coords])
    collection = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {"name": "UK land (Natural Earth 1:10m admin-0, GB + Isle of Man)"},
                "geometry": {"type": "MultiPolygon", "coordinates": polygons},
            }
        ],
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(collection, separators=(",", ":")) + "\n")
    print(f"{len(polygons)} polygons, {OUT.stat().st_size} bytes -> {OUT}")


if __name__ == "__main__":
    main(sys.argv[1])
