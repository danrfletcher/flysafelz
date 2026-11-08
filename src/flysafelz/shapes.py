"""The one GeoJSON shape of a NOTAM, and the one circle-to-polygon, for every map and overlay.

    notam_geometry(parsed)  -> (GeoJSON Polygon as a dict, approx)
    airspace_geometry(shape) -> GeoJSON Polygon or MultiPolygon as a dict, for an openAIP airspace's shape
    circle_ring(lat, lon, radius_nm) -> a closed ring of [lon, lat] positions

A NOTAM's shape is `notam_area.build_area(parsed)`, the code-only answer the reader and `assess` use (no model, no network), with
`approx` False. When `build_area` gives None (the E text names no area, or the area built from it was a parser miss and was replaced)
the shape is the Q-circle of the NOTAM and `approx` is True: the map draws something bigger than the real area, and says so.

A circle becomes a polygon of `CIRCLE_VERTICES` points, one every equal step starting due north, on the local flat map geo.py measures
with, and the ring is closed. The only inputs are the circle and that one constant, so the same NOTAM always gives the same
coordinates, to the last bit. The LZ's local-area circle uses the same function.

Positions are [lon, lat] as GeoJSON has them. This module imports nothing from `agent/`, LangChain or a web framework, so a route can use it.
"""

import math

from flysafelz.contracts import Circle, ParsedNotam
from flysafelz.geo import METRES_PER_NM, WGS84_A, WGS84_E2, Ring, Shape
from flysafelz.notam_area import build_area

CIRCLE_VERTICES = 72  # a circle is drawn as a polygon of this many points (one every 5 degrees)

Position = list[float]


def circle_ring(lat: float, lon: float, radius_nm: float) -> list[Position]:
    """A closed ring of `CIRCLE_VERTICES` [lon, lat] points round (lat, lon), starting due north."""
    radius_m = radius_nm * METRES_PER_NM
    sin_lat = math.sin(math.radians(lat))
    w = math.sqrt(1 - WGS84_E2 * sin_lat * sin_lat)
    north_radius = WGS84_A * (1 - WGS84_E2) / (w * w * w)
    east_radius = WGS84_A / w * math.cos(math.radians(lat))
    ring = []
    for step in range(CIRCLE_VERTICES):
        angle = math.tau * step / CIRCLE_VERTICES
        ring.append([lon + math.degrees(radius_m * math.sin(angle) / east_radius), lat + math.degrees(radius_m * math.cos(angle) / north_radius)])
    ring.append(list(ring[0]))
    return ring


def _closed(ring: Ring) -> list[Position]:
    """A geo.py ring ((lat, lon) points) as a closed GeoJSON ring of [lon, lat]."""
    positions = [[lon, lat] for lat, lon in ring]
    if positions[0] != positions[-1]:
        positions.append(list(positions[0]))
    return positions


def notam_geometry(parsed: ParsedNotam) -> tuple[dict, bool]:
    """(the NOTAM's GeoJSON Polygon, approx). See the module text."""
    area, _note = build_area(parsed)
    if area is None:
        return {"type": "Polygon", "coordinates": [circle_ring(parsed.centre_lat, parsed.centre_lon, parsed.radius_nm)]}, True
    if isinstance(area, Circle):
        return {"type": "Polygon", "coordinates": [circle_ring(area.lat, area.lon, area.radius_nm)]}, False
    return {"type": "Polygon", "coordinates": [_closed([(lat, lon) for lat, lon in area.points])]}, False


def airspace_geometry(shape: Shape) -> dict:
    """The GeoJSON of an airspace's shape: a Polygon for one polygon, a MultiPolygon for more. Holes are kept."""
    polygons = [[_closed(ring) for ring in poly] for poly in shape]
    if len(polygons) == 1:
        return {"type": "Polygon", "coordinates": polygons[0]}
    return {"type": "MultiPolygon", "coordinates": polygons}
