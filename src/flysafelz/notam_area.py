"""The area a NOTAM names in its E text, built by code alone (G3). No model, no network.

    build_area(notam) -> (Circle | Polygon | None, note)

None means "use the Q-circle": the circle in the Q-line, which is larger than any real NOTAM area (the area in
the E text is a median 25 % of it), so None is the cautious answer. The model has no geometry field.

What is built, by what the E text holds:

- one coordinate and one radius              -> Circle. "5NM RADIUS OF", "RADIUS 2000M CENTRED ON" and
                                                "WI 0.5NM OF PSN" (no word RADIUS) all count. NM, km and metres.
- three or more points and no radius         -> Polygon, in the order written (a closing repeat of the first
                                                point is dropped)
- no coordinates, one or two points and no radius, a radius of zero, or more than one radius -> None
- a corridor either side of a track, a buffer, an arc, two separate points or areas, a polygon whose
  text is cut off, a coordinate that is not a coordinate (minutes or seconds of 60 or more, a latitude past
  90, a longitude past 180)                  -> None

A built area that reaches more than `reader_area_slack_nm` outside the Q-circle is a parser miss, not a bigger
NOTAM: it is replaced by the Q-circle and logged once. Exactly that far out is kept (the rule's own comparison).
"Once" is once per NOTAM (id, E text, Q-circle): `build_area` remembers its answer, and the reader, the sort and each
segment of the action table all ask it.

Coordinates are DDMMN DDDMME or DDMMSSN DDDMMSSE, the seconds with decimals if the NOTAM has them.

Where a NOTAM is (G16): `area_from_item_e(notam)` sorts the E text into one of four answers, on top of `build_area`.

    Shape    a valid polygon or circle (`build_area`'s answer): what the action table measures the LZ against, however large
    Route    two or more coordinates each with a time on its line (a transit list): a path, not an area (limitation 20)
    Broken   coordinate-looking text that did not make a valid shape, with a one-line reason: the Q-circle, never the cap
    NoArea   no coordinate-looking text at all

Coordinate-looking text is a coordinate pair, or any run of 6 or 7 digits written straight before N, S, E or W. Frequencies
(128.925), AR and SUP references, FL195 and telephone numbers have no such letter after the digits and are not coordinates.
"""

import logging
import re
from dataclasses import dataclass
from itertools import pairwise

from flysafelz.contracts import Circle, ParsedNotam, Polygon
from flysafelz.geo import HALF_TURN, METRES_PER_KM, METRES_PER_NM, MIN_POLYGON_POINTS, distance_nm
from flysafelz.rules import RULES

log = logging.getLogger(__name__)

Area = Circle | Polygon | None

MINUTES_PER_DEGREE = 60
SECONDS_PER_MINUTE = 60
_REMEMBER_LIMIT = 20_000  # more than the 14,609 distinct NOTAMs of the archive; past it the list starts again
_REMEMBERED: dict[tuple[str, str, float, float, float, str], tuple[Area, str]] = {}

_COORDINATE = re.compile(
    r"(?<![\d.])(\d{2})(\d{2})(\d{2}(?:\.\d+)?)?([NS])\s*(\d{3})(\d{2})(\d{2}(?:\.\d+)?)?([EW])(?![A-Z0-9])"
)
_NUMBER = r"(?<![\d.])(\d+(?:\.\d+)?)"
_UNIT = r"(NM|KM|METRES|METERS|M)\b"
_RADIUS_BEFORE = re.compile(rf"{_NUMBER}\s*{_UNIT}\s+RADIUS\b")  # 2.2NM RADIUS OF, 185M RADIUS PSN
_RADIUS_AFTER = re.compile(rf"\bRADIUS\s+(?:OF\s+)?{_NUMBER}\s*{_UNIT}")  # CIRCLE RADIUS 2000M CENTRED ON
_WITHIN_OF = re.compile(rf"\b(?:WI|WITHIN)\s+{_NUMBER}\s*{_UNIT}\s+OF\s+(?:PSN\s+)?(?=\d{{4}})")  # WI 0.5NM OF PSN 5347..
_LOOKS_LIKE_A_COORDINATE = re.compile(r"(?<![\d.])\d{6,7}(?:\.\d+)?[NSEW](?![A-Z0-9])")
_TIME = re.compile(r"(?<![\d.])(?:[01]\d|2[0-3])[0-5]\d(?:Z)?(?![\d.A-Z])")  # HHMM or HHMMZ
_RADIUS_WORD = re.compile(r"\bRADIUS\b")
_MESSY = re.compile(r"\b(?:EITHER SIDE|BOTH SIDES|CORRIDOR|ARCS?|BUFFER(?:ED)?|CENTRE ?LINE|CENTER ?LINE|ALONG|TRACK)\b")
_CUT_OFF = re.compile(r"[-\u2013\sNSEW\d.]*")  # what may follow the last whole coordinate of a list that was cut off mid-way

Point = tuple[float, float]


def _degrees(degrees: str, minutes: str, seconds: str | None, hemisphere: str) -> float | None:
    """Decimal degrees (south and west negative), or None when minutes, seconds or the whole value are out of range."""
    minute = int(minutes)
    second = float(seconds) if seconds else 0
    if minute >= MINUTES_PER_DEGREE or second >= SECONDS_PER_MINUTE:
        return None
    value = int(degrees) + minute / MINUTES_PER_DEGREE + second / (MINUTES_PER_DEGREE * SECONDS_PER_MINUTE)
    if value > HALF_TURN:  # no longitude is past 180, and no latitude either (the latitude's own limit is below)
        return None
    return -value if hemisphere in "SW" else value


def _points(text: str) -> list[Point] | None:
    """Every coordinate pair in the text in order written; None when one of them is not a possible place."""
    points = []
    for lat_deg, lat_min, lat_sec, ns, lon_deg, lon_min, lon_sec, ew in _COORDINATE.findall(text):
        lat = _degrees(lat_deg, lat_min, lat_sec or None, ns)
        lon = _degrees(lon_deg, lon_min, lon_sec or None, ew)
        if lat is None or lon is None or abs(lat) > HALF_TURN - abs(lat):  # the last test is "latitude past 90"
            return None
        points.append((lat, lon))
    return points


def _radii_nm(text: str) -> list[float]:
    """The distinct radii the text states, in NM. Metres and km are converted here."""
    found = set()
    for pattern in (_RADIUS_BEFORE, _RADIUS_AFTER, _WITHIN_OF):
        for value, unit in pattern.findall(text):
            number = float(value)
            if unit == "NM":
                found.add(number)
            elif unit == "KM":
                found.add(number * METRES_PER_KM / METRES_PER_NM)
            else:
                found.add(number / METRES_PER_NM)
    return sorted(found)


def _is_cut_off(text: str) -> bool:
    """True when the text stops inside the coordinate list: after the last whole coordinate come only hyphens and digits.

    A full stop on its own, or any words, mean the list ended where the author meant it to.
    """
    after = text[list(_COORDINATE.finditer(text))[-1].end() :].strip()
    return after not in ("", ".") and _CUT_OFF.fullmatch(after) is not None and any(c.isdigit() or c in "-\u2013" for c in after)


def _polygon(points: list[Point], text: str) -> tuple[Area, str]:
    """Three or more points and no radius: a polygon in the order written, or None when it cannot be one."""
    ring = points[:-1] if len(points) > 1 and points[0] == points[-1] else points
    if len(set(ring)) != len(ring):
        return None, "Q-circle: the E text repeats a point inside the shape (two separate areas)"
    if len(ring) < MIN_POLYGON_POINTS:
        return None, f"Q-circle: {len(points)} point{'s' if len(points) != 1 else ''} and no radius"
    if _is_cut_off(text):
        return None, "Q-circle: the E text stops in the middle of the shape"
    return Polygon(points=ring), f"E text: polygon of {len(ring)} points"


def _outside_by(notam: ParsedNotam, area: Area) -> float | None:
    """How far in NM the area reaches outside the Q-circle, when that is more than the slack; else None."""
    if area is None:
        return None
    if isinstance(area, Circle):
        reach = distance_nm(notam.centre_lat, notam.centre_lon, area.lat, area.lon) + area.radius_nm
    else:
        reach = max(distance_nm(notam.centre_lat, notam.centre_lon, lat, lon) for lat, lon in area.points)
    outside = reach - notam.radius_nm
    return outside if RULES.reader_area_slack_nm.breached(outside) else None


def build_area(notam: ParsedNotam) -> tuple[Area, str]:
    """(the area the E text names or None for the Q-circle, a one-line note on where it came from).

    Never raises on text, however odd or cut off. The answer is worked out once per NOTAM (id, E text, Q-circle and the slack
    rule) and remembered, so the reader, the sort and every segment of the action table share one answer and a parser miss
    is logged once, not once per segment. `forget_areas()` forgets them (tests).
    """
    key = (notam.id, notam.item_e, notam.centre_lat, notam.centre_lon, notam.radius_nm, repr(RULES.reader_area_slack_nm.value))
    if key not in _REMEMBERED:
        if len(_REMEMBERED) >= _REMEMBER_LIMIT:
            _REMEMBERED.clear()
        _REMEMBERED[key] = _build_area(notam)
    area, note = _REMEMBERED[key]
    return (area.model_copy(deep=True) if area is not None else None), note  # a copy: callers never share a mutable model


def forget_areas() -> None:
    """Forget every remembered area (and so log a parser miss again)."""
    _REMEMBERED.clear()


def _build_area(notam: ParsedNotam) -> tuple[Area, str]:
    text = notam.item_e.upper()
    points = _points(text)
    if points is None:
        return None, "Q-circle: a coordinate in the E text is not a possible place"
    if _MESSY.search(text):
        return None, "Q-circle: the E text describes a corridor, buffer or arc"
    if not points:
        return None, "Q-circle: no coordinates in the E text"

    radii = _radii_nm(text)
    if len(radii) > 1:
        area, note = None, "Q-circle: the E text gives more than one radius"
    elif radii:
        if radii[0] <= 0:  # a circle of no size is no area; a radius cannot be written negative, so <= 0 is a zero
            area, note = None, "Q-circle: the radius in the E text is zero or less, so there is no area"
        elif len(points) == 1:
            area, note = Circle(lat=points[0][0], lon=points[0][1], radius_nm=radii[0]), f"E text: {radii[0]:.4g} NM radius round one point"
        else:
            area, note = None, "Q-circle: a radius with more than one point (separate areas)"
    elif _RADIUS_WORD.search(text):
        area, note = None, "Q-circle: the word RADIUS is there but no radius could be read"
    else:
        area, note = _polygon(points, text)

    outside = _outside_by(notam, area)
    if outside is not None:
        slack = RULES.reader_area_slack_nm.value
        log.warning(
            "parser miss: %s: the area built from its E text reaches %.2f NM outside the Q-circle, more than the %g NM slack; using the Q-circle",
            notam.id,
            outside,
            slack,
        )
        return None, f"Q-circle: the area built from the E text reaches {outside:.2f} NM outside the Q-circle, more than the {slack:g} NM slack (parser miss)"
    return area, note


@dataclass(frozen=True)
class Shape:
    """A valid area from the E text: a circle or a polygon."""

    area: Circle | Polygon

    @property
    def kind(self) -> str:
        return "circle" if isinstance(self.area, Circle) else "polygon"


@dataclass(frozen=True)
class Route:
    """Coordinates each with a time: where an aircraft goes, not an area it fills."""

    waypoints: int


@dataclass(frozen=True)
class Broken:
    """Coordinate-looking text that did not make a valid shape."""

    reason: str


@dataclass(frozen=True)
class NoArea:
    """No coordinate-looking text."""


ItemEArea = Shape | Route | Broken | NoArea


def _is_route(text: str, pairs: list[re.Match[str]]) -> bool:
    """Two or more coordinates, each alone on its line with a time (HHMM or HHMMZ, the coordinates themselves left out).

    One coordinate per line is what tells a transit list from a polygon written on one line with a schedule after it
    ("... 550500N 0013000W. SFC-UNL. 0800-1200 DAILY"): there the line holds every corner, so it is an area, not a route.
    A place name between the coordinate and its time is fine.
    """
    if len(pairs) <= 1:  # one coordinate is a place; a route has a leg
        return False
    for match in pairs:
        start = text.rfind("\n", 0, match.start()) + 1
        end = text.find("\n", match.end())
        line = text[start : len(text) if end < 0 else end]
        if len(_COORDINATE.findall(line)) != 1 or not _TIME.search(_COORDINATE.sub(" ", line)):
            return False
    return True


def _crossing(a: Point, b: Point, c: Point, d: Point) -> bool:
    """True when segment a-b and segment c-d cross at a point inside both (touching and overlapping do not count)."""

    def side(p: Point, q: Point, r: Point) -> float:
        return (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])

    return side(a, b, c) * side(a, b, d) < 0 and side(c, d, a) * side(c, d, b) < 0


def _crosses_itself(corners: list[Point]) -> bool:
    """True when two edges of the closed outline that do not share a corner cross each other (a bow tie)."""
    edges = list(pairwise([*corners, corners[0]]))
    for i, (a, b) in enumerate(edges):
        for j in range(i + 1, len(edges)):
            if j == i + 1 or (i == 0 and j == len(edges) - 1):  # neighbours meet at their shared corner
                continue
            if _crossing(a, b, *edges[j]):
                return True
    return False


def area_from_item_e(notam: ParsedNotam) -> ItemEArea:
    """The area item E draws, sorted into Shape, Route, Broken or NoArea. Never raises, whatever the text."""
    text = notam.item_e.upper()
    pairs = list(_COORDINATE.finditer(text))
    unpaired = [m for m in _LOOKS_LIKE_A_COORDINATE.finditer(text) if not any(p.start() <= m.start() and m.end() <= p.end() for p in pairs)]
    if not pairs and not unpaired:
        return NoArea()
    if unpaired:
        return Broken("a latitude or longitude has no partner")
    if _is_route(text, pairs):
        return Route(waypoints=len(pairs))
    area, note = build_area(notam)
    if area is None:
        return Broken(note.removeprefix("Q-circle: "))
    if isinstance(area, Polygon) and _crosses_itself(area.points):
        return Broken("the outline crosses itself")
    return Shape(area)
