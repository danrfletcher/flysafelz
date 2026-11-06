"""G7: fetch_airspace and parse_airspace against httpx.MockTransport. No socket opens; `now` is frozen; no key is read from the environment."""

import json
from pathlib import Path

import httpx
import pytest
from airspace_helpers import LZ, utc, with_rules

from flysafelz.clients import openaip
from flysafelz.clients.openaip import KEY_HEADER, URL, OpenAipError, fetch_airspace, parse_airspace
from flysafelz.contracts import Datum, LimitUnit, SourceName, SourceState
from flysafelz.geo import distance_nm, shape_over

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "openaip"
NOW = utc(7, 30)
FAKE_KEY = "test-key-not-real"


def page(name: str) -> dict:
    return json.loads((FIXTURES / f"{name}.json").read_text(encoding="utf-8"))


class Server:
    """A fake openAIP. Serves page 1 and page 2 from the fixtures and records every request."""

    def __init__(self, pages=None, *, status=200, body=None, raises=None):
        self.pages = pages if pages is not None else {"1": page("page1"), "2": page("page2")}
        self.status = status
        self.body = body
        self.raises = raises
        self.seen: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.seen.append(request)
        if self.raises:
            raise self.raises
        if self.status != 200:
            return httpx.Response(self.status)
        if self.body is not None:
            return httpx.Response(200, content=self.body)
        return httpx.Response(200, json=self.pages[request.url.params["page"]])

    def client(self) -> httpx.Client:
        return httpx.Client(transport=httpx.MockTransport(self))


def fetch(server: Server, **kwargs):
    kwargs.setdefault("api_key", FAKE_KEY)
    return fetch_airspace(*LZ, client=server.client(), now=NOW, **kwargs)


# --- the request -----------------------------------------------------------------


def test_the_request_goes_to_the_airspaces_endpoint_with_the_key_header():
    server = Server()
    fetch(server)
    first = server.seen[0]
    assert str(first.url).split("?")[0] == URL == "https://api.core.openaip.net/api/airspaces"
    assert first.method == "GET"
    assert first.headers[KEY_HEADER] == FAKE_KEY and KEY_HEADER == "x-openaip-api-key"
    assert FAKE_KEY not in str(first.url)  # the key is a header, never in the URL


def test_the_bbox_is_the_box_rule_either_side_of_the_lz():
    server = Server()
    fetch(server)
    min_lon, min_lat, max_lon, max_lat = (float(x) for x in server.seen[0].url.params["bbox"].split(","))
    assert min_lon < LZ[1] < max_lon and min_lat < LZ[0] < max_lat
    half_width = 10
    assert distance_nm(LZ[0], LZ[1], max_lat, LZ[1]) == pytest.approx(half_width, rel=0.005)
    assert distance_nm(LZ[0], LZ[1], min_lat, LZ[1]) == pytest.approx(half_width, rel=0.005)
    # the box is at least the half-width wide to the east and west too
    assert distance_nm(max_lat, LZ[1], max_lat, max_lon) >= half_width and distance_nm(max_lat, LZ[1], max_lat, min_lon) >= half_width
    assert half_width == openaip.RULES.airspace_box_nm.value


def test_a_changed_box_rule_changes_the_bbox():
    server = Server()
    fetch(server, rules=with_rules(airspace_box_nm=5))
    min_lon, min_lat, max_lon, max_lat = (float(x) for x in server.seen[0].url.params["bbox"].split(","))
    assert distance_nm(LZ[0], LZ[1], max_lat, LZ[1]) == pytest.approx(5, rel=0.005)


def test_every_page_is_followed_and_a_repeated_item_is_kept_once():
    server = Server()
    result = fetch(server)
    assert [r.url.params["page"] for r in server.seen] == ["1", "2"]
    assert [a.id[-1] for a in result.areas] == ["1", "2", "3", "4", "5", "6"]  # item 3 is on both pages
    assert result.status.state is SourceState.OK
    assert result.status.source is SourceName.AIRSPACE and result.status.fetched_at == NOW


def test_the_airspaces_are_read_into_models():
    result = fetch(Server())
    by_name = {a.name: a for a in result.areas}
    egd = by_name["EGD496A"]
    assert egd.upper is not None
    assert (egd.type, egd.by_notam, egd.designator, egd.lower.datum, egd.upper.value) == ("danger", True, "EGD496A", Datum.SFC, 1300)
    assert shape_over(*LZ, egd.shape)  # Blue Mill is inside the area
    cta = by_name["NEWCASTLE CTA 3"]
    assert (cta.type, cta.icao_class, cta.lower.value, cta.lower.datum) == ("cta", "D", 1500, Datum.AMSL)


def test_an_empty_result_is_ok_with_no_airspaces():
    result = fetch(Server(pages={"1": page("empty")}))
    assert result.status.state is SourceState.OK and result.areas == []


# --- failure: never an exception -----------------------------------------------------


@pytest.mark.parametrize("status", [401, 403, 429, 500, 503])
def test_an_http_error_is_a_failed_status_with_the_code(status):
    result = fetch(Server(status=status))
    assert result.status.state is SourceState.FAILED
    assert f"HTTP {status}" in result.status.reason and result.areas == []
    assert result.status.fetched_at == NOW


@pytest.mark.parametrize("error", [httpx.ReadTimeout("slow"), httpx.ConnectError("down")])
def test_a_timeout_or_connection_error_is_failed_with_the_exception_class(error):
    result = fetch(Server(raises=error))
    assert result.status.state is SourceState.FAILED
    assert type(error).__name__ in result.status.reason


@pytest.mark.parametrize("body", [b"<html>not json", b"", b'{"items": '])
def test_invalid_json_is_failed(body):
    result = fetch(Server(body=body))
    assert result.status.state is SourceState.FAILED and "not JSON" in result.status.reason


@pytest.mark.parametrize("body", [b"[]", b'{"items": "nope"}', b'{"page": 1}'])
def test_a_body_that_is_not_an_airspace_page_is_failed(body):
    result = fetch(Server(body=body))
    assert result.status.state is SourceState.FAILED and "not an airspace page" in result.status.reason


def test_a_bad_total_pages_is_failed():
    result = fetch(Server(pages={"1": {"items": [], "totalPages": "many"}}))
    assert result.status.state is SourceState.FAILED and "totalPages" in result.status.reason


def test_a_server_that_never_finishes_is_stopped():
    result = fetch(Server(pages={str(n): {"items": [], "totalPages": 10_000} for n in range(1, 100)}))
    assert result.status.state is SourceState.FAILED and "more pages" in result.status.reason


def test_an_item_with_no_usable_polygon_fails_the_fetch():
    bad = {"items": [{"_id": "x", "name": "NO SHAPE", "type": 4}], "totalPages": 1}
    result = fetch(Server(pages={"1": bad}))
    assert result.status.state is SourceState.FAILED
    assert "NO SHAPE" in result.status.reason and result.areas == []


@pytest.mark.parametrize("key", [None, "", "   "])
def test_a_missing_key_fails_with_a_reason_naming_the_variable_and_makes_no_request(key):
    server = Server()
    result = fetch(server, api_key=key)
    assert result.status.state is SourceState.FAILED
    assert result.status.reason == "OPENAIP_API_KEY is not set" and "\n" not in result.status.reason
    assert server.seen == []


def test_the_key_is_never_in_a_reason():
    for server in (Server(status=401), Server(raises=httpx.ReadTimeout("slow")), Server(body=b"x")):
        assert FAKE_KEY not in fetch(server).status.reason


def test_without_a_client_a_missing_key_still_fails_before_any_connection():
    result = fetch_airspace(*LZ, api_key=None, now=NOW)  # sockets are disabled: a request would raise
    assert result.status.state is SourceState.FAILED and "OPENAIP_API_KEY" in result.status.reason


def test_the_clock_is_read_only_when_no_now_is_given():
    result = fetch_airspace(*LZ, api_key=None)
    assert result.status.fetched_at is not None and result.status.fetched_at.tzinfo is not None


# --- parse_airspace: the code table ------------------------------------------------------


SQUARE = {"type": "Polygon", "coordinates": [[[0.0, 53.0], [0.1, 53.0], [0.1, 53.1], [0.0, 53.1], [0.0, 53.0]]]}


def item(**fields):
    base = {"_id": "i1", "name": "Test Airspace", "type": 4, "icaoClass": 3, "geometry": SQUARE}
    base.update(fields)
    return base


@pytest.mark.parametrize(
    ("code", "name"),
    [(0, "other"), (1, "restricted"), (2, "danger"), (3, "prohibited"), (4, "ctr"), (5, "tmz"), (6, "rmz"), (7, "tma"), (13, "atz"), (14, "matz"), (15, "airway"), (21, "gliding"), (26, "cta"), (36, "mctr")],
)
def test_type_codes(code, name):
    area = parse_airspace(item(type=code))
    assert area.type == name and area.type_code == code
    assert not any("unknown airspace type" in n for n in area.notes)


@pytest.mark.parametrize("code", [99, -1, None, "4", 4.5])
def test_an_unknown_type_code_is_other_and_noted(code):
    area = parse_airspace(item(type=code))
    assert area.type == "other"
    assert any("unknown airspace type" in n for n in area.notes)


@pytest.mark.parametrize(("code", "letter"), [(0, "A"), (1, "B"), (2, "C"), (3, "D"), (4, "E"), (5, "F"), (6, "G"), (8, "SUA"), (7, None), (42, None), (None, None)])
def test_class_codes(code, letter):
    assert parse_airspace(item(icaoClass=code)).icao_class == letter


@pytest.mark.parametrize(
    ("raw", "value", "unit", "datum"),
    [
        ({"value": 0, "unit": 1, "referenceDatum": 0}, 0, LimitUnit.FT, Datum.SFC),
        ({"value": 1500, "unit": 1, "referenceDatum": 1}, 1500, LimitUnit.FT, Datum.AMSL),
        ({"value": 400, "unit": 1, "referenceDatum": 0}, 400, LimitUnit.FT, Datum.AGL),
        ({"value": 65, "unit": 6, "referenceDatum": 2}, 65, LimitUnit.FL, Datum.AMSL),
        ({"value": 3000, "unit": 1, "referenceDatum": 2}, 3000, LimitUnit.FT, Datum.AMSL),
    ],
)
def test_limit_codes(raw, value, unit, datum):
    area = parse_airspace(item(lowerLimit=raw))
    assert (area.lower.value, area.lower.unit, area.lower.datum) == (value, unit, datum)
    assert not any("lower limit" in n for n in area.notes)


def test_a_limit_in_metres_is_converted_to_feet():
    area = parse_airspace(item(lowerLimit={"value": 300, "unit": 0, "referenceDatum": 1}))
    assert area.lower.unit is LimitUnit.FT and area.lower.value == pytest.approx(984.25, abs=0.01)


def test_ground_with_a_zero_in_metres_is_sfc():
    area = parse_airspace(item(lowerLimit={"value": 0, "unit": 0, "referenceDatum": 0}))
    assert area.lower.datum is Datum.SFC


def test_a_missing_lower_limit_is_sfc_and_a_missing_upper_is_unlimited_and_both_are_noted():
    area = parse_airspace(item())
    assert area.lower.datum is Datum.SFC and area.upper is None
    assert any("lower limit missing" in n and "SFC" in n for n in area.notes)
    assert any("upper limit missing" in n and "unlimited" in n for n in area.notes)


@pytest.mark.parametrize(
    "raw",
    [{"value": "high", "unit": 1, "referenceDatum": 1}, {"value": 5, "unit": 9, "referenceDatum": 1}, {"value": 5, "unit": 1, "referenceDatum": 9}, {"unit": 1}, "FL65", {"value": True, "unit": 1, "referenceDatum": 1}],
)
def test_an_unreadable_limit_is_read_the_fail_closed_way(raw):
    area = parse_airspace(item(lowerLimit=raw, upperLimit=raw))
    assert area.lower.datum is Datum.SFC and area.upper is None
    assert len(area.notes) == 2


# --- parse_airspace: hours, designators, geometry -----------------------------------------


def hours(*entries):
    return {"operatingHours": list(entries)}


def entry(day=0, start="08:00", end="18:00", **extra):
    return {"dayOfWeek": day, "startTime": start, "endTime": end, "sunrise": False, "sunset": False, "byNotam": False, **extra}


@pytest.mark.parametrize(
    ("raw", "text"),
    [
        (hours(entry()), "DAILY 0800-1800"),
        (hours(entry(1), entry(2, "09:00", "10:00")), "MON 0800-1800, TUE 0900-1000"),
        (hours(entry(7, sunrise=True, sunset=True)), "SUN SR-SS"),
        (hours(entry(0, "00:00", "23:59")), "H24"),
        (hours(entry(0, "00:00", "24:00")), "H24"),
        (None, None),
        ({}, None),
        (hours(), None),
    ],
)
def test_hours_of_operation_become_schedule_text(raw, text):
    area = parse_airspace(item(hoursOfOperation=raw))
    assert area.hours == text and area.hours_readable is True


@pytest.mark.parametrize("raw", [hours(entry(9)), hours(entry(start="8am")), hours("never"), {"operatingHours": "daily"}])
def test_hours_that_cannot_be_read_are_marked_and_noted(raw):
    area = parse_airspace(item(hoursOfOperation=raw))
    assert area.hours is None and area.hours_readable is False
    assert any("hoursOfOperation could not be read" in n for n in area.notes)


def test_by_notam_comes_from_the_flag_or_from_the_hours():
    assert parse_airspace(item(byNotam=True)).by_notam is True
    assert parse_airspace(item(hoursOfOperation=hours(entry(byNotam=True)))).by_notam is True
    only_notam = parse_airspace(item(hoursOfOperation=hours({"dayOfWeek": 0, "byNotam": True})))
    assert only_notam.by_notam is True and only_notam.hours is None and only_notam.hours_readable is True
    assert parse_airspace(item()).by_notam is False


@pytest.mark.parametrize("name", ["D496A", "EGD 496A", "EGD496A", "EG D496A", "D 496A TDA"])
def test_a_designator_written_any_way_is_the_same_designator(name):
    assert parse_airspace(item(name=name)).designator == "EGD496A"


def test_no_designator_in_the_name_is_none():
    assert parse_airspace(item(name="HUMBERSIDE CTR")).designator is None


def test_a_valid_polygon_is_kept_as_lat_lon_points():
    area = parse_airspace(item())
    assert area.shape[0][0][0] == (53.0, 0.0)  # the GeoJSON [lon, lat] is flipped to (lat, lon)
    assert not any("polygon" in n for n in area.notes)


def test_a_self_intersecting_polygon_is_repaired_with_buffer_zero():
    bow = {"type": "Polygon", "coordinates": [[[0, 0], [1, 1], [1, 0], [0, 1], [0, 0]]]}
    area = parse_airspace(item(geometry=bow))
    assert any("repaired with buffer(0)" in n for n in area.notes)
    assert len(area.shape) >= 1 and all(len(ring) >= 4 for poly in area.shape for ring in poly)


def test_a_polygon_that_cannot_be_repaired_becomes_its_bounding_box():
    line = {"type": "Polygon", "coordinates": [[[0, 0], [1, 1], [2, 2], [0, 0]]]}  # three points on one line: no area
    area = parse_airspace(item(geometry=line))
    assert any("bounding box" in n and "fail closed" in n for n in area.notes)
    assert shape_over(1.0, 0.5, area.shape)  # the box covers what the line only touched


def test_a_multipolygon_keeps_every_part_and_a_hole():
    ring = [[0, 0], [4, 0], [4, 4], [0, 4], [0, 0]]
    hole = [[1, 1], [2, 1], [2, 2], [1, 2], [1, 1]]
    far = [[10, 10], [11, 10], [11, 11], [10, 11], [10, 10]]
    area = parse_airspace(item(geometry={"type": "MultiPolygon", "coordinates": [[ring, hole], [far]]}))
    assert len(area.shape) == 2 and len(area.shape[0]) == 2
    assert not shape_over(1.5, 1.5, area.shape) and shape_over(3.0, 3.0, area.shape) and shape_over(10.5, 10.5, area.shape)


@pytest.mark.parametrize("geometry", [None, {}, {"type": "Point", "coordinates": [0, 0]}, {"type": "Polygon", "coordinates": []}, {"type": "Polygon", "coordinates": [[[0, 0], [1, 1]]]}, {"type": "Polygon", "coordinates": [[[0, 95], [1, 95], [1, 96]]]}])
def test_no_usable_polygon_raises(geometry):
    with pytest.raises(OpenAipError):
        parse_airspace(item(geometry=geometry))


def test_an_item_that_is_not_an_object_raises():
    with pytest.raises(OpenAipError):
        parse_airspace("not an item")  # type: ignore[arg-type]


def test_an_unnamed_item_gets_a_name_and_an_id():
    area = parse_airspace(item(_id=None, name=None))
    assert area.name == "(unnamed airspace)" and area.id == area.name


def test_the_fixture_with_an_unknown_type_and_a_bowtie_reads_through_the_fetch():
    result = fetch(Server())
    by_name = {a.name: a for a in result.areas}
    assert by_name["WIDGET MYSTERY ZONE"].type == "other"
    assert any("unknown airspace type code 99" in n for n in by_name["WIDGET MYSTERY ZONE"].notes)
    assert by_name["WIDGET MYSTERY ZONE"].lower.datum is Datum.SFC and by_name["WIDGET MYSTERY ZONE"].upper is None
    rmz = by_name["BOWTIE RMZ"]
    assert rmz.hours == "DAILY SR-SS" and any("buffer(0)" in n for n in rmz.notes)
    assert by_name["D323A WEEKDAY RANGE"].designator == "EGD323A"
    assert by_name["D323A WEEKDAY RANGE"].hours == "MON 0800-1700, TUE 0800-1700, WED 0800-1700, THU 0800-1700, FRI 0800-1700"
