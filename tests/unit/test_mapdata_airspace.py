"""G19: GET /map/airspace's data. Every GB airspace from openAIP as GeoJSON features, all pages joined, never boxed, and a failed page fails the refresh.

openAIP answers through httpx.MockTransport (tests/fixtures/openaip); no socket opens and the key is a made-up string.
"""

import copy
import logging

import httpx
import pytest
from mapdata_helpers import FAKE_KEY, Upstream, page
from shapely.geometry import Polygon

from flysafelz.clients.openaip import parse_airspace
from flysafelz.contracts import Datum, Limit, LimitUnit
from flysafelz.mapdata.airspace import airspace_features, limit_feet, limit_text
from flysafelz.mapdata.cache import RefreshFailed
from flysafelz.shapes import airspace_geometry

PROPERTIES = {"id", "name", "type", "class", "lower", "upper", "lower_ft", "upper_ft", "hours", "action"}


def features(upstream: Upstream | None = None) -> list[dict]:
    upstream = upstream or Upstream()
    return airspace_features(upstream.factory(), FAKE_KEY)


def by_name(found: list[dict]) -> dict[str, dict]:
    return {f["properties"]["name"]: f for f in found}


# --- the shape ----------------------------------------------------------------------------------------------------------


def test_the_two_fixture_pages_are_joined_into_geojson_features():
    upstream = Upstream()
    found = features(upstream)
    assert [r.url.params["page"] for r in upstream.airspace_requests] == ["1", "2"]
    assert all(f["type"] == "Feature" for f in found)
    assert [f["properties"]["name"] for f in found] == [
        "EGD496A",
        "NEWCASTLE CTA 3",
        "HUMBERSIDE CTR",
        "EDINBURGH CTR",
        "JUST OUTSIDE TMA",
        "D323A WEEKDAY RANGE",
        "WIDGET MYSTERY ZONE",
        "BOWTIE RMZ",
    ]  # HUMBERSIDE CTR is on both pages and appears once; FAR NO POLYGON cannot be drawn and is left out


def test_every_feature_has_exactly_the_listed_properties_and_a_closed_polygon_in_lon_lat_order():
    for feature in features():
        assert set(feature["properties"]) == PROPERTIES
        geometry = feature["geometry"]
        assert geometry["type"] in {"Polygon", "MultiPolygon"}
        rings = geometry["coordinates"] if geometry["type"] == "Polygon" else [ring for polygon in geometry["coordinates"] for ring in polygon]
        for ring in rings:
            assert ring[0] == ring[-1] and len(ring) >= 4
            for lon, lat in ring:
                assert 49 < lat < 61 and -9 < lon < 3  # GB: latitude is the second number, so a swap would fail here


def test_the_coordinates_are_the_fixtures_own_lon_lat_pairs():
    humberside = by_name(features())["HUMBERSIDE CTR"]
    source = next(i for i in page("page1")["items"] if i["name"] == "HUMBERSIDE CTR")
    assert humberside["geometry"]["coordinates"][0][0] == pytest.approx(source["geometry"]["coordinates"][0][0])


def test_the_properties_of_a_ctr_a_range_with_hours_and_a_zone_with_no_class():
    found = by_name(features())
    assert found["HUMBERSIDE CTR"]["properties"] == {
        "id": "aaa000000000000000000003",
        "name": "HUMBERSIDE CTR",
        "type": "ctr",
        "class": "D",
        "lower": "SFC",
        "upper": "3500 ft AMSL",
        "lower_ft": 0,
        "upper_ft": 3500,
        "hours": None,
        "action": None,
    }
    assert found["D323A WEEKDAY RANGE"]["properties"]["hours"] == "MON 0800-1700, TUE 0800-1700, WED 0800-1700, THU 0800-1700, FRI 0800-1700"
    assert found["EGD496A"]["properties"]["hours"] == "by NOTAM"
    assert found["BOWTIE RMZ"]["properties"]["upper"] == "FL60" and found["BOWTIE RMZ"]["properties"]["upper_ft"] == 6000


def test_the_base_layer_maps_no_action():
    assert {f["properties"]["action"] for f in features()} == {None}


def test_an_unknown_type_is_other_and_an_item_without_limits_is_unlimited_from_the_surface():
    widget = by_name(features())["WIDGET MYSTERY ZONE"]["properties"]
    assert (widget["type"], widget["class"]) == ("other", None)
    assert (widget["lower"], widget["upper"], widget["lower_ft"], widget["upper_ft"]) == ("SFC", "UNL", 0, None)


def repair_page(name: str, ring: list[list[float]]) -> dict:
    """One openAIP page with one airspace whose polygon is `ring` (lon, lat pairs), taken from a fixture item so every other field is real."""
    body = one_item_page(None, None)
    item = body["items"][0]
    item.update(name=name, geometry={"type": "Polygon", "coordinates": [ring]})
    return body


def test_an_item_whose_invalid_polygon_parse_airspace_repairs_with_buffer_zero_keeps_its_feature_as_the_parser_gives_it():
    bowtie = [[-1, 54], [-0.5, 54.5], [-0.5, 54], [-1, 54.5], [-1, 54]]  # two edges cross: invalid, and buffer(0) repairs it
    body = repair_page("BOWTIE REPAIRED ZONE", bowtie)
    parsed = parse_airspace(body["items"][0])
    assert any("repaired with buffer(0)" in note for note in parsed.notes)
    found = features(Upstream(airspace={"1": body}))
    assert [f["properties"]["name"] for f in found] == ["BOWTIE REPAIRED ZONE"]
    assert found[0]["geometry"] == airspace_geometry(parsed.shape)
    ring = found[0]["geometry"]["coordinates"][0]
    assert ring != bowtie and Polygon(ring).is_valid  # the repaired shape, not the item's own crossing ring


def test_an_item_whose_invalid_polygon_falls_back_on_its_bounding_box_keeps_its_feature_with_that_box():
    line = [[-1, 54], [-0.5, 54.5], [0, 55], [-1, 54]]  # all on one diagonal: no area, buffer(0) leaves nothing, so the bounding box is used
    body = repair_page("DIAGONAL BOX ZONE", line)
    parsed = parse_airspace(body["items"][0])
    assert any("bounding box is used" in note for note in parsed.notes)
    found = features(Upstream(airspace={"1": body}))
    assert [f["properties"]["name"] for f in found] == ["DIAGONAL BOX ZONE"]
    assert found[0]["geometry"] == airspace_geometry(parsed.shape)
    ring = found[0]["geometry"]["coordinates"][0]
    assert {tuple(point) for point in ring} == {(-1, 54), (0, 54), (0, 55), (-1, 55)}  # the box round the item's own points, in lon, lat order


def test_an_item_with_no_polygon_is_left_out_and_the_log_gives_the_count_not_the_item(caplog):
    body = repair_page("NO POLYGON ZONE", [])
    body["items"][0]["geometry"] = {}
    with caplog.at_level(logging.WARNING, logger="flysafelz.mapdata"):
        assert features(Upstream(airspace={"1": body})) == []
    assert "1 airspace items have no usable polygon" in caplog.text and "NO POLYGON ZONE" not in caplog.text


def test_an_empty_page_is_an_empty_list():
    assert features(Upstream(airspace={"1": page("empty")})) == []


# --- limits: SFC / GND / FL / ALT ---------------------------------------------------------------------------------------

LIMITS = [
    (Limit(value=0, unit=LimitUnit.FT, datum=Datum.SFC), "SFC", 0),
    (Limit(value=1500, unit=LimitUnit.FT, datum=Datum.AGL), "1500 ft AGL", 1500),
    (Limit(value=4500, unit=LimitUnit.FT, datum=Datum.AMSL), "4500 ft AMSL", 4500),
    (Limit(value=65, unit=LimitUnit.FL, datum=Datum.AMSL), "FL65", 6500),
    (Limit(value=195, unit=LimitUnit.FL, datum=Datum.AMSL), "FL195", 19500),
    (Limit(value=1312.34, unit=LimitUnit.FT, datum=Datum.AMSL), "1312 ft AMSL", 1312),  # a metre limit converted to feet
    (None, "UNL", None),
]


@pytest.mark.parametrize("limit,text,feet", LIMITS, ids=[t for _, t, _ in LIMITS])
def test_limit_text_and_feet(limit, text, feet):
    assert limit_text(limit) == text and limit_feet(limit) == feet


def one_item_page(lower: dict | None, upper: dict | None) -> dict:
    body = copy.deepcopy(page("page1"))
    item = body["items"][0]
    body.update(nextPage=None, totalPages=1, items=[item])
    for key, value in (("lowerLimit", lower), ("upperLimit", upper)):
        if value is None:
            item.pop(key, None)
        else:
            item[key] = value
    return body


OPENAIP_LIMITS = [
    ("GND at zero", {"value": 0, "unit": 1, "referenceDatum": 0}, "SFC", 0),
    ("GND above zero", {"value": 500, "unit": 1, "referenceDatum": 0}, "500 ft AGL", 500),
    ("ALT", {"value": 2500, "unit": 1, "referenceDatum": 1}, "2500 ft AMSL", 2500),
    ("FL", {"value": 85, "unit": 6, "referenceDatum": 2}, "FL85", 8500),
    ("metres", {"value": 1000, "unit": 0, "referenceDatum": 1}, "3281 ft AMSL", 3281),
]


@pytest.mark.parametrize("label,raw,text,feet", OPENAIP_LIMITS, ids=[row[0] for row in OPENAIP_LIMITS])
def test_a_limit_read_from_openaip_reaches_the_properties_as_text_and_feet(label, raw, text, feet):
    found = features(Upstream(airspace={"1": one_item_page(None, raw)}))
    assert (found[0]["properties"]["upper"], found[0]["properties"]["upper_ft"]) == (text, feet)
    found = features(Upstream(airspace={"1": one_item_page(raw, None)}))
    assert (found[0]["properties"]["lower"], found[0]["properties"]["lower_ft"]) == (text, feet)


# --- all of GB, not a box -----------------------------------------------------------------------------------------------


def test_the_request_asks_for_the_whole_country_and_never_sends_a_box_or_a_position():
    upstream = Upstream()
    features(upstream)
    for request in upstream.airspace_requests:
        assert set(request.url.params) == {"country", "limit", "page"} and request.url.params["country"] == "GB"
        assert request.headers["x-openaip-api-key"] == FAKE_KEY and FAKE_KEY not in str(request.url)


def test_an_airspace_far_from_any_landing_zone_is_kept():
    names = set(by_name(features()))
    assert {"EDINBURGH CTR", "JUST OUTSIDE TMA"} <= names  # ticket 2's fetch_airspace drops these as outside the box


def test_the_route_does_not_call_the_boxed_fetch():
    import flysafelz.mapdata.airspace as module

    assert not hasattr(module, "fetch_airspace")


# --- a failure fails the whole refresh ----------------------------------------------------------------------------------


def test_a_second_page_that_fails_fails_the_refresh_and_returns_no_partial_list():
    upstream = Upstream()
    original = upstream.__call__

    def second_page_down(request: httpx.Request) -> httpx.Response:
        if request.url.params["page"] == "2":
            return httpx.Response(500)
        return original(request)

    client = httpx.Client(transport=httpx.MockTransport(second_page_down))
    with pytest.raises(RefreshFailed, match="openAIP"):
        airspace_features(client, FAKE_KEY)


@pytest.mark.parametrize("status", [401, 403, 429, 500, 503])
def test_an_error_status_fails_the_refresh_with_a_one_line_reason_without_the_key(status):
    upstream = Upstream()
    upstream.airspace_status = status
    with pytest.raises(RefreshFailed) as raised:
        features(upstream)
    assert "\n" not in raised.value.reason and FAKE_KEY not in raised.value.reason and str(status) in raised.value.reason


def test_an_html_page_with_status_200_fails_the_refresh():
    upstream = Upstream()
    upstream.airspace_html = True
    with pytest.raises(RefreshFailed, match="not JSON"):
        features(upstream)


@pytest.mark.parametrize("body", [[], "text", {"items": "no"}, {"nothing": 1}, {"items": None}])
def test_a_json_body_that_is_not_an_openaip_page_fails_the_refresh(body):
    with pytest.raises(RefreshFailed, match="not an airspace page"):
        features(Upstream(airspace={"1": body}))


def test_a_transport_error_fails_the_refresh():
    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    with pytest.raises(RefreshFailed, match="ConnectError"):
        airspace_features(httpx.Client(transport=httpx.MockTransport(refuse)), FAKE_KEY)


@pytest.mark.parametrize("key", [None, "", "   "])
def test_no_key_fails_the_refresh_before_any_request(key):
    upstream = Upstream()
    with pytest.raises(RefreshFailed, match="OPENAIP_API_KEY is not set"):
        airspace_features(upstream.factory(), key)
    assert upstream.calls == 0


def test_the_key_never_reaches_a_reason_or_a_log_even_when_openaip_echoes_it(caplog):
    def echoing(request: httpx.Request) -> httpx.Response:
        return httpx.Response(403, text=f"denied for {request.headers['x-openaip-api-key']}")

    with caplog.at_level(logging.DEBUG), pytest.raises(RefreshFailed) as raised:
        airspace_features(httpx.Client(transport=httpx.MockTransport(echoing)), FAKE_KEY)
    assert FAKE_KEY not in raised.value.reason and FAKE_KEY not in caplog.text
    assert raised.value.reason == "openAIP: HTTP 403"  # the reason is the status line; the response body is never quoted


def test_a_good_refresh_logs_the_count_of_unusable_items_and_not_the_items(caplog):
    with caplog.at_level(logging.INFO):
        features()
    assert "1 airspace items have no usable polygon" in caplog.text
    assert "FAR NO POLYGON" not in caplog.text


def test_a_next_page_that_does_not_move_forward_fails_the_refresh():
    body = {**page("page1"), "nextPage": 1}
    with pytest.raises(RefreshFailed, match="nextPage is not a later page number"):
        features(Upstream(airspace={"1": body}))


def test_a_server_that_never_says_it_is_done_fails_the_refresh_after_the_page_cap():
    from flysafelz.clients.openaip import MAX_PAGES

    seen = []

    def endless(request: httpx.Request) -> httpx.Response:
        number = int(request.url.params["page"])
        seen.append(number)
        return httpx.Response(200, json={**page("page1"), "nextPage": number + 1})

    with pytest.raises(RefreshFailed, match="more pages than expected"):
        airspace_features(httpx.Client(transport=httpx.MockTransport(endless)), FAKE_KEY)
    assert seen == list(range(1, MAX_PAGES + 1))
