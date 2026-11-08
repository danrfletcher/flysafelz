"""G20: GET /map/notams's data. The live PIB's NOTAMs active at some moment of [now, now + 24 h) as GeoJSON features, one shape each.

The PIB answers through httpx.MockTransport: the committed trimmed PIB, or one built PIB per case. No socket opens.
"""

import ast
import logging
from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx
import pytest
from mapdata_helpers import FIXTURES, Upstream, notam, pib_of
from pib_helpers import LZ

from flysafelz.clients.pib_source import utc_text
from flysafelz.mapdata.cache import RefreshFailed
from flysafelz.mapdata.notams import WINDOW, in_window, notam_features
from flysafelz.pib import parse_pib
from flysafelz.shapes import notam_geometry

NOW = datetime(2026, 11, 8, 12, 0, tzinfo=UTC)
HOUR = timedelta(hours=1)
MINUTE = timedelta(minutes=1)  # the PIB's times are to the minute
TRIMMED = (FIXTURES / "pib" / "pib_trimmed.xml").read_bytes()
PROPERTIES = {"id", "q_code", "lower_fl", "upper_fl", "item_f", "item_g", "valid_from", "valid_to", "permanent", "est", "item_d", "text", "approx"}


def features(pib: bytes, now: datetime = NOW) -> list[dict]:
    return notam_features(Upstream(pib=pib).factory(), now)


def ids(found: list[dict]) -> list[str]:
    return [f["properties"]["id"] for f in found]


# --- the 24 h window ----------------------------------------------------------------------------------------------------
# (label, start, end, estimated, kept?) -- start and end are offsets from NOW; end None is a permanent NOTAM

WINDOW_CASES = [
    ("active now", -2 * HOUR, 3 * HOUR, False, True),
    ("started a month ago, ends next week", -30 * 24 * HOUR, 7 * 24 * HOUR, False, True),
    ("not yet started, starts in 10 h", 10 * HOUR, 12 * HOUR, False, True),
    ("starts 1 min inside the window", WINDOW - MINUTE, WINDOW + HOUR, False, True),
    ("starts exactly at now + 24 h", WINDOW, WINDOW + HOUR, False, False),
    ("starts after now + 24 h", WINDOW + HOUR, WINDOW + 2 * HOUR, False, False),
    ("ends 1 min after now", -2 * HOUR, MINUTE, False, True),
    ("ends exactly at now", -2 * HOUR, timedelta(0), False, False),
    ("ended an hour ago", -3 * HOUR, -HOUR, False, False),
    ("permanent, started long ago", -90 * 24 * HOUR, None, False, True),
    ("permanent, starts in 10 h", 10 * HOUR, None, False, True),
    ("permanent, starts after now + 24 h", WINDOW + HOUR, None, False, False),
    ("EST, still running", -2 * HOUR, 3 * HOUR, True, True),
    ("EST, estimated end passed", -3 * HOUR, -HOUR, True, False),
]


@pytest.mark.parametrize("label,start,end,estimated,kept", WINDOW_CASES, ids=[row[0] for row in WINDOW_CASES])
def test_the_24_hour_window_rule(label, start, end, estimated, kept):
    element = notam(1, NOW + start, None if end is None else NOW + end, estimation=estimated)
    found = features(pib_of(element))
    assert (ids(found) == ["H1/26"]) is kept
    _issued, parsed = parse_pib(pib_of(element))
    assert in_window(parsed[0], NOW) is kept  # the rule itself, on the parsed NOTAM


def test_an_est_notam_is_served_with_est_true_and_a_firm_one_with_false():
    found = features(pib_of(notam(1, NOW - HOUR, NOW + HOUR, estimation=True), notam(2, NOW - HOUR, NOW + HOUR)))
    assert {f["properties"]["id"]: f["properties"]["est"] for f in found} == {"H1/26": True, "H2/26": False}


def test_a_notam_with_no_end_element_is_open_ended_and_kept():
    from pib_helpers import notam_xml

    open_ended = notam_xml(number=1, start="2611080800", end=None)
    found = features(pib_of(open_ended))
    assert ids(found) == ["H1/26"] and found[0]["properties"]["valid_to"] is None


def test_a_permanent_notam_has_no_end_and_is_marked_permanent():
    found = features(pib_of(notam(1, NOW - HOUR, None)))
    assert found[0]["properties"]["valid_to"] is None and found[0]["properties"]["permanent"] is True


def test_a_replaced_notam_is_dropped_and_its_replacement_kept():
    old, new = notam(10, NOW - HOUR, NOW + HOUR), notam(11, NOW - HOUR, NOW + HOUR, kind="R", referred="H10/26")
    assert ids(features(pib_of(old, new))) == ["H11/26"]


def test_a_cancelled_notam_is_dropped_and_so_is_the_cancelling_notam_when_it_has_no_window_of_its_own():
    old = notam(10, NOW - HOUR, NOW + HOUR)
    cancel = notam(11, NOW - HOUR, NOW + HOUR, kind="C", referred="H10/26")
    assert "H10/26" not in ids(features(pib_of(old, cancel)))


def test_a_replacement_whose_window_has_ended_is_dropped_too():
    old = notam(10, NOW - 5 * HOUR, NOW + HOUR)
    new = notam(11, NOW - 5 * HOUR, NOW - 2 * HOUR, kind="R", referred="H10/26")
    assert features(pib_of(old, new)) == []


def test_a_pib_with_nothing_active_in_24_hours_is_an_empty_list():
    assert features(pib_of(notam(1, NOW - 5 * HOUR, NOW - HOUR), notam(2, NOW + 2 * WINDOW, NOW + 3 * WINDOW))) == []
    assert features(pib_of()) == []  # a PIB with no NOTAM at all


# --- every NOTAM at every height, anywhere ------------------------------------------------------------------------------


def test_a_high_notam_at_fl250_is_in_the_payload_with_its_levels():
    found = features(pib_of(notam(1, NOW - HOUR, NOW + HOUR, lower="250", upper="660")))
    assert ids(found) == ["H1/26"]
    assert (found[0]["properties"]["lower_fl"], found[0]["properties"]["upper_fl"]) == (250, 660)


def test_a_notam_far_from_any_landing_zone_is_in_the_payload():
    far = notam(1, NOW - HOUR, NOW + HOUR, coordinates="5557N00311W", e="DANGER AREA ACTIVE")  # Edinburgh, 200 miles from Blue Mill
    assert LZ[0] < 55 and ids(features(pib_of(far))) == ["H1/26"]


def names_used(module) -> set[str]:
    """Every name the module imports, reads or calls (docstrings and comments are not code)."""
    tree = ast.parse(Path(module.__file__).read_text(encoding="utf-8"))
    found = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            found.add(node.id)
        elif isinstance(node, ast.Attribute):
            found.add(node.attr)
        elif isinstance(node, ast.ImportFrom):
            found.update([node.module or "", *(a.name for a in node.names)])
        elif isinstance(node, ast.Import):
            found.update(a.name for a in node.names)
    return found


def test_the_agents_per_zone_prefilter_is_not_applied():
    import flysafelz.mapdata.notams as module

    assert not {"notam_max_lower_fl", "prefilter", "load_pib", "_drop_reason", "RULES"} & names_used(module)


# --- the feature ----------------------------------------------------------------------------------------------------------


def test_a_feature_has_exactly_the_listed_properties():
    found = features(pib_of(notam(1, NOW - HOUR, NOW + HOUR, d="DAILY 0800-1700", f="SFC", g="800FT AMSL")))
    assert found[0]["type"] == "Feature" and set(found[0]["properties"]) == PROPERTIES
    assert found[0]["properties"] == {
        "id": "H1/26",
        "q_code": "QWULW",
        "lower_fl": 0,
        "upper_fl": 8,
        "item_f": "SFC",
        "item_g": "800FT AMSL",
        "valid_from": utc_text(NOW - HOUR),
        "valid_to": utc_text(NOW + HOUR),
        "permanent": False,
        "est": False,
        "item_d": "DAILY 0800-1700",
        "text": "UAS OPR WI 0.5NM RADIUS OF 534732N 0000459W (GARTON).",
        "approx": False,
    }


def test_absent_f_g_and_d_are_null():
    found = features(pib_of(notam(1, NOW - HOUR, NOW + HOUR, d=None, f=None, g=None)))
    properties = found[0]["properties"]
    assert (properties["item_f"], properties["item_g"], properties["item_d"]) == (None, None, None)


TEXTS = [
    "UAS OPR WI 0.5NM RADIUS OF 534732N 0000459W (GARTON).",
    "LINE ONE\nLINE TWO  WITH  DOUBLE SPACES\n  AND INDENT",
    'CTC 01632 960003 FOR INFORMATION. R&D AREA <TEST> "QUOTED"',  # a fiction-range number (Ofcom drama range), as published
    "TRAILING SPACE ",
]


@pytest.mark.parametrize("text", TEXTS, ids=["circle", "lines", "phone_and_markup", "trailing_space"])
def test_text_is_item_e_exactly_as_published(text):
    from xml.sax.saxutils import escape

    pib = pib_of(notam(1, NOW - HOUR, NOW + HOUR, e=escape(text)))
    _issued, parsed = parse_pib(pib)
    found = features(pib)
    assert found[0]["properties"]["text"] == parsed[0].item_e == text


def test_text_of_the_committed_pib_is_byte_equal_to_item_e_for_every_feature():
    _issued, parsed = parse_pib(TRIMMED)
    wanted = {n.id: n.item_e for n in parsed}
    found = features(TRIMMED, datetime(2026, 11, 5, 18, 0, tzinfo=UTC))
    assert found
    for feature in found:
        assert feature["properties"]["text"].encode("utf-8") == wanted[feature["properties"]["id"]].encode("utf-8")


# --- the committed PIB ------------------------------------------------------------------------------------------------------

TRIMMED_NOW = datetime(2026, 11, 5, 18, 0, tzinfo=UTC)


def test_the_committed_pib_keeps_what_is_active_and_drops_what_ended_or_was_replaced():
    found = features(TRIMMED, TRIMMED_NOW)
    assert sorted(ids(found)) == ["A3857/26", "C7072/26", "H7171/26", "Q172/26", "V34/26"]
    # J3857, J3864 and H7297 ended before now; J3377 ended on 31 Oct; V33 (replaced by V34) is not in the file


def test_the_committed_pib_a_day_earlier_adds_the_notam_that_has_not_yet_started():
    """A3857/26 starts 2026-11-05 18:45Z. It is in the window from one minute after 2026-11-04 18:45Z, and not at that instant."""
    for now, kept in [
        (datetime(2026, 11, 4, 20, 0, tzinfo=UTC), True),  # starts in 22 h 45 min
        (datetime(2026, 11, 4, 18, 46, tzinfo=UTC), True),  # 23 h 59 min
        (datetime(2026, 11, 4, 18, 45, tzinfo=UTC), False),  # exactly 24 h: out
        (datetime(2026, 11, 4, 18, 0, tzinfo=UTC), False),  # 24 h 45 min
    ]:
        assert ("A3857/26" in ids(features(TRIMMED, now))) is kept, now


# --- one shape, and approx --------------------------------------------------------------------------------------------------


def test_every_geometry_and_approx_flag_is_what_shapes_notam_geometry_returns():
    _issued, parsed = parse_pib(TRIMMED)
    by_id = {n.id: n for n in parsed}
    found = features(TRIMMED, TRIMMED_NOW)
    for feature in found:
        geometry, approx = notam_geometry(by_id[feature["properties"]["id"]])
        assert feature["geometry"] == geometry and feature["properties"]["approx"] is approx


def test_a_built_shape_is_not_approx_and_a_q_circle_fallback_is():
    flags = {f["properties"]["id"]: f["properties"]["approx"] for f in features(TRIMMED, TRIMMED_NOW)}
    assert flags["A3857/26"] is False  # item E gives a circle at a point
    assert flags["C7072/26"] is True  # nothing in item E to draw: the Q-line's circle
    assert True in flags.values() and False in flags.values()


def test_the_route_never_builds_a_shape_itself():
    import flysafelz.mapdata.notams as module

    used = names_used(module)
    assert not {"build_area", "flysafelz.notam_area"} & used and "notam_geometry" in used


# --- the PIB cannot be read ---------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("status", [500, 502, 503, 504, 404])
def test_a_pib_that_answers_5xx_or_404_fails_the_refresh_with_the_status(status):
    upstream = Upstream(pib=pib_of())
    upstream.pib_status = status
    with pytest.raises(RefreshFailed) as raised:
        notam_features(upstream.factory(), NOW)
    assert raised.value.reason == f"NATS PIB: HTTP {status}"


def test_a_pib_that_is_unreachable_fails_the_refresh():
    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectTimeout("timed out")

    with pytest.raises(RefreshFailed, match="ConnectTimeout"):
        notam_features(httpx.Client(transport=httpx.MockTransport(refuse)), NOW)


@pytest.mark.parametrize("body", [b"", b"<html>maintenance</html>", b"<Pib><oops", b'{"json": true}'])
def test_a_body_that_is_not_a_pib_fails_the_refresh_with_a_fixed_one_line_reason(body):
    with pytest.raises(RefreshFailed) as raised:
        features(body)
    assert raised.value.reason == "NATS PIB: unreadable"


def test_a_notam_that_cannot_be_read_fails_the_refresh_without_quoting_the_pib(caplog):
    secret = "SECRET E TEXT 01632960000"
    broken = pib_of(notam(1, NOW - HOUR, NOW + HOUR, e=secret, omit=("QLine",)))
    with caplog.at_level(logging.DEBUG), pytest.raises(RefreshFailed) as raised:
        features(broken)
    assert secret not in raised.value.reason and secret not in caplog.text and "\n" not in raised.value.reason


def test_nothing_in_a_good_refresh_logs_the_notam_text(caplog):
    secret = "SECRET E TEXT 01632960000"
    with caplog.at_level(logging.DEBUG):
        found = features(pib_of(notam(1, NOW - HOUR, NOW + HOUR, e=secret)))
    assert found[0]["properties"]["text"] == secret and secret not in caplog.text
