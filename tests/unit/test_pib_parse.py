"""G12, E4, E5: parse_pib reads the real PIB shape into Parsed NOTAMs; dedupe, linkage, PERM/EST, failures."""

from datetime import UTC, datetime

import pytest
from pib_helpers import TRIMMED, notam_xml, pib_xml

from flysafelz.contracts import NotamType
from flysafelz.pib import PibError, drop_superseded, parse_pib, read_notams


def by_id(xml: bytes | str) -> dict:
    _issued, notams = parse_pib(xml)
    return {n.id: n for n in notams}


# --- the real (trimmed) PIB ----------------------------------------------------------------------


def test_issued_at_comes_from_the_area_pib_header():
    issued, _notams = parse_pib(TRIMMED.read_bytes())
    assert issued == datetime(2026, 10, 31, 7, 0, 37, tzinfo=UTC)


def test_h7171_fields_are_extracted():
    n = by_id(TRIMMED.read_bytes())["H7171/26"]
    assert (n.series, n.number, n.year, n.type, n.replaces, n.nof) == ("H", 7171, 26, NotamType.NEW, None, "EGGN")
    assert (n.fir, n.q_code, n.subject, n.condition) == ("EGTT", "QWULW", "WU", "LW")
    assert (n.traffic, n.purpose, n.scope, n.lower_fl, n.upper_fl) == ("IV", "BO", "W", 0, 8)
    assert (n.centre_lat, n.centre_lon, n.radius_nm) == pytest.approx((53 + 48 / 60, -5 / 60, 1.0), abs=1e-9)
    assert n.item_a == "EGTT"
    assert n.valid_from == datetime(2026, 10, 29, 8, 0, tzinfo=UTC)
    assert n.valid_to == datetime(2026, 11, 10, 18, 0, tzinfo=UTC)
    assert (n.permanent, n.est) == (False, False)
    assert n.item_d == "0800-1800"
    assert n.item_e.startswith("UAS OPR WI 0.5NM RADIUS OF 534732N 0000459W (GARTON).")
    assert (n.item_f, n.item_g) == ("SFC", "800FT AMSL")
    assert (n.section, n.sections) == ("WAR", ["WAR"])


def test_j3857_is_the_tda_activation_with_its_designator_in_item_e_verbatim():
    n = by_id(TRIMMED.read_bytes())["J3857/26"]
    assert (n.q_code, n.radius_nm, n.centre_lat) == ("QRDCA", 10.0, pytest.approx(53 + 56 / 60))
    assert n.centre_lon == pytest.approx(1 / 60)
    assert n.item_d is None
    assert "TEMPO DANGER AREA (TDA) EGD496A ACTIVATED. AIP SUP 108/2026\nREFERS." in n.item_e
    assert n.item_e.endswith("AR-2026-7404/05.")


def test_a_repeated_item_a_is_joined():
    n = by_id(TRIMMED.read_bytes())["Q172/26"]
    assert n.item_a == "EGTT EGPX"
    assert n.radius_nm == 999.0


def test_the_same_id_in_two_sections_is_one_notam_first_seen_with_every_section():
    issued, notams = read_notams(TRIMMED.read_bytes())
    assert issued.year == 2026
    assert len(notams) == 9  # 14 Notam elements in the file, 9 distinct ids
    a3857 = [n for n in notams if n.id == "A3857/26"]
    assert len(a3857) == 1
    assert (a3857[0].section, a3857[0].sections) == ("AD", ["AD", "WAR"])
    assert [n.id for n in notams].count("J3377/26") == 1


def test_dedupe_keeps_the_first_seen_notam_and_does_not_repeat_a_section():
    first = notam_xml(section="AD", e="FIRST")
    again_ad = notam_xml(section="AD", e="SECOND")
    in_war = notam_xml(section="WAR", e="THIRD")
    _issued, notams = read_notams(pib_xml(first, again_ad, in_war))
    assert len(notams) == 1
    assert (notams[0].item_e, notams[0].sections) == ("FIRST", ["AD", "WAR"])


def test_the_trimmed_fixture_links_a_replacement_and_drops_what_it_replaces_only_when_present():
    # J3377/26 replaces J2265/26, which is not in the PIB: that is not an error.
    notams = by_id(TRIMMED.read_bytes())
    assert notams["J3377/26"].type is NotamType.REPLACE
    assert notams["J3377/26"].replaces == "J2265/26"
    assert "J2265/26" not in notams
    assert notams["V34/26"].replaces == "V33/26"


def test_permanent_and_estimated_ends_in_the_real_fixture():
    notams = by_id(TRIMMED.read_bytes())
    assert (notams["V34/26"].permanent, notams["V34/26"].valid_to) == (True, None)
    assert notams["C7072/26"].est is True
    assert notams["C7072/26"].valid_to == datetime(2026, 11, 13, 22, 0, tzinfo=UTC)


# --- types and linkage (E4) ------------------------------------------------------------------------


def test_a_notamr_drops_the_notam_it_replaces_and_stays_itself():
    old = notam_xml(number=100)
    replacement = notam_xml(number=101, kind="R", referred="H100/26")
    other = notam_xml(number=102)
    _issued, kept = parse_pib(pib_xml(old, replacement, other))
    assert [n.id for n in kept] == ["H101/26", "H102/26"]
    assert kept[0].type is NotamType.REPLACE and kept[0].replaces == "H100/26"


def test_a_notamc_drops_the_notam_it_cancels_and_stays_in_the_list_as_type_c():
    old = notam_xml(number=100)
    cancel = notam_xml(number=103, kind="C", referred="H100/26")
    _issued, kept = parse_pib(pib_xml(old, cancel))
    assert [(n.id, n.type) for n in kept] == [("H103/26", NotamType.CANCEL)]
    assert kept[0].replaces == "H100/26"


def test_a_replaced_or_cancelled_id_missing_from_the_pib_is_not_an_error():
    xml = pib_xml(notam_xml(number=101, kind="R", referred="H5/26"), notam_xml(number=102, kind="C", referred="J6/25"))
    _issued, kept = parse_pib(xml)
    assert [n.id for n in kept] == ["H101/26", "H102/26"]


def test_drop_superseded_says_why_and_ignores_an_r_that_refers_to_itself():
    xml = pib_xml(
        notam_xml(number=100),
        notam_xml(number=101, kind="R", referred="H100/26"),
        notam_xml(number=102),
        notam_xml(number=103, kind="C", referred="H102/26"),
        notam_xml(number=104, kind="R", referred="H104/26"),
    )
    _issued, notams = read_notams(xml)
    kept, dropped = drop_superseded(notams)
    assert [n.id for n in kept] == ["H101/26", "H103/26", "H104/26"]
    assert [(d.notam_id, d.reason) for d in dropped] == [("H100/26", "replaced by H101/26"), ("H102/26", "cancelled by H103/26")]


def test_an_r_without_a_referred_id_replaces_nothing():
    _issued, kept = parse_pib(pib_xml(notam_xml(number=100), notam_xml(number=101, kind="R")))
    assert [(n.id, n.replaces) for n in kept] == [("H100/26", None), ("H101/26", None)]


def test_a_notam_in_the_obsolete_list_is_never_read():
    xml = pib_xml(notam_xml(number=1), obsolete=(notam_xml(number=2),))
    _issued, kept = parse_pib(xml)
    assert [n.id for n in kept] == ["H1/26"]


# --- B, C, EST, PERM ---------------------------------------------------------------------------------


def test_perm_is_open_ended():
    n = by_id(pib_xml(notam_xml(end="PERM")))["H7171/26"]
    assert (n.valid_to, n.permanent, n.est) == (None, True, False)


@pytest.mark.parametrize(
    ("end", "estimation", "expected_end"),
    [
        ("2611132200", True, datetime(2026, 11, 13, 22, 0, tzinfo=UTC)),  # the real shape: an Estimation element
        ("2611132200EST", False, datetime(2026, 11, 13, 22, 0, tzinfo=UTC)),  # an EST suffix on the time
        ("2611132200 EST", False, datetime(2026, 11, 13, 22, 0, tzinfo=UTC)),
        ("2611132200", False, datetime(2026, 11, 13, 22, 0, tzinfo=UTC)),
    ],
)
def test_est_is_kept_as_a_flag(end, estimation, expected_end):
    n = by_id(pib_xml(notam_xml(end=end, estimation=estimation)))["H7171/26"]
    assert n.valid_to == expected_end
    assert n.est is (estimation or "EST" in end)


def test_perm_with_est_keeps_both():
    n = by_id(pib_xml(notam_xml(end="PERM", estimation=True)))["H7171/26"]
    assert (n.permanent, n.est, n.valid_to) == (True, True, None)


def test_a_missing_end_is_open_ended_and_not_permanent():
    n = by_id(pib_xml(notam_xml(omit=("EndValidity",))))["H7171/26"]
    assert (n.valid_to, n.permanent) == (None, False)


# --- optional and required elements ---------------------------------------------------------------


def test_missing_optional_elements_give_none():
    n = by_id(pib_xml(notam_xml(d=None, f=None, g=None)))["H7171/26"]
    assert (n.item_d, n.item_f, n.item_g) == (None, None, None)


def test_blank_optional_elements_give_none():
    n = by_id(pib_xml(notam_xml(d="  \n ", f="", g=" ")))["H7171/26"]
    assert (n.item_d, n.item_f, n.item_g) == (None, None, None)


def test_a_missing_item_e_or_item_a_is_empty_not_a_failure():
    n = by_id(pib_xml(notam_xml(e=None, item_a=())))["H7171/26"]
    assert (n.item_e, n.item_a) == ("", "")


@pytest.mark.parametrize(
    "omitted",
    ["Series", "Number", "Year", "Type", "QLine", "FIR", "Code23", "Code45", "Traffic", "Purpose", "Scope", "Lower", "Upper", "Coordinates", "Radius", "StartValidity"],
)
def test_a_missing_required_element_raises_pib_error_with_a_one_line_reason(omitted):
    with pytest.raises(PibError) as caught:
        parse_pib(pib_xml(notam_xml(omit=(omitted,))))
    message = str(caught.value)
    assert "\n" not in message
    assert f"missing {omitted}" in message
    assert message.startswith(("NOTAM #1", "H7171/26"))


@pytest.mark.parametrize(
    ("overrides", "fragment"),
    [
        ({"kind": "X"}, "Type"),
        ({"start": "26102908"}, "StartValidity"),
        ({"start": "2613290800"}, "StartValidity"),
        ({"end": "soon"}, "EndValidity"),
        ({"coordinates": "5348N0005W"}, "Coordinates"),
        ({"radius": "big"}, "Radius"),
        ({"lower": "ground"}, "Lower"),
        ({"upper": ""}, "Upper"),
        ({"number": "seven"}, "Number"),
    ],
)
def test_an_unreadable_required_value_raises_pib_error_naming_it(overrides, fragment):
    with pytest.raises(PibError, match=fragment):
        parse_pib(pib_xml(notam_xml(**overrides)))


def test_an_unreadable_referred_number_raises_pib_error():
    bad = notam_xml(kind="R", referred="H100/26").replace("<ReferredNumber>100", "<ReferredNumber>one")
    with pytest.raises(PibError, match="ReferredNumber"):
        parse_pib(pib_xml(bad))


# --- the Q-circle ------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("coordinates", "radius", "expected"),
    [
        ("5356N00001E", "10", (53 + 56 / 60, 1 / 60, 10.0)),  # the PIB's separate centre and radius elements
        ("5356N00001E010", None, (53 + 56 / 60, 1 / 60, 10.0)),  # DDMMN DDDMME RRR
        ("5356N00001E 010", None, (53 + 56 / 60, 1 / 60, 10.0)),
        ("5356N 00001E010", None, (53 + 56 / 60, 1 / 60, 10.0)),
        ("5356N00001E010", "10", (53 + 56 / 60, 1 / 60, 10.0)),  # both, agreeing
        ("5356N00001E010", "12", (53 + 56 / 60, 1 / 60, 12.0)),  # both, disagreeing: the larger wins
        ("5130S00230W", "3", (-(51 + 30 / 60), -(2 + 30 / 60), 3.0)),
        ("5504N00500W", "999", (55 + 4 / 60, -5.0, 999.0)),
    ],
)
def test_the_q_circle_centre_and_radius(coordinates, radius, expected):
    n = by_id(pib_xml(notam_xml(coordinates=coordinates, radius=radius)))["H7171/26"]
    assert (n.centre_lat, n.centre_lon, n.radius_nm) == pytest.approx(expected)


def test_no_radius_anywhere_raises_pib_error():
    with pytest.raises(PibError, match="Radius"):
        parse_pib(pib_xml(notam_xml(coordinates="5356N00001E", radius=None)))


# --- item E and the ids ------------------------------------------------------------------------------


def test_unicode_and_stray_whitespace_in_item_e_are_kept_verbatim():
    text = "  CRANE \u2013 50M\u00a0AGL  \n\n   AT \u00c5BERDEEN \u2192 CTC 07784 546062  \n"
    n = by_id(pib_xml(notam_xml(e=text)))["H7171/26"]
    assert n.item_e == text


def test_item_e_is_kept_verbatim_when_the_xml_is_bytes_in_utf8():
    text = "TOWER \u00e9\u00e8  \n  2 SPACES"
    xml = pib_xml(notam_xml(e=text)).encode("utf-8")
    assert by_id(xml)["H7171/26"].item_e == text


def test_a_short_year_keeps_two_digits_in_the_id():
    assert list(by_id(pib_xml(notam_xml(year=7, number=12))).keys()) == ["H12/07"]


def test_the_series_letter_is_upper_case():
    assert list(by_id(pib_xml(notam_xml(series="q", number=172))).keys()) == ["Q172/26"]


def test_a_missing_nof_is_empty():
    n = by_id(pib_xml(notam_xml().replace("<NOF>EGGN</NOF>", "")))["H7171/26"]
    assert n.nof == ""


def test_a_notam_without_a_pib_section_attribute_gets_a_placeholder_section():
    n = by_id(pib_xml(notam_xml().replace(' PIBSection="WAR"', "")))["H7171/26"]
    assert n.sections == ["unknown"]


# --- the whole PIB ------------------------------------------------------------------------------------


def test_an_empty_pib_is_a_header_and_no_notams():
    issued, notams = parse_pib(pib_xml())
    assert (issued, notams) == (datetime(2026, 10, 31, 7, 0, 37, tzinfo=UTC), [])


def test_a_naive_issued_time_is_read_as_utc():
    issued, _notams = parse_pib(pib_xml(issued="2026-10-31T07:00:37"))
    assert issued == datetime(2026, 10, 31, 7, 0, 37, tzinfo=UTC)


@pytest.mark.parametrize(
    ("xml", "fragment"),
    [
        (pib_xml(notam_xml(), issued=None), "Issued"),
        ("<Pib><AreaPIBHeader><ShortSystemTitle>EAD</ShortSystemTitle></AreaPIBHeader></Pib>", "Issued"),
        (pib_xml(issued="yesterday"), "Issued"),
        ("<html><body>Service unavailable</body></html>", "Issued"),
        ("<Pib><AreaPIBHeader><Issued>2026-10-31T07:00:37+00:00", "not valid XML"),
        ("this is not xml", "not valid XML"),
        ("", "not valid XML"),
    ],
)
def test_a_pib_without_issued_or_with_malformed_xml_raises_pib_error(xml, fragment):
    with pytest.raises(PibError, match=fragment) as caught:
        parse_pib(xml)
    assert "\n" not in str(caught.value)


def test_malformed_bytes_raise_pib_error_not_a_crash():
    with pytest.raises(PibError, match="not valid XML"):
        parse_pib(b"\xff\xfe<Pib")
