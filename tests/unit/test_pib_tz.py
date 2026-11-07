"""tz on the PIB module: the load_pib stale reason, and the prefilter / sort_notam reasons that quote B, C and the window.

No socket (MockTransport), frozen `now`. A BST date (24 Oct 2026) shows the shift; the request parameter stays UTC.
"""

from datetime import UTC, datetime, timedelta

import httpx
import pytest
from pib_helpers import LZ, notam_xml, parsed, pib_xml
from test_pib_source import Server

from flysafelz.clients.pib_source import ARCHIVE_COMMITS_URL
from flysafelz.contracts import SourceState
from flysafelz.pib import SortAction, load_pib, prefilter, sort_notam

LONDON = "Europe/London"
BST_NOW = datetime(2026, 10, 24, 7, 30, tzinfo=UTC)  # London is on BST: 08:30 on the clock
WINDOW = (datetime(2026, 10, 24, 7, 0, tzinfo=UTC), datetime(2026, 10, 24, 18, 0, tzinfo=UTC))


def stale_body() -> bytes:
    issued = (BST_NOW - timedelta(hours=3, seconds=1)).isoformat()
    return pib_xml(notam_xml(), issued=issued).encode("utf-8")


def when(day: int, hour: int, month: int = 10) -> datetime:
    return datetime(2026, month, day, hour, tzinfo=UTC)


# --- load_pib ------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("tz", "stamp"),
    [(None, "2026-10-24T04:29:59Z"), ("UTC", "2026-10-24T04:29:59Z"), (LONDON, "2026-10-24 05:29 BST"), ("America/New_York", "2026-10-24 00:29 EDT")],
    ids=["None", "UTC", "London", "New York"],
)
def test_the_stale_reason_quotes_the_issued_time_in_the_zone(tz, stamp):
    got = load_pib("live", BST_NOW, client=Server(live=httpx.Response(200, content=stale_body())).client(), tz=tz)
    assert got.status.state is SourceState.STALE
    assert got.status.reason == f"PIB issued {stamp}, more than 3 h before now"


def test_the_status_fields_stay_utc_whatever_the_zone():
    plain = load_pib("live", BST_NOW, client=Server(live=httpx.Response(200, content=stale_body())).client())
    zoned = load_pib("live", BST_NOW, client=Server(live=httpx.Response(200, content=stale_body())).client(), tz=LONDON)
    assert plain.status.issued_at == zoned.status.issued_at == BST_NOW - timedelta(hours=3, seconds=1)
    assert plain.status.issued_at.utcoffset() == timedelta(0) and zoned.status.issued_at.utcoffset() == timedelta(0)  # type: ignore[union-attr]
    assert plain.status.fetched_at == zoned.status.fetched_at == BST_NOW
    assert plain.status.model_copy(update={"reason": zoned.status.reason}) == zoned.status  # only the reason text differs
    assert plain.notams == zoned.notams


def test_the_archive_request_still_asks_for_a_utc_until_when_a_zone_is_set():
    server = Server(commits=[("a" * 40, "2026-10-24T05:00:00Z")], files={"a" * 40: pib_xml(notam_xml(), issued="2026-10-24T05:00:00+00:00").encode()})
    got = load_pib("archive", BST_NOW, client=server.client(), tz=LONDON)
    assert got.status.state is SourceState.OK
    [listing] = [r for r in server.seen if str(r.url).startswith(ARCHIVE_COMMITS_URL)]
    assert listing.url.params["until"] == "2026-10-24T07:30:00Z"  # not 08:30 BST


@pytest.mark.parametrize("tz", ["", "Mars/Olympus"])
def test_load_pib_rejects_an_unknown_zone_before_any_request(tz):
    server = Server()
    with pytest.raises(ValueError, match="unknown time zone"):
        load_pib("live", BST_NOW, client=server.client(), tz=tz)
    assert server.seen == []


# --- prefilter -----------------------------------------------------------------------------------------------

DROPS = [
    # (id, the NOTAM, the reason in UTC, the reason in Europe/London)
    (
        "starts after the window",
        parsed(valid_from=when(24, 20), valid_to=when(25, 20)),
        "starts 2026-10-24T20:00:00Z, at or after the window end 2026-10-24T18:00:00Z",
        "starts 2026-10-24 21:00 BST, at or after the window end 2026-10-24 19:00 BST",
    ),
    (
        "ended before the window",
        parsed(valid_from=when(20, 8), valid_to=when(24, 6)),
        "ended 2026-10-24T06:00:00Z, at or before the window start 2026-10-24T07:00:00Z",
        "ended 2026-10-24 07:00 BST, at or before the window start 2026-10-24 08:00 BST",
    ),
]


@pytest.mark.parametrize(("notam", "utc_reason", "london_reason"), [d[1:] for d in DROPS], ids=[d[0] for d in DROPS])
def test_prefilter_quotes_b_c_and_the_window_in_the_zone(notam, utc_reason, london_reason):
    assert [d.reason for d in prefilter([notam], LZ, WINDOW)[1]] == [utc_reason]
    for tz in ("UTC", None):
        assert [d.reason for d in prefilter([notam], LZ, WINDOW, tz=tz)[1]] == [utc_reason]
    kept, dropped = prefilter([notam], LZ, WINDOW, tz=LONDON)
    assert kept == [] and [d.reason for d in dropped] == [london_reason]


def test_prefilter_reasons_that_quote_no_time_are_the_same_in_any_zone():
    far = parsed(centre_lat=54.8)  # about 60 NM north: the distance rule drops it, with no time in the reason
    assert prefilter([far], LZ, WINDOW)[1] == prefilter([far], LZ, WINDOW, tz=LONDON)[1]
    assert "NM" in prefilter([far], LZ, WINDOW)[1][0].reason


def test_prefilter_keeps_the_same_notams_whatever_the_zone():
    notams = [parsed("H1/26"), parsed("H2/26", valid_from=when(24, 20)), parsed("H3/26", valid_to=when(24, 6))]
    assert prefilter(notams, LZ, WINDOW, tz=LONDON)[0] == prefilter(notams, LZ, WINDOW)[0]


# --- sort_notam ----------------------------------------------------------------------------------------------


def test_sort_notam_quotes_the_start_and_the_end_in_the_zone():
    later = parsed(valid_from=when(24, 20), valid_to=when(25, 20))
    earlier = parsed(valid_from=when(20, 8), valid_to=when(24, 6))
    assert sort_notam(later, WINDOW, LZ).reason == "not active in the span: starts 2026-10-24T20:00:00Z"
    assert sort_notam(later, WINDOW, LZ, tz=LONDON).reason == "not active in the span: starts 2026-10-24 21:00 BST"
    assert sort_notam(earlier, WINDOW, LZ).reason == "not active in the span: ended 2026-10-24T06:00:00Z"
    assert sort_notam(earlier, WINDOW, LZ, tz=LONDON).reason == "not active in the span: ended 2026-10-24 07:00 BST"
    assert sort_notam(later, WINDOW, LZ, tz=LONDON).action is SortAction.IGNORE


def test_sort_notam_never_reformats_item_d_or_item_e():
    saturday = (datetime(2026, 10, 31, 9, tzinfo=UTC), datetime(2026, 10, 31, 12, tzinfo=UTC))
    weekdays_only = parsed(item_d="MON-FRI 0800-1700", valid_from=when(29, 8), valid_to=when(10, 18, 11))
    plain = sort_notam(weekdays_only, saturday, LZ)
    zoned = sort_notam(weekdays_only, saturday, LZ, tz=LONDON)
    assert plain.action is zoned.action is SortAction.IGNORE
    assert "MON-FRI 0800-1700" in plain.reason and plain.reason == zoned.reason  # the published schedule, unchanged

    text = parsed(item_e="UAS OPR 0800-1800 UTC. CALL 07700 900123.")
    assert sort_notam(text, saturday, LZ).reason == sort_notam(text, saturday, LZ, tz=LONDON).reason  # no time of ours in it


def test_sort_notam_rejects_an_unknown_zone_even_for_a_notam_with_no_time_to_quote():
    with pytest.raises(ValueError, match="unknown time zone"):
        sort_notam(parsed(), WINDOW, LZ, tz="Mars/Olympus")
    with pytest.raises(ValueError, match="unknown time zone"):
        prefilter([], LZ, WINDOW, tz="")
