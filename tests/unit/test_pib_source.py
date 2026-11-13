"""G11, E1, E2, E19: fetch_pib and load_pib against httpx.MockTransport. No socket opens; `now` is frozen."""

from datetime import UTC, datetime, timedelta

import httpx
import pytest
from pib_helpers import ISSUED, NOW, TRIMMED, notam_xml, pib_xml, with_rules

from flysafelz.clients import pib_source
from flysafelz.clients.pib_source import ARCHIVE_COMMITS_URL, LIVE_URL, fetch_pib, fetch_pib_at
from flysafelz.contracts import SourceName, SourceState
from flysafelz.pib import load_pib

NEWEST = "bdab14658f0270dc41460eebcd03332abfb230ec"
MIDDLE = "4375391a6241653e9893cd6db63aeb6a1d80cd06"
OLDEST = "7d0f59a7350e85547a4a9f221a024203e41d2824"
# The real archive's commits for data/PIB.xml around the 31 Oct case, newest first (committer time, UTC).
COMMITS = [
    (NEWEST, "2026-10-31T07:56:28Z"),
    (MIDDLE, "2026-10-31T01:23:47Z"),
    (OLDEST, "2026-10-30T21:20:57Z"),
]
XML = TRIMMED.read_bytes()


def raw_url(sha: str) -> str:
    return f"https://raw.githubusercontent.com/Jonty/uk-notam-archive/{sha}/data/PIB.xml"


def commits_json(commits=COMMITS) -> list[dict]:
    return [{"sha": sha, "commit": {"committer": {"date": date}}} for sha, date in commits]


class Server:
    """A fake NATS + GitHub. Records every request. `commits` honours the `until` parameter like the real API."""

    def __init__(self, *, live=None, commits=COMMITS, files=None, commits_response=None, raise_on=None):
        self.live = live
        self.commits = commits
        self.files = files if files is not None else {sha: XML for sha, _ in commits}
        self.commits_response = commits_response
        self.raise_on = raise_on or {}
        self.seen: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.seen.append(request)
        url = str(request.url).split("?")[0]
        if url in self.raise_on:
            raise self.raise_on[url]
        if url == LIVE_URL:
            return self.live if self.live is not None else httpx.Response(200, content=XML)
        if url == ARCHIVE_COMMITS_URL:
            if self.commits_response is not None:
                return self.commits_response
            until = request.url.params["until"]
            return httpx.Response(200, json=commits_json([c for c in self.commits if c[1] <= until]))
        for sha, body in self.files.items():
            if url == raw_url(sha):
                return httpx.Response(200, content=body)
        return httpx.Response(404, text="not found")

    def client(self) -> httpx.Client:
        return httpx.Client(transport=httpx.MockTransport(self))


# --- live ------------------------------------------------------------------------------------------------


def test_live_gets_the_nats_url_and_returns_the_bytes_ok():
    server = Server()
    got = fetch_pib("live", NOW, client=server.client())
    assert [str(r.url) for r in server.seen] == [LIVE_URL]
    assert LIVE_URL == "https://pibs.nats.co.uk/operational/pibs/PIB.xml"
    assert got.xml == XML
    assert got.status.sha is None
    assert (got.status.source, got.status.state, got.status.reason) == (SourceName.NOTAMS, SourceState.OK, "")
    assert got.status.fetched_at == NOW
    assert got.status.issued_at is None  # the issued time is read from the XML by load_pib


def test_no_request_carries_a_key_or_token():
    server = Server()
    fetch_pib("live", NOW, client=server.client())
    fetch_pib("archive", NOW, client=server.client())
    for request in server.seen:
        assert not {"authorization", "x-api-key", "x-openaip-api-key"} & set(request.headers)


# --- archive ---------------------------------------------------------------------------------------------


def test_archive_asks_the_commits_api_for_the_pib_path_until_now_and_fetches_the_file_at_that_sha():
    server = Server()
    got = fetch_pib("archive", datetime(2026, 10, 31, 8, 0, tzinfo=UTC), client=server.client())
    commits_request, file_request = server.seen
    assert str(commits_request.url).startswith("https://api.github.com/repos/Jonty/uk-notam-archive/commits?")
    assert dict(commits_request.url.params) == {"path": "data/PIB.xml", "until": "2026-10-31T08:00:00Z", "per_page": "5"}
    assert commits_request.headers["accept"] == "application/vnd.github+json"
    assert str(file_request.url) == raw_url(NEWEST)
    assert (got.status.sha, got.xml, got.status.state) == (NEWEST, XML, SourceState.OK)
    assert got.status.fetched_at == datetime(2026, 10, 31, 8, 0, tzinfo=UTC)


def test_archive_picks_the_newest_commit_at_or_before_now_and_records_its_sha():
    # At 07:30Z the 07:56Z commit does not exist yet: the newest at or before now is the 01:23Z one.
    server = Server()
    got = fetch_pib("archive", NOW, client=server.client())
    assert got.status.sha == MIDDLE
    assert str(server.seen[-1].url) == raw_url(MIDDLE)


def test_a_commit_exactly_at_now_counts():
    got = fetch_pib("archive", datetime(2026, 10, 31, 7, 56, 28, tzinfo=UTC), client=Server().client())
    assert got.status.sha == NEWEST


def test_a_commit_after_now_is_never_used_even_if_the_api_returns_it():
    listing = httpx.Response(200, json=commits_json(COMMITS))  # the API ignores `until`
    server = Server(commits_response=listing)
    got = fetch_pib("archive", NOW, client=server.client())
    assert got.status.sha == MIDDLE


def test_a_commit_without_a_readable_time_is_skipped_not_trusted():
    listing = httpx.Response(200, json=[{"sha": NEWEST}])
    got = fetch_pib("archive", NOW, client=Server(commits_response=listing, files={NEWEST: XML}).client())
    assert got.xml is None and got.status.state is SourceState.FAILED
    assert "no archive commit at or before" in got.status.reason


@pytest.mark.parametrize("date", [None, "", "not a time", 12])
def test_an_unreadable_commit_time_is_skipped_and_the_next_readable_commit_is_used(date):
    listing = httpx.Response(200, json=[{"sha": NEWEST, "commit": {"committer": {"date": date}}}, *commits_json([(MIDDLE, "2026-10-31T01:23:00Z")])])
    server = Server(commits_response=listing, files={MIDDLE: XML})
    got = fetch_pib("archive", NOW, client=server.client())
    assert got.status.sha == MIDDLE
    assert str(server.seen[-1].url) == raw_url(MIDDLE)


def test_a_commit_time_without_a_zone_is_read_as_utc():
    listing = httpx.Response(200, json=commits_json([(NEWEST, "2026-10-31T07:56:28")]))
    server = Server(commits_response=listing, files={NEWEST: XML})
    assert fetch_pib("archive", NOW, client=server.client()).status.state is SourceState.FAILED  # 07:56 is after 07:30
    assert fetch_pib("archive", datetime(2026, 10, 31, 8, 0, tzinfo=UTC), client=server.client()).status.sha == NEWEST


def test_e19_an_empty_commit_list_fails_with_no_xml():
    server = Server(commits=[])
    got = fetch_pib("archive", NOW, client=server.client())
    assert got.xml is None and got.status.sha is None
    assert got.status.state is SourceState.FAILED
    assert got.status.reason == "no archive commit at or before 2026-10-31T07:30:00Z"
    assert len(server.seen) == 1  # no file was asked for


def test_e19_now_before_the_archive_starts_fails_with_no_xml():
    early = datetime(2026, 10, 1, 0, 0, tzinfo=UTC)
    got = fetch_pib("archive", early, client=Server().client())
    assert got.xml is None
    assert got.status.state is SourceState.FAILED
    assert got.status.reason == "no archive commit at or before 2026-10-01T00:00:00Z"


@pytest.mark.parametrize("body", [{"message": "Not Found"}, "nonsense", [], [{}], [{"sha": ""}], [None], [{"sha": 7}]])
def test_e19_a_commits_response_without_a_usable_sha_is_no_commit(body):
    got = fetch_pib("archive", NOW, client=Server(commits_response=httpx.Response(200, json=body)).client())
    assert (got.xml, got.status.state) == (None, SourceState.FAILED)
    assert got.status.reason.startswith("no archive commit at or before")


def test_a_commits_response_that_is_not_json_fails():
    got = fetch_pib("archive", NOW, client=Server(commits_response=httpx.Response(200, text="<html>")).client())
    assert got.status.state is SourceState.FAILED
    assert got.status.reason == "archive commits: response is not JSON"


# --- E1: failures never raise -----------------------------------------------------------------------------


@pytest.mark.parametrize("code", [404, 429, 500, 502, 503])
def test_e1_live_http_errors_fail_with_the_status_code(code):
    got = fetch_pib("live", NOW, client=Server(live=httpx.Response(code, text="no")).client())
    assert got.xml is None and got.status.sha is None
    assert got.status.state is SourceState.FAILED
    assert str(code) in got.status.reason
    assert got.status.fetched_at == NOW


@pytest.mark.parametrize(
    "error",
    [httpx.ReadTimeout("slow"), httpx.ConnectTimeout("slow"), httpx.ConnectError("refused"), httpx.RemoteProtocolError("bad"), httpx.TooManyRedirects("loop")],
    ids=lambda e: type(e).__name__,
)
def test_e1_live_timeouts_and_transport_errors_fail_with_the_exception_class(error):
    got = fetch_pib("live", NOW, client=Server(raise_on={LIVE_URL: error}).client())
    assert got.status.state is SourceState.FAILED
    assert type(error).__name__ in got.status.reason
    assert got.xml is None


@pytest.mark.parametrize("code", [403, 404, 500, 503])
def test_e1_archive_commit_listing_http_errors_fail_with_the_status_code(code):
    """403 is GitHub's rate limit."""
    got = fetch_pib("archive", NOW, client=Server(commits_response=httpx.Response(code, json={"message": "API rate limit exceeded"})).client())
    assert got.xml is None and got.status.sha is None
    assert got.status.state is SourceState.FAILED
    assert str(code) in got.status.reason
    assert "archive commits" in got.status.reason


@pytest.mark.parametrize("error", [httpx.ReadTimeout("slow"), httpx.ConnectError("refused")], ids=lambda e: type(e).__name__)
def test_e1_archive_commit_listing_exceptions_fail_with_the_class(error):
    got = fetch_pib("archive", NOW, client=Server(raise_on={ARCHIVE_COMMITS_URL: error}).client())
    assert got.status.state is SourceState.FAILED
    assert type(error).__name__ in got.status.reason


@pytest.mark.parametrize("code", [403, 404, 500])
def test_e1_archive_file_http_errors_fail_but_keep_the_sha_that_was_tried(code):
    original = Server()

    def handler(request: httpx.Request) -> httpx.Response:
        if str(request.url) == raw_url(MIDDLE):
            return httpx.Response(code, text="no")
        return original(request)

    got = fetch_pib("archive", NOW, client=httpx.Client(transport=httpx.MockTransport(handler)))
    assert got.xml is None
    assert got.status.sha == MIDDLE
    assert got.status.state is SourceState.FAILED
    assert str(code) in got.status.reason and MIDDLE in got.status.reason


def test_e1_archive_file_timeout_fails_with_the_class():
    server = Server(raise_on={raw_url(MIDDLE): httpx.ReadTimeout("slow")})
    got = fetch_pib("archive", NOW, client=server.client())
    assert (got.xml, got.status.sha, got.status.state) == (None, MIDDLE, SourceState.FAILED)
    assert "ReadTimeout" in got.status.reason


def test_an_unknown_source_is_a_programming_error():
    with pytest.raises(ValueError, match="'live' or 'archive'"):
        fetch_pib("ftp", NOW, client=Server().client())  # type: ignore[arg-type]


def test_without_a_client_one_is_made_with_a_timeout_and_closed(monkeypatch):
    made = []
    real_client = httpx.Client

    def make(**kwargs):
        client = real_client(transport=httpx.MockTransport(Server()), **kwargs)
        made.append((kwargs, client))
        return client

    monkeypatch.setattr(pib_source.httpx, "Client", make)
    got = fetch_pib("live", NOW)
    assert got.status.state is SourceState.OK
    kwargs, client = made[0]
    assert kwargs["timeout"] == pib_source.TIMEOUT_SECONDS
    assert kwargs["follow_redirects"] is True
    assert client.is_closed


# --- load_pib: the status with the issued time, E2 stale, and unreadable XML ---------------------------------------


def pib_issued(delta: timedelta) -> bytes:
    issued = (NOW - delta).isoformat()
    return pib_xml(notam_xml(), notam_xml(number=2), issued=issued).encode("utf-8")


def test_load_live_ok_has_the_issued_time_the_notams_and_no_sha():
    got = load_pib("live", NOW, client=Server(live=httpx.Response(200, content=XML)).client())
    assert got.status.state is SourceState.OK
    assert got.status.reason == ""
    assert got.status.issued_at == datetime(2026, 10, 31, 7, 0, 37, tzinfo=UTC)
    assert got.status.fetched_at == NOW
    assert got.status.sha is None
    assert len(got.notams) == 9  # 14 elements, 9 distinct ids; what J3377 and V34 replace is not in the file


def test_load_archive_records_the_sha_next_to_the_status():
    got = load_pib("archive", datetime(2026, 10, 31, 8, 0, tzinfo=UTC), client=Server().client())
    assert (got.status.sha, got.status.state) == (NEWEST, SourceState.OK)


def test_e2_a_pib_issued_just_over_three_hours_before_now_is_stale_and_still_returns_its_notams():
    body = pib_issued(timedelta(hours=3, seconds=1))
    got = load_pib("live", NOW, client=Server(live=httpx.Response(200, content=body)).client())
    assert got.status.state is SourceState.STALE
    assert got.status.reason == "PIB issued 2026-10-31T04:29:59Z, more than 3 h before now"
    assert got.status.issued_at == NOW - timedelta(hours=3, seconds=1)
    assert [n.id for n in got.notams] == ["H7171/26", "H2/26"]


def test_e2_exactly_three_hours_is_ok():
    body = pib_issued(timedelta(hours=3))
    got = load_pib("live", NOW, client=Server(live=httpx.Response(200, content=body)).client())
    assert (got.status.state, got.status.reason) == (SourceState.OK, "")


def test_e2_a_pib_issued_just_under_three_hours_ago_is_ok():
    body = pib_issued(timedelta(hours=3) - timedelta(seconds=1))
    assert load_pib("live", NOW, client=Server(live=httpx.Response(200, content=body)).client()).status.state is SourceState.OK


def test_e2_the_real_archive_pib_at_07_30_is_stale_because_the_01_23_commit_is_old():
    # The commit at or before 07:30Z is the 01:23Z one; its PIB was issued hours earlier than the case PIB.
    old_pib = pib_xml(notam_xml(), issued="2026-10-31T01:00:00+00:00").encode()
    server = Server(files={MIDDLE: old_pib, NEWEST: XML, OLDEST: old_pib})
    got = load_pib("archive", NOW, client=server.client())
    assert (got.status.sha, got.status.state) == (MIDDLE, SourceState.STALE)
    assert "PIB issued 2026-10-31T01:00:00Z" in got.status.reason


def test_the_stale_limit_comes_from_the_rules():
    body = pib_issued(timedelta(hours=2))
    server = Server(live=httpx.Response(200, content=body))
    assert load_pib("live", NOW, client=server.client()).status.state is SourceState.OK
    got = load_pib("live", NOW, client=server.client(), rules=with_rules(pib_stale_hours=1))
    assert got.status.state is SourceState.STALE
    assert got.status.reason.endswith("more than 1 h before now")


def test_replaced_and_cancelled_notams_are_dropped_and_listed():
    body = pib_xml(notam_xml(number=1), notam_xml(number=2, kind="R", referred="H1/26"), notam_xml(number=3)).encode()
    got = load_pib("live", NOW, client=Server(live=httpx.Response(200, content=body)).client())
    assert [n.id for n in got.notams] == ["H2/26", "H3/26"]
    assert [(s.notam_id, s.reason) for s in got.superseded] == [("H1/26", "replaced by H2/26")]


def test_an_empty_pib_is_ok_with_no_notams():
    got = load_pib("live", NOW, client=Server(live=httpx.Response(200, content=pib_xml().encode())).client())
    assert (got.status.state, got.notams) == (SourceState.OK, [])


@pytest.mark.parametrize(
    ("body", "fragment"),
    [
        (pib_xml(notam_xml(), issued=None).encode(), "Issued"),
        (b"<Pib><AreaPIBHeader>", "not valid XML"),
        (b"", "not valid XML"),
        (pib_xml(notam_xml(omit=("QLine",))).encode(), "missing QLine"),
    ],
)
def test_unreadable_xml_fails_the_source_with_a_one_line_reason_and_no_notams(body, fragment):
    got = load_pib("live", NOW, client=Server(live=httpx.Response(200, content=body)).client())
    assert got.status.state is SourceState.FAILED
    assert fragment in got.status.reason and "\n" not in got.status.reason
    assert got.notams == []
    assert got.status.fetched_at == NOW


def test_a_failed_archive_fetch_passes_through_with_its_sha():
    server = Server(raise_on={raw_url(MIDDLE): httpx.ReadTimeout("slow")})
    got = load_pib("archive", NOW, client=server.client())
    assert (got.status.state, got.status.sha, got.notams) == (SourceState.FAILED, MIDDLE, [])


def test_the_issued_header_constant_matches_the_builder():
    assert datetime.fromisoformat(ISSUED) == datetime(2026, 10, 31, 7, 0, 37, tzinfo=UTC)


# --- status_from_xml: the half of load_pib that needs no request ---------------------------------------------------


def test_status_from_xml_gives_the_same_status_as_load_pib_for_the_same_bytes():
    from flysafelz.pib import status_from_xml

    fetched = load_pib("archive", datetime(2026, 10, 31, 8, 0, tzinfo=UTC), client=Server().client())
    direct = status_from_xml(XML, datetime(2026, 10, 31, 8, 0, tzinfo=UTC), fetched_at=fetched.status.fetched_at, sha=NEWEST)
    assert direct.status == fetched.status
    assert [n.id for n in direct.notams] == [n.id for n in fetched.notams]


def test_status_from_xml_applies_the_stale_check_and_the_case_mode_switch():
    from flysafelz.pib import CASE_MODE_NOTE, status_from_xml

    issued = datetime.fromisoformat(ISSUED)
    late = issued + timedelta(hours=3, minutes=1)
    stale = status_from_xml(XML, late)
    assert stale.status.state is SourceState.STALE and stale.status.issued_at == issued and stale.status.fetched_at == late
    assert stale.notams  # a stale PIB still returns its NOTAMs
    case = status_from_xml(XML, late, stale_rule=False)
    assert case.status.state is SourceState.OK and case.status.reason == CASE_MODE_NOTE


def test_status_from_xml_fails_on_unreadable_bytes_and_keeps_the_sha():
    from flysafelz.pib import status_from_xml

    got = status_from_xml(b"<not a pib", NOW, sha=NEWEST)
    assert got.status.state is SourceState.FAILED and got.status.sha == NEWEST and got.notams == []


# --- fetch_pib_at: a pinned archive commit ----------------------------------------------------------------------------


def test_fetch_pib_at_gets_the_raw_file_at_the_sha_and_asks_no_commits_api():
    server = Server()
    got = fetch_pib_at(OLDEST, NOW, client=server.client())
    assert (got.xml, got.status.sha, got.status.state, got.status.fetched_at) == (XML, OLDEST, SourceState.OK, NOW)
    assert [str(r.url) for r in server.seen] == [raw_url(OLDEST)]


def test_fetch_pib_at_words_a_failure_as_fetch_pib_does_and_keeps_the_sha():
    got = fetch_pib_at(NEWEST, NOW, client=Server(files={}).client())
    assert got.xml is None and got.status.state is SourceState.FAILED and got.status.sha == NEWEST
    assert got.status.reason == f"archive file at {NEWEST}: HTTP 404"
    timed_out = fetch_pib_at(NEWEST, NOW, client=Server(raise_on={raw_url(NEWEST): httpx.ReadTimeout("slow")}).client())
    assert timed_out.status.reason == f"archive file at {NEWEST}: ReadTimeout"
    # the same file through the commits lookup is worded the same way
    via_lookup = fetch_pib("archive", datetime(2026, 10, 31, 8, 0, tzinfo=UTC), client=Server(files={}).client())
    assert via_lookup.status.reason == f"archive file at {NEWEST}: HTTP 404"
