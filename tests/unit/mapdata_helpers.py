"""Shared builders for the map-data tests: a fake openAIP and a fake NATS PIB that answer through httpx.MockTransport, and a counting client factory.

    Upstream(airspace=, pib=)    one object that answers both hosts, counts every request and can be told to fail
    Upstream.factory             the `client_factory` create_app takes: a new httpx.Client on the same fake each call
    notam(...)                   one Notam element in the real PIB's shape (pib_helpers.notam_xml) with the times given as datetimes

No socket opens (the suite runs with --disable-socket) and no key is read: the tests pass the fake key in `settings`.
"""

import json
from datetime import datetime
from pathlib import Path

import httpx
from pib_helpers import ISSUED, notam_xml, pib_xml

from flysafelz.clients.openaip import KEY_HEADER, URL
from flysafelz.clients.pib_source import LIVE_URL

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
FAKE_KEY = "test-key-not-real"
SETTINGS = {"OPENAIP_API_KEY": FAKE_KEY}


def page(name: str) -> dict:
    return json.loads((FIXTURES / "openaip" / f"{name}.json").read_text(encoding="utf-8"))


def pib_time(t: datetime) -> str:
    """The PIB's own YYMMDDHHMM time text."""
    return t.strftime("%y%m%d%H%M")


def notam(number: int, start: datetime, end: datetime | None, **fields: object) -> str:
    """One Notam element. `end` None is a permanent NOTAM (EndValidity PERM); `estimation=True` makes the end an estimate. Times are to the minute."""
    return notam_xml(number=number, start=pib_time(start), end="PERM" if end is None else pib_time(end), **fields)  # type: ignore[arg-type]


def pib_of(*notams: str) -> bytes:
    return pib_xml(*notams, issued=ISSUED).encode("utf-8")


class Upstream:
    """A fake openAIP (page1 and page2 by default) and a fake NATS PIB, one counter for each.

    `airspace` is {page number: body}, `pib` the XML bytes. `airspace_status` and `pib_status` make that host answer with an error status;
    `airspace_html` makes openAIP answer 200 with a web page instead of JSON. Change any of them between requests.
    """

    def __init__(self, airspace: dict[str, dict] | None = None, pib: bytes | None = None) -> None:
        self.airspace = airspace if airspace is not None else {"1": page("page1"), "2": page("page2")}
        self.pib = pib if pib is not None else pib_of()
        self.airspace_status = 200
        self.pib_status = 200
        self.airspace_html = False
        self.airspace_requests: list[httpx.Request] = []
        self.pib_requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        if str(request.url).startswith(URL):
            self.airspace_requests.append(request)
            if self.airspace_status != 200:
                return httpx.Response(self.airspace_status, text="<html>blocked</html>")
            if self.airspace_html:
                return httpx.Response(200, text="<html>Access denied</html>", headers={"content-type": "text/html"})
            return httpx.Response(200, json=self.airspace[request.url.params["page"]])
        if str(request.url) == LIVE_URL:
            self.pib_requests.append(request)
            if self.pib_status != 200:
                return httpx.Response(self.pib_status)
            return httpx.Response(200, content=self.pib)
        raise AssertionError(f"a request to an unexpected URL: {request.url}")

    def factory(self) -> httpx.Client:
        return httpx.Client(transport=httpx.MockTransport(self))

    @property
    def calls(self) -> int:
        return len(self.airspace_requests) + len(self.pib_requests)


__all__ = ["FAKE_KEY", "KEY_HEADER", "SETTINGS", "Upstream", "notam", "page", "pib_of", "pib_time"]
