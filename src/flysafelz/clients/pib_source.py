"""Get the NATS PIB (the UK NOTAM briefing XML), live or from Jonty's archive. No key, no .env.

    fetch_pib("live", now)      GET the NATS PIB
    fetch_pib("archive", now)   the PIB in the last commit to data/PIB.xml at or before `now`

The archive is github.com/Jonty/uk-notam-archive. Two requests: the commits
API (path=data/PIB.xml, until=now, newest first) for the SHA, then the raw file
at that SHA. The SHA is recorded on the status so a replay can be reproduced.
"At or before now" is the commit's time, not the PIB's Issued time: a commit
made at 07:56Z can hold a PIB issued at 07:00Z.

Nothing is raised to the caller. An HTTP error (4xx, 5xx, a GitHub rate-limit
403), a timeout or any other httpx error gives a result whose status is
`failed`, with the status code or the exception class in the reason. An
archive with no commit at or before `now` is `failed` too, and has no XML.

This module only fetches. It does not read the XML: pib.py parses it and
adds the issued time and the stale check to the status.
"""

from datetime import UTC, datetime
from typing import Literal

import httpx
from pydantic import BaseModel, ConfigDict

from flysafelz.contracts import SourceName, SourceState, SourceStatus

LIVE_URL = "https://pibs.nats.co.uk/operational/pibs/PIB.xml"
ARCHIVE_COMMITS_URL = "https://api.github.com/repos/Jonty/uk-notam-archive/commits"
ARCHIVE_RAW_URL = "https://raw.githubusercontent.com/Jonty/uk-notam-archive/{sha}/data/PIB.xml"
ARCHIVE_PATH = "data/PIB.xml"

TIMEOUT_SECONDS = 60  # the PIB is a few MB
COMMITS_PER_PAGE = 5  # newest first; the first one at or before `now` wins

Source = Literal["live", "archive"]


class PibFetch(BaseModel):
    """What one fetch returned: the raw XML (None on failure) and the status (which carries the archive SHA)."""

    model_config = ConfigDict(extra="forbid")

    xml: bytes | None = None
    status: SourceStatus


def utc_text(t: datetime) -> str:
    """A UTC time as 2026-10-31T07:30:00Z, for reasons and request parameters."""
    return t.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _failed(reason: str, now: datetime, sha: str | None = None) -> PibFetch:
    status = SourceStatus(source=SourceName.NOTAMS, state=SourceState.FAILED, reason=reason, fetched_at=now, sha=sha)
    return PibFetch(status=status)


def _ok(xml: bytes, now: datetime, sha: str | None) -> PibFetch:
    status = SourceStatus(source=SourceName.NOTAMS, state=SourceState.OK, fetched_at=now, sha=sha)
    return PibFetch(xml=xml, status=status)


def _get(
    client: httpx.Client, url: str, what: str, params: dict[str, str | int] | None = None, headers: dict[str, str] | None = None
) -> httpx.Response | str:
    """The response, or a one-line failure reason (status code or exception class)."""
    try:
        response = client.get(url, params=params, headers=headers)
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        return f"{what}: HTTP {exc.response.status_code}"
    except httpx.HTTPError as exc:
        return f"{what}: {type(exc).__name__}"
    return response


def _commit_time(commit: object) -> datetime | None:
    """The committer time of a commits-API entry, or None when it is not there or unreadable."""
    try:
        when = datetime.fromisoformat(commit["commit"]["committer"]["date"])  # type: ignore[index]
    except (KeyError, TypeError, ValueError):
        return None
    return when if when.tzinfo else when.replace(tzinfo=UTC)


def _last_commit_sha(commits: object, now: datetime) -> str | None:
    """The SHA of the newest commit at or before `now` in a newest-first commits list, else None.

    The API already filters with `until`; the time is checked here too, so a
    commit after `now` is never used.
    """
    if not isinstance(commits, list):
        return None
    for commit in commits:
        sha = commit.get("sha") if isinstance(commit, dict) else None
        when = _commit_time(commit)
        if isinstance(sha, str) and sha and (when is None or when <= now):
            return sha
    return None


def _fetch_live(now: datetime, client: httpx.Client) -> PibFetch:
    got = _get(client, LIVE_URL, "NATS PIB")
    if isinstance(got, str):
        return _failed(got, now)
    return _ok(got.content, now, None)


def _fetch_archive(now: datetime, client: httpx.Client) -> PibFetch:
    params = {"path": ARCHIVE_PATH, "until": utc_text(now), "per_page": COMMITS_PER_PAGE}
    headers = {"Accept": "application/vnd.github+json"}
    listing = _get(client, ARCHIVE_COMMITS_URL, "archive commits", params=params, headers=headers)
    if isinstance(listing, str):
        return _failed(listing, now)
    try:
        commits = listing.json()
    except ValueError:
        return _failed("archive commits: response is not JSON", now)
    sha = _last_commit_sha(commits, now)
    if sha is None:
        return _failed(f"no archive commit at or before {utc_text(now)}", now)
    raw = _get(client, ARCHIVE_RAW_URL.format(sha=sha), f"archive file at {sha}")
    if isinstance(raw, str):
        return _failed(raw, now, sha)
    return _ok(raw.content, now, sha)


def fetch_pib(source: Source, now: datetime, *, client: httpx.Client | None = None) -> PibFetch:
    """Fetch the PIB from `source` ("live" or "archive"). `now` is the fetched-at time and the archive cut-off.

    Pass a client to control the transport (tests use httpx.MockTransport).
    Without one, a client with a timeout is made and closed here.
    """
    if source not in ("live", "archive"):
        raise ValueError(f"source must be 'live' or 'archive', got {source!r}")
    fetch = _fetch_live if source == "live" else _fetch_archive
    if client is not None:
        return fetch(now, client)
    with httpx.Client(timeout=TIMEOUT_SECONDS, follow_redirects=True) as own_client:
        return fetch(now, own_client)
