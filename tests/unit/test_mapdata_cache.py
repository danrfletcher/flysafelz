"""G22, E1, the cache rule: (cache: none / fresh / expired / corrupt ...) x (upstream: ok / fails) -> (status, stale, upstream calls, file rewritten).

The clock is a fixed instant passed in; nothing sleeps. The refresh is a real function that counts its calls. Both layers' TTLs (24 h and 15 min) are run.
"""

import asyncio
import json
import logging
from datetime import UTC, datetime, timedelta

import pytest

from flysafelz.mapdata.app import AIRSPACE_TTL, NOTAM_TTL
from flysafelz.mapdata.cache import Layer, RefreshFailed, write_payload

NOW = datetime(2026, 11, 8, 12, 0, tzinfo=UTC)
OLD = [{"type": "Feature", "geometry": None, "properties": {"id": "old"}}]
NEW = [{"type": "Feature", "geometry": None, "properties": {"id": "new"}}]
SECOND = timedelta(seconds=1)
LAYERS = [("airspace", AIRSPACE_TTL), ("notams", NOTAM_TTL)]


class Upstream:
    """A refresh function that answers NEW, or fails when told, and counts how often it was called."""

    def __init__(self, *, works: bool = True, reason: str = "upstream is down") -> None:
        self.works = works
        self.reason = reason
        self.calls = 0

    def __call__(self, now: datetime) -> list[dict]:
        self.calls += 1
        if not self.works:
            raise RefreshFailed(self.reason)
        return NEW


def put_cache(folder, name: str, updated_at: datetime) -> None:
    write_payload(folder / f"{name}.json", updated_at, OLD)


def cache_text(folder, name: str) -> str | None:
    path = folder / f"{name}.json"
    return path.read_text(encoding="utf-8") if path.exists() else None


def cache_json(folder, name: str) -> dict:
    return json.loads((folder / f"{name}.json").read_text(encoding="utf-8"))


def ids(body: dict) -> list[str]:
    return [feature["properties"]["id"] for feature in body["features"]]


# --- the table: cache state x upstream -> answer ------------------------------------------------------------------------

AGES = {
    "fresh": lambda ttl: ttl - SECOND,
    "just_inside": lambda ttl: timedelta(0),
    "expired": lambda ttl: ttl + SECOND,
    "exactly_at_ttl": lambda ttl: ttl,
    "long_expired": lambda ttl: ttl * 100,
}


@pytest.mark.parametrize("name,ttl", LAYERS)
@pytest.mark.parametrize("state", ["fresh", "just_inside"])
@pytest.mark.parametrize("works", [True, False])
def test_a_cache_inside_its_ttl_is_served_with_no_call_upstream(tmp_path, name, ttl, state, works):
    updated = NOW - AGES[state](ttl)
    put_cache(tmp_path, name, updated)
    before = cache_text(tmp_path, name)
    upstream = Upstream(works=works)
    served = Layer(name, ttl, upstream).resolve(tmp_path, NOW)
    assert served.status == 200 and served.body["stale"] is False
    assert ids(served.body) == ["old"] and served.body["updated_at"] == updated.strftime("%Y-%m-%dT%H:%M:%SZ")
    assert upstream.calls == 0 and cache_text(tmp_path, name) == before


@pytest.mark.parametrize("name,ttl", LAYERS)
@pytest.mark.parametrize("state", ["expired", "exactly_at_ttl", "long_expired"])
def test_an_expired_cache_makes_one_refresh_and_a_good_one_replaces_the_file(tmp_path, name, ttl, state):
    put_cache(tmp_path, name, NOW - AGES[state](ttl))
    upstream = Upstream()
    served = Layer(name, ttl, upstream).resolve(tmp_path, NOW)
    assert served.status == 200 and served.body["stale"] is False and ids(served.body) == ["new"]
    assert served.body["updated_at"] == "2026-11-08T12:00:00Z"
    assert upstream.calls == 1
    assert cache_json(tmp_path, name)["features"] == NEW  # the file was rewritten


@pytest.mark.parametrize("name,ttl", LAYERS)
@pytest.mark.parametrize("state", ["expired", "exactly_at_ttl", "long_expired"])
def test_an_expired_cache_with_a_failing_upstream_is_served_stale_and_left_untouched(tmp_path, name, ttl, state):
    updated = NOW - AGES[state](ttl)
    put_cache(tmp_path, name, updated)
    before = cache_text(tmp_path, name)
    upstream = Upstream(works=False)
    served = Layer(name, ttl, upstream).resolve(tmp_path, NOW)
    assert served.status == 200 and served.body["stale"] is True and ids(served.body) == ["old"]
    assert served.body["updated_at"] == updated.strftime("%Y-%m-%dT%H:%M:%SZ")  # the time of the last good refresh, not now
    assert upstream.calls == 1 and cache_text(tmp_path, name) == before


@pytest.mark.parametrize("name,ttl", LAYERS)
def test_no_cache_and_a_good_upstream_is_a_fresh_200_and_writes_the_file(tmp_path, name, ttl):
    upstream = Upstream()
    served = Layer(name, ttl, upstream).resolve(tmp_path, NOW)
    assert served.status == 200 and served.body["stale"] is False and ids(served.body) == ["new"]
    assert upstream.calls == 1 and cache_json(tmp_path, name)["updated_at"] == "2026-11-08T12:00:00Z"


@pytest.mark.parametrize("name,ttl", LAYERS)
def test_no_cache_and_a_failing_upstream_is_a_503_with_a_one_line_reason_and_no_file(tmp_path, name, ttl):
    upstream = Upstream(works=False, reason="NATS PIB: HTTP 503\nsecond line")
    served = Layer(name, ttl, upstream).resolve(tmp_path, NOW)
    assert served.status == 503 and served.body == {"reason": "NATS PIB: HTTP 503 second line"}
    assert upstream.calls == 1 and cache_text(tmp_path, name) is None


# --- a file that is not a usable cache ----------------------------------------------------------------------------------

BROKEN = {
    "empty": "",
    "truncated": '{"updated_at": "2026-11-08T11:59:00Z", "features": [{"type": "Fea',
    "not_json": "<html>nope</html>",
    "json_but_not_a_payload": '["a", "list"]',
    "no_features": '{"updated_at": "2026-11-08T11:59:00Z"}',
    "features_not_a_list": '{"updated_at": "2026-11-08T11:59:00Z", "features": {}}',
    "no_time": '{"features": []}',
    "time_without_zone": '{"updated_at": "2026-11-08T11:59:00", "features": []}',
    "time_not_a_time": '{"updated_at": "yesterday", "features": []}',
}


@pytest.mark.parametrize("name,ttl", LAYERS)
@pytest.mark.parametrize("kind", sorted(BROKEN))
def test_a_corrupt_cache_is_no_cache_and_the_next_good_refresh_overwrites_it(tmp_path, name, ttl, kind):
    (tmp_path / f"{name}.json").write_text(BROKEN[kind], encoding="utf-8")
    failing = Upstream(works=False)
    served = Layer(name, ttl, failing).resolve(tmp_path, NOW)
    assert served.status == 503 and failing.calls == 1  # no crash, and nothing to serve stale
    assert cache_text(tmp_path, name) == BROKEN[kind]  # a failed refresh does not touch it
    working = Upstream()
    served = Layer(name, ttl, working).resolve(tmp_path, NOW)
    assert served.status == 200 and ids(served.body) == ["new"]
    assert cache_json(tmp_path, name)["features"] == NEW


@pytest.mark.parametrize("name,ttl", LAYERS)
def test_a_cache_dated_in_the_future_is_not_trusted_as_fresh(tmp_path, name, ttl):
    put_cache(tmp_path, name, NOW + timedelta(hours=1))
    upstream = Upstream()
    served = Layer(name, ttl, upstream).resolve(tmp_path, NOW)
    assert upstream.calls == 1 and ids(served.body) == ["new"] and served.body["updated_at"] == "2026-11-08T12:00:00Z"


def test_the_two_layers_keep_separate_files(tmp_path):
    put_cache(tmp_path, "notams", NOW - NOTAM_TTL - SECOND)
    put_cache(tmp_path, "airspace", NOW - SECOND)
    airspace, notams = Upstream(), Upstream(works=False)
    assert Layer("airspace", AIRSPACE_TTL, airspace).resolve(tmp_path, NOW).body["stale"] is False
    assert Layer("notams", NOTAM_TTL, notams).resolve(tmp_path, NOW).body["stale"] is True
    assert (airspace.calls, notams.calls) == (0, 1)  # one layer failing did not touch the other
    assert sorted(p.name for p in tmp_path.iterdir()) == ["airspace.json", "notams.json"]


# --- what is written, and what is logged ---------------------------------------------------------------------------------


def test_the_file_is_written_through_a_temp_file_in_the_same_folder_then_renamed(tmp_path, monkeypatch):
    import os

    renames = []
    real_replace = os.replace

    def spy(source, target):
        renames.append((os.path.dirname(source), os.path.basename(source), target))
        assert (tmp_path / "notams.json").exists() is False  # at the moment of the rename the target does not hold a half file
        real_replace(source, target)

    monkeypatch.setattr(os, "replace", spy)
    Layer("notams", NOTAM_TTL, Upstream()).resolve(tmp_path, NOW)
    assert len(renames) == 1
    folder, temp_name, target = renames[0]
    assert folder == str(tmp_path) and temp_name.startswith(".tmp-") and target == tmp_path / "notams.json"
    assert [p.name for p in tmp_path.iterdir()] == ["notams.json"]  # no temp file left behind


def test_a_write_that_fails_leaves_the_old_file_and_no_temp_file(tmp_path, monkeypatch):
    import os

    put_cache(tmp_path, "notams", NOW - NOTAM_TTL - SECOND)
    before = cache_text(tmp_path, "notams")

    def broken(source, target):
        raise OSError("disk full")

    monkeypatch.setattr(os, "replace", broken)
    served = Layer("notams", NOTAM_TTL, Upstream()).resolve(tmp_path, NOW)
    assert served.status == 200 and ids(served.body) == ["new"]  # the answer still goes out
    assert cache_text(tmp_path, "notams") == before and [p.name for p in tmp_path.iterdir()] == ["notams.json"]


def test_a_refresh_that_raises_anything_else_is_a_failed_refresh_and_its_message_is_not_logged_or_served(tmp_path, caplog):
    secret = "SECRET NOTAM TEXT 01632960000"

    def explode(now):
        raise ValueError(secret)

    with caplog.at_level(logging.DEBUG):
        served = Layer("notams", NOTAM_TTL, explode).resolve(tmp_path, NOW)
    assert served.status == 503 and secret not in json.dumps(served.body) and served.body["reason"] == "notams refresh failed (ValueError)"
    assert secret not in caplog.text and "ValueError" in caplog.text


def test_the_folder_is_made_when_it_is_missing(tmp_path):
    folder = tmp_path / "data" / "map"
    Layer("notams", NOTAM_TTL, Upstream()).resolve(folder, NOW)
    assert (folder / "notams.json").is_file()


# --- two requests at the same time make one upstream call -----------------------------------------------------------------


def test_requests_that_arrive_while_a_refresh_runs_share_it(tmp_path):
    import threading

    started, release = threading.Event(), threading.Event()
    calls = []

    def slow(now):
        calls.append(now)
        started.set()
        assert release.wait(5)
        return NEW

    layer = Layer("notams", NOTAM_TTL, slow)

    async def go():
        first = asyncio.ensure_future(layer.resolve_once(tmp_path, lambda: NOW))
        await asyncio.to_thread(started.wait, 5)
        others = [asyncio.ensure_future(layer.resolve_once(tmp_path, lambda: NOW)) for _ in range(4)]
        await asyncio.sleep(0)  # let them all reach the refresh
        release.set()
        return await asyncio.gather(first, *others)

    served = asyncio.run(go())
    assert len(calls) == 1
    assert all(s.status == 200 and ids(s.body) == ["new"] for s in served)


def test_a_request_after_the_refresh_finished_reads_the_new_file_and_calls_nothing(tmp_path):
    upstream = Upstream()
    layer = Layer("notams", NOTAM_TTL, upstream)

    async def go():
        return [await layer.resolve_once(tmp_path, lambda: NOW), await layer.resolve_once(tmp_path, lambda: NOW + timedelta(minutes=14))]

    first, second = asyncio.run(go())
    assert upstream.calls == 1 and ids(second.body) == ["new"] and second.body["updated_at"] == first.body["updated_at"]


def test_a_request_after_the_ttl_calls_upstream_once_more(tmp_path):
    upstream = Upstream()
    layer = Layer("notams", NOTAM_TTL, upstream)

    async def go():
        await layer.resolve_once(tmp_path, lambda: NOW)
        return await layer.resolve_once(tmp_path, lambda: NOW + NOTAM_TTL)

    served = asyncio.run(go())
    assert upstream.calls == 2 and served.body["updated_at"] == "2026-11-08T12:15:00Z"
