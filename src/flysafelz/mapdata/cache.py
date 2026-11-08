"""The disk cache of one map layer, its freshness rule, and the one answer a route gives.

    Layer(name, ttl, refresh, key=)   one route's layer: `refresh(now)` returns the items (`key`: "features" by default) or raises RefreshFailed
    Layer.resolve(folder, now)        -> Served(status, body): the rule below, with at most one call to `refresh`
    Layer.resolve_once(folder, clock) the same, shared: a request that arrives while a refresh is running waits for its answer

The rule, for the cache file <folder>/<name>.json:
    fresh (age from zero up to the TTL, the TTL itself excluded)   200 from the file, stale false, no call upstream
    expired, missing, truncated or not JSON, or dated in the future   one refresh attempt
        it works     200 with the new items, stale false, updated_at = now, the file rewritten
        it fails     with a cache: 200 with the old items, stale true, updated_at unchanged, the file untouched
                     without one: 503 and {"reason": one line}
A file that cannot be read as a payload is treated as no cache and is overwritten by the next good refresh.

The file is written to a temporary file in the same folder, then renamed over the old one, so a reader never sees half a file.
Nothing is written anywhere else. The body is `updated_at` (ISO UTC), `stale` and the items under `key`. The default key "features" makes the body a
GeoJSON FeatureCollection (the airspace and NOTAM routes); the wind route's key is "grid" (its two leaflet-velocity records).
"""

import asyncio
import json
import logging
import os
import tempfile
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from http import HTTPStatus
from pathlib import Path

from flysafelz.clients.pib_source import utc_text

log = logging.getLogger("flysafelz.mapdata")

OK = HTTPStatus.OK.value
UNAVAILABLE = HTTPStatus.SERVICE_UNAVAILABLE.value
FEATURES = "features"
FILE_SUFFIX = ".json"
TEMP_PREFIX = ".tmp-"
NO_AGE = timedelta()


def one_line(reason: str) -> str:
    return " ".join(reason.split())


class RefreshFailed(Exception):
    """The upstream could not be read. `reason` is one line, with no key and no NOTAM text."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = one_line(reason)


@dataclass(frozen=True)
class Served:
    status: int
    body: dict


def read_payload(path: Path, key: str = FEATURES) -> tuple[datetime, list[dict]] | None:
    """(updated_at, items under `key`) from a cache file, or None when it is missing, truncated, not JSON or not a payload."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        updated = datetime.fromisoformat(data["updated_at"])
        features = data[key]
    except (OSError, ValueError, KeyError, TypeError):
        return None
    if updated.tzinfo is None or not isinstance(features, list):
        return None
    return updated.astimezone(UTC), features


def write_payload(path: Path, updated_at: datetime, features: list[dict], key: str = FEATURES) -> None:
    """Write the payload to a temporary file beside `path`, then rename it over `path`."""
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps({"updated_at": utc_text(updated_at), key: features}, separators=(",", ":"))
    handle, temp_name = tempfile.mkstemp(prefix=TEMP_PREFIX, suffix=FILE_SUFFIX, dir=path.parent)
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as temp:
            temp.write(text)
        os.replace(temp_name, path)
    except BaseException:
        Path(temp_name).unlink(missing_ok=True)
        raise


def body_of(updated_at: datetime, features: list[dict], *, stale: bool, key: str = FEATURES) -> dict:
    envelope = {key: features, "updated_at": utc_text(updated_at), "stale": stale}
    return {"type": "FeatureCollection", **envelope} if key == FEATURES else envelope


@dataclass
class Layer:
    name: str
    ttl: timedelta
    refresh: Callable[[datetime], list[dict]]
    key: str = FEATURES
    _flight: asyncio.Future | None = field(default=None, init=False, repr=False)

    def path(self, folder: Path) -> Path:
        return folder / f"{self.name}{FILE_SUFFIX}"

    def resolve(self, folder: Path, now: datetime) -> Served:
        path = self.path(folder)
        cached = read_payload(path, self.key)
        if cached is not None and NO_AGE <= now - cached[0] < self.ttl:  # a cache dated in the future is not trusted as fresh
            return Served(OK, body_of(cached[0], cached[1], stale=False, key=self.key))
        try:
            features = self.refresh(now)
        except RefreshFailed as failure:
            log.warning("map layer %s: refresh failed: %s", self.name, failure.reason)
            return self._failed(cached, failure.reason)
        except Exception as error:  # noqa: BLE001 - whatever a refresh raises is a failed refresh; only the class name is logged, it may hold upstream text
            log.warning("map layer %s: refresh raised %s", self.name, type(error).__name__)
            return self._failed(cached, f"{self.name} refresh failed ({type(error).__name__})")
        try:
            write_payload(path, now, features, self.key)
        except OSError as error:
            log.warning("map layer %s: the cache file could not be written (%s)", self.name, type(error).__name__)
        return Served(OK, body_of(now, features, stale=False, key=self.key))

    def _failed(self, cached: tuple[datetime, list[dict]] | None, reason: str) -> Served:
        if cached is None:
            return Served(UNAVAILABLE, {"reason": one_line(reason)})
        return Served(OK, body_of(cached[0], cached[1], stale=True, key=self.key))

    async def resolve_once(self, folder: Path, clock: Callable[[], datetime]) -> Served:
        """`resolve` in a worker thread. Requests that arrive while one is running share its answer, so two at once make one upstream call."""
        if self._flight is None or self._flight.done():
            self._flight = asyncio.ensure_future(asyncio.to_thread(lambda: self.resolve(folder, clock())))
        return await asyncio.shield(self._flight)
