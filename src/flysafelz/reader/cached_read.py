"""A read that asks the cache first (G9): `read_cached` for one NOTAM, `read_set` for a set and its counts.

    read_cached(parsed, llm, cache, now) -> CachedRead
    read_set(parsed_notams, llm, cache, now) -> ReadSet

`cache` is the path of the SQLite file, or None for bypass: nothing is read from it and nothing is written (the eval's
mode). A hit costs no model call and has status STORED. A miss goes to `read_notam`, and a READ or REPAIRED result is
stored; FAILED and UNAVAILABLE never are, so the next question asks the model about them again (R3.2).

A cache that cannot be used (garbage bytes at the path, a directory there, a full disk, a lock that never clears) never
stops a read: one warning is logged and the NOTAM is read without it.

Counts: `calls` is model calls, a repair included. `retries` is the NOTAMs sent to the model again because their last
answer was FAILED or UNAVAILABLE (R3.2, GP5). The cache keeps the ids of those in its `failures` table (cache.py), and a
miss whose id is there is a retry. A NOTAM never seen before, a re-issue under a new id, a changed text read for the
first time and a corrupt row are misses but not retries. Bypass records nothing, so it has no retries.
"""

import logging
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from langchain_core.language_models.chat_models import BaseChatModel

from flysafelz.contracts import ParsedNotam
from flysafelz.reader.cache import CACHE_ERRORS, STORABLE, cache_key, lookup, record_failure, store, was_failed
from flysafelz.reader.contract import ReaderResult, ReaderStatus
from flysafelz.reader.read import model_name, read_notam

log = logging.getLogger(__name__)

_REPAIR_ASKED = (ReaderStatus.REPAIRED, ReaderStatus.FAILED)


@dataclass(frozen=True)
class CachedRead:
    """One NOTAM's result, whether it came from the cache, the model calls it cost (0 for a hit) and whether it is a retry."""

    result: ReaderResult
    hit: bool
    calls: int
    retry: bool = False  # a miss for a NOTAM whose last answer was FAILED or UNAVAILABLE


@dataclass(frozen=True)
class ReadSet:
    """The reads of a set of NOTAMs by id, with the model calls they cost and how many NOTAMs were retries."""

    reads: dict[str, CachedRead]
    calls: int
    retries: int


def model_calls(result: ReaderResult) -> int:
    """How many `invoke` calls `read_notam` made for this result: one, plus the repair when it asked for one.

    The repair was asked for when the result is REPAIRED or FAILED, or UNAVAILABLE with failed checks left over (the
    first answer failed and the repair got none). STORED took none.
    """
    if result.status is ReaderStatus.STORED:
        return 0
    repaired = result.status in _REPAIR_ASKED or (result.status is ReaderStatus.UNAVAILABLE and bool(result.failed_checks))
    return 1 + int(repaired)


def read_cached(parsed: ParsedNotam, llm: BaseChatModel, cache: Path | None, now: datetime) -> CachedRead:
    """Read one NOTAM through the cache. `now` stamps a row it writes. Never raises for the cache."""
    if cache is None:
        result = read_notam(parsed, llm)
        return CachedRead(result, False, model_calls(result))
    key = cache_key(parsed, model_name(llm))
    usable, retry = True, False
    try:
        stored = lookup(cache, parsed, key)
        retry = stored is None and was_failed(cache, parsed.id)
    except CACHE_ERRORS as error:
        log.warning("reading cache unusable: %s: reading without it: %s", parsed.id, type(error).__name__)
        stored, usable = None, False
    if stored is not None:
        return CachedRead(stored, True, 0)
    result = read_notam(parsed, llm)
    if usable:
        try:
            if result.status in STORABLE:
                store(cache, key, result, now)
            else:
                record_failure(cache, parsed.id, now)
        except CACHE_ERRORS as error:
            log.warning("reading cache unusable: %s: the result was not stored: %s", parsed.id, type(error).__name__)
    return CachedRead(result, False, model_calls(result), retry)


def read_set(parsed_notams: Iterable[ParsedNotam], llm: BaseChatModel, cache: Path | None, now: datetime) -> ReadSet:
    """Read each NOTAM in turn through the cache. The pool that runs them `concurrency` at a time, after the prefilter and the sort, is read_for_lz's (reader_set.py)."""
    reads = {parsed.id: read_cached(parsed, llm, cache, now) for parsed in parsed_notams}
    return ReadSet(reads, sum(read.calls for read in reads.values()), sum(read.retry for read in reads.values()))
