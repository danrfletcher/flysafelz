"""The on-demand read for one landing zone and window (G8, G23).

    reader_set(parsed_notams, lz, window) -> list[ParsedNotam]
    read_for_lz(parsed_notams, lz, window, llm, cache) -> dict[id, ReaderResult]
    failed_before(parsed_notams, lz, window, cache) -> set[id]

Which NOTAMs are read is ticket 2's code-only answer and nothing else (G23). `prefilter` runs first. `sort_notam` then
runs on each NOTAM it kept, with the whole window as the span, and a NOTAM is read when the sort says TO_READER, or
VIA_AIRSPACE (read for its contact). IGNORE and AWARENESS NOTAMs are never sent to the model. This module neither filters
nor ranks again, and no reading, however unimportant it finds the NOTAM, takes one out of the set: facts can downgrade,
opinions can't.

Each NOTAM in the set goes through `read_cached` (cached_read.py): the cache first (a hit is status STORED and costs no
model call), else `read_notam`, and only a READ or REPAIRED result is stored. FAILED and UNAVAILABLE come back to the
caller but are never stored, so the next call retries just those.

One `ThreadPoolExecutor` of `settings.concurrency` workers is shared by every read in the call. A worker does a NOTAM's first
read and its repair one after the other inside its own slot (that is `read_notam`), and a repair is never submitted back to
the pool, so the limit counts first reads and repairs together and a full pool can never wait on work queued behind it.
An empty set makes no pool. A NOTAM that raises costs only its own entry, which comes back UNAVAILABLE.

The same id twice in the input is read once (the first one).
"""

import logging
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

from langchain_core.language_models.chat_models import BaseChatModel

from flysafelz.contracts import ParsedNotam
from flysafelz.pib import LatLon, SortAction, Window, prefilter, sort_notam
from flysafelz.reader.cache import CACHE_ERRORS, was_failed
from flysafelz.reader.cached_read import read_cached
from flysafelz.reader.contract import ReaderResult, ReaderStatus
from flysafelz.reader.prompt import PROMPT_VERSION
from flysafelz.reader.read import model_name
from flysafelz.reader.settings import SETTINGS
from flysafelz.redact import redact_phones

log = logging.getLogger(__name__)

READ_ACTIONS = (SortAction.TO_READER, SortAction.VIA_AIRSPACE)


def reader_set(parsed_notams: list[ParsedNotam], lz: LatLon, window: Window) -> list[ParsedNotam]:
    """The NOTAMs to read for this landing zone and window, once each by id, in input order. Decided by the prefilter and the sort alone."""
    unique: dict[str, ParsedNotam] = {}
    for parsed in parsed_notams:
        unique.setdefault(parsed.id, parsed)  # the first of a repeated id wins
    kept, _dropped = prefilter(list(unique.values()), lz, window)
    return [parsed for parsed in kept if sort_notam(parsed, window, lz).action in READ_ACTIONS]


def _unavailable(parsed: ParsedNotam, llm: BaseChatModel, error: Exception) -> ReaderResult:
    """The entry for a NOTAM whose read raised: unread and not stored, so the next call retries it. Only the error's class and its redacted text are logged."""
    log.warning("reader unavailable: %s: the read raised %s: %s", parsed.id, type(error).__name__, redact_phones(str(error))[0])
    return ReaderResult(notam_id=parsed.id, reading=None, status=ReaderStatus.UNAVAILABLE, failed_checks=[], model=model_name(llm), prompt_version=PROMPT_VERSION, latency_s=0.0, tokens=None)


def read_for_lz(
    parsed_notams: list[ParsedNotam],
    lz: LatLon,
    window: Window,
    llm: BaseChatModel,
    cache: Path | None,
    *,
    now: datetime | None = None,
) -> dict[str, ReaderResult]:
    """Read every NOTAM in the reader set for this landing zone and window. One entry per NOTAM in the set, keyed by id.

    `cache` is the SQLite file of the reading cache, or None to bypass it (nothing read, nothing written). `now` stamps a
    row written to it and is the clock when not given.
    """
    chosen = reader_set(parsed_notams, lz, window)
    if not chosen:
        return {}
    stamp = datetime.now(UTC) if now is None else now
    results: dict[str, ReaderResult] = {}
    with ThreadPoolExecutor(max_workers=SETTINGS.concurrency) as pool:
        futures = [(parsed, pool.submit(read_cached, parsed, llm, cache, stamp)) for parsed in chosen]
        for parsed, future in futures:
            try:
                results[parsed.id] = future.result().result
            except Exception as error:  # noqa: BLE001 - one NOTAM's failure, whatever it is, must not stop the others
                results[parsed.id] = _unavailable(parsed, llm, error)
    return results


def failed_before(parsed_notams: list[ParsedNotam], lz: LatLon, window: Window, cache: Path | None) -> set[str]:
    """The ids in this reader set whose last answer was FAILED or UNAVAILABLE and that have not been read since.

    Ask before `read_for_lz`. Those still not STORED afterwards are the retries of that call (R3.2, GP5). `cache` None
    (bypass) records nothing, so it gives the empty set. A file that cannot be used gives the empty set too: the read
    itself warns about it once and goes on without it.
    """
    if cache is None:
        return set()
    found: set[str] = set()
    for parsed in reader_set(parsed_notams, lz, window):
        try:
            if was_failed(cache, parsed.id):
                found.add(parsed.id)
        except CACHE_ERRORS:
            return set()
    return found
