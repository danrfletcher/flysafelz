"""The reading cache (G9): a SQLite file of NOTAM readings the model has already given and the checks have passed.

    cache_key(parsed, model)            -> CacheKey
    lookup(path, parsed, key)           -> ReaderResult (status STORED) | None
    store(path, key, result, now)       -> bool
    record_failure(path, notam_id, now) -> None
    was_failed(path, notam_id)          -> bool

The key is the NOTAM id, a sha256 of the user text the model was sent (the redacted E text and the coded fields: Q-line,
B, C, D, F, G), the model name, and a sha256 of the system text it was sent (the abbreviation rows and PROMPT_VERSION
are in it). Both texts come from `render_prompt`, the function `read_notam` builds its messages with, so the key cannot
drift from what the model saw. The hashed E text is the redacted one, so no real phone number feeds a key (F3).

A row is the key, the model's semantic fields as JSON (ReaderOutput: no id, no geometry, no schedule) and `created_at`.
On load code merges the id, the area (G3) and item D (G4) back in with `merge_reading`, so a parser fix takes effect
without clearing anything, and a model can never leave a coded field behind in a row (F1). The loaded reading is checked
again with `check_reading` before it is returned. A row that is bad JSON, has an unknown enum, an extra or missing key,
or fails a check counts as a miss: the NOTAM is read again and `store` replaces the row. It is logged with the NOTAM id only.

Only READ and REPAIRED results are stored. FAILED and UNAVAILABLE never are (R3.2), so the next question retries them.

To count those retries (GP5: "retries: n"), a second table `failures` holds the ids that last came back FAILED or
UNAVAILABLE, with `created_at` and nothing else: no key, no reading, no text. `lookup` never reads it, so it can never
return a reading. `record_failure` adds an id, and `store` removes it in the same transaction as the reading is filed.
A NOTAM that is a miss and whose id is in `failures` is a retry; a NOTAM never seen before, a re-issue (new id) and a
row that was corrupt are misses but not retries.

The path is an argument, `now` is an argument (this module never reads the clock), and a connection lives for one
operation, so threads share nothing but the file. Writes are serialised by a lock as well. `data/` is created on the
first write. Errors from SQLite or the file system are for the caller (cached_read.py), which reads uncached instead.
"""

import hashlib
import json
import logging
import sqlite3
import threading
from collections.abc import Mapping
from contextlib import closing
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from flysafelz import notam_area
from flysafelz.contracts import NotamReading, ParsedNotam
from flysafelz.reader import prompt
from flysafelz.reader.checks import check_reading
from flysafelz.reader.contract import ReaderOutput, ReaderResult, ReaderStatus, merge_reading
from flysafelz.reader.prompt import ABBREVIATIONS, render_prompt

log = logging.getLogger(__name__)

STORABLE = (ReaderStatus.READ, ReaderStatus.REPAIRED)
CACHE_ERRORS = (sqlite3.Error, OSError)  # what a file that cannot be opened, read or written raises

_WRITE_LOCK = threading.Lock()
_CREATE = "CREATE TABLE IF NOT EXISTS readings (key TEXT PRIMARY KEY, json TEXT NOT NULL, created_at TEXT NOT NULL)"
_CREATE_FAILURES = "CREATE TABLE IF NOT EXISTS failures (notam_id TEXT PRIMARY KEY, created_at TEXT NOT NULL)"
_SELECT = "SELECT json FROM readings WHERE key = ?"
_SELECT_FAILURE = "SELECT 1 FROM failures WHERE notam_id = ?"
_UPSERT_FAILURE = "INSERT INTO failures (notam_id, created_at) VALUES (?, ?) ON CONFLICT(notam_id) DO UPDATE SET created_at = excluded.created_at"
_DELETE_FAILURE = "DELETE FROM failures WHERE notam_id = ?"
_UPSERT = "INSERT INTO readings (key, json, created_at) VALUES (?, ?, ?) ON CONFLICT(key) DO UPDATE SET json = excluded.json, created_at = excluded.created_at"


@dataclass(frozen=True)
class CacheKey:
    """What a stored reading is filed under. Any one part changing is a miss."""

    notam_id: str
    input_sha256: str  # of the user text: the redacted E text and the coded fields sent
    model: str
    prompt_sha256: str  # of the system text: the abbreviation rows and PROMPT_VERSION included


def key_text(key: CacheKey) -> str:
    """The key as the one string the table is keyed by: a JSON list, so no part can run into the next."""
    return json.dumps([key.notam_id, key.input_sha256, key.model, key.prompt_sha256])


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def cache_key(parsed: ParsedNotam, model: str, abbreviations: Mapping[str, str] = ABBREVIATIONS) -> CacheKey:
    """The key for one NOTAM read by one model. Hashes exactly the strings `render_prompt` gives `read_notam`."""
    rendered = render_prompt(parsed, abbreviations)
    return CacheKey(parsed.id, _sha256(rendered.user), model, _sha256(rendered.system))


def _semantic_json(reading: NotamReading) -> str:
    """The model's fields of a reading as JSON, and nothing else: no id, no geometry, no schedule."""
    output = ReaderOutput(**{name: getattr(reading, name) for name in ReaderOutput.model_fields})
    return output.model_dump_json()


def store(path: Path, key: CacheKey, result: ReaderResult, now: datetime) -> bool:
    """File a READ or REPAIRED result under the key (the latest wins). True when a row was written, False for any other result.

    Raises sqlite3.Error or OSError when the file cannot be written; the caller decides what that means.
    """
    if result.status not in STORABLE or result.reading is None:
        return False
    row = (key_text(key), _semantic_json(result.reading), now.isoformat())
    path.parent.mkdir(parents=True, exist_ok=True)
    with _WRITE_LOCK, closing(sqlite3.connect(path)) as connection, connection:  # `with connection` commits, or rolls back on an error
        _create_tables(connection)
        connection.execute(_UPSERT, row)
        connection.execute(_DELETE_FAILURE, (key.notam_id,))  # read now, so no longer a failure to retry
    return True


def _create_tables(connection: sqlite3.Connection) -> None:
    connection.execute(_CREATE)
    connection.execute(_CREATE_FAILURES)


def record_failure(path: Path, notam_id: str, now: datetime) -> None:
    """Note that this NOTAM last came back FAILED or UNAVAILABLE, so asking again is a retry. Holds the id and `now` only.

    Raises sqlite3.Error or OSError when the file cannot be written; the caller decides what that means.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with _WRITE_LOCK, closing(sqlite3.connect(path)) as connection, connection:
        _create_tables(connection)
        connection.execute(_UPSERT_FAILURE, (notam_id, now.isoformat()))


def was_failed(path: Path, notam_id: str) -> bool:
    """True when the NOTAM's last answer was FAILED or UNAVAILABLE and it has not been read since. False for no file.

    Raises sqlite3.Error when the file is not a database; the caller decides what that means.
    """
    if not path.is_file():
        return False
    with closing(sqlite3.connect(path)) as connection:
        _create_tables(connection)
        return connection.execute(_SELECT_FAILURE, (notam_id,)).fetchone() is not None


def lookup(path: Path, parsed: ParsedNotam, key: CacheKey) -> ReaderResult | None:
    """The stored reading for the key as a STORED result, or None for a miss (no file, no row, or a row that cannot be used).

    The area and schedule are built from `parsed` now. The model is the key's, the tokens are None and the latency is 0.
    Raises sqlite3.Error when the file is not a database; the caller decides what that means.
    """
    if not path.is_file():
        return None
    with closing(sqlite3.connect(path)) as connection:
        _create_tables(connection)  # a file with no table yet (empty, or made by something else) is a file with no rows
        found = connection.execute(_SELECT, (key_text(key),)).fetchone()
    if found is None:
        return None
    try:
        output = ReaderOutput.model_validate_json(found[0])
        reading = merge_reading(parsed, notam_area.build_area(parsed)[0], output)
    except (ValueError, TypeError):  # a ValidationError is a ValueError; the error text is not logged: it quotes the row's values
        log.warning("reading cache: %s: the stored row does not fit the form, so it is read again", key.notam_id)
        return None
    failures = check_reading(parsed, reading, render_prompt(parsed).redaction)
    if failures:
        log.warning("reading cache: %s: the stored row fails %d check(s), so it is read again", key.notam_id, len(failures))
        return None
    return ReaderResult(
        notam_id=key.notam_id,
        reading=reading,
        status=ReaderStatus.STORED,
        failed_checks=[],
        model=key.model,
        prompt_version=prompt.PROMPT_VERSION,
        latency_s=0.0,
        tokens=None,
    )
