"""What the model fills (ReaderOutput) and what a read hands back (ReaderResult).

ReaderOutput is the semantic subset of NotamReading and nothing else. It has no field for the id, the Q-line, B/C, D,
F/G or the area: code builds those and merges them in (read.py), so the model can neither fill nor override one (F1).
It is STRICT (`extra="forbid"`), so a reply with a field outside the subset is rejected, not trimmed.
"""

from enum import StrEnum

from pydantic import BaseModel

from flysafelz.contracts import (
    STRICT,
    AppliesTo,
    Category,
    Contact,
    Exemption,
    Limit,
    NotamReading,
    ParsedNotam,
    ProhibitiveReason,
)
from flysafelz.notam_area import Area


class ReaderOutput(BaseModel):
    """The fields the model fills from the E text. Every one is required; floor and ceiling may be null (E states none)."""

    model_config = STRICT

    category: Category
    activity: str
    floor: Limit | None
    ceiling: Limit | None
    applies_to: AppliesTo
    exemptions: list[Exemption]
    contact: Contact
    activates_designators: list[str]
    prohibitive_reason: ProhibitiveReason


def merge_reading(parsed: ParsedNotam, area: Area, output: ReaderOutput) -> NotamReading:
    """The reading: the model's fields plus the coded ones code supplies (F1): id, the area (None is the Q-circle) and item D.

    The one place a reading is put together, for a model's answer (read.py) and for a stored row (cache.py) alike. The
    schedule is item D as written ("" when there is none, so notam_action() keeps using item D).
    """
    return NotamReading(id=parsed.id, geometry=area, schedule=parsed.item_d or "", **output.model_dump())


class ReaderStatus(StrEnum):
    """How a read ended. read_notam produces READ, REPAIRED, FAILED and UNAVAILABLE; STORED is a reading loaded from the cache (cache.py).

    FAILED means the model answered but its answer was wrong twice. UNAVAILABLE means no usable answer came back at all.
    Either way there is no reading and the NOTAM is unread. Their values, "failed" and "unavailable", are the two literals
    the assessment takes as `reader_status`.
    """

    READ = "read"
    REPAIRED = "repaired"
    FAILED = "failed"
    UNAVAILABLE = "unavailable"
    STORED = "stored"


class TokenUsage(BaseModel):
    """Tokens the endpoint reported for one call."""

    model_config = STRICT

    input_tokens: int
    output_tokens: int
    total_tokens: int


class ReaderResult(BaseModel):
    """One NOTAM's read: the reading (None when there is none), how it ended, and what the calls cost.

    reading is set for READ and REPAIRED only. failed_checks are plain sentences: empty for READ; for REPAIRED the checks
    the first answer failed; for FAILED the checks the second answer failed; for UNAVAILABLE the checks the first answer
    failed when the repair was the call that got no answer, else empty. latency_s is the seconds the model calls took, the
    repair included. tokens is the usage of every answer added up, or None when any answer reported none.
    """

    model_config = STRICT

    notam_id: str
    reading: NotamReading | None
    status: ReaderStatus
    failed_checks: list[str]
    model: str
    prompt_version: str
    latency_s: float
    tokens: TokenUsage | None
