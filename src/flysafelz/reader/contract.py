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
    ProhibitiveReason,
)


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


class ReaderOutputError(ValueError):
    """The model's reply is not a valid ReaderOutput: not JSON, cut off, a wrong enum, or a field it may not fill.

    `raw` is the reply text as received. Nothing in this package catches this or turns it into a reading.
    """

    def __init__(self, reason: str, raw: str):
        super().__init__(reason)
        self.reason = reason
        self.raw = raw


class ReaderStatus(StrEnum):
    """How a read ended. This module produces READ; the later steps produce the others."""

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
    """One NOTAM's read: the reading (None when there is none), how it ended, and what the call cost.

    tokens is None when the endpoint returned no usage. latency_s is the time the model call took, in seconds.
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
