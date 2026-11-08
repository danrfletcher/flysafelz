"""One place to build sheet rows, examples and answers for the evaluator tests (5a PR-5.F1): a few lines each, synthetic, no held-out id.

t(hour, minute, day=, month=)      an aware UTC time (default 31 Oct 2026, a GMT day)
row(start, end, verdict, limits)   a sheet row with a verdict (`edge` names a cut, as in the sheet)
tier_row(start, end, tier)         a sheet row with a `tier` and no verdict
example(...)                       a loader record for an assess example with its sheet labels, or a bare record for another path
seg(start, end, verdict, limits)   an answer segment
answer(...)                        a ScoredAnswer
"""

from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any

from evals.agent.evaluators.answer import ScoredAnswer, ScoredSegment

DAY = (2026, 10, 31)


def t(hour: int, minute: int = 0, *, day: int = DAY[2], month: int = DAY[1]) -> datetime:
    return datetime(DAY[0], month, day, hour, minute, tzinfo=UTC)


def iso(moment: datetime) -> str:
    return moment.isoformat().replace("+00:00", "Z")


def row(start: datetime, end: datetime, verdict: str, limits: bool = False, edge: str | None = None) -> dict[str, Any]:
    found: dict[str, Any] = {"start": iso(start), "end": iso(end)}
    if edge is not None:
        found["edge"] = edge
    return {**found, "verdict": verdict, "approaching_limits": limits}


def tier_row(start: datetime, end: datetime, tier: str = "OUTLOOK") -> dict[str, Any]:
    return {"start": iso(start), "end": iso(end), "tier": tier}


def example(
    rows: list[dict[str, Any]] | None = None,
    *,
    id: str = "syn_1",
    path: str = "assess",
    window: tuple[datetime, datetime] | None = None,
    tiers: list[dict[str, Any]] | None = None,
    actions: Mapping[str, str | None] | None = None,
    now: datetime | str | None = None,
    drop: tuple[str, ...] = (),
    compound: bool = False,
    sheet: bool = True,
) -> dict[str, Any]:
    """An assess example with the sheet labels the loader joins in (`sheet=False` leaves them out). Any other path is just its `path`."""
    reference: dict[str, Any] = {"path": path}
    if path == "assess":
        start, end = window or (t(8), t(18))
        reference["window"] = [iso(start), iso(end)]
        if sheet:
            reference |= {"verdict_by_segment": rows or [], "tier": tiers or [], "notam_actions": dict(actions or {})}
    return {
        "id": id,
        "inputs": {"now": now if isinstance(now, str) else iso(now or t(7, 30)), "question": "Can I fly?"},
        "fixtures": {"case": "synthetic", "drop": list(drop)},
        "reference": reference,
        "metadata": {"split": "dev", "compound": compound},
    }


def seg(start: datetime, end: datetime, verdict: str, limits: bool = False) -> ScoredSegment:
    return ScoredSegment(start, end, verdict, limits)  # type: ignore[arg-type]


def answer(segments: list[ScoredSegment] | None = None, *, path: str = "assess", actions: dict[str, Any] | None = None, text: str = "") -> ScoredAnswer:
    return ScoredAnswer(path, tuple(segments or []), actions or {}, text)
