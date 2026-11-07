"""The verdict sheet's draft mode (4b, PR-4.F1): code's own verdicts and NOTAM actions, pre-filled for Dan to correct.

Model-free. Everything here comes from the Assessment that produced the rows, never from a second run.

    why_text(segment)                 the read-only `# why:` text: blockers, cautions, then conditions, joined with '; '
    unread_ids(segment)               the unread NOTAMs behind an UNVERIFIED segment, sorted (the row's `unread` tag)
    worst_action(a, b)                the more severe of two NotamActions
    draft_actions(assessment)         {NOTAM id: worst action value} over the entry's segments, severity then id
    ignored_by_code(...)              (id, reason) for every NOTAM within the local area that code ignored (not for distance)
    fill_rows(rows, assessment)       blind rows -> draft rows (verdict, approaching_limits, unread, and a why each)
    DraftEntry, DraftRow, why_of      the entry and rows as dicts, carrying the comment text the file gets
    render_entry(entry)               the YAML text of a draft entry, with the `# why:` and `# ignored by code` comments

`# why:` and the ignored block are YAML comments only: after yaml.safe_load there is no key for either (F4). The
text of a comment is one line however the reason is worded (newlines and control characters become a space).
A blind entry never comes through here (F2); blind rows carry nothing derived from code.
"""

from collections.abc import Collection, Iterable, Sequence
from typing import Any

import yaml

from flysafelz.clients.pib_source import utc_text
from flysafelz.contracts import Assessment, IgnoredNotam, NotamAction, Segment, Tier, Verdict

# Most severe first. A NOTAM's line in `notam_actions` is the first of these any segment gives it.
SEVERITY: tuple[NotamAction, ...] = (
    NotamAction.NO_FLY,
    NotamAction.UNREAD_OVER_LZ,
    NotamAction.CONTACT_FIRST,
    NotamAction.AVOID_SECTOR,
    NotamAction.AWARENESS,
    NotamAction.AWARENESS_MAP,
    NotamAction.IGNORE,
)
_RANK = {action: rank for rank, action in enumerate(SEVERITY)}

NO_WHY = "no blockers, cautions or conditions"
WHY_PREFIX = "# why: "
IGNORED_HEADER = "# ignored by code (add a line above to overrule):"
UNREAD_NOTE = "# unread: read the NOTAM and set its action"
_LISTED = "notam_actions"
_ROWS = "rows"


# --- one row ---------------------------------------------------------------------------------------


def one_line(text: str) -> str:
    """The text on a single comment line: any line break or non-printing character becomes a space, runs of spaces collapse."""
    return " ".join("".join(c if c.isprintable() or c == " " else " " for c in text).split())


def why_text(segment: Segment) -> str:
    """The segment's blocker texts, then caution texts, then conditions (which keep their 'UNVERIFIED: ' prefix), joined with '; '."""
    texts = [*(b.text for b in segment.blockers), *(c.text for c in segment.cautions), *segment.conditions]
    return one_line("; ".join(texts)) or NO_WHY


def unread_ids(segment: Segment) -> list[str]:
    """The NOTAMs behind an UNVERIFIED segment that are unread over the LZ, sorted. Empty for any other verdict."""
    if segment.verdict is not Verdict.UNVERIFIED:
        return []
    return sorted({a.notam_id for a in segment.notam_actions if a.action is NotamAction.UNREAD_OVER_LZ})


class DraftRow(dict):
    """A labelable row as a plain dict, plus the `# why:` text its comment line carries (not a key)."""

    why: str | None = None


class DraftEntry(dict):
    """A draft entry as a plain dict, plus the block of NOTAMs code ignored, as (id, reason), for the comment under notam_actions."""

    ignored: Sequence[tuple[str, str]] = ()


def why_of(row: dict[str, Any]) -> str | None:
    """The `# why:` text a row carries; None for a row that has none (a tier-only row, or a blind one)."""
    return row.why if isinstance(row, DraftRow) else None


def segments_of(assessment: Assessment) -> list[Segment]:
    """Every segment of the assessment's VERDICT hours, in time order: the order of the labelable rows."""
    return [s for hour in assessment.hours if hour.tier is Tier.VERDICT for s in hour.segments]


def fill_rows(rows: Sequence[dict[str, Any]], assessment: Assessment) -> list[dict[str, Any]]:
    """Draft rows from the blind rows of the same assessment: a labelable row takes its segment's own verdict and flag.

    A tier-only row (OUTLOOK, TOO_FAR) is kept as it is: tier and nothing else. Raises ValueError when the rows
    are not this assessment's segments (they must come from the same call).
    """
    segments = iter(segments_of(assessment))
    filled: list[dict[str, Any]] = []
    for row in rows:
        if "tier" in row:
            filled.append(row)
            continue
        segment = next(segments, None)
        if segment is None or (row["start"], row["end"]) != (utc_text(segment.start), utc_text(segment.end)):
            raise ValueError(f"row {row['start']} to {row['end']} is not a segment of the assessment")
        draft = DraftRow(row)
        draft["verdict"] = segment.verdict.value
        draft["approaching_limits"] = segment.approaching_limits
        unread = unread_ids(segment)
        if unread:
            draft["unread"] = unread
        draft.why = why_text(segment)
        filled.append(draft)
    if next(segments, None) is not None:
        raise ValueError("the rows leave out a segment of the assessment")
    return filled


# --- notam_actions ---------------------------------------------------------------------------------


def worst_action(first: NotamAction, second: NotamAction) -> NotamAction:
    """The more severe: NO_FLY > UNREAD_OVER_LZ > CONTACT_FIRST > AVOID_SECTOR > AWARENESS > AWARENESS_MAP > IGNORE."""
    return first if _RANK[first] <= _RANK[second] else second


def draft_actions(assessment: Assessment) -> dict[str, str]:
    """One entry per NOTAM in any segment's notam_actions: its worst action (as text), in severity order then by id.

    IGNORE is not written: a NOTAM that is not listed counts as IGNORE.
    """
    worst: dict[str, NotamAction] = {}
    for segment in segments_of(assessment):
        for result in segment.notam_actions:
            worst[result.notam_id] = worst_action(worst[result.notam_id], result.action) if result.notam_id in worst else result.action
    listed = sorted(((i, a) for i, a in worst.items() if a is not NotamAction.IGNORE), key=lambda item: (_RANK[item[1]], item[0]))
    return {notam_id: action.value for notam_id, action in listed}


def ignored_by_code(ignored: Iterable[IgnoredNotam], local_ids: Collection[str], superseded: Iterable[IgnoredNotam]) -> list[tuple[str, str]]:
    """(NOTAM id, code's reason) sorted by id, for each ignored NOTAM within the local area and not superseded.

    Chosen by id only, never by matching the reason text. `local_ids` are the NOTAMs whose Q-circle passed the
    prefilter's distance check (within the local area of the LZ): the NOTAMs code dropped for distance are not
    in it, so they never reach the sheet. A NOTAM the prefilter dropped for time or height (it starts after the
    window, say) is in it, and so is one ignored as 'beyond the local area' from a reading's own geometry (its
    Q-circle is near). `superseded` are the NOTAMs replaced or cancelled. A repeated id is listed once, with its
    first reason.
    """
    gone = {s.notam_id for s in superseded}
    chosen: dict[str, str] = {}
    for entry in ignored:
        if entry.notam_id in local_ids and entry.notam_id not in gone:
            chosen.setdefault(entry.notam_id, entry.reason)
    return sorted(chosen.items())


# --- the text ----------------------------------------------------------------------------------------


class _Dumper(yaml.SafeDumper):
    """SafeDumper that writes a _Flow list on one line, e.g. unread: [J3857/26]."""


class _Flow(list):
    pass


_Dumper.add_representer(_Flow, lambda dumper, data: dumper.represent_sequence("tag:yaml.org,2002:seq", data, flow_style=True))


def _dump(value: Any) -> list[str]:
    return yaml.dump(value, Dumper=_Dumper, sort_keys=False, default_flow_style=False, allow_unicode=True).splitlines()


def _row_lines(row: dict[str, Any]) -> list[str]:
    """One row as the '- start: ...' lines of a sequence under a key, then its `# why:` comment (aligned with the keys)."""
    plain = {k: (_Flow(v) if k == "unread" else v) for k, v in row.items()}
    lines = _dump([plain])
    why = why_of(row)
    return lines if why is None else [*lines, f"  {WHY_PREFIX}{one_line(why)}"]


def _action_lines(actions: dict[str, str], ignored: Sequence[tuple[str, str]]) -> list[str]:
    """notam_actions, then the ignored block under it (one comment line each). `{}` when no action is listed."""
    lines = [f"{_LISTED}:"] if actions else [f"{_LISTED}: {{}}"]
    for notam_id, action in actions.items():
        line = "  " + _dump({notam_id: action})[0]
        lines.append(f"{line}  {UNREAD_NOTE}" if action == NotamAction.UNREAD_OVER_LZ.value else line)
    if ignored:
        lines.append(f"  {IGNORED_HEADER}")
        lines += [f"  # {one_line(notam_id)}: {one_line(reason)}" for notam_id, reason in ignored]
    return lines


def render_entry(entry: DraftEntry) -> str:
    """The YAML text of one draft entry as an item of the sheet's list, with its comments. safe_load of it equals dict(entry)."""
    lines: list[str] = []
    for key, value in entry.items():
        if key == _LISTED:
            lines += _action_lines(value, entry.ignored)
        elif key == _ROWS and value:
            lines.append(f"{_ROWS}:")
            for row in value:
                lines += _row_lines(row)
        else:
            lines += _dump({key: value})
    return "\n".join(("- " if i == 0 else "  ") + line for i, line in enumerate(lines)) + "\n"
