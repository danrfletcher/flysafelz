"""Reading, validating and loading the verdict sheet (4b, PR-5.F1). Data in, data out: no model, no scoring, no upload.

    SheetError                       one line, for anything wrong with the sheet file
    parse_sheet(text, name)          the sheet's entries, each with the exact text span it occupies in the file
    check_entry(entry, ...)          validate one entry (G10) and count its rows: an EntryReport
    report_line(report, stale, ...)  the one line `--check` prints for an entry (GP5)
    load_sheet(path)                 the labelled sheet as Spec section 4 data, per example id (GP6)

The sheet is a YAML list, one entry per example (see verdict_sheet). Validation is shared by `--check` and
load_sheet (G10):
  * `verdict` is one of contracts.Verdict or null; `approaching_limits` is true, false or null;
  * every `notam_actions` value is one of contracts.NotamAction or null;
  * the rows are contiguous and cover assess()'s whole-UTC-hour span of the entry's window (a 07:30 window
    starts at 07:00); a `window: default` entry is checked for contiguity only, unless the caller knows the
    default window (--check does, from the examples file);
  * a row with a verdict but no `approaching_limits`, or the reverse, is incomplete: it is counted, it is not an
    error, and load_sheet skips it.
Every error names the example id and the row (its 1-based number and its start), or the NOTAM id.

load_sheet returns, per example id, and raises SheetError listing every error when the sheet does not validate:
    verdict_by_segment   [{start, end, verdict, approaching_limits}] for fully labelled rows only
    tier                 [{start, end, tier}] for the OUTLOOK / TOO_FAR rows (no label keys)
    notam_actions        {NOTAM id: action} only when `actions_done` is true and no UNREAD_OVER_LZ is left (G4);
                         a null value loads as IGNORE, and so does any NOTAM id that is not listed
Times are 'YYYY-MM-DDTHH:MM:SSZ' text, as in the sheet. An entry whose example has gone from the examples file
(an orphan) still loads.
"""

import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from flysafelz.clients.pib_source import utc_text
from flysafelz.contracts import NotamAction, Tier, Verdict
from flysafelz.daylight import ONE_HOUR, utc_hours

TIER_ONLY = (Tier.OUTLOOK.value, Tier.TOO_FAR.value)
WINDOW_DEFAULT = "default"
_YAML_WORD = re.compile(r"\s+")


class SheetError(Exception):
    """Something about an example, the examples file, the sheet or the output stops the build. The message is the one line printed."""


# --- the file ----------------------------------------------------------------------------------------


@dataclass(frozen=True)
class SheetEntry:
    """One item of the sheet's list, as parsed, with the exact text it occupies in the file.

    `span` runs from the start of the line the entry begins on to the start of the next entry's line (or the end
    of the file), so it holds the entry's own comments and blank lines. It is what a rerun must never change.
    """

    position: int  # 1-based, in file order
    data: Any  # whatever the item parsed to; a well-formed entry is a mapping
    line: int  # 1-based line the entry starts on
    span: str

    @property
    def id(self) -> str | None:
        value = self.data.get("id") if isinstance(self.data, dict) else None
        return value if isinstance(value, str) and value.strip() else None


def _problem_line(exc: yaml.YAMLError) -> tuple[int | None, str]:
    mark = getattr(exc, "problem_mark", None) or getattr(exc, "context_mark", None)
    problem = getattr(exc, "problem", None) or getattr(exc, "context", None) or "invalid YAML"
    return (None if mark is None else mark.line + 1), _YAML_WORD.sub(" ", str(problem)).strip()


def parse_sheet(text: str, name: str | Path) -> list[SheetEntry]:
    """The entries of a sheet's text, in file order. An empty (or comment-only) text has none.

    Raises SheetError, one line with the YAML line number, when the text is not valid YAML or is not a list.
    """
    loader = yaml.SafeLoader(text)
    try:
        node = loader.get_single_node()
        if node is None:
            return []
        data = loader.construct_document(node)
    except yaml.YAMLError as exc:
        line, problem = _problem_line(exc)
        where = "" if line is None else f" at line {line}"
        raise SheetError(f"{name}: invalid YAML{where}: {problem}") from None
    finally:
        loader.dispose()
    if not isinstance(node, yaml.SequenceNode) or not isinstance(data, list):
        raise SheetError(f"{name}: line {node.start_mark.line + 1}: the sheet must be a YAML list of entries")
    starts = [text.rfind("\n", 0, item.start_mark.index) + 1 for item in node.value]
    ends = [*starts[1:], len(text)] if starts else []
    return [SheetEntry(i, data[i - 1], text.count("\n", 0, a) + 1, text[a:b]) for i, (a, b) in enumerate(zip(starts, ends, strict=True), start=1)]


def read_sheet(path: Path) -> tuple[str, list[SheetEntry]]:
    """The sheet's text (decoded as it is, line endings kept) and its entries. Raises SheetError, one line."""
    try:
        text = Path(path).read_bytes().decode("utf-8")
    except FileNotFoundError:
        raise SheetError(f"{path}: no such file") from None
    except (OSError, UnicodeDecodeError) as exc:
        raise SheetError(f"{path}: cannot be read ({type(exc).__name__})") from None
    return text, parse_sheet(text, path)


# --- one entry ---------------------------------------------------------------------------------------


@dataclass
class EntryReport:
    """What validating one entry found: its counts (GP5) and its errors (G10)."""

    id: str | None
    mode: str | None
    rows: int = 0
    tier_rows: int = 0
    labelled: int = 0
    unlabelled: int = 0  # labelable rows that are not fully labelled, incomplete ones included
    incomplete: int = 0  # a verdict without approaching_limits, or the reverse
    unread_left: int = 0
    actions_done: bool = False
    errors: list[str] = field(default_factory=list)
    edge_errors: list[str] = field(default_factory=list)  # the errors in `errors` about where rows start and end: a gap, an overlap, a row not after its start, rows not covering the window


def _time(value: object) -> datetime | None:
    """A sheet time (quoted text, or what YAML made of an unquoted one) as aware UTC; None when it is not one."""
    if isinstance(value, str):
        try:
            value = datetime.fromisoformat(value)
        except ValueError:
            return None
    if not isinstance(value, datetime) or value.tzinfo is None:
        return None
    return value.astimezone(UTC)


def entry_span(entry_window: object, default_span: tuple[datetime, datetime] | None) -> tuple[datetime, datetime] | None:
    """The whole-hour span the rows must cover for this entry's `window`, or None when it cannot be known (no default given)."""
    if entry_window == WINDOW_DEFAULT:
        window = default_span
    elif isinstance(entry_window, dict):
        start, end = _time(entry_window.get("start")), _time(entry_window.get("end"))
        window = (start, end) if start is not None and end is not None else None
    else:
        window = None
    if window is None:
        return None
    hours = utc_hours(*window)
    return (hours[0], hours[-1] + ONE_HOUR) if hours else None


def _enum_text(enum: type[Verdict] | type[NotamAction]) -> str:
    return ", ".join(member.value for member in enum)


def check_entry(entry: SheetEntry, default_span: tuple[datetime, datetime] | None = None) -> EntryReport:
    """Validate one entry (G10) and count it (GP5). Never raises: every problem is an error in the report.

    `default_span` is the window of an entry whose `window` is `default`, when the caller knows it.
    """
    data = entry.data
    if not isinstance(data, dict):
        return EntryReport(None, None, errors=[f"entry #{entry.position} (line {entry.line}): must be a mapping"])
    name = entry.id
    report = EntryReport(name, data.get("mode") if isinstance(data.get("mode"), str) else None, actions_done=data.get("actions_done") is True)
    label = f"example {name}" if name else f"entry #{entry.position} (line {entry.line})"
    if name is None:
        report.errors.append(f"{label}: missing id")
    if "actions_done" in data and not isinstance(data["actions_done"], bool):
        report.errors.append(f"{label}: actions_done must be true or false")
    _check_actions(data.get("notam_actions"), label, report)
    rows = data.get("rows")
    if not isinstance(rows, list):
        report.errors.append(f"{label}: rows must be a list")
        return report
    report.rows = len(rows)
    if not rows:
        report.errors.append(f"{label}: has no rows")
        return report
    span = entry_span(data.get("window"), default_span)
    if span is None and data.get("window") != WINDOW_DEFAULT:
        report.errors.append(f'{label}: window must be "default" or {{start, end}} with times')
    previous_end: datetime | None = None
    bounds: list[tuple[datetime, datetime] | None] = []
    for number, row in enumerate(rows, start=1):
        bounds.append(_check_row(row, number, label, report, previous_end))
        previous_end = bounds[-1][1] if bounds[-1] else None
    if span is not None:
        _check_coverage(rows, bounds, span, label, report)
    return report


def _check_actions(actions: object, label: str, report: EntryReport) -> None:
    if actions is None:
        return
    if not isinstance(actions, dict):
        report.errors.append(f"{label}: notam_actions must be a mapping of NOTAM id to an action")
        return
    for notam_id, action in actions.items():
        if action == NotamAction.UNREAD_OVER_LZ.value:
            report.unread_left += 1  # a count, not an error: Dan has to read it (G4)
        elif action is not None and not (isinstance(action, str) and action in {member.value for member in NotamAction}):
            report.errors.append(f"{label}, NOTAM {notam_id}: action {action!r} is not one of {_enum_text(NotamAction)} or null")


def _check_row(row: object, number: int, label: str, report: EntryReport, previous_end: datetime | None) -> tuple[datetime, datetime] | None:
    """Validate and count one row; the row's (start, end) when both are good times."""
    if not isinstance(row, dict):
        report.errors.append(f"{label}, row {number}: must be a mapping")
        return None
    start, end = _time(row.get("start")), _time(row.get("end"))
    where = f"{label}, row {number} ({utc_text(start) if start else row.get('start')})"
    if start is None or end is None:
        report.errors.append(f"{where}: start and end must be times with a timezone, e.g. 2026-10-31T07:00:00Z")
        bounds = None
    elif start >= end:
        report.errors.append(f"{where}: ends at {utc_text(end)}, which is not after its start")
        report.edge_errors.append(report.errors[-1])
        bounds = (start, end)
    else:
        bounds = (start, end)
    if start is not None and previous_end is not None and start != previous_end:
        kind = "gap" if start > previous_end else "overlap"
        report.errors.append(f"{where}: {kind}: the row before ends at {utc_text(previous_end)} but this one starts at {utc_text(start)}")
        report.edge_errors.append(report.errors[-1])
    if "tier" in row:
        if row["tier"] not in TIER_ONLY:
            report.errors.append(f"{where}: tier {row['tier']!r} is not one of {', '.join(TIER_ONLY)}")
        report.tier_rows += 1
        return bounds
    _count_labels(row, where, report)
    return bounds


def _count_labels(row: dict, where: str, report: EntryReport) -> None:
    verdict, limits = row.get("verdict"), row.get("approaching_limits")
    valid = True
    if verdict is not None and not (isinstance(verdict, str) and verdict in {member.value for member in Verdict}):
        report.errors.append(f"{where}: verdict {verdict!r} is not one of {_enum_text(Verdict)} or null")
        valid = False
    if limits is not None and not isinstance(limits, bool):
        report.errors.append(f"{where}: approaching_limits must be true, false or null, not {limits!r}")
        valid = False
    if valid and verdict is not None and limits is not None:
        report.labelled += 1
        return
    report.unlabelled += 1
    if valid and (verdict is None) != (limits is None):
        report.incomplete += 1


def _check_coverage(rows: list, bounds: list, span: tuple[datetime, datetime], label: str, report: EntryReport) -> None:
    first, last = bounds[0], bounds[-1]
    if first is not None and first[0] != span[0]:
        report.errors.append(f"{label}, row 1 ({utc_text(first[0])}): the window starts in the hour {utc_text(span[0])}, so the first row must start there")
        report.edge_errors.append(report.errors[-1])
    if last is not None and last[1] != span[1]:
        report.errors.append(f"{label}, row {len(rows)} ({utc_text(last[0])}): the last row ends at {utc_text(last[1])}; the window needs the rows to end at {utc_text(span[1])}")
        report.edge_errors.append(report.errors[-1])


def report_line(report: EntryReport, stale: str = "no", *, orphan: bool = False) -> str:
    """The `--check` line for one entry (GP5): rows, labelled, unlabelled, unread left, actions done, stale.

    `stale` is 'no', or the parenthesised change, e.g. '(inputs)'. Tier-only and incomplete rows are shown in brackets
    when there are any; an entry whose NOTAM actions are not done says so.
    """
    rows = f"{report.rows} ({report.tier_rows} tier-only)" if report.tier_rows else str(report.rows)
    unlabelled = f"{report.unlabelled} ({report.incomplete} incomplete)" if report.incomplete else str(report.unlabelled)
    done = "yes" if report.actions_done else "no (NOTAM actions not done)"
    line = f"{report.id} ({report.mode}): rows {rows}, labelled {report.labelled}, unlabelled {unlabelled}, unread left {report.unread_left}, actions done {done}, stale {stale}"
    return f"{line}, orphan (not in the examples file)" if orphan else line


# --- load_sheet (GP6) ----------------------------------------------------------------------------------


class NotamActions(dict):
    """{NOTAM id: action} where a NOTAM that is not listed counts as IGNORE: `actions[any_id]` never raises."""

    def __missing__(self, notam_id: str) -> str:
        return NotamAction.IGNORE.value


def _stamp(value: object) -> str:
    """A row time that validation has already accepted, as 'YYYY-MM-DDTHH:MM:SSZ' text."""
    moment = _time(value)
    assert moment is not None  # check_entry reports any row time that is not a time
    return utc_text(moment)


def _entry_data(entry: SheetEntry) -> dict[str, Any]:
    rows = entry.data["rows"]
    labelled = [
        {"start": _stamp(r["start"]), "end": _stamp(r["end"]), "verdict": r["verdict"], "approaching_limits": r["approaching_limits"]}
        for r in rows
        if "tier" not in r and r.get("verdict") is not None and r.get("approaching_limits") is not None
    ]
    tiers = [{"start": _stamp(r["start"]), "end": _stamp(r["end"]), "tier": r["tier"]} for r in rows if "tier" in r]
    loaded: dict[str, Any] = {"verdict_by_segment": labelled, "tier": tiers}
    actions = entry.data.get("notam_actions") or {}
    if entry.data.get("actions_done") is True and NotamAction.UNREAD_OVER_LZ.value not in actions.values():
        loaded["notam_actions"] = NotamActions({str(i): (a if a is not None else NotamAction.IGNORE.value) for i, a in actions.items()})
    return loaded


def check_entries(entries: list[SheetEntry], defaults: dict[str, tuple[datetime, datetime]] | None = None) -> list[EntryReport]:
    """check_entry for each entry, plus the one error a single entry cannot see: an id used twice."""
    reports = [check_entry(e, (defaults or {}).get(e.id or "")) for e in entries]
    first_at: dict[str, int] = {}
    for entry, report in zip(entries, reports, strict=True):
        if entry.id is None:
            continue
        if entry.id in first_at:
            report.errors.append(f"example {entry.id}: duplicate id (entries #{first_at[entry.id]} and #{entry.position})")
        first_at.setdefault(entry.id, entry.position)
    return reports


def load_sheet(path: str | Path) -> dict[str, dict[str, Any]]:
    """The sheet at `path` as data for scoring: {example id: {verdict_by_segment, tier, notam_actions?}} (see the module doc).

    Raises SheetError (one error per line, each naming the example and the row) when the sheet is missing, is not
    valid YAML (the message has the line number) or has an error in it. Incomplete rows and unfinished NOTAM actions
    are not errors: the incomplete rows are left out and the actions are not returned.
    """
    _text, entries = read_sheet(Path(path))
    errors = [e for report in check_entries(entries) for e in report.errors]
    if errors:
        raise SheetError("\n".join(errors))
    return {str(entry.id): _entry_data(entry) for entry in entries}
