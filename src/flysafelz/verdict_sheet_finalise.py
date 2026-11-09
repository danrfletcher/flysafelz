"""Finalise the verdict sheet (5a PR-2.F1, G1, G2): Dan's labelled NOTAM actions in place of code's, then the rows' verdicts.

    python scripts/build_verdict_sheet.py --finalise          (make dataset runs it)

    finalised_problems(sheet, root, ...)   {entry id: "verdict sheet is stale (<part> changed): run make dataset"} for every
                                           `actions_done: true` entry that is unfinalised or stale. The one source `--check` and the
                                           eval loader share.
    recombine(built, actions)              the (verdict, approaching_limits) of every labelable segment of one assessed example
    run_finalise(...)                      the command: rewrite the sheet's row labels and stamps, print the summary

For each entry with `actions_done: true` the example's committed case is assessed again (verdict_sheet._assess_example: the same call
the sheet's rows came from, the case's own readings.yaml, no reader and no agent). Each VERDICT-tier segment is then put together again:

    weather, daylight and airspace blockers          code's, kept
    non-NOTAM UNVERIFIED reasons                     code's, kept (a failed, stale or dropped source, a weather hole, other airspace UNVERIFIED)
    NOTAM blockers and UNREAD_OVER_LZ reasons        code's, removed, and so is a danger area's "activated by NOTAM <id>, which has not been
                                                     read" reason: it is the same unread NOTAM, which Dan has now read (a NOTAM he did not list
                                                     is IGNORE). Without this the dev_20 J3219 rows would stay UNVERIFIED after he set CHECK_SUP.
    each NOTAM active in the segment                 Dan's action: NO_FLY is a blocker; CONTACT_FIRST, CHECK_SUP, AVOID_SECTOR,
                                                     AWARENESS, AWARENESS_MAP and IGNORE add none (a NOTAM he did not label counts as IGNORE)
    verdict                                          assess's own precedence: any blocker NO_GO, else any reason UNVERIFIED, else PILOT_DISCRETION
    approaching_limits                               code's (cautions come from weather and daylight only)

"Active in the segment" is decided with the code assess cuts on: notams_in_hour() over the prefiltered NOTAMs, then active_intervals().
There is no second time rule here.

Only each labelable row's `verdict` and `approaching_limits` and the entry's `finalised` stamp are written, as text edits at the
values' own places in the file, so every comment, key order and the 4b `fingerprint` stay as they were. Before anything is written
the new text is parsed again and must equal the old data except for those keys. An entry whose rows code no longer cuts the same way
(a boundary moved, a row added or dropped) is an error naming the entry and the first differing row; nothing is written for it.

The stamp (`finalised`) is a sealed set of hashes, kept apart from the 4b `fingerprint`:
    rules      src/flysafelz/rules.yaml
    modules    {src/flysafelz/<module>.py: hash} for every flysafelz module `flysafelz.assess` imports, found by walking the imports
               at run time, and this module
    <file>     weather.json, airspace.json, pib.xml, readings.yaml of the case
    labels     the entry's own notam_actions and the start and end of its rows (so a hand edit of Dan's labels shows)
    seal       a hash over all of the above (verdict_sheet_fingerprint.sealed)
`code_root` (default: the src/ folder of this package) is where the rules and module bytes are read from; tests and the stale demonstration
point it at a copy.
"""

import ast
import hashlib
import json
import os
import re
import sys
import tempfile
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from flysafelz.assess import notams_in_hour
from flysafelz.clients.pib_source import utc_text
from flysafelz.contracts import NotamAction, Tier, Verdict
from flysafelz.daylight import ONE_HOUR
from flysafelz.notam_action import active_intervals
from flysafelz.sources import FIXTURES_DIR
from flysafelz.verdict_sheet import Example, SheetError, _assess_example, _default_windows, read_examples
from flysafelz.verdict_sheet_check import SheetEntry, _time, check_entries, parse_sheet, read_sheet
from flysafelz.verdict_sheet_fingerprint import ABSENT, FIXTURE_FILES, SEAL, file_parts, sealed

STAMP_KEY = "finalised"
PART_RULES = "rules.yaml"
KEY_RULES = "rules"
KEY_MODULES = "modules"
KEY_LABELS = "labels"
PART_LABELS = "labels"
STAMP_MISSING = "finalised stamp missing"
STAMP_UNREADABLE = "finalised stamp unreadable"
STALE_TEXT = "verdict sheet is stale ({what}): run make dataset"
NOT_FINALISED = "not finalised (NOTAM actions not done)"
NOTAM_SOURCE = "notam"  # contracts.Blocker.source of a NOTAM blocker
# airspace._danger_over_lz words an activation nobody has read like this; a test runs the real airspace code against it.
UNREAD_ACTIVATION = "NOTAM {notam_id}, which has not been read"
UNVERIFIED_PREFIX = "UNVERIFIED: "  # assess's prefix on an UNVERIFIED reason in a segment's conditions
UNLABELLED = "unlabelled"
PACKAGE = "flysafelz"
SRC_PREFIX = "src/"
SELF_MODULE = f"{PACKAGE}/verdict_sheet_finalise.py"
ENTRY_ROOT = f"{PACKAGE}.assess"
SRC_DIR = Path(__file__).resolve().parent.parent
_HEX = re.compile(r"[0-9a-f]{64}")

# What each of Dan's actions adds to a segment: only NO_FLY is a blocker.
BLOCKING_ACTIONS = frozenset({NotamAction.NO_FLY})

Label = tuple[str | None, bool | None]  # (verdict, approaching_limits) of one row; (None, None) when it has none


# --- the modules `assess` imports (G2) -----------------------------------------------------------------------------------------------


def _module_file(name: str, src_dir: Path) -> Path | None:
    base = src_dir.joinpath(*name.split("."))
    if base.with_suffix(".py").is_file():
        return base.with_suffix(".py")
    init = base / "__init__.py"
    return init if init.is_file() else None


def _imported_names(tree: ast.AST, module: str, is_package: bool) -> set[str]:
    """Every dotted name a module's source imports, relative imports resolved; a `from a import b` gives `a` and `a.b`."""
    found: set[str] = set()
    package = module if is_package else module.rpartition(".")[0]
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            base = node.module or ""
            if node.level:
                anchor = package.split(".")
                if node.level > len(anchor):
                    continue
                anchor = anchor[: len(anchor) - (node.level - 1)]
                base = ".".join([*anchor, *([base] if base else [])])
            found.add(base)
            found.update(f"{base}.{alias.name}" for alias in node.names)
    return found


def imported_modules(src_dir: Path = SRC_DIR, start: str = ENTRY_ROOT) -> list[str]:
    """The package-relative file of every flysafelz module `start` imports, itself and the parent packages' __init__ included (sorted).

    Found by walking the imports in each module's source (ast), never by a fixed list, so a module added to assess's graph
    later is covered without an edit here. Imports inside functions and under TYPE_CHECKING count too. Nothing is imported.
    """
    seen: dict[str, Path] = {}
    queue = [start]
    while queue:
        name = queue.pop()
        if name in seen or not (name == PACKAGE or name.startswith(f"{PACKAGE}.")):
            continue
        path = _module_file(name, src_dir)
        if path is None:
            continue  # `from flysafelz.x import name` where `name` is not a module
        seen[name] = path
        parts = name.split(".")
        queue.extend(".".join(parts[:i]) for i in range(1, len(parts)))  # a parent package runs its __init__ first
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        queue.extend(_imported_names(tree, name, path.name == "__init__.py"))
    return sorted(path.relative_to(src_dir).as_posix() for path in seen.values())


def _display(relative: str) -> str:
    return f"{SRC_PREFIX}{relative}"


def module_files(src_dir: Path = SRC_DIR) -> list[str]:
    """The files the stamp covers, package-relative: the modules `assess` imports, and this one."""
    return sorted({*imported_modules(src_dir), SELF_MODULE})


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def _file_hash(path: Path) -> str:
    try:
        return _sha256(path.read_bytes())
    except FileNotFoundError:
        return ABSENT


@dataclass(frozen=True)
class CodeHashes:
    """The hashes of the code a finalise depended on: rules.yaml and each module, read once per run."""

    rules: str
    modules: dict[str, str]  # display path -> hash

    @classmethod
    def read(cls, code_root: Path | None = None) -> "CodeHashes":
        """Hash the files under `code_root` (default: this package's src/). Which files is decided by the real package's import graph."""
        files = module_files()
        base = SRC_DIR if code_root is None else Path(code_root)
        return cls(_file_hash(base / PACKAGE / "rules.yaml"), {_display(f): _file_hash(base / f) for f in files})


# --- the stamp (G2) --------------------------------------------------------------------------------------------------------------------


def labels_part(entry: Mapping[str, Any]) -> str:
    """The hash of Dan's labels: the entry's notam_actions as written, and the start and end of each row."""
    rows = entry.get("rows")
    spans = []
    for row in rows if isinstance(rows, list) else []:
        times = [row.get(k) if isinstance(row, dict) else None for k in ("start", "end")]
        spans.append([utc_text(t) if (t := _time(v)) is not None else repr(v) for v in times])
    actions = entry.get("notam_actions")
    return _sha256(_canonical({"notam_actions": None if actions is None else {str(k): v for k, v in actions.items()}, "rows": spans}))


def stamp_parts(entry: Mapping[str, Any], case_dir: Path, code: CodeHashes) -> dict[str, Any]:
    """The unsealed stamp for an entry over the case in case_dir: rules, modules, the four fixture files, labels."""
    return {KEY_RULES: code.rules, KEY_MODULES: dict(code.modules), **file_parts(case_dir), KEY_LABELS: labels_part(entry)}


def stamp_for(entry: Mapping[str, Any], case_dir: Path, code: CodeHashes) -> dict[str, Any]:
    """The `finalised` value to store: the parts and their seal."""
    return sealed(stamp_parts(entry, case_dir, code))


def _is_hash(value: object, *, allow_absent: bool = True) -> bool:
    return isinstance(value, str) and (_HEX.fullmatch(value) is not None or (allow_absent and value == ABSENT))


def read_stamp(stored: object) -> dict[str, Any] | None:
    """The parts of a stored stamp, or None when it is unreadable: wrong shape, a bad hash, no seal or a seal that does not match."""
    if not isinstance(stored, dict) or SEAL not in stored:
        return None
    parts = {k: v for k, v in stored.items() if k != SEAL}
    modules = parts.get(KEY_MODULES)
    shape = {KEY_RULES, KEY_MODULES, *FIXTURE_FILES, KEY_LABELS}
    if set(parts) != shape or not isinstance(modules, dict) or not all(isinstance(k, str) and _is_hash(v) for k, v in modules.items()):
        return None
    if not all(_is_hash(v) for k, v in parts.items() if k != KEY_MODULES):
        return None
    return parts if stored[SEAL] == sealed(parts)[SEAL] else None


def changed_stamp_parts(stored: Mapping[str, Any], current: Mapping[str, Any]) -> list[str]:
    """The parts that differ, in order: rules.yaml, each module path, each fixture file, labels."""
    out = [PART_RULES] if stored[KEY_RULES] != current[KEY_RULES] else []
    old, new = stored[KEY_MODULES], current[KEY_MODULES]
    out += [path for path in sorted({*old, *new}) if old.get(path) != new.get(path)]
    out += [name for name in FIXTURE_FILES if stored[name] != current[name]]
    return out + ([PART_LABELS] if stored[KEY_LABELS] != current[KEY_LABELS] else [])


def stale_message(parts: Sequence[str]) -> str:
    """`verdict sheet is stale (<part> changed): run make dataset`; the stamp being missing or unreadable is named instead of a change."""
    what = parts[0] if parts[0] in (STAMP_MISSING, STAMP_UNREADABLE) else f"{', '.join(parts)} changed"
    return STALE_TEXT.format(what=what)


def entry_problem(entry: Mapping[str, Any], case_dir: Path, code: CodeHashes) -> str | None:
    """The stale message for one `actions_done: true` entry, or None when its stamp is there, sealed and matches what it depends on."""
    if STAMP_KEY not in entry:
        return stale_message([STAMP_MISSING])
    stored = read_stamp(entry[STAMP_KEY])
    if stored is None:
        return stale_message([STAMP_UNREADABLE])
    changed = changed_stamp_parts(stored, stamp_parts(entry, case_dir, code))
    return stale_message(changed) if changed else None


def problems_for(entries: Sequence[SheetEntry], examples: Mapping[str, Example], root: Path, code: CodeHashes) -> dict[str, str]:
    """{entry id: message} for every `actions_done: true` entry that is unfinalised or stale. An entry with no example is not checked."""
    problems: dict[str, str] = {}
    for entry in entries:
        data = entry.data
        if entry.id is None or not isinstance(data, dict) or data.get("actions_done") is not True:
            continue
        example = examples.get(entry.id)
        if example is None or example.path != "assess":
            continue
        message = entry_problem(data, root / FIXTURES_DIR / example.case, code)
        if message:
            problems[entry.id] = message
    return problems


def finalised_problems(sheet: Path, root: Path, examples: Path | None = None, *, code_root: Path | None = None) -> dict[str, str]:
    """The single source for `--check` and the eval loader: {entry id: stale message} for each `actions_done: true` entry that is not finalised.

    `sheet` is the sheet file, `root` the folder holding fixtures/. `examples` (default: examples.yaml beside the sheet) says which case
    each entry is for; an entry whose id is not in it is not checked. `actions_done: false` entries are never in the answer: they are
    not finalised and are not an error. An empty dict is a sheet the loader may use. Raises SheetError for an unreadable sheet or examples file.
    """
    _text, entries = read_sheet(Path(sheet))
    by_id = {e.id: e for e in read_examples(Path(examples) if examples is not None else Path(sheet).parent / "examples.yaml")}
    return problems_for(entries, by_id, Path(root), CodeHashes.read(code_root))


# --- recombining a segment (G1) ----------------------------------------------------------------------------------------------------------


def dan_actions(entry: Mapping[str, Any]) -> dict[str, NotamAction]:
    """Dan's action per NOTAM id. A null value, a NOTAM not listed or a null `notam_actions` map counts as IGNORE (as load_sheet does).

    Raises ValueError naming every NOTAM still marked UNREAD_OVER_LZ: it is not Dan's reading, so the entry cannot be finalised.
    """
    raw = entry.get("notam_actions") or {}
    unread = sorted(str(i) for i, a in raw.items() if a == NotamAction.UNREAD_OVER_LZ.value)
    if unread:
        raise ValueError(f"NOTAM {', '.join(unread)} is still UNREAD_OVER_LZ: read it and set its action")
    return {str(i): NotamAction(a) if a is not None else NotamAction.IGNORE for i, a in raw.items()}


def _active_by_hour(built: Any) -> dict[Any, list[tuple[str, list[tuple[Any, Any]]]]]:
    """For each VERDICT hour: [(NOTAM id, its active intervals inside the hour)] for the NOTAMs assess cuts the hour on."""
    out: dict[Any, list[tuple[str, list[tuple[Any, Any]]]]] = {}
    for hour in built.assessment.hours:
        if hour.tier is not Tier.VERDICT:
            continue
        in_hour = notams_in_hour(hour.hour, built.near, built.readings, built.lz)
        out[hour.hour] = [(n.id, active_intervals(n, built.readings.get(n.id), built.lz, hour.hour, hour.hour + ONE_HOUR)) for n in in_hour]
    return out


def kept_unverified(segment: Any, near_ids: Sequence[str]) -> list[str]:
    """The UNVERIFIED reasons of a segment's conditions that code keeps: all but the unread-NOTAM ones Dan's reading replaces.

    Removed: the UNREAD_OVER_LZ reasons (the segment's own notam_actions rows say which) and a danger area's "activated by NOTAM <id>,
    which has not been read" for a NOTAM in the prefiltered set.
    """
    unread = {UNVERIFIED_PREFIX + r.reason for r in segment.notam_actions if r.action is NotamAction.UNREAD_OVER_LZ}
    marks = [UNREAD_ACTIVATION.format(notam_id=i) for i in near_ids]
    return [c for c in segment.conditions if c.startswith(UNVERIFIED_PREFIX) and c not in unread and not any(m in c for m in marks)]


def recombine(built: Any, actions: Mapping[str, NotamAction]) -> list[tuple[Verdict, bool]]:
    """(verdict, approaching_limits) of every VERDICT-tier segment of an assessed example, in time order, with Dan's `actions` in place of code's.

    `built` is a verdict_sheet._Assessed. See the module doc for the rule. A NOTAM counts in a segment when one of its active
    intervals overlaps the segment; one Dan did not label is IGNORE.
    """
    active = _active_by_hour(built)
    near_ids = [n.id for n in built.near]
    labels: list[tuple[Verdict, bool]] = []
    for hour in built.assessment.hours:
        if hour.tier is not Tier.VERDICT:
            continue
        for segment in hour.segments:
            blocked = any(b.source != NOTAM_SOURCE for b in segment.blockers)
            unverified = bool(kept_unverified(segment, near_ids))
            for notam_id, spans in active[hour.hour]:
                if actions.get(notam_id, NotamAction.IGNORE) in BLOCKING_ACTIONS and any(a < segment.end and b > segment.start for a, b in spans):
                    blocked = True
            verdict = Verdict.NO_GO if blocked else Verdict.UNVERIFIED if unverified else Verdict.PILOT_DISCRETION
            labels.append((verdict, segment.approaching_limits))
    return labels


# --- reading the sheet text ----------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Edit:
    """Replace text[start:end] with `text`."""

    start: int
    end: int
    text: str


class EntryProblem(Exception):
    """Why one entry cannot be finalised; the message is the one line printed. Nothing is written for the entry."""


def _pairs(node: yaml.MappingNode) -> dict[str, yaml.Node]:
    return {k.value: v for k, v in node.value if isinstance(k, yaml.ScalarNode)}


def _last_index(node: yaml.Node) -> int:
    """The text index where the last scalar under `node` ends."""
    if isinstance(node, yaml.ScalarNode) or not node.value:
        return node.end_mark.index
    last = node.value[-1]
    return _last_index(last[1] if isinstance(last, tuple) else last)


def _line_end(text: str, index: int) -> int:
    end = text.find("\n", index)
    return len(text) if end < 0 else end


def _value_edit(node: yaml.Node, value: str) -> Edit:
    if not isinstance(node, yaml.ScalarNode):
        raise EntryProblem("a row's verdict or approaching_limits is not a plain value")
    if node.start_mark.index == node.end_mark.index:  # `verdict:` with nothing after the colon
        return Edit(node.start_mark.index, node.end_mark.index, f" {value}")
    return Edit(node.start_mark.index, node.end_mark.index, value)


def _text_of(label: Label) -> tuple[str, str]:
    verdict, flag = label
    return str(verdict), "true" if flag else "false"


def _stamp_lines(stamp: Mapping[str, Any], indent: int) -> list[str]:
    body = yaml.safe_dump({STAMP_KEY: dict(stamp)}, sort_keys=False, default_flow_style=False, allow_unicode=True, width=sys.maxsize)
    return [" " * indent + line for line in body.splitlines()]


def entry_edits(text: str, node: yaml.Node, data: Mapping[str, Any], labels: Sequence[Label], stamp: Mapping[str, Any]) -> list[Edit]:
    """The text edits that put `labels` on the entry's labelable rows and `stamp` on the entry. Only what differs is edited.

    Raises EntryProblem when the entry is not written as block YAML with a verdict and approaching_limits on each labelable row.
    """
    if not isinstance(node, yaml.MappingNode) or node.flow_style:
        raise EntryProblem("the entry is not written as a block mapping")
    keys = _pairs(node)
    rows_node = keys.get("rows")
    if not isinstance(rows_node, yaml.SequenceNode):
        raise EntryProblem("the entry has no rows list")
    edits: list[Edit] = []
    wanted = iter(labels)
    for number, (row_node, row) in enumerate(zip(rows_node.value, data["rows"], strict=True), start=1):
        if "tier" in row:
            continue
        verdict_text, flag_text = _text_of(next(wanted))
        row_keys = _pairs(row_node)
        for key, new_text, old in (("verdict", verdict_text, row.get("verdict")), ("approaching_limits", flag_text, row.get("approaching_limits"))):
            if key not in row_keys:
                raise EntryProblem(f"row {number} has no {key} key")
            if old != (new_text if key == "verdict" else new_text == "true"):
                edits.append(_value_edit(row_keys[key], new_text))
    if data.get(STAMP_KEY) != stamp:
        edits.append(_stamp_edit(text, node, keys, stamp))
    return edits


def _stamp_edit(text: str, node: yaml.MappingNode, keys: Mapping[str, yaml.Node], stamp: Mapping[str, Any]) -> Edit:
    indent = node.start_mark.column
    lines = _stamp_lines(stamp, indent)
    if STAMP_KEY in keys:  # replace the old stamp, key line to the end of its last line
        key_node = next(k for k, _v in node.value if isinstance(k, yaml.ScalarNode) and k.value == STAMP_KEY)
        start = text.rfind("\n", 0, key_node.start_mark.index) + 1
        return Edit(start, _line_end(text, _last_index(keys[STAMP_KEY])), "\n".join(lines))
    after = keys.get("fingerprint") or keys.get("actions_done")
    if after is None:
        raise EntryProblem("the entry has neither a fingerprint nor actions_done to put the stamp after")
    return Edit(_line_end(text, _last_index(after)), _line_end(text, _last_index(after)), "\n" + "\n".join(lines))


def apply_edits(text: str, edits: Sequence[Edit]) -> str:
    """The text with the edits made, last first so the indices hold. The edits must not overlap."""
    out = text
    previous = len(text) + 1
    for edit in sorted(edits, key=lambda e: (e.start, e.end), reverse=True):
        if edit.end > previous:
            raise EntryProblem("two edits overlap")
        out = out[: edit.start] + edit.text + out[edit.end :]
        previous = edit.start
    return out


# --- the command ---------------------------------------------------------------------------------------------------------------------


@dataclass
class Report:
    """What a finalise run found: the counts and the lines to print."""

    finalised: int = 0
    skipped: int = 0
    failed: int = 0
    before: Counter = field(default_factory=Counter)
    after: Counter = field(default_factory=Counter)
    moved: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def summary(self) -> list[str]:
        lines = [f"finalised {self.finalised} entries, {self.skipped} skipped (actions not done), {self.failed} failed"]
        names = [v.value for v in Verdict] + [UNLABELLED]
        lines.append("rows by verdict, before -> after:")
        lines += [f"  {name}: {self.before[name]} -> {self.after[name]}" for name in names if self.before[name] or self.after[name]]
        lines.append(f"{len(self.moved)} rows moved" + (":" if self.moved else ""))
        return lines + [f"  {line}" for line in self.moved]


def _shown(label: Label) -> str:
    verdict, flag = label
    return UNLABELLED if verdict is None else f"{verdict} (approaching limits)" if flag else str(verdict)


def _stored_labels(data: Mapping[str, Any]) -> list[Label]:
    return [(row.get("verdict"), row.get("approaching_limits")) for row in data["rows"] if "tier" not in row]


def _row_key(row: Mapping[str, Any]) -> tuple[str, str, Any]:
    start, end = _time(row.get("start")), _time(row.get("end"))
    return (utc_text(start) if start else repr(row.get("start")), utc_text(end) if end else repr(row.get("end")), row.get("tier"))


def _row_text(key: tuple[str, str, Any]) -> str:
    start, end, tier = key
    return f"{start} to {end}" + (f" (tier {tier})" if tier else "")


def edge_problem(stored: Sequence[Mapping[str, Any]], code_rows: Sequence[Mapping[str, Any]]) -> str | None:
    """None when code's rows have the stored rows' start, end and tier; else a sentence naming the first row that differs."""
    for number in range(1, max(len(stored), len(code_rows)) + 1):
        mine = _row_key(stored[number - 1]) if number <= len(stored) else None
        theirs = _row_key(code_rows[number - 1]) if number <= len(code_rows) else None
        if mine == theirs:
            continue
        if mine is None:
            return f"row {number} is not on the sheet but code cuts {_row_text(theirs)}"  # type: ignore[arg-type]
        if theirs is None:
            return f"row {number} ({_row_text(mine)}) is not a row of code's answer"
        return f"row {number} is {_row_text(mine)} on the sheet but code cuts {_row_text(theirs)}"
    return None


def _finalise_entry(entry: SheetEntry, example: Example | None, root: Path) -> list[Label]:
    """The new labels of one entry's labelable rows. Raises EntryProblem (one line) for anything that stops it."""
    data = entry.data
    if example is None or example.path != "assess":
        raise EntryProblem("it is not an assess example in the examples file")
    try:
        actions = dan_actions(data)
        built = _assess_example(example, root / FIXTURES_DIR / example.case, readings_optional=True)
    except ValueError as exc:
        raise EntryProblem(str(exc)) from None
    except SheetError as exc:
        raise EntryProblem(str(exc).removeprefix(f"example {example.id}: ")) from None
    problem = edge_problem(data["rows"], built.rows())
    if problem:
        raise EntryProblem(f"code's rows no longer match the stored rows: {problem}")
    labels = recombine(built, actions)
    if len(labels) != len(_stored_labels(data)):  # edge_problem has compared every row, so this is a bug
        raise EntryProblem("code and the sheet disagree on the number of labelable rows")
    return [(verdict.value, flag) for verdict, flag in labels]


def _without_labels(data: Any) -> Any:
    """An entry's data with the stamp and every labelable row's verdict and flag taken out: what a finalise must leave as it was."""
    if not isinstance(data, dict):
        return data
    rows = [r if "tier" in r else {k: v for k, v in r.items() if k not in ("verdict", "approaching_limits")} for r in data["rows"]]
    return {k: (rows if k == "rows" else v) for k, v in data.items() if k != STAMP_KEY}


def _check_result(old: Sequence[SheetEntry], new: Sequence[SheetEntry], done: Mapping[int, tuple[list[Label], Mapping[str, Any]]]) -> None:
    """Parse the new text's entries against the old: equal except the labels and stamps just written, which must be as planned."""
    if len(old) != len(new):
        raise SheetError("the finalised sheet would not have the same entries; nothing written")
    for before, after in zip(old, new, strict=True):
        if before.position in done:
            labels, stamp = done[before.position]
            if _without_labels(before.data) != _without_labels(after.data) or _stored_labels(after.data) != list(labels) or after.data.get(STAMP_KEY) != stamp:
                raise SheetError(f"example {before.id}: the finalised entry would differ from the old one beyond its labels and stamp; nothing written")
        elif before.data != after.data:
            raise SheetError(f"example {before.id}: an entry that was not finalised would change; nothing written")


def _write(path: Path, text: str) -> None:
    """Replace the file with `text` through a temporary file beside it (same mode). The only write finalise makes."""
    mode = path.stat().st_mode & 0o777
    handle, temp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(handle, "wb") as out:
            out.write(text.encode("utf-8"))
        os.chmod(temp, mode)
        os.replace(temp, path)
    finally:
        Path(temp).unlink(missing_ok=True)


def finalise_sheet(sheet: Path, examples_path: Path, root: Path, *, code_root: Path | None = None) -> tuple[Report, str | None]:
    """Finalise every `actions_done: true` entry of the sheet. Returns the report; writes the sheet only when its text changes.

    An entry that fails is an error line in the report and is left exactly as it is; the others are still finalised and written.
    Raises SheetError, with nothing written, for an unreadable or invalid sheet or examples file, or a done entry whose case folder is missing.
    """
    text, entries = read_sheet(sheet)
    examples = {e.id: e for e in read_examples(examples_path)}
    reports = check_entries(entries, _default_windows(list(examples.values())))
    # A done entry whose rows overlap, leave a gap or miss the window is an E15 entry: it fails alone (below), naming the first row that
    # differs from code's, and the other entries are still finalised. Every other sheet error stops the run before anything is read.
    label_errors = [
        m for entry, report in zip(entries, reports, strict=True) for m in report.errors if not (isinstance(entry.data, dict) and entry.data.get("actions_done") is True and m in report.edge_errors)
    ]
    if label_errors:
        raise SheetError("\n".join(label_errors))
    if "\r" in text:
        raise SheetError(f"{sheet}: has CR line endings; finalise edits LF text only")
    # A done entry whose case folder is missing means the checkout is not the one the sheet was built from: that stops the whole run with
    # one line and nothing written, not that entry alone (the other entries would be written from a repository that is not whole).
    for entry in entries:
        example = examples.get(entry.id or "")
        is_done = isinstance(entry.data, dict) and entry.data.get("actions_done") is True
        if is_done and example is not None and example.path == "assess" and not (root / FIXTURES_DIR / example.case).is_dir():
            raise SheetError(f"example {entry.id}: no case folder {FIXTURES_DIR}/{example.case}/ to finalise its entry from; nothing written")
    nodes = yaml.compose(text, Loader=yaml.SafeLoader)
    code = CodeHashes.read(code_root)
    report = Report()
    edits: list[Edit] = []
    done: dict[int, tuple[list[Label], Mapping[str, Any]]] = {}
    for entry, node, entry_report in zip(entries, nodes.value, reports, strict=True):
        data = entry.data
        if data.get("actions_done") is not True:
            report.skipped += 1
            continue
        try:
            example = examples.get(entry.id or "")
            labels = _finalise_entry(entry, example, root)
            if entry_report.edge_errors:  # code's rows equal the stored ones, but the entry's own window says they should cover more or less
                raise EntryProblem(entry_report.edge_errors[0])
            stamp = stamp_for(data, root / FIXTURES_DIR / example.case, code)  # type: ignore[union-attr]  # _finalise_entry refused a missing example
            entry_changes = entry_edits(text, node, data, labels, stamp)
        except EntryProblem as exc:
            report.failed += 1
            report.errors.append(f"example {entry.id}: {exc}")
            continue
        report.finalised += 1
        edits += entry_changes
        done[entry.position] = (labels, stamp)
        labelable = [row for row in data["rows"] if "tier" not in row]
        for row, old, new in zip(labelable, _stored_labels(data), labels, strict=True):
            report.before[old[0] or UNLABELLED] += 1
            report.after[new[0] or UNLABELLED] += 1
            if old != new:
                report.moved.append(f"{entry.id} {_row_key(row)[0]}–{_row_key(row)[1]}: {_shown(old)} → {_shown(new)}")
    new_text = None
    if edits:
        new_text = apply_edits(text, edits)
        _check_result(entries, parse_sheet(new_text, sheet), done)
        _write(sheet, new_text)
    return report, new_text


def run_finalise(sheet: Path, examples_path: Path, root: Path, *, code_root: Path | None = None) -> int:
    """`--finalise`: print the summary (counts by verdict, each moved row) and any error lines. Exit 0, or 1 when any entry failed or the sheet is invalid."""
    try:
        report, _written = finalise_sheet(sheet, examples_path, root, code_root=code_root)
    except SheetError as exc:
        print(exc, file=sys.stderr)
        return 1
    except OSError as exc:
        print(f"{sheet}: cannot be written ({type(exc).__name__})", file=sys.stderr)
        return 1
    for line in report.summary():
        print(line)
    for line in report.errors:
        print(line, file=sys.stderr)
    return 1 if report.failed else 0
