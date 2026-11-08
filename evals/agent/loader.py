"""The eval loader (5a PR-3.F1, G5, G17, E1): the repo copy of the dataset in, LangSmith examples out. It only reads; it runs nothing.

    load_examples(split="dev")        one record per example of the split: id, inputs, fixtures, reference, metadata
    load_dataset(split="dev")         the same records plus the examples refused (actions_done: false), by id
    upload_dataset(split="dev", client=None)   create or reuse the split's LangSmith dataset and make it match the records, by id
    experiment_target(split="dev")    the split's dataset name and LangSmith project name (fixed; no environment variable changes them)
    check_dataset()                   the `--check` line; raises DatasetError for anything wrong
    make_client()                     the real client, built only when LANGSMITH_API_KEY is set

    python -m evals.agent.loader --check             validate everything, print the counts line
    python -m evals.agent.loader --dry-run           upload dev twice to an in-memory RecordingClient (nothing leaves the machine)
    uv run --env-file .env python -m evals.agent.loader --upload [--split dev]    the real upload; needs LANGSMITH_API_KEY

The files read are <data_dir>/examples.yaml and <data_dir>/verdict_sheet.yaml (default evals/agent/data/), and the case folders under
<root>/fixtures/ for their presence. Nothing is written, in the repository or elsewhere.

Order of the checks, all before any LangSmith call and before any run:
  1. every example is well formed (examples.yaml is a list; ids unique; read_examples' own checks; the fields below), and `split` is dev or held_out;
  2. every example's fixture folder, of both splits, has a case.yaml (E1): one message lists all that do not;
  3. the finalised-sheet check of PR-2.F1 (verdict_sheet_finalise.finalised_problems), its message passed through unchanged;
  4. the verdict sheet loads (verdict_sheet_check.load_sheet), and the counts are exactly 29 dev, 15 held_out and 24 sheet entries.
The fixture check comes before the sheet check on purpose: a finalised stamp is sealed over the case's files, so a missing folder would otherwise
show up as a stale sheet and hide which folders are missing.

A record (G5):
    id          the dataset example id
    inputs      {lat, lon, question, now}: `now` is the frozen ISO UTC text from the file, never the clock; the question is kept byte for byte
    fixtures    {case, drop}: the case folder under fixtures/ and the sources dropped for the example (default [])
    reference   every key of the file's `reference` (judge_notes included when present); for an assess example also the sheet's labels from
                verdict_sheet_check.load_sheet: verdict_by_segment, tier, notam_actions
    metadata    split, case_type, path (= reference.path), failure_source, notam_category, airspace_context, constructed, compound (default
                False), parts (default [])

An assess example whose sheet entry has `actions_done: false` (or a NOTAM still UNREAD_OVER_LZ) is refused by id; the others still load.
In LangSmith an example is {inputs + fixtures, outputs = reference, metadata}: the target function only sees `inputs`, so the case and the
drop list travel inside it under `fixtures`. Its LangSmith id is uuid5 of the dataset example id, so a re-upload finds the same examples.

Splits (G17): dev is the default everywhere; held_out is reached only by passing the literal "held_out". Each split has its own dataset and its
own LangSmith project, both fixed here; no environment variable (LANGSMITH_PROJECT included) is read to choose them.
"""

import argparse
import json
import os
import re
import sys
import uuid
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from evals.agent.import_dataset import DATA_DIR, DatasetError
from evals.agent.recording_client import RecordingClient
from flysafelz.sources import FIXTURES_DIR
from flysafelz.verdict_sheet import DROP_SOURCES, SheetError, read_examples
from flysafelz.verdict_sheet_check import load_sheet
from flysafelz.verdict_sheet_finalise import finalised_problems

REPO = Path(__file__).resolve().parents[2]
EXAMPLES_NAME = "examples.yaml"
SHEET_NAME = "verdict_sheet.yaml"
CASE_FILE = "case.yaml"
API_KEY_VARIABLE = "LANGSMITH_API_KEY"
DEFAULT_SPLIT = "dev"
SHEET_KEY = "sheet"
CHUNK = 25  # examples per create or update call, so an error halfway through an upload leaves a partly written dataset that a rerun completes
ASSESS = "assess"
SHEET_LABELS = ("verdict_by_segment", "tier", "notam_actions")
METADATA_REQUIRED = ("case_type", "failure_source", "notam_category", "airspace_context")  # the key must be there; the value may be null
ID_NAMESPACE = uuid.uuid5(uuid.NAMESPACE_URL, "https://github.com/danrfletcher/flysafelz/evals/agent/dataset")
CASE_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")


@dataclass(frozen=True)
class SplitTarget:
    """Where a split's examples live (dataset) and where its experiments are logged (project)."""

    dataset: str
    project: str


SPLITS: dict[str, SplitTarget] = {
    "dev": SplitTarget(dataset="flysafelz-dev", project="flysafelz-dev"),
    "held_out": SplitTarget(dataset="flysafelz-held-out", project="flysafelz-held-out"),
}
# The counts the dataset must have: examples per split, and verdict sheet entries (one per assess example).
EXPECTED: dict[str, int] = {"dev": 29, "held_out": 15, SHEET_KEY: 24}

Record = dict[str, Any]


def experiment_target(split: str = DEFAULT_SPLIT) -> SplitTarget:
    """The fixed dataset and project of a split. The environment is never read, so LANGSMITH_PROJECT cannot reroute a run."""
    return SPLITS[_split(split)]


def _split(split: str) -> str:
    if split not in SPLITS:
        raise DatasetError(f"unknown split {split!r}: use {' or '.join(SPLITS)}")
    return split


# --- reading the examples ----------------------------------------------------------------------------------------------


def _fail(name: str, problem: str) -> DatasetError:
    return DatasetError(f"example {name}: {problem}")


def _read_text(path: Path) -> str:
    try:
        return path.read_bytes().decode("utf-8")
    except FileNotFoundError:
        raise DatasetError(f"{path.name}: no such file in {path.parent}") from None
    except (OSError, UnicodeDecodeError) as exc:
        raise DatasetError(f"{path.name}: cannot be read ({type(exc).__name__})") from None


def _mapping(raw: Mapping[str, Any], key: str, name: str) -> Mapping[str, Any]:
    value = raw.get(key)
    if not isinstance(value, Mapping):
        raise _fail(name, f"{key} must be a mapping")
    return value


def _number(value: object, field_name: str, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise _fail(name, f"{field_name} must be a number")
    return float(value)


def _drop(value: object, name: str) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise _fail(name, f"fixtures.drop must be a list of {', '.join(DROP_SOURCES)}")
    for item in value:
        if item not in DROP_SOURCES:
            raise _fail(name, f"fixtures.drop has {item!r}; it must be one of {', '.join(DROP_SOURCES)}")
    return list(dict.fromkeys(value))


def _frozen_now(value: object, name: str) -> str:
    """`inputs.now` as the file has it. It must be text that is a time with a timezone; it is never replaced by the clock."""
    problem = _fail(name, 'inputs.now must be a date and time with a timezone, as quoted text, e.g. "2026-10-31T07:30Z"')
    if not isinstance(value, str):
        raise problem
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        raise problem from None
    if parsed.tzinfo is None:
        raise problem
    return value


def _base_record(raw: Mapping[str, Any], name: str) -> Record:
    """One example of the file as a record, without the sheet's labels. Raises DatasetError naming the example."""
    inputs, fixtures = _mapping(raw, "inputs", name), _mapping(raw, "fixtures", name)
    reference, meta = _mapping(raw, "reference", name), _mapping(raw, "metadata", name)
    question = inputs.get("question")
    if not isinstance(question, str) or not question.strip():
        raise _fail(name, "inputs.question must be text")
    case = fixtures.get("case")
    if not isinstance(case, str) or not CASE_NAME.fullmatch(case):
        raise _fail(name, f"fixtures.case must be a case folder name under {FIXTURES_DIR}/, not a path")
    path = reference.get("path")
    if not isinstance(path, str) or not path.strip():
        raise _fail(name, "reference.path must be text")
    split = meta.get("split")
    if split not in SPLITS:
        raise _fail(name, f"metadata.split is {split!r}; it must be {' or '.join(SPLITS)}")
    for key in METADATA_REQUIRED:
        if key not in meta:
            raise _fail(name, f"metadata.{key} is missing (write ~ for none)")
    if not isinstance(meta.get("constructed"), bool):
        raise _fail(name, "metadata.constructed must be true or false")
    compound, parts = meta.get("compound", False), meta.get("parts", [])
    if not isinstance(compound, bool):
        raise _fail(name, "metadata.compound must be true or false")
    if not isinstance(parts, list):
        raise _fail(name, "metadata.parts must be a list")
    return {
        "id": name,
        "inputs": {
            "lat": _number(inputs.get("lat"), "inputs.lat", name),
            "lon": _number(inputs.get("lon"), "inputs.lon", name),
            "question": question,
            "now": _frozen_now(inputs.get("now"), name),
        },
        "fixtures": {"case": case, "drop": _drop(fixtures.get("drop"), name)},
        "reference": dict(reference),
        "metadata": {
            "split": split,
            "case_type": meta["case_type"],
            "path": path,
            "failure_source": meta["failure_source"],
            "notam_category": meta["notam_category"],
            "airspace_context": meta["airspace_context"],
            "constructed": meta["constructed"],
            "compound": compound,
            "parts": list(parts),
        },
    }


def _read_records(examples_path: Path) -> list[Record]:
    """Every example of examples.yaml, well formed, in file order. Raises DatasetError (one line, naming the example)."""
    try:
        data = yaml.safe_load(_read_text(examples_path))
    except yaml.YAMLError:
        raise DatasetError(f"{examples_path.name}: not valid YAML") from None
    if not isinstance(data, list):
        raise DatasetError(f"{examples_path.name}: must be a list of examples")
    try:
        read_examples(examples_path)  # the shared checks: a mapping each, a unique id, and an assess example's own fields
    except SheetError as exc:
        raise DatasetError(str(exc)) from None
    return [_base_record(item, str(item["id"])) for item in data]


# --- the checks ----------------------------------------------------------------------------------------------------


def _missing_fixtures(records: Sequence[Record], root: Path, split: str) -> str | None:
    """One message for every case folder without a case.yaml, sorted, each with the ids of the `split` examples that use it. None when all are there.

    A folder only the other split uses is named without ids, so a dev load does not list held-out example ids.
    """
    users: dict[str, list[str]] = {}
    for record in records:
        case_ids = users.setdefault(record["fixtures"]["case"], [])
        if record["metadata"]["split"] == split:
            case_ids.append(record["id"])
    missing = []
    for case in sorted(users):
        folder = root / FIXTURES_DIR / case
        if not (folder.is_dir() and (folder / CASE_FILE).is_file()):
            missing.append(f"  {FIXTURES_DIR}/{case}/" + (f" (used by {', '.join(users[case])})" if users[case] else ""))
    if not missing:
        return None
    return "\n".join([f"{len(missing)} fixture folder(s) missing or without {CASE_FILE}:", *missing])


def _stale_check(sheet_path: Path, examples_path: Path, root: Path) -> None:
    """PR-2.F1's check: the sheet must be finalised for the committed code and fixtures. Its message goes through unchanged."""
    try:
        problems = finalised_problems(sheet_path, root, examples_path)
    except SheetError as exc:
        raise DatasetError(str(exc)) from None
    if problems:
        raise DatasetError("\n".join(dict.fromkeys(problems.values())))


def _count_check(counts: Mapping[str, int], expected: Mapping[str, int], unlabelled: Sequence[str]) -> None:
    """Raise one DatasetError for wrong counts (expected vs found) and for assess examples with no sheet entry (by id).

    The two are reported together: a missing sheet entry also shows as a wrong sheet count, and the ids are what the fix needs.
    """
    problems = []
    wrong = [f"{name} expected {expected[name]}, found {counts[name]}" for name in expected if counts[name] != expected[name]]
    if wrong:
        problems.append("dataset counts are wrong: " + "; ".join(wrong))
    if unlabelled:
        problems.append(f"assess example(s) with no verdict sheet entry: {', '.join(unlabelled)}")
    if problems:
        raise DatasetError("; ".join(problems))


@dataclass
class _Checked:
    records: list[Record]  # every example, both splits, without the sheet's labels
    sheet: dict[str, dict[str, Any]]
    counts: dict[str, int]
    cases: int


def _checked(data_dir: Path, root: Path, expected: Mapping[str, int], split: str) -> _Checked:
    examples_path, sheet_path = Path(data_dir) / EXAMPLES_NAME, Path(data_dir) / SHEET_NAME
    records = _read_records(examples_path)
    problem = _missing_fixtures(records, Path(root), split)
    if problem:
        raise DatasetError(problem)
    _stale_check(sheet_path, examples_path, Path(root))
    try:
        sheet = load_sheet(sheet_path)
    except SheetError as exc:
        raise DatasetError(str(exc)) from None
    found = Counter(r["metadata"]["split"] for r in records)
    counts = {name: found[name] for name in SPLITS} | {SHEET_KEY: len(sheet)}
    unlabelled = [r["id"] for r in records if r["metadata"]["path"] == ASSESS and r["id"] not in sheet]
    _count_check(counts, expected, unlabelled)
    return _Checked(records, sheet, counts, len({r["fixtures"]["case"] for r in records}))


# --- loading ---------------------------------------------------------------------------------------------------------


@dataclass
class Dataset:
    """A split as loaded: its records, and the assess examples refused (id -> why)."""

    split: str
    records: list[Record]
    refused: dict[str, str] = field(default_factory=dict)


def _join(checked: _Checked, split: str) -> Dataset:
    """The split's records with the sheet's labels joined in. _checked has already refused any assess example with no sheet entry."""
    loaded = Dataset(split, [])
    for record in checked.records:
        if record["metadata"]["split"] != split:
            continue
        if record["metadata"]["path"] == ASSESS:
            labels = checked.sheet[record["id"]]
            if "notam_actions" not in labels:
                loaded.refused[record["id"]] = "its verdict sheet entry has actions_done: false (or a NOTAM still UNREAD_OVER_LZ)"
                continue
            clash = [key for key in SHEET_LABELS if key in record["reference"]]
            if clash:
                raise _fail(record["id"], f"reference already has {', '.join(clash)}, which come from the verdict sheet")
            record["reference"] = {**record["reference"], **{key: _plain(labels[key]) for key in SHEET_LABELS}}
        loaded.records.append(record)
    return loaded


def load_dataset(split: str = DEFAULT_SPLIT, *, data_dir: Path = DATA_DIR, root: Path = REPO, expected: Mapping[str, int] = EXPECTED) -> Dataset:
    """The split's records with the sheet's labels joined in, after every check in the module doc. Needs no LangSmith key."""
    split = _split(split)
    return _join(_checked(data_dir, root, expected, split), split)


def _plain(value: Any) -> Any:
    """The sheet's labels as plain dicts and lists (NotamActions is a dict subclass), ready for JSON."""
    if isinstance(value, Mapping):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_plain(v) for v in value]
    return value


def load_examples(split: str = DEFAULT_SPLIT, *, data_dir: Path = DATA_DIR, root: Path = REPO, expected: Mapping[str, int] = EXPECTED) -> list[Record]:
    """The records of a split (dev unless asked for held_out by name). Examples refused by the sheet are left out: see load_dataset."""
    return load_dataset(split, data_dir=data_dir, root=root, expected=expected).records


def check_dataset(*, data_dir: Path = DATA_DIR, root: Path = REPO, expected: Mapping[str, int] = EXPECTED) -> str:
    """Run every check of the loader over both splits and return the line `--check` prints. Raises DatasetError. No example is shown."""
    checked = _checked(data_dir, root, expected, DEFAULT_SPLIT)
    refused = sum(len(_join(checked, name).refused) for name in SPLITS)
    line = f"dev {checked.counts['dev']} · held_out {checked.counts['held_out']} · sheet {checked.counts[SHEET_KEY]} · fixtures {checked.cases}/{checked.cases} present · sheet finalised"
    return line + (f" · {refused} refused (actions_done false)" if refused else "")


# --- uploading ----------------------------------------------------------------------------------------------------


def example_uuid(example_id: str) -> uuid.UUID:
    """The LangSmith id of a dataset example: the same id every time, so a re-upload finds the example it made."""
    return uuid.uuid5(ID_NAMESPACE, example_id)


def _json(value: Any) -> Any:
    return json.loads(json.dumps(value))


def _langsmith_example(record: Record) -> dict[str, Any]:
    return {
        "id": example_uuid(record["id"]),
        "inputs": _json({**record["inputs"], "fixtures": record["fixtures"]}),
        "outputs": _json(record["reference"]),
        "metadata": _json(record["metadata"]),
    }


def _same(wanted: Mapping[str, Any], remote: Any) -> bool:
    """Whether a stored example already has what is wanted. Inputs and outputs must be equal; every wanted metadata key must be there and equal."""
    meta = _json(remote.metadata or {})
    return _json(remote.inputs or {}) == wanted["inputs"] and _json(remote.outputs or {}) == wanted["outputs"] and all(key in meta and meta[key] == value for key, value in wanted["metadata"].items())


@dataclass(frozen=True)
class UploadSummary:
    dataset: str
    total: int
    created: int = 0
    updated: int = 0
    deleted: int = 0
    unchanged: int = 0

    def line(self) -> str:
        return f"{self.dataset}: {self.total} examples ({self.created} created, {self.updated} updated, {self.deleted} deleted, {self.unchanged} unchanged)"


def make_client(environ: Mapping[str, str] | None = None, factory: Callable[..., Any] | None = None) -> Any:
    """The real LangSmith client. Raises DatasetError naming LANGSMITH_API_KEY, before anything is built or sent, when it is not set."""
    env = os.environ if environ is None else environ
    key = env.get(API_KEY_VARIABLE, "").strip()
    if not key:
        raise DatasetError(f"{API_KEY_VARIABLE} is not set: it is needed to upload to LangSmith; nothing was sent")
    if factory is None:
        from langsmith import Client

        factory = Client
    return factory(api_key=key)


def _call(dataset: str, what: str, call: Callable[..., Any], **kwargs: Any) -> Any:
    """One LangSmith call; any error is raised as a DatasetError naming the dataset and the step."""
    try:
        return call(**kwargs)
    except Exception as exc:
        raise DatasetError(f"{dataset}: LangSmith error while {what}: {type(exc).__name__}: {exc}") from exc


def _chunks(items: Sequence[Any]) -> list[Sequence[Any]]:
    return [items[i : i + CHUNK] for i in range(0, len(items), CHUNK)]


def upload_dataset(
    split: str = DEFAULT_SPLIT,
    client: Any = None,
    *,
    data_dir: Path = DATA_DIR,
    root: Path = REPO,
    expected: Mapping[str, int] = EXPECTED,
    environ: Mapping[str, str] | None = None,
) -> UploadSummary:
    """Create or reuse the split's dataset and make its examples match the records, by id; print and return the summary.

    The records are loaded and checked first, then the client is made (`client=None` builds the real one and needs LANGSMITH_API_KEY), so a data
    problem or a missing key stops the upload before a single call. Examples are created, then changed ones updated, then the ones that left the
    file deleted (a refused example is neither written nor deleted). An error halfway is a DatasetError naming the dataset; the dataset is then
    partly written, and a rerun finds what is already there and does the rest, ending in the same state as an upload that never failed.
    """
    loaded = load_dataset(split, data_dir=data_dir, root=root, expected=expected)
    name = experiment_target(loaded.split).dataset
    client = make_client(environ) if client is None else client
    if _call(name, "looking for the dataset", client.has_dataset, dataset_name=name):
        dataset = _call(name, "reading the dataset", client.read_dataset, dataset_name=name)
    else:
        dataset = _call(name, "creating the dataset", client.create_dataset, dataset_name=name, description=f"FlySafeLZ eval examples, split {loaded.split}")
    wanted = {example_uuid(r["id"]): _langsmith_example(r) for r in loaded.records}
    kept = {example_uuid(i) for i in loaded.refused}
    remote = {e.id: e for e in _call(name, "listing the examples", client.list_examples, dataset_id=dataset.id)}
    new = [item for key, item in wanted.items() if key not in remote]
    changed = [item for key, item in wanted.items() if key in remote and not _same(item, remote[key])]
    gone = [key for key in remote if key not in wanted and key not in kept]
    for chunk in _chunks(new):
        _call(name, "creating examples", client.create_examples, dataset_id=dataset.id, examples=list(chunk))
    for chunk in _chunks(changed):
        _call(name, "updating examples", client.update_examples, dataset_id=dataset.id, updates=list(chunk))
    if gone:
        _call(name, "deleting examples", client.delete_examples, example_ids=gone)
    summary = UploadSummary(name, len(wanted), len(new), len(changed), len(gone), len(wanted) - len(new) - len(changed))
    print(summary.line())
    for example_id, why in loaded.refused.items():
        print(f"refused {example_id}: {why}")
    return summary


# --- the command ----------------------------------------------------------------------------------------------------


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m evals.agent.loader", description="Check the eval dataset, or upload a split to LangSmith.")
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--check", action="store_true", help="validate the dataset, the sheet and the fixtures; print the counts")
    action.add_argument("--dry-run", action="store_true", help="upload twice to an in-memory client; nothing is sent")
    action.add_argument("--upload", action="store_true", help=f"upload to LangSmith (needs {API_KEY_VARIABLE})")
    parser.add_argument("--split", default=DEFAULT_SPLIT, choices=list(SPLITS), help="the split to upload; dev unless held_out is named")
    return parser


def main(argv: Sequence[str] | None = None, *, environ: Mapping[str, str] | None = None, data_dir: Path = DATA_DIR, root: Path = REPO) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.check:
            print(check_dataset(data_dir=data_dir, root=root))
        elif args.dry_run:
            client = RecordingClient()
            print(f"dry run: {args.split} uploaded twice to an in-memory RecordingClient; nothing is sent to LangSmith")
            for number in (1, 2):
                before = len(client.writes())
                print(f"run {number}:")
                upload_dataset(args.split, client, data_dir=data_dir, root=root)
                print(f"run {number}: {len(client.writes()) - before} write call(s)")
        else:
            upload_dataset(args.split, None, data_dir=data_dir, root=root, environ=environ)
    except DatasetError as exc:
        print(f"evals.agent.loader: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
