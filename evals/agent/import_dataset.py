"""`make dataset` (5a PR-1.S1, G3): check Dan's three dataset files against dataset.sha256, then copy them into evals/agent/data/.

    DATASET_SRC=<folder> python -m evals.agent.import_dataset

The folder holds examples.yaml, verdict_sheet.yaml and guidance_questions.yaml; the checksums are in `<DATASET_SRC>/../dataset.sha256`.
DATASET_SRC is read from the environment, never from the command line, so the shell never parses a path with spaces or a quote
(as for `make fixtures` and CASE).

Everything is checked before anything is written: any missing file, differing file or bad checksum list exits 1 with a message that
names the file, and evals/agent/data/ is left exactly as it was. Each file is then written to a temporary file in evals/agent/data/
and renamed into place. The source is only ever opened for reading. Finalising the sheet is not done here: the `dataset` target runs
`build_verdict_sheet.py --finalise` after this module (PR-2.F1).
"""

import hashlib
import os
import sys
import tempfile
from collections.abc import Mapping
from pathlib import Path

from evals.agent.checksums import DIFFERS, MISSING, UNREADABLE, ChecksumError, read_sums, verify

SOURCE_VARIABLE = "DATASET_SRC"
SUMS_NAME = "dataset.sha256"
REQUIRED = ("examples.yaml", "verdict_sheet.yaml", "guidance_questions.yaml")
DATA_DIR = Path(__file__).resolve().parent / "data"


class DatasetError(Exception):
    """Why the dataset cannot be imported; the message names the file or setting."""


def source_from(environ: Mapping[str, str]) -> Path:
    value = environ.get(SOURCE_VARIABLE, "")
    if not value.strip():
        raise DatasetError(f"{SOURCE_VARIABLE} is not set: point it at the folder holding {', '.join(REQUIRED)}")
    src = Path(value).expanduser()
    if not src.is_dir():
        raise DatasetError(f"{SOURCE_VARIABLE} is not a folder: {src}")
    return src


def _problem(relative: str, kind: str) -> str:
    if kind == MISSING:
        return f"{relative} is missing"
    if kind == UNREADABLE:
        return f"{relative} cannot be read"
    if kind == DIFFERS:
        return f"{relative} does not match its checksum in {SUMS_NAME}"
    return f"{relative}: {kind}"


def verified_bytes(src: Path) -> dict[str, bytes]:
    """The bytes of the three dataset files, each checked against `<src>/../dataset.sha256`. Reads only. Raises naming every bad file."""
    try:
        sums = read_sums(src.parent / SUMS_NAME)
    except ChecksumError as error:
        raise DatasetError(str(error)) from None
    absent = [name for name in REQUIRED if name not in sums]
    if absent:
        raise DatasetError(f"{SUMS_NAME} does not list {', '.join(absent)}")
    problems = verify(src, sums)
    if problems:
        raise DatasetError("; ".join(_problem(relative, kind) for relative, kind in problems))
    data: dict[str, bytes] = {}
    for name in REQUIRED:
        try:
            data[name] = (src / name).read_bytes()
        except OSError as error:
            raise DatasetError(f"{name} cannot be read: {error}") from None
        if hashlib.sha256(data[name]).hexdigest() != sums[name]:  # the file changed between the check and this read
            raise DatasetError(_problem(name, DIFFERS))
    return data


def write_all(data: Mapping[str, bytes], dest: Path) -> list[tuple[str, bool]]:
    """Write each file to a temporary file beside its place, then rename them all in. (name, changed) per file.

    A file whose bytes are already in place is left alone, so a second run with an unchanged source touches nothing.
    """
    staged: list[tuple[str, Path]] = []
    changed: dict[str, bool] = {}
    try:
        dest.mkdir(parents=True, exist_ok=True)
        for name, body in data.items():
            target = dest / name
            changed[name] = not (target.is_file() and target.read_bytes() == body)
            if not changed[name]:
                continue
            handle, temp = tempfile.mkstemp(prefix=f".{name}.", suffix=".tmp", dir=dest)
            staged.append((name, Path(temp)))
            with os.fdopen(handle, "wb") as out:
                out.write(body)
            os.chmod(temp, 0o644)
        for name, temp in staged:
            os.replace(temp, dest / name)
    except OSError as error:
        raise DatasetError(f"cannot write {dest}: {error}") from None
    finally:
        for _name, temp in staged:
            temp.unlink(missing_ok=True)
    return list(changed.items())


def import_dataset(src: Path, dest: Path = DATA_DIR) -> list[tuple[str, bool]]:
    return write_all(verified_bytes(src), dest)


def main(environ: Mapping[str, str] | None = None, dest: Path = DATA_DIR) -> int:
    try:
        src = source_from(os.environ if environ is None else environ)
        results = import_dataset(src, dest)
    except DatasetError as error:
        print(f"make dataset: {error}", file=sys.stderr)
        return 1
    print(f"verified {', '.join(REQUIRED)} against {SUMS_NAME}")
    for name, changed in results:
        print(f"{'copied' if changed else 'unchanged'}: {name} -> {dest.name}/{name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
