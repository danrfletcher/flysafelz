"""Copy Dan's labelled case folders into fixtures/ byte for byte, then check all of them (5a PR-1.S1, G4).

    python -m evals.agent.import_fixtures --src <labelled-fixtures folder> [--dest fixtures] [--sums <list>] [--exceptions <list>]

Nothing here captures a case: the bytes come from the labelled source and from nowhere else, and no archive, openAIP or Open-Meteo
call is made. The source is only read. The checks are the committed `labelled-fixtures.sha256` (a verbatim copy of the source's list)
and `labelled_exceptions.yaml` (the named files allowed to differ, each with its reason), both in the destination by default.

Order: check the source against the list (a bad source copies nothing); copy each source folder that is absent from the destination,
file by file with shutil.copyfile into a temporary folder that is then renamed into place; a folder that already exists is never
touched, even when its bytes differ, and is named in the output; then check every folder the list describes. Exit 0 only if that
check is clean: a missing, differing or unlisted file, or a stale exception, exits 1 naming the file.
"""

import argparse
import shutil
import sys
import tempfile
from pathlib import Path

from evals.agent.checksums import STALE_EXCEPTION, ChecksumError, check_labelled, describe, read_exceptions, read_sums, verify

REPO = Path(__file__).resolve().parents[2]
FIXTURES = REPO / "fixtures"
SUMS_NAME = "labelled-fixtures.sha256"
EXCEPTIONS_NAME = "labelled_exceptions.yaml"


class FixtureImportError(Exception):
    """Why the labelled folders cannot be imported or do not check clean; the message names the folder or file."""


def labelled_folders(sums: dict[str, str]) -> list[str]:
    """The case folders the checksum list describes."""
    return sorted({relative.split("/")[0] for relative in sums if "/" in relative})


def copy_folder(src: Path, dest: Path) -> int:
    """Copy one case folder with shutil.copyfile into a temporary folder in `dest`'s parent, then rename it to `dest`. The number of files."""
    files = sorted(p for p in src.rglob("*") if p.is_file())
    temp = Path(tempfile.mkdtemp(prefix=f".{dest.name}.", suffix=".tmp", dir=dest.parent))
    try:
        for path in files:
            target = temp / path.relative_to(src)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
        temp.rename(dest)
    except OSError as error:
        raise FixtureImportError(f"{dest.name}: cannot copy ({error})") from None
    finally:
        if temp.exists():
            shutil.rmtree(temp, ignore_errors=True)
    return len(files)


def import_fixtures(src: Path, dest: Path, sums_path: Path, exceptions_path: Path) -> tuple[list[str], list[str], int]:
    """(copied folders, skipped folders, files checked). Raises FixtureImportError for a bad source or a folder that does not check clean."""
    try:
        sums = read_sums(sums_path)
        exceptions = read_exceptions(exceptions_path)
    except ChecksumError as error:
        raise FixtureImportError(str(error)) from None
    if not src.is_dir():
        raise FixtureImportError(f"--src is not a folder: {src}")
    bad_source = verify(src, sums)
    if bad_source:
        raise FixtureImportError(f"the source does not match {sums_path.name}; nothing was copied:\n{describe(bad_source)}")
    copied: list[str] = []
    skipped: list[str] = []
    dest.mkdir(parents=True, exist_ok=True)
    for name in labelled_folders(sums):
        if (dest / name).exists():
            skipped.append(name)
        else:
            copy_folder(src / name, dest / name)
            copied.append(name)
    problems = check_labelled(dest, sums, exceptions)
    if problems:
        stale = [relative for relative, kind in problems if kind == STALE_EXCEPTION]
        note = f"\n  (a stale exception is an entry in {exceptions_path.name} for a file that matches its checksum: remove it)" if stale else ""
        raise FixtureImportError(f"the fixtures do not match {sums_path.name}:\n{describe(problems)}{note}")
    return copied, skipped, len(sums)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m evals.agent.import_fixtures", description="Copy Dan's labelled case folders into fixtures/ byte for byte, then check all of them.")
    parser.add_argument("--src", type=Path, required=True, help="the labelled-fixtures folder to copy from (read only)")
    parser.add_argument("--dest", type=Path, default=FIXTURES, help="the fixtures folder (default: the repo's fixtures/)")
    parser.add_argument("--sums", type=Path, help=f"default: <dest>/{SUMS_NAME}")
    parser.add_argument("--exceptions", type=Path, help=f"default: <dest>/{EXCEPTIONS_NAME}")
    args = parser.parse_args(argv)
    sums_path = args.sums or args.dest / SUMS_NAME
    exceptions_path = args.exceptions or args.dest / EXCEPTIONS_NAME
    try:
        copied, skipped, files = import_fixtures(args.src, args.dest, sums_path, exceptions_path)
        exceptions = read_exceptions(exceptions_path)
    except (FixtureImportError, ChecksumError) as error:
        print(f"import_fixtures: {error}", file=sys.stderr)
        return 1
    for name in copied:
        print(f"copied: {name}")
    for name in skipped:
        print(f"already in {args.dest.name}/, not touched: {name}")
    print(f"{len(copied)} folders copied, {len(skipped)} already present and left alone")
    print(f"verified {len(copied) + len(skipped)} folders, {files} files against {sums_path.name}; {len(exceptions)} named exception(s):")
    for relative, reason in exceptions.items():
        print(f"  {relative}: {reason}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
