"""Reading and checking shasum lists (5a PR-1.S1): the dataset's dataset.sha256 and the labelled fixtures' labelled-fixtures.sha256.

    read_sums(path)                              {relative path: lower-case sha256 hex} from a `shasum -a 256` list
    verify(root, sums, exceptions)               every mismatch as (relative path, kind): missing, differs or unlisted
    stale_exceptions(root, sums, exceptions)     the excepted paths that match their checksum, or that the list does not hold
    read_exceptions(path)                        {relative path: reason} from the named exception list
    compare_fixtures(...)                        the fixture mismatches against the sums and the sheet fingerprints, one line each
    sha256_file(path)                            the hex digest of a file's bytes

Nothing here writes a file or touches the network.
"""

import hashlib
import re
from collections.abc import Collection, Mapping
from pathlib import Path, PurePosixPath

import yaml

MISSING = "missing"  # listed in the sums, absent on disk
DIFFERS = "differs"  # on disk, but its bytes do not match the sums
UNREADABLE = "unreadable"  # on disk, but it cannot be opened for reading
UNLISTED = "unlisted"  # in a case folder the sums describe, but with no sums line
STALE_EXCEPTION = "stale_exception"  # an exception entry for a file that matches its checksum (or that the sums do not list)

_LINE = re.compile(r"(?P<digest>[0-9a-fA-F]{64}) [ *](?P<path>.+)")


class ChecksumError(Exception):
    """A sums or exceptions file that cannot be read: the message names the file (and the line)."""


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _clean(raw: str) -> str | None:
    """`./case/file` and `case/file` are the same path; None for one that leaves its root (absolute, or with `..`)."""
    path = PurePosixPath(raw.removeprefix("./"))
    if path.is_absolute() or ".." in path.parts or not path.parts:
        return None
    return str(path)


def read_sums(path: Path) -> dict[str, str]:
    """Parse shasum's `<hex>  <path>` lines. The path may be `./case/file` or a bare `examples.yaml`; hex may be upper case.

    Blank lines are skipped. An empty list, a malformed line (its number is in the message), a path that leaves the root and a
    path listed twice raise ChecksumError.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        raise ChecksumError(f"{path.name}: the checksum file is missing ({path})") from None
    except (OSError, UnicodeDecodeError) as error:
        raise ChecksumError(f"{path.name}: the checksum file cannot be read ({error})") from None
    sums: dict[str, str] = {}
    for number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        match = _LINE.fullmatch(line.rstrip("\r"))
        relative = None if match is None else _clean(match.group("path"))
        if match is None or relative is None:
            raise ChecksumError(f"{path.name}: line {number} is not '<sha256>  <path>'")
        if relative in sums:
            raise ChecksumError(f"{path.name}: line {number} lists {relative} a second time")
        sums[relative] = match.group("digest").lower()
    if not sums:
        raise ChecksumError(f"{path.name}: the checksum file lists no files")
    return sums


def read_exceptions(path: Path) -> dict[str, str]:
    """The named exception list: a YAML list of `{path, reason}`. A missing file is an error, so is an entry without both."""
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise ChecksumError(f"{path.name}: the exception list is missing ({path})") from None
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as error:
        raise ChecksumError(f"{path.name}: the exception list cannot be read ({error})") from None
    if data is None:
        data = []
    if not isinstance(data, list):
        raise ChecksumError(f"{path.name}: the exception list must be a list of {{path, reason}}")
    found: dict[str, str] = {}
    for number, entry in enumerate(data, start=1):
        ok = isinstance(entry, dict) and set(entry) == {"path", "reason"} and all(isinstance(v, str) and v.strip() for v in entry.values())
        relative = _clean(entry["path"]) if ok else None
        if relative is None:
            raise ChecksumError(f"{path.name}: entry {number} needs a path inside the fixtures and a reason")
        if relative in found:
            raise ChecksumError(f"{path.name}: entry {number} lists {relative} a second time")
        found[relative] = entry["reason"]
    return found


def verify(root: Path, sums: Mapping[str, str], exceptions: Collection[str] = ()) -> list[tuple[str, str]]:
    """Every mismatch between `sums` and the files under `root`, as (relative path, kind), sorted. The full list, not the first.

    missing   a listed file is absent (an excepted file that is absent is still missing)
    differs   a listed file's bytes do not match (not reported for a path in `exceptions`)
    unreadable  a listed file is there but cannot be opened
    unlisted  a file in a folder that the sums describe (the first path part of some listed file) has no sums line.
              Folders the sums do not mention are not scanned, and neither are files at the top of `root`.
    """
    found: list[tuple[str, str]] = []
    for relative, digest in sums.items():
        path = root / relative
        if not path.is_file():
            found.append((relative, MISSING))
            continue
        try:
            matches = sha256_file(path) == digest
        except OSError:
            found.append((relative, UNREADABLE))
            continue
        if not matches and relative not in exceptions:
            found.append((relative, DIFFERS))
    folders = {PurePosixPath(relative).parts[0] for relative in sums if len(PurePosixPath(relative).parts) > 1}
    for folder in sorted(folders):
        base = root / folder
        if not base.is_dir():
            continue
        for path in sorted(p for p in base.rglob("*") if p.is_file()):
            relative = path.relative_to(root).as_posix()
            if relative not in sums:
                found.append((relative, UNLISTED))
    return sorted(found)


def stale_exceptions(root: Path, sums: Mapping[str, str], exceptions: Collection[str]) -> list[str]:
    """The excepted paths that no longer need an exception: the file matches its checksum, or the sums do not list it.

    An excepted file that is absent is not stale: verify reports it as missing.
    """
    stale = []
    for relative in sorted(exceptions):
        path = root / relative
        try:
            matches = path.is_file() and sha256_file(path) == sums.get(relative)
        except OSError:
            matches = False
        if relative not in sums or matches:
            stale.append(relative)
    return stale


def check_labelled(root: Path, sums: Mapping[str, str], exceptions: Collection[str]) -> list[tuple[str, str]]:
    """verify plus the stale exceptions (kind `stale_exception`): everything that makes the labelled folders fail the check."""
    return sorted([*verify(root, sums, exceptions), *((relative, STALE_EXCEPTION) for relative in stale_exceptions(root, sums, exceptions))])


def describe(mismatches: list[tuple[str, str]]) -> str:
    """One line per mismatch for a failure message."""
    return "\n".join(f"  {relative}: {kind}" for relative, kind in mismatches)


def compare_fixtures(
    root: Path,
    sums: Mapping[str, str],
    exceptions: Collection[str],
    drafts: Mapping[str, Mapping[str, str]],
) -> list[str]:
    """The fixture failures against both references, one line each, `<case>/<file>: ...`, saying which reference it fails.

    `sums` is the labelled-fixtures list (with `exceptions` allowed to differ). `drafts` is {case folder: {file name: sha256}}: the
    fixture-file hashes in the sheet fingerprint of a draft entry that uses the case. A file is checked against whichever of
    the two hold it. The line says "differs from the sha256 list", "differs from the sheet fingerprint" or both.
    """
    reasons: dict[str, list[str]] = {}
    for relative, kind in check_labelled(root, sums, exceptions):
        reasons.setdefault(relative, []).append(f"{kind} against the sha256 list")
    for case, wanted in sorted(drafts.items()):
        for name, digest in sorted(wanted.items()):
            path = root / case / name
            try:
                ok = (path.is_file() and sha256_file(path) == digest) or (not path.is_file() and digest == "absent")
            except OSError:
                ok = False
            if not ok:
                reasons.setdefault(f"{case}/{name}", []).append("does not match the sheet fingerprint")
    return [f"{relative}: {' and '.join(why)}" for relative, why in sorted(reasons.items())]
