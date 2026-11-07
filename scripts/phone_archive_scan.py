"""Run the phone detector (flysafelz.redact.redact_phones) over the real NOTAM archive and report what it missed (G12).

One-off script, not used by any test (the tests call its functions with a fake HTTP transport and tiny constructed PIBs; none
runs `main`). Run it with:

    uv run python scripts/phone_archive_scan.py
    GITHUB_TOKEN=<token> uv run python scripts/phone_archive_scan.py

What it fetches: for each date D from 2026-07-01 to 2026-10-31 (123 dates) it asks the GitHub API for the latest
Jonty/uk-notam-archive commit that touched data/PIB.xml up to D (23:59:59Z), then downloads that commit's PIB from
raw.githubusercontent.com. Dates that resolve to the same commit share one download. A date with no commit (before the archive
starts) is logged and skipped.

The 123 commit lookups are over GitHub's anonymous limit of 60 an hour. Set GITHUB_TOKEN (any token, no scopes needed, for
example from `gh auth token`) and the lookups are sent with `Authorization: Bearer <token>`. Without one, or when the limit is hit
anyway, the run stops at the first 403 or 429, names the date it reached, and keeps everything already cached; run it again
later and it resumes from that date.

Cache: .cache/pib_archive/ (gitignored, because the archive PIBs hold real phone numbers) holds <sha>.xml for each PIB and
days.json, the map of date to sha (null for a date with no commit). Files are written to a temp name and renamed, so an
interrupted download leaves no partial file. A rerun with a full cache makes no request at all. A cached PIB that does not parse
is reported by sha and date and left in place; delete it to fetch it again.

What it prints (stdout; progress and the request count go to stderr, so two runs over one cache give identical stdout):
the date, commit and PIB counts, the distinct NOTAM count (deduped by id, the earliest date wins), the hit count (NOTAMs whose
item E holds a number), how many ids had different E text on different days, then the miss candidates and the over-match
candidates, one per line: NOTAM id, date, a shape mask (every digit becomes #), and the number itself written as [phone].
No real number is printed, logged or written. A candidate is a lead for a person to check, not a verdict.

Exit code: 0 even when it finds misses. 1 on a network failure, a rate limit or a PIB that does not parse, naming the date or sha.
"""

import argparse
import json
import os
import re
import sys
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import IO

import httpx

from flysafelz.pib import PibError, read_notams
from flysafelz.redact import _PHONE, redact_phones

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / ".cache" / "pib_archive"
REPO = "Jonty/uk-notam-archive"
FIRST_DATE = date(2026, 7, 1)
LAST_DATE = date(2026, 10, 31)
COMMITS_URL = f"https://api.github.com/repos/{REPO}/commits"
RAW_URL = f"https://raw.githubusercontent.com/{REPO}/{{sha}}/data/PIB.xml"
RATE_LIMITED = (403, 429)
TIMEOUT_S = 120.0
DAYS_FILE = "days.json"
PHONE = "[phone]"
MISS_MIN_DIGITS = 9


class RateLimited(Exception):
    """GitHub answered 403 or 429. The message names the date the run reached."""


class FetchError(Exception):
    """A request failed for another reason. The message names the date or sha."""


# --- the dates and the requests ----------------------------------------------------------


def date_plan(first: date = FIRST_DATE, last: date = LAST_DATE) -> list[date]:
    """Every date from first to last, both included."""
    return [first + timedelta(days=n) for n in range((last - first).days + 1)]


def commits_url(day: date) -> str:
    """The latest commit touching data/PIB.xml up to the end of the day (UTC)."""
    return f"{COMMITS_URL}?path=data/PIB.xml&until={day.isoformat()}T23:59:59Z&per_page=1"


def raw_url(sha: str) -> str:
    return RAW_URL.format(sha=sha)


def api_headers() -> dict[str, str]:
    """The GitHub API headers. The bearer header is there only when GITHUB_TOKEN is set; it is read on every call and never logged."""
    headers = {"Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28"}
    token = os.environ.get("GITHUB_TOKEN", "").strip()
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def _get(client: httpx.Client, url: str, what: str, *, headers: dict[str, str] | None = None) -> httpx.Response:
    """GET, with a rate limit as RateLimited and any other failure as FetchError. `what` names the date or sha."""
    try:
        response = client.get(url, headers=headers, timeout=TIMEOUT_S)
    except httpx.HTTPError as exc:
        raise FetchError(f"{what}: {type(exc).__name__}: {exc}") from exc
    if response.status_code in RATE_LIMITED:
        raise RateLimited(f"{what}: GitHub answered {response.status_code} (rate limit; set GITHUB_TOKEN, or wait and rerun)")
    if response.status_code != httpx.codes.OK:
        raise FetchError(f"{what}: GitHub answered {response.status_code}")
    return response


def lookup_sha(client: httpx.Client, day: date) -> str | None:
    """The sha of the latest archive commit touching data/PIB.xml up to the day, or None when there is none."""
    response = _get(client, commits_url(day), f"commit lookup for {day}", headers=api_headers())
    try:
        commits = response.json()
        return str(commits[0]["sha"]) if commits else None
    except (ValueError, KeyError, TypeError, IndexError) as exc:
        raise FetchError(f"commit lookup for {day}: unexpected answer ({type(exc).__name__})") from exc


def download_pib(client: httpx.Client, sha: str, day: date) -> bytes:
    return _get(client, raw_url(sha), f"PIB {sha[:12]} (first needed for {day})").content


# --- the cache ---------------------------------------------------------------------------


def write_atomically(path: Path, data: bytes) -> None:
    """Write to a temp file in the same folder, then rename: the file is there whole or not at all."""
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temp = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(handle, "wb") as out:
            out.write(data)
        os.replace(temp, path)
    except BaseException:
        Path(temp).unlink(missing_ok=True)
        raise


def load_days(cache: Path) -> dict[date, str | None]:
    path = cache / DAYS_FILE
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        return {date.fromisoformat(day): sha for day, sha in raw.items()}
    except (ValueError, AttributeError) as exc:
        raise FetchError(f"{path} is not a map of date to sha ({exc}); delete it to start the lookups again") from exc


def save_days(cache: Path, days: dict[date, str | None]) -> None:
    text = json.dumps({day.isoformat(): sha for day, sha in sorted(days.items())}, indent=0) + "\n"
    write_atomically(cache / DAYS_FILE, text.encode("utf-8"))


@dataclass
class Fetched:
    """What the fetch step leaves in the cache."""

    days: dict[date, str | None]
    requests: int = 0
    failures: list[str] = field(default_factory=list)


def fetch_archive(client: httpx.Client, dates: Sequence[date], cache: Path, log: IO[str]) -> Fetched:
    """Resolve every date to a sha and cache every PIB not cached yet. A full cache makes no request.

    Raises RateLimited or FetchError, naming the date or sha; what was cached before stays valid, and a rerun resumes.
    A downloaded body that is not a PIB is reported in `failures`, not cached, and the run goes on.
    """
    result = Fetched(days=load_days(cache))
    for day in dates:
        if day not in result.days:
            result.requests += 1
            print(f"{day}: looking up the commit", file=log)
            result.days[day] = lookup_sha(client, day)
            save_days(cache, result.days)
        sha = result.days[day]
        if sha is None:
            print(f"{day}: no archive commit up to this date, skipped", file=log)
            continue
        path = cache / f"{sha}.xml"
        if path.exists():
            continue
        result.requests += 1
        print(f"{day}: downloading PIB {sha[:12]}", file=log)
        body = download_pib(client, sha, day)
        try:
            read_notams(body)
        except PibError as exc:
            result.failures.append(f"{day}: PIB {sha[:12]} downloaded but is not a valid PIB, not cached: {exc}")
            continue
        write_atomically(path, body)
    return result


# --- the scan ----------------------------------------------------------------------------

_BREAK = r"(?:\r?\n)"
# A run of digits joined by spaces, brackets, a leading + (itself maybe inside a bracket: "(+44)") and at most one line break:
# blunter than the detector on purpose.
_RUN = re.compile(rf"\(*\+?[\d(][\d() \t]*(?:{_BREAK}[\d() \t]*)?")
_COORDINATE_AFTER = re.compile(r"\s*(?:\.\d+)?[NSEW](?![A-Za-z])")
_DAY_DIGITS = 2  # a day of the month
_TIME_DIGITS = 4  # hhmm
_DATE_TIME_DIGITS = 10  # YYMMDDhhmm
_LAST_HOUR = 24
_MINUTES_IN_HOUR = 60
_BEFORE_COORDINATE = re.compile(r"\d[NSEW]\s*$")
_DOT_DIGIT = re.compile(r"\.\d")
_AR_BEFORE = re.compile(r"AR-[\d-]*$")
_AFTER_COORDINATE = re.compile(r"[NSEW](?![A-Za-z])")
_NOTAM_ID_BEFORE = re.compile(r"[A-Z]\d{4}/\d{2}[\s,/-]*$")
_NOTAM_ID_AFTER = re.compile(r"[\s,/-]*[A-Z]\d{4}/\d{2}\b")
_CONTEXT = 12


def mask(text: str) -> str:
    """The shape of a run: every digit becomes #, a line break is written \\n, nothing else changes."""
    return re.sub(r"\d", "#", text).replace("\r", "").replace("\n", "\\n")


@dataclass(frozen=True)
class Candidate:
    """A lead: where, the shape, and (kept out of repr, never printed) the text it came from."""

    notam_id: str
    day: date
    text: str = field(repr=False)
    kind: str = ""  # over-match candidates: why the match looks wrong


def _is_time(group: str) -> bool:
    """Four digits that read as hhmm (0000 to 2400)."""
    return len(group) == _TIME_DIGITS and int(group[:2]) <= _LAST_HOUR and int(group[2:]) < _MINUTES_IN_HOUR


def _is_schedule_or_date(groups: list[str]) -> bool:
    """Only day-of-month numbers and times (OCT 01 02 03 0900, 1900 05 0800), or only date-time groups: the exclusions test_redact.py names."""
    if all(len(g) == _DATE_TIME_DIGITS and g.isdigit() for g in groups):
        return True
    return all(g.isdigit() and (len(g) == _DAY_DIGITS or _is_time(g)) for g in groups)


def miss_runs(redacted: str) -> list[str]:
    """The runs left in redacted item E with 9 or more digits that are not a day or time list, a date-time group or a coordinate.

    A group of digits written straight before N, S, E or W (534732N, 0000459.5W) is a coordinate and leaves the run.
    """
    found = []
    for match in _RUN.finditer(redacted):
        run = match.group(0).rstrip()
        groups = [g for g in re.split(r"[\s()+]+", run) if g]
        if _COORDINATE_AFTER.match(redacted, match.start() + len(run)) and groups:
            groups.pop()
        if sum(len(g) for g in groups) >= MISS_MIN_DIGITS and not _is_schedule_or_date(groups):
            found.append(run)
    return found


def over_match_kind(text: str, start: int, end: int) -> str | None:
    """Why a detector match looks like something else (a coordinate, a frequency, an AR or NOTAM reference), or None."""
    before, after = text[max(0, start - _CONTEXT) : start], text[end : end + _CONTEXT]
    if _AFTER_COORDINATE.match(after):
        return "N/S/E/W right after the match"
    if _BEFORE_COORDINATE.search(before):
        return "N/S/E/W right after a digit, before the match"
    if before.endswith(".") or _DOT_DIGIT.match(after):
        return "a . between digits"
    if _AR_BEFORE.search(before):
        return "AR- reference"
    if _NOTAM_ID_BEFORE.search(before) or _NOTAM_ID_AFTER.match(after):
        return "next to a NOTAM id"
    return None


@dataclass
class Scan:
    pibs: int = 0
    distinct: int = 0
    hits: int = 0
    changed_e: int = 0
    misses: list[Candidate] = field(default_factory=list)
    over_matches: list[Candidate] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)


def scan_texts(texts: dict[str, tuple[date, str]]) -> Scan:
    """The detector over item E of each distinct NOTAM: {id: (first date, item E)}. Fills hits, misses and over-matches."""
    scan = Scan(distinct=len(texts))
    for notam_id, (day, item_e) in texts.items():
        redacted, mapping = redact_phones(item_e)
        if mapping:
            scan.hits += 1
        scan.misses += [Candidate(notam_id, day, run) for run in miss_runs(redacted)]
        for match in _PHONE.finditer(item_e):
            kind = over_match_kind(item_e, match.start(), match.end())
            if kind:
                scan.over_matches.append(Candidate(notam_id, day, match.group(0), kind))
    return scan


def scan_archive(days: dict[date, str | None], cache: Path) -> Scan:
    """Read each distinct cached PIB once, in date order, and scan the distinct NOTAMs (the earliest date wins).

    A PIB that is missing or does not parse is put in `failures`, naming the sha and the first date, and the rest carry on.
    """
    texts: dict[str, tuple[date, str]] = {}
    variants: dict[str, set[str]] = {}
    failures: list[str] = []
    seen: set[str] = set()
    for day, sha in sorted(days.items()):
        if sha is None or sha in seen:
            continue
        seen.add(sha)
        path = cache / f"{sha}.xml"
        try:
            notams = read_notams(path.read_bytes())[1]
        except OSError as exc:
            failures.append(f"{day}: cached PIB {sha[:12]} cannot be read ({type(exc).__name__}); run again to fetch it")
            continue
        except PibError as exc:
            failures.append(f"{day}: cached PIB {sha[:12]} does not parse and is skipped ({exc}); delete {path.name} to fetch it again")
            continue
        for notam in notams:
            texts.setdefault(notam.id, (day, notam.item_e))
            variants.setdefault(notam.id, set()).add(" ".join(notam.item_e.split()))
    scan = scan_texts(texts)
    scan.pibs = len(seen) - len(failures)
    scan.changed_e = sum(1 for forms in variants.values() if len(forms) > 1)
    scan.failures = failures
    return scan


# --- the report --------------------------------------------------------------------------


def _candidate_line(candidate: Candidate) -> str:
    """One line: date, id, the number as [phone], its shape. The text of the number is used for the mask only."""
    why = f"  ({candidate.kind})" if candidate.kind else ""
    return f"  {candidate.day}  {candidate.notam_id}  {PHONE}  shape {mask(candidate.text)}{why}"


def _section(title: str, candidates: list[Candidate]) -> list[str]:
    ordered = sorted(candidates, key=lambda c: (c.day, c.notam_id, mask(c.text)))
    return [f"{title}: {len(ordered)}", *(_candidate_line(c) for c in ordered)]


def format_report(scan: Scan, days: dict[date, str | None]) -> str:
    """The report, in a fixed order, so two runs over one cache give the same bytes. `days` is the plan: date to sha or None."""
    resolved = sorted((day, sha) for day, sha in days.items() if sha is not None)
    shas = {sha for _day, sha in resolved}
    lines = [
        f"Phone detector archive scan: {REPO}",
        f"Dates: {len(days)}, with an archive commit: {len(resolved)}, skipped with none: {len(days) - len(resolved)}",
        f"Distinct commits: {len(shas)}" + (f", first {resolved[0][1]} ({resolved[0][0]}), last {resolved[-1][1]} ({resolved[-1][0]})" if resolved else ""),
        f"PIBs read: {scan.pibs}",
        f"Distinct NOTAMs: {scan.distinct}",
        f"Hits (NOTAMs with a phone number in item E): {scan.hits}",
        f"Ids with different E text on different days: {scan.changed_e}",
        *_section("Miss candidates", scan.misses),
        *_section("Over-match candidates", scan.over_matches),
    ]
    return "\n".join(lines) + "\n"


# --- the run -----------------------------------------------------------------------------


def run(client: httpx.Client, cache: Path, dates: Sequence[date], out: IO[str], err: IO[str]) -> int:
    """Fetch, scan and print. 0 even when there are misses; 1 on a rate limit, a failed request or a PIB that does not parse."""
    try:
        fetched = fetch_archive(client, dates, cache, err)
    except RateLimited as exc:
        print(f"stopped: {exc}. Everything fetched so far is cached; run again to resume from this date.", file=err)
        return 1
    except FetchError as exc:
        print(f"error: {exc}", file=err)
        return 1
    print(f"requests made: {fetched.requests}", file=err)
    plan = {day: fetched.days[day] for day in dates}
    scan = scan_archive(plan, cache)
    out.write(format_report(scan, plan))
    problems = [*fetched.failures, *scan.failures]
    for problem in problems:
        print(f"error: {problem}", file=err)
    return 1 if problems else 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the phone detector over the daily archive PIBs, 2026-07-01 to 2026-10-31.")
    parser.add_argument("--cache", type=Path, default=CACHE, help="the cache folder (default .cache/pib_archive/, gitignored)")
    args = parser.parse_args(argv)
    with httpx.Client(follow_redirects=True) as client:
        return run(client, args.cache, date_plan(), sys.stdout, sys.stderr)


if __name__ == "__main__":
    sys.exit(main())
