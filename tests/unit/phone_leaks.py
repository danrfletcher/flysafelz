"""F3: the list of real phone numbers in the committed PIBs, built WITHOUT the redactor.

If this used `redact_phones` to find the numbers, a number the redactor misses would be missing from the list too, and the
leak test would pass on exactly the failure it exists to catch. So the list comes from a different, blunter rule: every run
of 10 or 11 digits in a raw E text once all whitespace (spaces and line breaks) is removed.

A date-time group that parses as YYMMDDhhmm (`2610311200`) is excluded: it is ten digits but a date, and the NOTAMs write
them in the open. Every other run is kept, so the list holds some runs that are not phone numbers; none is in the sheet.
"""

import re
from datetime import datetime
from pathlib import Path

from flysafelz.pib import read_notams

ROOT = Path(__file__).resolve().parents[2]
PIBS = ROOT / "evals" / "reader" / "pibs"
DIGIT_RUN = re.compile(r"\d{10,11}")
DATE_TIME_GROUP = "%y%m%d%H%M"  # YYMMDDhhmm


def is_date_time_group(digits: str) -> bool:
    try:
        datetime.strptime(digits, DATE_TIME_GROUP)
    except ValueError:
        return False
    return len(digits) == len("YYMMDDhhmm")


def digit_runs(text: str) -> list[str]:
    """Every 10-11 digit run in the text with whitespace removed, date-time groups excepted."""
    return [run for run in DIGIT_RUN.findall(re.sub(r"\s+", "", text)) if not is_date_time_group(run)]


def leak_list(pibs: Path = PIBS) -> dict[str, set[str]]:
    """{digits: {NOTAM ids it was found in}} over every raw E text of the committed PIBs."""
    found: dict[str, set[str]] = {}
    for path in sorted(pibs.glob("*.xml")):
        for notam in read_notams(path.read_bytes())[1]:
            for run in digit_runs(notam.item_e):
                found.setdefault(run, set()).add(notam.id)
    return found
