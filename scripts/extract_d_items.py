"""Rebuild tests/fixtures/schedule/d_items_<date>.txt from the public NOTAM archive.

One-off script, not used by the tests (they read only the committed text file).
It needs no key. Run it with:

    uv run python scripts/extract_d_items.py            # downloads the archive PIB
    uv run python scripts/extract_d_items.py PIB.xml    # or reads a local copy

The output has a header of `#` lines naming the archive SHA, then one distinct
item D string per line, sorted, with real newlines written as the two
characters backslash-n.
"""

import sys
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

ARCHIVE_SHA = "bdab14658f0270dc41460eebcd03332abfb230ec"
ARCHIVE_URL = f"https://raw.githubusercontent.com/Jonty/uk-notam-archive/{ARCHIVE_SHA}/data/PIB.xml"
OUT = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "schedule" / "d_items_2026-10-31.txt"


def d_items(xml_bytes: bytes) -> list[str]:
    """Every ItemD text in the PIB, in document order, one entry per NOTAM that has one."""
    root = ET.fromstring(xml_bytes)
    return [e.text for e in root.iter("ItemD") if e.text and e.text.strip()]


def escape(text: str) -> str:
    if "\\" in text:
        raise ValueError(f"item D contains a backslash, so the \\n escape is ambiguous: {text!r}")
    return text.replace("\r\n", "\n").replace("\n", "\\n")


def main(argv: list[str]) -> int:
    if argv:
        xml_bytes = Path(argv[0]).read_bytes()
    else:
        with urllib.request.urlopen(ARCHIVE_URL, timeout=60) as response:
            xml_bytes = response.read()
    items = d_items(xml_bytes)
    distinct = sorted({escape(i) for i in items})
    header = [
        "# Distinct NOTAM item D strings from the NATS PIB archive (public, no key).",
        f"# Archive: github.com/Jonty/uk-notam-archive, commit {ARCHIVE_SHA} (PIB issued 2026-10-31T07:00:37Z).",
        f"# {len(items)} D items, {len(distinct)} distinct. One per line; a newline inside a string is written \\n.",
        "# Rebuild: uv run python scripts/extract_d_items.py",
    ]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join([*header, *distinct]) + "\n", encoding="utf-8")
    print(f"{len(items)} D items, {len(distinct)} distinct -> {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
