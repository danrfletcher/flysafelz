"""Build the verdict labelling sheet from Dan's examples file. All the logic is in flysafelz.verdict_sheet.

    uv run python scripts/build_verdict_sheet.py --examples examples.yaml --out verdict_sheet.yaml

An example marked `metadata.blind: true` gets a blind entry; any other gets a draft entry (code's verdicts and
NOTAM actions, to correct). `--blind` or `--draft` builds only that mode's examples.

`--examples` and `--out` may be anywhere. The repository root (where fixtures/ is) is found from this
file's own location, never from the working directory.

A rerun on an existing `--out` only appends the entries of new examples (text appended; nothing already in the file
changes) and reports any entry whose example has changed as stale, leaving it alone. `--check` validates the sheet and
prints the counts per example; it exits 1 when it finds an error. `flysafelz.verdict_sheet.load_sheet(path)` reads a
labelled sheet into data for scoring.
"""

import sys
from pathlib import Path

from flysafelz.verdict_sheet import main

ROOT = Path(__file__).resolve().parents[1]

if __name__ == "__main__":
    sys.exit(main(root=ROOT))
