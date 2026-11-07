"""Build the verdict labelling sheet from Dan's examples file (blind mode). All the logic is in flysafelz.verdict_sheet.

    uv run python scripts/build_verdict_sheet.py --blind --examples examples.yaml --out verdict_sheet.yaml

`--examples` and `--out` may be anywhere. The repository root (where fixtures/ is) is found from this
file's own location, never from the working directory. Writes a new sheet only; an existing `--out` is refused.
"""

import sys
from pathlib import Path

from flysafelz.verdict_sheet import main

ROOT = Path(__file__).resolve().parents[1]

if __name__ == "__main__":
    sys.exit(main(root=ROOT))
