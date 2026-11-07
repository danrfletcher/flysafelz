"""The integration tests share tests/factories.py, which imports the unit tests' helper modules by name."""

import sys
from pathlib import Path

UNIT = str(Path(__file__).resolve().parents[1] / "unit")
if UNIT not in sys.path:
    sys.path.insert(0, UNIT)
