"""The equals-a-rules-value scan shared by test_fence.py and test_helpers_fence.py (F6).

A numeric literal that equals a rules.yaml value is a hard-coded threshold, with two exemptions:

- structural 0 and 1 (`1 - WGS84_E2`, a loop start): `reader_area_slack_nm` is 1.0, so without this every
  `1` in the code would flag;
- a reviewed named constant in geo.py (METRES_PER_KM = 1000, MIN_POLYGON_POINTS = 3). They are a unit and a
  structure, not thresholds, and happen to equal `cape_no_go` and `pib_stale_hours`. The skip is for that
  assignment in geo.py only: the same literal anywhere else, or under another name in geo.py, still flags.
"""

import ast
import re
from pathlib import Path

STRUCTURAL = (0, 1)

# F7/F2: no registry, DI, provider class or memoising decorator anywhere in src/. The word `cache` is banned too, except in
# the reader package (ticket 3 F2): G9's SQLite cache is there, and the CLI says "stored reading", so cli.py still passes.
BANNED_EVERYWHERE = re.compile(r"lru_cache|functools\.cache|registry|Registry|\binject\b|Provider\b")
BANNED_OUTSIDE_THE_READER = re.compile(r"\bcache\b")


def banned_words(path: Path, reader: Path) -> list[str]:
    """The banned words in a source file. `reader` is the one directory where the word `cache` is allowed."""
    text = path.read_text(encoding="utf-8")
    pattern = BANNED_EVERYWHERE if reader in path.parents else re.compile(f"{BANNED_EVERYWHERE.pattern}|{BANNED_OUTSIDE_THE_READER.pattern}")
    return [match.group(0) for match in pattern.finditer(text)]


# F10/F2: no LLM env variable, READER_* name or /models path anywhere in src/, except the reader package (ticket 3 F2). OPENAIP_API_KEY is the
# openAIP key, not an LLM variable.
LLM_ENV_PATTERN = re.compile(r"OPENAI(?!P_API_KEY)|ANTHROPIC|LLM_|MODEL_|READER_|/models")


def llm_env_words(path: Path, reader: Path) -> list[str]:
    """The LLM env variables, READER_ names and /models paths in a source file. `reader` is the one directory where they are allowed."""
    if reader in path.parents:
        return []
    return [match.group(0) for match in LLM_ENV_PATTERN.finditer(path.read_text(encoding="utf-8"))]


# F2: environment and `.env` reads. The reader's config.py is the only place that reads READER_* (and the only reader file that may
# touch the environment). The openAIP key is read by the capture path before this ticket; those files are reviewed here, each
# with its reason, and a new file that reads the environment fails until it is added.
ENV_READ = re.compile(r"environ|getenv|\.env\b|dotenv")
OPENAIP_KEY_READERS = {
    "capture.py": "reads OPENAIP_API_KEY from the environment or .env (find_key)",
    "sources.py": "passes find_key the environment for the openAIP key",
    "verdict_sheet.py": "passes the environment on to capture for the openAIP key",
    "clients/openaip.py": "its docstring says the caller reads OPENAIP_API_KEY; the module reads none",
    "clients/pib_source.py": "its docstring says there is no .env; the module reads none",
}
READER_CONFIG = "reader/config.py"


def env_read_files(src: Path, files: list[Path]) -> list[str]:
    """The files (as paths under `src`) whose text matches an environment or `.env` read, less the reviewed openAIP key readers and reader/config.py."""
    found = []
    for path in files:
        relative = path.relative_to(src).as_posix()
        if relative != READER_CONFIG and relative not in OPENAIP_KEY_READERS and ENV_READ.search(path.read_text(encoding="utf-8")):
            found.append(relative)
    return found


# The numeric literals the reader's modules may hold besides 0 and 1, per module, each with its reason (F6). None is a
# threshold. This is the scan that catches a planted `slack = 1.5`: 1.5 is not a rules.yaml value, so the equals-a-rules-value
# scan cannot see it (and reader_area_slack_nm = 1.0 is exempt as structural 1). A new literal fails until it is added here.
READER_REVIEWED_LITERALS = {
    "redact.py": {},  # the digit counts are written in its patterns, not as numbers
    "notam_area.py": {
        60: "minutes per degree and seconds per minute (unit): MINUTES_PER_DEGREE, SECONDS_PER_MINUTE",
        20_000: "memory bound on the remembered areas (_REMEMBER_LIMIT), more than the archive's distinct NOTAMs; not a threshold",
    },
    "reader/cache.py": {},  # the cache (G9) and its read: structural 0 and 1 only
    "reader/cached_read.py": {},
    "reader/config.py": {},  # F6: the reader's package holds structural 0 and 1 only; the knobs are in settings.yaml
    "reader/reader_set.py": {},
    "reader/settings.py": {},
    "build_reader_items.py": {
        40: "how many NOTAMs the sheet holds (structure): ITEM_COUNT, the selection's fixed size",
        8: "characters of a commit sha shown in an error message (display)",
    },
}
# Every float literal in src/flysafelz other than 0.0 and 1.0, keyed by path under src/flysafelz, each with its reason (F6).
# This is the scan that catches a planted `slack = 1.5` in ANY module: 1.5 is not a rules.yaml value, so the
# equals-a-rules-value scan cannot see it. A new float literal fails until it is added here with a reason.
REVIEWED_FLOATS = {
    "geo.py": {
        6378137.0: "WGS84 equatorial radius in metres (geodesy constant, WGS84_A)",
        0.00669437999014: "WGS84 first eccentricity squared (geodesy constant, WGS84_E2)",
        1e-06: "ON_EDGE_M: float-noise tolerance for 'on the edge' (1e-6 m), not a flight threshold",
        0.5: "half: a mean of two latitudes, half a compass sector, half of pi (arithmetic)",
    },
    "airspace.py": {0.5: "half: the midpoint of a bounding box (arithmetic)"},
    "notam_action.py": {0.5: "half of pi: the latitude bound of the circle sanity check (arithmetic)"},
    "clients/openaip.py": {3.280839895: "feet per metre (unit conversion, FEET_PER_METRE)"},
}
REVIEWED_NAMED_CONSTANTS = {"METRES_PER_KM": 1000, "MIN_POLYGON_POINTS": 3}
GEO = "geo.py"


def number(node: ast.AST) -> float | None:
    """The value of a numeric literal (a bool is not one), else None."""
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
        return node.value
    return None


def _reviewed_constant_nodes(path: Path, tree: ast.Module) -> set[int]:
    """The ids of the literal nodes of a reviewed named constant, when the file is geo.py."""
    if path.name != GEO:
        return set()
    skipped = set()
    for stmt in tree.body:
        if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1 and isinstance(stmt.targets[0], ast.Name):
            name, value = stmt.targets[0].id, stmt.value
        elif isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name) and stmt.value is not None:
            name, value = stmt.target.id, stmt.value
        else:
            continue
        if name in REVIEWED_NAMED_CONSTANTS and number(value) == REVIEWED_NAMED_CONSTANTS[name]:
            skipped.add(id(value))
    return skipped


def hard_coded_rule_values(path: Path, tree: ast.Module, rule_values: set[float]) -> list[tuple[int, float]]:
    """(line, value) for every literal that equals a rules value, less the two exemptions above."""
    skipped = _reviewed_constant_nodes(path, tree)
    found = []
    for node in ast.walk(tree):
        value = number(node)  # a negated literal is found through its operand: the scan compares abs(value)
        if value is None or id(node) in skipped or abs(value) in STRUCTURAL:
            continue
        if abs(value) in rule_values:
            found.append((node.lineno, value))
    return found


def unreviewed_literals(path: Path, tree: ast.Module) -> list[tuple[int, float]]:
    """(line, value) for every numeric literal that is not 0, 1 or reviewed for this file in READER_REVIEWED_LITERALS."""
    reviewed = READER_REVIEWED_LITERALS.get(path.name, {})
    found = []
    for node in ast.walk(tree):
        value = number(node)
        if value is not None and abs(value) not in STRUCTURAL and abs(value) not in reviewed:
            found.append((node.lineno, value))
    return found


def unreviewed_floats(relative: str, tree: ast.Module) -> list[tuple[int, float]]:
    """(line, value) for every float literal that is not 0.0, 1.0 or reviewed for this module in REVIEWED_FLOATS.

    `relative` is the module's path under src/flysafelz, with forward slashes (`clients/openaip.py`).
    """
    reviewed = REVIEWED_FLOATS.get(relative, {})
    found = []
    for node in ast.walk(tree):
        value = number(node)
        if isinstance(value, float) and abs(value) not in (0.0, 1.0) and abs(value) not in reviewed:
            found.append((node.lineno, value))
    return found
