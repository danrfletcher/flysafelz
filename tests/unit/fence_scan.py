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
# touch the environment); the agent's config.py is the only place that reads AGENT_*. The openAIP key is read by the capture path before this ticket; those files are reviewed here, each
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
AGENT_CONFIG = "agent/config.py"  # reads AGENT_* from os.environ, and nothing else (ticket 4 PR-1.S1)
GUIDANCE_INGEST = "guidance/ingest.py"  # main() reads GUIDANCE_SOURCES, a different register for a test run, and nothing else (ticket 4 PR-7.F1)


def env_read_files(src: Path, files: list[Path]) -> list[str]:
    """The files (as paths under `src`) whose text matches an environment or `.env` read, less the reviewed openAIP key readers, reader/config.py, agent/config.py and guidance/ingest.py."""
    found = []
    for path in files:
        relative = path.relative_to(src).as_posix()
        if relative not in (READER_CONFIG, AGENT_CONFIG, GUIDANCE_INGEST) and relative not in OPENAIP_KEY_READERS and ENV_READ.search(path.read_text(encoding="utf-8")):
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
    "agent/config.py": {},  # F7: the agent's and guidance's modules hold structural 0 and 1 only; the knobs are in agent/settings.yaml
    "agent/settings.py": {},
    "guidance/embedding_model.py": {},
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
REVIEWED_NAMED_CONSTANTS = {"METRES_PER_KM": 1000, "MIN_POLYGON_POINTS": 3, "QUARTER_TURN": 90}
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


# F6: which package may import which framework. The scan is per package, by the file's place under src/flysafelz:
#   agent/      langchain, langchain_core, langchain_openai, langgraph*, langsmith (the model only through langchain_openai)
#   guidance/   fastembed, pypdf, and flysafelz.agent.settings (the knobs, which load without LangChain)
#   reader/     langchain_openai and langchain_core only (ticket 3's exemption, unchanged)
#   elsewhere   none of them, and no flysafelz.agent or flysafelz.guidance
# openai and anthropic are banned in every package.
AGENT_PACKAGE = "agent"
GUIDANCE_PACKAGE = "guidance"
READER_PACKAGE = "reader"
FRAMEWORK_PREFIXES = ("langchain", "langgraph", "langsmith")
GUIDANCE_PREFIXES = ("fastembed", "pypdf")
BANNED_CLIENTS = ("openai", "anthropic")
AGENT_LANGCHAIN_MODULES = ("langchain", "langchain_core", "langchain_openai")
READER_LANGCHAIN_MODULES = ("langchain_openai", "langchain_core")
INTERNAL_PACKAGES = ("flysafelz.agent", "flysafelz.guidance")
GUIDANCE_MAY_IMPORT = "flysafelz.agent.settings"
PROVIDER_SHORTCUT = "init_chat_model"  # langchain's way to build any provider's model from a string: it would bypass langchain_openai


def _top(module: str) -> str:
    return module.split(".")[0]


def _is_or_under(name: str, package: str) -> bool:
    return name == package or name.startswith(package + ".")


def _own_package(path: Path, src: Path) -> list[str] | None:
    """The dotted package a file sits in, as parts (`["flysafelz", "guidance"]` for guidance/index.py), or None for a file outside `src`."""
    try:
        parts = path.relative_to(src).parts
    except ValueError:
        return None
    return [src.name, *parts[:-1]]


def _absolute_module(node: ast.ImportFrom, own_package: list[str] | None) -> str | None:
    """The absolute dotted module an `ImportFrom` names: `from .agent import x` in a top-level module is `flysafelz.agent`, `from . import x`
    is `flysafelz`, `from ..agent.config import x` in guidance/ is `flysafelz.agent.config`. None when it cannot be resolved (a relative
    import in a file outside `src`, or one that climbs past the top package), which Python itself would refuse to run."""
    if node.level == 0:
        return node.module
    if own_package is None or node.level > len(own_package):
        return None
    base = own_package[: len(own_package) - (node.level - 1)]
    return ".".join([*base, node.module] if node.module else base)


def _imports_of(tree: ast.Module, own_package: list[str] | None = None) -> list[tuple[str, list[str]]]:
    """(absolute module, the dotted names it brings in). `from a import b` brings in `a.b`; `import a.b` brings in `a.b`.

    A relative import is resolved against `own_package` first, so `from .agent import config` is checked as `flysafelz.agent.config`.
    """
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found += [(a.name, [a.name]) for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            module = _absolute_module(node, own_package)
            if module:
                found.append((module, [f"{module}.{a.name}" for a in node.names]))
    return found


def _package_of(path: Path, src: Path) -> str:
    """The first folder of `path` under `src` (agent, guidance, reader, clients, ...), or "" for a top-level module or a file outside `src`."""
    try:
        parts = path.relative_to(src).parts
    except ValueError:
        return ""
    return parts[0] if len(parts) > 1 else ""


def _framework_violation(module: str, package: str) -> bool:
    top = _top(module)
    if top in BANNED_CLIENTS:
        return True
    if package == AGENT_PACKAGE:
        framework_ok = top in AGENT_LANGCHAIN_MODULES or top.startswith(("langgraph", "langsmith"))
        return (top.startswith(FRAMEWORK_PREFIXES) and not framework_ok) or top.startswith(GUIDANCE_PREFIXES)
    if package == GUIDANCE_PACKAGE:
        return top.startswith(FRAMEWORK_PREFIXES)
    if package == READER_PACKAGE:
        return (top.startswith(FRAMEWORK_PREFIXES) and top not in READER_LANGCHAIN_MODULES) or top.startswith(GUIDANCE_PREFIXES)
    return top.startswith(FRAMEWORK_PREFIXES + GUIDANCE_PREFIXES)


def import_violations(path: Path, src: Path) -> list[str]:
    """The imports in the file at `path` that its package under `src` (the flysafelz package folder) may not make, in order.

    A file outside `src` (a script) is held to the same rule as a top-level module. Each hit is the module as written, except an import of
    flysafelz.agent or flysafelz.guidance, which is reported as the absolute dotted name it brings in (a relative import is resolved first).
    """
    package = _package_of(path, src)
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    found = []
    for module, names in _imports_of(tree, _own_package(path, src)):
        if _framework_violation(module, package):
            found.append(module)
        for name in names:
            if not any(_is_or_under(name, p) for p in INTERNAL_PACKAGES) or package == AGENT_PACKAGE:
                continue
            if package == GUIDANCE_PACKAGE and (_is_or_under(name, GUIDANCE_MAY_IMPORT) or _is_or_under(name, "flysafelz.guidance")):
                continue
            found.append(name)
    if package == AGENT_PACKAGE:
        shortcut = [
            n
            for n in ast.walk(tree)
            if (isinstance(n, ast.alias) and n.name == PROVIDER_SHORTCUT) or (isinstance(n, ast.Name) and n.id == PROVIDER_SHORTCUT) or (isinstance(n, ast.Attribute) and n.attr == PROVIDER_SHORTCUT)
        ]
        found += [PROVIDER_SHORTCUT for _ in shortcut]
    return found


def tree_import_violations(src: Path, extra: list[Path] | None = None) -> dict[str, list[str]]:
    """{path relative to src's parent folder: violations} for every .py file under `src` (and any `extra` files), omitting the clean ones."""
    hits = {}
    for path in sorted(src.rglob("*.py")) + (extra or []):
        found = import_violations(path, src)
        if found:
            hits[path.relative_to(src.parent).as_posix() if src in path.parents else path.name] = found
    return hits
