"""Where the map routes keep their files, which browser origins may read them, and the openAIP key.

    project_root()                 the repository root (this file is src/flysafelz/mapdata/config.py)
    cache_dir(root)                <root>/data/map, the only folder these routes write to
    process_settings()             the process's settings (the variables the server was started with)
    ui_origins(environ)            the two dev UI origins plus FLYSAFELZ_UI_ORIGIN when it is set and is a plain origin
    openaip_key(environ, root)     the openAIP key, read the way capture.find_key reads it (environment, then .env)

CORS never uses a wildcard: an origin is allowed only when it is on that list, character for character.
"""

import os
import re
from collections.abc import Mapping
from pathlib import Path

from flysafelz.capture import find_key

DEV_UI_ORIGIN = "http://localhost:5173"  # `make dev`: Vite's dev server
PREVIEW_UI_ORIGIN = "http://localhost:4173"  # `vite preview`, which the tester uses
UI_ORIGIN_VARIABLE = "FLYSAFELZ_UI_ORIGIN"
CACHE_FOLDER = ("data", "map")
ORIGIN = re.compile(r"https?://[A-Za-z0-9.-]+(?::\d+)?")  # scheme, host and optional port: no path, no wildcard


def project_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent.parent


def cache_dir(root: Path) -> Path:
    return root.joinpath(*CACHE_FOLDER)


def ui_origins(environ: Mapping[str, str]) -> list[str]:
    """The origins that may call the map routes. FLYSAFELZ_UI_ORIGIN is added when it is one plain origin (a trailing slash is dropped); anything else is ignored."""
    origins = [DEV_UI_ORIGIN, PREVIEW_UI_ORIGIN]
    extra = environ.get(UI_ORIGIN_VARIABLE, "").strip().rstrip("/")
    if ORIGIN.fullmatch(extra) and extra not in origins:
        origins.append(extra)
    return origins


def process_settings() -> Mapping[str, str]:
    return os.environ


def openaip_key(environ: Mapping[str, str], root: Path) -> str | None:
    return find_key(environ, root)
