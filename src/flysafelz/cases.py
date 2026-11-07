"""Load a committed fixture case (fixtures/<case>/) into what assess() takes. Offline, no key, no clock.

    load_case_inputs(case_dir)                      -> (Case, Forecast, AirspaceFetch, PibResult, readings)
    load_case_inputs_by_name(name, *, root=None)    the same for fixtures/<name>/ under the repository root

The files are replayed through the real client code by sources.replay_files (the same path as
sources.load_case_sources), so the openAIP filter and the PIB parser run as in a live fetch.

`today` for the case's own check is the case's `now`: a committed case is never refused as
"in the future", and the wall clock is never read. A malformed case.yaml raises CaptureError
naming the field; a missing or unreadable data file raises SourceError naming the file.

Unlike load_case_sources, the PIB status keeps the sha the parser gave it (none, offline);
case.yaml's recorded sha is not copied in. This is what the tests have always loaded.
"""

from datetime import UTC, datetime
from pathlib import Path

import yaml
from pydantic import AwareDatetime, TypeAdapter, ValidationError

from flysafelz.capture import ROOT, Case, load_case
from flysafelz.clients.open_meteo import Forecast
from flysafelz.clients.openaip import AirspaceFetch
from flysafelz.contracts import NotamReading
from flysafelz.pib import PibResult
from flysafelz.sources import FIXTURES_DIR, SourceError, replay_files

CaseInputs = tuple[Case, Forecast, AirspaceFetch, PibResult, dict[str, NotamReading]]

_AWARE = TypeAdapter(AwareDatetime)
_NEVER = datetime.max.replace(tzinfo=UTC)


def _own_now(case_yaml: Path) -> datetime:
    """The `now` written in case.yaml, to use as the case's own `today`.

    When it cannot be read or is not a valid aware time, the check is switched off instead, so that
    load_case reports the real problem (a missing file, bad YAML, the bad `now` field) in its usual words.
    """
    try:
        raw = yaml.safe_load(case_yaml.read_text(encoding="utf-8"))
        return _AWARE.validate_python(raw["now"])
    except (OSError, UnicodeDecodeError, yaml.YAMLError, TypeError, KeyError, ValidationError):
        return _NEVER


def _load(folder: Path, root: Path, readings_optional: bool = False) -> CaseInputs:
    case, _raw = load_case(folder / "case.yaml", today=_own_now(folder / "case.yaml"))
    forecast, airspace, pib, readings = replay_files(case, folder, root, readings_optional=readings_optional)
    return case, forecast, airspace, pib, readings


def load_case_inputs(case_dir: Path, *, readings_optional: bool = False) -> CaseInputs:
    """The case in this folder, ready for assess(): (case, forecast, airspace, pib, readings by NOTAM id).

    readings_optional: a folder with no readings.yaml gives no readings instead of a SourceError (the verdict
    sheet's draft mode, Q3). A readings.yaml that is there is still read and checked.
    """
    return _load(case_dir, ROOT, readings_optional)


def load_case_inputs_by_name(name: str, *, root: Path | None = None) -> CaseInputs:
    """fixtures/<name>/ under the repository root (capture.ROOT, found from this package's own location).

    The working directory is never used, so this gives the same result from anywhere. `root` replaces the
    repository root, for tests. Raises SourceError for a name that is not a folder under fixtures/.
    """
    root = ROOT if root is None else root
    folder = root / FIXTURES_DIR / name
    if not name or name != Path(name).name or name.startswith(".") or not folder.is_dir():
        raise SourceError(f"unknown case {name!r}: no folder {FIXTURES_DIR}/{name}/")
    return _load(folder, root)
