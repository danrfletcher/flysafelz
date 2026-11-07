"""The reader's operational settings, read from `settings.yaml` by name. No number is written in this module."""

from pathlib import Path

import yaml
from pydantic import BaseModel, PositiveFloat, PositiveInt

from flysafelz.contracts import STRICT

SETTINGS_PATH = Path(__file__).with_name("settings.yaml")


class ReaderSettings(BaseModel):
    """Every key is required and none may be added.

    timeout_s: seconds one model call may take before it counts as unavailable.
    concurrency: model calls in flight at once, first reads and repairs together (read_for_lz).
    runs: how many times the reader eval reads each labelled item.
    """

    model_config = STRICT

    timeout_s: PositiveFloat
    concurrency: PositiveInt
    runs: PositiveInt


def load_settings(path: Path = SETTINGS_PATH) -> ReaderSettings:
    """The settings in a YAML file. Raises ValueError (a Pydantic ValidationError is one) naming a missing, unknown or non-positive key."""
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}  # an empty file names every missing key
    return ReaderSettings.model_validate(data)


SETTINGS = load_settings()
