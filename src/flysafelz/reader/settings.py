"""The reader's operational settings, read from `settings.yaml` by name. No number is written in this module."""

from pathlib import Path

import yaml
from pydantic import BaseModel, PositiveFloat

from flysafelz.contracts import STRICT

SETTINGS_PATH = Path(__file__).with_name("settings.yaml")


class ReaderSettings(BaseModel):
    """timeout_s: seconds one model call may take before it counts as unavailable."""

    model_config = STRICT

    timeout_s: PositiveFloat


def load_settings(path: Path = SETTINGS_PATH) -> ReaderSettings:
    """The settings in a YAML file. Raises ValueError (a Pydantic ValidationError is one) for a missing or unknown key."""
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return ReaderSettings.model_validate(data)


SETTINGS = load_settings()
