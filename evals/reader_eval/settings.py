"""The reader eval's scoring knobs, read from `settings.yaml` by name. No number is written in this module."""

from pathlib import Path

import yaml
from pydantic import BaseModel, NonNegativeFloat, PositiveFloat

from flysafelz.contracts import STRICT
from flysafelz.rules import RULES

SETTINGS_PATH = Path(__file__).with_name("settings.yaml")


class EvalSettings(BaseModel):
    """Every key is required and none may be added.

    ring_inner_nm: NM from the area's edge to the spots just inside it and just outside it.
    ring_outer_nm: NM from the edge to the spots near the edge of the local area.
    site_elevation_ft: the ground height at every spot.
    height_tolerance_ft: how far a top or a bottom may be from the label's before it counts (harmful or over-cautious).
    field_height_tolerance_ft: how far apart two heights may be to match in the per-field accuracy.
    """

    model_config = STRICT

    ring_inner_nm: PositiveFloat
    ring_outer_nm: PositiveFloat
    site_elevation_ft: NonNegativeFloat
    height_tolerance_ft: NonNegativeFloat
    field_height_tolerance_ft: NonNegativeFloat


def load_settings(path: Path = SETTINGS_PATH) -> EvalSettings:
    """The settings in a YAML file. Raises ValueError (a Pydantic ValidationError is one) naming a missing, unknown or invalid key."""
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}  # an empty file names every missing key
    return EvalSettings.model_validate(data)


SETTINGS = load_settings()


def reference_pressure_hpa() -> float:
    """The pressure every spot is judged at, on both sides of a comparison (R3.11): the rules' reference pressure, standard QNH."""
    value = RULES.fl_reference_hpa.value
    if isinstance(value, list):
        raise TypeError("rule fl_reference_hpa is a list, not a number")
    return float(value)
