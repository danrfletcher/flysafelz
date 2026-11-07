"""The agent's operational settings, read from `settings.yaml` by name. No number is written in this module.

Imports only the standard library, yaml and pydantic, so `flysafelz.guidance` can read the knobs without loading LangChain.
A missing, unknown or wrongly typed key raises SettingsError naming the key (as rules.py does for a rule).
"""

from pathlib import Path
from typing import Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

SETTINGS_PATH = Path(__file__).with_name("settings.yaml")


class SettingsError(ValueError):
    """settings.yaml is missing a key, has an unknown one, or a key is malformed. The message names the key."""


class AgentSettings(BaseModel):
    """Every key is required and none may be added. Whole numbers are strict: 1.5 and "1" are refused, not rounded or converted.

    top_k: chunks returned for one guidance query.
    chunk_tokens_min, chunk_tokens_max: the token band the chunker aims at (min may not exceed max).
    guard_retries: extra model calls after a failed answer.
    model_max_retries: the chat model client's own retries.
    source_reuse_minutes: how long a fetched source is reused within a thread.
    embedding_model: the fastembed model name.
    embedding_cache_dir: where that model is stored, relative to the repository root (see embedding_folder).
    chunker_version: part of the corpus key.
    langsmith_project: the LangSmith project used when LANGSMITH_PROJECT is not set.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    top_k: int = Field(strict=True, gt=0)
    chunk_tokens_min: int = Field(strict=True, gt=0)
    chunk_tokens_max: int = Field(strict=True, gt=0)
    guard_retries: int = Field(strict=True, ge=0)
    model_max_retries: int = Field(strict=True, ge=0)
    source_reuse_minutes: int = Field(strict=True, gt=0)
    embedding_model: str = Field(strict=True, min_length=1)
    embedding_cache_dir: str = Field(strict=True, min_length=1)
    chunker_version: int = Field(strict=True, gt=0)
    langsmith_project: str = Field(strict=True, min_length=1)

    @model_validator(mode="after")
    def _band_is_not_inverted(self) -> Self:
        if self.chunk_tokens_min > self.chunk_tokens_max:
            raise ValueError("chunk_tokens_min must not be more than chunk_tokens_max")
        return self


def _describe(exc: ValidationError) -> str:
    return "; ".join(f"{'.'.join(str(p) for p in e['loc']) or 'settings'}: {e['msg']}" for e in exc.errors())


def load_settings(path: Path = SETTINGS_PATH) -> AgentSettings:
    """The settings in a YAML file. Raises SettingsError naming a missing, unknown or malformed key."""
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise SettingsError(f"agent settings file is not valid YAML: {exc}") from exc
    if data is None:
        data = {}  # an empty file names every missing key
    if not isinstance(data, dict):
        raise SettingsError("agent settings file must be a mapping of key to value")
    try:
        return AgentSettings.model_validate(data)
    except ValidationError as exc:
        raise SettingsError(f"agent settings: {_describe(exc)}") from exc


SETTINGS = load_settings()


def repository_root() -> Path:
    """The nearest folder above this file that holds pyproject.toml."""
    return next(folder for folder in Path(__file__).resolve().parents if (folder / "pyproject.toml").is_file())


def embedding_folder(settings: AgentSettings = SETTINGS) -> Path:
    """The folder the embedding model is stored in: `embedding_cache_dir` under the repository root (git-ignored)."""
    return repository_root() / settings.embedding_cache_dir
