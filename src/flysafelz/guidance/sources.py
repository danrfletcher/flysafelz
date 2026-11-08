"""The guidance register (G16): which documents `make guidance` may fetch, read from `data/guidance/sources.yaml`.

    load_sources(path)        -> list[Source]   every entry, in file order; raises SourcesError naming the entry that is wrong
    indexed(sources)          -> list[Source]   the entries with `indexed: true` (the only ones ever downloaded or embedded)
    url_is_allowed(url)       -> bool           https, or http to this machine only (a local test server)

An entry that is `indexed: false` is link-only: the register names it, nothing downloads or embeds it. Nothing in this module touches the network.
"""

import re
from datetime import date
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

DOCUMENT_ID = re.compile(r"[a-z0-9]+(?:[-_][a-z0-9]+)*")  # the id is a folder name under data/guidance/, so it can hold no path part
LOOPBACK_HOSTS = ("localhost", "127.0.0.1", "::1")  # the one place plain http is allowed: a test server on this machine
ISO_DATE = re.compile(r"\d{4}-\d{2}-\d{2}")


class SourcesError(ValueError):
    """sources.yaml is unreadable, or an entry is wrong. The message names the entry."""


def url_is_allowed(url: str) -> bool:
    """https with a host, or http to a loopback host. Anything else (http elsewhere, ftp, file, no host) is refused."""
    parts = urlsplit(url)
    if not parts.hostname:
        return False
    return parts.scheme == "https" or (parts.scheme == "http" and parts.hostname in LOOPBACK_HOSTS)


class Source(BaseModel):
    """One document: id (also its folder), title, url, edition, licence, review_date (YYYY-MM-DD) and whether it is indexed."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(strict=True)
    title: str = Field(strict=True, min_length=1)
    url: str = Field(strict=True)
    edition: str = Field(strict=True, min_length=1)
    licence: str = Field(strict=True, min_length=1)
    review_date: date
    indexed: bool = Field(strict=True)

    @field_validator("id")
    @classmethod
    def _id_is_a_safe_name(cls, value: str) -> str:
        if not DOCUMENT_ID.fullmatch(value):
            raise ValueError("id must be lower-case letters and digits, joined by - or _ (it is a folder name)")
        return value

    @field_validator("url")
    @classmethod
    def _url_is_https(cls, value: str) -> str:
        if not url_is_allowed(value):
            raise ValueError("url must be an https URL")
        return value

    @field_validator("review_date", mode="before")
    @classmethod
    def _date_is_iso(cls, value: Any) -> Any:
        """A YAML date, or a YYYY-MM-DD string that is a real day. A number, a datetime or any other text is refused."""
        if isinstance(value, date) and not hasattr(value, "hour"):
            return value
        if isinstance(value, str) and ISO_DATE.fullmatch(value):
            return value
        raise ValueError("review_date must be a date written YYYY-MM-DD")


def _first_message(error: ValidationError) -> str:
    first = error.errors()[0]
    field = ".".join(str(part) for part in first["loc"])
    return f"{field}: {first['msg']}" if field else first["msg"]


def _label(entry: Any, position: int) -> str:
    """How an error names an entry: its id when it has a usable one, else its place in the file."""
    if isinstance(entry, dict) and isinstance(entry.get("id"), str) and entry["id"]:
        return f"entry '{entry['id']}'"
    return f"entry #{position}"


def load_sources(path: Path) -> list[Source]:
    """The register in `path`. Raises SourcesError for a missing field, a duplicate id, a URL that is not https or a malformed date, naming the entry."""
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except OSError as error:
        raise SourcesError(f"{path.name}: cannot read the register: {error.strerror or error}") from error
    except yaml.YAMLError as error:
        raise SourcesError(f"{path.name}: not valid YAML: {error}") from error
    entries = data.get("documents") if isinstance(data, dict) else None
    if not isinstance(entries, list) or not entries:
        raise SourcesError(f"{path.name}: expected a non-empty list under `documents:`")
    sources: list[Source] = []
    first_seen: dict[str, int] = {}
    for position, entry in enumerate(entries, start=1):
        label = _label(entry, position)
        if not isinstance(entry, dict):
            raise SourcesError(f"{path.name}: {label}: each entry must be a mapping")
        try:
            source = Source.model_validate(entry)
        except ValidationError as error:
            raise SourcesError(f"{path.name}: {label}: {_first_message(error)}") from error
        if source.id in first_seen:
            raise SourcesError(f"{path.name}: duplicate id '{source.id}' (entries #{first_seen[source.id]} and #{position})")
        first_seen[source.id] = position
        sources.append(source)
    return sources


def indexed(sources: list[Source]) -> list[Source]:
    """The entries that are downloaded and embedded."""
    return [source for source in sources if source.indexed]
