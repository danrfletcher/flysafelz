"""The one place a tool's output loses its phone numbers.

    redact_text(text, phone_map)         -> (text, phone_map)
    redact_artifact(artifact, phone_map) -> (artifact, phone_map)

Every string a tool returns, whether the text the model reads or the artifact the guard and the UI read, goes through here with the
thread's `phone_map` ({"<PHONE_1>": number as written}). A phone number becomes `<PHONE_n>`; a number already in the map keeps its
placeholder, so numbering is by first appearance and carries on across turns of one thread. What counts as a number is ticket 3's
`redact_phones` pattern, through its `redact_known`, so a NOTAM id, a DMS coordinate, a timestamp or a frequency is left alone.

Each call returns the map it ends with and never changes the one it was given: the tool writes the new map back to the thread's state
with a `Command` update. The real numbers live in that map and nowhere else the model, the trace or the log can see; they go back only
into the final answer, after the guard (rehydration is not here).

`redact_artifact` walks dicts (keys and values), lists, tuples, sets, Pydantic models (private attributes too) and dataclasses (fields
that are not `__init__` arguments too), and returns the same kind of
object with every string redacted (any mapping comes back as a plain dict). A string that holds no number is returned as the same object. A value of a kind it does not know
raises TypeError rather than pass through unread, so a new artifact shape cannot leak a number quietly.
"""

import dataclasses
from collections.abc import Mapping
from datetime import date, datetime, time, timedelta
from decimal import Decimal
from enum import Enum
from pathlib import PurePath
from typing import Any

from pydantic import BaseModel

from flysafelz.redact import redact_known

# Values that hold no text a number could be in.
PLAIN_VALUES = (Enum, type(None), bool, int, float, Decimal, datetime, date, time, timedelta, PurePath)


def redact_text(text: str, phone_map: Mapping[str, str]) -> tuple[str, dict[str, str]]:
    """The text with every phone number swapped for its `<PHONE_n>`, and the map extended with the numbers that were new."""
    return redact_known(text, dict(phone_map))


class _Walk:
    """One pass over an artifact. The map grows as strings are met, so numbers are numbered in the order the walk reaches them."""

    def __init__(self, phone_map: Mapping[str, str]):
        self.phone_map = dict(phone_map)

    def string(self, value: str) -> str:
        redacted, self.phone_map = redact_known(value, self.phone_map)
        return value if redacted == value else redacted  # an unchanged string (or str-valued enum) is kept as it is

    def visit(self, value: Any) -> Any:
        if isinstance(value, str):
            return self.string(value)
        if isinstance(value, PLAIN_VALUES):
            return value
        if isinstance(value, BaseModel):
            fields = {name: self.visit(getattr(value, name)) for name in type(value).model_fields}
            fields.update({name: self.visit(extra) for name, extra in (value.model_extra or {}).items()})
            copy = value.model_copy(update=fields)
            for name, private in (value.__pydantic_private__ or {}).items():  # private attributes are data too
                setattr(copy, name, self.visit(private))
            return copy
        if dataclasses.is_dataclass(value) and not isinstance(value, type):
            copy = dataclasses.replace(value, **{f.name: self.visit(getattr(value, f.name)) for f in dataclasses.fields(value) if f.init})
            for f in dataclasses.fields(value):
                if not f.init and hasattr(value, f.name):  # `replace` rebuilds these from their defaults: set the redacted value back
                    object.__setattr__(copy, f.name, self.visit(getattr(value, f.name)))
            return copy
        if isinstance(value, Mapping):
            return {(self.string(k) if isinstance(k, str) else k): self.visit(v) for k, v in value.items()}
        if isinstance(value, list | tuple | set | frozenset):
            items = [self.visit(item) for item in value]
            return type(value)(*items) if hasattr(value, "_fields") else type(value)(items)  # a named tuple takes its fields as arguments
        raise TypeError(f"cannot redact a {type(value).__name__}: it is not text, a number, a list, a dict or a model")


def redact_artifact(artifact: Any, phone_map: Mapping[str, str]) -> tuple[Any, dict[str, str]]:
    """The artifact with every string redacted, and the map extended with the numbers that were new. Dict keys are redacted too."""
    walk = _Walk(phone_map)
    redacted = walk.visit(artifact)
    return redacted, walk.phone_map
