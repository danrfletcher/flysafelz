"""How the online job reads a LangSmith run (7 PR-1.F1). One place, so `pull` and `reference` read a run the same way.

A run is the object `client.list_runs` returns (`langsmith.schemas.Run`). Ticket 4 writes `source` and `path` into the root run's metadata, which LangSmith holds as
`run.extra["metadata"]`, and also copies `path` into the root run's outputs when the run ends (I5, I6). The delivered answer is the root run's `outputs["answer"]`.
Anything missing or of the wrong shape reads as empty: a run is never rejected here, the caller decides what an empty value means.
"""

from collections.abc import Mapping
from typing import Any

SOURCE_KEY = "source"
PATH_KEY = "path"
ANSWER_KEY = "answer"


def mapping(value: object) -> Mapping[str, Any]:
    """`value` when it is a mapping, else an empty one."""
    return value if isinstance(value, Mapping) else {}


def metadata_of(run: Any) -> Mapping[str, Any]:
    """The run's metadata (`extra.metadata`), or an empty mapping."""
    return mapping(mapping(getattr(run, "extra", None)).get("metadata"))


def inputs_of(run: Any) -> Mapping[str, Any]:
    return mapping(getattr(run, "inputs", None))


def outputs_of(run: Any) -> Mapping[str, Any]:
    return mapping(getattr(run, "outputs", None))


def final_answer(run: Any) -> str:
    """The delivered answer in the run's outputs, or an empty string when there is none (a blank answer is none)."""
    answer = outputs_of(run).get(ANSWER_KEY)
    return answer if isinstance(answer, str) and answer.strip() else ""
