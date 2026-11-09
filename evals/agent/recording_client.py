"""An in-memory stand-in for the LangSmith client, for the loader's tests and for `python -m evals.agent.loader --dry-run` (5a PR-3.F1).

It implements only the seven methods the loader calls, with the keyword arguments the loader passes, keeps the datasets and examples in
memory, and records every call in `calls` as (method name, keyword arguments). Nothing is sent anywhere: it opens no socket and reads no key.

`fail_on` makes the n-th call (1-based, counting every method) of a named method raise, to model a LangSmith error halfway through an
upload; the earlier calls have already taken effect, as they would have on the real service.
"""

import uuid
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from typing import Any


@dataclass
class StoredExample:
    """What list_examples returns for one example: the attributes the loader reads."""

    id: uuid.UUID
    inputs: dict[str, Any]
    outputs: dict[str, Any]
    metadata: dict[str, Any]


@dataclass
class StoredDataset:
    id: uuid.UUID
    name: str
    examples: dict[uuid.UUID, StoredExample] = field(default_factory=dict)


class RecordingClient:
    """Records each call; keeps datasets in memory. `fail_on={"create_examples": 2}` raises RuntimeError on the second create_examples call."""

    def __init__(self, fail_on: Mapping[str, int] | None = None) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.datasets: dict[str, StoredDataset] = {}
        self._fail_on = dict(fail_on or {})
        self._seen: dict[str, int] = {}

    # --- the record -------------------------------------------------------------------------------------------------------------

    def _record(self, method: str, **kwargs: Any) -> None:
        self.calls.append((method, kwargs))
        self._seen[method] = self._seen.get(method, 0) + 1
        if self._fail_on.get(method) == self._seen[method]:
            raise RuntimeError(f"simulated LangSmith error in {method} (call {self._seen[method]})")

    def methods_called(self) -> list[str]:
        return [method for method, _kwargs in self.calls]

    def writes(self) -> list[str]:
        """The calls that change a dataset: creating a dataset, creating, updating or deleting examples."""
        return [m for m in self.methods_called() if m in ("create_dataset", "create_examples", "update_examples", "delete_examples")]

    def state(self, dataset_name: str) -> dict[str, tuple[Any, Any, Any]]:
        """{example id: (inputs, outputs, metadata)} of one dataset, for comparing two runs' final states."""
        return {str(i): (e.inputs, e.outputs, e.metadata) for i, e in self.datasets[dataset_name].examples.items()}

    # --- the seven methods ------------------------------------------------------------------------------------------------------

    def has_dataset(self, *, dataset_name: str) -> bool:
        self._record("has_dataset", dataset_name=dataset_name)
        return dataset_name in self.datasets

    def read_dataset(self, *, dataset_name: str) -> StoredDataset:
        self._record("read_dataset", dataset_name=dataset_name)
        return self.datasets[dataset_name]

    def create_dataset(self, dataset_name: str, *, description: str | None = None) -> StoredDataset:
        self._record("create_dataset", dataset_name=dataset_name, description=description)
        made = StoredDataset(uuid.uuid4(), dataset_name)
        self.datasets[dataset_name] = made
        return made

    def _by_id(self, dataset_id: uuid.UUID) -> StoredDataset:
        return next(d for d in self.datasets.values() if d.id == dataset_id)

    def list_examples(self, *, dataset_id: uuid.UUID) -> Iterator[StoredExample]:
        self._record("list_examples", dataset_id=dataset_id)
        return iter(list(self._by_id(dataset_id).examples.values()))

    def create_examples(self, *, dataset_id: uuid.UUID, examples: list[dict[str, Any]]) -> None:
        self._record("create_examples", dataset_id=dataset_id, examples=examples)
        stored = self._by_id(dataset_id).examples
        for item in examples:
            stored[item["id"]] = StoredExample(item["id"], item["inputs"], item["outputs"], item["metadata"])

    def update_examples(self, *, dataset_id: uuid.UUID, updates: list[dict[str, Any]]) -> None:
        self._record("update_examples", dataset_id=dataset_id, updates=updates)
        stored = self._by_id(dataset_id).examples
        for item in updates:
            stored[item["id"]] = StoredExample(item["id"], item["inputs"], item["outputs"], item["metadata"])

    def delete_examples(self, *, example_ids: list[uuid.UUID]) -> None:
        self._record("delete_examples", example_ids=example_ids)
        for dataset in self.datasets.values():
            for example_id in example_ids:
                dataset.examples.pop(example_id, None)
