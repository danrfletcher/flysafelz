"""The embedding model, loaded from the folder `make setup` filled and never downloaded (F5).

    Embedder                      the two methods of fastembed's TextEmbedding that are used (`embed` for documents, `query_embed` for a query)
    load_embedder(model, folder)  -> Embedder   `TextEmbedding(model, cache_dir=folder, local_files_only=True)`; raises EmbeddingModelMissing if it is not stored there
    load_settings_embedder()      -> Embedder   the same, with the model and folder settings.yaml names

`make guidance` (ingest.py) and the search index (index.py) both build the embedder here, so there is one place that says how it is loaded.
Nothing here runs on import: fastembed is constructed only when a function is called.
"""

from collections.abc import Iterable
from pathlib import Path
from typing import Any, Protocol

from fastembed import TextEmbedding

from flysafelz.agent.settings import SETTINGS, embedding_folder

MISSING_MESSAGE = "embedding model missing: run make setup"


class EmbeddingModelMissing(RuntimeError):
    """The embedding model is not stored in the model folder; `make setup` stores it. It is never downloaded here (F5)."""


class Embedder(Protocol):
    """The two methods of fastembed's TextEmbedding that are used. Vectors are sequences of numbers."""

    def embed(self, documents: Iterable[str], **kwargs: Any) -> Iterable[Any]: ...

    def query_embed(self, query: Iterable[str], **kwargs: Any) -> Iterable[Any]: ...


def load_embedder(model_name: str, folder: Path) -> Embedder:
    """The embedding model from the folder `make setup` filled. Raises EmbeddingModelMissing if it is not there; never downloads."""
    try:
        return TextEmbedding(model_name, cache_dir=str(folder), local_files_only=True)
    except ValueError as error:  # fastembed's "could not load model from any source" for a model that is not stored here
        raise EmbeddingModelMissing(MISSING_MESSAGE) from error


def load_settings_embedder() -> Embedder:
    """The embedder for the model and folder in settings.yaml. This is the factory the search index uses unless a test gives it another."""
    return load_embedder(SETTINGS.embedding_model, embedding_folder())
