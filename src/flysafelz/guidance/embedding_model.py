"""`make setup` fetches the embedding model here, so the app never downloads at runtime (F5).

    ensure_embedding_model(name, folder)  -> bool   load the model from `folder`; download it first only if it is not stored there
    main()                                -> int    the `make setup` step: 0 when the model is ready, 1 (with a message) when it is not

The model name and the folder come from the agent's settings.yaml. The folder is git-ignored and is not fastembed's default,
which is a temp folder that gets wiped. With the model already stored, nothing touches the network.
"""

import sys
from pathlib import Path

from fastembed import TextEmbedding

from flysafelz.agent.settings import SETTINGS, embedding_folder


def ensure_embedding_model(model_name: str, folder: Path) -> bool:
    """Load the model from `folder`, downloading it into `folder` first when it is not stored there. True when a download was needed.

    Raises whatever fastembed raises when the download fails; `main` turns that into a message.
    """
    try:
        TextEmbedding(model_name, cache_dir=str(folder), local_files_only=True)
    except ValueError:  # fastembed's "could not load model from any source" for a model that is not stored here
        TextEmbedding(model_name, cache_dir=str(folder))
        return True
    return False


def main(model_name: str | None = None, folder: Path | None = None) -> int:
    """Make sure the model is stored. Prints one line saying what happened. Never fails quietly: a failed download names the model and says to rerun online."""
    model_name = SETTINGS.embedding_model if model_name is None else model_name
    folder = embedding_folder() if folder is None else folder
    try:
        downloaded = ensure_embedding_model(model_name, folder)
    except Exception as error:  # noqa: BLE001 - the one boundary: fastembed and huggingface_hub raise their own classes, whatever the network does
        reason = str(error).strip().rstrip(".") or type(error).__name__
        print(f"Could not download the embedding model {model_name} into {folder}: {reason}. Rerun `make setup` when you are online.", file=sys.stderr)
        return 1
    print(f"Embedding model {model_name} {'downloaded to' if downloaded else 'already stored in'} {folder}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
