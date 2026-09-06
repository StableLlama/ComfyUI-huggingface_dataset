"""ComfyUI node that loads a Hugging Face dataset and makes it available in ComfyUI.

The loaded dataset is exposed in two forms:

* ``dataset``  - the raw object returned by :func:`datasets.load_dataset` (a
  :class:`datasets.Dataset` for a single split). It is handed to ComfyUI as an
  opaque ``HUGGINGFACE_DATASET`` value that other nodes can consume.
* ``rows``     - the rows of the selected split materialized as a ComfyUI
  *Data List* (a Python ``list`` of ``dict`` rows), so each record can be
  processed further with generic data-handling nodes (for example the
  "Basic data handling" node pack).

Loading is delegated to the `datasets <https://huggingface.co/docs/datasets>`_
library (``datasets.load_dataset``), which is imported lazily so that ComfyUI
keeps starting even when the dependency is not installed yet.
"""

from inspect import cleandoc
from os import path as os_path
from typing import Any


class IO:
    """Node I/O type constants.

    The values mirror ``comfy.comfy_types.node_typing.IO``. When ComfyUI is
    available they are replaced at runtime by the real constants (see below);
    otherwise these fallbacks keep the module importable outside ComfyUI.
    """

    BOOLEAN: str = "BOOLEAN"
    INT: str = "INT"
    FLOAT: str = "FLOAT"
    STRING: str = "STRING"
    NUMBER: str = "FLOAT,INT"
    ANY: str = "*"


class ComfyNodeABC:
    """Node base class.

    Replaced at runtime by ComfyUI's ``ComfyNodeABC`` when ComfyUI is available;
    a plain (empty) class keeps the module importable outside ComfyUI and gives
    mypy a concrete, non-``Any`` base to type against.
    """


try:  # pragma: no cover - ComfyUI-only
    from comfy.comfy_types.node_typing import IO as _ComfyIO
    from comfy.comfy_types.node_typing import ComfyNodeABC as _ComfyNodeABC

    IO = _ComfyIO  # type: ignore[misc]  # swap in the real ComfyUI IO at runtime
    ComfyNodeABC = _ComfyNodeABC  # type: ignore[misc]
except Exception:
    pass

INT_MAX = 2**31 - 1

#: file builders of the ``datasets`` library and the extensions they can read.
#: ``auto`` and ``hub`` are special "loaders": see :func:`_load_dataset`.
_FILE_BUILDERS = ("arrow", "csv", "json", "parquet", "text")
_LOADER_CHOICES = ("auto", "hub", *_FILE_BUILDERS)

_EXTENSION_TO_BUILDER = {
    ".arrow": "arrow",
    ".csv": "csv",
    ".tsv": "csv",
    ".json": "json",
    ".jsonl": "json",
    ".ndjson": "json",
    ".parquet": "parquet",
    ".txt": "text",
}


def _require_datasets() -> Any:
    """Import and return the ``datasets`` module (lazy import).

    The dependency is imported at call time instead of at module import time so
    that ComfyUI still starts when ``datasets`` is missing; the user only gets a
    clear, actionable error when they actually try to load a dataset.
    """
    try:
        import datasets  # noqa: PLC0415 - deliberately imported lazily
    except ImportError as exc:  # pragma: no cover - exercised when dependency missing
        raise ImportError(
            "The Hugging Face 'datasets' package is required to load datasets.\nInstall it with:  pip install datasets"
        ) from exc
    return datasets


def _infer_file_builder(path: str) -> str | None:
    """Guess the ``datasets`` file builder from a path/glob/URL extension.

    Returns ``None`` when no known file extension can be detected (typical for a
    Hub repository id or a local dataset directory).
    """
    base = path.split("?", 1)[0].split("#", 1)[0]  # strip URL query / fragment
    extension = os_path.splitext(base)[1].lower()
    return _EXTENSION_TO_BUILDER.get(extension)


def _to_python(value: Any) -> Any:
    """Recursively convert a (nested) row value into plain Python objects.

    numpy scalars/arrays are duck-typed via ``.item()`` / ``.tolist()`` so that
    numpy itself does not need to be imported here; any other object (PIL image,
    audio, ...) is passed through unchanged.
    """
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, dict):
        return {key: _to_python(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_python(item) for item in value]
    if isinstance(value, bytes):
        try:
            return value.decode("utf-8")
        except UnicodeDecodeError:
            return value.hex()
    if hasattr(value, "tolist") and hasattr(value, "ndim"):  # numpy array
        try:
            return _to_python(value.tolist())
        except Exception:  # noqa: BLE001 - be tolerant of exotic objects
            pass
    if hasattr(value, "item"):  # numpy scalar (and similar)
        try:
            return _to_python(value.item())
        except Exception:  # noqa: BLE001 - be tolerant of exotic objects
            pass
    return value


def _load_dataset(
    path: str,
    loader: str = "auto",
    split: str = "train",
    config: str = "",
    revision: str = "",
    trust_remote_code: bool = False,
) -> Any:
    """Load a single-split dataset with :func:`datasets.load_dataset`.

    ``path`` is either a Hub repository id (e.g. ``"rotten_tomatoes"`` or
    ``"lhoestq/demo1"``), a local dataset directory, or a local/remote/glob
    file path (``csv``/``json``/``jsonl``/``parquet``/``arrow``/``txt``).
    """
    datasets = _require_datasets()

    if not path or not path.strip():
        raise ValueError("'path' must not be empty: give a Hub dataset id or a file path.")

    active_split = split.strip() or "train"
    kwargs: dict[str, Any] = {"split": active_split, "trust_remote_code": bool(trust_remote_code)}
    if revision.strip():
        kwargs["revision"] = revision.strip()

    active_loader = loader
    if active_loader == "auto":
        active_loader = _infer_file_builder(path) or "hub"

    if active_loader in _FILE_BUILDERS:
        # Local / remote / glob data files read through the matching file builder.
        return datasets.load_dataset(active_loader, data_files=path, **kwargs)

    # "hub" loader: Hub repository id (optionally with config subset) or a local
    # dataset directory.
    if config.strip():
        return datasets.load_dataset(path, config.strip(), **kwargs)
    return datasets.load_dataset(path, **kwargs)


def _materialize_rows(dataset: Any, limit: int = -1) -> list[dict[str, Any]]:
    """Return up to ``limit`` rows of ``dataset`` as a list of plain dicts.

    A negative ``limit`` (or ``None``) returns every row. Rows are fetched as a
    single batched slice (column-wise), then re-joined into per-row dicts.
    """
    length = len(dataset)
    stop = length if limit is None or limit < 0 else min(int(limit), length)
    if stop <= 0:
        return []
    batch = dataset[:stop]  # dict of column name -> list of values
    columns = list(batch.keys())
    return [{column: _to_python(batch[column][index]) for column in columns} for index in range(stop)]


class LoadHuggingFaceDataset(ComfyNodeABC):
    """Loads a Hugging Face dataset (Hub or local/remote files) into ComfyUI.

    Loading is done through the ``datasets`` library (``datasets.load_dataset``).

    Sources:
      - **Hugging Face Hub** repository id, e.g. ``rotten_tomatoes``,
        ``glue`` (with config ``mrpc``), ``lhoestq/custom_squad`` (with revision).
      - **Local / remote files**: ``csv``, ``tsv``, ``json``, ``jsonl``,
        ``parquet``, ``arrow`` and ``txt``. With ``loader`` set to ``auto`` the
        format is inferred from the file extension; otherwise pick the matching
        file builder explicitly. Globs and ``https://`` / ``hf://`` URLs work too.

    Outputs:
      - ``dataset``: the raw ``datasets.Dataset`` of the selected split.
      - ``rows``: a *Data List* of row dicts (one dict per row) ready for further
        processing with generic data-handling nodes. ``limit`` caps how many
        rows are materialized (``-1`` = all).
    """

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "path": (IO.STRING, {"default": "", "multiline": False}),
                "loader": (_LOADER_CHOICES, {"default": "auto"}),
                "split": (IO.STRING, {"default": "train", "multiline": False}),
                "config": (IO.STRING, {"default": "", "multiline": False}),
                "revision": (IO.STRING, {"default": "", "multiline": False}),
                "trust_remote_code": (IO.BOOLEAN, {"default": False}),
                "limit": (IO.INT, {"default": -1, "min": -1, "max": INT_MAX, "step": 1}),
            }
        }

    RETURN_TYPES = ("HUGGINGFACE_DATASET", IO.ANY)
    RETURN_NAMES = ("dataset", "rows")
    OUTPUT_IS_LIST = (False, True)
    CATEGORY = "HuggingFace/Dataset"
    DESCRIPTION = cleandoc(__doc__ or "")
    FUNCTION = "load"

    def load(
        self,
        path: str,
        loader: str = "auto",
        split: str = "train",
        config: str = "",
        revision: str = "",
        trust_remote_code: bool = False,
        limit: int = -1,
    ) -> tuple[Any, list[dict[str, Any]]]:
        """Load the dataset and return ``(dataset, rows)``."""
        dataset = _load_dataset(
            path=path,
            loader=loader,
            split=split,
            config=config,
            revision=revision,
            trust_remote_code=trust_remote_code,
        )
        rows = _materialize_rows(dataset, limit)
        return (dataset, rows)


NODE_CLASS_MAPPINGS = {
    "LoadHuggingFaceDataset": LoadHuggingFaceDataset,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadHuggingFaceDataset": "Hugging Face Dataset Loader",
}
