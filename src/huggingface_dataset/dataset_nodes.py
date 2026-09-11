"""ComfyUI node that loads a Hugging Face dataset and makes it available in ComfyUI.

The loaded dataset is exposed in two forms:

* ``dataset``  - the raw object returned by :func:`datasets.load_dataset` (a
  :class:`datasets.Dataset`, or a :class:`datasets.IterableDataset` when
  ``streaming`` is enabled, for a single split). It is handed to ComfyUI as an
  opaque ``HUGGINGFACE_DATASET`` value that other nodes can consume.
* ``rows``     - the rows of the selected split materialized as a ComfyUI
  *Data List* (a Python ``list`` of ``dict`` rows), so each record can be
  processed further with generic data-handling nodes (for example the
  "Basic data handling" node pack).

Loading is delegated to the `datasets <https://huggingface.co/docs/datasets>`_
library (``datasets.load_dataset``), which is imported lazily so that ComfyUI
keeps starting even when the dependency is not installed yet.
"""

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from inspect import cleandoc
from itertools import islice
from os import path as os_path
from typing import Any


# Optional JPEG XL image support.
#
# ``datasets`` decodes images with Pillow, which only auto-registers its
# *built-in* formats. JPEG XL (``.jxl``) support comes from a third-party
# Pillow plugin - ``pillow-jxl-plugin`` (import name ``pillow_jxl``) - whose
# decoder registers itself with Pillow as an import side effect. Nothing in the
# ``datasets`` load path imports that module, so this pack does it here: with
# the plugin installed, JPEG XL images (even extensionless cached blobs) decode
# fine; without it, loading still works and only the JPEG XL images fail to
# decode (see huggingface/datasets#8537).
def _pillow_jxl_importable() -> bool:
    """Import ``pillow_jxl`` and report whether Pillow's JPEG XL decoder is available.

    The import itself is the whole point: it registers the JPEG XL codec with
    Pillow. The plugin is optional, so a missing one is silently ignored.
    """
    try:
        import pillow_jxl  # noqa: F401  # type: ignore[import-not-found, import-untyped]  # side-effect: registers the JPEG XL codec
    except ImportError:
        return False
    return True


#: Whether Pillow can decode JPEG XL images (``pip install pillow-jxl-plugin``).
JPEG_XL_AVAILABLE = _pillow_jxl_importable()


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

#: Split used when the user does not pick one. It is a member of
#: ``_FALLBACK_SPLITS`` so ComfyUI can offer it before the dataset is known.
_DEFAULT_SPLIT = "train"

#: Options offered by the ``split`` dropdown until the dataset's real splits are
#: known (the frontend replaces them with the actual split names once a source
#: is chosen). Keep this in sync with the JS fallback list in ``web/js``.
_FALLBACK_SPLITS = ("train", "test", "validation")

#: Order in which a "sensible" split is picked when the requested one is absent
#: (e.g. a dataset that only has a ``test`` split but no ``train``).
_PREFERRED_SPLITS = ("train", "validation", "test")

#: Loggers of the Hugging Face stack that spam the console with per-request INFO
#: lines ("HTTP Request: ..." via ``httpx``) while a dataset is looked up/loaded.
_HF_LOGGER_NAMES = ("httpx", "huggingface_hub", "datasets", "filelock", "httpcore", "urllib3")


@contextmanager
def _quiet_http_logs() -> Iterator[None]:
    """Temporarily quieten the noisy HTTP/datasets loggers during our own calls.

    The ``datasets``/``huggingface_hub`` stack logs every HTTP round-trip at
    INFO level (through ``httpx``), which clutters the ComfyUI console while the
    node discovers a dataset's splits or loads data. Only the loggers listed in
    :data:`_HF_LOGGER_NAMES` (and their sub-loggers) are raised to WARNING, and
    only for the duration of the wrapped library call; their previous level is
    restored afterwards.
    """
    names = list(_HF_LOGGER_NAMES)
    names.extend(name for name in logging.Logger.manager.loggerDict if name.startswith(_HF_LOGGER_NAMES) and name not in names)
    affected: list[tuple[logging.Logger, int]] = []
    for name in names:
        logger = logging.getLogger(name)
        if logger.level == logging.NOTSET or logger.level < logging.WARNING:
            affected.append((logger, logger.level))
            logger.setLevel(logging.WARNING)
    try:
        yield
    finally:
        for logger, level in affected:
            logger.setLevel(level)


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


def _split_name(split: str) -> str:
    """Return the plain split name of ``split``, dropping any slicing suffix.

    ``datasets`` slicing syntax (``"train[:100]"``, ``"train[:10%]"``) encodes
    the split to use in the part before the ``[``; that part is what has to be
    present in the dataset's split list.
    """
    return split.split("[", 1)[0].strip()


def _sensible_split(available: list[str]) -> str:
    """Pick the friendliest default among ``available`` split names."""
    for name in _PREFERRED_SPLITS:
        if name in available:
            return name
    return available[0] if available else _DEFAULT_SPLIT


def _available_splits(
    path: str,
    loader: str = "auto",
    config: str = "",
    revision: str = "",
) -> list[str] | None:
    """Return the split names a dataset exposes, or ``None`` when they cannot be determined.

    * Empty ``path`` yields an empty list (nothing to look up yet).
    * A single file/glob/URL (``csv``/``json``/``parquet``/``arrow``/``text``)
      always maps to exactly one split (``train``), matching
      :func:`datasets.load_dataset` when a single string is given as
      ``data_files``.
    * Hub ids and local dataset directories are looked up through
      ``datasets.get_dataset_split_names``.

    Returns ``None`` when the splits cannot be determined (network errors, a
    multi-config Hub dataset without a ``config``, ...) so callers can fall back
    to ``datasets`` itself.
    Raises :class:`ImportError` when the ``datasets`` package is missing so
    callers can surface the actionable install message.
    """
    clean_path = (path or "").strip()
    if not clean_path:
        return []

    datasets = _require_datasets()  # ImportError propagates to the caller

    active_loader = loader
    if active_loader == "auto":
        active_loader = _infer_file_builder(clean_path) or "hub"

    if active_loader in _FILE_BUILDERS:
        # A single file / glob is always loaded as one "train" split.
        return [_DEFAULT_SPLIT]

    try:
        kwargs: dict[str, Any] = {}
        if (config or "").strip():
            kwargs["config_name"] = config.strip()
        if (revision or "").strip():
            kwargs["revision"] = revision.strip()
        with _quiet_http_logs():
            raw_names = datasets.get_dataset_split_names(clean_path, **kwargs)
        return [str(name) for name in raw_names]
    except Exception:
        return None


def _resolve_split(
    path: str,
    split: str,
    loader: str = "auto",
    config: str = "",
    revision: str = "",
) -> str:
    """Resolve the split to actually load, giving clean feedback on bad splits.

    The dropdown normally prevents invalid splits, but this still guards cases
    where the requested value comes from an older workflow or an upstream value:

    * when the split (or the split part of a slicing expression like
      ``"train[:100]"``) exists in the dataset it is passed through unchanged;
    * when the *default* split (``"train"``) does not exist - e.g. a dataset
      that only ships a ``test`` split - a sensible alternative is picked;
    * any other unknown split raises a clear :class:`ValueError` listing the
      available splits instead of an opaque ``datasets`` traceback;
    * when the available splits cannot be determined (offline, missing
      ``datasets``, ...) the request is passed through untouched so
      :func:`datasets.load_dataset` keeps its usual behaviour.
    """
    requested = (split or "").strip() or _DEFAULT_SPLIT
    try:
        available = _available_splits(
            path=path,
            loader=loader,
            config=config,
            revision=revision,
        )
    except ImportError:
        return requested  # "pip install datasets" surfaces later, from _load_dataset()

    if not available:
        return requested  # unknown -> let datasets load / report

    if _split_name(requested) in available:
        return requested
    if requested == _DEFAULT_SPLIT:
        return _sensible_split(available)
    listed = ", ".join(available)
    raise ValueError(f"Unknown split {requested!r}: this dataset has no {_split_name(requested)!r} split. Available splits: {listed}.")


def split_info(
    path: str,
    loader: str = "auto",
    config: str = "",
    revision: str = "",
) -> dict[str, Any]:
    """Return the dataset's splits for the frontend dropdown (HTTP-friendly).

    Returns ``{"splits": [...], "default": str, "error": None}`` on success and
    ``{"splits": None, "default": None, "error": str}`` when the splits cannot be
    determined (e.g. ``datasets`` is not installed).
    """
    try:
        available = _available_splits(
            path=path,
            loader=loader,
            config=config,
            revision=revision,
        )
    except ImportError as exc:  # datasets missing -> actionable message
        return {"splits": None, "default": None, "error": str(exc)}

    if available is None:
        return {
            "splits": None,
            "default": None,
            "error": "Could not determine the splits of this dataset (network, missing config, or loading-code issues).",
        }
    if not available:
        return {
            "splits": None,
            "default": None,
            "error": "No splits were found for this dataset source.",
        }
    return {"splits": available, "default": _sensible_split(available), "error": None}


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
    streaming: bool = False,
) -> Any:
    """Load a single-split dataset with :func:`datasets.load_dataset`.

    ``path`` is either a Hub repository id (e.g. ``"stanfordnlp/imdb"``), a local
    dataset directory, or a local/remote/glob file path
    (``csv``/``json``/``jsonl``/``parquet``/``arrow``/``txt``). With
    ``streaming=True`` the split comes back as a lazily-loaded
    :class:`datasets.IterableDataset` (nothing is downloaded until iterated)
    instead of a materialized :class:`datasets.Dataset`.
    """
    datasets = _require_datasets()

    if not path or not path.strip():
        raise ValueError("'path' must not be empty: give a Hub dataset id or a file path.")

    active_split = split.strip() or "train"
    kwargs: dict[str, Any] = {"split": active_split, "streaming": bool(streaming)}
    if revision.strip():
        kwargs["revision"] = revision.strip()

    active_loader = loader
    if active_loader == "auto":
        active_loader = _infer_file_builder(path) or "hub"

    with _quiet_http_logs():
        if active_loader in _FILE_BUILDERS:
            # Local / remote / glob data files read through the matching file builder.
            return datasets.load_dataset(active_loader, data_files=path, **kwargs)

        # "hub" loader: Hub repository id (optionally with config subset) or a
        # local dataset directory.
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


def _materialize_rows_stream(dataset: Any, limit: int = -1) -> list[dict[str, Any]]:
    """Return up to ``limit`` rows from a streaming ``IterableDataset``.

    ``datasets.IterableDataset`` (what :func:`datasets.load_dataset` returns with
    ``streaming=True``) has no ``len()`` and cannot be sliced, so rows are pulled
    by iterating and each row dict is converted with :func:`_to_python`. A
    negative ``limit`` (or ``None``) consumes the whole stream.
    """
    stream = islice(dataset, int(limit)) if limit is not None and limit >= 0 else dataset
    return [{key: _to_python(value) for key, value in row.items()} for row in stream]


class LoadHuggingFaceDataset(ComfyNodeABC):
    """Loads a Hugging Face dataset (Hub or local/remote files) into ComfyUI.

    Loading is done through the ``datasets`` library (``datasets.load_dataset``).

    Sources:
      - **Hugging Face Hub** repository id (`namespace/name`), e.g.
        ``stanfordnlp/imdb``, ``nyu-mll/glue`` (with config ``mrpc``), or
        ``HuggingFaceFW/fineweb`` (with revision).
      - **Local / remote files**: ``csv``, ``tsv``, ``json``, ``jsonl``,
        ``parquet``, ``arrow`` and ``txt``. With ``loader`` set to ``auto`` the
        format is inferred from the file extension; otherwise pick the matching
        file builder explicitly. Globs and ``https://`` / ``hf://`` URLs work too.

    Modes:
      - **Default** (``streaming`` off) downloads and caches the whole split and
        returns a ``datasets.Dataset``.
      - **Streaming** (``streaming`` on) returns a lazily-loaded
        ``datasets.IterableDataset`` instead; data files are only fetched as the
        dataset is iterated, so no full download/cache happens. Ideal for very
        large datasets when only the first ``limit`` rows are needed as a
        *Data List*.

    Outputs:
      - ``dataset``: the raw ``datasets.Dataset`` (or
        ``datasets.IterableDataset`` when ``streaming`` is on) of the selected
        split. It is passed to ComfyUI as an opaque ``HUGGINGFACE_DATASET``
        value; consumer nodes must handle both types.
      - ``rows``: a *Data List* of row dicts (one dict per row) ready for further
        processing with generic data-handling nodes. ``limit`` caps how many
        rows are materialized (``-1`` = all).
    """

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "path": (
                    IO.STRING,
                    {
                        "default": "",
                        "multiline": False,
                        "tooltip": "Hub dataset id (e.g. stanfordnlp/imdb), or a local/remote file, glob or dataset directory.",
                    },
                ),
                "loader": (
                    _LOADER_CHOICES,
                    {
                        "default": "auto",
                        "tooltip": "How to read the source: 'auto' infers from the file extension, 'hub' loads a Hub id, or pick a file builder (csv/json/parquet/arrow/text).",
                    },
                ),
                # A dropdown; the frontend swaps the fallback options for the
                # dataset's real split names once `path`/`config`/... are known.
                "split": (
                    _FALLBACK_SPLITS,
                    {
                        "default": _DEFAULT_SPLIT,
                        "tooltip": "Split to load. The dropdown lists the splits of the selected source; slicing like 'train[:100]' still works.",
                    },
                ),
                "config": (
                    IO.STRING,
                    {
                        "default": "",
                        "multiline": False,
                        "tooltip": "Config/subset name for Hub datasets that have several configs (e.g. glue + mrpc).",
                    },
                ),
                "revision": (
                    IO.STRING,
                    {
                        "default": "",
                        "multiline": False,
                        "tooltip": "Optional Hub revision: tag, branch name, or commit hash.",
                    },
                ),
                "streaming": (
                    IO.BOOLEAN,
                    {
                        "default": False,
                        "tooltip": "Load the split lazily as an IterableDataset instead of downloading/caching it fully.",
                    },
                ),
                "limit": (
                    IO.INT,
                    {
                        "default": -1,
                        "min": -1,
                        "max": INT_MAX,
                        "step": 1,
                        "tooltip": "Maximum number of rows to materialize into the 'rows' Data List; -1 = all.",
                    },
                ),
                # Cache-buster used by the frontend "Force reload" button. The
                # frontend hides this widget and increments it on click; because
                # the value is part of this node's inputs, ComfyUI treats the
                # loader as changed and re-fetches the dataset on the next run
                # even when the other inputs are unchanged. The value has no 
                # effect on the data.
                "reload_tick": (
                    IO.INT,
                    {
                        "default": 0,
                        "min": 0,
                        "max": INT_MAX,
                        "step": 1,
                        "tooltip": "Incremented by the 'Force reload' button so the next run triggers a fresh load (hidden; has no effect on the data).",
                    },
                ),
            }
        }

    RETURN_TYPES = ("HUGGINGFACE_DATASET", IO.ANY)
    RETURN_NAMES = ("dataset", "rows")
    OUTPUT_IS_LIST = (False, True)
    OUTPUT_TOOLTIPS = (
        "Raw datasets.Dataset of the split (a streaming IterableDataset with 'streaming' on); feed it into the 🤗 dataset nodes.",
        "ComfyUI Data List of row dicts (one dict per row), capped by 'limit'.",
    )
    CATEGORY = "Hugging Face 🤗"
    DESCRIPTION = cleandoc(__doc__ or "")
    FUNCTION = "load"

    def load(
        self,
        path: str,
        loader: str = "auto",
        split: str = "train",
        config: str = "",
        revision: str = "",
        streaming: bool = False,
        limit: int = -1,
        reload_tick: int = 0,
    ) -> tuple[Any, list[dict[str, Any]]]:
        """Load the dataset and return ``(dataset, rows)``.

        ``reload_tick`` is a cache-buster: the frontend "Force reload" button
        increments it so ComfyUI treats this node as changed and re-fetches the
        dataset on the next run even when every other input stays the same. Its
        value is otherwise ignored.
        """
        requested_split = (split or "").strip() or _DEFAULT_SPLIT
        active_split = _resolve_split(
            path=path,
            split=requested_split,
            loader=loader,
            config=config,
            revision=revision,
        )
        dataset = _load_dataset(
            path=path,
            loader=loader,
            split=active_split,
            config=config,
            revision=revision,
            streaming=streaming,
        )
        if streaming:
            rows = _materialize_rows_stream(dataset, limit)
        else:
            rows = _materialize_rows(dataset, limit)
        return (dataset, rows)


NODE_CLASS_MAPPINGS = {
    "LoadHuggingFaceDataset": LoadHuggingFaceDataset,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadHuggingFaceDataset": "🤗 Dataset Loader",
}
