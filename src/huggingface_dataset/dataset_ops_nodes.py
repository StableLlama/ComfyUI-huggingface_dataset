"""ComfyUI nodes that operate on a ``HUGGINGFACE_DATASET`` value.

The ``LoadHuggingFaceDataset`` node exposes the split it loads as an opaque
``HUGGINGFACE_DATASET`` value (a ``datasets.Dataset``, or a lazy
``datasets.IterableDataset`` when ``streaming`` is enabled). The nodes in this
module consume that value and expose the data-wrangling methods of the
``datasets`` library as graph nodes, so datasets can be transformed *before*
they are turned into a *Data List* (for further processing with the "Basic data
handling" pack) or exported to a file.

Two kinds of nodes live here:

* **Transform nodes** take a ``HUGGINGFACE_DATASET`` and hand back a new one
  (they call the matching method on the underlying ``datasets`` object, so
  chains keep working on ``Dataset`` *and* ``IterableDataset`` where the
  operation exists). Operations that only exist on a fully-loaded
  ``datasets.Dataset`` (``sort``, ``select`` by index, ``flatten``,
  ``train_test_split``, ``unique``) raise a friendly error when given a
  streaming dataset instead of failing with an opaque traceback.

* **Conversion nodes** turn a ``HUGGINGFACE_DATASET`` into plain Python data:
  a Basic-data-handling ``LIST`` (one Python list value) or a ComfyUI *Data
  List* (``OUTPUT_IS_LIST``) of row dicts - or of a single column's values when
  a ``column`` is given - so rows can be processed further with the nodes of
  the **Basic data handling** pack. Image cells become ComfyUI ``IMAGE``
  batches of one, so an image entry is valid on its own and can also be wired
  straight into the image nodes.

Because the objects are opaque and the operations are exposed declaratively
(widgets, no free-form Python), none of these nodes need to import the
``datasets`` package itself: they only call methods on the object they receive.

The conversion nodes make sure every value they hand out is valid on its own in
ComfyUI: ``datasets`` hands image cells over as PIL images (lazy JPEG XL ones
included), which ComfyUI's image nodes cannot consume, so those become a regular
ComfyUI ``IMAGE`` batch of one (``[1, H, W, C]``) - built with the
Pillow/numpy/torch stack ComfyUI already ships, imported lazily. An image column
whose cells carry an alpha channel additionally gets a virtual
``<column>_mask`` field holding the alpha as a real ComfyUI ``MASK`` (``1`` =
transparent), so transparency is never lost silently and no non-standard
4-channel trick is needed.
"""

import io
import logging
from inspect import cleandoc
from typing import Any

from .dataset_nodes import ComfyNodeABC, IO
from .dataset_nodes import INT_MAX
from .dataset_nodes import _materialize_rows, _materialize_rows_stream, _to_python

#: The opaque type the loader outputs and every node here consumes.
HUGGINGFACE_DATASET = "HUGGINGFACE_DATASET"

#: Category shared with the loader, so all dataset nodes group together.
_CATEGORY = "Hugging Face 🤗"

#: Declarative operators offered by the filter node. Compare/membership
#: operators coerce the widget ``value`` to the type of the actual cell.
_FILTER_OPERATORS = (
    "==",
    "!=",
    "<",
    "<=",
    ">",
    ">=",
    "contains",
    "not contains",
    "starts with",
    "ends with",
    "in",
    "not in",
    "is null",
    "is not null",
)

#: Operators that need a comparison value from the ``value`` widget.
_FILTER_VALUE_OPS = (
    "==",
    "!=",
    "<",
    "<=",
    ">",
    ">=",
    "contains",
    "not contains",
    "starts with",
    "ends with",
    "in",
    "not in",
)

#: Operations for adding/replacing a column in the map node.
_COLUMN_OPS = ("constant", "copy column", "row index")

#: Logger used for user-facing notes (e.g. a renamed virtual mask column).
_LOGGER = logging.getLogger("huggingface_dataset")


# --------------------------------------------------------------------------- #
# Shared helpers
# --------------------------------------------------------------------------- #


def _require_method(dataset: Any, name: str) -> None:
    """Raise a friendly error unless ``dataset`` looks like a real dataset object.

    The ``dataset`` output of the loader is an opaque ``datasets.Dataset`` /
    ``datasets.IterableDataset``. If a user accidentally wires something else in
    (for example the loader's *Data List* ``rows`` output, which expands per
    row), calling a method on it would produce a confusing traceback; this guard
    turns that into a clear message instead.
    """
    if not callable(getattr(dataset, name, None)):
        raise ValueError(
            "The value passed to this node is not a Hugging Face dataset "
            f"(it has no {name!r} method). Connect the 'dataset' output of the "
            "'🤗 Dataset Loader' node, or of another dataset node."
        )


def _is_streaming(dataset: Any) -> bool:
    """Return whether ``dataset`` is a lazy (streaming) dataset.

    ``datasets.IterableDataset`` (what ``streaming=True`` returns) deliberately
    has no ``__len__``; a fully-loaded ``datasets.Dataset`` does. Duck-typed so
    ``datasets`` never has to be imported here.
    """
    return not callable(getattr(dataset, "__len__", None))


def _ensure_materialized(dataset: Any) -> None:
    """Raise a friendly error when an operation needs a fully-loaded dataset."""
    if _is_streaming(dataset):
        raise ValueError(
            "This operation needs a fully-loaded (materialized) dataset, but the input is a "
            "streaming one (datasets.IterableDataset). Disable 'streaming' on the 'Hugging "
            "Face Dataset Loader' node that created this dataset."
        )


def _columns(dataset: Any) -> list[str] | None:
    """Best-effort list of the dataset's columns, or ``None`` when unknown.

    A streaming dataset loaded without known features reports no columns.
    """
    try:
        columns = getattr(dataset, "column_names", None)
    except Exception:  # noqa: BLE001 - be tolerant of exotic dataset objects
        return None
    if isinstance(columns, (list, tuple)):
        return [str(name) for name in columns]
    return None


def _check_column(dataset: Any, column: str, *, subject: str = "column") -> None:
    """Validate ``column`` is non-empty and (when known) present on ``dataset``."""
    column = (column or "").strip()
    if not column:
        raise ValueError(f"Provide a {subject} to operate on.")
    available = _columns(dataset)
    if available is not None and column not in available:
        listed = ", ".join(available)
        raise ValueError(f"Unknown {subject} {column!r}: this dataset has columns: {listed}.")


def _parse_columns(text: str) -> list[str]:
    """Split a comma-separated column list into trimmed names."""
    columns = [name.strip() for name in (text or "").split(",") if name.strip()]
    if not columns:
        raise ValueError("Provide at least one column name (comma-separated for several).")
    return columns


def _check_columns_known(dataset: Any, columns: list[str], *, subject: str = "columns") -> None:
    """When the dataset's columns are known, make sure every requested one exists."""
    available = _columns(dataset)
    if available is None:
        return
    unknown = [name for name in columns if name not in available]
    if unknown:
        listed = ", ".join(available)
        raise ValueError(f"Unknown {subject} {', '.join(repr(name) for name in unknown)}: this dataset has columns: {listed}.")


def _parse_indices(text: str, length: int) -> list[int]:
    """Parse a row-selection expression into a list of explicit indices.

    Accepts comma-separated indices (``0, 2, 4``) and/or Python-style slices
    (``0:100``, ``0:100:2``); an open-ended slice runs to the last row. Used for
    ``datasets.Dataset.select``, which needs a concrete index iterable.
    """
    tokens = [token.strip() for token in (text or "").split(",") if token.strip()]
    if not tokens:
        raise ValueError("Provide row indices, e.g. '0,2,4' or a slice like '0:100' or '0:100:2'.")

    def _to_int(part: str, default: int | None) -> int | None:
        part = part.strip()
        if not part:
            return default
        try:
            return int(part)
        except ValueError as exc:
            raise ValueError(f"Bad row index {token!r} (part {part!r} is not an integer).") from exc

    indices: list[int] = []
    for token in tokens:
        if ":" in token:
            parts = token.split(":")
            if len(parts) > 3:
                raise ValueError(f"Bad row slice {token!r}: use 'start:stop[:step]'.")
            start = _to_int(parts[0], 0)
            stop = _to_int(parts[1], length) if len(parts) > 1 else length
            step = _to_int(parts[2], 1) if len(parts) > 2 else 1
            assert start is not None and stop is not None and step is not None
            if step == 0:
                raise ValueError(f"Bad row slice {token!r}: step cannot be 0.")
            indices.extend(range(start, stop, step))
        else:
            try:
                indices.append(int(token))
            except ValueError as exc:
                raise ValueError(f"Bad row index {token!r}: not an integer.") from exc
    for index in indices:
        if index < 0 or index >= length:
            raise ValueError(f"Row index {index} is out of range (dataset has {length} rows).")
    return indices


def _typed_value(value: str, actual: Any) -> Any:
    """Best-effort coerce a widget ``value`` to the type of an actual cell.

    Lets ``5 == "5"`` and ``3 < "4"`` compare numerically while leaving text
    columns alone. Falls back to the raw string when the value does not parse.
    """
    value = value.strip()
    if isinstance(actual, bool):
        lowered = value.lower()
        if lowered in ("1", "true", "yes", "y"):
            return True
        if lowered in ("0", "false", "no", "n", ""):
            return False
        return value
    if isinstance(actual, int):
        try:
            return int(value)
        except ValueError:
            try:
                return float(value)
            except ValueError:
                return value
    if isinstance(actual, float):
        try:
            return float(value)
        except ValueError:
            return value
    return value


def _compare(operator: str, actual: Any, typed: Any) -> bool:
    """Apply a comparison operator, tolerating non-comparable types."""
    try:
        if operator == "==":
            return bool(actual == typed)
        if operator == "!=":
            return bool(actual != typed)
        if operator == "<":
            return bool(actual < typed)
        if operator == "<=":
            return bool(actual <= typed)
        if operator == ">":
            return bool(actual > typed)
        return bool(actual >= typed)
    except TypeError:
        return False


def _build_filter_predicate(column: str, operator: str, value: str) -> Any:
    """Return a ``row -> bool`` predicate implementing a declarative filter.

    ``column`` must be non-empty. Operators in :data:`_FILTER_VALUE_OPS` need a
    ``value``; ``is null`` / ``is not null`` ignore it.
    """
    column = column.strip()
    if not column:
        raise ValueError("Provide a 'column' to filter on.")

    if operator in _FILTER_VALUE_OPS and not value.strip() and operator not in ("==", "!=", "in", "not in"):
        raise ValueError(f"The filter operator {operator!r} needs a 'value' to compare against.")
    if operator in ("in", "not in") and not value.strip():
        raise ValueError(f"The filter operator {operator!r} needs comma-separated values, e.g. 'a, b, c'.")

    if operator in ("==", "!=", "<", "<=", ">", ">="):

        def predicate(row: dict[str, Any]) -> bool:
            try:
                actual = row[column]
            except (KeyError, TypeError):
                return False
            return _compare(operator, actual, _typed_value(value, actual))

        return predicate

    if operator in ("contains", "not contains", "starts with", "ends with"):
        needle = value

        def predicate(row: dict[str, Any]) -> bool:
            try:
                text = str(row[column])
            except (KeyError, TypeError):
                return False
            if operator == "contains":
                return needle in text
            if operator == "not contains":
                return needle not in text
            if operator == "starts with":
                return text.startswith(needle)
            return text.endswith(needle)

        return predicate

    if operator in ("in", "not in"):
        allowed = [part.strip() for part in value.split(",") if part.strip()]

        def predicate(row: dict[str, Any]) -> bool:
            try:
                actual = row[column]
            except (KeyError, TypeError):
                return False
            matches = any(_typed_value(part, actual) == actual for part in allowed)
            return matches if operator == "in" else not matches

        return predicate

    if operator in ("is null", "is not null"):

        def predicate(row: dict[str, Any]) -> bool:
            try:
                is_null = row.get(column) is None
            except (KeyError, TypeError):
                is_null = True
            return is_null if operator == "is null" else not is_null

        return predicate

    raise ValueError(f"Unknown filter operator {operator!r}.")


def _identity(value: Any) -> Any:
    """Return ``value`` unchanged - the "materialize raw values" conversion."""
    return value


def _raw_rows(dataset: Any, limit: int = -1) -> list[dict[str, Any]]:
    """Materialize up to ``limit`` rows of a dataset (either kind), values left raw.

    The conversion nodes need the untouched cells: an image cell becomes an
    ``IMAGE`` *and* - when it carries an alpha channel - a ``MASK``, which can
    only be decided from the cell itself.
    """
    if _is_streaming(dataset):
        return _materialize_rows_stream(dataset, limit, convert=_identity)
    return _materialize_rows(dataset, limit, convert=_identity)


def _column_help(dataset: Any, masks: dict[str, str]) -> str:
    """Describe the columns a user may ask for, including the virtual mask ones."""
    available = _columns(dataset) or []
    listed = ", ".join(available) or "unknown"
    if masks:
        virtual = ", ".join(sorted(masks.values()))
        return f"this dataset has columns: {listed} (virtual mask columns: {virtual})."
    return f"this dataset has columns: {listed}."


def _materialize_values(dataset: Any, column: str, limit: int) -> list[Any]:
    """Materialize up to ``limit`` values: whole rows, one column, or a virtual mask column.

    Without a ``column`` each row dict is returned with its values converted
    (image cells become ``IMAGE``, and an image column that carries an alpha
    channel additionally gets a virtual ``<column>_mask`` entry holding the
    ``MASK``). With a ``column`` only that column's values are converted, so
    asking for a text column of an image dataset stays cheap; ``column`` may
    also name a virtual mask column to get that column's masks.
    """
    active_column = (column or "").strip()
    rows = _raw_rows(dataset, limit)
    masks = _mask_columns(rows)
    if not active_column:
        return _convert_rows(rows, masks)

    if active_column in masks:
        # A real image column with alpha: its (RGB) images, the mask lives in the virtual column.
        return [_to_comfy_image(row[active_column]) for row in rows]

    available = _columns(dataset)
    if available is not None and active_column not in available:
        base = next((image_column for image_column, mask_name in masks.items() if mask_name == active_column), None)
        if base is not None:
            return [_to_comfy_mask(row[base]) for row in rows]
        raise ValueError(f"Unknown column {active_column!r}: {_column_help(dataset, masks)}")

    values: list[Any] = []
    for index, row in enumerate(rows):
        if active_column not in row:
            listed = ", ".join(str(key) for key in row) or "unknown"
            raise ValueError(f"Row {index} has no column {active_column!r}. Available columns there: {listed}.")
        values.append(_to_comfy_value(row[active_column]))
    return values


# --------------------------------------------------------------------------- #
# Row ordering / counting
# --------------------------------------------------------------------------- #


class HFDatasetShuffle(ComfyNodeABC):
    """Shuffles a Hugging Face dataset.

    Randomly reorders the rows (``datasets.Dataset.shuffle`` /
    ``datasets.IterableDataset.shuffle``). Pass a ``seed`` for reproducible
    shuffles; streaming datasets require one.
    """

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "dataset": (
                    HUGGINGFACE_DATASET,
                    {"tooltip": "Input dataset to shuffle (from the 🤗 Dataset Loader or another dataset node)."},
                ),
                "seed": (
                    IO.INT,
                    {
                        "default": 0,
                        "min": 0,
                        "max": INT_MAX,
                        "step": 1,
                        "tooltip": "Random seed for a reproducible shuffle; required when the input is streaming.",
                    },
                ),
            }
        }

    RETURN_TYPES = (HUGGINGFACE_DATASET,)
    RETURN_NAMES = ("dataset",)
    OUTPUT_TOOLTIPS = ("The shuffled dataset.",)
    CATEGORY = _CATEGORY
    DESCRIPTION = cleandoc(__doc__ or "")
    FUNCTION = "shuffle"

    def shuffle(self, dataset: Any, seed: int) -> tuple[Any]:
        _require_method(dataset, "shuffle")
        return (dataset.shuffle(seed=int(seed)),)


class HFDatasetSkip(ComfyNodeABC):
    """Skips the first ``n`` rows of a Hugging Face dataset.

    Returns a new dataset without the first ``n`` rows (``skip``). Works on both
    fully-loaded and streaming datasets.
    """

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "dataset": (
                    HUGGINGFACE_DATASET,
                    {"tooltip": "Input dataset to drop rows from (from the 🤗 Dataset Loader or another dataset node)."},
                ),
                "n": (
                    IO.INT,
                    {
                        "default": 1,
                        "min": 0,
                        "max": INT_MAX,
                        "step": 1,
                        "tooltip": "Number of leading rows to drop.",
                    },
                ),
            }
        }

    RETURN_TYPES = (HUGGINGFACE_DATASET,)
    RETURN_NAMES = ("dataset",)
    OUTPUT_TOOLTIPS = ("The dataset without its first n rows.",)
    CATEGORY = _CATEGORY
    DESCRIPTION = cleandoc(__doc__ or "")
    FUNCTION = "skip_rows"

    def skip_rows(self, dataset: Any, n: int) -> tuple[Any]:
        _require_method(dataset, "skip")
        return (dataset.skip(int(n)),)


class HFDatasetTake(ComfyNodeABC):
    """Keeps only the first ``n`` rows of a Hugging Face dataset.

    Returns a new dataset limited to the first ``n`` rows (``take``). Works on
    both fully-loaded and streaming datasets. Combine with
    ``HFDatasetToDataList`` to cap how much of a large dataset is materialized.
    """

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "dataset": (
                    HUGGINGFACE_DATASET,
                    {"tooltip": "Input dataset to keep rows from (from the 🤗 Dataset Loader or another dataset node)."},
                ),
                "n": (
                    IO.INT,
                    {
                        "default": 100,
                        "min": 1,
                        "max": INT_MAX,
                        "step": 1,
                        "tooltip": "Number of leading rows to keep.",
                    },
                ),
            }
        }

    RETURN_TYPES = (HUGGINGFACE_DATASET,)
    RETURN_NAMES = ("dataset",)
    OUTPUT_TOOLTIPS = ("The dataset limited to its first n rows.",)
    CATEGORY = _CATEGORY
    DESCRIPTION = cleandoc(__doc__ or "")
    FUNCTION = "take_rows"

    def take_rows(self, dataset: Any, n: int) -> tuple[Any]:
        _require_method(dataset, "take")
        return (dataset.take(int(n)),)


class HFDatasetSort(ComfyNodeABC):
    """Sorts a fully-loaded dataset by a column.

    Sorts the rows by ``column`` (``datasets.Dataset.sort``). Only available on
    a fully-loaded dataset - streaming datasets cannot be sorted this way, so
    disable ``streaming`` on the loader (or load the split normally).
    """

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "dataset": (
                    HUGGINGFACE_DATASET,
                    {"tooltip": "Fully-loaded dataset to sort (from the 🤗 Dataset Loader or another dataset node)."},
                ),
                "column": (
                    IO.STRING,
                    {"default": "", "multiline": False, "tooltip": "Column to sort the rows by."},
                ),
                "reverse": (
                    IO.BOOLEAN,
                    {"default": False, "tooltip": "Sort in descending order when enabled."},
                ),
            }
        }

    RETURN_TYPES = (HUGGINGFACE_DATASET,)
    RETURN_NAMES = ("dataset",)
    OUTPUT_TOOLTIPS = ("The sorted dataset.",)
    CATEGORY = _CATEGORY
    DESCRIPTION = cleandoc(__doc__ or "")
    FUNCTION = "sort_rows"

    def sort_rows(self, dataset: Any, column: str, reverse: bool) -> tuple[Any]:
        _ensure_materialized(dataset)
        _require_method(dataset, "sort")
        _check_column(dataset, column, subject="sort column")
        return (dataset.sort(column_names=column.strip(), reverse=bool(reverse)),)


class HFDatasetShard(ComfyNodeABC):
    """Splits a dataset into ``num_shards`` and keeps shard ``index``.

    Hands back the ``index``-th shard of a dataset split into ``num_shards``
    roughly equal pieces (``shard``). Works on both fully-loaded and streaming
    datasets.
    """

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "dataset": (
                    HUGGINGFACE_DATASET,
                    {"tooltip": "Dataset to split into shards (from the 🤗 Dataset Loader or another dataset node)."},
                ),
                "num_shards": (
                    IO.INT,
                    {
                        "default": 2,
                        "min": 1,
                        "max": INT_MAX,
                        "step": 1,
                        "tooltip": "How many roughly equal shards to split the dataset into.",
                    },
                ),
                "index": (
                    IO.INT,
                    {
                        "default": 0,
                        "min": 0,
                        "max": INT_MAX,
                        "step": 1,
                        "tooltip": "Which shard to keep (0 .. num_shards-1).",
                    },
                ),
                "contiguous": (
                    IO.BOOLEAN,
                    {"default": True, "tooltip": "Keep contiguous row blocks (true) or interleave rows (false)."},
                ),
            }
        }

    RETURN_TYPES = (HUGGINGFACE_DATASET,)
    RETURN_NAMES = ("dataset",)
    OUTPUT_TOOLTIPS = ("The requested shard of the dataset.",)
    CATEGORY = _CATEGORY
    DESCRIPTION = cleandoc(__doc__ or "")
    FUNCTION = "take_shard"

    def take_shard(self, dataset: Any, num_shards: int, index: int, contiguous: bool) -> tuple[Any]:
        _require_method(dataset, "shard")
        num_shards = int(num_shards)
        index = int(index)
        if index >= num_shards:
            raise ValueError(f"Shard index {index} must be smaller than num_shards ({num_shards}).")
        return (dataset.shard(num_shards=num_shards, index=index, contiguous=bool(contiguous)),)


# --------------------------------------------------------------------------- #
# Row selection (fully-loaded datasets only)
# --------------------------------------------------------------------------- #


class HFDatasetSelect(ComfyNodeABC):
    """Selects specific rows by index from a fully-loaded dataset.

    Keeps the rows whose indices match ``indices`` (``datasets.Dataset.select``).
    ``indices`` accepts comma-separated indices (``0, 2, 4``) and/or Python-style
    slices (``0:100``, ``0:100:2``); an open slice runs to the last row. Only
    available on a fully-loaded dataset - disable ``streaming`` on the loader.
    """

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "dataset": (
                    HUGGINGFACE_DATASET,
                    {"tooltip": "Fully-loaded dataset to select rows from (from the 🤗 Dataset Loader or another dataset node)."},
                ),
                "indices": (
                    IO.STRING,
                    {
                        "default": "0:100",
                        "multiline": False,
                        "tooltip": "Rows to keep: comma-separated indices (0, 2, 4) and/or Python slices (0:100, 0:100:2); an open slice runs to the last row.",
                    },
                ),
            }
        }

    RETURN_TYPES = (HUGGINGFACE_DATASET,)
    RETURN_NAMES = ("dataset",)
    OUTPUT_TOOLTIPS = ("The dataset with only the selected rows.",)
    CATEGORY = _CATEGORY
    DESCRIPTION = cleandoc(__doc__ or "")
    FUNCTION = "select_rows"

    def select_rows(self, dataset: Any, indices: str) -> tuple[Any]:
        _ensure_materialized(dataset)
        _require_method(dataset, "select")
        selected = _parse_indices(indices, len(dataset))
        return (dataset.select(selected),)


# --------------------------------------------------------------------------- #
# Column operations
# --------------------------------------------------------------------------- #


class HFDatasetSelectColumns(ComfyNodeABC):
    """Keeps only the listed columns of a dataset.

    ``columns`` is a comma-separated list (``select_columns``). Works on both
    fully-loaded and streaming datasets.
    """

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "dataset": (
                    HUGGINGFACE_DATASET,
                    {"tooltip": "Dataset to select columns from (from the 🤗 Dataset Loader or another dataset node)."},
                ),
                "columns": (
                    IO.STRING,
                    {
                        "default": "",
                        "multiline": False,
                        "tooltip": "Comma-separated columns to keep, e.g. text, label.",
                    },
                ),
            }
        }

    RETURN_TYPES = (HUGGINGFACE_DATASET,)
    RETURN_NAMES = ("dataset",)
    OUTPUT_TOOLTIPS = ("The dataset restricted to the given columns.",)
    CATEGORY = _CATEGORY
    DESCRIPTION = cleandoc(__doc__ or "")
    FUNCTION = "select_columns"

    def select_columns(self, dataset: Any, columns: str) -> tuple[Any]:
        _require_method(dataset, "select_columns")
        names = _parse_columns(columns)
        _check_columns_known(dataset, names)
        return (dataset.select_columns(names),)


class HFDatasetRemoveColumns(ComfyNodeABC):
    """Removes the listed columns from a dataset.

    ``columns`` is a comma-separated list (``remove_columns``). Works on both
    fully-loaded and streaming datasets.
    """

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "dataset": (
                    HUGGINGFACE_DATASET,
                    {"tooltip": "Dataset to remove columns from (from the 🤗 Dataset Loader or another dataset node)."},
                ),
                "columns": (
                    IO.STRING,
                    {
                        "default": "",
                        "multiline": False,
                        "tooltip": "Comma-separated columns to remove, e.g. text, label.",
                    },
                ),
            }
        }

    RETURN_TYPES = (HUGGINGFACE_DATASET,)
    RETURN_NAMES = ("dataset",)
    OUTPUT_TOOLTIPS = ("The dataset without the removed columns.",)
    CATEGORY = _CATEGORY
    DESCRIPTION = cleandoc(__doc__ or "")
    FUNCTION = "remove_columns"

    def remove_columns(self, dataset: Any, columns: str) -> tuple[Any]:
        _require_method(dataset, "remove_columns")
        names = _parse_columns(columns)
        _check_columns_known(dataset, names)
        available = _columns(dataset)
        if available is not None and not set(available) - set(names):
            raise ValueError("Removing every column would leave the dataset empty.")
        return (dataset.remove_columns(names),)


class HFDatasetRenameColumn(ComfyNodeABC):
    """Renames a single column of a dataset.

    Renames ``original_column`` to ``new_column`` (``rename_column``). Works on
    both fully-loaded and streaming datasets.
    """

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "dataset": (
                    HUGGINGFACE_DATASET,
                    {"tooltip": "Dataset whose column to rename (from the 🤗 Dataset Loader or another dataset node)."},
                ),
                "original_column": (
                    IO.STRING,
                    {"default": "", "multiline": False, "tooltip": "Name of the column to rename."},
                ),
                "new_column": (
                    IO.STRING,
                    {"default": "", "multiline": False, "tooltip": "New name for the column."},
                ),
            }
        }

    RETURN_TYPES = (HUGGINGFACE_DATASET,)
    RETURN_NAMES = ("dataset",)
    OUTPUT_TOOLTIPS = ("The dataset with the renamed column.",)
    CATEGORY = _CATEGORY
    DESCRIPTION = cleandoc(__doc__ or "")
    FUNCTION = "rename_column"

    def rename_column(self, dataset: Any, original_column: str, new_column: str) -> tuple[Any]:
        _require_method(dataset, "rename_column")
        original = (original_column or "").strip()
        new = (new_column or "").strip()
        if not original or not new:
            raise ValueError("Provide both an 'original_column' and a 'new_column' to rename.")
        if original == new:
            return (dataset,)
        _check_column(dataset, original, subject="column to rename")
        return (dataset.rename_column(original, new),)


class HFDatasetFlatten(ComfyNodeABC):
    """Flattens nested columns of a fully-loaded dataset.

    Expands columns holding nested dicts/lists into top-level columns
    (``datasets.Dataset.flatten``). Only available on a fully-loaded dataset -
    disable ``streaming`` on the loader.
    """

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "dataset": (
                    HUGGINGFACE_DATASET,
                    {"tooltip": "Fully-loaded dataset whose nested columns to flatten (from the 🤗 Dataset Loader)."},
                ),
            }
        }

    RETURN_TYPES = (HUGGINGFACE_DATASET,)
    RETURN_NAMES = ("dataset",)
    OUTPUT_TOOLTIPS = ("The dataset with nested columns expanded to top level.",)
    CATEGORY = _CATEGORY
    DESCRIPTION = cleandoc(__doc__ or "")
    FUNCTION = "flatten"

    def flatten(self, dataset: Any) -> tuple[Any]:
        _ensure_materialized(dataset)
        _require_method(dataset, "flatten")
        return (dataset.flatten(),)


# --------------------------------------------------------------------------- #
# Row filtering / mapping
# --------------------------------------------------------------------------- #


class HFDatasetFilter(ComfyNodeABC):
    """Filters rows of a dataset with a declarative condition.

    Keeps the rows of ``column`` that satisfy ``operator`` compared to
    ``value`` (``datasets.Dataset.filter`` / ``datasets.IterableDataset.filter``).

    Operators:
      - equality / ordering: ``==``, ``!=``, ``<``, ``<=``, ``>``, ``>=`` - the
        ``value`` text is coerced to the column's type, so numeric columns can
        be compared with plain numbers.
      - text: ``contains``, ``not contains``, ``starts with``, ``ends with``.
      - membership: ``in`` / ``not in`` with a comma-separated ``value`` list.
      - presence: ``is null`` / ``is not null`` (``value`` is ignored).

    Works on both fully-loaded and streaming datasets.
    """

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "dataset": (
                    HUGGINGFACE_DATASET,
                    {"tooltip": "Dataset to filter rows of (from the 🤗 Dataset Loader or another dataset node)."},
                ),
                "column": (
                    IO.STRING,
                    {"default": "", "multiline": False, "tooltip": "Column the condition is evaluated on."},
                ),
                "operator": (
                    _FILTER_OPERATORS,
                    {
                        "default": "==",
                        "tooltip": "Comparison: == != < <= > >=, contains / not contains / starts with / ends with, in / not in, is null / is not null.",
                    },
                ),
                "value": (
                    IO.STRING,
                    {
                        "default": "",
                        "multiline": False,
                        "tooltip": "Value to compare (comma-separated for 'in' / 'not in'); ignored for the null checks.",
                    },
                ),
            }
        }

    RETURN_TYPES = (HUGGINGFACE_DATASET,)
    RETURN_NAMES = ("dataset",)
    OUTPUT_TOOLTIPS = ("The dataset with only the matching rows.",)
    CATEGORY = _CATEGORY
    DESCRIPTION = cleandoc(__doc__ or "")
    FUNCTION = "filter_rows"

    def filter_rows(self, dataset: Any, column: str, operator: str, value: str) -> tuple[Any]:
        _require_method(dataset, "filter")
        _check_column(dataset, column)
        predicate = _build_filter_predicate(column, operator, value)
        return (dataset.filter(predicate),)


class HFDatasetMapColumn(ComfyNodeABC):
    """Adds or replaces a column computed per row (``datasets`` ``map``).

    Adds (or replaces) ``column`` on every row, with values chosen by
    ``operation``:
      - ``constant``: every row gets the literal ``value`` text.
      - ``copy column``: ``value`` is the name of an existing column whose
        values are copied into ``column``.
      - ``row index``: every row gets its 0-based row index as an integer.

    Works on both fully-loaded and streaming datasets. Replacing an existing
    column with a value of another type is supported (the old column is dropped
    first).
    """

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "dataset": (
                    HUGGINGFACE_DATASET,
                    {"tooltip": "Dataset whose column to add or replace (from the 🤗 Dataset Loader or another dataset node)."},
                ),
                "column": (
                    IO.STRING,
                    {"default": "new_column", "multiline": False, "tooltip": "Name of the column to add or replace."},
                ),
                "operation": (
                    _COLUMN_OPS,
                    {"default": "constant", "tooltip": "How to compute the value: constant, copy column, or row index."},
                ),
                "value": (
                    IO.STRING,
                    {
                        "default": "",
                        "multiline": False,
                        "tooltip": "Literal value for 'constant', or the source column name for 'copy column'.",
                    },
                ),
            }
        }

    RETURN_TYPES = (HUGGINGFACE_DATASET,)
    RETURN_NAMES = ("dataset",)
    OUTPUT_TOOLTIPS = ("The dataset with the added or replaced column.",)
    CATEGORY = _CATEGORY
    DESCRIPTION = cleandoc(__doc__ or "")
    FUNCTION = "map_column"

    def map_column(self, dataset: Any, column: str, operation: str, value: str) -> tuple[Any]:
        _require_method(dataset, "map")
        target = (column or "").strip()
        if not target:
            raise ValueError("Provide a name for the column to add or replace ('column').")
        operation = operation.strip()

        if operation == "copy column":
            _check_column(dataset, value, subject="source column")
            if value.strip() == target:
                return (dataset,)  # copying a column onto itself is a no-op

        # Replacing an existing column with a value of a different type: drop the
        # old column first so datasets can infer the new feature cleanly.
        existing = _columns(dataset)
        if existing is not None and target in existing:
            _require_method(dataset, "remove_columns")
            dataset = dataset.remove_columns([target])

        if operation == "row index":

            def with_index(row: dict[str, Any], index: int) -> dict[str, Any]:
                out = dict(row)
                out[target] = index
                return out

            return (dataset.map(with_index, with_indices=True),)

        if operation == "constant":
            constant = value

            def with_constant(row: dict[str, Any]) -> dict[str, Any]:
                out = dict(row)
                out[target] = constant
                return out

            return (dataset.map(with_constant),)

        if operation == "copy column":
            source = value.strip()

            def with_copy(row: dict[str, Any]) -> dict[str, Any]:
                out = dict(row)
                out[target] = row[source]
                return out

            return (dataset.map(with_copy),)

        raise ValueError(f"Unknown column operation {operation!r}.")


class HFDatasetSplit(ComfyNodeABC):
    """Splits a fully-loaded dataset into train and test parts.

    Randomly splits the rows into ``train`` and ``test`` datasets
    (``datasets.Dataset.train_test_split``). ``test_size`` is the fraction of
    rows held out for ``test``. Only available on a fully-loaded dataset -
    disable ``streaming`` on the loader.
    """

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "dataset": (
                    HUGGINGFACE_DATASET,
                    {"tooltip": "Fully-loaded dataset to split (from the 🤗 Dataset Loader or another dataset node)."},
                ),
                "test_size": (
                    IO.FLOAT,
                    {
                        "default": 0.2,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.05,
                        "tooltip": "Fraction of rows held out for the 'test' part (0.0 - 1.0).",
                    },
                ),
                "seed": (
                    IO.INT,
                    {
                        "default": 0,
                        "min": 0,
                        "max": INT_MAX,
                        "step": 1,
                        "tooltip": "Random seed for a reproducible split.",
                    },
                ),
                "shuffle": (
                    IO.BOOLEAN,
                    {"default": True, "tooltip": "Shuffle the rows before splitting when enabled."},
                ),
            }
        }

    RETURN_TYPES = (HUGGINGFACE_DATASET, HUGGINGFACE_DATASET)
    RETURN_NAMES = ("train", "test")
    OUTPUT_TOOLTIPS = ("The training part of the split.", "The held-out test part of the split.")
    CATEGORY = _CATEGORY
    DESCRIPTION = cleandoc(__doc__ or "")
    FUNCTION = "train_test_split"

    def train_test_split(self, dataset: Any, test_size: float, seed: int, shuffle: bool) -> tuple[Any, Any]:
        _ensure_materialized(dataset)
        _require_method(dataset, "train_test_split")
        split = dataset.train_test_split(test_size=float(test_size), seed=int(seed), shuffle=bool(shuffle))
        return (split["train"], split["test"])


# --------------------------------------------------------------------------- #
# Introspection & conversions
# --------------------------------------------------------------------------- #


class HFDatasetCount(ComfyNodeABC):
    """Returns the number of entries (rows) of a fully-loaded dataset.

    Counts the rows of the dataset (``len``) and hands the number back as an
    INT. Only available on a fully-loaded dataset - a streaming one has no
    ``len`` and counting it would consume every row - so disable ``streaming``
    on the loader (or feed in a materialized output of another dataset node).
    """

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "dataset": (
                    HUGGINGFACE_DATASET,
                    {"tooltip": "Fully-loaded dataset whose number of entries to count (from the 🤗 Dataset Loader or another dataset node)."},
                ),
            }
        }

    RETURN_TYPES = (IO.INT,)
    RETURN_NAMES = ("count",)
    OUTPUT_TOOLTIPS = ("The number of rows (entries) in the dataset.",)
    CATEGORY = _CATEGORY
    DESCRIPTION = cleandoc(__doc__ or "")
    FUNCTION = "count_rows"

    def count_rows(self, dataset: Any) -> tuple[int]:
        _ensure_materialized(dataset)
        return (len(dataset),)


class HFDatasetUnique(ComfyNodeABC):
    """Returns the unique values of a column of a fully-loaded dataset.

    Computes the distinct values present in ``column``
    (``datasets.Dataset.unique``) and hands them back as a *Data List* of plain
    values. Only available on a fully-loaded dataset - disable ``streaming`` on
    the loader.
    """

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "dataset": (
                    HUGGINGFACE_DATASET,
                    {"tooltip": "Fully-loaded dataset whose distinct values to collect (from the 🤗 Dataset Loader)."},
                ),
                "column": (
                    IO.STRING,
                    {"default": "", "multiline": False, "tooltip": "Column whose unique values to return."},
                ),
            }
        }

    RETURN_TYPES = (IO.ANY,)
    RETURN_NAMES = ("values",)
    OUTPUT_IS_LIST = (True,)
    OUTPUT_TOOLTIPS = ("Data List of the distinct values found in the column.",)
    CATEGORY = _CATEGORY
    DESCRIPTION = cleandoc(__doc__ or "")
    FUNCTION = "unique_values"

    def unique_values(self, dataset: Any, column: str) -> tuple[list[Any]]:
        _ensure_materialized(dataset)
        _require_method(dataset, "unique")
        _check_column(dataset, column)
        values = [_to_python(value) for value in dataset.unique(column.strip())]
        return (values,)


class HFDatasetToList(ComfyNodeABC):
    """Converts a Hugging Face dataset into a Basic-data-handling LIST.

    Materializes the dataset into one Python list - a *LIST* as defined by the
    **Basic data handling** pack - so it can feed nodes such as ``List length``,
    ``List get item``, etc. Without a ``column`` each item is a row dict; with a
    ``column`` each item is that column's value. ``limit`` caps how many rows are
    materialized (``-1`` = all). Works on fully-loaded and streaming datasets.

    Every item is converted into a value that is valid on its own in ComfyUI:
    numpy scalars/arrays become plain Python, image cells (PIL images - lazy
    JPEG XL ones included - undecoded ``{"bytes": ..., "path": ...}`` mappings,
    or raw encoded bytes) become a single-image ``IMAGE`` batch, and an image
    column whose cells carry an alpha channel additionally gets a virtual
    ``<column>_mask`` entry holding the alpha as a ``MASK`` (``1`` =
    transparent). Only the requested column is converted, so asking for a text
    column of an image dataset stays cheap.
    """

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "dataset": (
                    HUGGINGFACE_DATASET,
                    {"tooltip": "Dataset to materialize (from the 🤗 Dataset Loader or another dataset node)."},
                ),
                "column": (
                    IO.STRING,
                    {
                        "default": "",
                        "multiline": False,
                        "tooltip": "When set, each list item is that column's value instead of a whole row dict (a virtual '<image>_mask' column can be requested too).",
                    },
                ),
                "limit": (
                    IO.INT,
                    {
                        "default": -1,
                        "min": -1,
                        "max": INT_MAX,
                        "step": 1,
                        "tooltip": "Max rows to materialize; -1 = all.",
                    },
                ),
            }
        }

    RETURN_TYPES = ("LIST",)
    RETURN_NAMES = ("list",)
    OUTPUT_TOOLTIPS = ("One Python list of all rows (or of a column's values), as a Basic-data-handling LIST.",)
    CATEGORY = _CATEGORY
    DESCRIPTION = cleandoc(__doc__ or "")
    FUNCTION = "to_list"

    def to_list(self, dataset: Any, column: str, limit: int) -> tuple[list[Any]]:
        _require_method(dataset, "map")
        return (_materialize_values(dataset, column, int(limit)),)


class HFDatasetToDataList(ComfyNodeABC):
    """Converts a Hugging Face dataset into a ComfyUI *Data List*.

    Materializes the dataset as a *Data List* of items - one item per row
    (or one value per row of ``column`` when one is given) - the same shape the
    loader's ``dataset`` output has once materialized. Downstream *Data List*
    nodes of the **Basic data handling** pack receive the whole list in one
    call, while other nodes run once per item. ``limit`` caps how many rows are
    materialized (``-1`` = all). Works on fully-loaded and streaming datasets.

    Every item is converted into a value that is valid on its own in ComfyUI:
    numpy scalars/arrays become plain Python, image cells (PIL images - lazy
    JPEG XL ones included - undecoded ``{"bytes": ..., "path": ...}`` mappings,
    or raw encoded bytes) become a single-image ``IMAGE`` batch, so an image
    entry can be wired straight into the image nodes. An image column whose cells
    carry an alpha channel additionally gets a virtual ``<column>_mask`` entry
    holding the alpha as a ``MASK`` (``1`` = transparent), and a name collision
    with an existing column is resolved to ``<column>_mask1``, ``_mask2``, ...
    (logged on the console).
    """

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "dataset": (
                    HUGGINGFACE_DATASET,
                    {"tooltip": "Dataset to materialize (from the 🤗 Dataset Loader or another dataset node)."},
                ),
                "column": (
                    IO.STRING,
                    {
                        "default": "",
                        "multiline": False,
                        "tooltip": "When set, each item is that column's value instead of a whole row dict (a virtual '<image>_mask' column can be requested too).",
                    },
                ),
                "limit": (
                    IO.INT,
                    {
                        "default": -1,
                        "min": -1,
                        "max": INT_MAX,
                        "step": 1,
                        "tooltip": "Max rows to materialize; -1 = all.",
                    },
                ),
            }
        }

    RETURN_TYPES = (IO.ANY,)
    RETURN_NAMES = ("rows",)
    OUTPUT_IS_LIST = (True,)
    OUTPUT_TOOLTIPS = ("Data List of row dicts (or of a column's values), one item per row.",)
    CATEGORY = _CATEGORY
    DESCRIPTION = cleandoc(__doc__ or "")
    FUNCTION = "to_data_list"

    def to_data_list(self, dataset: Any, column: str, limit: int) -> tuple[list[Any]]:
        _require_method(dataset, "map")
        return (_materialize_values(dataset, column, int(limit)),)


# --------------------------------------------------------------------------- #
# Image conversion
# --------------------------------------------------------------------------- #


def _require_numpy() -> Any:
    """Import and return ``numpy`` (lazy import; ComfyUI always ships it)."""
    try:
        import numpy  # noqa: PLC0415 - deliberately imported lazily
    except ImportError as exc:  # pragma: no cover - ComfyUI always ships numpy
        raise ImportError("The 'numpy' package is required to convert dataset images into ComfyUI images.") from exc
    return numpy


def _require_torch() -> Any:
    """Import and return ``torch`` (lazy import; ComfyUI always ships it)."""
    try:
        import torch  # noqa: PLC0415 - deliberately imported lazily
    except ImportError as exc:  # pragma: no cover - ComfyUI always ships torch
        raise ImportError("ComfyUI's 'torch' package is required to convert dataset images into ComfyUI images.") from exc
    return torch


#: Cached ``PIL.Image.Image`` class (imported on first use, see :func:`_is_pil_image`).
_PIL_IMAGE_TYPE: Any = None


def _pil_image_from(value: Any) -> Any:
    """Return a PIL image for a dataset image cell.

    Handles every shape ``datasets`` produces for an image column: a decoded
    PIL image (lazy formats such as JPEG XL included, opened through
    ``pillow-jxl-plugin``), the undecoded ``{"bytes": ..., "path": ...}``
    mapping, and raw encoded image bytes.
    """
    from PIL import Image  # noqa: PLC0415 - Pillow ships with ComfyUI/datasets

    if isinstance(value, Image.Image):
        return value

    raw: Any = None
    if isinstance(value, dict):
        raw = value.get("bytes")
        if raw is None and value.get("path"):
            return Image.open(value["path"])
    else:
        raw = value
    if not isinstance(raw, (bytes, bytearray, memoryview)):
        raise ValueError(f"Cannot read an image from a {type(value).__name__}.")
    return Image.open(io.BytesIO(bytes(raw)))


def _is_image_mapping(value: Any) -> bool:
    """Whether ``value`` is an undecoded image cell (``{"bytes": .., "path": ..}``)."""
    return (
        isinstance(value, dict)
        and set(value) <= {"bytes", "path"}
        and (isinstance(value.get("bytes"), (bytes, bytearray, memoryview)) or bool(value.get("path")))
    )


def _is_pil_image(value: Any) -> bool:
    """Whether ``value`` is a PIL image (importing Pillow only when needed)."""
    global _PIL_IMAGE_TYPE
    if _PIL_IMAGE_TYPE is None:
        from PIL import Image  # noqa: PLC0415 - Pillow ships with ComfyUI/datasets

        _PIL_IMAGE_TYPE = Image.Image
    return isinstance(value, _PIL_IMAGE_TYPE)


def _to_comfy_image_and_mask(value: Any) -> tuple[Any, Any]:
    """Decode one image cell into a ComfyUI ``IMAGE`` batch of one and its ``MASK``.

    Returns ``([1, H, W, 3]`` float RGB ``0..1``, ``[1, H, W]`` float mask)``.
    The mask follows ComfyUI's convention - ``1`` = transparent - the same one
    ``LoadImage`` and ``SplitImageWithAlpha`` use, so it can be fed straight
    into the ``MASK`` inputs (``ImageCompositeMasked``, ``SetLatentNoiseMask``,
    ``GrowMask``, ...).
    """
    numpy = _require_numpy()
    torch = _require_torch()
    try:
        image = _pil_image_from(value)
        # ``numpy.array`` copies: Pillow hands out read-only buffers, which
        # ``torch.from_numpy`` cannot wrap.
        array = numpy.array(image.convert("RGBA"), dtype=numpy.uint8)
    except Exception as exc:  # noqa: BLE001 - surface a friendly decode error
        raise ValueError(
            f"Could not decode a dataset image: {exc}. For JPEG XL images install the optional "
            "'pillow-jxl-plugin' (pip install pillow-jxl-plugin)."
        ) from exc
    rgba = torch.from_numpy(array)
    images = rgba[..., :3].to(torch.float32).unsqueeze(0) / 255.0
    masks = (1.0 - rgba[..., 3].to(torch.float32) / 255.0).unsqueeze(0)
    return (images, masks)


def _to_comfy_image(value: Any) -> Any:
    """Decode one image cell into a ComfyUI ``IMAGE`` batch of one (RGB).

    The alpha channel - when there is one - is *not* dropped silently: it is
    exposed as its own virtual ``<column>_mask`` value (see
    :func:`_mask_columns`).
    """
    return _to_comfy_image_and_mask(value)[0]


def _to_comfy_mask(value: Any) -> Any:
    """Decode one image cell into its ComfyUI ``MASK`` (``[1, H, W]``, ``1`` = transparent)."""
    return _to_comfy_image_and_mask(value)[1]


def _is_image_cell(value: Any) -> bool:
    """Whether a raw dataset value is an image (PIL image, mapping, or encoded bytes)."""
    if value is None:
        return False
    if _is_pil_image(value) or _is_image_mapping(value):
        return True
    if isinstance(value, (bytes, bytearray, memoryview)):
        try:
            _pil_image_from(value)
        except Exception:  # noqa: BLE001 - just not an image
            return False
        return True
    return False


def _image_has_alpha(value: Any) -> bool:
    """Whether an image cell carries an alpha channel."""
    try:
        image = _pil_image_from(value)
        return "A" in image.getbands() or bool(getattr(image, "info", {}).get("transparency"))
    except Exception:  # noqa: BLE001 - not an image / not decodable
        return False


def _mask_columns(rows: list[dict[str, Any]]) -> dict[str, str]:
    """Map each image column that carries transparency to its virtual mask column name.

    The name is ``<column>_mask``. When the dataset already has a column of that
    name (or another virtual column claimed it) a counter is appended -
    ``<column>_mask1``, ``<column>_mask2``, ... - and the chosen name is logged
    to the ComfyUI console, because a workflow has to use it explicitly.
    """
    if not rows:
        return {}
    taken: set[str] = set()
    for row in rows:
        taken.update(row)

    masks: dict[str, str] = {}
    for column in rows[0]:
        images = [cell for cell in (row.get(column) for row in rows) if _is_image_cell(cell)]
        if not images or not any(_image_has_alpha(cell) for cell in images):
            continue
        name = f"{column}_mask"
        counter = 1
        while name in taken:
            name = f"{column}_mask{counter}"
            counter += 1
        if name != f"{column}_mask":
            _LOGGER.info(
                "Another column is already named %r, so the alpha mask of %r is exposed as %r instead.",
                f"{column}_mask",
                column,
                name,
            )
        taken.add(name)
        masks[column] = name
    return masks


def _convert_rows(rows: list[dict[str, Any]], masks: dict[str, str]) -> list[dict[str, Any]]:
    """Convert raw rows into ComfyUI-valid values, adding the virtual mask columns."""
    converted: list[dict[str, Any]] = []
    for row in rows:
        out: dict[str, Any] = {}
        for key, value in row.items():
            mask_name = masks.get(key)
            if mask_name is None:
                out[key] = _to_comfy_value(value)
            else:
                out[key], out[mask_name] = _to_comfy_image_and_mask(value)
        converted.append(out)
    return converted


def _to_comfy_value(value: Any) -> Any:
    """Convert a materialized dataset value into ComfyUI-usable values.

    Does everything :func:`_to_python` does (numpy scalars/arrays, nested
    lists/dicts, ...) and additionally turns image cells into a ComfyUI
    ``IMAGE`` batch of one: a decoded PIL image (lazy JPEG XL ones included),
    the ``{"bytes": ..., "path": ...}`` mapping of an undecoded image cell, or
    raw encoded image bytes. Every *Data List* / *LIST* entry therefore stays
    valid on its own, so an image entry can be wired straight into the image
    nodes.

    Bytes that are not an image keep the previous behaviour (UTF-8 text or a hex
    string) via :func:`_to_python`.
    """
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if _is_pil_image(value) or _is_image_mapping(value):
        return _to_comfy_image(value)
    if isinstance(value, dict):
        return {key: _to_comfy_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_comfy_value(item) for item in value]
    if isinstance(value, (bytes, bytearray, memoryview)):
        try:
            return _to_comfy_image(value)
        except ValueError:
            return _to_python(value)
    return _to_python(value)


NODE_CLASS_MAPPINGS = {
    "HFDatasetShuffle": HFDatasetShuffle,
    "HFDatasetSkip": HFDatasetSkip,
    "HFDatasetTake": HFDatasetTake,
    "HFDatasetSort": HFDatasetSort,
    "HFDatasetShard": HFDatasetShard,
    "HFDatasetSelect": HFDatasetSelect,
    "HFDatasetSelectColumns": HFDatasetSelectColumns,
    "HFDatasetRemoveColumns": HFDatasetRemoveColumns,
    "HFDatasetRenameColumn": HFDatasetRenameColumn,
    "HFDatasetFlatten": HFDatasetFlatten,
    "HFDatasetFilter": HFDatasetFilter,
    "HFDatasetMapColumn": HFDatasetMapColumn,
    "HFDatasetSplit": HFDatasetSplit,
    "HFDatasetCount": HFDatasetCount,
    "HFDatasetUnique": HFDatasetUnique,
    "HFDatasetToList": HFDatasetToList,
    "HFDatasetToDataList": HFDatasetToDataList,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "HFDatasetShuffle": "🤗 Dataset Shuffle",
    "HFDatasetSkip": "🤗 Dataset Skip",
    "HFDatasetTake": "🤗 Dataset Take",
    "HFDatasetSort": "🤗 Dataset Sort",
    "HFDatasetShard": "🤗 Dataset Shard",
    "HFDatasetSelect": "🤗 Dataset Select Rows",
    "HFDatasetSelectColumns": "🤗 Dataset Select Columns",
    "HFDatasetRemoveColumns": "🤗 Dataset Remove Columns",
    "HFDatasetRenameColumn": "🤗 Dataset Rename Column",
    "HFDatasetFlatten": "🤗 Dataset Flatten",
    "HFDatasetFilter": "🤗 Dataset Filter",
    "HFDatasetMapColumn": "🤗 Dataset Map Column",
    "HFDatasetSplit": "🤗 Dataset Train/Test Split",
    "HFDatasetCount": "🤗 Dataset Count",
    "HFDatasetUnique": "🤗 Dataset Unique",
    "HFDatasetToList": "🤗 Dataset To LIST",
    "HFDatasetToDataList": "🤗 Dataset To Data List",
}
