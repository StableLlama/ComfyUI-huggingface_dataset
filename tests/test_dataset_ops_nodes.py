"""Tests for the Hugging Face dataset transform/conversion nodes.

Like the loader tests, these run hermetically: no network and no real
``datasets`` install. The nodes operate on whatever object they receive (they
never import ``datasets``), so small behaviorful stand-ins that mimic the
``Dataset`` / ``IterableDataset`` method surface are enough to exercise them.
"""

from typing import Any

import pytest

from huggingface_dataset import NODE_CLASS_MAPPINGS as PACKAGE_MAPPINGS
from huggingface_dataset import NODE_DISPLAY_NAME_MAPPINGS as PACKAGE_DISPLAY_NAMES
from huggingface_dataset import dataset_ops_nodes as ops


def _rows(count: int) -> list[dict[str, Any]]:
    return [{"text": f"row-{index}", "label": index} for index in range(count)]


class FakeDataset:
    """Behaviorful stand-in for ``datasets.Dataset``.

    Implements just enough of the materialized-Dataset surface (``__len__``,
    slicing, ``column_names`` and the transform methods the nodes call) to test
    the glue. Every call is recorded on ``calls``; transforms return a new
    ``FakeDataset`` sharing the same call log so chaining keeps working.
    """

    def __init__(self, rows: list[dict[str, Any]]):
        self._rows = list(rows)
        self.calls: list[tuple[Any, ...]] = []

    def __len__(self) -> int:
        return len(self._rows)

    def __iter__(self) -> Any:
        return iter(self._rows)

    def __getitem__(self, key: Any) -> Any:
        if isinstance(key, slice):
            subset = self._rows[key]
            columns = list(self._rows[0]) if self._rows else []
            return {column: [row[column] for row in subset] for column in columns}
        return self._rows[key]

    @property
    def column_names(self) -> list[str]:
        return list(self._rows[0]) if self._rows else []

    def _clone(self, rows: list[dict[str, Any]]) -> "FakeDataset":
        result = FakeDataset(rows)
        result.calls = self.calls
        return result

    def shuffle(self, seed: int | None = None, **kwargs: Any) -> "FakeDataset":
        self.calls.append(("shuffle", seed, kwargs))
        return self._clone(list(self._rows))

    def skip(self, n: int, **kwargs: Any) -> "FakeDataset":
        self.calls.append(("skip", n, kwargs))
        return self._clone(self._rows[int(n) :])

    def take(self, n: int, **kwargs: Any) -> "FakeDataset":
        self.calls.append(("take", n, kwargs))
        return self._clone(self._rows[: int(n)])

    def sort(self, column_names: str | None = None, reverse: bool = False, **kwargs: Any) -> "FakeDataset":
        assert column_names is not None
        self.calls.append(("sort", column_names, reverse, kwargs))
        return self._clone(sorted(self._rows, key=lambda row: row[column_names], reverse=bool(reverse)))

    def shard(self, num_shards: int = 1, index: int = 0, contiguous: bool = True, **kwargs: Any) -> "FakeDataset":
        self.calls.append(("shard", num_shards, index, contiguous, kwargs))
        start = index * len(self._rows) // num_shards
        end = (index + 1) * len(self._rows) // num_shards
        return self._clone(self._rows[start:end])

    def select(self, indices: Any, **kwargs: Any) -> "FakeDataset":
        self.calls.append(("select", list(indices), kwargs))
        return self._clone([self._rows[int(index)] for index in indices])

    def select_columns(self, column_names: Any, **kwargs: Any) -> "FakeDataset":
        names = list(column_names)
        self.calls.append(("select_columns", names, kwargs))
        return self._clone([{name: row[name] for name in names} for row in self._rows])

    def remove_columns(self, column_names: Any, **kwargs: Any) -> "FakeDataset":
        names = set(column_names)
        self.calls.append(("remove_columns", list(names), kwargs))
        kept = [name for name in self.column_names if name not in names]
        return self._clone([{name: row[name] for name in kept} for row in self._rows])

    def rename_column(self, original_column_name: str, new_column_name: str, **kwargs: Any) -> "FakeDataset":
        self.calls.append(("rename_column", original_column_name, new_column_name, kwargs))
        rows: list[dict[str, Any]] = []
        for row in self._rows:
            rows.append({(new_column_name if name == original_column_name else name): value for name, value in row.items()})
        return self._clone(rows)

    def flatten(self, **kwargs: Any) -> "FakeDataset":
        self.calls.append(("flatten", kwargs))
        return self._clone(self._rows)

    def filter(self, function: Any = None, **kwargs: Any) -> "FakeDataset":
        self.calls.append(("filter", function, kwargs))
        if function is None:
            return self._clone(self._rows)
        return self._clone([row for row in self._rows if function(row)])

    def map(self, function: Any = None, with_indices: bool = False, **kwargs: Any) -> "FakeDataset":
        self.calls.append(("map", function, with_indices, kwargs))
        if function is None:
            return self._clone(self._rows)
        rows = []
        for index, row in enumerate(self._rows):
            rows.append(function(row, index) if with_indices else function(row))
        return self._clone(rows)

    def unique(self, column: str) -> list[Any]:
        self.calls.append(("unique", column, {}))
        seen: list[Any] = []
        for row in self._rows:
            if row[column] not in seen:
                seen.append(row[column])
        return seen

    def train_test_split(
        self, test_size: float | None = None, seed: int | None = None, shuffle: bool = True, **kwargs: Any
    ) -> dict[str, "FakeDataset"]:
        self.calls.append(("train_test_split", test_size, seed, shuffle, kwargs))
        count = len(self._rows)
        test_count = int(round(count * float(test_size))) if test_size is not None else 0
        cut = count - test_count
        return {"train": FakeDataset(self._rows[:cut]), "test": FakeDataset(self._rows[cut:])}


class FakeIterableDataset:
    """Behaviorful stand-in for a streaming ``datasets.IterableDataset``.

    Iterable, no ``__len__``/slicing, and only the transform methods that exist
    on the real streaming object.
    """

    def __init__(self, rows: list[dict[str, Any]]):
        self._rows = list(rows)
        self.calls: list[tuple[Any, ...]] = []

    def __iter__(self) -> Any:
        return iter(self._rows)

    @property
    def column_names(self) -> list[str]:
        return list(self._rows[0]) if self._rows else []

    def _clone(self, rows: list[dict[str, Any]]) -> "FakeIterableDataset":
        result = FakeIterableDataset(rows)
        result.calls = self.calls
        return result

    def shuffle(self, seed: int | None = None, **kwargs: Any) -> "FakeIterableDataset":
        self.calls.append(("shuffle", seed, kwargs))
        return self._clone(list(self._rows))

    def skip(self, n: int, **kwargs: Any) -> "FakeIterableDataset":
        self.calls.append(("skip", n, kwargs))
        return self._clone(self._rows[int(n) :])

    def take(self, n: int, **kwargs: Any) -> "FakeIterableDataset":
        self.calls.append(("take", n, kwargs))
        return self._clone(self._rows[: int(n)])

    def shard(self, num_shards: int = 1, index: int = 0, contiguous: bool = True, **kwargs: Any) -> "FakeIterableDataset":
        self.calls.append(("shard", num_shards, index, contiguous, kwargs))
        start = index * len(self._rows) // num_shards
        end = (index + 1) * len(self._rows) // num_shards
        return self._clone(self._rows[start:end])

    def select_columns(self, column_names: Any, **kwargs: Any) -> "FakeIterableDataset":
        names = list(column_names)
        self.calls.append(("select_columns", names, kwargs))
        return self._clone([{name: row[name] for name in names} for row in self._rows])

    def remove_columns(self, column_names: Any, **kwargs: Any) -> "FakeIterableDataset":
        names = set(column_names)
        self.calls.append(("remove_columns", list(names), kwargs))
        kept = [name for name in self.column_names if name not in names]
        return self._clone([{name: row[name] for name in kept} for row in self._rows])

    def rename_column(self, original_column_name: str, new_column_name: str, **kwargs: Any) -> "FakeIterableDataset":
        self.calls.append(("rename_column", original_column_name, new_column_name, kwargs))
        rows = [{(new_column_name if name == original_column_name else name): value for name, value in row.items()} for row in self._rows]
        return self._clone(rows)

    def filter(self, function: Any = None, **kwargs: Any) -> "FakeIterableDataset":
        self.calls.append(("filter", function, kwargs))
        if function is None:
            return self._clone(self._rows)
        return self._clone([row for row in self._rows if function(row)])

    def map(self, function: Any = None, with_indices: bool = False, **kwargs: Any) -> "FakeIterableDataset":
        self.calls.append(("map", function, with_indices, kwargs))
        if function is None:
            return self._clone(self._rows)
        rows = []
        for index, row in enumerate(self._rows):
            rows.append(function(row, index) if with_indices else function(row))
        return self._clone(rows)


# --------------------------------------------------------------------------- #
# Registration / metadata
# --------------------------------------------------------------------------- #


def test_all_ops_registered():
    assert set(ops.NODE_CLASS_MAPPINGS) == set(ops.NODE_DISPLAY_NAME_MAPPINGS)
    for name, node_class in ops.NODE_CLASS_MAPPINGS.items():
        assert name in PACKAGE_MAPPINGS
        assert PACKAGE_MAPPINGS[name] is node_class
        assert PACKAGE_DISPLAY_NAMES[name] == ops.NODE_DISPLAY_NAME_MAPPINGS[name]


def test_op_metadata_shape():
    dataset_nodes = [
        name
        for name in ops.NODE_CLASS_MAPPINGS
        if name not in ("HFDatasetSplit", "HFDatasetCount", "HFDatasetUnique", "HFDatasetToList", "HFDatasetToDataList")
    ]
    for name in dataset_nodes:
        node_class = ops.NODE_CLASS_MAPPINGS[name]
        inputs = node_class.INPUT_TYPES()["required"]
        assert "dataset" in inputs
        assert inputs["dataset"][0] == "HUGGINGFACE_DATASET"
        assert node_class.RETURN_TYPES == ("HUGGINGFACE_DATASET",)
        assert node_class.RETURN_NAMES == ("dataset",)
        assert node_class.CATEGORY == "Hugging Face 🤗"
        assert callable(getattr(node_class(), node_class.FUNCTION))


def test_ops_tooltips_present():
    for name, node_class in ops.NODE_CLASS_MAPPINGS.items():
        for kind in ("required", "optional"):
            for input_name, spec in node_class.INPUT_TYPES().get(kind, {}).items():
                if isinstance(spec, tuple) and len(spec) > 1:
                    options = spec[1]
                    assert options.get("tooltip"), f"{name}.{input_name} is missing a tooltip"
        tips = getattr(node_class, "OUTPUT_TOOLTIPS", None)
        assert tips is not None, f"{name} is missing OUTPUT_TOOLTIPS"
        assert len(tips) == len(node_class.RETURN_TYPES), f"{name} OUTPUT_TOOLTIPS length mismatch"
        assert all(tip.strip() for tip in tips)


def test_special_nodes_metadata():
    split = ops.HFDatasetSplit()
    assert split.RETURN_TYPES == ("HUGGINGFACE_DATASET", "HUGGINGFACE_DATASET")
    assert split.RETURN_NAMES == ("train", "test")

    count = ops.HFDatasetCount()
    assert count.RETURN_TYPES == (ops.IO.INT,)
    assert count.RETURN_NAMES == ("count",)

    unique = ops.HFDatasetUnique()
    assert unique.OUTPUT_IS_LIST == (True,)
    assert unique.RETURN_NAMES == ("values",)

    to_list = ops.HFDatasetToList()
    assert to_list.RETURN_TYPES == ("LIST",)
    assert to_list.RETURN_NAMES == ("list",)

    to_data_list = ops.HFDatasetToDataList()
    assert to_data_list.OUTPUT_IS_LIST == (True,)
    assert to_data_list.RETURN_NAMES == ("rows",)


# --------------------------------------------------------------------------- #
# Row ordering / counting
# --------------------------------------------------------------------------- #


def test_shuffle_forwards_seed():
    ds = FakeDataset(_rows(5))
    (out,) = ops.HFDatasetShuffle().shuffle(dataset=ds, seed=42)
    assert ds.calls[-1] == ("shuffle", 42, {})
    assert isinstance(out, FakeDataset)
    assert len(out) == 5


def test_skip_rows():
    ds = FakeDataset(_rows(5))
    (out,) = ops.HFDatasetSkip().skip_rows(dataset=ds, n=2)
    assert ds.calls[-1][0] == "skip"
    assert [row["text"] for row in out] == ["row-2", "row-3", "row-4"]


def test_take_rows():
    ds = FakeDataset(_rows(5))
    (out,) = ops.HFDatasetTake().take_rows(dataset=ds, n=2)
    assert ds.calls[-1][0] == "take"
    assert [row["text"] for row in out] == ["row-0", "row-1"]


def test_skip_and_take_work_on_streaming():
    stream = FakeIterableDataset(_rows(5))
    (skipped,) = ops.HFDatasetSkip().skip_rows(dataset=stream, n=1)
    (taken,) = ops.HFDatasetTake().take_rows(dataset=skipped, n=2)
    assert [row["text"] for row in taken] == ["row-1", "row-2"]


def test_sort_rows_on_materialized():
    ds = FakeDataset([{"label": 3}, {"label": 1}, {"label": 2}])
    (out,) = ops.HFDatasetSort().sort_rows(dataset=ds, column="label", reverse=True)
    assert ds.calls[-1] == ("sort", "label", True, {})
    assert [row["label"] for row in out] == [3, 2, 1]


def test_sort_rejects_streaming_and_unknown_column():
    with pytest.raises(ValueError, match="streaming"):
        ops.HFDatasetSort().sort_rows(dataset=FakeIterableDataset(_rows(3)), column="label", reverse=False)
    with pytest.raises(ValueError, match="Unknown sort column"):
        ops.HFDatasetSort().sort_rows(dataset=FakeDataset(_rows(3)), column="missing", reverse=False)


def test_shard_slices_contiguously():
    ds = FakeDataset(_rows(6))
    (out,) = ops.HFDatasetShard().take_shard(dataset=ds, num_shards=3, index=1, contiguous=True)
    assert ds.calls[-1][0] == "shard"
    assert [row["text"] for row in out] == ["row-2", "row-3"]


def test_shard_validates_index():
    with pytest.raises(ValueError, match="num_shards"):
        ops.HFDatasetShard().take_shard(dataset=FakeDataset(_rows(6)), num_shards=3, index=3, contiguous=True)


# --------------------------------------------------------------------------- #
# Row selection (materialized only)
# --------------------------------------------------------------------------- #


def test_select_rows_by_list_and_slice():
    ds = FakeDataset(_rows(10))
    (listed,) = ops.HFDatasetSelect().select_rows(dataset=ds, indices="0, 2, 4")
    assert [row["text"] for row in listed] == ["row-0", "row-2", "row-4"]
    (sliced,) = ops.HFDatasetSelect().select_rows(dataset=ds, indices="0:10:2")
    assert [row["text"] for row in sliced] == [f"row-{i}" for i in range(0, 10, 2)]
    assert ds.calls[-1][0] == "select"


def test_select_open_ended_slice_runs_to_end():
    ds = FakeDataset(_rows(10))
    (out,) = ops.HFDatasetSelect().select_rows(dataset=ds, indices="7:")
    assert [row["text"] for row in out] == ["row-7", "row-8", "row-9"]


def test_select_rejects_bad_and_out_of_range_indices():
    ds = FakeDataset(_rows(5))
    with pytest.raises(ValueError, match="out of range"):
        ops.HFDatasetSelect().select_rows(dataset=ds, indices="9")
    with pytest.raises(ValueError, match="not an integer"):
        ops.HFDatasetSelect().select_rows(dataset=ds, indices="a,b")
    with pytest.raises(ValueError, match="streaming"):
        ops.HFDatasetSelect().select_rows(dataset=FakeIterableDataset(_rows(5)), indices="0:3")


# --------------------------------------------------------------------------- #
# Column operations
# --------------------------------------------------------------------------- #


def test_select_columns():
    ds = FakeDataset(_rows(3))
    (out,) = ops.HFDatasetSelectColumns().select_columns(dataset=ds, columns="text")
    assert set(out.column_names) == {"text"}
    with pytest.raises(ValueError, match="Unknown columns"):
        ops.HFDatasetSelectColumns().select_columns(dataset=ds, columns="nope")


def test_remove_columns():
    ds = FakeDataset(_rows(3))
    (out,) = ops.HFDatasetRemoveColumns().remove_columns(dataset=ds, columns="label")
    assert set(out.column_names) == {"text"}
    with pytest.raises(ValueError, match="empty"):
        ops.HFDatasetRemoveColumns().remove_columns(dataset=ds, columns="text,label")


def test_rename_column():
    ds = FakeDataset(_rows(3))
    (out,) = ops.HFDatasetRenameColumn().rename_column(dataset=ds, original_column="label", new_column="score")
    assert "score" in out.column_names and "label" not in out.column_names
    assert [row["score"] for row in out] == [0, 1, 2]
    with pytest.raises(ValueError, match="Unknown column to rename"):
        ops.HFDatasetRenameColumn().rename_column(dataset=ds, original_column="nope", new_column="x")


def test_rename_same_name_is_noop():
    ds = FakeDataset(_rows(3))
    (out,) = ops.HFDatasetRenameColumn().rename_column(dataset=ds, original_column="label", new_column="label")
    assert out is ds
    assert not ds.calls


def test_flatten_on_materialized_only():
    ds = FakeDataset(_rows(2))
    (out,) = ops.HFDatasetFlatten().flatten(dataset=ds)
    assert ds.calls[-1][0] == "flatten"
    assert len(out) == 2
    with pytest.raises(ValueError, match="streaming"):
        ops.HFDatasetFlatten().flatten(dataset=FakeIterableDataset(_rows(2)))


# --------------------------------------------------------------------------- #
# Filtering
# --------------------------------------------------------------------------- #


def _filter_rows() -> list[dict[str, Any]]:
    return [
        {"text": "apple pie", "label": 1, "score": 4.0, "tag": "a"},
        {"text": "banana", "label": 2, "score": 5.0, "tag": "b"},
        {"text": "cherry pie", "label": 3, "score": 6.0, "tag": None},
    ]


def test_filter_equality_coerces_numbers():
    ds = FakeDataset(_filter_rows())
    (out,) = ops.HFDatasetFilter().filter_rows(dataset=ds, column="label", operator="==", value="2")
    assert [row["text"] for row in out] == ["banana"]


def test_filter_numeric_comparison():
    ds = FakeDataset(_filter_rows())
    (out,) = ops.HFDatasetFilter().filter_rows(dataset=ds, column="score", operator=">=", value="5.5")
    assert [row["text"] for row in out] == ["cherry pie"]


def test_filter_text_contains_and_bounds():
    ds = FakeDataset(_filter_rows())
    (contains,) = ops.HFDatasetFilter().filter_rows(dataset=ds, column="text", operator="contains", value="pie")
    assert [row["text"] for row in contains] == ["apple pie", "cherry pie"]
    (bounds,) = ops.HFDatasetFilter().filter_rows(dataset=ds, column="text", operator="starts with", value="b")
    assert [row["text"] for row in bounds] == ["banana"]


def test_filter_null_and_membership():
    ds = FakeDataset(_filter_rows())
    (nulls,) = ops.HFDatasetFilter().filter_rows(dataset=ds, column="tag", operator="is null", value="")
    assert [row["text"] for row in nulls] == ["cherry pie"]
    (member,) = ops.HFDatasetFilter().filter_rows(dataset=ds, column="tag", operator="in", value="a, b")
    assert [row["text"] for row in member] == ["apple pie", "banana"]


def test_filter_on_streaming_dataset():
    stream = FakeIterableDataset(_filter_rows())
    (out,) = ops.HFDatasetFilter().filter_rows(dataset=stream, column="label", operator=">", value="1")
    assert isinstance(out, FakeIterableDataset)
    assert [row["text"] for row in out] == ["banana", "cherry pie"]


def test_filter_validation_errors():
    ds = FakeDataset(_filter_rows())
    with pytest.raises(ValueError, match="column"):
        ops.HFDatasetFilter().filter_rows(dataset=ds, column="", operator="==", value="1")
    with pytest.raises(ValueError, match="Unknown column"):
        ops.HFDatasetFilter().filter_rows(dataset=ds, column="missing", operator="==", value="1")
    with pytest.raises(ValueError, match="needs a 'value'"):
        ops.HFDatasetFilter().filter_rows(dataset=ds, column="text", operator="contains", value="")
    with pytest.raises(ValueError, match="Unknown filter operator"):
        ops._build_filter_predicate("text", "bogus", "x")


# --------------------------------------------------------------------------- #
# Mapping a column
# --------------------------------------------------------------------------- #


def test_map_constant_adds_column():
    ds = FakeDataset(_rows(3))
    (out,) = ops.HFDatasetMapColumn().map_column(dataset=ds, column="kind", operation="constant", value="hi")
    assert ds.calls[-1][0] == "map"
    assert [row["kind"] for row in out] == ["hi", "hi", "hi"]


def test_map_constant_replaces_existing_column_of_other_type():
    ds = FakeDataset(_rows(3))
    (out,) = ops.HFDatasetMapColumn().map_column(dataset=ds, column="label", operation="constant", value="x")
    # The old int column is dropped first, then re-added as a string.
    assert ("remove_columns", ["label"], {}) in ds.calls
    assert [row["label"] for row in out] == ["x", "x", "x"]


def test_map_row_index_adds_ints():
    ds = FakeDataset(_rows(3))
    (out,) = ops.HFDatasetMapColumn().map_column(dataset=ds, column="idx", operation="row index", value="")
    method, function, with_indices, kwargs = ds.calls[-1]
    assert (method, with_indices, kwargs) == ("map", True, {})
    assert callable(function)
    assert [row["idx"] for row in out] == [0, 1, 2]


def test_map_copy_column():
    ds = FakeDataset(_rows(3))
    (out,) = ops.HFDatasetMapColumn().map_column(dataset=ds, column="copy", operation="copy column", value="text")
    assert [row["copy"] for row in out] == ["row-0", "row-1", "row-2"]


def test_map_copy_onto_itself_is_noop():
    ds = FakeDataset(_rows(3))
    (out,) = ops.HFDatasetMapColumn().map_column(dataset=ds, column="text", operation="copy column", value="text")
    assert out is ds
    assert not ds.calls


def test_map_column_on_streaming():
    stream = FakeIterableDataset(_rows(3))
    (out,) = ops.HFDatasetMapColumn().map_column(dataset=stream, column="idx", operation="row index", value="")
    assert isinstance(out, FakeIterableDataset)
    assert [row["idx"] for row in out] == [0, 1, 2]


def test_map_column_validation_errors():
    ds = FakeDataset(_rows(3))
    with pytest.raises(ValueError, match="Provide a name"):
        ops.HFDatasetMapColumn().map_column(dataset=ds, column="", operation="constant", value="x")
    with pytest.raises(ValueError, match="source column"):
        ops.HFDatasetMapColumn().map_column(dataset=ds, column="copy", operation="copy column", value="missing")
    with pytest.raises(ValueError, match="Unknown column operation"):
        ops.HFDatasetMapColumn().map_column(dataset=ds, column="kind", operation="bogus", value="")


# --------------------------------------------------------------------------- #
# Train/test split
# --------------------------------------------------------------------------- #


def test_train_test_split_outputs_two_datasets():
    ds = FakeDataset(_rows(10))
    train, test = ops.HFDatasetSplit().train_test_split(dataset=ds, test_size=0.25, seed=0, shuffle=True)
    assert isinstance(train, FakeDataset) and isinstance(test, FakeDataset)
    assert len(test) == 2
    assert len(train) == 8
    assert ds.calls[-1][0] == "train_test_split"
    with pytest.raises(ValueError, match="streaming"):
        ops.HFDatasetSplit().train_test_split(dataset=FakeIterableDataset(_rows(10)), test_size=0.25, seed=0, shuffle=True)


# --------------------------------------------------------------------------- #
# Unique
# --------------------------------------------------------------------------- #


def test_unique_values():
    ds = FakeDataset([{"label": 1}, {"label": 2}, {"label": 1}, {"label": 3}])
    (values,) = ops.HFDatasetUnique().unique_values(dataset=ds, column="label")
    assert values == [1, 2, 3]
    assert ds.calls[-1][0] == "unique"
    with pytest.raises(ValueError, match="streaming"):
        ops.HFDatasetUnique().unique_values(dataset=FakeIterableDataset([{"label": 1}]), column="label")
    with pytest.raises(ValueError, match="Unknown column"):
        ops.HFDatasetUnique().unique_values(dataset=ds, column="nope")


# --------------------------------------------------------------------------- #
# Counting
# --------------------------------------------------------------------------- #


def test_count_rows_on_materialized():
    ds = FakeDataset(_rows(5))
    (count,) = ops.HFDatasetCount().count_rows(dataset=ds)
    assert count == 5
    assert isinstance(count, int)


def test_count_empty_dataset():
    (count,) = ops.HFDatasetCount().count_rows(dataset=FakeDataset([]))
    assert count == 0


def test_count_rejects_streaming():
    with pytest.raises(ValueError, match="streaming"):
        ops.HFDatasetCount().count_rows(dataset=FakeIterableDataset(_rows(5)))


# --------------------------------------------------------------------------- #
# Conversions to LIST / Data List
# --------------------------------------------------------------------------- #


def test_to_list_returns_row_dicts():
    data = _rows(3)
    (result,) = ops.HFDatasetToList().to_list(dataset=FakeDataset(data), column="", limit=-1)
    assert result == data


def test_to_list_flat_column():
    data = _rows(3)
    (result,) = ops.HFDatasetToList().to_list(dataset=FakeDataset(data), column="text", limit=-1)
    assert result == ["row-0", "row-1", "row-2"]


def test_to_list_honours_limit():
    data = _rows(5)
    (result,) = ops.HFDatasetToList().to_list(dataset=FakeDataset(data), column="", limit=2)
    assert result == data[:2]


def test_to_data_list_returns_rows():
    data = _rows(3)
    (result,) = ops.HFDatasetToDataList().to_data_list(dataset=FakeDataset(data), column="label", limit=-1)
    assert result == [0, 1, 2]


def test_conversions_work_on_streaming():
    stream = FakeIterableDataset(_rows(4))
    (listed,) = ops.HFDatasetToList().to_list(dataset=stream, column="text", limit=2)
    assert listed == ["row-0", "row-1"]
    stream2 = FakeIterableDataset(_rows(4))
    (rows,) = ops.HFDatasetToDataList().to_data_list(dataset=stream2, column="", limit=3)
    assert rows == _rows(3)


def test_conversions_validate_column():
    data = _rows(3)
    with pytest.raises(ValueError, match="Unknown column"):
        ops.HFDatasetToList().to_list(dataset=FakeDataset(data), column="missing", limit=-1)


# --------------------------------------------------------------------------- #
# Guards against non-dataset input
# --------------------------------------------------------------------------- #


def test_transform_rejects_non_dataset():
    with pytest.raises(ValueError, match="not a Hugging Face dataset"):
        ops.HFDatasetShuffle().shuffle(dataset={"text": "x"}, seed=1)
    with pytest.raises(ValueError, match="not a Hugging Face dataset"):
        ops.HFDatasetToList().to_list(dataset=[{"text": "x"}], column="", limit=-1)
