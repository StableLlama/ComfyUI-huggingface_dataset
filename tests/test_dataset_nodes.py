"""Tests for the Hugging Face dataset loading node.

The ``datasets`` library and the network are deliberately not required: the
module-level ``_require_datasets`` helper is monkeypatched with a fake so all
loading logic can be tested hermetically.
"""

from typing import Any

import pytest

from huggingface_dataset import dataset_nodes as nodes
from huggingface_dataset.dataset_nodes import (
    LoadHuggingFaceDataset,
    NODE_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS,
    _available_splits,
    _infer_file_builder,
    _sensible_split,
    _to_python,
    split_info,
)


class FakeDataset:
    """Minimal stand-in for ``datasets.Dataset`` backed by a list of row dicts."""

    def __init__(self, rows: list[dict[str, Any]]):
        self._rows = list(rows)
        self._columns = list(rows[0]) if rows else []

    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(self, key: Any) -> Any:
        if isinstance(key, slice):
            subset = self._rows[key]
            return {column: [row[column] for row in subset] for column in self._columns}
        return self._rows[key]


class FakeIterableDataset:
    """Minimal stand-in for a streaming ``datasets.IterableDataset``.

    Iterable (like the real streaming object) but with no ``len()`` and no
    slicing - exactly what ``streaming=True`` hands back.
    """

    def __init__(self, rows: list[dict[str, Any]]):
        self._rows = list(rows)

    def __iter__(self) -> Any:
        return iter(self._rows)


class FakeDatasets:
    """Stand-in for the ``datasets`` module; records ``load_dataset`` calls.

    ``split_names`` controls what ``get_dataset_split_names`` reports. When it is
    ``None`` the lookup raises, mimicking a dataset whose splits cannot be
    determined (offline / needs config / custom loader script).
    """

    def __init__(self, result: Any = None, split_names: list[str] | None = None):
        self.calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
        self.split_calls: list[tuple[str, dict[str, Any]]] = []
        self._result = result
        self.split_names = split_names

    def load_dataset(self, *args: Any, **kwargs: Any) -> Any:
        self.calls.append((args, kwargs))
        return self._result

    def get_dataset_split_names(self, path: str, **kwargs: Any) -> list[str]:
        self.split_calls.append((path, kwargs))
        if self.split_names is None:
            raise RuntimeError("cannot determine splits")
        return list(self.split_names)


class _Scalar:
    """Object exposing ``.item()`` like a numpy scalar."""

    def __init__(self, value: int):
        self._value = value

    def item(self) -> int:
        return self._value


class _Array:
    """Object exposing ``.ndim``/``.tolist()`` like a numpy array."""

    ndim = 1

    def __init__(self, values: list[int]):
        self._values = values

    def tolist(self) -> list[int]:
        return self._values


def _rows(count: int) -> list[dict[str, Any]]:
    return [{"text": f"row-{index}", "label": index} for index in range(count)]


def _make_node() -> LoadHuggingFaceDataset:
    return LoadHuggingFaceDataset()


def _install_fake(monkeypatch: pytest.MonkeyPatch, result: Any = None) -> FakeDatasets:
    fake = FakeDatasets(result=result)
    monkeypatch.setattr(nodes, "_require_datasets", lambda: fake)
    return fake


# --------------------------------------------------------------------------- #
# Registration / metadata
# --------------------------------------------------------------------------- #


def test_node_is_registered():
    assert NODE_CLASS_MAPPINGS["LoadHuggingFaceDataset"] is LoadHuggingFaceDataset
    assert NODE_DISPLAY_NAME_MAPPINGS["LoadHuggingFaceDataset"] == "🤗 Dataset Loader"


def test_node_metadata():
    node = _make_node()
    inputs = node.INPUT_TYPES()["required"]
    assert set(inputs) == {"path", "loader", "split", "config", "revision", "streaming", "limit"}
    assert node.RETURN_TYPES[0] == "HUGGINGFACE_DATASET"
    assert node.RETURN_NAMES == ("dataset", "rows")
    assert node.OUTPUT_IS_LIST == (False, True)
    assert node.FUNCTION == "load"
    assert callable(getattr(node, node.FUNCTION))


def test_node_tooltips():
    node = _make_node()
    for name, (_type, options) in node.INPUT_TYPES()["required"].items():
        assert options.get("tooltip"), f"{name} is missing a tooltip"
    assert len(node.OUTPUT_TOOLTIPS) == len(node.RETURN_TYPES)
    assert all(tip.strip() for tip in node.OUTPUT_TOOLTIPS)


def test_jpeg_xl_flag_is_boolean():
    assert isinstance(nodes.JPEG_XL_AVAILABLE, bool)


def test_jpeg_xl_enabled_when_plugin_is_importable(monkeypatch):
    import sys
    import types

    monkeypatch.setitem(sys.modules, "pillow_jxl", types.ModuleType("pillow_jxl"))

    assert nodes._pillow_jxl_importable() is True


def test_jpeg_xl_disabled_when_plugin_is_missing(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def block_pillow_jxl(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "pillow_jxl":
            raise ImportError("pillow-jxl-plugin is not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", block_pillow_jxl)

    assert nodes._pillow_jxl_importable() is False


# --------------------------------------------------------------------------- #
# Hub loading
# --------------------------------------------------------------------------- #


def test_hub_load_default(monkeypatch):
    data = _rows(3)
    fake = _install_fake(monkeypatch, result=FakeDataset(data))

    dataset, rows = _make_node().load(path="myorg/my_dataset")

    assert dataset._rows == data
    assert rows == data
    (args, kwargs) = fake.calls[0]
    assert args == ("myorg/my_dataset",)
    assert kwargs == {"split": "train", "streaming": False}


def test_hub_load_with_config_split_and_revision(monkeypatch):
    fake = _install_fake(monkeypatch, result=FakeDataset(_rows(1)))

    _make_node().load(path="glue", config="mrpc", split="validation", revision="abc123")

    (args, kwargs) = fake.calls[0]
    assert args == ("glue", "mrpc")
    assert kwargs == {"split": "validation", "revision": "abc123", "streaming": False}


def test_empty_split_falls_back_to_train(monkeypatch):
    fake = _install_fake(monkeypatch, result=FakeDataset(_rows(1)))

    _make_node().load(path="myorg/my_dataset", split="")

    (_, kwargs) = fake.calls[0]
    assert kwargs["split"] == "train"


def test_empty_revision_is_omitted(monkeypatch):
    fake = _install_fake(monkeypatch, result=FakeDataset(_rows(1)))

    _make_node().load(path="myorg/my_dataset", revision="")

    (_, kwargs) = fake.calls[0]
    assert "revision" not in kwargs


# --------------------------------------------------------------------------- #
# Local / remote file loading
# --------------------------------------------------------------------------- #


def test_auto_infers_csv_builder(monkeypatch):
    fake = _install_fake(monkeypatch, result=FakeDataset(_rows(1)))

    _make_node().load(path="/data/my_file.csv")

    (args, kwargs) = fake.calls[0]
    assert args == ("csv",)
    assert kwargs["data_files"] == "/data/my_file.csv"
    assert kwargs["split"] == "train"


def test_auto_infers_parquet_from_url(monkeypatch):
    fake = _install_fake(monkeypatch, result=FakeDataset(_rows(1)))

    url = "https://example.com/data/train-00000-of-00001.parquet?x=1"
    _make_node().load(path=url)

    (args, kwargs) = fake.calls[0]
    assert args == ("parquet",)
    assert kwargs["data_files"] == url


def test_explicit_loader_wins_over_extension(monkeypatch):
    fake = _install_fake(monkeypatch, result=FakeDataset(_rows(1)))

    _make_node().load(path="/data/data.parquet", loader="json")

    (args, _) = fake.calls[0]
    assert args == ("json",)


def test_hub_is_used_when_no_extension(monkeypatch):
    fake = _install_fake(monkeypatch, result=FakeDataset(_rows(1)))

    _make_node().load(path="dataset_folder", loader="auto")

    (args, _) = fake.calls[0]
    assert args == ("dataset_folder",)


def test_explicit_hub_loader_with_config(monkeypatch):
    fake = _install_fake(monkeypatch, result=FakeDataset(_rows(1)))

    _make_node().load(path="glue", loader="hub", config="sst2")

    (args, _) = fake.calls[0]
    assert args == ("glue", "sst2")


def test_extension_inference_helper():
    assert _infer_file_builder("/a/b.csv") == "csv"
    assert _infer_file_builder("/a/b.tsv") == "csv"
    assert _infer_file_builder("data.jsonl") == "json"
    assert _infer_file_builder("https://x/y.parquet?token=abc") == "parquet"
    assert _infer_file_builder("myorg/dataset") is None
    assert _infer_file_builder("/a/dir_without_ext") is None


# --------------------------------------------------------------------------- #
# rows output
# --------------------------------------------------------------------------- #


def test_limit_caps_rows(monkeypatch):
    data = _rows(5)
    _install_fake(monkeypatch, result=FakeDataset(data))

    _, rows = _make_node().load(path="myorg/my_dataset", limit=2)

    assert rows == data[:2]


def test_negative_limit_returns_all_rows(monkeypatch):
    data = _rows(5)
    _install_fake(monkeypatch, result=FakeDataset(data))

    _, rows = _make_node().load(path="myorg/my_dataset", limit=-1)

    assert rows == data


def test_limit_larger_than_dataset_is_clamped(monkeypatch):
    data = _rows(3)
    _install_fake(monkeypatch, result=FakeDataset(data))

    _, rows = _make_node().load(path="myorg/my_dataset", limit=1000)

    assert rows == data


def test_empty_dataset_returns_empty_rows(monkeypatch):
    _install_fake(monkeypatch, result=FakeDataset([]))

    dataset, rows = _make_node().load(path="myorg/my_dataset")

    assert len(dataset) == 0
    assert rows == []


# --------------------------------------------------------------------------- #
# streaming loading
# --------------------------------------------------------------------------- #


def test_streaming_forwards_flag_and_returns_iterable(monkeypatch):
    data = _rows(3)
    fake = _install_fake(monkeypatch, result=FakeIterableDataset(data))

    dataset, rows = _make_node().load(path="myorg/my_dataset", streaming=True)

    assert isinstance(dataset, FakeIterableDataset)
    assert rows == data
    (_, kwargs) = fake.calls[0]
    assert kwargs == {"split": "train", "streaming": True}


def test_streaming_limit_caps_rows(monkeypatch):
    data = _rows(5)
    _install_fake(monkeypatch, result=FakeIterableDataset(data))

    _, rows = _make_node().load(path="myorg/my_dataset", streaming=True, limit=2)

    assert rows == data[:2]


def test_streaming_zero_limit_returns_no_rows(monkeypatch):
    _install_fake(monkeypatch, result=FakeIterableDataset(_rows(5)))

    _, rows = _make_node().load(path="myorg/my_dataset", streaming=True, limit=0)

    assert rows == []


def test_streaming_off_uses_materialized_dataset(monkeypatch):
    data = _rows(3)
    fake = _install_fake(monkeypatch, result=FakeDataset(data))

    dataset, rows = _make_node().load(path="myorg/my_dataset", streaming=False)

    assert isinstance(dataset, FakeDataset)
    assert rows == data
    (_, kwargs) = fake.calls[0]
    assert kwargs["streaming"] is False


# --------------------------------------------------------------------------- #
# row value conversion
# --------------------------------------------------------------------------- #


def test_to_python_converts_exotic_values():
    row = {
        "text": "hello",
        "flag": True,
        "count": _Scalar(3),
        "vector": _Array([1, 2]),
        "payload": b"caf\xc3\xa9",
        "nested": {"a": _Scalar(7), "b": [_Array([9])]},
        "opaque": object(),
    }
    converted = _to_python(row)
    assert converted["text"] == "hello"
    assert converted["flag"] is True
    assert converted["count"] == 3
    assert converted["vector"] == [1, 2]
    assert converted["payload"] == "café"
    assert converted["nested"] == {"a": 7, "b": [[9]]}
    assert isinstance(converted["opaque"], object)


def test_to_python_keeps_unknown_objects():
    opaque = object()
    assert _to_python(opaque) is opaque


# --------------------------------------------------------------------------- #
# error handling
# --------------------------------------------------------------------------- #


def test_empty_path_raises_value_error(monkeypatch):
    _install_fake(monkeypatch)
    with pytest.raises(ValueError, match="path"):
        _make_node().load(path="")


def test_missing_datasets_raises_actionable_error(monkeypatch):
    def missing() -> Any:
        raise ImportError("The Hugging Face 'datasets' package is required to load datasets.\nInstall it with:  pip install datasets")

    monkeypatch.setattr(nodes, "_require_datasets", missing)
    with pytest.raises(ImportError, match="pip install datasets"):
        _make_node().load(path="myorg/my_dataset")


# --------------------------------------------------------------------------- #
# split dropdown & discovery
# --------------------------------------------------------------------------- #


def test_split_input_is_a_combo_with_fallback_options():
    split_input = LoadHuggingFaceDataset.INPUT_TYPES()["required"]["split"]
    options = split_input[0]
    assert isinstance(options, tuple)
    assert set(options) == {"train", "test", "validation"}
    assert split_input[1]["default"] == "train"


def test_available_splits_hub(monkeypatch):
    fake = FakeDatasets(split_names=["train", "test"])
    monkeypatch.setattr(nodes, "_require_datasets", lambda: fake)

    assert _available_splits(path="myorg/my_dataset") == ["train", "test"]
    (path, kwargs) = fake.split_calls[0]
    assert path == "myorg/my_dataset"
    assert kwargs == {}


def test_available_splits_hub_with_config_and_revision(monkeypatch):
    fake = FakeDatasets(split_names=["test"])
    monkeypatch.setattr(nodes, "_require_datasets", lambda: fake)

    assert _available_splits(path="glue", config="mrpc", revision="abc123") == ["test"]
    (path, kwargs) = fake.split_calls[0]
    assert path == "glue"
    assert kwargs == {"config_name": "mrpc", "revision": "abc123"}


def test_available_splits_file_builder_is_single_train_split(monkeypatch):
    fake = FakeDatasets()
    monkeypatch.setattr(nodes, "_require_datasets", lambda: fake)

    assert _available_splits(path="/data/my_file.csv") == ["train"]
    assert fake.split_calls == []


def test_available_splits_empty_path_returns_empty_list(monkeypatch):
    monkeypatch.setattr(nodes, "_require_datasets", lambda: FakeDatasets())
    assert _available_splits(path="") == []


def test_available_splits_unknown_returns_none(monkeypatch):
    monkeypatch.setattr(nodes, "_require_datasets", lambda: FakeDatasets())
    assert _available_splits(path="myorg/my_dataset") is None


def test_sensible_split_prefers_train_then_validation_then_test():
    assert _sensible_split(["test"]) == "test"
    assert _sensible_split(["test", "validation"]) == "validation"
    assert _sensible_split(["test", "train"]) == "train"
    assert _sensible_split(["other"]) == "other"
    assert _sensible_split([]) == "train"


def test_default_split_auto_corrects_when_no_train(monkeypatch):
    fake = FakeDatasets(result=FakeDataset(_rows(1)), split_names=["test"])
    monkeypatch.setattr(nodes, "_require_datasets", lambda: fake)

    _make_node().load(path="myorg/my_dataset")

    (_, kwargs) = fake.calls[0]
    assert kwargs["split"] == "test"


def test_default_split_kept_when_train_present(monkeypatch):
    fake = FakeDatasets(result=FakeDataset(_rows(1)), split_names=["train", "test"])
    monkeypatch.setattr(nodes, "_require_datasets", lambda: fake)

    _make_node().load(path="myorg/my_dataset")

    (_, kwargs) = fake.calls[0]
    assert kwargs["split"] == "train"


def test_explicit_invalid_split_raises_clean_error(monkeypatch):
    fake = FakeDatasets(result=FakeDataset(_rows(1)), split_names=["test"])
    monkeypatch.setattr(nodes, "_require_datasets", lambda: fake)

    with pytest.raises(ValueError, match=r"Available splits: test"):
        _make_node().load(path="myorg/my_dataset", split="validation")


def test_slicing_split_is_passed_through(monkeypatch):
    fake = FakeDatasets(result=FakeDataset(_rows(1)), split_names=["train", "test"])
    monkeypatch.setattr(nodes, "_require_datasets", lambda: fake)

    _make_node().load(path="myorg/my_dataset", split="train[:10%]")

    (_, kwargs) = fake.calls[0]
    assert kwargs["split"] == "train[:10%]"


def test_unknown_splits_pass_through_unchanged(monkeypatch):
    # split_names=None => cannot introspect => request is forwarded unchanged
    fake = FakeDatasets(result=FakeDataset(_rows(1)))
    monkeypatch.setattr(nodes, "_require_datasets", lambda: fake)

    _make_node().load(path="myorg/my_dataset", split="validation")

    (_, kwargs) = fake.calls[0]
    assert kwargs["split"] == "validation"


def test_split_info_success(monkeypatch):
    fake = FakeDatasets(split_names=["train", "test"])
    monkeypatch.setattr(nodes, "_require_datasets", lambda: fake)

    assert split_info(path="myorg/my_dataset") == {
        "splits": ["train", "test"],
        "default": "train",
        "error": None,
    }


def test_split_info_default_is_sensible_for_test_only_dataset(monkeypatch):
    fake = FakeDatasets(split_names=["test"])
    monkeypatch.setattr(nodes, "_require_datasets", lambda: fake)

    info = split_info(path="myorg/my_dataset")
    assert info["splits"] == ["test"]
    assert info["default"] == "test"
    assert info["error"] is None


def test_split_info_reports_missing_datasets(monkeypatch):
    def missing() -> Any:
        raise ImportError("The Hugging Face 'datasets' package is required to load datasets.\nInstall it with:  pip install datasets")

    monkeypatch.setattr(nodes, "_require_datasets", missing)

    info = split_info(path="myorg/my_dataset")
    assert info["splits"] is None
    assert info["default"] is None
    assert "pip install datasets" in info["error"]


def test_split_info_reports_unknown_splits(monkeypatch):
    monkeypatch.setattr(nodes, "_require_datasets", lambda: FakeDatasets())

    info = split_info(path="myorg/my_dataset")
    assert info["splits"] is None
    assert info["default"] is None
    assert info["error"]
