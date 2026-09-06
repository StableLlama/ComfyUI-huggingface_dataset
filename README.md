# Hugging Face Dataset Loader

A custom [ComfyUI](https://docs.comfy.org/get_started) node that loads a
[Hugging Face](https://huggingface.co/) dataset and makes it available inside
ComfyUI for further processing.

It loads a dataset from the Hugging Face Hub or from local/remote files using the
[`datasets`](https://huggingface.co/docs/datasets/loading) library and exposes it
in two forms:

- **`dataset`** — the raw `datasets.Dataset` object of the selected split,
  handed to ComfyUI as an opaque `HUGGINGFACE_DATASET` value.
- **`rows`** — a ComfyUI *Data List* of row dicts (one dict per row), so every
  record can be processed further with generic data-handling nodes, for example
  from the **Basic data handling** node pack (convert to a ComfyUI `LIST` or
  `Data List`, access fields, filter, ...).

## Requirements

- [ComfyUI](https://docs.comfy.org/get_started)
- Python package `datasets` (Hugging Face). The node imports it **lazily**, so
  ComfyUI still starts when it is missing — you only get a clear error when you
  actually try to load a dataset:

  ```bash
  pip install datasets
  ```

## Quickstart

### Recommended Installation (ComfyUI-Manager)

1. Install [ComfyUI-Manager](https://github.com/ltdrdata/ComfyUI-Manager).
2. Look up the **"Hugging Face dataset"** extension in ComfyUI-Manager and install it.
3. Restart ComfyUI.

### Alternative (Manual Installation)

1. Install [ComfyUI](https://docs.comfy.org/get_started).
2. Clone this repository under `ComfyUI/custom_nodes`:
   ```bash
   git clone https://github.com/StableLlama/ComfyUI-huggingface_dataset.git
   ```
3. Install the `datasets` dependency into ComfyUI's Python environment:
   ```bash
   pip install datasets
   ```
4. Restart ComfyUI.

## Node

### Load Hugging Face Dataset

Loads a dataset and outputs the loaded `dataset` plus a `rows` Data List.

| Input | Type | Default | Description |
| --- | --- | --- | --- |
| `path` | STRING | `""` | Hub dataset id (e.g. `rotten_tomatoes`, `lhoestq/demo1`) **or** a local/remote file or directory (see [Sources](#sources)). |
| `loader` | STRING | `auto` | `auto` (infer from file extension), `hub`, or one of `csv`, `json`, `parquet`, `arrow`, `text`. |
| `split` | STRING | `train` | Split to load, e.g. `train`, `test`, `validation`. Supports `datasets` slicing syntax, e.g. `train[:100]`, `train[:10%]`. |
| `config` | STRING | `""` | Config/subset name for Hub datasets that have several configs (e.g. `glue` + config `mrpc`). |
| `revision` | STRING | `""` | Optional Hub revision: tag, branch name, or commit hash. |
| `trust_remote_code` | BOOLEAN | `False` | Allow executing dataset loading code from the Hub. |
| `limit` | INT | `-1` | Maximum number of rows to materialize into `rows`. `-1` = all rows. |

| Output | Type | Description |
| --- | --- | --- |
| `dataset` | `HUGGINGFACE_DATASET` | The raw `datasets.Dataset` of the selected split. |
| `rows` | `*` (Data List) | List of row dicts (one dict per row), capped by `limit`. |

### Sources

- **Hugging Face Hub** — pass a repository id as `path`, e.g.
  `rotten_tomatoes`, `glue` (with `config`), or `lhoestq/custom_squad` (with a
  `revision`). Leave `loader` at `auto`.
- **Local/remote files** — `csv`, `tsv`, `json`, `jsonl`, `parquet`, `arrow`,
  `txt`. With `loader = auto` the format is inferred from the file extension,
  otherwise pick the matching file builder explicitly. Globs (e.g.
  `data/*.parquet`) and remote `https://`/`hf://` URLs work too.
- **Local dataset directory** — a directory in Hugging Face dataset format.

### Example workflows

Load the IMDb-style `rotten_tomatoes` dataset and count its rows:

```mermaid
flowchart LR
    A[Load Hugging Face Dataset<br/>path=rotten_tomatoes<br/>split=train] -->|rows| B[Data List length]
```

Feed the rows into the **Basic data handling** node pack for per-row processing,
e.g. turn each row dict into a string, or take fields out of a specific row:

```mermaid
flowchart LR
    A[Load Hugging Face Dataset] -->|rows| B[Basic: get item<br/>index=0]
    B --> C[Basic: DICT get<br/>key=text]
```

> [!TIP]
> `rows` is already a ComfyUI *Data List* — it can be connected straight into the
> **Data List** and **LIST** nodes of the **Basic data handling** pack.

## Development

```bash
# Lint & format
python -m ruff check .
python -m ruff format .

# Type check (strict)
python -m mypy .

# Tests (no network / no datasets required)
python -m pytest tests/
```

## License

[GPL-3.0](./LICENSE)
