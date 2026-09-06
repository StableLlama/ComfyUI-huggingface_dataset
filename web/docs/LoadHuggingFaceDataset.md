# 🤗 Dataset Loader

Loads a Hugging Face dataset — from the **Hub** or from **local / remote files** — and makes it available in your graph for further processing.

Add this node from *Hugging Face 🤗 → 🤗 Dataset Loader*.

## What it does

The node loads one split of a dataset with the Hugging Face
[`datasets`](https://huggingface.co/docs/datasets/) library and hands it to the
rest of the graph in two forms:

- **`dataset`** — the raw `datasets.Dataset` of the selected split, exposed as an
  opaque `HUGGINGFACE_DATASET` value. Feed this into the other *Hugging Face*
  nodes (**Shuffle**, **Filter**, **To LIST**, …) to shape the dataset *before*
  its rows are turned into plain data.
- **`rows`** — a ComfyUI *Data List* of row dicts (one dict per row), capped by
  `limit`. This connects straight into the *Data List* / *LIST* nodes of packs
  like **Basic data handling**.

## Inputs

| Input | Type | Default | Description |
| --- | --- | --- | --- |
| `path` | STRING | `""` | Hub dataset id (e.g. `stanfordnlp/imdb`), a local/remote file, a glob, or a local dataset directory. |
| `loader` | COMBO | `auto` | `auto` (infer), `hub`, or a file builder: `csv`, `json`, `parquet`, `arrow`, `text`. |
| `split` | COMBO | `train` | The split to load. A dropdown filled with the splits the source actually exposes (see below). |
| `config` | STRING | `""` | Config/subset name for Hub datasets with several configs (e.g. `nyu-mll/glue` + config `mrpc`). |
| `revision` | STRING | `""` | Optional Hub revision: tag, branch name or commit hash. |
| `streaming` | BOOLEAN | `false` | Load as a lazy `IterableDataset` instead of downloading/caching the whole split. |
| `limit` | INT | `-1` | Max rows materialized into the `rows` Data List. `-1` = all. |

## Outputs

| Output | Type | Description |
| --- | --- | --- |
| `dataset` | `HUGGINGFACE_DATASET` | The raw `datasets.Dataset` of the split (an `IterableDataset` when `streaming` is on). |
| `rows` | `*` (Data List) | List of row dicts, capped by `limit`. |

## Split selection

The `split` dropdown lists the splits of the selected source. It refreshes
automatically when `path` / `config` / `revision` / `loader` change (use the
**Refresh splits** button to re-query manually). When the dataset has no
`train` split a sensible default is picked (`train` → `validation` → `test`).

If the splits can't be determined (offline, or a multi-config dataset without a
`config`), the dropdown falls back to `train`/`test`/`validation` and the node
validates the split when it runs — the error message lists the available splits.

## Streaming vs. full load

With `streaming` **off** (default) the whole split is downloaded and cached
before the first rows are materialized; `limit` caps the size of `rows`, but
*not* the download.

With `streaming` **on** the split loads as a lazy `datasets.IterableDataset`:
nothing is fetched until it is iterated. This is the memory/bandwidth-friendly
way to pull only the first rows of a very large dataset. Note that the
`dataset` output is then an `IterableDataset` (no `len()`, single pass) — some
downstream operations need a fully-loaded dataset, disable `streaming` for
those.

## Sources

- **Hub** — dataset id, e.g. `stanfordnlp/imdb`, `nyu-mll/glue` (with `config`),
  or `HuggingFaceFW/fineweb` (with a `revision`). Leave `loader` at `auto`.
- **Local / remote files** — `csv`, `tsv`, `json`, `jsonl`, `parquet`, `arrow`,
  `txt`. Format is inferred from the extension with `auto`, or forced by picking
  the matching file builder. Globs and `https://`/`hf://` URLs work too.
- **Local dataset directory** — a folder in Hugging Face dataset format.

## Example

Load IMDb, keep only positive reviews, and pass them on as a *Data List*:

```text
Loader (path=stanfordnlp/imdb, split=train)
  └─ dataset → 🤗 Dataset Filter (column=label, ==, value=1)
                 └─ dataset → 🤗 Dataset To Data List (limit=100)
```

> [!NOTE]
> Requires the Python package `datasets` (`pip install datasets`). It is
> installed automatically when the node is installed through ComfyUI-Manager or
> the Comfy Registry; a missing one shows a clear install message in the GUI.

### Use with Basic data handling

`rows` is already a ComfyUI *Data List*, so the **Basic data handling** pack can
consume it directly:

- `Data List → length` / `count` — how many rows were materialised.
- `Data List → get item (index=0)` → `DICT → get (key=text)` — read one field of
  one row.
- Connect `rows` straight into a `DICT → get`, `STRING` or `cast` node — such
  nodes run once per row and return a Data List of results (per-row mapping).

See the README's *Working with Basic data handling* section for more recipes.
