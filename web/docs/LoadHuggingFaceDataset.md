# 🤗 Dataset Loader

Loads a Hugging Face dataset — from the **Hub** or from **local / remote files** — and makes it available in your graph for further processing.

Add this node from *Hugging Face 🤗 → 🤗 Dataset Loader*.

## What it does

The node loads one split of a dataset with the Hugging Face
[`datasets`](https://huggingface.co/docs/datasets/) library and hands the raw
`datasets.Dataset` (or a lazy `datasets.IterableDataset` with `streaming` on) to
the rest of the graph as an opaque `HUGGINGFACE_DATASET` value.

The node itself materializes nothing: shape the dataset with the other *Hugging
Face* nodes (**Take**, **Filter**, **Shuffle**, …) and turn it into plain
ComfyUI data with the conversion nodes (**🤗 Dataset To Data List** /
**To LIST**). A graph only pays for the rows it actually uses.

## Inputs

| Input | Type | Default | Description |
| --- | --- | --- | --- |
| `path` | STRING | `""` | Hub dataset id (e.g. `stanfordnlp/imdb`), a local/remote file, a glob, or a local dataset directory. |
| `loader` | COMBO | `auto` | `auto` (infer), `hub`, or a file builder: `csv`, `json`, `parquet`, `arrow`, `text`. |
| `split` | COMBO | `train` | The split to load. A dropdown filled with the splits the source actually exposes (see below). |
| `config` | STRING | `""` | Config/subset name for Hub datasets with several configs (e.g. `nyu-mll/glue` + config `mrpc`). |
| `revision` | STRING | `""` | Optional Hub revision: tag, branch name or commit hash. |
| `streaming` | BOOLEAN | `false` | Load as a lazy `IterableDataset` instead of downloading/caching the whole split. |

## Outputs

| Output | Type | Description |
| --- | --- | --- |
| `dataset` | `HUGGINGFACE_DATASET` | The raw `datasets.Dataset` of the split (an `IterableDataset` when `streaming` is on). |

> [!IMPORTANT]
> **Upgrading from 1.x:** there is no `rows` output and no `limit` widget any
> more. Connect the `dataset` output to a **🤗 Dataset To Data List** node and
> bound the rows with **🤗 Dataset Take** (or the conversion node's `limit`).

## Split selection

The `split` dropdown lists the splits of the selected source. It refreshes
automatically when `path` / `config` / `revision` / `loader` change, when a
workflow is loaded, and again when you click **Force reload**. When the dataset
has no `train` split a sensible default is picked (`train` → `validation` →
`test`).

If the splits can't be determined (offline, or a multi-config dataset without a
`config`), the dropdown falls back to `train`/`test`/`validation` and the node
validates the split when it runs — the error message lists the available splits.

## Streaming vs. full load

With `streaming` **off** (default) the whole split is downloaded and cached and
a `datasets.Dataset` is returned; `Take` / `Filter` / … then work on local data.

With `streaming` **on** the split loads as a lazy `datasets.IterableDataset`:
nothing is fetched until it is iterated. This is the memory/bandwidth-friendly
way to pull only the first rows of a very large dataset — put a **🤗 Dataset
Take** (or a `Select` / `Shard`) in front of **🤗 Dataset To Data List**. Note
that the `dataset` output is then an `IterableDataset` (no `len()`, single pass)
— some downstream operations need a fully-loaded dataset, disable `streaming`
for those.

## Force reload

The **Force reload** button does two things at once: it re-queries the dataset's
splits (so the `split` dropdown is in sync) and marks the loader to re-fetch the
dataset on the **next run**. ComfyUI caches a node's output on its inputs, so
re-running a workflow normally reuses the dataset you already loaded — even if
the source changed on the Hub or on disk. Clicking **Force reload** bumps an
internal `reload_tick` counter (a hidden input of the node) to invalidate that
cache; the next time you run the workflow the loader fetches the dataset fresh.

Use it after the source dataset has been updated (for example offline/local
changes on disk) and you want the loader to pick up the new rows without having
to change `path` / `revision` / … yourself. It also replaces the old manual
"Refresh splits" button — the split dropdown still refreshes automatically when
the source inputs change.

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

Add a **🤗 Dataset To Data List** (or **To LIST**) node behind the loader; its
output is a ComfyUI *Data List* that the **Basic data handling** pack can
consume directly:

- `Data List → length` / `count` — how many rows were materialised.
- `Data List → get item (index=0)` → `DICT → get (key=text)` — read one field of
  one row.
- Connect the Data List straight into a `DICT → get`, `STRING` or `cast` node —
  such nodes run once per row and return a Data List of results (per-row
  mapping).

See the README's *Working with Basic data handling* section for more recipes.
