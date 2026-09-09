# 🤗 Dataset Count

Returns the number of entries (rows) of a fully-loaded Hugging Face dataset as
an `INT`.

## Inputs

| Input | Type | Default | Description |
| --- | --- | --- | --- |
| `dataset` | `HUGGINGFACE_DATASET` | — | A dataset from the Loader (or another dataset node). |

## Outputs

| Output | Type | Description |
| --- | --- | --- |
| `count` | INT | The number of rows (entries) in the dataset. |

## Notes

- **Loaded only.** Counting needs a fully-loaded `datasets.Dataset` (`len`);
  a streaming (`IterableDataset`) input raises a clear error, because it has no
  `len` and counting it would consume every row. Disable `streaming` on the
  Loader (or feed in a materialized output of another dataset node).

## Example

Get the number of rows of the IMDb train split:

```text
Loader (path=stanfordnlp/imdb, split=train) ── dataset → Count ── count
```

Feed the `count` into any `INT`-accepting node (e.g. a `Basic/INT` node) to
loop or branch on the dataset size.
