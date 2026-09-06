# 🤗 Dataset Select Rows

Keeps specific rows of a fully-loaded dataset by index
(`datasets.Dataset.select`).

## Inputs

| Input | Type | Default | Description |
| --- | --- | --- | --- |
| `dataset` | `HUGGINGFACE_DATASET` | — | A dataset from the Loader (or another dataset node). |
| `indices` | STRING | `0:100` | Comma-separated indices (`0, 2, 4`) and/or Python-style slices (`0:100`, `0:100:2`); an open slice runs to the last row. |

## Outputs

| Output | Type | Description |
| --- | --- | --- |
| `dataset` | `HUGGINGFACE_DATASET` | The dataset with only the selected rows. |

## Notes

- **Loaded only.** Selecting by explicit index needs a fully-loaded
  `datasets.Dataset`; streaming (`IterableDataset`) inputs raise a clear error.
  Disable `streaming` on the Loader.
- Row indices out of range raise a clear error.

## Example

Keep every third row of the first 100 rows:

```text
Loader ── dataset → Select Rows (indices=0:100:2) ── dataset → 🤗 Dataset To Data List
```
