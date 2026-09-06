# 🤗 Dataset Shuffle

Randomly reorders the rows of a `HUGGINGFACE_DATASET`
(`datasets.Dataset.shuffle` / `datasets.IterableDataset.shuffle`).

## Inputs

| Input | Type | Default | Description |
| --- | --- | --- | --- |
| `dataset` | `HUGGINGFACE_DATASET` | — | A dataset from the Loader (or another dataset node). |
| `seed` | INT | `0` | Random seed for reproducible shuffles. **Required** when `streaming` is enabled. |

## Outputs

| Output | Type | Description |
| --- | --- | --- |
| `dataset` | `HUGGINGFACE_DATASET` | The shuffled dataset. |

## Notes

- Works on fully-loaded **and** streaming datasets.
- With a streaming (`IterableDataset`) input the `seed` is mandatory — without
  one the rows could not be reordered deterministically.

## Example

```text
Loader ── dataset → Shuffle (seed=42) ── dataset → 🤗 Dataset To Data List
```
