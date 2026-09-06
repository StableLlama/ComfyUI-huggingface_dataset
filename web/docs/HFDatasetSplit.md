# 🤗 Dataset Train/Test Split

Randomly splits a fully-loaded dataset into `train` and `test` parts
(`datasets.Dataset.train_test_split`).

## Inputs

| Input | Type | Default | Description |
| --- | --- | --- | --- |
| `dataset` | `HUGGINGFACE_DATASET` | — | A dataset from the Loader (or another dataset node). |
| `test_size` | FLOAT | `0.2` | Fraction of rows held out for `test` (0.0 – 1.0). |
| `seed` | INT | `0` | Random seed for a reproducible split. |
| `shuffle` | BOOLEAN | `true` | Shuffle before splitting when enabled. |

## Outputs

| Output | Type | Description |
| --- | --- | --- |
| `train` | `HUGGINGFACE_DATASET` | The training part. |
| `test` | `HUGGINGFACE_DATASET` | The held-out test part. |

## Notes

- **Loaded only.** Splitting needs a fully-loaded `datasets.Dataset`;
  streaming (`IterableDataset`) inputs raise a clear error. Disable `streaming`
  on the Loader.
- Useful for preparing a dataset for training downstream.

## Example

Split the (pre-filtered) IMDb positives into 80% train / 20% test:

```text
Loader ── dataset → Filter (==1) ── dataset → Train/Test Split (test_size=0.2, seed=42)
  ├─ train → 🤗 Dataset To Data List
  └─ test  → 🤗 Dataset To Data List
```
