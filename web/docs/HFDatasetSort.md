# 🤗 Dataset Sort

Sorts a fully-loaded dataset by a column (`datasets.Dataset.sort`).

## Inputs

| Input | Type | Default | Description |
| --- | --- | --- | --- |
| `dataset` | `HUGGINGFACE_DATASET` | — | A dataset from the Loader (or another dataset node). |
| `column` | STRING | `""` | The column to sort by. |
| `reverse` | BOOLEAN | `false` | Sort descending when enabled. |

## Outputs

| Output | Type | Description |
| --- | --- | --- |
| `dataset` | `HUGGINGFACE_DATASET` | The sorted dataset. |

## Notes

- **Loaded only.** Sorting needs a fully-loaded `datasets.Dataset`; streaming
  (`IterableDataset`) inputs raise a clear error. Disable `streaming` on the
  Loader that created the dataset.

## Example

Sort reviews by their numeric `label` column:

```text
Loader ── dataset → Sort (column=label, reverse=false) ── dataset → 🤗 Dataset To Data List
```
