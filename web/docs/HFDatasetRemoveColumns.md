# 🤗 Dataset Remove Columns

Removes the listed columns from a `HUGGINGFACE_DATASET`
(`datasets` `remove_columns`). Keeps every other column.

## Inputs

| Input | Type | Default | Description |
| --- | --- | --- | --- |
| `dataset` | `HUGGINGFACE_DATASET` | — | A dataset from the Loader (or another dataset node). |
| `columns` | STRING | `""` | Comma-separated columns to remove, e.g. `id, timestamp`. |

## Outputs

| Output | Type | Description |
| --- | --- | --- |
| `dataset` | `HUGGINGFACE_DATASET` | The dataset without the given columns. |

## Notes

- Works on fully-loaded **and** streaming datasets.
- Unknown column names raise a clear error; removing *every* column is refused
  (that would leave an empty dataset).

## Example

Drop columns you don't need before converting to a Data List:

```text
Loader ── dataset → Remove Columns (columns=id) ── dataset → 🤗 Dataset To Data List
```
