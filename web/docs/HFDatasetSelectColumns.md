# 🤗 Dataset Select Columns

Keeps only the listed columns of a `HUGGINGFACE_DATASET`
(`datasets` `select_columns`). Drops every other column.

## Inputs

| Input | Type | Default | Description |
| --- | --- | --- | --- |
| `dataset` | `HUGGINGFACE_DATASET` | — | A dataset from the Loader (or another dataset node). |
| `columns` | STRING | `""` | Comma-separated columns to keep, e.g. `text, label`. |

## Outputs

| Output | Type | Description |
| --- | --- | --- |
| `dataset` | `HUGGINGFACE_DATASET` | The dataset restricted to the given columns. |

## Notes

- Works on fully-loaded **and** streaming datasets.
- Unknown column names raise a clear error listing the available columns.

## Example

Keep only the review text before converting to a plain LIST of strings:

```text
Loader ── dataset → Select Columns (columns=text) ── dataset → 🤗 Dataset To LIST (column=text)
```
