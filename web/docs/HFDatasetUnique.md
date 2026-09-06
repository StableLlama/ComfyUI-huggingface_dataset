# 🤗 Dataset Unique

Returns the distinct values present in a `column` of a fully-loaded dataset
(`datasets.Dataset.unique`) as a ComfyUI *Data List* of plain values.

## Inputs

| Input | Type | Default | Description |
| --- | --- | --- | --- |
| `dataset` | `HUGGINGFACE_DATASET` | — | A dataset from the Loader (or another dataset node). |
| `column` | STRING | `""` | The column whose distinct values to collect. |

## Outputs

| Output | Type | Description |
| --- | --- | --- |
| `values` | `*` (Data List) | The unique column values. Each value is an item of the Data List. |

## Notes

- **Loaded only.** `unique` needs a fully-loaded `datasets.Dataset`;
  streaming (`IterableDataset`) inputs raise a clear error. Disable `streaming`
  on the Loader.
- Unknown column names raise a clear error listing the available columns.

## Example

List every distinct label of the IMDb train split:

```text
Loader (path=stanfordnlp/imdb, split=train) ── dataset → Unique (column=label)
```

The `values` output is a ComfyUI *Data List* — count the distinct labels with
`Data List → length`, or turn them into a plain `LIST` with
`Data List → convert to LIST` (Basic data handling).
