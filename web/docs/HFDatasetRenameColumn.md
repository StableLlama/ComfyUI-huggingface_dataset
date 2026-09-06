# 🤗 Dataset Rename Column

Renames a single column of a `HUGGINGFACE_DATASET`
(`datasets` `rename_column`).

## Inputs

| Input | Type | Default | Description |
| --- | --- | --- | --- |
| `dataset` | `HUGGINGFACE_DATASET` | — | A dataset from the Loader (or another dataset node). |
| `original_column` | STRING | `""` | The current column name. |
| `new_column` | STRING | `""` | The desired column name. |

## Outputs

| Output | Type | Description |
| --- | --- | --- |
| `dataset` | `HUGGINGFACE_DATASET` | The dataset with the renamed column. |

## Notes

- Works on fully-loaded **and** streaming datasets.
- Renaming a column onto itself is a harmless no-op. An unknown
  `original_column` raises a clear error.

## Example

```text
Loader ── dataset → Rename Column (original_column=text, new_column=review) ── dataset → 🤗 Dataset To Data List
```
