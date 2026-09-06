# 🤗 Dataset Map Column

Adds (or replaces) one column on every row of a `HUGGINGFACE_DATASET`, computed
per row with the `datasets` `map` operation. No free-form Python needed.

## Inputs

| Input | Type | Default | Description |
| --- | --- | --- | --- |
| `dataset` | `HUGGINGFACE_DATASET` | — | A dataset from the Loader (or another dataset node). |
| `column` | STRING | `new_column` | Name of the column to add or replace. |
| `operation` | COMBO | `constant` | How to compute the value (see below). |
| `value` | STRING | `""` | Argument for the chosen operation. |

## Outputs

| Output | Type | Description |
| --- | --- | --- |
| `dataset` | `HUGGINGFACE_DATASET` | The dataset with the new/changed column. |

## Operations

- **`constant`** — every row gets the literal `value` text.
- **`copy column`** — `value` is the name of an existing column whose values
  are copied into `column`.
- **`row index`** — every row gets its 0-based row index as an integer.

## Notes

- Works on fully-loaded **and** streaming datasets.
- Replacing an existing column with a value of another type is supported (the
  old column is dropped first so `datasets` can infer the new feature cleanly).
- Copying a column onto itself is a harmless no-op.

## Example

Number the rows of a filtered dataset:

```text
Loader ── dataset → Filter (column=label, ==, value=1) ── dataset
  → Map Column (column=row_id, operation=row index) ── dataset → 🤗 Dataset To Data List
```
