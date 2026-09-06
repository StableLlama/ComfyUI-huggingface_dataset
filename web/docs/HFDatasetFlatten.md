# 🤗 Dataset Flatten

Flattens the nested columns of a fully-loaded dataset into top-level columns
(`datasets.Dataset.flatten`). Nested dict/list values are expanded, e.g.
`meta.user` becomes a `meta.user` column.

## Inputs

| Input | Type | Default | Description |
| --- | --- | --- | --- |
| `dataset` | `HUGGINGFACE_DATASET` | — | A dataset from the Loader (or another dataset node). |

## Outputs

| Output | Type | Description |
| --- | --- | --- |
| `dataset` | `HUGGINGFACE_DATASET` | The flattened dataset. |

## Notes

- **Loaded only.** Flattening needs a fully-loaded `datasets.Dataset`;
  streaming (`IterableDataset`) inputs raise a clear error. Disable `streaming`
  on the Loader.

## Example

Flatten a dataset whose rows contain a nested `info` dict, then keep one of the
new columns:

```text
Loader ── dataset → Flatten ── dataset → Select Columns (columns=info.title)
```
