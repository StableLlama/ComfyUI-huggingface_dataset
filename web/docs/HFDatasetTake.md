# 🤗 Dataset Take

Keeps only the first `n` rows of a `HUGGINGFACE_DATASET` and drops the rest
(`datasets` `take`).

## Inputs

| Input | Type | Default | Description |
| --- | --- | --- | --- |
| `dataset` | `HUGGINGFACE_DATASET` | — | A dataset from the Loader (or another dataset node). |
| `n` | INT | `100` | How many leading rows to keep. |

## Outputs

| Output | Type | Description |
| --- | --- | --- |
| `dataset` | `HUGGINGFACE_DATASET` | The dataset limited to its first `n` rows. |

## Notes

- Works on fully-loaded **and** streaming datasets.
- Combine with **To Data List** to cap how much of a large dataset is
  materialized into plain data.

## Example

```text
Loader (streaming=true) ── dataset → Take (n=20) ── dataset → 🤗 Dataset To Data List
```
