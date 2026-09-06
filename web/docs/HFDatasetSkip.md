# 🤗 Dataset Skip

Drops the first `n` rows of a `HUGGINGFACE_DATASET` and returns the rest
(`datasets` `skip`).

## Inputs

| Input | Type | Default | Description |
| --- | --- | --- | --- |
| `dataset` | `HUGGINGFACE_DATASET` | — | A dataset from the Loader (or another dataset node). |
| `n` | INT | `1` | How many leading rows to drop. |

## Outputs

| Output | Type | Description |
| --- | --- | --- |
| `dataset` | `HUGGINGFACE_DATASET` | The dataset without its first `n` rows. |

## Notes

- Works on fully-loaded **and** streaming datasets — handy to page past the
  first rows of a large streaming split.

## Example

Skip the first 10 rows of a streamed split, then keep 20:

```text
Loader (streaming=true) ── dataset → Skip (n=10) ── dataset → Take (n=20)
```
