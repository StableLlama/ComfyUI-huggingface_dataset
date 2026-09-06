# 🤗 Dataset Shard

Splits a `HUGGINGFACE_DATASET` into `num_shards` roughly equal pieces and hands
back shard `index` (`datasets` `shard`).

## Inputs

| Input | Type | Default | Description |
| --- | --- | --- | --- |
| `dataset` | `HUGGINGFACE_DATASET` | — | A dataset from the Loader (or another dataset node). |
| `num_shards` | INT | `2` | How many shards to split into (≥ 1). |
| `index` | INT | `0` | Which shard to keep (`0` .. `num_shards-1`). |
| `contiguous` | BOOLEAN | `true` | Keep contiguous blocks of rows (`true`) or interleave (`false`). |

## Outputs

| Output | Type | Description |
| --- | --- | --- |
| `dataset` | `HUGGINGFACE_DATASET` | Shard `index` of the split dataset. |

## Notes

- Works on fully-loaded **and** streaming datasets.
- `index` must be smaller than `num_shards`, otherwise a clear error is raised.

## Example

Keep the second half of a streamed split (2 shards, index 1):

```text
Loader (streaming=true) ── dataset → Shard (num_shards=2, index=1) ── dataset → 🤗 Dataset To Data List
```
