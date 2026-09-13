# 🤗 Dataset To LIST

Converts a `HUGGINGFACE_DATASET` (fully-loaded **or** streaming) into a single
Python list value — a *LIST* as defined by the **Basic data handling** pack — so
it feeds the Basic `LIST` nodes such as `length`, `get item`, etc.

## Inputs

| Input | Type | Default | Description |
| --- | --- | --- | --- |
| `dataset` | `HUGGINGFACE_DATASET` | — | A dataset from the Loader (or another dataset node). |
| `column` | STRING | `""` | When set, the list holds that column's values instead of whole row dicts. |
| `limit` | INT | `-1` | Max rows to materialize. `-1` = all. |

## Outputs

| Output | Type | Description |
| --- | --- | --- |
| `list` | `LIST` | One Python list: of row dicts, or of a single column's values when `column` is set. |

## Notes

- Works on fully-loaded **and** streaming datasets.
- Every item is a value that is valid on its own in ComfyUI: numpy
  scalars/arrays become plain Python, and image cells become a single-image
  `IMAGE` (a batch of one). An image column whose cells carry an alpha channel
  additionally gets a virtual `<column>_mask` entry holding the alpha as a
  `MASK` (`1` = transparent).
- Only the requested column is converted; a virtual `<image>_mask` column can be
  requested by name. See **To Data List** for the full alpha-channel details.
- Handy for pulling only the first rows of a large streaming dataset: set
  `column` and `limit`, or chain **Take** first.
- Use **To Data List** instead when you want the rows as a ComfyUI *Data List*
  (one item per row) rather than one whole list value.

## Example

Expose the review text of the first 100 rows as a LIST of strings:

```text
Loader (streaming=true) ── dataset → Take (n=100) ── dataset
  → To LIST (column=text) ── list → LIST → length / get item (Basic data handling)
```
