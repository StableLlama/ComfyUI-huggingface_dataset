# 🤗 Dataset To Data List

Converts a `HUGGINGFACE_DATASET` (fully-loaded **or** streaming) into a ComfyUI
*Data List* of items — one item per row (or one value per row of `column` when
one is given). This is the same shape as the Loader's `rows` output.

## Inputs

| Input | Type | Default | Description |
| --- | --- | --- | --- |
| `dataset` | `HUGGINGFACE_DATASET` | — | A dataset from the Loader (or another dataset node). |
| `column` | STRING | `""` | When set, each item is that column's value instead of a whole row dict. |
| `limit` | INT | `-1` | Max rows to materialize. `-1` = all. |

## Outputs

| Output | Type | Description |
| --- | --- | --- |
| `rows` | `*` (Data List) | Data List of row dicts (or of a single column's values). |

## Notes

- Works on fully-loaded **and** streaming datasets.
- Downstream *Data List* nodes of the **Basic data handling** pack receive the
  whole list in one call; most other nodes run once per item.
- Use **To LIST** instead when you want one whole Python list value rather than
  a Data List.

## Example

Filter positive IMDb reviews and hand the rows to a *Data List* node of Basic
data handling:

```text
Loader (path=stanfordnlp/imdb) ── dataset → Filter (column=label, ==, value=1)
  ── dataset → To Data List (limit=100) ── rows → Data List → length (Basic data handling)
```

You can also map over the rows: connect the `rows` Data List straight into a
`Basic/DICT → get`, `STRING` or `cast` node, and it runs once per row.
