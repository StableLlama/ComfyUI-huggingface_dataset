# 🤗 Dataset Filter

Keeps the rows of a `HUGGINGFACE_DATASET` whose `column` satisfies a
declarative condition (`datasets` `filter`). No free-form Python needed — pick
a column, an operator and a value.

## Inputs

| Input | Type | Default | Description |
| --- | --- | --- | --- |
| `dataset` | `HUGGINGFACE_DATASET` | — | A dataset from the Loader (or another dataset node). |
| `column` | STRING | `""` | The column to test. |
| `operator` | COMBO | `==` | The comparison operator (see below). |
| `value` | STRING | `""` | The value to compare against. |

## Outputs

| Output | Type | Description |
| --- | --- | --- |
| `dataset` | `HUGGINGFACE_DATASET` | The dataset with only the matching rows. |

## Operators

- **Equality / ordering** — `==`, `!=`, `<`, `<=`, `>`, `>=`. The `value` text
  is coerced to the column's type, so numeric columns compare with plain
  numbers (`label` `==` `1`).
- **Text** — `contains`, `not contains`, `starts with`, `ends with`.
- **Membership** — `in` / `not in` with a comma-separated `value` list, e.g.
  `value = apple, banana, cherry`.
- **Presence** — `is null` / `is not null` (`value` is ignored).

## Notes

- Works on fully-loaded **and** streaming datasets.
- Unknown columns and operators that are missing a required `value` raise clear
  errors.

## Example

Keep only positive IMDb reviews, then add a row index:

```text
Loader (path=stanfordnlp/imdb) ── dataset → Filter (column=label, ==, value=1)
  └─ dataset → 🤗 Dataset Map Column (column=row_id, operation=row index)
```
