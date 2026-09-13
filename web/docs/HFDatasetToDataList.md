# 🤗 Dataset To Data List

Converts a `HUGGINGFACE_DATASET` (fully-loaded **or** streaming) into a ComfyUI
*Data List* of items — one item per row (or one value per row of `column` when
one is given). Feed it the `dataset` output of the Loader (after shaping it with
the transform nodes) or of any other dataset node.

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
- Every item is a value that is valid on its own in ComfyUI: numpy
  scalars/arrays become plain Python, and image cells become a single-image
  `IMAGE` (a batch of one). `column=image` therefore yields a Data List you can
  wire straight into `Preview Image`, `Save Image`, `ImageScale`, …
  (JPEG XL images need the optional `pillow-jxl-plugin`).
- Only the requested column is converted, so asking for a text column of an
  image dataset stays cheap.
- Use **To LIST** instead when you want one whole Python list value rather than
  a Data List.

## Alpha channels (virtual `<image>_mask` columns)

ComfyUI's `IMAGE` is RGB — transparency belongs in a separate `MASK` (convention:
`1` = transparent, the same one `LoadImage` and `SplitImageWithAlpha` use). This
node follows that instead of inventing a 4-channel image: an image column whose
cells carry an **alpha channel** (PNG or JPEG XL with transparency) gets an
additional **virtual column** named `<column>_mask` holding the mask.

- Without a `column` (whole rows) every row dict contains both keys:

  ```text
  {"image": IMAGE, "image_mask": MASK, "label": …}
  ```

  so `DICT → get (key=image_mask)` feeds any `MASK` input
  (`ImageCompositeMasked`, `SetLatentNoiseMask`, `GrowMask`, …).
- The mask column is addressable like a real one: `column=image_mask` gives a
  Data List of masks.
- `image` + `image_mask` → `Join Image with Alpha` rebuilds the RGBA image in
  case a node needs one.
- An image requested as `column=image` stays RGB (the alpha is *not* composited
  away — the RGB values under transparent pixels are kept as stored).
- Images **without** an alpha channel produce no virtual column at all.
- **Name collision:** if the dataset already has a column called
  `<column>_mask`, the virtual one becomes `<column>_mask1`,
  `<column>_mask2`, … and the chosen name is logged to the ComfyUI console. A
  real column always keeps its name.

## Example

Filter positive IMDb reviews and hand the rows to a *Data List* node of Basic
data handling:

```text
Loader (path=stanfordnlp/imdb) ── dataset → Filter (column=label, ==, value=1)
  ── dataset → To Data List (limit=100) ── rows → Data List → length (Basic data handling)
```

You can also map over the rows: connect the `rows` Data List straight into a
`Basic/DICT → get`, `STRING` or `cast` node, and it runs once per row.

Preview images that have transparency, and reuse their alpha mask:

```text
Loader (path=…, limit via Take) ── dataset → To Data List (limit=8)
  ── rows → DICT → get (key=image)       → Preview Image
  ── rows → DICT → get (key=image_mask)  → SetLatentNoiseMask / ImageCompositeMasked
```
