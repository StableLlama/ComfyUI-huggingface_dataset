# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.1.0] - 2026-09-09

### Added

- `HFDatasetCount` (`🤗 Dataset Count`) — returns the number of entries (rows)
  of a fully-loaded dataset as an `INT` (`len`). Counting needs a materialized
  `datasets.Dataset`; a streaming `IterableDataset` input raises a clear
  "disable streaming" error instead of an opaque traceback, matching the other
  loaded-only operations (`sort`, `select`, `unique`, `flatten`, ...). Includes
  in-app documentation, a README entry, unit tests and a proper `tooltip` /
  `OUTPUT_TOOLTIPS` pair.

## [1.0.0] - 2026-09-06

Initial release of the `huggingface_dataset` pack for ComfyUI: load a Hugging
Face dataset (Hub, or local/remote files), shape it declaratively and hand its
rows to the "Basic data handling" pack — no free-form Python, no `datasets`
tracebacks in the GUI.

### Added

- `LoadHuggingFaceDataset`, the core node: loads a dataset from the Hugging
  Face Hub or from local/remote files (`csv`, `json`, `jsonl`, `parquet`,
  `arrow`, `txt`) via the `datasets` library, with support for a config
  subset, a `split` (incl. slicing), a `revision`, a `streaming` toggle and a
  row `limit`.
- The loader exposes the dataset two ways: as an opaque `HUGGINGFACE_DATASET`
  output (consumed by the transform / conversion nodes below) and as a ComfyUI
  *Data List* of row dicts, ready for further processing with generic
  data-handling nodes.
- A `streaming` toggle: when enabled, the split is loaded as a lazy
  `datasets.IterableDataset` (no full download/cache) and the Data List output
  is materialized by iterating up to `limit` rows. The opaque `dataset` output
  is then an `IterableDataset` instead of a `datasets.Dataset`; consumer nodes
  dispatch on both types.
- A `split` dropdown (COMBO) that lists the splits the selected dataset
  actually exposes, auto-refreshes when the source (`path`, `config`,
  `revision`, `loader`) changes and offers a manual "Refresh splits" button
  (small frontend extension plus a `/hfds/splits` endpoint). When a dataset
  has no `train` split, a sensible default is picked (`train` → `validation` →
  `test`).
- Friendly errors: a wrong/unknown split raises a clear message listing the
  available splits instead of a `datasets` traceback, and a missing `datasets`
  package yields an actionable "pip install datasets" message in the ComfyUI
  GUI.
- A family of transform nodes that consume the opaque `HUGGINGFACE_DATASET`
  value (the loader's `dataset` output) and expose the data-wrangling methods
  of the `datasets` library as ComfyUI nodes, so a dataset can be shaped
  *before* its rows are turned into a Data List: `HFDatasetShuffle`,
  `HFDatasetSkip`, `HFDatasetTake`, `HFDatasetSort`, `HFDatasetShard`,
  `HFDatasetSelect` (rows by index), `HFDatasetSelectColumns`,
  `HFDatasetRemoveColumns`, `HFDatasetRenameColumn`, `HFDatasetFlatten`,
  `HFDatasetFilter`, `HFDatasetMapColumn`, `HFDatasetSplit` (train/test) and
  `HFDatasetUnique`. Where the underlying method exists on both kinds of
  object they work on a fully-loaded `datasets.Dataset` *and* on a lazy
  streaming `datasets.IterableDataset`; operations that only exist on a
  materialized dataset (`sort`, `select`, `flatten`, `train_test_split`,
  `unique`) raise a friendly error when given a streaming one instead of an
  opaque traceback.
- Declarative per-row logic without free-form Python: `HFDatasetFilter` keeps
  rows by column + operator + value (`==`, `!=`, `<`, `<=`, `>`, `>=` with
  numeric coercion, `contains`/`not contains`/`starts with`/`ends with`,
  `in`/`not in`, `is null`/`is not null`), and `HFDatasetMapColumn` adds or
  replaces a column per row from a constant, a copy of another column, or the
  row index.
- Conversion nodes that turn any `HUGGINGFACE_DATASET` (fully-loaded *or*
  streaming) into plain data for the "Basic data handling" pack:
  `HFDatasetToList` materializes it as a single *LIST* value and
  `HFDatasetToDataList` as a ComfyUI *Data List* of rows. Both expose just one
  column's values when a `column` is given and honour a row `limit`.
- Tooltips on every input and output (the `tooltip` option in `INPUT_TYPES`
  plus `OUTPUT_TOOLTIPS`), so hovering a widget shows help and the ComfyUI
  node documentation panel lists rich parameter descriptions instead of empty
  rows.
- Per-node in-app documentation (`web/docs/<NodeName>.md` for every node):
  each node shows a rich help page in the ComfyUI UI (see ComfyUI's "Add node
  docs" help-page feature).
- Ready-made example workflows under `example_workflows/` that appear in
  ComfyUI's template browser (`Workflow → Browse Templates`): filter the
  positive IMDb reviews to a Data List, add a row index / keep columns and
  export to a LIST, and a streaming skip/take example — each with a thumbnail.
- Documentation: a README "Working with Basic data handling" guide plus in-node
  doc call-outs that show how the `rows` / `LIST` / `Data List` outputs plug
  straight into the "Basic data handling" pack (whole-list nodes and per-row
  mapping recipes, using its real menu labels).
- JPEG XL (`.jxl`) image decoding: the pack opportunistically imports
  `pillow_jxl` (the Pillow JPEG XL plugin from `pip install pillow-jxl-plugin`)
  so it registers its decoder with Pillow when installed. The plugin stays
  optional — without it, loading keeps working and only the JPEG XL images fail
  to decode.
- The ComfyUI startup log reports that the Hugging Face dataset custom nodes
  were loaded, together with whether JPEG XL image support is `enabled` or
  `disabled`.
- Node display names use a short `🤗` prefix for easier menus (e.g.
  `🤗 Dataset Loader`); the node menu category is `Hugging Face 🤗`.
- Packaging: `datasets` is declared as a runtime dependency but imported lazily
  (`_require_datasets()`) so ComfyUI still starts when it is missing, mirrored
  in a root `requirements.txt` so git-based installers (ComfyUI-Manager, manual
  clones) install it automatically; `requires-python = ">=3.10"` and a
  Documentation URL follow Comfy Registry metadata best practices. There is no
  `trust_remote_code` option — modern `datasets` no longer supports executing
  Hub loading scripts.
