# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Per-node in-app documentation (`web/docs/<NodeName>.md` for every node): each
  node now shows a rich help page in the ComfyUI UI (see ComfyUI's "Add node
  docs" help-page feature).
- Ready-made example workflows under `example_workflows/` that appear in
  ComfyUI's template browser (`Workflow → Browse Templates`): filter the
  positive IMDb reviews to a Data List, add a row index / keep columns and
  export to a LIST, and a streaming skip/take example — each with a thumbnail.
- A root `requirements.txt` so git-based installers (ComfyUI-Manager, manual
  clones) install the `datasets` dependency automatically, mirroring the
  `pyproject.toml` dependency already used by Comfy-Registry installs.
- Every node input and output now carries a tooltip (the `tooltip` option in
  `INPUT_TYPES` plus `OUTPUT_TOOLTIPS`), so hovering a widget shows help and the
  ComfyUI node documentation panel lists rich parameter descriptions instead of
  empty rows.
- A family of transform nodes that consume the opaque `HUGGINGFACE_DATASET`
  value (the loader's `dataset` output) and expose the data-wrangling methods of
  the `datasets` library as ComfyUI nodes, so a dataset can be shaped *before*
  its rows are turned into a Data List. The new nodes are `HFDatasetShuffle`,
  `HFDatasetSkip`, `HFDatasetTake`, `HFDatasetSort`, `HFDatasetShard`,
  `HFDatasetSelect` (rows by index), `HFDatasetSelectColumns`,
  `HFDatasetRemoveColumns`, `HFDatasetRenameColumn`, `HFDatasetFlatten`,
  `HFDatasetFilter`, `HFDatasetMapColumn`, `HFDatasetSplit` (train/test) and
  `HFDatasetUnique`. Where the underlying method exists on both kinds of object
  they work on a fully-loaded `datasets.Dataset` *and* on a lazy streaming
  `datasets.IterableDataset`; operations that only exist on a materialized
  dataset (`sort`, `select`, `flatten`, `train_test_split`, `unique`) raise a
  friendly error when given a streaming one instead of an opaque traceback.
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
- Datasets whose images are JPEG XL (`.jxl`) can now be decoded: the pack
  opportunistically imports `pillow_jxl` (the Pillow JPEG XL plugin from
  `pip install pillow-jxl-plugin`) so it registers its decoder with Pillow when
  installed. The plugin stays optional — without it, loading keeps working and
  only the JPEG XL images fail to decode.
- The ComfyUI startup log now reports that the Hugging Face dataset custom nodes
  were loaded, together with whether JPEG XL image support is `enabled` or
  `disabled`.

### Changed

- README: install instructions state that the `datasets` dependency is
  installed automatically, cover ComfyUI-Manager / Registry installs, document
  the new template workflows, and use `pip install -r requirements.txt` for
  manual installs (plus a corrected strict-`mypy` command).
- Packaging metadata (`pyproject.toml`): declare `requires-python = ">=3.10"`
  and add a Documentation URL to follow Comfy Registry metadata best
  practices.
- Node display names are shortened to a `🤗` prefix for easier menus (e.g.
  `🤗 Dataset Loader` instead of `Hugging Face Dataset Loader`); the node menu
  category stays `Hugging Face 🤗`. All docs and the README were updated to
  match.
- Drop the `trust_remote_code` input: modern `datasets` no longer supports
  executing Hub loading scripts and merely ignores the flag (logging a
  "trust_remote_code is not supported anymore" error), so the option was dead
  weight and could only confuse.
- Add a `streaming` toggle: when enabled, the split is loaded as a lazy
  `datasets.IterableDataset` (no full download/cache) and `rows` is
  materialized by iterating up to `limit` rows. The opaque `dataset` output is
  then an `IterableDataset` instead of a `datasets.Dataset`, and consumer nodes
  must handle both types.
- `split` is now a dropdown (COMBO) that lists the splits the selected dataset
  actually exposes instead of a free-text field. It auto-refreshes when the
  source (`path`, `config`, `revision`, `loader`) changes and offers a manual
  "Refresh splits" button (small frontend extension + a `/hfds/splits`
  endpoint).
- A sensible default split is selected (in the order `train` → `validation` →
  `test`) when the dataset has no `train` split, so datasets that only ship e.g.
  a `test` split no longer fail on the default.
- Errors are now user friendly: a wrong/unknown split raises a clear message
  listing the available splits instead of a `datasets` traceback, and a missing
  `datasets` package yields an actionable "pip install datasets" message in the
  ComfyUI GUI.

## [0.1.0] - 2026-09-06

- Add `LoadHuggingFaceDataset` node that loads a Hugging Face dataset from the
  Hub or from local/remote files (`csv`, `json`, `jsonl`, `parquet`, `arrow`,
  `txt`) via the `datasets` library.
- The node exposes the loaded dataset as an opaque `HUGGINGFACE_DATASET` output
  and as a ComfyUI *Data List* of row dicts, ready for further processing with
  generic data-handling nodes.
- Support config subset, split (incl. slicing), revision, `trust_remote_code`
  and a row `limit` for the Data List output.
