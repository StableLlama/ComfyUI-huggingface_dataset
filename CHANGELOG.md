# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

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
