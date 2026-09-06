# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

- `split` is now a dropdown (COMBO) that lists the splits the selected dataset
  actually exposes instead of a free-text field. It auto-refreshes when the
  source (`path`, `config`, `revision`, `loader`, `trust_remote_code`) changes
  and offers a manual "Refresh splits" button (small frontend extension + a
  `/hfds/splits` endpoint).
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
