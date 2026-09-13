# AGENTS.md

Guidance for AI coding assistants working in this repository.

## Project overview

`huggingface_dataset` is a **ComfyUI custom-node pack**: a node that loads a
Hugging Face dataset (Hub or local/remote files) via the `datasets` library and
makes it available inside ComfyUI as an opaque `HUGGINGFACE_DATASET` value, plus
nodes that shape it and turn it into ComfyUI data (a *Data List* / *LIST* of
rows, with image cells becoming single-image `IMAGE` values and their alpha a
separate `MASK`, following ComfyUI's `IMAGE` + `MASK` convention).

## Repository layout

- `src/huggingface_dataset/` — node implementation (`dataset_nodes.py` registers
the node; `__init__.py` merges the mappings). Plain `src` layout: `src/` and
`tests/` have **no** `__init__.py`.
- `__init__.py` (repo root) — ComfyUI entry point (ComfyUI imports the folder's
`__init__.py`); it adds `src/` to `sys.path` and imports the mappings from there
(the importable package lives under `src/`, and when the repo is cloned as its
GitHub name "ComfyUI-huggingface_dataset" the folder is not a valid Python
package name).
- `tests/` — pytest suite (no network / no `datasets` install needed; the lazy
  `_require_datasets()` helper is monkeypatched).
- `web/` — frontend assets: icon (`img/`), the split-dropdown / **Force reload**
  extension (`js/hf_dataset_splits.js`, auto-loaded by ComfyUI from the
  `WEB_DIRECTORY`), and per-node in-app help pages (`docs/<NodeName>.md` — one
  file per key in `NODE_CLASS_MAPPINGS`, shown by ComfyUI's "node docs"
  feature).
- `example_workflows/` — workflow templates (+ same-name `*.jpg` thumbnails)
  that appear in ComfyUI's template browser (`Workflow → Browse Templates`).
  Regenerate them from a live ComfyUI after changing a node's inputs/outputs: a
  stale template still loads, but the serialized node must carry the current
  widget list (e.g. the loader's hidden `reload_tick`), the `localized_name`s
  and `properties` (`cnr_id` + `ver`, not the legacy `aux_id`). Keep the `.jpg`
  thumbnails consistent with the node layout.
- `requirements.txt` — mirrors `[project] dependencies` in `pyproject.toml` so
  ComfyUI-Manager / git installers auto-install the runtime dependency; the
  registry reads the `pyproject.toml` declaration. Keep the two in sync.
- `pyproject.toml` — package metadata, Comfy registry config (`[tool.comfy]`),
  and tool config (ruff / mypy / pytest).
- `CHANGELOG.md` — release notes in Keep-a-Changelog format; the publish
  workflow reads the section matching the current version.
- `.github/workflows/` — CI (`build-pipeline.yml`, `validate.yml`) and registry
  publishing (`publish_node.yml`), copied from the `basic_data_handling` pack.

## Key design points (IMPORTANT)

- **Lazy dependency:** `datasets` is declared in `[project] dependencies` but is
  imported at call time via `_require_datasets()` so ComfyUI still starts when it
  is missing; a clear "pip install datasets" error is raised on use. Do **not**
  import `datasets` at module import time.
- **Row conversion:** rows are fetched as one batched slice and re-joined into
  per-row dicts for regular (`datasets.Dataset`) loads, or pulled one row at a
  time from the iterable for `streaming` loads (`_materialize_rows_stream`).
  `_to_python()` converts numpy scalars/arrays via duck-typing (no hard numpy
  import) and leaves exotic objects (images, audio) unchanged - the conversion
  nodes pass `_to_comfy_value` instead, which also handles images (see below).
- **Loader is a pure wrapper:** `LoadHuggingFaceDataset` outputs only the
  opaque `dataset` - there is deliberately no eager `rows` output and no
  `limit` widget. Materializing rows is the conversion nodes' job
  (`HFDatasetToList` / `HFDatasetToDataList`), so a graph only pays for the
  rows it actually uses (`Load(huge) -> Take(10) -> To Data List`). Do not
  re-add eager materialization to the loader.
- **Force reload / `reload_tick`:** a hidden `reload_tick` INT input, bumped by
  the frontend **Force reload** button, invalidates ComfyUI's output cache so
  the next run re-fetches the dataset even when nothing else changed. It is a
  real input and *is* serialized (see `example_workflows/`); the button itself
  is added by the `web/js` extension and must stay unserialized
  (`serialize = false`, internal name `__hfdsReloadButton`).
- **Conversion nodes emit ComfyUI-valid values:** every *Data List* / *LIST*
  entry must be valid on its own. `_to_comfy_value()` / `_to_comfy_image()` in
  `dataset_ops_nodes.py` do what `_to_python()` does and additionally turn image
  cells into a single-image `IMAGE` batch (`[1,H,W,C]`) - never a raw PIL
  object. Only the requested column is converted (`_materialize_values`), so a
  text column of an image dataset stays cheap. The shared `_materialize_rows` /
  `_materialize_rows_stream` take a `convert` callable (`_to_python` for the
  loader's default, `_identity` for raw cells the ops module converts itself)
  so the loader module stays free of torch/numpy.
- **Alpha channels:** alpha is *not* a 4th channel (that breaks `VAEEncode`).
  Following ComfyUI's `IMAGE` + `MASK` convention (`mask = 1 - alpha`, cf.
  `LoadImage` / `SplitImageWithAlpha`), `_mask_columns()` derives a virtual
  `<image>_mask` column (`_mask1`, `_mask2`, ... on collision, logged) for image
  columns that actually carry alpha; `_convert_rows()` emits the `MASK` under
  that name. Don't switch image cells to RGBA.
- **Images:** `datasets` hands image cells over as PIL images (lazy JPEG XL
  `JXLImageFile` ones included), which is *not* a ComfyUI `IMAGE`; feeding such
  a value into `Preview Image` failed with `'JXLImageFile' object is not
  subscriptable`. `_to_comfy_value()` decodes PIL images, the undecoded
  `{"bytes": ..., "path": ...}` mapping and raw image bytes, lazily importing
  PIL/numpy/torch (all shipped by ComfyUI). Note that `_to_python` hex-encodes
  bare `bytes`, so image decoding must happen before/around it - the ops module
  therefore reads raw cells via `_identity` / `_raw_rows` and converts them
  itself.
- **Split dropdown:** the `split` COMBO offers the dataset's real split names.
  Discovery uses `datasets.get_dataset_split_names` (see `_available_splits`);
  it returns `None` on any failure (offline, multi-config without `config`) and
  the loader then passes the requested value through unchanged. The
  `/hfds/splits` HTTP endpoint (registered in `src/huggingface_dataset/__init__`
  via ComfyUI's `PromptServer.routes`) serves splits + a sensible default; the
  frontend extension `web/js/hf_dataset_splits.js` refreshes the widget when
  `path`/`loader`/`config`/`revision` change, on workflow load, and via the
  loader's **Force reload** button.
- **Streaming:** `streaming` (BOOLEAN) loads the split via `streaming=True`,
  which returns a `datasets.IterableDataset` (no `len`, single-pass) instead of
  a materialized `datasets.Dataset`. The opaque `dataset` output deliberately
  exposes either type; consumer nodes must dispatch on it. The conversion nodes
  materialize rows by iterating (`_materialize_rows_stream`) for streaming
  inputs instead of using the batched-slice path (`_materialize_rows`); a
  `Take`/`Skip` (or the conversion node's `limit`) bounds what is fetched, since
  the loader has no `limit` of its own. `trust_remote_code` was removed: modern
  `datasets` no longer supports executing Hub loading scripts.
- **Friendly errors:** the loader validates the requested split against the
  dataset's real splits - it picks a sensible default when the default `train`
  is missing and raises a clean `ValueError` listing the available splits
  otherwise - and surfaces the lazy "pip install datasets" message. No opaque
  `datasets` tracebacks reach the ComfyUI GUI.
- **Node display names & tooltips:** `NODE_DISPLAY_NAME_MAPPINGS` use a short
  `🤗 Dataset …` prefix (e.g. `🤗 Dataset Loader`) while `CATEGORY` stays
  `Hugging Face 🤗`; every required input carries a `tooltip` in its options and
  every node sets `OUTPUT_TOOLTIPS` (shown in the ComfyUI help/docs panel).
  Keep this up when adding nodes.
- ComfyUI imports are guarded with a fallback `IO`/`ComfyNodeABC` so tests run
  without ComfyUI.

## Versioning & releases

- **Single source of truth:** `[project] version` in `pyproject.toml`.
- To release: bump the version, add a `## [X.Y.Z] - YYYY-MM-DD` section to
  `CHANGELOG.md` (Keep-a-Changelog), then push to `main`. The publish workflow
  auto-triggers on pushes touching `pyproject.toml` and publishes to the Comfy
  registry; the matching changelog section becomes the release note.

## Tooling

Python venv (repo lives under `ComfyUI/custom_nodes`): `<repo>/../../venv/bin/python`

- Lint: `python -m ruff check .`
- Format: `python -m ruff format .`
- Type check: `python -m mypy src tests` (strict; not `mypy .` — the importable
  package lives under `src`, and when cloned as the GitHub name the repo folder
  "ComfyUI-huggingface_dataset" is not a valid Python package name)
- Tests: `python -m pytest tests/`
- Pre-commit: `python -m pre_commit run --all-files`

## Workflow / commit policy

- The human reviews and commits all changes (commits are GPG-signed). Do **not**
  run `git commit` yourself; make the changes, optionally stage them, and hand
  off to the user.
