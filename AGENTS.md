# AGENTS.md

Guidance for AI coding assistants working in this repository.

## Project overview

`huggingface_dataset` is a **ComfyUI custom-node pack**: a node that loads a
Hugging Face dataset (Hub or local/remote files) via the `datasets` library and
makes it available inside ComfyUI as an opaque `HUGGINGFACE_DATASET` output plus
a ComfyUI *Data List* of row dicts for further processing.

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
- `web/` — frontend assets: icon (`img/`) and the split-dropdown extension
  (`js/hf_dataset_splits.js`, auto-loaded by ComfyUI from the `WEB_DIRECTORY`).
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
  import) and leaves exotic objects (images, audio) unchanged.
- **Data List output:** the `rows` output uses `OUTPUT_IS_LIST = (False, True)`
  so it behaves like the "Data List" producers of the Basic data handling pack.
- **Split dropdown:** `split` is a COMBO (fallback options `train`/`test`/
  `validation`) whose real options come from the dataset. Split discovery uses
  `datasets.get_dataset_split_names` (see `_available_splits`); it returns
  `None` on any failure (offline, multi-config without `config`) and the loader
  then passes the requested value through unchanged. The `/hfds/splits` HTTP
  endpoint (registered in `src/huggingface_dataset/__init__` via ComfyUI's
  `PromptServer.routes`) serves splits + a sensible default; the frontend
  extension `web/js/hf_dataset_splits.js` refreshes the widget on
  `path`/`config`/`revision`/`loader` changes, on workflow load and via its
  "Refresh splits" button.
- **Streaming:** `streaming` (BOOLEAN) loads the split via `streaming=True`,
  which returns a `datasets.IterableDataset` (no `len`, single-pass) instead of
  a materialized `datasets.Dataset`. The opaque `dataset` output deliberately
  exposes either type; consumer nodes must dispatch on it. `rows` is
  materialized by iterating (`_materialize_rows_stream`) when streaming rather
  than by the batched-slice path (`_materialize_rows`), and `limit` bounds what
  is fetched. `trust_remote_code` was removed: modern `datasets` no longer
  supports executing Hub loading scripts.
- **Friendly errors:** the loader validates the requested split against the
  dataset's real splits - it picks a sensible default when the default `train`
  is missing and raises a clean `ValueError` listing the available splits
  otherwise - and surfaces the lazy "pip install datasets" message. No opaque
  `datasets` tracebacks reach the ComfyUI GUI.
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
