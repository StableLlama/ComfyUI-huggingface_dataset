// ComfyUI extension: refresh the "split" dropdown of the Hugging Face Dataset
// Loader with the actual splits of the dataset entered in "path", and add a
// "Force reload" button that refreshes the dropdown and marks the dataset for
// a fresh re-fetch on the next run.
//
// * the split dropdown auto-refreshes (debounced) when
//   `path`/`loader`/`config`/`revision` change, and refreshes when a workflow
//   is loaded / the node is configured;
// * a single "Force reload" button re-queries the splits and bumps the hidden
//   `reload_tick` input so the *next* queue re-loads the data fresh even when
//   the other inputs are unchanged.
//
// The values below must stay in sync with `_DEFAULT_SPLIT` / `_FALLBACK_SPLITS`
// in src/huggingface_dataset/dataset_nodes.py.
// Note: read `app` from `window.comfyAPI` instead of importing
// "../../scripts/app.js" - the relative import would break because this file is
// served from a nested `web/js/` folder.
const { app } = window.comfyAPI.app;

const NODE_CLASS = "LoadHuggingFaceDataset";
const DEFAULT_SPLIT = "train";
const FALLBACK_SPLITS = ["train", "test", "validation"];
const PREFERRED_ORDER = [DEFAULT_SPLIT, "validation", "test"];
const TRIGGER_INPUTS = ["path", "loader", "config", "revision"];

function findWidget(node, name) {
  return (node.widgets || []).find((widget) => widget.name === name) || null;
}

// The part of a split value before any `datasets` slicing suffix
// (e.g. "train" for "train[:100]" or "train[:10%]").
function splitBase(value) {
  const text = String(value ?? "").trim();
  const index = text.indexOf("[");
  return index === -1 ? text : text.slice(0, index).trim();
}

function preferredSplit(splits) {
  if (!splits.length) return DEFAULT_SPLIT;
  for (const candidate of PREFERRED_ORDER) {
    if (splits.includes(candidate)) return candidate;
  }
  return splits[0];
}

function setSplitOptions(node, splitWidget, splits) {
  if (!splitWidget) return;
  splitWidget.options = splitWidget.options || {};
  splitWidget.options.values = splits.slice();

  const current = String(splitWidget.value ?? "").trim();
  const base = splitBase(current);

  // Keep the current choice when it is still valid: an exact split name, or a
  // slicing expression (e.g. "train[:100]") whose base split still exists.
  if (current && (splits.includes(current) || splits.includes(base))) {
    return;
  }

  // Otherwise fall back to a sensible default. Only replace a leftover generic
  // value (the default "train" on a dataset without a "train" split, an empty
  // value, or something not in the new list) so an explicit choice is kept.
  if (!current || FALLBACK_SPLITS.includes(current) || !splits.includes(current)) {
    splitWidget.value = preferredSplit(splits);
  }

  if (app.graph) app.graph.setDirtyCanvas(true, true);
}

async function refreshSplits(node) {
  const pathWidget = findWidget(node, "path");
  const splitWidget = findWidget(node, "split");
  if (!pathWidget || !splitWidget) return;

  const path = String(pathWidget.value ?? "").trim();
  if (!path) {
    setSplitOptions(node, splitWidget, FALLBACK_SPLITS);
    return;
  }

  const params = new URLSearchParams({
    path,
    loader: String(findWidget(node, "loader")?.value ?? "auto"),
    config: String(findWidget(node, "config")?.value ?? ""),
    revision: String(findWidget(node, "revision")?.value ?? ""),
  });

  let payload;
  try {
    const response = await fetch(`/hfds/splits?${params.toString()}`);
    payload = await response.json();
  } catch (error) {
    console.warn("[Hugging Face Dataset] Could not list splits:", error);
    return;
  }

  if (!payload || payload.error) {
    console.warn(
      "[Hugging Face Dataset] Could not list splits:",
      (payload && payload.error) || "unknown error",
    );
    return;
  }

  const splits =
    Array.isArray(payload.splits) && payload.splits.length ? payload.splits : FALLBACK_SPLITS;
  setSplitOptions(node, splitWidget, splits);
}

function scheduleRefresh(node) {
  clearTimeout(node.__hfdsRefreshTimer);
  node.__hfdsRefreshTimer = setTimeout(() => {
    refreshSplits(node);
  }, 350);
}

// The single "Force reload" button: refresh the split dropdown with the
// dataset's real splits, then mark the loader to re-fetch the dataset on the
// *next* run even when every other input is unchanged. ComfyUI caches a node's
// output on its inputs, so we bump the hidden `reload_tick` counter (a real
// input of the node) to invalidate that cache; the next queued run then loads
// the dataset fresh (including any offline/local updates to the source).
//
// The button deliberately does *not* queue a run itself: it only makes the
// cached dataset stale, so nothing starts computing behind the user's back.
async function forceReload(node) {
  // Keep the split dropdown in sync first. If a previously selected split no
  // longer exists, refreshSplits picks a sensible default, so the next run uses
  // a valid split. It warns on failure rather than throwing, but guard anyway so
  // a split-refresh hiccup can never block the cache-busting tick.
  try {
    await refreshSplits(node);
  } catch (error) {
    console.warn("[Hugging Face Dataset] Could not refresh splits while reloading:", error);
  }

  // Invalidate ComfyUI's output cache for this node: the bumped `reload_tick`
  // is a real input, so the next queue re-runs the loader and re-fetches the
  // dataset instead of reusing the cached one.
  const tick = findWidget(node, "reload_tick");
  if (tick) {
    tick.value = (Number(tick.value) || 0) + 1;
  }
  app.graph.setDirtyCanvas(true, true);
}

function hookUpNode(node) {
  if (!findWidget(node, "split")) return;

  for (const name of TRIGGER_INPUTS) {
    const widget = findWidget(node, name);
    if (!widget || widget.__hfdsRefreshWrapped) continue;
    widget.__hfdsRefreshWrapped = true;
    const previous = widget.callback;
    widget.callback = function (...args) {
      if (typeof previous === "function") previous.apply(this, args);
      scheduleRefresh(node);
    };
  }

  // "Force reload" button: refreshes the split dropdown and bumps the hidden
  // `reload_tick` counter to invalidate ComfyUI's output cache, so the next
  // queue re-fetches the dataset. It never queues a run itself. It is the only
  // button (it also covers the old "Refresh splits" action, so that manual
  // button is not needed). Never serialized.
  //
  // The widget keeps a stable "__" internal name so the duplicate-guard
  // (findWidget) can find it again when hookUpNode runs on the onConfigure pass
  // of the same node instance. ComfyUI renders button widgets as `label || name`,
  // so without an explicit label the raw internal name would be shown - set a
  // friendly caption instead.
  const reloadTick = findWidget(node, "reload_tick");
  if (reloadTick) {
    reloadTick.hidden = true;
    reloadTick.options = reloadTick.options || {};
    reloadTick.options.hidden = true;
  }
  if (!findWidget(node, "__hfdsReloadButton")) {
    const button = node.addWidget("button", "Force reload", null, () => {
      forceReload(node);
    });
    button.name = "__hfdsReloadButton";
    button.label = "Force reload";
    button.serialize = false;
    button.serializeValue = () => undefined;
  }
}

app.registerExtension({
  name: "StableLlama.HuggingFaceDataset.SplitRefresh",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== NODE_CLASS) return;

    const onNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function (...args) {
      const result = onNodeCreated ? onNodeCreated.apply(this, args) : undefined;
      hookUpNode(this);
      scheduleRefresh(this);
      return result;
    };

    // Refresh again once a saved workflow has been applied, so the dropdown
    // reflects the values that were just loaded (path/config/revision/...).
    const onConfigure = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function (...args) {
      const result = onConfigure ? onConfigure.apply(this, args) : undefined;
      hookUpNode(this);
      scheduleRefresh(this);
      return result;
    };
  },
});
