// ComfyUI extension: refresh the "split" dropdown of the Hugging Face Dataset
// Loader with the actual splits of the dataset entered in "path".
//
// * auto-refreshes (debounced) when `path`/`loader`/`config`/`revision`/
//   `trust_remote_code` change,
// * refreshes when a workflow is loaded / the node is configured,
// * offers a manual "Refresh splits" button.
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
const TRIGGER_INPUTS = ["path", "loader", "config", "revision", "trust_remote_code"];

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
    trust_remote_code: findWidget(node, "trust_remote_code")?.value ? "1" : "0",
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

  // Manual refresh button (never serialized into workflows).
  if (!findWidget(node, "__hfdsRefreshButton")) {
    const button = node.addWidget("button", "Refresh splits", null, () => {
      refreshSplits(node);
    });
    button.name = "__hfdsRefreshButton";
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
