"""Hugging Face dataset loading nodes for ComfyUI."""

from . import dataset_nodes

NODE_CLASS_MAPPINGS = {}
NODE_CLASS_MAPPINGS.update(dataset_nodes.NODE_CLASS_MAPPINGS)

NODE_DISPLAY_NAME_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS.update(dataset_nodes.NODE_DISPLAY_NAME_MAPPINGS)

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
