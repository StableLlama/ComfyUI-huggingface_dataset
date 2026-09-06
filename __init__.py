"""Top-level package for the ``huggingface_dataset`` ComfyUI custom node pack.

This is the entry point that ComfyUI imports when it loads a *directory*-based
custom node (ComfyUI imports the ``__init__.py`` of the folder under
``custom_nodes/``).
"""

import os
import sys

__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
    "WEB_DIRECTORY",
]

# The importable package lives under ``src/`` (src-layout), and the name of the
# folder this repository is cloned into is not guaranteed to be a valid Python
# package name (ComfyUI-Manager, for instance, keeps the repo name
# "ComfyUI-huggingface_dataset"). Add ``src`` to ``sys.path`` and import from
# there so the mappings below are always available.
_SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "src"))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from huggingface_dataset import NODE_CLASS_MAPPINGS  # noqa: E402
from huggingface_dataset import NODE_DISPLAY_NAME_MAPPINGS  # noqa: E402

WEB_DIRECTORY = "./web"
