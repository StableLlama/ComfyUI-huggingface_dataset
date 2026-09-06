"""Hugging Face dataset loading nodes for ComfyUI.

Importing this package also registers the small HTTP endpoint (``/hfds/splits``)
that the frontend extension uses to fill the node's ``split`` dropdown with the
selected dataset's real split names. Registration is guarded so the package
still imports cleanly outside ComfyUI (tests, tooling).
"""

from typing import Any

from . import dataset_nodes
from .dataset_nodes import split_info

NODE_CLASS_MAPPINGS = {}
NODE_CLASS_MAPPINGS.update(dataset_nodes.NODE_CLASS_MAPPINGS)

NODE_DISPLAY_NAME_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS.update(dataset_nodes.NODE_DISPLAY_NAME_MAPPINGS)

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]

_ROUTES_REGISTERED = False


def _register_routes() -> None:
    """Register the ``/hfds/splits`` endpoint used by the split dropdown.

    Only meaningful inside a running ComfyUI (where ``server`` is importable and
    ``PromptServer.instance`` already exists when custom nodes are loaded);
    anywhere else this is a silent no-op.
    """
    global _ROUTES_REGISTERED
    if _ROUTES_REGISTERED:
        return
    try:
        from aiohttp import web
        from server import PromptServer
    except Exception:
        return  # not running inside ComfyUI

    server = PromptServer.instance
    if server is None or server.routes is None:
        return
    _ROUTES_REGISTERED = True

    async def splits(request: Any) -> Any:
        query = request.query
        data = split_info(
            path=query.get("path", ""),
            loader=query.get("loader", "auto"),
            config=query.get("config", ""),
            revision=query.get("revision", ""),
        )
        return web.json_response(data)

    server.routes.get("/hfds/splits")(splits)


_register_routes()
