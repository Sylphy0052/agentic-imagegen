"""どのバックエンドの具象を使うかの選択。

composition root (CLI / MCP Server) から使う。ここだけが `IMAGEGEN_BACKEND` と
adapters層の具象クラスの対応を知っており、services / domain 層はProtocol越しの
ままでいられる (services層がadaptersをimportしないことは
tests/unit/test_service_layer_isolation.py が守る)。

diffusersバックエンドの具象はtorchを必要とするため、選ばれたときにだけimportする。
"""

from __future__ import annotations

from contextlib import AbstractAsyncContextManager

from agentic_imagegen.adapters.comfyui.client import ComfyUIClient
from agentic_imagegen.config import Settings
from agentic_imagegen.services.catalog import CatalogBackend
from agentic_imagegen.services.generation import GenerationBackend


def open_generation_backend(settings: Settings) -> AbstractAsyncContextManager[GenerationBackend]:
    """設定に従って生成用のバックエンドを開く。

    `GenerationBackendFactory` として渡せる形 (Settings を受け取り
    async context manager を返す)。
    """
    if settings.backend == "diffusers":
        from agentic_imagegen.adapters.diffusers.backend import DiffusersBackend

        return DiffusersBackend(settings)
    return ComfyUIClient(settings)


def open_catalog_backend(settings: Settings) -> AbstractAsyncContextManager[CatalogBackend]:
    """設定に従って列挙用のバックエンドを開く。

    `CatalogBackendFactory` として渡せる形。
    """
    if settings.backend == "diffusers":
        from agentic_imagegen.adapters.diffusers.catalog import DiffusersCatalog

        return DiffusersCatalog(settings)
    return ComfyUIClient(settings)


__all__ = ["open_catalog_backend", "open_generation_backend"]
