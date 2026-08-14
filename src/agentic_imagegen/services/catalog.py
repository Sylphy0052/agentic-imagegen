"""モデル一覧などの列挙系操作の抽象。

services/generation.py の GenerationBackend と同じ考え方で、列挙系の操作にも
Protocolを用意する。ComfyUIClient はこのProtocolを構造的に満たすが、
Service層はComfyUI固有の型を知らないままこのProtocol越しに扱う。
"""

from __future__ import annotations

from collections.abc import Callable
from contextlib import AbstractAsyncContextManager
from typing import Protocol

from agentic_imagegen.config import Settings
from agentic_imagegen.services.generation import GenerationBackend


class CatalogBackend(Protocol):
    """checkpoint / LoRA などの一覧取得に求める操作。

    ComfyUIClient はこのProtocolを構造的に満たす。
    将来 diffusers や remote API を足す場合も、この形に合わせれば
    Service層を変更せずに差し替えられる。
    """

    async def available_checkpoints(self) -> tuple[str, ...]: ...

    async def available_loras(self) -> tuple[str, ...]: ...

    async def available_controlnets(self) -> tuple[str, ...]: ...

    async def available_ipadapters(self) -> tuple[str, ...]: ...

    async def available_clip_visions(self) -> tuple[str, ...]: ...

    async def available_diffusion_models(self) -> tuple[str, ...]: ...

    async def available_text_encoders(self) -> tuple[str, ...]: ...

    async def available_vaes(self) -> tuple[str, ...]: ...

    async def available_upscale_models(self) -> tuple[str, ...]: ...

    async def available_embeddings(self) -> tuple[str, ...]: ...


#: バックエンドを開くファクトリ。`ComfyUIClient(settings)` のように Settings を
#: 受け取り、async context manager (使い終えたら接続を閉じられるもの) を返す
#: 呼び出し可能をこの形で表す。列挙系 (CatalogBackend) と生成系
#: (GenerationBackend) のどちらも同じ形になるため、2つとも定義しておく。
type CatalogBackendFactory = Callable[[Settings], AbstractAsyncContextManager[CatalogBackend]]
type GenerationBackendFactory = Callable[[Settings], AbstractAsyncContextManager[GenerationBackend]]


__all__ = ["CatalogBackend", "CatalogBackendFactory", "GenerationBackendFactory"]
