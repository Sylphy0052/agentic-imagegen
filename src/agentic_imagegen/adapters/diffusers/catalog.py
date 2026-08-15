"""diffusersバックエンドでの列挙系 (どのモデルが使えるか)。

ComfyUIバックエンドはHTTPで問い合わせるが、diffusersはローカルの
`IMAGEGEN_MODELS_ROOT` 配下を見るだけで済む。`CatalogBackend` Protocol
(services/catalog.py) を構造的に満たす。

diffusersバックエンドで使えない区分 (ControlNet / IPAdapter / DiT系) は、
ファイルが置いてあっても使えないため空を返す。
"""

from __future__ import annotations

from pathlib import Path
from types import TracebackType
from typing import Final

from agentic_imagegen.config import Settings
from agentic_imagegen.domain.models import ALLOWED_CHECKPOINT_SUFFIXES, ALLOWED_LORA_SUFFIXES
from agentic_imagegen.errors import InvalidConfiguration

#: embeddingは拡張子を外した名前で prompt から参照する。
_EMBEDDING_SUFFIXES: Final = frozenset({".safetensors", ".pt", ".bin"})


class DiffusersCatalog:
    """models_root 配下のファイル一覧を返す。

    ComfyUIClient と同じく async context manager として使う
    (`CatalogBackendFactory` が要求する形)。開く接続は無い。
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    async def __aenter__(self) -> DiffusersCatalog:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> bool:
        return False

    async def available_checkpoints(self) -> tuple[str, ...]:
        return self._entries("checkpoints", ALLOWED_CHECKPOINT_SUFFIXES)

    async def available_loras(self) -> tuple[str, ...]:
        return self._entries("loras", ALLOWED_LORA_SUFFIXES)

    async def available_vaes(self) -> tuple[str, ...]:
        """外部VAEは未対応だが、置いてあるものは見えるようにしておく。"""
        return self._entries("vae", ALLOWED_CHECKPOINT_SUFFIXES)

    async def available_upscale_models(self) -> tuple[str, ...]:
        """hires fixが未対応のため、選べるものは無い。"""
        return ()

    async def available_embeddings(self) -> tuple[str, ...]:
        """prompt中の embedding: 参照が未対応のため、選べるものは無い。"""
        return ()

    async def available_controlnets(self) -> tuple[str, ...]:
        """ControlNetが未対応のため、選べるものは無い。"""
        return ()

    async def available_ipadapters(self) -> tuple[str, ...]:
        """IPAdapterが未対応のため、選べるものは無い。"""
        return ()

    async def available_clip_visions(self) -> tuple[str, ...]:
        """IPAdapterと対で使うものなので、同じく空。"""
        return ()

    async def available_diffusion_models(self) -> tuple[str, ...]:
        """DiT系 (unet単体) が未対応のため、選べるものは無い。"""
        return ()

    async def available_text_encoders(self) -> tuple[str, ...]:
        """DiT系と対で使うものなので、同じく空。"""
        return ()

    def _entries(self, category: str, suffixes: frozenset[str]) -> tuple[str, ...]:
        """`<models_root>/<category>` 直下の該当ファイル名を返す。

        ComfyUIの一覧に合わせ、拡張子付きのファイル名をそのまま返す。
        ディレクトリが無い場合は未配置として空にする。
        """
        directory = self._root() / category
        if not directory.is_dir():
            return ()
        return tuple(
            sorted(
                entry.name
                for entry in directory.iterdir()
                if entry.is_file() and entry.suffix in suffixes
            )
        )

    def _root(self) -> Path:
        if self._settings.models_root is None:
            raise InvalidConfiguration(
                "diffusersバックエンドを使うには IMAGEGEN_MODELS_ROOT を設定してください "
                "(配下に checkpoints / loras が並ぶディレクトリ)"
            )
        return self._settings.models_root.expanduser()


__all__ = ["DiffusersCatalog"]
