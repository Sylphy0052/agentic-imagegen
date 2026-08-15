"""モデル一覧などの列挙系操作の抽象と、環境・在庫の一括収集。

services/generation.py の GenerationBackend と同じ考え方で、列挙系の操作にも
Protocolを用意する。ComfyUIClient はこのProtocolを構造的に満たすが、
Service層はComfyUI固有の型を知らないままこのProtocol越しに扱う。

collect_catalog は「いま何が使えるか」を1回で集める。ComfyUIへ到達できない
場合はmodelsディレクトリの直読みへ落とす。生成のたびにComfyUIを起動・停止
する運用のため、探索のためだけに起動を増やさない。
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal, Protocol

from agentic_imagegen.config import Settings
from agentic_imagegen.domain.models import (
    ALLOWED_CHECKPOINT_SUFFIXES,
    ALLOWED_CLIP_VISION_SUFFIXES,
    ALLOWED_CONTROLNET_SUFFIXES,
    ALLOWED_FONT_SUFFIXES,
    ALLOWED_IPADAPTER_SUFFIXES,
    ALLOWED_LORA_SUFFIXES,
    ALLOWED_UPSCALE_MODEL_SUFFIXES,
)
from agentic_imagegen.domain.presets import PresetKind
from agentic_imagegen.domain.results import CatalogSnapshot
from agentic_imagegen.errors import ComfyUIUnavailable
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


@dataclass(frozen=True, slots=True)
class CatalogKind:
    """列挙する種別1つ分。APIとファイルシステムの両方の見方をここへ寄せる。

    片方だけ足すとAPI経由とフォールバックで見える種別が食い違うため、
    対応関係をこの1箇所で持つ。
    """

    #: 表示とCatalogSnapshot.modelsのキー。
    name: str
    #: CatalogBackend のメソッド名。
    method: str
    #: ComfyUIのmodels配下のサブディレクトリ名。
    directory: str
    #: フォールバック時に拾う拡張子。
    suffixes: frozenset[str]
    #: ファイル名から拡張子を落として返すか。embeddingだけprompt中の表記に合わせる。
    strip_suffix: bool = False


#: 列挙する種別の一覧。CatalogBackend の available_* と1対1で対応する。
CATALOG_KINDS: Final = (
    CatalogKind("checkpoints", "available_checkpoints", "checkpoints", ALLOWED_CHECKPOINT_SUFFIXES),
    CatalogKind("loras", "available_loras", "loras", ALLOWED_LORA_SUFFIXES),
    CatalogKind("controlnets", "available_controlnets", "controlnet", ALLOWED_CONTROLNET_SUFFIXES),
    CatalogKind("ipadapters", "available_ipadapters", "ipadapter", ALLOWED_IPADAPTER_SUFFIXES),
    CatalogKind(
        "clip_visions", "available_clip_visions", "clip_vision", ALLOWED_CLIP_VISION_SUFFIXES
    ),
    CatalogKind(
        "diffusion_models",
        "available_diffusion_models",
        "diffusion_models",
        ALLOWED_CHECKPOINT_SUFFIXES,
    ),
    CatalogKind(
        "text_encoders", "available_text_encoders", "text_encoders", ALLOWED_CHECKPOINT_SUFFIXES
    ),
    CatalogKind("vaes", "available_vaes", "vae", ALLOWED_CHECKPOINT_SUFFIXES),
    CatalogKind(
        "upscale_models",
        "available_upscale_models",
        "upscale_models",
        ALLOWED_UPSCALE_MODEL_SUFFIXES,
    ),
    CatalogKind(
        "embeddings",
        "available_embeddings",
        "embeddings",
        ALLOWED_CHECKPOINT_SUFFIXES | frozenset({".pt", ".bin"}),
        strip_suffix=True,
    ),
)


async def collect_catalog(
    settings: Settings,
    *,
    backend_factory: CatalogBackendFactory,
    comfyui_home: Path,
    presets_root: Path,
    fonts_root: Path,
) -> CatalogSnapshot:
    """使えるモデル・preset・フォントを1回で集める。

    ComfyUIへ到達できればAPIから、できなければ `comfyui_home/models/` の
    直読みから組み立てる。presetとフォントはリポジトリ内にあるため、
    ComfyUIの状態によらず常にファイルシステムから読む。
    """
    try:
        models = await _models_from_api(settings, backend_factory=backend_factory)
        source: Literal["api", "filesystem"] = "api"
    except ComfyUIUnavailable:
        models = _models_from_filesystem(comfyui_home)
        source = "filesystem"

    return CatalogSnapshot(
        source=source,
        models=models,
        presets=_collect_presets(presets_root),
        fonts=_list_names(fonts_root, ALLOWED_FONT_SUFFIXES),
    )


async def _models_from_api(
    settings: Settings, *, backend_factory: CatalogBackendFactory
) -> dict[str, tuple[str, ...]]:
    async with backend_factory(settings) as backend:
        collected: dict[str, tuple[str, ...]] = {}
        for kind in CATALOG_KINDS:
            method = getattr(backend, kind.method)
            collected[kind.name] = tuple(await method())
        return collected


def _models_from_filesystem(comfyui_home: Path) -> dict[str, tuple[str, ...]]:
    models_root = comfyui_home / "models"
    return {
        kind.name: _list_names(
            models_root / kind.directory, kind.suffixes, strip_suffix=kind.strip_suffix
        )
        for kind in CATALOG_KINDS
    }


def _collect_presets(presets_root: Path) -> dict[str, tuple[str, ...]]:
    return {
        kind.value: tuple(
            sorted(path.stem for path in (presets_root / kind.directory).glob("*.yaml"))
        )
        for kind in PresetKind
    }


def _list_names(
    directory: Path, suffixes: Iterable[str], *, strip_suffix: bool = False
) -> tuple[str, ...]:
    """ディレクトリ配下のファイル名を、ComfyUIが受け付ける表記で列挙する。

    ComfyUIはサブフォルダを1階層まで `sub/name` の形で扱うため、そこまで見る。
    ディレクトリが無い場合は空を返す (未導入の種別と区別しない。どちらも
    「指定できるものがない」という結論が同じため)。
    """
    if not directory.is_dir():
        return ()

    allowed = frozenset(suffixes)
    names: list[str] = []

    def add(prefix: str, path: Path) -> None:
        # 絞り込みは常に元のファイル名の拡張子で行う。
        # 表示名を先に削ると、拡張子で弾く判断ができなくなる。
        if path.suffix.lower() not in allowed:
            return
        names.append(f"{prefix}{path.stem if strip_suffix else path.name}")

    for path in directory.iterdir():
        if path.is_file():
            add("", path)
            continue
        if not path.is_dir():
            continue
        for nested in path.iterdir():
            if nested.is_file():
                add(f"{path.name}/", nested)

    return tuple(sorted(names))


__all__ = [
    "CATALOG_KINDS",
    "CatalogBackend",
    "CatalogBackendFactory",
    "CatalogKind",
    "CatalogSnapshot",
    "GenerationBackendFactory",
    "collect_catalog",
]
