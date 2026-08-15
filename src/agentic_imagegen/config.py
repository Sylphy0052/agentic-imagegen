"""環境変数由来の設定。

ハードコードを避け、上限値や接続先をここへ集約する。
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Final
from urllib.parse import urlparse

from agentic_imagegen.errors import InvalidConfiguration

DEFAULT_BASE_URL: Final = "http://127.0.0.1:8188"
DEFAULT_MAX_WIDTH: Final = 2048
DEFAULT_MAX_HEIGHT: Final = 2048
DEFAULT_MAX_PIXELS: Final = 4194304

#: hires fixの拡大後に許す総pixel数 (4096x4096相当)。
#: ベース解像度の上限 (DEFAULT_MAX_PIXELS) とは目的が違うため別に持つ。
#: 前者は1段目の生成負荷を抑えるためのもので、こちらは拡大時のピークメモリを
#: 抑えるためのもの。モデル拡大は一度モデルの固有倍率 (4xなど) まで広げるため、
#: 同じ値で縛るとベース512x768 + 4xモデルのような実用的な組み合わせが通らない。
DEFAULT_MAX_UPSCALED_PIXELS: Final = 16777216
DEFAULT_MAX_BATCH: Final = 4
DEFAULT_TIMEOUT_SECONDS: Final = 300
DEFAULT_OUTPUT_ROOT: Final = "outputs"
DEFAULT_PRESETS_ROOT: Final = "presets"
DEFAULT_FONTS_ROOT: Final = "fonts"
#: キャラクタ台帳の置き場。生成には関与せず、手掛かりを引くためだけに読む。
DEFAULT_REGISTRY_ROOT: Final = "registry"
#: ComfyUIの置き場。models配下を直接見るときだけ使う。
DEFAULT_COMFYUI_HOME: Final = "~/ComfyUI"
#: 上をホーム展開した既定値。dataclassの既定でメソッド呼び出しを行わないためここで解く。
DEFAULT_COMFYUI_HOME_PATH: Final = Path(DEFAULT_COMFYUI_HOME).expanduser()

#: img2imgの入力画像として受け付ける最大バイト数 (32MiB)。
#: 巨大画像はそのままの解像度で生成されるため、時間とメモリを直撃する。
DEFAULT_MAX_SOURCE_BYTES: Final = 32 * 1024 * 1024

# batch_sizeのハード上限 (Phase 1)。設定でこれを超える値は許可しない。
BATCH_HARD_LIMIT: Final = 4

_ALLOWED_SCHEMES: Final = frozenset({"http", "https"})


def _positive_int(key: str, default: int, *, maximum: int | None = None) -> int:
    raw = os.environ.get(key)
    if raw is None or raw == "":
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise InvalidConfiguration(f"{key} は整数で指定してください (指定値: {raw!r})") from exc
    if value < 1:
        raise InvalidConfiguration(f"{key} は1以上で指定してください (指定値: {value})")
    if maximum is not None and value > maximum:
        raise InvalidConfiguration(f"{key} は{maximum}以下で指定してください (指定値: {value})")
    return value


def _base_url(key: str, default: str) -> str:
    raw = os.environ.get(key)
    if raw is None:
        return default
    url = raw.strip().rstrip("/")
    parsed = urlparse(url)
    if parsed.scheme not in _ALLOWED_SCHEMES or not parsed.netloc:
        raise InvalidConfiguration(
            f"{key} は http:// または https:// から始まるURLで指定してください (指定値: {raw!r})"
        )
    return url


@dataclass(frozen=True, slots=True)
class Settings:
    """実行時設定。

    生成される値はすべて検証済みであることを前提としてよい。
    """

    comfyui_base_url: str
    max_width: int
    max_height: int
    max_pixels: int
    max_batch: int
    timeout_seconds: int
    output_root: Path
    #: presetの探索ルート。既存の呼び出しを壊さないよう既定値を持たせている。
    presets_root: Path = Path(DEFAULT_PRESETS_ROOT)
    max_source_bytes: int = DEFAULT_MAX_SOURCE_BYTES
    #: hires fixの拡大後に許す総pixel数。既定を持たせて既存の呼び出しを壊さない。
    max_upscaled_pixels: int = DEFAULT_MAX_UPSCALED_PIXELS
    #: テキスト合成に使うフォントの探索ルート。
    fonts_root: Path = Path(DEFAULT_FONTS_ROOT)
    #: ComfyUIの置き場。ComfyUIへ到達できないときに models配下を直接見るために使う。
    #: 生成そのものはHTTP越しに行うため、通常の経路では参照しない。
    comfyui_home: Path = DEFAULT_COMFYUI_HOME_PATH
    #: キャラクタ台帳の探索ルート。
    registry_root: Path = Path(DEFAULT_REGISTRY_ROOT)

    @classmethod
    def from_env(cls) -> Settings:
        """環境変数から設定を組み立てる。不正値は InvalidConfiguration を送出する。"""
        output_root = os.environ.get("IMAGEGEN_OUTPUT_ROOT", DEFAULT_OUTPUT_ROOT).strip()
        if not output_root:
            raise InvalidConfiguration("IMAGEGEN_OUTPUT_ROOT に空文字は指定できません")

        presets_root = os.environ.get("IMAGEGEN_PRESETS_ROOT", DEFAULT_PRESETS_ROOT).strip()
        if not presets_root:
            raise InvalidConfiguration("IMAGEGEN_PRESETS_ROOT に空文字は指定できません")

        fonts_root = os.environ.get("IMAGEGEN_FONTS_ROOT", DEFAULT_FONTS_ROOT).strip()
        if not fonts_root:
            raise InvalidConfiguration("IMAGEGEN_FONTS_ROOT に空文字は指定できません")

        registry_root = os.environ.get("IMAGEGEN_REGISTRY_ROOT", DEFAULT_REGISTRY_ROOT).strip()
        if not registry_root:
            raise InvalidConfiguration("IMAGEGEN_REGISTRY_ROOT に空文字は指定できません")

        comfyui_home = os.environ.get("COMFYUI_HOME", DEFAULT_COMFYUI_HOME).strip()
        if not comfyui_home:
            raise InvalidConfiguration("COMFYUI_HOME に空文字は指定できません")

        return cls(
            comfyui_base_url=_base_url("COMFYUI_BASE_URL", DEFAULT_BASE_URL),
            max_width=_positive_int("IMAGEGEN_MAX_WIDTH", DEFAULT_MAX_WIDTH),
            max_height=_positive_int("IMAGEGEN_MAX_HEIGHT", DEFAULT_MAX_HEIGHT),
            max_pixels=_positive_int("IMAGEGEN_MAX_PIXELS", DEFAULT_MAX_PIXELS),
            max_batch=_positive_int(
                "IMAGEGEN_MAX_BATCH", DEFAULT_MAX_BATCH, maximum=BATCH_HARD_LIMIT
            ),
            timeout_seconds=_positive_int("IMAGEGEN_TIMEOUT", DEFAULT_TIMEOUT_SECONDS),
            output_root=Path(output_root),
            presets_root=Path(presets_root),
            max_source_bytes=_positive_int("IMAGEGEN_MAX_SOURCE_BYTES", DEFAULT_MAX_SOURCE_BYTES),
            max_upscaled_pixels=_positive_int(
                "IMAGEGEN_MAX_UPSCALED_PIXELS", DEFAULT_MAX_UPSCALED_PIXELS
            ),
            fonts_root=Path(fonts_root),
            comfyui_home=Path(comfyui_home).expanduser(),
            registry_root=Path(registry_root),
        )


__all__ = ["Settings"]
