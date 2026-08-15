"""生成処理の入出力を表すデータ構造。

バックエンド固有の型を上位層へ持ち込まないよう、ここで中立な形を定義する。
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal


@dataclass(frozen=True, slots=True)
class ImageRef:
    """バックエンド上に生成された画像1件への参照。"""

    filename: str
    subfolder: str
    type: str


@dataclass(frozen=True, slots=True)
class HealthStatus:
    """バックエンドへの到達確認結果。

    ComfyUI固有の情報ではなく「どの実行基盤で動いているか」を表す中立な値として扱う。
    生成時のmetadataへ記録し、あとから実行環境を追えるようにする。
    """

    base_url: str
    comfyui_version: str | None
    devices: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class GenerationResult:
    """1回の生成の結果。"""

    prompt_id: str
    seed: int
    directory: Path
    files: tuple[Path, ...]
    metadata_path: Path
    #: テキストを合成した画像。text を指定しなかった場合は空。
    #: files の生成そのままの画像は残したままにする。
    text_files: tuple[Path, ...] = ()


@dataclass(frozen=True, slots=True)
class CatalogSnapshot:
    """使えるモデル・preset・フォントの一覧。

    `source` は取得元。`api` はバックエンドが実際に読み込める名前で、
    `filesystem` はComfyUIのmodelsディレクトリを直接見た結果。後者は
    ComfyUIを起動せずに探せる代わりに、カスタムノード由来の種別 (IPAdapterなど)
    が未導入かどうかまでは分からない。どちらで見た値かで信頼度が変わるため、
    表示にも残す。
    """

    source: Literal["api", "filesystem"]
    #: 種別名 (CATALOG_KINDS の name) から、その種別で使える名前の一覧へ。
    models: Mapping[str, tuple[str, ...]]
    #: presetの軸 (character / scene / style) から、preset名の一覧へ。
    presets: Mapping[str, tuple[str, ...]]
    fonts: tuple[str, ...]


__all__ = ["CatalogSnapshot", "GenerationResult", "HealthStatus", "ImageRef"]
