"""生成処理の入出力を表すデータ構造。

バックエンド固有の型を上位層へ持ち込まないよう、ここで中立な形を定義する。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


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


__all__ = ["GenerationResult", "HealthStatus", "ImageRef"]
