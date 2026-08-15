"""diffusersバックエンドが読むモデルファイルの解決。

ComfyUIバックエンドはモデルの所在をComfyUI側が解決するが、diffusersは
プロセス内で読むため自分でパスを組み立てる必要がある。置き場は
`IMAGEGEN_MODELS_ROOT` (配下に checkpoints / loras が並ぶ、ComfyUIと同じ構成)。
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Final

from agentic_imagegen.config import Settings
from agentic_imagegen.errors import InvalidConfiguration, InvalidGenerationSpec

#: SDXLだけが持つ2つ目のtext encoder (OpenCLIP-G) を指すキーの接頭辞。
#: SD1.5は `cond_stage_model.` 配下に1つだけ持つ。
_SDXL_MARKER: Final = "conditioner.embedders.1."

#: kohya形式のLoRAでtext encoder側の重みを指すキーの接頭辞。
#: `lora_te_` / `lora_te1_` / `lora_te2_` (SDXLは2つ持つ) と、
#: 変換済みの形で配布されているものを見る。
_TEXT_ENCODER_MARKERS: Final = ("lora_te", "text_model.", "text_encoder.")


def resolve_model_path(settings: Settings, category: str, name: str) -> Path:
    """`<models_root>/<category>/<name>` を実在確認したうえで返す。

    category は "checkpoints" / "loras" など、ComfyUIのmodelsディレクトリと
    同じ区分を指す。
    """
    if settings.models_root is None:
        raise InvalidConfiguration(
            "diffusersバックエンドを使うには IMAGEGEN_MODELS_ROOT を設定してください "
            "(配下に checkpoints / loras が並ぶディレクトリ)"
        )

    root = settings.models_root.expanduser().resolve()
    candidate = (root / category / name).resolve()
    # Specでもファイル名を検証しているが、パスを組み立てるここでも root の外を拒否する
    if not candidate.is_relative_to(root):
        raise InvalidGenerationSpec(f"モデルの指定がモデルディレクトリの外を指しています: {name}")
    if not candidate.is_file():
        raise InvalidGenerationSpec(
            f"モデルファイルが見つかりません: {name} (探した場所: {root / category})"
        )
    return candidate


def has_sdxl_marker(keys: Iterable[str]) -> bool:
    """checkpointのテンソル名からSDXL系かどうかを判定する。

    SD1.5とSDXLではPipelineのクラスが別で、ファイル名からは区別できない。
    2つ目のtext encoderの有無で見分ける。
    """
    return any(key.startswith(_SDXL_MARKER) for key in keys)


def has_text_encoder_weights(keys: Iterable[str]) -> bool:
    """LoRAのテンソル名からtext encoder側の重みを含むかどうかを判定する。

    ComfyUIはUNet側とtext encoder側を別々の強度で当てられるが、diffusers側は
    kohya形式のtext encoder用キーを読めない (後述の `loads_text_encoder`)。
    """
    return any(key.startswith(_TEXT_ENCODER_MARKERS) for key in keys)


def loads_text_encoder(path: Path) -> bool:
    """LoRAファイルがtext encoder側の重みを持つかを、ヘッダだけ読んで判定する。

    diffusers 0.39はkohya形式 (`lora_te_*`) のtext encoder側を変換しきれず、
    読み込みの途中で `IndexError` になる (UNet側だけなら読める)。生成に入る前に
    ここで気づけるようにする。

    safetensors以外 (`.pt`) はヘッダだけを読む手段が無いため、判定せず False を返す。
    """
    if path.suffix != ".safetensors":
        return False

    from safetensors import safe_open

    with safe_open(str(path), framework="pt") as handle:
        return has_text_encoder_weights(handle.keys())


def is_sdxl_checkpoint(path: Path) -> bool:
    """checkpointファイルのヘッダだけを読んでSDXL系かどうかを判定する。

    safetensorsはヘッダにテンソル名を持つため、重みを読まずに判定できる。
    """
    from safetensors import safe_open

    with safe_open(str(path), framework="pt") as handle:
        return has_sdxl_marker(handle.keys())


__all__ = [
    "has_sdxl_marker",
    "has_text_encoder_weights",
    "is_sdxl_checkpoint",
    "loads_text_encoder",
    "resolve_model_path",
]
