"""設定由来のポリシー制約。

モデル定義のハード制約 (domain.models) とは別に、
環境変数で調整できる上限値と出力先の安全性をここで検証する。
"""

from __future__ import annotations

from pathlib import Path

from agentic_imagegen.config import Settings
from agentic_imagegen.domain.models import GenerationSpec
from agentic_imagegen.errors import InvalidGenerationSpec


def validate_against_limits(spec: GenerationSpec, settings: Settings) -> None:
    """Specが設定上の上限を超えていないか検証する。

    超過時は InvalidGenerationSpec を送出する。
    """
    params = spec.generation

    if params.width > settings.max_width:
        raise InvalidGenerationSpec(
            f"width が上限を超えています ({params.width} > {settings.max_width})"
        )
    if params.height > settings.max_height:
        raise InvalidGenerationSpec(
            f"height が上限を超えています ({params.height} > {settings.max_height})"
        )
    if params.batch_size > settings.max_batch:
        raise InvalidGenerationSpec(
            f"batch_size が上限を超えています ({params.batch_size} > {settings.max_batch})"
        )

    total_pixels = params.width * params.height * params.batch_size
    if total_pixels > settings.max_pixels:
        raise InvalidGenerationSpec(
            "総pixel数が上限を超えています "
            f"({params.width}x{params.height}x{params.batch_size} = {total_pixels} "
            f"> {settings.max_pixels})"
        )


def resolve_output_directory(directory: str, root: Path) -> Path:
    """出力先を root 配下の絶対パスへ解決する。

    rootの外へ脱出する指定は InvalidGenerationSpec で拒否する。
    """
    if not directory or directory != directory.strip():
        raise InvalidGenerationSpec("output.directory に空文字は指定できません")
    if "\\" in directory:
        raise InvalidGenerationSpec("output.directory にバックスラッシュは使用できません")
    if directory.startswith("~"):
        raise InvalidGenerationSpec("output.directory にホームディレクトリ参照は指定できません")

    candidate = Path(directory)
    if candidate.is_absolute():
        raise InvalidGenerationSpec("output.directory は相対パスで指定してください")

    resolved_root = root.resolve()
    resolved = (resolved_root / candidate).resolve()
    if resolved != resolved_root and resolved_root not in resolved.parents:
        raise InvalidGenerationSpec(
            f"output.directory が作業ルートの外を指しています (指定値: {directory})"
        )
    return resolved


def resolve_source_image(image: str, root: Path, *, max_bytes: int) -> Path:
    """img2imgの入力画像を root 配下の絶対パスへ解決する。

    パス形式のハード制約は SourceSpec 側で済んでいる。ここでは実体に触れる検証
    (rootの外を指していないか、実在するか、大きすぎないか) を担う。
    """
    resolved_root = root.resolve()
    resolved = (resolved_root / Path(image)).resolve()

    if resolved_root not in resolved.parents:
        raise InvalidGenerationSpec(
            f"source.image が作業ルートの外を指しています (指定値: {image})"
        )
    if not resolved.is_file():
        raise InvalidGenerationSpec(f"入力画像が見つかりません: {image}")

    size = resolved.stat().st_size
    if size > max_bytes:
        raise InvalidGenerationSpec(
            f"入力画像が大きすぎます ({size} bytes > {max_bytes} bytes): {image}"
        )
    return resolved


__all__ = [
    "resolve_output_directory",
    "resolve_source_image",
    "validate_against_limits",
]
