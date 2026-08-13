"""設定由来のポリシー制約。

モデル定義のハード制約 (domain.models) とは別に、
環境変数で調整できる上限値と出力先の安全性をここで検証する。
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

from agentic_imagegen.config import Settings
from agentic_imagegen.domain.models import ALLOWED_FONT_SUFFIXES, GenerationSpec
from agentic_imagegen.errors import InvalidGenerationSpec, TextCompositionError

#: フォントが見つからないときに列挙する候補の件数。多すぎると読めない。
_MAX_FONT_CANDIDATES: Final = 10


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

    _validate_upscaled_pixels(spec, settings)


def _validate_upscaled_pixels(spec: GenerationSpec, settings: Settings) -> None:
    """hires fixの拡大でピークになる解像度を上限と突き合わせる。

    見るのは最終解像度ではなくピーク。モデル拡大は要求された倍率が小さくても
    一度モデルの固有倍率まで広げてから縮小するため、途中がいちばん大きくなる。
    latent拡大ではピークと最終が同じになる。
    """
    upscale = spec.generation.upscale
    if upscale is None:
        return

    params = spec.generation
    peak_scale = upscale.effective_model_scale if upscale.uses_model else upscale.scale
    peak_width = int(params.width * peak_scale)
    peak_height = int(params.height * peak_scale)
    peak_pixels = peak_width * peak_height * params.batch_size

    if peak_pixels > settings.max_upscaled_pixels:
        route = "アップスケールモデルでの拡大" if upscale.uses_model else "latent拡大"
        raise InvalidGenerationSpec(
            f"拡大後の総pixel数が上限を超えています ({route}: "
            f"{peak_width}x{peak_height}x{params.batch_size} = {peak_pixels} "
            f"> {settings.max_upscaled_pixels})"
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


def resolve_compose_output(path: str | Path, root: Path) -> Path:
    """composeの出力先を root 配下の絶対パスへ解決する。

    resolve_output_directory と異なり、絶対パス指定であっても root 配下なら
    そのまま通す (`imagegen compose -o` は相対/絶対どちらでも受け付けるため)。
    見るのは root 配下かどうかだけで、symlinkも `.resolve()` で解決されるため
    同時に塞がれる。
    """
    resolved_root = root.resolve()
    resolved = (resolved_root / Path(path)).resolve()

    if resolved_root not in resolved.parents:
        raise InvalidGenerationSpec(f"出力先が作業ルートの外を指しています (指定値: {path})")
    return resolved


def resolve_font(name: str, root: Path, *, project_root: Path | None = None) -> Path:
    """フォント名を root 配下の絶対パスへ解決する。

    パス形式のハード制約は TextLayer 側で済んでいる。ここでは実体に触れる検証
    (rootの外を指していないか、実在するか) を担う。

    見つからない場合は、別の書体へ暗黙にフォールバックせず失敗させる。意図しない
    書体で出力されるより、置き場所と候補を示して止める方が扱いやすい。

    project_root を渡すと、見つからないときのメッセージに出す探索ルートを
    作業ルートからの相対パスへ丸める (作業ルートの外を指す場合は絶対パスのまま)。
    省略時は従来どおり絶対パスを出す。
    """
    resolved_root = root.resolve()
    resolved = (resolved_root / Path(name)).resolve()

    if resolved_root not in resolved.parents:
        raise TextCompositionError(f"フォントの指定がフォントルートの外を指しています: {name}")

    if not resolved.is_file():
        raise TextCompositionError(
            f"フォントが見つかりません: {name}\n"
            f"  探索ルート: {display_path(resolved_root, project_root)}\n"
            f"  {_describe_available_fonts(resolved_root)}"
        )
    return resolved


def display_path(path: Path, project_root: Path | None) -> str:
    """パスを表示用の文字列へ丸める。

    services/mcp_tools.py の `_relative` と同じ方式 (作業ルート配下なら相対パス、
    解決できなければそのまま) で表示形式を揃える。ただしエラーメッセージ用のため、
    ルート外を指す場合は `_relative` のようにファイル名だけへ丸めず、絶対パスを
    そのまま示す (どこを指しているか分からなくなるのを避けるため)。
    """
    if project_root is None:
        return str(path)
    try:
        return path.resolve().relative_to(project_root.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def _describe_available_fonts(root: Path) -> str:
    """フォントルート配下にある候補を人が読める形へまとめる。"""
    if not root.is_dir():
        return "フォントルートが存在しません。ディレクトリを作成してフォントを置いてください"

    candidates = sorted(
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in ALLOWED_FONT_SUFFIXES
    )
    if not candidates:
        return "フォントルートにフォントがありません。フォントを置いてください"

    shown = candidates[:_MAX_FONT_CANDIDATES]
    listed = " / ".join(shown)
    remainder = len(candidates) - len(shown)
    if remainder > 0:
        listed = f"{listed} 他{remainder}件"
    return f"利用できるフォント: {listed}"


__all__ = [
    "display_path",
    "resolve_compose_output",
    "resolve_font",
    "resolve_output_directory",
    "resolve_source_image",
    "validate_against_limits",
]
