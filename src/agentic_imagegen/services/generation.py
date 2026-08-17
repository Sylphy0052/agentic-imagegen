"""画像生成のユースケース。

Spec -> バックエンド実行 -> 保存 の流れをここで組み立てる。
バックエンドは Protocol 越しに扱い、ComfyUIやWorkflowといった特定バックエンドの
事情には依存しない (それらは adapters 層の責務)。
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Any, Final, Protocol

from agentic_imagegen.config import Settings
from agentic_imagegen.domain.models import GenerationSpec
from agentic_imagegen.domain.policy import resolve_output_directory
from agentic_imagegen.domain.results import GenerationResult, HealthStatus
from agentic_imagegen.errors import TextCompositionError
from agentic_imagegen.services.compose import ResolvedFont, compose_text

logger: Final = logging.getLogger(__name__)

METADATA_FILENAME: Final = "metadata.json"

#: テキストを合成した画像に付ける接尾辞。生成そのままの画像と並べて置く。
TEXT_SUFFIX: Final = "_text"

#: 同じ日・同じprefixで再実行したときに、既存の結果を上書きしないための試行上限。
_MAX_DIRECTORY_SUFFIX: Final = 1000


@dataclass(frozen=True, slots=True)
class BackendOutput:
    """バックエンドが1回の生成分として返す結果。

    ComfyUIの非同期キュー・prompt_id・Workflow dictのような特定バックエンドの
    事情はここには現れない。画像はファイルへ保存する前のバイト列のまま持ち、
    保存とmetadata書き出しは generate() (バックエンド非依存) が担う。
    """

    #: 生成された画像のバイト列 (保存前)。
    images: tuple[bytes, ...]
    #: 実際に使われたseed。
    seed: int
    #: バックエンド内で一意な識別子。ComfyUIならprompt_id。
    request_id: str
    #: metadataへ書くバックエンド固有の情報 (workflow名・実行基盤情報など)。
    info: dict[str, Any]
    #: 各画像の拡張子 (".png" など)。images と同数。
    suffixes: tuple[str, ...]


class GenerationBackend(Protocol):
    """画像生成バックエンドに求める操作。

    ComfyUIClient はこのProtocolを構造的に満たす。ComfyUIの非同期キュー前提の
    段取り (Workflow組み立て・投入・監視・出力取得) はすべてバックエンド内へ
    閉じ込め、Service層へは execute() 越しの結果だけを返す。
    将来 diffusers や remote API を足す場合も、この形に合わせれば
    Service層を変更せずに差し替えられる。
    """

    async def execute(
        self, spec: GenerationSpec, *, project_root: Path, timeout: float | None = None
    ) -> BackendOutput: ...

    async def health(self) -> HealthStatus: ...


async def generate(
    spec: GenerationSpec,
    settings: Settings,
    *,
    backend: GenerationBackend,
    project_root: Path,
    timeout: float | None = None,
) -> GenerationResult:
    """Specに従って画像を生成し、結果をプロジェクト配下へ保存する。"""
    directory = _prepare_directory(spec, settings, project_root)

    resolved_timeout = timeout if timeout is not None else float(settings.timeout_seconds)
    # 見積り係数 (services/estimate.py) を起こし直すための実測値。
    # バックエンドへの投入から出力の取得までを測る。テキスト合成は含めない。
    started = time.monotonic()
    output = await backend.execute(spec, project_root=project_root, timeout=resolved_timeout)
    elapsed = time.monotonic() - started

    directory.mkdir(parents=True, exist_ok=True)
    files = _save_images(directory, output.images, output.suffixes)

    write_metadata = partial(
        _write_metadata,
        directory,
        spec=spec,
        request_id=output.request_id,
        seed=output.seed,
        files=files,
        info=output.info,
        duration_seconds=elapsed,
    )

    text_files, text_info, text_error = _compose_text_layers(spec, files, settings, project_root)
    # 画像は取得済みなので、途中まで合成できていればその分の記録は残してから失敗させる。
    # 1件も成功していない場合は従来どおり text_info は None のまま。
    metadata_path = write_metadata(text_info=text_info)
    if text_error is not None:
        raise text_error

    logger.info(
        "generation done: prompt_id=%s files=%d dir=%s", output.request_id, len(files), directory
    )

    return GenerationResult(
        prompt_id=output.request_id,
        seed=output.seed,
        directory=directory,
        files=files,
        metadata_path=metadata_path,
        text_files=text_files,
    )


def _compose_text_layers(
    spec: GenerationSpec,
    files: tuple[Path, ...],
    settings: Settings,
    project_root: Path,
) -> tuple[tuple[Path, ...], dict[str, Any] | None, TextCompositionError | None]:
    """生成画像へテキストを合成する。

    生成そのままの画像は消さずに残し、合成結果を別ファイルとして書き出す。
    文字だけ差し替えて作り直せる状態を保つため。

    batch_size > 1 で途中の1件が失敗しても、それより前に成功した分は
    呼び出し元へ返す。生成そのものは既に完了しているため、後段の合成の失敗で
    成功済みの記録まで失うべきではない。1件も成功していない場合の戻り値は
    従来どおり `(( ), None, error)`。
    """
    if spec.text is None:
        return (), None, None

    fonts_root = resolve_fonts_root(settings, project_root)
    outputs: list[Path] = []
    fonts: tuple[ResolvedFont, ...] = ()
    for path in files:
        try:
            result = compose_text(
                image=path,
                spec=spec.text,
                fonts_root=fonts_root,
                output=path.with_name(f"{path.stem}{TEXT_SUFFIX}{path.suffix}"),
                max_pixels=settings.max_pixels,
                project_root=project_root,
            )
        except TextCompositionError as exc:
            info = _text_compose_info(outputs, fonts, error=str(exc)) if outputs else None
            return tuple(outputs), info, exc
        outputs.append(result.output)
        fonts = result.fonts

    logger.info("text composed: files=%d dir=%s", len(outputs), files[0].parent if files else "-")
    return tuple(outputs), _text_compose_info(outputs, fonts), None


def _text_compose_info(
    outputs: list[Path], fonts: tuple[ResolvedFont, ...], *, error: str | None = None
) -> dict[str, Any]:
    info: dict[str, Any] = {
        "fonts": [
            {"name": font.name, "path": str(font.path), "index": font.index} for font in fonts
        ],
        "outputs": [path.name for path in outputs],
    }
    if error is not None:
        # 部分失敗であることをmetadataから分かるようにする
        info["error"] = error
    return info


def resolve_fonts_root(settings: Settings, project_root: Path) -> Path:
    """フォントルートを作業ルート基準の絶対パスへ解く。"""
    if settings.fonts_root.is_absolute():
        return settings.fonts_root
    return project_root / settings.fonts_root


def _prepare_directory(spec: GenerationSpec, settings: Settings, project_root: Path) -> Path:
    """`<出力ルート>/<日付>/<時刻>_<prefix>` を作業ルート内に解決する。

    ディレクトリ名の先頭へ実行時刻 (HHMMSS) を置き、同じprefixの結果が時系列に並ぶようにする。
    同じ秒に再実行した場合は連番を付け、既存の結果を上書きしない。
    """
    directory = spec.output.directory or str(settings.output_root)
    base = resolve_output_directory(directory, project_root)
    now = datetime.now().astimezone()
    day = now.strftime("%Y-%m-%d")
    name = f"{now.strftime('%H%M%S')}_{spec.output.prefix}"
    candidate = base / day / name

    if not candidate.exists():
        return candidate
    for suffix in range(2, _MAX_DIRECTORY_SUFFIX):
        numbered = candidate.with_name(f"{name}-{suffix}")
        if not numbered.exists():
            return numbered
    return candidate


def _save_images(
    directory: Path, images: tuple[bytes, ...], suffixes: tuple[str, ...]
) -> tuple[Path, ...]:
    """バックエンドが返した画像バイト列を `image_0001.png` 形式で書き出す。

    ファイル名の形はバックエンドによらず固定する (metadataや後段のテキスト合成が
    この命名を前提にしているため)。
    """
    files: list[Path] = []
    for index, (data, suffix) in enumerate(zip(images, suffixes, strict=True), 1):
        path = directory / f"image_{index:04d}{suffix}"
        path.write_bytes(data)
        files.append(path)
    return tuple(files)


def _write_metadata(
    directory: Path,
    *,
    spec: GenerationSpec,
    request_id: str,
    seed: int,
    files: tuple[Path, ...],
    info: dict[str, Any],
    duration_seconds: float,
    text_info: dict[str, Any] | None,
) -> Path:
    metadata = {
        "prompt_id": request_id,
        # 論理タスク名 (spec.task) ではなく、実際に使ったテンプレート名を残す
        "workflow": info.get("workflow"),
        "workflow_hash": info.get("workflow_hash"),
        "created_at": datetime.now().astimezone().isoformat(),
        # 投入から出力取得までの実測秒。モデルのロードを含む場合があるため、
        # 係数を起こすときは同じモデルで2回目以降の値を使う
        "duration_seconds": round(duration_seconds, 2),
        "resolved_seed": seed,
        "backend": info.get("backend"),
        "spec": spec.model_dump(mode="json"),
        "outputs": [path.name for path in files],
        # 解決したフォントの実パスを残し、見た目が変わったときに切り分けられるようにする
        "text": text_info,
    }
    path = directory / METADATA_FILENAME
    path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


__all__ = ["BackendOutput", "GenerationBackend", "generate"]
