"""画像生成のユースケース。

Spec -> Workflow -> バックエンド実行 -> 保存 の流れをここで組み立てる。
バックエンドは Protocol 越しに扱い、ComfyUI固有の型には依存しない。
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Any, Final, Protocol

from agentic_imagegen.config import Settings
from agentic_imagegen.domain.embeddings import (
    extract_embedding_names,
    extract_unresolvable_embedding_refs,
    strip_embedding_extension,
)
from agentic_imagegen.domain.models import GenerationSpec
from agentic_imagegen.domain.policy import resolve_output_directory, resolve_source_image
from agentic_imagegen.domain.results import GenerationResult, HealthStatus, ImageRef
from agentic_imagegen.errors import InvalidGenerationSpec, TextCompositionError
from agentic_imagegen.services.compose import ResolvedFont, compose_text
from agentic_imagegen.workflows.injector import prepare_workflow

logger: Final = logging.getLogger(__name__)

METADATA_FILENAME: Final = "metadata.json"

#: テキストを合成した画像に付ける接尾辞。生成そのままの画像と並べて置く。
TEXT_SUFFIX: Final = "_text"

#: 同じ日・同じprefixで再実行したときに、既存の結果を上書きしないための試行上限。
_MAX_DIRECTORY_SUFFIX: Final = 1000


class GenerationBackend(Protocol):
    """画像生成バックエンドに求める操作。

    ComfyUIClient はこのProtocolを構造的に満たす。
    将来 diffusers や remote API を足す場合も、この形に合わせれば
    Service層を変更せずに差し替えられる。
    """

    async def submit(self, workflow: dict[str, Any]) -> str: ...

    async def wait_for_completion(
        self, prompt_id: str, *, timeout: float | None = None
    ) -> None: ...

    async def fetch_outputs(self, prompt_id: str) -> tuple[ImageRef, ...]: ...

    async def download(self, ref: ImageRef) -> bytes: ...

    async def health(self) -> HealthStatus: ...

    async def upload_image(self, path: Path) -> str: ...

    async def available_embeddings(self) -> tuple[str, ...]: ...


async def generate(
    spec: GenerationSpec,
    settings: Settings,
    *,
    backend: GenerationBackend,
    project_root: Path,
    timeout: float | None = None,
    workflows_dir: Path | None = None,
) -> GenerationResult:
    """Specに従って画像を生成し、結果をプロジェクト配下へ保存する。"""
    await _validate_embeddings(spec, backend)
    directory = _prepare_directory(spec, settings, project_root)

    source_image_name = await _upload_image(
        spec.source.image if spec.source is not None else None,
        settings,
        backend,
        project_root,
        label="source",
    )
    control_image_name = await _upload_image(
        spec.control.image if spec.control is not None else None,
        settings,
        backend,
        project_root,
        label="control",
    )
    reference_image_name = await _upload_image(
        spec.reference.image if spec.reference is not None else None,
        settings,
        backend,
        project_root,
        label="reference",
    )
    prepared = prepare_workflow(
        spec,
        workflows_dir=workflows_dir,
        project_root=project_root,
        source_image_name=source_image_name,
        control_image_name=control_image_name,
        reference_image_name=reference_image_name,
    )
    seed = prepared.seed
    logger.info(
        "generation start: workflow=%s prefix=%s seed=%s",
        prepared.workflow_name,
        spec.output.prefix,
        seed,
    )

    prompt_id = await backend.submit(prepared.workflow)
    await backend.wait_for_completion(
        prompt_id, timeout=timeout if timeout is not None else float(settings.timeout_seconds)
    )
    refs = await backend.fetch_outputs(prompt_id)

    directory.mkdir(parents=True, exist_ok=True)
    files = tuple(
        [await _save_image(backend, ref, directory, index) for index, ref in enumerate(refs, 1)]
    )

    backend_info = await _collect_backend_info(backend)
    write_metadata = partial(
        _write_metadata,
        directory,
        spec=spec,
        prompt_id=prompt_id,
        seed=seed,
        files=files,
        workflow_name=prepared.workflow_name,
        workflow_hash=prepared.template_hash,
        backend_info=backend_info,
    )

    text_files, text_info, text_error = _compose_text_layers(spec, files, settings, project_root)
    # 画像は取得済みなので、途中まで合成できていればその分の記録は残してから失敗させる。
    # 1件も成功していない場合は従来どおり text_info は None のまま。
    metadata_path = write_metadata(text_info=text_info)
    if text_error is not None:
        raise text_error

    logger.info("generation done: prompt_id=%s files=%d dir=%s", prompt_id, len(files), directory)

    return GenerationResult(
        prompt_id=prompt_id,
        seed=seed,
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


async def _save_image(
    backend: GenerationBackend, ref: ImageRef, directory: Path, index: int
) -> Path:
    data = await backend.download(ref)
    suffix = Path(ref.filename).suffix or ".png"
    path = directory / f"image_{index:04d}{suffix}"
    path.write_bytes(data)
    return path


async def _upload_image(
    relative_path: str | None,
    settings: Settings,
    backend: GenerationBackend,
    project_root: Path,
    *,
    label: str,
) -> str | None:
    """入力画像を検証し、ComfyUIへアップロードして参照名を返す。

    LoadImageが参照できるのはComfyUIのinput配下だけなので、リポジトリ内の画像は
    そのままでは使えない。img2imgの入力画像・ControlNetのcontrol画像・
    IPAdapterの参照画像で共通の手順。
    """
    if relative_path is None:
        return None

    path = resolve_source_image(relative_path, project_root, max_bytes=settings.max_source_bytes)
    try:
        name = await backend.upload_image(path)
    except InvalidGenerationSpec as exc:
        # adapterはファイル名しか知らない。どのフィールドの指定だったかはここで補う
        raise InvalidGenerationSpec(f"{label}.image を読み込めません: {relative_path}") from exc
    logger.info("%s image uploaded: %s -> %s", label, relative_path, name)
    return name


async def _validate_embeddings(spec: GenerationSpec, backend: GenerationBackend) -> None:
    """promptで参照しているembeddingがComfyUIに実在するか検証する。

    ComfyUI自身は未配置のembeddingを見つけても例外を出さず、警告ログを残して
    黙って無視するだけ (生成そのものは成功するがembeddingは効かない)。
    それではユーザーが気づけないため、投入前にここで検出する。

    prompt中に `embedding:` 記法が無ければComfyUIへ問い合わせない
    (無駄な往復を避ける)。
    """
    unresolvable = extract_unresolvable_embedding_refs(spec.prompt.positive, spec.prompt.negative)
    if unresolvable:
        raise InvalidGenerationSpec(
            "ComfyUIが解決しない書き方のembedding参照があります: "
            f"{', '.join(unresolvable)} "
            "(embedding: の直前に空白が要ります。"
            "`1girl,embedding:name` ではなく `1girl, embedding:name` と書きます)"
        )

    referenced = extract_embedding_names(spec.prompt.positive, spec.prompt.negative)
    if not referenced:
        return

    available = set(await backend.available_embeddings())
    missing = [name for name in referenced if strip_embedding_extension(name) not in available]
    if missing:
        raise InvalidGenerationSpec(
            "未配置のembeddingが指定されています: "
            f"{', '.join(missing)} "
            f"(配置済み: {', '.join(sorted(available)) if available else 'なし'})"
        )


async def _collect_backend_info(backend: GenerationBackend) -> dict[str, Any] | None:
    """metadataへ残す実行基盤の情報を集める。

    ここでの失敗は生成そのものを巻き戻す理由にならない (画像は既に取得済み)。
    記録を諦めるだけにして、理由はログへ残す。
    """
    try:
        status = await backend.health()
    except Exception:
        logger.warning(
            "実行基盤の情報を取得できませんでした。metadataへは記録しません", exc_info=True
        )
        return None
    return {"comfyui_version": status.comfyui_version, "devices": list(status.devices)}


def _write_metadata(
    directory: Path,
    *,
    spec: GenerationSpec,
    prompt_id: str,
    seed: int,
    files: tuple[Path, ...],
    workflow_name: str,
    workflow_hash: str,
    backend_info: dict[str, Any] | None,
    text_info: dict[str, Any] | None,
) -> Path:
    metadata = {
        "prompt_id": prompt_id,
        # 論理タスク名 (spec.task) ではなく、実際に使ったテンプレート名を残す
        "workflow": workflow_name,
        "workflow_hash": workflow_hash,
        "created_at": datetime.now().astimezone().isoformat(),
        "resolved_seed": seed,
        "backend": backend_info,
        "spec": spec.model_dump(mode="json"),
        "outputs": [path.name for path in files],
        # 解決したフォントの実パスを残し、見た目が変わったときに切り分けられるようにする
        "text": text_info,
    }
    path = directory / METADATA_FILENAME
    path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


__all__ = ["GenerationBackend", "generate"]
