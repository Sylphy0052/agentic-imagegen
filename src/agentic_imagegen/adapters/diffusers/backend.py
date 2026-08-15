"""diffusersでプロセス内推論するバックエンド。

ComfyUIバックエンドと違い、HTTP越しのキューを持たずプロセス内で完結する。
`GenerationBackend` Protocol (services/generation.py) を構造的に満たし、
Service層からは execute() / health() の2つだけが見える。

torch / diffusers は optional-dependencies の `diffusers` extra であり、
未インストールの環境でもこのモジュールをimportできるよう、重いimportは
実際に生成へ入る関数の中で行う。
"""

from __future__ import annotations

import asyncio
import gc
import io
import logging
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import TYPE_CHECKING, Any, Final

from agentic_imagegen.adapters.diffusers.models import (
    is_sdxl_checkpoint,
    loads_text_encoder,
    resolve_model_path,
)
from agentic_imagegen.adapters.diffusers.schedulers import resolve_scheduler
from agentic_imagegen.config import Settings
from agentic_imagegen.domain.models import GenerationSpec, resolve_seed
from agentic_imagegen.domain.results import HealthStatus
from agentic_imagegen.errors import GenerationFailed, GenerationTimeout, InvalidGenerationSpec
from agentic_imagegen.services.generation import BackendOutput

if TYPE_CHECKING:
    from PIL.Image import Image

logger: Final = logging.getLogger(__name__)

#: 生成した画像を書き出す形式。ComfyUIバックエンドの既定と揃える。
_IMAGE_SUFFIX: Final = ".png"

#: プロセス内で完結するため接続先URLを持たない。metadataの見た目を揃えるための値。
_LOCATION: Final = "in-process"


@dataclass(frozen=True, slots=True)
class _PipelineKey:
    """読み込み済みPipelineを使い回すための鍵。"""

    checkpoint: str
    task: str


class DiffusersBackend:
    """diffusersでプロセス内推論するバックエンド。

    ComfyUIClient と同じく async context manager として使う
    (`GenerationBackendFactory` が要求する形)。
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._pipelines: dict[_PipelineKey, Any] = {}

    async def __aenter__(self) -> DiffusersBackend:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> bool:
        self._pipelines.clear()
        _empty_device_cache()
        return False

    async def execute(
        self, spec: GenerationSpec, *, project_root: Path, timeout: float | None = None
    ) -> BackendOutput:
        """Specに従ってプロセス内で画像を生成し、結果をバイト列として返す。"""
        reject_unsupported(spec)
        reject_unsupported_loras(spec, self._settings)
        seed = resolve_seed(spec.generation.seed)

        logger.info(
            "generation start: backend=diffusers checkpoint=%s prefix=%s seed=%s",
            spec.model.checkpoint,
            spec.output.prefix,
            seed,
        )

        # 推論はCPU/GPUを占有する同期処理のため、イベントループを塞がないよう別スレッドで動かす
        try:
            images = await asyncio.wait_for(
                asyncio.to_thread(self._run, spec, project_root, seed), timeout
            )
        except TimeoutError as exc:
            raise GenerationTimeout(
                f"生成が{timeout}秒以内に終わりませんでした (backend=diffusers)"
            ) from exc

        return BackendOutput(
            images=images,
            seed=seed,
            # プロセス内で完結するためキュー由来のIDが無い。seedで実行を指せるようにする
            request_id=f"diffusers-{seed}",
            info={
                "backend": {
                    "name": "diffusers",
                    "checkpoint": spec.model.checkpoint,
                    "device": _select_device(),
                }
            },
            suffixes=tuple(_IMAGE_SUFFIX for _ in images),
        )

    async def health(self) -> HealthStatus:
        """torchが使えるか、どのデバイスで動くかを返す。"""
        return await asyncio.to_thread(self._health)

    def _health(self) -> HealthStatus:
        try:
            import torch
        except ImportError as exc:
            raise GenerationFailed(
                "diffusersバックエンドを使うには `uv sync --extra diffusers` が必要です"
            ) from exc
        return HealthStatus(
            base_url=_LOCATION,
            comfyui_version=None,
            devices=(f"{_select_device()} (torch {torch.__version__})",),
        )

    def _run(self, spec: GenerationSpec, project_root: Path, seed: int) -> tuple[bytes, ...]:
        """同期の推論本体。別スレッドから呼ばれる。"""
        pipeline = self._prepare_pipeline(spec)
        try:
            images = _generate_images(pipeline, spec, project_root, seed)
        finally:
            # 生成のたびに返す。抱えたままだとXPUのアロケータが断片化し、
            # 空き容量が十分あっても次の確保に失敗する。batchやMCP Serverの
            # ように1つのバックエンドで続けて生成する経路で効いてくる。
            _empty_device_cache()
        return tuple(_to_png(image) for image in images)

    def _prepare_pipeline(self, spec: GenerationSpec) -> Any:
        """Pipelineを用意する。同じcheckpoint / taskなら読み込み済みのものを使う。"""
        checkpoint = spec.model.checkpoint
        if checkpoint is None:  # pragma: no cover - reject_unsupported が先に弾く
            raise InvalidGenerationSpec("diffusersバックエンドは checkpoint の指定が必要です")

        key = _PipelineKey(checkpoint=checkpoint, task=spec.task)
        pipeline = self._pipelines.get(key)
        if pipeline is None:
            path = resolve_model_path(self._settings, "checkpoints", checkpoint)
            loaded = self._loaded_for(checkpoint)
            if loaded is None:
                pipeline = _load_pipeline(path, task=spec.task)
            else:
                pipeline = _derive_pipeline(loaded, path=path, task=spec.task)
            self._pipelines[key] = pipeline

        _apply_scheduler(pipeline, spec)
        _apply_loras(pipeline, spec, self._settings)
        return pipeline

    def _loaded_for(self, checkpoint: str) -> Any | None:
        """同じcheckpointを別のtaskで読み込み済みなら、そのPipelineを返す。"""
        for key, pipeline in self._pipelines.items():
            if key.checkpoint == checkpoint:
                return pipeline
        return None


def reject_unsupported(spec: GenerationSpec) -> None:
    """diffusersバックエンドが対応していない指定を拒否する。

    黙って無視すると、書いたのに効いていない状態になり結果から原因を追えない。
    ComfyUIバックエンドなら通るSpecなので、どちらで動かすかを示して伝える。
    """
    unsupported: list[str] = []
    if spec.model.uses_separate_loaders:
        unsupported.append("unet / clip / vae (DiT系モデル)")
    if spec.model.vae is not None and not spec.model.uses_separate_loaders:
        unsupported.append("model.vae (外部VAE)")
    if spec.control is not None:
        unsupported.append("control (ControlNet)")
    if spec.reference is not None:
        unsupported.append("reference (IPAdapter)")
    if spec.generation.upscale is not None:
        unsupported.append("generation.upscale (hires fix)")
    if "embedding:" in f"{spec.prompt.positive} {spec.prompt.negative or ''}":
        unsupported.append("prompt中の embedding: 参照")

    if unsupported:
        raise InvalidGenerationSpec(
            "diffusersバックエンドは次の指定に対応していません: "
            f"{', '.join(unsupported)} "
            "(IMAGEGEN_BACKEND=comfyui で実行してください)"
        )


def reject_unsupported_loras(spec: GenerationSpec, settings: Settings) -> None:
    """text encoder側の重みを持つLoRAを、読み込みに入る前に拒否する。

    diffusers 0.39はkohya形式 (`lora_te_*`) のtext encoder側を変換しきれず、
    読み込みの途中で `IndexError` になる。UNet側だけを当てて続けることはできるが、
    それでは指定したLoRAとは違うものが当たった状態になるため、ここで止める。
    SD1.5 / SDXL向けに配布されているLoRAの多くは両方を持つ。
    """
    for lora in spec.model.loras:
        path = resolve_model_path(settings, "loras", lora.name)
        if loads_text_encoder(path):
            raise InvalidGenerationSpec(
                "diffusersバックエンドは text encoder側の重みを持つLoRAに対応していません: "
                f"{lora.name} (UNet側だけのLoRAなら使えます。"
                "そのまま使うなら IMAGEGEN_BACKEND=comfyui で実行してください)"
            )


def _select_device() -> str:
    """使えるデバイスを選ぶ。XPU (Intel GPU) を優先し、無ければCUDA / CPU。"""
    import torch

    if torch.xpu.is_available():
        return "xpu"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _empty_device_cache() -> None:
    """デバイス側のキャッシュを解放する。torch未導入なら何もしない。

    Pipelineは内部で循環参照を持つため、参照を外しただけでは即座に解放されず、
    デバイス側のメモリも返らない。同一プロセスで続けてPipelineを読み込むと
    その分だけ確保に失敗するため、明示的にGCを回してから解放する。
    """
    try:
        import torch
    except ImportError:
        return
    gc.collect()
    if torch.xpu.is_available():
        torch.xpu.empty_cache()
    elif torch.cuda.is_available():
        torch.cuda.empty_cache()


def _pipeline_class(path: Path, *, task: str) -> Any:
    """checkpointとtaskからPipelineのクラスを決める。

    SD1.5系とSDXL系ではクラスが違い、ファイル名からは区別できないため
    checkpointのヘッダを見て決める。
    """
    from diffusers import (
        StableDiffusionImg2ImgPipeline,
        StableDiffusionPipeline,
        StableDiffusionXLImg2ImgPipeline,
        StableDiffusionXLPipeline,
    )

    is_sdxl = is_sdxl_checkpoint(path)
    if task == "img2img":
        return StableDiffusionXLImg2ImgPipeline if is_sdxl else StableDiffusionImg2ImgPipeline
    return StableDiffusionXLPipeline if is_sdxl else StableDiffusionPipeline


def _derive_pipeline(source: Any, *, path: Path, task: str) -> Any:
    """読み込み済みのPipelineから、別taskのPipelineを重みを共有したまま作る。

    同じcheckpointをtaskごとに読み直すと重みを二重に持つことになり、
    メモリの少ない環境では2回目の読み込みで確保に失敗する。
    from_pipe は構成要素 (UNet / VAE / text encoder) を共有したまま
    別のPipelineへ載せ替えるため、追加の確保が要らない。

    ただし dtype を渡さないと from_pipe は float32 へ載せ替えてしまい、
    共有しているにもかかわらず使用量が倍になる (実測でSD1.5が2.0 -> 4.0GiB)。
    元のPipelineのdtypeを明示して引き継ぐ。
    """
    derived = _pipeline_class(path, task=task).from_pipe(source, torch_dtype=source.dtype)
    derived.set_progress_bar_config(disable=True)
    return derived


def _load_pipeline(path: Path, *, task: str) -> Any:
    """checkpointを1ファイルから読み込んでPipelineを作る。"""
    import torch

    cls: Any = _pipeline_class(path, task=task)
    is_sdxl = "XL" in cls.__name__
    device = _select_device()
    # CPUはfloat16の対応が乏しく、そのまま流すと極端に遅いか落ちる
    dtype = torch.float32 if device == "cpu" else torch.float16
    logger.info(
        "loading checkpoint: %s (sdxl=%s device=%s dtype=%s)", path.name, is_sdxl, device, dtype
    )
    pipeline = cls.from_single_file(str(path), torch_dtype=dtype, safety_checker=None)
    pipeline.set_progress_bar_config(disable=True)
    return pipeline.to(device)


def _apply_scheduler(pipeline: Any, spec: GenerationSpec) -> None:
    """Specのsampler / schedulerに合わせてSchedulerを差し替える。"""
    import diffusers

    choice = resolve_scheduler(spec.generation.sampler, spec.generation.scheduler)
    scheduler_cls = getattr(diffusers, choice.class_name)
    pipeline.scheduler = scheduler_cls.from_config(pipeline.scheduler.config, **choice.options)


def _apply_loras(pipeline: Any, spec: GenerationSpec, settings: Settings) -> None:
    """LoRAを読み込んで適用する。

    Pipelineを使い回すため、前回のLoRAは必ず外してから読み直す。
    strength_model と strength_clip を別々に指定できるComfyUIと違い、
    diffusersのadapter weightは1つなので strength_model を採用する。
    """
    pipeline.unload_lora_weights()
    if not spec.model.loras:
        return

    names: list[str] = []
    weights: list[float] = []
    for index, lora in enumerate(spec.model.loras):
        path = resolve_model_path(settings, "loras", lora.name)
        adapter = f"lora{index}"
        pipeline.load_lora_weights(str(path.parent), weight_name=path.name, adapter_name=adapter)
        names.append(adapter)
        weights.append(lora.strength_model)
        if lora.strength_clip != lora.strength_model:
            logger.warning(
                "diffusersはLoRAの重みを1つしか持てないため strength_clip=%s は無視されます "
                "(strength_model=%s を使用): %s",
                lora.strength_clip,
                lora.strength_model,
                lora.name,
            )
    pipeline.set_adapters(names, adapter_weights=weights)


def _generate_images(
    pipeline: Any, spec: GenerationSpec, project_root: Path, seed: int
) -> list[Image]:
    """Pipelineを実行して画像を得る。"""
    import torch

    generation = spec.generation
    generator = torch.Generator(device="cpu").manual_seed(seed)
    arguments: dict[str, Any] = {
        "prompt": spec.prompt.positive,
        "negative_prompt": spec.prompt.negative or None,
        "num_inference_steps": generation.steps,
        "guidance_scale": generation.cfg,
        "num_images_per_prompt": generation.batch_size,
        "generator": generator,
        "output_type": "pil",
    }
    if spec.model.clip_skip is not None:
        # Specの clip_skip はA1111と同じ数え方 (1が既定)。
        # diffusersの clip_skip は「何層飛ばすか」なので1つずれる。
        arguments["clip_skip"] = spec.model.clip_skip - 1

    if spec.task == "img2img":
        if spec.source is None:  # pragma: no cover - Spec側で保証済み
            raise InvalidGenerationSpec("img2img には source が必要です")
        arguments["image"] = _load_source_image(spec.source.image, project_root)
        arguments["strength"] = spec.source.denoise
    else:
        arguments["width"] = generation.width
        arguments["height"] = generation.height

    try:
        result = pipeline(**arguments)
    except Exception as exc:
        raise GenerationFailed(f"diffusersでの生成に失敗しました: {exc}") from exc

    images: list[Image] = list(result.images)
    if not images:
        raise GenerationFailed("diffusersが画像を返しませんでした")
    return images


def _load_source_image(reference: str, project_root: Path) -> Image:
    """img2imgの入力画像を読む。"""
    from PIL import Image as PILImage

    path = (project_root / reference).resolve()
    if not path.is_file():
        raise InvalidGenerationSpec(f"入力画像が見つかりません: {reference}")
    with PILImage.open(path) as handle:
        return handle.convert("RGB")


def _to_png(image: Image) -> bytes:
    """PIL画像をPNGのバイト列にする。保存は services 層が行う。"""
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


__all__ = ["DiffusersBackend", "reject_unsupported", "reject_unsupported_loras"]
