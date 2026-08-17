"""DiffusersBackend のテスト。

実際のcheckpointを読むと数GB・数分かかるため、Pipelineは差し替える。
ここで見るのは「Specの各項目がPipelineへどう渡るか」と「対応していない指定を
黙って無視しないこと」の2点。実際に絵が出るかは integration test が見る。
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from PIL import Image

from agentic_imagegen.adapters.diffusers import backend as backend_module
from agentic_imagegen.adapters.diffusers.backend import DiffusersBackend, reject_unsupported
from agentic_imagegen.config import Settings
from agentic_imagegen.domain.models import GenerationSpec
from agentic_imagegen.errors import GenerationFailed, InvalidGenerationSpec

#: torchは `[diffusers]` extra にしか無い (数GBあるため既定では入れない)。
#: Pipelineの実行まで進むテストは実装側の実行時 `import torch` を踏むため、
#: extraを入れていない環境では外す。LoRAファイルの不在やSpecの検証で手前で
#: 弾かれるテストはtorchが無くても意味を持つため、そちらには付けない。
requires_torch = pytest.mark.skipif(
    importlib.util.find_spec("torch") is None,
    reason="torchが無い環境。`uv sync --extra diffusers` で入る",
)

BASE_SPEC: dict[str, Any] = {
    "version": "1",
    "task": "txt2img",
    "prompt": {"positive": "1girl, blue hair", "negative": "worst quality"},
    "generation": {"width": 512, "height": 768, "seed": 42, "steps": 20, "cfg": 7.0},
    "model": {"checkpoint": "hassakuSD15_v13.safetensors"},
}


def _spec(**overrides: Any) -> GenerationSpec:
    return GenerationSpec.model_validate({**BASE_SPEC, **overrides})


def write_safetensors(path: Path, keys: list[str]) -> None:
    """テンソル名だけを持つ最小のsafetensorsファイルを書く。

    実装はヘッダ (JSON) しか読まないため、重みは0バイト分で足りる。
    torchを入れていない環境でも書けるよう、フォーマットを直接組み立てる。
    """
    header = {key: {"dtype": "F32", "shape": [0], "data_offsets": [0, 0]} for key in keys}
    blob = json.dumps(header).encode("utf-8")
    path.write_bytes(len(blob).to_bytes(8, "little") + blob)


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    (tmp_path / "checkpoints").mkdir()
    (tmp_path / "checkpoints" / "hassakuSD15_v13.safetensors").write_bytes(b"x")
    (tmp_path / "loras").mkdir()
    # UNet側だけを持つLoRA。text encoder側を持つものは別に用意する
    write_safetensors(
        tmp_path / "loras" / "add_detail.safetensors",
        ["lora_unet_down_blocks_0_attentions_0_proj_in.lora_down.weight"],
    )
    return Settings(
        comfyui_base_url="http://127.0.0.1:8188",
        max_width=2048,
        max_height=2048,
        max_pixels=4194304,
        max_batch=4,
        timeout_seconds=30,
        output_root=Path("outputs"),
        backend="diffusers",
        models_root=tmp_path,
    )


class FakePipeline:
    """Pipelineの代わり。呼ばれた引数を記録するだけで推論はしない。"""

    def __init__(self, images: int = 1) -> None:
        self.calls: list[dict[str, Any]] = []
        self.loaded_loras: list[tuple[str, str]] = []
        self.adapters: tuple[list[str], list[float]] | None = None
        self.unload_count = 0
        self._images = images
        #: 設定すると生成時にこの例外を送出する (バックエンド側の失敗を模す)。
        self.error: Exception | None = None

    def __call__(self, **kwargs: Any) -> SimpleNamespace:
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        count = kwargs.get("num_images_per_prompt", 1)
        return SimpleNamespace(
            images=[Image.new("RGB", (8, 8), "blue") for _ in range(count * self._images)]
        )

    def unload_lora_weights(self) -> None:
        self.unload_count += 1

    def load_lora_weights(self, directory: str, *, weight_name: str, adapter_name: str) -> None:
        self.loaded_loras.append((weight_name, adapter_name))

    def set_adapters(self, names: list[str], *, adapter_weights: list[float]) -> None:
        self.adapters = (names, adapter_weights)


@pytest.fixture
def pipeline(monkeypatch: pytest.MonkeyPatch) -> FakePipeline:
    """Pipelineの読み込みとScheduler差し替えを差し替える。

    どちらもtorch / diffusersの実体を必要とするため、ここでは通さない。
    Schedulerの対応づけ自体は test_diffusers_scheduler.py が見ている。
    """
    fake = FakePipeline()
    monkeypatch.setattr(backend_module, "_load_pipeline", lambda path, *, task: fake)
    monkeypatch.setattr(backend_module, "_apply_scheduler", lambda pipeline, spec: None)
    monkeypatch.setattr(backend_module, "_select_device", lambda: "xpu")
    monkeypatch.setattr(backend_module, "_empty_device_cache", lambda: None)
    return fake


class TestRejectUnsupported:
    """対応していない指定は、生成に入る前に理由を添えて拒否する。"""

    def test_plain_txt2img_is_allowed(self) -> None:
        reject_unsupported(_spec())

    @pytest.mark.parametrize(
        ("overrides", "expected"),
        [
            (
                {
                    "control": {
                        "image": "inputs/pose.png",
                        "model": "control_v11p_sd15_canny_fp16.safetensors",
                    }
                },
                "ControlNet",
            ),
            (
                {
                    "reference": {
                        "image": "inputs/ref.png",
                        "model": "ip-adapter-plus_sd15.safetensors",
                        "clip_vision": "CLIP-ViT-H-14-laion2B-s32B-b79K.safetensors",
                    }
                },
                "IPAdapter",
            ),
            (
                {"generation": {"width": 512, "height": 512, "upscale": {"scale": 1.5}}},
                "hires fix",
            ),
            (
                {
                    "model": {
                        "unet": "anima_v1.safetensors",
                        "clip": "t5.safetensors",
                        "vae": "anima_vae.safetensors",
                    }
                },
                "DiT系モデル",
            ),
            (
                {
                    "model": {
                        "checkpoint": "hassakuSD15_v13.safetensors",
                        "vae": "vae-ft-mse-840000.safetensors",
                    }
                },
                "外部VAE",
            ),
            (
                {"prompt": {"positive": "1girl, embedding:EasyNegative"}},
                "embedding",
            ),
        ],
    )
    def test_unsupported_is_rejected(self, overrides: dict[str, Any], expected: str) -> None:
        with pytest.raises(InvalidGenerationSpec) as exc:
            reject_unsupported(_spec(**overrides))

        assert expected in str(exc.value)
        # どうすれば動くかまで示す
        assert "IMAGEGEN_BACKEND=comfyui" in str(exc.value)


@requires_torch
class TestExecute:
    @pytest.mark.asyncio
    async def test_returns_png_bytes(
        self, settings: Settings, pipeline: FakePipeline, tmp_path: Path
    ) -> None:
        async with DiffusersBackend(settings) as backend:
            output = await backend.execute(_spec(), project_root=tmp_path)

        assert len(output.images) == 1
        assert output.images[0].startswith(b"\x89PNG")
        assert output.suffixes == (".png",)
        assert output.seed == 42
        assert output.info["backend"]["name"] == "diffusers"

    @pytest.mark.asyncio
    async def test_passes_generation_parameters(
        self, settings: Settings, pipeline: FakePipeline, tmp_path: Path
    ) -> None:
        async with DiffusersBackend(settings) as backend:
            await backend.execute(_spec(), project_root=tmp_path)

        call = pipeline.calls[0]
        assert call["prompt"] == "1girl, blue hair"
        assert call["negative_prompt"] == "worst quality"
        assert call["num_inference_steps"] == 20
        assert call["guidance_scale"] == 7.0
        assert call["width"] == 512
        assert call["height"] == 768

    @pytest.mark.asyncio
    async def test_batch_size_becomes_images_per_prompt(
        self, settings: Settings, pipeline: FakePipeline, tmp_path: Path
    ) -> None:
        spec = _spec(generation={"width": 512, "height": 512, "seed": 1, "batch_size": 3})

        async with DiffusersBackend(settings) as backend:
            output = await backend.execute(spec, project_root=tmp_path)

        assert pipeline.calls[0]["num_images_per_prompt"] == 3
        assert len(output.images) == 3
        assert len(output.suffixes) == 3

    @pytest.mark.asyncio
    async def test_random_seed_is_resolved(
        self, settings: Settings, pipeline: FakePipeline, tmp_path: Path
    ) -> None:
        """seed -1 は実行時に決めた値をそのまま結果へ返す (再現できるように)。"""
        spec = _spec(generation={"width": 512, "height": 512, "seed": -1})

        async with DiffusersBackend(settings) as backend:
            output = await backend.execute(spec, project_root=tmp_path)

        assert output.seed != -1
        assert output.request_id == f"diffusers-{output.seed}"

    @pytest.mark.asyncio
    async def test_clip_skip_is_shifted_by_one(
        self, settings: Settings, pipeline: FakePipeline, tmp_path: Path
    ) -> None:
        """SpecのclipskipはA1111と同じ数え方。diffusersは「何層飛ばすか」で1つずれる。"""
        spec = _spec(model={"checkpoint": "hassakuSD15_v13.safetensors", "clip_skip": 2})

        async with DiffusersBackend(settings) as backend:
            await backend.execute(spec, project_root=tmp_path)

        assert pipeline.calls[0]["clip_skip"] == 1

    @pytest.mark.asyncio
    async def test_clip_skip_absent_is_not_passed(
        self, settings: Settings, pipeline: FakePipeline, tmp_path: Path
    ) -> None:
        async with DiffusersBackend(settings) as backend:
            await backend.execute(_spec(), project_root=tmp_path)

        assert "clip_skip" not in pipeline.calls[0]

    @pytest.mark.asyncio
    async def test_backend_failure_becomes_generation_failed(
        self, settings: Settings, pipeline: FakePipeline, tmp_path: Path
    ) -> None:
        """diffusers由来の例外をそのまま外へ出さず、exit codeの決まった型へ寄せる。"""
        pipeline.error = RuntimeError("out of memory")

        async with DiffusersBackend(settings) as backend:
            with pytest.raises(GenerationFailed) as exc:
                await backend.execute(_spec(), project_root=tmp_path)

        assert "out of memory" in str(exc.value)


class TestDerivePipeline:
    """同じcheckpointを別のtaskで使うときの載せ替え。

    実際のメモリ使用量はデバイスが要るため見られないが、倍化の原因になる
    dtypeの受け渡しだけはここで固定しておく。
    """

    def test_keeps_source_dtype(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """dtypeを渡さないと from_pipe が float32 へ載せ替え、共有していても倍になる。"""
        recorded: dict[str, Any] = {}

        class FakeDerived:
            @classmethod
            def from_pipe(cls, source: Any, **kwargs: Any) -> Any:
                recorded.update(kwargs)
                return SimpleNamespace(set_progress_bar_config=lambda **_: None)

        monkeypatch.setattr(backend_module, "_pipeline_class", lambda path, *, task: FakeDerived)
        source = SimpleNamespace(dtype="float16")

        backend_module._derive_pipeline(source, path=Path("checkpoint.safetensors"), task="img2img")

        assert recorded["torch_dtype"] == "float16"


@requires_torch
class TestDeviceMemory:
    """デバイス側のメモリを生成のたびに返すこと。

    XPUのアロケータは断片化しやすく、返さないまま同じプロセスで生成を続けると
    空き容量が十分あっても確保に失敗する。batchやMCP Serverのように1つの
    バックエンドを使い回す経路で効いてくる。
    """

    @pytest.fixture
    def releases(self, monkeypatch: pytest.MonkeyPatch) -> list[None]:
        calls: list[None] = []
        monkeypatch.setattr(backend_module, "_empty_device_cache", lambda: calls.append(None))
        return calls

    @pytest.mark.asyncio
    async def test_released_after_each_generation(
        self, settings: Settings, pipeline: FakePipeline, tmp_path: Path, releases: list[None]
    ) -> None:
        async with DiffusersBackend(settings) as backend:
            await backend.execute(_spec(), project_root=tmp_path)
            await backend.execute(_spec(), project_root=tmp_path)
            during = len(releases)

        assert during == 2
        # 閉じるときにも返す (Pipeline本体の解放)
        assert len(releases) == 3

    @pytest.mark.asyncio
    async def test_released_even_when_generation_fails(
        self, settings: Settings, pipeline: FakePipeline, tmp_path: Path, releases: list[None]
    ) -> None:
        """失敗したときこそ返す。落ちた分を抱えたまま次を試すと続けて落ちる。"""
        pipeline.error = RuntimeError("boom")

        async with DiffusersBackend(settings) as backend:
            with pytest.raises(GenerationFailed):
                await backend.execute(_spec(), project_root=tmp_path)
            assert len(releases) == 1


class TestLoras:
    @requires_torch
    @pytest.mark.asyncio
    async def test_loras_are_loaded_with_weights(
        self, settings: Settings, pipeline: FakePipeline, tmp_path: Path
    ) -> None:
        spec = _spec(
            model={
                "checkpoint": "hassakuSD15_v13.safetensors",
                "loras": [{"name": "add_detail.safetensors", "strength_model": 0.8}],
            }
        )

        async with DiffusersBackend(settings) as backend:
            await backend.execute(spec, project_root=tmp_path)

        assert pipeline.loaded_loras == [("add_detail.safetensors", "lora0")]
        assert pipeline.adapters == (["lora0"], [0.8])

    @requires_torch
    @pytest.mark.asyncio
    async def test_previous_loras_are_unloaded(
        self, settings: Settings, pipeline: FakePipeline, tmp_path: Path
    ) -> None:
        """Pipelineを使い回すため、前回のLoRAが残ったままにならないこと。"""
        async with DiffusersBackend(settings) as backend:
            await backend.execute(_spec(), project_root=tmp_path)
            await backend.execute(_spec(), project_root=tmp_path)

        assert pipeline.unload_count == 2
        assert pipeline.adapters is None

    @pytest.mark.asyncio
    async def test_missing_lora_file_is_rejected(
        self, settings: Settings, pipeline: FakePipeline, tmp_path: Path
    ) -> None:
        spec = _spec(
            model={
                "checkpoint": "hassakuSD15_v13.safetensors",
                "loras": [{"name": "absent.safetensors"}],
            }
        )

        async with DiffusersBackend(settings) as backend:
            with pytest.raises(InvalidGenerationSpec) as exc:
                await backend.execute(spec, project_root=tmp_path)

        assert "absent.safetensors" in str(exc.value)

    @requires_torch
    @pytest.mark.asyncio
    async def test_text_encoder_lora_is_rejected(
        self, settings: Settings, pipeline: FakePipeline, tmp_path: Path
    ) -> None:
        """diffusersが読めない側を含むLoRAは、UNet側だけ当てて続けず拒否する。"""
        write_safetensors(
            tmp_path / "loras" / "with_te.safetensors",
            [
                "lora_unet_down_blocks_0_attentions_0_proj_in.lora_down.weight",
                "lora_te_text_model_encoder_layers_0_mlp_fc1.lora_down.weight",
            ],
        )
        spec = _spec(
            model={
                "checkpoint": "hassakuSD15_v13.safetensors",
                "loras": [{"name": "with_te.safetensors"}],
            }
        )

        async with DiffusersBackend(settings) as backend:
            with pytest.raises(InvalidGenerationSpec) as exc:
                await backend.execute(spec, project_root=tmp_path)

        assert "with_te.safetensors" in str(exc.value)
        assert "IMAGEGEN_BACKEND=comfyui" in str(exc.value)
        # 読み込みに入る前に止める (Pipelineへは渡らない)
        assert pipeline.loaded_loras == []


@requires_torch
class TestImg2Img:
    @pytest.mark.asyncio
    async def test_source_image_and_strength_are_passed(
        self, settings: Settings, pipeline: FakePipeline, tmp_path: Path
    ) -> None:
        source = tmp_path / "inputs" / "ref.png"
        source.parent.mkdir()
        Image.new("RGB", (64, 64), "red").save(source)
        spec = _spec(
            task="img2img",
            generation={"seed": 7},
            source={"image": "inputs/ref.png", "denoise": 0.4},
        )

        async with DiffusersBackend(settings) as backend:
            await backend.execute(spec, project_root=tmp_path)

        call = pipeline.calls[0]
        assert call["strength"] == 0.4
        assert call["image"].size == (64, 64)
        # img2imgは入力画像のサイズを使うため解像度は渡さない
        assert "width" not in call
        assert "height" not in call

    @pytest.mark.asyncio
    async def test_missing_source_image_is_rejected(
        self, settings: Settings, pipeline: FakePipeline, tmp_path: Path
    ) -> None:
        spec = _spec(
            task="img2img",
            generation={"seed": 7},
            source={"image": "inputs/absent.png", "denoise": 0.4},
        )

        async with DiffusersBackend(settings) as backend:
            with pytest.raises(InvalidGenerationSpec) as exc:
                await backend.execute(spec, project_root=tmp_path)

        assert "inputs/absent.png" in str(exc.value)
