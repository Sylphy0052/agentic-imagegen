"""ComfyUI実機に対するIntegration Test。

実行:
    uv run pytest -m integration

通常の `uv run pytest` ではskipされる (tests/conftest.py)。
ComfyUIが起動していない場合は、失敗ではなくskipして理由を表示する。

GPU負荷を最小化するため、解像度は512x512・steps 2・batch_size 1に固定する。
hires fixを含むケースは1段目を256x256まで落とし、2段目も steps 2 で回す。

各ケースは必要なモデルがComfyUIに無ければskipする。ControlNetモデルや
DiT系のUNet / text encoder / VAEは環境によって置いていないため。
"""

from __future__ import annotations

import json
import os
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest
import pytest_asyncio

from agentic_imagegen.adapters.comfyui.client import ComfyUIClient
from agentic_imagegen.config import Settings
from agentic_imagegen.domain.models import GenerationSpec
from agentic_imagegen.errors import ComfyUIUnavailable
from agentic_imagegen.services.generation import generate

pytestmark = pytest.mark.integration

#: 生成の待ち時間。CPU推論では1枚あたり十数秒かかるため既定を長めに取る。
DEFAULT_TIMEOUT_SECONDS = 180


@pytest.fixture(scope="session")
def settings() -> Settings:
    timeout = int(os.environ.get("IMAGEGEN_TIMEOUT", DEFAULT_TIMEOUT_SECONDS))
    return Settings(
        comfyui_base_url=os.environ.get("COMFYUI_BASE_URL", "http://127.0.0.1:8188"),
        max_width=2048,
        max_height=2048,
        max_pixels=4194304,
        max_batch=4,
        timeout_seconds=timeout,
        output_root=Path("outputs"),
    )


@pytest_asyncio.fixture
async def client(settings: Settings) -> AsyncIterator[ComfyUIClient]:
    async with ComfyUIClient(settings) as connected:
        try:
            await connected.health()
        except ComfyUIUnavailable as exc:
            pytest.skip(f"ComfyUIへ到達できません ({settings.comfyui_base_url}): {exc}")
        yield connected


#: CPU推論では軽いモデルほど所要時間が短いため、SD1.5系を優先して選ぶ。
#: SDXL/Illustrious系は1枚あたり10分以上かかることがあり、Integration Testには向かない。
_PREFERRED_CHECKPOINT_HINTS = ("v1-5", "sd15", "meinamix", "dreamshaper")


@pytest_asyncio.fixture
async def checkpoint(client: ComfyUIClient) -> str:
    """ComfyUIが実際に持っているcheckpointを1つ選ぶ。

    `IMAGEGEN_TEST_CHECKPOINT` で明示指定できる。未指定ならSD1.5系を優先する。
    """
    names = await client.available_checkpoints()
    if not names:
        pytest.skip("ComfyUIに利用可能なcheckpointがありません")

    explicit = os.environ.get("IMAGEGEN_TEST_CHECKPOINT")
    if explicit:
        if explicit not in names:
            pytest.skip(f"指定されたcheckpointがComfyUIにありません: {explicit}")
        return explicit

    for hint in _PREFERRED_CHECKPOINT_HINTS:
        for name in names:
            if hint in name.lower():
                return name
    return names[0]


@pytest_asyncio.fixture
async def controlnet(client: ComfyUIClient) -> str:
    """ComfyUIが持っているControlNetモデルを1つ選ぶ。無ければskipする。"""
    names = await client.available_controlnets()
    if not names:
        pytest.skip("ComfyUIに利用可能なControlNetモデルがありません")
    return names[0]


@pytest_asyncio.fixture
async def separate_loaders(client: ComfyUIClient) -> dict[str, str]:
    """DiT系の UNet / text encoder / VAE を1組選ぶ。揃っていなければskipする。"""
    unets = await client.available_diffusion_models()
    clips = await client.available_text_encoders()
    vaes = await client.available_vaes()
    if not (unets and clips and vaes):
        pytest.skip("ComfyUIにDiT系のUNet / text encoder / VAEが揃っていません")
    return {"unet": unets[0], "clip": clips[0], "vae": vaes[0]}


def _write_source_image(project_root: Path, relative: str) -> str:
    """入力画像・control画像をproject_root配下へ作る。

    実写やイラストである必要はない。ここで見たいのは、アップロードから
    ノードへの受け渡しまでが通ることだけ。
    """
    from PIL import Image, ImageDraw

    path = project_root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (256, 256), (240, 240, 240))
    draw = ImageDraw.Draw(image)
    # Cannyが線を拾えるよう、はっきりしたエッジを置く
    draw.rectangle((64, 64, 192, 192), fill=(30, 30, 30))
    draw.line((0, 128, 256, 128), fill=(200, 40, 40), width=4)
    image.save(path)
    return relative


def _spec(checkpoint: str, prefix: str) -> GenerationSpec:
    payload: dict[str, Any] = {
        "prompt": {"positive": "a red apple on a table", "negative": "blurry"},
        "generation": {
            "width": 512,
            "height": 512,
            "steps": 2,
            "cfg": 7.0,
            "seed": 1,
            "batch_size": 1,
        },
        "model": {"checkpoint": checkpoint},
        "output": {"prefix": prefix},
    }
    return GenerationSpec.model_validate(payload)


async def test_health(client: ComfyUIClient) -> None:
    status = await client.health()

    assert status.base_url
    assert isinstance(status.devices, tuple)


async def test_available_checkpoints(client: ComfyUIClient, checkpoint: str) -> None:
    assert checkpoint
    assert checkpoint.endswith((".safetensors", ".ckpt"))


async def test_generate_end_to_end(
    client: ComfyUIClient, checkpoint: str, settings: Settings, tmp_path: Path
) -> None:
    """submit -> 実行完了 -> 画像取得 -> 保存 まで通しで確認する。"""
    result = await generate(
        _spec(checkpoint, "integration"),
        settings,
        backend=client,
        project_root=tmp_path,
    )

    assert result.prompt_id
    assert result.seed == 1
    assert result.files
    for path in result.files:
        assert path.is_file()
        assert path.stat().st_size > 0
        assert path.read_bytes().startswith(b"\x89PNG")

    metadata = json.loads(result.metadata_path.read_text(encoding="utf-8"))
    assert metadata["prompt_id"] == result.prompt_id
    assert metadata["workflow"] == "txt2img"
    assert metadata["outputs"] == [path.name for path in result.files]


async def test_submit_rejects_unknown_checkpoint(client: ComfyUIClient) -> None:
    """存在しないcheckpointはComfyUI側で拒否され、原因が分かる形で返る。"""
    from agentic_imagegen.errors import WorkflowSubmissionError
    from agentic_imagegen.workflows.injector import prepare_workflow

    prepared = prepare_workflow(_spec("definitely-missing-model.safetensors", "integration"))

    with pytest.raises(WorkflowSubmissionError):
        await client.submit(prepared.workflow)


async def _run(
    spec: GenerationSpec,
    settings: Settings,
    client: ComfyUIClient,
    project_root: Path,
    *,
    expected_workflow: str,
) -> dict[str, Any]:
    """生成を1件流し、出力とmetadataの基本的な整合まで確認して返す。"""
    result = await generate(spec, settings, backend=client, project_root=project_root)

    assert result.files
    for path in result.files:
        assert path.is_file()
        assert path.read_bytes().startswith(b"\x89PNG")

    metadata: dict[str, Any] = json.loads(result.metadata_path.read_text(encoding="utf-8"))
    assert metadata["workflow"] == expected_workflow
    assert metadata["outputs"] == [path.name for path in result.files]
    return metadata


def _light_generation(**extra: Any) -> dict[str, Any]:
    """Integration Testで使い回す低負荷な generation ブロック。"""
    params: dict[str, Any] = {
        "width": 256,
        "height": 256,
        "steps": 2,
        "cfg": 7.0,
        "seed": 1,
        "batch_size": 1,
    }
    params.update(extra)
    return params


async def test_generate_with_hires_fix(
    client: ComfyUIClient, checkpoint: str, settings: Settings, tmp_path: Path
) -> None:
    """1段目の結果をlatentのまま拡大し、2段目で描き足すところまで通す。"""
    spec = GenerationSpec.model_validate(
        {
            "prompt": {"positive": "a red apple on a table"},
            "generation": _light_generation(upscale={"scale": 1.5, "denoise": 0.4, "steps": 2}),
            "model": {"checkpoint": checkpoint},
            "output": {"prefix": "integration-hires"},
        }
    )

    await _run(spec, settings, client, tmp_path, expected_workflow="txt2img_hires")

    from PIL import Image

    directory = tmp_path / "outputs"
    produced = sorted(directory.rglob("*.png"))
    with Image.open(produced[0]) as image:
        assert image.size == (384, 384)


async def test_generate_with_controlnet(
    client: ComfyUIClient,
    checkpoint: str,
    controlnet: str,
    settings: Settings,
    tmp_path: Path,
) -> None:
    """control画像のアップロードから Canny -> ControlNetApplyAdvanced まで通す。"""
    image = _write_source_image(tmp_path, "inputs/control.png")
    spec = GenerationSpec.model_validate(
        {
            "prompt": {"positive": "a red apple on a table"},
            "generation": _light_generation(),
            "model": {"checkpoint": checkpoint},
            "control": {"image": image, "model": controlnet, "strength": 0.8},
            "output": {"prefix": "integration-controlnet"},
        }
    )

    await _run(spec, settings, client, tmp_path, expected_workflow="txt2img_controlnet")


async def test_generate_with_hires_fix_and_controlnet(
    client: ComfyUIClient,
    checkpoint: str,
    controlnet: str,
    settings: Settings,
    tmp_path: Path,
) -> None:
    """hires fix と ControlNet の併用 (#38)。ControlNetが効くのは1段目だけ。"""
    image = _write_source_image(tmp_path, "inputs/control.png")
    spec = GenerationSpec.model_validate(
        {
            "prompt": {"positive": "a red apple on a table"},
            "generation": _light_generation(upscale={"scale": 1.5, "denoise": 0.4, "steps": 2}),
            "model": {"checkpoint": checkpoint},
            "control": {"image": image, "model": controlnet, "strength": 0.8},
            "output": {"prefix": "integration-hires-controlnet"},
        }
    )

    await _run(spec, settings, client, tmp_path, expected_workflow="txt2img_hires_controlnet")


async def test_generate_img2img(
    client: ComfyUIClient, checkpoint: str, settings: Settings, tmp_path: Path
) -> None:
    """入力画像のアップロードから VAEEncode を経由した生成まで通す。"""
    image = _write_source_image(tmp_path, "inputs/source.png")
    spec = GenerationSpec.model_validate(
        {
            "task": "img2img",
            "prompt": {"positive": "a red apple on a table"},
            # img2imgは入力画像のサイズを使うため width / height は指定できない
            "generation": {"steps": 2, "cfg": 7.0, "seed": 1, "batch_size": 1},
            "model": {"checkpoint": checkpoint},
            "source": {"image": image, "denoise": 0.5},
            "output": {"prefix": "integration-img2img"},
        }
    )

    await _run(spec, settings, client, tmp_path, expected_workflow="img2img")


async def test_generate_with_separate_loaders(
    client: ComfyUIClient,
    separate_loaders: dict[str, str],
    settings: Settings,
    tmp_path: Path,
) -> None:
    """DiT系 (UNet / CLIP / VAE を別々に読む形式) を通す。"""
    spec = GenerationSpec.model_validate(
        {
            "prompt": {"positive": "a red apple on a table"},
            "generation": _light_generation(cfg=4.0, sampler="euler", scheduler="simple"),
            "model": dict(separate_loaders),
            "output": {"prefix": "integration-unet"},
        }
    )

    await _run(spec, settings, client, tmp_path, expected_workflow="txt2img_unet")


async def test_generate_with_separate_loaders_img2img_and_hires(
    client: ComfyUIClient,
    separate_loaders: dict[str, str],
    settings: Settings,
    tmp_path: Path,
) -> None:
    """DiT系 + img2img + hires fix (#39)。3ローダーが2段目とVAEEncodeにも効く。"""
    image = _write_source_image(tmp_path, "inputs/source.png")
    spec = GenerationSpec.model_validate(
        {
            "task": "img2img",
            "prompt": {"positive": "a red apple on a table"},
            "generation": {
                "steps": 2,
                "cfg": 4.0,
                "seed": 1,
                "batch_size": 1,
                "sampler": "euler",
                "scheduler": "simple",
                "upscale": {"scale": 1.5, "denoise": 0.4, "steps": 2},
            },
            "model": dict(separate_loaders),
            "source": {"image": image, "denoise": 0.5},
            "output": {"prefix": "integration-unet-img2img-hires"},
        }
    )

    await _run(spec, settings, client, tmp_path, expected_workflow="img2img_unet_hires")
