"""ComfyUI実機に対するIntegration Test。

実行:
    uv run pytest -m integration

通常の `uv run pytest` ではskipされる (tests/conftest.py)。
ComfyUIが起動していない場合は、失敗ではなくskipして理由を表示する。

GPU負荷を最小化するため、解像度は512x512・steps 2・batch_size 1に固定する。
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

    workflow, _ = prepare_workflow(_spec("definitely-missing-model.safetensors", "integration"))

    with pytest.raises(WorkflowSubmissionError):
        await client.submit(workflow)
