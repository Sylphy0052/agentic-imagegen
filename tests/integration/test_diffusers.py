"""diffusersバックエンドの実機に対するIntegration Test。

実行:
    IMAGEGEN_MODELS_ROOT=~/ComfyUI/models uv run pytest -m integration tests/integration/test_diffusers.py

通常の `uv run pytest` ではskipされる (tests/conftest.py)。
torch / diffusers が未インストール (`uv sync --extra diffusers` 未実行) の場合や、
IMAGEGEN_MODELS_ROOT が未設定・必要なcheckpointが無い場合は失敗ではなくskipする。

負荷を最小化するため、解像度は512x512・steps 2・batch_size 1に固定する。
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest

from agentic_imagegen.config import Settings
from agentic_imagegen.domain.models import GenerationSpec
from agentic_imagegen.errors import InvalidGenerationSpec
from agentic_imagegen.services.generation import generate

# torch / transformers / diffusers は自分たちの内部で警告を出す
# (WSL2で初期化できないLevel Zero Sysman、tokenizersの非推奨API など)。
# 実機を動かす以上こちらでは消せないため、このモジュールに限って
# filterwarnings = error を緩める。Unit Test側は厳格なまま。
pytestmark = [
    pytest.mark.integration,
    pytest.mark.filterwarnings("ignore::DeprecationWarning"),
    pytest.mark.filterwarnings("ignore::UserWarning"),
    pytest.mark.filterwarnings("ignore::FutureWarning"),
]

#: 生成の待ち時間。checkpointの読み込みに1分近くかかることがあるため長めに取る。
DEFAULT_TIMEOUT_SECONDS = 900

#: 実機確認に使うSD1.5系checkpoint。無ければskipする。
CHECKPOINT = "hassakuSD15_v13.safetensors"


@pytest.fixture(scope="session")
def models_root() -> Path:
    raw = os.environ.get("IMAGEGEN_MODELS_ROOT", "").strip()
    if not raw:
        pytest.skip("IMAGEGEN_MODELS_ROOT が未設定のためskipします")
    root = Path(raw).expanduser()
    if not (root / "checkpoints" / CHECKPOINT).is_file():
        pytest.skip(f"{CHECKPOINT} が {root / 'checkpoints'} に無いためskipします")
    return root


@pytest.fixture(scope="session")
def settings(models_root: Path) -> Settings:
    pytest.importorskip("torch", reason="uv sync --extra diffusers が必要です")
    pytest.importorskip("diffusers", reason="uv sync --extra diffusers が必要です")
    timeout = int(os.environ.get("IMAGEGEN_TIMEOUT", DEFAULT_TIMEOUT_SECONDS))
    return Settings(
        comfyui_base_url="http://127.0.0.1:8188",
        max_width=2048,
        max_height=2048,
        max_pixels=4194304,
        max_batch=4,
        timeout_seconds=timeout,
        output_root=Path("outputs"),
        backend="diffusers",
        models_root=models_root,
    )


@pytest.fixture(scope="session")
def backend(settings: Settings) -> Any:
    """セッションを通して1つだけ使うバックエンド。

    checkpointの読み込みはメモリを大きく使い、内蔵GPU (共有メモリ) の環境では
    プロセス内で2回目の読み込みが確保に失敗する。テストごとに開き直さず、
    読み込み済みのPipelineを共有する。
    """
    from agentic_imagegen.adapters.diffusers.backend import DiffusersBackend

    return DiffusersBackend(settings)


def _spec(**overrides: Any) -> GenerationSpec:
    base: dict[str, Any] = {
        "version": "1",
        "task": "txt2img",
        "prompt": {"positive": "1girl, solo, blue hair", "negative": "worst quality"},
        "generation": {"width": 512, "height": 512, "steps": 2, "cfg": 7.0, "seed": 4242},
        "model": {"checkpoint": CHECKPOINT},
        "output": {"prefix": "it_diffusers"},
    }
    return GenerationSpec.model_validate({**base, **overrides})


@pytest.mark.asyncio
async def test_health_reports_device(settings: Settings) -> None:
    from agentic_imagegen.adapters.diffusers.backend import DiffusersBackend

    async with DiffusersBackend(settings) as backend:
        status = await backend.health()

    assert status.devices
    # ComfyUIのバージョンに相当するものは持たない
    assert status.comfyui_version is None


@pytest.mark.asyncio
async def test_txt2img_generates_image(settings: Settings, backend: Any, tmp_path: Path) -> None:
    """実際に絵が出て、生成フロー (services.generate) が結果を保存できること。"""
    result = await generate(_spec(), settings, backend=backend, project_root=tmp_path)

    assert result.files
    for path in result.files:
        assert path.is_file()
        assert path.stat().st_size > 0
    assert result.seed == 4242
    assert result.metadata_path.is_file()


@pytest.mark.asyncio
async def test_same_seed_reproduces(settings: Settings, backend: Any, tmp_path: Path) -> None:
    """同じseedなら同じ絵になる (seedが実際に効いていることの確認)。"""
    first = await generate(
        _spec(output={"prefix": "it_seed_a"}), settings, backend=backend, project_root=tmp_path
    )
    second = await generate(
        _spec(output={"prefix": "it_seed_b"}), settings, backend=backend, project_root=tmp_path
    )

    assert first.files[0].read_bytes() == second.files[0].read_bytes()


@pytest.mark.asyncio
async def test_img2img_generates_image(settings: Settings, backend: Any, tmp_path: Path) -> None:
    from PIL import Image

    source = tmp_path / "inputs" / "ref.png"
    source.parent.mkdir(parents=True)
    Image.new("RGB", (512, 512), "navy").save(source)
    spec = _spec(
        task="img2img",
        generation={"steps": 2, "cfg": 7.0, "seed": 4242},
        source={"image": "inputs/ref.png", "denoise": 0.5},
        output={"prefix": "it_img2img"},
    )

    result = await generate(spec, settings, backend=backend, project_root=tmp_path)

    assert result.files
    assert result.files[0].stat().st_size > 0


@pytest.mark.asyncio
async def test_text_encoder_lora_is_rejected(
    settings: Settings, backend: Any, tmp_path: Path
) -> None:
    """text encoder側を持つLoRAは、読み込みに入る前に拒否すること。

    配布されているLoRAの多くは両方を持つ。diffusers 0.39はkohya形式の
    text encoder側を変換しきれず読み込みの途中で落ちるため、UNet側だけを
    当てて続けるのではなく生成前に止める (実ファイルで確かめる)。
    """
    lora = "add_detail.safetensors"
    path = settings.models_root / "loras" / lora  # type: ignore[operator]
    if not path.is_file():
        pytest.skip(f"{lora} が無いためskipします")

    spec = _spec(
        model={"checkpoint": CHECKPOINT, "loras": [{"name": lora, "strength_model": 1.0}]},
        output={"prefix": "it_lora_te"},
    )

    with pytest.raises(InvalidGenerationSpec) as exc:
        await generate(spec, settings, backend=backend, project_root=tmp_path)

    assert lora in str(exc.value)


@pytest.mark.asyncio
async def test_catalog_lists_local_models(settings: Settings) -> None:
    from agentic_imagegen.adapters.diffusers.catalog import DiffusersCatalog

    async with DiffusersCatalog(settings) as catalog:
        checkpoints = await catalog.available_checkpoints()
        controlnets = await catalog.available_controlnets()

    assert CHECKPOINT in checkpoints
    # 未対応の区分は置いてあっても選ばせない
    assert controlnets == ()
