"""生成サービス (Spec -> バックエンド実行 -> 保存) のテスト。

バックエンド固有の段取り (Workflow組み立て・embedding検証・入力画像アップロード・
ComfyUIの非同期キュー) は GenerationBackend.execute() の内側へ移った (Issue #31)。
それらの検証は tests/unit/test_comfyui_execute.py が担い、ここでは
「backendから受け取った結果をどう保存し、どうmetadataへ残すか」という
バックエンド非依存の振る舞いだけを検証する。
"""

import json
import re
from pathlib import Path
from typing import Any

import pytest

from agentic_imagegen.config import Settings
from agentic_imagegen.domain.models import GenerationSpec
from agentic_imagegen.domain.results import HealthStatus
from agentic_imagegen.errors import GenerationTimeout, InvalidGenerationSpec, OutputNotFound
from agentic_imagegen.services.generation import BackendOutput, generate

REQUEST_ID = "b3f0a1c2-0000-4000-8000-000000000001"
PNG = b"\x89PNG\r\n\x1a\n"

DEFAULT_INFO: dict[str, Any] = {
    "workflow": "txt2img",
    "workflow_hash": "sha256:" + "0" * 64,
    "backend": {
        "comfyui_version": "0.32.0",
        "devices": ("xpu:0 Intel(R) Graphics [0x7d55]",),
    },
}


class FakeBackend:
    """テスト用のバックエンド。実際のバックエンドへは接続しない。"""

    def __init__(
        self,
        *,
        images: tuple[bytes, ...] = (),
        suffixes: tuple[str, ...] | None = None,
        seed: int = 4242,
        request_id: str = REQUEST_ID,
        info: dict[str, Any] | None = None,
        execute_error: Exception | None = None,
    ) -> None:
        self.images = images
        self.suffixes = suffixes if suffixes is not None else (".png",) * len(images)
        self.seed = seed
        self.request_id = request_id
        self.info = info if info is not None else dict(DEFAULT_INFO)
        self.execute_error = execute_error
        self.received_spec: GenerationSpec | None = None
        self.received_project_root: Path | None = None
        self.received_timeout: float | None = None

    async def execute(
        self, spec: GenerationSpec, *, project_root: Path, timeout: float | None = None
    ) -> BackendOutput:
        self.received_spec = spec
        self.received_project_root = project_root
        self.received_timeout = timeout
        if self.execute_error is not None:
            raise self.execute_error
        return BackendOutput(
            images=self.images,
            seed=self.seed,
            request_id=self.request_id,
            info=self.info,
            suffixes=self.suffixes,
        )

    async def health(self) -> HealthStatus:
        return HealthStatus(
            base_url="http://127.0.0.1:8188",
            comfyui_version="0.32.0",
            devices=("xpu:0 Intel(R) Graphics [0x7d55]",),
        )


def _settings(tmp_path: Path, **overrides: Any) -> Settings:
    defaults: dict[str, Any] = {
        "comfyui_base_url": "http://127.0.0.1:8188",
        "max_width": 2048,
        "max_height": 2048,
        "max_pixels": 4194304,
        "max_batch": 4,
        "timeout_seconds": 30,
        "output_root": Path("outputs"),
    }
    return Settings(**{**defaults, **overrides})


def _spec(**overrides: Any) -> GenerationSpec:
    payload: dict[str, Any] = {
        "prompt": {"positive": "1girl, blue hair", "negative": "low quality"},
        "model": {"checkpoint": "v1-5-pruned-emaonly.safetensors"},
        "generation": {"width": 512, "height": 768, "seed": 4242},
        "output": {"prefix": "blue_hair"},
    }
    for section, values in overrides.items():
        if isinstance(values, dict) and isinstance(payload.get(section), dict):
            payload[section] = {**payload[section], **values}
        else:
            payload[section] = values
    return GenerationSpec.model_validate(payload)


def _images(count: int = 2) -> tuple[bytes, ...]:
    return tuple(PNG + f"image{i}".encode() for i in range(1, count + 1))


async def test_generate_saves_images(tmp_path: Path) -> None:
    backend = FakeBackend(images=_images())

    result = await generate(_spec(), _settings(tmp_path), backend=backend, project_root=tmp_path)

    assert result.prompt_id == REQUEST_ID
    assert result.seed == 4242
    assert [path.name for path in result.files] == ["image_0001.png", "image_0002.png"]
    assert all(path.is_file() for path in result.files)
    assert result.files[0].read_bytes() == _images()[0]
    assert result.files[1].read_bytes() == _images()[1]


async def test_generate_uses_dated_directory(tmp_path: Path) -> None:
    backend = FakeBackend(images=_images(1))

    result = await generate(_spec(), _settings(tmp_path), backend=backend, project_root=tmp_path)

    assert result.directory.parent.parent.name == "outputs"
    # ディレクトリ名は <時刻>_<prefix>
    assert re.fullmatch(r"\d{6}_blue_hair", result.directory.name), result.directory.name
    assert result.directory.is_relative_to(tmp_path)
    # 日付ディレクトリは YYYY-MM-DD 形式
    date_part = result.directory.parent.name
    assert len(date_part) == 10
    assert date_part.count("-") == 2


async def test_generate_writes_metadata(tmp_path: Path) -> None:
    backend = FakeBackend(images=_images(1))

    result = await generate(_spec(), _settings(tmp_path), backend=backend, project_root=tmp_path)

    metadata = json.loads(result.metadata_path.read_text(encoding="utf-8"))

    assert metadata["prompt_id"] == REQUEST_ID
    assert metadata["workflow"] == "txt2img"
    assert metadata["resolved_seed"] == 4242
    assert metadata["outputs"] == ["image_0001.png"]
    assert metadata["spec"]["prompt"]["positive"] == "1girl, blue hair"
    assert metadata["created_at"]


async def test_generate_relays_backend_info_verbatim(tmp_path: Path) -> None:
    """backend.execute() が返した info をそのままmetadataへ展開する。

    workflow名・workflow_hashの実体はComfyUIアダプタの責務であり、
    services層はどんな値であってもそのまま書き出すだけでよい。
    """
    info = {
        "workflow": "img2img",
        "workflow_hash": "sha256:" + "1" * 64,
        "backend": {"comfyui_version": "0.99.0", "devices": ["cpu"]},
    }
    backend = FakeBackend(images=_images(1), info=info)

    result = await generate(_spec(), _settings(tmp_path), backend=backend, project_root=tmp_path)

    metadata = json.loads(result.metadata_path.read_text(encoding="utf-8"))

    assert metadata["workflow"] == "img2img"
    assert metadata["workflow_hash"] == "sha256:" + "1" * 64
    assert metadata["backend"] == {"comfyui_version": "0.99.0", "devices": ["cpu"]}


async def test_generate_keeps_images_when_backend_info_missing(tmp_path: Path) -> None:
    """実行基盤の情報取得に失敗した場合、backendは info["backend"] を None にする。

    その場合でも生成済みの画像・metadataそのものは失わない。
    """
    backend = FakeBackend(images=_images(1), info={**DEFAULT_INFO, "backend": None})

    result = await generate(_spec(), _settings(tmp_path), backend=backend, project_root=tmp_path)

    metadata = json.loads(result.metadata_path.read_text(encoding="utf-8"))

    assert metadata["backend"] is None
    assert result.files[0].is_file()


async def test_generate_passes_timeout(tmp_path: Path) -> None:
    backend = FakeBackend(images=_images(1))

    await generate(_spec(), _settings(tmp_path), backend=backend, project_root=tmp_path, timeout=12)

    assert backend.received_timeout == 12


async def test_generate_defaults_timeout_from_settings(tmp_path: Path) -> None:
    backend = FakeBackend(images=_images(1))

    await generate(_spec(), _settings(tmp_path), backend=backend, project_root=tmp_path)

    assert backend.received_timeout == 30


async def test_generate_passes_spec_and_project_root_to_backend(tmp_path: Path) -> None:
    backend = FakeBackend(images=_images(1))
    spec = _spec()

    await generate(spec, _settings(tmp_path), backend=backend, project_root=tmp_path)

    assert backend.received_spec is spec
    assert backend.received_project_root == tmp_path


async def test_generate_propagates_timeout_error(tmp_path: Path) -> None:
    backend = FakeBackend(execute_error=GenerationTimeout("timed out"))

    with pytest.raises(GenerationTimeout):
        await generate(_spec(), _settings(tmp_path), backend=backend, project_root=tmp_path)


async def test_generate_without_images_raises(tmp_path: Path) -> None:
    backend = FakeBackend(execute_error=OutputNotFound("no images"))

    with pytest.raises(OutputNotFound):
        await generate(_spec(), _settings(tmp_path), backend=backend, project_root=tmp_path)


async def test_generate_rejects_output_outside_project_root(tmp_path: Path) -> None:
    backend = FakeBackend(images=_images(1))

    with pytest.raises(InvalidGenerationSpec):
        await generate(
            _spec(output={"directory": "../escape"}),
            _settings(tmp_path),
            backend=backend,
            project_root=tmp_path,
        )


async def test_generate_uses_settings_output_root_when_spec_omits_it(tmp_path: Path) -> None:
    backend = FakeBackend(images=_images(1))
    settings = _settings(tmp_path, output_root=Path("generated"))

    result = await generate(_spec(), settings, backend=backend, project_root=tmp_path)

    assert result.directory.parent.parent.name == "generated"


async def test_generate_does_not_overwrite_existing_run(tmp_path: Path) -> None:
    backend = FakeBackend(images=_images(1))
    settings = _settings(tmp_path)

    first = await generate(_spec(), settings, backend=backend, project_root=tmp_path)
    second = await generate(_spec(), settings, backend=backend, project_root=tmp_path)

    assert first.directory != second.directory
    assert first.files[0].is_file()
    assert second.files[0].is_file()
