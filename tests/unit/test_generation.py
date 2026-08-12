"""生成サービス (Spec -> Workflow -> バックエンド -> 保存) のテスト。"""

import json
from pathlib import Path
from typing import Any

import pytest

from agentic_imagegen.config import Settings
from agentic_imagegen.domain.models import GenerationSpec
from agentic_imagegen.domain.results import ImageRef
from agentic_imagegen.errors import GenerationTimeout, OutputNotFound
from agentic_imagegen.services.generation import generate

PROMPT_ID = "b3f0a1c2-0000-4000-8000-000000000001"
PNG = b"\x89PNG\r\n\x1a\n"


class FakeBackend:
    """テスト用のバックエンド。ComfyUIへは接続しない。"""

    def __init__(
        self,
        *,
        images: tuple[ImageRef, ...] = (),
        wait_error: Exception | None = None,
        outputs_error: Exception | None = None,
    ) -> None:
        self.images = images
        self.wait_error = wait_error
        self.outputs_error = outputs_error
        self.submitted: dict[str, Any] | None = None
        self.wait_timeout: float | None = None

    async def submit(self, workflow: dict[str, Any]) -> str:
        self.submitted = workflow
        return PROMPT_ID

    async def wait_for_completion(self, prompt_id: str, *, timeout: float | None = None) -> None:
        self.wait_timeout = timeout
        if self.wait_error is not None:
            raise self.wait_error

    async def fetch_outputs(self, prompt_id: str) -> tuple[ImageRef, ...]:
        if self.outputs_error is not None:
            raise self.outputs_error
        return self.images

    async def download(self, ref: ImageRef) -> bytes:
        return PNG + ref.filename.encode()


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


def _images(count: int = 2) -> tuple[ImageRef, ...]:
    return tuple(
        ImageRef(filename=f"blue_hair_{i:05d}_.png", subfolder="", type="output")
        for i in range(1, count + 1)
    )


async def test_generate_saves_images(tmp_path: Path) -> None:
    backend = FakeBackend(images=_images())

    result = await generate(_spec(), _settings(tmp_path), backend=backend, project_root=tmp_path)

    assert result.prompt_id == PROMPT_ID
    assert result.seed == 4242
    assert [path.name for path in result.files] == ["image_0001.png", "image_0002.png"]
    assert all(path.is_file() for path in result.files)
    assert result.files[0].read_bytes().startswith(PNG)


async def test_generate_uses_dated_directory(tmp_path: Path) -> None:
    backend = FakeBackend(images=_images(1))

    result = await generate(_spec(), _settings(tmp_path), backend=backend, project_root=tmp_path)

    assert result.directory.parent.parent.name == "outputs"
    assert result.directory.name == "blue_hair"
    assert result.directory.is_relative_to(tmp_path)
    # 日付ディレクトリは YYYY-MM-DD 形式
    date_part = result.directory.parent.name
    assert len(date_part) == 10
    assert date_part.count("-") == 2


async def test_generate_writes_metadata(tmp_path: Path) -> None:
    backend = FakeBackend(images=_images(1))

    result = await generate(_spec(), _settings(tmp_path), backend=backend, project_root=tmp_path)

    metadata = json.loads(result.metadata_path.read_text(encoding="utf-8"))

    assert metadata["prompt_id"] == PROMPT_ID
    assert metadata["workflow"] == "txt2img"
    assert metadata["resolved_seed"] == 4242
    assert metadata["outputs"] == ["image_0001.png"]
    assert metadata["spec"]["prompt"]["positive"] == "1girl, blue hair"
    assert metadata["created_at"]


async def test_generate_resolves_random_seed(tmp_path: Path) -> None:
    backend = FakeBackend(images=_images(1))

    result = await generate(
        _spec(generation={"seed": -1}),
        _settings(tmp_path),
        backend=backend,
        project_root=tmp_path,
    )

    assert result.seed >= 0
    assert backend.submitted is not None
    assert backend.submitted["3"]["inputs"]["seed"] == result.seed


async def test_generate_injects_spec_into_workflow(tmp_path: Path) -> None:
    backend = FakeBackend(images=_images(1))

    await generate(_spec(), _settings(tmp_path), backend=backend, project_root=tmp_path)

    assert backend.submitted is not None
    assert backend.submitted["6"]["inputs"]["text"] == "1girl, blue hair"
    assert backend.submitted["5"]["inputs"]["height"] == 768
    assert backend.submitted["9"]["inputs"]["filename_prefix"] == "blue_hair"


async def test_generate_does_not_overwrite_existing_run(tmp_path: Path) -> None:
    backend = FakeBackend(images=_images(1))
    settings = _settings(tmp_path)

    first = await generate(_spec(), settings, backend=backend, project_root=tmp_path)
    second = await generate(_spec(), settings, backend=backend, project_root=tmp_path)

    assert first.directory != second.directory
    assert first.files[0].is_file()
    assert second.files[0].is_file()


async def test_generate_passes_timeout(tmp_path: Path) -> None:
    backend = FakeBackend(images=_images(1))

    await generate(_spec(), _settings(tmp_path), backend=backend, project_root=tmp_path, timeout=12)

    assert backend.wait_timeout == 12


async def test_generate_defaults_timeout_from_settings(tmp_path: Path) -> None:
    backend = FakeBackend(images=_images(1))

    await generate(_spec(), _settings(tmp_path), backend=backend, project_root=tmp_path)

    assert backend.wait_timeout == 30


async def test_generate_propagates_timeout_error(tmp_path: Path) -> None:
    backend = FakeBackend(wait_error=GenerationTimeout("timed out"))

    with pytest.raises(GenerationTimeout):
        await generate(_spec(), _settings(tmp_path), backend=backend, project_root=tmp_path)


async def test_generate_without_images_raises(tmp_path: Path) -> None:
    backend = FakeBackend(outputs_error=OutputNotFound("no images"))

    with pytest.raises(OutputNotFound):
        await generate(_spec(), _settings(tmp_path), backend=backend, project_root=tmp_path)


async def test_generate_rejects_output_outside_project_root(tmp_path: Path) -> None:
    from agentic_imagegen.errors import InvalidGenerationSpec

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
