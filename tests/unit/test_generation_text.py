"""生成後のテキスト合成が generate へ組み込まれていることの検証。"""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any

import pytest
from PIL import Image

from agentic_imagegen.config import Settings
from agentic_imagegen.domain.models import GenerationSpec
from agentic_imagegen.domain.results import HealthStatus
from agentic_imagegen.errors import TextCompositionError
from agentic_imagegen.services.generation import BackendOutput, generate

PROMPT_ID = "b3f0a1c2-0000-4000-8000-000000000002"
CANVAS: tuple[int, int] = (128, 96)


def _png_bytes() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", CANVAS, (0, 0, 0)).save(buffer, format="PNG")
    return buffer.getvalue()


class FakeBackend:
    """実際のPNGを返すテスト用バックエンド。"""

    def __init__(self, *, count: int = 1) -> None:
        self.count = count

    async def execute(
        self, spec: GenerationSpec, *, project_root: Path, timeout: float | None = None
    ) -> BackendOutput:
        return BackendOutput(
            images=tuple(_png_bytes() for _ in range(self.count)),
            seed=1,
            request_id=PROMPT_ID,
            info={},
            suffixes=(".png",) * self.count,
        )

    async def health(self) -> HealthStatus:
        return HealthStatus(
            base_url="http://127.0.0.1:8188", comfyui_version="0.32.0", devices=("cpu",)
        )


class PartiallyBrokenBackend(FakeBackend):
    """指定したindexの画像として、画像として開けないバイト列を返すバックエンド。

    batch_size > 1 での途中失敗 (2件目のテキスト合成が失敗する状況) を再現するために使う。
    """

    def __init__(self, *, count: int, broken_indexes: frozenset[int]) -> None:
        super().__init__(count=count)
        self._broken_indexes = broken_indexes

    async def execute(
        self, spec: GenerationSpec, *, project_root: Path, timeout: float | None = None
    ) -> BackendOutput:
        images = tuple(
            b"not an image" if index in self._broken_indexes else _png_bytes()
            for index in range(self.count)
        )
        return BackendOutput(
            images=images, seed=1, request_id=PROMPT_ID, info={}, suffixes=(".png",) * self.count
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


def _spec(text: dict[str, Any] | None) -> GenerationSpec:
    payload: dict[str, Any] = {
        "prompt": {"positive": "a street"},
        "model": {"checkpoint": "v1-5-pruned-emaonly.safetensors"},
        "generation": {"width": CANVAS[0], "height": CANVAS[1], "seed": 1},
    }
    if text is not None:
        payload["text"] = text
    return GenerationSpec.model_validate(payload)


def _text(**overrides: Any) -> dict[str, Any]:
    layer: dict[str, Any] = {"content": "AB", "font": "test.ttf", "size": 24, "color": "#ff0000"}
    layer.update(overrides)
    return {"layers": [layer]}


def _backend() -> FakeBackend:
    return FakeBackend(count=1)


def _metadata(result: Any) -> dict[str, Any]:
    loaded: dict[str, Any] = json.loads(result.metadata_path.read_text(encoding="utf-8"))
    return loaded


@pytest.mark.asyncio
class TestGenerateWithoutText:
    async def test_creates_no_composed_file(self, tmp_path: Path) -> None:
        result = await generate(
            _spec(None),
            _settings(tmp_path),
            backend=_backend(),
            project_root=tmp_path,
        )

        assert result.text_files == ()
        assert list(result.directory.glob("*_text.png")) == []

    async def test_metadata_text_is_null(self, tmp_path: Path) -> None:
        result = await generate(
            _spec(None),
            _settings(tmp_path),
            backend=_backend(),
            project_root=tmp_path,
        )

        assert _metadata(result)["text"] is None


@pytest.mark.asyncio
class TestGenerateWithText:
    async def test_writes_composed_file(self, tmp_path: Path, fonts_root: Path) -> None:
        result = await generate(
            _spec(_text()),
            _settings(tmp_path, fonts_root=fonts_root),
            backend=_backend(),
            project_root=tmp_path,
        )

        assert len(result.text_files) == 1
        composed = result.text_files[0]
        assert composed.name == "image_0001_text.png"
        assert composed.is_file()

    async def test_keeps_raw_generated_image(self, tmp_path: Path, fonts_root: Path) -> None:
        result = await generate(
            _spec(_text()),
            _settings(tmp_path, fonts_root=fonts_root),
            backend=_backend(),
            project_root=tmp_path,
        )

        assert len(result.files) == 1
        assert result.files[0].name == "image_0001.png"
        assert result.files[0].read_bytes() == _png_bytes()

    async def test_metadata_records_compose_result(self, tmp_path: Path, fonts_root: Path) -> None:
        result = await generate(
            _spec(_text()),
            _settings(tmp_path, fonts_root=fonts_root),
            backend=_backend(),
            project_root=tmp_path,
        )

        text = _metadata(result)["text"]
        assert text["outputs"] == ["image_0001_text.png"]
        assert text["fonts"] == [
            {"name": "test.ttf", "path": str(fonts_root / "test.ttf"), "index": 0}
        ]

    async def test_metadata_spec_keeps_text(self, tmp_path: Path, fonts_root: Path) -> None:
        result = await generate(
            _spec(_text()),
            _settings(tmp_path, fonts_root=fonts_root),
            backend=_backend(),
            project_root=tmp_path,
        )

        layers = _metadata(result)["spec"]["text"]["layers"]
        assert layers[0]["content"] == "AB"

    async def test_resolves_fonts_root_from_project_root(
        self, tmp_path: Path, fonts_root: Path
    ) -> None:
        # fonts_root fixture の中身を作業ルート配下へ写し、相対指定で引けることを確認する
        local_root = tmp_path / "assets" / "fonts"
        local_root.mkdir(parents=True)
        (local_root / "test.ttf").write_bytes((fonts_root / "test.ttf").read_bytes())

        result = await generate(
            _spec(_text()),
            _settings(tmp_path, fonts_root=Path("assets/fonts")),
            backend=_backend(),
            project_root=tmp_path,
        )

        assert result.text_files[0].is_file()


@pytest.mark.asyncio
class TestComposeFailure:
    async def test_fails_when_font_missing(self, tmp_path: Path) -> None:
        with pytest.raises(TextCompositionError):
            await generate(
                _spec(_text()),
                _settings(tmp_path, fonts_root=tmp_path / "absent"),
                backend=_backend(),
                project_root=tmp_path,
            )

    async def test_keeps_outputs_and_metadata(self, tmp_path: Path) -> None:
        with pytest.raises(TextCompositionError):
            await generate(
                _spec(_text()),
                _settings(tmp_path, fonts_root=tmp_path / "absent"),
                backend=_backend(),
                project_root=tmp_path,
            )

        images = sorted((tmp_path / "outputs").rglob("image_0001.png"))
        metadata = sorted((tmp_path / "outputs").rglob("metadata.json"))
        assert len(images) == 1
        assert len(metadata) == 1
        assert json.loads(metadata[0].read_text(encoding="utf-8"))["text"] is None


@pytest.mark.asyncio
class TestPartialComposeFailure:
    """batch_size > 1 で2枚目以降の合成が失敗しても、1枚目の成功分は残る。"""

    def _backend(self) -> PartiallyBrokenBackend:
        return PartiallyBrokenBackend(count=2, broken_indexes=frozenset({1}))

    async def test_keeps_earlier_success_as_text_file(
        self, tmp_path: Path, fonts_root: Path
    ) -> None:
        with pytest.raises(TextCompositionError):
            await generate(
                _spec(_text()),
                _settings(tmp_path, fonts_root=fonts_root),
                backend=self._backend(),
                project_root=tmp_path,
            )

        text_files = sorted((tmp_path / "outputs").rglob("*_text.png"))
        assert [path.name for path in text_files] == ["image_0001_text.png"]

    async def test_metadata_records_partial_success_with_error(
        self, tmp_path: Path, fonts_root: Path
    ) -> None:
        with pytest.raises(TextCompositionError):
            await generate(
                _spec(_text()),
                _settings(tmp_path, fonts_root=fonts_root),
                backend=self._backend(),
                project_root=tmp_path,
            )

        metadata_path = sorted((tmp_path / "outputs").rglob("metadata.json"))[0]
        text_info = json.loads(metadata_path.read_text(encoding="utf-8"))["text"]
        assert text_info is not None
        assert text_info["outputs"] == ["image_0001_text.png"]
        assert "error" in text_info
