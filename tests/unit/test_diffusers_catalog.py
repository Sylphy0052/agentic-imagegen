"""DiffusersCatalog のテスト。

ComfyUIバックエンドはHTTPで問い合わせるが、diffusersは models_root 配下を
そのまま見る。ここで見るのは「置いてあるものを拾えること」と
「使えない区分は置いてあっても選ばせないこと」の2点。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from agentic_imagegen.adapters.diffusers.catalog import DiffusersCatalog
from agentic_imagegen.config import Settings
from agentic_imagegen.errors import InvalidConfiguration


def _settings(models_root: Path | None) -> Settings:
    return Settings(
        comfyui_base_url="http://127.0.0.1:8188",
        max_width=2048,
        max_height=2048,
        max_pixels=4194304,
        max_batch=4,
        timeout_seconds=30,
        output_root=Path("outputs"),
        backend="diffusers",
        models_root=models_root,
    )


@pytest.fixture
def models_root(tmp_path: Path) -> Path:
    checkpoints = tmp_path / "checkpoints"
    checkpoints.mkdir()
    (checkpoints / "b.safetensors").write_bytes(b"x")
    (checkpoints / "a.ckpt").write_bytes(b"x")
    # 拡張子が対象外のものと、ディレクトリは拾わない
    (checkpoints / "notes.txt").write_bytes(b"x")
    (checkpoints / "nested").mkdir()

    loras = tmp_path / "loras"
    loras.mkdir()
    (loras / "add_detail.safetensors").write_bytes(b"x")

    controlnets = tmp_path / "controlnet"
    controlnets.mkdir()
    (controlnets / "canny.safetensors").write_bytes(b"x")
    return tmp_path


class TestAvailableEntries:
    async def test_lists_checkpoints_sorted(self, models_root: Path) -> None:
        async with DiffusersCatalog(_settings(models_root)) as catalog:
            assert await catalog.available_checkpoints() == ("a.ckpt", "b.safetensors")

    async def test_lists_loras(self, models_root: Path) -> None:
        async with DiffusersCatalog(_settings(models_root)) as catalog:
            assert await catalog.available_loras() == ("add_detail.safetensors",)

    async def test_missing_directory_is_empty(self, tmp_path: Path) -> None:
        """モデルを置く前でも列挙そのものは通す (未配置と壊れた設定を区別する)。"""
        async with DiffusersCatalog(_settings(tmp_path)) as catalog:
            assert await catalog.available_checkpoints() == ()

    async def test_missing_models_root_is_rejected(self) -> None:
        catalog = DiffusersCatalog(_settings(None))

        with pytest.raises(InvalidConfiguration) as exc:
            await catalog.available_checkpoints()

        assert "IMAGEGEN_MODELS_ROOT" in str(exc.value)


class TestUnsupportedKinds:
    """未対応の区分は、ファイルが置いてあっても選ばせない。

    一覧に出すと指定できるように見えてしまい、生成の直前まで気づけない。
    """

    async def test_controlnets_are_empty_even_when_present(self, models_root: Path) -> None:
        async with DiffusersCatalog(_settings(models_root)) as catalog:
            assert await catalog.available_controlnets() == ()

    async def test_every_unsupported_kind_is_empty(self, models_root: Path) -> None:
        async with DiffusersCatalog(_settings(models_root)) as catalog:
            assert await catalog.available_ipadapters() == ()
            assert await catalog.available_clip_visions() == ()
            assert await catalog.available_diffusion_models() == ()
            assert await catalog.available_text_encoders() == ()
            assert await catalog.available_upscale_models() == ()
            assert await catalog.available_embeddings() == ()


async def test_vaes_are_listed_for_reference(models_root: Path) -> None:
    """外部VAEへの差し替えは未対応だが、置いてあるものは見えるようにしておく。"""
    vaes = models_root / "vae"
    vaes.mkdir()
    (vaes / "kl-f8.safetensors").write_bytes(b"x")

    async with DiffusersCatalog(_settings(models_root)) as catalog:
        assert await catalog.available_vaes() == ("kl-f8.safetensors",)
