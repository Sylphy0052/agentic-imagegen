"""diffusersバックエンドのモデルファイル解決のテスト。

ComfyUIバックエンドはモデルの所在をComfyUIが解決するが、diffusersは
プロセス内で読むため自分でパスを組み立てる。置き場は IMAGEGEN_MODELS_ROOT。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from agentic_imagegen.adapters.diffusers.models import (
    has_sdxl_marker,
    has_text_encoder_weights,
    resolve_model_path,
)
from agentic_imagegen.config import Settings
from agentic_imagegen.errors import InvalidConfiguration, InvalidGenerationSpec


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


class TestResolveModelPath:
    def test_returns_path_under_category(self, tmp_path: Path) -> None:
        checkpoints = tmp_path / "checkpoints"
        checkpoints.mkdir()
        (checkpoints / "model.safetensors").write_bytes(b"x")

        path = resolve_model_path(_settings(tmp_path), "checkpoints", "model.safetensors")

        assert path == checkpoints / "model.safetensors"

    def test_missing_models_root_is_rejected(self) -> None:
        """diffusersを選んだのに置き場が未設定なら、生成に入る前に気づかせる。"""
        with pytest.raises(InvalidConfiguration) as exc:
            resolve_model_path(_settings(None), "checkpoints", "model.safetensors")

        assert "IMAGEGEN_MODELS_ROOT" in str(exc.value)

    def test_missing_file_is_rejected(self, tmp_path: Path) -> None:
        (tmp_path / "checkpoints").mkdir()

        with pytest.raises(InvalidGenerationSpec) as exc:
            resolve_model_path(_settings(tmp_path), "checkpoints", "absent.safetensors")

        assert "absent.safetensors" in str(exc.value)

    def test_escaping_the_root_is_rejected(self, tmp_path: Path) -> None:
        """Spec側でも弾いているが、パスを組み立てるここでも確かめる。"""
        outside = tmp_path / "outside.safetensors"
        outside.write_bytes(b"x")
        root = tmp_path / "root"
        (root / "checkpoints").mkdir(parents=True)

        with pytest.raises(InvalidGenerationSpec):
            resolve_model_path(_settings(root), "checkpoints", "../../outside.safetensors")

    def test_symlink_escaping_the_root_is_rejected(self, tmp_path: Path) -> None:
        """実体が外にあるsymlinkも root の外として扱う。"""
        outside = tmp_path / "outside.safetensors"
        outside.write_bytes(b"x")
        checkpoints = tmp_path / "root" / "checkpoints"
        checkpoints.mkdir(parents=True)
        (checkpoints / "linked.safetensors").symlink_to(outside)

        with pytest.raises(InvalidGenerationSpec):
            resolve_model_path(_settings(tmp_path / "root"), "checkpoints", "linked.safetensors")


class TestHasSdxlMarker:
    def test_sdxl_checkpoint(self) -> None:
        """SDXLは2つ目のtext encoder (OpenCLIP-G) を持つ。"""
        keys = [
            "model.diffusion_model.input_blocks.0.0.weight",
            "conditioner.embedders.0.transformer.text_model.embeddings.position_embedding.weight",
            "conditioner.embedders.1.model.token_embedding.weight",
        ]

        assert has_sdxl_marker(keys) is True

    def test_sd15_checkpoint(self) -> None:
        keys = [
            "model.diffusion_model.input_blocks.0.0.weight",
            "cond_stage_model.transformer.text_model.embeddings.position_embedding.weight",
        ]

        assert has_sdxl_marker(keys) is False

    def test_prefix_must_match_exactly(self) -> None:
        """`conditioner.embedders.10` のような別の番号を取り違えない。"""
        assert has_sdxl_marker(["conditioner.embedders.11.model.weight"]) is False


class TestHasTextEncoderWeights:
    """LoRAがtext encoder側の重みを持つかの判定。

    diffusers 0.39はkohya形式のtext encoder側を読めないため、生成前に見分ける。
    """

    def test_kohya_text_encoder_keys(self) -> None:
        keys = [
            "lora_unet_down_blocks_0_attentions_0_proj_in.lora_down.weight",
            "lora_te_text_model_encoder_layers_0_mlp_fc1.lora_down.weight",
        ]

        assert has_text_encoder_weights(keys) is True

    def test_sdxl_has_two_text_encoders(self) -> None:
        assert has_text_encoder_weights(["lora_te2_text_model_encoder_layers_0_mlp_fc1.alpha"])

    def test_unet_only_lora(self) -> None:
        keys = [
            "lora_unet_down_blocks_0_attentions_0_proj_in.lora_down.weight",
            "lora_unet_mid_block_attentions_0_proj_out.alpha",
        ]

        assert has_text_encoder_weights(keys) is False
