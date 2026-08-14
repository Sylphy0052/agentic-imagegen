"""MCP toolの中身 (services層) のテスト。

MCP層は薄いアダプタに留め、ロジックはここでテストできる形にしておく。
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

import pytest

from agentic_imagegen.config import Settings
from agentic_imagegen.services import mcp_tools
from agentic_imagegen.services.mcp_tools import (
    list_workflows,
    validate_generation,
)
from agentic_imagegen.workflows.axes import ALL_TEMPLATE_NAMES

VALID_SPEC: dict[str, Any] = {
    "version": "1",
    "task": "txt2img",
    "prompt": {"positive": "1girl, blue hair"},
    "generation": {"width": 512, "height": 768, "seed": 42},
    "model": {"checkpoint": "v1-5-pruned-emaonly.safetensors"},
}


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        comfyui_base_url="http://127.0.0.1:8188",
        max_width=2048,
        max_height=2048,
        max_pixels=4194304,
        max_batch=4,
        timeout_seconds=30,
        output_root=Path("outputs"),
        presets_root=tmp_path / "presets",
    )


class TestValidateGeneration:
    def test_valid_spec(self, settings: Settings, tmp_path: Path) -> None:
        result = validate_generation(VALID_SPEC, settings=settings, project_root=tmp_path)

        assert result["valid"] is True
        assert result["workflow"] == "txt2img"
        assert result["resolution"] == {"width": 512, "height": 768, "batch_size": 1}
        assert result["checkpoint"] == "v1-5-pruned-emaonly.safetensors"
        assert result["errors"] == []

    def test_reports_errors_instead_of_raising(self, settings: Settings, tmp_path: Path) -> None:
        """検証結果を得るのがtoolの目的なので、不正でも例外にせず結果として返す。"""
        broken = {**VALID_SPEC, "generation": {"width": 511}}

        result = validate_generation(broken, settings=settings, project_root=tmp_path)

        assert result["valid"] is False
        assert result["errors"]
        assert any("width" in message for message in result["errors"])

    def test_reports_policy_violation(self, settings: Settings, tmp_path: Path) -> None:
        """設定由来の上限超過も検証結果として返す。"""
        oversized = {**VALID_SPEC, "generation": {"width": 4096, "height": 4096}}

        result = validate_generation(oversized, settings=settings, project_root=tmp_path)

        assert result["valid"] is False
        assert result["errors"]

    def test_reports_lora_workflow(self, settings: Settings, tmp_path: Path) -> None:
        spec = {
            **VALID_SPEC,
            "model": {
                "checkpoint": "v1-5-pruned-emaonly.safetensors",
                "loras": [{"name": "add_detail.safetensors", "strength_model": 0.8}],
            },
        }

        result = validate_generation(spec, settings=settings, project_root=tmp_path)

        assert result["valid"] is True
        assert result["workflow"] == "txt2img_lora"
        assert result["loras"] == [
            {"name": "add_detail.safetensors", "strength_model": 0.8, "strength_clip": 1.0}
        ]

    def test_expands_presets(self, settings: Settings, tmp_path: Path) -> None:
        characters = settings.presets_root / "characters"
        characters.mkdir(parents=True)
        (characters / "kaede.yaml").write_text(
            "prompt:\n  positive: 1girl, solo, blue hair\n", encoding="utf-8"
        )
        spec = {**VALID_SPEC, "presets": {"character": "kaede"}}

        result = validate_generation(spec, settings=settings, project_root=tmp_path)

        assert result["valid"] is True
        assert result["presets"] == {"character": "kaede"}
        assert result["prompt"]["positive"].startswith("1girl, solo, blue hair")

    def test_reports_img2img_source(self, settings: Settings, tmp_path: Path) -> None:
        spec = {
            **VALID_SPEC,
            "task": "img2img",
            "generation": {"seed": 42},
            "source": {"image": "inputs/ref.png", "denoise": 0.4},
        }

        result = validate_generation(spec, settings=settings, project_root=tmp_path)

        assert result["valid"] is True
        assert result["workflow"] == "img2img"
        assert result["source"] == {"image": "inputs/ref.png", "denoise": 0.4}
        # img2imgは入力画像のサイズを使うため解像度は返さない
        assert result["resolution"] is None

    def test_rejects_non_mapping(self, settings: Settings, tmp_path: Path) -> None:
        result = validate_generation(
            ["not", "a", "mapping"], settings=settings, project_root=tmp_path
        )

        assert result["valid"] is False
        assert result["errors"]


class TestListWorkflows:
    def test_returns_allowed_workflows(self) -> None:
        """許可済みテンプレートの全件を返す。

        期待値をここへ手で並べると、軸を1本足すたびに組み合わせの数だけ
        書き足すことになる (Issue #84 で他の3か所から潰した重複)。
        `ALL_TEMPLATE_NAMES` は `workflows.axes` の列挙、`list_workflows()` が
        返すのは `adapters.comfyui.workflow` のbinding由来で、別経路のため
        突き合わせる意味がある。
        """
        names = list_workflows()

        assert set(names) == set(ALL_TEMPLATE_NAMES)
        assert names == sorted(names)
        # 代表例。軸の組み合わせが名前へどう並ぶかを目で見て分かるようにしておく
        assert {"txt2img", "txt2img_controlnet", "txt2img_controlnet_raw"} <= set(names)


class FakeCatalogClient:
    """列挙系テスト用のバックエンド代替。

    `mcp_tools._connect` の差し替え先。HTTPは行わず、呼ばれたメソッド名から
    固定値を組み立てて返すため、toolとバックエンドのメソッドの対応も検証できる。
    """

    def __init__(self) -> None:
        self.entered = 0
        self.exited = 0

    async def __aenter__(self) -> FakeCatalogClient:
        self.entered += 1
        return self

    async def __aexit__(self, *exc_info: object) -> bool:
        self.exited += 1
        return False

    def __getattr__(self, name: str) -> Callable[[], Awaitable[tuple[str, ...]]]:
        if not name.startswith("available_"):
            raise AttributeError(name)

        async def query() -> tuple[str, ...]:
            return (f"{name}-1", f"{name}-2")

        return query


#: 列挙系tool -> それが問い合わせるバックエンドのメソッド名。
CATALOG_CASES: list[tuple[Any, str]] = [
    (mcp_tools.list_models, "available_checkpoints"),
    (mcp_tools.list_loras, "available_loras"),
    (mcp_tools.list_controlnets, "available_controlnets"),
    (mcp_tools.list_ipadapters, "available_ipadapters"),
    (mcp_tools.list_clip_visions, "available_clip_visions"),
    (mcp_tools.list_diffusion_models, "available_diffusion_models"),
    (mcp_tools.list_text_encoders, "available_text_encoders"),
    (mcp_tools.list_vaes, "available_vaes"),
    (mcp_tools.list_upscale_models, "available_upscale_models"),
    (mcp_tools.list_embeddings, "available_embeddings"),
]


class TestCatalog:
    """列挙系がバックエンド接続を1か所 (`_connect`) 経由にしていること。"""

    @pytest.mark.parametrize(("tool", "method"), CATALOG_CASES)
    @pytest.mark.asyncio
    async def test_returns_names_from_backend(
        self,
        tool: Any,
        method: str,
        settings: Settings,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        fake = FakeCatalogClient()
        monkeypatch.setattr(mcp_tools, "_connect", lambda _settings: fake)

        names = await tool(settings)

        assert names == [f"{method}-1", f"{method}-2"]
        assert isinstance(names, list)

    @pytest.mark.parametrize(("tool", "method"), CATALOG_CASES)
    @pytest.mark.asyncio
    async def test_closes_connection(
        self,
        tool: Any,
        method: str,
        settings: Settings,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """接続は使い終わったら閉じる。"""
        fake = FakeCatalogClient()
        monkeypatch.setattr(mcp_tools, "_connect", lambda _settings: fake)

        await tool(settings)

        assert fake.entered == 1
        assert fake.exited == 1

    def test_covers_every_catalog_tool(self) -> None:
        """列挙系を足したらこのテーブルへも足す。

        バックエンドへ接続しない `list_workflows` だけを除いた全件を並べる。
        """
        exported = {
            name for name in dir(mcp_tools) if name.startswith("list_") and name != "list_workflows"
        }

        assert {tool.__name__ for tool, _ in CATALOG_CASES} == exported


class TestValidateReportsAdvancedOptions:
    """ControlNet と hires fix の指定が検証結果に現れることを見る。

    workflow名だけでは、どのパラメータで効いているのかまでは分からない。
    """

    def test_reports_control(self, settings: Settings, tmp_path: Path) -> None:
        spec = {
            **VALID_SPEC,
            "control": {
                "image": "inputs/pose.png",
                "model": "control_v11p_sd15_canny_fp16.safetensors",
                "strength": 0.9,
            },
        }

        result = validate_generation(spec, settings=settings, project_root=tmp_path)

        assert result["valid"] is True
        assert result["workflow"] == "txt2img_controlnet"
        assert result["control"] is not None
        assert result["control"]["image"] == "inputs/pose.png"
        assert result["control"]["model"] == "control_v11p_sd15_canny_fp16.safetensors"
        assert result["control"]["strength"] == 0.9

    def test_reports_reference(self, settings: Settings, tmp_path: Path) -> None:
        spec = {
            **VALID_SPEC,
            "reference": {
                "image": "inputs/character.png",
                "model": "ip-adapter-plus_sd15.safetensors",
                "clip_vision": "CLIP-ViT-H-14-laion2B-s32B-b79K.safetensors",
                "weight": 0.8,
            },
        }

        result = validate_generation(spec, settings=settings, project_root=tmp_path)

        assert result["valid"] is True
        assert result["workflow"] == "txt2img_ipadapter"
        assert result["reference"] is not None
        assert result["reference"]["image"] == "inputs/character.png"
        assert result["reference"]["model"] == "ip-adapter-plus_sd15.safetensors"
        assert result["reference"]["weight"] == 0.8

    def test_reports_upscale(self, settings: Settings, tmp_path: Path) -> None:
        spec = {
            **VALID_SPEC,
            "generation": {
                "width": 512,
                "height": 512,
                "seed": 42,
                "upscale": {"scale": 1.5, "denoise": 0.45},
            },
        }

        result = validate_generation(spec, settings=settings, project_root=tmp_path)

        assert result["valid"] is True
        assert result["workflow"] == "txt2img_hires"
        assert result["upscale"] is not None
        assert result["upscale"]["scale"] == 1.5
        assert result["upscale"]["denoise"] == 0.45

    def test_absent_options_are_none(self, settings: Settings, tmp_path: Path) -> None:
        result = validate_generation(VALID_SPEC, settings=settings, project_root=tmp_path)

        assert result["control"] is None
        assert result["upscale"] is None

    def test_failure_payload_has_same_keys(self, settings: Settings, tmp_path: Path) -> None:
        """成功と失敗で鍵の集合を揃える。呼び出し側の分岐を増やさないため。"""
        ok = validate_generation(VALID_SPEC, settings=settings, project_root=tmp_path)
        ng = validate_generation({"task": "txt2img"}, settings=settings, project_root=tmp_path)

        assert set(ok) == set(ng)
