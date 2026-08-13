"""MCP Serverの登録内容と、toolを実際に呼べることの確認。

ここではMCP層が薄いアダプタとして機能しているか (tool名・スキーマ・呼び出し) を見る。
中身のロジックは test_mcp_tools.py が担う。
"""

from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any

import pytest
from mcp.server.mcpserver.exceptions import ToolError

from agentic_imagegen import mcp_server

EXPECTED_TOOLS = {
    "validate_generation",
    "generate_image",
    "get_generation_status",
    "generate_batch",
    "get_batch_status",
    "list_models",
    "list_loras",
    "list_controlnets",
    "list_ipadapters",
    "list_clip_visions",
    "list_diffusion_models",
    "list_text_encoders",
    "list_vaes",
    "list_upscale_models",
    "list_embeddings",
    "list_workflows",
}


async def test_registers_expected_tools() -> None:
    tools = await mcp_server.server.list_tools()

    assert {tool.name for tool in tools} == EXPECTED_TOOLS


async def test_tools_have_descriptions() -> None:
    """toolの説明はクライアントが使い方を判断する材料になるため、空にしない。"""
    tools = await mcp_server.server.list_tools()

    for tool in tools:
        assert tool.description, f"{tool.name} に説明がありません"


async def test_validate_generation_schema_takes_spec() -> None:
    tools = {tool.name: tool for tool in await mcp_server.server.list_tools()}

    schema = tools["validate_generation"].input_schema

    assert "spec" in schema["properties"]
    assert schema["required"] == ["spec"]


async def test_generate_image_schema_takes_spec() -> None:
    tools = {tool.name: tool for tool in await mcp_server.server.list_tools()}

    schema = tools["generate_image"].input_schema

    assert "spec" in schema["properties"]
    assert schema["required"] == ["spec"]


async def test_get_generation_status_schema_takes_job_id() -> None:
    tools = {tool.name: tool for tool in await mcp_server.server.list_tools()}

    schema = tools["get_generation_status"].input_schema

    assert "job_id" in schema["properties"]
    assert schema["required"] == ["job_id"]


def test_generate_image_tool_is_async() -> None:
    """生成を投入するtoolは非同期にしておく。

    同期関数として登録すると、実行中のイベントループが無い文脈で呼ばれ、
    asyncio.create_task が `no running event loop` で失敗する。
    ユニットテストはasync文脈から呼ぶため気づけず、実サーバー経路でだけ壊れる。
    """
    assert inspect.iscoroutinefunction(mcp_server.generate_image)


def test_generate_batch_tool_is_async() -> None:
    """一括生成も同じ理由で非同期にしておく。"""
    assert inspect.iscoroutinefunction(mcp_server.generate_batch)


async def test_generate_batch_schema_takes_specs() -> None:
    tools = {tool.name: tool for tool in await mcp_server.server.list_tools()}

    schema = tools["generate_batch"].input_schema

    assert "specs" in schema["properties"]
    assert "seeds" in schema["properties"]
    # seedsは省略できる (指定しなければ掃引しない)
    assert schema["required"] == ["specs"]


async def test_get_batch_status_rejects_unknown_job() -> None:
    with pytest.raises(ToolError, match="job_id"):
        await mcp_server.server.call_tool("get_batch_status", {"job_id": "does-not-exist"})


async def test_get_generation_status_rejects_unknown_job() -> None:
    """未知のjob_idはエラーにする (黙って running を返さない)。

    サーバーを直接呼ぶと例外がそのまま上がる。クライアント経由では
    これが isError の応答へ変換される。
    """
    with pytest.raises(ToolError, match="job_id"):
        await mcp_server.server.call_tool("get_generation_status", {"job_id": "does-not-exist"})


async def test_list_workflows_returns_allowlist() -> None:
    result = await mcp_server.server.call_tool("list_workflows", {})

    payload = _payload(result)
    assert set(payload) == {
        "txt2img",
        "txt2img_lora",
        "txt2img_hires",
        "txt2img_lora_hires",
        "txt2img_controlnet",
        "txt2img_lora_controlnet",
        "img2img",
        "img2img_lora",
        "img2img_hires",
        "img2img_lora_hires",
        "img2img_controlnet",
        "img2img_lora_controlnet",
        "txt2img_hires_controlnet",
        "txt2img_lora_hires_controlnet",
        "img2img_hires_controlnet",
        "img2img_lora_hires_controlnet",
        "txt2img_ipadapter",
        "txt2img_lora_ipadapter",
        "img2img_ipadapter",
        "img2img_lora_ipadapter",
        "txt2img_controlnet_ipadapter",
        "txt2img_lora_controlnet_ipadapter",
        "img2img_controlnet_ipadapter",
        "img2img_lora_controlnet_ipadapter",
        "txt2img_unet",
        "txt2img_unet_hires",
        "img2img_unet",
        "img2img_unet_hires",
        "txt2img_hires_model",
        "txt2img_lora_hires_model",
        "img2img_hires_model",
        "img2img_lora_hires_model",
        "txt2img_unet_hires_model",
        "img2img_unet_hires_model",
        "txt2img_hires_model_controlnet",
        "txt2img_lora_hires_model_controlnet",
        "img2img_hires_model_controlnet",
        "img2img_lora_hires_model_controlnet",
        "img2img_vae",
        "img2img_vae_controlnet",
        "img2img_vae_controlnet_ipadapter",
        "img2img_vae_hires",
        "img2img_vae_hires_controlnet",
        "img2img_vae_hires_model",
        "img2img_vae_hires_model_controlnet",
        "img2img_vae_ipadapter",
        "img2img_vae_lora",
        "img2img_vae_lora_controlnet",
        "img2img_vae_lora_controlnet_ipadapter",
        "img2img_vae_lora_hires",
        "img2img_vae_lora_hires_controlnet",
        "img2img_vae_lora_hires_model",
        "img2img_vae_lora_hires_model_controlnet",
        "img2img_vae_lora_ipadapter",
        "txt2img_vae",
        "txt2img_vae_controlnet",
        "txt2img_vae_controlnet_ipadapter",
        "txt2img_vae_hires",
        "txt2img_vae_hires_controlnet",
        "txt2img_vae_hires_model",
        "txt2img_vae_hires_model_controlnet",
        "txt2img_vae_ipadapter",
        "txt2img_vae_lora",
        "txt2img_vae_lora_controlnet",
        "txt2img_vae_lora_controlnet_ipadapter",
        "txt2img_vae_lora_hires",
        "txt2img_vae_lora_hires_controlnet",
        "txt2img_vae_lora_hires_model",
        "txt2img_vae_lora_hires_model_controlnet",
        "txt2img_vae_lora_ipadapter",
    }


async def test_validate_generation_reports_valid(tmp_path: Path) -> None:
    spec = {
        "version": "1",
        "task": "txt2img",
        "prompt": {"positive": "1girl"},
        "model": {"checkpoint": "v1-5-pruned-emaonly.safetensors"},
    }

    result = await mcp_server.server.call_tool("validate_generation", {"spec": spec})

    payload = _payload(result)
    assert payload["valid"] is True
    assert payload["workflow"] == "txt2img"


async def test_validate_generation_reports_errors() -> None:
    """不正なSpecでもtool呼び出し自体は成功し、結果として理由を返す。"""
    result = await mcp_server.server.call_tool("validate_generation", {"spec": {"task": "txt2img"}})

    payload = _payload(result)
    assert payload["valid"] is False
    assert payload["errors"]


class TestProjectRoot:
    def test_defaults_to_cwd(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("IMAGEGEN_PROJECT_ROOT", raising=False)

        assert mcp_server.project_root() == Path.cwd()

    def test_honours_env(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        monkeypatch.setenv("IMAGEGEN_PROJECT_ROOT", str(tmp_path))

        assert mcp_server.project_root() == tmp_path.resolve()


def _payload(result: Any) -> Any:
    """call_tool の戻り (CallToolResult) から構造化データを取り出す。

    dictを返すtoolは structured_content に入る。listを返すtoolは要素ごとの
    TextContent へ分解されるため、そちらから組み立てる。
    """
    assert not result.is_error, [item.text for item in result.content]

    structured = result.structured_content
    if isinstance(structured, dict) and "result" in structured:
        return structured["result"]
    if isinstance(structured, dict):
        return structured
    return [item.text for item in result.content]
