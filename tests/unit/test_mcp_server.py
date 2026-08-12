"""MCP Serverの登録内容と、toolを実際に呼べることの確認。

ここではMCP層が薄いアダプタとして機能しているか (tool名・スキーマ・呼び出し) を見る。
中身のロジックは test_mcp_tools.py が担う。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from agentic_imagegen import mcp_server

EXPECTED_TOOLS = {"validate_generation", "list_models", "list_loras", "list_workflows"}


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


async def test_list_workflows_returns_allowlist() -> None:
    result = await mcp_server.server.call_tool("list_workflows", {})

    payload = _payload(result)
    assert set(payload) == {"txt2img", "txt2img_lora", "img2img", "img2img_lora"}


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
