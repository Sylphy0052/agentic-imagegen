"""MCP Server。Claude Code / Codex の双方から同じ基盤を使うための入口。

ここは薄いアダプタに留める。検証や生成のロジックは services / domain 側にあり、
CLIと同じ経路を通る。MCP経由で検証を迂回できる抜け道は作らない。

起動:
    uv run imagegen-mcp

作業ルートはカレントディレクトリを既定とし、IMAGEGEN_PROJECT_ROOT で上書きできる。
クライアントが任意のディレクトリからサーバーを起動するため、明示できる余地を残す。
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Final

from mcp.server.mcpserver import MCPServer

from agentic_imagegen import __version__
from agentic_imagegen.config import Settings
from agentic_imagegen.services import mcp_tools
from agentic_imagegen.services.jobs import JobRegistry

SERVER_NAME: Final = "agentic-imagegen"

#: 実行中および完了済みの生成ジョブ。プロセス内に保持する。
_registry: Final = JobRegistry()

server: Final = MCPServer(
    name=SERVER_NAME,
    version=__version__,
    instructions=(
        "ComfyUI経由でStable Diffusion系モデルの画像生成を行う。"
        "生成前に validate_generation でSpecを確認し、"
        "checkpointやLoRAは list_models / list_loras で実在するものだけを指定すること。"
    ),
)


def project_root() -> Path:
    """作業ルート。出力先や入力画像の解決に使う。"""
    override = os.environ.get("IMAGEGEN_PROJECT_ROOT", "").strip()
    return Path(override).resolve() if override else Path.cwd()


@server.tool()
def validate_generation(spec: dict[str, Any]) -> dict[str, Any]:
    """GenerationSpecを検証する。画像は生成しない。

    presetの展開結果、選択されるWorkflowテンプレート、解像度、LoRA構成を返す。
    不正な場合も例外にせず valid: false と理由を返す。
    """
    return mcp_tools.validate_generation(
        spec, settings=Settings.from_env(), project_root=project_root()
    )


@server.tool()
async def list_models() -> list[str]:
    """ComfyUIが持っているcheckpoint名の一覧を返す。"""
    return await mcp_tools.list_models(Settings.from_env())


@server.tool()
async def list_loras() -> list[str]:
    """ComfyUIが持っているLoRA名の一覧を返す。"""
    return await mcp_tools.list_loras(Settings.from_env())


@server.tool()
async def generate_image(spec: dict[str, Any]) -> dict[str, Any]:
    """GenerationSpecに従って画像生成を開始する。

    生成には数十秒から数分かかるため、完了は待たずに job_id を返す。
    結果は get_generation_status で受け取る。不正なSpecはここで拒否される。

    このtoolは非同期にしておく必要がある。同期関数として登録すると
    実行中のイベントループが無い文脈で呼ばれ、ジョブを起動できない。
    """
    return mcp_tools.submit_generation(
        spec,
        settings=Settings.from_env(),
        project_root=project_root(),
        registry=_registry,
    )


@server.tool()
def get_generation_status(job_id: str) -> dict[str, Any]:
    """generate_image で開始した生成の状態を返す。

    status は running / completed / failed。完了時は出力ファイルのパスとseedを、
    失敗時は理由と exit_code (CLIと同じ体系) を返す。パスは作業ルートからの相対。
    """
    return mcp_tools.get_generation_status(job_id, registry=_registry, project_root=project_root())


@server.tool()
def list_workflows() -> list[str]:
    """実行を許可しているWorkflowテンプレート名の一覧を返す。

    どのテンプレートを使うかは task と LoRA指定の有無から自動的に決まるため、
    呼び出し側が明示する必要はない。
    """
    return mcp_tools.list_workflows()


def main() -> None:
    """stdioでMCP Serverを起動する。"""
    server.run(transport="stdio")


__all__ = ["main", "project_root", "server"]
