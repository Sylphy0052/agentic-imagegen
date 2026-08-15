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
from agentic_imagegen.backends import open_catalog_backend, open_generation_backend
from agentic_imagegen.config import Settings
from agentic_imagegen.domain.results import GenerationResult
from agentic_imagegen.services import mcp_tools
from agentic_imagegen.services.batch import BatchOutcome
from agentic_imagegen.services.jobs import JobRegistry

SERVER_NAME: Final = "agentic-imagegen"

#: 実行中および完了済みの生成ジョブ。プロセス内に保持する。
_registry: Final = JobRegistry[GenerationResult]()

#: 一括生成のジョブ。結果の形が違うため単発とは別に持つ。
_batch_registry: Final = JobRegistry[list[BatchOutcome]]()

#: 使うバックエンドは IMAGEGEN_BACKEND で決まる (既定はComfyUI)。
#: 具象の選択は backends へ寄せ、composition root (このファイル) は
#: 「どちらのファクトリを渡すか」だけを知る。列挙系と生成系で開くものが
#: 違う場合があるため、ファクトリは2つに分けて渡す。
_CATALOG_FACTORY: Final = open_catalog_backend
_GENERATION_FACTORY: Final = open_generation_backend

server: Final = MCPServer(
    name=SERVER_NAME,
    version=__version__,
    instructions=(
        "ComfyUI経由でStable Diffusion系モデルの画像生成を行う。"
        "生成前に validate_generation でSpecを確認し、"
        "checkpointやLoRAは list_models / list_loras で、"
        "ControlNetモデルは list_controlnets で、"
        "IPAdapterは list_ipadapters / list_clip_visions で、"
        "generation.upscale.model は list_upscale_models で、"
        "promptに embedding:<name> を書く場合は list_embeddings で"
        "実在するものだけを指定すること。"
    ),
)


def project_root() -> Path:
    """作業ルート。出力先や入力画像の解決に使う。"""
    override = os.environ.get("IMAGEGEN_PROJECT_ROOT", "").strip()
    return Path(override).resolve() if override else Path.cwd()


@server.tool()
def validate_generation(spec: dict[str, Any]) -> dict[str, Any]:
    """GenerationSpecを検証する。画像は生成しない。

    presetの展開結果、選択されるWorkflowテンプレート、解像度、LoRA構成、
    ControlNet (control)、IPAdapter (reference)、hires fix (generation.upscale)
    の設定を返す。
    不正な場合も例外にせず valid: false と理由を返す。
    """
    return mcp_tools.validate_generation(
        spec, settings=Settings.from_env(), project_root=project_root()
    )


@server.tool()
async def list_models() -> list[str]:
    """ComfyUIが持っているcheckpoint名の一覧を返す。"""
    return await mcp_tools.list_models(Settings.from_env(), backend_factory=_CATALOG_FACTORY)


@server.tool()
async def list_loras() -> list[str]:
    """ComfyUIが持っているLoRA名の一覧を返す。"""
    return await mcp_tools.list_loras(Settings.from_env(), backend_factory=_CATALOG_FACTORY)


@server.tool()
async def list_controlnets() -> list[str]:
    """ComfyUIが持っているControlNetモデル名の一覧を返す。"""
    return await mcp_tools.list_controlnets(Settings.from_env(), backend_factory=_CATALOG_FACTORY)


@server.tool()
async def list_ipadapters() -> list[str]:
    """ComfyUIが持っているIPAdapterモデル名の一覧を返す。

    空の場合はカスタムノードが未導入で、reference (IPAdapter) を使えない。
    """
    return await mcp_tools.list_ipadapters(Settings.from_env(), backend_factory=_CATALOG_FACTORY)


@server.tool()
async def list_clip_visions() -> list[str]:
    """ComfyUIが持っているCLIP Visionモデル名の一覧を返す。

    reference.clip_vision にはここに出る名前だけを指定する。
    """
    return await mcp_tools.list_clip_visions(Settings.from_env(), backend_factory=_CATALOG_FACTORY)


@server.tool()
async def list_diffusion_models() -> list[str]:
    """ComfyUIが持っているUNet単体のモデル名の一覧を返す。

    DiT系モデル (Anima など) を使うときに model.unet へ指定する。
    """
    return await mcp_tools.list_diffusion_models(
        Settings.from_env(), backend_factory=_CATALOG_FACTORY
    )


@server.tool()
async def list_text_encoders() -> list[str]:
    """ComfyUIが持っているtext encoder名の一覧を返す。model.clip へ指定する。"""
    return await mcp_tools.list_text_encoders(Settings.from_env(), backend_factory=_CATALOG_FACTORY)


@server.tool()
async def list_vaes() -> list[str]:
    """ComfyUIが持っているVAE名の一覧を返す。model.vae へ指定する。"""
    return await mcp_tools.list_vaes(Settings.from_env(), backend_factory=_CATALOG_FACTORY)


@server.tool()
async def list_upscale_models() -> list[str]:
    """ComfyUIが持っているアップスケールモデル名の一覧を返す。

    generation.upscale.model へ指定する。空ならlatent拡大だけが使える。
    """
    return await mcp_tools.list_upscale_models(
        Settings.from_env(), backend_factory=_CATALOG_FACTORY
    )


@server.tool()
async def list_embeddings() -> list[str]:
    """ComfyUIが持っているTextual Inversion embedding名 (拡張子なし) の一覧を返す。

    prompt中の `embedding:<name>` にはここに出る名前だけを指定する。
    未配置のembeddingを指定した場合は生成前に拒否される。
    """
    return await mcp_tools.list_embeddings(Settings.from_env(), backend_factory=_CATALOG_FACTORY)


@server.tool()
async def generate_image(spec: dict[str, Any]) -> dict[str, Any]:
    """GenerationSpecに従って画像生成を開始する。

    生成には数十秒から数分かかるため、完了は待たずに job_id を返す。
    結果は get_generation_status で受け取る。不正なSpecはここで拒否される。

    Specに control を書けばControlNet (Canny) で構図を制御でき、
    reference を書けばIPAdapterで参照画像の特徴を引き継げる。
    generation.upscale を書けば hires fix で解像度を上げられる。
    使うWorkflowテンプレートはSpecの内容から自動的に決まる。

    このtoolは非同期にしておく必要がある。同期関数として登録すると
    実行中のイベントループが無い文脈で呼ばれ、ジョブを起動できない。
    """
    return mcp_tools.submit_generation(
        spec,
        settings=Settings.from_env(),
        project_root=project_root(),
        registry=_registry,
        backend_factory=_GENERATION_FACTORY,
    )


@server.tool()
async def generate_batch(
    specs: list[dict[str, Any]], seeds: list[int] | None = None
) -> dict[str, Any]:
    """複数のGenerationSpecをまとめて生成する。

    seeds を指定すると、Specごとに各seedを当てたものへ展開する
    (Spec1件 + seed3つなら3枚)。完了は待たずに job_id を返し、
    結果は get_batch_status で受け取る。

    検証は投入前に全件行う。1件でも不正なら1件も生成しない。
    生成は順に実行され、1件失敗しても残りは続く。

    generate_image と同じく非同期にしておく必要がある。
    """
    return mcp_tools.submit_batch(
        specs,
        seeds=seeds,
        settings=Settings.from_env(),
        project_root=project_root(),
        registry=_batch_registry,
        backend_factory=_GENERATION_FACTORY,
    )


@server.tool()
def get_batch_status(job_id: str) -> dict[str, Any]:
    """generate_batch で開始した一括生成の状態を返す。

    完了時は total / succeeded / failed と、1件ごとの結果を items で返す。
    1件失敗しても残りは続くため、失敗が混ざっていても status は completed になる。
    """
    return mcp_tools.get_batch_status(job_id, registry=_batch_registry, project_root=project_root())


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

    どのテンプレートを使うかは task と LoRA / control / upscale の指定有無から
    自動的に決まるため、呼び出し側が明示する必要はない。
    """
    return mcp_tools.list_workflows()


def main() -> None:
    """stdioでMCP Serverを起動する。"""
    server.run(transport="stdio")


__all__ = ["main", "project_root", "server"]
