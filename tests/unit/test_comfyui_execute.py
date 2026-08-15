"""ComfyUIアダプタの execute() (services.generation.GenerationBackend実装) のテスト。

Workflow組み立て・embedding検証・入力画像アップロード・投入から出力取得までの
一連の段取りは、以前は services.generation.generate() が直接担っていたが、
GenerationBackend Protocolの抽象を上げるためComfyUIアダプタ (ComfyUIClient.execute())
へ移した (Issue #31)。ここではその結線が壊れていないことを、実ComfyUIへは
接続せずMockTransportで確認する。ComfyUIの低レベルHTTP呼び出し自体
(submit / wait_for_completion / fetch_outputs / download) の単体テストは
test_comfyui_execution.py が引き続き担う。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
import pytest

from agentic_imagegen.adapters.comfyui.client import ComfyUIClient
from agentic_imagegen.adapters.comfyui.workflow import IMG2IMG_BINDING
from agentic_imagegen.config import Settings
from agentic_imagegen.domain.models import GenerationSpec
from agentic_imagegen.errors import ComfyUIUnavailable, InvalidGenerationSpec

PROMPT_ID = "b3f0a1c2-0000-4000-8000-000000000003"
PNG = b"\x89PNG\r\n\x1a\n"

SYSTEM_STATS: dict[str, Any] = {
    "system": {"comfyui_version": "0.32.0"},
    "devices": [{"name": "cpu"}],
}

HISTORY_SUCCESS: dict[str, Any] = {
    PROMPT_ID: {
        "outputs": {
            "9": {
                "images": [{"filename": "blue_hair_00001_.png", "subfolder": "", "type": "output"}]
            }
        },
        "status": {"status_str": "success", "completed": True},
    }
}


def _settings(**overrides: Any) -> Settings:
    defaults: dict[str, Any] = {
        "comfyui_base_url": "http://127.0.0.1:8188",
        "max_width": 2048,
        "max_height": 2048,
        "max_pixels": 4194304,
        "max_batch": 4,
        "timeout_seconds": 5,
        "output_root": Path("outputs"),
        "max_source_bytes": 1024,
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


def _client(handler: Any, **settings_overrides: Any) -> ComfyUIClient:
    return ComfyUIClient(_settings(**settings_overrides), transport=httpx.MockTransport(handler))


def _disable_websocket(monkeypatch: pytest.MonkeyPatch) -> None:
    """WebSocket監視をスキップし、historyポーリングへ即フォールバックさせる。

    MockTransportはHTTPしか差し替えないため、そのままだと実際のWebSocket接続を
    試みてテストが不安定になる。既存の test_comfyui_execution.py と同じ手当て。
    """

    async def broken_websocket(self: ComfyUIClient, prompt_id: str, timeout: float) -> None:
        raise OSError("websocket unavailable")

    monkeypatch.setattr(ComfyUIClient, "_wait_via_websocket", broken_websocket)


def _happy_path_handler(captured: dict[str, Any]) -> Any:
    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/embeddings":
            return httpx.Response(200, json=[])
        if path == "/prompt":
            captured["submitted"] = json.loads(request.content)["prompt"]
            return httpx.Response(200, json={"prompt_id": PROMPT_ID})
        if path == f"/history/{PROMPT_ID}":
            return httpx.Response(200, json=HISTORY_SUCCESS)
        if path == "/view":
            return httpx.Response(200, content=PNG)
        if path == "/system_stats":
            return httpx.Response(200, json=SYSTEM_STATS)
        raise AssertionError(f"想定外のリクエスト: {path}")  # pragma: no cover

    return handler


async def test_execute_returns_images_seed_and_request_id(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _disable_websocket(monkeypatch)
    captured: dict[str, Any] = {}

    async with _client(_happy_path_handler(captured)) as client:
        output = await client.execute(_spec(), project_root=tmp_path, timeout=5)

    assert output.images == (PNG,)
    assert output.suffixes == (".png",)
    assert output.seed == 4242
    assert output.request_id == PROMPT_ID


async def test_execute_injects_spec_into_workflow(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """seed・prompt・解像度・出力prefixが実際に投入されるWorkflowへ反映される。"""
    _disable_websocket(monkeypatch)
    captured: dict[str, Any] = {}

    async with _client(_happy_path_handler(captured)) as client:
        await client.execute(_spec(), project_root=tmp_path, timeout=5)

    submitted = captured["submitted"]
    assert submitted["3"]["inputs"]["seed"] == 4242
    assert submitted["6"]["inputs"]["text"] == "1girl, blue hair"
    assert submitted["5"]["inputs"]["height"] == 768
    assert submitted["9"]["inputs"]["filename_prefix"] == "blue_hair"


async def test_execute_resolves_random_seed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _disable_websocket(monkeypatch)
    captured: dict[str, Any] = {}

    async with _client(_happy_path_handler(captured)) as client:
        output = await client.execute(
            _spec(generation={"seed": -1}), project_root=tmp_path, timeout=5
        )

    assert output.seed >= 0
    assert captured["submitted"]["3"]["inputs"]["seed"] == output.seed


async def test_execute_records_workflow_and_backend_info(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _disable_websocket(monkeypatch)
    captured: dict[str, Any] = {}

    async with _client(_happy_path_handler(captured)) as client:
        output = await client.execute(_spec(), project_root=tmp_path, timeout=5)

    assert output.info["workflow"] == "txt2img"
    assert output.info["workflow_hash"].startswith("sha256:")
    assert output.info["backend"] == {"comfyui_version": "0.32.0", "devices": ["cpu"]}


async def test_execute_keeps_images_when_backend_info_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """実行基盤の情報取得 (health) に失敗しても、生成そのものは失わない。"""
    _disable_websocket(monkeypatch)
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/system_stats":
            return httpx.Response(500, text="boom")
        return _happy_path_handler(captured)(request)

    async with _client(handler) as client:
        output = await client.execute(_spec(), project_root=tmp_path, timeout=5)

    assert output.info["backend"] is None
    assert output.images == (PNG,)


async def test_execute_rejects_missing_embedding_without_submitting(
    tmp_path: Path,
) -> None:
    """embedding:<name> が未配置なら、投入前にInvalidGenerationSpecで止める。

    ComfyUI自身は未配置のembeddingを見つけても例外にせず、警告ログを残して
    黙って無視するだけ (効かないことにユーザーが気づけない)。
    """
    submitted = {"called": False}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/embeddings":
            return httpx.Response(200, json=[])
        if request.url.path == "/prompt":  # pragma: no cover - 到達しないはず
            submitted["called"] = True
            return httpx.Response(200, json={"prompt_id": PROMPT_ID})
        raise AssertionError(f"想定外のリクエスト: {request.url.path}")  # pragma: no cover

    async with _client(handler) as client:
        with pytest.raises(InvalidGenerationSpec, match="easynegative"):
            await client.execute(
                _spec(prompt={"negative": "embedding:easynegative, worst quality"}),
                project_root=tmp_path,
                timeout=5,
            )

    assert submitted["called"] is False


async def test_execute_allows_placed_embedding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _disable_websocket(monkeypatch)
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/embeddings":
            return httpx.Response(200, json=["easynegative"])
        return _happy_path_handler(captured)(request)

    async with _client(handler) as client:
        output = await client.execute(
            _spec(prompt={"negative": "embedding:easynegative, worst quality"}),
            project_root=tmp_path,
            timeout=5,
        )

    assert output.request_id == PROMPT_ID


async def test_execute_reports_multiple_missing_embeddings(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/embeddings":
            return httpx.Response(200, json=["easynegative"])
        raise AssertionError(f"想定外のリクエスト: {request.url.path}")  # pragma: no cover

    async with _client(handler) as client:
        with pytest.raises(InvalidGenerationSpec, match="badhandv4"):
            await client.execute(
                _spec(
                    prompt={
                        "positive": "1girl, embedding:foo_style",
                        "negative": "embedding:easynegative, embedding:badhandv4",
                    }
                ),
                project_root=tmp_path,
                timeout=5,
            )


async def test_execute_allows_embedding_with_extension(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """拡張子付きで書いてもComfyUIは解決するため、こちらで拒んではいけない。"""
    _disable_websocket(monkeypatch)
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/embeddings":
            return httpx.Response(200, json=["easynegative"])
        return _happy_path_handler(captured)(request)

    async with _client(handler) as client:
        output = await client.execute(
            _spec(prompt={"negative": "embedding:easynegative.safetensors, worst quality"}),
            project_root=tmp_path,
            timeout=5,
        )

    assert output.request_id == PROMPT_ID


async def test_execute_rejects_unresolvable_embedding_reference(tmp_path: Path) -> None:
    """`1girl,embedding:easynegative` のように空白が無い書き方はComfyUIが解決しない。"""

    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError(f"想定外のリクエスト: {request.url.path}")  # pragma: no cover

    async with _client(handler) as client:
        with pytest.raises(InvalidGenerationSpec, match="空白"):
            await client.execute(
                _spec(prompt={"negative": "1girl,embedding:easynegative"}),
                project_root=tmp_path,
                timeout=5,
            )


async def test_execute_propagates_backend_error_during_embedding_lookup(tmp_path: Path) -> None:
    """embeddingの問い合わせでComfyUIへ到達できなければ、握り潰さず伝える。"""

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/embeddings":
            raise httpx.ConnectError("refused", request=request)
        raise AssertionError(f"想定外のリクエスト: {request.url.path}")  # pragma: no cover

    async with _client(handler) as client:
        with pytest.raises(ComfyUIUnavailable):
            await client.execute(
                _spec(prompt={"negative": "embedding:easynegative"}),
                project_root=tmp_path,
                timeout=5,
            )


async def test_execute_skips_embedding_lookup_when_not_referenced(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """promptにembedding:記法が無ければ、ComfyUIへ問い合わせない。"""
    _disable_websocket(monkeypatch)
    captured: dict[str, Any] = {}
    queried = {"embeddings": False}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/embeddings":
            queried["embeddings"] = True
        return _happy_path_handler(captured)(request)

    async with _client(handler) as client:
        await client.execute(_spec(), project_root=tmp_path, timeout=5)

    assert queried["embeddings"] is False


async def test_execute_uploads_source_image_and_injects_it_into_workflow(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """img2imgの入力画像がアップロードされ、そのアップロード名がWorkflowへ注入される。"""
    _disable_websocket(monkeypatch)
    (tmp_path / "inputs").mkdir()
    (tmp_path / "inputs" / "ref.png").write_bytes(PNG)

    img2img_spec = GenerationSpec.model_validate(
        {
            "version": "1",
            "task": "img2img",
            "prompt": {"positive": "1girl, blue hair"},
            "source": {"image": "inputs/ref.png", "denoise": 0.45},
            "generation": {"seed": 12345},
            "model": {"checkpoint": "v1-5-pruned-emaonly.safetensors"},
            "output": {"prefix": "img2img_test"},
        }
    )

    uploaded_name = "imagegen_test_ref.png"
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/embeddings":
            return httpx.Response(200, json=[])
        if path == "/upload/image":
            return httpx.Response(200, json={"name": uploaded_name})
        if path == "/prompt":
            captured["submitted"] = json.loads(request.content)["prompt"]
            return httpx.Response(200, json={"prompt_id": PROMPT_ID})
        if path == f"/history/{PROMPT_ID}":
            return httpx.Response(200, json=HISTORY_SUCCESS)
        if path == "/view":
            return httpx.Response(200, content=PNG)
        if path == "/system_stats":
            return httpx.Response(200, json=SYSTEM_STATS)
        raise AssertionError(f"想定外のリクエスト: {path}")  # pragma: no cover

    async with _client(handler) as client:
        output = await client.execute(img2img_spec, project_root=tmp_path, timeout=5)

    load_image = IMG2IMG_BINDING.nodes["source_image"].node_id
    assert captured["submitted"][load_image]["inputs"]["image"] == uploaded_name
    assert output.info["workflow"] == "img2img"


async def test_execute_wraps_upload_failure_with_label_and_no_absolute_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """upload_imageの失敗は `<label>.image を読み込めません: <指定値>` へ包み直す。

    adapterはComfyUI固有のファイル名しか知らないため、Specのどのフィールドの
    指定だったか (source/control/reference) をここで補う。作業ルート配下の
    絶対パスは利用者のディレクトリ構成を露出するため出さない。
    """
    (tmp_path / "inputs").mkdir()
    (tmp_path / "inputs" / "ref.png").write_bytes(PNG)

    async def failing_upload_image(self: ComfyUIClient, path: Path) -> str:
        raise InvalidGenerationSpec(f"入力画像を読み込めません: {path.name}")

    monkeypatch.setattr(ComfyUIClient, "upload_image", failing_upload_image)

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/embeddings":
            return httpx.Response(200, json=[])
        raise AssertionError(f"想定外のリクエスト: {request.url.path}")  # pragma: no cover

    img2img_spec = GenerationSpec.model_validate(
        {
            "version": "1",
            "task": "img2img",
            "prompt": {"positive": "1girl"},
            "model": {"checkpoint": "meinamix_v12Final.safetensors"},
            "source": {"image": "inputs/ref.png", "denoise": 0.45},
        }
    )

    async with _client(handler) as client:
        with pytest.raises(InvalidGenerationSpec) as excinfo:
            await client.execute(img2img_spec, project_root=tmp_path, timeout=5)

    message = str(excinfo.value)
    assert "source.image を読み込めません: inputs/ref.png" in message
    assert str(tmp_path) not in message
