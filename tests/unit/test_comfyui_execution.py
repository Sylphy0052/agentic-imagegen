"""ComfyUIへのWorkflow投入・実行監視・出力取得のテスト。"""

import json
from pathlib import Path
from typing import Any

import httpx
import pytest

from agentic_imagegen.adapters.comfyui.client import (
    ComfyUIClient,
    ImageRef,
    _is_completion_message,
    _to_websocket_url,
)
from agentic_imagegen.config import Settings
from agentic_imagegen.errors import (
    ComfyUIUnavailable,
    GenerationFailed,
    GenerationTimeout,
    OutputNotFound,
    WorkflowSubmissionError,
)

PROMPT_ID = "b3f0a1c2-0000-4000-8000-000000000001"

WORKFLOW: dict[str, Any] = {"3": {"class_type": "KSampler", "inputs": {"seed": 1}}}

HISTORY_SUCCESS: dict[str, Any] = {
    PROMPT_ID: {
        "outputs": {
            "9": {
                "images": [
                    {"filename": "blue_hair_00001_.png", "subfolder": "", "type": "output"},
                    {"filename": "blue_hair_00002_.png", "subfolder": "sub", "type": "output"},
                ]
            }
        },
        "status": {"status_str": "success", "completed": True},
    }
}

HISTORY_ERROR: dict[str, Any] = {
    PROMPT_ID: {
        "outputs": {},
        "status": {
            "status_str": "error",
            "completed": False,
            "messages": [["execution_error", {"exception_message": "OOM on node 3"}]],
        },
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
    }
    return Settings(**{**defaults, **overrides})


def _client(handler: Any, **overrides: Any) -> ComfyUIClient:
    return ComfyUIClient(_settings(**overrides), transport=httpx.MockTransport(handler))


# --- submit -----------------------------------------------------------------


async def test_submit_returns_prompt_id() -> None:
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        import json

        captured["path"] = request.url.path
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json={"prompt_id": PROMPT_ID, "number": 1})

    async with _client(handler) as client:
        prompt_id = await client.submit(WORKFLOW)

    assert prompt_id == PROMPT_ID
    assert captured["path"] == "/prompt"
    assert captured["body"]["prompt"] == WORKFLOW
    assert captured["body"]["client_id"] == client.client_id


async def test_submit_without_prompt_id_raises() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"number": 1})

    async with _client(handler) as client:
        with pytest.raises(WorkflowSubmissionError):
            await client.submit(WORKFLOW)


async def test_submit_validation_error_includes_node_details() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            400,
            json={
                "error": {"type": "prompt_outputs_failed_validation", "message": "invalid"},
                "node_errors": {"4": {"errors": [{"message": "ckpt_name not in list"}]}},
            },
        )

    async with _client(handler) as client:
        with pytest.raises(WorkflowSubmissionError, match="ckpt_name"):
            await client.submit(WORKFLOW)


async def test_submit_server_error_raises() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="boom")

    async with _client(handler) as client:
        with pytest.raises(WorkflowSubmissionError, match="500"):
            await client.submit(WORKFLOW)


async def test_submit_connection_error_raises_unavailable() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    async with _client(handler) as client:
        with pytest.raises(ComfyUIUnavailable):
            await client.submit(WORKFLOW)


# --- 実行監視 ---------------------------------------------------------------


async def test_wait_via_polling_completes() -> None:
    calls = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["count"] += 1
        if calls["count"] < 3:
            return httpx.Response(200, json={})
        return httpx.Response(200, json=HISTORY_SUCCESS)

    async with _client(handler) as client:
        await client.wait_for_completion(
            PROMPT_ID, timeout=5, poll_interval=0.01, use_websocket=False
        )

    assert calls["count"] == 3


async def test_wait_via_polling_detects_execution_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=HISTORY_ERROR)

    async with _client(handler) as client:
        with pytest.raises(GenerationFailed, match="OOM"):
            await client.wait_for_completion(
                PROMPT_ID, timeout=5, poll_interval=0.01, use_websocket=False
            )


async def test_wait_times_out() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={})

    async with _client(handler) as client:
        with pytest.raises(GenerationTimeout, match=r"0\.1"):
            await client.wait_for_completion(
                PROMPT_ID, timeout=0.1, poll_interval=0.01, use_websocket=False
            )


async def test_wait_falls_back_to_polling_when_websocket_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """WebSocketが使えない環境ではポーリングへ自動フォールバックする。"""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=HISTORY_SUCCESS)

    async def broken_websocket(self: ComfyUIClient, prompt_id: str, timeout: float) -> None:
        raise OSError("websocket unavailable")

    monkeypatch.setattr(ComfyUIClient, "_wait_via_websocket", broken_websocket)

    async with _client(handler) as client:
        await client.wait_for_completion(PROMPT_ID, timeout=5, poll_interval=0.01)


# --- 出力取得 ---------------------------------------------------------------


async def test_fetch_outputs() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == f"/history/{PROMPT_ID}"
        return httpx.Response(200, json=HISTORY_SUCCESS)

    async with _client(handler) as client:
        images = await client.fetch_outputs(PROMPT_ID)

    assert images == (
        ImageRef(filename="blue_hair_00001_.png", subfolder="", type="output"),
        ImageRef(filename="blue_hair_00002_.png", subfolder="sub", type="output"),
    )


async def test_fetch_outputs_missing_history_raises() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={})

    async with _client(handler) as client:
        with pytest.raises(OutputNotFound):
            await client.fetch_outputs(PROMPT_ID)


async def test_fetch_outputs_without_images_raises() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={PROMPT_ID: {"outputs": {}, "status": {}}})

    async with _client(handler) as client:
        with pytest.raises(OutputNotFound):
            await client.fetch_outputs(PROMPT_ID)


async def test_download_image() -> None:
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["path"] = request.url.path
        captured["params"] = dict(request.url.params)
        return httpx.Response(200, content=b"\x89PNG\r\n")

    ref = ImageRef(filename="blue_hair_00001_.png", subfolder="sub", type="output")

    async with _client(handler) as client:
        data = await client.download(ref)

    assert data == b"\x89PNG\r\n"
    assert captured["path"] == "/view"
    assert captured["params"] == {
        "filename": "blue_hair_00001_.png",
        "subfolder": "sub",
        "type": "output",
    }


async def test_download_image_not_found_raises() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, text="not found")

    ref = ImageRef(filename="absent.png", subfolder="", type="output")

    async with _client(handler) as client:
        with pytest.raises(OutputNotFound):
            await client.download(ref)


# --- WebSocketメッセージ解析 ------------------------------------------------


@pytest.mark.parametrize(
    ("base_url", "expected_scheme"),
    [
        ("http://127.0.0.1:8188", "ws"),
        ("https://example.test", "wss"),
    ],
)
def test_to_websocket_url(base_url: str, expected_scheme: str) -> None:
    url = _to_websocket_url(base_url, "abc-123")

    assert url.startswith(f"{expected_scheme}://")
    assert url.endswith("/ws?clientId=abc-123")


def test_completion_message_detected() -> None:
    raw = json.dumps({"type": "executing", "data": {"node": None, "prompt_id": PROMPT_ID}})

    assert _is_completion_message(raw, PROMPT_ID) is True


@pytest.mark.parametrize(
    "raw",
    [
        json.dumps({"type": "executing", "data": {"node": "3", "prompt_id": PROMPT_ID}}),
        json.dumps({"type": "executing", "data": {"node": None, "prompt_id": "other"}}),
        json.dumps({"type": "status", "data": {"prompt_id": PROMPT_ID}}),
        json.dumps({"type": "executing"}),
        json.dumps(["not", "a", "mapping"]),
        "not json at all",
    ],
)
def test_non_completion_messages_ignored(raw: str) -> None:
    assert _is_completion_message(raw, PROMPT_ID) is False


def test_execution_error_message_raises() -> None:
    raw = json.dumps(
        {
            "type": "execution_error",
            "data": {
                "prompt_id": PROMPT_ID,
                "node_type": "KSampler",
                "exception_message": "CUDA out of memory",
            },
        }
    )

    with pytest.raises(GenerationFailed, match="CUDA out of memory"):
        _is_completion_message(raw, PROMPT_ID)
