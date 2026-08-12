"""ComfyUI Clientのテスト。実ComfyUIへは接続せずMockTransportで代替する。"""

from typing import Any

import httpx
import pytest

from agentic_imagegen.adapters.comfyui.client import ComfyUIClient
from agentic_imagegen.config import Settings
from agentic_imagegen.errors import ComfyUIUnavailable

SYSTEM_STATS: dict[str, Any] = {
    "system": {
        "os": "posix",
        "comfyui_version": "0.3.40",
        "python_version": "3.12.13",
    },
    "devices": [{"name": "cpu", "type": "cpu", "vram_total": 0}],
}

OBJECT_INFO: dict[str, Any] = {
    "CheckpointLoaderSimple": {
        "input": {
            "required": {
                "ckpt_name": [["v1-5-pruned-emaonly.safetensors", "sd15/anything.safetensors"], {}]
            }
        }
    }
}


def _settings(**overrides: Any) -> Settings:
    from pathlib import Path

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


def _client(handler: Any, **settings_overrides: Any) -> ComfyUIClient:
    return ComfyUIClient(_settings(**settings_overrides), transport=httpx.MockTransport(handler))


async def test_health_returns_status() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json=SYSTEM_STATS)

    async with _client(handler) as client:
        status = await client.health()

    assert status.base_url == "http://127.0.0.1:8188"
    assert status.comfyui_version == "0.3.40"
    assert status.devices == ("cpu",)
    assert str(requests[0].url) == "http://127.0.0.1:8188/system_stats"


async def test_health_tolerates_missing_fields() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={})

    async with _client(handler) as client:
        status = await client.health()

    assert status.comfyui_version is None
    assert status.devices == ()


async def test_health_connection_error_raises() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    async with _client(handler) as client:
        with pytest.raises(ComfyUIUnavailable, match="接続できません"):
            await client.health()


async def test_health_timeout_raises() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("timed out", request=request)

    async with _client(handler) as client:
        with pytest.raises(ComfyUIUnavailable):
            await client.health()


@pytest.mark.parametrize("status_code", [404, 500, 503])
async def test_health_http_error_raises(status_code: int) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status_code, text="error")

    async with _client(handler) as client:
        with pytest.raises(ComfyUIUnavailable, match=str(status_code)):
            await client.health()


async def test_health_invalid_json_raises() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="not json")

    async with _client(handler) as client:
        with pytest.raises(ComfyUIUnavailable):
            await client.health()


async def test_base_url_trailing_slash_is_normalized() -> None:
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(str(request.url))
        return httpx.Response(200, json=SYSTEM_STATS)

    async with _client(handler, comfyui_base_url="http://127.0.0.1:9000") as client:
        await client.health()

    assert seen == ["http://127.0.0.1:9000/system_stats"]


async def test_available_checkpoints() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/object_info/CheckpointLoaderSimple"
        return httpx.Response(200, json=OBJECT_INFO)

    async with _client(handler) as client:
        checkpoints = await client.available_checkpoints()

    assert checkpoints == (
        "v1-5-pruned-emaonly.safetensors",
        "sd15/anything.safetensors",
    )


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"CheckpointLoaderSimple": {}},
        {"CheckpointLoaderSimple": {"input": {"required": {}}}},
        {"CheckpointLoaderSimple": {"input": {"required": {"ckpt_name": []}}}},
        {"CheckpointLoaderSimple": {"input": {"required": {"ckpt_name": ["x"]}}}},
    ],
)
async def test_available_checkpoints_unexpected_shape_returns_empty(
    payload: dict[str, Any],
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=payload)

    async with _client(handler) as client:
        assert await client.available_checkpoints() == ()


async def test_available_checkpoints_unreachable_raises() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    async with _client(handler) as client:
        with pytest.raises(ComfyUIUnavailable):
            await client.available_checkpoints()


CONTROLNET_OBJECT_INFO: dict[str, Any] = {
    "ControlNetLoader": {
        "input": {
            "required": {
                "control_net_name": [
                    ["control_v11p_sd15_canny_fp16.safetensors", "control_v11f1p_sd15_depth.pth"],
                    {},
                ]
            }
        }
    }
}


async def test_available_controlnets() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/object_info/ControlNetLoader"
        return httpx.Response(200, json=CONTROLNET_OBJECT_INFO)

    async with _client(handler) as client:
        names = await client.available_controlnets()

    assert names == (
        "control_v11p_sd15_canny_fp16.safetensors",
        "control_v11f1p_sd15_depth.pth",
    )


async def test_available_controlnets_empty_when_none_installed() -> None:
    """ControlNetモデルが1つも置かれていない場合も空タプルになる。"""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"ControlNetLoader": {"input": {"required": {}}}})

    async with _client(handler) as client:
        assert await client.available_controlnets() == ()
